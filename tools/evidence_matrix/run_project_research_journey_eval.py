from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.main import app
from api.routers import chat as chat_router
from api.routers import evidence_matrices as matrix_router
from api.routers import research_briefs as brief_router
from api.routers import research_gaps as gap_router
from kb.chat_store import ChatStore
from kb.store import load_all_chunks


DEFAULT_FIXTURE = ROOT / "docs" / "project_research_journey_eval_v1.json"
EXPECTED_ACTIONS = [
    "add_project_sources",
    "create_evidence_matrix",
    "fill_evidence_gaps",
    "review_comparison_candidates",
    "create_research_brief",
    "export_current_brief",
]


def _records(value: object) -> list[dict[str, Any]]:
    return [item for item in list(value or []) if isinstance(item, dict)]


def _normal_path(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return str(Path(raw).expanduser().resolve(strict=False)).replace("\\", "/").casefold()


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version") or 0) != 1:
        raise ValueError("project research journey fixture version must be 1")
    sources = dict(payload.get("sources") or {})
    if not 3 <= len(sources) <= 5:
        raise ValueError("project research journey fixture requires three to five real sources")
    actions = [str(item or "") for item in list(payload.get("expected_actions") or [])]
    if actions != EXPECTED_ACTIONS:
        raise ValueError("project research journey fixture must cover the fixed six-stage action sequence")
    allowed = _records(payload.get("allowed_deferred_gaps"))
    allowed_keys = {(str(item.get("source_key") or ""), str(item.get("field") or "")) for item in allowed}
    if len(allowed_keys) != len(allowed):
        raise ValueError("deferred gap identities must be unique")
    if any(not source_key or source_key not in sources or not field for source_key, field in allowed_keys):
        raise ValueError("every deferred gap must name a fixture source and field")
    if int(payload.get("expected_matrix_rows") or 0) != len(sources):
        raise ValueError("expected matrix rows must cover every reviewed source")
    if int(payload.get("expected_comparison_candidates") or 0) <= 0:
        raise ValueError("expected comparison candidate coverage must be positive")
    review_groups = int(payload.get("expected_review_groups") or 0)
    confirmation_prompts = int(payload.get("expected_confirmation_prompts_before_reuse") or 0)
    confirmation_signatures = int(payload.get("expected_confirmation_signatures_after_reuse") or 0)
    if review_groups <= 0:
        raise ValueError("expected review group coverage must be positive")
    if confirmation_prompts <= 0 or not 0 < confirmation_signatures <= confirmation_prompts:
        raise ValueError("confirmation reuse expectations must preserve at least one reviewed signature")
    if int(payload.get("minimum_matrix_evidence") or 0) <= 0:
        raise ValueError("minimum matrix evidence must be positive")
    if int(payload.get("minimum_brief_evidence") or 0) <= 0:
        raise ValueError("minimum brief evidence must be positive")
    return payload


def _source_records(fixture: dict[str, Any], *, fixture_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for key, relative in dict(fixture.get("sources") or {}).items():
        path = (fixture_path.parent.parent / str(relative or "")).resolve(strict=False)
        if not path.is_file():
            missing.append(str(path))
            continue
        records.append(
            {
                "key": str(key),
                "title": path.stem,
                "sourceName": path.stem,
                "sourcePath": str(path),
                "libraryMatchPath": str(path),
            }
        )
    if missing:
        raise FileNotFoundError("missing reviewed source files: " + ", ".join(missing))
    return records


def _locator_present(item: dict[str, Any]) -> bool:
    return bool(
        item.get("page_start") is not None
        or item.get("page") is not None
        or item.get("heading_path")
        or item.get("headingPath")
        or item.get("block_id")
        or item.get("blockId")
        or item.get("anchor_id")
        or item.get("anchorId")
    )


def _review_value(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _candidate_dimension(candidate: dict[str, Any], name: str) -> dict[str, Any]:
    return next(
        (
            item
            for item in _records(candidate.get("dimensions"))
            if str(item.get("dimension") or "") == name
        ),
        {},
    )


def _candidate_review_group_signature(candidate: dict[str, Any]) -> tuple[str, ...]:
    task = _candidate_dimension(candidate, "task")
    dataset = _candidate_dimension(candidate, "dataset")
    return (
        str(candidate.get("matrix_id") or ""),
        str(candidate.get("left_row_id") or ""),
        str(candidate.get("right_row_id") or ""),
        _review_value(task.get("left_value")),
        _review_value(task.get("right_value")),
        _review_value(dataset.get("left_value")),
        _review_value(dataset.get("right_value")),
    )


def _candidate_confirmation_signature(
    candidate: dict[str, Any],
    dimension_name: str,
) -> tuple[str, ...]:
    dimension = _candidate_dimension(candidate, dimension_name)
    return (
        *_candidate_review_group_signature(candidate),
        dimension_name,
        _review_value(dimension.get("left_value")),
        _review_value(dimension.get("right_value")),
    )


def _corpus_text_by_source() -> dict[str, list[str]]:
    by_source: dict[str, list[str]] = {}
    for chunk in _records(load_all_chunks(ROOT / "db")):
        meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
        identity = _normal_path(meta.get("source_path"))
        text = " ".join(str(chunk.get("text") or "").split()).casefold()
        if identity and text:
            by_source.setdefault(identity, []).append(text)
    return by_source


def _evidence_exact_and_locatable(
    items: list[dict[str, Any]],
    *,
    corpus_text: dict[str, list[str]],
) -> bool:
    existing_sources: set[str] = set()
    for item in items:
        source_path = str(item.get("source_path") or item.get("sourcePath") or "").strip()
        quote = str(
            item.get("source_evidence_quote")
            or item.get("sourceEvidenceQuote")
            or item.get("evidence_quote")
            or item.get("evidenceQuote")
            or ""
        ).strip()
        if not source_path or not quote or not _locator_present(item):
            return False
        identity = _normal_path(source_path)
        if identity not in existing_sources:
            path = Path(source_path).expanduser().resolve(strict=False)
            if not path.is_file():
                return False
            existing_sources.add(identity)
        normalized_quote = " ".join(quote.split()).casefold()
        indexed_source_chunks = corpus_text.get(identity, [])
        if not any(normalized_quote in chunk for chunk in indexed_source_chunks):
            return False
    return bool(items)


@contextmanager
def _isolated_store(path: Path) -> Iterator[ChatStore]:
    store = ChatStore(path)
    modules = (chat_router, matrix_router, gap_router, brief_router)
    original = {module: module.get_chat_store for module in modules}
    try:
        for module in modules:
            module.get_chat_store = lambda store=store: store
        yield store
    finally:
        for module, getter in original.items():
            module.get_chat_store = getter


class JourneyClient:
    def __init__(self, client: TestClient) -> None:
        self.client = client
        self.timings: dict[str, list[float]] = {}

    def request(self, stage: str, method: str, path: str, **kwargs: Any):
        started = time.perf_counter()
        response = self.client.request(method, path, **kwargs)
        self.timings.setdefault(stage, []).append(round((time.perf_counter() - started) * 1_000.0, 3))
        if response.status_code >= 400:
            raise RuntimeError(f"{stage}: HTTP {response.status_code}: {response.text}")
        return response

    def timing_summary(self) -> dict[str, dict[str, float | int]]:
        return {
            stage: {
                "count": len(values),
                "total_ms": round(sum(values), 3),
                "median_ms": round(statistics.median(values), 3),
                "max_ms": round(max(values), 3),
            }
            for stage, values in self.timings.items()
            if values
        }


def _status(journey: JourneyClient, project_id: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = journey.request(
        "refresh_project_status",
        "POST",
        f"/api/projects/{project_id}/research-status/refresh",
    ).json()
    transition = {
        "stage": label,
        "action": str((payload.get("recommended_action") or {}).get("code") or ""),
        "readiness": str(payload.get("readiness") or ""),
        "active_gap_count": int(payload.get("active_gap_count") or 0),
        "pending_candidate_count": int((payload.get("stages") or {}).get("comparisons", {}).get("pending_candidate_count") or 0),
        "phase_timings_ms": dict(payload.get("phase_timings_ms") or {}),
    }
    return payload, transition


def _allowed_deferred_gaps(
    fixture: dict[str, Any],
    source_records: list[dict[str, Any]],
) -> dict[tuple[str, str], str]:
    paths = {str(item["key"]): _normal_path(item["sourcePath"]) for item in source_records}
    return {
        (paths[str(item.get("source_key") or "")], str(item.get("field") or "")): str(item.get("reason") or "")
        for item in _records(fixture.get("allowed_deferred_gaps"))
    }


def run_eval(*, fixture_path: Path, max_tokens: int) -> dict[str, Any]:
    fixture = _load_fixture(fixture_path)
    sources = _source_records(fixture, fixture_path=fixture_path)
    allowed_deferred = _allowed_deferred_gaps(fixture, sources)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    total_started = time.perf_counter()
    corpus_started = time.perf_counter()
    corpus_text = _corpus_text_by_source()
    corpus_load_ms = round((time.perf_counter() - corpus_started) * 1_000.0, 3)
    with TemporaryDirectory(prefix="pi_zaya_project_journey_") as temp_dir:
        with _isolated_store(Path(temp_dir) / "chat.sqlite3"):
            journey = JourneyClient(TestClient(app))
            project_id = journey.request(
                "create_project",
                "POST",
                "/api/projects",
                json={"name": str(fixture.get("project_name") or "Reviewed project journey")},
            ).json()["id"]
            conversation_id = journey.request(
                "create_project_conversation",
                "POST",
                "/api/conversations",
                json={"title": "Reviewed source workflow", "project_id": project_id},
            ).json()["id"]
            transitions: list[dict[str, Any]] = []
            _payload, transition = _status(journey, project_id, "empty_project")
            transitions.append(transition)

            journey.request(
                "add_reviewed_sources",
                "PATCH",
                f"/api/chat/citation-shelf?project_id={project_id}&scope=project",
                json={
                    "items": sources,
                    "open": True,
                    "project_id": project_id,
                    "scope": "project",
                },
            )
            _payload, transition = _status(journey, project_id, "sources_added")
            transitions.append(transition)

            matrix = journey.request(
                "generate_evidence_matrix",
                "POST",
                f"/api/projects/{project_id}/evidence-matrices/generate",
                json={
                    "title": str(fixture.get("matrix_title") or "Evidence matrix"),
                    "objective": str(fixture.get("matrix_objective") or ""),
                    "item_keys": [str(item["key"]) for item in sources],
                    "source_conv_id": conversation_id,
                },
            ).json()
            _payload, transition = _status(journey, project_id, "matrix_generated")
            transitions.append(transition)

            gap_decisions: list[dict[str, Any]] = []
            while True:
                scan = journey.request(
                    "scan_research_gaps",
                    "POST",
                    f"/api/projects/{project_id}/research-gaps/scan",
                ).json()
                gaps = _records(scan.get("items"))
                if not gaps:
                    break
                gap = gaps[0]
                repairs = journey.request(
                    "search_same_source_repairs",
                    "GET",
                    f"/api/projects/{project_id}/research-gaps/{gap['id']}/repairs?limit=8",
                ).json()
                repair_items = _records(repairs.get("items"))
                if repair_items:
                    repair = repair_items[0]
                    result = journey.request(
                        "apply_same_source_repair",
                        "POST",
                        f"/api/projects/{project_id}/research-gaps/{gap['id']}/repairs/{repair['id']}/apply",
                        json={"expected_matrix_revision": int(gap.get("matrix_revision") or 0)},
                    ).json()
                    matrix = result["matrix"]
                    gap_decisions.append(
                        {
                            "kind": gap.get("kind"),
                            "source_path": gap.get("source_path"),
                            "field": gap.get("field"),
                            "decision": "same_source_repair",
                            "candidate_count": len(repair_items),
                            "evidence_exact_and_locatable": _evidence_exact_and_locatable(
                                [repair],
                                corpus_text=corpus_text,
                            ),
                        }
                    )
                    continue
                identity = (_normal_path(gap.get("source_path")), str(gap.get("field") or ""))
                reason = allowed_deferred.get(identity)
                if not reason:
                    raise ValueError(
                        "an unreviewed gap has no strict repair and is not an allowed explicit deferral: "
                        f"{gap.get('source_name')} / {gap.get('field')}"
                    )
                journey.request(
                    "defer_reviewed_unavailable_gap",
                    "POST",
                    f"/api/projects/{project_id}/research-gaps/{gap['id']}/ignore",
                    json={"reason": reason},
                )
                gap_decisions.append(
                    {
                        "kind": gap.get("kind"),
                        "source_path": gap.get("source_path"),
                        "field": gap.get("field"),
                        "decision": "explicit_reviewed_deferral",
                        "candidate_count": 0,
                        "reason": reason,
                    }
                )
            _payload, transition = _status(journey, project_id, "evidence_gaps_reviewed")
            transitions.append(transition)

            audited: list[dict[str, Any]] = []
            initial_candidate_count: int | None = None
            review_group_count = 0
            confirmation_prompt_count = 0
            confirmation_signature_count = 0
            for _index in range(100):
                candidate_result = journey.request(
                    "scan_comparison_candidates",
                    "GET",
                    f"/api/projects/{project_id}/evidence-matrices/{matrix['id']}/comparison-candidates?limit=50",
                ).json()
                candidates = _records(candidate_result.get("items"))
                if initial_candidate_count is None:
                    initial_candidate_count = len(candidates)
                    review_group_count = len({_candidate_review_group_signature(item) for item in candidates})
                    confirmation_prompt_count = sum(
                        len(list(item.get("required_confirmations") or []))
                        for item in candidates
                    )
                    confirmation_signature_count = len(
                        {
                            _candidate_confirmation_signature(item, str(dimension or ""))
                            for item in candidates
                            for dimension in list(item.get("required_confirmations") or [])
                            if str(dimension or "")
                        }
                    )
                if not candidates:
                    break
                candidate = candidates[0]
                if not _evidence_exact_and_locatable(
                    _records(candidate.get("evidence")),
                    corpus_text=corpus_text,
                ):
                    raise ValueError("a comparison candidate lacks exact two-sided locatable evidence")
                result = journey.request(
                    "audit_comparison_candidate",
                    "POST",
                    f"/api/projects/{project_id}/evidence-matrices/{matrix['id']}/comparison-candidates/{candidate['id']}/audit",
                    json={
                        "expected_revision": int(matrix.get("revision") or 0),
                        "confirmed_mappings": list(candidate.get("required_confirmations") or []),
                    },
                ).json()
                matrix = result["matrix"]
                audit = result["audit"]
                audited.append(
                    {
                        "candidate_id": candidate.get("id"),
                        "status": audit.get("status"),
                        "evidence_count": len(_records(audit.get("evidence"))),
                        "evidence_exact_and_locatable": _evidence_exact_and_locatable(
                            _records(audit.get("evidence")),
                            corpus_text=corpus_text,
                        ),
                    }
                )
            else:
                raise RuntimeError("comparison candidate review did not converge")
            _payload, transition = _status(journey, project_id, "comparisons_reviewed")
            transitions.append(transition)

            brief = journey.request(
                "generate_verified_brief",
                "POST",
                f"/api/projects/{project_id}/research-briefs/generate",
                json={
                    "title": str(fixture.get("brief_title") or "Research brief"),
                    "objective": str(fixture.get("brief_objective") or ""),
                    "matrix_id": matrix["id"],
                    "source_conv_id": conversation_id,
                    "locale": "en",
                    "max_tokens": max_tokens,
                },
            ).json()
            final_status, transition = _status(journey, project_id, "brief_created")
            transitions.append(transition)
            exported = journey.request(
                "export_verified_brief",
                "GET",
                f"/api/research-briefs/{brief['id']}/export?format=markdown",
            )

            transition_actions = [str(item.get("action") or "") for item in transitions]
            matrix_evidence = _records(matrix.get("evidence"))
            brief_evidence = _records(brief.get("evidence"))
            checks = {
                "expected_action_sequence": transition_actions == list(fixture.get("expected_actions") or []),
                "three_to_five_real_sources": 3 <= len(sources) <= 5,
                "matrix_verified": str(matrix.get("quality_status") or "") == "verified",
                "matrix_row_count": len(_records(matrix.get("rows"))) == int(fixture.get("expected_matrix_rows") or 0),
                "matrix_evidence_floor": len(matrix_evidence) >= int(fixture.get("minimum_matrix_evidence") or 0),
                "matrix_evidence_exact_and_locatable": _evidence_exact_and_locatable(
                    matrix_evidence,
                    corpus_text=corpus_text,
                ),
                "no_unsupported_matrix_cells": not _records((matrix.get("quality") or {}).get("unsupported_cells")),
                "all_remaining_gaps_explicitly_reviewed": all(
                    str(item.get("decision") or "") in {"same_source_repair", "explicit_reviewed_deferral"}
                    for item in gap_decisions
                ),
                "expected_candidate_coverage": int(initial_candidate_count or 0)
                == int(fixture.get("expected_comparison_candidates") or 0),
                "expected_review_group_coverage": review_group_count
                == int(fixture.get("expected_review_groups") or 0),
                "conservative_confirmation_reuse": confirmation_prompt_count
                == int(fixture.get("expected_confirmation_prompts_before_reuse") or 0)
                and confirmation_signature_count
                == int(fixture.get("expected_confirmation_signatures_after_reuse") or 0),
                "all_candidates_strictly_audited": len(audited) == int(initial_candidate_count or 0)
                and all(str(item.get("status") or "") == "verified" for item in audited),
                "all_comparison_evidence_exact_and_locatable": bool(audited)
                and all(bool(item.get("evidence_exact_and_locatable")) for item in audited),
                "brief_verified": str(brief.get("quality_status") or "") == "verified",
                "brief_current": str((brief.get("lineage") or {}).get("status") or "") == "current",
                "brief_evidence_floor": len(brief_evidence) >= int(fixture.get("minimum_brief_evidence") or 0),
                "brief_evidence_exact_and_locatable": _evidence_exact_and_locatable(
                    brief_evidence,
                    corpus_text=corpus_text,
                ),
                "brief_bibliography_covers_sources": len(_records(brief.get("bibliography"))) == len(sources),
                "ready_only_after_verified_brief": str(final_status.get("readiness") or "") == "ready"
                and str((final_status.get("recommended_action") or {}).get("code") or "") == "export_current_brief",
                "markdown_export_contains_evidence_appendix": "Evidence appendix" in exported.text,
            }
            return {
                "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "fixture": str(fixture_path),
                "isolated_store": True,
                "source_count": len(sources),
                "corpus_source_count": len(corpus_text),
                "corpus_load_ms": corpus_load_ms,
                "sources": sources,
                "transitions": transitions,
                "gap_decisions": gap_decisions,
                "matrix": {
                    "id": matrix.get("id"),
                    "revision": matrix.get("revision"),
                    "quality_status": matrix.get("quality_status"),
                    "row_count": len(_records(matrix.get("rows"))),
                    "evidence_count": len(matrix_evidence),
                },
                "comparisons": {
                    "initial_candidate_count": int(initial_candidate_count or 0),
                    "audited_count": len(audited),
                    "review_efficiency": {
                        "group_count": review_group_count,
                        "confirmation_prompts_before_reuse": confirmation_prompt_count,
                        "confirmation_signatures_after_reuse": confirmation_signature_count,
                        "confirmation_actions_saved": max(
                            0,
                            confirmation_prompt_count - confirmation_signature_count,
                        ),
                    },
                    "items": audited,
                },
                "brief": {
                    "id": brief.get("id"),
                    "quality_status": brief.get("quality_status"),
                    "lineage_status": (brief.get("lineage") or {}).get("status"),
                    "evidence_count": len(brief_evidence),
                    "bibliography_count": len(_records(brief.get("bibliography"))),
                    "export_bytes": len(exported.content),
                },
                "phase_timings_ms": journey.timing_summary(),
                "total_elapsed_ms": round((time.perf_counter() - total_started) * 1_000.0, 3),
                "checks": checks,
                "summary": {
                    "passed": sum(1 for value in checks.values() if value),
                    "total": len(checks),
                    "failed": sorted(key for key, value in checks.items() if not value),
                },
            }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an isolated, real-paper project journey from project creation through verified brief export."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "test_results" / "project_research_journey",
    )
    parser.add_argument("--max-tokens", type=int, default=1_800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    fixture_path = args.fixture.resolve(strict=False)
    fixture = _load_fixture(fixture_path)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "ok": True,
                    "fixture": str(fixture_path),
                    "source_count": len(dict(fixture.get("sources") or {})),
                    "expected_actions": list(fixture.get("expected_actions") or []),
                    "expected_comparison_candidates": int(fixture.get("expected_comparison_candidates") or 0),
                    "expected_review_groups": int(fixture.get("expected_review_groups") or 0),
                    "expected_confirmation_signatures_after_reuse": int(
                        fixture.get("expected_confirmation_signatures_after_reuse") or 0
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    report = run_eval(fixture_path=fixture_path, max_tokens=max(400, min(4_096, int(args.max_tokens))))
    folder = args.out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report["summary"]}, ensure_ascii=False))
    return 0 if not report["summary"]["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
