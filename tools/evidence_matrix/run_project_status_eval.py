from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.evidence_matrix import audit_evidence_comparison, find_evidence_comparison_candidates
from kb.evidence_watch import source_identity
from kb.project_status import build_project_research_status
from kb.store import load_all_chunks


DEFAULT_FIXTURE = ROOT / "docs" / "project_research_status_eval_v1.json"


def _normal(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version") or 0) != 1:
        raise ValueError("project research status fixture version must be 1")
    cases = [item for item in list(payload.get("cases") or []) if isinstance(item, dict)]
    if len(cases) != 5:
        raise ValueError("project research status fixture requires exactly five reviewed states")
    expected_states = {
        "source_change",
        "unsupported_evidence",
        "comparison_candidates",
        "stale_brief",
        "ready",
    }
    if {str(item.get("state") or "") for item in cases} != expected_states:
        raise ValueError("project research status fixture does not cover the five required states")
    return payload


def _candidate_spec(candidate: dict[str, Any]) -> dict[str, Any]:
    confirmed = set(candidate.get("required_confirmations") or [])
    return {
        "mode": "ranking",
        "left_row_id": str(candidate.get("left_row_id") or ""),
        "right_row_id": str(candidate.get("right_row_id") or ""),
        "dimensions": [
            {
                "dimension": str(item.get("dimension") or ""),
                "left_value": str(item.get("left_value") or ""),
                "right_value": str(item.get("right_value") or ""),
                "mapping_confirmed": str(item.get("dimension") or "") in confirmed,
            }
            for item in list(candidate.get("dimensions") or [])
            if isinstance(item, dict)
        ],
        "left_target": str(candidate.get("left_target") or ""),
        "right_target": str(candidate.get("right_target") or ""),
        "left_result": str(candidate.get("left_result") or ""),
        "right_result": str(candidate.get("right_result") or ""),
    }


def _candidate_evidence_is_exact_and_locatable(
    candidates: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> bool:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
        by_source.setdefault(source_identity(meta.get("source_path")), []).append(chunk)
    for candidate in candidates:
        evidence = [item for item in list(candidate.get("evidence") or []) if isinstance(item, dict)]
        if len(evidence) < 2:
            return False
        for item in evidence:
            quote = _normal(item.get("evidence_quote"))
            source_chunks = by_source.get(source_identity(item.get("source_path")), [])
            if not quote or not any(quote in _normal(chunk.get("text")) for chunk in source_chunks):
                return False
            if not (
                item.get("page_start") is not None
                or item.get("heading_path")
                or item.get("block_id")
                or item.get("anchor_id")
            ):
                return False
    return True


def _gap(kind: str) -> dict[str, Any]:
    return {
        "id": f"gap-{kind}",
        "kind": kind,
        "status": "open",
        "matrix_id": "real-status-matrix",
        "brief_id": "real-status-brief" if kind == "brief_stale" else "",
    }


def run_eval(*, fixture_path: Path, db_root: Path) -> dict[str, Any]:
    fixture = _load_fixture(fixture_path)
    source_fixture_path = fixture_path.parent / str(fixture.get("source_fixture") or "")
    source_fixture = json.loads(source_fixture_path.read_text(encoding="utf-8"))
    source_paths = {
        key: (db_root.parent / str(relative)).resolve(strict=False)
        for key, relative in dict(source_fixture.get("sources") or {}).items()
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing reviewed source files: " + ", ".join(missing))
    load_started = time.perf_counter()
    chunks = [item for item in load_all_chunks(db_root) if isinstance(item, dict)]
    corpus_load_ms = round((time.perf_counter() - load_started) * 1_000.0, 3)
    rows = [
        {
            "id": "row-scigs",
            "paper": "SCIGS",
            "source_name": "SCIGS",
            "source_path": str(source_paths["scigs"]),
            "source_status": "active",
            "cells": {},
        },
        {
            "id": "row-scinerf",
            "paper": "SCINeRF",
            "source_name": "SCINeRF",
            "source_path": str(source_paths["scinerf"]),
            "source_status": "active",
            "cells": {},
        },
    ]
    base_matrix = {
        "id": "real-status-matrix",
        "title": "Reviewed SCIGS / SCINeRF matrix",
        "revision": 4,
        "quality_status": "verified",
        "updated_at": 40,
        "rows": rows,
        "source_items": [
            {"key": row["id"], "sourceName": row["source_name"], "sourcePath": row["source_path"]}
            for row in rows
        ],
        "comparison_audits": [],
    }
    pending_scan = find_evidence_comparison_candidates(
        base_matrix,
        db_dir=db_root,
        corpus_chunks=chunks,
        limit=100,
    )
    pending_candidates = [
        item for item in list(pending_scan.get("items") or []) if isinstance(item, dict)
    ]
    if len(pending_candidates) < 5:
        raise ValueError("reviewed real sources must expose at least five comparison candidates")
    if not _candidate_evidence_is_exact_and_locatable(pending_candidates, chunks):
        raise ValueError("a project-status comparison candidate lacks exact, locatable real evidence")

    resolved_matrix = copy.deepcopy(base_matrix)
    resolved_matrix["comparison_audits"] = [
        audit_evidence_comparison(
            rows=rows,
            spec=_candidate_spec(candidate),
            db_dir=db_root,
            corpus_chunks=chunks,
        )
        for candidate in pending_candidates
    ]
    if any(str(item.get("status") or "") != "verified" for item in resolved_matrix["comparison_audits"]):
        raise ValueError("reviewed comparison candidates did not all pass strict audit")
    resolved_scan = find_evidence_comparison_candidates(
        resolved_matrix,
        db_dir=db_root,
        corpus_chunks=chunks,
        limit=100,
    )
    if int(resolved_scan.get("candidate_count") or 0) != 0:
        raise ValueError("audited comparisons must not remain pending project actions")

    shelf = {
        "items": [
            {"key": row["id"], "sourcePath": row["source_path"], "sourceName": row["source_name"]}
            for row in rows
        ]
    }
    current_brief = {
        "id": "real-status-brief",
        "title": "Reviewed comparison brief",
        "revision": 2,
        "quality_status": "verified",
        "quality": {"source_matrix_id": base_matrix["id"]},
        "lineage": {"status": "current"},
        "updated_at": 50,
    }
    pending_scan_summary = {
        "candidate_count": int(pending_scan.get("candidate_count") or 0),
        "first_candidate_matrix_id": base_matrix["id"],
        "eligible_matrix_count": 1,
        "scanned_matrix_count": 1,
        "skipped_stale_matrix_count": 0,
        "scan_complete": True,
    }
    resolved_scan_summary = {
        "candidate_count": 0,
        "first_candidate_matrix_id": "",
        "eligible_matrix_count": 1,
        "scanned_matrix_count": 1,
        "skipped_stale_matrix_count": 0,
        "scan_complete": True,
    }

    results: list[dict[str, Any]] = []
    elapsed_values: list[float] = []
    for case in list(fixture.get("cases") or []):
        state = str(case.get("state") or "")
        matrix = copy.deepcopy(base_matrix if state in {"source_change", "unsupported_evidence", "comparison_candidates"} else resolved_matrix)
        gaps: list[dict[str, Any]] = []
        briefs = [copy.deepcopy(current_brief)]
        scan = pending_scan_summary if state in {"source_change", "unsupported_evidence", "comparison_candidates"} else resolved_scan_summary
        if state == "source_change":
            gaps = [_gap("source_change"), _gap("unsupported_cell")]
        elif state == "unsupported_evidence":
            gaps = [_gap("unsupported_cell")]
        elif state == "stale_brief":
            briefs[0]["lineage"] = {"status": "matrix_updated"}
            gaps = [_gap("brief_stale")]
        started = time.perf_counter()
        status = build_project_research_status(
            project={"id": "real-status-project", "name": "Reviewed real-paper project"},
            citation_shelf=shelf,
            matrices=[matrix],
            briefs=briefs,
            gaps=gaps,
            comparison_scan=scan,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1_000.0, 3)
        elapsed_values.append(elapsed_ms)
        checks = {
            "expected_action": str(status["recommended_action"]["code"])
            == str(case.get("expected_action") or ""),
            "expected_readiness": str(status["readiness"])
            == str(case.get("expected_readiness") or ""),
            "single_recommendation": isinstance(status.get("recommended_action"), dict),
            "real_source_count": int(status["stages"]["sources"]["project_source_count"]) == 2,
            "comparison_scan_complete": bool(status["stages"]["comparisons"]["scan_complete"]),
        }
        results.append(
            {
                "id": case.get("id"),
                "state": state,
                "passed": all(checks.values()),
                "checks": checks,
                "action": status["recommended_action"],
                "readiness": status["readiness"],
                "elapsed_ms": elapsed_ms,
            }
        )
    return {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "fixture": str(fixture_path),
        "source_fixture": str(source_fixture_path),
        "db_root": str(db_root),
        "corpus_chunk_count": len(chunks),
        "corpus_load_ms": corpus_load_ms,
        "real_pending_candidate_count": len(pending_candidates),
        "candidate_scan_ms": float((pending_scan.get("phase_timings_ms") or {}).get("total") or 0.0),
        "candidate_evidence_exact_and_locatable": True,
        "cases": results,
        "summary": {
            "passed": sum(1 for item in results if item.get("passed")),
            "total": len(results),
            "status_build_median_ms": round(statistics.median(elapsed_values), 3),
            "status_build_max_ms": round(max(elapsed_values), 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate deterministic project next actions on reviewed real-paper states."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--db-root", type=Path, default=ROOT / "db")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "test_results" / "project_research_status",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    fixture_path = args.fixture.resolve(strict=False)
    fixture = _load_fixture(fixture_path)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "ok": True,
                    "cases": len(fixture["cases"]),
                    "states": [item["state"] for item in fixture["cases"]],
                    "fixture": str(fixture_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    report = run_eval(
        fixture_path=fixture_path,
        db_root=args.db_root.resolve(strict=False),
    )
    folder = args.out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report["summary"]}, ensure_ascii=False))
    return 0 if int(report["summary"]["passed"]) == int(report["summary"]["total"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
