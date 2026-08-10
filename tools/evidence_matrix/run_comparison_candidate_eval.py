from __future__ import annotations

import argparse
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

from kb.evidence_matrix import (
    audit_evidence_comparison,
    find_evidence_comparison_candidates,
)
from kb.evidence_watch import source_identity
from kb.store import load_all_chunks


DEFAULT_FIXTURE = ROOT / "docs" / "evidence_comparison_candidate_eval_v1.json"


def _normal(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version") or 0) != 1:
        raise ValueError("comparison candidate fixture version must be 1")
    if not isinstance(payload.get("sources"), dict) or len(list(payload.get("cases") or [])) < 5:
        raise ValueError("comparison candidate fixture requires sources and at least five cases")
    return payload


def _rows(source_paths: dict[str, Path]) -> list[dict[str, Any]]:
    return [
        {
            "id": "row-left",
            "source_item_key": "left",
            "paper": "SCIGS",
            "source_name": "SCIGS",
            "source_path": str(source_paths["scigs"]),
            "source_status": "active",
            "cells": {},
        },
        {
            "id": "row-right",
            "source_item_key": "right",
            "paper": "SCINeRF",
            "source_name": "SCINeRF",
            "source_path": str(source_paths["scinerf"]),
            "source_status": "active",
            "cells": {},
        },
    ]


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


def _evidence_exact_and_locatable(
    evidence: list[dict[str, Any]],
    chunks_by_source: dict[str, list[dict[str, Any]]],
) -> tuple[bool, bool]:
    exact = bool(evidence)
    locatable = bool(evidence)
    for item in evidence:
        source_chunks = chunks_by_source.get(source_identity(item.get("source_path")), [])
        quote = _normal(item.get("evidence_quote"))
        if not quote or not any(quote in _normal(chunk.get("text")) for chunk in source_chunks):
            exact = False
        if not (
            item.get("page_start") is not None
            or item.get("heading_path")
            or item.get("block_id")
            or item.get("anchor_id")
        ):
            locatable = False
    return exact, locatable


def run_eval(*, fixture_path: Path, db_root: Path) -> dict[str, Any]:
    fixture = _load_fixture(fixture_path)
    source_paths = {
        key: (db_root.parent / str(relative)).resolve(strict=False)
        for key, relative in dict(fixture["sources"]).items()
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing reviewed source files: " + ", ".join(missing))
    load_started = time.perf_counter()
    chunks = [item for item in load_all_chunks(db_root) if isinstance(item, dict)]
    corpus_load_ms = round((time.perf_counter() - load_started) * 1_000, 3)
    chunks_by_source: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
        chunks_by_source.setdefault(source_identity(meta.get("source_path")), []).append(chunk)
    rows = _rows(source_paths)
    matrix = {
        "id": "reviewed-comparison-candidate-matrix",
        "revision": 1,
        "rows": rows,
        "comparison_audits": [],
        "quality": {},
    }
    candidate_result = find_evidence_comparison_candidates(
        matrix,
        db_dir=db_root,
        corpus_chunks=chunks,
        limit=100,
    )
    candidates = [
        item for item in list(candidate_result.get("items") or []) if isinstance(item, dict)
    ]
    results: list[dict[str, Any]] = []
    audit_times: list[float] = []
    for case in list(fixture.get("cases") or []):
        expected = dict(case)
        candidate = next(
            (
                item
                for item in candidates
                if any(
                    str(dimension.get("dimension") or "") == "dataset"
                    and _normal(dimension.get("left_value")) == _normal(expected.get("dataset"))
                    and _normal(dimension.get("right_value")) == _normal(expected.get("dataset"))
                    for dimension in list(item.get("dimensions") or [])
                    if isinstance(dimension, dict)
                )
                and any(
                    str(dimension.get("dimension") or "") == "metric"
                    and _normal(dimension.get("left_value")) == _normal(expected.get("metric"))
                    and _normal(dimension.get("right_value")) == _normal(expected.get("metric"))
                    for dimension in list(item.get("dimensions") or [])
                    if isinstance(dimension, dict)
                )
            ),
            None,
        )
        if not isinstance(candidate, dict):
            results.append({"id": expected.get("id"), "passed": False, "error": "candidate_not_found"})
            continue
        exact, locatable = _evidence_exact_and_locatable(
            [item for item in list(candidate.get("evidence") or []) if isinstance(item, dict)],
            chunks_by_source,
        )
        audit = audit_evidence_comparison(
            rows=rows,
            spec=_candidate_spec(candidate),
            db_dir=db_root,
            corpus_chunks=chunks,
        )
        audit_ms = float((audit.get("phase_timings_ms") or {}).get("total") or 0.0)
        audit_times.append(audit_ms)
        checks = {
            "left_target": _normal(candidate.get("left_target")) == _normal(expected.get("left_target")),
            "right_target": _normal(candidate.get("right_target")) == _normal(expected.get("right_target")),
            "left_result": _normal(candidate.get("left_result")) == _normal(expected.get("left_result")),
            "right_result": _normal(candidate.get("right_result")) == _normal(expected.get("right_result")),
            "required_human_mapping": candidate.get("required_confirmations") == ["evaluation_protocol"],
            "candidate_same_source_exact": exact,
            "candidate_reader_locator": locatable,
            "audit_verified": str(audit.get("status") or "") == "verified",
            "preferred_side": str(audit.get("preferred_side") or "")
            == str(expected.get("expected_preferred_side") or ""),
            "audit_evidence_exact": _evidence_exact_and_locatable(
                [item for item in list(audit.get("evidence") or []) if isinstance(item, dict)],
                chunks_by_source,
            )[0],
        }
        results.append(
            {
                "id": expected.get("id"),
                "candidate_id": candidate.get("id"),
                "passed": all(checks.values()),
                "checks": checks,
                "audit_status": audit.get("status"),
                "relation": audit.get("relation"),
                "preferred_side": audit.get("preferred_side"),
                "audit_ms": audit_ms,
            }
        )

    audited_candidates = [
        audit_evidence_comparison(
            rows=rows,
            spec=_candidate_spec(candidate),
            db_dir=db_root,
            corpus_chunks=chunks,
        )
        for candidate in candidates
    ]
    candidate_evidence_failures = sum(
        1
        for candidate in candidates
        if not all(
            _evidence_exact_and_locatable(
                [
                    item
                    for item in list(candidate.get("evidence") or [])
                    if isinstance(item, dict)
                ],
                chunks_by_source,
            )
        )
    )
    candidate_prefill_failures = sum(
        1
        for candidate in candidates
        if len([item for item in list(candidate.get("dimensions") or []) if isinstance(item, dict)]) != 4
        or any(
            not str(item.get(side) or "").strip()
            for item in list(candidate.get("dimensions") or [])
            if isinstance(item, dict)
            for side in ("left_value", "right_value")
        )
        or any(
            not str(candidate.get(field) or "").strip()
            for field in ("left_target", "right_target", "left_result", "right_result")
        )
    )
    cross_dataset_candidates = sum(
        1
        for candidate in candidates
        for dimension in list(candidate.get("dimensions") or [])
        if isinstance(dimension, dict)
        and str(dimension.get("dimension") or "") == "dataset"
        and _normal(dimension.get("left_value")) != _normal(dimension.get("right_value"))
    )
    uncontrolled_metric_candidates = sum(
        1
        for candidate in candidates
        for dimension in list(candidate.get("dimensions") or [])
        if isinstance(dimension, dict)
        and str(dimension.get("dimension") or "") == "metric"
        and str(dimension.get("match_type") or "") != "controlled_alias"
    )
    return {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "fixture": str(fixture_path),
        "db_root": str(db_root),
        "corpus_chunk_count": len(chunks),
        "corpus_load_ms": corpus_load_ms,
        "candidate_scan": candidate_result,
        "cases": results,
        "summary": {
            "passed": sum(1 for item in results if item.get("passed")),
            "total": len(results),
            "discovered_candidate_count": len(candidates),
            "candidate_contract_failures": sum(
                1 for audit in audited_candidates if str(audit.get("status") or "") != "verified"
            ),
            "candidate_evidence_failures": candidate_evidence_failures,
            "candidate_prefill_failures": candidate_prefill_failures,
            "cross_dataset_candidates": cross_dataset_candidates,
            "uncontrolled_metric_candidates": uncontrolled_metric_candidates,
            "candidate_scan_ms": float((candidate_result.get("phase_timings_ms") or {}).get("total") or 0.0),
            "audit_median_ms": round(statistics.median(audit_times), 3) if audit_times else None,
            "audit_max_ms": round(max(audit_times), 3) if audit_times else None,
            "prefilled_contract_values_per_candidate": 12,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate evidence-bound comparison candidate discovery on reviewed real papers."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--db-root", type=Path, default=ROOT / "db")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "test_results" / "evidence_comparison_candidates",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    fixture = _load_fixture(args.fixture.resolve(strict=False))
    if args.dry_run:
        print(json.dumps({"ok": True, "cases": len(fixture["cases"]), "fixture": str(args.fixture)}, indent=2))
        return 0
    report = run_eval(
        fixture_path=args.fixture.resolve(strict=False),
        db_root=args.db_root.resolve(strict=False),
    )
    folder = args.out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report["summary"]}, ensure_ascii=False))
    summary = report["summary"]
    return 0 if (
        int(summary["passed"]) == int(summary["total"])
        and int(summary["candidate_contract_failures"]) == 0
        and int(summary["candidate_evidence_failures"]) == 0
        and int(summary["candidate_prefill_failures"]) == 0
        and int(summary["cross_dataset_candidates"]) == 0
        and int(summary["uncontrolled_metric_candidates"]) == 0
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
