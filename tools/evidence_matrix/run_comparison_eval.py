from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.evidence_matrix import COMPARISON_DIMENSIONS, audit_evidence_comparison
from kb.store import load_all_chunks


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version") or 0) != 1:
        raise ValueError("comparison eval fixture version must be 1")
    sources = payload.get("sources")
    cases = payload.get("cases")
    if not isinstance(sources, dict) or not isinstance(cases, list) or len(cases) < 5:
        raise ValueError("comparison eval fixture requires a source map and at least five cases")
    expected_dimensions = set(COMPARISON_DIMENSIONS)
    seen_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("comparison eval cases must be objects")
        case_id = str(case.get("id") or "").strip()
        if not case_id or case_id in seen_ids:
            raise ValueError("comparison eval case IDs must be unique and non-empty")
        seen_ids.add(case_id)
        if case.get("left_source") not in sources or case.get("right_source") not in sources:
            raise ValueError(f"{case_id}: unknown source key")
        spec = case.get("spec") if isinstance(case.get("spec"), dict) else {}
        dimensions = spec.get("dimensions") if isinstance(spec.get("dimensions"), list) else []
        if {str(item.get("dimension") or "") for item in dimensions if isinstance(item, dict)} != expected_dimensions:
            raise ValueError(f"{case_id}: all four comparison dimensions are required")
        if not isinstance(case.get("expected"), dict):
            raise ValueError(f"{case_id}: expected result is required")
    return payload


def _normal(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _source_key(value: object) -> str:
    try:
        return str(Path(str(value or "")).resolve(strict=False)).replace("\\", "/").casefold()
    except Exception:
        return str(value or "").replace("\\", "/").casefold()


def _rows(left_path: Path, right_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "id": "row-left",
            "source_item_key": "left",
            "paper": left_path.stem,
            "source_name": left_path.stem,
            "source_path": str(left_path),
            "source_status": "active",
            "cells": {},
        },
        {
            "id": "row-right",
            "source_item_key": "right",
            "paper": right_path.stem,
            "source_name": right_path.stem,
            "source_path": str(right_path),
            "source_status": "active",
            "cells": {},
        },
    ]


def _evidence_is_exact(
    audit: dict[str, Any],
    chunks_by_source: dict[str, list[dict[str, Any]]],
) -> tuple[bool, bool]:
    exact = True
    locatable = True
    for evidence in list(audit.get("evidence") or []):
        if not isinstance(evidence, dict):
            exact = False
            continue
        source_chunks = chunks_by_source.get(_source_key(evidence.get("source_path")), [])
        quote = _normal(evidence.get("evidence_quote"))
        if not quote or not any(quote in _normal(chunk.get("text")) for chunk in source_chunks):
            exact = False
        if not (
            evidence.get("page_start")
            or evidence.get("heading_path")
            or evidence.get("block_id")
            or evidence.get("anchor_id")
        ):
            locatable = False
    return exact, locatable


def run_eval(*, fixture_path: Path, db_root: Path) -> dict[str, Any]:
    fixture = _load_fixture(fixture_path)
    source_paths = {
        key: (db_root.parent / str(relative_path)).resolve(strict=False)
        if str(relative_path).replace("\\", "/").startswith("db/")
        else (db_root / str(relative_path)).resolve(strict=False)
        for key, relative_path in dict(fixture["sources"]).items()
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing reviewed source files: " + ", ".join(missing))
    load_started = time.perf_counter()
    chunks = [item for item in load_all_chunks(db_root) if isinstance(item, dict)]
    corpus_load_ms = round((time.perf_counter() - load_started) * 1000, 3)
    chunks_by_source: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
        chunks_by_source.setdefault(_source_key(meta.get("source_path")), []).append(chunk)

    results: list[dict[str, Any]] = []
    for case in fixture["cases"]:
        left_path = source_paths[str(case["left_source"])]
        right_path = source_paths[str(case["right_source"])]
        spec = dict(case["spec"])
        spec.update({"left_row_id": "row-left", "right_row_id": "row-right"})
        audit = audit_evidence_comparison(
            rows=_rows(left_path, right_path),
            spec=spec,
            db_dir=db_root,
            corpus_chunks=chunks,
        )
        expected = dict(case["expected"])
        checks = {
            "status": str(audit.get("status") or "") == str(expected.get("status") or ""),
            "relation": str(audit.get("relation") or "") == str(expected.get("relation") or ""),
            "preferred_side": str(audit.get("preferred_side") or "") == str(expected.get("preferred_side") or ""),
            "confirmed_conflict": bool(audit.get("confirmed_conflict")) is bool(expected.get("confirmed_conflict")),
        }
        required_reason = str(expected.get("required_reason") or "")
        if required_reason:
            checks["required_reason"] = required_reason in list(audit.get("reasons") or [])
        exact, locatable = _evidence_is_exact(audit, chunks_by_source)
        checks["same_source_exact_evidence"] = exact
        checks["reader_locator"] = locatable
        results.append(
            {
                "id": case["id"],
                "passed": all(checks.values()),
                "checks": checks,
                "actual": {
                    "status": audit.get("status"),
                    "relation": audit.get("relation"),
                    "preferred_side": audit.get("preferred_side"),
                    "confirmed_conflict": audit.get("confirmed_conflict"),
                    "reasons": audit.get("reasons"),
                    "evidence_count": len(list(audit.get("evidence") or [])),
                },
                "phase_timings_ms": audit.get("phase_timings_ms"),
            }
        )
    audit_times = [float((item.get("phase_timings_ms") or {}).get("total") or 0.0) for item in results]
    return {
        "fixture": str(fixture_path),
        "db_root": str(db_root),
        "corpus_chunk_count": len(chunks),
        "corpus_load_ms": corpus_load_ms,
        "cases": results,
        "summary": {
            "passed": sum(1 for item in results if item["passed"]),
            "total": len(results),
            "false_comparisons": sum(
                1
                for case, result in zip(fixture["cases"], results)
                if str(case["expected"].get("status") or "") == "not_comparable"
                and str(result["actual"].get("status") or "") == "verified"
            ),
            "audit_median_ms": round(statistics.median(audit_times), 3) if audit_times else 0.0,
            "audit_max_ms": round(max(audit_times), 3) if audit_times else 0.0,
            "cold_median_estimate_ms": round(corpus_load_ms + statistics.median(audit_times), 3) if audit_times else corpus_load_ms,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the reviewed evidence-comparison acceptance set.")
    parser.add_argument("--fixture", default="docs/evidence_comparison_eval_v1.json")
    parser.add_argument("--db-root", default="db")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    fixture_path = Path(args.fixture).resolve(strict=False)
    fixture = _load_fixture(fixture_path)
    if args.dry_run:
        print(json.dumps({"ok": True, "fixture": str(fixture_path), "cases": len(fixture["cases"])}, indent=2))
        return 0
    report = run_eval(fixture_path=fixture_path, db_root=Path(args.db_root).resolve(strict=False))
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output).resolve(strict=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if int(report["summary"]["passed"]) == int(report["summary"]["total"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
