from __future__ import annotations

import argparse
import copy
import json
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
    MATRIX_CELL_FIELDS,
    apply_evidence_matrix_cell_repair,
    evidence_matrix_cell_repair_candidates,
    evidence_matrix_quality,
)
from kb.evidence_watch import source_identity
from kb.research_gap import build_project_research_gaps
from kb.store import load_all_chunks


DEFAULT_FIXTURE = ROOT / "test_results" / "evidence_matrix" / "20260806_012452" / "deterministic_report.json"


def _normalized(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _write_report(root: Path, payload: dict[str, Any]) -> Path:
    folder = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _missing_identities(matrix: dict[str, Any]) -> set[tuple[str, str]]:
    quality = matrix.get("quality") if isinstance(matrix.get("quality"), dict) else {}
    return {
        (str(item.get("row_id") or ""), str(item.get("field") or ""))
        for item in list(quality.get("missing_cells") or [])
        if isinstance(item, dict)
    }


def _gap(matrix: dict[str, Any], row: dict[str, Any], field: str, *, suffix: str) -> dict[str, Any]:
    return {
        "id": f"repair-eval-{suffix}",
        "gap_key": f"repair-eval-key-{suffix}",
        "project_id": str(matrix.get("project_id") or ""),
        "kind": "missing_cell",
        "matrix_id": str(matrix.get("id") or ""),
        "matrix_revision": int(matrix.get("revision") or 1),
        "row_id": str(row.get("id") or ""),
        "field": field,
        "source_path": str(row.get("source_path") or ""),
    }


def _holdout(
    matrix: dict[str, Any],
    *,
    row_id: str,
    field: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    held = copy.deepcopy(matrix)
    row = next(item for item in held["rows"] if str(item.get("id") or "") == row_id)
    cell = dict(row["cells"][field])
    expected = str(cell.get("value") or "")
    removed_ids = {str(item or "") for item in list(cell.get("evidence_ids") or []) if str(item or "")}
    row["cells"][field] = {
        "field": field,
        "value": "",
        "support_status": "missing",
        "evidence_ids": [],
        "manual_override": False,
    }
    held["evidence"] = [
        item
        for item in list(held.get("evidence") or [])
        if isinstance(item, dict) and str(item.get("id") or "") not in removed_ids
    ]
    status, quality = evidence_matrix_quality(
        rows=held["rows"],
        evidence=held["evidence"],
        selected_items=held["source_items"],
        comparison_flags=held.get("comparison_flags"),
        comparison_audits=held.get("comparison_audits"),
    )
    held["quality_status"] = status
    held["quality"] = quality
    return held, row, expected


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay strict same-source research-gap repairs on five real matrices.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--db-root", type=Path, default=ROOT / "db")
    parser.add_argument("--out-root", type=Path, default=ROOT / "test_results" / "research_gap_repair")
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    cases = [item for item in list(fixture.get("cases") or []) if isinstance(item, dict)]
    if len(cases) != 5:
        raise RuntimeError(f"reviewed fixture must contain five real matrices, got {len(cases)}")
    chunks = [item for item in load_all_chunks(args.db_root) if isinstance(item, dict)]
    chunks_by_id = {str(item.get("id") or ""): item for item in chunks if str(item.get("id") or "")}

    results: list[dict[str, Any]] = []
    search_times: list[float] = []
    apply_times: list[float] = []
    honest_missing_before = 0
    honest_missing_after = 0
    for case in cases:
        case_id = str(case.get("id") or "")
        matrix = copy.deepcopy(dict(case.get("matrix") or {}))
        matrix.update(
            {
                "project_id": f"repair-eval-{case_id}",
                "objective": str(case.get("objective") or matrix.get("objective") or ""),
                "revision": int(matrix.get("revision") or 1),
                "quality": copy.deepcopy(dict(case.get("quality") or matrix.get("quality") or {})),
                "quality_status": str(matrix.get("quality_status") or "verified"),
                "source_items": copy.deepcopy(list(case.get("selected_items") or matrix.get("source_items") or [])),
                "comparison_audits": copy.deepcopy(list(matrix.get("comparison_audits") or [])),
            }
        )
        before_missing = _missing_identities(matrix)
        honest_missing_before += len(before_missing)
        # Searching honest gaps must not mutate the reviewed matrix or clear any
        # missing identity without a human-confirmed application.
        original_snapshot = json.dumps(matrix, ensure_ascii=False, sort_keys=True)
        for row_id, field in before_missing:
            row = next((item for item in matrix.get("rows", []) if str(item.get("id") or "") == row_id), None)
            if isinstance(row, dict):
                evidence_matrix_cell_repair_candidates(
                    matrix,
                    _gap(matrix, row, field, suffix=f"{case_id}-honest-{row_id}-{field}"),
                    db_dir=args.db_root,
                    limit=3,
                    chunks=chunks,
                )
        unchanged_after_search = json.dumps(matrix, ensure_ascii=False, sort_keys=True) == original_snapshot
        gaps_after_search = build_project_research_gaps(
            project_id=str(matrix["project_id"]),
            matrices=[matrix],
        )
        observed_after = {
            (str(item.get("row_id") or ""), str(item.get("field") or ""))
            for item in gaps_after_search
            if str(item.get("kind") or "") == "missing_cell"
        }
        honest_missing_after += len(observed_after)

        selected: dict[str, Any] | None = None
        for row in [item for item in list(matrix.get("rows") or []) if isinstance(item, dict)]:
            cells = row.get("cells") if isinstance(row.get("cells"), dict) else {}
            for field in MATRIX_CELL_FIELDS:
                cell = cells.get(field) if isinstance(cells.get(field), dict) else {}
                if str(cell.get("support_status") or "") != "grounded" or not str(cell.get("value") or ""):
                    continue
                held, held_row, expected = _holdout(matrix, row_id=str(row.get("id") or ""), field=field)
                gap = _gap(held, held_row, field, suffix=f"{case_id}-{field}")
                started = time.perf_counter()
                candidates = evidence_matrix_cell_repair_candidates(
                    held,
                    gap,
                    db_dir=args.db_root,
                    limit=8,
                    chunks=chunks,
                )
                search_ms = round((time.perf_counter() - started) * 1_000, 3)
                exact = next(
                    (
                        item
                        for item in candidates
                        if _normalized(item.get("value")) == _normalized(expected)
                        and source_identity(item.get("source_path")) == source_identity(held_row.get("source_path"))
                    ),
                    None,
                )
                if isinstance(exact, dict):
                    selected = {
                        "held": held,
                        "row": held_row,
                        "field": field,
                        "expected": expected,
                        "gap": gap,
                        "candidate": exact,
                        "candidate_count": len(candidates),
                        "search_ms": search_ms,
                    }
                    break
            if selected:
                break
        if not selected:
            results.append(
                {
                    "id": case_id,
                    "passed": False,
                    "error": "no reviewed grounded cell was recoverable from its exact same-source passage",
                    "honest_missing_preserved": unchanged_after_search and observed_after == before_missing,
                }
            )
            continue

        repair = selected["candidate"]
        started = time.perf_counter()
        applied = apply_evidence_matrix_cell_repair(
            selected["held"],
            selected["gap"],
            repair,
            db_dir=args.db_root,
        )
        apply_ms = round((time.perf_counter() - started) * 1_000, 3)
        search_times.append(float(selected["search_ms"]))
        apply_times.append(apply_ms)
        row = next(item for item in applied["rows"] if str(item.get("id") or "") == str(selected["row"].get("id") or ""))
        cell = row["cells"][selected["field"]]
        chunk = chunks_by_id.get(str(repair.get("chunk_id") or ""), {})
        exact_locator = bool(
            _normalized(repair.get("evidence_quote"))
            and _normalized(repair.get("evidence_quote")) in _normalized(chunk.get("text"))
            and source_identity((chunk.get("meta") or {}).get("source_path")) == source_identity(repair.get("source_path"))
            and (
                str(repair.get("anchor_id") or "")
                or str(repair.get("block_id") or "")
                or str(repair.get("heading_path") or "")
                or repair.get("page_start") is not None
            )
        )
        passed = bool(
            unchanged_after_search
            and observed_after == before_missing
            and exact_locator
            and _normalized(cell.get("value")) == _normalized(selected["expected"])
            and str(cell.get("support_status") or "") == "grounded"
            and not bool(cell.get("manual_override"))
            and int(applied["quality"].get("unsupported_cell_count") or 0) == 0
        )
        results.append(
            {
                "id": case_id,
                "row_id": str(selected["row"].get("id") or ""),
                "field": selected["field"],
                "source_path": str(repair.get("source_path") or ""),
                "chunk_id": str(repair.get("chunk_id") or ""),
                "candidate_count": int(selected["candidate_count"]),
                "exact_same_source_with_locator": exact_locator,
                "recovered_original_value": _normalized(cell.get("value")) == _normalized(selected["expected"]),
                "honest_missing_count": len(before_missing),
                "honest_missing_preserved": unchanged_after_search and observed_after == before_missing,
                "search_ms": selected["search_ms"],
                "apply_and_reaudit_ms": apply_ms,
                "quality_status": applied["quality_status"],
                "passed": passed,
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "fixture": str(args.fixture.resolve()),
        "db_root": str(args.db_root.resolve()),
        "indexed_chunk_count": len(chunks),
        "cases": results,
        "summary": {
            "passed": sum(1 for item in results if item.get("passed")),
            "total": len(results),
            "exact_same_source_repairs": sum(1 for item in results if item.get("exact_same_source_with_locator")),
            "honest_missing_before": honest_missing_before,
            "honest_missing_after_search": honest_missing_after,
            "search_median_ms": round(statistics.median(search_times), 3) if search_times else None,
            "search_max_ms": round(max(search_times), 3) if search_times else None,
            "apply_and_reaudit_median_ms": round(statistics.median(apply_times), 3) if apply_times else None,
            "apply_and_reaudit_max_ms": round(max(apply_times), 3) if apply_times else None,
        },
    }
    report_path = _write_report(args.out_root, payload)
    print(json.dumps({"report": str(report_path), **payload["summary"]}, ensure_ascii=False))
    return 0 if payload["summary"]["passed"] == payload["summary"]["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
