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
    apply_evidence_matrix_source_expansion,
    evidence_matrix_source_expansion_preview,
)
from kb.evidence_watch import source_identity
from kb.research_gap import build_project_research_gaps, find_research_gap_candidates
from kb.store import load_all_chunks


DEFAULT_FIXTURE = ROOT / "test_results" / "evidence_matrix" / "20260806_012452" / "deterministic_report.json"
TARGET_CASES = 5


def _normalized(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _write_report(root: Path, payload: dict[str, Any]) -> Path:
    folder = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _candidate_source_item(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": f"research-gap:{candidate.get('id') or ''}",
        "title": str(candidate.get("title") or candidate.get("source_name") or ""),
        "main": str(candidate.get("title") or candidate.get("source_name") or ""),
        "sourceName": str(candidate.get("source_name") or ""),
        "sourcePath": str(candidate.get("source_path") or ""),
        "shelfItemKind": "citation",
        "shelfOrigin": "research_gap",
        "shelfExcerpt": str(candidate.get("evidence_quote") or ""),
        "evidenceQuote": str(candidate.get("evidence_quote") or ""),
        "headingPath": str(candidate.get("heading_path") or ""),
        "locationLabel": str(candidate.get("location_label") or ""),
        "pageStart": candidate.get("page_start"),
        "pageEnd": candidate.get("page_end"),
        "blockId": str(candidate.get("block_id") or ""),
        "anchorId": str(candidate.get("anchor_id") or ""),
    }


def _candidate_is_exact(candidate: dict[str, Any], chunks_by_id: dict[str, dict[str, Any]]) -> bool:
    chunk = chunks_by_id.get(str(candidate.get("chunk_id") or ""), {})
    meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
    return bool(
        _normalized(candidate.get("evidence_quote"))
        and _normalized(candidate.get("evidence_quote")) in _normalized(chunk.get("text"))
        and source_identity(meta.get("source_path")) == source_identity(candidate.get("source_path"))
        and (
            str(candidate.get("anchor_id") or "")
            or str(candidate.get("block_id") or "")
            or str(candidate.get("heading_path") or "")
            or candidate.get("page_start") is not None
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate five real cross-paper research-gap expansions without cross-source attribution."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--db-root", type=Path, default=ROOT / "db")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "test_results" / "research_gap_expansion",
    )
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    fixture_cases = [item for item in list(fixture.get("cases") or []) if isinstance(item, dict)]
    chunks = [item for item in load_all_chunks(args.db_root) if isinstance(item, dict)]
    chunks_by_id = {str(item.get("id") or ""): item for item in chunks if str(item.get("id") or "")}
    results: list[dict[str, Any]] = []
    preview_times: list[float] = []
    apply_times: list[float] = []
    seen_gap_keys: set[str] = set()

    for case in fixture_cases:
        if len(results) >= TARGET_CASES:
            break
        case_id = str(case.get("id") or "")
        matrix = copy.deepcopy(dict(case.get("matrix") or {}))
        matrix.update(
            {
                "id": str(matrix.get("id") or f"matrix-expansion-{case_id}"),
                "project_id": f"expansion-eval-{case_id}",
                "objective": str(case.get("objective") or matrix.get("objective") or ""),
                "revision": int(matrix.get("revision") or 1),
                "quality": copy.deepcopy(dict(case.get("quality") or matrix.get("quality") or {})),
                "quality_status": str(matrix.get("quality_status") or "verified"),
                "source_items": copy.deepcopy(
                    list(case.get("selected_items") or matrix.get("source_items") or [])
                ),
                "comparison_audits": copy.deepcopy(list(matrix.get("comparison_audits") or [])),
            }
        )
        gaps = [
            item
            for item in build_project_research_gaps(
                project_id=str(matrix["project_id"]),
                matrices=[matrix],
            )
            if isinstance(item, dict)
            and str(item.get("kind") or "") in {"missing_cell", "unsupported_cell"}
            and bool(item.get("candidate_searchable"))
        ]
        excluded_sources = [
            str(item.get("source_path") or "")
            for item in list(matrix.get("rows") or [])
            if isinstance(item, dict)
        ]
        for gap in gaps:
            if len(results) >= TARGET_CASES:
                break
            gap_key = str(gap.get("gap_key") or "")
            if not gap_key or gap_key in seen_gap_keys:
                continue
            candidates = find_research_gap_candidates(
                gap,
                db_dir=args.db_root,
                excluded_source_paths=excluded_sources,
                limit=12,
                chunks=chunks,
            )
            selected: tuple[dict[str, Any], dict[str, Any], float] | None = None
            errors: list[str] = []
            for candidate in candidates:
                started = time.perf_counter()
                try:
                    preview = evidence_matrix_source_expansion_preview(
                        matrix,
                        gap,
                        candidate,
                        _candidate_source_item(candidate),
                        db_dir=args.db_root,
                        chunks=chunks,
                    )
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                preview_ms = round((time.perf_counter() - started) * 1_000, 3)
                selected = candidate, preview, preview_ms
                break
            if selected is None:
                continue
            candidate, preview, preview_ms = selected
            original_rows = copy.deepcopy(list(matrix.get("rows") or []))
            original_evidence = copy.deepcopy(list(matrix.get("evidence") or []))
            original_source_ids = {
                source_identity(item.get("source_path"))
                for item in original_rows
                if isinstance(item, dict)
            }
            started = time.perf_counter()
            applied = apply_evidence_matrix_source_expansion(
                matrix,
                gap,
                preview,
                db_dir=args.db_root,
            )
            apply_ms = round((time.perf_counter() - started) * 1_000, 3)
            expanded_matrix = {
                **matrix,
                "revision": int(matrix.get("revision") or 1) + 1,
                "rows": applied["rows"],
                "evidence": applied["evidence"],
                "source_items": applied["source_items"],
                "comparison_flags": applied["comparison_flags"],
                "comparison_audits": applied["comparison_audits"],
                "quality_status": applied["quality_status"],
                "quality": applied["quality"],
            }
            rescanned = build_project_research_gaps(
                project_id=str(matrix["project_id"]),
                matrices=[expanded_matrix],
            )
            original_gap_preserved = any(
                str(item.get("gap_key") or "") == gap_key for item in rescanned if isinstance(item, dict)
            )
            new_identity = source_identity(candidate.get("source_path"))
            new_evidence = [
                item
                for item in list(applied.get("evidence") or [])
                if isinstance(item, dict) and source_identity(item.get("source_path")) == new_identity
            ]
            exact_new_evidence = bool(new_evidence) and all(
                any(
                    source_identity((chunk.get("meta") or {}).get("source_path")) == new_identity
                    and _normalized(item.get("evidence_quote")) in _normalized(chunk.get("text"))
                    for chunk in chunks
                    if isinstance(chunk.get("meta"), dict)
                )
                for item in new_evidence
            )
            passed = bool(
                _candidate_is_exact(candidate, chunks_by_id)
                and new_identity not in original_source_ids
                and applied["rows"][: len(original_rows)] == original_rows
                and applied["evidence"][: len(original_evidence)] == original_evidence
                and source_identity(preview["row"].get("source_path")) == new_identity
                and exact_new_evidence
                and int(applied["quality"].get("unsupported_cell_count") or 0) == 0
                and len(applied["rows"]) == len(original_rows) + 1
                and original_gap_preserved
            )
            results.append(
                {
                    "id": f"{case_id}:{gap.get('row_id')}:{gap.get('field')}",
                    "matrix_case": case_id,
                    "gap_key": gap_key,
                    "field": str(gap.get("field") or ""),
                    "candidate_id": str(candidate.get("id") or ""),
                    "candidate_source_path": str(candidate.get("source_path") or ""),
                    "candidate_exact_with_locator": _candidate_is_exact(candidate, chunks_by_id),
                    "candidate_outside_matrix": new_identity not in original_source_ids,
                    "grounded_fields": list(preview.get("grounded_fields") or []),
                    "missing_fields": list(preview.get("missing_fields") or []),
                    "old_rows_preserved": applied["rows"][: len(original_rows)] == original_rows,
                    "old_evidence_preserved": applied["evidence"][: len(original_evidence)] == original_evidence,
                    "new_evidence_exact_same_source": exact_new_evidence,
                    "original_gap_preserved": original_gap_preserved,
                    "preview_ms": preview_ms,
                    "apply_and_reaudit_ms": apply_ms,
                    "quality_status": applied["quality_status"],
                    "passed": passed,
                    "candidate_rejections_before_selection": errors,
                }
            )
            seen_gap_keys.add(gap_key)
            preview_times.append(preview_ms)
            apply_times.append(apply_ms)

    while len(results) < TARGET_CASES:
        results.append(
            {
                "id": f"missing-real-gap-{len(results) + 1}",
                "passed": False,
                "error": "no additional reviewed real gap produced a verified external-source row",
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
            "exact_cross_source_candidates": sum(
                1 for item in results if item.get("candidate_exact_with_locator")
            ),
            "old_rows_preserved": sum(1 for item in results if item.get("old_rows_preserved")),
            "original_gaps_preserved": sum(1 for item in results if item.get("original_gap_preserved")),
            "preview_median_ms": round(statistics.median(preview_times), 3) if preview_times else None,
            "preview_max_ms": round(max(preview_times), 3) if preview_times else None,
            "apply_and_reaudit_median_ms": round(statistics.median(apply_times), 3) if apply_times else None,
            "apply_and_reaudit_max_ms": round(max(apply_times), 3) if apply_times else None,
        },
    }
    report_path = _write_report(args.out_root, payload)
    print(json.dumps({"report": str(report_path), **payload["summary"]}, ensure_ascii=False))
    return 0 if payload["summary"]["passed"] == TARGET_CASES else 1


if __name__ == "__main__":
    raise SystemExit(main())
