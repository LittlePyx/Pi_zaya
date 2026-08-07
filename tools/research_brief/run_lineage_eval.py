from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.evidence_matrix import MATRIX_CELL_FIELDS, evidence_matrix_hits, evidence_matrix_quality
from kb.research_brief import research_brief_evidence
from kb.research_brief_lineage import matrix_contract_fingerprint, research_brief_lineage


DEFAULT_MATRIX_REPORT = ROOT / "test_results" / "evidence_matrix" / "20260806_012452" / "deterministic_report.json"


def _source_identity(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").casefold()


def _matrix_from_case(case: dict[str, Any]) -> dict[str, Any]:
    matrix = copy.deepcopy(case.get("matrix") or {})
    matrix["title"] = str(case.get("id") or matrix.get("id") or "Evidence matrix")
    matrix["source_items"] = copy.deepcopy(case.get("selected_items") or [])
    matrix["comparison_audits"] = list(matrix.get("comparison_audits") or [])
    matrix["quality_status"], matrix["quality"] = evidence_matrix_quality(
        rows=list(matrix.get("rows") or []),
        evidence=list(matrix.get("evidence") or []),
        selected_items=list(matrix.get("source_items") or []),
        comparison_flags=list(matrix.get("comparison_flags") or []),
        comparison_audits=list(matrix.get("comparison_audits") or []),
    )
    return matrix


def _brief_for_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    hits = evidence_matrix_hits(matrix, limit=20)
    return {
        "id": f"brief-{matrix.get('id')}",
        "quality_status": "verified",
        "quality": {
            "source_matrix_id": str(matrix.get("id") or ""),
            "source_matrix_title": str(matrix.get("title") or ""),
            "source_matrix_revision": int(matrix.get("revision") or 1),
            "source_matrix_quality_status": str(matrix.get("quality_status") or ""),
            "source_matrix_fingerprint": matrix_contract_fingerprint(matrix),
        },
        "evidence": research_brief_evidence(hits),
    }


def _grounded_cell_count(row: dict[str, Any]) -> int:
    cells = row.get("cells") if isinstance(row.get("cells"), dict) else {}
    return sum(
        1
        for field in MATRIX_CELL_FIELDS
        if isinstance(cells.get(field), dict)
        and str(cells[field].get("value") or "").strip()
        and str(cells[field].get("support_status") or "") == "grounded"
    )


def _remove_one_used_cell(
    current: dict[str, Any],
    brief: dict[str, Any],
) -> tuple[str, str, int]:
    rows = [row for row in list(current.get("rows") or []) if isinstance(row, dict)]
    for citation in list(brief.get("evidence") or []):
        if not isinstance(citation, dict):
            continue
        field = str(citation.get("matrix_field") or "")
        source = _source_identity(citation.get("source_path") or citation.get("source_name"))
        if field not in MATRIX_CELL_FIELDS or not source:
            continue
        row = next(
            (
                item
                for item in rows
                if _source_identity(item.get("source_path") or item.get("source_name")) == source
                and _grounded_cell_count(item) > 1
            ),
            None,
        )
        if not isinstance(row, dict):
            continue
        cells = row.get("cells") if isinstance(row.get("cells"), dict) else {}
        cell = cells.get(field) if isinstance(cells.get(field), dict) else None
        if not isinstance(cell, dict) or not str(cell.get("value") or "").strip():
            continue
        cell.update(
            {
                "value": "",
                "support_status": "missing",
                "evidence_ids": [],
                "manual_override": False,
            }
        )
        return str(row.get("id") or ""), field, int(citation.get("citation_number") or 0)
    raise RuntimeError("no source-balanced grounded cell used by the brief could be removed safely")


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    historical = _matrix_from_case(case)
    if historical.get("quality_status") != "verified":
        raise RuntimeError(f"{case.get('id')}: historical real matrix is not verified")
    brief = _brief_for_matrix(historical)
    current = copy.deepcopy(historical)
    current["revision"] = int(historical.get("revision") or 1) + 1
    row_id, field, citation_number = _remove_one_used_cell(current, brief)
    current["quality_status"], current["quality"] = evidence_matrix_quality(
        rows=list(current.get("rows") or []),
        evidence=list(current.get("evidence") or []),
        selected_items=list(current.get("source_items") or []),
        comparison_flags=list(current.get("comparison_flags") or []),
        comparison_audits=list(current.get("comparison_audits") or []),
    )
    if current.get("quality_status") != "verified":
        raise RuntimeError(f"{case.get('id')}: honest missing-cell refresh unexpectedly lost verification")

    changed = research_brief_lineage(
        brief,
        current_matrix=current,
        historical_matrix=historical,
        include_impact=True,
    )
    equivalent = copy.deepcopy(historical)
    equivalent["revision"] = int(current.get("revision") or 2) + 1
    equivalent_lineage = research_brief_lineage(
        brief,
        current_matrix=equivalent,
        historical_matrix=historical,
        include_impact=True,
    )
    missing_lineage = research_brief_lineage(brief, current_matrix=None)
    impact = changed.get("impact") if isinstance(changed.get("impact"), dict) else {}
    passed = bool(
        changed.get("status") == "matrix_updated"
        and changed.get("historical_verified") is True
        and changed.get("export_allowed") is True
        and citation_number in list(impact.get("affected_citation_numbers") or [])
        and equivalent_lineage.get("status") == "current_equivalent"
        and equivalent_lineage.get("latest_verified") is True
        and missing_lineage.get("status") == "matrix_missing"
        and missing_lineage.get("export_allowed") is False
    )
    return {
        "id": str(case.get("id") or ""),
        "passed": passed,
        "historical_supported_cells": int((historical.get("quality") or {}).get("supported_cell_count") or 0),
        "current_supported_cells": int((current.get("quality") or {}).get("supported_cell_count") or 0),
        "removed_row_id": row_id,
        "removed_field": field,
        "affected_citation_number": citation_number,
        "lineage_status": changed.get("status"),
        "changed_field_count": impact.get("changed_field_count"),
        "affected_citation_numbers": impact.get("affected_citation_numbers"),
        "equivalent_status": equivalent_lineage.get("status"),
        "missing_status": missing_lineage.get("status"),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay research-brief lineage against five real evidence matrices.")
    parser.add_argument("--matrix-report", type=Path, default=DEFAULT_MATRIX_REPORT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "test_results" / "research_brief_lineage")
    args = parser.parse_args()

    payload = json.loads(args.matrix_report.read_text(encoding="utf-8"))
    cases = [item for item in list(payload.get("cases") or []) if isinstance(item, dict)]
    results = [_evaluate_case(case) for case in cases]
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "matrix_report": str(args.matrix_report),
        "total": len(results),
        "passed": sum(1 for item in results if item.get("passed")),
        "all_passed": bool(results) and all(bool(item.get("passed")) for item in results),
        "cases": results,
    }
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output": str(output_path)}, ensure_ascii=False, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
