from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.deps import get_settings
from kb.agent.tools import generate_grounded_answer, verify_answer_citations
from kb.evidence_matrix import MATRIX_CELL_FIELDS, evidence_matrix_hits, evidence_matrix_quality
from kb.research_brief import research_brief_evidence, research_brief_quality
from kb.research_brief_lineage import matrix_contract_fingerprint, research_brief_lineage
from kb.research_brief_update import (
    apply_research_brief_update_decisions,
    build_research_brief_update_plan,
    stable_matrix_hits,
)


DEFAULT_MATRIX_REPORT = ROOT / "test_results" / "evidence_matrix" / "20260806_012452" / "deterministic_report.json"
DEFAULT_FULL_BASELINE = ROOT / "test_results" / "evidence_matrix_brief_latency" / "20260806_022537" / "live_report.json"


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
    lines = ["## Evidence"]
    for number, hit in enumerate(hits, start=1):
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        source = str(meta.get("source_name") or meta.get("title") or "Source").strip()
        quote = re.sub(r"\[(?:\d+(?:\s*[-,]\s*\d+)*)\]", "", str(hit.get("text") or ""))
        quote = " ".join(quote.split()).rstrip(".!?。！？")
        lines.append(f"- {source}: {quote} [{number}].")
    content = "\n".join(lines)
    return {
        "id": f"brief-{matrix.get('id')}",
        "revision": 3,
        "title": f"Brief for {matrix.get('title')}",
        "objective": "Preserve reviewed content while updating changed evidence.",
        "content_markdown": content,
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


def _change_one_used_cell(current: dict[str, Any], brief: dict[str, Any]) -> tuple[str, str, int]:
    evidence_by_id = {
        str(item.get("id") or ""): item
        for item in list(current.get("evidence") or [])
        if isinstance(item, dict) and str(item.get("id") or "")
    }
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
            ),
            None,
        )
        if not isinstance(row, dict):
            continue
        cells = row.get("cells") if isinstance(row.get("cells"), dict) else {}
        cell = cells.get(field) if isinstance(cells.get(field), dict) else None
        if not isinstance(cell, dict):
            continue
        evidence_id = next((str(item) for item in list(cell.get("evidence_ids") or []) if str(item)), "")
        evidence = evidence_by_id.get(evidence_id)
        quote = " ".join(str((evidence or {}).get("evidence_quote") or "").split())
        old_value = " ".join(str(cell.get("value") or "").split())
        if not quote:
            continue
        new_value = quote
        if new_value.casefold() == old_value.casefold():
            sentences = [part.strip() for part in quote.replace("。", ".").split(".") if part.strip()]
            new_value = sentences[0] if sentences and sentences[0].casefold() != old_value.casefold() else quote[: max(24, len(quote) - 1)]
        if not new_value or new_value.casefold() == old_value.casefold():
            continue
        cell["value"] = new_value
        return str(row.get("id") or ""), field, int(citation.get("citation_number") or 0)
    raise RuntimeError("no brief-used grounded cell could be safely refreshed")


def _audit(content: str, hits: list[dict[str, Any]], selected_items: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    verified = verify_answer_citations(content, hits, answer_mode="evidence_grounded")
    verification = verified.get("verification") if isinstance(verified.get("verification"), dict) else {}
    trace = {
        "status": "done",
        "errors": [],
        "verification": verification,
        "summary": {
            "query_scope": "basket",
            "quality_gate_status": "passed",
            **{
                key: verification.get(key)
                for key in ("total_claims", "supported_claims", "unsupported_claims", "support_ratio", "evidence_status")
            },
        },
    }
    return research_brief_quality(
        answer=content,
        agent_trace=trace,
        selected_items=selected_items,
        evidence=research_brief_evidence(hits),
    )


def _evaluate_case(case: dict[str, Any], *, live: bool, settings: Any) -> dict[str, Any]:
    started = time.perf_counter()
    historical = _matrix_from_case(case)
    if historical.get("quality_status") != "verified":
        raise RuntimeError(f"{case.get('id')}: historical matrix is not verified")
    brief = _brief_for_matrix(historical)
    current = copy.deepcopy(historical)
    current["revision"] = int(historical.get("revision") or 1) + 1
    row_id, field, citation_number = _change_one_used_cell(current, brief)
    current["quality_status"], current["quality"] = evidence_matrix_quality(
        rows=list(current.get("rows") or []),
        evidence=list(current.get("evidence") or []),
        selected_items=list(current.get("source_items") or []),
        comparison_flags=list(current.get("comparison_flags") or []),
        comparison_audits=list(current.get("comparison_audits") or []),
    )
    if current.get("quality_status") != "verified":
        raise RuntimeError(f"{case.get('id')}: refreshed real matrix lost verification")
    lineage = research_brief_lineage(
        brief,
        current_matrix=current,
        historical_matrix=historical,
        include_impact=True,
    )
    impact = lineage.get("impact") if isinstance(lineage.get("impact"), dict) else {}
    plan = build_research_brief_update_plan(
        brief,
        historical_matrix=historical,
        current_matrix=current,
        impact=impact,
        locale="en",
        settings=settings,
        max_tokens=600,
        model_generator=generate_grounded_answer if live else None,
    )
    decisions = {str(item.get("id") or ""): "accept" for item in list(plan.get("items") or [])}
    merged = apply_research_brief_update_decisions(
        str(brief.get("content_markdown") or ""),
        list(plan.get("items") or []),
        decisions,
    )
    hits = stable_matrix_hits(list(brief.get("evidence") or []), current)
    quality_status, quality = _audit(
        str(merged.get("content_markdown") or ""),
        hits,
        [item for item in list(current.get("source_items") or []) if isinstance(item, dict)],
    )
    items = [item for item in list(plan.get("items") or []) if isinstance(item, dict)]
    base_content = str(brief.get("content_markdown") or "")
    if len(items) == 1:
        item = items[0]
        start = int(item.get("start") or 0)
        end = int(item.get("end") or 0)
        expected_content = f"{base_content[:start]}{item.get('proposed_markdown') or ''}{base_content[end:]}"
        exact_preservation = str(merged.get("content_markdown") or "") == expected_content
    else:
        exact_preservation = False
    passed = bool(
        lineage.get("status") == "matrix_updated"
        and citation_number in list(impact.get("affected_citation_numbers") or [])
        and len(items) == 1
        and quality_status == "verified"
        and not list(quality.get("reasons") or [])
        and merged.get("all_accepted") is True
        and exact_preservation
    )
    return {
        "id": str(case.get("id") or ""),
        "passed": passed,
        "changed_row_id": row_id,
        "changed_field": field,
        "affected_citation_number": citation_number,
        "change_item_count": len(items),
        "candidate_generation_mode": str((plan.get("generation") or {}).get("mode") or ""),
        "candidate_generation_ms": float((plan.get("generation") or {}).get("elapsed_ms") or 0.0),
        "candidate_generation_reason": str((plan.get("generation") or {}).get("reason") or ""),
        "plan_total_ms": float(plan.get("elapsed_ms") or 0.0),
        "unaffected_preservation_ratio": float((plan.get("preservation") or {}).get("unaffected_preservation_ratio") or 0.0),
        "unaffected_exactly_preserved": exact_preservation,
        "quality_status": quality_status,
        "supported_claims": int(quality.get("supported_claims") or 0),
        "total_claims": int(quality.get("total_claims") or 0),
        "reasons": list(quality.get("reasons") or []),
        "unresolved_citations": list(quality.get("unresolved_citations") or []),
        "evidence_hit_count": int(quality.get("evidence_hit_count") or 0),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def _full_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    timings = [
        float(((item.get("phase_timings_ms") or {}).get("total") or 0.0))
        for item in list(payload.get("cases") or [])
        if float(((item.get("phase_timings_ms") or {}).get("total") or 0.0)) > 0
    ]
    return {
        "report": str(path),
        "case_count": len(timings),
        "median_ms": round(statistics.median(timings), 2) if timings else 0.0,
        "max_ms": round(max(timings), 2) if timings else 0.0,
        "unaffected_manual_content_preserved": False,
        "reason": "legacy same-matrix regeneration replaced the complete content_markdown value",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate reviewable incremental brief updates on five real matrices.")
    parser.add_argument("--matrix-report", type=Path, default=DEFAULT_MATRIX_REPORT)
    parser.add_argument("--full-baseline", type=Path, default=DEFAULT_FULL_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "test_results" / "research_brief_incremental_update")
    parser.add_argument("--live", action="store_true", help="Use the configured paid model for focused candidate synthesis.")
    args = parser.parse_args()

    payload = json.loads(args.matrix_report.read_text(encoding="utf-8"))
    cases = [item for item in list(payload.get("cases") or []) if isinstance(item, dict)]
    settings = get_settings() if args.live else None
    results = [_evaluate_case(case, live=args.live, settings=settings) for case in cases]
    plan_times = [float(item.get("plan_total_ms") or 0.0) for item in results]
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": "live_model" if args.live else "deterministic_extractive",
        "matrix_report": str(args.matrix_report),
        "full_regeneration_baseline": _full_baseline(args.full_baseline),
        "total": len(results),
        "passed": sum(1 for item in results if item.get("passed")),
        "all_passed": bool(results) and all(bool(item.get("passed")) for item in results),
        "summary": {
            "median_plan_ms": round(statistics.median(plan_times), 2) if plan_times else 0.0,
            "max_plan_ms": round(max(plan_times), 2) if plan_times else 0.0,
            "minimum_unaffected_preservation_ratio": round(
                min((float(item.get("unaffected_preservation_ratio") or 0.0) for item in results), default=0.0),
                4,
            ),
            "verified_after_apply": sum(1 for item in results if item.get("quality_status") == "verified"),
            "model_synthesis": sum(1 for item in results if item.get("candidate_generation_mode") == "model_synthesis"),
        },
        "cases": results,
    }
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
