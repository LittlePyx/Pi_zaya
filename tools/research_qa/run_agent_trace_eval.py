from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.agent.runner import build_agent_trace_for_completed_answer
from kb.agent.schema import validate_agent_trace
from kb.agent.verifier import verify_answer_citations
from tools.research_qa.validate_research_agent_golden import DEFAULT_DATASET, load_cases, validate_case

DEFAULT_QUALITY_DATASET = Path("docs/research_agent_eval_v1.jsonl")
QUALITY_REQUIRED_FIELDS = {
    "id",
    "query",
    "answer",
    "evidence_hits",
    "expected_retrieval_hit",
    "should_use_local_evidence",
    "external_fallback_allowed",
    "expected_answer_points",
    "expected_user_notice",
}
DEFAULT_FORBIDDEN_ANSWER_TERMS = [
    "Research Agent Trace",
    "agent_trace",
    "tool calls",
    "verification statistics",
]


def _synthetic_hit(case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case.get("id") or "synthetic")
    query = str(case.get("query") or "")
    return {
        "text": f"{query} Synthetic evidence snippet for trace validation.",
        "score": 1.0,
        "meta": {
            "source_name": f"{case_id}.md",
            "source_path": f"synthetic/{case_id}.md",
            "heading_path": "Trace Eval",
        },
    }


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    cases: list[dict[str, Any]] = []
    for line_no, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{target}:{line_no}: case must be an object")
        item["_line_no"] = line_no
        cases.append(item)
    return cases


def load_quality_cases(path: str | Path = DEFAULT_QUALITY_DATASET) -> list[dict[str, Any]]:
    return _load_jsonl(path)


def evaluate_cases(path: str | Path = DEFAULT_DATASET) -> dict[str, Any]:
    cases = load_cases(path)
    planning_errors: list[str] = []
    schema_errors: list[str] = []
    question_types: dict[str, int] = {}
    scope_context_present = 0

    for case in cases:
        case_id = str(case.get("id") or f"line:{case.get('_line_no')}")
        planning_errors.extend(validate_case(case))
        query = str(case.get("query") or "")
        trace = build_agent_trace_for_completed_answer(
            query,
            f"Synthetic grounded answer for {case_id} [1].",
            evidence_hits=[_synthetic_hit(case)],
            scope_context={"query_scope": "library", "scope_source": "agent_trace_eval"},
        )
        validation = validate_agent_trace(trace)
        if not bool(validation.get("ok")):
            for error in list(validation.get("errors") or []):
                schema_errors.append(f"{case_id}: {error}")
        if isinstance(trace.get("context"), dict) and trace["context"]:
            scope_context_present += 1
        qtype = str(trace.get("question_type") or "unknown")
        question_types[qtype] = question_types.get(qtype, 0) + 1

    return {
        "ok": not planning_errors and not schema_errors,
        "case_count": len(cases),
        "question_types": question_types,
        "scope_context_present": scope_context_present,
        "schema_errors": schema_errors,
        "planning_errors": planning_errors,
        "metrics_note": "Schema and planner checks only; no quality scores or fabricated answer metrics.",
    }


def _norm(value: object) -> str:
    return " ".join(str(value or "").replace("\\", "/").lower().split())


def _contains(haystack: object, needle: object) -> bool:
    term = _norm(needle)
    return bool(term and term in _norm(haystack))


def _payload_text(value: object) -> str:
    if isinstance(value, dict):
        return " ".join([str(k) for k in value.keys()] + [_payload_text(v) for v in value.values()])
    if isinstance(value, list):
        return " ".join(_payload_text(item) for item in value)
    return str(value or "")


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(max(0.0, min(1.0, numerator / denominator)), 4)


def _answer_has_notice(answer: str, notice_type: str) -> bool:
    text = _norm(answer)
    notice = str(notice_type or "none").strip()
    if notice == "none":
        return True
    if notice == "hybrid_notice":
        return "local citations" in text and "knowledge base" in text and "external model" in text
    if notice == "external_not_local":
        return (
            "no matching local knowledge-base evidence" in text
            and "external model answer" in text
            and "not a knowledge-base-grounded answer" in text
        )
    if notice == "insufficient_local_evidence":
        return (
            "does not contain enough evidence" in text
            or "no supporting local snippets" in text
            or "no matching local knowledge-base evidence" in text
        )
    return False


def _trace_clutter_free(answer: str, case: dict[str, Any]) -> bool:
    terms = [
        str(term or "")
        for term in list(case.get("forbidden_answer_terms") or DEFAULT_FORBIDDEN_ANSWER_TERMS)
        if str(term or "").strip()
    ]
    return not any(_contains(answer, term) for term in terms)


def _validate_quality_case(case: dict[str, Any]) -> list[str]:
    case_id = str(case.get("id") or f"line:{case.get('_line_no')}")
    errors: list[str] = []
    missing = sorted(QUALITY_REQUIRED_FIELDS - set(case))
    if missing:
        errors.append(f"{case_id}: missing fields: {', '.join(missing)}")
    if not isinstance(case.get("evidence_hits", []), list):
        errors.append(f"{case_id}: evidence_hits must be a list")
    if not isinstance(case.get("expected_answer_points", []), list):
        errors.append(f"{case_id}: expected_answer_points must be a list")
    return errors


def evaluate_quality_cases(path: str | Path = DEFAULT_QUALITY_DATASET) -> dict[str, Any]:
    cases = load_quality_cases(path)
    errors: list[str] = []
    case_results: list[dict[str, Any]] = []
    retrieval_expected_total = 0
    retrieval_hit_count = 0
    expected_source_total = 0
    expected_source_hit_count = 0
    answer_point_total = 0
    answer_point_hit_count = 0
    local_claim_total = 0
    supported_claim_total = 0
    unsupported_claim_total = 0
    cited_local_claim_total = 0
    supported_cited_claim_total = 0
    no_evidence_notice_total = 0
    no_evidence_notice_ok = 0
    fallback_notice_total = 0
    fallback_notice_ok = 0
    trace_clutter_total = 0
    trace_clutter_ok = 0

    for case in cases:
        case_id = str(case.get("id") or f"line:{case.get('_line_no')}")
        errors.extend(_validate_quality_case(case))
        answer = str(case.get("answer") or "")
        hits = [hit for hit in list(case.get("evidence_hits") or []) if isinstance(hit, dict)]
        answer_mode = str(case.get("answer_mode") or "").strip()
        expected_retrieval_hit = bool(case.get("expected_retrieval_hit"))
        should_use_local = bool(case.get("should_use_local_evidence"))
        notice_type = str(case.get("expected_user_notice") or "none").strip()

        retrieval_hit_ok = bool(hits) == expected_retrieval_hit
        if expected_retrieval_hit:
            retrieval_expected_total += 1
            if hits:
                retrieval_hit_count += 1
            else:
                errors.append(f"{case_id}: expected retrieved evidence but evidence_hits is empty")

        evidence_payload = _payload_text(hits)
        expected_sources = [str(item or "") for item in list(case.get("expected_source_keywords") or []) if str(item or "").strip()]
        source_hit_ok = True
        if expected_sources:
            expected_source_total += 1
            source_hit_ok = all(_contains(evidence_payload, item) for item in expected_sources)
            if source_hit_ok:
                expected_source_hit_count += 1
            else:
                errors.append(f"{case_id}: expected source keywords not found in evidence hits")

        expected_points = [str(item or "") for item in list(case.get("expected_answer_points") or []) if str(item or "").strip()]
        point_hits = [point for point in expected_points if _contains(answer, point)]
        answer_point_total += len(expected_points)
        answer_point_hit_count += len(point_hits)
        if len(point_hits) < len(expected_points):
            missing_points = [point for point in expected_points if point not in point_hits]
            errors.append(f"{case_id}: missing expected answer points: {', '.join(missing_points)}")

        verification = verify_answer_citations(answer, hits, answer_mode=answer_mode)
        if should_use_local:
            local_claim_total += int(verification.total_claims or 0)
            supported_claim_total += int(verification.supported_claims or 0)
            unsupported_claim_total += int(verification.unsupported_claims or 0)
            if int(verification.supported_claims or 0) <= 0:
                errors.append(f"{case_id}: expected at least one locally supported claim")
            for row in list(verification.claims or []):
                if not isinstance(row, dict) or row.get("verification_scope") != "local_evidence":
                    continue
                if row.get("citation_present"):
                    cited_local_claim_total += 1
                    if row.get("supported") is True:
                        supported_cited_claim_total += 1

        notice_ok = _answer_has_notice(answer, notice_type)
        if notice_type in {"external_not_local", "insufficient_local_evidence"}:
            no_evidence_notice_total += 1
            if notice_ok:
                no_evidence_notice_ok += 1
            else:
                errors.append(f"{case_id}: expected user notice {notice_type!r} was not found")
        elif notice_type != "none" and not notice_ok:
            errors.append(f"{case_id}: expected user notice {notice_type!r} was not found")

        fallback_notice_required = bool(case.get("external_fallback_allowed")) and answer_mode in {
            "external_academic_llm",
            "hybrid_local_external",
        }
        if fallback_notice_required:
            fallback_notice_total += 1
            if notice_ok:
                fallback_notice_ok += 1
            else:
                errors.append(f"{case_id}: external/hybrid fallback did not disclose its source mode")

        clutter_free = _trace_clutter_free(answer, case)
        trace_clutter_total += 1
        if clutter_free:
            trace_clutter_ok += 1
        else:
            errors.append(f"{case_id}: answer includes trace/tool/debug clutter")

        case_results.append(
            {
                "id": case_id,
                "retrieval_hit_ok": retrieval_hit_ok,
                "expected_source_hit_ok": source_hit_ok,
                "expected_answer_point_hits": len(point_hits),
                "expected_answer_point_total": len(expected_points),
                "notice_ok": notice_ok,
                "trace_clutter_free": clutter_free,
                "local_evidence_evaluated": should_use_local,
                "supported_claims": int(verification.supported_claims or 0) if should_use_local else None,
                "unsupported_claims": int(verification.unsupported_claims or 0) if should_use_local else None,
                "evidence_status": verification.evidence_status if should_use_local else "not_applicable",
            }
        )

    citation_precision = _ratio(supported_cited_claim_total, cited_local_claim_total)
    claim_support_rate = _ratio(supported_claim_total, local_claim_total)
    unsupported_claim_rate = _ratio(unsupported_claim_total, local_claim_total)
    return {
        "ok": not errors,
        "case_count": len(cases),
        "errors": errors,
        "retrieval_hit_rate": _ratio(retrieval_hit_count, retrieval_expected_total),
        "expected_source_hit_rate": _ratio(expected_source_hit_count, expected_source_total),
        "expected_answer_point_coverage": _ratio(answer_point_hit_count, answer_point_total),
        "citation_precision": citation_precision,
        "claim_support_rate": claim_support_rate,
        "unsupported_claim_rate": unsupported_claim_rate,
        "no_evidence_refusal_accuracy": _ratio(no_evidence_notice_ok, no_evidence_notice_total),
        "external_fallback_disclosure_accuracy": _ratio(fallback_notice_ok, fallback_notice_total),
        "trace_clutter_free_rate": _ratio(trace_clutter_ok, trace_clutter_total),
        "local_claim_count": local_claim_total,
        "supported_claim_count": supported_claim_total,
        "unsupported_claim_count": unsupported_claim_total,
        "metrics_note": (
            "Fixture-based answer quality checks over recorded cases; not a live model benchmark."
        ),
        "cases": case_results,
    }


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def build_eval_report(
    summary: dict[str, Any],
    *,
    quality_summary: dict[str, Any] | None = None,
    commit: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    quality = quality_summary if isinstance(quality_summary, dict) else {}
    retrieval_recall_at_5 = quality.get("expected_source_hit_rate")
    citation_precision = quality.get("citation_precision")
    claim_support_rate = quality.get("claim_support_rate")
    unsupported_claim_rate = quality.get("unsupported_claim_rate")
    no_evidence_refusal_accuracy = quality.get("no_evidence_refusal_accuracy")
    if not quality:
        retrieval_recall_at_5 = None
        citation_precision = None
        claim_support_rate = None
        unsupported_claim_rate = None
        no_evidence_refusal_accuracy = None
    return {
        "commit": str(commit if commit is not None else _git_commit()),
        "date": str(date or datetime.now(timezone.utc).isoformat()),
        "num_cases": int(summary.get("case_count") or 0),
        "num_quality_cases": int(quality.get("case_count") or 0),
        "planner_validation_ok": bool(summary.get("ok")),
        "quality_eval_ok": bool(quality.get("ok")) if quality else None,
        "planner_error_count": len(list(summary.get("planning_errors") or [])),
        "trace_schema_error_count": len(list(summary.get("schema_errors") or [])),
        "quality_error_count": len(list(quality.get("errors") or [])) if quality else 0,
        "question_types": dict(summary.get("question_types") or {}),
        "retrieval_recall_at_5": retrieval_recall_at_5,
        "retrieval_hit_rate": quality.get("retrieval_hit_rate") if quality else None,
        "expected_answer_point_coverage": quality.get("expected_answer_point_coverage") if quality else None,
        "citation_precision": citation_precision,
        "claim_support_rate": claim_support_rate,
        "unsupported_claim_rate": unsupported_claim_rate,
        "no_evidence_refusal_accuracy": no_evidence_refusal_accuracy,
        "external_fallback_disclosure_accuracy": quality.get("external_fallback_disclosure_accuracy") if quality else None,
        "trace_clutter_free_rate": quality.get("trace_clutter_free_rate") if quality else None,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "cost_per_query_usd": None,
        "notes": (
            "Quality metrics are fixture-based over recorded eval cases; latency and cost remain null until live runs are instrumented."
            if quality
            else "Quality metrics are null until the eval suite is run on a labeled dataset with expected evidence and human-reviewed answers."
        ),
        "details": summary,
        "quality_details": quality or None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lightweight Research Agent trace evaluation.")
    parser.add_argument("--path", default=str(DEFAULT_DATASET), help="JSONL golden dataset path.")
    parser.add_argument(
        "--quality-path",
        default=str(DEFAULT_QUALITY_DATASET),
        help="JSONL answer-quality fixture path.",
    )
    parser.add_argument("--skip-quality", action="store_true", help="Skip fixture answer-quality checks.")
    parser.add_argument("--json-out", default="", help="Optional path for the portfolio/eval JSON report.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the legacy summary instead of the report shape.",
    )
    args = parser.parse_args()
    summary = evaluate_cases(args.path)
    quality_summary = None if args.skip_quality else evaluate_quality_cases(args.quality_path)
    report = build_eval_report(summary, quality_summary=quality_summary)
    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary if args.summary_only else report, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] and (quality_summary is None or quality_summary["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
