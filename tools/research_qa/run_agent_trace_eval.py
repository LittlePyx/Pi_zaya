from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.agent.runner import build_agent_trace_for_completed_answer
from kb.agent.schema import validate_agent_trace
from tools.research_qa.validate_research_agent_golden import DEFAULT_DATASET, load_cases, validate_case


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lightweight Research Agent trace evaluation.")
    parser.add_argument("--path", default=str(DEFAULT_DATASET), help="JSONL golden dataset path.")
    args = parser.parse_args()
    summary = evaluate_cases(args.path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
