from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.agent.planner import plan_research_question


DEFAULT_DATASET = Path("docs/research_agent_golden_v0.jsonl")
REQUIRED_FIELDS = {"id", "query", "expected_question_type", "required_tools", "manual_review"}


def load_cases(path: str | Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    target = Path(path)
    cases: list[dict[str, Any]] = []
    for line_no, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(case, dict):
            raise ValueError(f"{target}:{line_no}: case must be an object")
        case["_line_no"] = line_no
        cases.append(case)
    return cases


def _ordered_contains(items: list[str], required: list[str]) -> bool:
    pos = 0
    for item in items:
        if pos < len(required) and item == required[pos]:
            pos += 1
    return pos == len(required)


def validate_case(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    case_id = str(case.get("id") or f"line:{case.get('_line_no')}")
    missing = sorted(REQUIRED_FIELDS - set(case))
    if missing:
        errors.append(f"{case_id}: missing fields: {', '.join(missing)}")
        return errors

    query = str(case.get("query") or "")
    expected_question_type = str(case.get("expected_question_type") or "")
    required_tools = [str(tool or "") for tool in list(case.get("required_tools") or []) if str(tool or "")]
    if not required_tools:
        errors.append(f"{case_id}: required_tools must not be empty")
    if not isinstance(case.get("manual_review"), list) or not case.get("manual_review"):
        errors.append(f"{case_id}: manual_review must be a non-empty list")

    question_type, plan = plan_research_question(query)
    planned_tools = [step.tool for step in plan]
    if question_type != expected_question_type:
        errors.append(f"{case_id}: expected {expected_question_type}, planned {question_type}")
    if required_tools and not _ordered_contains(planned_tools, required_tools):
        errors.append(f"{case_id}: required tools {required_tools} not in planned order {planned_tools}")

    structured_fields = case.get("expected_structured_fields")
    if structured_fields is not None:
        fields = [str(field or "") for field in list(structured_fields or []) if str(field or "")]
        required_compare_fields = {"paper", "method", "evidence", "limitation", "relation_to_question"}
        if not required_compare_fields.issubset(set(fields)):
            errors.append(f"{case_id}: comparison fields must include {sorted(required_compare_fields)}")
    return errors


def validate_cases(path: str | Path = DEFAULT_DATASET) -> dict[str, Any]:
    cases = load_cases(path)
    errors: list[str] = []
    question_types: dict[str, int] = {}
    for case in cases:
        errors.extend(validate_case(case))
        qtype = str(case.get("expected_question_type") or "unknown")
        question_types[qtype] = question_types.get(qtype, 0) + 1
    return {
        "ok": not errors,
        "case_count": len(cases),
        "question_types": question_types,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Research Agent golden prompt set.")
    parser.add_argument("--path", default=str(DEFAULT_DATASET), help="JSONL golden dataset path.")
    args = parser.parse_args()
    summary = validate_cases(args.path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
