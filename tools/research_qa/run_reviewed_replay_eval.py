from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_qa.run_agent_trace_eval import build_eval_report, evaluate_quality_cases, load_quality_cases


DEFAULT_REVIEWED_REPLAY_PATHS = [
    Path("docs/research_agent_reviewed_replay.jsonl"),
    Path("test_results/research_agent_answer_reviewed.jsonl"),
]


def _minimal_planner_summary() -> dict[str, Any]:
    return {
        "ok": True,
        "case_count": 0,
        "question_types": {},
        "planning_errors": [],
        "schema_errors": [],
    }


def _is_reviewed_case(case: dict[str, Any]) -> bool:
    return (
        str(case.get("sample_kind") or "").strip() == "real_chat_reviewed"
        or str(case.get("review_status") or "").strip() == "accepted"
    ) and case.get("replay_unlabeled") is not True


def _candidate_paths(paths: list[Path] | None) -> list[Path]:
    if paths:
        return [Path(path) for path in paths]
    return list(DEFAULT_REVIEWED_REPLAY_PATHS)


def run_reviewed_replay_eval(
    *,
    paths: list[Path] | None = None,
    require_reviewed: bool = False,
) -> dict[str, Any]:
    candidates = _candidate_paths(paths)
    evaluated: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[str] = []
    total_reviewed = 0

    for path in candidates:
        path = Path(path)
        if not path.exists():
            skipped.append({"path": str(path), "reason": "missing"})
            continue
        try:
            cases = load_quality_cases(path)
        except Exception as exc:
            errors.append(f"{path}: failed to load reviewed replay dataset: {exc}")
            continue

        reviewed_count = sum(1 for case in cases if _is_reviewed_case(case))
        non_reviewed_count = len(cases) - reviewed_count
        if reviewed_count <= 0:
            skipped.append(
                {
                    "path": str(path),
                    "reason": "no_reviewed_cases",
                    "case_count": len(cases),
                }
            )
            continue
        if non_reviewed_count:
            errors.append(
                f"{path}: reviewed replay gate expects only accepted reviewed cases; "
                f"found {non_reviewed_count} unreviewed cases"
            )
            continue

        quality = evaluate_quality_cases(path)
        report = build_eval_report(_minimal_planner_summary(), quality_summary=quality)
        evaluated.append(
            {
                "path": str(path),
                "case_count": int(quality.get("case_count") or 0),
                "reviewed_case_count": reviewed_count,
                "ok": bool(quality.get("ok")),
                "error_count": len(list(quality.get("errors") or [])),
                "report": report,
            }
        )
        total_reviewed += reviewed_count
        if not bool(quality.get("ok")):
            for error in list(quality.get("errors") or []):
                errors.append(f"{path}: {error}")

    if require_reviewed and total_reviewed <= 0:
        errors.append("no reviewed replay cases found")

    return {
        "ok": not errors,
        "reviewed_case_count": total_reviewed,
        "evaluated_dataset_count": len(evaluated),
        "skipped_dataset_count": len(skipped),
        "evaluated": evaluated,
        "skipped": skipped,
        "errors": errors,
        "metrics_note": (
            "Optional reviewed replay gate. Missing local or unpublished reviewed datasets are skipped; "
            "accepted reviewed datasets are evaluated strictly."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the optional reviewed Research Agent replay quality gate.")
    parser.add_argument(
        "--path",
        action="append",
        type=Path,
        default=None,
        help="Reviewed replay JSONL path. Can be passed more than once.",
    )
    parser.add_argument(
        "--require-reviewed",
        action="store_true",
        help="Fail when no reviewed replay cases are found.",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON summary output path.")
    args = parser.parse_args()

    summary = run_reviewed_replay_eval(paths=args.path, require_reviewed=args.require_reviewed)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
