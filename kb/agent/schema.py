from __future__ import annotations

from typing import Any

from .types import EvidenceStatus, QuestionType, StepStatus, ToolName


VALID_QUESTION_TYPES = set(QuestionType.__args__)
VALID_STATUSES = set(StepStatus.__args__)
VALID_TOOLS = set(ToolName.__args__)
VALID_EVIDENCE_STATUSES = set(EvidenceStatus.__args__)


def _is_int_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        int(value)
        return True
    except Exception:
        return False


def _check_mapping_list(value: Any, *, name: str, errors: list[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        errors.append(f"{name} must be a list")
        return []
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"{name}[{idx}] must be an object")
            continue
        out.append(item)
    return out


def validate_agent_trace(trace: Any) -> dict[str, Any]:
    """Validate the public Research Agent trace shape without extra dependencies."""
    errors: list[str] = []
    if not isinstance(trace, dict):
        return {
            "ok": False,
            "errors": ["trace must be an object"],
            "summary": {"plan_steps": 0, "execution_steps": 0, "supported_claims": 0, "total_claims": 0},
        }

    if trace.get("mode") != "research_agent":
        errors.append("mode must be research_agent")

    question_type = str(trace.get("question_type") or "")
    if question_type not in VALID_QUESTION_TYPES:
        errors.append(f"question_type must be one of {sorted(VALID_QUESTION_TYPES)}")

    status = str(trace.get("status") or "")
    if status not in VALID_STATUSES:
        errors.append(f"status must be one of {sorted(VALID_STATUSES)}")

    context = trace.get("context", {})
    if context is not None and not isinstance(context, dict):
        errors.append("context must be an object")
    if isinstance(context, dict) and "planner_intent" in context:
        intent = context.get("planner_intent")
        if not isinstance(intent, dict):
            errors.append("context.planner_intent must be an object")
        else:
            if str(intent.get("task_type") or "") not in VALID_QUESTION_TYPES:
                errors.append(f"context.planner_intent.task_type must be one of {sorted(VALID_QUESTION_TYPES)}")
            try:
                confidence = float(intent.get("confidence", 0.0) or 0.0)
            except Exception:
                errors.append("context.planner_intent.confidence must be numeric")
            else:
                if confidence < 0.0 or confidence > 1.0:
                    errors.append("context.planner_intent.confidence must be between 0 and 1")
            if "required_tools" in intent:
                required_tools = intent.get("required_tools") or []
                if not isinstance(required_tools, list):
                    errors.append("context.planner_intent.required_tools must be a list")
                    required_tools = []
                for idx, tool in enumerate(required_tools):
                    if str(tool or "") not in VALID_TOOLS:
                        errors.append(f"context.planner_intent.required_tools[{idx}] is invalid")

    plan = _check_mapping_list(trace.get("plan"), name="plan", errors=errors)
    for idx, step in enumerate(plan):
        tool = str(step.get("tool") or "")
        step_status = str(step.get("status") or "")
        if not str(step.get("goal") or "").strip():
            errors.append(f"plan[{idx}].goal is required")
        if tool not in VALID_TOOLS:
            errors.append(f"plan[{idx}].tool is invalid")
        if step_status not in VALID_STATUSES:
            errors.append(f"plan[{idx}].status is invalid")

    steps = _check_mapping_list(trace.get("steps"), name="steps", errors=errors)
    for idx, step in enumerate(steps):
        tool = str(step.get("tool") or "")
        step_status = str(step.get("status") or "")
        if tool not in VALID_TOOLS:
            errors.append(f"steps[{idx}].tool is invalid")
        if step_status not in VALID_STATUSES:
            errors.append(f"steps[{idx}].status is invalid")
        if "output" in step and not isinstance(step.get("output"), dict):
            errors.append(f"steps[{idx}].output must be an object")

    verification = trace.get("verification")
    if not isinstance(verification, dict):
        errors.append("verification must be an object")
        verification = {}
    for field in (
        "total_claims",
        "supported_claims",
        "unsupported_claims",
        "local_claims",
        "external_background_claims",
        "source_notice_count",
    ):
        if not _is_int_like(verification.get(field, 0)):
            errors.append(f"verification.{field} must be an integer")
    if "evidence_status" in verification and str(verification.get("evidence_status") or "") not in VALID_EVIDENCE_STATUSES:
        errors.append(f"verification.evidence_status must be one of {sorted(VALID_EVIDENCE_STATUSES)}")
    if "evidence_hit_count" in verification and not _is_int_like(verification.get("evidence_hit_count", 0)):
        errors.append("verification.evidence_hit_count must be an integer")
    claims = verification.get("claims", [])
    if claims is not None and not isinstance(claims, list):
        errors.append("verification.claims must be a list")

    research_run = trace.get("research_run", {})
    if research_run is not None and research_run != {}:
        if not isinstance(research_run, dict):
            errors.append("research_run must be an object")
            research_run = {}
        else:
            run_status = str(research_run.get("status") or "")
            if run_status and run_status not in {"planning", "retrieving", "extracting", "synthesizing", "verified", "failed"}:
                errors.append("research_run.status is invalid")
            source_policy = str(research_run.get("source_policy") or "")
            if source_policy and source_policy not in {
                "local_only",
                "local_plus_external_background",
                "external_allowed_with_notice",
                "trusted_sites_only",
            }:
                errors.append("research_run.source_policy is invalid")
            subtasks = research_run.get("subtasks", [])
            if subtasks is not None and not isinstance(subtasks, list):
                errors.append("research_run.subtasks must be a list")
                subtasks = []
            for idx, item in enumerate(list(subtasks or [])):
                if not isinstance(item, dict):
                    errors.append(f"research_run.subtasks[{idx}] must be an object")
                    continue
                if str(item.get("status") or "") and str(item.get("status") or "") not in VALID_STATUSES:
                    errors.append(f"research_run.subtasks[{idx}].status is invalid")
            matrix = research_run.get("evidence_matrix", [])
            if matrix is not None and not isinstance(matrix, list):
                errors.append("research_run.evidence_matrix must be a list")
                matrix = []
            for idx, item in enumerate(list(matrix or [])):
                if not isinstance(item, dict):
                    errors.append(f"research_run.evidence_matrix[{idx}] must be an object")
                    continue
                if "support_status" in item and str(item.get("support_status") or "") not in VALID_EVIDENCE_STATUSES:
                    errors.append(f"research_run.evidence_matrix[{idx}].support_status is invalid")

    summary = trace.get("summary", {})
    if summary is not None and not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    for field in (
        "total_claims",
        "supported_claims",
        "unsupported_claims",
        "local_claims",
        "external_background_claims",
        "source_notice_count",
        "usable_hit_count",
        "subtask_count",
        "evidence_matrix_rows",
        "plan_step_count",
        "tool_call_count",
    ):
        if isinstance(summary, dict) and field in summary and not _is_int_like(summary.get(field, 0)):
            errors.append(f"summary.{field} must be an integer")
    if isinstance(summary, dict) and "evidence_status" in summary and str(summary.get("evidence_status") or "") not in VALID_EVIDENCE_STATUSES:
        errors.append(f"summary.evidence_status must be one of {sorted(VALID_EVIDENCE_STATUSES)}")
    if isinstance(summary, dict) and "evidence_hit_count" in summary and not _is_int_like(summary.get("evidence_hit_count", 0)):
        errors.append("summary.evidence_hit_count must be an integer")

    return {
        "ok": not errors,
        "errors": errors,
        "summary": {
            "question_type": question_type,
            "plan_steps": len(plan),
            "execution_steps": len(steps),
            "supported_claims": int(verification.get("supported_claims") or 0)
            if _is_int_like(verification.get("supported_claims", 0))
            else 0,
            "total_claims": int(verification.get("total_claims") or 0)
            if _is_int_like(verification.get("total_claims", 0))
            else 0,
            "unsupported_claims": int(verification.get("unsupported_claims") or 0)
            if _is_int_like(verification.get("unsupported_claims", 0))
            else 0,
            "evidence_status": str(verification.get("evidence_status") or summary.get("evidence_status") or ""),
            "evidence_hit_count": int(verification.get("evidence_hit_count") or summary.get("evidence_hit_count") or 0)
            if _is_int_like(verification.get("evidence_hit_count", summary.get("evidence_hit_count", 0)))
            else 0,
            "research_run_status": str(research_run.get("status") or "") if isinstance(research_run, dict) else "",
            "evidence_matrix_rows": len(research_run.get("evidence_matrix") or []) if isinstance(research_run, dict) and isinstance(research_run.get("evidence_matrix"), list) else 0,
            "tool_call_count": len(steps),
            "has_errors": bool(trace.get("errors")) if isinstance(trace.get("errors"), list) else False,
            "has_context": isinstance(context, dict) and bool(context),
        },
    }
