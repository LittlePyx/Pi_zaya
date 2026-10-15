from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from kb.local_observability import emit_local_observation


ANSWER_FINALIZATION_OUTCOME_VERSION = 1


def audit_final_claim_evidence(
    answer: str,
    answer_hits: list[dict],
    prompt: str,
    audit: Callable[..., tuple[str, dict[str, Any]]],
) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    finalized, metadata = audit(
        answer, answer_hits=answer_hits, allow_citation_repairs=True, prompt=prompt
    )
    metadata["final_gate_applied"] = True
    metadata["pass_timings_ms"] = {
        "initial": round((time.perf_counter() - started) * 1000.0, 3),
        "recheck": 0.0,
    }
    return finalized, metadata

REPAIR_ACTION_FIELDS = (
    "repaired_citations",
    "rebound_citations",
    "section_source_rebound_citations",
    "repaired_modality_boundaries",
    "relocated_midphrase_citations",
    "restored_prompt_terms",
    "restored_evidence_numbers",
    "restored_source_facts",
    "added_requested_facet_boundaries",
)

NARROW_ACTION_FIELDS = (
    "trimmed_unsupported_inferences",
    "dropped_cross_source_claims",
    "dropped_hard_mismatch_claims",
    "stripped_weak_citations",
    "renderer_rejected_citations",
    "removed_unplanned_citations",
    "removed_heading_citations",
    "dropped_unsupported_unplanned_claims",
    "dropped_unsupported_inferences",
    "dropped_placeholder_sections",
)

DETECTION_FIELDS = (
    "uncited_high_risk_claims",
    "citation_mismatch_claims",
)


def _record(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _count(record: dict[str, Any], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _counts(record: dict[str, Any], fields: tuple[str, ...]) -> dict[str, int]:
    return {field: _count(record, field) for field in fields}


def _gate_pass_count(claim_evidence: dict[str, Any]) -> int:
    if not bool(claim_evidence.get("final_gate_applied")):
        return 0
    pass_count = 1
    timings = _record(claim_evidence.get("pass_timings_ms"))
    try:
        recheck_ms = float(timings.get("recheck") or 0.0)
    except (TypeError, ValueError):
        recheck_ms = 0.0
    if (
        recheck_ms > 0
        or bool(claim_evidence.get("post_gate_term_normalization"))
        or bool(claim_evidence.get("post_gate_citation_rebinding"))
    ):
        pass_count += 1
    return pass_count


def _duration_ms(claim_evidence: dict[str, Any]) -> float:
    timings = _record(claim_evidence.get("pass_timings_ms"))
    total = 0.0
    for key in ("initial", "recheck"):
        try:
            total += max(0.0, float(timings.get(key) or 0.0))
        except (TypeError, ValueError):
            continue
    return round(total, 3)


def build_answer_finalization_outcome(answer_quality: dict[str, Any] | None) -> dict[str, Any]:
    """Return compact, local-only outcome counters for one finalized answer.

    The payload contains no prompt, answer, evidence text, paths, or claim rows.
    It makes future repair rates measurable without exposing internal verifier
    states to ordinary users.
    """
    quality = _record(answer_quality)
    claim_evidence = _record(quality.get("claim_evidence"))
    gate_applied = bool(claim_evidence.get("final_gate_applied"))
    repair_counts = _counts(claim_evidence, REPAIR_ACTION_FIELDS)
    narrow_counts = _counts(claim_evidence, NARROW_ACTION_FIELDS)
    detection_counts = _counts(claim_evidence, DETECTION_FIELDS)
    repair_count = sum(repair_counts.values())
    narrow_count = sum(narrow_counts.values())
    action_count = repair_count + narrow_count
    final_ok = bool(quality.get("minimum_ok", True)) and bool(
        claim_evidence.get("minimum_ok", True)
    )

    if not gate_applied:
        status = "not_applicable"
    elif not final_ok:
        status = "failed"
    elif repair_count and narrow_count:
        status = "repaired_and_narrowed"
    elif repair_count:
        status = "repaired"
    elif narrow_count:
        status = "narrowed"
    else:
        status = "passed"

    repair_required = bool(gate_applied and action_count > 0)
    return {
        "schema_version": ANSWER_FINALIZATION_OUTCOME_VERSION,
        "status": status,
        "gate_applied": gate_applied,
        "gate_pass_count": _gate_pass_count(claim_evidence),
        "duration_ms": _duration_ms(claim_evidence),
        "final_ok": final_ok if gate_applied else None,
        "repair_required": repair_required,
        "repair_succeeded": final_ok if repair_required else None,
        "action_count": action_count,
        "repair_action_count": repair_count,
        "narrow_action_count": narrow_count,
        "detected_issue_count": sum(detection_counts.values()),
        "repair_actions": repair_counts,
        "narrow_actions": narrow_counts,
        "detections": detection_counts,
        "scope": "source_grounded_answer_finalization",
    }


def record_answer_finalization_outcome(
    answer_quality: dict[str, Any] | None,
) -> dict[str, Any]:
    outcome = build_answer_finalization_outcome(answer_quality)
    emit_local_observation(
        "pi_zaya.answer_finalization",
        {
            "outcome_id": f"answer_finalization.{outcome['status']}",
            "status": outcome["status"],
            "duration_ms": outcome["duration_ms"],
            "gate_pass_count": outcome["gate_pass_count"],
            "repair_action_count": outcome["repair_action_count"],
            "narrow_action_count": outcome["narrow_action_count"],
            "detected_issue_count": outcome["detected_issue_count"],
            "repair_required": outcome["repair_required"],
            "repair_succeeded": outcome["repair_succeeded"],
        },
    )
    return outcome
