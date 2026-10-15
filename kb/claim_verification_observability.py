from __future__ import annotations

from typing import Any

from kb.local_observability import emit_local_observation


_EVIDENCE_OUTCOMES = {
    "grounded",
    "needs_review",
    "insufficient",
    "not_applicable",
}
_STEP_STATUSES = {"done", "error", "canceled", "cancelled", "skipped"}


def record_production_claim_verification(
    *,
    evidence_status: str,
    total_claim_count: int,
    supported_claim_count: int,
    unsupported_claim_count: int,
    duration_ms: float,
    step_status: str = "done",
) -> bool:
    outcome = str(evidence_status or "").strip().lower()
    if outcome not in _EVIDENCE_OUTCOMES:
        outcome = "insufficient"
    status = str(step_status or "").strip().lower()
    if status not in _STEP_STATUSES:
        status = "error"
    if status == "canceled":
        status = "cancelled"
    attributes: dict[str, Any] = {
        "outcome_id": f"claim_verification.{outcome}",
        "status": status,
        "duration_ms": max(0.0, float(duration_ms)),
        "verification_outcome_id": outcome,
        "total_claim_count": max(0, int(total_claim_count)),
        "supported_claim_count": max(0, int(supported_claim_count)),
        "unsupported_claim_count": max(0, int(unsupported_claim_count)),
    }
    if status == "skipped" or outcome == "not_applicable":
        attributes["fallback_reason_id"] = "non_local_answer"
    elif status == "error":
        attributes["fallback_reason_id"] = "verification_error"
    elif status == "cancelled":
        attributes["fallback_reason_id"] = "task_cancelled"
    else:
        # This identifies the production check honestly: it is lexical and
        # structural, not the benchmark-only semantic NLI shadow candidate.
        attributes["model_revision"] = "lexical_structural_v1"
    return emit_local_observation("pi_zaya.claim_verification", attributes)
