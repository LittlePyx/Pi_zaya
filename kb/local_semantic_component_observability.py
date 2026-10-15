from __future__ import annotations

import time

from .local_observability import emit_local_observation
from .local_semantic_component_lifecycle_contract import (
    LocalSemanticComponentLifecycleRecord,
)


_OPERATIONS = {"enable", "resume", "recover", "remove"}
_TERMINAL_OUTCOME_BY_STATE = {
    "ready": "local_semantic_component.ready",
    "paused_by_user": "local_semantic_component.paused",
    "needs_space": "local_semantic_component.needs_space",
    "degraded_bm25": "local_semantic_component.bm25_contained",
    "not_installed": "local_semantic_component.removed",
    "unavailable_in_build": "local_semantic_component.unavailable",
}


def _fallback_category(record: LocalSemanticComponentLifecycleRecord) -> str:
    reason = str(record.reason_id or "").strip().lower()
    if record.state == "paused_by_user":
        return "user_pause"
    if record.state == "needs_space":
        return "storage_capacity"
    if record.state != "degraded_bm25":
        return ""
    if "download" in reason or "transport" in reason:
        return "component_download"
    if "archive" in reason or "payload" in reason or "quarantin" in reason:
        return "component_integrity"
    if "health" in reason:
        return "component_health"
    if "index" in reason:
        return "semantic_index"
    if "space" in reason or "storage" in reason or "path_too_long" in reason:
        return "storage_capacity"
    if "remov" in reason or "cleanup" in reason:
        return "component_removal"
    return "component_internal"


def record_local_semantic_component_observation(
    record: LocalSemanticComponentLifecycleRecord,
    *,
    phase: str,
    started_at: float | None = None,
) -> bool:
    """Record one redacted lifecycle start or terminal outcome, fail-open."""

    clean_phase = str(phase or "").strip().lower()
    if clean_phase == "started":
        outcome_id = "local_semantic_component.started"
        status = "running"
    elif clean_phase == "terminal":
        outcome_id = _TERMINAL_OUTCOME_BY_STATE.get(record.state, "")
        if not outcome_id:
            return False
        status = "error" if record.state in {
            "degraded_bm25",
            "unavailable_in_build",
        } else "done"
    else:
        return False

    operation = str(record.operation or "").strip().lower()
    attributes: dict[str, object] = {
        "outcome_id": outcome_id,
        "status": status,
        "lifecycle_operation_id": (
            operation if operation in _OPERATIONS else "unknown_operation"
        ),
        "lifecycle_state_id": record.state,
        "attempt_count": max(0, int(record.attempt)),
        "consented": bool(record.consented),
        "bm25_available": True,
    }
    if started_at is not None:
        attributes["duration_ms"] = max(
            0.0,
            (time.perf_counter() - float(started_at)) * 1000.0,
        )
    fallback_category = _fallback_category(record)
    if fallback_category:
        attributes["fallback_reason_id"] = fallback_category
    return emit_local_observation("pi_zaya.local_semantic_component", attributes)


__all__ = ["record_local_semantic_component_observation"]
