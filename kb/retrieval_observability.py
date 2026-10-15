from __future__ import annotations

import time
from typing import Any

from kb.local_observability import emit_local_observation


_SCOPE_IDS = {
    "library": "full_library",
    "full_library": "full_library",
    "basket": "research_basket",
    "research_basket": "research_basket",
    "current": "current_paper",
    "current_paper": "current_paper",
    "selected_sources": "selected_sources",
    "scoped": "selected_sources",
}
_CHANNEL_IDS = {
    "bm25_product",
    "hybrid_local_product",
    "agent_bm25",
    "structured_exact",
    "structured_section",
}
_SHADOW_CHANNEL_IDS = {
    "product_chat_shadow",
    "controlled_replay_shadow",
}
_RETRIEVAL_MODE_IDS = {"bm25", "hybrid_local"}


def _scope_id(value: str) -> str:
    return _SCOPE_IDS.get(str(value or "").strip().lower(), "unknown_scope")


def _channel_id(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _CHANNEL_IDS else "unknown_channel"


def _retrieval_mode_id(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    return normalized if normalized in _RETRIEVAL_MODE_IDS else "unknown_mode"


def record_retrieval_observation(
    *,
    started_at: float,
    channel: str,
    source_scope: str,
    hit_count: int = 0,
    query_variant_count: int = 0,
    translation_used: bool = False,
    fallback_reason: str | None = None,
    requested_mode: str = "",
    used_mode: str = "",
    automatic_fallback: bool = False,
    cold_start: bool = False,
    index_cache_hit: bool = False,
    status: str = "done",
) -> bool:
    safe_status = str(status or "").strip().lower()
    if safe_status not in {"done", "error", "cancelled"}:
        safe_status = "error"
    safe_hits = max(0, int(hit_count))
    if safe_status == "done":
        outcome_id = "retrieval.ready" if safe_hits else "retrieval.empty"
    elif safe_status == "cancelled":
        outcome_id = "retrieval.cancelled"
    else:
        outcome_id = "retrieval.error"
    reason = str(fallback_reason or "").strip().lower() or None
    attributes: dict[str, Any] = {
        "outcome_id": outcome_id,
        "status": safe_status,
        "duration_ms": max(0.0, (time.perf_counter() - float(started_at)) * 1000.0),
        "retrieval_channel_id": _channel_id(channel),
        "source_scope_id": _scope_id(source_scope),
        "retrieval_hit_count": safe_hits,
        "query_variant_count": max(0, int(query_variant_count)),
        "translation_used": bool(translation_used),
    }
    if requested_mode or used_mode:
        attributes.update(
            {
                "requested_retrieval_mode_id": _retrieval_mode_id(requested_mode),
                "used_retrieval_mode_id": _retrieval_mode_id(used_mode),
                "automatic_fallback": bool(automatic_fallback),
                "cold_start": bool(cold_start),
                "index_cache_hit": bool(index_cache_hit),
            }
        )
    if reason:
        attributes["fallback_reason_id"] = reason
    return emit_local_observation("pi_zaya.retrieval", attributes)


def record_retrieval_shadow_observation(
    *,
    started_at: float,
    channel: str,
    source_scope: str,
    status: str,
    product_hit_count: int,
    shadow_hit_count: int = 0,
    overlap_count_at_5: int = 0,
    overlap_count_at_10: int = 0,
    changed_count_at_5: int = 0,
    changed_count_at_10: int = 0,
    dense_novelty_count: int = 0,
    source_escape_count: int = 0,
    protected_prefix_preserved: bool = True,
    product_output_unchanged: bool = True,
    cold_start: bool = False,
    index_cache_hit: bool = False,
    fallback_reason: str | None = None,
) -> bool:
    safe_status = str(status or "").strip().lower()
    if safe_status not in {"done", "error", "cancelled"}:
        safe_status = "error"
    if safe_status == "done":
        outcome_id = "retrieval_shadow.compared"
    elif safe_status == "cancelled":
        outcome_id = "retrieval_shadow.skipped"
    else:
        outcome_id = "retrieval_shadow.error"
    normalized_channel = str(channel or "").strip().lower()
    attributes: dict[str, Any] = {
        "outcome_id": outcome_id,
        "status": safe_status,
        "duration_ms": max(0.0, (time.perf_counter() - float(started_at)) * 1000.0),
        "retrieval_channel_id": (
            normalized_channel
            if normalized_channel in _SHADOW_CHANNEL_IDS
            else "unknown_channel"
        ),
        "source_scope_id": _scope_id(source_scope),
        "product_hit_count": max(0, int(product_hit_count)),
        "shadow_hit_count": max(0, int(shadow_hit_count)),
        "overlap_count_at_5": max(0, int(overlap_count_at_5)),
        "overlap_count_at_10": max(0, int(overlap_count_at_10)),
        "changed_count_at_5": max(0, int(changed_count_at_5)),
        "changed_count_at_10": max(0, int(changed_count_at_10)),
        "dense_novelty_count": max(0, int(dense_novelty_count)),
        "source_escape_count": max(0, int(source_escape_count)),
        "protected_prefix_preserved": bool(protected_prefix_preserved),
        "product_output_unchanged": bool(product_output_unchanged),
        "cold_start": bool(cold_start),
        "index_cache_hit": bool(index_cache_hit),
    }
    reason = str(fallback_reason or "").strip().lower()
    if reason:
        attributes["fallback_reason_id"] = reason
    return emit_local_observation("pi_zaya.retrieval_shadow", attributes)
