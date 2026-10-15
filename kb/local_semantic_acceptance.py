from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


LOCAL_SEMANTIC_ACCEPTANCE_REPORT_SCHEMA_VERSION = 1
_LIFECYCLE_EVENT = "pi_zaya.local_semantic_component"
_RETRIEVAL_EVENT = "pi_zaya.retrieval"
_LIFECYCLE_OUTCOMES = (
    "local_semantic_component.ready",
    "local_semantic_component.paused",
    "local_semantic_component.needs_space",
    "local_semantic_component.bm25_contained",
    "local_semantic_component.removed",
    "local_semantic_component.unavailable",
)
_PREPARATION_FAILURE_OUTCOMES = frozenset(
    {
        "local_semantic_component.needs_space",
        "local_semantic_component.bm25_contained",
        "local_semantic_component.unavailable",
    }
)
_LIFECYCLE_OPERATIONS = frozenset({"enable", "resume", "recover", "remove"})


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(max(0, int(numerator)) / int(denominator), 4)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * min(1.0, max(0.0, float(quantile)))
    lower = math.floor(position)
    upper = math.ceil(position)
    value = ordered[lower]
    if upper != lower:
        value += (ordered[upper] - ordered[lower]) * (position - lower)
    return round(value, 3)


def _fallback_category(reason: object) -> str:
    clean = str(reason or "").strip().lower()
    if not clean:
        return "unspecified"
    if clean == "local_semantic_not_promoted":
        return "not_promoted"
    if "scope" in clean or "protected_prefix" in clean:
        return "source_safety_boundary"
    if "index" in clean or "database_mismatch" in clean:
        return "semantic_index"
    if "dependenc" in clean or "site_packages" in clean:
        return "runtime_dependency"
    if clean == "dense_query_failed":
        return "dense_query"
    if "component" in clean or "runtime" in clean or "artifact" in clean:
        return "component_runtime"
    return "other_internal"


def build_local_semantic_acceptance_report(
    events: Iterable[Mapping[str, Any]],
    *,
    input_line_count: int | None = None,
) -> dict[str, Any]:
    """Aggregate v2 redacted spans without copying event identifiers or content."""

    accepted = 0
    ignored = 0
    lifecycle_starts: Counter[str] = Counter()
    lifecycle_terminals: Counter[str] = Counter()
    enable_ready = 0
    preparation_failures = 0
    bm25_contained_failures = 0
    hybrid_requests = 0
    hybrid_used = 0
    automatic_fallbacks = 0
    fallback_categories: Counter[str] = Counter()
    warm_latencies: list[float] = []

    for event in events:
        if not isinstance(event, Mapping):
            ignored += 1
            continue
        try:
            source_schema_version = int(event.get("schema_version") or 0)
        except (TypeError, ValueError, OverflowError):
            ignored += 1
            continue
        if source_schema_version < 2:
            ignored += 1
            continue
        name = str(event.get("name") or "")
        attributes = event.get("attributes")
        if name not in {_LIFECYCLE_EVENT, _RETRIEVAL_EVENT} or not isinstance(
            attributes, Mapping
        ):
            ignored += 1
            continue
        accepted += 1
        if name == _LIFECYCLE_EVENT:
            outcome = str(attributes.get("outcome_id") or "")
            operation = str(attributes.get("lifecycle_operation_id") or "")
            if operation not in _LIFECYCLE_OPERATIONS:
                operation = "unknown_operation"
            if outcome == "local_semantic_component.started":
                lifecycle_starts[operation] += 1
                continue
            if outcome not in _LIFECYCLE_OUTCOMES:
                ignored += 1
                accepted -= 1
                continue
            lifecycle_terminals[outcome] += 1
            if outcome == "local_semantic_component.ready" and operation == "enable":
                enable_ready += 1
            if outcome in _PREPARATION_FAILURE_OUTCOMES:
                preparation_failures += 1
                if attributes.get("bm25_available") is True:
                    bm25_contained_failures += 1
            continue

        requested = str(attributes.get("requested_retrieval_mode_id") or "")
        used = str(attributes.get("used_retrieval_mode_id") or "")
        if requested != "hybrid_local":
            continue
        hybrid_requests += 1
        if used == "hybrid_local":
            hybrid_used += 1
            duration = attributes.get("duration_ms")
            if (
                attributes.get("cold_start") is False
                and attributes.get("index_cache_hit") is True
                and isinstance(duration, (int, float))
                and not isinstance(duration, bool)
                and math.isfinite(float(duration))
                and float(duration) >= 0
            ):
                warm_latencies.append(float(duration))
        if attributes.get("automatic_fallback") is True and used == "bm25":
            automatic_fallbacks += 1
            fallback_categories[_fallback_category(attributes.get("fallback_reason_id"))] += 1

    enable_starts = lifecycle_starts["enable"]
    observed_lines = accepted + ignored if input_line_count is None else max(
        0, int(input_line_count)
    )
    ignored_lines = max(ignored, observed_lines - accepted)
    return {
        "schema_version": LOCAL_SEMANTIC_ACCEPTANCE_REPORT_SCHEMA_VERSION,
        "source_contract_schema_version": 2,
        "privacy": {
            "aggregate_only": True,
            "raw_content_included": False,
            "event_ids_included": False,
        },
        "window": {
            "input_line_count": observed_lines,
            "accepted_event_count": accepted,
            "ignored_event_count": ignored_lines,
            "window_boundary_incomplete": enable_ready > enable_starts,
        },
        "lifecycle": {
            "operation_started_counts": dict(sorted(lifecycle_starts.items())),
            "terminal_outcome_counts": {
                outcome: lifecycle_terminals[outcome]
                for outcome in _LIFECYCLE_OUTCOMES
            },
            "opt_in_started_count": enable_starts,
            "opt_in_ready_count": enable_ready,
            "observed_opt_in_completion_rate": _rate(enable_ready, enable_starts),
            "preparation_failure_count": preparation_failures,
            "bm25_contained_failure_count": bm25_contained_failures,
            "automatic_containment_rate": _rate(
                bm25_contained_failures,
                preparation_failures,
            ),
        },
        "retrieval": {
            "hybrid_request_count": hybrid_requests,
            "actual_hybrid_use_count": hybrid_used,
            "actual_hybrid_use_rate": _rate(hybrid_used, hybrid_requests),
            "automatic_fallback_count": automatic_fallbacks,
            "automatic_fallback_rate": _rate(automatic_fallbacks, hybrid_requests),
            "automatic_fallback_category_counts": dict(
                sorted(fallback_categories.items())
            ),
            "warm_hybrid_latency_ms": {
                "sample_count": len(warm_latencies),
                "p50": _percentile(warm_latencies, 0.5),
                "p95": _percentile(warm_latencies, 0.95),
                "max": round(max(warm_latencies), 3) if warm_latencies else None,
            },
        },
        "sample_sufficiency": {
            "opt_in_completion_available": enable_starts > 0,
            "actual_mode_rate_available": hybrid_requests > 0,
            "warm_latency_available": bool(warm_latencies),
        },
    }


def load_local_observation_events(path: Path) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    line_count = 0
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line_count += 1
            try:
                payload = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events, line_count


__all__ = [
    "LOCAL_SEMANTIC_ACCEPTANCE_REPORT_SCHEMA_VERSION",
    "build_local_semantic_acceptance_report",
    "load_local_observation_events",
]
