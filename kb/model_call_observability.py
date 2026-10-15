from __future__ import annotations

import time
from typing import Any, Mapping

from kb.local_observability import emit_local_observation


_TOKEN_ALIASES = {
    "input_token_count": ("input_token_count", "input_tokens", "prompt_tokens"),
    "output_token_count": ("output_token_count", "output_tokens", "completion_tokens"),
    "total_token_count": ("total_token_count", "total_tokens"),
}
_MODEL_ROUTES = {
    "text": "text_generation",
    "vision": "vision_generation",
    "web_search": "web_search_generation",
}


def _read_field(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def extract_provider_token_usage(value: Any) -> dict[str, int]:
    """Return only explicit provider usage fields; never tokenize or estimate."""

    usage = _read_field(value, "usage")
    if usage is None:
        usage = value
    counts: dict[str, int] = {}
    for output_key, aliases in _TOKEN_ALIASES.items():
        for alias in aliases:
            parsed = _nonnegative_int(_read_field(usage, alias))
            if parsed is not None:
                counts[output_key] = parsed
                break
    if "total_token_count" not in counts and {
        "input_token_count",
        "output_token_count",
    } <= counts.keys():
        counts["total_token_count"] = (
            counts["input_token_count"] + counts["output_token_count"]
        )
    return counts


def stable_model_error_category(error: BaseException | None) -> str | None:
    if error is None:
        return None
    name = type(error).__name__.lower()
    status_code = _nonnegative_int(getattr(error, "status_code", None))
    if isinstance(error, TimeoutError) or "timeout" in name:
        return "timeout"
    if status_code in {401, 403} or "authentication" in name or "permission" in name:
        return "authentication"
    if status_code == 429 or "ratelimit" in name or "rate_limit" in name:
        return "rate_limit"
    if isinstance(error, ConnectionError) or "connection" in name:
        return "connection"
    return "provider_error"


def record_model_call_observation(
    *,
    route_kind: str,
    started_at: float,
    outcome: str,
    attempt_count: int,
    streaming: bool,
    usage_source: Any = None,
    error: BaseException | None = None,
) -> bool:
    route_id = _MODEL_ROUTES.get(str(route_kind or "").strip().lower(), "unknown_route")
    outcome_id = str(outcome or "").strip().lower()
    status = "done" if outcome_id == "completed" else outcome_id
    usage = extract_provider_token_usage(usage_source)
    attributes: dict[str, Any] = {
        "outcome_id": outcome_id,
        "status": status,
        "duration_ms": max(0.0, (time.monotonic() - float(started_at)) * 1000.0),
        "model_route_id": route_id,
        "attempt_count": max(1, int(attempt_count)),
        "usage_reported": bool(usage),
        "streaming": bool(streaming),
        **usage,
    }
    error_category = stable_model_error_category(error)
    if error_category:
        attributes["error_category_id"] = error_category
    return emit_local_observation("pi_zaya.model_call", attributes)
