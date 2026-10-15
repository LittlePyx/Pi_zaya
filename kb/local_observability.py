from __future__ import annotations

import json
import math
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Mapping


LOCAL_OBSERVATION_SCHEMA_VERSION = 2
DEFAULT_LOCAL_OBSERVATION_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_LOCAL_OBSERVATION_BACKUP_COUNT = 1
MIN_LOCAL_OBSERVATION_MAX_BYTES = 64 * 1024
MAX_LOCAL_OBSERVATION_MAX_BYTES = 1024 * 1024 * 1024
_STABLE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,79}$")
_EVENT_FIELDS = {
    "pi_zaya.answer_finalization": {
        "outcome_id",
        "status",
        "duration_ms",
        "gate_pass_count",
        "repair_action_count",
        "narrow_action_count",
        "detected_issue_count",
        "repair_required",
        "repair_succeeded",
    },
    "pi_zaya.agent_run": {
        "outcome_id",
        "status",
        "duration_ms",
        "question_type_id",
        "verification_outcome_id",
        "retrieval_hit_count",
        "tool_call_count",
        "tool_error_count",
        "supported_claim_count",
        "unsupported_claim_count",
        "input_token_count",
        "output_token_count",
    },
    "pi_zaya.retrieval": {
        "outcome_id",
        "status",
        "duration_ms",
        "retrieval_channel_id",
        "source_scope_id",
        "retrieval_hit_count",
        "query_variant_count",
        "cache_hit",
        "translation_used",
        "fallback_reason_id",
        "input_token_count",
        "requested_retrieval_mode_id",
        "used_retrieval_mode_id",
        "automatic_fallback",
        "cold_start",
        "index_cache_hit",
    },
    "pi_zaya.local_semantic_component": {
        "outcome_id",
        "status",
        "duration_ms",
        "lifecycle_operation_id",
        "lifecycle_state_id",
        "fallback_reason_id",
        "attempt_count",
        "consented",
        "bm25_available",
    },
    "pi_zaya.retrieval_shadow": {
        "outcome_id",
        "status",
        "duration_ms",
        "retrieval_channel_id",
        "source_scope_id",
        "product_hit_count",
        "shadow_hit_count",
        "overlap_count_at_5",
        "overlap_count_at_10",
        "changed_count_at_5",
        "changed_count_at_10",
        "dense_novelty_count",
        "source_escape_count",
        "protected_prefix_preserved",
        "product_output_unchanged",
        "cold_start",
        "index_cache_hit",
        "fallback_reason_id",
    },
    "pi_zaya.claim_verification": {
        "outcome_id",
        "status",
        "duration_ms",
        "verification_outcome_id",
        "model_revision",
        "total_claim_count",
        "supported_claim_count",
        "unsupported_claim_count",
        "fallback_reason_id",
    },
    "pi_zaya.model_call": {
        "outcome_id",
        "status",
        "duration_ms",
        "model_route_id",
        "error_category_id",
        "attempt_count",
        "input_token_count",
        "output_token_count",
        "total_token_count",
        "usage_reported",
        "streaming",
    },
}
_ID_FIELDS = {
    "outcome_id",
    "status",
    "question_type_id",
    "verification_outcome_id",
    "retrieval_channel_id",
    "source_scope_id",
    "fallback_reason_id",
    "model_revision",
    "model_route_id",
    "error_category_id",
    "requested_retrieval_mode_id",
    "used_retrieval_mode_id",
    "lifecycle_operation_id",
    "lifecycle_state_id",
}
_BOOL_FIELDS = {
    "repair_required",
    "repair_succeeded",
    "cache_hit",
    "translation_used",
    "usage_reported",
    "streaming",
    "protected_prefix_preserved",
    "product_output_unchanged",
    "cold_start",
    "index_cache_hit",
    "automatic_fallback",
    "consented",
    "bm25_available",
}
_COUNT_FIELDS = {
    "gate_pass_count",
    "repair_action_count",
    "narrow_action_count",
    "detected_issue_count",
    "retrieval_hit_count",
    "query_variant_count",
    "tool_call_count",
    "tool_error_count",
    "supported_claim_count",
    "unsupported_claim_count",
    "total_claim_count",
    "input_token_count",
    "output_token_count",
    "total_token_count",
    "attempt_count",
    "product_hit_count",
    "shadow_hit_count",
    "overlap_count_at_5",
    "overlap_count_at_10",
    "changed_count_at_5",
    "changed_count_at_10",
    "dense_novelty_count",
    "source_escape_count",
}
_PROHIBITED_ATTRIBUTE_NAMES = {
    "api_key",
    "claim",
    "claim_text",
    "credential",
    "evidence",
    "evidence_text",
    "exception",
    "exception_message",
    "file_path",
    "model",
    "prompt",
    "query",
    "response",
    "source_path",
    "source_title",
    "translated_query",
    "url",
}
_recorders: dict[tuple[str, int, int], "LocalJsonlObservationRecorder"] = {}
_recorders_lock = threading.Lock()


def local_observability_contract() -> dict[str, Any]:
    return {
        "schema_version": LOCAL_OBSERVATION_SCHEMA_VERSION,
        "service_name": "pi_zaya",
        "sink": {
            "format": "otel_aligned_jsonl",
            "default_enabled": False,
            "path_environment_variable": "KB_LOCAL_OBSERVABILITY_PATH",
            "external_export_enabled": False,
        },
        "retention": {
            "strategy": "size_rotation",
            "default_max_bytes": DEFAULT_LOCAL_OBSERVATION_MAX_BYTES,
            "default_backup_count": DEFAULT_LOCAL_OBSERVATION_BACKUP_COUNT,
            "max_bytes_environment_variable": "KB_LOCAL_OBSERVABILITY_MAX_BYTES",
            "backup_count_environment_variable": "KB_LOCAL_OBSERVABILITY_BACKUP_COUNT",
        },
        "privacy": {
            "raw_content_allowed": False,
            "prohibited_attribute_names": sorted(_PROHIBITED_ATTRIBUTE_NAMES),
        },
        "attribute_types": {
            "stable_id": sorted(_ID_FIELDS),
            "boolean": sorted(_BOOL_FIELDS),
            "non_negative_count": sorted(_COUNT_FIELDS),
            "duration_ms": ["duration_ms"],
        },
        "events": {
            name: sorted(fields) for name, fields in sorted(_EVENT_FIELDS.items())
        },
    }


def sanitize_observation_attributes(
    name: str,
    attributes: Mapping[str, Any],
) -> dict[str, Any]:
    allowed = _EVENT_FIELDS.get(str(name))
    if allowed is None:
        raise ValueError(f"unsupported local observation name: {name}")
    unknown = set(attributes) - allowed
    if unknown:
        raise ValueError(f"unapproved local observation attributes: {sorted(unknown)}")
    sanitized: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if key in _ID_FIELDS:
            clean = str(value or "").strip().lower()
            if not _STABLE_ID_RE.fullmatch(clean):
                raise ValueError(f"{key} must be a stable redacted identifier")
            sanitized[key] = clean
        elif key in _BOOL_FIELDS:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be boolean")
            sanitized[key] = value
        elif key in _COUNT_FIELDS:
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"{key} must be a non-negative count")
            sanitized[key] = int(value)
        elif key == "duration_ms":
            duration = float(value)
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("duration_ms must be finite and non-negative")
            sanitized[key] = round(duration, 3)
        else:
            raise ValueError(f"{key} has no approved local observation type")
    return sanitized


class LocalJsonlObservationRecorder:
    """Thread-safe, explicit-path JSONL sink for already-redacted spans."""

    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int = DEFAULT_LOCAL_OBSERVATION_MAX_BYTES,
        backup_count: int = DEFAULT_LOCAL_OBSERVATION_BACKUP_COUNT,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.max_bytes = max(
            MIN_LOCAL_OBSERVATION_MAX_BYTES,
            min(MAX_LOCAL_OBSERVATION_MAX_BYTES, int(max_bytes)),
        )
        self.backup_count = max(0, min(5, int(backup_count)))
        self._lock = threading.Lock()

    def _backup_path(self, index: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{int(index)}")

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        if not self.path.is_file():
            return
        if self.path.stat().st_size + max(0, int(incoming_bytes)) <= self.max_bytes:
            return
        if self.backup_count <= 0:
            self.path.unlink(missing_ok=True)
            return
        oldest = self._backup_path(self.backup_count)
        oldest.unlink(missing_ok=True)
        for index in range(self.backup_count - 1, 0, -1):
            source = self._backup_path(index)
            if source.is_file():
                source.replace(self._backup_path(index + 1))
        self.path.replace(self._backup_path(1))

    def emit(self, name: str, attributes: Mapping[str, Any]) -> dict[str, Any]:
        clean = sanitize_observation_attributes(name, attributes)
        finished_ns = time.time_ns()
        duration_ms = float(clean.get("duration_ms") or 0.0)
        event = {
            "schema_version": LOCAL_OBSERVATION_SCHEMA_VERSION,
            "service": {"name": "pi_zaya"},
            "name": name,
            "trace_id": secrets.token_hex(16),
            "span_id": secrets.token_hex(8),
            "start_time_unix_nano": max(0, finished_ns - int(duration_ms * 1_000_000)),
            "end_time_unix_nano": finished_ns,
            "status": {
                "code": (
                    "ERROR"
                    if clean.get("status") in {"error", "failed", "cancelled"}
                    else "OK"
                )
            },
            "attributes": clean,
        }
        rendered = json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed(len(rendered.encode("utf-8")))
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered)
        return event


def _bounded_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(str(os.environ.get(name, default) or default).strip())
    except (TypeError, ValueError, OverflowError):
        parsed = int(default)
    return max(int(minimum), min(int(maximum), parsed))


def emit_local_observation(name: str, attributes: Mapping[str, Any]) -> bool:
    """Emit only when an explicit local path is configured; never break the task."""

    raw_path = str(os.environ.get("KB_LOCAL_OBSERVABILITY_PATH") or "").strip()
    if not raw_path:
        return False
    try:
        max_bytes = _bounded_env_int(
            "KB_LOCAL_OBSERVABILITY_MAX_BYTES",
            DEFAULT_LOCAL_OBSERVATION_MAX_BYTES,
            minimum=MIN_LOCAL_OBSERVATION_MAX_BYTES,
            maximum=MAX_LOCAL_OBSERVATION_MAX_BYTES,
        )
        backup_count = _bounded_env_int(
            "KB_LOCAL_OBSERVABILITY_BACKUP_COUNT",
            DEFAULT_LOCAL_OBSERVATION_BACKUP_COUNT,
            minimum=0,
            maximum=5,
        )
        resolved_path = str(Path(raw_path).expanduser().resolve())
        recorder_key = (resolved_path, max_bytes, backup_count)
        with _recorders_lock:
            recorder = _recorders.get(recorder_key)
            if recorder is None:
                recorder = LocalJsonlObservationRecorder(
                    Path(resolved_path),
                    max_bytes=max_bytes,
                    backup_count=backup_count,
                )
                _recorders[recorder_key] = recorder
        recorder.emit(name, attributes)
        return True
    except (OSError, TypeError, ValueError):
        return False
