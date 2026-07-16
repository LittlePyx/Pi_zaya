from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit


VisionCircuitKey = tuple[str, str, str]


@dataclass(frozen=True)
class VisionCircuitPolicy:
    enabled: bool
    failure_threshold: int
    cooldown_s: float
    state_ttl_s: float
    max_entries: int


@dataclass(frozen=True)
class VisionCircuitDecision:
    allow_request: bool
    reason: str
    retry_after_s: float = 0.0


@dataclass
class _VisionCircuitState:
    consecutive_failures: int = 0
    opened_until: float = 0.0
    probe_inflight: bool = False
    updated_at: float = 0.0
    last_failure_kind: str = ""


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0") or "").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(str(os.environ.get(name, str(default)) or str(default)).strip())
    except Exception:
        value = int(default)
    return max(int(low), min(int(high), int(value)))


def _env_float(name: str, default: float, *, low: float, high: float) -> float:
    try:
        value = float(str(os.environ.get(name, str(default)) or str(default)).strip())
    except Exception:
        value = float(default)
    return max(float(low), min(float(high), float(value)))


def load_vision_circuit_policy(speed_mode: str) -> VisionCircuitPolicy:
    mode = "ultra_fast" if str(speed_mode or "").strip().lower() == "ultra_fast" else "normal"
    prefix = "KB_PDF_VISION_CIRCUIT_ULTRA_FAST" if mode == "ultra_fast" else "KB_PDF_VISION_CIRCUIT_NORMAL"
    default_threshold = 2 if mode == "ultra_fast" else 3
    default_cooldown_s = 45.0 if mode == "ultra_fast" else 30.0
    threshold = _env_int(f"{prefix}_THRESHOLD", default_threshold, low=0, high=20)
    cooldown_s = _env_float(f"{prefix}_COOLDOWN_S", default_cooldown_s, low=0.0, high=600.0)
    enabled = _env_bool("KB_PDF_VISION_CIRCUIT_BREAKER", True) and threshold > 0 and cooldown_s > 0
    state_ttl_s = _env_float("KB_PDF_VISION_CIRCUIT_STATE_TTL_S", 180.0, low=5.0, high=3600.0)
    # Preserve half-open recovery even if an operator configures a TTL shorter
    # than the cooldown. Closed, below-threshold failure states still use the
    # short TTL unchanged.
    state_ttl_s = max(float(state_ttl_s), float(cooldown_s) + 1.0)
    return VisionCircuitPolicy(
        enabled=bool(enabled),
        failure_threshold=max(1, int(threshold or 1)),
        cooldown_s=max(1.0, float(cooldown_s or 1.0)),
        state_ttl_s=state_ttl_s,
        max_entries=_env_int("KB_PDF_VISION_CIRCUIT_MAX_ENTRIES", 64, low=4, high=512),
    )


def build_vision_circuit_key(*, base_url: str, model: str, speed_mode: str) -> VisionCircuitKey:
    raw_url = str(base_url or "").strip()
    try:
        parsed = urlsplit(raw_url)
        hostname = str(parsed.hostname or "").strip().lower()
        port = f":{int(parsed.port)}" if parsed.port is not None else ""
        provider = f"{str(parsed.scheme or '').lower()}://{hostname}{port}{parsed.path.rstrip('/')}"
    except Exception:
        provider = raw_url.split("?", 1)[0].split("#", 1)[0].rstrip("/").lower()
    if not provider or provider == "://":
        provider = "unknown-provider"
    mode = "ultra_fast" if str(speed_mode or "").strip().lower() == "ultra_fast" else "normal"
    return provider, str(model or "").strip().lower() or "unknown-model", mode


def vision_circuit_failure_kind(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        if int(status_code) == 429:
            return "rate_limited"
    except Exception:
        pass
    name = type(exc).__name__.lower()
    message = str(exc or "").strip().lower()
    explicit_429 = bool(
        re.search(r"(?:http|status(?:_code)?|error\s+code)\s*[:=]?\s*429\b", message)
    )
    if "ratelimit" in name or "rate limit" in message or explicit_429:
        return "rate_limited"
    if isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in message or "hard timeout" in message:
        return "timeout"
    return ""


class VisionCircuitBreaker:
    def __init__(self, *, clock: Callable[[], float] | None = None):
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._states: dict[VisionCircuitKey, _VisionCircuitState] = {}

    def _prune_locked(self, *, now: float, policy: VisionCircuitPolicy) -> None:
        stale_before = float(now) - float(policy.state_ttl_s)
        for key, state in list(self._states.items()):
            if float(state.opened_until) > float(now):
                continue
            if float(state.updated_at) < stale_before:
                self._states.pop(key, None)
        while len(self._states) > int(policy.max_entries):
            oldest_key = min(
                self._states,
                key=lambda item: float(self._states[item].updated_at),
                default=None,
            )
            if oldest_key is None:
                break
            self._states.pop(oldest_key, None)

    def before_request(self, key: VisionCircuitKey, policy: VisionCircuitPolicy) -> VisionCircuitDecision:
        if not policy.enabled:
            self.reset(key)
            return VisionCircuitDecision(True, "disabled")
        now = float(self._clock())
        with self._lock:
            self._prune_locked(now=now, policy=policy)
            state = self._states.get(key)
            if state is None:
                return VisionCircuitDecision(True, "closed")
            state.updated_at = now
            if float(state.opened_until) > now:
                return VisionCircuitDecision(
                    False,
                    "open",
                    max(0.0, float(state.opened_until) - now),
                )
            if float(state.opened_until) > 0.0:
                if state.probe_inflight:
                    return VisionCircuitDecision(False, "probe_inflight")
                state.probe_inflight = True
                return VisionCircuitDecision(True, "half_open_probe")
            return VisionCircuitDecision(True, "closed")

    def record_failure(
        self,
        key: VisionCircuitKey,
        *,
        failure_kind: str,
        policy: VisionCircuitPolicy,
    ) -> bool:
        kind = str(failure_kind or "").strip().lower()
        if kind not in {"timeout", "rate_limited"}:
            self.record_neutral(key)
            return False
        if not policy.enabled:
            self.reset(key)
            return False
        now = float(self._clock())
        with self._lock:
            self._prune_locked(now=now, policy=policy)
            state = self._states.get(key)
            if state is None:
                state = _VisionCircuitState(updated_at=now)
                self._states[key] = state
            already_open = float(state.opened_until) > now
            was_half_open = bool(state.probe_inflight or float(state.opened_until) > 0.0)
            state.probe_inflight = False
            state.updated_at = now
            state.last_failure_kind = kind
            if was_half_open:
                state.consecutive_failures = int(policy.failure_threshold)
            else:
                state.consecutive_failures += 1
            opened = int(state.consecutive_failures) >= int(policy.failure_threshold)
            if opened:
                state.opened_until = now + float(policy.cooldown_s)
            self._prune_locked(now=now, policy=policy)
            return bool(opened and not already_open)

    def record_success(self, key: VisionCircuitKey) -> None:
        self.reset(key)

    def record_neutral(self, key: VisionCircuitKey) -> None:
        # Only consecutive timeout/429 failures count.  Any ordinary response or
        # business error breaks that sequence and must release a half-open probe.
        self.reset(key)

    def reset(self, key: VisionCircuitKey) -> None:
        with self._lock:
            self._states.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._states.clear()

    def state_count(self) -> int:
        with self._lock:
            return len(self._states)
