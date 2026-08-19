from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable


_STATE_SCHEMA_VERSION = 1


def _clamp_int(value: object, *, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    return max(int(low), min(int(high), parsed))


def _clamp_float(value: object, *, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    return max(float(low), min(float(high), parsed))


def _open_lock_file(path: Path) -> BinaryIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b", buffering=0)
    handle.seek(0, os.SEEK_END)
    if handle.tell() < 1:
        handle.write(b"\0")
        handle.flush()
    return handle


def _try_lock(handle: BinaryIO) -> bool:
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _unlock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_global_inflight_snapshot(coordinator_dir: Path | str) -> dict:
    state_path = Path(coordinator_dir).expanduser() / "state.json"
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


@dataclass
class GlobalInflightLease:
    _limiter: "CrossProcessInflightLimiter"
    _handle: BinaryIO
    slot_index: int
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._limiter._release_lease(self)

    def __enter__(self) -> "GlobalInflightLease":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()


class CrossProcessInflightLimiter:
    """Work-conserving provider request budget shared by converter processes.

    Slot ownership is represented by OS file locks. Locks are released by the
    operating system if a converter process exits unexpectedly, so one crashed
    document cannot permanently strand provider capacity. A small shared state
    file carries conservative timeout/rate-limit backoff across sibling
    converters. The limiter is opt-in through an explicit coordinator path.
    """

    def __init__(
        self,
        coordinator_dir: Path | str,
        *,
        max_limit: int,
        owner_id: str,
        min_limit: int | None = None,
        cooldown_s: float | None = None,
        recovery_successes: int | None = None,
        poll_interval_s: float | None = None,
        waiter_ttl_s: float | None = None,
        clock: Callable[[], float] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ):
        self.coordinator_dir = Path(coordinator_dir).expanduser().resolve()
        self.coordinator_dir.mkdir(parents=True, exist_ok=True)
        self.max_limit = _clamp_int(max_limit, default=8, low=1, high=32)
        default_min = max(1, self.max_limit // 2)
        env_min = os.environ.get("KB_LLM_GLOBAL_MIN_INFLIGHT", "")
        requested_min = min_limit if min_limit is not None else (env_min or default_min)
        self.min_limit = _clamp_int(requested_min, default=default_min, low=1, high=self.max_limit)
        self.cooldown_s = _clamp_float(
            cooldown_s if cooldown_s is not None else os.environ.get("KB_LLM_GLOBAL_BACKOFF_S", "30"),
            default=30.0,
            low=1.0,
            high=600.0,
        )
        default_recovery = max(12, self.max_limit * 3)
        self.recovery_successes = _clamp_int(
            recovery_successes
            if recovery_successes is not None
            else os.environ.get("KB_LLM_GLOBAL_RECOVERY_SUCCESSES", str(default_recovery)),
            default=default_recovery,
            low=1,
            high=10000,
        )
        self.poll_interval_s = _clamp_float(
            poll_interval_s
            if poll_interval_s is not None
            else os.environ.get("KB_LLM_GLOBAL_POLL_INTERVAL_S", "0.05"),
            default=0.05,
            low=0.01,
            high=0.5,
        )
        self.waiter_ttl_s = _clamp_float(
            waiter_ttl_s
            if waiter_ttl_s is not None
            else os.environ.get("KB_LLM_GLOBAL_WAITER_TTL_S", "180"),
            default=180.0,
            low=15.0,
            high=3600.0,
        )
        raw_owner = str(owner_id or f"pid-{os.getpid()}").strip() or f"pid-{os.getpid()}"
        self.owner_id = hashlib.sha256(raw_owner.encode("utf-8", errors="replace")).hexdigest()[:20]
        self._clock = clock or time.time
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleeper or time.sleep
        self._state_path = self.coordinator_dir / "state.json"
        self._state_lock_path = self.coordinator_dir / ".state.lock"
        self._waiters_dir = self.coordinator_dir / "waiters"
        self._waiters_dir.mkdir(parents=True, exist_ok=True)
        self._cursor = 0
        self._cursor_lock = threading.Lock()
        self._cached_effective_limit = self.max_limit
        self._initialize_state()

    def _default_state(self) -> dict:
        now = float(self._clock())
        return {
            "schema_version": _STATE_SCHEMA_VERSION,
            "configured_limit": int(self.max_limit),
            "minimum_limit": int(self.min_limit),
            "effective_limit": int(self.max_limit),
            "min_effective_limit": int(self.max_limit),
            "cooldown_until": 0.0,
            "success_streak": 0,
            "pressure_events": 0,
            "rate_limited_events": 0,
            "timeout_events": 0,
            "limit_reductions": 0,
            "limit_recoveries": 0,
            "updated_at": now,
        }

    def _read_state_unlocked(self) -> dict:
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            payload = {}
        state = self._default_state()
        if isinstance(payload, dict):
            state.update(payload)
        configured = _clamp_int(
            state.get("configured_limit"),
            default=self.max_limit,
            low=1,
            high=self.max_limit,
        )
        minimum = _clamp_int(
            state.get("minimum_limit"),
            default=self.min_limit,
            low=1,
            high=configured,
        )
        effective = _clamp_int(
            state.get("effective_limit"),
            default=configured,
            low=minimum,
            high=configured,
        )
        state.update(
            {
                "schema_version": _STATE_SCHEMA_VERSION,
                "configured_limit": configured,
                "minimum_limit": minimum,
                "effective_limit": effective,
                "min_effective_limit": min(
                    effective,
                    _clamp_int(
                        state.get("min_effective_limit"),
                        default=effective,
                        low=minimum,
                        high=configured,
                    ),
                ),
            }
        )
        return state

    def _write_state_unlocked(self, state: dict) -> None:
        state = dict(state)
        state["updated_at"] = float(self._clock())
        temp_path = self.coordinator_dir / f".state-{os.getpid()}-{uuid.uuid4().hex}.tmp"
        temp_path.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(temp_path, self._state_path)

    def _acquire_state_lock(self, *, timeout_s: float = 2.0) -> BinaryIO | None:
        deadline = self._monotonic() + max(0.0, float(timeout_s))
        while True:
            handle = _open_lock_file(self._state_lock_path)
            if _try_lock(handle):
                return handle
            handle.close()
            if self._monotonic() >= deadline:
                return None
            self._sleep(min(self.poll_interval_s, max(0.0, deadline - self._monotonic())))

    @staticmethod
    def _release_state_lock(handle: BinaryIO | None) -> None:
        if handle is None:
            return
        try:
            _unlock(handle)
        finally:
            handle.close()

    def _initialize_state(self) -> None:
        handle = self._acquire_state_lock()
        if handle is None:
            raise RuntimeError("unable to initialize global LLM inflight coordinator")
        try:
            state = self._read_state_unlocked()
            # A shared directory must never expand because one child supplied a
            # larger value. The first/smaller configured ceiling wins safely.
            state["configured_limit"] = min(int(state["configured_limit"]), self.max_limit)
            state["minimum_limit"] = min(
                int(state["configured_limit"]),
                max(1, min(int(state["minimum_limit"]), self.min_limit)),
            )
            state["effective_limit"] = max(
                int(state["minimum_limit"]),
                min(int(state["effective_limit"]), int(state["configured_limit"])),
            )
            state["min_effective_limit"] = min(
                int(state["min_effective_limit"]),
                int(state["effective_limit"]),
            )
            self._cached_effective_limit = int(state["effective_limit"])
            self._write_state_unlocked(state)
        finally:
            self._release_state_lock(handle)

    def snapshot(self) -> dict:
        handle = self._acquire_state_lock(timeout_s=0.5)
        if handle is None:
            return load_global_inflight_snapshot(self.coordinator_dir)
        try:
            return self._read_state_unlocked()
        finally:
            self._release_state_lock(handle)

    def get_effective_limit(self) -> int:
        handle = self._acquire_state_lock(timeout_s=0.5)
        if handle is None:
            return int(self._cached_effective_limit)
        try:
            state = self._read_state_unlocked()
            self._cached_effective_limit = int(state["effective_limit"])
            return int(self._cached_effective_limit)
        finally:
            self._release_state_lock(handle)

    def _slot_path(self, slot_index: int) -> Path:
        return self.coordinator_dir / f"slot-{int(slot_index):02d}.lock"

    def _waiter_path(self) -> Path:
        return self._waiters_dir / f"waiter-{self.owner_id}-{uuid.uuid4().hex}.json"

    def _create_waiter(self) -> Path:
        waiter_path = self._waiter_path()
        waiter_path.write_text(
            json.dumps(
                {
                    "owner_id": self.owner_id,
                    "pid": os.getpid(),
                    "created_at": float(self._clock()),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return waiter_path

    def _waiting_owners(self) -> set[str]:
        now = float(self._clock())
        owners: set[str] = set()
        try:
            paths = list(self._waiters_dir.glob("waiter-*.json"))
        except Exception:
            return owners
        for path in paths:
            try:
                age = max(0.0, now - float(path.stat().st_mtime))
                if age > self.waiter_ttl_s:
                    path.unlink(missing_ok=True)
                    continue
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                owner = str(payload.get("owner_id") or "").strip()
                if owner:
                    owners.add(owner)
            except Exception:
                continue
        return owners

    def _active_owner_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        now = float(self._clock())
        for slot_index in range(self.max_limit):
            path = self._slot_path(slot_index)
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace").lstrip("\0"))
                owner = str(payload.get("owner_id") or "").strip()
                acquired_at = float(payload.get("acquired_at") or 0.0)
            except Exception:
                continue
            # Metadata is advisory for fairness only. The file lock remains the
            # authoritative global-cap guard and is crash-released by the OS.
            if owner and acquired_at > 0.0 and (now - acquired_at) <= 1800.0:
                counts[owner] = int(counts.get(owner, 0)) + 1
        return counts

    def _owner_can_acquire(self, *, effective_limit: int) -> bool:
        waiting_owners = self._waiting_owners()
        other_waiters = waiting_owners - {self.owner_id}
        if not other_waiters:
            return True
        active_counts = self._active_owner_counts()
        participants = set(active_counts) | waiting_owners | {self.owner_id}
        fair_cap = max(1, int(math.ceil(float(effective_limit) / float(max(1, len(participants))))))
        return int(active_counts.get(self.owner_id, 0)) < fair_cap

    def _next_slot_order(self, effective_limit: int) -> list[int]:
        with self._cursor_lock:
            start = self._cursor % max(1, int(effective_limit))
            self._cursor = (self._cursor + 1) % max(1, int(effective_limit))
        return list(range(start, effective_limit)) + list(range(0, start))

    def _try_acquire_slot(self, slot_index: int) -> GlobalInflightLease | None:
        handle = _open_lock_file(self._slot_path(slot_index))
        if not _try_lock(handle):
            handle.close()
            return None
        try:
            handle.seek(0)
            handle.truncate(0)
            handle.write(
                json.dumps(
                    {
                        "owner_id": self.owner_id,
                        "pid": os.getpid(),
                        "thread_id": threading.get_ident(),
                        "acquired_at": float(self._clock()),
                    },
                    sort_keys=True,
                ).encode("utf-8")
            )
            handle.flush()
            return GlobalInflightLease(self, handle, int(slot_index))
        except Exception:
            try:
                _unlock(handle)
            finally:
                handle.close()
            raise

    def acquire(self, timeout: float | None = None) -> GlobalInflightLease | None:
        deadline = None
        if timeout is not None:
            try:
                deadline = self._monotonic() + max(0.0, float(timeout))
            except Exception:
                deadline = None
        waiter_path = self._create_waiter()
        try:
            while True:
                effective_limit = self.get_effective_limit()
                if self._owner_can_acquire(effective_limit=effective_limit):
                    for slot_index in self._next_slot_order(effective_limit):
                        lease = self._try_acquire_slot(slot_index)
                        if lease is not None:
                            return lease
                if deadline is not None and self._monotonic() >= deadline:
                    return None
                sleep_s = self.poll_interval_s
                if deadline is not None:
                    sleep_s = min(sleep_s, max(0.0, deadline - self._monotonic()))
                    if sleep_s <= 0.0:
                        return None
                self._sleep(sleep_s)
        finally:
            try:
                waiter_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _release_lease(self, lease: GlobalInflightLease) -> None:
        handle = lease._handle
        try:
            handle.seek(0)
            handle.truncate(0)
            handle.write(b"\0")
            handle.flush()
        except Exception:
            pass
        try:
            _unlock(handle)
        finally:
            handle.close()

    def record_failure(self, failure_kind: str) -> dict:
        kind = str(failure_kind or "").strip().lower()
        if kind not in {"rate_limited", "timeout"}:
            return self.snapshot()
        handle = self._acquire_state_lock()
        if handle is None:
            return self.snapshot()
        try:
            state = self._read_state_unlocked()
            previous = int(state["effective_limit"])
            minimum = int(state["minimum_limit"])
            if kind == "rate_limited":
                target = max(minimum, int(math.floor(previous * 0.5)))
                state["rate_limited_events"] = int(state.get("rate_limited_events") or 0) + 1
            else:
                target = max(minimum, min(previous - 1, int(math.floor(previous * 0.75))))
                state["timeout_events"] = int(state.get("timeout_events") or 0) + 1
            target = max(minimum, min(previous, target))
            state["pressure_events"] = int(state.get("pressure_events") or 0) + 1
            state["success_streak"] = 0
            state["cooldown_until"] = max(
                float(state.get("cooldown_until") or 0.0),
                float(self._clock()) + self.cooldown_s,
            )
            if target < previous:
                state["effective_limit"] = target
                state["limit_reductions"] = int(state.get("limit_reductions") or 0) + 1
            state["min_effective_limit"] = min(
                int(state.get("min_effective_limit") or previous),
                int(state["effective_limit"]),
            )
            self._cached_effective_limit = int(state["effective_limit"])
            self._write_state_unlocked(state)
            return state
        finally:
            self._release_state_lock(handle)

    def record_success(self) -> dict:
        handle = self._acquire_state_lock(timeout_s=0.5)
        if handle is None:
            return self.snapshot()
        try:
            state = self._read_state_unlocked()
            configured = int(state["configured_limit"])
            effective = int(state["effective_limit"])
            if effective >= configured:
                if int(state.get("success_streak") or 0) != 0:
                    state["success_streak"] = 0
                    self._write_state_unlocked(state)
                self._cached_effective_limit = effective
                return state
            if float(self._clock()) < float(state.get("cooldown_until") or 0.0):
                self._cached_effective_limit = effective
                return state
            streak = int(state.get("success_streak") or 0) + 1
            if streak >= self.recovery_successes:
                state["effective_limit"] = min(configured, effective + 1)
                state["success_streak"] = 0
                state["limit_recoveries"] = int(state.get("limit_recoveries") or 0) + 1
            else:
                state["success_streak"] = streak
            self._cached_effective_limit = int(state["effective_limit"])
            self._write_state_unlocked(state)
            return state
        finally:
            self._release_state_lock(handle)
