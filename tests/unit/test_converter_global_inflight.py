from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

from kb.converter.global_inflight import CrossProcessInflightLimiter


class _FakeClock:
    def __init__(self, value: float = 1000.0):
        self.value = float(value)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


def _hold_global_slot(coordinator: str, ready) -> None:
    limiter = CrossProcessInflightLimiter(
        coordinator,
        max_limit=1,
        min_limit=1,
        owner_id="child",
    )
    lease = limiter.acquire(timeout=2.0)
    if lease is None:
        return
    ready.set()
    time.sleep(30.0)


def test_global_inflight_limit_is_shared_and_work_conserving(tmp_path: Path) -> None:
    coordinator = tmp_path / "coordinator"
    first = CrossProcessInflightLimiter(
        coordinator,
        max_limit=2,
        min_limit=1,
        owner_id="document-a",
        poll_interval_s=0.01,
    )
    second = CrossProcessInflightLimiter(
        coordinator,
        max_limit=2,
        min_limit=1,
        owner_id="document-b",
        poll_interval_s=0.01,
    )

    lease_a1 = first.acquire(timeout=0.2)
    lease_a2 = first.acquire(timeout=0.2)
    assert lease_a1 is not None
    assert lease_a2 is not None
    assert second.acquire(timeout=0.05) is None

    lease_a1.release()
    lease_b = second.acquire(timeout=0.2)
    assert lease_b is not None
    lease_b.release()
    lease_a2.release()


def test_global_inflight_pressure_reduces_then_recovers_shared_limit(tmp_path: Path) -> None:
    clock = _FakeClock()
    coordinator = tmp_path / "coordinator"
    first = CrossProcessInflightLimiter(
        coordinator,
        max_limit=8,
        min_limit=4,
        owner_id="document-a",
        cooldown_s=10.0,
        recovery_successes=2,
        clock=clock,
        monotonic=clock,
        sleeper=lambda seconds: clock.advance(seconds),
    )

    reduced = first.record_failure("rate_limited")
    assert reduced["effective_limit"] == 4
    assert reduced["rate_limited_events"] == 1
    assert reduced["limit_reductions"] == 1

    sibling = CrossProcessInflightLimiter(
        coordinator,
        max_limit=8,
        min_limit=4,
        owner_id="document-b",
        cooldown_s=10.0,
        recovery_successes=2,
        clock=clock,
        monotonic=clock,
        sleeper=lambda seconds: clock.advance(seconds),
    )
    assert sibling.get_effective_limit() == 4

    clock.advance(11.0)
    assert sibling.record_success()["effective_limit"] == 4
    recovered = first.record_success()
    assert recovered["effective_limit"] == 5
    assert recovered["limit_recoveries"] == 1


def test_global_inflight_slot_is_released_when_process_exits(tmp_path: Path) -> None:
    coordinator = tmp_path / "coordinator"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_hold_global_slot, args=(str(coordinator), ready))
    process.start()
    try:
        assert ready.wait(timeout=8.0)
        parent = CrossProcessInflightLimiter(
            coordinator,
            max_limit=1,
            min_limit=1,
            owner_id="parent",
            poll_interval_s=0.01,
        )
        assert parent.acquire(timeout=0.05) is None
        process.terminate()
        process.join(timeout=8.0)
        lease = parent.acquire(timeout=1.0)
        assert lease is not None
        lease.release()
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=8.0)
