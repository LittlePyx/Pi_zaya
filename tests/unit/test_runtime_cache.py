from __future__ import annotations

from kb import runtime_state as RUNTIME
from kb import retrieval_engine
from kb.runtime_cache import cache_get, cache_set


def test_retrieval_engine_runtime_cache_callbacks_store_values():
    with RUNTIME.CACHE_LOCK:
        old_bucket = RUNTIME.CACHE.pop("unit_cache_test", None)
    try:
        retrieval_engine.configure_cache(cache_get, cache_set)

        retrieval_engine._cache_set("unit_cache_test", "a", {"value": 1}, max_items=2)

        assert retrieval_engine._cache_get("unit_cache_test", "a") == {"value": 1}
        with RUNTIME.CACHE_LOCK:
            assert RUNTIME.CACHE["unit_cache_test"]["a"] == {"value": 1}
    finally:
        with RUNTIME.CACHE_LOCK:
            RUNTIME.CACHE.pop("unit_cache_test", None)
            if old_bucket is not None:
                RUNTIME.CACHE["unit_cache_test"] = old_bucket


def test_generation_task_prune_keeps_running_and_removes_old_done():
    with RUNTIME.GEN_LOCK:
        old_tasks = dict(RUNTIME.GEN_TASKS)
        RUNTIME.GEN_TASKS.clear()
        RUNTIME.GEN_TASKS["old-done"] = {"status": "done", "finished_at": 10.0}
        RUNTIME.GEN_TASKS["running"] = {"status": "running", "created_at": 10.0}
    try:
        removed = RUNTIME.prune_generation_tasks(now=100.0, ttl_s=20.0, max_items=10)

        assert removed == 1
        with RUNTIME.GEN_LOCK:
            assert "old-done" not in RUNTIME.GEN_TASKS
            assert "running" in RUNTIME.GEN_TASKS
    finally:
        with RUNTIME.GEN_LOCK:
            RUNTIME.GEN_TASKS.clear()
            RUNTIME.GEN_TASKS.update(old_tasks)
