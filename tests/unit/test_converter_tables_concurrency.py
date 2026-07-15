from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import kb.converter.tables as tables_module


class _ConcurrentProbePage:
    def __init__(self, *, start_barrier: threading.Barrier, state: dict, state_lock: threading.Lock):
        self.rect = tables_module.fitz.Rect(0, 0, 600, 800)
        self._start_barrier = start_barrier
        self._state = state
        self._state_lock = state_lock

    def get_text(self, mode: str, **_kwargs):
        assert mode == "dict"
        self._start_barrier.wait(timeout=2)
        return {"blocks": []}

    def find_tables(self, **_kwargs):
        with self._state_lock:
            self._state["active"] += 1
            self._state["max_active"] = max(self._state["max_active"], self._state["active"])
        try:
            time.sleep(0.05)
            return []
        finally:
            with self._state_lock:
                self._state["active"] -= 1


def test_pymupdf_table_finder_calls_are_serialized_across_pages():
    start_barrier = threading.Barrier(2)
    state = {"active": 0, "max_active": 0}
    state_lock = threading.Lock()
    pages = [
        _ConcurrentProbePage(start_barrier=start_barrier, state=state, state_lock=state_lock)
        for _ in range(2)
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(tables_module._extract_tables_by_layout, pages))

    assert results == [[], []]
    assert state["max_active"] == 1
