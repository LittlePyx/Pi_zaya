from __future__ import annotations

import time
import threading
from typing import Optional


BG_LOCK = threading.Lock()
BG_STATE = {
    "queue": [],
    "active_tasks": [],
    "active_count": 0,
    "running": False,
    "done": 0,
    "total": 0,
    "current": "",
    "cur_page_done": 0,
    "cur_page_total": 0,
    "cur_page_msg": "",
    "cancel": False,
    "last": "",
}
BG_THREAD: Optional[threading.Thread] = None
BG_THREADS: list[threading.Thread] = []


GEN_LOCK = threading.Lock()
GEN_TASKS: dict[str, dict] = {}
GEN_TASK_TTL_S = 60 * 60
GEN_TASK_MAX_ITEMS = 256


def _gen_task_running(task: dict) -> bool:
    return str(task.get("status") or "") == "running" and not bool(task.get("answer_ready") or False)


def prune_generation_tasks(*, now: float | None = None, ttl_s: float | None = None, max_items: int | None = None) -> int:
    current_time = time.time() if now is None else float(now)
    ttl = float(ttl_s if ttl_s is not None else GEN_TASK_TTL_S)
    cap = int(max_items if max_items is not None else GEN_TASK_MAX_ITEMS)
    cutoff = current_time - max(1.0, ttl)
    removed = 0
    with GEN_LOCK:
        for session_id, task in list(GEN_TASKS.items()):
            if not isinstance(task, dict) or _gen_task_running(task):
                continue
            try:
                ts = float(task.get("finished_at") or task.get("updated_at") or task.get("created_at") or current_time)
            except Exception:
                ts = current_time
            if ts < cutoff:
                GEN_TASKS.pop(session_id, None)
                removed += 1
        if len(GEN_TASKS) > cap:
            candidates: list[tuple[float, str]] = []
            for session_id, task in GEN_TASKS.items():
                if not isinstance(task, dict) or _gen_task_running(task):
                    continue
                try:
                    ts = float(task.get("finished_at") or task.get("updated_at") or task.get("created_at") or current_time)
                except Exception:
                    ts = current_time
                candidates.append((ts, session_id))
            overflow = len(GEN_TASKS) - cap
            for _ts, session_id in sorted(candidates)[: max(0, overflow)]:
                GEN_TASKS.pop(session_id, None)
                removed += 1
    return removed


# Citation metadata async worker state (used by ui/refs_renderer.py).
CITATION_LOCK = threading.Lock()
CITATION_TASKS: dict[str, dict] = {}


CACHE_LOCK = threading.Lock()
CACHE: dict[str, dict] = {
    "file_text": {},
    "deep_read": {},
    "trans": {},
    "rerank": {},
    "refs_pack": {},
}
