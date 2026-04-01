from __future__ import annotations

from pathlib import Path
from threading import Lock
import time
from typing import Any


def _active_tasks(state: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = state.get("active_tasks")
    if isinstance(tasks, list):
        return tasks
    tasks = []
    state["active_tasks"] = tasks
    return tasks


def _task_info_from_active_record(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "_tid": str(rec.get("_tid") or ""),
        "pdf": str(rec.get("pdf") or ""),
        "name": str(rec.get("name") or ""),
        "replace": bool(rec.get("replace", False)),
        "started_at": float(rec.get("started_at") or 0.0),
        "cur_page_done": int(rec.get("cur_page_done", 0) or 0),
        "cur_page_total": int(rec.get("cur_page_total", 0) or 0),
        "cur_page_msg": str(rec.get("cur_page_msg") or ""),
        "cur_profile": str(rec.get("cur_profile") or ""),
        "cur_llm_profile": str(rec.get("cur_llm_profile") or ""),
        "cur_log_tail": list(rec.get("cur_log_tail") or []),
    }


def _sync_legacy_summary_fields(state: dict[str, Any]) -> None:
    active = _active_tasks(state)
    primary = active[0] if active else {}
    state["running"] = bool(active)
    state["active_count"] = len(active)
    state["current"] = str(primary.get("name") or "")
    state["cur_task_id"] = str(primary.get("_tid") or "")
    state["cur_task_replace"] = bool(primary.get("replace", False))
    state["cur_page_done"] = int(primary.get("cur_page_done", 0) or 0)
    state["cur_page_total"] = int(primary.get("cur_page_total", 0) or 0)
    state["cur_page_msg"] = str(primary.get("cur_page_msg") or "")
    state["cur_profile"] = str(primary.get("cur_profile") or "")
    state["cur_llm_profile"] = str(primary.get("cur_llm_profile") or "")
    state["cur_log_tail"] = list(primary.get("cur_log_tail") or [])


def enqueue(state: dict[str, Any], lock: Lock, task: dict[str, Any]) -> None:
    with lock:
        if (not bool(state.get("running"))) and (not state.get("queue")) and (not _active_tasks(state)):
            state["done"] = 0
            state["total"] = 0
            state["last"] = ""
        state.setdefault("queue", []).append(task)
        state["total"] = int(state.get("total", 0)) + 1


def remove_queued_tasks_for_pdf(state: dict[str, Any], lock: Lock, pdf_path: Path) -> int:
    target = str(Path(pdf_path))
    removed = 0
    with lock:
        queue = list(state.get("queue") or [])
        kept: list[dict[str, Any]] = []
        for task in queue:
            try:
                if str(task.get("pdf") or "") == target:
                    removed += 1
                else:
                    kept.append(task)
            except Exception:
                    kept.append(task)
        state["queue"] = kept
        done = int(state.get("done", 0) or 0)
        active_n = len(_active_tasks(state))
        total = int(state.get("total", 0) or 0) - int(removed)
        state["total"] = max(done + active_n, total)
    return removed


def cancel_all(state: dict[str, Any], lock: Lock, message: str) -> None:
    with lock:
        state["cancel"] = True
        state["cur_page_msg"] = message
        for task in _active_tasks(state):
            try:
                task["cur_page_msg"] = str(message or "")
            except Exception:
                pass


def snapshot(state: dict[str, Any], lock: Lock) -> dict[str, Any]:
    with lock:
        _sync_legacy_summary_fields(state)
        snap = dict(state)
        try:
            snap["queue"] = list(state.get("queue") or [])
        except Exception:
            snap["queue"] = []
        try:
            snap["active_tasks"] = [_task_info_from_active_record(rec) for rec in _active_tasks(state)]
        except Exception:
            snap["active_tasks"] = []
        return snap


def begin_next_task_or_idle(state: dict[str, Any], lock: Lock) -> dict[str, Any] | None:
    with lock:
        if state.get("cancel"):
            state.setdefault("queue", []).clear()
            state["total"] = int(state.get("done", 0)) + len(_active_tasks(state))
        state["cancel"] = False

        queue = state.get("queue") or []
        if queue:
            task = queue.pop(0)
            _active_tasks(state).append(
                {
                    "_tid": str(task.get("_tid") or ""),
                    "pdf": str(task.get("pdf") or ""),
                    "name": str(task.get("name") or ""),
                    "replace": bool(task.get("replace", False)),
                    "started_at": float(time.time()),
                    "cur_page_done": 0,
                    "cur_page_total": 0,
                    "cur_page_msg": "",
                    "cur_profile": "",
                    "cur_llm_profile": "",
                    "cur_log_tail": [],
                }
            )
            _sync_legacy_summary_fields(state)
            return task

        _sync_legacy_summary_fields(state)
        return None


def update_page_progress(
    state: dict[str, Any],
    lock: Lock,
    page_done: int,
    page_total: int,
    msg: str = "",
    *,
    task_id: str = "",
) -> None:
    with lock:
        tid = str(task_id or "")
        active = _active_tasks(state)
        target: dict[str, Any] | None = None
        if tid:
            for rec in active:
                if str(rec.get("_tid") or "") == tid:
                    target = rec
                    break
            if target is None:
                return
        elif len(active) == 1:
            target = active[0]
        else:
            return

        old_done = int(target.get("cur_page_done", 0) or 0)
        old_total = int(target.get("cur_page_total", 0) or 0)
        new_done = max(0, int(page_done or 0))
        new_total = max(0, int(page_total or 0))

        total = max(old_total, new_total)
        done = max(old_done, new_done)
        if total > 0:
            done = min(done, total)
        target["cur_page_done"] = int(done)
        target["cur_page_total"] = int(total)
        line = str(msg or "")[:220]
        is_profile = line.startswith("converter profile:") or line.startswith("LLM concurrency:")
        regressed = (new_done < old_done) and (new_total <= old_total) and (not is_profile)
        if regressed:
            line = str(target.get("cur_page_msg") or "")
        target["cur_page_msg"] = line

        if line.startswith("converter profile:"):
            target["cur_profile"] = line
            target["cur_profile_ts"] = float(time.time())
        elif line.startswith("LLM concurrency:"):
            target["cur_llm_profile"] = line
            target["cur_llm_profile_ts"] = float(time.time())

        tail = list(target.get("cur_log_tail") or [])
        if line and (not regressed):
            tail.append(line)
            if len(tail) > 24:
                tail = tail[-24:]
        target["cur_log_tail"] = tail
        _sync_legacy_summary_fields(state)


def should_cancel(state: dict[str, Any], lock: Lock) -> bool:
    with lock:
        return bool(state.get("cancel"))


def finish_task(state: dict[str, Any], lock: Lock, message: str, *, task_id: str = "") -> None:
    with lock:
        tid = str(task_id or "")
        active = _active_tasks(state)
        if tid:
            keep = [rec for rec in active if str(rec.get("_tid") or "") != tid]
            if len(keep) == len(active):
                return
            state["active_tasks"] = keep
        elif active:
            state["active_tasks"] = active[1:]
        state["done"] = int(state.get("done", 0)) + 1
        done = int(state.get("done", 0) or 0)
        total = int(state.get("total", 0) or 0)
        if done > total:
            state["total"] = done
        state["last"] = message
        _sync_legacy_summary_fields(state)


def is_running_snapshot(snap: dict[str, Any]) -> bool:
    if list(snap.get("active_tasks") or []):
        return True
    if bool(snap.get("running")):
        return True
    if str(snap.get("current") or "").strip():
        return True
    if list(snap.get("queue") or []):
        return True
    return False
