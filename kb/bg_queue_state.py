from __future__ import annotations

from pathlib import Path
from threading import Lock
import time
from typing import Any


PUBLIC_CONVERSION_STAGES = {
    "queued",
    "converting",
    "finalizing",
    "indexing",
    "retrying",
    "cancelling",
}
PUBLIC_RUNNING_PAGES_LIMIT = 12
RECENT_CONVERSION_TASKS_LIMIT = 50
PUBLIC_CONVERSION_OUTCOMES = {
    "success",
    "cancelled",
    "conversion_failed",
    "quality_blocked",
    "index_failed",
}
PUBLIC_CONVERSION_OPERATIONS = {"conversion", "index_retry"}


def _public_conversion_stage(value: Any, *, fallback: str = "") -> str:
    stage = str(value or "").strip().lower()
    if stage in PUBLIC_CONVERSION_STAGES:
        return stage
    fallback_stage = str(fallback or "").strip().lower()
    return fallback_stage if fallback_stage in PUBLIC_CONVERSION_STAGES else ""


def _stage_from_progress_message(message: str, *, current: str = "") -> str:
    line = str(message or "").strip().lower()
    if not line:
        return _public_conversion_stage(current)
    if "cancel" in line:
        return "cancelling"
    if line.startswith("ingesting:"):
        return "indexing"
    if line.startswith("quality gate:") and "retry" in line:
        return "retrying"
    if line.startswith("quality gate:"):
        return "finalizing"
    if line.startswith("[converter_stage]") or line.startswith("post-processing after pages"):
        return "finalizing"
    if line.startswith("processing page") or line.startswith("finished page"):
        if _public_conversion_stage(current) != "retrying":
            return "converting"
    return _public_conversion_stage(current)


def _active_tasks(state: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = state.get("active_tasks")
    if isinstance(tasks, list):
        return tasks
    tasks = []
    state["active_tasks"] = tasks
    return tasks


def _recent_tasks(state: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = state.get("recent_tasks")
    if isinstance(tasks, list):
        return tasks
    tasks = []
    state["recent_tasks"] = tasks
    return tasks


def _terminal_outcome(message: str) -> str:
    text = str(message or "").strip()
    upper = text.upper()
    if upper == "CANCELLED" or upper.startswith("CANCELLED:"):
        return "cancelled"
    if "SOURCE_RETRY_QUALITY_BLOCKED" in upper or "QUALITY_BLOCKED" in upper:
        return "quality_blocked"
    if "INGEST_BLOCKED" in upper:
        return "index_failed"
    if upper.startswith("FAIL"):
        return "conversion_failed"
    return "success" if upper.startswith("OK") else "conversion_failed"


def _terminal_retry_action(outcome: str) -> str:
    clean = str(outcome or "").strip().lower()
    if clean == "index_failed":
        return "reindex"
    if clean in {"cancelled", "conversion_failed", "quality_blocked"}:
        return "reconvert"
    return ""


def _terminal_public_message(outcome: str, detail: str = "", *, operation: str = "conversion") -> str:
    clean = str(outcome or "").strip().lower()
    clean_operation = str(operation or "").strip().lower()
    if clean == "success":
        return "Index retry completed." if clean_operation == "index_retry" else "Conversion and index update completed."
    if clean == "cancelled":
        return "Conversion was cancelled."
    if clean == "quality_blocked":
        return "Conversion completed, but source quality checks blocked indexing."
    if clean == "index_failed":
        return "Index retry failed." if clean_operation == "index_retry" else "Conversion completed, but the index update failed."
    raw = str(detail or "").strip()
    if raw.upper().startswith("FAIL"):
        raw = raw.split(":", 1)[-1].strip() if ":" in raw else raw
    return raw[:500] or "Conversion failed."


def _append_task_result_unlocked(
    state: dict[str, Any],
    *,
    task_id: str,
    pdf: str,
    name: str,
    outcome: str,
    message: str,
    operation: str = "conversion",
    replace: bool = False,
    speed_mode: str = "",
    started_at: float = 0.0,
    finished_at: float | None = None,
    page_done: int = 0,
    page_total: int = 0,
    detail: str = "",
) -> dict[str, Any]:
    clean_outcome = str(outcome or "").strip().lower()
    if clean_outcome not in PUBLIC_CONVERSION_OUTCOMES:
        clean_outcome = "conversion_failed"
    clean_operation = str(operation or "").strip().lower()
    if clean_operation not in PUBLIC_CONVERSION_OPERATIONS:
        clean_operation = "conversion"
    ended = float(finished_at if finished_at is not None else time.time())
    started = max(0.0, float(started_at or 0.0))
    raw_detail = str(detail or "").strip()
    record = {
        "task_id": str(task_id or "").strip(),
        "pdf": str(pdf or ""),
        "name": str(name or Path(str(pdf or "")).name),
        "outcome": clean_outcome,
        "operation": clean_operation,
        "message": _terminal_public_message(
            clean_outcome,
            raw_detail or message,
            operation=clean_operation,
        ),
        "detail": raw_detail[:500],
        "retry_action": _terminal_retry_action(clean_outcome),
        "replace": bool(replace),
        "speed_mode": str(speed_mode or ""),
        "started_at": started,
        "finished_at": ended,
        "duration_s": max(0.0, ended - started) if started > 0 else 0.0,
        "page_done": max(0, int(page_done or 0)),
        "page_total": max(0, int(page_total or 0)),
    }
    recent = _recent_tasks(state)
    tid = str(record.get("task_id") or "")
    if tid:
        recent[:] = [item for item in recent if str((item or {}).get("task_id") or "") != tid]
    recent.insert(0, record)
    del recent[RECENT_CONVERSION_TASKS_LIMIT:]
    return record


def record_task_result(
    state: dict[str, Any],
    lock: Lock,
    **kwargs: Any,
) -> dict[str, Any]:
    """Record a bounded task result produced outside the conversion worker."""
    with lock:
        return dict(_append_task_result_unlocked(state, **kwargs))


def _task_pdf_key(task: dict[str, Any]) -> str:
    raw = str((task or {}).get("pdf") or "").strip()
    if not raw:
        return ""
    try:
        return str(Path(raw).expanduser().resolve(strict=False)).casefold()
    except Exception:
        return raw.casefold()


def _compact_repair_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    issue_codes = [
        str(item or "").strip()
        for item in list(value.get("issue_codes") or [])
        if str(item or "").strip()
    ]
    out = {
        "action": str(value.get("action") or ""),
        "scope": str(value.get("scope") or ""),
        "reason": str(value.get("reason") or ""),
        "source": str(value.get("source") or ""),
        "repair_run_id": str(value.get("repair_run_id") or ""),
        "issue_codes": issue_codes[:30],
    }
    return {
        key: item
        for key, item in out.items()
        if (item if not isinstance(item, list) else bool(item))
    }


def _normalize_running_pages(
    value: Any,
    *,
    total: int = 0,
    limit: int | None = PUBLIC_RUNNING_PAGES_LIMIT,
) -> list[int]:
    rows = value if isinstance(value, (list, tuple, set)) else []
    clean: set[int] = set()
    upper = max(0, int(total or 0))
    for item in rows:
        try:
            page = int(item)
        except Exception:
            continue
        if page <= 0 or (upper > 0 and page > upper):
            continue
        clean.add(page)
    ordered = sorted(clean)
    return ordered if limit is None else ordered[:max(0, int(limit))]


def _task_info_from_active_record(rec: dict[str, Any]) -> dict[str, Any]:
    info = {
        "_tid": str(rec.get("_tid") or ""),
        "pdf": str(rec.get("pdf") or ""),
        "name": str(rec.get("name") or ""),
        "replace": bool(rec.get("replace", False)),
        "started_at": float(rec.get("started_at") or 0.0),
        "cur_page_done": int(rec.get("cur_page_done", 0) or 0),
        "cur_page_total": int(rec.get("cur_page_total", 0) or 0),
        "cur_page_msg": str(rec.get("cur_page_msg") or ""),
        "running_pages": _normalize_running_pages(
            rec.get("running_pages"),
            total=int(rec.get("cur_page_total", 0) or 0),
        ),
        "running_page_count": max(0, int(rec.get("running_page_count", 0) or 0)),
        "conversion_stage": _public_conversion_stage(rec.get("conversion_stage"), fallback="converting"),
        "cur_profile": str(rec.get("cur_profile") or ""),
        "cur_llm_profile": str(rec.get("cur_llm_profile") or ""),
        "cur_log_tail": list(rec.get("cur_log_tail") or []),
    }
    repair_context = _compact_repair_context(rec.get("repair_context"))
    if repair_context:
        info["repair_context"] = repair_context
    return info


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
    state["running_pages"] = _normalize_running_pages(
        primary.get("running_pages"),
        total=int(primary.get("cur_page_total", 0) or 0),
    )
    state["running_page_count"] = max(0, int(primary.get("running_page_count", 0) or 0))
    state["conversion_stage"] = _public_conversion_stage(primary.get("conversion_stage"))
    state["cur_profile"] = str(primary.get("cur_profile") or "")
    state["cur_llm_profile"] = str(primary.get("cur_llm_profile") or "")
    state["cur_log_tail"] = list(primary.get("cur_log_tail") or [])


def enqueue(state: dict[str, Any], lock: Lock, task: dict[str, Any]) -> bool:
    with lock:
        task_key = _task_pdf_key(task)
        if task_key:
            queued_keys = {_task_pdf_key(item) for item in list(state.get("queue") or [])}
            if task_key in queued_keys:
                _sync_legacy_summary_fields(state)
                return False
            if not bool(state.get("cancel")):
                active_keys = {
                    _task_pdf_key(item)
                    for item in _active_tasks(state)
                    if not bool(item.get("cancel"))
                }
                if task_key in active_keys:
                    _sync_legacy_summary_fields(state)
                    return False
        if (not bool(state.get("running"))) and (not state.get("queue")) and (not _active_tasks(state)):
            state["done"] = 0
            state["total"] = 0
            state["last"] = ""
        state.setdefault("queue", []).append(task)
        state["total"] = int(state.get("total", 0)) + 1
        _sync_legacy_summary_fields(state)
        return True


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
        active = _active_tasks(state)
        queued = list(state.get("queue") or [])
        state["cancel"] = bool(active)
        state["cur_page_msg"] = message
        state["last"] = message
        state["queue"] = []
        for task in queued:
            _append_task_result_unlocked(
                state,
                task_id=str((task or {}).get("_tid") or ""),
                pdf=str((task or {}).get("pdf") or ""),
                name=str((task or {}).get("name") or ""),
                outcome="cancelled",
                message="CANCELLED",
                replace=bool((task or {}).get("replace", False)),
                speed_mode=str((task or {}).get("speed_mode") or ""),
            )
        done = int(state.get("done", 0) or 0)
        state["total"] = done + len(active)
        for task in active:
            try:
                task["cancel"] = True
                task["cur_page_msg"] = str(message or "")
                task["conversion_stage"] = "cancelling"
                task["running_pages"] = []
                task["running_page_count"] = 0
            except Exception:
                pass
        _sync_legacy_summary_fields(state)


def cancel_task(
    state: dict[str, Any],
    lock: Lock,
    task_id: str,
    message: str,
) -> dict[str, Any]:
    """Cancel one queued or active task without affecting its siblings."""
    tid = str(task_id or "").strip()
    if not tid:
        return {
            "matched": False,
            "task_id": "",
            "state": "not_found",
            "removed_queued": 0,
        }

    with lock:
        queue = list(state.get("queue") or [])
        kept: list[dict[str, Any]] = []
        removed = 0
        for task in queue:
            if str((task or {}).get("_tid") or "") == tid:
                removed += 1
            else:
                kept.append(task)
        if removed:
            state["queue"] = kept
            done = int(state.get("done", 0) or 0)
            active_n = len(_active_tasks(state))
            total = int(state.get("total", 0) or 0) - removed
            state["total"] = max(done + active_n, total)
            state["last"] = str(message or "")
            removed_task = next(
                (task for task in queue if str((task or {}).get("_tid") or "") == tid),
                {},
            )
            _append_task_result_unlocked(
                state,
                task_id=tid,
                pdf=str((removed_task or {}).get("pdf") or ""),
                name=str((removed_task or {}).get("name") or ""),
                outcome="cancelled",
                message="CANCELLED",
                replace=bool((removed_task or {}).get("replace", False)),
                speed_mode=str((removed_task or {}).get("speed_mode") or ""),
            )
            _sync_legacy_summary_fields(state)
            return {
                "matched": True,
                "task_id": tid,
                "state": "queued_removed",
                "removed_queued": removed,
            }

        for task in _active_tasks(state):
            if str(task.get("_tid") or "") != tid:
                continue
            task["cancel"] = True
            task["cur_page_msg"] = str(message or "")
            task["conversion_stage"] = "cancelling"
            task["running_pages"] = []
            task["running_page_count"] = 0
            state["last"] = str(message or "")
            _sync_legacy_summary_fields(state)
            return {
                "matched": True,
                "task_id": tid,
                "state": "cancelling",
                "removed_queued": 0,
            }

        _sync_legacy_summary_fields(state)
        return {
            "matched": False,
            "task_id": tid,
            "state": "not_found",
            "removed_queued": 0,
        }


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
        try:
            snap["recent_tasks"] = [dict(rec) for rec in _recent_tasks(state)]
        except Exception:
            snap["recent_tasks"] = []
        return snap


def begin_next_task_or_idle(state: dict[str, Any], lock: Lock) -> dict[str, Any] | None:
    with lock:
        if state.get("cancel"):
            if _active_tasks(state):
                _sync_legacy_summary_fields(state)
                return None
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
                    "repair_context": _compact_repair_context(task.get("repair_context")),
                    "started_at": float(time.time()),
                    "cur_page_done": 0,
                    "cur_page_total": 0,
                    "cur_page_msg": "",
                    "running_pages": [],
                    "running_page_count": 0,
                    "conversion_stage": "converting",
                    "cur_profile": "",
                    "cur_llm_profile": "",
                    "cur_log_tail": [],
                    "cancel": False,
                    "speed_mode": str(task.get("speed_mode") or ""),
                    "no_llm": bool(task.get("no_llm", False)),
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

        if bool(target.get("cancel")):
            target["cur_page_msg"] = str(target.get("cur_page_msg") or "")
            target["conversion_stage"] = "cancelling"
            target["running_pages"] = []
            target["running_page_count"] = 0
            _sync_legacy_summary_fields(state)
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
        stripped_line = line.strip()
        is_log_separator = len(stripped_line) >= 8 and set(stripped_line).issubset({"=", "-", "_"})
        is_private_diagnostic = (
            is_profile
            or line.startswith("converter pid=")
            or line.startswith("[CONVERTER_TIMING]")
            or is_log_separator
        )
        if line.startswith("converter profile:"):
            target["cur_profile"] = line
            target["cur_profile_ts"] = float(time.time())
        elif line.startswith("LLM concurrency:"):
            target["cur_llm_profile"] = line
            target["cur_llm_profile_ts"] = float(time.time())
        if is_private_diagnostic:
            _sync_legacy_summary_fields(state)
            return
        regressed = (new_done < old_done) and (new_total <= old_total) and (not is_profile)
        if regressed:
            line = str(target.get("cur_page_msg") or "")
        target["cur_page_msg"] = line
        target["conversion_stage"] = _stage_from_progress_message(
            line,
            current=str(target.get("conversion_stage") or ""),
        )
        if target["conversion_stage"] in {"finalizing", "indexing", "cancelling"}:
            target["running_pages"] = []
            target["running_page_count"] = 0

        tail = list(target.get("cur_log_tail") or [])
        if line and (not regressed):
            tail.append(line)
            if len(tail) > 24:
                tail = tail[-24:]
        target["cur_log_tail"] = tail
        _sync_legacy_summary_fields(state)


def update_running_pages(
    state: dict[str, Any],
    lock: Lock,
    pages: list[int],
    *,
    task_id: str = "",
) -> None:
    with lock:
        tid = str(task_id or "")
        active = _active_tasks(state)
        target: dict[str, Any] | None = None
        if tid:
            target = next((rec for rec in active if str(rec.get("_tid") or "") == tid), None)
        elif len(active) == 1:
            target = active[0]
        if target is None:
            return
        if bool(target.get("cancel")):
            target["running_pages"] = []
            target["running_page_count"] = 0
            target["conversion_stage"] = "cancelling"
            _sync_legacy_summary_fields(state)
            return
        total = int(target.get("cur_page_total", 0) or 0)
        all_pages = _normalize_running_pages(pages, total=total, limit=None)
        target["running_pages"] = all_pages[:PUBLIC_RUNNING_PAGES_LIMIT]
        target["running_page_count"] = len(all_pages)
        _sync_legacy_summary_fields(state)


def update_conversion_stage(
    state: dict[str, Any],
    lock: Lock,
    stage: str,
    *,
    task_id: str = "",
) -> None:
    clean_stage = _public_conversion_stage(stage)
    if not clean_stage:
        raise ValueError(f"unsupported public conversion stage: {stage}")
    with lock:
        tid = str(task_id or "")
        active = _active_tasks(state)
        target: dict[str, Any] | None = None
        if tid:
            target = next((rec for rec in active if str(rec.get("_tid") or "") == tid), None)
        elif len(active) == 1:
            target = active[0]
        if target is None:
            return
        if bool(target.get("cancel")):
            target["conversion_stage"] = "cancelling"
            target["running_pages"] = []
            target["running_page_count"] = 0
            _sync_legacy_summary_fields(state)
            return
        target["conversion_stage"] = clean_stage
        if clean_stage != "converting":
            target["running_pages"] = []
            target["running_page_count"] = 0
        _sync_legacy_summary_fields(state)


def should_cancel(state: dict[str, Any], lock: Lock, *, task_id: str = "") -> bool:
    with lock:
        if bool(state.get("cancel")):
            return True
        tid = str(task_id or "").strip()
        if not tid:
            return False
        return any(
            str(task.get("_tid") or "") == tid and bool(task.get("cancel"))
            for task in _active_tasks(state)
        )


def finish_task(state: dict[str, Any], lock: Lock, message: str, *, task_id: str = "") -> None:
    with lock:
        tid = str(task_id or "")
        active = _active_tasks(state)
        finished: dict[str, Any] | None = None
        if tid:
            finished = next((rec for rec in active if str(rec.get("_tid") or "") == tid), None)
            keep = [rec for rec in active if str(rec.get("_tid") or "") != tid]
            if len(keep) == len(active):
                return
            state["active_tasks"] = keep
        elif active:
            finished = active[0]
            state["active_tasks"] = active[1:]
        if finished is None:
            return
        outcome = _terminal_outcome(message)
        failure_detail = str(message or "") if outcome == "conversion_failed" else ""
        _append_task_result_unlocked(
            state,
            task_id=str(finished.get("_tid") or tid),
            pdf=str(finished.get("pdf") or ""),
            name=str(finished.get("name") or ""),
            outcome=outcome,
            message=str(message or ""),
            replace=bool(finished.get("replace", False)),
            speed_mode=str(finished.get("speed_mode") or ""),
            started_at=float(finished.get("started_at") or 0.0),
            page_done=int(finished.get("cur_page_done", 0) or 0),
            page_total=int(finished.get("cur_page_total", 0) or 0),
            detail=failure_detail,
        )
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
