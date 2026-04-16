from __future__ import annotations

from threading import Lock

from kb.bg_queue_state import (
    begin_next_task_or_idle,
    enqueue,
    finish_task,
    is_running_snapshot,
    remove_queued_tasks_for_pdf,
    snapshot,
    update_page_progress,
)


def _make_state() -> dict:
    return {
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


def test_bg_queue_state_tracks_multiple_active_tasks_and_legacy_summary():
    state = _make_state()
    lock = Lock()

    enqueue(state, lock, {"_tid": "t1", "pdf": "a.pdf", "name": "a.pdf", "replace": False})
    enqueue(state, lock, {"_tid": "t2", "pdf": "b.pdf", "name": "b.pdf", "replace": True})
    enqueue(state, lock, {"_tid": "t3", "pdf": "c.pdf", "name": "c.pdf", "replace": False})

    task1 = begin_next_task_or_idle(state, lock)
    task2 = begin_next_task_or_idle(state, lock)
    assert task1 is not None and task1["_tid"] == "t1"
    assert task2 is not None and task2["_tid"] == "t2"

    update_page_progress(state, lock, 1, 4, "page 1", task_id="t1")
    update_page_progress(state, lock, 2, 5, "page 2", task_id="t2")

    snap = snapshot(state, lock)
    assert snap["running"] is True
    assert snap["active_count"] == 2
    assert snap["current"] == "a.pdf"
    assert len(list(snap.get("active_tasks") or [])) == 2
    second = list(snap.get("active_tasks") or [])[1]
    assert second["name"] == "b.pdf"
    assert second["cur_page_done"] == 2
    assert second["cur_page_total"] == 5

    finish_task(state, lock, "OK: a", task_id="t1")
    snap = snapshot(state, lock)
    assert snap["done"] == 1
    assert snap["running"] is True
    assert snap["active_count"] == 1
    assert snap["current"] == "b.pdf"

    finish_task(state, lock, "OK: b", task_id="t2")
    begin_next_task_or_idle(state, lock)
    finish_task(state, lock, "OK: c", task_id="t3")
    snap = snapshot(state, lock)
    assert snap["done"] == 3
    assert snap["running"] is False
    assert snap["active_count"] == 0
    assert snap["current"] == ""
    assert is_running_snapshot(snap) is False


def test_remove_queued_tasks_for_pdf_preserves_done_plus_active_total():
    state = _make_state()
    lock = Lock()

    enqueue(state, lock, {"_tid": "t1", "pdf": "a.pdf", "name": "a.pdf", "replace": False})
    enqueue(state, lock, {"_tid": "t2", "pdf": "b.pdf", "name": "b.pdf", "replace": False})

    begin_next_task_or_idle(state, lock)
    removed = remove_queued_tasks_for_pdf(state, lock, "b.pdf")
    snap = snapshot(state, lock)

    assert removed == 1
    assert snap["total"] == 1
    assert snap["active_count"] == 1
    assert len(list(snap.get("queue") or [])) == 0


def test_update_page_progress_allows_stage_message_after_pages_finish():
    state = _make_state()
    lock = Lock()

    enqueue(state, lock, {"_tid": "t1", "pdf": "a.pdf", "name": "a.pdf", "replace": False})
    begin_next_task_or_idle(state, lock)

    update_page_progress(state, lock, 11, 11, "page 11/11", task_id="t1")
    update_page_progress(state, lock, 11, 11, "ingesting: updating knowledge base index", task_id="t1")

    snap = snapshot(state, lock)
    assert snap["cur_page_done"] == 11
    assert snap["cur_page_total"] == 11
    assert snap["cur_page_msg"] == "ingesting: updating knowledge base index"
