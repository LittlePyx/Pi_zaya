from __future__ import annotations

from threading import Lock

from kb.bg_queue_state import (
    begin_next_task_or_idle,
    cancel_all,
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


def test_bg_queue_state_preserves_active_task_repair_context():
    state = _make_state()
    lock = Lock()

    enqueue(
        state,
        lock,
        {
            "_tid": "t1",
            "pdf": "a.pdf",
            "name": "a.pdf",
            "replace": True,
            "repair_context": {
                "action": "reconvert",
                "source": "library_quality_repair",
                "repair_run_id": "run-123",
                "issue_codes": ["weak_structure"],
            },
        },
    )
    begin_next_task_or_idle(state, lock)

    snap = snapshot(state, lock)
    active = list(snap.get("active_tasks") or [])
    assert active[0]["repair_context"]["repair_run_id"] == "run-123"
    assert active[0]["repair_context"]["issue_codes"] == ["weak_structure"]


def test_enqueue_skips_duplicate_queued_pdf():
    state = _make_state()
    lock = Lock()

    first = enqueue(state, lock, {"_tid": "t1", "pdf": "a.pdf", "name": "a.pdf", "replace": False})
    second = enqueue(state, lock, {"_tid": "t2", "pdf": "a.pdf", "name": "a.pdf", "replace": True})
    snap = snapshot(state, lock)

    assert first is True
    assert second is False
    assert snap["total"] == 1
    assert [task["_tid"] for task in snap["queue"]] == ["t1"]


def test_enqueue_skips_duplicate_active_pdf_until_cancel_requested():
    state = _make_state()
    lock = Lock()

    assert enqueue(state, lock, {"_tid": "t1", "pdf": "a.pdf", "name": "a.pdf", "replace": False}) is True
    begin_next_task_or_idle(state, lock)

    duplicate = enqueue(state, lock, {"_tid": "t2", "pdf": "a.pdf", "name": "a.pdf", "replace": True})
    cancel_all(state, lock, "Canceling current background conversion")
    retry_after_cancel = enqueue(state, lock, {"_tid": "t3", "pdf": "a.pdf", "name": "a.pdf", "replace": True})
    snap = snapshot(state, lock)

    assert duplicate is False
    assert retry_after_cancel is True
    assert snap["active_count"] == 1
    assert [task["_tid"] for task in snap["queue"]] == ["t3"]
    assert snap["total"] == 2


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


def test_cancel_all_clears_queued_tasks_but_keeps_active_cancelable():
    state = _make_state()
    lock = Lock()

    enqueue(state, lock, {"_tid": "t1", "pdf": "a.pdf", "name": "a.pdf", "replace": False})
    enqueue(state, lock, {"_tid": "t2", "pdf": "b.pdf", "name": "b.pdf", "replace": False})
    enqueue(state, lock, {"_tid": "t3", "pdf": "c.pdf", "name": "c.pdf", "replace": False})
    begin_next_task_or_idle(state, lock)

    cancel_all(state, lock, "Canceling current background conversion")
    snap = snapshot(state, lock)

    assert snap["cancel"] is True
    assert snap["queue"] == []
    assert snap["total"] == 1
    assert snap["done"] == 0
    assert snap["active_count"] == 1
    assert snap["cur_page_msg"] == "Canceling current background conversion"
    assert snap["last"] == "Canceling current background conversion"
    assert is_running_snapshot(snap) is True

    finish_task(state, lock, "CANCELLED", task_id="t1")
    assert begin_next_task_or_idle(state, lock) is None
    done_snap = snapshot(state, lock)
    assert done_snap["queue"] == []
    assert done_snap["done"] == 1
    assert done_snap["total"] == 1
    assert done_snap["running"] is False


def test_cancel_all_clears_queue_when_no_task_has_started():
    state = _make_state()
    lock = Lock()

    enqueue(state, lock, {"_tid": "t1", "pdf": "a.pdf", "name": "a.pdf", "replace": False})
    enqueue(state, lock, {"_tid": "t2", "pdf": "b.pdf", "name": "b.pdf", "replace": False})

    cancel_all(state, lock, "Canceling queued background conversions")
    snap = snapshot(state, lock)

    assert snap["queue"] == []
    assert snap["active_count"] == 0
    assert snap["cancel"] is False
    assert snap["done"] == 0
    assert snap["total"] == 0
    assert snap["running"] is False
    assert is_running_snapshot(snap) is False


def test_enqueue_after_cancel_waits_for_active_task_without_losing_new_work():
    state = _make_state()
    lock = Lock()

    enqueue(state, lock, {"_tid": "t1", "pdf": "a.pdf", "name": "a.pdf", "replace": False})
    begin_next_task_or_idle(state, lock)
    cancel_all(state, lock, "Canceling current background conversion")
    enqueue(state, lock, {"_tid": "t2", "pdf": "b.pdf", "name": "b.pdf", "replace": False})

    assert begin_next_task_or_idle(state, lock) is None
    canceling_snap = snapshot(state, lock)
    assert canceling_snap["cancel"] is True
    assert [task["_tid"] for task in canceling_snap["queue"]] == ["t2"]
    assert canceling_snap["active_count"] == 1
    assert canceling_snap["total"] == 2

    finish_task(state, lock, "CANCELLED", task_id="t1")
    next_task = begin_next_task_or_idle(state, lock)
    assert next_task is not None
    assert next_task["_tid"] == "t2"
    running_snap = snapshot(state, lock)
    assert running_snap["cancel"] is False
    assert running_snap["queue"] == []
    assert running_snap["active_count"] == 1
    assert running_snap["current"] == "b.pdf"
