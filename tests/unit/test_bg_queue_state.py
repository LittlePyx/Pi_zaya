from __future__ import annotations

from threading import Lock

from kb.bg_queue_state import (
    begin_next_task_or_idle,
    cancel_all,
    cancel_task,
    enqueue,
    finish_task,
    is_running_snapshot,
    record_task_result,
    remove_queued_tasks_for_pdf,
    should_cancel,
    snapshot,
    update_conversion_stage,
    update_page_progress,
    update_running_pages,
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
        "conversion_stage": "",
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
    assert snap["conversion_stage"] == "indexing"


def test_conversion_public_stage_progresses_without_changing_completed_pages():
    state = _make_state()
    lock = Lock()

    enqueue(state, lock, {"_tid": "t1", "pdf": "a.pdf", "name": "a.pdf", "replace": False})
    begin_next_task_or_idle(state, lock)
    update_page_progress(state, lock, 21, 21, "Finished page 21/21", task_id="t1")
    update_page_progress(state, lock, 21, 21, "[CONVERTER_STAGE] source_check", task_id="t1")

    finalizing = snapshot(state, lock)
    assert finalizing["conversion_stage"] == "finalizing"
    assert finalizing["cur_page_done"] == 21
    assert finalizing["cur_page_total"] == 21

    update_conversion_stage(state, lock, "indexing", task_id="t1")
    indexing = snapshot(state, lock)
    assert indexing["conversion_stage"] == "indexing"
    assert indexing["cur_page_done"] == 21
    assert indexing["cur_page_total"] == 21


def test_update_conversion_stage_rejects_private_or_unknown_values():
    state = _make_state()
    lock = Lock()

    enqueue(state, lock, {"_tid": "t1", "pdf": "a.pdf", "name": "a.pdf", "replace": False})
    begin_next_task_or_idle(state, lock)

    import pytest

    with pytest.raises(ValueError):
        update_conversion_stage(state, lock, "quality_gate_model_failure", task_id="t1")


def test_update_page_progress_keeps_converter_profile_out_of_public_progress_message():
    state = _make_state()
    lock = Lock()

    enqueue(state, lock, {"_tid": "t1", "pdf": "a.pdf", "name": "a.pdf", "replace": False})
    begin_next_task_or_idle(state, lock)
    update_page_progress(state, lock, 0, 12, "converter starting...", task_id="t1")
    raw_profile = "converter profile: script=C:/private/convert.py, workers=8, llm_timeout=300s"
    update_page_progress(state, lock, 0, 0, raw_profile, task_id="t1")

    snap = snapshot(state, lock)
    task = snap["active_tasks"][0]
    assert snap["cur_page_msg"] == "converter starting..."
    assert task["cur_profile"] == raw_profile
    assert raw_profile not in task["cur_log_tail"]

    update_page_progress(state, lock, 0, 12, "converter pid=37664", task_id="t1")
    pid_snap = snapshot(state, lock)
    assert pid_snap["cur_page_msg"] == "converter starting..."
    assert "converter pid=37664" not in pid_snap["active_tasks"][0]["cur_log_tail"]

    update_page_progress(state, lock, 0, 12, "=" * 80, task_id="t1")
    separator_snap = snapshot(state, lock)
    assert separator_snap["cur_page_msg"] == "converter starting..."
    assert "=" * 80 not in separator_snap["active_tasks"][0]["cur_log_tail"]

    update_page_progress(state, lock, 12, 12, "[CONVERTER_TIMING] stage=verifying elapsed_s=1.234", task_id="t1")
    timing_snap = snapshot(state, lock)
    assert timing_snap["cur_page_msg"] == "converter starting..."
    assert not any("CONVERTER_TIMING" in line for line in timing_snap["active_tasks"][0]["cur_log_tail"])


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


def test_cancel_task_isolates_one_active_conversion_and_preserves_sibling():
    state = _make_state()
    lock = Lock()
    enqueue(state, lock, {"_tid": "t1", "pdf": "a.pdf", "name": "a.pdf", "replace": False})
    enqueue(state, lock, {"_tid": "t2", "pdf": "b.pdf", "name": "b.pdf", "replace": False})
    begin_next_task_or_idle(state, lock)
    begin_next_task_or_idle(state, lock)
    update_page_progress(state, lock, 1, 4, "Processing page 2/4 ...", task_id="t1")
    update_page_progress(state, lock, 2, 5, "Processing page 3/5 ...", task_id="t2")

    result = cancel_task(state, lock, "t1", "Canceling selected background conversion")

    assert result == {
        "matched": True,
        "task_id": "t1",
        "state": "cancelling",
        "removed_queued": 0,
    }
    assert should_cancel(state, lock, task_id="t1") is True
    assert should_cancel(state, lock, task_id="t2") is False
    assert should_cancel(state, lock) is False

    # Late callbacks from the cancelled child cannot restore a running stage.
    update_page_progress(state, lock, 3, 4, "Finished page 3/4", task_id="t1")
    update_running_pages(state, lock, [4], task_id="t1")
    update_conversion_stage(state, lock, "indexing", task_id="t1")
    canceling = snapshot(state, lock)
    first, second = canceling["active_tasks"]
    assert first["conversion_stage"] == "cancelling"
    assert first["running_pages"] == []
    assert second["conversion_stage"] == "converting"
    assert second["cur_page_done"] == 2

    finish_task(state, lock, "CANCELLED", task_id="t1")
    update_page_progress(state, lock, 5, 5, "Finished page 5/5", task_id="t2")
    finish_task(state, lock, "OK: b", task_id="t2")
    done = snapshot(state, lock)
    assert done["done"] == 2
    assert done["running"] is False
    assert done["last"] == "OK: b"
    assert [item["task_id"] for item in done["recent_tasks"][:2]] == ["t2", "t1"]
    assert done["recent_tasks"][0]["outcome"] == "success"
    assert done["recent_tasks"][1]["outcome"] == "cancelled"


def test_cancel_task_removes_only_selected_queued_conversion():
    state = _make_state()
    lock = Lock()
    enqueue(state, lock, {"_tid": "t1", "pdf": "a.pdf", "name": "a.pdf", "replace": False})
    enqueue(state, lock, {"_tid": "t2", "pdf": "b.pdf", "name": "b.pdf", "replace": False})
    enqueue(state, lock, {"_tid": "t3", "pdf": "c.pdf", "name": "c.pdf", "replace": False})
    begin_next_task_or_idle(state, lock)

    result = cancel_task(state, lock, "t2", "Canceling selected background conversion")
    missing = cancel_task(state, lock, "missing", "Canceling selected background conversion")
    snap = snapshot(state, lock)

    assert result["state"] == "queued_removed"
    assert result["removed_queued"] == 1
    assert missing["matched"] is False
    assert [task["_tid"] for task in snap["queue"]] == ["t3"]
    assert snap["active_tasks"][0]["_tid"] == "t1"
    assert snap["total"] == 2
    assert snap["recent_tasks"][0]["task_id"] == "t2"
    assert snap["recent_tasks"][0]["outcome"] == "cancelled"
    assert snap["recent_tasks"][0]["retry_action"] == "reconvert"


def test_finish_task_classifies_terminal_results_and_keeps_original_task_metadata():
    cases = [
        ("OK+INGEST: out", "success", ""),
        ("CANCELLED", "cancelled", "reconvert"),
        ("FAIL: provider timed out", "conversion_failed", "reconvert"),
        ("OK+QUALITY_BLOCKED: out", "quality_blocked", "reconvert"),
        ("OK+INGEST_BLOCKED: out", "index_failed", "reindex"),
    ]

    for idx, (message, outcome, retry_action) in enumerate(cases, start=1):
        state = _make_state()
        lock = Lock()
        task_id = f"t{idx}"
        enqueue(
            state,
            lock,
            {
                "_tid": task_id,
                "pdf": f"paper-{idx}.pdf",
                "name": f"paper-{idx}.pdf",
                "replace": True,
                "speed_mode": "balanced",
            },
        )
        begin_next_task_or_idle(state, lock)
        update_page_progress(state, lock, 3, 4, "Finished page 3/4", task_id=task_id)
        finish_task(state, lock, message, task_id=task_id)

        result = snapshot(state, lock)["recent_tasks"][0]
        assert result["task_id"] == task_id
        assert result["name"] == f"paper-{idx}.pdf"
        assert result["outcome"] == outcome
        assert result["retry_action"] == retry_action
        assert result["speed_mode"] == "balanced"
        assert result["page_done"] == 3
        assert result["page_total"] == 4
        assert result["finished_at"] >= result["started_at"]


def test_external_index_retry_result_replaces_duplicate_task_id_and_is_bounded():
    state = _make_state()
    lock = Lock()

    for idx in range(55):
        record_task_result(
            state,
            lock,
            task_id=f"idx-{idx}",
            pdf=f"paper-{idx}.pdf",
            name=f"paper-{idx}.pdf",
            outcome="success",
            message="OK+INDEX_RETRY",
            operation="index_retry",
            started_at=10.0,
            finished_at=12.5,
        )
    record_task_result(
        state,
        lock,
        task_id="idx-54",
        pdf="paper-54.pdf",
        name="paper-54.pdf",
        outcome="index_failed",
        message="FAIL+INDEX_RETRY",
        operation="index_retry",
        detail="disk full",
    )

    recent = snapshot(state, lock)["recent_tasks"]
    assert len(recent) == 50
    assert recent[0]["task_id"] == "idx-54"
    assert recent[0]["outcome"] == "index_failed"
    assert recent[0]["operation"] == "index_retry"
    assert recent[0]["retry_action"] == "reindex"
    assert recent[0]["detail"] == "disk full"
    assert sum(1 for item in recent if item["task_id"] == "idx-54") == 1


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


def test_running_pages_are_structured_isolated_and_cleared_by_later_stages():
    state = _make_state()
    lock = Lock()
    enqueue(state, lock, {"_tid": "t1", "pdf": "a.pdf", "name": "a.pdf", "replace": False})
    enqueue(state, lock, {"_tid": "t2", "pdf": "b.pdf", "name": "b.pdf", "replace": False})
    begin_next_task_or_idle(state, lock)
    begin_next_task_or_idle(state, lock)
    update_page_progress(state, lock, 20, 21, "Processing page 17/21 ...", task_id="t1")
    update_page_progress(state, lock, 4, 8, "Processing page 6/8 ...", task_id="t2")

    update_running_pages(state, lock, [21, 17, 17, -1, 22], task_id="t1")
    update_running_pages(state, lock, [6, 5], task_id="t2")
    snap = snapshot(state, lock)

    assert snap["running_pages"] == [17, 21]
    assert snap["active_tasks"][0]["running_pages"] == [17, 21]
    assert snap["active_tasks"][0]["running_page_count"] == 2
    assert snap["active_tasks"][1]["running_pages"] == [5, 6]

    # Snapshots own their list values and cannot mutate live queue state.
    snap["active_tasks"][0]["running_pages"].append(3)
    assert snapshot(state, lock)["active_tasks"][0]["running_pages"] == [17, 21]

    update_conversion_stage(state, lock, "finalizing", task_id="t1")
    finalizing = snapshot(state, lock)
    assert finalizing["active_tasks"][0]["running_pages"] == []
    assert finalizing["active_tasks"][0]["running_page_count"] == 0


def test_cancel_clears_running_pages():
    state = _make_state()
    lock = Lock()
    enqueue(state, lock, {"_tid": "t1", "pdf": "a.pdf", "name": "a.pdf", "replace": False})
    begin_next_task_or_idle(state, lock)
    update_page_progress(state, lock, 1, 4, "Processing page 2/4 ...", task_id="t1")
    update_running_pages(state, lock, [2, 3], task_id="t1")

    cancel_all(state, lock, "Canceling current background conversion")

    snap = snapshot(state, lock)
    assert snap["running_pages"] == []
    assert snap["active_tasks"][0]["running_pages"] == []
