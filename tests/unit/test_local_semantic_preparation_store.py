from __future__ import annotations

from pathlib import Path

from kb.local_semantic_preparation_store import LocalSemanticPreparationStore


def test_empty_store_has_actionable_not_prepared_status(tmp_path: Path) -> None:
    status = LocalSemanticPreparationStore(tmp_path).current()
    assert status.state == "not_prepared"
    assert status.action_id == "prepare"
    assert status.available is False


def test_duplicate_start_returns_active_task(tmp_path: Path) -> None:
    store = LocalSemanticPreparationStore(tmp_path)
    first, created = store.start_or_get(
        operation="prepare",
        owner_session="one",
        total_units=6,
        total_bytes=10,
    )
    duplicate, duplicate_created = store.start_or_get(
        operation="retry",
        owner_session="two",
        total_units=6,
        total_bytes=10,
    )
    assert created is True
    assert duplicate_created is False
    assert duplicate.task_id == first.task_id


def test_cancel_wins_over_late_progress_and_finishes_resumable(tmp_path: Path) -> None:
    store = LocalSemanticPreparationStore(tmp_path)
    status, _ = store.start_or_get(operation="prepare", owner_session="one")
    assert store.request_cancel(status.task_id) is True
    assert store.should_cancel(status.task_id) is True
    assert store.update_progress(
        status.task_id,
        state="downloading",
        reason_id="artifact_downloading",
        phase="artifacts",
        completed_units=1,
        total_units=6,
        completed_bytes=4,
        total_bytes=10,
        eta_seconds=2,
    ) is False
    assert store.finish(
        status.task_id,
        state="interrupted",
        reason_id="cancelled_by_user",
    ) is True
    current = store.current()
    assert current.state == "interrupted"
    assert current.action_id == "resume"


def test_restart_reconciles_foreign_active_task_once(tmp_path: Path) -> None:
    store = LocalSemanticPreparationStore(tmp_path)
    status, _ = store.start_or_get(operation="prepare", owner_session="old")
    reconciled = store.reconcile_after_restart(owner_session="new")
    assert reconciled.task_id == status.task_id
    assert reconciled.state == "interrupted"
    assert reconciled.reason_id == "app_restarted"
    assert store.reconcile_after_restart(owner_session="new").state == "interrupted"


def test_progress_and_ready_counts_are_persisted_without_paths(tmp_path: Path) -> None:
    store = LocalSemanticPreparationStore(tmp_path)
    status, _ = store.start_or_get(operation="prepare", owner_session="one")
    assert store.update_progress(
        status.task_id,
        state="indexing",
        reason_id="index_building",
        phase="index",
        completed_units=2,
        total_units=3,
        completed_bytes=8,
        total_bytes=10,
        eta_seconds=1,
        document_count=2,
        chunk_count=7,
    ) is True
    assert store.finish(
        status.task_id,
        state="ready",
        reason_id="ready",
        document_count=2,
        chunk_count=7,
    ) is True
    payload = store.current().public_payload()
    assert payload["available"] is True
    assert payload["action_id"] == "none"
    assert payload["document_count"] == 2
    assert payload["chunk_count"] == 7
    assert not any("path" in key or "error" in key for key in payload)


def test_corrupt_future_state_fails_closed_with_retry_action(tmp_path: Path) -> None:
    store = LocalSemanticPreparationStore(tmp_path)
    status, _ = store.start_or_get(operation="prepare", owner_session="one")
    with store._connect() as connection:
        connection.execute(
            "UPDATE local_semantic_preparation_jobs SET state = 'future' WHERE task_id = ?",
            (status.task_id,),
        )
    current = store.current()
    assert current.state == "repairable_error"
    assert current.reason_id == "preparation_state_invalid"
    assert current.action_id == "retry"
