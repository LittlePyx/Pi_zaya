from __future__ import annotations

from kb.local_semantic_component_lifecycle_store import (
    LocalSemanticComponentLifecycleStore,
)


def test_default_state_is_bm25_safe_not_installed(tmp_path) -> None:
    record = LocalSemanticComponentLifecycleStore(tmp_path).current()
    assert record.state == "not_installed"
    assert record.public_payload() == {
        "state": "not_installed",
        "available": False,
        "mode": "hybrid_local",
        "reason_id": "semantic_component_not_installed",
        "action_id": "enable",
        "task_id": "",
        "completed_bytes": 0,
        "total_bytes": 0,
        "eta_seconds": None,
        "required_additional_bytes": 0,
        "bm25_available": True,
    }


def test_duplicate_active_start_returns_existing_task(tmp_path) -> None:
    store = LocalSemanticComponentLifecycleStore(tmp_path)
    first, created = store.start_or_get(
        operation="enable",
        initial_state="checking_capacity",
        manifest_identity="manifest-v1",
        owner_session="one",
        consented=True,
        total_bytes=100,
    )
    duplicate, duplicate_created = store.start_or_get(
        operation="enable",
        initial_state="checking_capacity",
        manifest_identity="manifest-v1",
        owner_session="two",
        consented=True,
    )
    assert created is True and duplicate_created is False
    assert duplicate.task_id == first.task_id


def test_public_payload_hides_internal_integrity_failure(tmp_path) -> None:
    store = LocalSemanticComponentLifecycleStore(tmp_path)
    record, _created = store.start_or_get(
        operation="enable",
        initial_state="checking_capacity",
        manifest_identity="manifest-v1",
        owner_session="one",
        consented=True,
    )
    store.transition(
        record.task_id,
        state="degraded_bm25",
        reason_id="component_payload_hash_mismatch",
        finished=True,
    )
    current = store.current()
    assert current.reason_id == "component_payload_hash_mismatch"
    assert current.public_payload()["reason_id"] == (
        "semantic_component_temporarily_unavailable"
    )
    assert current.public_payload()["action_id"] == "none"
    assert current.public_payload()["bm25_available"] is True


def test_cancel_remains_paused_instead_of_auto_resuming_after_restart(tmp_path) -> None:
    store = LocalSemanticComponentLifecycleStore(tmp_path)
    record, _created = store.start_or_get(
        operation="enable",
        initial_state="downloading",
        manifest_identity="manifest-v1",
        owner_session="old",
        consented=True,
    )
    assert store.request_cancel(record.task_id) is True
    reconciled, should_resume = LocalSemanticComponentLifecycleStore(
        tmp_path
    ).reconcile_after_restart(owner_session="new")
    assert reconciled.state == "paused_by_user"
    assert reconciled.public_payload()["action_id"] == "resume"
    assert should_resume is False


def test_non_user_interruption_recovers_automatically_after_restart(tmp_path) -> None:
    store = LocalSemanticComponentLifecycleStore(tmp_path)
    record, _created = store.start_or_get(
        operation="enable",
        initial_state="installing",
        manifest_identity="manifest-v1",
        owner_session="old",
        consented=True,
    )
    reconciled, should_resume = LocalSemanticComponentLifecycleStore(
        tmp_path
    ).reconcile_after_restart(owner_session="new")
    assert reconciled.task_id == record.task_id
    assert reconciled.state == "recovering"
    assert reconciled.public_payload()["state"] == "preparing"
    assert should_resume is True


def test_remove_request_survives_restart_and_is_not_treated_as_user_pause(
    tmp_path,
) -> None:
    store = LocalSemanticComponentLifecycleStore(tmp_path)
    record, _created = store.start_or_get(
        operation="enable",
        initial_state="downloading",
        manifest_identity="manifest-v1",
        owner_session="old",
        consented=True,
    )
    assert store.request_remove(record.task_id) is True
    reconciled, should_resume = LocalSemanticComponentLifecycleStore(
        tmp_path
    ).reconcile_after_restart(owner_session="new")
    assert reconciled.operation == "remove"
    assert reconciled.state == "recovering"
    assert should_resume is True


def test_quarantine_is_scoped_to_exact_manifest_identity(tmp_path) -> None:
    store = LocalSemanticComponentLifecycleStore(tmp_path)
    store.quarantine("release-one:sha", reason_id="component_archive_hash_mismatch")
    assert store.quarantine_reason("release-one:sha") == (
        "component_archive_hash_mismatch"
    )
    assert store.quarantine_reason("release-two:sha") == ""


def test_worker_progress_cannot_erase_an_accepted_cancel_or_remove(tmp_path):
    for operation in ("cancel", "remove"):
        store = LocalSemanticComponentLifecycleStore(tmp_path / operation)
        record, _ = store.start_or_get(operation="enable", initial_state="indexing",
            manifest_identity="manifest-v1", owner_session="one", consented=True)
        request = store.request_cancel if operation == "cancel" else store.request_remove
        assert request(record.task_id)
        for state in ("indexing", "ready_to_index", "ready", "recovering"):
            assert not store.transition(record.task_id, state=state, reason_id="late_worker_progress")
            assert store.should_cancel(record.task_id)
        terminal = "paused_by_user" if operation == "cancel" else "removing"
        assert store.transition(record.task_id, state=terminal, reason_id="accepted_user_action")
        assert store.current().state == terminal


def test_explicit_removal_can_finish_without_an_intermediate_worker_state(tmp_path):
    store = LocalSemanticComponentLifecycleStore(tmp_path)
    record, _ = store.start_or_get(operation="enable", initial_state="indexing",
        manifest_identity="manifest-v1", owner_session="one", consented=True)
    assert store.request_remove(record.task_id)
    assert store.transition(record.task_id, state="not_installed", reason_id="component_removed", finished=True)
    assert store.current().state == "not_installed"
