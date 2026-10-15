from __future__ import annotations

import pytest

from kb.local_semantic_preparation_contract import (
    LOCAL_SEMANTIC_STATES,
    LocalSemanticPreparationStatus,
    normalize_preparation_state,
    preparation_action,
    preparation_is_active,
    reconcile_preparation_state_after_restart,
)


EXPECTED_ACTIONS = {
    "unavailable_in_build": "update_app",
    "not_prepared": "prepare",
    "downloading": "cancel",
    "verifying": "cancel",
    "ready_to_index": "resume",
    "indexing": "cancel",
    "cancelling": "wait",
    "interrupted": "resume",
    "stale": "rebuild",
    "repairable_error": "retry",
    "ready": "none",
}


def test_every_state_has_one_action_and_only_ready_is_available() -> None:
    assert set(LOCAL_SEMANTIC_STATES) == set(EXPECTED_ACTIONS)
    for state in LOCAL_SEMANTIC_STATES:
        status = LocalSemanticPreparationStatus(state=state, reason_id=state)
        assert preparation_action(state) == EXPECTED_ACTIONS[state]
        assert status.available is (state == "ready")
        assert status.public_payload()["action_id"] == EXPECTED_ACTIONS[state]


@pytest.mark.parametrize("state", ["downloading", "verifying", "indexing", "cancelling"])
def test_active_state_becomes_resumable_interrupted_after_restart(state: str) -> None:
    assert preparation_is_active(state) is True
    assert reconcile_preparation_state_after_restart(state) == "interrupted"
    assert preparation_action(reconcile_preparation_state_after_restart(state)) == "resume"


def test_ready_and_stale_survive_restart_reconciliation() -> None:
    assert reconcile_preparation_state_after_restart("ready") == "ready"
    assert reconcile_preparation_state_after_restart("stale") == "stale"


def test_unknown_state_fails_closed_to_repairable_error() -> None:
    assert normalize_preparation_state("future_state") == "repairable_error"
    assert preparation_action("future_state") == "retry"


def test_public_payload_is_path_free_and_has_deterministic_remediation() -> None:
    payload = LocalSemanticPreparationStatus(
        state="downloading",
        reason_id="artifact_download_in_progress",
        task_id="task-1",
        phase="model",
        completed_units=1,
        total_units=6,
        completed_bytes=1024,
        total_bytes=492421554,
        eta_seconds=42,
    ).public_payload()
    assert payload == {
        "state": "downloading",
        "reason_id": "artifact_download_in_progress",
        "task_id": "task-1",
        "phase": "model",
        "completed_units": 1,
        "total_units": 6,
        "completed_bytes": 1024,
        "total_bytes": 492421554,
        "eta_seconds": 42,
        "document_count": 0,
        "chunk_count": 0,
        "available": False,
        "mode": "hybrid_local",
        "action_id": "cancel",
    }
    assert not any("path" in key or "error" in key for key in payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"completed_units": -1},
        {"completed_units": 2, "total_units": 1},
        {"completed_bytes": -1},
        {"completed_bytes": 2, "total_bytes": 1},
        {"eta_seconds": -1},
    ],
)
def test_invalid_progress_is_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        LocalSemanticPreparationStatus(state="downloading", **kwargs)
