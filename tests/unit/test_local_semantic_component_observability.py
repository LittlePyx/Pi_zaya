from __future__ import annotations

import json
import time

from kb.local_semantic_component_lifecycle_contract import (
    LocalSemanticComponentLifecycleRecord,
)
from kb.local_semantic_component_observability import (
    record_local_semantic_component_observation,
)


def _lines(path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_component_observation_redacts_identity_and_records_closed_outcomes(
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "component.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))
    started = LocalSemanticComponentLifecycleRecord(
        state="checking_capacity",
        reason_id="component_capacity_check",
        task_id="private-task-id",
        operation="enable",
        manifest_identity="private-manifest",
        owner_session="private-session",
        consented=True,
    )
    failed = LocalSemanticComponentLifecycleRecord(
        state="degraded_bm25",
        reason_id="component_payload_hash_mismatch",
        task_id="private-task-id",
        operation="enable",
        manifest_identity="private-manifest",
        owner_session="private-session",
        attempt=2,
        consented=True,
    )

    assert record_local_semantic_component_observation(started, phase="started")
    assert record_local_semantic_component_observation(
        failed,
        phase="terminal",
        started_at=time.perf_counter(),
    )

    stored_text = path.read_text(encoding="utf-8")
    events = _lines(path)
    assert [event["attributes"]["outcome_id"] for event in events] == [
        "local_semantic_component.started",
        "local_semantic_component.bm25_contained",
    ]
    terminal = events[-1]["attributes"]
    assert terminal["fallback_reason_id"] == "component_integrity"
    assert terminal["attempt_count"] == 2
    assert terminal["consented"] is True
    assert terminal["bm25_available"] is True
    assert terminal["duration_ms"] >= 0
    assert "private-task-id" not in stored_text
    assert "private-manifest" not in stored_text
    assert "private-session" not in stored_text
    assert "component_payload_hash_mismatch" not in stored_text


def test_component_observation_ignores_nonterminal_progress_and_unknown_phase(
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "component.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))
    progress = LocalSemanticComponentLifecycleRecord(
        state="downloading",
        operation="enable",
        consented=True,
    )

    assert not record_local_semantic_component_observation(progress, phase="terminal")
    assert not record_local_semantic_component_observation(progress, phase="progress")
    assert not path.exists()


def test_component_observation_uses_closed_storage_and_user_pause_categories(
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "component.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))

    for state, reason in (
        ("needs_space", "semantic_component_needs_space"),
        ("paused_by_user", "cancelled_by_user"),
    ):
        assert record_local_semantic_component_observation(
            LocalSemanticComponentLifecycleRecord(
                state=state,
                reason_id=reason,
                operation="resume",
                consented=True,
            ),
            phase="terminal",
        )

    attributes = [event["attributes"] for event in _lines(path)]
    assert [item["fallback_reason_id"] for item in attributes] == [
        "storage_capacity",
        "user_pause",
    ]
    assert all(item["bm25_available"] is True for item in attributes)
