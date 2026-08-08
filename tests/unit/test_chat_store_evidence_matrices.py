from pathlib import Path

from kb.chat_store import ChatStore


def test_evidence_matrix_revision_conflict_restore_and_project_cascade(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Imaging review")
    conv_id = store.create_conversation("Evidence", project_id=project_id)
    row = {
        "id": "row-a",
        "paper": "Paper A",
        "source_path": "F:/papers/a.md",
        "notes": "Initial note",
        "cells": {},
    }
    created = store.create_evidence_matrix(
        project_id=project_id,
        source_conv_id=conv_id,
        title="Evidence matrix",
        objective="Compare methods.",
        rows=[row],
        evidence=[{"id": "ev-a", "source_path": "F:/papers/a.md"}],
        source_items=[{"key": "a", "sourcePath": "F:/papers/a.md"}],
        quality_status="verified",
        quality={"supported_cell_count": 1},
    )
    assert created is not None
    assert created["revision"] == 1
    assert created["rows"][0]["notes"] == "Initial note"

    listed = store.list_evidence_matrices(project_id)
    assert listed[0]["id"] == created["id"]
    assert listed[0]["rows"] == []

    row["notes"] = "Updated note"
    updated, conflict = store.update_evidence_matrix(
        created["id"],
        expected_revision=1,
        rows=[row],
    )
    assert conflict is False
    assert updated is not None
    assert updated["revision"] == 2

    stale, conflict = store.update_evidence_matrix(
        created["id"],
        expected_revision=1,
        title="Stale title",
    )
    assert conflict is True
    assert stale is not None
    assert stale["revision"] == 2

    assert [item["revision"] for item in store.list_evidence_matrix_revisions(created["id"])] == [2, 1]
    restored, conflict = store.restore_evidence_matrix_revision(created["id"], 1, expected_revision=2)
    assert conflict is False
    assert restored is not None
    assert restored["revision"] == 3
    assert restored["rows"][0]["notes"] == "Initial note"

    assert store.delete_project(project_id) is True
    assert store.get_evidence_matrix(created["id"]) is None


def test_evidence_watch_events_are_deduplicated_acknowledged_and_cascaded(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Living review")
    matrix = store.create_evidence_matrix(project_id=project_id, title="Tracked matrix")
    assert matrix is not None
    snapshot = {"contract_version": 1, "sources": [], "fingerprint": "base"}
    baseline = store.set_evidence_watch_baseline(
        matrix["id"],
        project_id=project_id,
        matrix_revision=1,
        snapshot=snapshot,
    )
    assert baseline is not None
    assert baseline["snapshot"]["fingerprint"] == "base"
    event = {
        "event_key": "same-change",
        "kind": "source_added",
        "actionable": True,
        "source_identity": "f:/papers/a.md",
        "impact": {"affected_row_ids": []},
    }

    first = store.sync_evidence_watch_events(
        project_id=project_id,
        matrix_id=matrix["id"],
        matrix_revision=1,
        events=[event],
    )
    second = store.sync_evidence_watch_events(
        project_id=project_id,
        matrix_id=matrix["id"],
        matrix_revision=1,
        events=[event],
    )
    assert len(first) == len(second) == 1
    assert first[0]["id"] == second[0]["id"]

    ignored = store.set_evidence_watch_event_status(
        first[0]["id"],
        project_id=project_id,
        status="ignored",
    )
    assert ignored is not None
    assert ignored["status"] == "ignored"
    assert store.sync_evidence_watch_events(
        project_id=project_id,
        matrix_id=matrix["id"],
        matrix_revision=2,
        events=[event],
    ) == []
    assert store.get_evidence_watch_event(first[0]["id"])["matrix_revision"] == 2

    transient = {**event, "event_key": "transient-change", "kind": "source_unavailable"}
    opened = store.sync_evidence_watch_events(
        project_id=project_id,
        matrix_id=matrix["id"],
        matrix_revision=2,
        events=[transient],
    )
    assert len(opened) == 1
    transient_id = opened[0]["id"]
    assert store.sync_evidence_watch_events(
        project_id=project_id,
        matrix_id=matrix["id"],
        matrix_revision=2,
        events=[],
    ) == []
    assert store.get_evidence_watch_event(transient_id)["status"] == "resolved"
    reopened = store.sync_evidence_watch_events(
        project_id=project_id,
        matrix_id=matrix["id"],
        matrix_revision=2,
        events=[transient],
    )
    assert len(reopened) == 1
    assert reopened[0]["id"] == transient_id
    assert reopened[0]["status"] == "open"

    assert store.delete_project(project_id) is True
    assert store.get_evidence_watch_event(first[0]["id"]) is None
