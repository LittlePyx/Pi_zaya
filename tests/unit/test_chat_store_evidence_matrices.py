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
