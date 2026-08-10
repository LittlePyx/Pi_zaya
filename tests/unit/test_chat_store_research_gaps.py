from __future__ import annotations

from pathlib import Path

from kb.chat_store import ChatStore


def test_research_gap_status_persists_and_resolved_gap_reopens(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Gap project")
    gap = {
        "gap_key": "gap-one",
        "project_id": project_id,
        "kind": "missing_cell",
        "priority_score": 48,
        "priority": "low",
        "title": "Missing limitation",
        "dismissible": True,
    }
    opened = store.sync_research_gap_items(project_id=project_id, gaps=[gap])
    assert len(opened) == 1
    gap_id = opened[0]["id"]

    in_progress = store.set_research_gap_status(
        gap_id,
        project_id=project_id,
        status="in_progress",
        action={"candidate_id": "candidate-1"},
    )
    assert in_progress is not None
    assert in_progress["status"] == "in_progress"
    assert in_progress["action"]["candidate_id"] == "candidate-1"
    rescanned = store.sync_research_gap_items(project_id=project_id, gaps=[gap])
    assert rescanned[0]["status"] == "in_progress"
    assert rescanned[0]["action"]["candidate_id"] == "candidate-1"

    assert store.sync_research_gap_items(project_id=project_id, gaps=[]) == []
    assert store.get_research_gap(gap_id)["status"] == "resolved"
    reopened = store.sync_research_gap_items(project_id=project_id, gaps=[gap])
    assert reopened[0]["id"] == gap_id
    assert reopened[0]["status"] == "open"

    ignored = store.set_research_gap_status(
        gap_id,
        project_id=project_id,
        status="ignored",
        action={"ignore_reason": "out of scope"},
    )
    assert ignored is not None
    assert store.sync_research_gap_items(project_id=project_id, gaps=[gap]) == []
    assert store.get_research_gap(gap_id)["status"] == "ignored"

    assert store.delete_project(project_id) is True
    assert store.get_research_gap(gap_id) is None
