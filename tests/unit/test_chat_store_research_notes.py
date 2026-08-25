from pathlib import Path
import sqlite3

from kb.chat_store import ChatStore


def test_research_notes_persist_source_links_and_guard_revisions(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Imaging notes")
    conv_id = store.create_conversation("Grounded answer", project_id=project_id)
    created = store.create_research_note(
        project_id=project_id,
        source_conv_id=conv_id,
        title="SPAD noise",
        content_markdown="## Research question\n\nWhy model noise?",
        source_state={
            "version": 1,
            "links": [
                {
                    "kind": "answer",
                    "conversation_id": conv_id,
                    "message_id": 7,
                    "label": "Why model noise?",
                }
            ],
        },
    )

    assert created is not None
    assert created["revision"] == 1
    assert created["source_state"]["links"][0]["message_id"] == 7
    listed = store.list_research_notes(project_id=project_id)
    assert [item["id"] for item in listed] == [created["id"]]
    assert listed[0]["content_markdown"] == ""
    assert listed[0]["source_state"]["links"][0]["conversation_id"] == conv_id

    updated, conflict = store.update_research_note(
        created["id"],
        expected_revision=1,
        title="SPAD noise model",
        content_markdown="Edited note",
    )
    assert conflict is False
    assert updated is not None
    assert updated["revision"] == 2
    assert updated["title"] == "SPAD noise model"

    stale, conflict = store.update_research_note(
        created["id"],
        expected_revision=1,
        content_markdown="Stale overwrite",
    )
    assert conflict is True
    assert stale is not None
    assert stale["revision"] == 2
    assert stale["content_markdown"] == "Edited note"


def test_research_notes_follow_project_scope_and_survive_project_deletion(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Project")
    project_conv = store.create_conversation("Project conversation", project_id=project_id)
    root_conv = store.create_conversation("Root conversation")
    project_note = store.create_research_note(
        project_id=project_id,
        source_conv_id=project_conv,
        title="Project note",
        content_markdown="Project content",
    )
    root_note = store.create_research_note(
        source_conv_id=root_conv,
        title="Root note",
        content_markdown="Root content",
    )

    assert project_note is not None
    assert root_note is not None
    assert [item["id"] for item in store.list_research_notes(project_id=project_id)] == [project_note["id"]]
    assert [item["id"] for item in store.list_research_notes()] == [root_note["id"]]

    assert store.delete_project(project_id) is True
    preserved = store.get_research_note(project_note["id"])
    assert preserved is not None
    assert preserved["project_id"] is None
    assert preserved["source_conv_id"] == project_conv
    assert {item["id"] for item in store.list_research_notes()} == {project_note["id"], root_note["id"]}


def test_research_note_delete_is_explicit(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chat.sqlite3")
    created = store.create_research_note(title="Delete me", content_markdown="Content")
    assert created is not None
    assert store.delete_research_note(created["id"]) is True
    assert store.get_research_note(created["id"]) is None
    assert store.delete_research_note(created["id"]) is False


def test_research_notes_workspace_metadata_global_search_and_archive(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_a = store.create_project("Imaging")
    project_b = store.create_project("Diffusion")
    note_a = store.create_research_note(
        project_id=project_a,
        title="SPAD model",
        content_markdown="Poisson shot noise evidence",
        tags=["Noise", "noise", "Detector"],
    )
    note_b = store.create_research_note(
        project_id=project_b,
        title="Manifold diffusion",
        content_markdown="First hitting process",
    )
    assert note_a is not None and note_b is not None
    assert note_a["tags"] == ["Noise", "Detector"]

    updated, conflict = store.update_research_note(
        note_b["id"],
        expected_revision=1,
        tags=["Generative models"],
        pinned=True,
        archived=True,
        project_id=None,
        project_id_is_set=True,
    )
    assert conflict is False
    assert updated is not None
    assert updated["project_id"] is None
    assert updated["pinned"] is True
    assert updated["archived"] is True

    active = store.list_research_notes(scope="all", query="poisson", archived=False)
    archived = store.list_research_notes(scope="all", query="generative", archived=True)
    all_notes = store.list_research_notes(scope="all", archived=None)
    assert [item["id"] for item in active] == [note_a["id"]]
    assert [item["id"] for item in archived] == [note_b["id"]]
    assert {item["id"] for item in all_notes} == {note_a["id"], note_b["id"]}


def test_research_notes_old_table_is_migrated_without_data_loss(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE research_notes (
              id TEXT PRIMARY KEY,
              project_id TEXT,
              source_conv_id TEXT,
              title TEXT NOT NULL,
              content_markdown TEXT NOT NULL DEFAULT '',
              source_state_json TEXT NOT NULL DEFAULT '{}',
              revision INTEGER NOT NULL DEFAULT 1,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO research_notes VALUES (?, NULL, NULL, ?, ?, '{}', 1, 1, 1)",
            ("legacy-note", "Legacy note", "Preserved content"),
        )

    store = ChatStore(db_path)
    record = store.get_research_note("legacy-note")
    assert record is not None
    assert record["content_markdown"] == "Preserved content"
    assert record["tags"] == []
    assert record["pinned"] is False
    assert record["archived"] is False
