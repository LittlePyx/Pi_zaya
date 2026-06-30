from __future__ import annotations

from pathlib import Path

from kb.chat_store import ChatStore


def test_sidebar_snapshot_groups_conversations_and_honors_limit(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Project A")
    other_project_id = store.create_project("Project B")
    root_old = store.create_conversation("root old")
    root_new = store.create_conversation("root new")
    project_old = store.create_conversation("project old", project_id=project_id)
    project_new = store.create_conversation("project new", project_id=project_id)
    other_project = store.create_conversation("other project", project_id=other_project_id)

    with store._connect() as conn:
        for idx, conv_id in enumerate([root_old, root_new, project_old, project_new, other_project], start=1):
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (idx, conv_id))

    snapshot = store.sidebar_snapshot(limit=1)

    assert {project["id"] for project in snapshot["projects"]} == {project_id, other_project_id}
    assert [conv["id"] for conv in snapshot["root_conversations"]] == [root_new]
    assert [conv["id"] for conv in snapshot["project_conversations"][project_id]] == [project_new]
    assert [conv["id"] for conv in snapshot["project_conversations"][other_project_id]] == [other_project]


def test_sidebar_snapshot_filters_archived_by_default(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    active_id = store.create_conversation("active")
    archived_id = store.create_conversation("archived")
    with store._connect() as conn:
        conn.execute("UPDATE conversations SET archived = 1, archived_at = 10, updated_at = 20 WHERE id = ?", (archived_id,))
        conn.execute("UPDATE conversations SET updated_at = 10 WHERE id = ?", (active_id,))

    visible = store.sidebar_snapshot(limit=5)
    with_archived = store.sidebar_snapshot(limit=5, include_archived=True)

    assert [conv["id"] for conv in visible["root_conversations"]] == [active_id]
    assert [conv["id"] for conv in with_archived["root_conversations"]] == [archived_id, active_id]


def test_sidebar_snapshot_keeps_empty_project_group_and_excludes_orphans(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Empty Project")
    orphan_id = store.create_conversation("orphan")
    with store._connect() as conn:
        conn.execute("UPDATE conversations SET project_id = ? WHERE id = ?", ("missing-project", orphan_id))

    snapshot = store.sidebar_snapshot(limit=5)

    assert project_id in snapshot["project_conversations"]
    assert snapshot["project_conversations"][project_id] == []
    assert snapshot["root_conversations"] == []


def test_sidebar_snapshot_archives_excess_conversations_across_all_scopes(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Project")
    root_ids = [store.create_conversation(f"root {idx}") for idx in range(402)]
    project_ids = [store.create_conversation(f"project {idx}", project_id=project_id) for idx in range(402)]

    snapshot = store.sidebar_snapshot(limit=300)

    assert len(snapshot["root_conversations"]) == 300
    assert len(snapshot["project_conversations"][project_id]) == 300
    with store._connect() as conn:
        archived_root = conn.execute(
            "SELECT COUNT(*) AS n FROM conversations WHERE project_id IS NULL AND COALESCE(archived, 0) = 1"
        ).fetchone()["n"]
        archived_project = conn.execute(
            "SELECT COUNT(*) AS n FROM conversations WHERE project_id = ? AND COALESCE(archived, 0) = 1",
            (project_id,),
        ).fetchone()["n"]
        visible_root_ids = {
            row["id"]
            for row in conn.execute(
                "SELECT id FROM conversations WHERE project_id IS NULL AND COALESCE(archived, 0) = 0"
            ).fetchall()
        }
        visible_project_ids = {
            row["id"]
            for row in conn.execute(
                "SELECT id FROM conversations WHERE project_id = ? AND COALESCE(archived, 0) = 0",
                (project_id,),
            ).fetchall()
        }

    assert archived_root == 2
    assert archived_project == 2
    assert root_ids[-1] in visible_root_ids
    assert root_ids[0] not in visible_root_ids
    assert project_ids[-1] in visible_project_ids
    assert project_ids[0] not in visible_project_ids


def test_create_conversation_with_missing_project_falls_back_to_ungrouped(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")

    conv_id = store.create_conversation("stale project", project_id="missing-project")

    conv = store.get_conversation(conv_id)
    assert conv is not None
    assert conv["project_id"] is None
    snapshot = store.sidebar_snapshot(limit=5)
    assert [item["id"] for item in snapshot["root_conversations"]] == [conv_id]


def test_project_names_are_cleaned_for_sidebar_display(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("  Project\x00\nName  " + ("x" * 200))

    project = store.get_project(project_id)
    assert project is not None
    assert project["name"].startswith("Project Name")
    assert "\x00" not in project["name"]
    assert "\n" not in project["name"]
    assert len(project["name"]) == 120

    assert store.rename_project(project_id, "  ") is False
    assert store.rename_project(project_id, " Renamed\tProject ") is True
    snapshot = store.sidebar_snapshot(limit=5)
    names = {item["id"]: item["name"] for item in snapshot["projects"]}
    assert names[project_id] == "Renamed Project"


def test_move_conversation_rejects_missing_project_without_orphaning(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Project")
    conv_id = store.create_conversation("movable", project_id=project_id)

    ok = store.set_conversation_project(conv_id, "missing-project")

    assert ok is False
    assert store.get_conversation(conv_id)["project_id"] == project_id
    snapshot = store.sidebar_snapshot(limit=5)
    assert [item["id"] for item in snapshot["project_conversations"][project_id]] == [conv_id]


def test_delete_project_moves_conversations_to_ungrouped_and_rearchives_root(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Project")
    root_ids = [store.create_conversation(f"root {idx}") for idx in range(399)]
    project_ids = [store.create_conversation(f"project {idx}", project_id=project_id) for idx in range(5)]

    assert store.delete_project(project_id) is True

    with store._connect() as conn:
        active_root = conn.execute(
            "SELECT COUNT(*) AS n FROM conversations WHERE project_id IS NULL AND COALESCE(archived, 0) = 0"
        ).fetchone()["n"]
        newest_project_row = conn.execute(
            "SELECT project_id, archived FROM conversations WHERE id = ?",
            (project_ids[-1],),
        ).fetchone()
        oldest_root_row = conn.execute(
            "SELECT archived FROM conversations WHERE id = ?",
            (root_ids[0],),
        ).fetchone()

    assert active_root == 400
    assert newest_project_row["project_id"] is None
    assert newest_project_row["archived"] == 0
    assert oldest_root_row["archived"] == 1
