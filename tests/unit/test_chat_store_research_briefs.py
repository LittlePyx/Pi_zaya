from pathlib import Path

from kb.chat_store import ChatStore


def test_research_brief_revision_conflict_and_restore(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Imaging review")
    conv_id = store.create_conversation("Evidence", project_id=project_id)

    created = store.create_research_brief(
        project_id=project_id,
        source_conv_id=conv_id,
        title="Evidence brief",
        objective="Compare acquisition methods.",
        content_markdown="## Finding\n\nMethod A is faster [1].",
        evidence=[{"citation_number": 1, "source_name": "Paper A"}],
        bibliography=[{"title": "Paper A"}],
        quality_status="verified",
        quality={"support_ratio": 1.0},
    )

    assert created is not None
    assert created["revision"] == 1
    assert created["source_conv_id"] == conv_id
    assert created["evidence"][0]["citation_number"] == 1

    updated, conflict = store.update_research_brief(
        created["id"],
        expected_revision=1,
        content_markdown="## Finding\n\nEdited conclusion [1].",
        quality_status="draft",
    )
    assert conflict is False
    assert updated is not None
    assert updated["revision"] == 2
    assert updated["content_markdown"].endswith("Edited conclusion [1].")

    stale, conflict = store.update_research_brief(
        created["id"],
        expected_revision=1,
        title="Stale overwrite",
    )
    assert conflict is True
    assert stale is not None
    assert stale["revision"] == 2
    assert stale["title"] == "Evidence brief"

    revisions = store.list_research_brief_revisions(created["id"])
    assert [item["revision"] for item in revisions] == [2, 1]

    restored, conflict = store.restore_research_brief_revision(
        created["id"],
        1,
        expected_revision=2,
    )
    assert conflict is False
    assert restored is not None
    assert restored["revision"] == 3
    assert restored["quality_status"] == "verified"
    assert "Method A is faster" in restored["content_markdown"]


def test_research_briefs_are_project_scoped_and_deleted_with_project(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_a = store.create_project("A")
    project_b = store.create_project("B")
    brief_a = store.create_research_brief(project_id=project_a, title="A brief")
    brief_b = store.create_research_brief(project_id=project_b, title="B brief")

    assert brief_a is not None
    assert brief_b is not None
    assert [item["id"] for item in store.list_research_briefs(project_a)] == [brief_a["id"]]
    assert [item["id"] for item in store.list_research_briefs(project_b)] == [brief_b["id"]]

    assert store.delete_project(project_a) is True
    assert store.get_research_brief(brief_a["id"]) is None
    assert store.get_research_brief(brief_b["id"]) is not None


def test_research_brief_update_plans_are_persistent_version_bound_and_superseded(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Incremental updates")
    brief = store.create_research_brief(project_id=project_id, title="Brief", content_markdown="Old [1].")
    assert brief is not None

    first, conflict = store.create_research_brief_update_plan(
        brief["id"],
        expected_revision=1,
        matrix_id="matrix-1",
        matrix_revision=2,
        matrix_fingerprint="fingerprint-2",
        payload={"items": [{"id": "change-1"}], "base_content_hash": "hash"},
    )
    assert conflict is False
    assert first is not None
    assert first["status"] == "open"
    assert store.get_open_research_brief_update_plan(brief["id"])["id"] == first["id"]

    second, conflict = store.create_research_brief_update_plan(
        brief["id"],
        expected_revision=1,
        matrix_id="matrix-1",
        matrix_revision=2,
        matrix_fingerprint="fingerprint-2",
        payload={"items": []},
    )
    assert conflict is False
    assert second is not None
    assert second["id"] != first["id"]
    assert store.get_research_brief_update_plan(brief["id"], first["id"])["status"] == "superseded"
    assert store.get_open_research_brief_update_plan(brief["id"])["id"] == second["id"]

    updated, conflict = store.update_research_brief(
        brief["id"],
        expected_revision=1,
        content_markdown="New [1].",
    )
    assert conflict is False
    assert updated is not None
    stale, conflict = store.create_research_brief_update_plan(
        brief["id"],
        expected_revision=1,
        matrix_id="matrix-1",
        matrix_revision=3,
        matrix_fingerprint="fingerprint-3",
        payload={},
    )
    assert stale["base_brief_revision"] == 2
    assert conflict is True
