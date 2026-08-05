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
