from pathlib import Path

from kb.chat_store import ChatStore


def test_citation_shelf_project_scope_persists_across_conversations(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("paper project")
    conv_a = store.create_conversation("guide a", project_id=project_id)
    conv_b = store.create_conversation("guide b", project_id=project_id)

    saved = store.save_citation_shelf(
        conv_id=conv_a,
        items=[
            {
                "key": "ref-1",
                "main": "High-resolution single-photon imaging",
                "title": "High-resolution single-photon imaging",
                "doi": "10.1038/demo",
                "tags": ["method"],
                "note": "Important upstream method.",
            }
        ],
        open=True,
    )

    assert saved is not None
    assert saved["scope"] == "project"
    assert saved["scope_id"] == project_id
    assert saved["open"] is True
    assert saved["revision"] == 1

    loaded = store.get_citation_shelf(conv_id=conv_b)

    assert loaded is not None
    assert loaded["scope_id"] == project_id
    assert loaded["open"] is True
    assert loaded["items"][0]["key"] == "ref-1"
    assert loaded["items"][0]["tags"] == ["method"]
    assert loaded["items"][0]["note"] == "Important upstream method."


def test_citation_shelf_default_scope_and_delete(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("no project")

    saved = store.save_citation_shelf(
        conv_id=conv_id,
        items=[{"key": "ref-2", "main": "Default shelf item"}],
        open=False,
    )

    assert saved is not None
    assert saved["scope_id"] == "__default__"
    assert saved["items"][0]["main"] == "Default shelf item"

    deleted = store.delete_citation_shelf(conv_id=conv_id)
    assert deleted is not None
    assert deleted["items"] == []
    assert deleted["open"] is False

    loaded = store.get_citation_shelf(conv_id=conv_id)
    assert loaded is not None
    assert loaded["items"] == []
    assert loaded["revision"] == 0


def test_citation_shelf_rejects_missing_conversation_scope(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")

    assert store.get_citation_shelf(conv_id="missing") is None
    assert store.save_citation_shelf(conv_id="missing", items=[]) is None
