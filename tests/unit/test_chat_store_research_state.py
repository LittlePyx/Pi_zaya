from pathlib import Path

from kb.chat_store import ChatStore


def test_conversation_research_state_roundtrip_and_clear(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("research state")

    empty = store.get_conversation_research_state(conv_id)
    assert empty is not None
    assert empty["conv_id"] == conv_id
    assert empty["state"] == {}

    saved = store.patch_conversation_research_state(
        conv_id,
        {
            "selected_research_context": {
                "version": 1,
                "id": "ctx-1",
                "source": "citation_shelf",
                "items": [{"key": "ref-1", "title": "Reference one"}],
            },
            "reader": {"open": True},
        },
    )
    assert saved is not None
    assert saved["state"]["selected_research_context"]["id"] == "ctx-1"
    assert saved["state"]["reader"]["open"] is True

    loaded = store.get_conversation_research_state(conv_id)
    assert loaded is not None
    assert loaded["state"]["selected_research_context"]["items"][0]["key"] == "ref-1"

    cleared = store.patch_conversation_research_state(conv_id, {"selected_research_context": None})
    assert cleared is not None
    assert "selected_research_context" not in cleared["state"]
    assert cleared["state"]["reader"]["open"] is True


def test_conversation_research_state_rejects_missing_conversation(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")

    assert store.get_conversation_research_state("missing") is None
    assert store.patch_conversation_research_state("missing", {"selected_research_context": {}}) is None
