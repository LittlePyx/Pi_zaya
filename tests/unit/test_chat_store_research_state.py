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


def test_conversation_research_state_sanitizes_unstable_payload(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("research payload")
    long_text = "r" * 6000

    saved = store.patch_conversation_research_state(
        conv_id,
        {
            "selected_research_context": {
                "id": "ctx-large",
                "score": float("nan"),
                "items": [{"key": "ref-1", "quote": long_text}],
                "values": list(range(600)),
            },
            "empty_list": [],
        },
    )

    assert saved is not None
    state = saved["state"]
    context = state["selected_research_context"]
    assert context["score"] is None
    assert len(context["items"][0]["quote"]) == 4000
    assert len(context["values"]) == 500
    assert state["empty_list"] == []

    loaded = store.get_conversation_research_state(conv_id)
    assert loaded is not None
    assert loaded["state"]["selected_research_context"]["score"] is None


def test_conversation_reader_state_sanitizes_payload_without_dropping_empty_values(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("reader payload")
    long_text = "h" * 6000

    saved = store.patch_conversation_reader_state(
        conv_id,
        "paper.en.md",
        {
            "highlights": [],
            "scroll": {"ratio": float("inf"), "block": "intro"},
            "debug": long_text,
            "ids": list(range(600)),
        },
    )

    assert saved is not None
    state = saved["state"]
    assert state["highlights"] == []
    assert state["scroll"]["ratio"] is None
    assert state["scroll"]["block"] == "intro"
    assert len(state["debug"]) == 4000
    assert len(state["ids"]) == 500

    loaded = store.get_conversation_reader_state(conv_id, "paper.en.md")
    assert loaded is not None
    assert loaded["state"]["scroll"]["ratio"] is None


def test_conversation_reader_state_patch_none_clears_top_level_key(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("reader clear")

    saved = store.patch_conversation_reader_state(
        conv_id,
        "paper.en.md",
        {
            "highlights": [{"id": "h1", "text": "important"}],
            "selection": {"text": "selected"},
        },
    )
    assert saved is not None
    assert "highlights" in saved["state"]

    cleared = store.patch_conversation_reader_state(conv_id, "paper.en.md", {"highlights": None})

    assert cleared is not None
    assert "highlights" not in cleared["state"]
    assert cleared["state"]["selection"]["text"] == "selected"


def test_conversation_reader_state_matches_source_path_variants(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("reader path variants")
    canonical = "F:/Research Papers/Fixture/Paper.en.md"
    file_url = "file:///F:/Research%20Papers/Fixture/Paper.en.md?download=1#selection"

    saved = store.patch_conversation_reader_state(
        conv_id,
        file_url,
        {"highlights": [{"id": "h1", "text": "important sentence"}]},
    )
    loaded = store.get_conversation_reader_state(
        conv_id,
        r"F:\Research Papers\.\Fixture\Paper.en.md",
    )
    updated = store.patch_conversation_reader_state(
        conv_id,
        canonical,
        {"scroll": {"block": "intro"}},
    )

    assert saved is not None
    assert saved["source_path"] == file_url
    assert loaded is not None
    assert loaded["source_path"] == r"F:\Research Papers\.\Fixture\Paper.en.md"
    assert loaded["state"]["highlights"][0]["id"] == "h1"
    assert updated is not None
    assert updated["state"]["highlights"][0]["text"] == "important sentence"
    assert updated["state"]["scroll"]["block"] == "intro"

    with store._connect() as conn:
        rows = conn.execute(
            "SELECT source_path FROM conversation_reader_states WHERE conv_id = ?",
            (conv_id,),
        ).fetchall()
    assert [row["source_path"] for row in rows] == ["f:/research papers/fixture/paper.en.md"]


def test_conversation_reader_state_preserves_literal_hash_in_file_identity(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("reader hash path")
    file_url = "file:///F:/Research%20Papers/Fixture/A%23B.en.md?download=1#selection"
    canonical = "F:/Research Papers/Fixture/A#B.en.md"

    saved = store.patch_conversation_reader_state(
        conv_id,
        file_url,
        {"highlights": [{"id": "hash", "text": "important sentence"}]},
    )
    loaded = store.get_conversation_reader_state(conv_id, canonical)

    assert saved is not None
    assert loaded is not None
    assert loaded["state"]["highlights"][0]["id"] == "hash"
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT source_path FROM conversation_reader_states WHERE conv_id = ?",
            (conv_id,),
        ).fetchall()
    assert [row["source_path"] for row in rows] == ["f:/research papers/fixture/a#b.en.md"]
