from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from kb.chat_store import ChatStore


def test_chat_store_rejects_message_for_missing_conversation(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")

    with pytest.raises(sqlite3.IntegrityError):
        store.append_message("missing-conversation", "user", "hello")


def test_chat_store_rejects_source_binding_for_missing_conversation(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")

    assert store.bind_conversation_source("missing-conversation", "paper.en.md", "paper.pdf") is False
    assert store.list_conversation_sources("missing-conversation") == []

    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversation_sources").fetchone()[0] == 0


def test_chat_store_cleans_legacy_orphan_conversation_sources(tmp_path: Path):
    db_path = tmp_path / "chat.sqlite3"
    store = ChatStore(db_path)
    conv_id = store.create_conversation("with source")
    assert store.bind_conversation_source(conv_id, "paper.en.md", "paper.pdf") is True

    with store._connect() as conn:
        conn.execute("PRAGMA foreign_keys=OFF;")
        conn.execute(
            """
            INSERT INTO conversation_sources (conv_id, source_path, source_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("missing-conversation", "orphan.en.md", "orphan.pdf", 1.0, 1.0),
        )

    reopened = ChatStore(db_path)

    assert reopened.list_conversation_sources(conv_id)[0]["source_path"] == "paper.en.md"
    assert reopened.list_conversation_sources("missing-conversation") == []
    with reopened._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM conversation_sources WHERE conv_id = ?",
            ("missing-conversation",),
        ).fetchone()[0] == 0


def test_chat_store_cleans_legacy_orphan_conversation_state_and_refs(tmp_path: Path):
    db_path = tmp_path / "chat.sqlite3"
    store = ChatStore(db_path)
    conv_id = store.create_conversation("kept conversation")
    user_msg_id = store.append_message(conv_id, "user", "What is the method?")
    assistant_msg_id = store.append_message(conv_id, "assistant", "A draft answer.")
    assert user_msg_id > 0
    assert assistant_msg_id > 0
    assert store.upsert_message_refs(
        user_msg_id=user_msg_id,
        conv_id=conv_id,
        prompt="What is the method?",
        prompt_sig="sig",
        hits=[{"source": "paper.en.md"}],
        scores=[1.0],
        used_query="method",
        used_translation=False,
    ) is True
    assert store.patch_conversation_reader_state(conv_id, "paper.en.md", {"scroll": {"block": "intro"}}) is not None
    assert store.patch_conversation_research_state(conv_id, {"selected": {"id": "ctx-1"}}) is not None
    assert store.save_citation_shelf(
        conv_id=conv_id,
        scope="conversation",
        items=[{"key": "keep-ref", "main": "Kept reference"}],
        open=True,
    ) is not None

    with store._connect() as conn:
        conn.execute("PRAGMA foreign_keys=OFF;")
        conn.execute(
            """
            INSERT INTO conversation_reader_states
            (conv_id, source_path, state_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("missing-conversation", "orphan.en.md", '{"scroll": true}', 1.0, 1.0),
        )
        conn.execute(
            """
            INSERT INTO conversation_research_states
            (conv_id, state_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            ("missing-conversation", '{"selected": true}', 1.0, 1.0),
        )
        conn.execute(
            """
            INSERT INTO citation_shelves
            (scope, scope_id, items_json, open, revision, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("conversation", "missing-conversation", '[{"key":"orphan-ref"}]', 1, 1, 1.0, 1.0),
        )
        conn.execute(
            """
            INSERT INTO message_refs
            (user_msg_id, conv_id, prompt, prompt_sig, hits_json, scores_json,
             used_query, used_translation, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (assistant_msg_id, conv_id, "assistant refs", "bad", "[]", "[]", "assistant", 0, 1.0, 1.0),
        )
        conn.execute(
            """
            INSERT INTO message_refs
            (user_msg_id, conv_id, prompt, prompt_sig, hits_json, scores_json,
             used_query, used_translation, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (9999, "missing-conversation", "orphan refs", "bad", "[]", "[]", "orphan", 0, 1.0, 1.0),
        )

    reopened = ChatStore(db_path)

    assert reopened.get_conversation_reader_state(conv_id, "paper.en.md") is not None
    assert reopened.get_conversation_research_state(conv_id) is not None
    assert reopened.get_citation_shelf(conv_id=conv_id, scope="conversation")["items"][0]["key"] == "keep-ref"
    assert user_msg_id in reopened.list_message_refs(conv_id)
    with reopened._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM conversation_reader_states WHERE conv_id = ?",
            ("missing-conversation",),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM conversation_research_states WHERE conv_id = ?",
            ("missing-conversation",),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM citation_shelves WHERE scope = 'conversation' AND scope_id = ?",
            ("missing-conversation",),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM message_refs WHERE user_msg_id IN (?, ?)",
            (assistant_msg_id, 9999),
        ).fetchone()[0] == 0
