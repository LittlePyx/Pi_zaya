from __future__ import annotations

import json
from pathlib import Path

from api.provider_connection_state import load_connection_check, remember_connection_check
from kb.reader_session_store import ReaderSessionStore
from kb.store import ready_document_ids, save_docs_index, write_doc_chunks
from kb.chat_store import ChatStore


def test_connection_proof_survives_reload_and_rejects_changed_configuration(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "checks.json"
    monkeypatch.setenv("KB_MODEL_CONNECTION_STATE_PATH", str(path))
    remember_connection_check("text", {"ok": True, "checked_at": 12, "fingerprint": "config-a", "reply": "private", "api_key": "secret"})
    assert load_connection_check("text", "config-a")["ok"] is True
    assert load_connection_check("text", "config-b") is None
    assert "secret" not in path.read_text() and "private" not in path.read_text()
    remember_connection_check("text", {"ok": False, "checked_at": 13, "fingerprint": "config-a", "error_type": "authentication"})
    assert load_connection_check("text", "config-a")["ok"] is False


def test_ready_document_count_requires_published_usable_chunk_artifact(tmp_path: Path) -> None:
    save_docs_index(tmp_path, {
        "empty": {"num_chunks": 0},
        "blocked": {"num_chunks": 1, "index_status": "quality_blocked"},
        "missing": {"num_chunks": 1, "index_status": "ready"},
        "good": {"num_chunks": 1, "index_status": "ready"},
    })
    write_doc_chunks(tmp_path, "good", [{"id": "good:1", "text": "evidence", "meta": {}}])
    write_doc_chunks(tmp_path, "blocked", [{"id": "blocked:1", "text": "unsafe", "meta": {}}])
    assert ready_document_ids(tmp_path) == ["good"]


def test_reader_open_does_not_count_until_bound_evidence_is_located(tmp_path: Path) -> None:
    store = ReaderSessionStore(tmp_path / "readers.json")
    record = store.create({"sourcePath": "C:/papers/a.md", "snippet": "exact source quote"}, conversation_id="conversation")
    assert store.located_sources() == []
    store.update_state(record["id"], {"locateResult": {"ok": True, "status": "source_only", "sourcePath": "C:/papers/a.md"}})
    assert store.located_sources() == []
    store.update_state(record["id"], {"locateResult": {"ok": True, "status": "exact", "sourcePath": "C:/papers/b.md"}})
    assert store.located_sources() == []
    store.update_state(record["id"], {"locateResult": {"ok": True, "status": "exact", "sourcePath": "c:\\papers\\a.md"}})
    assert store.located_sources() == [("conversation", "c:/papers/a.md")]
    assert json.loads((tmp_path / "readers.json").read_text())["sessions"]


def test_locate_must_belong_to_an_answer_evidence_source(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chat.sqlite3")
    cid = store.create_conversation()
    question = store.append_message(cid, "user", "question")
    store.append_message(cid, "assistant", "source-backed answer [1]")
    store.upsert_message_refs(user_msg_id=question, conv_id=cid, prompt="question", prompt_sig="test", scores=[1.0], used_query="question", used_translation=False, hits=[{"id": "a:1", "text": "source quote", "meta": {"source_path": "C:/papers/a.md"}}])
    assert store.count_located_answer_sources([(cid, "c:\\papers\\a.md")]) == 1
    assert store.count_located_answer_sources([(cid, "C:/papers/b.md")]) == 0
    assert store.count_located_answer_sources([("other-conversation", "C:/papers/a.md")]) == 0
