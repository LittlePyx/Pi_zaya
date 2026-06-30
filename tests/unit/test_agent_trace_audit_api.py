from pathlib import Path

import pytest
from fastapi import HTTPException

from kb.agent.runner import build_agent_trace_for_completed_answer
from kb.chat_store import ChatStore


def test_message_agent_trace_endpoint_returns_stored_trace(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("agent audit")
    trace = build_agent_trace_for_completed_answer(
        "What is grounded?",
        "The answer is grounded by retrieval [1].",
        evidence_hits=[
            {
                "text": "The answer is grounded by retrieval.",
                "meta": {"source_name": "demo.md", "source_path": "demo.md"},
            }
        ],
        scope_context={"query_scope": "library"},
    )
    msg_id = store.append_message(conv_id, "assistant", "answer", meta={"agent_trace": trace})
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    payload = chat_router.get_message_agent_trace(msg_id, conv_id=conv_id)

    assert payload["available"] is True
    assert payload["agent_trace"]["mode"] == "research_agent"
    assert payload["summary"]["schema_ok"] is True
    assert payload["summary"]["question_type"] == "single_paper_qa"
    assert payload["summary"]["query_scope"] == "library"


def test_message_agent_trace_endpoint_returns_empty_for_message_without_trace(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("agent audit")
    msg_id = store.append_message(conv_id, "assistant", "answer", meta={"agent_mode": "research_agent"})
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    payload = chat_router.get_message_agent_trace(msg_id, conv_id=conv_id)

    assert payload["available"] is False
    assert payload["agent_trace"] == {}
    assert payload["summary"] == {"available": False}


def test_message_agent_trace_endpoint_rejects_wrong_conversation(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("agent audit")
    other_id = store.create_conversation("other")
    msg_id = store.append_message(conv_id, "assistant", "answer")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    with pytest.raises(HTTPException) as exc:
        chat_router.get_message_agent_trace(msg_id, conv_id=other_id)

    assert exc.value.status_code == 404
