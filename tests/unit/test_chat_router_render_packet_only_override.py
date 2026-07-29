from pathlib import Path

from kb.chat_store import ChatStore
from kb import runtime_state as RUNTIME
from kb import task_runtime


def test_paper_guide_messages_page_can_disable_render_packet_only(monkeypatch, tmp_path: Path):
    # Regression: paper_guide mode defaults to render_packet_only=1, but the query param
    # should be able to explicitly disable it (render_packet_only=0) for debugging.
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("guide", mode="paper_guide", bound_source_path="demo.md", bound_source_name="demo.pdf", bound_source_ready=True)
    user_id = store.append_message(conv_id, "user", "explain")
    store.append_message(
        conv_id,
        "assistant",
        "SPI relies on compressive sensing [[CITE:s1234abcd:1]].",
        meta={"paper_guide_contracts": {"version": 1, "intent": {"family": "citation_lookup"}}},
    )
    # Minimal refs entry to allow annotator path to run.
    store.upsert_message_refs(
        user_msg_id=user_id,
        conv_id=conv_id,
        prompt="explain",
        prompt_sig="sig-1",
        hits=[{"text": "dummy", "meta": {"source_path": r"db\\doc\\doc.en.md"}}],
        scores=[],
        used_query="explain",
        used_translation=False,
    )

    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.delenv("KB_CHAT_RENDER_PACKET_ONLY", raising=False)

    # Default (no query override): paper_guide should strip legacy fields.
    default_page = chat_router.get_messages_page(conv_id, limit=24, before_id=None, render_packet_only=None)
    default_msg = default_page["messages"][-1]
    assert "rendered_body" not in default_msg
    assert "notice" not in default_msg

    # Explicit override off: keep legacy fields in payload.
    compat_page = chat_router.get_messages_page(conv_id, limit=24, before_id=None, render_packet_only=0)
    compat_msg = compat_page["messages"][-1]
    assert "rendered_body" in compat_msg
    assert "rendered_content" in compat_msg


def test_messages_page_assistant_only_slice_keeps_reference_packet(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("assistant-only page")
    user_id = store.append_message(conv_id, "user", "what helps SPI reconstruction?")
    store.append_message(
        conv_id,
        "assistant",
        "Learning-based SPI improves reconstruction quality [1].",
    )
    store.upsert_message_refs(
        user_msg_id=user_id,
        conv_id=conv_id,
        prompt="what helps SPI reconstruction?",
        prompt_sig="sig-assistant-only-route",
        hits=[
            {
                "text": "Deep learning improves reconstruction quality in single-pixel imaging.",
                "meta": {
                    "source_path": r"db\LPR-2025\LPR-2025.en.md",
                    "heading_path": "Benefits / Reconstruction quality",
                },
            }
        ],
        scores=[],
        used_query="SPI reconstruction",
        used_translation=False,
    )

    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    primed = chat_router.get_messages(conv_id, render_packet_only=0)
    assert primed[-1]["refs_user_msg_id"] == user_id
    assert len(primed[-1].get("cite_details") or []) == 1

    page = chat_router.get_messages_page(
        conv_id,
        limit=1,
        before_id=None,
        render_packet_only=0,
    )

    assert len(page["messages"]) == 1
    assistant = page["messages"][0]
    assert assistant["role"] == "assistant"
    assert assistant["refs_user_msg_id"] == user_id
    assert len(assistant.get("cite_details") or []) == 1
    persisted = store.get_messages(conv_id)[-1]
    render_cache = ((persisted.get("meta") or {}).get("render_cache") or {})
    packet = (
        (((persisted.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet"))
        or {}
    )
    assert render_cache.get("refs_user_msg_id") == user_id
    assert len(packet.get("cite_details") or []) == 1


def test_messages_page_recovers_stale_live_assistant_marker(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("normal")
    task_id = "task-stale-live-marker"
    store.append_message(conv_id, "user", "explain this")
    assistant_id = store.append_message(conv_id, "assistant", task_runtime._live_assistant_text(task_id))

    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(chat_router, "load_prefs", lambda: {"ui_locale": "en"})
    with RUNTIME.GEN_LOCK:
        old_tasks = dict(RUNTIME.GEN_TASKS)
        RUNTIME.GEN_TASKS.clear()

    try:
        page = chat_router.get_messages_page(conv_id, limit=24, before_id=None, render_packet_only=0)
        assistant = page["messages"][-1]

        assert assistant["id"] == assistant_id
        assert assistant["content"] == task_runtime.generation_interrupted_message("en")
        assert assistant["meta"]["generation_status"] == "interrupted"
        assert assistant["meta"]["generation_task_id"] == task_id

        stored = store.get_messages(conv_id)[-1]
        assert stored["content"] == task_runtime.generation_interrupted_message("en")
        assert stored["meta"]["generation_status"] == "interrupted"
    finally:
        with RUNTIME.GEN_LOCK:
            RUNTIME.GEN_TASKS.clear()
            RUNTIME.GEN_TASKS.update(old_tasks)


def test_messages_page_keeps_active_live_assistant_marker(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("normal")
    session_id = "session-active-live-marker"
    task_id = "task-active-live-marker"
    live_text = task_runtime._live_assistant_text(task_id)
    store.append_message(conv_id, "user", "explain this")
    store.append_message(conv_id, "assistant", live_text)

    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(chat_router, "load_prefs", lambda: {"ui_locale": "en"})
    with RUNTIME.GEN_LOCK:
        RUNTIME.GEN_TASKS[session_id] = {
            "id": task_id,
            "session_id": session_id,
            "conv_id": conv_id,
            "chat_db": str(tmp_path / "chat.db"),
            "status": "running",
            "answer_ready": False,
            "cancel": False,
            "created_at": 1.0,
            "updated_at": 1.0,
        }

    try:
        page = chat_router.get_messages_page(conv_id, limit=24, before_id=None, render_packet_only=0)
        assert page["messages"][-1]["content"] == live_text
        assert "generation_status" not in dict(page["messages"][-1].get("meta") or {})
    finally:
        with RUNTIME.GEN_LOCK:
            RUNTIME.GEN_TASKS.pop(session_id, None)
