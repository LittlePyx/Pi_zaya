from pathlib import Path

from kb.chat_store import ChatStore


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
