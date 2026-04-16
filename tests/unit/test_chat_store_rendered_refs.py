from pathlib import Path

from kb.chat_store import ChatStore


def test_message_refs_rendered_payload_roundtrip(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("refs")
    user_msg_id = store.append_message(conv_id, "user", "Which paper defines dynamic supersampling?")

    store.upsert_message_refs(
        user_msg_id=user_msg_id,
        conv_id=conv_id,
        prompt="Which paper defines dynamic supersampling?",
        prompt_sig="sig-dyn",
        hits=[{"text": "hit", "meta": {"source_path": r"db\SciAdv-2017\SciAdv-2017.en.md"}}],
        scores=[8.5],
        used_query="dynamic supersampling",
        used_translation=False,
        rendered_payload={"hits": [{"ui_meta": {"summary_line": "full"}}]},
        rendered_payload_sig="render-sig-1",
        render_status="full",
        render_built_at=123.0,
        render_attempts=1,
        render_evidence_sig="evidence-sig-1",
    )

    listed = store.list_message_refs(conv_id)
    pack = listed[user_msg_id]
    assert pack["rendered_payload"] == {"hits": [{"ui_meta": {"summary_line": "full"}}]}
    assert pack["rendered_payload_sig"] == "render-sig-1"
    assert pack["render_status"] == "full"
    assert pack["render_built_at"] == 123.0
    assert pack["render_attempts"] == 1
    assert pack["render_evidence_sig"] == "evidence-sig-1"

    ok = store.set_message_refs_rendered_payload(
        user_msg_id=user_msg_id,
        rendered_payload={"hits": [{"ui_meta": {"summary_line": "full-v2"}}]},
        rendered_payload_sig="render-sig-2",
        render_status="failed",
        render_error="render_payload_empty",
        render_error_detail="empty payload",
        render_attempts=2,
    )
    assert ok is True

    listed2 = store.list_message_refs(conv_id)
    pack2 = listed2[user_msg_id]
    assert pack2["rendered_payload"] == {"hits": [{"ui_meta": {"summary_line": "full-v2"}}]}
    assert pack2["rendered_payload_sig"] == "render-sig-2"
    assert pack2["render_status"] == "failed"
    assert pack2["render_error"] == "render_payload_empty"
    assert pack2["render_error_detail"] == "empty payload"
    assert pack2["render_attempts"] == 2


def test_message_refs_render_state_roundtrip(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("refs")
    user_msg_id = store.append_message(conv_id, "user", "Which paper compares Hadamard and Fourier?")

    store.upsert_message_refs(
        user_msg_id=user_msg_id,
        conv_id=conv_id,
        prompt="Which paper compares Hadamard and Fourier?",
        prompt_sig="sig-compare",
        hits=[{"text": "hit", "meta": {"source_path": r"db\OE-2017\OE-2017.en.md"}}],
        scores=[9.1],
        used_query="Hadamard Fourier compare",
        used_translation=False,
    )

    ok = store.set_message_refs_render_state(
        user_msg_id=user_msg_id,
        render_status="pending",
        render_error="",
        render_error_detail="",
        render_attempts=1,
        render_locale="zh",
    )
    assert ok is True

    pack = store.list_message_refs(conv_id)[user_msg_id]
    assert pack["render_status"] == "pending"
    assert pack["render_attempts"] == 1
    assert pack["render_locale"] == "zh"
