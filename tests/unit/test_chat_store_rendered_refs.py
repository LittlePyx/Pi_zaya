import json
from pathlib import Path

from kb.chat_store import ChatStore, _MESSAGE_REFS_NESTED_PAYLOAD_REPAIR_KEY


def _insert_orphan_message_ref(store: ChatStore, *, conv_id: str, user_msg_id: int) -> None:
    with store._connect() as conn:
        conn.execute(
            """
            INSERT INTO message_refs (
                user_msg_id, conv_id, prompt, prompt_sig, hits_json, scores_json,
                used_query, used_translation, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_msg_id,
                conv_id,
                "orphan prompt",
                "orphan-sig",
                "[]",
                "[]",
                "orphan query",
                0,
                1.0,
                1.0,
            ),
        )


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


def test_message_refs_state_snapshot_does_not_load_rendered_payload(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("refs")
    user_msg_id = store.append_message(conv_id, "user", "Which paper is relevant?")
    store.upsert_message_refs(
        user_msg_id=user_msg_id,
        conv_id=conv_id,
        prompt="Which paper is relevant?",
        prompt_sig="sig-state",
        hits=[{"text": "hit", "meta": {"source_path": "paper.md"}}],
        scores=[8.0],
        used_query="relevant paper",
        used_translation=False,
        rendered_payload={
            "hits": [{"ui_meta": {"summary_line": "grounded"}}],
            "large_unused_field": "x" * 20_000,
        },
        rendered_payload_sig="render-state",
        render_status="full",
    )

    state = store.list_message_refs_state(conv_id)

    assert set(state) == {"rows", "messages"}
    row = list(state["rows"])[0]
    assert row["user_msg_id"] == user_msg_id
    assert row["rendered_payload_sig"] == "render-state"
    assert row["rendered_payload_json_chars"] > 20_000
    assert "rendered_payload" not in row
    assert state["messages"]["message_count"] == 1


def test_chat_store_repairs_legacy_nested_rendered_payload_once(tmp_path: Path):
    db_path = tmp_path / "chat.sqlite3"
    store = ChatStore(db_path)
    conv_id = store.create_conversation("refs")
    user_msg_id = store.append_message(conv_id, "user", "Which paper is relevant?")
    store.upsert_message_refs(
        user_msg_id=user_msg_id,
        conv_id=conv_id,
        prompt="Which paper is relevant?",
        prompt_sig="sig-nested",
        hits=[{"text": "hit", "meta": {"source_path": "paper.md"}}],
        scores=[8.0],
        used_query="relevant paper",
        used_translation=False,
        rendered_payload={
            "hits": [{"ui_meta": {"summary_line": "current"}}],
            "rendered_payload": {
                "hits": [{"ui_meta": {"summary_line": "stale"}}],
                "large_unused_field": "x" * 50_000,
            },
        },
        rendered_payload_sig="render-nested",
        render_status="full",
    )
    with store._connect() as conn:
        conn.execute(
            "DELETE FROM chat_store_repairs WHERE repair_key = ?",
            (_MESSAGE_REFS_NESTED_PAYLOAD_REPAIR_KEY,),
        )

    repaired_store = ChatStore(db_path)
    with repaired_store._connect() as conn:
        raw = conn.execute(
            "SELECT rendered_payload_json FROM message_refs WHERE user_msg_id = ?",
            (user_msg_id,),
        ).fetchone()["rendered_payload_json"]
        repair_row = conn.execute(
            "SELECT 1 FROM chat_store_repairs WHERE repair_key = ?",
            (_MESSAGE_REFS_NESTED_PAYLOAD_REPAIR_KEY,),
        ).fetchone()

    payload = json.loads(raw)
    assert payload["hits"][0]["ui_meta"]["summary_line"] == "current"
    assert "rendered_payload" not in payload
    assert repair_row is not None


def test_message_refs_reject_late_write_after_user_message_deleted(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("refs")
    user_msg_id = store.append_message(conv_id, "user", "Which paper defines dynamic supersampling?")

    ok = store.upsert_message_refs(
        user_msg_id=user_msg_id,
        conv_id=conv_id,
        prompt="Which paper defines dynamic supersampling?",
        prompt_sig="sig-dyn",
        hits=[{"text": "hit", "meta": {"source_path": "paper.md"}}],
        scores=[8.5],
        used_query="dynamic supersampling",
        used_translation=False,
    )
    assert ok is True
    assert user_msg_id in store.list_message_refs(conv_id)

    assert store.delete_message(user_msg_id) is True
    assert store.list_message_refs(conv_id) == {}

    late_ok = store.upsert_message_refs(
        user_msg_id=user_msg_id,
        conv_id=conv_id,
        prompt="Which paper defines dynamic supersampling?",
        prompt_sig="sig-dyn-late",
        hits=[{"text": "late", "meta": {"source_path": "paper.md"}}],
        scores=[7.0],
        used_query="dynamic supersampling late",
        used_translation=False,
    )
    assert late_ok is False
    assert store.list_message_refs(conv_id) == {}


def test_message_refs_orphans_are_hidden_and_cleaned_on_render_updates(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("refs")
    orphan_msg_id = 987654

    _insert_orphan_message_ref(store, conv_id=conv_id, user_msg_id=orphan_msg_id)
    assert store.list_message_refs(conv_id) == {}

    state_ok = store.set_message_refs_render_state(
        user_msg_id=orphan_msg_id,
        render_status="pending",
        render_attempts=1,
    )
    assert state_ok is False
    with store._connect() as conn:
        assert conn.execute("SELECT 1 FROM message_refs WHERE user_msg_id = ?", (orphan_msg_id,)).fetchone() is None

    _insert_orphan_message_ref(store, conv_id=conv_id, user_msg_id=orphan_msg_id)
    payload_ok = store.set_message_refs_rendered_payload(
        user_msg_id=orphan_msg_id,
        rendered_payload={"hits": []},
        rendered_payload_sig="orphan-render",
        render_status="full",
    )
    assert payload_ok is False
    with store._connect() as conn:
        assert conn.execute("SELECT 1 FROM message_refs WHERE user_msg_id = ?", (orphan_msg_id,)).fetchone() is None
