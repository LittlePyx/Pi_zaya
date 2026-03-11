from pathlib import Path

from kb.chat_store import ChatStore


def test_chat_store_persists_message_attachments(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation()
    msg_id = store.append_message(
        conv_id,
        "user",
        "[Image attachment x1]",
        attachments=[
            {
                "sha1": "abc123",
                "path": str(tmp_path / "img.png"),
                "name": "img.png",
                "mime": "image/png",
                "url": "/api/chat/uploads/image?path=test",
            }
        ],
    )

    messages = store.get_messages(conv_id)

    assert msg_id > 0
    assert len(messages) == 1
    assert messages[0]["attachments"][0]["name"] == "img.png"
    assert messages[0]["attachments"][0]["url"].startswith("/api/chat/uploads/image")


def test_chat_store_persists_message_meta_and_provenance(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation()
    msg_id = store.append_message(
        conv_id,
        "assistant",
        "answer",
        meta={
            "trace_id": "t-1",
            "provenance": {
                "version": 1,
                "segments": [{"segment_id": "seg_001"}],
            },
        },
    )

    messages = store.get_messages(conv_id)

    assert msg_id > 0
    assert len(messages) == 1
    assert messages[0]["meta"]["trace_id"] == "t-1"
    assert messages[0]["provenance"]["version"] == 1
    assert messages[0]["provenance"]["segments"][0]["segment_id"] == "seg_001"


def test_chat_store_merge_message_meta_keeps_previous_fields(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation()
    msg_id = store.append_message(
        conv_id,
        "assistant",
        "answer",
        meta={"trace_id": "t-1", "stage": "initial"},
    )

    ok = store.merge_message_meta(
        msg_id,
        {
            "stage": "merged",
            "provenance": {"status": "ready"},
        },
    )
    messages = store.get_messages(conv_id)

    assert ok is True
    assert len(messages) == 1
    assert messages[0]["meta"]["trace_id"] == "t-1"
    assert messages[0]["meta"]["stage"] == "merged"
    assert messages[0]["provenance"]["status"] == "ready"
