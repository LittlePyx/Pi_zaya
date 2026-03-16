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


def test_chat_store_get_messages_page_returns_recent_slice(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation()
    for idx in range(5):
        store.append_message(conv_id, "user" if idx % 2 == 0 else "assistant", f"msg-{idx + 1}")

    messages, has_more_before, oldest_loaded_id, newest_loaded_id = store.get_messages_page(conv_id, limit=2)

    assert [msg["content"] for msg in messages] == ["msg-4", "msg-5"]
    assert has_more_before is True
    assert oldest_loaded_id == messages[0]["id"]
    assert newest_loaded_id == messages[-1]["id"]


def test_chat_store_get_messages_page_before_id_paginates_older_messages(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation()
    for idx in range(5):
        store.append_message(conv_id, "user", f"msg-{idx + 1}")

    first_page, has_more_first, oldest_first, _ = store.get_messages_page(conv_id, limit=2)
    second_page, has_more_second, oldest_second, newest_second = store.get_messages_page(
        conv_id,
        limit=2,
        before_id=oldest_first,
    )
    third_page, has_more_third, oldest_third, newest_third = store.get_messages_page(
        conv_id,
        limit=2,
        before_id=oldest_second,
    )

    assert [msg["content"] for msg in first_page] == ["msg-4", "msg-5"]
    assert has_more_first is True
    assert [msg["content"] for msg in second_page] == ["msg-2", "msg-3"]
    assert has_more_second is True
    assert oldest_second == second_page[0]["id"]
    assert newest_second == second_page[-1]["id"]
    assert [msg["content"] for msg in third_page] == ["msg-1"]
    assert has_more_third is False
    assert oldest_third == third_page[0]["id"]
    assert newest_third == third_page[-1]["id"]
