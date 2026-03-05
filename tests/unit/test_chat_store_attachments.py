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
