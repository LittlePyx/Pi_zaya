from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from kb.chat_store import ChatStore


def test_chat_store_rejects_message_for_missing_conversation(tmp_path: Path):
    store = ChatStore(tmp_path / "chat.sqlite3")

    with pytest.raises(sqlite3.IntegrityError):
        store.append_message("missing-conversation", "user", "hello")
