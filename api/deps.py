from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache

from kb.config import Settings, load_settings
from kb.chat_store import ChatStore


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=1)
def get_chat_store() -> ChatStore:
    return ChatStore(get_settings().chat_db_path)


_HERE = Path(__file__).resolve().parent.parent
_PREFS_PATH = _HERE / "user_prefs.json"


def prefs_path() -> Path:
    return _PREFS_PATH


def load_prefs() -> dict:
    p = prefs_path()
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            return {}
    return {}


def save_prefs(data: dict) -> None:
    p = prefs_path()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
