from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

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
    configured = str(os.environ.get("KB_USER_PREFS_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return _PREFS_PATH


def load_prefs() -> dict:
    p = prefs_path()
    if p.exists():
        try:
            data = json.loads(p.read_text("utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def save_prefs(data: dict) -> None:
    p = prefs_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp_path), str(p))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
