from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from api.deps import prefs_path
from kb.store import atomic_write_json
from kb.test_isolation import isolated_test_mode_enabled


_LOCK = threading.RLock()
_LOGGER = logging.getLogger(__name__)


def _state_path() -> Path:
    configured = str(os.environ.get("KB_MODEL_CONNECTION_STATE_PATH") or "").strip()
    path = Path(configured).expanduser() if configured else prefs_path().with_name("model_connection_checks.json")
    if isolated_test_mode_enabled():
        root = Path(os.environ["KB_TEST_DATA_ROOT"]).resolve()
        if not path.resolve().is_relative_to(root):
            raise RuntimeError("model connection state escapes isolated test root")
    return path


def _load() -> dict:
    try:
        value = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def load_connection_check(target: str, fingerprint: str) -> dict | None:
    with _LOCK:
        record = _load().get(target)
    if not isinstance(record, dict) or record.get("fingerprint") != fingerprint:
        return None
    return dict(record)


def remember_connection_check(target: str, result: dict) -> None:
    """Persist a configuration-bound outcome without credentials or response text."""
    if target not in {"text", "vision"}:
        raise ValueError("unknown provider target")
    record = {key: result.get(key) for key in ("ok", "checked_at", "error_type", "fingerprint")}
    try:
        with _LOCK:
            current = _load()
            current[target] = record
            atomic_write_json(_state_path(), current)
    except OSError:
        _LOGGER.warning("Could not persist model connection check; verification remains session-local")
