from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any


_LOCK = threading.RLock()
_MAX_SESSIONS = 240
_MAX_STRING_LEN = 12_000
_MAX_LIST_ITEMS = 120
_MAX_DICT_ITEMS = 120
_MAX_TITLE_LEN = 240
_MAX_ID_LEN = 120


def _now() -> float:
    return time.time()


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, str):
        return value[:_MAX_STRING_LEN]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth=depth + 1) for item in list(value)[:_MAX_LIST_ITEMS]]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for raw_key, raw_value in list(value.items())[:_MAX_DICT_ITEMS]:
            key = str(raw_key or "").strip()
            if not key:
                continue
            out[key[:120]] = _json_safe(raw_value, depth=depth + 1)
        return out
    return str(value)[:_MAX_STRING_LEN]


def _safe_text(value: Any, *, limit: int = _MAX_STRING_LEN) -> str:
    return str(value or "").replace("\x00", " ").strip()[:limit]


def _safe_timestamp(value: Any, fallback: float) -> float:
    try:
        ts = float(value)
    except Exception:
        return fallback
    if ts != ts or ts in (float("inf"), float("-inf")) or ts < 0:
        return fallback
    return ts


def _safe_message_id(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        mid = int(value)
    except Exception:
        return None
    return mid if mid >= 0 else None


def _apply_state_patch(state: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    current = dict(state or {})
    for raw_key, value in (patch or {}).items():
        key = _safe_text(raw_key, limit=120)
        if not key:
            continue
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value
    safe_state = _json_safe(current)
    return safe_state if isinstance(safe_state, dict) else {}


def _sanitize_record(session_id: str, record: dict[str, Any]) -> dict[str, Any]:
    now = _now()
    safe_payload = _json_safe(record.get("payload") if isinstance(record, dict) else {})
    if not isinstance(safe_payload, dict):
        safe_payload = {}
    safe_state = _json_safe(record.get("state") if isinstance(record, dict) else {})
    if not isinstance(safe_state, dict):
        safe_state = {}
    created_at = _safe_timestamp(record.get("created_at") if isinstance(record, dict) else None, now)
    updated_at = _safe_timestamp(record.get("updated_at") if isinstance(record, dict) else None, created_at)
    return {
        "id": _safe_text(record.get("id") if isinstance(record, dict) else "", limit=_MAX_ID_LEN)
        or _safe_text(session_id, limit=_MAX_ID_LEN),
        "title": _safe_text(record.get("title") if isinstance(record, dict) else "", limit=_MAX_TITLE_LEN),
        "conversation_id": _safe_text(record.get("conversation_id") if isinstance(record, dict) else "", limit=_MAX_ID_LEN),
        "message_id": _safe_message_id(record.get("message_id") if isinstance(record, dict) else None),
        "payload": safe_payload,
        "state": safe_state,
        "created_at": created_at,
        "updated_at": max(created_at, updated_at),
    }


class ReaderSessionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "sessions": {}}
        try:
            data = json.loads(self.path.read_text("utf-8"))
        except Exception:
            return {"version": 1, "sessions": {}}
        if not isinstance(data, dict):
            return {"version": 1, "sessions": {}}
        sessions = data.get("sessions")
        if not isinstance(sessions, dict):
            data["sessions"] = {}
        return data

    def _save_unlocked(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(self.path)

    def _prune_unlocked(self, sessions: dict[str, Any]) -> None:
        if len(sessions) <= _MAX_SESSIONS:
            return
        ordered = sorted(
            sessions.items(),
            key=lambda item: float((item[1] if isinstance(item[1], dict) else {}).get("updated_at") or 0),
            reverse=True,
        )
        keep_ids = {sid for sid, _ in ordered[:_MAX_SESSIONS]}
        for sid in list(sessions.keys()):
            if sid not in keep_ids:
                sessions.pop(sid, None)

    def create(
        self,
        payload: dict[str, Any],
        *,
        title: str = "",
        conversation_id: str = "",
        message_id: int | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        safe_payload = _json_safe(payload)
        if not isinstance(safe_payload, dict):
            safe_payload = {}
        safe_state = _json_safe(state or {})
        if not isinstance(safe_state, dict):
            safe_state = {}
        session_id = uuid.uuid4().hex
        now = _now()
        record = {
            "id": session_id,
            "title": _safe_text(title, limit=_MAX_TITLE_LEN),
            "conversation_id": _safe_text(conversation_id, limit=_MAX_ID_LEN),
            "message_id": _safe_message_id(message_id),
            "payload": safe_payload,
            "state": safe_state,
            "created_at": now,
            "updated_at": now,
        }
        with _LOCK:
            data = self._load_unlocked()
            sessions = data.setdefault("sessions", {})
            if not isinstance(sessions, dict):
                sessions = {}
                data["sessions"] = sessions
            sessions[session_id] = record
            self._prune_unlocked(sessions)
            self._save_unlocked(data)
        return record

    def get(self, session_id: str) -> dict[str, Any] | None:
        sid = str(session_id or "").strip()
        if not sid:
            return None
        with _LOCK:
            data = self._load_unlocked()
            sessions = data.get("sessions")
            if not isinstance(sessions, dict):
                return None
            record = sessions.get(sid)
            if not isinstance(record, dict):
                return None
            return _sanitize_record(sid, record)

    def update_state(self, session_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
        sid = str(session_id or "").strip()
        if not sid:
            return None
        safe_patch = _json_safe(patch or {})
        if not isinstance(safe_patch, dict):
            safe_patch = {}
        with _LOCK:
            data = self._load_unlocked()
            sessions = data.get("sessions")
            if not isinstance(sessions, dict):
                return None
            record = sessions.get(sid)
            if not isinstance(record, dict):
                return None
            record = _sanitize_record(sid, record)
            state = _apply_state_patch(record.get("state") if isinstance(record.get("state"), dict) else {}, safe_patch)
            record["state"] = state
            record["updated_at"] = _now()
            sessions[sid] = record
            self._save_unlocked(data)
            return dict(record)
