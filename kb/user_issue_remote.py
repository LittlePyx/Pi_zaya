from __future__ import annotations

import hashlib
import os
import re
import threading
from typing import Any, Mapping

import requests


_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s\"'<>|]+|\\\\[^\s\"'<>|]+|/(?:Users|home|mnt|var|tmp|private)/[^\s\"'<>]+)",
    flags=re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_TOKEN_RE = re.compile(r"\b(?:sk|pk|ghp|github_pat|xoxb|xoxp|ya29|AIza)[A-Za-z0-9_\-]{12,}\b")
_LONG_HASH_RE = re.compile(r"\b[A-Fa-f0-9]{32,}\b")
_URL_QUERY_RE = re.compile(r"(https?://[^\s?#]+)(?:\?[^ \t\r\n\"'<>]*)?")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|token|secret|password|authorization|cookie|pdf[_-]?path|md[_-]?path|"
    r"source[_-]?path|absolute[_-]?path|local[_-]?path|file[_-]?path|path)$",
    flags=re.IGNORECASE,
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except Exception:
        return float(default)


def _clean_text(value: Any, *, limit: int = 2000) -> str:
    text = str(value if value is not None else "").replace("\x00", " ")
    text = _PATH_RE.sub("[local-path]", text)
    text = _EMAIL_RE.sub("[email]", text)
    text = _TOKEN_RE.sub("[token]", text)
    text = _LONG_HASH_RE.sub("[hash]", text)
    text = _URL_QUERY_RE.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[: int(max(0, limit))]


def _safe_scalar(value: Any, *, limit: int = 1000) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _clean_text(value, limit=limit)


def _safe_payload(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[depth-limit]"
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, raw_val in value.items():
            clean_key = _clean_text(key, limit=120)
            if not clean_key:
                continue
            if _SENSITIVE_KEY_RE.search(clean_key):
                out[clean_key] = "[redacted]"
                continue
            out[clean_key] = _safe_payload(raw_val, depth=depth + 1)
        return out
    if isinstance(value, list):
        return [_safe_payload(item, depth=depth + 1) for item in value[:100]]
    return _safe_scalar(value, limit=1000)


def _stable_client_id(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:32]


def user_issue_remote_enabled() -> bool:
    return bool(
        _env_bool("KB_USER_ISSUES_REMOTE_ENABLED", False)
        and str(os.environ.get("KB_USER_ISSUES_REMOTE_URL") or "").strip()
    )


def build_remote_issue_payload(issue: Mapping[str, Any]) -> dict[str, Any]:
    client_id = _stable_client_id(os.environ.get("KB_USER_ISSUES_CLIENT_ID") or "")
    project_channel = _clean_text(os.environ.get("KB_USER_ISSUES_CLIENT_CHANNEL") or "", limit=120)
    payload = {
        "schema": "pi-zaya.user_issue.v1",
        "client": {
            "installation_id": client_id,
            "channel": project_channel,
        },
        "issue": {
            "fingerprint": _clean_text(issue.get("fingerprint"), limit=128),
            "source": _clean_text(issue.get("source"), limit=120),
            "domain": _clean_text(issue.get("domain"), limit=120),
            "severity": _clean_text(issue.get("severity"), limit=40),
            "status": _clean_text(issue.get("status"), limit=40),
            "summary": _clean_text(issue.get("summary"), limit=500),
            "detail": _clean_text(issue.get("detail"), limit=1200),
            "route": _clean_text(issue.get("route"), limit=500),
            "first_seen_at": _safe_scalar(issue.get("first_seen_at")),
            "last_seen_at": _safe_scalar(issue.get("last_seen_at")),
            "occurrence_count": _safe_scalar(issue.get("occurrence_count")),
            "context": _safe_payload(issue.get("context") if isinstance(issue.get("context"), Mapping) else {}),
            "payload": _safe_payload(issue.get("payload") if isinstance(issue.get("payload"), Mapping) else {}),
        },
    }
    return payload


def _post_remote_issue(payload: dict[str, Any]) -> None:
    url = str(os.environ.get("KB_USER_ISSUES_REMOTE_URL") or "").strip()
    if not url:
        return
    token = str(os.environ.get("KB_USER_ISSUES_REMOTE_TOKEN") or "").strip()
    timeout_s = max(0.5, min(15.0, _env_float("KB_USER_ISSUES_REMOTE_TIMEOUT_S", 2.5)))
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Pi-zaya-KB/1.0 user-issue-reporter",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        requests.post(url, json=payload, headers=headers, timeout=timeout_s)
    except Exception:
        return


def report_user_issue_remote(issue: Mapping[str, Any], *, async_send: bool = True) -> dict[str, Any]:
    if not user_issue_remote_enabled():
        return {"enabled": False, "queued": False}
    payload = build_remote_issue_payload(issue)
    if async_send:
        thread = threading.Thread(target=_post_remote_issue, args=(payload,), daemon=True)
        thread.start()
        return {"enabled": True, "queued": True}
    _post_remote_issue(payload)
    return {"enabled": True, "queued": False}
