from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.deps import get_settings
from api.internal_access import require_internal_api
from kb.user_issue_remote import build_remote_smoke_test_payload, post_remote_issue_payload, user_issue_remote_status
from kb.user_issue_store import UserIssueStore


router = APIRouter(prefix="/api/user-issues", tags=["user-issues"])
_REMOTE_SCHEMA = "pi-zaya.user_issue.v1"
_REMOTE_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9_.:-]+")
_REMOTE_WINDOWS_PATH_RE = re.compile(r"(^|[\s(\"'=])([A-Za-z]:[\\/][^\s\"'<>|]+)")
_REMOTE_FILE_URL_RE = re.compile(r"file:\/\/\/[^\s\"'<>|]+", flags=re.IGNORECASE)
_REMOTE_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|]+")
_REMOTE_UNIX_PATH_RE = re.compile(
    r"(^|[\s(\"'=])(/(?:Users|home|mnt|var|tmp|private)/[^\s\"'<>]+)",
    flags=re.IGNORECASE,
)
_REMOTE_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_REMOTE_AUTH_SECRET_RE = re.compile(
    r"\b(?:authorization|x[-_]?api[-_]?key|api[-_]?key|access[-_]?token|refresh[-_]?token|cookie|set-cookie)\s*[:=]\s*"
    r"(?:bearer\s+)?[A-Za-z0-9._~+/=\-]{8,}",
    flags=re.IGNORECASE,
)
_REMOTE_BEARER_RE = re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=\-]{8,}", flags=re.IGNORECASE)
_REMOTE_TOKEN_RE = re.compile(r"\b(?:sk|pk|ghp|github_pat|xoxb|xoxp|ya29|AIza)[A-Za-z0-9_\-]{12,}\b")
_REMOTE_HTTP_URL_RE = re.compile(r"https?://[^\s\"'<>]+", flags=re.IGNORECASE)
_REMOTE_URL_QUERY_RE = re.compile(r"(https?://[^\s?#]+)(?:\?[^ \t\r\n\"'<>]*)?")
_LOCAL_ISSUE_DICT_MAX_JSON_CHARS = 30_000
_REMOTE_INGEST_DICT_MAX_JSON_CHARS = 40_000
_RATE_LIMIT_WINDOW_S = 60.0
_RATE_LIMIT_MAX_KEYS = 4096
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_BUCKETS: dict[tuple[str, str], deque[float]] = {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except Exception:
        return int(default)


def _bounded_env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    value = _env_int(name, default)
    return max(int(min_value), min(int(max_value), int(value)))


def _user_issue_rate_limit_per_min(kind: str) -> int:
    default = 180 if kind == "local" else 600
    specific_name = (
        "KB_USER_ISSUES_LOCAL_RATE_LIMIT_PER_MIN"
        if kind == "local"
        else "KB_USER_ISSUES_INGEST_RATE_LIMIT_PER_MIN"
    )
    if str(os.environ.get(specific_name) or "").strip():
        return _bounded_env_int(specific_name, default, min_value=0, max_value=10_000)
    return _bounded_env_int("KB_USER_ISSUES_RATE_LIMIT_PER_MIN", default, min_value=0, max_value=10_000)


def _client_rate_key(request: Request) -> str:
    try:
        host = str(request.client.host if request.client else "" or "").strip().lower()
    except Exception:
        host = ""
    return host or "unknown"


def _prune_rate_limit_buckets(now: float) -> None:
    if len(_RATE_LIMIT_BUCKETS) <= _RATE_LIMIT_MAX_KEYS:
        return
    stale_before = now - _RATE_LIMIT_WINDOW_S
    for key, bucket in list(_RATE_LIMIT_BUCKETS.items()):
        while bucket and bucket[0] <= stale_before:
            bucket.popleft()
        if not bucket:
            _RATE_LIMIT_BUCKETS.pop(key, None)
        if len(_RATE_LIMIT_BUCKETS) <= _RATE_LIMIT_MAX_KEYS:
            break


def _enforce_user_issue_rate_limit(request: Request, *, kind: str) -> None:
    limit = _user_issue_rate_limit_per_min(kind)
    if limit <= 0:
        return
    now = time.monotonic()
    stale_before = now - _RATE_LIMIT_WINDOW_S
    bucket_key = (kind, _client_rate_key(request))
    with _RATE_LIMIT_LOCK:
        _prune_rate_limit_buckets(now)
        bucket = _RATE_LIMIT_BUCKETS.setdefault(bucket_key, deque())
        while bucket and bucket[0] <= stale_before:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(math.ceil(_RATE_LIMIT_WINDOW_S - (now - bucket[0]))))
            detail = (
                "too many local user issue reports; try again later"
                if kind == "local"
                else "too many remote user issue ingest requests; try again later"
            )
            raise HTTPException(
                status_code=429,
                detail=detail,
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)


def _bounded_dict(value: Any, *, name: str, max_json_chars: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc
    if len(text) > max_json_chars:
        raise ValueError(f"{name} is too large")
    return value


class UserIssueBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str = Field("frontend", max_length=120)
    domain: str = Field("general", max_length=120)
    severity: str = Field("info", max_length=40)
    summary: str = Field(..., min_length=1, max_length=2_000)
    detail: str = Field("", max_length=12_000)
    route: str = Field("", max_length=1_000)
    context: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str = Field("", max_length=512)

    @field_validator("context", "payload")
    @classmethod
    def _bound_local_dicts(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_dict(value, name="user issue payload", max_json_chars=_LOCAL_ISSUE_DICT_MAX_JSON_CHARS)


class RemoteUserIssueIngestBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_name: str = Field(_REMOTE_SCHEMA, alias="schema", max_length=80)
    client: dict[str, Any] = Field(default_factory=dict)
    issue: dict[str, Any] = Field(default_factory=dict)

    @field_validator("client", "issue")
    @classmethod
    def _bound_remote_dicts(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_dict(value, name="remote user issue payload", max_json_chars=_REMOTE_INGEST_DICT_MAX_JSON_CHARS)


def _issue_db_path() -> Path:
    settings = get_settings()
    configured = getattr(settings, "user_issues_db_path", None)
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(getattr(settings, "db_dir", Path.cwd() / "db")).expanduser().resolve().parent / "user_issues.sqlite3")


def _store() -> UserIssueStore:
    return UserIssueStore(_issue_db_path())


def _require_ingest_token(request: Request) -> None:
    settings = get_settings()
    expected = str(getattr(settings, "user_issues_ingest_token", "") or "").strip()
    if not expected:
        raise HTTPException(403, "remote user issue ingest is not configured")
    auth = str(request.headers.get("authorization") or "").strip()
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    header_token = str(request.headers.get("x-pi-zaya-issue-token") or "").strip()
    if not (
        secrets.compare_digest(expected, bearer)
        or secrets.compare_digest(expected, header_token)
    ):
        raise HTTPException(401, "invalid user issue ingest token")


def _clean_remote_id(value: Any, *, limit: int = 96) -> str:
    text = str(value if value is not None else "").replace("\x00", " ").strip()
    text = _REMOTE_ID_SAFE_RE.sub("-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[: max(0, int(limit))]


def _remote_id_contains_sensitive_text(value: Any) -> bool:
    text = str(value if value is not None else "")
    if not text:
        return False
    if (
        _REMOTE_WINDOWS_PATH_RE.search(text)
        or _REMOTE_FILE_URL_RE.search(text)
        or _REMOTE_UNC_PATH_RE.search(text)
        or _REMOTE_UNIX_PATH_RE.search(text)
        or _REMOTE_EMAIL_RE.search(text)
        or _REMOTE_AUTH_SECRET_RE.search(text)
        or _REMOTE_BEARER_RE.search(text)
        or _REMOTE_TOKEN_RE.search(text)
        or _REMOTE_HTTP_URL_RE.search(text)
    ):
        return True
    return _REMOTE_URL_QUERY_RE.sub(r"\1", text) != text


def _remote_ingest_client_id(value: Any) -> str:
    clean = _clean_remote_id(value, limit=64)
    if not clean:
        return "unknown"
    raw = str(value if value is not None else "")
    hash_input = raw if _remote_id_contains_sensitive_text(raw) else clean
    digest = hashlib.sha256(hash_input.encode("utf-8", "ignore")).hexdigest()[:24]
    return f"client-{digest}"


def _remote_ingest_fingerprint(client_id: Any, raw_fingerprint: Any) -> str:
    raw_fp = str(raw_fingerprint if raw_fingerprint is not None else "")
    fp = _clean_remote_id(raw_fp, limit=96)
    if not fp:
        return ""
    client = _remote_ingest_client_id(client_id)
    if _remote_id_contains_sensitive_text(raw_fp):
        digest = hashlib.sha256(f"{client}\n{raw_fp}".encode("utf-8", "ignore")).hexdigest()[:32]
        return f"remote:{client[:40]}:fp-{digest}"
    candidate = f"remote:{client}:{fp}"
    if len(candidate) <= 128:
        return candidate
    digest = hashlib.sha256(f"{client}\n{fp}".encode("utf-8", "ignore")).hexdigest()[:32]
    return f"remote:{client[:40]}:{digest}"


def _remote_client_has_quality_data_consent(client: dict[str, Any]) -> bool:
    value = client.get("quality_data_sharing")
    if isinstance(value, bool):
        return value
    raw = str(value if value is not None else "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _remote_ingest_client_context(client: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {
        "installation_id": _remote_ingest_client_id(client.get("installation_id")),
        "quality_data_sharing": _remote_client_has_quality_data_consent(client),
    }
    for key, limit in {
        "channel": 80,
        "app_version": 80,
        "platform": 80,
    }.items():
        value = _clean_remote_id(client.get(key), limit=limit)
        if value:
            context[key] = value
    return context


def _remote_ingest_summary(source: str, summary: Any) -> str:
    clean_source = _clean_remote_id(source, limit=120).lower()
    if clean_source == "research_qa_failure_case":
        return "Research QA failure"
    if clean_source == "frontend":
        return "Frontend issue"
    return str(summary or "Remote quality issue")


def _remote_ingest_detail(source: str, detail: Any) -> str:
    clean_source = _clean_remote_id(source, limit=120).lower()
    if clean_source == "frontend":
        return ""
    return str(detail or "")


def _issue_ack(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(issue.get("id") or 0),
        "fingerprint": str(issue.get("fingerprint") or ""),
        "status": str(issue.get("status") or "open"),
        "severity": str(issue.get("severity") or "info"),
        "occurrence_count": int(issue.get("occurrence_count") or 0),
        "last_seen_at": float(issue.get("last_seen_at") or 0.0),
    }


@router.post("")
def record_user_issue(body: UserIssueBody, request: Request):
    _enforce_user_issue_rate_limit(request, kind="local")
    user_agent = str(request.headers.get("user-agent") or "")
    issue = _store().record_issue(
        source=body.source,
        domain=body.domain,
        severity=body.severity,
        summary=body.summary,
        detail=body.detail,
        route=body.route,
        user_agent=user_agent,
        context=body.context,
        payload=body.payload,
        fingerprint=body.fingerprint,
    )
    return {"ok": True, "issue": _issue_ack(issue)}


@router.post("/ingest")
def ingest_remote_user_issue(body: RemoteUserIssueIngestBody, request: Request):
    _enforce_user_issue_rate_limit(request, kind="ingest")
    _require_ingest_token(request)
    if str(body.schema_name or "").strip() != _REMOTE_SCHEMA:
        raise HTTPException(400, "unsupported user issue schema")
    raw_issue = dict(body.issue or {})
    client = dict(body.client or {})
    if not _remote_client_has_quality_data_consent(client):
        raise HTTPException(403, "quality data sharing consent is required")
    client_id = str(client.get("installation_id") or "").strip()
    raw_fp = str(raw_issue.get("fingerprint") or "").strip()
    fingerprint = _remote_ingest_fingerprint(client_id, raw_fp)
    context = raw_issue.get("context") if isinstance(raw_issue.get("context"), dict) else {}
    payload = raw_issue.get("payload") if isinstance(raw_issue.get("payload"), dict) else {}
    source = str(raw_issue.get("source") or "remote")
    remote_client = _remote_ingest_client_context(client)
    remote_context = {"remote_client": remote_client, **context}
    remote_context["remote_client"] = remote_client
    issue = _store().record_issue(
        source=source,
        domain=str(raw_issue.get("domain") or "general"),
        severity=str(raw_issue.get("severity") or "info"),
        summary=_remote_ingest_summary(source, raw_issue.get("summary")),
        detail=_remote_ingest_detail(source, raw_issue.get("detail")),
        route=str(raw_issue.get("route") or ""),
        user_agent=str(request.headers.get("user-agent") or ""),
        context=remote_context,
        payload=payload,
        fingerprint=fingerprint,
        forward_remote=False,
    )
    return {"ok": True, "issue": _issue_ack(issue)}


@router.get("")
def list_user_issues(request: Request, limit: int = 100, status: str = "open"):
    require_internal_api(request)
    return {
        "ok": True,
        "items": _store().list_issues(limit=limit, status=status),
    }


@router.get("/summary")
def user_issues_summary(request: Request):
    require_internal_api(request)
    return {
        "ok": True,
        **_store().summary(),
    }


@router.get("/outbox/summary")
def user_issues_outbox_summary(request: Request):
    require_internal_api(request)
    return {
        "ok": True,
        **_store().remote_outbox_summary(),
    }


@router.post("/outbox/flush")
def flush_user_issues_outbox(request: Request, limit: int = 20):
    require_internal_api(request)
    return _store().flush_remote_outbox(limit=limit)


@router.get("/remote/status")
def user_issues_remote_status(request: Request):
    require_internal_api(request)
    return {
        "ok": True,
        **user_issue_remote_status(),
        "outbox": _store().remote_outbox_summary(),
    }


@router.post("/remote/test")
def test_user_issues_remote(request: Request):
    require_internal_api(request)
    remote_status = user_issue_remote_status()
    if not bool(remote_status.get("enabled")):
        return {
            "ok": False,
            "enabled": False,
            "status_code": 0,
            "error": str(remote_status.get("remote_block_reason") or "remote reporting is disabled"),
            "remote": remote_status,
            "outbox": _store().remote_outbox_summary(),
        }
    result = post_remote_issue_payload(build_remote_smoke_test_payload())
    return {
        "ok": bool(result.get("ok")),
        **result,
        "remote": user_issue_remote_status(),
        "outbox": _store().remote_outbox_summary(),
    }
