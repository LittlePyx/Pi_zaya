from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.deps import get_settings
from kb.user_issue_store import UserIssueStore


router = APIRouter(prefix="/api/user-issues", tags=["user-issues"])


class UserIssueBody(BaseModel):
    source: str = "frontend"
    domain: str = "general"
    severity: str = "info"
    summary: str
    detail: str = ""
    route: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str = ""


class RemoteUserIssueIngestBody(BaseModel):
    schema_name: str = Field("pi-zaya.user_issue.v1", alias="schema")
    client: dict[str, Any] = Field(default_factory=dict)
    issue: dict[str, Any] = Field(default_factory=dict)


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
    if expected not in {bearer, header_token}:
        raise HTTPException(401, "invalid user issue ingest token")


@router.post("")
def record_user_issue(body: UserIssueBody, request: Request):
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
    return {"ok": True, "issue": issue}


@router.post("/ingest")
def ingest_remote_user_issue(body: RemoteUserIssueIngestBody, request: Request):
    _require_ingest_token(request)
    raw_issue = dict(body.issue or {})
    client = dict(body.client or {})
    client_id = str(client.get("installation_id") or "").strip()
    raw_fp = str(raw_issue.get("fingerprint") or "").strip()
    fingerprint = f"remote:{client_id or 'unknown'}:{raw_fp}"[:128] if raw_fp else ""
    context = raw_issue.get("context") if isinstance(raw_issue.get("context"), dict) else {}
    payload = raw_issue.get("payload") if isinstance(raw_issue.get("payload"), dict) else {}
    issue = _store().record_issue(
        source=str(raw_issue.get("source") or "remote"),
        domain=str(raw_issue.get("domain") or "general"),
        severity=str(raw_issue.get("severity") or "info"),
        summary=str(raw_issue.get("summary") or "Remote quality issue"),
        detail=str(raw_issue.get("detail") or ""),
        route=str(raw_issue.get("route") or ""),
        user_agent=str(request.headers.get("user-agent") or ""),
        context={**context, "remote_client": client},
        payload=payload,
        fingerprint=fingerprint,
        forward_remote=False,
    )
    return {"ok": True, "issue": issue}


@router.get("")
def list_user_issues(limit: int = 100, status: str = "open"):
    return {
        "ok": True,
        "items": _store().list_issues(limit=limit, status=status),
    }


@router.get("/summary")
def user_issues_summary():
    return {
        "ok": True,
        **_store().summary(),
    }
