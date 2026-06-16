from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
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


def _issue_db_path() -> Path:
    settings = get_settings()
    configured = getattr(settings, "user_issues_db_path", None)
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(getattr(settings, "db_dir", Path.cwd() / "db")).expanduser().resolve().parent / "user_issues.sqlite3")


def _store() -> UserIssueStore:
    return UserIssueStore(_issue_db_path())


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
