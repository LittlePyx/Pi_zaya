from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
from api.routers import user_issues as user_issues_router


def test_user_issue_api_records_to_configured_db(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        user_issues_router,
        "get_settings",
        lambda: SimpleNamespace(user_issues_db_path=tmp_path / "user_issues.sqlite3"),
    )
    client = TestClient(app)

    response = client.post(
        "/api/user-issues",
        json={
            "source": "frontend",
            "domain": "runtime",
            "severity": "error",
            "summary": "Visible workflow failed",
            "detail": "button click rejected",
            "route": "/library",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["issue"]["summary"] == "Visible workflow failed"

    listed = client.get("/api/user-issues", params={"status": "all"}).json()
    assert listed["items"][0]["summary"] == "Visible workflow failed"
