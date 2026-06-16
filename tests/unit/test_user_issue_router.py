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


def test_user_issue_ingest_requires_token_and_records_remote_payload(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        user_issues_router,
        "get_settings",
        lambda: SimpleNamespace(
            user_issues_db_path=tmp_path / "collector.sqlite3",
            user_issues_ingest_token="collect-secret",
        ),
    )
    client = TestClient(app)
    body = {
        "schema": "pi-zaya.user_issue.v1",
        "client": {"installation_id": "client-a", "channel": "beta"},
        "issue": {
            "fingerprint": "issue-fp",
            "source": "library_quality_overview",
            "domain": "conversion",
            "severity": "warning",
            "summary": "Missing image assets",
            "payload": {"issue": {"code": "missing_images", "count": 3}},
        },
    }

    denied = client.post("/api/user-issues/ingest", json=body)
    assert denied.status_code == 401

    accepted = client.post(
        "/api/user-issues/ingest",
        json=body,
        headers={"Authorization": "Bearer collect-secret"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["issue"]["summary"] == "Missing image assets"

    listed = client.get("/api/user-issues", params={"status": "all"}).json()
    assert listed["items"][0]["fingerprint"].startswith("remote:client-a:issue-fp")
    assert listed["items"][0]["context"]["remote_client"]["channel"] == "beta"
