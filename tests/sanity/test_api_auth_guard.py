from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api import deps
from api.main import app
from api.routers import user_issues as user_issues_router


def _clear_settings_cache() -> None:
    try:
        deps.get_settings.cache_clear()
    except Exception:
        pass


def _set_auth_env(monkeypatch, *, token: str = "secret-token") -> None:
    monkeypatch.setenv("KB_REQUIRE_AUTH", "1")
    monkeypatch.setenv("KB_AUTH_COOKIE_SECURE", "0")
    monkeypatch.setenv("KB_ACCESS_TOKEN", token)
    monkeypatch.setenv("KB_API_TOKEN", "")
    monkeypatch.setenv("KB_AUTH_TOKEN", "")
    monkeypatch.setenv("KB_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_API_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_AUTH_TOKEN_SHA256", "")
    _clear_settings_cache()


def test_api_guard_rejects_protected_api_without_token(monkeypatch):
    _set_auth_env(monkeypatch)
    client = TestClient(app)

    assert client.get("/api/health").status_code == 200
    res = client.get("/api/settings/readiness")

    assert res.status_code == 401
    assert res.json()["detail"] == "Authentication required"


def test_api_guard_accepts_access_token_header(monkeypatch):
    _set_auth_env(monkeypatch)
    client = TestClient(app)

    res = client.get("/api/settings/readiness", headers={"X-KB-Access-Token": "secret-token"})

    assert res.status_code == 200
    assert res.json()["overall"]["status"] in {"ok", "warning", "error"}


def test_api_guard_rejects_access_token_query_parameter(monkeypatch):
    _set_auth_env(monkeypatch)
    client = TestClient(app)

    res = client.get("/api/settings/readiness?access_token=secret-token")

    assert res.status_code == 401


def test_user_issue_ingest_uses_own_token_when_api_auth_is_required(monkeypatch, tmp_path: Path):
    _set_auth_env(monkeypatch)
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
        "client": {"installation_id": "client-a"},
        "issue": {"fingerprint": "smoke", "summary": "Collector smoke test"},
    }

    denied = client.post("/api/user-issues/ingest", json=body)
    assert denied.status_code == 401
    assert denied.json()["detail"] == "invalid user issue ingest token"

    accepted = client.post(
        "/api/user-issues/ingest",
        json=body,
        headers={"Authorization": "Bearer collect-secret"},
    )

    assert accepted.status_code == 200
    assert accepted.json()["ok"] is True


def test_auth_login_sets_cookie_for_subsequent_api_calls(monkeypatch):
    _set_auth_env(monkeypatch)
    client = TestClient(app)

    login = client.post("/api/auth/login", json={"token": "secret-token"})
    assert login.status_code == 200
    assert login.json()["authenticated"] is True

    res = client.get("/api/settings/readiness")
    assert res.status_code == 200


def test_api_guard_reports_missing_required_access_token(monkeypatch):
    _set_auth_env(monkeypatch, token="")
    client = TestClient(app)

    status = client.get("/api/auth/status")
    assert status.status_code == 200
    assert status.json()["configured"] is False

    res = client.get("/api/settings/readiness")
    assert res.status_code == 503
    assert res.json()["detail"] == "API access token is not configured"
