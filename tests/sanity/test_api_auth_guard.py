from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api import deps
from api.main import app
from api.routers import generate as generate_router
from api.routers import settings as settings_router
from api.routers import user_issues as user_issues_router
from kb.user_issue_store import UserIssueStore


def _clear_settings_cache() -> None:
    try:
        deps.get_settings.cache_clear()
    except Exception:
        pass


def _set_auth_env(monkeypatch, *, token: str = "secret-token") -> None:
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "1")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "1")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "1")
    monkeypatch.setenv("KB_ALLOW_LOCAL_AUTH_GATE", "1")
    monkeypatch.setenv("KB_AUTH_COOKIE_SECURE", "0")
    monkeypatch.setenv("KB_ACCESS_TOKEN", token)
    monkeypatch.setenv("KB_API_TOKEN", "")
    monkeypatch.setenv("KB_AUTH_TOKEN", "")
    monkeypatch.setenv("KB_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_API_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_AUTH_TOKEN_SHA256", "")
    _clear_settings_cache()


def _set_management_env(monkeypatch, *, token: str = "management-secret") -> None:
    monkeypatch.setenv("KB_ENV", "production")
    monkeypatch.setenv("KB_APP_ENV", "")
    monkeypatch.setenv("KB_REQUIRE_MANAGEMENT_AUTH", "1")
    monkeypatch.setenv("KB_MANAGEMENT_ACCESS_TOKEN", token)
    monkeypatch.setenv("KB_MANAGEMENT_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_AUTH_COOKIE_SECURE", "0")
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "0")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "0")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "0")
    _clear_settings_cache()


def _asgi_post_without_content_length(
    path: str,
    chunks: list[bytes],
    *,
    content_type: str = "application/json",
) -> tuple[int, dict]:
    async def run() -> tuple[int, dict]:
        pending = [
            {
                "type": "http.request",
                "body": chunk,
                "more_body": idx < len(chunks) - 1,
            }
            for idx, chunk in enumerate(chunks)
        ]
        sent: list[dict] = []
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", content_type.encode("latin-1")),
            ],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        async def receive():
            if pending:
                return pending.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        await app(scope, receive, send)
        status = next(message["status"] for message in sent if message.get("type") == "http.response.start")
        raw = b"".join(message.get("body", b"") for message in sent if message.get("type") == "http.response.body")
        return int(status), json.loads(raw.decode("utf-8"))

    return asyncio.run(run())


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


def test_configured_access_token_does_not_lock_public_instance(monkeypatch):
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "0")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "0")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "0")
    monkeypatch.setenv("KB_ACCESS_TOKEN", "configured-but-public")
    monkeypatch.setenv("KB_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_API_TOKEN", "")
    monkeypatch.setenv("KB_AUTH_TOKEN", "")
    monkeypatch.setenv("KB_API_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_AUTH_TOKEN_SHA256", "")
    _clear_settings_cache()
    client = TestClient(app)

    status = client.get("/api/auth/status")
    health = client.get("/api/health")
    assert status.status_code == 200
    assert status.json()["required"] is False
    assert status.json()["configured"] is False
    assert health.status_code == 200
    assert health.json()["auth"] == {"required": False, "configured": False}

    readiness = client.get("/api/settings/readiness")
    assert readiness.status_code == 200

    login = client.post("/api/auth/login", json={"token": "wrong-token"})
    assert login.status_code == 200
    assert login.json()["authenticated"] is False


def test_accidental_require_auth_flag_does_not_lock_user_app(monkeypatch):
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "0")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "0")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "1")
    monkeypatch.setenv("KB_ACCESS_TOKEN", "configured-but-user-app")
    monkeypatch.setenv("KB_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_API_TOKEN", "")
    monkeypatch.setenv("KB_AUTH_TOKEN", "")
    monkeypatch.setenv("KB_API_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_AUTH_TOKEN_SHA256", "")
    _clear_settings_cache()
    client = TestClient(app)

    status = client.get("/api/auth/status")
    readiness = client.get("/api/settings/readiness")

    assert status.status_code == 200
    assert status.json()["required"] is False
    assert status.json()["configured"] is False
    assert readiness.status_code == 200


def test_accidental_local_auth_gate_flags_do_not_lock_user_app(monkeypatch):
    monkeypatch.setenv("KB_ENV", "development")
    monkeypatch.setenv("KB_APP_ENV", "")
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "0")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "1")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "1")
    monkeypatch.setenv("KB_ALLOW_LOCAL_AUTH_GATE", "0")
    monkeypatch.setenv("KB_ACCESS_TOKEN", "configured-but-public-local")
    monkeypatch.setenv("KB_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_API_TOKEN", "")
    monkeypatch.setenv("KB_AUTH_TOKEN", "")
    monkeypatch.setenv("KB_API_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_AUTH_TOKEN_SHA256", "")
    _clear_settings_cache()
    client = TestClient(app)

    status = client.get("/api/auth/status")
    readiness = client.get("/api/settings/readiness")

    assert status.status_code == 200
    assert status.json()["required"] is False
    assert status.json()["configured"] is False
    assert readiness.status_code == 200


def test_accidental_production_auth_flags_without_private_marker_do_not_lock_user_app(monkeypatch):
    monkeypatch.setenv("KB_ENV", "production")
    monkeypatch.setenv("KB_APP_ENV", "")
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "0")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "1")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "1")
    monkeypatch.setenv("KB_ACCESS_TOKEN", "configured-but-public-production")
    monkeypatch.setenv("KB_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_API_TOKEN", "")
    monkeypatch.setenv("KB_AUTH_TOKEN", "")
    monkeypatch.setenv("KB_API_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_AUTH_TOKEN_SHA256", "")
    _clear_settings_cache()
    client = TestClient(app)

    status = client.get("/api/auth/status")
    readiness = client.get("/api/settings/readiness")

    assert status.status_code == 200
    assert status.json()["required"] is False
    assert status.json()["configured"] is False
    assert readiness.status_code == 200


def test_server_file_picker_allowed_for_local_user_app(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_ENV", "development")
    monkeypatch.setenv("KB_APP_ENV", "")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "0")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "0")
    monkeypatch.setattr(settings_router, "_pick_directory_dialog", lambda initial: str(tmp_path))
    _clear_settings_cache()
    client = TestClient(app)

    response = client.post("/api/settings/pick-dir", json={"target": "pdf", "initial_dir": ""})

    assert response.status_code == 200
    assert response.json() == {"ok": True, "path": str(tmp_path)}


def test_server_file_picker_hidden_for_public_production(monkeypatch):
    monkeypatch.setenv("KB_ENV", "production")
    monkeypatch.setenv("KB_APP_ENV", "")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "0")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "0")
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "0")
    monkeypatch.setenv("KB_REQUIRE_MANAGEMENT_AUTH", "1")
    monkeypatch.setenv("KB_MANAGEMENT_ACCESS_TOKEN", "")
    monkeypatch.setenv("KB_MANAGEMENT_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_ACCESS_TOKEN", "")
    monkeypatch.setenv("KB_ACCESS_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_API_TOKEN", "")
    monkeypatch.setenv("KB_API_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_AUTH_TOKEN", "")
    monkeypatch.setenv("KB_AUTH_TOKEN_SHA256", "")

    def fail_picker(initial: str) -> str:
        raise AssertionError("public production should not open a server-side picker")

    monkeypatch.setattr(settings_router, "_pick_directory_dialog", fail_picker)
    _clear_settings_cache()
    client = TestClient(app)

    response = client.post("/api/settings/pick-dir", json={"target": "pdf", "initial_dir": ""})

    assert response.status_code == 503
    assert response.json()["detail"] == "Management access token is not configured"


def test_public_production_blocks_management_writes_but_allows_management_unlock(monkeypatch, tmp_path: Path):
    _set_management_env(monkeypatch)
    monkeypatch.setattr(settings_router, "_pick_directory_dialog", lambda initial: str(tmp_path))
    client = TestClient(app)

    status = client.get("/api/auth/status")
    assert status.status_code == 200
    assert status.json()["required"] is False
    assert status.json()["management_required"] is True
    assert status.json()["management_configured"] is True
    assert status.json()["management_authenticated"] is False

    denied = client.post("/api/settings/pick-dir", json={"target": "pdf", "initial_dir": ""})
    assert denied.status_code == 401
    assert denied.json()["detail"] == "Management authentication required"
    assert denied.headers["X-KB-Management-Auth"] == "required"

    unlocked = client.post(
        "/api/settings/pick-dir",
        json={"target": "pdf", "initial_dir": ""},
        headers={"X-KB-Management-Token": "management-secret"},
    )
    assert unlocked.status_code == 200
    assert unlocked.json()["path"] == str(tmp_path)

    safe_settings = client.get("/api/settings")
    assert safe_settings.status_code == 200
    assert safe_settings.json()["db_dir"] == ""


def test_management_login_unlocks_public_management_routes(monkeypatch, tmp_path: Path):
    _set_management_env(monkeypatch, token="login-management-secret")
    monkeypatch.setattr(settings_router, "_pick_directory_dialog", lambda initial: str(tmp_path))
    client = TestClient(app)

    login = client.post("/api/auth/login", json={"token": "login-management-secret"})
    assert login.status_code == 200
    assert login.json()["authenticated"] is False
    assert login.json()["management_authenticated"] is True

    response = client.post("/api/settings/pick-dir", json={"target": "pdf", "initial_dir": ""})
    assert response.status_code == 200
    assert response.json()["path"] == str(tmp_path)


def test_server_file_picker_allowed_for_authenticated_private_production(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_ENV", "production")
    monkeypatch.setenv("KB_APP_ENV", "")
    _set_auth_env(monkeypatch)
    monkeypatch.setattr(settings_router, "_pick_directory_dialog", lambda initial: str(tmp_path))
    client = TestClient(app)

    denied = client.post("/api/settings/pick-dir", json={"target": "pdf", "initial_dir": ""})
    accepted = client.post(
        "/api/settings/pick-dir",
        json={"target": "pdf", "initial_dir": ""},
        headers={"X-KB-Access-Token": "secret-token"},
    )

    assert denied.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json()["path"] == str(tmp_path)


def test_internal_admin_routes_are_hidden_when_public_auth_is_off(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_REQUIRE_AUTH", "0")
    monkeypatch.delenv("KB_ENABLE_INTERNAL_API", raising=False)
    _clear_settings_cache()
    monkeypatch.setattr(
        user_issues_router,
        "get_settings",
        lambda: SimpleNamespace(user_issues_db_path=tmp_path / "user_issues.sqlite3"),
    )
    client = TestClient(app)

    maintenance = client.get("/api/maintenance/status")
    issue_listing = client.get("/api/user-issues", params={"status": "all"})
    issue_summary = client.get("/api/user-issues/summary")
    issue_record = client.post(
        "/api/user-issues",
        json={
            "source": "frontend",
            "domain": "runtime",
            "severity": "error",
            "summary": "Visible workflow failed",
        },
    )
    library_quality_admin = [
        client.get("/api/library/quality/overview"),
        client.post("/api/library/quality/artifact/open", json={}),
        client.get("/api/library/quality/action-history", params={"limit": 1}),
        client.post(
            "/api/library/quality/action-history",
            json={"stage_key": "conversion", "summary": "hidden admin event"},
        ),
        client.get("/api/library/quality/reader-locate", params={"limit": 1}),
        client.post(
            "/api/library/quality/reader-locate",
            json={
                "source_path": "hidden.md",
                "locate_feedback_key": "hidden",
                "status": "failed",
                "precision": "failed",
                "ok": False,
            },
        ),
        client.get("/api/library/quality/repair-runs", params={"limit": 1}),
        client.get("/api/library/quality/repair-runs/missing-run"),
        client.post("/api/library/quality/repair-runs/missing-run", json={}),
        client.post("/api/library/quality/repair-runs/missing-run/advance", json={}),
        client.post("/api/library/quality/research-qa/rerun", json={}),
        client.post("/api/library/quality/sources", json={"sources": []}),
        client.post("/api/library/quality/conversion/batch", json={}),
        client.post("/api/library/quality/figure-assets/scan", json={}),
        client.post("/api/library/quality/figure-assets/refresh", json={}),
        client.post("/api/library/quality/repair", json={}),
        client.get("/api/generate/quality/summary"),
        client.get("/api/generate/internal-session/trace"),
    ]

    assert maintenance.status_code == 404
    assert issue_listing.status_code == 404
    assert issue_summary.status_code == 404
    assert [res.status_code for res in library_quality_admin] == [404] * len(library_quality_admin)
    assert issue_record.status_code == 200
    assert issue_record.json()["ok"] is True


def test_internal_admin_routes_can_be_enabled_for_local_development(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_REQUIRE_AUTH", "0")
    monkeypatch.setenv("KB_ENABLE_INTERNAL_API", "1")
    _clear_settings_cache()
    monkeypatch.setattr(
        user_issues_router,
        "get_settings",
        lambda: SimpleNamespace(user_issues_db_path=tmp_path / "user_issues.sqlite3"),
    )
    monkeypatch.setattr(
        generate_router,
        "_gen_get_task",
        lambda session_id: {"research_trace": {"session": session_id, "events": [{"name": "retrieval"}]}},
    )
    client = TestClient(app)

    response = client.get("/api/user-issues", params={"status": "all"})
    quality_history = client.get("/api/library/quality/action-history", params={"limit": 1})
    quality_sources = client.post("/api/library/quality/sources", json={"sources": []})
    answer_quality = client.get("/api/generate/quality/summary", params={"limit": 5})
    generation_trace = client.get("/api/generate/internal-session/trace")
    reader_locate = client.post(
        "/api/library/quality/reader-locate",
        json={
            "source_path": "diagnostic.md",
            "locate_feedback_key": "diagnostic",
            "status": "failed",
            "precision": "failed",
            "ok": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert quality_history.status_code == 200
    assert quality_history.json()["ok"] is True
    assert quality_sources.status_code == 200
    assert quality_sources.json()["ok"] is True
    assert answer_quality.status_code == 200
    assert "total" in answer_quality.json()
    assert generation_trace.status_code == 200
    assert generation_trace.json()["session"] == "internal-session"
    assert reader_locate.status_code == 200
    assert reader_locate.json()["ok"] is True


def test_authenticated_public_user_does_not_receive_internal_generation_diagnostics_by_default(monkeypatch):
    _set_auth_env(monkeypatch)
    monkeypatch.setenv("KB_ENV", "production")
    monkeypatch.setenv("KB_APP_ENV", "")
    monkeypatch.delenv("KB_ENABLE_INTERNAL_API", raising=False)
    _clear_settings_cache()
    monkeypatch.setattr(
        generate_router,
        "_gen_get_task",
        lambda session_id: {
            "stage": "done",
            "partial": "public answer",
            "status": "done",
            "answer": "public answer",
            "answer_quality": {"citation_plan": {"internal": True}},
            "paper_guide_debug": {"source_path": "F:\\private\\paper.en.md"},
            "research_trace": {"trace_id": "internal-trace"},
        },
    )

    response = TestClient(app).get(
        "/api/generate/public-session/stream",
        headers={"X-KB-Access-Token": "secret-token"},
    )

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payload = json.loads(lines[-1][6:])
    assert payload["answer"] == "public answer"
    assert payload["answer_quality"] == {}
    assert payload["paper_guide_debug"] == {}
    assert payload["research_trace"] == {}


def test_user_issue_local_routes_require_main_auth_when_api_auth_is_required(monkeypatch, tmp_path: Path):
    _set_auth_env(monkeypatch)
    monkeypatch.setenv("KB_ENABLE_INTERNAL_API", "1")
    monkeypatch.setattr(
        user_issues_router,
        "get_settings",
        lambda: SimpleNamespace(user_issues_db_path=tmp_path / "user_issues.sqlite3"),
    )
    client = TestClient(app)

    denied = client.get("/api/user-issues", params={"status": "all"})
    accepted = client.get(
        "/api/user-issues",
        params={"status": "all"},
        headers={"X-KB-Access-Token": "secret-token"},
    )

    assert denied.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json()["ok"] is True


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
        "client": {"installation_id": "client-a", "quality_data_sharing": True},
        "issue": {"fingerprint": "smoke", "summary": "Collector smoke test"},
    }

    denied = client.post("/api/user-issues/ingest", json=body)
    assert denied.status_code == 401
    assert denied.json()["detail"] == "invalid user issue ingest token"

    main_token_denied = client.post(
        "/api/user-issues/ingest",
        json=body,
        headers={"Authorization": "Bearer secret-token"},
    )
    assert main_token_denied.status_code == 401
    assert main_token_denied.json()["detail"] == "invalid user issue ingest token"

    kb_header_denied = client.post(
        "/api/user-issues/ingest",
        json=body,
        headers={"X-KB-Access-Token": "secret-token"},
    )
    assert kb_header_denied.status_code == 401
    assert kb_header_denied.json()["detail"] == "invalid user issue ingest token"

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


def test_auth_login_accepts_sha256_only_access_token(monkeypatch):
    token = "hash-only-secret"
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "1")
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "1")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "1")
    monkeypatch.setenv("KB_ALLOW_LOCAL_AUTH_GATE", "1")
    monkeypatch.setenv("KB_AUTH_COOKIE_SECURE", "0")
    monkeypatch.setenv("KB_ACCESS_TOKEN", "")
    monkeypatch.setenv("KB_API_TOKEN", "")
    monkeypatch.setenv("KB_AUTH_TOKEN", "")
    monkeypatch.setenv("KB_ACCESS_TOKEN_SHA256", hashlib.sha256(token.encode("utf-8")).hexdigest())
    monkeypatch.setenv("KB_API_TOKEN_SHA256", "")
    monkeypatch.setenv("KB_AUTH_TOKEN_SHA256", "")
    _clear_settings_cache()
    client = TestClient(app)

    denied = client.post("/api/auth/login", json={"token": "wrong-token"})
    assert denied.status_code == 401

    login = client.post("/api/auth/login", json={"token": token})
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


def test_api_json_body_size_guard_rejects_large_json_before_route_access(monkeypatch):
    from api.routers import generate as generate_router

    def fail_store():
        raise AssertionError("route should not be touched for an oversized JSON body")

    monkeypatch.setenv("KB_API_JSON_MAX_BODY_BYTES", "70000")
    monkeypatch.setattr(generate_router, "get_chat_store", fail_store)
    client = TestClient(app)
    body = '{"conv_id":"conv-1","prompt":"' + ("x" * 72_000) + '"}'

    response = client.post(
        "/api/generate",
        content=body.encode("utf-8"),
        headers={"content-type": "application/vnd.pi-zaya+json"},
    )

    assert response.status_code == 413
    assert "JSON request body is too large" in response.json()["detail"]


def test_api_json_body_size_guard_counts_streamed_body_without_content_length(monkeypatch):
    from api.routers import generate as generate_router

    def fail_store():
        raise AssertionError("route should not be touched for an oversized streamed JSON body")

    monkeypatch.setenv("KB_API_JSON_MAX_BODY_BYTES", "70000")
    monkeypatch.setattr(generate_router, "get_chat_store", fail_store)
    body = ('{"conv_id":"conv-1","prompt":"' + ("x" * 72_000) + '"}').encode("utf-8")

    status, payload = _asgi_post_without_content_length(
        "/api/generate",
        [body[:30_000], body[30_000:60_000], body[60_000:]],
    )

    assert status == 413
    assert "JSON request body is too large" in payload["detail"]


def test_user_issue_body_size_guard_counts_streamed_body_without_content_length(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_ISSUES_MAX_BODY_BYTES", "2048")
    monkeypatch.setattr(
        user_issues_router,
        "get_settings",
        lambda: SimpleNamespace(user_issues_db_path=tmp_path / "user_issues.sqlite3"),
    )
    body = json.dumps(
        {
            "source": "frontend",
            "domain": "runtime",
            "severity": "error",
            "summary": "Oversized streamed issue",
            "payload": {"bulk": "x" * 3000},
        },
        separators=(",", ":"),
    ).encode("utf-8")

    status, payload = _asgi_post_without_content_length(
        "/api/user-issues",
        [body[:1500], body[1500:2600], body[2600:]],
    )

    assert status == 413
    assert "user issue payload is too large" in payload["detail"]
    assert UserIssueStore(tmp_path / "user_issues.sqlite3").summary()["total"] == 0
