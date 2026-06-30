from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routers import user_issues as user_issues_router
from kb.user_issue_store import UserIssueStore


def _enable_remote_issue_reporting(monkeypatch, tmp_path: Path) -> None:
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": True}), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "1")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_URL", "https://collector.example/api/user-issues/ingest")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_TOKEN", "collect-secret")


@pytest.fixture(autouse=True)
def _disable_general_api_auth(monkeypatch):
    monkeypatch.setenv("KB_REQUIRE_AUTH", "0")
    monkeypatch.setenv("KB_ENABLE_INTERNAL_API", "1")
    with user_issues_router._RATE_LIMIT_LOCK:
        user_issues_router._RATE_LIMIT_BUCKETS.clear()


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
    assert payload["issue"]["id"] >= 1
    assert payload["issue"]["status"] == "open"
    assert payload["issue"]["severity"] == "error"
    assert payload["issue"]["occurrence_count"] == 1
    assert "summary" not in payload["issue"]
    assert "detail" not in payload["issue"]
    assert "context" not in payload["issue"]
    assert "payload" not in payload["issue"]

    listed = client.get("/api/user-issues", params={"status": "all"}).json()
    assert listed["items"][0]["summary"] == "Visible workflow failed"


def test_user_issue_api_rejects_oversized_request_before_recording(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_ISSUES_MAX_BODY_BYTES", "2048")
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
            "summary": "Oversized issue",
            "payload": {"bulk": "x" * 3000},
        },
    )

    assert response.status_code == 413
    assert UserIssueStore(tmp_path / "user_issues.sqlite3").summary()["total"] == 0


def test_user_issue_api_rejects_unbounded_summary(monkeypatch, tmp_path: Path):
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
            "summary": "x" * 2001,
        },
    )

    assert response.status_code == 422
    assert UserIssueStore(tmp_path / "user_issues.sqlite3").summary()["total"] == 0


def test_user_issue_api_rejects_large_context_payload(monkeypatch, tmp_path: Path):
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
            "summary": "Large local issue context",
            "context": {"bulk": "x" * 31_000},
        },
    )

    assert response.status_code == 422
    assert UserIssueStore(tmp_path / "user_issues.sqlite3").summary()["total"] == 0


def test_user_issue_api_rate_limits_local_records_before_writing(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_ISSUES_LOCAL_RATE_LIMIT_PER_MIN", "2")
    monkeypatch.setattr(
        user_issues_router,
        "get_settings",
        lambda: SimpleNamespace(user_issues_db_path=tmp_path / "user_issues.sqlite3"),
    )
    client = TestClient(app)

    first = client.post(
        "/api/user-issues",
        json={"summary": "first issue", "fingerprint": "rate-local-1"},
    )
    second = client.post(
        "/api/user-issues",
        json={"summary": "second issue", "fingerprint": "rate-local-2"},
    )
    third = client.post(
        "/api/user-issues",
        json={"summary": "third issue", "fingerprint": "rate-local-3"},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.headers.get("retry-after")
    assert third.json()["detail"] == "too many local user issue reports; try again later"
    assert UserIssueStore(tmp_path / "user_issues.sqlite3").summary()["total"] == 2


def test_user_issue_api_records_remote_outbox_when_user_opts_in(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    monkeypatch.setattr(
        user_issues_router,
        "get_settings",
        lambda: SimpleNamespace(user_issues_db_path=tmp_path / "user_issues.sqlite3"),
    )
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)
    client = TestClient(app)

    response = client.post(
        "/api/user-issues",
        json={
            "source": "library_quality_overview",
            "domain": "conversion",
            "severity": "warning",
            "summary": "Missing image assets",
        },
    )
    summary = client.get("/api/user-issues/outbox/summary").json()

    assert response.status_code == 200
    assert summary["ok"] is True
    assert summary["total"] == 1
    assert summary["pending"] == 1
    assert summary["retryable"] == 1


def test_user_issue_api_flushes_remote_outbox(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    monkeypatch.setattr(
        user_issues_router,
        "get_settings",
        lambda: SimpleNamespace(user_issues_db_path=tmp_path / "user_issues.sqlite3"),
    )
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)
    sent: list[dict] = []

    def fake_post(payload: dict) -> dict:
        sent.append(dict(payload))
        return {"ok": True, "enabled": True, "status_code": 200, "error": ""}

    monkeypatch.setattr("kb.user_issue_store.post_remote_issue_payload", fake_post)
    client = TestClient(app)

    client.post(
        "/api/user-issues",
        json={
            "source": "frontend",
            "domain": "runtime",
            "severity": "error",
            "summary": "Visible workflow failed",
            "fingerprint": "visible-workflow-failed",
        },
    )
    flushed = client.post("/api/user-issues/outbox/flush", params={"limit": 5}).json()
    summary = client.get("/api/user-issues/outbox/summary").json()

    assert flushed["sent"] == 1
    assert flushed["failed"] == 0
    assert sent[0]["issue"]["summary"] == "Frontend issue"
    assert sent[0]["issue"]["detail"] == ""
    assert summary["pending"] == 0
    assert summary["sent"] == 1


def test_user_issue_api_reports_remote_status(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_TOKEN", "collect-secret")
    monkeypatch.setattr(
        user_issues_router,
        "get_settings",
        lambda: SimpleNamespace(user_issues_db_path=tmp_path / "user_issues.sqlite3"),
    )
    client = TestClient(app)

    payload = client.get("/api/user-issues/remote/status").json()

    assert payload["ok"] is True
    assert payload["enabled"] is True
    assert payload["remote_enabled"] is True
    assert payload["remote_url_configured"] is True
    assert payload["remote_url_host"] == "collector.example"
    assert payload["remote_token_configured"] is True
    assert "collect-secret" not in str(payload)
    assert payload["outbox"]["pending"] == 0


def test_user_issue_api_sends_remote_smoke_test(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    monkeypatch.setattr(
        user_issues_router,
        "get_settings",
        lambda: SimpleNamespace(user_issues_db_path=tmp_path / "user_issues.sqlite3"),
    )
    sent: list[dict] = []

    def fake_post(payload: dict) -> dict:
        sent.append(dict(payload))
        return {"ok": True, "enabled": True, "status_code": 200, "error": ""}

    monkeypatch.setattr(user_issues_router, "post_remote_issue_payload", fake_post)
    client = TestClient(app)

    response = client.post("/api/user-issues/remote/test").json()

    assert response["ok"] is True
    assert response["status_code"] == 200
    assert sent[0]["issue"]["source"] == "collector_smoke_test"
    assert sent[0]["issue"]["payload"]["test"] is True


def test_user_issue_api_remote_smoke_test_requires_opt_in(monkeypatch, tmp_path: Path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": False}), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "1")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_URL", "https://collector.example/api/user-issues/ingest")
    monkeypatch.setattr(
        user_issues_router,
        "get_settings",
        lambda: SimpleNamespace(user_issues_db_path=tmp_path / "user_issues.sqlite3"),
    )
    sent: list[dict] = []

    def fake_post(payload: dict) -> dict:
        sent.append(dict(payload))
        return {"ok": True, "enabled": True, "status_code": 200, "error": ""}

    monkeypatch.setattr(user_issues_router, "post_remote_issue_payload", fake_post)
    client = TestClient(app)

    response = client.post("/api/user-issues/remote/test").json()

    assert response["ok"] is False
    assert response["enabled"] is False
    assert response["remote"]["remote_block_reason"] == "user_opt_out"
    assert sent == []


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
        "client": {"installation_id": "client-a", "channel": "beta", "quality_data_sharing": True},
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
    assert accepted.json()["issue"]["id"] >= 1
    assert "summary" not in accepted.json()["issue"]
    assert "context" not in accepted.json()["issue"]
    assert "payload" not in accepted.json()["issue"]

    listed = client.get("/api/user-issues", params={"status": "all"}).json()
    assert listed["items"][0]["summary"] == "Missing image assets"
    remote_client = listed["items"][0]["context"]["remote_client"]
    assert listed["items"][0]["fingerprint"].startswith(f"remote:{remote_client['installation_id']}:issue-fp")
    assert remote_client["installation_id"].startswith("client-")
    assert remote_client["installation_id"] != "client-a"
    assert remote_client["channel"] == "beta"


def test_user_issue_ingest_rate_limits_token_guessing_before_recording(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_ISSUES_INGEST_RATE_LIMIT_PER_MIN", "2")
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
        "issue": {"fingerprint": "issue-fp", "summary": "Collector smoke"},
    }

    first = client.post("/api/user-issues/ingest", json=body, headers={"Authorization": "Bearer wrong-1"})
    second = client.post("/api/user-issues/ingest", json=body, headers={"Authorization": "Bearer wrong-2"})
    third = client.post("/api/user-issues/ingest", json=body, headers={"Authorization": "Bearer wrong-3"})

    assert first.status_code == 401
    assert second.status_code == 401
    assert third.status_code == 429
    assert third.headers.get("retry-after")
    assert third.json()["detail"] == "too many remote user issue ingest requests; try again later"
    assert UserIssueStore(tmp_path / "collector.sqlite3").summary()["total"] == 0


def test_user_issue_ingest_rejects_payload_without_remote_consent(monkeypatch, tmp_path: Path):
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
        "issue": {"fingerprint": "issue-fp", "summary": "Missing consent"},
    }

    missing = client.post(
        "/api/user-issues/ingest",
        json=body,
        headers={"Authorization": "Bearer collect-secret"},
    )
    denied = client.post(
        "/api/user-issues/ingest",
        json={**body, "client": {**body["client"], "quality_data_sharing": False}},
        headers={"Authorization": "Bearer collect-secret"},
    )

    assert missing.status_code == 403
    assert missing.json()["detail"] == "quality data sharing consent is required"
    assert denied.status_code == 403
    assert denied.json()["detail"] == "quality data sharing consent is required"
    assert UserIssueStore(tmp_path / "collector.sqlite3").summary()["total"] == 0


def test_user_issue_ingest_rejects_large_remote_issue_payload(monkeypatch, tmp_path: Path):
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
        "issue": {
            "fingerprint": "issue-fp",
            "summary": "Large remote payload",
            "payload": {"bulk": "x" * 41_000},
        },
    }

    response = client.post(
        "/api/user-issues/ingest",
        json=body,
        headers={"Authorization": "Bearer collect-secret"},
    )

    assert response.status_code == 422
    assert UserIssueStore(tmp_path / "collector.sqlite3").summary()["total"] == 0


def test_user_issue_ingest_preserves_remote_client_when_context_is_large(monkeypatch, tmp_path: Path):
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
        "client": {"installation_id": "client-a", "channel": "beta", "quality_data_sharing": True},
        "issue": {
            "fingerprint": "issue-fp",
            "summary": "Large remote context",
            "context": {
                "remote_client": {"installation_id": "spoofed"},
                **{f"k{i:03d}": i for i in range(150)},
            },
        },
    }

    response = client.post(
        "/api/user-issues/ingest",
        json=body,
        headers={"Authorization": "Bearer collect-secret"},
    )

    assert response.status_code == 200
    listed = client.get("/api/user-issues", params={"status": "all"}).json()
    remote_client = listed["items"][0]["context"]["remote_client"]
    assert remote_client["installation_id"].startswith("client-")
    assert remote_client["installation_id"] != "client-a"
    assert remote_client["channel"] == "beta"


def test_user_issue_ingest_keeps_remote_client_first_when_context_has_no_existing_key(monkeypatch, tmp_path: Path):
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
        "client": {"installation_id": "client-a", "channel": "beta", "quality_data_sharing": True},
        "issue": {
            "fingerprint": "issue-fp",
            "summary": "Large remote context without client key",
            "context": {f"k{i:03d}": i for i in range(150)},
        },
    }

    response = client.post(
        "/api/user-issues/ingest",
        json=body,
        headers={"Authorization": "Bearer collect-secret"},
    )

    assert response.status_code == 200
    listed = client.get("/api/user-issues", params={"status": "all"}).json()
    issue_context = listed["items"][0]["context"]
    assert issue_context["remote_client"]["installation_id"].startswith("client-")
    assert issue_context["remote_client"]["installation_id"] != "client-a"
    assert issue_context["remote_client"]["channel"] == "beta"


def test_user_issue_ingest_redacts_untrusted_research_freeform_fields(monkeypatch, tmp_path: Path):
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
        "issue": {
            "fingerprint": "issue-fp",
            "source": "research_qa_failure_case",
            "domain": "research_qa",
            "severity": "error",
            "summary": "Why does Secret Lab Draft mention Alice?",
            "payload": {
                "case": {
                    "question": "Why does Secret Lab Draft mention Alice?",
                    "question_text": "What does Secret Lab Draft say?",
                    "prompt_text": "Summarize Secret Lab Draft",
                    "answer": "It says the private prototype failed.",
                    "answer_text": "The private answer should not be collected.",
                    "quote_text": "A private quote from the paper.",
                    "title": "Secret Lab Draft",
                    "markdown_text": "Full converted markdown should not be collected.",
                    "code": "citation_missing",
                    "count": 1,
                }
            },
        },
    }

    response = client.post(
        "/api/user-issues/ingest",
        json=body,
        headers={"Authorization": "Bearer collect-secret"},
    )

    assert response.status_code == 200
    listed = client.get("/api/user-issues", params={"status": "all"}).json()
    issue = listed["items"][0]
    case_payload = issue["payload"]["case"]
    assert issue["summary"] == "Research QA failure"
    assert case_payload["question"] == "[redacted]"
    assert case_payload["question_text"] == "[redacted]"
    assert case_payload["prompt_text"] == "[redacted]"
    assert case_payload["answer"] == "[redacted]"
    assert case_payload["answer_text"] == "[redacted]"
    assert case_payload["quote_text"] == "[redacted]"
    assert case_payload["title"] == "[redacted]"
    assert case_payload["markdown_text"] == "[redacted]"
    assert case_payload["code"] == "citation_missing"
    assert case_payload["count"] == 1
    assert "Secret Lab" not in str(issue)
    assert "private prototype" not in str(issue)


def test_user_issue_ingest_generalizes_untrusted_frontend_freeform_text(monkeypatch, tmp_path: Path):
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
        "issue": {
            "fingerprint": "frontend-private-message",
            "source": "frontend",
            "domain": "runtime",
            "severity": "error",
            "summary": "Why does Secret Lab Draft mention Alice?",
            "detail": "It says the private prototype failed.",
            "payload": {
                "message": "Secret Lab Draft failed to render",
                "error_kind": "render_failed",
            },
        },
    }

    response = client.post(
        "/api/user-issues/ingest",
        json=body,
        headers={"Authorization": "Bearer collect-secret"},
    )

    assert response.status_code == 200
    listed = client.get("/api/user-issues", params={"status": "all"}).json()
    issue = listed["items"][0]
    assert issue["summary"] == "Frontend issue"
    assert issue["detail"] == ""
    assert issue["payload"]["message"] == "[redacted]"
    assert issue["payload"]["error_kind"] == "render_failed"
    assert "Secret Lab" not in str(issue)
    assert "private prototype" not in str(issue)


def test_user_issue_ingest_rejects_unsupported_schema(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        user_issues_router,
        "get_settings",
        lambda: SimpleNamespace(
            user_issues_db_path=tmp_path / "collector.sqlite3",
            user_issues_ingest_token="collect-secret",
        ),
    )
    client = TestClient(app)

    response = client.post(
        "/api/user-issues/ingest",
        json={
            "schema": "pi-zaya.user_issue.v0",
            "client": {"installation_id": "client-a", "quality_data_sharing": True},
            "issue": {"fingerprint": "issue-fp", "summary": "Old schema payload"},
        },
        headers={"Authorization": "Bearer collect-secret"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "unsupported user issue schema"
    assert UserIssueStore(tmp_path / "collector.sqlite3").summary()["total"] == 0


def test_user_issue_ingest_bounds_remote_fingerprint(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        user_issues_router,
        "get_settings",
        lambda: SimpleNamespace(
            user_issues_db_path=tmp_path / "collector.sqlite3",
            user_issues_ingest_token="collect-secret",
        ),
    )
    client = TestClient(app)
    long_client = "client-" + ("a" * 160)
    body = {
        "schema": "pi-zaya.user_issue.v1",
        "client": {"installation_id": long_client, "quality_data_sharing": True},
        "issue": {
            "fingerprint": "issue/" + ("b" * 160),
            "source": "frontend",
            "summary": "Long remote fingerprint",
        },
    }

    response = client.post(
        "/api/user-issues/ingest",
        json=body,
        headers={"Authorization": "Bearer collect-secret"},
    )

    assert response.status_code == 200
    fingerprint = response.json()["issue"]["fingerprint"]
    assert fingerprint.startswith("remote:client-")
    assert len(fingerprint) <= 128
    assert "/" not in fingerprint


def test_user_issue_ingest_hashes_sensitive_remote_identifiers(monkeypatch, tmp_path: Path):
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
        "client": {"installation_id": "alice@example.com", "quality_data_sharing": True},
        "issue": {
            "fingerprint": "frontend|C:/Users/Alice/private-paper.pdf|sk-secretsecretsecret",
            "source": "frontend",
            "summary": "Sensitive remote fingerprint",
        },
    }

    response = client.post(
        "/api/user-issues/ingest",
        json=body,
        headers={"Authorization": "Bearer collect-secret"},
    )

    assert response.status_code == 200
    fingerprint = response.json()["issue"]["fingerprint"]
    assert fingerprint.startswith("remote:client-")
    assert ":fp-" in fingerprint
    assert "Alice" not in fingerprint
    assert "private-paper" not in fingerprint
    assert "alice@example.com" not in fingerprint
    assert "sk-secret" not in fingerprint


def test_user_issue_ingest_hashes_authorization_like_remote_fingerprint(monkeypatch, tmp_path: Path):
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
        "issue": {
            "fingerprint": "frontend|Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
            "source": "frontend",
            "summary": "Authorization-like remote fingerprint",
        },
    }

    response = client.post(
        "/api/user-issues/ingest",
        json=body,
        headers={"Authorization": "Bearer collect-secret"},
    )

    assert response.status_code == 200
    fingerprint = response.json()["issue"]["fingerprint"]
    assert fingerprint.startswith("remote:client-")
    assert ":fp-" in fingerprint
    assert "Authorization" not in fingerprint
    assert "abcdefghijklmnopqrstuvwxyz" not in fingerprint


def test_user_issue_ingest_hashes_plain_https_url_fingerprints(monkeypatch, tmp_path: Path):
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
        "client": {"installation_id": "https://client.example/install", "quality_data_sharing": True},
        "issue": {
            "fingerprint": "frontend|https://client.example/v1",
            "source": "frontend",
            "summary": "URL-like fingerprint",
        },
    }

    response = client.post(
        "/api/user-issues/ingest",
        json=body,
        headers={"Authorization": "Bearer collect-secret"},
    )

    assert response.status_code == 200
    fingerprint = response.json()["issue"]["fingerprint"]
    assert fingerprint.startswith("remote:client-")
    assert ":fp-" in fingerprint
    assert "client.example" not in fingerprint
    assert "[local-path]" not in str(response.json()["issue"])


def test_user_issue_ingest_whitelists_remote_client_context(monkeypatch, tmp_path: Path):
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
        "client": {
            "installation_id": "alice@example.com",
            "quality_data_sharing": True,
            "channel": "beta cohort",
            "app_version": "1.2.3",
            "platform": "windows desktop",
            "notes": "Private Lab Study",
            "authorization": "sk-secretsecretsecret",
        },
        "issue": {
            "fingerprint": "issue-fp",
            "source": "frontend",
            "summary": "Sensitive remote client context",
            "context": {
                "remote_client": {"installation_id": "spoofed"},
                "code": "missing_images",
            },
        },
    }

    response = client.post(
        "/api/user-issues/ingest",
        json=body,
        headers={"Authorization": "Bearer collect-secret"},
    )

    assert response.status_code == 200
    listed = client.get("/api/user-issues", params={"status": "all"}).json()
    issue_context = listed["items"][0]["context"]
    remote_client = issue_context["remote_client"]
    assert set(remote_client) == {
        "installation_id",
        "quality_data_sharing",
        "channel",
        "app_version",
        "platform",
    }
    assert remote_client["installation_id"].startswith("client-")
    assert remote_client["quality_data_sharing"] is True
    assert remote_client["channel"] == "beta-cohort"
    assert remote_client["app_version"] == "1.2.3"
    assert remote_client["platform"] == "windows-desktop"
    assert issue_context["code"] == "missing_images"
    assert "Private Lab" not in str(issue_context)
    assert "sk-secret" not in str(issue_context)
    assert "alice@example.com" not in str(issue_context)
    assert "spoofed" not in str(issue_context)
