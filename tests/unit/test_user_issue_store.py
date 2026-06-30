from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from kb import user_issue_remote
from kb.user_issue_remote import build_remote_issue_payload
from kb.user_issue_store import UserIssueStore, record_library_quality_issues


def test_user_issue_store_deduplicates_occurrences(tmp_path: Path):
    store = UserIssueStore(tmp_path / "user_issues.sqlite3")

    first = store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        detail="stack",
        route="/library",
        fingerprint="same-error",
    )
    second = store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        detail="stack",
        route="/library",
        fingerprint="same-error",
    )

    assert first["id"] == second["id"]
    assert second["occurrence_count"] == 2
    assert store.summary()["open"] == 1


def test_user_issue_store_coalesces_recent_duplicate_events_and_remote_outbox(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    monkeypatch.setenv("KB_USER_ISSUES_EVENT_COALESCE_S", "60")
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)
    store = UserIssueStore(tmp_path / "user_issues.sqlite3")

    first = store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        detail="stack",
        route="/library",
        fingerprint="same-render-error",
    )
    second = store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        detail="stack",
        route="/library",
        fingerprint="same-render-error",
    )

    with store._connect() as conn:
        event_count = int(conn.execute("SELECT COUNT(*) FROM user_issue_events;").fetchone()[0] or 0)
    summary = store.remote_outbox_summary()

    assert first["id"] == second["id"]
    assert second["occurrence_count"] == 2
    assert event_count == 1
    assert summary["total"] == 1
    assert summary["pending"] == 1


def test_user_issue_store_queues_after_opt_in_despite_recent_local_event(monkeypatch, tmp_path: Path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": False}), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "1")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_URL", "https://collector.example/api/user-issues/ingest")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_TOKEN", "collect-secret")
    monkeypatch.setenv("KB_USER_ISSUES_EVENT_COALESCE_S", "60")
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)
    store = UserIssueStore(tmp_path / "user_issues.sqlite3")

    first = store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        detail="stack",
        route="/library",
        fingerprint="same-render-error",
    )
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": True}), encoding="utf-8")
    second = store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        detail="stack",
        route="/library",
        fingerprint="same-render-error",
    )

    with store._connect() as conn:
        event_count = int(conn.execute("SELECT COUNT(*) FROM user_issue_events;").fetchone()[0] or 0)
    summary = store.remote_outbox_summary()

    assert first["id"] == second["id"]
    assert second["occurrence_count"] == 2
    assert event_count == 2
    assert summary["total"] == 1
    assert summary["pending"] == 1


def test_user_issue_store_prunes_old_local_history(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(tmp_path / "missing-user-prefs.json"))
    monkeypatch.setenv("KB_USER_ISSUES_EVENT_COALESCE_S", "0")
    monkeypatch.setenv("KB_USER_ISSUES_MAX_ISSUES", "3")
    monkeypatch.setenv("KB_USER_ISSUES_MAX_EVENTS", "3")
    store = UserIssueStore(tmp_path / "user_issues.sqlite3")

    for idx in range(5):
        store.record_issue(
            source="frontend",
            domain="runtime",
            severity="error",
            summary=f"Render failed {idx}",
            fingerprint=f"issue-{idx}",
        )

    with store._connect() as conn:
        issue_count = int(conn.execute("SELECT COUNT(*) FROM user_issues;").fetchone()[0] or 0)
        event_count = int(conn.execute("SELECT COUNT(*) FROM user_issue_events;").fetchone()[0] or 0)
    fingerprints = {item["fingerprint"] for item in store.list_issues(status="all", limit=10)}

    assert issue_count == 3
    assert event_count == 3
    assert fingerprints == {"issue-2", "issue-3", "issue-4"}


def test_user_issue_store_pruning_preserves_pending_remote_outbox(monkeypatch, tmp_path: Path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(
        json.dumps({"quality_data_sharing_enabled": True, "quality_data_client_id": "pending-client"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "0")
    monkeypatch.setenv("KB_USER_ISSUES_EVENT_COALESCE_S", "0")
    monkeypatch.setenv("KB_USER_ISSUES_MAX_ISSUES", "1")
    monkeypatch.setenv("KB_USER_ISSUES_MAX_EVENTS", "1")
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)
    store = UserIssueStore(tmp_path / "user_issues.sqlite3")

    for idx in range(2):
        store.record_issue(
            source="frontend",
            domain="runtime",
            severity="error",
            summary=f"Render failed while collector pending {idx}",
            fingerprint=f"pending-issue-{idx}",
        )

    with store._connect() as conn:
        issue_count = int(conn.execute("SELECT COUNT(*) FROM user_issues;").fetchone()[0] or 0)
        event_count = int(conn.execute("SELECT COUNT(*) FROM user_issue_events;").fetchone()[0] or 0)
    summary = store.remote_outbox_summary()

    assert issue_count == 2
    assert event_count == 2
    assert summary["pending"] == 2


def test_user_issue_store_prunes_sent_remote_outbox(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    monkeypatch.setenv("KB_USER_ISSUES_EVENT_COALESCE_S", "0")
    monkeypatch.setenv("KB_USER_ISSUES_MAX_SENT_OUTBOX", "1")
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)

    def fake_post(payload: dict) -> dict:
        return {"ok": True, "enabled": True, "status_code": 200, "error": ""}

    monkeypatch.setattr("kb.user_issue_store.post_remote_issue_payload", fake_post)
    store = UserIssueStore(tmp_path / "user_issues.sqlite3")

    for idx in range(3):
        store.record_issue(
            source="frontend",
            domain="runtime",
            severity="error",
            summary=f"Render failed sent outbox {idx}",
            fingerprint=f"sent-outbox-{idx}",
        )
    result = store.flush_remote_outbox(limit=10)
    summary = store.remote_outbox_summary()

    assert result["sent"] == 3
    assert summary["pending"] == 0
    assert summary["sent"] == 1
    assert summary["total"] == 1


def test_user_issue_store_redacts_and_bounds_local_issue_payload(tmp_path: Path):
    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    fingerprint = "0123456789abcdef" * 4

    issue = store.record_issue(
        source="frontend C:/Users/Alice/private-paper.pdf",
        domain="runtime api_key=localapikeysecret",
        severity="error",
        summary="Failed reading C:/Users/Alice/private-paper.pdf",
        detail=(
            "Error: token sk-secretsecretsecret at alice@example.com "
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz "
            "x-api-key=localapikeysecret "
            "from C:/Users/Alice/private-paper.pdf "
            f"cache {fingerprint}"
        ),
        route="/library?token=secret#debug",
        user_agent="Pi-zaya test sk-agentsecretsecretsecret",
        context={
            "user_agent": "Mozilla/5.0 Secret Lab Browser alice@example.com",
            "ua": "Secret Lab UA",
            "source_path": "C:/Users/Alice/private-paper.pdf",
            "route": "/library?token=secret",
            "prompt_text": "Explain Secret Lab Draft in detail",
            "nested": {
                "filename": "Secret Lab Draft.pdf",
                "answer_text": "The private prototype failed.",
                "code": "missing_images",
            },
        },
        payload={
            "filename": "file:///C:/Users/Alice/private-paper.pdf",
            "pdf_name": "Secret Lab Draft.pdf",
            "question_text": "Why does Secret Lab Draft mention Alice?",
            "quote_text": "The private draft says the prototype failed.",
            "code": "missing_images",
            "count": 3,
            "nan_score": float("nan"),
            "inf_score": float("inf"),
            "nested_metrics": {"bad": float("-inf")},
            "huge": ["x" * 2000 for _ in range(120)],
        },
        fingerprint=fingerprint,
    )

    assert issue["fingerprint"] == fingerprint
    assert issue["source"] == "frontend [local-path]"
    assert issue["domain"] == "runtime api_key=[token]"
    assert "Alice" not in str(issue)
    assert "sk-secret" not in str(issue)
    assert "abcdefghijklmnopqrstuvwxyz" not in str(issue)
    assert "localapikeysecret" not in str(issue)
    assert "alice@example.com" not in str(issue)
    assert "[local-path]" in issue["summary"]
    assert "[token]" in issue["detail"]
    assert "[email]" in issue["detail"]
    assert "[hash]" in issue["detail"]
    assert issue["context"]["source_path"] == "[redacted]"
    assert issue["context"]["user_agent"] == "[redacted]"
    assert issue["context"]["ua"] == "[redacted]"
    assert issue["context"]["prompt_text"] == "[redacted]"
    assert issue["context"]["nested"]["filename"] == "[redacted]"
    assert issue["context"]["nested"]["answer_text"] == "[redacted]"
    assert issue["context"]["nested"]["code"] == "missing_images"
    assert issue["payload"]["filename"] == "[redacted]"
    assert issue["payload"]["pdf_name"] == "[redacted]"
    assert issue["payload"]["question_text"] == "[redacted]"
    assert issue["payload"]["quote_text"] == "[redacted]"
    assert issue["payload"]["code"] == "missing_images"
    assert issue["payload"]["count"] == 3
    assert issue["payload"]["nan_score"] is None
    assert issue["payload"]["inf_score"] is None
    assert issue["payload"]["nested_metrics"]["bad"] is None
    assert "Secret Lab" not in str(issue)
    assert "Mozilla/5.0" not in str(issue)
    assert "private prototype" not in str(issue)

    with store._connect() as conn:
        row = conn.execute("SELECT route, user_agent, payload_json FROM user_issue_events LIMIT 1;").fetchone()
    assert row["route"] == "/library"
    assert "secret" not in row["route"]
    assert "[token]" in row["user_agent"]
    assert len(str(row["payload_json"] or "")) <= 21_000


def test_user_issue_store_preserves_https_urls_while_redacting_local_paths(tmp_path: Path):
    store = UserIssueStore(tmp_path / "user_issues.sqlite3")

    issue = store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary=(
            "Callback https://proxy.example/v1?token=private failed while reading "
            "C:/Users/Alice/private-paper.pdf"
        ),
        detail=(
            "Retry https://collector.example/api/user-issues/ingest?api_key=secret "
            "and callback https://collector.example/oauth/callback#access_token=fragmentsecret "
            "and /Users/alice/project/.env plus file:///C:/Users/Alice/private.env"
        ),
        route="/settings#token=fragmentsecret",
        fingerprint="url-and-local-path-redaction",
    )

    assert "https://proxy.example/v1" in issue["summary"]
    assert "https://collector.example/api/user-issues/ingest" in issue["detail"]
    assert "https://collector.example/oauth/callback" in issue["detail"]
    assert "?token=private" not in issue["summary"]
    assert "?api_key=secret" not in issue["detail"]
    assert "#access_token=" not in issue["detail"]
    assert "fragmentsecret" not in str(issue)
    assert "http[local-path]" not in str(issue)
    assert "Alice" not in str(issue)
    assert "/Users/alice" not in str(issue)
    assert "file:///C:" not in str(issue)
    assert "[local-path]" in issue["summary"]
    assert "[local-path]" in issue["detail"]

    with store._connect() as conn:
        row = conn.execute("SELECT route FROM user_issue_events LIMIT 1;").fetchone()
    assert row["route"] == "/settings"


def test_user_issue_store_hashes_sensitive_supplied_fingerprint(tmp_path: Path):
    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    raw_fingerprint = "frontend|C:/Users/Alice/private-paper.pdf|sk-secretsecretsecret|alice@example.com"

    issue = store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Failed reading C:/Users/Alice/private-paper.pdf",
        detail="token sk-secretsecretsecret at alice@example.com",
        fingerprint=raw_fingerprint,
    )

    assert issue["fingerprint"] != raw_fingerprint
    assert re.fullmatch(r"[0-9a-f]{64}", issue["fingerprint"])
    assert "Alice" not in issue["fingerprint"]
    assert "sk-secret" not in issue["fingerprint"]
    assert "alice@example.com" not in issue["fingerprint"]


def test_user_issue_store_hashes_authorization_like_supplied_fingerprint(tmp_path: Path):
    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    raw_fingerprint = "frontend|Authorization: Bearer abcdefghijklmnopqrstuvwxyz|render-failed"

    issue = store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        fingerprint=raw_fingerprint,
    )

    assert issue["fingerprint"] != raw_fingerprint
    assert re.fullmatch(r"[0-9a-f]{64}", issue["fingerprint"])
    assert "Authorization" not in issue["fingerprint"]
    assert "abcdefghijklmnopqrstuvwxyz" not in issue["fingerprint"]


def test_user_issue_store_hashes_url_like_supplied_fingerprint(tmp_path: Path):
    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    raw_fingerprint = "frontend|https://private.example/workspace/issues/render-failed"

    issue = store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        fingerprint=raw_fingerprint,
    )

    assert issue["fingerprint"] != raw_fingerprint
    assert re.fullmatch(r"[0-9a-f]{64}", issue["fingerprint"])
    assert "private.example" not in issue["fingerprint"]


def test_user_issue_store_redacts_research_freeform_payload_fields(tmp_path: Path):
    store = UserIssueStore(tmp_path / "user_issues.sqlite3")

    issue = store.record_issue(
        source="research_qa_failure_case",
        domain="research_qa",
        severity="error",
        summary="Research QA failure",
        payload={
            "case": {
                "question": "Why does Secret Lab Draft mention Alice?",
                "answer": "It says the private prototype failed.",
                "title": "Secret Lab Draft",
                "markdown_text": "Full converted markdown should not be collected.",
                "code": "citation_missing",
                "count": 1,
            }
        },
        fingerprint="research-qa-freeform",
    )

    case_payload = issue["payload"]["case"]
    assert case_payload["question"] == "[redacted]"
    assert case_payload["answer"] == "[redacted]"
    assert case_payload["title"] == "[redacted]"
    assert case_payload["markdown_text"] == "[redacted]"
    assert case_payload["code"] == "citation_missing"
    assert case_payload["count"] == 1
    assert "Secret Lab" not in str(issue)
    assert "private prototype" not in str(issue)


def _enable_remote_issue_reporting(monkeypatch, tmp_path: Path) -> Path:
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": True}), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "1")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_URL", "https://collector.example/api/user-issues/ingest")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_TOKEN", "collect-secret")
    return prefs_path


def test_user_issue_store_keeps_remote_outbox_empty_without_opt_in(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(tmp_path / "missing-user-prefs.json"))
    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    store.record_issue(
        source="library_quality_overview",
        domain="conversion",
        severity="warning",
        summary="Missing image assets",
    )

    assert store.remote_outbox_summary()["total"] == 0


def test_user_issue_store_queues_opted_in_issue_before_collector_is_ready(monkeypatch, tmp_path: Path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(
        json.dumps({"quality_data_sharing_enabled": True, "quality_data_client_id": "pending-collector-client"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "0")
    monkeypatch.delenv("KB_USER_ISSUES_REMOTE_URL", raising=False)
    monkeypatch.delenv("KB_USER_ISSUES_REMOTE_TOKEN", raising=False)
    flush_calls: list[int] = []
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: flush_calls.append(limit))

    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    store.record_issue(
        source="library_quality_overview",
        domain="conversion",
        severity="warning",
        summary="Missing image assets while collector is not ready",
        fingerprint="collector-not-ready",
    )

    summary = store.remote_outbox_summary()
    assert summary["total"] == 1
    assert summary["pending"] == 1
    assert summary["retryable"] == 1
    assert flush_calls == []

    disabled_flush = store.flush_remote_outbox()
    assert disabled_flush["ok"] is False
    assert disabled_flush["enabled"] is False
    assert store.remote_outbox_summary()["pending"] == 1


def test_user_issue_store_queues_recent_duplicate_after_user_opts_in(monkeypatch, tmp_path: Path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": False}), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_EVENT_COALESCE_S", "60")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "0")
    monkeypatch.delenv("KB_USER_ISSUES_REMOTE_URL", raising=False)
    monkeypatch.delenv("KB_USER_ISSUES_REMOTE_TOKEN", raising=False)
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)

    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed before opt in",
        fingerprint="render-failed-after-opt-in",
    )
    assert store.remote_outbox_summary()["total"] == 0

    prefs_path.write_text(
        json.dumps({"quality_data_sharing_enabled": True, "quality_data_client_id": "new-consent-client"}),
        encoding="utf-8",
    )
    store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed after opt in",
        fingerprint="render-failed-after-opt-in",
    )

    summary = store.remote_outbox_summary()
    assert summary["total"] == 1
    assert summary["pending"] == 1


def test_user_issue_store_queues_recorded_issue_for_remote_outbox(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)

    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    store.record_issue(
        source="library_quality_overview",
        domain="conversion",
        severity="warning",
        summary="Missing image assets",
    )

    summary = store.remote_outbox_summary()
    assert summary["total"] == 1
    assert summary["pending"] == 1
    assert summary["retryable"] == 1


def test_user_issue_remote_outbox_preserves_fingerprint_and_route(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)
    sent: list[dict] = []

    def fake_post(payload: dict) -> dict:
        sent.append(dict(payload))
        return {"ok": True, "enabled": True, "status_code": 200, "error": ""}

    monkeypatch.setattr("kb.user_issue_store.post_remote_issue_payload", fake_post)

    fingerprint = "a" * 64
    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        route="/library?tab=quality",
        fingerprint=fingerprint,
    )
    result = store.flush_remote_outbox()

    assert result["sent"] == 1
    assert sent[0]["issue"]["fingerprint"] == fingerprint
    assert sent[0]["issue"]["fingerprint"] != "[hash]"
    assert sent[0]["issue"]["route"] == "/library"


def test_user_issue_store_flushes_remote_outbox_success(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)
    sent: list[dict] = []

    def fake_post(payload: dict) -> dict:
        sent.append(dict(payload))
        return {"ok": True, "enabled": True, "status_code": 200, "error": ""}

    monkeypatch.setattr("kb.user_issue_store.post_remote_issue_payload", fake_post)

    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        detail="stack",
        route="/library",
        fingerprint="render-failed",
    )
    result = store.flush_remote_outbox()

    assert result["sent"] == 1
    assert result["failed"] == 0
    assert len(sent) == 1
    assert sent[0]["issue"]["summary"] == "Frontend issue"
    assert sent[0]["issue"]["detail"] == ""
    assert store.remote_outbox_summary()["pending"] == 0
    assert store.remote_outbox_summary()["sent"] == 1


def test_user_issue_store_keeps_failed_remote_outbox_for_retry(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)
    attempts: list[dict] = []

    def fake_post(payload: dict) -> dict:
        attempts.append(dict(payload))
        return {"ok": False, "enabled": True, "status_code": 503, "error": "collector down"}

    monkeypatch.setattr("kb.user_issue_store.post_remote_issue_payload", fake_post)

    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        detail="stack",
        route="/library",
        fingerprint="render-failed",
    )
    first = store.flush_remote_outbox()
    second = store.flush_remote_outbox()
    summary = store.remote_outbox_summary()

    assert first["failed"] == 1
    assert second["sent"] == 0
    assert second["failed"] == 0
    assert len(attempts) == 1
    assert summary["pending"] == 1
    assert summary["sent"] == 0
    assert summary["latest_error"] == "collector down"


def test_user_issue_store_concurrent_flush_claims_outbox_once(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)
    entered = threading.Event()
    release = threading.Event()
    sent: list[dict] = []
    first_result: dict = {}

    def fake_post(payload: dict) -> dict:
        sent.append(dict(payload))
        entered.set()
        assert release.wait(timeout=3.0)
        return {"ok": True, "enabled": True, "status_code": 200, "error": ""}

    monkeypatch.setattr("kb.user_issue_store.post_remote_issue_payload", fake_post)

    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        fingerprint="render-failed",
    )

    worker = threading.Thread(target=lambda: first_result.update(store.flush_remote_outbox()))
    worker.start()
    assert entered.wait(timeout=3.0)

    second_result = store.flush_remote_outbox()
    release.set()
    worker.join(timeout=3.0)

    assert first_result["sent"] == 1
    assert second_result["sent"] == 0
    assert second_result["failed"] == 0
    assert len(sent) == 1
    summary = store.remote_outbox_summary()
    assert summary["pending"] == 0
    assert summary["sent"] == 1


def test_user_issue_store_flush_releases_claim_when_post_raises(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)

    def fake_post(payload: dict) -> dict:
        raise RuntimeError("collector exploded")

    monkeypatch.setattr("kb.user_issue_store.post_remote_issue_payload", fake_post)

    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        fingerprint="render-failed",
    )
    result = store.flush_remote_outbox()
    summary = store.remote_outbox_summary()

    assert result["ok"] is False
    assert result["sent"] == 0
    assert result["failed"] == 1
    assert summary["pending"] == 1
    assert summary["sent"] == 0
    assert "collector exploded" in summary["latest_error"]


def test_user_issue_store_redacts_sensitive_remote_outbox_errors(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)

    def fake_post(payload: dict) -> dict:
        return {
            "ok": False,
            "enabled": True,
            "status_code": 401,
            "error": (
                "collector rejected Authorization: Bearer abcdefghijklmnopqrstuvwxyz "
                "at https://collector.example/api/user-issues/ingest?token=secret"
            ),
        }

    monkeypatch.setattr("kb.user_issue_store.post_remote_issue_payload", fake_post)

    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        fingerprint="render-failed-redacted-error",
    )

    result = store.flush_remote_outbox()
    summary = store.remote_outbox_summary()

    assert result["failed"] == 1
    assert "Authorization: [token]" in summary["latest_error"]
    assert "abcdefghijklmnopqrstuvwxyz" not in summary["latest_error"]
    assert "https://collector.example/api/user-issues/ingest" in summary["latest_error"]
    assert "token=secret" not in summary["latest_error"]


def test_user_issue_store_flush_rechecks_opt_in_after_claim(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)
    enabled_calls = {"count": 0}
    sent: list[dict] = []

    def fake_enabled() -> bool:
        enabled_calls["count"] += 1
        return enabled_calls["count"] <= 2

    def fake_post(payload: dict) -> dict:
        sent.append(dict(payload))
        return {"ok": True, "enabled": True, "status_code": 200, "error": ""}

    monkeypatch.setattr("kb.user_issue_store.user_issue_remote_enabled", fake_enabled)
    monkeypatch.setattr("kb.user_issue_store.post_remote_issue_payload", fake_post)

    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        fingerprint="render-failed",
    )
    result = store.flush_remote_outbox()
    summary = store.remote_outbox_summary()

    assert result["enabled"] is False
    assert result["sent"] == 0
    assert result["failed"] == 0
    assert result["released"] == 1
    assert sent == []
    assert summary["pending"] == 1
    assert summary["retryable"] == 1
    assert summary["sent"] == 0
    assert summary["latest_error"] == "remote reporting is disabled"


def test_user_issue_store_discards_unsent_remote_outbox(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    monkeypatch.setattr(UserIssueStore, "flush_remote_outbox_async", lambda self, limit=20: None)

    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    store.record_issue(
        source="frontend",
        domain="runtime",
        severity="error",
        summary="Render failed",
        fingerprint="render-failed",
    )

    assert store.remote_outbox_summary()["pending"] == 1
    result = store.discard_unsent_remote_outbox()
    summary = store.remote_outbox_summary()

    assert result["removed"] == 1
    assert summary["pending"] == 0
    assert summary["total"] == 0


def test_remote_issue_payload_redacts_local_paths_and_tokens(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_ISSUES_CLIENT_ID", "lab-machine-01")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(tmp_path / "missing-user-prefs.json"))
    payload = build_remote_issue_payload(
        {
            "fingerprint": "fp1",
            "source": "frontend C:/Users/Alice/private.pdf",
            "domain": "runtime api_key=remoteapikeysecret",
            "severity": "error",
            "summary": "Failed reading C:/Users/Alice/private.pdf",
            "detail": (
                "token sk-secretsecretsecret at alice@example.com "
                "Authorization: Bearer abcdefghijklmnopqrstuvwxyz "
                "api_key=remoteapikeysecret"
            ),
            "context": {
                "source_path": r"C:\Users\Alice\paper.pdf",
                "route": "/library",
                "user_agent": "Mozilla/5.0 Secret Lab Browser alice@example.com",
                "ua": "Secret Lab UA",
            },
            "payload": {"filename": "paper.pdf", "code": "missing_images"},
        }
    )

    text = str(payload)
    assert "Alice" not in text
    assert "sk-secret" not in text
    assert "abcdefghijklmnopqrstuvwxyz" not in text
    assert "remoteapikeysecret" not in text
    assert "alice@example.com" not in text
    assert payload["issue"]["source"] == "frontend [local-path]"
    assert payload["issue"]["domain"] == "runtime api_key=[token]"
    assert "Authorization: [token]" in payload["issue"]["detail"]
    assert "api_key=[token]" in payload["issue"]["detail"]
    assert payload["issue"]["payload"]["code"] == "missing_images"
    assert payload["issue"]["context"]["source_path"] == "[redacted]"
    assert payload["issue"]["context"]["user_agent"] == "[redacted]"
    assert payload["issue"]["context"]["ua"] == "[redacted]"
    assert payload["client"]["installation_id"]
    assert payload["client"]["quality_data_sharing"] is False


def test_remote_issue_payload_preserves_https_urls_while_stripping_queries(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(tmp_path / "missing-user-prefs.json"))
    payload = build_remote_issue_payload(
        {
            "fingerprint": "remote-url-redaction",
            "source": "frontend",
            "domain": "runtime",
            "severity": "error",
            "summary": "Callback https://proxy.example/v1?token=private failed",
            "detail": (
                "Collector https://collector.example/api/user-issues/ingest?api_key=secret "
                "and https://collector.example/oauth/callback#access_token=fragmentsecret "
                "failed beside C:/Users/Alice/private-paper.pdf and file:///C:/Users/Alice/private.env"
            ),
            "payload": {
                "code": "collector_failed",
                "callback": "https://proxy.example/v1#token=fragmentsecret",
            },
        }
    )

    text = str(payload)
    assert payload["issue"]["summary"] == "Frontend issue"
    assert payload["issue"]["detail"] == ""
    assert payload["issue"]["payload"]["callback"] == "https://proxy.example/v1"
    assert payload["issue"]["payload"]["code"] == "collector_failed"
    assert "http[local-path]" not in text
    assert "?token=private" not in text
    assert "?api_key=secret" not in text
    assert "#access_token=" not in text
    assert "#token=" not in text
    assert "fragmentsecret" not in text
    assert "Alice" not in text
    assert "file:///C:" not in text


def test_remote_issue_payload_generalizes_frontend_freeform_text(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(tmp_path / "missing-user-prefs.json"))
    payload = build_remote_issue_payload(
        {
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
        }
    )

    assert payload["issue"]["summary"] == "Frontend issue"
    assert payload["issue"]["detail"] == ""
    assert payload["issue"]["payload"]["message"] == "[redacted]"
    assert payload["issue"]["payload"]["error_kind"] == "render_failed"
    assert "Secret Lab" not in str(payload)
    assert "private prototype" not in str(payload)


def test_remote_issue_payload_redacts_file_name_fields_but_keeps_quality_codes(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(tmp_path / "missing-user-prefs.json"))
    payload = build_remote_issue_payload(
        {
            "fingerprint": "fp-file-name",
            "summary": "Frontend runtime error",
            "payload": {
                "filename": "file:///C:/Users/Alice/private-paper.pdf",
                "pdf_name": "Secret Lab Draft.pdf",
                "source_name": "private-paper.en.md",
                "issue": {
                    "code": "missing_images",
                    "count": 3,
                    "document_name": "Secret Lab Draft.pdf",
                },
            },
        }
    )

    issue_payload = payload["issue"]["payload"]
    assert issue_payload["filename"] == "[redacted]"
    assert issue_payload["pdf_name"] == "[redacted]"
    assert issue_payload["source_name"] == "[redacted]"
    assert issue_payload["issue"]["document_name"] == "[redacted]"
    assert issue_payload["issue"]["code"] == "missing_images"
    assert issue_payload["issue"]["count"] == 3
    assert "Secret Lab" not in str(payload)
    assert "Alice" not in str(payload)


def test_remote_issue_payload_redacts_user_freeform_fields(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(tmp_path / "missing-user-prefs.json"))
    payload = build_remote_issue_payload(
        {
            "fingerprint": "qa-private-question",
            "source": "research_qa_failure_case",
            "domain": "research_qa",
            "severity": "error",
            "summary": "Why does Secret Lab Draft mention alice@example.com?",
            "detail": "citation_missing",
            "payload": {
                "case": {
                    "question": "Why does Secret Lab Draft mention Alice?",
                    "question_text": "What does the Secret Lab Draft say?",
                    "prompt_text": "Summarize Secret Lab Draft for Alice",
                    "answer": "It says the private prototype failed.",
                    "answer_text": "The private answer should not leave the machine.",
                    "quote_text": "A private quote from the converted paper.",
                    "markdown_text": "Full converted markdown should never leave the machine.",
                    "title": "Secret Lab Draft",
                    "code": "citation_missing",
                    "count": 1,
                },
            },
        }
    )

    case_payload = payload["issue"]["payload"]["case"]
    assert payload["issue"]["summary"] == "Research QA failure"
    assert case_payload["question"] == "[redacted]"
    assert case_payload["question_text"] == "[redacted]"
    assert case_payload["prompt_text"] == "[redacted]"
    assert case_payload["answer"] == "[redacted]"
    assert case_payload["answer_text"] == "[redacted]"
    assert case_payload["quote_text"] == "[redacted]"
    assert case_payload["markdown_text"] == "[redacted]"
    assert case_payload["title"] == "[redacted]"
    assert case_payload["code"] == "citation_missing"
    assert case_payload["count"] == 1
    assert "Secret Lab" not in str(payload)
    assert "Alice" not in str(payload)
    assert "private prototype" not in str(payload)


def test_remote_issue_payload_bounds_large_nested_payload_but_keeps_codes(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(tmp_path / "missing-user-prefs.json"))
    payload = build_remote_issue_payload(
        {
            "fingerprint": "large-quality-payload",
            "summary": "Quality scan produced a large payload",
            "payload": {
                "filename": "C:/Users/Alice/private-paper.pdf",
                "documents": ["Secret Lab Draft.pdf"],
                "paper_count": 1,
                "issue": {
                    "code": "table_fragmentation",
                    "count": 7,
                    "samples": ["x" * 2000 for _ in range(120)],
                    "example_text": "A private example from Secret Lab Draft.",
                },
            },
        }
    )

    issue_payload = payload["issue"]["payload"]
    assert issue_payload["filename"] == "[redacted]"
    assert issue_payload["documents"] == "[redacted]"
    assert issue_payload["paper_count"] == 1
    assert issue_payload["issue"]["code"] == "table_fragmentation"
    assert issue_payload["issue"]["count"] == 7
    assert issue_payload["issue"]["samples"] == "[redacted]"
    assert issue_payload["issue"]["example_text"] == "[redacted]"
    assert len(json.dumps(payload, ensure_ascii=False)) < 20_000
    assert "Alice" not in str(payload)
    assert "Secret Lab" not in str(payload)


def test_remote_issue_payload_preserves_hash_fingerprint_for_collector_dedup(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(tmp_path / "missing-user-prefs.json"))
    fingerprint = "0123456789abcdef" * 4
    payload = build_remote_issue_payload(
        {
            "fingerprint": fingerprint,
            "summary": "Conversion warning",
            "detail": "A local cache hash 0123456789abcdef0123456789abcdef should still be redacted here.",
        }
    )

    assert payload["issue"]["fingerprint"] == fingerprint
    assert "[hash]" in payload["issue"]["detail"]


def test_remote_issue_payload_sanitizes_non_finite_numbers(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(tmp_path / "missing-user-prefs.json"))
    payload = build_remote_issue_payload(
        {
            "summary": "Conversion warning",
            "first_seen_at": float("nan"),
            "last_seen_at": float("inf"),
            "payload": {
                "score": float("-inf"),
                "nested": {"confidence": float("nan")},
            },
        }
    )

    assert payload["issue"]["first_seen_at"] is None
    assert payload["issue"]["last_seen_at"] is None
    assert payload["issue"]["payload"]["score"] is None
    assert payload["issue"]["payload"]["nested"]["confidence"] is None
    assert "NaN" not in json.dumps(payload, allow_nan=False)


def test_remote_issue_payload_hashes_sensitive_fingerprint(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(tmp_path / "missing-user-prefs.json"))
    payload = build_remote_issue_payload(
        {
            "fingerprint": "frontend|C:/Users/Alice/private-paper.pdf|sk-secretsecretsecret|alice@example.com",
            "summary": "Conversion warning",
        }
    )

    fingerprint = payload["issue"]["fingerprint"]
    assert fingerprint.startswith("fp-")
    assert "Alice" not in fingerprint
    assert "sk-secret" not in fingerprint
    assert "alice@example.com" not in fingerprint


def test_remote_issue_payload_hashes_url_like_fingerprint(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(tmp_path / "missing-user-prefs.json"))
    payload = build_remote_issue_payload(
        {
            "fingerprint": "frontend|https://private.example/workspace/issues/render-failed",
            "summary": "Conversion warning",
        }
    )

    fingerprint = payload["issue"]["fingerprint"]
    assert fingerprint.startswith("fp-")
    assert "private.example" not in fingerprint


def test_remote_issue_reporting_requires_user_opt_in(monkeypatch, tmp_path: Path):
    prefs_path = tmp_path / "user_prefs.json"
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "1")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_URL", "https://collector.example/api/user-issues/ingest")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_TOKEN", "collect-secret")

    assert user_issue_remote.user_issue_remote_enabled() is False

    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": True}), encoding="utf-8")

    assert user_issue_remote.user_issue_remote_enabled() is True
    payload = build_remote_issue_payload({"summary": "Conversion warning"})
    assert payload["client"]["quality_data_sharing"] is True


def test_remote_issue_payload_backfills_missing_anonymous_client_id(monkeypatch, tmp_path: Path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": True}), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))

    first_payload = build_remote_issue_payload({"summary": "Conversion warning"})
    stored = json.loads(prefs_path.read_text(encoding="utf-8"))
    second_payload = build_remote_issue_payload({"summary": "Another conversion warning"})

    raw_client_id = str(stored.get("quality_data_client_id") or "")
    assert raw_client_id
    assert first_payload["client"]["installation_id"] == user_issue_remote._stable_client_id(raw_client_id)
    assert second_payload["client"]["installation_id"] == first_payload["client"]["installation_id"]
    assert first_payload["client"]["installation_id"] != raw_client_id


def test_remote_issue_reporting_requires_sender_token_by_default(monkeypatch, tmp_path: Path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": True}), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "1")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_URL", "https://collector.example/api/user-issues/ingest")

    status = user_issue_remote.user_issue_remote_status()
    result = user_issue_remote.post_remote_issue_payload(build_remote_issue_payload({"summary": "Conversion warning"}))

    assert user_issue_remote.user_issue_remote_enabled() is False
    assert status["enabled"] is False
    assert status["remote_token_configured"] is False
    assert status["remote_token_required"] is True
    assert status["remote_block_reason"] == "missing_remote_token"
    assert result["ok"] is False
    assert result["error"] == "missing_remote_token"

    monkeypatch.setenv("KB_USER_ISSUES_ALLOW_UNAUTHENTICATED_REMOTE", "1")

    status = user_issue_remote.user_issue_remote_status()
    assert status["remote_unauthenticated_allowed"] is True
    assert status["remote_token_required"] is False
    assert user_issue_remote.user_issue_remote_enabled() is True


def test_remote_issue_reporting_blocks_localhost_collector_by_default(monkeypatch, tmp_path: Path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": True}), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "1")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_URL", "http://127.0.0.1:9000/api/user-issues/ingest")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_TOKEN", "collect-secret")

    status = user_issue_remote.user_issue_remote_status()

    assert user_issue_remote.user_issue_remote_enabled() is False
    assert status["enabled"] is False
    assert status["remote_url_is_local"] is True
    assert status["remote_url_allowed"] is False
    assert status["remote_block_reason"] == "local_remote_url"

    monkeypatch.setenv("KB_USER_ISSUES_ALLOW_LOCAL_REMOTE", "1")

    assert user_issue_remote.user_issue_remote_enabled() is True
    assert user_issue_remote.user_issue_remote_status()["remote_url_local_allowed"] is True


def test_remote_issue_reporting_blocks_private_network_collector_by_default(monkeypatch, tmp_path: Path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": True}), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "1")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_URL", "https://192.168.1.10/api/user-issues/ingest")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_TOKEN", "collect-secret")

    status = user_issue_remote.user_issue_remote_status()

    assert user_issue_remote.user_issue_remote_enabled() is False
    assert status["enabled"] is False
    assert status["remote_url_is_local"] is True
    assert status["remote_url_allowed"] is False
    assert status["remote_block_reason"] == "local_remote_url"

    monkeypatch.setenv("KB_USER_ISSUES_ALLOW_LOCAL_REMOTE", "1")

    assert user_issue_remote.user_issue_remote_enabled() is True
    assert user_issue_remote.user_issue_remote_status()["remote_url_local_allowed"] is True


def test_remote_issue_reporting_blocks_insecure_public_collector(monkeypatch, tmp_path: Path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": True}), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "1")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_URL", "http://collector.example/api/user-issues/ingest")

    status = user_issue_remote.user_issue_remote_status()

    assert user_issue_remote.user_issue_remote_enabled() is False
    assert status["enabled"] is False
    assert status["remote_url_scheme"] == "http"
    assert status["remote_url_has_valid_scheme"] is True
    assert status["remote_url_secure"] is False
    assert status["remote_url_allowed"] is False
    assert status["remote_block_reason"] == "insecure_remote_url"


def test_remote_issue_reporting_blocks_collector_without_http_scheme(monkeypatch, tmp_path: Path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": True}), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "1")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_URL", "collector.example/api/user-issues/ingest")

    status = user_issue_remote.user_issue_remote_status()

    assert user_issue_remote.user_issue_remote_enabled() is False
    assert status["enabled"] is False
    assert status["remote_url_has_valid_scheme"] is False
    assert status["remote_url_allowed"] is False
    assert status["remote_block_reason"] == "invalid_remote_url"


def test_remote_issue_reporting_blocks_collector_url_credentials(monkeypatch, tmp_path: Path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": True}), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "1")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_URL", "https://user:pass@collector.example/api/user-issues/ingest")
    calls: list[dict] = []

    def fake_post(*args: object, **kwargs: object) -> object:
        calls.append({"args": args, "kwargs": kwargs})
        raise AssertionError("collector URL with credentials should not be called")

    monkeypatch.setattr(user_issue_remote.requests, "post", fake_post)

    status = user_issue_remote.user_issue_remote_status()
    result = user_issue_remote.post_remote_issue_payload(build_remote_issue_payload({"summary": "Conversion warning"}))

    assert user_issue_remote.user_issue_remote_enabled() is False
    assert status["enabled"] is False
    assert status["remote_url_host"] == "collector.example"
    assert status["remote_url_has_credentials"] is True
    assert status["remote_url_allowed"] is False
    assert status["remote_block_reason"] == "remote_url_credentials"
    assert "user:pass" not in str(status)
    assert result["ok"] is False
    assert result["error"] == "remote_url_credentials"
    assert calls == []


def test_remote_issue_reporting_blocks_collector_url_with_invalid_port(monkeypatch, tmp_path: Path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(json.dumps({"quality_data_sharing_enabled": True}), encoding="utf-8")
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "1")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_URL", "https://collector.example:bad/api/user-issues/ingest")

    status = user_issue_remote.user_issue_remote_status()

    assert user_issue_remote.user_issue_remote_enabled() is False
    assert status["enabled"] is False
    assert status["remote_url_has_valid_port"] is False
    assert status["remote_url_allowed"] is False
    assert status["remote_block_reason"] == "invalid_remote_url"


def test_remote_issue_post_blocks_public_host_that_resolves_private(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    calls: list[dict] = []

    def fake_getaddrinfo(host: str, port: object, **kwargs: object) -> list[tuple]:
        assert host == "collector.example"
        assert port is None
        assert kwargs["type"] == user_issue_remote.socket.SOCK_STREAM
        return [(user_issue_remote.socket.AF_INET, user_issue_remote.socket.SOCK_STREAM, 6, "", ("10.0.0.12", 0))]

    def fake_post(*args: object, **kwargs: object) -> object:
        calls.append({"args": args, "kwargs": kwargs})
        raise AssertionError("private collector target should not be called")

    monkeypatch.setattr(user_issue_remote.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(user_issue_remote.requests, "post", fake_post)

    result = user_issue_remote.post_remote_issue_payload(build_remote_issue_payload({"summary": "Conversion warning"}))

    assert result["ok"] is False
    assert result["enabled"] is True
    assert result["status_code"] == 0
    assert result["error"] == "remote host resolves to local/private address"
    assert calls == []


def test_remote_issue_post_allows_public_resolved_collector(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    calls: list[dict] = []

    def fake_getaddrinfo(host: str, port: object, **kwargs: object) -> list[tuple]:
        assert host == "collector.example"
        assert port is None
        assert kwargs["type"] == user_issue_remote.socket.SOCK_STREAM
        return [(user_issue_remote.socket.AF_INET, user_issue_remote.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    class FakeResponse:
        status_code = 204
        text = ""

    def fake_post(*args: object, **kwargs: object) -> FakeResponse:
        calls.append({"args": args, "kwargs": kwargs})
        return FakeResponse()

    monkeypatch.setattr(user_issue_remote.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(user_issue_remote.requests, "post", fake_post)

    result = user_issue_remote.post_remote_issue_payload(build_remote_issue_payload({"summary": "Conversion warning"}))

    assert result["ok"] is True
    assert result["status_code"] == 204
    assert len(calls) == 1
    assert calls[0]["args"][0] == "https://collector.example/api/user-issues/ingest"
    assert calls[0]["kwargs"]["allow_redirects"] is False


def test_remote_issue_post_rejects_collector_redirects(monkeypatch, tmp_path: Path):
    _enable_remote_issue_reporting(monkeypatch, tmp_path)
    calls: list[dict] = []

    def fake_getaddrinfo(host: str, port: object, **kwargs: object) -> list[tuple]:
        assert host == "collector.example"
        assert port is None
        assert kwargs["type"] == user_issue_remote.socket.SOCK_STREAM
        return [(user_issue_remote.socket.AF_INET, user_issue_remote.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    class FakeResponse:
        status_code = 302
        text = ""

    def fake_post(*args: object, **kwargs: object) -> FakeResponse:
        calls.append({"args": args, "kwargs": kwargs})
        return FakeResponse()

    monkeypatch.setattr(user_issue_remote.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(user_issue_remote.requests, "post", fake_post)

    result = user_issue_remote.post_remote_issue_payload(build_remote_issue_payload({"summary": "Conversion warning"}))

    assert result == {
        "ok": False,
        "enabled": True,
        "status_code": 302,
        "error": "remote redirects are not allowed",
    }
    assert len(calls) == 1
    assert calls[0]["args"][0] == "https://collector.example/api/user-issues/ingest"
    assert calls[0]["kwargs"]["allow_redirects"] is False


def test_remote_issue_payload_uses_anonymous_quality_data_client_id(monkeypatch, tmp_path: Path):
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(
        json.dumps(
            {
                "quality_data_sharing_enabled": True,
                "quality_data_client_id": "local-random-client-id",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KB_USER_PREFS_PATH", str(prefs_path))

    payload = build_remote_issue_payload({"summary": "Conversion warning"})

    assert payload["client"]["installation_id"]
    assert payload["client"]["installation_id"] != "local-random-client-id"


def test_record_library_quality_issues_captures_hidden_quality_problems(tmp_path: Path):
    result = record_library_quality_issues(
        tmp_path / "user_issues.sqlite3",
        {
            "status": "error",
            "scope": "all",
            "summary": {"converted": 2, "review": 1},
            "top_issues": [
                {
                    "code": "missing_images",
                    "label": "Missing image assets",
                    "severity": "error",
                    "papers": 1,
                    "count": 3,
                    "repair_strategy": "Rebuild figures",
                    "samples": [
                        {
                            "document": "Secret Lab Draft.pdf",
                            "snippet": "A private paragraph from the converted paper.",
                            "code": "missing_images",
                        }
                    ],
                }
            ],
            "domains": {
                "research_qa": {
                    "available": True,
                    "status": "error",
                    "summary": {"failed": 1},
                    "top_failures": [{"name": "citation_missing", "count": 1}],
                }
            },
            "failure_cases": [
                {
                    "id": "case-1",
                    "question": "Why did citation lookup fail?",
                    "failures": [{"name": "citation_missing"}],
                }
            ],
        },
    )

    assert result["recorded"] == 3
    issues = UserIssueStore(tmp_path / "user_issues.sqlite3").list_issues(status="all")
    summaries = {str(item["summary"]) for item in issues}
    assert "Missing image assets" in summaries
    assert "Research QA failure" in summaries
    assert "Why did citation lookup fail?" not in str(issues)
    assert "Secret Lab Draft" not in str(issues)
    assert "private paragraph" not in str(issues)
    quality_issue = next(item for item in issues if item["summary"] == "Missing image assets")
    assert quality_issue["payload"]["issue"]["papers"] == 1
    assert quality_issue["payload"]["issue"]["samples"] == "[redacted]"
    research_issue = next(item for item in issues if item["source"] == "research_qa_failure_case")
    assert research_issue["detail"] == "citation_missing"
    assert research_issue["payload"]["case"]["question"] == "[redacted]"
