from __future__ import annotations

from pathlib import Path

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


def test_user_issue_store_forwards_recorded_issue_to_remote_reporter(monkeypatch, tmp_path: Path):
    sent: list[dict] = []
    monkeypatch.setattr("kb.user_issue_store.report_user_issue_remote", lambda issue: sent.append(dict(issue)))

    store = UserIssueStore(tmp_path / "user_issues.sqlite3")
    store.record_issue(
        source="library_quality_overview",
        domain="conversion",
        severity="warning",
        summary="Missing image assets",
    )

    assert len(sent) == 1
    assert sent[0]["summary"] == "Missing image assets"


def test_remote_issue_payload_redacts_local_paths_and_tokens(monkeypatch):
    monkeypatch.setenv("KB_USER_ISSUES_CLIENT_ID", "lab-machine-01")
    payload = build_remote_issue_payload(
        {
            "fingerprint": "fp1",
            "source": "frontend",
            "domain": "runtime",
            "severity": "error",
            "summary": "Failed reading C:/Users/Alice/private.pdf",
            "detail": "token sk-secretsecretsecret at alice@example.com",
            "context": {"source_path": r"C:\Users\Alice\paper.pdf", "route": "/library"},
            "payload": {"filename": "paper.pdf", "code": "missing_images"},
        }
    )

    text = str(payload)
    assert "Alice" not in text
    assert "sk-secret" not in text
    assert "alice@example.com" not in text
    assert payload["issue"]["payload"]["code"] == "missing_images"
    assert payload["issue"]["context"]["source_path"] == "[redacted]"
    assert payload["client"]["installation_id"]


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
    assert "Why did citation lookup fail?" in summaries
