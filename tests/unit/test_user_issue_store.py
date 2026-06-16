from __future__ import annotations

from pathlib import Path

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
