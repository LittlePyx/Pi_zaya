from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
from kb.chat_store import ChatStore


def test_project_research_status_refresh_returns_one_auditable_next_action(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from api.routers import research_gaps as gap_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Status workflow")
    source_a = tmp_path / "paper-a.md"
    source_b = tmp_path / "paper-b.md"
    source_a.write_text("Paper A evidence", encoding="utf-8")
    source_b.write_text("Paper B evidence", encoding="utf-8")
    matrix = store.create_evidence_matrix(
        project_id=project_id,
        title="Status matrix",
        objective="Compare Paper A and Paper B.",
        rows=[
            {
                "id": "row-a",
                "paper": "Paper A",
                "source_name": "Paper A",
                "source_path": str(source_a),
                "source_status": "active",
                "cells": {},
            },
            {
                "id": "row-b",
                "paper": "Paper B",
                "source_name": "Paper B",
                "source_path": str(source_b),
                "source_status": "active",
                "cells": {},
            },
        ],
        evidence=[],
        source_items=[
            {"key": "paper-a", "sourceName": "Paper A", "sourcePath": str(source_a)},
            {"key": "paper-b", "sourceName": "Paper B", "sourcePath": str(source_b)},
        ],
        quality_status="verified",
        quality={"missing_cells": [], "unsupported_cells": [], "reasons": []},
    )
    assert matrix is not None

    monkeypatch.setattr(gap_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(gap_router, "get_settings", lambda: SimpleNamespace(db_dir=tmp_path))
    monkeypatch.setattr(gap_router, "_scan_project_evidence_changes", lambda _project_id: {"items": []})
    monkeypatch.setattr(
        gap_router,
        "_project_comparison_candidate_scan",
        lambda matrices: {
            "contract_version": 1,
            "candidate_count": 2,
            "first_candidate_matrix_id": str(matrices[0]["id"]),
            "eligible_matrix_count": 1,
            "scanned_matrix_count": 1,
            "skipped_stale_matrix_count": 0,
            "scan_complete": True,
            "examined_row_pairs": 1,
            "structured_observation_count": 4,
            "matrix_results": [],
            "elapsed_ms": 4.25,
        },
    )
    client = TestClient(app)

    refreshed = client.post(f"/api/projects/{project_id}/research-status/refresh")
    assert refreshed.status_code == 200
    payload = refreshed.json()
    assert payload["contract_version"] == 1
    assert payload["project"]["id"] == project_id
    assert payload["stages"]["sources"]["project_source_count"] == 2
    assert payload["stages"]["comparisons"]["pending_candidate_count"] == 2
    assert payload["comparison_scan"]["scanned_matrix_count"] == 1
    assert payload["recommended_action"] == {
        "code": "review_comparison_candidates",
        "target": "evidence_matrix",
        "priority": 75,
        "reason": "evidence_bound_comparisons_await_human_confirmation",
        "matrix_id": matrix["id"],
        "matrix_title": matrix["title"],
        "matrix_revision": matrix["revision"],
        "brief_id": "",
        "brief_title": "",
        "brief_revision": 0,
        "gap_count": 0,
        "candidate_count": 2,
        "workspace_tab": "comparisons",
    }
    assert payload["phase_timings_ms"]["scan_comparison_candidates"] == 4.25
    assert payload["phase_timings_ms"]["total"] >= 0

    snapshot = client.get(f"/api/projects/{project_id}/research-status")
    assert snapshot.status_code == 200
    assert snapshot.json()["recommended_action"]["code"] == "refresh_project_status"
    assert snapshot.json()["comparison_scan"]["scan_complete"] is False

    missing = client.post("/api/projects/missing/research-status/refresh")
    assert missing.status_code == 404
