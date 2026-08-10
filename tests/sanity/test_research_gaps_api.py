from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
from kb.chat_store import ChatStore


def test_research_gap_scan_ignore_search_and_confirm(monkeypatch, tmp_path: Path) -> None:
    from api.routers import research_gaps as gap_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Gap workflow")
    source = tmp_path / "paper-a.md"
    source.write_text("Paper A", encoding="utf-8")
    matrix = store.create_evidence_matrix(
        project_id=project_id,
        title="Gap matrix",
        objective="Compare imaging methods.",
        rows=[
            {
                "id": "row-a",
                "paper": "Paper A",
                "source_name": "Paper A",
                "source_path": str(source),
                "cells": {},
            }
        ],
        source_items=[{"key": "paper-a", "sourceName": "Paper A", "sourcePath": str(source)}],
        quality_status="verified",
        quality={
            "missing_cells": [
                {"row_id": "row-a", "field": "limitation"},
                {"row_id": "row-a", "field": "metric"},
            ],
            "unsupported_cells": [],
            "reasons": [],
        },
    )
    assert matrix is not None
    candidate_path = tmp_path / "paper-b.md"
    candidate_path.write_text("Paper B candidate", encoding="utf-8")
    candidate = {
        "id": "candidate-1",
        "source_path": str(candidate_path),
        "source_name": "Paper B",
        "title": "Paper B",
        "evidence_quote": "The method reports a limitation under motion.",
        "heading_path": "Discussion",
        "location_label": "Discussion",
        "page_start": 5,
        "page_end": 5,
        "block_id": "blk-5",
        "anchor_id": "",
    }

    monkeypatch.setattr(gap_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(gap_router, "get_settings", lambda: SimpleNamespace(db_dir=tmp_path))
    monkeypatch.setattr(gap_router, "_scan_project_evidence_changes", lambda _project_id: {"items": []})
    monkeypatch.setattr(gap_router, "_gap_candidates", lambda _gap, limit: [candidate])
    client = TestClient(app)

    scanned = client.post(f"/api/projects/{project_id}/research-gaps/scan")
    assert scanned.status_code == 200
    items = scanned.json()["items"]
    assert len(items) == 2
    assert scanned.json()["summary"]["searchable"] == 2
    limitation = next(item for item in items if item["field"] == "limitation")
    metric = next(item for item in items if item["field"] == "metric")

    candidates = client.get(f"/api/projects/{project_id}/research-gaps/{limitation['id']}/candidates")
    assert candidates.status_code == 200
    assert candidates.json()["items"][0]["evidence_quote"] == candidate["evidence_quote"]

    confirmed = client.post(
        f"/api/projects/{project_id}/research-gaps/{limitation['id']}/candidates/{candidate['id']}/confirm"
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["shelf"]["items"][0]["anchor"] == "blk-5"
    assert confirmed.json()["gap"]["status"] == "in_progress"
    assert confirmed.json()["shelf"]["items"][0]["sourcePath"] == str(candidate_path)
    assert confirmed.json()["shelf"]["items"][0]["shelfExcerpt"] == candidate["evidence_quote"]

    ignored = client.post(
        f"/api/projects/{project_id}/research-gaps/{metric['id']}/ignore",
        json={"reason": "Metric is outside this review."},
    )
    assert ignored.status_code == 200
    assert ignored.json()["status"] == "ignored"
    active = client.get(f"/api/projects/{project_id}/research-gaps").json()["items"]
    assert [item["id"] for item in active] == [limitation["id"]]
