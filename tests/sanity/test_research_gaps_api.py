from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
from kb.chat_store import ChatStore
from kb.research_brief_lineage import matrix_contract_fingerprint


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
                "source_status": "active",
                "cells": {
                    "method": {
                        "field": "method",
                        "value": "We propose a coded imaging method.",
                        "support_status": "grounded",
                        "evidence_ids": ["ev-method"],
                        "manual_override": False,
                    },
                    "dataset_or_experiment": {
                        "field": "dataset_or_experiment",
                        "value": "Experiments use a dynamic imaging benchmark.",
                        "support_status": "grounded",
                        "evidence_ids": ["ev-dataset"],
                        "manual_override": False,
                    },
                },
            }
        ],
        evidence=[
            {
                "id": "ev-method",
                "field": "method",
                "source_path": str(source),
                "evidence_quote": "We propose a coded imaging method.",
            },
            {
                "id": "ev-dataset",
                "field": "dataset_or_experiment",
                "source_path": str(source),
                "evidence_quote": "Experiments use a dynamic imaging benchmark.",
            },
        ],
        source_items=[{"key": "paper-a", "sourceName": "Paper A", "sourcePath": str(source)}],
        quality_status="verified",
        quality={
            "missing_cells": [
                {"row_id": "row-a", "field": "limitation"},
                {"row_id": "row-a", "field": "metric"},
                {"row_id": "row-a", "field": "key_result"},
            ],
            "unsupported_cells": [],
            "reasons": [],
        },
    )
    assert matrix is not None
    brief = store.create_research_brief(
        project_id=project_id,
        title="Living gap brief",
        objective="Compare imaging methods.",
        content_markdown="# Brief\n\nPaper A uses a coded imaging method [1].",
        evidence=[
            {
                "citation_number": 1,
                "source_path": str(source),
                "source_name": "Paper A",
                "evidence_quote": "We propose a coded imaging method.",
            }
        ],
        quality_status="verified",
        quality={
            "source_matrix_id": str(matrix["id"]),
            "source_matrix_revision": int(matrix["revision"]),
            "source_matrix_quality_status": str(matrix["quality_status"]),
            "source_matrix_title": str(matrix["title"]),
            "source_matrix_fingerprint": matrix_contract_fingerprint(matrix),
        },
    )
    assert brief is not None
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
    repair = {
        "id": "repair-1",
        "gap_id": "",
        "gap_key": "",
        "matrix_id": str(matrix["id"]),
        "matrix_revision": int(matrix["revision"]),
        "row_id": "row-a",
        "field": "limitation",
        "value": "However, our method remains limited under motion.",
        "source_path": str(source),
        "source_name": "Paper A",
        "title": "Paper A",
        "chunk_id": "paper-a:7",
        "evidence_id": "ev-repair-1",
        "evidence_quote": "However, our method remains limited under motion.",
        "heading_path": "Discussion",
        "location_label": "Discussion",
        "page_start": 7,
        "page_end": 7,
        "block_id": "blk-7",
        "anchor_id": "",
        "score": 8.0,
        "same_source_verified": True,
        "match_reason": "Exact same-source field evidence.",
    }

    monkeypatch.setattr(gap_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(gap_router, "get_settings", lambda: SimpleNamespace(db_dir=tmp_path))
    monkeypatch.setattr(gap_router, "_scan_project_evidence_changes", lambda _project_id: {"items": []})
    monkeypatch.setattr(gap_router, "_gap_candidates", lambda _gap, limit: [candidate])
    monkeypatch.setattr(gap_router, "_indexed_source_is_fresh", lambda _source_path: True)
    monkeypatch.setattr(
        gap_router,
        "_gap_repairs",
        lambda gap, _matrix, limit: [{**repair, "gap_id": gap["id"], "gap_key": gap["gap_key"]}],
    )
    client = TestClient(app)

    scanned = client.post(f"/api/projects/{project_id}/research-gaps/scan")
    assert scanned.status_code == 200
    items = scanned.json()["items"]
    assert len(items) == 3
    assert scanned.json()["summary"]["searchable"] == 3
    limitation = next(item for item in items if item["field"] == "limitation")
    metric = next(item for item in items if item["field"] == "metric")
    key_result = next(item for item in items if item["field"] == "key_result")

    repairs = client.get(f"/api/projects/{project_id}/research-gaps/{limitation['id']}/repairs")
    assert repairs.status_code == 200
    assert repairs.json()["items"][0]["source_path"] == str(source)
    applied = client.post(
        f"/api/projects/{project_id}/research-gaps/{limitation['id']}/repairs/{repair['id']}/apply",
        json={"expected_revision": matrix["revision"]},
    )
    assert applied.status_code == 200
    assert applied.json()["gap"]["status"] == "resolved"
    assert applied.json()["matrix"]["revision"] == matrix["revision"] + 1
    repaired_cell = applied.json()["matrix"]["rows"][0]["cells"]["limitation"]
    assert repaired_cell["support_status"] == "grounded"
    assert repaired_cell["manual_override"] is False
    assert repaired_cell["evidence_ids"] == [repair["evidence_id"]]
    affected_briefs = applied.json()["affected_briefs"]
    assert len(affected_briefs) == 1
    assert affected_briefs[0]["id"] == brief["id"]
    assert affected_briefs[0]["lineage_status"] == "matrix_updated"
    assert affected_briefs[0]["update_ready"] is True
    assert affected_briefs[0]["impact"]["changed_field_count"] == 1

    candidates = client.get(f"/api/projects/{project_id}/research-gaps/{metric['id']}/candidates")
    assert candidates.status_code == 200
    assert candidates.json()["items"][0]["evidence_quote"] == candidate["evidence_quote"]

    confirmed = client.post(
        f"/api/projects/{project_id}/research-gaps/{metric['id']}/candidates/{candidate['id']}/confirm"
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["shelf"]["items"][0]["anchor"] == "blk-5"
    assert confirmed.json()["gap"]["status"] == "in_progress"
    assert confirmed.json()["shelf"]["items"][0]["sourcePath"] == str(candidate_path)
    assert confirmed.json()["shelf"]["items"][0]["shelfExcerpt"] == candidate["evidence_quote"]

    ignored = client.post(
        f"/api/projects/{project_id}/research-gaps/{key_result['id']}/ignore",
        json={"reason": "Metric is outside this review."},
    )
    assert ignored.status_code == 200
    assert ignored.json()["status"] == "ignored"
    active = client.get(f"/api/projects/{project_id}/research-gaps").json()["items"]
    assert {item["kind"] for item in active} == {"brief_stale", "missing_cell"}
    assert next(item for item in active if item["kind"] == "missing_cell")["id"] == metric["id"]
