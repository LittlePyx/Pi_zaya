from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
from kb.chat_store import ChatStore


def _generated_matrix(source_path: str) -> tuple[list[dict], list[dict], list[dict]]:
    evidence = {
        "id": "ev-method",
        "field": "method",
        "source_item_key": "paper-a",
        "source_path": source_path,
        "source_name": "Paper A",
        "title": "Paper A",
        "heading_path": "Method / Architecture",
        "page_start": 3,
        "block_id": "block-method",
        "evidence_quote": "The method uses a coded optical network.",
        "score": 8.0,
    }
    row = {
        "id": "row-a",
        "source_item_key": "paper-a",
        "paper": "Paper A",
        "source_name": "Paper A",
        "source_path": source_path,
        "notes": "",
        "source_status": "active",
        "cells": {
            "method": {
                "field": "method",
                "value": evidence["evidence_quote"],
                "support_status": "grounded",
                "evidence_ids": [evidence["id"]],
                "manual_override": False,
            }
        },
    }
    return [row], [evidence], []


def test_evidence_matrix_api_generates_versions_edits_and_exports(monkeypatch, tmp_path: Path) -> None:
    from api.routers import evidence_matrices as matrix_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Matrix project")
    conv_id = store.create_conversation("Evidence", project_id=project_id)
    source_path = str(tmp_path / "paper-a.md")
    store.save_citation_shelf(
        project_id=project_id,
        scope="project",
        items=[{"key": "paper-a", "title": "Paper A", "sourcePath": source_path}],
        open=True,
    )
    monkeypatch.setattr(matrix_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(matrix_router, "get_settings", lambda: SimpleNamespace(db_dir=tmp_path))
    def fake_build(*args, **kwargs):
        rows, evidence, flags = _generated_matrix(source_path)
        existing = list(kwargs.get("existing_rows") or [])
        if existing:
            rows[0]["notes"] = str(existing[0].get("notes") or "")
        return rows, evidence, flags

    monkeypatch.setattr(matrix_router, "build_project_evidence_matrix", fake_build)
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{project_id}/evidence-matrices/generate",
        json={
            "title": "Optical comparison",
            "objective": "Compare methods.",
            "item_keys": ["paper-a"],
            "source_conv_id": conv_id,
        },
    )
    assert response.status_code == 200
    record = response.json()
    assert record["quality_status"] == "verified"
    assert record["quality"]["covered_source_count"] == 1
    assert record["rows"][0]["cells"]["method"]["support_status"] == "grounded"
    assert record["evidence"][0]["heading_path"] == "Method / Architecture"
    matrix_id = record["id"]

    edited = client.patch(
        f"/api/evidence-matrices/{matrix_id}",
        json={
            "expected_revision": 1,
            "row_updates": [
                {
                    "row_id": "row-a",
                    "notes": "Keep this note on refresh.",
                    "cells": [{"field": "method", "value": "Manual method summary."}],
                }
            ],
        },
    )
    assert edited.status_code == 200
    edited_record = edited.json()
    assert edited_record["revision"] == 2
    assert edited_record["quality_status"] == "needs_review"
    assert edited_record["rows"][0]["cells"]["method"]["manual_override"] is True
    assert "edited_after_verification" in edited_record["quality"]["reasons"]

    conflict = client.patch(
        f"/api/evidence-matrices/{matrix_id}",
        json={"expected_revision": 1, "title": "Stale title"},
    )
    assert conflict.status_code == 409

    refreshed = client.post(
        f"/api/projects/{project_id}/evidence-matrices/generate",
        json={
            "title": "Optical comparison",
            "objective": "Compare methods.",
            "item_keys": ["paper-a"],
            "matrix_id": matrix_id,
            "expected_revision": 2,
        },
    )
    assert refreshed.status_code == 200
    assert refreshed.json()["revision"] == 3
    assert refreshed.json()["quality_status"] == "verified"
    assert refreshed.json()["rows"][0]["notes"] == "Keep this note on refresh."

    revisions = client.get(f"/api/evidence-matrices/{matrix_id}/revisions")
    assert revisions.status_code == 200
    assert [item["revision"] for item in revisions.json()] == [3, 2, 1]

    markdown = client.get(f"/api/evidence-matrices/{matrix_id}/export?format=markdown")
    assert markdown.status_code == 200
    assert "Evidence appendix" in markdown.text
    csv_response = client.get(f"/api/evidence-matrices/{matrix_id}/export?format=csv")
    assert csv_response.status_code == 200
    assert "Paper A" in csv_response.content.decode("utf-8-sig")
    xlsx = client.get(f"/api/evidence-matrices/{matrix_id}/export?format=xlsx")
    assert xlsx.status_code == 200
    assert xlsx.content.startswith(b"PK")


def test_research_brief_can_use_only_a_verified_matrix(monkeypatch, tmp_path: Path) -> None:
    from api.routers import research_briefs as brief_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Matrix-backed brief")
    source_path = str(tmp_path / "paper-a.md")
    rows, evidence, _flags = _generated_matrix(source_path)
    matrix = store.create_evidence_matrix(
        project_id=project_id,
        title="Verified matrix",
        objective="Compare methods.",
        rows=rows,
        evidence=evidence,
        source_items=[{"key": "paper-a", "title": "Paper A", "sourcePath": source_path}],
        quality_status="verified",
        quality={"supported_cell_count": 1},
    )
    assert matrix is not None
    monkeypatch.setattr(brief_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(brief_router, "get_settings", lambda: SimpleNamespace(db_dir=tmp_path))
    monkeypatch.setattr(
        brief_router,
        "generate_research_brief_from_matrix",
        lambda *args, **kwargs: {
            "answer": "The method uses a coded optical network [1].",
            "hits": [
                {
                    "text": "The method uses a coded optical network.",
                    "score": 8.0,
                    "meta": {
                        "source_path": source_path,
                        "source_name": "Paper A",
                        "heading_path": "Method / Architecture",
                    },
                }
            ],
            "agent_trace": {
                "status": "done",
                "errors": [],
                "verification": {
                    "total_claims": 1,
                    "supported_claims": 1,
                    "unsupported_claims": 0,
                    "support_ratio": 1.0,
                    "evidence_status": "grounded",
                },
                "summary": {"query_scope": "basket", "quality_gate_status": "passed"},
            },
        },
    )
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{project_id}/research-briefs/generate",
        json={"title": "Matrix brief", "objective": "Compare methods.", "matrix_id": matrix["id"]},
    )
    assert response.status_code == 200
    brief = response.json()
    assert brief["quality_status"] == "verified"
    assert brief["quality"]["source_matrix_id"] == matrix["id"]
    assert brief["quality"]["source_matrix_revision"] == 1

    draft = store.create_evidence_matrix(project_id=project_id, title="Draft matrix")
    assert draft is not None
    rejected = client.post(
        f"/api/projects/{project_id}/research-briefs/generate",
        json={"title": "Rejected", "matrix_id": draft["id"]},
    )
    assert rejected.status_code == 400
    assert "verified evidence matrix" in rejected.json()["detail"]
