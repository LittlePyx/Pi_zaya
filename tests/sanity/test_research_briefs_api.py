from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
from kb.chat_store import ChatStore


def _agent_payload(source_path: str) -> dict:
    return {
        "answer": "## Core finding\n\nThe selected paper reports a measured improvement [1].",
        "hits": [
            {
                "text": "The selected paper reports a measured improvement in the experiment.",
                "score": 8.5,
                "meta": {
                    "source_path": source_path,
                    "source_name": "Measured imaging paper",
                    "title": "Measured imaging paper",
                    "heading_path": "Results / Measurement",
                    "page": 5,
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
    }


def test_research_brief_api_generates_versions_and_exports(monkeypatch, tmp_path: Path) -> None:
    from api.routers import research_briefs as brief_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Imaging project")
    conv_id = store.create_conversation("Evidence", project_id=project_id)
    source_path = str(tmp_path / "measured-imaging.md")
    store.save_citation_shelf(
        project_id=project_id,
        scope="project",
        items=[
            {
                "key": "paper-a",
                "title": "Measured imaging paper",
                "sourcePath": source_path,
                "authors": "Ada Author",
                "year": "2025",
                "venue": "Optics Letters",
                "doi": "10.1000/measured",
            }
        ],
        open=True,
    )
    monkeypatch.setattr(brief_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(
        brief_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=tmp_path),
    )
    monkeypatch.setattr(
        brief_router,
        "run_research_agent",
        lambda *args, **kwargs: _agent_payload(source_path),
    )
    client = TestClient(app)

    generated = client.post(
        f"/api/projects/{project_id}/research-briefs/generate",
        json={
            "title": "Measured-imaging brief",
            "objective": "Compare measured performance.",
            "item_keys": ["paper-a"],
            "source_conv_id": conv_id,
            "locale": "en",
        },
    )
    assert generated.status_code == 200
    record = generated.json()
    assert record["quality_status"] == "verified"
    assert record["quality"]["support_ratio"] == 1.0
    assert record["quality"]["unexpected_sources"] == []
    assert record["evidence"][0]["heading_path"] == "Results / Measurement"
    assert record["bibliography"][0]["doi"] == "10.1000/measured"

    brief_id = record["id"]
    edited = client.patch(
        f"/api/research-briefs/{brief_id}",
        json={
            "expected_revision": 1,
            "content_markdown": f"{record['content_markdown']}\n\nEditorial note.",
        },
    )
    assert edited.status_code == 200
    edited_record = edited.json()
    assert edited_record["revision"] == 2
    assert edited_record["quality_status"] == "draft"
    assert edited_record["quality"]["edited_after_verification"] is True

    conflict = client.patch(
        f"/api/research-briefs/{brief_id}",
        json={"expected_revision": 1, "title": "Stale title"},
    )
    assert conflict.status_code == 409

    revisions = client.get(f"/api/research-briefs/{brief_id}/revisions")
    assert revisions.status_code == 200
    assert [item["revision"] for item in revisions.json()] == [2, 1]

    restored = client.post(
        f"/api/research-briefs/{brief_id}/restore",
        json={"revision": 1, "expected_revision": 2},
    )
    assert restored.status_code == 200
    assert restored.json()["revision"] == 3
    assert restored.json()["quality_status"] == "verified"

    markdown = client.get(f"/api/research-briefs/{brief_id}/export?format=markdown")
    assert markdown.status_code == 200
    assert "Evidence appendix" in markdown.text
    assert "filename=" in markdown.headers["content-disposition"]

    docx = client.get(f"/api/research-briefs/{brief_id}/export?format=docx")
    assert docx.status_code == 200
    assert docx.content.startswith(b"PK")
    bibtex = client.get(f"/api/research-briefs/{brief_id}/export?format=bibtex")
    assert "10.1000/measured" in bibtex.text
    ris = client.get(f"/api/research-briefs/{brief_id}/export?format=ris")
    assert "TY  - JOUR" in ris.text


def test_research_brief_api_rejects_nonlocal_shelf_entries(monkeypatch, tmp_path: Path) -> None:
    from api.routers import research_briefs as brief_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Metadata-only project")
    store.save_citation_shelf(
        project_id=project_id,
        scope="project",
        items=[
            {"key": "local", "title": "Local", "sourcePath": str(tmp_path / "local.md")},
            {"key": "metadata-only", "title": "Metadata only", "doi": "10.1000/metadata"},
        ],
        open=True,
    )
    monkeypatch.setattr(brief_router, "get_chat_store", lambda: store)
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{project_id}/research-briefs/generate",
        json={"title": "Should not generate", "item_keys": ["local", "metadata-only"]},
    )
    assert response.status_code == 400
    assert "local full-text evidence" in response.json()["detail"]


def test_research_brief_api_requires_revision_before_regeneration(monkeypatch, tmp_path: Path) -> None:
    from api.routers import research_briefs as brief_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Versioned project")
    source_path = str(tmp_path / "paper.md")
    store.save_citation_shelf(
        project_id=project_id,
        scope="project",
        items=[{"key": "paper", "title": "Paper", "sourcePath": source_path}],
        open=True,
    )
    brief = store.create_research_brief(project_id=project_id, title="Existing")
    assert brief is not None
    monkeypatch.setattr(brief_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(
        brief_router,
        "run_research_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("agent must not run")),
    )
    client = TestClient(app)

    response = client.post(
        f"/api/projects/{project_id}/research-briefs/generate",
        json={"title": "Regenerate", "item_keys": ["paper"], "brief_id": brief["id"]},
    )

    assert response.status_code == 400
    assert "expected_revision" in response.json()["detail"]


def test_research_brief_incremental_update_plan_preserves_unaffected_content_and_reaudits(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from api.routers import research_briefs as brief_router
    from kb.evidence_matrix import evidence_matrix_hits
    from kb.research_brief import research_brief_evidence
    from kb.research_brief_lineage import matrix_contract_fingerprint

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Incremental brief project")
    row = {
        "id": "row-a",
        "paper": "Paper A",
        "source_name": "Paper A",
        "source_path": "F:/papers/a.md",
        "source_status": "active",
        "notes": "",
        "cells": {
            "method": {
                "field": "method",
                "value": "calibrated acquisition",
                "support_status": "grounded",
                "evidence_ids": ["ev-method"],
                "manual_override": False,
            },
            "key_result": {
                "field": "key_result",
                "value": "1.0 dB",
                "support_status": "grounded",
                "evidence_ids": ["ev-result-old"],
                "manual_override": False,
            },
        },
    }
    evidence = [
        {
            "id": "ev-method",
            "field": "method",
            "source_path": "F:/papers/a.md",
            "source_name": "Paper A",
            "evidence_quote": "Paper A uses calibrated acquisition.",
        },
        {
            "id": "ev-result-old",
            "field": "key_result",
            "source_path": "F:/papers/a.md",
            "source_name": "Paper A",
            "evidence_quote": "Paper A reports 1.0 dB.",
        },
    ]
    source_items = [
        {"key": "paper-a", "title": "Paper A", "sourcePath": "F:/papers/a.md"}
    ]
    matrix = store.create_evidence_matrix(
        project_id=project_id,
        title="Verified matrix",
        rows=[row],
        evidence=evidence,
        source_items=source_items,
        quality_status="verified",
        quality={"supported_cell_count": 2},
    )
    assert matrix is not None
    original_content = (
        "## Findings\n\n"
        "- Paper A uses calibrated acquisition [1].\n\n"
        "### Quantitative evidence\n\n"
        "- Paper A reports 1.0 dB [2]."
    )
    brief = store.create_research_brief(
        project_id=project_id,
        title="Measured brief",
        objective="Summarize Paper A.",
        content_markdown=original_content,
        evidence=research_brief_evidence(evidence_matrix_hits(matrix, limit=20)),
        quality_status="verified",
        quality={
            "source_matrix_id": matrix["id"],
            "source_matrix_revision": 1,
            "source_matrix_quality_status": "verified",
            "source_matrix_title": matrix["title"],
            "source_matrix_fingerprint": matrix_contract_fingerprint(matrix),
        },
    )
    assert brief is not None
    rejected_brief = store.create_research_brief(
        project_id=project_id,
        title="Measured brief with retained text",
        objective="Summarize Paper A.",
        content_markdown=original_content,
        evidence=research_brief_evidence(evidence_matrix_hits(matrix, limit=20)),
        quality_status="verified",
        quality={
            "source_matrix_id": matrix["id"],
            "source_matrix_revision": 1,
            "source_matrix_quality_status": "verified",
            "source_matrix_title": matrix["title"],
            "source_matrix_fingerprint": matrix_contract_fingerprint(matrix),
        },
    )
    assert rejected_brief is not None

    current_row = dict(row)
    current_row["cells"] = {**row["cells"]}
    current_row["cells"]["key_result"] = {
        **row["cells"]["key_result"],
        "value": "1.4 dB",
        "evidence_ids": ["ev-result-new"],
    }
    current_evidence = [
        evidence[0],
        {
            **evidence[1],
            "id": "ev-result-new",
            "evidence_quote": "Paper A reports 1.4 dB.",
        },
    ]
    updated_matrix, conflict = store.update_evidence_matrix(
        matrix["id"],
        expected_revision=1,
        rows=[current_row],
        evidence=current_evidence,
    )
    assert conflict is False
    assert updated_matrix is not None

    monkeypatch.setattr(brief_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(brief_router, "get_settings", lambda: SimpleNamespace(db_dir=tmp_path))
    monkeypatch.setattr(
        brief_router,
        "generate_grounded_answer",
        lambda *args, **kwargs: {"answer": "- Paper A reports 1.4 dB [1]."},
    )
    client = TestClient(app)

    full_replace = client.post(
        f"/api/projects/{project_id}/research-briefs/generate",
        json={
            "title": brief["title"],
            "objective": brief["objective"],
            "brief_id": brief["id"],
            "matrix_id": matrix["id"],
            "expected_revision": 1,
            "locale": "en",
        },
    )
    assert full_replace.status_code == 409
    assert "incremental update plan" in full_replace.json()["detail"]

    planned = client.post(
        f"/api/research-briefs/{brief['id']}/update-plans",
        json={"expected_revision": 1, "locale": "en"},
    )
    assert planned.status_code == 200, planned.text
    plan = planned.json()
    assert len(plan["items"]) == 1
    assert plan["items"][0]["citation_numbers_before"] == [2]
    assert "1.4 dB" in plan["items"][0]["proposed_markdown"]
    assert "Paper A uses calibrated acquisition [1]." in plan["preview_content_markdown"]
    assert "### Quantitative evidence" in plan["preview_content_markdown"]

    applied = client.post(
        f"/api/research-briefs/{brief['id']}/update-plans/{plan['id']}/apply",
        json={
            "expected_revision": 1,
            "decisions": [{"item_id": plan["items"][0]["id"], "decision": "accept"}],
        },
    )
    assert applied.status_code == 200, applied.text
    record = applied.json()
    assert record["revision"] == 2
    assert record["quality_status"] == "verified", record["quality"]["reasons"]
    assert record["lineage"]["status"] == "current"
    assert record["quality"]["incremental_update"]["rejected_item_ids"] == []
    assert "Paper A uses calibrated acquisition [1]." in record["content_markdown"]
    assert "### Quantitative evidence" in record["content_markdown"]
    assert "1.4 dB [2]" in record["content_markdown"]
    assert "1.0 dB [2]" not in record["content_markdown"]

    reused = client.post(
        f"/api/research-briefs/{brief['id']}/update-plans/{plan['id']}/apply",
        json={"expected_revision": 2, "decisions": []},
    )
    assert reused.status_code == 404

    rejected_plan_response = client.post(
        f"/api/research-briefs/{rejected_brief['id']}/update-plans",
        json={"expected_revision": 1, "locale": "en"},
    )
    assert rejected_plan_response.status_code == 200, rejected_plan_response.text
    rejected_plan = rejected_plan_response.json()
    rejected_apply = client.post(
        f"/api/research-briefs/{rejected_brief['id']}/update-plans/{rejected_plan['id']}/apply",
        json={"expected_revision": 1, "decisions": []},
    )
    assert rejected_apply.status_code == 200, rejected_apply.text
    retained = rejected_apply.json()
    assert retained["revision"] == 2
    assert retained["content_markdown"] == original_content
    assert retained["quality_status"] == "needs_review"
    assert "incremental_update_rejected_changes" in retained["quality"]["reasons"]
    assert retained["quality"]["incremental_update"]["accepted_item_ids"] == []
    assert retained["quality"]["incremental_update"]["rejected_item_ids"] == [
        rejected_plan["items"][0]["id"]
    ]
    assert retained["agent_trace"]["summary"]["quality_gate_status"] == "passed"
