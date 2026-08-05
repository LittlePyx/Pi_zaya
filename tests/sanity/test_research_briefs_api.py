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
