from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from kb.chat_store import ChatStore


def test_chat_research_notes_api_roundtrip_and_revision_conflict(monkeypatch, tmp_path: Path) -> None:
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Notes")
    conv_id = store.create_conversation("Answers", project_id=project_id)
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(
        chat_router,
        "_dangerous_auto_snapshot",
        lambda *args, **kwargs: {"ok": True, "block_operation": False},
    )
    client = TestClient(app)

    created_response = client.post(
        "/api/chat/research-notes",
        json={
            "project_id": project_id,
            "source_conv_id": conv_id,
            "title": "Traceable note",
            "content_markdown": "## Finding\n\nGrounded answer [1].",
            "source_state": {
                "links": [
                    {
                        "kind": "answer",
                        "conversation_id": conv_id,
                        "message_id": 12,
                        "label": "Question",
                    }
                ]
            },
        },
    )
    assert created_response.status_code == 200
    created = created_response.json()
    note_id = created["id"]
    assert created["revision"] == 1

    listed = client.get(f"/api/chat/research-notes?project_id={project_id}")
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == note_id
    assert listed.json()[0]["content_markdown"] == ""
    assert listed.json()[0]["source_state"]["links"][0]["message_id"] == 12

    fetched = client.get(f"/api/chat/research-notes/{note_id}")
    assert fetched.status_code == 200
    assert "Grounded answer" in fetched.json()["content_markdown"]

    updated = client.patch(
        f"/api/chat/research-notes/{note_id}",
        json={
            "expected_revision": 1,
            "title": "Renamed note",
            "content_markdown": "Updated content",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["revision"] == 2
    assert updated.json()["title"] == "Renamed note"

    conflict = client.patch(
        f"/api/chat/research-notes/{note_id}",
        json={"expected_revision": 1, "content_markdown": "Stale"},
    )
    assert conflict.status_code == 409
    assert store.get_research_note(note_id)["content_markdown"] == "Updated content"

    deleted = client.delete(f"/api/chat/research-notes/{note_id}")
    assert deleted.status_code == 200
    assert deleted.json()["auto_backup"]["ok"] is True
    assert client.get(f"/api/chat/research-notes/{note_id}").status_code == 404


def test_chat_research_notes_api_rejects_invalid_scope_and_blank_content(monkeypatch, tmp_path: Path) -> None:
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    client = TestClient(app)

    invalid_scope = client.post(
        "/api/chat/research-notes",
        json={
            "project_id": "missing-project",
            "title": "Note",
            "content_markdown": "Content",
        },
    )
    blank = client.post(
        "/api/chat/research-notes",
        json={"title": "Note", "content_markdown": "   "},
    )

    assert invalid_scope.status_code == 404
    assert blank.status_code == 422


def test_chat_research_note_evidence_context_verifies_source_without_exposing_internal_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from api.routers import chat as chat_router

    md_path = tmp_path / "converted.md"
    md_path.write_text(
        """# Paper

<!-- kb_page: 4 -->

## Results

Context immediately before the evidence.

The measured response remained linear across the tested range.

Context immediately after the evidence.
""",
        encoding="utf-8",
    )
    from kb.source_blocks import load_source_blocks

    target = next(
        block
        for block in load_source_blocks(md_path)
        if "remained linear" in str(block.get("text") or "")
    )
    monkeypatch.setattr(chat_router, "_resolve_allowed_reader_source_path", lambda _path: str(md_path))
    monkeypatch.setattr(chat_router, "_resolve_paper_guide_md_path", lambda *args, **kwargs: md_path)
    monkeypatch.setattr(chat_router, "_md_dir", lambda: tmp_path)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/chat/research-note/evidence-context",
        json={
            "source_path": "library/paper.pdf",
            "source_name": "Paper",
            "heading_path": "Results",
            "evidence_quote": "The measured response remained linear across the tested range.",
            "block_id": target["block_id"],
            "page_start": 4,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "exact"
    assert payload["page_start"] == 4
    assert payload["reader_open"]["sourcePath"] == "library/paper.pdf"
    assert str(md_path) not in str(payload)
    assert client.post(
        "/api/chat/research-note/evidence-context",
        json={"source_path": "   "},
    ).status_code == 422


def test_chat_research_notes_workspace_list_and_metadata(monkeypatch, tmp_path: Path) -> None:
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Workspace")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    client = TestClient(app)

    created = client.post(
        "/api/chat/research-notes",
        json={
            "project_id": project_id,
            "title": "Evidence note",
            "content_markdown": "A rare-search-token appears in the body.",
            "tags": ["evidence", "review"],
            "pinned": True,
        },
    )
    assert created.status_code == 200
    note_id = created.json()["id"]

    found = client.get(
        "/api/chat/research-notes",
        params={"scope": "all", "query": "rare-search-token", "archived": "active"},
    )
    assert found.status_code == 200
    assert [item["id"] for item in found.json()] == [note_id]
    assert found.json()[0]["tags"] == ["evidence", "review"]
    assert found.json()[0]["pinned"] is True

    archived = client.patch(
        f"/api/chat/research-notes/{note_id}",
        json={
            "expected_revision": 1,
            "project_id": None,
            "tags": ["final"],
            "archived": True,
        },
    )
    assert archived.status_code == 200
    assert archived.json()["project_id"] is None
    assert archived.json()["tags"] == ["final"]
    assert archived.json()["archived"] is True

    assert client.get("/api/chat/research-notes", params={"scope": "all"}).json() == []
    archived_list = client.get(
        "/api/chat/research-notes",
        params={"scope": "all", "archived": "archived"},
    )
    assert [item["id"] for item in archived_list.json()] == [note_id]

    invalid_query = client.get(
        "/api/chat/research-notes",
        params={"scope": "workspace", "archived": "later"},
    )
    assert invalid_query.status_code == 422


def test_chat_research_outline_composition_survives_store_reopen(monkeypatch, tmp_path: Path) -> None:
    from api.routers import chat as chat_router

    db_path = tmp_path / "chat.sqlite3"
    store = ChatStore(db_path)
    first = store.create_research_note(title="First source note", content_markdown="First finding")
    second = store.create_research_note(title="Second source note", content_markdown="Second finding")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    client = TestClient(app)

    response = client.post(
        "/api/chat/research-notes",
        json={
            "title": "Persistent evidence outline",
            "content_markdown": "# Persistent evidence outline\n\n## 1. First source note",
            "source_state": {
                "version": 1,
                "links": [
                    {
                        "kind": "source",
                        "label": "Grounded source",
                        "source_path": "db/paper/source.md",
                        "block_id": "block-source",
                    }
                ],
                "composition": {
                    "kind": "research_note_outline",
                    "version": 1,
                    "source_notes": [
                        {"id": first["id"], "title": first["title"], "revision": first["revision"]},
                        {"id": second["id"], "title": second["title"], "revision": second["revision"]},
                    ],
                },
            },
            "tags": ["outline", "evidence"],
        },
    )
    assert response.status_code == 200
    outline_id = response.json()["id"]

    latest_first, conflicted = store.update_research_note(
        first["id"],
        expected_revision=1,
        content_markdown="First finding with calibrated evidence",
    )
    assert conflicted is False
    assert latest_first is not None
    refresh_response = client.patch(
        f"/api/chat/research-notes/{outline_id}",
        json={
            "expected_revision": 1,
            "content_markdown": "# Persistent evidence outline\n\n## 1. First source note\n\nFirst finding with calibrated evidence",
            "source_state": {
                "version": 1,
                "links": [
                    {
                        "kind": "source",
                        "label": "Grounded source",
                        "source_path": "db/paper/source.md",
                        "block_id": "block-source",
                    }
                ],
                "composition": {
                    "kind": "research_note_outline",
                    "version": 1,
                    "source_notes": [
                        {
                            "id": first["id"],
                            "title": first["title"],
                            "revision": 2,
                            "content_fingerprint": "fnv1a32:1234abcd:39",
                            "last_refreshed_at": 1_800_000_000,
                        },
                        {"id": second["id"], "title": second["title"], "revision": 1},
                    ],
                    "refresh_history": [
                        {
                            "source_id": first["id"],
                            "from_revision": 1,
                            "to_revision": 2,
                            "refreshed_at": 1_800_000_000,
                            "replaced_manual_edits": True,
                        }
                    ],
                },
            },
        },
    )
    assert refresh_response.status_code == 200
    assert refresh_response.json()["revision"] == 2

    reopened = ChatStore(db_path)
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: reopened)
    restored = client.get(f"/api/chat/research-notes/{outline_id}")

    assert restored.status_code == 200
    record = restored.json()
    assert record["tags"] == ["outline", "evidence"]
    assert record["source_state"]["links"][0]["block_id"] == "block-source"
    assert record["source_state"]["composition"] == {
        "kind": "research_note_outline",
        "version": 1,
        "source_notes": [
            {
                "id": first["id"],
                "title": "First source note",
                "revision": 2,
                "content_fingerprint": "fnv1a32:1234abcd:39",
                "last_refreshed_at": 1_800_000_000,
            },
            {"id": second["id"], "title": "Second source note", "revision": 1},
        ],
        "refresh_history": [
            {
                "source_id": first["id"],
                "from_revision": 1,
                "to_revision": 2,
                "refreshed_at": 1_800_000_000,
                "replaced_manual_edits": True,
            }
        ],
    }
