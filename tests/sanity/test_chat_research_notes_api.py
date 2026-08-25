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
