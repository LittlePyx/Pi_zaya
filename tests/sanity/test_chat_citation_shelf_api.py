from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from kb.chat_store import ChatStore


def test_chat_sidebar_api_returns_grouped_snapshot(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("papers")
    root_id = store.create_conversation("root")
    project_conv_id = store.create_conversation("project", project_id=project_id)
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    client = TestClient(app)

    response = client.get("/api/sidebar?limit=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["projects"][0]["id"] == project_id
    assert [conv["id"] for conv in payload["root_conversations"]] == [root_id]
    assert [conv["id"] for conv in payload["project_conversations"][project_id]] == [project_conv_id]


def test_chat_citation_shelf_api_roundtrip(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("papers")
    conv_a = store.create_conversation("a", project_id=project_id)
    conv_b = store.create_conversation("b", project_id=project_id)
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    client = TestClient(app)

    saved = client.patch(
        f"/api/chat/citation-shelf?conv_id={conv_a}",
        json={
            "items": [
                {
                    "key": "ref-api",
                    "main": "Physics-informed deep learning",
                    "doi": "10.1038/api-demo",
                }
            ],
            "open": True,
        },
    )
    assert saved.status_code == 200
    saved_payload = saved.json()
    assert saved_payload["scope"] == "project"
    assert saved_payload["scope_id"] == project_id
    assert saved_payload["items"][0]["key"] == "ref-api"

    loaded = client.get(f"/api/chat/citation-shelf?conv_id={conv_b}")
    assert loaded.status_code == 200
    loaded_payload = loaded.json()
    assert loaded_payload["open"] is True
    assert loaded_payload["items"][0]["doi"] == "10.1038/api-demo"

    cleared = client.delete(f"/api/chat/citation-shelf?conv_id={conv_b}")
    assert cleared.status_code == 200
    assert cleared.json()["items"] == []


def test_chat_citation_shelf_rejects_accidental_empty_overwrite(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("papers")
    conv_a = store.create_conversation("a", project_id=project_id)
    conv_b = store.create_conversation("b", project_id=project_id)
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    client = TestClient(app)

    saved = client.patch(
        f"/api/chat/citation-shelf?conv_id={conv_a}",
        json={
            "items": [
                {
                    "key": "ref-1",
                    "anchor": "ref-1",
                    "main": "Physics-informed deep learning",
                    "title": "Physics-informed deep learning",
                    "doi": "10.1038/demo",
                    "raw": "Physics-informed deep learning reference entry.",
                    "shelf_item_kind": "reference",
                    "tags": [" idea ", "", "related-work"],
                },
                {
                    "key": "ref-duplicate",
                    "anchor": "ref-duplicate",
                    "main": "Duplicate DOI",
                    "title": "Duplicate DOI",
                    "doi": "https://doi.org/10.1038/demo",
                },
            ],
            "open": True,
        },
    )
    assert saved.status_code == 200
    saved_payload = saved.json()
    assert len(saved_payload["items"]) == 1
    assert saved_payload["items"][0]["shelfItemKind"] == "reference"
    assert saved_payload["items"][0]["shelfExcerpt"] == "Physics-informed deep learning reference entry."
    assert saved_payload["items"][0]["tags"] == ["idea", "related-work"]

    accidental_clear = client.patch(
        f"/api/chat/citation-shelf?conv_id={conv_b}",
        json={"items": [], "open": False},
    )
    assert accidental_clear.status_code == 200
    accidental_payload = accidental_clear.json()
    assert accidental_payload["items"][0]["doi"] == "10.1038/demo"
    assert accidental_payload["revision"] == saved_payload["revision"]

    explicit_clear = client.patch(
        f"/api/chat/citation-shelf?conv_id={conv_b}",
        json={"items": [], "open": False, "allow_empty_overwrite": True},
    )
    assert explicit_clear.status_code == 200
    assert explicit_clear.json()["items"] == []


def test_chat_citation_shelf_append_item_merges_project_scope(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("papers")
    conv_a = store.create_conversation("a", project_id=project_id)
    conv_b = store.create_conversation("b", project_id=project_id)
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    client = TestClient(app)

    first = client.post(
        f"/api/chat/citation-shelf/items?conv_id={conv_a}",
        json={
            "item": {
                "key": "reader-ref-1",
                "main": "Reader reference",
                "title": "Reader reference",
                "doi": "10.1000/reader-ref",
                "raw": "Reader reference entry.",
                "shelf_item_kind": "reference",
            },
            "open": True,
        },
    )
    assert first.status_code == 200
    assert first.json()["open"] is True
    assert first.json()["items"][0]["doi"] == "10.1000/reader-ref"

    duplicate = client.post(
        f"/api/chat/citation-shelf/items?conv_id={conv_b}",
        json={
            "item": {
                "key": "reader-ref-duplicate",
                "main": "Duplicate reader reference",
                "title": "Duplicate reader reference",
                "doi": "https://doi.org/10.1000/reader-ref",
            },
            "open": True,
        },
    )
    assert duplicate.status_code == 200
    assert len(duplicate.json()["items"]) == 1

    loaded = client.get(f"/api/chat/citation-shelf?conv_id={conv_b}")
    assert loaded.status_code == 200
    assert loaded.json()["scope_id"] == project_id
    assert loaded.json()["items"][0]["shelfItemKind"] == "reference"


def test_chat_citation_shelf_append_keeps_distinct_reader_selections(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("reader notes")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    client = TestClient(app)

    for idx, text in enumerate(["first selected passage", "second selected passage"]):
        response = client.post(
            f"/api/chat/citation-shelf/items?conv_id={conv_id}",
            json={
                "item": {
                    "key": f"reader-selection-{idx}",
                    "main": "Demo paper",
                    "title": "Demo paper",
                    "sourcePath": "demo.md",
                    "blockId": "blk-1",
                    "startOffset": idx * 10,
                    "endOffset": idx * 10 + len(text),
                    "raw": text,
                    "shelfItemKind": "reader_selection",
                    "shelfExcerpt": text,
                },
                "open": True,
            },
        )
        assert response.status_code == 200

    loaded = client.get(f"/api/chat/citation-shelf?conv_id={conv_id}")
    assert loaded.status_code == 200
    items = loaded.json()["items"]
    assert len(items) == 2
    assert {item["shelfExcerpt"] for item in items} == {"first selected passage", "second selected passage"}


def test_chat_citation_shelf_rejects_malformed_empty_items(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("a")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    client = TestClient(app)

    saved = client.patch(
        f"/api/chat/citation-shelf?conv_id={conv_id}",
        json={
            "items": [
                {},
                {"key": "   "},
                {"key": "valid-ref", "main": "Stable reference item"},
            ],
            "open": True,
        },
    )

    assert saved.status_code == 200
    items = saved.json()["items"]
    assert len(items) == 1
    assert items[0]["key"] == "valid-ref"
    assert items[0]["main"] == "Stable reference item"


def test_chat_citation_shelf_api_rejects_oversized_payload(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("a")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    client = TestClient(app)

    too_many = client.patch(
        f"/api/chat/citation-shelf?conv_id={conv_id}",
        json={
            "items": [{"key": f"ref-{idx}", "main": f"Reference {idx}"} for idx in range(121)],
            "open": True,
        },
    )
    huge_item = client.post(
        f"/api/chat/citation-shelf/items?conv_id={conv_id}",
        json={"item": {"key": "huge", "main": "x" * 45_000}, "open": True},
    )

    assert too_many.status_code == 422
    assert huge_item.status_code == 422
    assert store.get_citation_shelf(conv_id=conv_id)["items"] == []


def test_chat_message_api_rejects_oversized_content_before_writing(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("Draft")
    msg_id = store.append_message(conv_id, "user", "short")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    client = TestClient(app)

    created = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"role": "user", "content": "x" * 80_001},
    )
    updated = client.patch(
        f"/api/messages/{msg_id}",
        json={"content": "y" * 80_001},
    )

    assert created.status_code == 422
    assert updated.status_code == 422
    messages = store.get_messages(conv_id)
    assert len(messages) == 1
    assert messages[0]["content"] == "short"


def test_conversation_title_update_rejects_blank_title(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("Draft")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    client = TestClient(app)

    blank = client.patch(f"/api/conversations/{conv_id}/title", json={"title": " \n "})
    assert blank.status_code == 400

    missing = client.patch("/api/conversations/missing/title", json={"title": "Valid title"})
    assert missing.status_code == 404

    renamed = client.patch(f"/api/conversations/{conv_id}/title", json={"title": "Valid title"})
    assert renamed.status_code == 200
    assert store.get_conversation(conv_id)["title"] == "Valid title"


def test_conversation_title_and_project_api_reject_oversized_names(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Project")
    conv_id = store.create_conversation("Draft", project_id=project_id)
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    client = TestClient(app)

    created_project = client.post("/api/projects", json={"name": "p" * 121})
    renamed_project = client.patch(f"/api/projects/{project_id}", json={"name": "p" * 121})
    created_conv = client.post("/api/conversations", json={"title": "t" * 241})
    renamed_conv = client.patch(f"/api/conversations/{conv_id}/title", json={"title": "t" * 241})

    assert created_project.status_code == 422
    assert renamed_project.status_code == 422
    assert created_conv.status_code == 422
    assert renamed_conv.status_code == 422
    assert store.get_project(project_id)["name"] == "Project"
    assert store.get_conversation(conv_id)["title"] == "Draft"


def test_conversation_project_update_rejects_missing_project(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Project")
    conv_id = store.create_conversation("Draft", project_id=project_id)
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    client = TestClient(app)

    created_missing = client.post(
        "/api/conversations",
        json={"title": "Stale project", "project_id": "missing-project"},
    )
    moved_missing = client.patch(
        f"/api/conversations/{conv_id}/project",
        json={"project_id": "missing-project"},
    )

    assert created_missing.status_code == 404
    assert created_missing.json()["detail"] == "project not found"
    assert moved_missing.status_code == 404
    assert moved_missing.json()["detail"] == "project not found"
    assert store.get_conversation(conv_id)["project_id"] == project_id

    moved_valid = client.patch(
        f"/api/conversations/{conv_id}/project",
        json={"project_id": None},
    )
    assert moved_valid.status_code == 200
    assert store.get_conversation(conv_id)["project_id"] is None


def test_delete_missing_conversation_and_project_return_404_without_snapshot(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    def fail_snapshot(*args, **kwargs):
        raise AssertionError("snapshot should not run for a missing resource")

    monkeypatch.setattr(chat_router, "_dangerous_auto_snapshot", fail_snapshot)
    client = TestClient(app)

    missing_conversation = client.delete("/api/conversations/missing-conversation")
    missing_project = client.delete("/api/projects/missing-project")
    missing_message = client.delete("/api/messages/123456")
    missing_citation_shelf = client.delete("/api/chat/citation-shelf?conv_id=missing-conversation")

    assert missing_conversation.status_code == 404
    assert missing_project.status_code == 404
    assert missing_message.status_code == 404
    assert missing_citation_shelf.status_code == 404


def test_delete_conversation_and_project_report_success_for_existing_records(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("papers")
    conv_id = store.create_conversation("delete me", project_id=project_id)
    msg_id = store.append_message(conv_id, "user", "temporary message")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(chat_router, "_dangerous_auto_snapshot", lambda *args, **kwargs: {"ok": True})

    client = TestClient(app)

    message_deleted = client.delete(f"/api/messages/{msg_id}")
    conversation_deleted = client.delete(f"/api/conversations/{conv_id}")
    project_deleted = client.delete(f"/api/projects/{project_id}")

    assert message_deleted.status_code == 200
    assert message_deleted.json()["ok"] is True
    assert store.message_exists(msg_id) is False
    assert conversation_deleted.status_code == 200
    assert conversation_deleted.json()["ok"] is True
    assert store.get_conversation(conv_id) is None
    assert project_deleted.status_code == 200
    assert project_deleted.json()["ok"] is True
    assert store.get_project(project_id) is None


def test_conversation_research_state_api_roundtrip(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("research state")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    client = TestClient(app)

    empty = client.get(f"/api/conversations/{conv_id}/research-state")
    assert empty.status_code == 200
    assert empty.json()["state"] == {}

    saved = client.patch(
        f"/api/conversations/{conv_id}/research-state",
        json={
            "state": {
                "selected_research_context": {
                    "version": 1,
                    "id": "ctx-api",
                    "source": "citation_shelf",
                    "items": [{"key": "ref-api", "title": "API reference"}],
                }
            }
        },
    )
    assert saved.status_code == 200
    assert saved.json()["state"]["selected_research_context"]["id"] == "ctx-api"

    loaded = client.get(f"/api/conversations/{conv_id}/research-state")
    assert loaded.status_code == 200
    assert loaded.json()["state"]["selected_research_context"]["items"][0]["key"] == "ref-api"

    cleared = client.patch(
        f"/api/conversations/{conv_id}/research-state",
        json={"state": {"selected_research_context": None}},
    )
    assert cleared.status_code == 200
    assert "selected_research_context" not in cleared.json()["state"]

    missing = client.get("/api/conversations/missing/research-state")
    assert missing.status_code == 404


def test_conversation_research_state_api_rejects_oversized_state(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("research state")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

    client = TestClient(app)

    oversized = client.patch(
        f"/api/conversations/{conv_id}/research-state",
        json={"state": {"debug": "x" * 170_000}},
    )

    assert oversized.status_code == 422
    assert store.get_conversation_research_state(conv_id)["state"] == {}
