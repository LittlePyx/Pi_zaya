from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from kb.chat_store import ChatStore


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
