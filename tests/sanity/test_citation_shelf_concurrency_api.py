from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routers import chat as router
from kb.chat_store import ChatStore


def test_api_conflict_duplicate_and_guarded_clear(monkeypatch, tmp_path):
    store = ChatStore(tmp_path / "chat.sqlite3")
    monkeypatch.setattr(router, "get_chat_store", lambda: store)
    client = TestClient(app)
    url = "/api/chat/citation-shelf"
    body = {"items": [{"key": "A", "main": "A"}], "expected_revision": 0, "mutation_id": "add"}
    saved = client.patch(url, json=body)
    assert saved.status_code == 200
    assert client.patch(url, json={**body, "mutation_id": "stale"}).status_code == 409
    assert client.patch(url, json={"items": [], "allow_empty_overwrite": True}).status_code == 409
    assert client.delete(url).status_code == 409
    cleared = client.patch(url, json={"items": [], "allow_empty_overwrite": True,
        "expected_revision": saved.json()["revision"], "mutation_id": "clear"})
    assert cleared.status_code == 200
    duplicate = client.patch(url, json=body)
    assert duplicate.status_code == 200
    assert duplicate.json()["items"] == []
    assert duplicate.json()["revision"] == cleared.json()["revision"]
    assert client.get(url).json()["concurrency_guarded"]


@pytest.mark.parametrize("operation", [
    {"id": "", "upserts": []}, {"id": "A", "upserts": "not a list"},
    {"id": "A", "removedKeys": [None]}, {"id": "A", "clear": "yes"},
    {"id": "A", "open": None},
])
def test_api_rejects_invalid_operations(monkeypatch, tmp_path, operation):
    store = ChatStore(tmp_path / "chat.sqlite3")
    monkeypatch.setattr(router, "get_chat_store", lambda: store)
    response = TestClient(app).patch("/api/chat/citation-shelf", json={
        "items": [], "expected_revision": 0, "mutation_id": "test", "mutations": [operation],
    })
    assert response.status_code == 422
    assert store.get_citation_shelf()["revision"] == 0
