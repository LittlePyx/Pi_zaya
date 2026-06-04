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
