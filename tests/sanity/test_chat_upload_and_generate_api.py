from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app

TINY_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
)


def test_reader_sessions_persist_payload(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    db_dir = tmp_path / "db"
    md_root = tmp_path / "md_output"
    pdf_root = tmp_path / "pdfs"
    source_md = md_root / "source" / "source.en.md"
    source_md.parent.mkdir(parents=True, exist_ok=True)
    source_md.write_text("# Source\n\nimportant sentence\n", encoding="utf-8")
    pdf_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        chat_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=db_dir),
    )
    monkeypatch.setattr(chat_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: pdf_root)

    client = TestClient(app)
    missing = client.post("/api/reader/sessions", json={"payload": {}})
    assert missing.status_code == 400

    response = client.post(
        "/api/reader/sessions",
        json={
            "title": "Reader source",
            "conversation_id": "conv-1",
            "payload": {
                "sourcePath": str(source_md),
                "sourceName": "source.pdf",
                "headingPath": "Methods / Reader",
                "locateTarget": {"snippet": "important sentence"},
            },
            "state": {
                "highlights": [{"id": "h1", "text": "important sentence"}],
            },
        },
    )
    assert response.status_code == 200
    created = response.json()
    assert created["id"]
    assert created["payload"]["sourcePath"].endswith("source.en.md")
    assert created["conversation_id"] == "conv-1"
    assert (db_dir / "_reader_sessions.json").exists()

    loaded = client.get(f"/api/reader/sessions/{created['id']}")
    assert loaded.status_code == 200
    payload = loaded.json()["payload"]
    assert payload["sourceName"] == "source.pdf"
    assert payload["locateTarget"]["snippet"] == "important sentence"
    assert loaded.json()["state"]["highlights"][0]["id"] == "h1"

    patched = client.patch(
        f"/api/reader/sessions/{created['id']}/state",
        json={"state": {"selection": {"text": "selected text"}}},
    )
    assert patched.status_code == 200
    assert patched.json()["state"]["selection"]["text"] == "selected text"
    assert patched.json()["state"]["highlights"][0]["text"] == "important sentence"

    cleared = client.patch(
        f"/api/reader/sessions/{created['id']}/state",
        json={"state": {"highlights": None}},
    )
    assert cleared.status_code == 200
    assert "highlights" not in cleared.json()["state"]
    assert cleared.json()["state"]["selection"]["text"] == "selected text"

    loaded_after_clear = client.get(f"/api/reader/sessions/{created['id']}")
    assert loaded_after_clear.status_code == 200
    assert "highlights" not in loaded_after_clear.json()["state"]


def test_reader_sessions_reject_source_outside_allowed_roots(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    db_dir = tmp_path / "db"
    md_root = tmp_path / "md_output"
    pdf_root = tmp_path / "pdfs"
    tmp_reader_root = tmp_path / "tmp"
    tmp_reader_md = tmp_reader_root / "reconvert" / "source.en.md"
    outside = tmp_path / "outside" / "source.en.md"
    tmp_reader_md.parent.mkdir(parents=True, exist_ok=True)
    outside.parent.mkdir(parents=True, exist_ok=True)
    tmp_reader_md.write_text("# Temp Reader\n", encoding="utf-8")
    outside.write_text("# Outside\n", encoding="utf-8")
    md_root.mkdir(parents=True, exist_ok=True)
    pdf_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(chat_router, "get_settings", lambda: SimpleNamespace(db_dir=db_dir))
    monkeypatch.setattr(chat_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: pdf_root)
    monkeypatch.setattr(chat_router, "_reader_markdown_roots", lambda: [md_root.resolve(), tmp_reader_root.resolve()])

    client = TestClient(app)
    ok_response = client.post(
        "/api/reader/sessions",
        json={"payload": {"sourcePath": str(tmp_reader_md), "sourceName": "source.pdf"}},
    )
    assert ok_response.status_code == 200
    assert ok_response.json()["payload"]["sourcePath"] == str(tmp_reader_md.resolve(strict=False))

    response = client.post(
        "/api/reader/sessions",
        json={"payload": {"sourcePath": str(outside), "sourceName": "outside.pdf"}},
    )

    assert response.status_code == 400


def test_conversation_reader_state_persists_by_source(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router
    from kb.chat_store import ChatStore

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("Guide")
    db_dir = tmp_path / "db"
    md_root = tmp_path / "md_output"
    pdf_root = tmp_path / "pdfs"
    source_md = md_root / "paper" / "paper.en.md"
    other_md = md_root / "other" / "other.en.md"
    source_md.parent.mkdir(parents=True, exist_ok=True)
    other_md.parent.mkdir(parents=True, exist_ok=True)
    source_md.write_text("# Paper\n\nimportant sentence\n", encoding="utf-8")
    other_md.write_text("# Other\n", encoding="utf-8")
    pdf_root.mkdir(parents=True, exist_ok=True)
    source_path = str(source_md)
    other_source_path = str(other_md)
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(chat_router, "get_settings", lambda: SimpleNamespace(db_dir=db_dir))
    monkeypatch.setattr(chat_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: pdf_root)

    client = TestClient(app)
    missing_source = client.get(f"/api/conversations/{conv_id}/reader-state")
    assert missing_source.status_code == 400

    initial = client.get(
        f"/api/conversations/{conv_id}/reader-state",
        params={"source_path": source_path},
    )
    assert initial.status_code == 200
    assert initial.json()["state"] == {}

    patched = client.patch(
        f"/api/conversations/{conv_id}/reader-state",
        params={"source_path": source_path},
        json={"state": {"highlights": [{"id": "h1", "text": "important sentence"}]}},
    )
    assert patched.status_code == 200
    assert patched.json()["source_path"] == source_path
    assert patched.json()["state"]["highlights"][0]["id"] == "h1"

    loaded = client.get(
        f"/api/conversations/{conv_id}/reader-state",
        params={"source_path": source_path},
    )
    assert loaded.status_code == 200
    assert loaded.json()["state"]["highlights"][0]["text"] == "important sentence"

    file_url_source_path = "file:///" + str(source_md).replace("\\", "/").replace(" ", "%20") + "?download=1#reader"
    variant_patch = client.patch(
        f"/api/conversations/{conv_id}/reader-state",
        params={"source_path": file_url_source_path},
        json={"state": {"scroll": {"block": "intro"}}},
    )
    assert variant_patch.status_code == 200
    assert variant_patch.json()["source_path"] == str(source_md.resolve(strict=False))
    assert variant_patch.json()["state"]["highlights"][0]["id"] == "h1"
    assert variant_patch.json()["state"]["scroll"]["block"] == "intro"

    variant_loaded = client.get(
        f"/api/conversations/{conv_id}/reader-state",
        params={"source_path": str(source_md).replace("\\", "/") + "#reader"},
    )
    assert variant_loaded.status_code == 200
    assert variant_loaded.json()["state"]["highlights"][0]["text"] == "important sentence"
    assert variant_loaded.json()["state"]["scroll"]["block"] == "intro"

    hash_md = md_root / "paper#variant" / "paper#variant.en.md"
    hash_md.parent.mkdir(parents=True, exist_ok=True)
    hash_md.write_text("# Hash Paper\n\nhash in filename\n", encoding="utf-8")
    hash_file_url_source_path = "file:///" + str(hash_md).replace("\\", "/").replace("#", "%23") + "?download=1#reader"
    hash_patch = client.patch(
        f"/api/conversations/{conv_id}/reader-state",
        params={"source_path": hash_file_url_source_path},
        json={"state": {"scroll": {"block": "hash-intro"}}},
    )
    assert hash_patch.status_code == 200
    assert hash_patch.json()["source_path"] == str(hash_md.resolve(strict=False))
    assert hash_patch.json()["state"]["scroll"]["block"] == "hash-intro"

    cleared = client.patch(
        f"/api/conversations/{conv_id}/reader-state",
        params={"source_path": source_path},
        json={"state": {"highlights": None}},
    )
    assert cleared.status_code == 200
    assert "highlights" not in cleared.json()["state"]

    oversized = client.patch(
        f"/api/conversations/{conv_id}/reader-state",
        params={"source_path": source_path},
        json={"state": {"debug": "z" * 6000, "ids": list(range(600)), "empty": []}},
    )
    assert oversized.status_code == 200
    oversized_state = oversized.json()["state"]
    assert len(oversized_state["debug"]) == 4000
    assert len(oversized_state["ids"]) == 500
    assert oversized_state["empty"] == []

    legacy_md = md_root / "legacy" / "legacy.en.md"
    legacy_md.parent.mkdir(parents=True, exist_ok=True)
    legacy_md.write_text("# Legacy\n\nold highlight\n", encoding="utf-8")
    legacy_source_path = str(legacy_md).replace("\\", "/")
    store.patch_conversation_reader_state(
        conv_id,
        legacy_source_path,
        {"highlights": [{"id": "legacy", "text": "old highlight"}]},
    )
    legacy_loaded = client.get(
        f"/api/conversations/{conv_id}/reader-state",
        params={"source_path": legacy_source_path},
    )
    assert legacy_loaded.status_code == 200
    assert legacy_loaded.json()["source_path"] == str(legacy_md.resolve(strict=False))
    assert legacy_loaded.json()["state"]["highlights"][0]["id"] == "legacy"

    other = client.get(
        f"/api/conversations/{conv_id}/reader-state",
        params={"source_path": other_source_path},
    )
    assert other.status_code == 200
    assert other.json()["state"] == {}

    outside = tmp_path / "outside" / "paper.en.md"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("# Outside\n", encoding="utf-8")
    invalid = client.get(
        f"/api/conversations/{conv_id}/reader-state",
        params={"source_path": str(outside)},
    )
    assert invalid.status_code == 400


def test_conversation_reader_state_api_rejects_oversized_state(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router
    from kb.chat_store import ChatStore

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("Guide")
    db_dir = tmp_path / "db"
    md_root = tmp_path / "md_output"
    pdf_root = tmp_path / "pdfs"
    source_md = md_root / "paper" / "paper.en.md"
    source_md.parent.mkdir(parents=True, exist_ok=True)
    source_md.write_text("# Paper\n", encoding="utf-8")
    pdf_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(chat_router, "get_settings", lambda: SimpleNamespace(db_dir=db_dir))
    monkeypatch.setattr(chat_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: pdf_root)

    client = TestClient(app)

    oversized = client.patch(
        f"/api/conversations/{conv_id}/reader-state",
        params={"source_path": str(source_md)},
        json={"state": {"debug": "x" * 170_000}},
    )

    assert oversized.status_code == 422
    assert store.get_conversation_reader_state(conv_id, str(source_md.resolve(strict=False)))["state"] == {}


def test_append_message_rejects_missing_conversation(monkeypatch):
    from api.routers import chat as chat_router

    class FakeStore:
        def __init__(self) -> None:
            self.append_calls = 0

        def get_conversation(self, conv_id: str):
            return None

        def append_message(self, *args, **kwargs):
            self.append_calls += 1
            return 1

    fake_store = FakeStore()
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: fake_store)

    client = TestClient(app)
    response = client.post("/api/conversations/missing/messages", json={"role": "user", "content": "hello"})

    assert response.status_code == 404
    assert fake_store.append_calls == 0


def test_chat_uploads_route_handles_pdf_and_image(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    image_dir = tmp_path / "chat_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    saved_pdf = tmp_path / "paper.pdf"
    saved_pdf.write_bytes(b"%PDF-1.4 test")

    def fake_save_pdf_to_library(*, file_name: str, data: bytes, base_name: str = "", fast_mode: bool = False) -> dict:
        assert fast_mode is True
        return {
            "duplicate": False,
            "path": str(saved_pdf),
            "name": "paper.pdf",
            "sha1": "pdfsha1",
        }

    seen_sha1: list[str] = []

    def fake_start_chat_pdf_ingest_job(
        *,
        pdf_path: Path,
        speed_mode: str,
        display_name: str,
        sha1: str = "",
        conv_id: str = "",
    ) -> str:
        assert pdf_path == saved_pdf
        assert speed_mode == "balanced"
        assert display_name == "paper.pdf"
        assert sha1
        assert conv_id == ""
        seen_sha1.append(sha1)
        chat_router._CHAT_UPLOAD_JOBS["job-1"] = {
            "name": display_name,
            "sha1": sha1,
            "path": str(pdf_path),
            "ready": True,
            "ingest_status": "ready",
            "md_path": str(tmp_path / "paper" / "paper.en.md"),
            "error": "",
        }
        return "job-1"

    monkeypatch.setattr(chat_router, "save_pdf_to_library", fake_save_pdf_to_library)
    monkeypatch.setattr(chat_router, "_start_chat_pdf_ingest_job", fake_start_chat_pdf_ingest_job)
    monkeypatch.setattr(chat_router, "_chat_image_dir", lambda: image_dir)

    client = TestClient(app)
    response = client.post(
      "/api/chat/uploads",
      files=[
        ("files", ("paper.pdf", b"%PDF-1.4 test", "application/pdf")),
        ("files", ("figure.png", TINY_PNG_BYTES, "image/png")),
      ],
      data={"quick_ingest": "true"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 2

    pdf_item = next(item for item in payload["items"] if item["kind"] == "pdf")
    image_item = next(item for item in payload["items"] if item["kind"] == "image")

    assert pdf_item["status"] == "saved"
    assert pdf_item["ready"] is False
    assert pdf_item["ingest_status"] == "renaming"
    assert pdf_item["ingest_job_id"] == "job-1"
    assert pdf_item["name"] == "paper.pdf"
    assert pdf_item["sha1"] == seen_sha1[0]

    status_response = client.get("/api/chat/uploads/status?job_ids=job-1")
    assert status_response.status_code == 200
    status_item = status_response.json()["items"][0]
    assert status_item["sha1"] == seen_sha1[0]
    assert status_item["ready"] is True
    assert status_item["ingest_status"] == "ready"

    assert image_item["status"] == "saved"
    assert image_item["attachment"]["name"] == "figure.png"
    assert Path(image_item["attachment"]["path"]).name == image_item["attachment"]["path"]
    assert (image_dir / image_item["attachment"]["path"]).exists()
    assert image_item["attachment"]["url"].startswith("/api/chat/uploads/image?path=")
    assert str(tmp_path) not in image_item["attachment"]["url"]

    image_response = client.get(image_item["attachment"]["url"])
    assert image_response.status_code == 200
    assert image_response.headers["content-type"].startswith("image/png")


def test_chat_upload_image_content_signature_controls_saved_type(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    image_dir = tmp_path / "chat_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(chat_router, "_chat_image_dir", lambda: image_dir)

    client = TestClient(app)
    response = client.post(
      "/api/chat/uploads",
      files=[("files", ("figure.jpg", TINY_PNG_BYTES, "image/jpeg"))],
      data={"quick_ingest": "false"},
    )

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["kind"] == "image"
    assert item["status"] == "saved"
    assert item["mime"] == "image/png"
    assert item["attachment"]["mime"] == "image/png"
    assert Path(item["attachment"]["path"]).suffix == ".png"
    assert Path(item["attachment"]["path"]).name == item["attachment"]["path"]

    image_response = client.get(item["attachment"]["url"])
    assert image_response.status_code == 200
    assert image_response.headers["content-type"].startswith("image/png")


def test_chat_upload_pdf_requires_header_signature(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    called = {"save": False}

    def fake_save_pdf_to_library(*args, **kwargs):
        called["save"] = True
        raise AssertionError("fake PDF content must not be saved")

    monkeypatch.setattr(chat_router, "save_pdf_to_library", fake_save_pdf_to_library)

    client = TestClient(app)
    response = client.post(
      "/api/chat/uploads",
      files=[("files", ("paper.pdf", b"not really a pdf, just mentions %PDF-1.4", "application/pdf"))],
      data={"quick_ingest": "false"},
    )

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["kind"] == "pdf"
    assert item["status"] == "error"
    assert item["error"] == "invalid PDF file"
    assert called["save"] is False


def test_chat_upload_image_filename_is_collapsed_to_basename(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    image_dir = tmp_path / "chat_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(chat_router, "_chat_image_dir", lambda: image_dir)

    client = TestClient(app)
    response = client.post(
      "/api/chat/uploads",
      files=[("files", (r"C:\Users\Alice\secret-figure.png", TINY_PNG_BYTES, "image/png"))],
    )

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["kind"] == "image"
    assert item["name"] == "secret-figure.png"
    assert item["attachment"]["name"] == "secret-figure.png"
    assert "Alice" not in Path(item["path"]).name
    assert Path(item["path"]).name.startswith("secret-figure-")
    assert Path(item["attachment"]["path"]).name == item["attachment"]["path"]


def test_chat_upload_image_display_name_removes_control_characters(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    image_dir = tmp_path / "chat_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(chat_router, "_chat_image_dir", lambda: image_dir)

    client = TestClient(app)
    response = client.post(
      "/api/chat/uploads",
      files=[("files", ("figure\nwith\tcontrol.png", TINY_PNG_BYTES, "image/png"))],
    )

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["kind"] == "image"
    assert item["status"] == "saved"
    assert item["name"] == "figure with control.png"
    assert item["attachment"]["name"] == "figure with control.png"
    assert "\n" not in item["name"]
    assert "\t" not in item["name"]


def test_chat_upload_image_replaces_corrupt_duplicate_cache(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    image_dir = tmp_path / "chat_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    sha1 = hashlib.sha1(TINY_PNG_BYTES).hexdigest()
    cached = image_dir / f"figure-{sha1[:10]}.png"
    cached.write_bytes(b"not a valid cached image")
    monkeypatch.setattr(chat_router, "_chat_image_dir", lambda: image_dir)

    client = TestClient(app)
    response = client.post(
      "/api/chat/uploads",
      files=[("files", ("figure.png", TINY_PNG_BYTES, "image/png"))],
    )

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["kind"] == "image"
    assert item["status"] == "saved"
    assert image_dir / item["path"] == cached
    assert Path(item["path"]).name == item["path"]
    assert cached.read_bytes() == TINY_PNG_BYTES

    image_response = client.get(item["attachment"]["url"])
    assert image_response.status_code == 200
    assert image_response.headers["content-type"].startswith("image/png")


def test_chat_upload_rejects_invalid_or_unsupported_images(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    image_dir = tmp_path / "chat_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(chat_router, "_chat_image_dir", lambda: image_dir)

    client = TestClient(app)
    response = client.post(
      "/api/chat/uploads",
      files=[
        ("files", ("not-image.png", b"not really an image", "image/png")),
        ("files", ("fake-header.png", b"\x89PNG\r\n\x1a\nnot-a-real-full-image", "image/png")),
        ("files", ("figure.svg", b"<svg></svg>", "image/svg+xml")),
      ],
      data={"quick_ingest": "false"},
    )

    assert response.status_code == 200
    items = response.json()["items"]
    invalid_item = next(item for item in items if item["name"] == "not-image.png")
    fake_header_item = next(item for item in items if item["name"] == "fake-header.png")
    unsupported_item = next(item for item in items if item["name"] == "figure.svg")
    assert invalid_item["kind"] == "image"
    assert invalid_item["status"] == "error"
    assert invalid_item["error"] == "invalid image file"
    assert fake_header_item["kind"] == "image"
    assert fake_header_item["status"] == "error"
    assert fake_header_item["error"] == "invalid image file"
    assert unsupported_item["kind"] == "unknown"
    assert unsupported_item["status"] == "unsupported"
    assert not any(image_dir.iterdir())


def test_chat_upload_image_route_rejects_paths_outside_upload_root(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    image_dir = tmp_path / "chat_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    in_root = image_dir / "ok.png"
    in_root.write_bytes(TINY_PNG_BYTES)
    outside = tmp_path / "outside.png"
    outside.write_bytes(TINY_PNG_BYTES)
    fake = image_dir / "fake.png"
    fake.write_bytes(b"not really an image")
    monkeypatch.setattr(chat_router, "_chat_image_dir", lambda: image_dir)

    client = TestClient(app)
    ok_response = client.get("/api/chat/uploads/image", params={"path": str(in_root)})
    assert ok_response.status_code == 200

    leaf_response = client.get("/api/chat/uploads/image", params={"path": in_root.name})
    assert leaf_response.status_code == 200

    fake_response = client.get("/api/chat/uploads/image", params={"path": str(fake)})
    assert fake_response.status_code == 404

    outside_response = client.get("/api/chat/uploads/image", params={"path": str(outside)})
    assert outside_response.status_code == 404

    traversal_response = client.get("/api/chat/uploads/image", params={"path": str(image_dir / ".." / "outside.png")})
    assert traversal_response.status_code == 404


def test_chat_uploads_reject_oversized_file(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    monkeypatch.setattr(
        chat_router,
        "get_settings",
        lambda: SimpleNamespace(
            db_dir=tmp_path / "db",
            max_pdf_upload_bytes=64,
            max_image_upload_bytes=8,
            max_chat_upload_files=4,
        ),
    )

    client = TestClient(app)
    response = client.post(
      "/api/chat/uploads",
      files=[("files", ("figure.png", b"123456789", "image/png"))],
      data={"quick_ingest": "false"},
    )

    assert response.status_code == 413


def test_generate_accepts_image_only(monkeypatch, tmp_path: Path):
    from api.routers import generate as generate_router

    class FakeStore:
        def __init__(self) -> None:
            self.messages: list[tuple[str, str, str, list[dict] | None]] = []
            self.titles: list[tuple[str, str]] = []

        def append_message(
            self,
            conv_id: str,
            role: str,
            content: str,
            attachments: list[dict] | None = None,
            meta: dict | None = None,
        ) -> int:
            self.messages.append((conv_id, role, content, attachments))
            return len(self.messages)

        def set_title_if_default(self, conv_id: str, title: str) -> None:
            self.titles.append((conv_id, title))
            return True

        def get_conversation(self, conv_id: str) -> dict:
            return {"title": "新会话 · 06/05 10:30"}

    fake_store = FakeStore()
    started_tasks: list[dict] = []

    class FakeSettings:
        chat_db_path = tmp_path / "chat.sqlite3"
        db_dir = tmp_path / "db"

    image_dir = FakeSettings.db_dir / "_chat_uploads" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "img.png"
    image_path.write_bytes(TINY_PNG_BYTES)

    monkeypatch.setattr(generate_router, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(generate_router, "get_chat_store", lambda: fake_store)
    monkeypatch.setattr(generate_router, "_gen_start_task", lambda task: started_tasks.append(task) or True)

    client = TestClient(app)
    response = client.post(
      "/api/generate",
      json={
        "conv_id": "conv-1",
        "prompt": "",
        "image_attachments": [
            {
              "sha1": "imgsha1",
              "path": image_path.name,
              "name": "img.png",
              "mime": "image/png",
            }
        ],
      },
    )

    assert response.status_code == 200
    assert fake_store.messages[0][0:3] == ("conv-1", "user", "[Image attachment x1]")
    assert fake_store.messages[0][3]
    assert fake_store.messages[0][3][0]["name"] == "img.png"
    assert fake_store.messages[1][1] == "assistant"
    assert fake_store.titles == [("conv-1", "图片提问 x1")]
    assert response.json()["conversation_title"] == "图片提问 x1"
    assert started_tasks
    assert started_tasks[0]["image_attachments"][0]["name"] == "img.png"


def test_generate_rejects_image_attachment_outside_upload_root(monkeypatch, tmp_path: Path):
    from api.routers import generate as generate_router

    class FakeStore:
        def __init__(self) -> None:
            self.append_calls = 0

        def append_message(self, *args, **kwargs):
            self.append_calls += 1
            return 1

        def get_conversation(self, conv_id: str) -> dict:
            return {"title": "New chat"}

    fake_store = FakeStore()

    class FakeSettings:
        chat_db_path = tmp_path / "chat.sqlite3"
        db_dir = tmp_path / "db"

    outside_image = tmp_path / "outside.png"
    outside_image.write_bytes(TINY_PNG_BYTES)
    started_tasks: list[dict] = []

    monkeypatch.setattr(generate_router, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(generate_router, "get_chat_store", lambda: fake_store)
    monkeypatch.setattr(generate_router, "_gen_start_task", lambda task: started_tasks.append(task) or True)
    monkeypatch.setattr(generate_router, "load_prefs", lambda: {})

    client = TestClient(app)
    response = client.post(
        "/api/generate",
        json={
            "conv_id": "conv-1",
            "prompt": "",
            "image_attachments": [
                {
                    "sha1": "bad",
                    "path": str(outside_image),
                    "name": "outside.png",
                    "mime": "image/png",
                }
            ],
        },
    )

    assert response.status_code == 400
    assert fake_store.append_calls == 0
    assert started_tasks == []


def test_generate_rejects_invalid_image_content_inside_upload_root(monkeypatch, tmp_path: Path):
    from api.routers import generate as generate_router

    class FakeStore:
        def __init__(self) -> None:
            self.append_calls = 0

        def append_message(self, *args, **kwargs):
            self.append_calls += 1
            return 1

        def get_conversation(self, conv_id: str) -> dict:
            return {"title": "New chat"}

    fake_store = FakeStore()

    class FakeSettings:
        chat_db_path = tmp_path / "chat.sqlite3"
        db_dir = tmp_path / "db"

    image_dir = FakeSettings.db_dir / "_chat_uploads" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    fake_image = image_dir / "fake.png"
    fake_image.write_bytes(b"not really an image")
    started_tasks: list[dict] = []

    monkeypatch.setattr(generate_router, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(generate_router, "get_chat_store", lambda: fake_store)
    monkeypatch.setattr(generate_router, "_gen_start_task", lambda task: started_tasks.append(task) or True)
    monkeypatch.setattr(generate_router, "load_prefs", lambda: {})

    client = TestClient(app)
    response = client.post(
        "/api/generate",
        json={
            "conv_id": "conv-1",
            "prompt": "",
            "image_attachments": [
                {
                    "sha1": "fake",
                    "path": str(fake_image),
                    "name": "fake.png",
                    "mime": "image/png",
                }
            ],
        },
    )

    assert response.status_code == 400
    assert fake_store.append_calls == 0
    assert started_tasks == []


def test_generate_rejects_missing_conversation_before_appending(monkeypatch, tmp_path: Path):
    from api.routers import generate as generate_router

    class FakeStore:
        def __init__(self) -> None:
            self.append_calls = 0

        def get_conversation(self, conv_id: str):
            return None

        def append_message(self, *args, **kwargs):
            self.append_calls += 1
            return 1

    fake_store = FakeStore()

    class FakeSettings:
        chat_db_path = tmp_path / "chat.sqlite3"
        db_dir = tmp_path / "db"

    monkeypatch.setattr(generate_router, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(generate_router, "get_chat_store", lambda: fake_store)
    monkeypatch.setattr(generate_router, "load_prefs", lambda: {})

    client = TestClient(app)
    response = client.post("/api/generate", json={"conv_id": "missing", "prompt": "hello"})

    assert response.status_code == 404
    assert fake_store.append_calls == 0


def test_generate_api_rejects_oversized_prompt_before_store_access(monkeypatch):
    from api.routers import generate as generate_router

    def fail_store():
        raise AssertionError("store should not be touched for invalid generate body")

    monkeypatch.setattr(generate_router, "get_chat_store", fail_store)

    client = TestClient(app)
    response = client.post(
        "/api/generate",
        json={"conv_id": "conv-1", "prompt": "x" * 80_001},
    )

    assert response.status_code == 422


def test_generate_api_rejects_unbounded_generation_parameters(monkeypatch):
    from api.routers import generate as generate_router

    def fail_store():
        raise AssertionError("store should not be touched for invalid generate body")

    monkeypatch.setattr(generate_router, "get_chat_store", fail_store)
    client = TestClient(app)

    top_k = client.post("/api/generate", json={"conv_id": "conv-1", "prompt": "hello", "top_k": 10_000})
    temperature = client.post("/api/generate", json={"conv_id": "conv-1", "prompt": "hello", "temperature": 9})
    max_tokens = client.post("/api/generate", json={"conv_id": "conv-1", "prompt": "hello", "max_tokens": 1_000_000})

    assert top_k.status_code == 422
    assert temperature.status_code == 422
    assert max_tokens.status_code == 422


def test_generate_api_rejects_oversized_context_and_source_hints(monkeypatch):
    from api.routers import generate as generate_router

    def fail_store():
        raise AssertionError("store should not be touched for invalid generate body")

    monkeypatch.setattr(generate_router, "get_chat_store", fail_store)
    client = TestClient(app)

    huge_context = client.post(
        "/api/generate",
        json={
            "conv_id": "conv-1",
            "prompt": "hello",
            "prompt_context": {"items": [{"summary": "x" * 270_000}]},
        },
    )
    too_many_sources = client.post(
        "/api/generate",
        json={
            "conv_id": "conv-1",
            "prompt": "hello",
            "preferred_sources": [f"source-{idx}" for idx in range(13)],
        },
    )
    too_long_source = client.post(
        "/api/generate",
        json={
            "conv_id": "conv-1",
            "prompt": "hello",
            "preferred_sources": ["s" * 1_201],
        },
    )
    too_many_images = client.post(
        "/api/generate",
        json={
            "conv_id": "conv-1",
            "prompt": "",
            "image_attachments": [{"path": f"img-{idx}.png"} for idx in range(5)],
        },
    )

    assert huge_context.status_code == 422
    assert too_many_sources.status_code == 422
    assert too_long_source.status_code == 422
    assert too_many_images.status_code == 422


def test_generate_rejects_source_lock_markdown_outside_allowed_roots(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router
    from api.routers import generate as generate_router

    class FakeStore:
        def __init__(self) -> None:
            self.append_calls = 0

        def append_message(self, *args, **kwargs):
            self.append_calls += 1
            return 1

        def get_conversation(self, conv_id: str) -> dict:
            return {"title": "New chat", "mode": "normal"}

    fake_store = FakeStore()

    class FakeSettings:
        chat_db_path = tmp_path / "chat.sqlite3"
        db_dir = tmp_path / "db"

    md_root = tmp_path / "md_output"
    pdf_root = tmp_path / "pdfs"
    md_root.mkdir(parents=True, exist_ok=True)
    pdf_root.mkdir(parents=True, exist_ok=True)
    outside_md = tmp_path / "outside" / "secret.md"
    outside_md.parent.mkdir(parents=True, exist_ok=True)
    outside_md.write_text("# Secret\n\nOutside content.", encoding="utf-8")
    started_tasks: list[dict] = []

    monkeypatch.setattr(generate_router, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(generate_router, "get_chat_store", lambda: fake_store)
    monkeypatch.setattr(generate_router, "_gen_start_task", lambda task: started_tasks.append(task) or True)
    monkeypatch.setattr(generate_router, "load_prefs", lambda: {})
    monkeypatch.setattr(chat_router, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(chat_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: pdf_root)

    client = TestClient(app)
    response = client.post(
        "/api/generate",
        json={
            "conv_id": "conv-1",
            "prompt": "summarize this",
            "source_lock_path": str(outside_md),
        },
    )

    assert response.status_code == 400
    assert fake_store.append_calls == 0
    assert started_tasks == []


def test_generate_accepts_selected_research_context(monkeypatch, tmp_path: Path):
    from api.routers import generate as generate_router

    class FakeStore:
        def __init__(self) -> None:
            self.messages: list[dict] = []

        def append_message(
            self,
            conv_id: str,
            role: str,
            content: str,
            attachments: list[dict] | None = None,
            meta: dict | None = None,
        ) -> int:
            self.messages.append({
                "conv_id": conv_id,
                "role": role,
                "content": content,
                "attachments": attachments,
                "meta": meta,
            })
            return len(self.messages)

        def set_title_if_default(self, conv_id: str, title: str) -> None:
            return False

        def get_conversation(self, conv_id: str) -> dict:
            return {"title": "Manual"}

    fake_store = FakeStore()
    started_tasks: list[dict] = []

    class FakeSettings:
        chat_db_path = tmp_path / "chat.sqlite3"
        db_dir = tmp_path / "db"

    monkeypatch.setattr(generate_router, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(generate_router, "get_chat_store", lambda: fake_store)
    monkeypatch.setattr(generate_router, "_gen_start_task", lambda task: started_tasks.append(task) or True)

    client = TestClient(app)
    response = client.post(
        "/api/generate",
        json={
            "conv_id": "conv-ctx",
            "prompt": "Use my selected excerpts to compare the baseline.",
            "query_scope": "basket",
            "prompt_context": {
                "id": "ctx-1",
                "source": "citation_shelf",
                "tokenEstimate": 5000,
                "items": [
                    {
                        "key": "r12",
                        "kind": "reference",
                        "title": "Sparse 3-D transform-domain filtering",
                        "sourceName": "reader.pdf",
                        "locationLabel": "References / [12]",
                        "refNum": 12,
                        "doi": "10.1109/tip.2007.901238",
                        "summary": "A denoising baseline frequently used for comparison.",
                        "excerpt": "x" * 1200,
                    },
                    {"key": "empty"},
                ],
            },
        },
    )

    assert response.status_code == 200
    assert started_tasks
    selected = started_tasks[0]["selected_research_context"]
    assert selected["source"] == "citation_shelf"
    assert selected["itemCount"] == 1
    assert selected["tokenEstimate"] == 1600
    assert selected["items"][0]["refNum"] == 12
    assert selected["items"][0]["title"] == "Sparse 3-D transform-domain filtering"
    assert len(selected["items"][0]["excerpt"]) <= 900
    assert started_tasks[0]["query_scope"] == "basket"
    assert fake_store.messages[0]["meta"]["prompt_context"]["itemCount"] == 1
    assert fake_store.messages[0]["meta"]["query_scope"] == "basket"


def test_generate_auto_titles_default_conversation(monkeypatch, tmp_path: Path):
    from api.routers import generate as generate_router
    from kb.chat_store import ChatStore

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("新会话 · 06/05 10:30")
    started_tasks: list[dict] = []

    class FakeSettings:
        chat_db_path = tmp_path / "chat.sqlite3"
        db_dir = tmp_path / "db"

    monkeypatch.setattr(generate_router, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(generate_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(generate_router, "_gen_start_task", lambda task: started_tasks.append(task) or True)

    client = TestClient(app)
    response = client.post(
        "/api/generate",
        json={
            "conv_id": conv_id,
            "prompt": "帮我看看 SPAD 阵列为什么需要物理先验，重点看方法部分",
        },
    )

    assert response.status_code == 200
    title = store.get_conversation(conv_id)["title"]
    assert title == "SPAD 阵列为什么需要物理先验"
    assert response.json()["conversation_title"] == title
    assert started_tasks


def test_generate_does_not_auto_title_manual_conversation(monkeypatch, tmp_path: Path):
    from api.routers import generate as generate_router
    from kb.chat_store import ChatStore

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("Manual literature plan")

    class FakeSettings:
        chat_db_path = tmp_path / "chat.sqlite3"
        db_dir = tmp_path / "db"

    monkeypatch.setattr(generate_router, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(generate_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(generate_router, "_gen_start_task", lambda task: True)

    client = TestClient(app)
    response = client.post(
        "/api/generate",
        json={
            "conv_id": conv_id,
            "prompt": "summarize the evidence map for this paper",
        },
    )

    assert response.status_code == 200
    assert store.get_conversation(conv_id)["title"] == "Manual literature plan"
    assert response.json()["conversation_title"] == "Manual literature plan"


def test_generate_reports_start_failure_and_persists_assistant_error(monkeypatch, tmp_path: Path):
    from api.routers import generate as generate_router
    from kb.chat_store import ChatStore

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("New chat")

    class FakeSettings:
        chat_db_path = tmp_path / "chat.sqlite3"
        db_dir = tmp_path / "db"

    monkeypatch.setattr(generate_router, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(generate_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(generate_router, "load_prefs", lambda: {})
    monkeypatch.setattr(generate_router, "_gen_start_task", lambda task: False)
    monkeypatch.setattr(generate_router, "_gen_get_task", lambda session_id: None)

    client = TestClient(app)
    response = client.post(
        "/api/generate",
        json={
            "conv_id": conv_id,
            "prompt": "summarize the paper",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["started"] is False
    assert payload["start_error"] == "generation_start_failed"
    messages = store.get_messages(conv_id)
    assistant = next(item for item in messages if item["role"] == "assistant")
    assert assistant["content"] == "Generation could not be started. Please retry."


def test_generate_reports_start_failure_in_ui_locale(monkeypatch, tmp_path: Path):
    from api.routers import generate as generate_router
    from kb.chat_store import ChatStore

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("新会话")

    class FakeSettings:
        chat_db_path = tmp_path / "chat.sqlite3"
        db_dir = tmp_path / "db"

    monkeypatch.setattr(generate_router, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(generate_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(generate_router, "load_prefs", lambda: {"ui_locale": "zh"})
    monkeypatch.setattr(generate_router, "_gen_start_task", lambda task: False)
    monkeypatch.setattr(generate_router, "_gen_get_task", lambda session_id: None)

    client = TestClient(app)
    response = client.post(
        "/api/generate",
        json={
            "conv_id": conv_id,
            "prompt": "总结这篇论文",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["started"] is False
    assert payload["start_error"] == "generation_start_failed"
    messages = store.get_messages(conv_id)
    assistant = next(item for item in messages if item["role"] == "assistant")
    assert assistant["content"] == "回答任务未能启动，请稍后重试。"


def test_generate_cleans_messages_when_start_task_loses_running_race(monkeypatch, tmp_path: Path):
    from api.routers import generate as generate_router
    from kb.chat_store import ChatStore

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("New chat")

    class FakeSettings:
        chat_db_path = tmp_path / "chat.sqlite3"
        db_dir = tmp_path / "db"

    running_checks = {"count": 0}

    def fake_has_running_for_conversation(*args, **kwargs):
        running_checks["count"] += 1
        return running_checks["count"] > 1

    monkeypatch.setattr(generate_router, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(generate_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(generate_router, "load_prefs", lambda: {})
    monkeypatch.setattr(generate_router, "_gen_has_running_for_conversation", fake_has_running_for_conversation)
    monkeypatch.setattr(generate_router, "_gen_start_task", lambda task: False)

    client = TestClient(app)
    response = client.post(
        "/api/generate",
        json={
            "conv_id": conv_id,
            "prompt": "second prompt while first answer is still starting",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "generation already running for this conversation"
    assert store.get_messages(conv_id) == []
    assert store.get_conversation(conv_id)["title"] == "New chat"
    assert running_checks["count"] == 2


def test_generate_rejects_running_conversation_before_appending_messages(monkeypatch, tmp_path: Path):
    from api.routers import generate as generate_router
    from kb import runtime_state as RUNTIME
    from kb.chat_store import ChatStore

    chat_db = tmp_path / "chat.sqlite3"
    store = ChatStore(chat_db)
    conv_id = store.create_conversation("Active generation")

    class FakeSettings:
        chat_db_path = chat_db
        db_dir = tmp_path / "db"

    monkeypatch.setattr(generate_router, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(generate_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(generate_router, "load_prefs", lambda: {})
    monkeypatch.setattr(
        generate_router,
        "_gen_start_task",
        lambda task: (_ for _ in ()).throw(AssertionError("start task should not be called")),
    )

    session_id = "existing-session"
    with RUNTIME.GEN_LOCK:
        RUNTIME.GEN_TASKS[session_id] = {
            "id": "existing-task",
            "session_id": session_id,
            "conv_id": conv_id,
            "chat_db": str(chat_db),
            "status": "running",
            "answer_ready": False,
            "created_at": 1.0,
            "updated_at": 1.0,
        }

    try:
        client = TestClient(app)
        response = client.post(
            "/api/generate",
            json={
                "conv_id": conv_id,
                "prompt": "second prompt while first answer is still running",
            },
        )

        assert response.status_code == 409
        assert response.json()["detail"] == "generation already running for this conversation"
        assert store.get_messages(conv_id) == []
    finally:
        with RUNTIME.GEN_LOCK:
            RUNTIME.GEN_TASKS.pop(session_id, None)


def test_generate_allows_retry_after_cancel_request(monkeypatch, tmp_path: Path):
    from api.routers import generate as generate_router
    from kb import runtime_state as RUNTIME
    from kb.chat_store import ChatStore

    chat_db = tmp_path / "chat.sqlite3"
    store = ChatStore(chat_db)
    conv_id = store.create_conversation("Cancel then retry")
    started_tasks: list[dict] = []

    class FakeSettings:
        chat_db_path = chat_db
        db_dir = tmp_path / "db"

    monkeypatch.setattr(generate_router, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(generate_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(generate_router, "load_prefs", lambda: {})
    monkeypatch.setattr(generate_router, "_gen_start_task", lambda task: started_tasks.append(dict(task)) or True)

    session_id = "existing-cancel-requested"
    with RUNTIME.GEN_LOCK:
        RUNTIME.GEN_TASKS[session_id] = {
            "id": "task-cancel-requested",
            "session_id": session_id,
            "conv_id": conv_id,
            "chat_db": str(chat_db),
            "status": "running",
            "answer_ready": False,
            "cancel": True,
            "created_at": 1.0,
            "updated_at": 1.0,
        }

    try:
        client = TestClient(app)
        response = client.post(
            "/api/generate",
            json={
                "conv_id": conv_id,
                "prompt": "new prompt immediately after cancel",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["started"] is True
        assert started_tasks and started_tasks[0]["conv_id"] == conv_id
        messages = store.get_messages(conv_id)
        assert [item["role"] for item in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "new prompt immediately after cancel"
    finally:
        with RUNTIME.GEN_LOCK:
            RUNTIME.GEN_TASKS.pop(session_id, None)


def test_generate_stream_exposes_public_answer_contract_fields(monkeypatch):
    from api.routers import generate as generate_router

    monkeypatch.setattr(
        generate_router,
        "_gen_get_task",
        lambda session_id: {
            "stage": "done",
            "partial": "ok",
            "char_count": 2,
            "status": "done",
            "answer": "ok",
            "answer_intent": "reading",
            "answer_depth": "L2",
            "answer_output_mode": "fact_answer",
            "answer_contract_v1": True,
            "answer_quality": {"minimum_ok": True, "core_section_coverage": 1.0},
            "paper_guide_debug": {"retrieval_mode": "internal"},
            "research_trace": {"trace_id": "trace-internal", "retrieval": {"raw_hit_count": 3}},
        },
    )

    client = TestClient(app)
    response = client.get("/api/generate/sid-1/stream")
    assert response.status_code == 200
    lines = [ln for ln in response.text.splitlines() if ln.startswith("data: ")]
    assert lines
    payload = json.loads(lines[-1][6:])
    assert payload["done"] is True
    assert payload["status"] == "done"
    assert payload["answer_intent"] == "reading"
    assert payload["answer_depth"] == "L2"
    assert payload["answer_output_mode"] == "fact_answer"
    assert payload["answer_contract_v1"] is True
    assert payload["answer_quality"] == {}
    assert payload["paper_guide_debug"] == {}
    assert payload["research_trace"] == {}


def test_generate_stream_keeps_agent_runtime_check_out_of_visible_answer(monkeypatch):
    from api.routers import generate as generate_router

    raw_visible = "Useful final answer.\n\nResearch Agent Trace\nPlan\n- retrieve_evidence debug"
    monkeypatch.setattr(
        generate_router,
        "_gen_get_task",
        lambda session_id: {
            "stage": "done",
            "partial": raw_visible,
            "char_count": len(raw_visible),
            "status": "done",
            "answer": raw_visible,
            "agent_mode": True,
            "agent_trace": {"mode": "research_agent", "summary": {"answer_source_blend": "general_llm"}},
            "agent_source_summary": {"kind": "general_api", "label": "Not from KB", "should_show": True},
            "answer_runtime_check": {
                "status": "passed",
                "repair": {"changed": True, "reasons": ["debug_content_removed"]},
            },
            "answer_contract": {
                "schema_version": 1,
                "source_summary": {"kind": "general_api", "label": "Not from KB", "should_show": True},
                "runtime_check": {
                    "status": "passed",
                    "repair": {"changed": True, "reasons": ["debug_content_removed"]},
                },
            },
        },
    )

    client = TestClient(app)
    response = client.get("/api/generate/sid-agent-visible/stream")
    assert response.status_code == 200
    lines = [ln for ln in response.text.splitlines() if ln.startswith("data: ")]
    assert lines
    payload = json.loads(lines[-1][6:])

    assert payload["done"] is True
    assert payload["answer"] == "Useful final answer."
    assert payload["partial"] == "Useful final answer."
    assert payload["answer_runtime_check"]["repair"]["changed"] is True
    assert payload["answer_contract"]["source_summary"]["kind"] == "general_api"
    assert payload["answer_contract"]["runtime_check"]["repair"]["changed"] is True
    assert "Research Agent Trace" not in payload["answer"]
    assert "answer_runtime_check" not in payload["answer"]


def test_generate_stream_exposes_internal_trace_only_when_internal_api_is_enabled(monkeypatch):
    from api.routers import generate as generate_router

    monkeypatch.setenv("KB_ENV", "development")
    monkeypatch.setenv("KB_APP_ENV", "")
    monkeypatch.setenv("KB_ENABLE_INTERNAL_API", "1")
    monkeypatch.setattr(
        generate_router,
        "_gen_get_task",
        lambda session_id: {
            "stage": "done",
            "partial": "ok",
            "char_count": 2,
            "status": "done",
            "answer": "ok",
            "paper_guide_debug": {"retrieval_mode": "internal"},
            "research_trace": {"trace_id": "trace-internal", "retrieval": {"raw_hit_count": 3}},
        },
    )

    client = TestClient(app)
    response = client.get("/api/generate/sid-1/stream")
    assert response.status_code == 200
    lines = [ln for ln in response.text.splitlines() if ln.startswith("data: ")]
    assert lines
    payload = json.loads(lines[-1][6:])

    assert payload["paper_guide_debug"] == {"retrieval_mode": "internal"}
    assert payload["research_trace"]["trace_id"] == "trace-internal"


def test_generate_stream_not_found_returns_error_frame(monkeypatch):
    from api.routers import generate as generate_router

    monkeypatch.setattr(generate_router, "_gen_get_task", lambda session_id: None)
    monkeypatch.setattr(generate_router, "load_prefs", lambda: {})

    client = TestClient(app)
    response = client.get("/api/generate/missing-session/stream")
    assert response.status_code == 200
    lines = [ln for ln in response.text.splitlines() if ln.startswith("data: ")]
    assert lines
    payload = json.loads(lines[-1][6:])
    assert payload["stream_schema_version"] == 2
    assert payload["done"] is True
    assert payload["status"] == "error"
    assert payload["stage"] == "error"
    assert payload["error"] == "not_found"
    assert payload["partial"] == "Generation could not be started. Please retry."
    assert payload["answer"] == payload["partial"]


def test_generate_stream_not_found_uses_ui_locale(monkeypatch):
    from api.routers import generate as generate_router

    monkeypatch.setattr(generate_router, "_gen_get_task", lambda session_id: None)
    monkeypatch.setattr(generate_router, "load_prefs", lambda: {"ui_locale": "zh"})

    client = TestClient(app)
    response = client.get("/api/generate/missing-session/stream")
    assert response.status_code == 200
    lines = [ln for ln in response.text.splitlines() if ln.startswith("data: ")]
    assert lines
    payload = json.loads(lines[-1][6:])
    assert payload["done"] is True
    assert payload["status"] == "error"
    assert payload["partial"] == "回答任务未能启动，请稍后重试。"
    assert payload["answer"] == payload["partial"]


def test_generate_stream_sanitizes_internal_support_markers(monkeypatch):
    from api.routers import generate as generate_router

    monkeypatch.setattr(
        generate_router,
        "_gen_get_task",
        lambda session_id: {
            "stage": "answer",
            "partial": "draft [[SUPPORT:DOC-1-S2]] still cites [[CITE:ref-1:12]]",
            "char_count": 58,
            "status": "done",
            "answer": "final [[SUPPORT:DOC-2]] still cites [[CITE:ref-2:34]]",
        },
    )

    client = TestClient(app)
    response = client.get("/api/generate/sid-support/stream")
    assert response.status_code == 200
    lines = [ln for ln in response.text.splitlines() if ln.startswith("data: ")]
    assert lines
    payload = json.loads(lines[-1][6:])

    assert "SUPPORT:" not in payload["partial"]
    assert "SUPPORT:" not in payload["answer"]
    assert "[[CITE:ref-1:12]]" in payload["partial"]
    assert "[[CITE:ref-2:34]]" in payload["answer"]
    assert payload["char_count"] == len(payload["partial"])


def test_generate_cancel_accepts_query_body_and_missing_task_id():
    from kb import runtime_state as RUNTIME

    def _put_running(session_id: str, task_id: str) -> None:
        with RUNTIME.GEN_LOCK:
            RUNTIME.GEN_TASKS[session_id] = {
                "id": task_id,
                "session_id": session_id,
                "conv_id": "conv-cancel",
                "chat_db": "chat.sqlite3",
                "status": "running",
                "answer_ready": False,
                "stage": "answer",
                "created_at": 1.0,
                "updated_at": 1.0,
            }

    client = TestClient(app)
    body_session = "session-cancel-body"
    query_session = "session-cancel-query"
    try:
        _put_running(body_session, "task-body")
        body_response = client.post(f"/api/generate/{body_session}/cancel", json={"task_id": "task-body"})
        assert body_response.status_code == 200
        assert body_response.json() == {"ok": True}
        with RUNTIME.GEN_LOCK:
            assert RUNTIME.GEN_TASKS[body_session]["cancel"] is True
            assert RUNTIME.GEN_TASKS[body_session]["stage"] == "canceled"

        _put_running(query_session, "task-query")
        query_response = client.post(f"/api/generate/{query_session}/cancel?task_id=task-query")
        assert query_response.status_code == 200
        assert query_response.json() == {"ok": True}

        missing_response = client.post(f"/api/generate/{query_session}/cancel")
        assert missing_response.status_code == 200
        assert missing_response.json() == {"ok": False}
    finally:
        with RUNTIME.GEN_LOCK:
            RUNTIME.GEN_TASKS.pop(body_session, None)
            RUNTIME.GEN_TASKS.pop(query_session, None)


def test_generate_quality_summary_route(monkeypatch):
    from api.routers import generate as generate_router

    monkeypatch.setenv("KB_ENABLE_INTERNAL_API", "1")
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        generate_router,
        "_gen_answer_quality_summary",
        lambda limit=200, intent="", depth="", only_failed=False: seen.update({"limit": limit, "intent": intent, "depth": depth, "only_failed": only_failed}) or {
            "limit": limit,
            "filters": {"intent": intent, "depth": depth, "only_failed": only_failed},
            "total": 2,
            "failed_count": 0,
            "failed_rate": 0.0,
            "structure_complete_rate": 1.0,
            "evidence_coverage_rate": 1.0,
            "next_steps_coverage_rate": 1.0,
            "minimum_ok_rate": 1.0,
            "avg_core_section_coverage": 1.0,
            "by_intent": {"reading": {"count": 2}},
            "by_depth": {"L2": {"count": 2, "minimum_ok_rate": 1.0, "avg_char_count": 120.0}},
            "fail_reasons": {},
        },
    )

    client = TestClient(app)
    response = client.get("/api/generate/quality/summary?limit=77&intent=reading&depth=L2&only_failed=true")
    assert response.status_code == 200
    payload = response.json()
    assert payload["limit"] == 77
    assert payload["filters"]["intent"] == "reading"
    assert payload["filters"]["depth"] == "L2"
    assert payload["filters"]["only_failed"] is True
    assert payload["total"] == 2
    assert payload["by_intent"]["reading"]["count"] == 2
    assert payload["by_depth"]["L2"]["count"] == 2
    assert seen["intent"] == "reading"
    assert seen["depth"] == "L2"
    assert bool(seen["only_failed"]) is True


def test_chat_uploads_route_marks_pdf_ingest_start_failure(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    missing_pdf = tmp_path / "missing.pdf"

    def fake_save_pdf_to_library(*, file_name: str, data: bytes, base_name: str = "", fast_mode: bool = False) -> dict:
        return {
            "duplicate": False,
            "path": str(missing_pdf),
            "name": "missing.pdf",
            "sha1": "pdfsha1",
        }

    monkeypatch.setattr(chat_router, "save_pdf_to_library", fake_save_pdf_to_library)
    monkeypatch.setattr(chat_router, "_path_exists", lambda path: False)

    client = TestClient(app)
    response = client.post(
      "/api/chat/uploads",
      files=[("files", ("missing.pdf", b"%PDF-1.4 test", "application/pdf"))],
      data={"quick_ingest": "true", "speed_mode": "ultra_fast"},
    )

    assert response.status_code == 200
    pdf_item = response.json()["items"][0]
    assert pdf_item["kind"] == "pdf"
    assert pdf_item["status"] == "error"
    assert pdf_item["ingest_status"] == "error"
    assert "not started" in pdf_item["error"]


def test_chat_upload_cancel_and_retry_routes(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    client = TestClient(app)

    class DummyProc:
        def __init__(self) -> None:
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout: float | None = None):
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = 0

    proc = DummyProc()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: tmp_path)
    chat_router._CHAT_UPLOAD_JOBS["job-1"] = {
        "job_id": "job-1",
        "name": "paper.pdf",
        "sha1": "sha1",
        "path": str(pdf_path),
        "ready": False,
        "ingest_status": "converting",
        "speed_mode": "ultra_fast",
        "cancel_requested": False,
        "ingest_proc": proc,
    }

    cancel_response = client.post("/api/chat/uploads/cancel", json={"job_id": "job-1"})
    assert cancel_response.status_code == 200
    cancelled_item = cancel_response.json()["item"]
    assert cancelled_item["ingest_status"] == "cancelled"
    assert cancelled_item["status"] == "error"
    assert proc.terminated is True

    def fake_start_chat_pdf_ingest_job(
        *,
        pdf_path: Path,
        speed_mode: str,
        display_name: str,
        sha1: str = "",
        conv_id: str = "",
    ) -> str:
        chat_router._CHAT_UPLOAD_JOBS["job-2"] = {
            "job_id": "job-2",
            "name": display_name,
            "sha1": sha1,
            "path": str(pdf_path),
            "ready": False,
            "ingest_status": "converting",
            "speed_mode": speed_mode,
            "conv_id": conv_id,
            "cancel_requested": False,
            "error": "",
        }
        return "job-2"

    monkeypatch.setattr(chat_router, "_start_chat_pdf_ingest_job", fake_start_chat_pdf_ingest_job)

    retry_response = client.post("/api/chat/uploads/retry", json={"job_id": "job-1"})
    assert retry_response.status_code == 200
    retried_item = retry_response.json()["item"]
    assert retried_item["ingest_job_id"] == "job-2"
    assert retried_item["ingest_status"] == "converting"
    assert retried_item["path"] == str(pdf_path)


def test_chat_upload_retry_rejects_pdf_path_outside_library(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    pdf_root = tmp_path / "pdfs"
    outside = tmp_path / "outside" / "paper.pdf"
    pdf_root.mkdir()
    outside.parent.mkdir()
    outside.write_bytes(b"%PDF-1.4 test")
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: pdf_root)
    chat_router._CHAT_UPLOAD_JOBS["job-outside"] = {
        "job_id": "job-outside",
        "name": "paper.pdf",
        "sha1": "sha1",
        "path": str(outside),
        "ready": False,
        "ingest_status": "error",
        "speed_mode": "ultra_fast",
        "cancel_requested": False,
        "error": "previous failure",
    }

    client = TestClient(app)
    retry_response = client.post("/api/chat/uploads/retry", json={"job_id": "job-outside"})

    assert retry_response.status_code == 400
    assert "configured PDF directory" in retry_response.json()["detail"]


def test_chat_upload_retry_rejects_invalid_pdf_content(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir()
    fake_pdf = pdf_root / "fake.pdf"
    fake_pdf.write_bytes(b"not really a pdf")
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: pdf_root)
    chat_router._CHAT_UPLOAD_JOBS["job-fake"] = {
        "job_id": "job-fake",
        "name": "fake.pdf",
        "sha1": "sha1",
        "path": str(fake_pdf),
        "ready": False,
        "ingest_status": "error",
        "speed_mode": "ultra_fast",
        "cancel_requested": False,
        "error": "previous failure",
    }

    client = TestClient(app)
    retry_response = client.post("/api/chat/uploads/retry", json={"job_id": "job-fake"})

    assert retry_response.status_code == 400
    assert retry_response.json()["detail"] == "pdf file is not a valid PDF"


def test_chat_upload_quality_retry_and_cancel_routes(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    client = TestClient(app)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: tmp_path)

    chat_router._CHAT_UPLOAD_JOBS["job-q"] = {
        "job_id": "job-q",
        "name": "paper.pdf",
        "sha1": "sha1",
        "path": str(pdf_path),
        "ready": True,
        "ingest_status": "ready",
        "speed_mode": "ultra_fast",
        "quality_status": "error",
        "quality_stage": "error",
        "quality_error": "mock fail",
        "cancel_requested": False,
        "error": "",
    }

    def fake_start_chat_pdf_quality_refine(job_id: str) -> None:
        assert job_id == "job-q"
        chat_router._set_chat_pdf_ingest_job(
            job_id,
            {
                "quality_status": "running",
                "quality_stage": "refining",
                "quality_error": "",
            },
        )

    monkeypatch.setattr(chat_router, "_start_chat_pdf_quality_refine", fake_start_chat_pdf_quality_refine)

    retry_response = client.post("/api/chat/uploads/quality/retry", json={"job_id": "job-q"})
    assert retry_response.status_code == 200
    retried_item = retry_response.json()["item"]
    assert retried_item["ingest_job_id"] == "job-q"
    assert retried_item["ingest_status"] == "ready"
    assert retried_item["quality_status"] == "running"

    cancel_response = client.post("/api/chat/uploads/cancel", json={"job_id": "job-q"})
    assert cancel_response.status_code == 200
    cancelled_item = cancel_response.json()["item"]
    assert cancelled_item["ingest_status"] == "ready"
    assert cancelled_item["quality_status"] == "cancelled"


def test_chat_upload_duplicate_binds_conversation_source(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    existing_pdf = tmp_path / "dup.pdf"
    existing_pdf.write_bytes(b"%PDF-1.4 test")

    class FakeStore:
        def __init__(self) -> None:
            self.bound: list[tuple[str, str, str]] = []

        def get_conversation(self, conv_id: str):
            return {"id": conv_id}

        def bind_conversation_source(self, conv_id: str, source_path: str, source_name: str = "") -> bool:
            self.bound.append((conv_id, source_path, source_name))
            return True

    fake_store = FakeStore()

    def fake_save_pdf_to_library(*, file_name: str, data: bytes, base_name: str = "", fast_mode: bool = False) -> dict:
        return {
            "duplicate": True,
            "path": str(existing_pdf),
            "name": existing_pdf.name,
            "sha1": "dup-sha1",
            "existing": existing_pdf.name,
        }

    monkeypatch.setattr(chat_router, "save_pdf_to_library", fake_save_pdf_to_library)
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: fake_store)

    client = TestClient(app)
    response = client.post(
      "/api/chat/uploads",
      files=[("files", ("dup.pdf", b"%PDF-1.4 test", "application/pdf"))],
      data={"quick_ingest": "true", "conv_id": "conv-1"},
    )

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["status"] == "duplicate"
    assert item["ingest_status"] == "ready"
    assert fake_store.bound == [("conv-1", str(existing_pdf), existing_pdf.name)]


def test_references_asset_route_serves_md_assets_only(monkeypatch, tmp_path: Path):
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    asset_dir = md_root / "DocA" / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    in_root_asset = asset_dir / "page_1_fig_2.png"
    in_root_asset.write_bytes(TINY_PNG_BYTES)
    fake_in_root_asset = asset_dir / "not_really.png"
    fake_in_root_asset.write_bytes(b"not really an image")
    out_root_asset = tmp_path / "outside.png"
    out_root_asset.write_bytes(TINY_PNG_BYTES)

    monkeypatch.setattr(refs_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(refs_router, "_reference_asset_roots", lambda: [md_root.resolve()])
    client = TestClient(app)

    ok_resp = client.get("/api/references/asset", params={"path": str(in_root_asset)})
    assert ok_resp.status_code == 200
    assert ok_resp.headers["content-type"].startswith("image/png")
    assert ok_resp.headers["cache-control"] == "no-cache, max-age=0"

    bad_resp = client.get("/api/references/asset", params={"path": str(out_root_asset)})
    assert bad_resp.status_code == 404

    fake_resp = client.get("/api/references/asset", params={"path": str(fake_in_root_asset)})
    assert fake_resp.status_code == 404


def test_references_asset_route_rejects_unbounded_path(monkeypatch, tmp_path: Path):
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    md_root.mkdir(parents=True)
    monkeypatch.setattr(refs_router, "_reference_asset_roots", lambda: [md_root.resolve()])

    client = TestClient(app)
    response = client.get("/api/references/asset", params={"path": "x" * 1300})

    assert response.status_code == 404


def test_references_reader_doc_rewrites_tmp_assets_and_serves(monkeypatch, tmp_path: Path):
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    tmp_root = tmp_path / "tmp"
    doc_dir = tmp_root / "reconvert_doc"
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    fig = assets_dir / "page_2_fig_1.png"
    fig.write_bytes(TINY_PNG_BYTES)
    md_path = doc_dir / "output.md"
    md_path.write_text("![Figure 1](./assets/page_2_fig_1.png)\n", encoding="utf-8")

    monkeypatch.setattr(refs_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(refs_router, "_reference_asset_roots", lambda: [md_root.resolve(), tmp_root.resolve()])
    client = TestClient(app)

    doc_resp = client.post("/api/references/reader/doc", json={"source_path": str(md_path)})
    assert doc_resp.status_code == 200
    markdown = str(doc_resp.json().get("markdown") or "")
    assert "/api/references/asset?path=" in markdown
    assert "&v=" in markdown

    m = re.search(r"\((/api/references/asset\?path=[^)]+)\)", markdown)
    assert m is not None
    asset_url = m.group(1)
    asset_resp = client.get(asset_url)
    assert asset_resp.status_code == 200
    assert asset_resp.headers["content-type"].startswith("image/png")
    assert asset_resp.headers["cache-control"] == "no-cache, max-age=0"


def test_public_projected_nested_source_opens_without_absolute_path_leak(
    monkeypatch,
    tmp_path: Path,
):
    import api.reference_ui as reference_ui
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    source_path = md_root / "collection" / "paper" / "Paper.en.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("# Paper\n\nDirect evidence.\n", encoding="utf-8")
    roots = [md_root.resolve()]
    monkeypatch.setattr(refs_router, "_reference_asset_roots", lambda: roots)

    projected = reference_ui.public_refs_payload_projection(
        {
            1: {
                "hits": [
                    {
                        "meta": {"source_path": str(source_path)},
                        "ui_meta": {
                            "source_path": str(source_path),
                            "reader_open": {"sourcePath": str(source_path)},
                        },
                    }
                ]
            }
        },
        source_roots=roots,
    )
    source_id = projected[1]["hits"][0]["ui_meta"]["reader_open"]["sourcePath"]
    assert source_id == "kb-source/0/collection/paper/Paper.en.md"
    assert str(tmp_path) not in json.dumps(projected)

    client = TestClient(app)
    doc_resp = client.post("/api/references/reader/doc", json={"source_path": source_id})

    assert doc_resp.status_code == 200
    payload = doc_resp.json()
    assert payload["source_path"] == source_id
    assert payload["md_path"] == source_id
    assert "Direct evidence" in payload["markdown"]
    assert str(tmp_path) not in json.dumps(payload)

    from api.routers import chat as chat_router
    from kb.chat_store import ChatStore

    db_dir = tmp_path / "db"
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_root / "Paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% safe test fixture\n")
    store = ChatStore(tmp_path / "chat.sqlite3")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(chat_router, "get_settings", lambda: SimpleNamespace(db_dir=db_dir))
    monkeypatch.setattr(chat_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: pdf_root)
    monkeypatch.setattr(chat_router, "_kickoff_paper_guide_prefetch_if_needed", lambda **_kwargs: None)

    guide_resp = client.post(
        "/api/conversations",
        json={
            "title": "Paper guide",
            "mode": "paper_guide",
            "bound_source_path": source_id,
            "bound_source_name": "Paper.pdf",
            "bound_source_ready": True,
        },
    )

    assert guide_resp.status_code == 200
    guide = store.get_conversation(guide_resp.json()["id"])
    assert guide is not None
    assert guide["bound_source_path"] == str(source_path.resolve(strict=False))

    session_resp = client.post(
        "/api/reader/sessions",
        json={
            "payload": {"sourcePath": source_id, "sourceName": "Paper.pdf"},
            "conversation_id": guide_resp.json()["id"],
        },
    )
    assert session_resp.status_code == 200
    session_payload = session_resp.json()
    assert session_payload["payload"]["sourcePath"] == source_id
    session_id = session_payload["id"]
    session_get = client.get(f"/api/reader/sessions/{session_id}")
    assert session_get.status_code == 200
    assert session_get.json()["payload"]["sourcePath"] == source_id
    session_patch = client.patch(
        f"/api/reader/sessions/{session_id}/state",
        json={"state": {"scrollTop": 24}},
    )
    assert session_patch.status_code == 200
    assert session_patch.json()["payload"]["sourcePath"] == source_id

    reader_state_get = client.get(
        f"/api/conversations/{guide_resp.json()['id']}/reader-state",
        params={"source_path": source_id},
    )
    assert reader_state_get.status_code == 200
    assert reader_state_get.json()["source_path"] == source_id
    reader_state_patch = client.patch(
        f"/api/conversations/{guide_resp.json()['id']}/reader-state",
        params={"source_path": source_id},
        json={"state": {"scrollTop": 48}},
    )
    assert reader_state_patch.status_code == 200
    assert reader_state_patch.json()["source_path"] == source_id

    citation_calls: list[str] = []
    monkeypatch.setattr(refs_router, "_pdf_dir", lambda: pdf_root)
    monkeypatch.setattr(refs_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(refs_router, "_lib_store", lambda: None)
    monkeypatch.setattr(
        refs_router,
        "ensure_source_citation_meta",
        lambda **kwargs: citation_calls.append(str(kwargs["source_path"]))
        or {"source_path": str(kwargs["source_path"]), "title": "Paper"},
    )
    citation_resp = client.post(
        "/api/references/citation-meta",
        json={"source_path": source_id},
    )

    assert citation_resp.status_code == 200
    assert citation_calls == [str(source_path.resolve(strict=False))]
    assert citation_resp.json()["source_path"] == source_id

    import api.reference_ui as reference_ui_module

    opened: dict[str, object] = {}
    monkeypatch.setattr(
        reference_ui_module,
        "_open_pdf_at",
        lambda path, page=None: opened.update({"path": path, "page": page})
        or (True, f"Opened: {path}"),
    )
    open_result = refs_router.open_reference(
        refs_router.OpenReferenceBody(source_path=source_id, page=2)
    )

    assert open_result == {"ok": True, "message": "PDF opened"}
    assert opened == {"path": pdf_path.resolve(strict=False), "page": 2}


def test_references_reader_doc_accepts_file_url_source_path_variants(monkeypatch, tmp_path: Path):
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    doc_dir = md_root / "Doc With Spaces"
    doc_dir.mkdir(parents=True, exist_ok=True)
    md_path = doc_dir / "Doc With Spaces.en.md"
    md_path.write_text("# Paper\n\nimportant sentence\n", encoding="utf-8")

    monkeypatch.setattr(refs_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(refs_router, "_reference_asset_roots", lambda: [md_root.resolve()])
    client = TestClient(app)

    file_url_source_path = "file:///" + str(md_path).replace("\\", "/").replace(" ", "%20") + "?download=1#reader"
    doc_resp = client.post("/api/references/reader/doc", json={"source_path": file_url_source_path})

    assert doc_resp.status_code == 200
    payload = doc_resp.json()
    assert payload["source_path"] == "kb-source/0/Doc With Spaces/Doc With Spaces.en.md"
    assert payload["md_path"] == payload["source_path"]
    assert "important sentence" in payload["markdown"]


def test_references_citation_meta_accepts_file_url_source_path_variants(monkeypatch, tmp_path: Path):
    import api.reference_ui as reference_ui
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    doc_dir = md_root / "Doc With Spaces"
    doc_dir.mkdir(parents=True, exist_ok=True)
    md_path = doc_dir / "Doc With Spaces.en.md"
    md_path.write_text(
        "# A Better Local Paper Title\n\n"
        "This local abstract gives the citation metadata route a title source.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(refs_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(refs_router, "_lib_store", lambda: None)
    monkeypatch.setattr(reference_ui, "fetch_crossref_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_enrich_bibliometrics", lambda meta: meta)
    client = TestClient(app)

    file_url_source_path = "file:///" + str(md_path).replace("\\", "/").replace(" ", "%20") + "?download=1#reader"
    meta_resp = client.post("/api/references/citation-meta", json={"source_path": file_url_source_path})

    assert meta_resp.status_code == 200
    payload = meta_resp.json()
    assert payload["title"] == "A Better Local Paper Title"


def test_references_reader_doc_rewrites_angle_bracket_asset_path_with_spaces(monkeypatch, tmp_path: Path):
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    doc_dir = md_root / "Doc With Spaces"
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    fig = assets_dir / "Figure 1 Space.png"
    fig.write_bytes(TINY_PNG_BYTES)
    md_path = doc_dir / "Doc With Spaces.en.md"
    md_path.write_text('![Figure 1](<./assets/Figure 1 Space.png> "figure title")\n', encoding="utf-8")

    monkeypatch.setattr(refs_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(refs_router, "_reference_asset_roots", lambda: [md_root.resolve()])
    client = TestClient(app)

    doc_resp = client.post("/api/references/reader/doc", json={"source_path": str(md_path)})
    assert doc_resp.status_code == 200
    markdown = str(doc_resp.json().get("markdown") or "")
    assert "/api/references/asset?path=" in markdown
    assert "Figure%201%20Space.png" in markdown

    m = re.search(r"\((/api/references/asset\?path=[^)]+)\)", markdown)
    assert m is not None
    asset_resp = client.get(m.group(1))
    assert asset_resp.status_code == 200
    assert asset_resp.headers["content-type"].startswith("image/png")


def test_references_reader_doc_rewrites_asset_path_with_parentheses(monkeypatch, tmp_path: Path):
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    doc_dir = md_root / "DocA"
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    fig = assets_dir / "Figure (1).png"
    fig.write_bytes(TINY_PNG_BYTES)
    md_path = doc_dir / "DocA.en.md"
    md_path.write_text('![Figure 1](./assets/Figure (1).png "panel title")\n', encoding="utf-8")

    monkeypatch.setattr(refs_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(refs_router, "_reference_asset_roots", lambda: [md_root.resolve()])
    client = TestClient(app)

    doc_resp = client.post("/api/references/reader/doc", json={"source_path": str(md_path)})
    assert doc_resp.status_code == 200
    markdown = str(doc_resp.json().get("markdown") or "")
    assert "/api/references/asset?path=" in markdown
    assert "Figure%20%281%29.png" in markdown

    m = re.search(r"\((/api/references/asset\?path=[^)]+)\)", markdown)
    assert m is not None
    asset_resp = client.get(m.group(1))
    assert asset_resp.status_code == 200
    assert asset_resp.headers["content-type"].startswith("image/png")


def test_references_reader_doc_rewrites_percent_encoded_asset_path(monkeypatch, tmp_path: Path):
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    doc_dir = md_root / "DocA"
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    fig = assets_dir / "Figure 2.png"
    fig.write_bytes(TINY_PNG_BYTES)
    md_path = doc_dir / "DocA.en.md"
    md_path.write_text("![Figure 2](./assets/Figure%202.png)\n", encoding="utf-8")

    monkeypatch.setattr(refs_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(refs_router, "_reference_asset_roots", lambda: [md_root.resolve()])
    client = TestClient(app)

    doc_resp = client.post("/api/references/reader/doc", json={"source_path": str(md_path)})
    assert doc_resp.status_code == 200
    markdown = str(doc_resp.json().get("markdown") or "")
    assert "/api/references/asset?path=" in markdown
    assert "Figure%202.png" in markdown

    m = re.search(r"\((/api/references/asset\?path=[^)]+)\)", markdown)
    assert m is not None
    asset_resp = client.get(m.group(1))
    assert asset_resp.status_code == 200
    assert asset_resp.headers["content-type"].startswith("image/png")


def test_references_reader_doc_rewrites_escaped_space_asset_path(monkeypatch, tmp_path: Path):
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    doc_dir = md_root / "DocA"
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    fig = assets_dir / "Figure 3.png"
    fig.write_bytes(TINY_PNG_BYTES)
    md_path = doc_dir / "DocA.en.md"
    md_path.write_text("![Figure 3](./assets/Figure\\ 3.png)\n", encoding="utf-8")

    monkeypatch.setattr(refs_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(refs_router, "_reference_asset_roots", lambda: [md_root.resolve()])
    client = TestClient(app)

    doc_resp = client.post("/api/references/reader/doc", json={"source_path": str(md_path)})
    assert doc_resp.status_code == 200
    markdown = str(doc_resp.json().get("markdown") or "")
    assert "/api/references/asset?path=" in markdown
    assert "Figure%203.png" in markdown

    m = re.search(r"\((/api/references/asset\?path=[^)]+)\)", markdown)
    assert m is not None
    asset_resp = client.get(m.group(1))
    assert asset_resp.status_code == 200
    assert asset_resp.headers["content-type"].startswith("image/png")


def test_references_reader_doc_does_not_rewrite_fake_image_asset(monkeypatch, tmp_path: Path):
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    doc_dir = md_root / "DocA"
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    fake_fig = assets_dir / "page_2_fig_1.png"
    fake_fig.write_bytes(b"not really an image")
    md_path = doc_dir / "DocA.en.md"
    md_path.write_text("![Figure 1](./assets/page_2_fig_1.png)\n", encoding="utf-8")

    monkeypatch.setattr(refs_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(refs_router, "_reference_asset_roots", lambda: [md_root.resolve()])
    client = TestClient(app)

    doc_resp = client.post("/api/references/reader/doc", json={"source_path": str(md_path)})
    assert doc_resp.status_code == 200
    markdown = str(doc_resp.json().get("markdown") or "")
    assert "/api/references/asset?path=" not in markdown
    assert "![Figure 1](./assets/page_2_fig_1.png)" in markdown


def test_references_reader_doc_rejects_oversized_source_path_before_resolution(monkeypatch, tmp_path: Path):
    from api.routers import references as refs_router

    called = {"resolve": 0}

    def fail_resolve(source_path: str):
        called["resolve"] += 1
        raise AssertionError("reader path resolution should not run for invalid body")

    monkeypatch.setattr(refs_router, "_resolve_reader_md_path", fail_resolve)
    client = TestClient(app)

    response = client.post("/api/references/reader/doc", json={"source_path": "x" * 1300})

    assert response.status_code == 422
    assert called["resolve"] == 0


def test_references_reader_doc_rejects_markdown_outside_allowed_roots(monkeypatch, tmp_path: Path):
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    outside = tmp_path / "outside" / "paper.md"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("# Outside\n", encoding="utf-8")

    monkeypatch.setattr(refs_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(refs_router, "_reference_asset_roots", lambda: [md_root.resolve()])
    client = TestClient(app)

    doc_resp = client.post("/api/references/reader/doc", json={"source_path": str(outside)})
    assert doc_resp.status_code == 404


def test_references_reader_doc_rejects_pdf_outside_pdf_root_even_with_matching_markdown(monkeypatch, tmp_path: Path):
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir(parents=True, exist_ok=True)
    inside_pdf = pdf_root / "Paper.pdf"
    outside_pdf = tmp_path / "outside" / "Paper.pdf"
    inside_pdf.write_bytes(b"%PDF-1.4\n")
    outside_pdf.parent.mkdir(parents=True, exist_ok=True)
    outside_pdf.write_bytes(b"%PDF-1.4\n")
    doc_dir = md_root / "Paper"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "Paper.en.md").write_text("# Paper\n\nAllowed conversion.\n", encoding="utf-8")

    monkeypatch.setattr(refs_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(refs_router, "_pdf_dir", lambda: pdf_root)
    monkeypatch.setattr(refs_router, "_reference_asset_roots", lambda: [md_root.resolve()])
    client = TestClient(app)

    ok_resp = client.post("/api/references/reader/doc", json={"source_path": str(inside_pdf)})
    assert ok_resp.status_code == 200

    bad_resp = client.post("/api/references/reader/doc", json={"source_path": str(outside_pdf)})
    assert bad_resp.status_code == 404


def test_references_open_rejects_pdf_outside_pdf_root(monkeypatch, tmp_path: Path):
    import api.reference_ui as reference_ui
    from api.routers import references as refs_router

    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir(parents=True)
    paper = pdf_root / "Paper.pdf"
    paper.write_bytes(b"%PDF-1.4\n")
    outside = tmp_path / "outside" / "Paper.pdf"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"%PDF-1.4\n")
    opened: list[Path] = []

    def fake_open_pdf_at(pdf_path: Path, *, page: int | None = None) -> tuple[bool, str]:
        opened.append(Path(pdf_path))
        return True, "opened"

    monkeypatch.setattr(refs_router, "_pdf_dir", lambda: pdf_root)
    monkeypatch.setattr(reference_ui, "_open_pdf_at", fake_open_pdf_at)
    client = TestClient(app)

    ok_resp = client.post("/api/references/open", json={"source_path": str(paper)})
    assert ok_resp.status_code == 200
    assert opened == [paper.resolve(strict=False)]

    bad_resp = client.post("/api/references/open", json={"source_path": str(outside)})
    assert bad_resp.status_code == 404
    assert opened == [paper.resolve(strict=False)]


def test_references_open_accepts_file_url_source_path_variants(monkeypatch, tmp_path: Path):
    import api.reference_ui as reference_ui
    from api.routers import references as refs_router

    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir(parents=True)
    paper = pdf_root / "Paper With Spaces.pdf"
    paper.write_bytes(b"%PDF-1.4\n")
    opened: list[Path] = []

    def fake_open_pdf_at(pdf_path: Path, *, page: int | None = None) -> tuple[bool, str]:
        opened.append(Path(pdf_path))
        return True, "opened"

    monkeypatch.setattr(refs_router, "_pdf_dir", lambda: pdf_root)
    monkeypatch.setattr(reference_ui, "_open_pdf_at", fake_open_pdf_at)
    client = TestClient(app)

    file_url_source_path = "file:///" + str(paper).replace("\\", "/").replace(" ", "%20") + "?download=1#viewer"
    ok_resp = client.post("/api/references/open", json={"source_path": file_url_source_path})

    assert ok_resp.status_code == 200
    assert opened == [paper.resolve(strict=False)]


def test_references_reader_doc_exposes_outline_contract(monkeypatch, tmp_path: Path):
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    doc_dir = md_root / "paper"
    doc_dir.mkdir(parents=True, exist_ok=True)
    md_path = doc_dir / "paper.en.md"
    md_path.write_text(
        "# Paper Title\n\n## Abstract\n\nA compact abstract.\n\n### Details\n\nMore detail.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(refs_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(refs_router, "_reference_asset_roots", lambda: [md_root.resolve()])
    client = TestClient(app)

    doc_resp = client.post("/api/references/reader/doc", json={"source_path": str(md_path)})
    assert doc_resp.status_code == 200
    payload = doc_resp.json()
    assert re.fullmatch(r"[0-9a-f]{40}", str(payload.get("doc_hash") or ""))
    quality = payload.get("outline_quality") or {}
    assert quality["ok"] is True
    assert quality["has_document_title"] is True
    assert quality["heading_count"] == 3

    heading_blocks = [row for row in payload["blocks"] if row.get("kind") == "heading"]
    assert [row.get("heading_level") for row in heading_blocks] == [1, 2, 3]
    heading_anchors = [row for row in payload["anchors"] if row.get("kind") == "heading"]
    assert [row.get("heading_level") for row in heading_anchors] == [1, 2, 3]


def test_references_reader_doc_exposes_reference_cite_details(monkeypatch, tmp_path: Path):
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    doc_dir = md_root / "paper"
    doc_dir.mkdir(parents=True, exist_ok=True)
    md_path = doc_dir / "paper.en.md"
    md_path.write_text(
        "# Paper Title\n\n"
        "Prior spectral imaging appears in this paper [1].\n\n"
        "## References\n\n"
        "[1] Gehm M, Brady D. Single-shot compressive spectral imaging with a dual-disperser architecture. Optics Express, 2007. doi:10.1364/OE.15.014013\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(refs_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(refs_router, "_reference_asset_roots", lambda: [md_root.resolve()])
    monkeypatch.setattr(refs_router, "_reader_reference_index_data", lambda: {})
    client = TestClient(app)

    doc_resp = client.post("/api/references/reader/doc", json={"source_path": str(md_path)})
    assert doc_resp.status_code == 200
    payload = doc_resp.json()
    details = payload.get("cite_details") or []
    assert len(details) == 1
    detail = details[0]
    assert detail["num"] == 1
    assert detail["is_inpaper"] is True
    assert detail["citation_route"] == "system_b"
    assert str(detail["anchor"]).startswith("kb-cite-reader-")
    assert "Single-shot compressive spectral imaging" in detail["raw"]
    assert "Single-shot compressive spectral imaging" in detail["card_reference_entry"]
