from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app


def test_reader_sessions_persist_payload(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    db_dir = tmp_path / "db"
    monkeypatch.setattr(
        chat_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=db_dir),
    )

    client = TestClient(app)
    missing = client.post("/api/reader/sessions", json={"payload": {}})
    assert missing.status_code == 400

    response = client.post(
        "/api/reader/sessions",
        json={
            "title": "Reader source",
            "conversation_id": "conv-1",
            "payload": {
                "sourcePath": str(tmp_path / "source.en.md"),
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


def test_conversation_reader_state_persists_by_source(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router
    from kb.chat_store import ChatStore

    store = ChatStore(tmp_path / "chat.sqlite3")
    conv_id = store.create_conversation("Guide")
    source_path = str(tmp_path / "paper.en.md")
    other_source_path = str(tmp_path / "other.en.md")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)

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

    other = client.get(
        f"/api/conversations/{conv_id}/reader-state",
        params={"source_path": other_source_path},
    )
    assert other.status_code == 200
    assert other.json()["state"] == {}


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
        ("files", ("figure.png", b"\x89PNG\r\n\x1a\nfake", "image/png")),
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
    assert Path(image_item["attachment"]["path"]).exists()
    assert image_item["attachment"]["url"].startswith("/api/chat/uploads/image?path=")

    image_response = client.get(image_item["attachment"]["url"])
    assert image_response.status_code == 200
    assert image_response.headers["content-type"].startswith("image/png")


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
            "path": str(tmp_path / "img.png"),
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
    assert fake_store.messages[0]["meta"]["prompt_context"]["itemCount"] == 1


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


def test_generate_stream_exposes_answer_probe_fields(monkeypatch):
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
    assert payload["answer_quality"]["minimum_ok"] is True


def test_generate_quality_summary_route(monkeypatch):
    from api.routers import generate as generate_router
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


def test_chat_upload_quality_retry_and_cancel_routes(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    client = TestClient(app)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

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
    in_root_asset.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    out_root_asset = tmp_path / "outside.png"
    out_root_asset.write_bytes(b"\x89PNG\r\n\x1a\nfake2")

    monkeypatch.setattr(refs_router, "_md_dir", lambda: md_root)
    monkeypatch.setattr(refs_router, "_reference_asset_roots", lambda: [md_root.resolve()])
    client = TestClient(app)

    ok_resp = client.get("/api/references/asset", params={"path": str(in_root_asset)})
    assert ok_resp.status_code == 200
    assert ok_resp.headers["content-type"].startswith("image/png")
    assert ok_resp.headers["cache-control"] == "no-cache, max-age=0"

    bad_resp = client.get("/api/references/asset", params={"path": str(out_root_asset)})
    assert bad_resp.status_code == 404


def test_references_reader_doc_rewrites_tmp_assets_and_serves(monkeypatch, tmp_path: Path):
    from api.routers import references as refs_router

    md_root = tmp_path / "md_output"
    tmp_root = tmp_path / "tmp"
    doc_dir = tmp_root / "reconvert_doc"
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    fig = assets_dir / "page_2_fig_1.png"
    fig.write_bytes(b"\x89PNG\r\n\x1a\nfake-fig")
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
