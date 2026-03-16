from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


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

        def append_message(self, conv_id: str, role: str, content: str, attachments: list[dict] | None = None) -> int:
            self.messages.append((conv_id, role, content, attachments))
            return len(self.messages)

        def set_title_if_default(self, conv_id: str, title: str) -> None:
            self.titles.append((conv_id, title))

        def get_conversation(self, conv_id: str) -> dict:
            return {}

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
    assert started_tasks
    assert started_tasks[0]["image_attachments"][0]["name"] == "img.png"


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
    client = TestClient(app)

    ok_resp = client.get("/api/references/asset", params={"path": str(in_root_asset)})
    assert ok_resp.status_code == 200
    assert ok_resp.headers["content-type"].startswith("image/png")

    bad_resp = client.get("/api/references/asset", params={"path": str(out_root_asset)})
    assert bad_resp.status_code == 404
