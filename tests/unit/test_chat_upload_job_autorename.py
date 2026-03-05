from __future__ import annotations

import time
from pathlib import Path


def test_chat_pdf_job_autorenames_before_ingest(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    src_pdf = tmp_path / "upload.pdf"
    src_pdf.write_bytes(b"%PDF-1.4 test")
    renamed_pdf = tmp_path / "Venue-2024-Title.pdf"
    renamed_pdf.write_bytes(b"%PDF-1.4 test")
    md_path = tmp_path / "paper" / "paper.en.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("# paper\n", encoding="utf-8")

    called: dict[str, object] = {}

    def fake_auto_rename_saved_pdf_in_library(*, pdf_path: Path, base_name: str = "") -> dict:
        assert pdf_path == src_pdf
        called["renamed"] = True
        return {
            "ok": True,
            "path": str(renamed_pdf),
            "name": renamed_pdf.name,
            "sha1": "sha1abc",
            "renamed": True,
        }

    def fake_quick_ingest_pdf(*, pdf_path: Path, speed_mode: str = "ultra_fast", progress_cb=None, cancel_cb=None, ingest_proc_cb=None) -> dict:
        called["ingest_path"] = str(pdf_path)
        if callable(progress_cb):
            progress_cb("converting")
            progress_cb("ingesting")
        return {"ready": True, "md_path": str(md_path)}

    monkeypatch.setattr(chat_router, "auto_rename_saved_pdf_in_library", fake_auto_rename_saved_pdf_in_library)
    monkeypatch.setattr(chat_router, "quick_ingest_pdf", fake_quick_ingest_pdf)

    job_id = chat_router._start_chat_pdf_ingest_job(
        pdf_path=src_pdf,
        speed_mode="ultra_fast",
        display_name="upload.pdf",
        sha1="sha1abc",
        conv_id="",
    )
    assert job_id

    deadline = time.time() + 2.0
    rec = None
    while time.time() < deadline:
        rec = chat_router._get_chat_pdf_ingest_job(job_id)
        if isinstance(rec, dict) and str(rec.get("ingest_status") or "") == "ready":
            break
        time.sleep(0.02)

    assert called.get("renamed") is True
    assert called.get("ingest_path") == str(renamed_pdf)
    assert isinstance(rec, dict)
    assert rec.get("name") == renamed_pdf.name
    assert rec.get("path") == str(renamed_pdf)
    assert rec.get("ingest_status") == "ready"


def test_chat_pdf_job_starts_background_quality_refine(monkeypatch, tmp_path: Path):
    from api.routers import chat as chat_router

    src_pdf = tmp_path / "upload.pdf"
    src_pdf.write_bytes(b"%PDF-1.4 test")
    md_path = tmp_path / "paper" / "paper.en.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("# quick\n", encoding="utf-8")

    monkeypatch.setattr(
        chat_router,
        "auto_rename_saved_pdf_in_library",
        lambda **kwargs: {
            "ok": True,
            "path": str(src_pdf),
            "name": src_pdf.name,
            "sha1": "sha1abc",
            "renamed": False,
        },
    )

    monkeypatch.setattr(
        chat_router,
        "quick_ingest_pdf",
        lambda **kwargs: {"ready": True, "md_path": str(md_path), "out_folder": str(tmp_path / "paper")},
    )

    called: dict[str, object] = {}

    def fake_refine_pdf_with_full_llm_replace(*, pdf_path: Path, progress_cb=None, cancel_cb=None, ingest_proc_cb=None):
        called["pdf_path"] = str(pdf_path)
        if callable(progress_cb):
            progress_cb("refining")
            progress_cb("refine_ingesting")
        return {"ready": True, "md_path": str(md_path)}

    monkeypatch.setattr(chat_router, "refine_pdf_with_full_llm_replace", fake_refine_pdf_with_full_llm_replace)

    job_id = chat_router._start_chat_pdf_ingest_job(
        pdf_path=src_pdf,
        speed_mode="ultra_fast",
        display_name=src_pdf.name,
        sha1="sha1abc",
        conv_id="",
    )
    assert job_id

    deadline = time.time() + 2.0
    rec = None
    while time.time() < deadline:
        rec = chat_router._get_chat_pdf_ingest_job(job_id)
        if isinstance(rec, dict) and str(rec.get("quality_status") or "") == "ready":
            break
        time.sleep(0.02)

    assert called.get("pdf_path") == str(src_pdf)
    assert isinstance(rec, dict)
    assert rec.get("ingest_status") == "ready"
    assert rec.get("quality_status") == "ready"
