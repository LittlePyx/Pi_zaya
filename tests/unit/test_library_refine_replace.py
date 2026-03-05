from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_refine_pdf_with_full_llm_replace_success(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")

    md_root = tmp_path / "md_output"
    target_folder = md_root / "paper"
    target_folder.mkdir(parents=True, exist_ok=True)
    old_md = target_folder / "paper.en.md"
    old_md.write_text("# old\n", encoding="utf-8")

    monkeypatch.setattr(library_router, "get_settings", lambda: SimpleNamespace(db_dir=str(tmp_path / "db")))
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_root)

    def fake_run_pdf_to_md(**kwargs):
        out_root = Path(kwargs["out_root"])
        out_folder = out_root / pdf.stem
        out_folder.mkdir(parents=True, exist_ok=True)
        (out_folder / f"{pdf.stem}.en.md").write_text("# refined\n", encoding="utf-8")
        return True, str(out_folder)

    called: dict[str, str] = {}

    def fake_ingest_markdown_incremental(*, md_main: Path, db_dir: Path, cancel_cb=None, ingest_proc_cb=None):
        called["md_main"] = str(md_main)
        called["db_dir"] = str(db_dir)
        return {"ready": True}

    monkeypatch.setattr(library_router, "run_pdf_to_md", fake_run_pdf_to_md)
    monkeypatch.setattr(library_router, "_ingest_markdown_incremental", fake_ingest_markdown_incremental)

    result = library_router.refine_pdf_with_full_llm_replace(pdf_path=pdf)

    assert result["ready"] is True
    assert old_md.read_text(encoding="utf-8").strip() == "# refined"
    assert called["md_main"] == str(old_md)


def test_refine_pdf_with_full_llm_replace_rolls_back_on_ingest_error(monkeypatch, tmp_path: Path):
    from api.routers import library as library_router

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")

    md_root = tmp_path / "md_output"
    target_folder = md_root / "paper"
    target_folder.mkdir(parents=True, exist_ok=True)
    old_md = target_folder / "paper.en.md"
    old_md.write_text("# old\n", encoding="utf-8")

    monkeypatch.setattr(library_router, "get_settings", lambda: SimpleNamespace(db_dir=str(tmp_path / "db")))
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_root)

    def fake_run_pdf_to_md(**kwargs):
        out_root = Path(kwargs["out_root"])
        out_folder = out_root / pdf.stem
        out_folder.mkdir(parents=True, exist_ok=True)
        (out_folder / f"{pdf.stem}.en.md").write_text("# refined\n", encoding="utf-8")
        return True, str(out_folder)

    monkeypatch.setattr(library_router, "run_pdf_to_md", fake_run_pdf_to_md)
    monkeypatch.setattr(
        library_router,
        "_ingest_markdown_incremental",
        lambda **kwargs: {"ready": False, "error": "ingest failed"},
    )

    result = library_router.refine_pdf_with_full_llm_replace(pdf_path=pdf)

    assert result["ready"] is False
    assert "ingest failed" in str(result.get("error") or "")
    assert old_md.read_text(encoding="utf-8").strip() == "# old"
