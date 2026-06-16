from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.routers import library as library_router


def test_library_pdf_path_arg_allows_relative_pdf_inside_configured_root(monkeypatch, tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)

    resolved = library_router._resolve_library_pdf_path_arg("paper.pdf")

    assert resolved == (pdf_dir / "paper.pdf").resolve(strict=False)


def test_library_pdf_path_arg_rejects_absolute_pdf_outside_configured_root(monkeypatch, tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    outside_dir = tmp_path / "outside"
    pdf_dir.mkdir()
    outside_dir.mkdir()
    outside_pdf = outside_dir / "paper.pdf"
    outside_pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)

    with pytest.raises(HTTPException) as exc_info:
        library_router._resolve_library_pdf_path_arg(str(outside_pdf))

    assert exc_info.value.status_code == 400
    assert "PDF directory" in str(exc_info.value.detail)


def test_start_convert_rejects_traversal_before_enqueue(monkeypatch, tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md"
    outside_dir = tmp_path / "outside"
    pdf_dir.mkdir()
    md_dir.mkdir()
    outside_dir.mkdir()
    (outside_dir / "paper.pdf").write_bytes(b"%PDF-1.4")
    enqueued: list[dict] = []

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "get_settings", lambda: SimpleNamespace(db_dir=tmp_path / "db"))
    monkeypatch.setattr(library_router, "_bg_enqueue", lambda task: enqueued.append(task))

    with pytest.raises(HTTPException) as exc_info:
        library_router.start_convert(library_router.ConvertBody(pdf_name="../outside/paper.pdf", replace=False))

    assert exc_info.value.status_code == 400
    assert enqueued == []


def test_start_convert_enqueues_resolved_pdf_inside_library(monkeypatch, tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md"
    pdf_dir.mkdir()
    md_dir.mkdir()
    pdf_path = pdf_dir / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    enqueued: list[dict] = []

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "get_settings", lambda: SimpleNamespace(db_dir=tmp_path / "db"))
    monkeypatch.setattr(library_router, "_bg_enqueue", lambda task: enqueued.append(task))

    out = library_router.start_convert(library_router.ConvertBody(pdf_name="paper.pdf", replace=False))

    assert out["ok"] is True
    assert enqueued
    assert Path(enqueued[0]["pdf"]).resolve(strict=False) == pdf_path.resolve(strict=False)


@pytest.mark.parametrize("bad_name", ["../paper.pdf", "subdir/paper.pdf", r"subdir\paper.pdf", ""])
def test_library_pdf_name_arg_rejects_non_leaf_names(monkeypatch, tmp_path: Path, bad_name: str):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)

    with pytest.raises(HTTPException) as exc_info:
        library_router._resolve_library_pdf_name_arg(bad_name)

    assert exc_info.value.status_code == 400


def test_library_pdf_name_arg_can_require_existing_file(monkeypatch, tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)

    with pytest.raises(HTTPException) as exc_info:
        library_router._resolve_library_pdf_name_arg("missing.pdf", require_exists=True)

    assert exc_info.value.status_code == 404
