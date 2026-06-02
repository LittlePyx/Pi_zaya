from pathlib import Path

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
