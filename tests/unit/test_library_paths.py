from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from kb.library_paths import resolve_library_paths


def test_library_paths_use_managed_defaults_when_no_override(tmp_path: Path):
    settings = SimpleNamespace(db_dir=tmp_path / "data" / "db")

    paths = resolve_library_paths(settings, {}, {})

    assert paths.pdf_dir == (tmp_path / "data" / "pdfs").resolve()
    assert paths.md_dir == (tmp_path / "data" / "md_output").resolve()
    assert paths.pdf_source == "default"
    assert paths.md_source == "default"
    assert paths.public_payload()["uses_managed_defaults"] is True


def test_library_paths_prefer_user_preferences_over_environment(tmp_path: Path):
    settings = SimpleNamespace(db_dir=tmp_path / "data" / "db")
    preferred_pdf = tmp_path / "chosen" / "pdfs"
    preferred_md = tmp_path / "chosen" / "markdown"

    paths = resolve_library_paths(
        settings,
        {"pdf_dir": str(preferred_pdf), "md_dir": str(preferred_md)},
        {"KB_PDF_DIR": str(tmp_path / "env-pdfs"), "KB_MD_DIR": str(tmp_path / "env-md")},
    )

    assert paths.pdf_dir == preferred_pdf.resolve()
    assert paths.md_dir == preferred_md.resolve()
    assert paths.pdf_source == "preference"
    assert paths.md_source == "preference"
    assert paths.public_payload()["uses_managed_defaults"] is False
