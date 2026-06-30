from __future__ import annotations

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


def test_library_pdf_path_arg_rejects_sibling_prefix(monkeypatch, tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    sibling_dir = tmp_path / "pdfs-extra"
    pdf_dir.mkdir()
    sibling_dir.mkdir()
    outside_pdf = sibling_dir / "paper.pdf"
    outside_pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)

    with pytest.raises(HTTPException) as exc_info:
        library_router._resolve_library_pdf_path_arg(str(outside_pdf))

    assert exc_info.value.status_code == 400
    assert "PDF directory" in str(exc_info.value.detail)


def test_library_pdf_name_arg_rejects_symlink_escape(monkeypatch, tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    outside_dir = tmp_path / "outside"
    pdf_dir.mkdir()
    outside_dir.mkdir()
    outside_pdf = outside_dir / "paper.pdf"
    outside_pdf.write_bytes(b"%PDF-1.4")
    link = pdf_dir / "linked.pdf"
    try:
        link.symlink_to(outside_pdf)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable in this environment: {exc}")
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)

    with pytest.raises(HTTPException) as exc_info:
        library_router._resolve_library_pdf_name_arg("linked.pdf", require_exists=True)

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


def test_save_pdf_to_library_ignores_same_sha1_record_outside_pdf_root(monkeypatch, tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    outside_dir = tmp_path / "outside"
    pdf_dir.mkdir()
    md_dir.mkdir()
    outside_dir.mkdir()
    data = b"%PDF-1.4 same bytes"
    outside_pdf = outside_dir / "external.pdf"
    outside_pdf.write_bytes(data)

    class FakeStore:
        def __init__(self) -> None:
            self.upserts: list[tuple[str, Path]] = []

        def get_by_sha1(self, _sha1: str) -> dict:
            return {"path": str(outside_pdf)}

        def upsert(self, sha1: str, path: Path, citation_meta: dict | None = None) -> None:
            self.upserts.append((sha1, Path(path)))

    fake_store = FakeStore()
    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: fake_store)
    monkeypatch.setattr(library_router, "extract_pdf_meta_suggestion", lambda *args, **kwargs: library_router.PdfMetaSuggestion())

    result = library_router.save_pdf_to_library(file_name="paper.pdf", data=data, base_name="paper")

    assert result["duplicate"] is False
    assert Path(result["path"]).parent == pdf_dir
    assert Path(result["path"]).name == "paper.pdf"
    assert fake_store.upserts


def test_existing_pdf_record_ignores_scan_result_outside_pdf_root(monkeypatch, tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    outside_dir = tmp_path / "outside"
    pdf_dir.mkdir()
    outside_dir.mkdir()
    data = b"%PDF-1.4 same bytes"
    outside_pdf = outside_dir / "external.pdf"
    outside_pdf.write_bytes(data)

    class EmptyStore:
        def get_by_sha1(self, _sha1: str) -> None:
            return None

    monkeypatch.setattr(library_router, "_list_pdf_paths_fast", lambda _pdf_dir: [outside_pdf])

    result = library_router._existing_pdf_record(
        pdf_dir,
        library_router._sha1_bytes(data),
        lib_store=EmptyStore(),
    )

    assert result is None


def test_auto_rename_saved_pdf_rejects_path_outside_pdf_root(monkeypatch, tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    outside_dir = tmp_path / "outside"
    pdf_dir.mkdir()
    md_dir.mkdir()
    outside_dir.mkdir()
    outside_pdf = outside_dir / "external.pdf"
    outside_pdf.write_bytes(b"%PDF-1.4 outside")

    class FakeStore:
        def upsert(self, *args, **kwargs) -> None:
            raise AssertionError("library store must not be updated for outside paths")

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "_library_store", lambda: FakeStore())

    result = library_router.auto_rename_saved_pdf_in_library(pdf_path=outside_pdf, use_llm=False, also_md=True)

    assert result["ok"] is False
    assert result["error"] == "pdf must be within the configured PDF directory"
    assert outside_pdf.exists()


def test_open_library_file_rejects_markdown_symlink_escape(monkeypatch, tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    outside_dir = tmp_path / "outside_md"
    pdf_dir.mkdir()
    md_dir.mkdir()
    outside_dir.mkdir()
    (pdf_dir / "paper.pdf").write_bytes(b"%PDF-1.4")
    (outside_dir / "paper.en.md").write_text("# outside", encoding="utf-8")
    link = md_dir / "paper"
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable in this environment: {exc}")
    opened: list[Path] = []

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(library_router, "open_in_explorer", lambda path: opened.append(Path(path)))

    with pytest.raises(HTTPException) as exc_info:
        library_router.open_library_file(library_router.OpenLibraryFileBody(pdf_name="paper.pdf", target="md"))

    assert exc_info.value.status_code == 400
    assert "Markdown directory" in str(exc_info.value.detail)
    assert opened == []


def test_guide_source_rejects_markdown_symlink_escape(monkeypatch, tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    md_dir = tmp_path / "md_output"
    outside_dir = tmp_path / "outside_md"
    pdf_dir.mkdir()
    md_dir.mkdir()
    outside_dir.mkdir()
    (pdf_dir / "paper.pdf").write_bytes(b"%PDF-1.4")
    (outside_dir / "paper.en.md").write_text("# outside", encoding="utf-8")
    link = md_dir / "paper"
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable in this environment: {exc}")

    monkeypatch.setattr(library_router, "_pdf_dir", lambda: pdf_dir)
    monkeypatch.setattr(library_router, "_md_dir", lambda: md_dir)

    with pytest.raises(HTTPException) as exc_info:
        library_router.resolve_library_guide_source(library_router.GuideSourceBody(pdf_name="paper.pdf"))

    assert exc_info.value.status_code == 400
    assert "markdown" in str(exc_info.value.detail).lower()


def test_sync_md_after_pdf_rename_rejects_markdown_symlink_escape(tmp_path: Path):
    md_dir = tmp_path / "md_output"
    outside_dir = tmp_path / "outside_md"
    md_dir.mkdir()
    outside_dir.mkdir()
    (outside_dir / "paper.en.md").write_text("# outside", encoding="utf-8")
    link = md_dir / "paper"
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable in this environment: {exc}")

    result = library_router._sync_md_after_pdf_rename_basic(
        md_root=md_dir,
        src_pdf=Path("paper.pdf"),
        dest_pdf=Path("renamed.pdf"),
    )

    assert result["ok"] is False
    assert "Markdown directory" in str(result["msg"])
    assert (outside_dir / "paper.en.md").is_file()
    assert not (outside_dir / "renamed.en.md").exists()
