from __future__ import annotations

from pathlib import Path

from kb.file_ops import _next_pdf_dest_path, _resolve_md_output_paths, _write_tmp_upload


def test_resolve_md_output_paths_prefers_canonical_markdown_over_legacy_output(tmp_path: Path) -> None:
    pdf = tmp_path / "pdfs" / "Paper.pdf"
    out_root = tmp_path / "md"
    folder = out_root / "Paper"
    pdf.parent.mkdir()
    folder.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4")
    canonical = folder / "Paper.en.md"
    legacy_output = folder / "output.md"
    canonical.write_text("# Canonical\n", encoding="utf-8")
    legacy_output.write_text("# Legacy output\n", encoding="utf-8")

    _, md_main, exists = _resolve_md_output_paths(out_root, pdf)

    assert exists is True
    assert md_main == canonical


def test_next_pdf_dest_path_sanitizes_base_name_inside_pdf_dir(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    dest = _next_pdf_dest_path(pdf_dir, "../CON")

    assert dest.parent == pdf_dir
    assert dest.name == "CON-paper.pdf"
    assert dest.resolve(strict=False).parent == pdf_dir.resolve(strict=False)


def test_next_pdf_dest_path_suffixes_sanitized_collisions(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "bad-name.pdf").write_bytes(b"%PDF-1.4")

    dest = _next_pdf_dest_path(pdf_dir, "bad:name")

    assert dest.parent == pdf_dir
    assert dest.name == "bad-name-2.pdf"


def test_write_tmp_upload_sanitizes_path_like_names_inside_pdf_dir(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    tmp = _write_tmp_upload(pdf_dir, "../bad:name?.pdf", b"%PDF-1.4")

    assert tmp.parent == pdf_dir
    assert tmp.name.startswith("__upload__bad-name-")
    assert tmp.suffix == ".pdf"
