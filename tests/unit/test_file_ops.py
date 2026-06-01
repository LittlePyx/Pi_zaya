from __future__ import annotations

from pathlib import Path

from kb.file_ops import _resolve_md_output_paths


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
