from __future__ import annotations

from pathlib import Path

import fitz

from kb import pdf_tools


def _write_text_native_paper(path: Path) -> None:
    doc = fitz.open()
    pages = [
        [
            (72, 72, "Traceable Evidence Retrieval", 20),
            (72, 112, "Abstract", 14),
            (72, 140, "We evaluate a source-grounded literature retrieval workflow.", 11),
            (72, 180, "1 Introduction", 14),
            (72, 208, "Readers need page-addressable evidence from papers in their library.", 11),
        ],
        [
            (72, 72, "2 Methods", 14),
            (72, 104, "The system preserves page anchors and compares retrieved evidence.", 11),
            (72, 150, "3 Results", 14),
            (72, 182, "Page-aware retrieval improved evidence location accuracy.", 11),
        ],
        [
            (72, 72, "4 Limitations", 14),
            (72, 104, "Scanned documents still require a visual extraction provider.", 11),
            (72, 150, "References", 14),
            (72, 182, "[1] A. Author. Evidence retrieval systems. 2024.", 11),
            (72, 208, "[2] B. Author. Page-aware document reading. 2025.", 11),
        ],
    ]
    for rows in pages:
        page = doc.new_page(width=595, height=842)
        for x, y, text, size in rows:
            page.insert_text((x, y), text, fontsize=size)
    doc.set_metadata({"title": "Traceable Evidence Retrieval"})
    doc.save(path)
    doc.close()


def test_text_fallback_preserves_page_and_structure_contract(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    out_root = tmp_path / "markdown"
    _write_text_native_paper(pdf_path)
    monkeypatch.setenv("KB_PDF_CONVERTER", str(tmp_path / "missing-converter.py"))

    ok, output = pdf_tools.run_pdf_to_md(
        pdf_path=pdf_path,
        out_root=out_root,
        no_llm=True,
        keep_debug=True,
        eq_image_fallback=False,
    )

    assert ok is True
    md_path = Path(output) / "paper.en.md"
    markdown = md_path.read_text(encoding="utf-8")
    assert markdown.count("<!-- kb_page:") == 3
    assert "<!-- kb_page: 1 -->" in markdown
    assert "<!-- kb_page: 2 -->" in markdown
    assert "<!-- kb_page: 3 -->" in markdown
    assert "## Abstract" in markdown
    assert "# 1 Introduction" in markdown or "## 1 Introduction" in markdown
    assert "References" in markdown
    assert "[1] A. Author" in markdown
    assert "\n\n---\n\n" not in markdown
