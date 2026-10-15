from __future__ import annotations

import fitz
import pytest

from kb.converter.config import ConvertConfig
from kb.converter.pipeline import PDFConverter


@pytest.mark.parametrize("lines", [
    [r"for each subset J ∈ $\mathcal{J}$."],
    [r"mean-zero noise in each pixel, we may choose $\mathcal{J}$ ="],
    [r"$\mathcal{J}$ minimizing (1)"],
    ["Definition. Let J be a partition of the dimensions",
     "{1, . . . , m} and let J ∈ J. A function f : R m → R m",
     "is J-invariant if f(x) does not depend on x J."],
    ["We model the signal y and its noisy measurement x as a pair",
     "of random variables in R m. If J ⊂ {1, . . . , m} is a subset",
     "of the dimensions, we write x J for x restricted to J."],
    ["Proposition 1. Suppose x is an unbiased estimator of y, i.e.",
     "E[x|y] = y, and the noise in each subset J ∈ J is indepen-",
     "dent from the noise in its complement J c, conditional on y.",
     "Let f be J-invariant. Then"],
])
def test_inline_mathematics_preserves_entire_condition_as_searchable_prose(tmp_path, lines):
    cfg = ConvertConfig(pdf_path=tmp_path / "input.pdf", out_dir=tmp_path,
                        translate_zh=False, start_page=0, end_page=-1,
                        skip_existing=False, keep_debug=False, llm=None)
    converter = PDFConverter(cfg)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    source_lines = [{"bbox": (55, 300 + i * 13, 291, 313 + i * 13),
                     "spans": [{"text": text, "size": 10, "font": "Times-Roman"}]}
                    for i, text in enumerate(lines)]
    blocks = converter._extract_text_blocks(
        page, page_index=1, body_size=10, tables=[], visual_rects=[],
        assets_dir=tmp_path, page_dict={"blocks": [{
            "bbox": (55, 300, 291, 300 + len(lines) * 13), "lines": source_lines}]},
    )
    assert blocks and all(not b.is_math for b in blocks)
    assert all(text in " ".join(b.text for b in blocks) for text in lines)
    doc.close()


def test_statement_begins_its_own_block_for_direct_recall(tmp_path):
    from kb.direct_question_evidence import direct_question_evidence_hits

    cfg = ConvertConfig(pdf_path=tmp_path / "input.pdf", out_dir=tmp_path,
                        translate_zh=False, skip_existing=False, llm=None,
                        start_page=0, end_page=-1, keep_debug=False)
    converter = PDFConverter(cfg)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    texts = ["We may search over all invariant functions for the global optimum:",
             r"Proposition 2. The $\mathcal{J}$-invariant function minimizing (1) satisfies",
             r"f(x)_J = E[y_J | x_{J^c}]", r"for each subset J ∈ $\mathcal{J}$."]
    lines = [{"bbox": (55, 300 + i * 15, 291, 313 + i * 15),
              "spans": [{"text": text, "size": 10, "font": "Times-Roman"}]}
             for i, text in enumerate(texts)]
    blocks = converter._extract_text_blocks(
        page, page_index=1, body_size=10, tables=[], visual_rects=[], assets_dir=tmp_path,
        page_dict={"blocks": [{"bbox": (55, 300, 291, 360), "lines": lines}]})
    source = tmp_path / "converted.md"
    source.write_text("<!-- kb_page: 2 -->\n\n" + "\n\n".join(
        "$$" + b.text + "$$" if b.is_math else b.text for b in blocks), encoding="utf8")
    hits = direct_question_evidence_hits(source, "Explain Proposition 2.")
    assert hits and hits[0]["text"].startswith("Proposition 2.")
    assert any("x_{J^c}" in h["text"] for h in hits)
    assert any("for each subset" in h["text"] for h in hits)
    assert all(h["meta"]["page_start"] == 2 for h in hits)
    assert not direct_question_evidence_hits(source, "Explain Proposition 3.")
    doc.close()
