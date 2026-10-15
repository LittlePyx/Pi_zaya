from __future__ import annotations

import fitz

from kb.converter.quality_repair import (
    _demote_source_proven_nonheading_numbered_headings,
    _plausible_reference_citation_indices,
)


def test_noise_interval_is_not_a_bibliographic_reference():
    text = r"Train with $\sigma \in [5,55]$; noise levels are in the range of [0,55]. Compare Gaussian denoisers [42]."
    assert _plausible_reference_citation_indices(text) == {42}


def test_source_proves_equation_tail_is_body_not_numbered_heading(tmp_path):
    path = tmp_path / "source.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    for y, text in [(50, "1. Introduction"), (100, "3. Methods"), (200, "4. Theory")]:
        page.insert_text((50, y), text, fontsize=12, fontname="hebo")
    page.insert_text((50, 150), "mse squared . They repeat this, permuting", fontsize=10)
    pdf.save(path)
    pdf.close()
    text = "<!-- kb_page: 1 -->\n## 1. Introduction\n## 3. Methods\n## 2. They repeat this, permuting\n## 4. Theory\n"
    fixed, changed = _demote_source_proven_nonheading_numbered_headings(text, tmp_path / "source.en.md", path)
    assert changed
    assert "## 2. They" not in fixed
    assert "## 3. Methods" in fixed and "## 4. Theory" in fixed
