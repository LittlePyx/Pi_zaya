from __future__ import annotations

from kb.converter.pipeline import PDFConverter


def test_remap_page_image_links_by_caption_uses_nearby_caption_number():
    md = "\n".join(
        [
            "![Figure](./assets/page_5_fig_1.png)",
            "**Figure 3.** Caption for fig 3.",
            "",
            "![Figure](./assets/page_5_fig_2.png)",
            "**Figure 2.** Caption for fig 2.",
        ]
    )
    meta = {
        "page_5_fig_1.png": {"fig_no": 2},
        "page_5_fig_2.png": {"fig_no": 3},
    }

    out = PDFConverter._remap_page_image_links_by_caption(
        md,
        page_index=4,
        figure_meta_by_asset=meta,
    )

    assert "![Figure](./assets/page_5_fig_2.png)" in out.splitlines()[0]
    assert "![Figure](./assets/page_5_fig_1.png)" in out.splitlines()[3]


def test_remap_page_image_links_by_caption_uses_alt_when_available():
    md = "![Fig. 3](./assets/page_5_fig_1.png)"
    meta = {
        "page_5_fig_1.png": {"fig_no": 2},
        "page_5_fig_2.png": {"fig_no": 3},
    }

    out = PDFConverter._remap_page_image_links_by_caption(
        md,
        page_index=4,
        figure_meta_by_asset=meta,
    )

    assert out.strip() == "![Fig. 3](./assets/page_5_fig_2.png)"


def test_remap_page_image_links_by_caption_skips_ambiguous_figure_number():
    md = "![Fig. 3](./assets/page_5_fig_1.png)"
    # Duplicate fig_no in metadata -> ambiguous, should keep original link.
    meta = {
        "page_5_fig_1.png": {"fig_no": 3},
        "page_5_fig_2.png": {"fig_no": 3},
    }

    out = PDFConverter._remap_page_image_links_by_caption(
        md,
        page_index=4,
        figure_meta_by_asset=meta,
    )

    assert out.strip() == md


def test_reorder_page_figure_pairs_by_number():
    md = "\n".join(
        [
            "![Fig. 3](./assets/page_5_fig_2.png)",
            "",
            "**Figure 3.** caption three",
            "",
            "![Fig. 2](./assets/page_5_fig_1.png)",
            "",
            "**Figure 2.** caption two",
            "",
            "## Next heading",
        ]
    )
    out = PDFConverter._reorder_page_figure_pairs_by_number(md, page_index=4)
    lines = out.splitlines()
    assert lines[0].startswith("![Fig. 2]")
    assert "Figure 2" in lines[2]
    assert any(ln.startswith("![Fig. 3]") for ln in lines)
