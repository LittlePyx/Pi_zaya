from __future__ import annotations

import pytest

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


def test_remap_page_image_links_by_caption_accepts_markdown_styled_caption_line():
    md = "\n".join(
        [
            "![Figure](./assets/page_5_fig_1.png)",
            "",
            "> **Figure 3.** Landscape result.",
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

    assert out.splitlines()[0].strip() == "![Figure](./assets/page_5_fig_2.png)"


def test_remap_page_image_links_by_caption_allows_one_non_caption_line_before_caption():
    md = "\n".join(
        [
            "![Figure](./assets/page_5_fig_1.png)",
            "This figure compares reconstruction quality.",
            "**Figure 3.** Caption for fig 3.",
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

    assert out.splitlines()[0].strip() == "![Figure](./assets/page_5_fig_2.png)"


def test_remap_page_image_links_by_caption_natphoton_style_swapped_links():
    md = "\n".join(
        [
            "![Fig. 2](./assets/page_5_fig_2.png)",
            "",
            "**Figure 2.** | Evaluating regularized and non-regularized image reconstructions.",
            "",
            "![Fig. 3](./assets/page_5_fig_1.png)",
            "",
            "**Figure 3.** | The experimental landscape for single-pixel imaging systems.",
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

    lines = out.splitlines()
    assert lines[0].strip() == "![Fig. 2](./assets/page_5_fig_1.png)"
    assert lines[4].strip() == "![Fig. 3](./assets/page_5_fig_2.png)"


@pytest.mark.parametrize(
    "paper_style,caption_line",
    [
        ("icip_style", "Figure 3: Multi-scale compressive reconstruction."),
        ("cvpr_style", "Fig. 3 - Snapshot compressive imaging pipeline."),
        ("nature_style", "[Figure 3] Experimental landscape and benchmarks."),
        ("ieee_style", "> Fig. 3. Recovered geometry and rendering quality."),
        (
            "cvpr_2024_style",
            "*Figure 3.** Given a single snapshot compressed image, recover the 3D scene.",
        ),
        (
            "nature_2025_style",
            "Fig. 3|Frequency response of the electrically driven dual-cavity perovskite laser.",
        ),
        (
            "natcommun_2021_style",
            "Fig. 3a Reconstruction of holographic images for the piece of mouse tail.",
        ),
        (
            "optica_2016_style",
            "Fig. 3. (a) LIA time domain data for steady BPSK states.",
        ),
    ],
)
def test_remap_page_image_links_by_caption_supports_multiple_paper_styles(
    paper_style: str,
    caption_line: str,
):
    md = "\n".join(
        [
            "![Figure](./assets/page_8_fig_1.png)",
            caption_line,
        ]
    )
    meta = {
        "page_8_fig_1.png": {"fig_no": 2},
        "page_8_fig_2.png": {"fig_no": 3},
    }

    out = PDFConverter._remap_page_image_links_by_caption(
        md,
        page_index=7,
        figure_meta_by_asset=meta,
    )

    first = out.splitlines()[0].strip()
    assert first == "![Figure](./assets/page_8_fig_2.png)", paper_style


def test_remap_page_image_links_by_caption_supports_subfigure_suffix_in_alt():
    md = "![Fig. 3a](./assets/page_8_fig_1.png)"
    meta = {
        "page_8_fig_1.png": {"fig_no": 2},
        "page_8_fig_2.png": {"fig_no": 3},
    }

    out = PDFConverter._remap_page_image_links_by_caption(
        md,
        page_index=7,
        figure_meta_by_asset=meta,
    )

    assert out.strip() == "![Fig. 3a](./assets/page_8_fig_2.png)"


def test_remap_page_image_links_by_caption_supports_subfigure_suffix_in_caption():
    md = "\n".join(
        [
            "![Figure](./assets/page_8_fig_1.png)",
            "Fig. 3a Reconstruction details.",
        ]
    )
    meta = {
        "page_8_fig_1.png": {"fig_no": 2},
        "page_8_fig_2.png": {"fig_no": 3},
    }

    out = PDFConverter._remap_page_image_links_by_caption(
        md,
        page_index=7,
        figure_meta_by_asset=meta,
    )

    assert out.splitlines()[0].strip() == "![Figure](./assets/page_8_fig_2.png)"


def test_remap_page_image_links_by_caption_supports_caption_above_image():
    md = "\n".join(
        [
            "**Figure 3.** Caption appears above image.",
            "![Figure](./assets/page_6_fig_1.png)",
        ]
    )
    meta = {
        "page_6_fig_1.png": {"fig_no": 2},
        "page_6_fig_2.png": {"fig_no": 3},
    }

    out = PDFConverter._remap_page_image_links_by_caption(
        md,
        page_index=5,
        figure_meta_by_asset=meta,
    )

    assert out.splitlines()[1].strip() == "![Figure](./assets/page_6_fig_2.png)"


def test_remap_page_image_links_by_caption_stops_at_heading_boundary():
    md = "\n".join(
        [
            "![Figure](./assets/page_6_fig_1.png)",
            "",
            "## Figure 3. This is a section heading, not a caption.",
        ]
    )
    meta = {
        "page_6_fig_1.png": {"fig_no": 2},
        "page_6_fig_2.png": {"fig_no": 3},
    }

    out = PDFConverter._remap_page_image_links_by_caption(
        md,
        page_index=5,
        figure_meta_by_asset=meta,
    )

    # Heading-like lines should not be treated as captions for remap.
    assert out.splitlines()[0].strip() == "![Figure](./assets/page_6_fig_1.png)"


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


def test_reorder_page_figure_pairs_by_number_supports_subfigure_suffix():
    md = "\n".join(
        [
            "![Fig. 3a](./assets/page_5_fig_2.png)",
            "",
            "**Figure 3a.** caption three-a",
            "",
            "![Fig. 2](./assets/page_5_fig_1.png)",
            "",
            "**Figure 2.** caption two",
        ]
    )
    out = PDFConverter._reorder_page_figure_pairs_by_number(md, page_index=4)
    lines = out.splitlines()
    assert lines[0].startswith("![Fig. 2]")
    assert any("Figure 3a" in ln for ln in lines)
