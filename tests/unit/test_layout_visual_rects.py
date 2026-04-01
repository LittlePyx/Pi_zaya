import pytest
from types import SimpleNamespace

try:
    import fitz
except Exception:  # pragma: no cover
    fitz = None

from kb.converter.layout_analysis import _merge_nearby_visual_rects, sort_blocks_reading_order
from kb.converter.pipeline import PDFConverter
from kb.converter.models import TextBlock


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_merge_stacked_multi_panel_figure():
    page_w, page_h = 600.0, 900.0
    rects = [
        fitz.Rect(100, 120, 380, 260),  # top panel group
        fitz.Rect(130, 298, 350, 450),  # bottom panel / chart
    ]
    out = _merge_nearby_visual_rects(rects, page_w=page_w, page_h=page_h)
    assert len(out) == 1
    r = out[0]
    assert r.x0 <= 100 and r.y0 <= 120
    assert r.x1 >= 380 and r.y1 >= 450


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_do_not_merge_distant_stacked_figures():
    page_w, page_h = 600.0, 900.0
    rects = [
        fitz.Rect(100, 120, 380, 260),
        fitz.Rect(130, 430, 350, 580),  # too far below
    ]
    out = _merge_nearby_visual_rects(rects, page_w=page_w, page_h=page_h)
    assert len(out) == 2


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_do_not_merge_top_journal_banner_with_real_figure_below():
    page_w, page_h = 612.0, 792.0
    rects = [
        fitz.Rect(116.28, 30.68, 495.72, 70.12),  # journal banner
        fitz.Rect(136.02, 93.60, 476.04, 325.50),  # actual figure below
    ]
    out = _merge_nearby_visual_rects(rects, page_w=page_w, page_h=page_h)
    assert len(out) == 2


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_expanded_visual_crop_rect_keeps_more_top_room_for_figure_internal_title():
    conv = PDFConverter.__new__(PDFConverter)
    rect = fitz.Rect(100, 60, 510, 326)
    out = conv._expanded_visual_crop_rect(
        rect=rect,
        page_w=595.0,
        page_h=842.0,
        is_full_width=True,
        line_boxes=[],
    )
    assert out.y0 <= 50.5


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_expanded_visual_crop_rect_does_not_eat_body_text_above():
    conv = PDFConverter.__new__(PDFConverter)
    rect = fitz.Rect(100, 60, 510, 326)
    body_line = (fitz.Rect(110, 44, 500, 57), "This is a normal body sentence directly above the figure")
    out = conv._expanded_visual_crop_rect(
        rect=rect,
        page_w=595.0,
        page_h=842.0,
        is_full_width=True,
        line_boxes=[body_line],
    )
    assert out.y0 >= 57.0


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_expanded_visual_crop_rect_keeps_wide_bold_figure_title_above():
    conv = PDFConverter.__new__(PDFConverter)
    rect = fitz.Rect(110, 61.8, 510, 326.6)
    panel_label = (fitz.Rect(92, 49.7, 98, 60.6), "a", 10.8, True)
    figure_title = (
        fitz.Rect(148.4, 51.4, 458.4, 59.6),
        "Network and feature map comparison between SwinIR and the reported network",
        8.1,
        True,
    )
    out = conv._expanded_visual_crop_rect(
        rect=rect,
        page_w=595.0,
        page_h=842.0,
        is_full_width=False,
        line_boxes=[panel_label, figure_title],
    )
    assert out.y0 <= 50.5


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_expanded_visual_crop_rect_does_not_absorb_ragged_paragraph_tail():
    conv = PDFConverter.__new__(PDFConverter)
    rect = fitz.Rect(121.74, 168.60, 490.20, 319.92)
    ragged_tail = (
        fitz.Rect(117.36, 151.32, 306.31, 162.42),
        "SLMs commonly have 256 quantization levels",
        10.02,
        False,
    )
    out = conv._expanded_visual_crop_rect(
        rect=rect,
        page_w=612.0,
        page_h=792.0,
        is_full_width=False,
        line_boxes=[ragged_tail],
    )
    assert out.y0 >= 162.0


def test_axis_or_panel_text_does_not_treat_section_heading_as_panel():
    assert PDFConverter._looks_axis_or_panel_text("2. Comparison of theory") is False
    assert PDFConverter._looks_axis_or_panel_text("2.1 Principle of HSI and FSI") is False


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_expanded_visual_crop_rect_does_not_absorb_plain_sentence_tail():
    conv = PDFConverter.__new__(PDFConverter)
    rect = fitz.Rect(152.78, 549.66, 458.78, 651.08)
    body_tail = (
        fitz.Rect(117.38, 532.92, 433.02, 544.03),
        "reconstructed images in the presented numerical simulations is 64 64",
        10.03,
        False,
    )
    out = conv._expanded_visual_crop_rect(
        rect=rect,
        page_w=612.0,
        page_h=792.0,
        is_full_width=False,
        line_boxes=[body_tail],
    )
    assert out.y0 >= 544.0


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_split_visual_rects_by_internal_captions_breaks_stacked_figures():
    conv = PDFConverter.__new__(PDFConverter)
    page = SimpleNamespace(rect=fitz.Rect(0, 0, 595.0, 842.0))
    visual_rects = [fitz.Rect(136.14, 93.54, 475.86, 601.26)]
    caption_candidates = [
        {
            "fig_no": 8,
            "caption": "Fig. 8. Noise-robustness comparison using the Lena image.",
            "bbox": [153.35, 335.84, 460.73, 353.86],
        }
    ]
    out = conv._split_visual_rects_by_internal_captions(
        page=page,
        visual_rects=visual_rects,
        caption_candidates=caption_candidates,
    )
    assert len(out) == 2
    assert out[0].y1 <= 336.0
    assert out[1].y0 >= 353.0


def test_sort_blocks_reading_order_handles_single_large_left_column_continuation():
    page_w = 595.0
    left = TextBlock(
        bbox=(56.7, 88.9, 290.7, 732.2),
        text="realized because most of the photons are discarded. " * 4,
        max_font_size=10.0,
        is_bold=False,
    )
    right = TextBlock(
        bbox=(304.7, 85.7, 538.8, 493.2),
        text="Results Principle of interferometric ISM (iISM) " * 4,
        max_font_size=10.0,
        is_bold=False,
    )
    eq = TextBlock(
        bbox=(339.8, 517.4, 501.6, 533.1),
        text="E = mc^2",
        max_font_size=10.0,
        is_bold=False,
        is_math=True,
    )

    out = sort_blocks_reading_order([right, left, eq], page_width=page_w)
    assert out[0].text == left.text
    assert out[1].text == right.text


def test_sort_blocks_reading_order_keeps_left_column_before_right_when_left_spills_into_gutter():
    page_w = 595.0
    header = TextBlock(
        bbox=(56.7, 33.6, 538.6, 42.6),
        text="Kuppers and Moerner Light: Science & Applications (2026) 15:129 Page 2 of 13",
        max_font_size=9.0,
        is_bold=False,
    )
    left = TextBlock(
        bbox=(56.7, 88.9, 290.7, 732.2),
        text="realized because most of the photons are discarded. " * 6,
        max_font_size=10.0,
        is_bold=False,
    )
    right = TextBlock(
        bbox=(304.7, 85.7, 538.8, 493.2),
        text="Results Principle of interferometric ISM (iISM) " * 6,
        max_font_size=10.0,
        is_bold=False,
    )
    eq = TextBlock(
        bbox=(339.8, 517.4, 501.6, 533.1),
        text="E = mc^2",
        max_font_size=10.0,
        is_bold=False,
        is_math=True,
    )

    out = sort_blocks_reading_order([header, right, left, eq], page_width=page_w)
    assert out[0].text == header.text
    assert out[1].text == left.text
    assert out[2].text == right.text
