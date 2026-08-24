import pytest
from types import SimpleNamespace

try:
    import fitz
except Exception:  # pragma: no cover
    fitz = None

from kb.converter.layout_analysis import (
    _collect_visual_rects,
    _collect_image_rects,
    _merge_nearby_visual_rects,
    _looks_like_running_header_image_rect,
    page_has_full_page_image_layer,
    sort_blocks_reading_order,
)
from kb.converter.page_figure_metadata import (
    expand_visual_crop_for_intersecting_caption,
    extract_page_figure_caption_candidates,
    infer_visual_rects_from_caption_candidates,
    match_figure_entries_with_captions,
)
from kb.converter.pipeline import PDFConverter
from kb.converter.models import TextBlock


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_collect_image_rects_filters_full_page_scan_background_with_ocr_text():
    class _Page:
        rect = fitz.Rect(0, 0, 400, 600)

        def get_image_info(self):
            return [
                {"bbox": (0, -1, 401, 600), "width": 1600, "height": 2400},
                {"bbox": (90, 140, 260, 280), "width": 600, "height": 500},
            ]

        def get_text(self, mode):
            assert mode == "text"
            return "OCR text layer " * 30

    page = _Page()

    assert page_has_full_page_image_layer(page) is True
    rects = _collect_image_rects(page)
    assert len(rects) == 1
    assert tuple(round(float(v), 1) for v in rects[0]) == (90.0, 140.0, 260.0, 280.0)


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_collect_image_rects_keeps_full_page_image_without_text_layer():
    class _Page:
        rect = fitz.Rect(0, 0, 400, 600)

        def get_image_info(self):
            return [{"bbox": (0, 0, 400, 600), "width": 1600, "height": 2400}]

        def get_text(self, mode):
            assert mode == "text"
            return ""

    page = _Page()

    assert page_has_full_page_image_layer(page) is False
    rects = _collect_image_rects(page)
    assert len(rects) == 1
    assert tuple(round(float(v), 1) for v in rects[0]) == (0.0, 0.0, 400.0, 600.0)


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_collect_image_rects_filters_top_journal_masthead_banner():
    class _Page:
        rect = fitz.Rect(0, 0, 612, 792)

        def get_image_info(self):
            return [
                {"bbox": (117.0, 69.8, 495.0, 90.2), "width": 1590, "height": 86},
                {"bbox": (192.6, 392.5, 419.4, 566.1), "width": 3216, "height": 2462},
            ]

        def get_text(self, mode):
            assert mode == "text"
            return "Optics Express article text " * 40

    rects = _collect_image_rects(_Page())

    assert len(rects) == 1
    assert tuple(round(float(v), 1) for v in rects[0]) == (192.6, 392.5, 419.4, 566.1)


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_collect_visual_rects_ignores_white_vector_figure_background():
    class _Page:
        rect = fitz.Rect(0, 0, 612, 792)

        def get_drawings(self):
            return [
                {
                    "rect": fitz.Rect(275.78, 43.24, 584.54, 216.91),
                    "type": "f",
                    "fill": (1.0, 1.0, 1.0),
                    "color": None,
                    "items": [("re", fitz.Rect(275.78, 43.24, 584.54, 216.91), 1)],
                },
                {
                    "rect": fitz.Rect(363.52, 72.71, 478.25, 187.44),
                    "type": "s",
                    "fill": None,
                    "color": (0.57, 0.57, 0.57),
                    "items": [("c",)],
                },
            ]

    rects = _collect_visual_rects(_Page(), image_rects=[])

    assert len(rects) == 1
    assert tuple(round(float(v), 2) for v in rects[0]) == (363.52, 72.71, 478.25, 187.44)


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_running_header_filter_keeps_taller_top_figure():
    rect = fitz.Rect(90, 62, 520, 155)

    assert _looks_like_running_header_image_rect(rect, page_w=612.0, page_h=792.0) is False


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_infer_visual_rects_from_caption_candidates_crops_above_caption_in_same_column():
    class _Page:
        rect = fitz.Rect(0, 0, 410, 625)

        def get_text(self, mode):
            assert mode == "dict"
            return {
                "blocks": [
                    {
                        "bbox": (205, 398, 370, 438),
                        "lines": [
                            {
                                "bbox": (205, 424, 370, 438),
                                "spans": [{"text": "which Shannon has studied the"}],
                            }
                        ],
                    },
                    {
                        "bbox": (211, 566, 355, 578),
                        "lines": [
                            {
                                "bbox": (211, 566, 355, 578),
                                "spans": [{"text": "FIG. 1. Illustration of redundant visual stimulation"}],
                            }
                        ],
                    },
                ]
            }

    cap = {
        "fig_no": 1,
        "fig_ident": "1",
        "caption": "FIG. 1. Illustration of redundant visual stimulation",
        "bbox": [211, 566, 355, 578],
    }

    out = infer_visual_rects_from_caption_candidates(_Page(), [cap])

    assert len(out) == 1
    rect = out[0]
    assert rect.x0 >= 190
    assert rect.x1 <= 410
    assert 438 < rect.y0 < 470
    assert rect.y1 < 566
    assert (rect.width * rect.height) < (410 * 625 * 0.25)


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_extract_caption_candidates_accepts_old_ocr_fie_prefix():
    class _Page:
        def get_text(self, mode):
            assert mode == "dict"
            return {
                "blocks": [
                    {
                        "bbox": (20, 210, 187, 258),
                        "lines": [
                            {
                                "bbox": (20, 210, 187, 222),
                                "spans": [{"text": "Fie. 2. Subjects attempted to approximate"}],
                            },
                            {
                                "bbox": (20, 224, 187, 236),
                                "spans": [{"text": "the closed figure shown above."}],
                            },
                        ],
                    }
                ]
            }

    captions = extract_page_figure_caption_candidates(_Page())

    assert len(captions) == 1
    assert captions[0]["fig_no"] == 2
    assert captions[0]["caption"].startswith("Fie. 2.")


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_extract_caption_candidates_finds_caption_after_vector_labels_in_same_block():
    class _Page:
        def get_text(self, mode):
            assert mode == "dict"
            return {
                "blocks": [
                    {
                        "bbox": (365, 126, 504, 223),
                        "lines": [
                            {"bbox": (475, 126, 503, 141), "spans": [{"text": "Rotate"}]},
                            {"bbox": (365, 184, 504, 194), "spans": [{"text": "Figure 3: To sample a conditioned"}]},
                            {"bbox": (365, 194, 504, 204), "spans": [{"text": "process, pick up a trajectory."}]},
                        ],
                    }
                ]
            }

    captions = extract_page_figure_caption_candidates(_Page())

    assert len(captions) == 1
    assert captions[0]["fig_no"] == 3
    assert captions[0]["caption"].startswith("Figure 3: To sample")
    assert captions[0]["bbox"] == [365.0, 184.0, 504.0, 204.0]


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_caption_inferred_visual_rect_tightens_to_rendered_ink_bounds():
    doc = fitz.open()
    page = doc.new_page(width=410, height=625)
    page.insert_text((210, 430), "The paragraph ends before the figure.", fontsize=10)
    page.draw_rect(fitz.Rect(246, 472, 338, 526), color=(0, 0, 0), width=2)
    page.draw_line(fitz.Point(246, 526), fitz.Point(338, 472), color=(0, 0, 0), width=2)
    page.insert_text((216, 566), "FIG. 1. A compact test figure.", fontsize=9)

    captions = extract_page_figure_caption_candidates(page)
    out = infer_visual_rects_from_caption_candidates(page, captions)
    doc.close()

    assert len(out) == 1
    rect = out[0]
    assert 230 <= rect.x0 <= 250
    assert 335 <= rect.x1 <= 355
    assert 455 <= rect.y0 <= 475
    assert 525 <= rect.y1 <= 545


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_intersecting_caption_is_fully_included_instead_of_cut_mid_line():
    visual = fitz.Rect(363.52, 72.71, 478.25, 187.44)
    crop = fitz.Rect(361.08, 61.62, 505.46, 193.30)
    caption = {
        "bbox": [365.40, 183.53, 504.00, 223.21],
        "caption": "Figure 3: To sample a conditioned process.",
    }

    out = expand_visual_crop_for_intersecting_caption(
        crop,
        visual_rect=visual,
        caption_candidates=[caption],
        page_w=612.0,
        page_h=792.0,
    )

    assert out.y1 >= 227.0
    assert out.x1 >= 507.0


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
def test_caption_matching_uses_visual_bbox_when_crop_contains_caption():
    page = SimpleNamespace(rect=fitz.Rect(0, 0, 612, 792))
    entries = [
        {
            "asset_name": "page_7_fig_1.png",
            "bbox": [363.52, 72.71, 478.25, 187.44],
            "crop_bbox": [361.08, 61.62, 507.06, 227.96],
        }
    ]
    captions = [
        {
            "fig_no": 3,
            "fig_ident": "3",
            "caption": "Figure 3: To sample a conditioned process.",
            "bbox": [365.40, 183.53, 504.00, 223.21],
        }
    ]

    matched = match_figure_entries_with_captions(
        page=page,
        figure_entries=entries,
        caption_candidates=captions,
    )

    assert matched[0]["fig_no"] == 3
    assert matched[0]["caption"].startswith("Figure 3:")


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
def test_expanded_visual_crop_rect_keeps_short_labels_beside_vector_plot():
    conv = PDFConverter.__new__(PDFConverter)
    rect = fitz.Rect(363.52, 72.71, 478.25, 187.44)
    line_boxes = [
        (fitz.Rect(475.43, 126.76, 503.01, 140.64), "Rotate", 10.0, False),
        (fitz.Rect(471.13, 151.08, 501.81, 162.77), "Desired", 9.0, False),
        (fitz.Rect(108.0, 126.55, 356.68, 140.18), "long adjacent body sentence that must stay out", 9.0, False),
        (fitz.Rect(365.4, 183.53, 504.0, 193.59), "Figure 3: To sample a conditioned", 9.0, False),
    ]

    out = conv._expanded_visual_crop_rect(
        rect=rect,
        page_w=612.0,
        page_h=792.0,
        is_full_width=False,
        line_boxes=line_boxes,
    )

    assert out.x0 > 350.0
    assert out.x1 >= 505.0


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


def test_sort_blocks_reading_order_keeps_upper_right_abstract_before_lower_two_column_intro():
    page_w = 595.0
    title = TextBlock(
        bbox=(39.7, 113.5, 553.8, 167.3),
        text="High-resolution single-photon imaging with physics-informed deep learning",
        max_font_size=17.0,
        is_bold=True,
    )
    abstract = TextBlock(
        bbox=(217.3, 270.4, 561.3, 545.4),
        text="High-resolution single-photon imaging remains a big challenge. " * 5,
        max_font_size=9.0,
        is_bold=False,
    )
    intro_left = TextBlock(
        bbox=(39.7, 573.8, 294.9, 678.5),
        text="Single-photon avalanche diode (SPAD) array has received wide attention. " * 4,
        max_font_size=8.2,
        is_bold=False,
    )
    intro_right = TextBlock(
        bbox=(306.1, 573.8, 561.4, 657.0),
        text="While early SPAD arrays were limited in imaging resolution. " * 4,
        max_font_size=8.2,
        is_bold=False,
    )

    out = sort_blocks_reading_order([intro_right, abstract, title, intro_left], page_width=page_w)

    assert [b.text for b in out] == [
        title.text,
        abstract.text,
        intro_left.text,
        intro_right.text,
    ]
