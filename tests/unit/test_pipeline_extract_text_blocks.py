from pathlib import Path

import fitz

from kb.converter.config import ConvertConfig
from kb.converter.pipeline import PDFConverter
import kb.converter.page_text_blocks as page_text_blocks


def _make_converter(tmp_path):
    cfg = ConvertConfig(
        pdf_path=tmp_path / "dummy.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None,
    )
    return PDFConverter(cfg)


class _DummyPixmap:
    def save(self, path: Path) -> None:
        path.write_bytes(b"x" * 512)


class _DummyPage:
    def __init__(self):
        self.rect = fitz.Rect(0, 0, 200, 300)

    def get_text(self, mode: str):
        assert mode == "dict"
        return {
            "blocks": [
                {
                    "bbox": (10, 20, 120, 40),
                    "lines": [
                        {
                            "bbox": (10, 20, 120, 40),
                            "spans": [{"text": "Body paragraph.", "size": 10.0, "font": "Times-Roman"}],
                        }
                    ],
                },
                {
                    "bbox": (10, 170, 120, 210),
                    "lines": [
                        {
                            "bbox": (10, 170, 120, 185),
                            "spans": [{"text": "Figure 1. Caption text.", "size": 9.0, "font": "Times-Roman"}],
                        }
                    ],
                },
            ]
        }

    def get_pixmap(self, *, clip, dpi):
        assert dpi == 200
        return _DummyPixmap()


class _DummyPageWithFigureLabel(_DummyPage):
    def get_text(self, mode: str):
        assert mode == "dict"
        return {
            "blocks": [
                {
                    "bbox": (10, 20, 120, 40),
                    "lines": [
                        {
                            "bbox": (10, 20, 120, 40),
                            "spans": [{"text": "Body paragraph.", "size": 10.0, "font": "Times-Roman"}],
                        }
                    ],
                },
                {
                    "bbox": (6, 86, 18, 98),
                    "lines": [
                        {
                            "bbox": (6, 86, 18, 98),
                            "spans": [{"text": "a", "size": 9.0, "font": "Times-Roman"}],
                        }
                    ],
                },
                {
                    "bbox": (10, 170, 120, 210),
                    "lines": [
                        {
                            "bbox": (10, 170, 120, 185),
                            "spans": [{"text": "Figure 1. Caption text.", "size": 9.0, "font": "Times-Roman"}],
                        }
                    ],
                },
            ]
        }


class _DummyPageWithRunningHeader(_DummyPage):
    def get_text(self, mode: str):
        assert mode == "dict"
        return {
            "blocks": [
                {
                    "bbox": (10, 4, 190, 16),
                    "lines": [
                        {
                            "bbox": (10, 4, 190, 16),
                            "spans": [{"text": "Kuppers and Moerner Light: Science & Applications (2026) 15:129 Page 3 of 13", "size": 8.0, "font": "Times-Roman"}],
                        }
                    ],
                },
                {
                    "bbox": (10, 20, 120, 40),
                    "lines": [
                        {
                            "bbox": (10, 20, 120, 40),
                            "spans": [{"text": "Body paragraph.", "size": 10.0, "font": "Times-Roman"}],
                        }
                    ],
                },
            ]
        }


class _DummyPageWithCompactRunningHeader(_DummyPage):
    def get_text(self, mode: str):
        assert mode == "dict"
        return {
            "blocks": [
                {
                    "bbox": (40, 4, 135, 16),
                    "lines": [
                        {
                            "bbox": (40, 4, 135, 16),
                            "spans": [{"text": "Light: Science & Applications (2026) 15:129 Page 3 of 13", "size": 8.0, "font": "Times-Roman"}],
                        }
                    ],
                },
                {
                    "bbox": (10, 20, 120, 40),
                    "lines": [
                        {
                            "bbox": (10, 20, 120, 40),
                            "spans": [{"text": "Body paragraph.", "size": 10.0, "font": "Times-Roman"}],
                        }
                    ],
                },
            ]
        }


class _DummyPageWithFooterNumber(_DummyPage):
    def get_text(self, mode: str):
        assert mode == "dict"
        return {
            "blocks": [
                {
                    "bbox": (10, 20, 120, 40),
                    "lines": [
                        {
                            "bbox": (10, 20, 120, 40),
                            "spans": [{"text": "Body paragraph.", "size": 10.0, "font": "Times-Roman"}],
                        }
                    ],
                },
                {
                    "bbox": (96, 284, 104, 294),
                    "lines": [
                        {
                            "bbox": (96, 284, 104, 294),
                            "spans": [{"text": "5", "size": 9.0, "font": "Times-Roman"}],
                        }
                    ],
                },
            ]
        }


class _DummyPageWithSectionHeading(_DummyPage):
    def get_text(self, mode: str):
        assert mode == "dict"
        return {
            "blocks": [
                {
                    "bbox": (10, 20, 160, 40),
                    "lines": [
                        {
                            "bbox": (10, 20, 160, 40),
                            "spans": [{"text": "4. Experiments", "size": 12.0, "font": "Times-Bold"}],
                        }
                    ],
                }
            ]
        }


class _DummyPageWithCitationRichProse(_DummyPage):
    def get_text(self, mode: str):
        assert mode == "dict"
        return {
            "blocks": [
                {
                    "bbox": (10, 20, 190, 45),
                    "lines": [
                        {
                            "bbox": (10, 20, 190, 45),
                            "spans": [{"text": "PyTorch [48]. We exploit the vanilla NeRF from [26] to represent the 3D scene.", "size": 10.0, "font": "Times-Roman"}],
                        }
                    ],
                }
            ]
        }


class _DummyPageReuseAsset(_DummyPage):
    def get_text(self, mode: str):
        assert mode == "dict"
        return {
            "blocks": [
                {
                    "bbox": (10, 170, 120, 210),
                    "lines": [
                        {
                            "bbox": (10, 170, 120, 185),
                            "spans": [{"text": "Figure 1. Caption text.", "size": 9.0, "font": "Times-Roman"}],
                        }
                    ],
                },
            ]
        }


def test_extract_text_blocks_keeps_caption_over_visual_and_appends_assets(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    page = _DummyPage()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    visual_rect = fitz.Rect(0, 80, 160, 190)
    table_rect = fitz.Rect(20, 220, 180, 260)
    persisted = {}

    monkeypatch.setattr(page_text_blocks, "_looks_like_math_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_code_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_equation_text", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_non_body_metadata_text", lambda *args, **kwargs: False)
    monkeypatch.setattr(page_text_blocks, "_is_frontmatter_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "detect_header_tag", lambda **kwargs: None)
    monkeypatch.setattr(page_text_blocks, "sort_blocks_reading_order", lambda blocks, page_width: blocks)
    monkeypatch.setattr(page_text_blocks, "_detect_column_split_x", lambda blocks, page_width: None)

    monkeypatch.setattr(converter, "_collect_page_text_line_boxes", lambda page: [])
    monkeypatch.setattr(converter, "_expanded_visual_crop_rect", lambda **kwargs: kwargs["rect"])
    monkeypatch.setattr(converter, "_extract_page_figure_caption_candidates", lambda page: [])
    monkeypatch.setattr(
        converter,
        "_match_figure_entries_with_captions",
        lambda **kwargs: kwargs["figure_entries"],
    )
    monkeypatch.setattr(
        converter,
        "_persist_page_figure_metadata",
        lambda **kwargs: persisted.setdefault("entries", kwargs["figure_entries"]),
    )

    blocks = converter._extract_text_blocks(
        page,
        page_index=0,
        body_size=10.0,
        tables=[(table_rect, "| H |\n| --- |\n| 1 |")],
        visual_rects=[visual_rect],
        assets_dir=assets_dir,
        is_references_page=False,
    )

    texts = [b.text for b in blocks]
    assert "Body paragraph." in texts
    assert "Figure 1. Caption text." in texts
    caption_block = next(b for b in blocks if b.text == "Figure 1. Caption text.")
    assert caption_block.is_caption is True
    assert any(b.is_table and b.table_markdown == "| H |\n| --- |\n| 1 |" for b in blocks)
    assert any("./assets/page_1_fig_1.png" in b.text for b in blocks)
    assert (assets_dir / "page_1_fig_1.png").exists()
    assert persisted["entries"][0]["asset_name"] == "page_1_fig_1.png"


def test_extract_text_blocks_reuses_existing_asset_without_resaving(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    page = _DummyPageReuseAsset()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "page_1_fig_1.png").write_bytes(b"x" * 512)
    visual_rect = fitz.Rect(0, 80, 160, 190)
    persisted = {}

    monkeypatch.setattr(page_text_blocks, "_looks_like_math_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_code_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_equation_text", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_non_body_metadata_text", lambda *args, **kwargs: False)
    monkeypatch.setattr(page_text_blocks, "_is_frontmatter_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "detect_header_tag", lambda **kwargs: None)
    monkeypatch.setattr(page_text_blocks, "sort_blocks_reading_order", lambda blocks, page_width: blocks)
    monkeypatch.setattr(page_text_blocks, "_detect_column_split_x", lambda blocks, page_width: None)

    monkeypatch.setattr(converter, "_collect_page_text_line_boxes", lambda page: [])
    monkeypatch.setattr(converter, "_expanded_visual_crop_rect", lambda **kwargs: kwargs["rect"])
    monkeypatch.setattr(converter, "_extract_page_figure_caption_candidates", lambda page: [])
    monkeypatch.setattr(converter, "_match_figure_entries_with_captions", lambda **kwargs: kwargs["figure_entries"])
    monkeypatch.setattr(
        converter,
        "_persist_page_figure_metadata",
        lambda **kwargs: persisted.setdefault("entries", kwargs["figure_entries"]),
    )

    def _unexpected_get_pixmap(*args, **kwargs):
        raise AssertionError("existing image asset should be reused")

    page.get_pixmap = _unexpected_get_pixmap

    blocks = converter._extract_text_blocks(
        page,
        page_index=0,
        body_size=10.0,
        tables=[],
        visual_rects=[visual_rect],
        assets_dir=assets_dir,
        is_references_page=False,
    )

    texts = [b.text for b in blocks]
    assert "![Figure](./assets/page_1_fig_1.png)" in texts
    assert persisted["entries"][0]["asset_name"] == "page_1_fig_1.png"


def test_extract_text_blocks_upgrades_image_alt_text_after_caption_match(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    page = _DummyPage()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    visual_rect = fitz.Rect(0, 80, 160, 190)

    monkeypatch.setattr(page_text_blocks, "_looks_like_math_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_code_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_equation_text", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_non_body_metadata_text", lambda *args, **kwargs: False)
    monkeypatch.setattr(page_text_blocks, "_is_frontmatter_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "detect_header_tag", lambda **kwargs: None)
    monkeypatch.setattr(page_text_blocks, "sort_blocks_reading_order", lambda blocks, page_width: blocks)
    monkeypatch.setattr(page_text_blocks, "_detect_column_split_x", lambda blocks, page_width: None)

    monkeypatch.setattr(converter, "_collect_page_text_line_boxes", lambda page: [])
    monkeypatch.setattr(converter, "_expanded_visual_crop_rect", lambda **kwargs: kwargs["rect"])
    monkeypatch.setattr(converter, "_extract_page_figure_caption_candidates", lambda page: [])
    monkeypatch.setattr(
        converter,
        "_match_figure_entries_with_captions",
        lambda **kwargs: [
            {
                **kwargs["figure_entries"][0],
                "fig_no": 4,
                "fig_ident": "4",
                "caption": "Figure 4. Caption text.",
            }
        ],
    )
    monkeypatch.setattr(converter, "_persist_page_figure_metadata", lambda **kwargs: {})

    blocks = converter._extract_text_blocks(
        page,
        page_index=0,
        body_size=10.0,
        tables=[],
        visual_rects=[visual_rect],
        assets_dir=assets_dir,
        is_references_page=False,
    )

    assert any(b.text == "![Figure 4](./assets/page_1_fig_1.png)" for b in blocks)


def test_extract_text_blocks_drops_short_figure_internal_labels_near_visual_rect(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    page = _DummyPageWithFigureLabel()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    visual_rect = fitz.Rect(10, 80, 160, 190)

    monkeypatch.setattr(page_text_blocks, "_looks_like_math_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_code_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_equation_text", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_non_body_metadata_text", lambda *args, **kwargs: False)
    monkeypatch.setattr(page_text_blocks, "_is_frontmatter_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "detect_header_tag", lambda **kwargs: None)
    monkeypatch.setattr(page_text_blocks, "sort_blocks_reading_order", lambda blocks, page_width: blocks)
    monkeypatch.setattr(page_text_blocks, "_detect_column_split_x", lambda blocks, page_width: None)

    monkeypatch.setattr(converter, "_collect_page_text_line_boxes", lambda page: [])
    monkeypatch.setattr(converter, "_expanded_visual_crop_rect", lambda **kwargs: kwargs["rect"])
    monkeypatch.setattr(converter, "_extract_page_figure_caption_candidates", lambda page: [])
    monkeypatch.setattr(converter, "_match_figure_entries_with_captions", lambda **kwargs: kwargs["figure_entries"])
    monkeypatch.setattr(converter, "_persist_page_figure_metadata", lambda **kwargs: {})

    blocks = converter._extract_text_blocks(
        page,
        page_index=0,
        body_size=10.0,
        tables=[],
        visual_rects=[visual_rect],
        assets_dir=assets_dir,
        is_references_page=False,
    )

    texts = [b.text for b in blocks]
    assert "Body paragraph." in texts
    assert "Figure 1. Caption text." in texts
    assert "a" not in texts


def test_extract_text_blocks_drops_label_just_below_visual_bottom_edge(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    page = _DummyPageWithFigureLabel()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    visual_rect = fitz.Rect(10, 80, 160, 190)

    page.get_text = lambda mode: {
        "blocks": [
            {
                "bbox": (10, 20, 120, 40),
                "lines": [
                    {
                        "bbox": (10, 20, 120, 40),
                        "spans": [{"text": "Body paragraph.", "size": 10.0, "font": "Times-Roman"}],
                    }
                ],
            },
            {
                "bbox": (20, 191, 140, 198),
                "lines": [
                    {
                        "bbox": (20, 191, 140, 198),
                        "spans": [{"text": "Interference contrast", "size": 8.0, "font": "Times-Roman"}],
                    }
                ],
            },
            {
                "bbox": (10, 205, 120, 220),
                "lines": [
                    {
                        "bbox": (10, 205, 120, 220),
                        "spans": [{"text": "Figure 1. Caption text.", "size": 9.0, "font": "Times-Roman"}],
                    }
                ],
            },
        ]
    }

    monkeypatch.setattr(page_text_blocks, "_looks_like_math_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_code_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_equation_text", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_non_body_metadata_text", lambda *args, **kwargs: False)
    monkeypatch.setattr(page_text_blocks, "_is_frontmatter_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "detect_header_tag", lambda **kwargs: None)
    monkeypatch.setattr(page_text_blocks, "sort_blocks_reading_order", lambda blocks, page_width: blocks)
    monkeypatch.setattr(page_text_blocks, "_detect_column_split_x", lambda blocks, page_width: None)
    monkeypatch.setattr(converter, "_collect_page_text_line_boxes", lambda page: [])
    monkeypatch.setattr(converter, "_expanded_visual_crop_rect", lambda **kwargs: kwargs["rect"])
    monkeypatch.setattr(converter, "_extract_page_figure_caption_candidates", lambda page: [])
    monkeypatch.setattr(converter, "_match_figure_entries_with_captions", lambda **kwargs: kwargs["figure_entries"])
    monkeypatch.setattr(converter, "_persist_page_figure_metadata", lambda **kwargs: {})

    blocks = converter._extract_text_blocks(
        page,
        page_index=0,
        body_size=10.0,
        tables=[],
        visual_rects=[visual_rect],
        assets_dir=assets_dir,
        is_references_page=False,
    )

    texts = [b.text for b in blocks]
    assert "Interference contrast" not in texts
    assert "Figure 1. Caption text." in texts


def test_extract_text_blocks_drops_running_header_lines(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    page = _DummyPageWithRunningHeader()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    monkeypatch.setattr(page_text_blocks, "_looks_like_math_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_code_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_equation_text", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_non_body_metadata_text", lambda *args, **kwargs: False)
    monkeypatch.setattr(page_text_blocks, "_is_frontmatter_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "detect_header_tag", lambda **kwargs: None)
    monkeypatch.setattr(page_text_blocks, "sort_blocks_reading_order", lambda blocks, page_width: blocks)
    monkeypatch.setattr(page_text_blocks, "_detect_column_split_x", lambda blocks, page_width: None)
    monkeypatch.setattr(converter, "_collect_page_text_line_boxes", lambda page: [])
    monkeypatch.setattr(converter, "_expanded_visual_crop_rect", lambda **kwargs: kwargs["rect"])
    monkeypatch.setattr(converter, "_extract_page_figure_caption_candidates", lambda page: [])
    monkeypatch.setattr(converter, "_match_figure_entries_with_captions", lambda **kwargs: kwargs["figure_entries"])
    monkeypatch.setattr(converter, "_persist_page_figure_metadata", lambda **kwargs: {})

    blocks = converter._extract_text_blocks(
        page,
        page_index=2,
        body_size=10.0,
        tables=[],
        visual_rects=[],
        assets_dir=assets_dir,
        is_references_page=False,
    )

    texts = [b.text for b in blocks]
    assert "Body paragraph." in texts
    assert not any("Light: Science & Applications" in text for text in texts)


def test_extract_text_blocks_drops_compact_running_header_lines(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    page = _DummyPageWithCompactRunningHeader()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    monkeypatch.setattr(page_text_blocks, "_looks_like_math_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_code_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_equation_text", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_non_body_metadata_text", lambda *args, **kwargs: False)
    monkeypatch.setattr(page_text_blocks, "_is_frontmatter_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "detect_header_tag", lambda **kwargs: None)
    monkeypatch.setattr(page_text_blocks, "sort_blocks_reading_order", lambda blocks, page_width: blocks)
    monkeypatch.setattr(page_text_blocks, "_detect_column_split_x", lambda blocks, page_width: None)
    monkeypatch.setattr(converter, "_collect_page_text_line_boxes", lambda page: [])
    monkeypatch.setattr(converter, "_expanded_visual_crop_rect", lambda **kwargs: kwargs["rect"])
    monkeypatch.setattr(converter, "_extract_page_figure_caption_candidates", lambda page: [])
    monkeypatch.setattr(converter, "_match_figure_entries_with_captions", lambda **kwargs: kwargs["figure_entries"])
    monkeypatch.setattr(converter, "_persist_page_figure_metadata", lambda **kwargs: {})

    blocks = converter._extract_text_blocks(
        page,
        page_index=2,
        body_size=10.0,
        tables=[],
        visual_rects=[],
        assets_dir=assets_dir,
        is_references_page=False,
    )

    texts = [b.text for b in blocks]
    assert "Body paragraph." in texts
    assert not any("Light: Science & Applications" in text for text in texts)


def test_extract_text_blocks_drops_center_footer_page_number_lines(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    page = _DummyPageWithFooterNumber()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    monkeypatch.setattr(page_text_blocks, "_looks_like_math_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_code_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_equation_text", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_non_body_metadata_text", lambda *args, **kwargs: False)
    monkeypatch.setattr(page_text_blocks, "_is_frontmatter_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "detect_header_tag", lambda **kwargs: None)
    monkeypatch.setattr(page_text_blocks, "sort_blocks_reading_order", lambda blocks, page_width: blocks)
    monkeypatch.setattr(page_text_blocks, "_detect_column_split_x", lambda blocks, page_width: None)
    monkeypatch.setattr(converter, "_collect_page_text_line_boxes", lambda page: [])
    monkeypatch.setattr(converter, "_expanded_visual_crop_rect", lambda **kwargs: kwargs["rect"])
    monkeypatch.setattr(converter, "_extract_page_figure_caption_candidates", lambda page: [])
    monkeypatch.setattr(converter, "_match_figure_entries_with_captions", lambda **kwargs: kwargs["figure_entries"])
    monkeypatch.setattr(converter, "_persist_page_figure_metadata", lambda **kwargs: {})

    blocks = converter._extract_text_blocks(
        page,
        page_index=4,
        body_size=10.0,
        tables=[],
        visual_rects=[],
        assets_dir=assets_dir,
        is_references_page=False,
    )

    texts = [b.text for b in blocks]
    assert "Body paragraph." in texts
    assert "5" not in texts


def test_extract_text_blocks_keeps_numbered_section_heading_out_of_math(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    page = _DummyPageWithSectionHeading()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    monkeypatch.setattr(page_text_blocks, "_looks_like_math_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_code_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_equation_text", lambda text: True)
    monkeypatch.setattr(page_text_blocks, "_is_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_non_body_metadata_text", lambda *args, **kwargs: False)
    monkeypatch.setattr(page_text_blocks, "_is_frontmatter_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "detect_header_tag", lambda **kwargs: "[H1]")
    monkeypatch.setattr(page_text_blocks, "sort_blocks_reading_order", lambda blocks, page_width: blocks)
    monkeypatch.setattr(page_text_blocks, "_detect_column_split_x", lambda blocks, page_width: None)
    monkeypatch.setattr(converter, "_collect_page_text_line_boxes", lambda page: [])
    monkeypatch.setattr(converter, "_expanded_visual_crop_rect", lambda **kwargs: kwargs["rect"])
    monkeypatch.setattr(converter, "_extract_page_figure_caption_candidates", lambda page: [])
    monkeypatch.setattr(converter, "_match_figure_entries_with_captions", lambda **kwargs: kwargs["figure_entries"])
    monkeypatch.setattr(converter, "_persist_page_figure_metadata", lambda **kwargs: {})

    blocks = converter._extract_text_blocks(
        page,
        page_index=4,
        body_size=10.0,
        tables=[],
        visual_rects=[],
        assets_dir=assets_dir,
        is_references_page=False,
    )

    assert len(blocks) == 1
    assert blocks[0].text == "4. Experiments"
    assert blocks[0].is_math is False
    assert blocks[0].heading_level == "[H1]"


def test_extract_text_blocks_keeps_citation_rich_prose_out_of_math(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    page = _DummyPageWithCitationRichProse()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    monkeypatch.setattr(page_text_blocks, "_looks_like_math_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_code_block", lambda lines: False)
    monkeypatch.setattr(page_text_blocks, "_looks_like_equation_text", lambda text: True)
    monkeypatch.setattr(page_text_blocks, "_is_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "_is_non_body_metadata_text", lambda *args, **kwargs: False)
    monkeypatch.setattr(page_text_blocks, "_is_frontmatter_noise_line", lambda text: False)
    monkeypatch.setattr(page_text_blocks, "detect_header_tag", lambda **kwargs: None)
    monkeypatch.setattr(page_text_blocks, "sort_blocks_reading_order", lambda blocks, page_width: blocks)
    monkeypatch.setattr(page_text_blocks, "_detect_column_split_x", lambda blocks, page_width: None)
    monkeypatch.setattr(converter, "_collect_page_text_line_boxes", lambda page: [])
    monkeypatch.setattr(converter, "_expanded_visual_crop_rect", lambda **kwargs: kwargs["rect"])
    monkeypatch.setattr(converter, "_extract_page_figure_caption_candidates", lambda page: [])
    monkeypatch.setattr(converter, "_match_figure_entries_with_captions", lambda **kwargs: kwargs["figure_entries"])
    monkeypatch.setattr(converter, "_persist_page_figure_metadata", lambda **kwargs: {})

    blocks = converter._extract_text_blocks(
        page,
        page_index=4,
        body_size=10.0,
        tables=[],
        visual_rects=[],
        assets_dir=assets_dir,
        is_references_page=False,
    )

    assert len(blocks) == 1
    assert blocks[0].is_math is False
    assert "PyTorch [48]. We exploit the vanilla NeRF from [26]" in blocks[0].text
