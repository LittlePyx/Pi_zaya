from pathlib import Path

import fitz

from kb.converter.config import ConvertConfig
from kb.converter.models import TextBlock
from kb.converter.pipeline import PDFConverter
import kb.converter.page_layout_crops as page_layout_crops


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


class _DummyLLMWorker:
    def __init__(self):
        self.calls = []

    def call_llm_page_to_markdown(
        self,
        png_bytes,
        *,
        page_number,
        total_pages,
        hint,
        speed_mode,
        is_references_page,
    ):
        tag = png_bytes.decode("utf-8")
        self.calls.append(
            {
                "tag": tag,
                "page_number": page_number,
                "total_pages": total_pages,
                "hint": hint,
                "speed_mode": speed_mode,
                "is_references_page": is_references_page,
            }
        )
        if tag == "left":
            return "Left prose only"
        if tag == "right":
            return ""
        return None


class _DummyPixmap:
    def __init__(self, tag: str):
        self._tag = tag

    def tobytes(self, fmt: str) -> bytes:
        assert fmt == "png"
        return self._tag.encode("utf-8")


class _DummyPage:
    def __init__(self):
        self.rect = fitz.Rect(0, 0, 200, 300)

    def get_text(self, mode: str):
        assert mode == "dict"
        return {"blocks": []}

    def get_pixmap(self, *, clip, dpi, alpha=False):
        assert dpi >= 220
        lane = "left" if float(clip.x0) < 50.0 else "right"
        return _DummyPixmap(lane)


def test_fallback_markdown_from_blocks_formats_tables_headings_and_captions():
    blocks = [
        TextBlock(bbox=(0, 0, 40, 10), text="Overview", heading_level="[H2]"),
        TextBlock(
            bbox=(0, 12, 40, 24),
            text="[TABLE]",
            is_table=True,
            table_markdown="| A |\n| --- |\n| 1 |",
        ),
        TextBlock(bbox=(0, 26, 40, 36), text="Fig. 1. Caption text.", is_caption=True),
        TextBlock(bbox=(0, 38, 40, 48), text="Body paragraph."),
    ]

    out = PDFConverter._fallback_markdown_from_blocks(blocks)

    assert "## Overview" in out
    assert "| A |\n| --- |\n| 1 |" in out
    assert "*Fig. 1. Caption text.*" in out
    assert out.endswith("Body paragraph.")


def test_layout_crops_preserve_region_order_and_restore_missing_tables(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    converter.llm_worker = _DummyLLMWorker()
    page = _DummyPage()
    monkeypatch.setenv("KB_PDF_VISION_LAYOUT_CROP_MODE", "1")

    monkeypatch.setattr(page_layout_crops, "_page_maybe_has_table_from_dict", lambda d: True)
    monkeypatch.setattr(page_layout_crops, "_extract_tables_by_layout", lambda *args, **kwargs: ["table"])
    monkeypatch.setattr(page_layout_crops, "detect_body_font_size", lambda pages: 10.0)
    monkeypatch.setattr(page_layout_crops, "_detect_column_split_x", lambda blocks, page_width: 100.0)

    blocks = [
        TextBlock(
            bbox=(10, 10, 90, 40),
            text="[TABLE]",
            is_table=True,
            table_markdown="| A |\n| --- |\n| 1 |",
        ),
        TextBlock(bbox=(10, 45, 90, 60), text="Left fallback"),
        TextBlock(bbox=(110, 15, 180, 30), text="Right fallback"),
    ]
    monkeypatch.setattr(
        converter,
        "_extract_text_blocks",
        lambda *args, **kwargs: blocks,
    )

    out = converter._convert_page_with_layout_crops(
        page=page,
        page_index=1,
        total_pages=6,
        page_hint="",
        speed_mode="normal",
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        image_names=[],
    )

    assert out is not None
    assert out.index("| A |\n| --- |\n| 1 |") < out.index("Left prose only")
    assert out.index("Left prose only") < out.index("Right fallback")
    assert [call["tag"] for call in converter.llm_worker.calls] == ["left", "right"]


def test_layout_crops_auto_enable_for_two_column_cross_reading_risk_without_tables(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    converter.llm_worker = _DummyLLMWorker()
    page = _DummyPage()

    monkeypatch.setattr(page_layout_crops, "_page_maybe_has_table_from_dict", lambda d: False)
    monkeypatch.setattr(page_layout_crops, "detect_body_font_size", lambda pages: 10.0)
    monkeypatch.setattr(page_layout_crops, "_detect_column_split_x", lambda blocks, page_width: 100.0)

    blocks = [
        TextBlock(
            bbox=(10, 10, 90, 90),
            text="realized because most of the photons are discarded and the introduction continues here " * 2,
        ),
        TextBlock(
            bbox=(110, 12, 190, 92),
            text="Results Principle of interferometric ISM (iISM) In this work we developed " * 2,
        ),
        TextBlock(
            bbox=(112, 105, 188, 128),
            text="where n is the refractive index of the medium and z is the axial position",
        ),
    ]
    monkeypatch.setattr(converter, "_extract_text_blocks", lambda *args, **kwargs: blocks)

    out = converter._convert_page_with_layout_crops(
        page=page,
        page_index=1,
        total_pages=6,
        page_hint="",
        speed_mode="normal",
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        image_names=[],
    )

    assert out is not None
    assert out.index("Left prose only") < out.index("Results Principle of interferometric ISM")
    assert [call["tag"] for call in converter.llm_worker.calls] == ["left", "right"]


def test_layout_crops_drop_running_header_footer_noise_blocks(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    converter.llm_worker = _DummyLLMWorker()
    page = _DummyPage()
    monkeypatch.setenv("KB_PDF_VISION_LAYOUT_CROP_MODE", "1")

    monkeypatch.setattr(page_layout_crops, "_page_maybe_has_table_from_dict", lambda d: False)
    monkeypatch.setattr(page_layout_crops, "detect_body_font_size", lambda pages: 10.0)
    monkeypatch.setattr(page_layout_crops, "_detect_column_split_x", lambda blocks, page_width: 100.0)

    blocks = [
        TextBlock(
            bbox=(10, 5, 190, 18),
            text="Kuppers and Moerner Light: Science & Applications (2026) 15:129 Page 2 of 13",
        ),
        TextBlock(
            bbox=(10, 25, 90, 90),
            text="realized because most of the photons are discarded and the introduction continues here " * 2,
        ),
        TextBlock(
            bbox=(110, 27, 190, 92),
            text="Results Principle of interferometric ISM (iISM) In this work we developed " * 2,
        ),
        TextBlock(
            bbox=(112, 100, 188, 128),
            text="where n is the refractive index of the medium and z is the axial position",
        ),
    ]
    monkeypatch.setattr(converter, "_extract_text_blocks", lambda *args, **kwargs: blocks)

    out = converter._convert_page_with_layout_crops(
        page=page,
        page_index=1,
        total_pages=6,
        page_hint="",
        speed_mode="normal",
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        image_names=[],
    )

    assert out is not None
    assert "Kuppers and Moerner" not in out
