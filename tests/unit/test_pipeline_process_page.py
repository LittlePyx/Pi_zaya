from pathlib import Path
from dataclasses import replace

import fitz

from kb.converter.config import ConvertConfig, LlmConfig
from kb.converter.models import TextBlock
from kb.converter.pipeline import PDFConverter
import kb.converter.page_local_pipeline as page_local_pipeline
from kb.converter.pipeline_render_markdown import render_blocks_to_markdown


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


class _DummyPage:
    def __init__(self):
        self.rect = fitz.Rect(0, 0, 200, 300)
        self.get_text_calls = {"dict": 0, "text": 0}

    def get_text(self, mode: str):
        assert mode in self.get_text_calls
        self.get_text_calls[mode] += 1
        if mode == "text":
            return "Body paragraph."
        return {"blocks": []}


class _EquationPixmap:
    def save(self, path):
        Path(path).write_bytes(b"equation-image" * 32)


class _EquationPage(_DummyPage):
    def __init__(self):
        self.rect = fitz.Rect(0, 0, 600, 800)

    def get_pixmap(self, **_kwargs):
        return _EquationPixmap()


def test_process_page_orchestrates_local_pipeline_steps(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    page = _DummyPage()

    kept_visual_rect = fitz.Rect(20, 80, 180, 180)
    header_rect = fitz.Rect(0, 0, 100, 20)
    table_rect = fitz.Rect(15, 190, 185, 240)

    monkeypatch.setattr(page_local_pipeline, "detect_body_font_size", lambda pages, **kwargs: 11.0)
    monkeypatch.setattr(page_local_pipeline, "_page_has_references_heading", lambda page, **kwargs: False)
    monkeypatch.setattr(page_local_pipeline, "_page_looks_like_references_content", lambda page, **kwargs: False)
    monkeypatch.setattr(
        page_local_pipeline,
        "_collect_visual_rects",
        lambda page, **kwargs: [header_rect, kept_visual_rect],
    )
    monkeypatch.setattr(page_local_pipeline, "_page_maybe_has_table_from_dict", lambda d: True)
    monkeypatch.setattr(
        page_local_pipeline,
        "_extract_tables_by_layout",
        lambda *args, **kwargs: [(table_rect, "| H |\n| --- |\n| 1 |")],
    )

    monkeypatch.setattr(converter, "_extract_page_figure_caption_candidates", lambda page, **kwargs: [])
    monkeypatch.setattr(
        converter,
        "_split_visual_rects_by_internal_captions",
        lambda **kwargs: kwargs["visual_rects"],
    )

    extracted = {}

    def _extract_text_blocks(*args, **kwargs):
        extracted["body_size"] = kwargs["body_size"]
        extracted["tables"] = kwargs["tables"]
        extracted["visual_rects"] = kwargs["visual_rects"]
        extracted["is_references_page"] = kwargs["is_references_page"]
        extracted["caption_candidates"] = kwargs["caption_candidates"]
        return [TextBlock(bbox=(10, 10, 50, 30), text="Body paragraph.")]

    merge_calls = {"count": 0}

    def _merge_adjacent_math_fragments(blocks, *, page_wh):
        merge_calls["count"] += 1
        return blocks

    rendered = {}

    def _render_blocks_to_markdown(blocks, page_index, **kwargs):
        rendered["blocks"] = blocks
        rendered["page_index"] = page_index
        rendered["is_references_page"] = kwargs["is_references_page"]
        return "FINAL_MD"

    monkeypatch.setattr(converter, "_extract_text_blocks", _extract_text_blocks)
    monkeypatch.setattr(converter, "_merge_adjacent_math_fragments", _merge_adjacent_math_fragments)
    monkeypatch.setattr(converter, "_render_blocks_to_markdown", _render_blocks_to_markdown)

    out = converter._process_page(
        page,
        page_index=2,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
    )

    assert out == "FINAL_MD"
    assert extracted["body_size"] == 11.0
    assert extracted["tables"] == [(table_rect, "| H |\n| --- |\n| 1 |")]
    assert extracted["visual_rects"] == [kept_visual_rect]
    assert extracted["is_references_page"] is False
    assert extracted["caption_candidates"] == []
    assert getattr(page, "has_table_hint") is True
    assert merge_calls["count"] == 1
    assert rendered["page_index"] == 2
    assert rendered["is_references_page"] is False
    assert len(rendered["blocks"]) == 1
    assert page.get_text_calls == {"dict": 1, "text": 1}


def test_render_prepared_page_uses_reference_text_fastpath(tmp_path):
    converter = _make_converter(tmp_path)
    page = _DummyPage()

    prepared = {
        "blocks": [TextBlock(bbox=(10, 10, 50, 30), text="Should not render from blocks.")],
        "is_references_page": True,
        "reference_page_text": "References\n[1] A. Author. First paper. Journal, 2020. 3\n10\n",
        "prepare_elapsed": 0.01,
    }

    out = page_local_pipeline.render_prepared_page(
        converter,
        prepared=prepared,
        page=page,
        page_index=0,
        assets_dir=tmp_path,
    )

    assert out.startswith("# References")
    assert "[1] A. Author. First paper. Journal, 2020. 3" in out
    assert "\n10\n" not in f"\n{out}\n"


def test_process_page_local_only_skips_llm_enhance(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    converter.cfg = replace(
        converter.cfg,
        llm=LlmConfig(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="vision-model",
        ),
    )
    page = _DummyPage()

    monkeypatch.setattr(page_local_pipeline, "detect_body_font_size", lambda pages, **kwargs: 11.0)
    monkeypatch.setattr(page_local_pipeline, "_page_has_references_heading", lambda page, **kwargs: False)
    monkeypatch.setattr(page_local_pipeline, "_page_looks_like_references_content", lambda page, **kwargs: False)
    monkeypatch.setattr(page_local_pipeline, "_collect_visual_rects", lambda page, **kwargs: [])
    monkeypatch.setattr(page_local_pipeline, "_page_maybe_has_table_from_dict", lambda d: False)
    monkeypatch.setattr(page_local_pipeline, "_extract_tables_by_layout", lambda *args, **kwargs: [])
    monkeypatch.setattr(converter, "_extract_page_figure_caption_candidates", lambda page, **kwargs: [])
    monkeypatch.setattr(converter, "_split_visual_rects_by_internal_captions", lambda **kwargs: [])
    monkeypatch.setattr(
        converter,
        "_extract_text_blocks",
        lambda *args, **kwargs: [TextBlock(bbox=(10, 10, 50, 30), text="Local body.")],
    )
    monkeypatch.setattr(converter, "_merge_adjacent_math_fragments", lambda blocks, *, page_wh: blocks)
    monkeypatch.setattr(
        converter,
        "_enhance_blocks_with_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM enhance must be skipped")),
    )
    monkeypatch.setattr(converter, "_render_blocks_to_markdown", lambda *args, **kwargs: "LOCAL_MD")

    out = converter._process_page_local_only(
        page,
        page_index=0,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
    )

    assert out == "LOCAL_MD"


def test_local_only_render_disables_math_and_text_llm_calls(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    converter.cfg = replace(
        converter.cfg,
        llm=LlmConfig(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="vision-model",
        ),
    )
    converter.llm_worker._client = object()
    converter._active_speed_config = {"use_llm_for_all": True, "use_llm_in_render": True}
    calls = {"math": 0, "math_image": 0, "body": 0, "raw": 0}

    monkeypatch.setattr(
        converter.llm_worker,
        "call_llm_repair_math",
        lambda *args, **kwargs: calls.__setitem__("math", calls["math"] + 1),
    )
    monkeypatch.setattr(
        converter.llm_worker,
        "call_llm_repair_math_from_image",
        lambda *args, **kwargs: calls.__setitem__("math_image", calls["math_image"] + 1),
    )
    monkeypatch.setattr(
        converter.llm_worker,
        "call_llm_repair_body_paragraph",
        lambda *args, **kwargs: calls.__setitem__("body", calls["body"] + 1),
    )
    monkeypatch.setattr(
        converter.llm_worker,
        "_llm_create",
        lambda **kwargs: calls.__setitem__("raw", calls["raw"] + 1),
    )

    out = converter._render_blocks_to_markdown(
        [
            TextBlock(bbox=(10, 20, 190, 50), text="x = sum of all sample values", is_math=True),
            TextBlock(bbox=(10, 60, 190, 90), text="鑶规 local extraction text"),
        ],
        0,
        page=_DummyPage(),
        assets_dir=tmp_path,
        is_references_page=False,
        allow_llm_calls=False,
    )

    assert out
    assert calls == {"math": 0, "math_image": 0, "body": 0, "raw": 0}


def test_safe_complex_fallback_requires_two_columns_and_source_formula_rect(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    page = _EquationPage()
    left_body = TextBlock(
        bbox=(40, 140, 275, 250),
        text="Left-column body text continues with enough content to establish a reliable reading lane.",
    )
    formula = TextBlock(bbox=(100, 270, 250, 300), text="Y = sum fragments", is_math=True)
    right_heading = TextBlock(bbox=(330, 120, 520, 145), text="3.3. Proposed Framework", heading_level="[H2]")
    right_body = TextBlock(
        bbox=(330, 150, 560, 260),
        text="Right-column body text provides overlapping vertical flow and a second reliable reading lane.",
    )
    prepared = {
        "blocks": [right_heading, left_body, formula, right_body],
        "is_references_page": False,
    }
    monkeypatch.setattr(
        converter,
        "_collect_display_math_candidates",
        lambda *args, **kwargs: [{"rect": fitz.Rect(90, 265, 270, 310), "text": "Y = sum_i X_i (3)"}],
    )

    page_local_pipeline._prepare_safe_complex_fallback(converter, prepared, page, page_index=3)

    assert prepared["safe_complex_fallback"] is True
    assert prepared["safe_formula_rects"] == [(90.0, 265.0, 270.0, 310.0)]
    assert prepared["blocks"].index(left_body) < prepared["blocks"].index(right_heading)
    assert prepared["blocks"].index(formula) < prepared["blocks"].index(right_body)


def test_safe_complex_fallback_does_not_change_table_or_single_column_pages(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    page = _EquationPage()
    monkeypatch.setattr(
        converter,
        "_collect_display_math_candidates",
        lambda *args, **kwargs: [{"rect": fitz.Rect(90, 265, 270, 310), "text": "Y = X (3)"}],
    )
    table_prepared = {
        "blocks": [
            TextBlock(bbox=(40, 140, 275, 250), text="Left-column reliable text that is long enough for the layout gate."),
            TextBlock(bbox=(330, 140, 560, 250), text="Right-column reliable text that is long enough for the layout gate."),
            TextBlock(bbox=(40, 280, 560, 380), text="[TABLE]", is_table=True, table_markdown="| A |\n| --- |\n| 1 |"),
        ],
        "is_references_page": False,
    }
    page_local_pipeline._prepare_safe_complex_fallback(converter, table_prepared, page, page_index=0)
    assert "safe_complex_fallback" not in table_prepared

    single_column_prepared = {
        "blocks": [
            TextBlock(bbox=(80, 120, 520, 260), text="A reliable single-column paragraph with an equation below it."),
            TextBlock(bbox=(180, 280, 420, 315), text="Y = X", is_math=True),
        ],
        "is_references_page": False,
    }
    page_local_pipeline._prepare_safe_complex_fallback(converter, single_column_prepared, page, page_index=0)
    assert "safe_complex_fallback" not in single_column_prepared


def test_safe_complex_render_uses_one_equation_image_and_never_emits_fragmented_latex(tmp_path):
    converter = _make_converter(tmp_path)
    page = _EquationPage()
    blocks = [
        TextBlock(bbox=(100, 270, 135, 290), text="N X", is_math=True),
        TextBlock(bbox=(120, 280, 235, 305), text="X_i + Z", is_math=True),
        TextBlock(bbox=(245, 280, 270, 300), text="(3)"),
        TextBlock(
            bbox=(40, 330, 275, 360),
            text="where the source text remains readable even though it was misclassified as math",
            is_math=True,
        ),
    ]

    out = render_blocks_to_markdown(
        converter,
        blocks,
        3,
        page=page,
        assets_dir=tmp_path,
        allow_llm_calls=False,
        safe_complex_fallback=True,
        safe_formula_rects=[(90, 265, 275, 315)],
    )

    assert out.count("![Equation](./assets/page_4_eq_1.png)") == 1
    assert "<!-- kb:conversion_retry kind=equation page=4 asset=page_4_eq_1.png number=3 -->" in out
    assert "where the source text remains readable" in out
    assert "$$" not in out
    assert (tmp_path / "page_4_eq_1.png").stat().st_size > 256
