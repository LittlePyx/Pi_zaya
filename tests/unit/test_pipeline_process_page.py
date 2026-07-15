from pathlib import Path
from dataclasses import replace

import fitz

from kb.converter.config import ConvertConfig, LlmConfig
from kb.converter.models import TextBlock
from kb.converter.pipeline import PDFConverter
import kb.converter.page_local_pipeline as page_local_pipeline


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

    def get_text(self, mode: str):
        assert mode == "dict"
        return {"blocks": []}


def test_process_page_orchestrates_local_pipeline_steps(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    page = _DummyPage()

    kept_visual_rect = fitz.Rect(20, 80, 180, 180)
    header_rect = fitz.Rect(0, 0, 100, 20)
    table_rect = fitz.Rect(15, 190, 185, 240)

    monkeypatch.setattr(page_local_pipeline, "detect_body_font_size", lambda pages: 11.0)
    monkeypatch.setattr(page_local_pipeline, "_page_has_references_heading", lambda page: False)
    monkeypatch.setattr(page_local_pipeline, "_page_looks_like_references_content", lambda page: False)
    monkeypatch.setattr(page_local_pipeline, "_collect_visual_rects", lambda page: [header_rect, kept_visual_rect])
    monkeypatch.setattr(page_local_pipeline, "_page_maybe_has_table_from_dict", lambda d: True)
    monkeypatch.setattr(
        page_local_pipeline,
        "_extract_tables_by_layout",
        lambda *args, **kwargs: [(table_rect, "| H |\n| --- |\n| 1 |")],
    )

    monkeypatch.setattr(converter, "_extract_page_figure_caption_candidates", lambda page: [])
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
    assert getattr(page, "has_table_hint") is True
    assert merge_calls["count"] == 1
    assert rendered["page_index"] == 2
    assert rendered["is_references_page"] is False
    assert len(rendered["blocks"]) == 1


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

    monkeypatch.setattr(page_local_pipeline, "detect_body_font_size", lambda pages: 11.0)
    monkeypatch.setattr(page_local_pipeline, "_page_has_references_heading", lambda page: False)
    monkeypatch.setattr(page_local_pipeline, "_page_looks_like_references_content", lambda page: False)
    monkeypatch.setattr(page_local_pipeline, "_collect_visual_rects", lambda page: [])
    monkeypatch.setattr(page_local_pipeline, "_page_maybe_has_table_from_dict", lambda d: False)
    monkeypatch.setattr(page_local_pipeline, "_extract_tables_by_layout", lambda *args, **kwargs: [])
    monkeypatch.setattr(converter, "_extract_page_figure_caption_candidates", lambda page: [])
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
