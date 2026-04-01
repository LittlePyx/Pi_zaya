from types import SimpleNamespace
from pathlib import Path

import kb.converter.page_vision_direct_page as page_module


class _DummyPix:
    def tobytes(self, fmt: str) -> bytes:
        return b"png-bytes"


class _DummyPage:
    def __init__(self):
        self.rect = SimpleNamespace(width=612.0, height=792.0)
        self.render_calls = []
        self.clip_render_calls = []

    def get_pixmap(self, matrix=None, alpha=False):
        if matrix is None and alpha is False:
            return _DummyPix()
        self.render_calls.append({"matrix": matrix, "alpha": alpha})
        return _DummyPix()

    def get_text(self, mode: str):
        if mode == "text":
            return ""
        if mode == "dict":
            return {"blocks": []}
        raise AssertionError(mode)


class _DummyConverter:
    def _mask_rects_on_png(self, png_bytes, rects, page_width, page_height):
        return png_bytes

    def _convert_page_with_vision_guardrails(self, **kwargs):
        return "raw-md"

    def _postprocess_vision_page_markdown(self, md, **kwargs):
        return md + "-post"


def test_extract_page_visual_assets_reuses_existing_asset_without_resaving(tmp_path, monkeypatch):
    page = _DummyPage()
    existing = tmp_path / "page_1_fig_1.png"
    existing.write_bytes(b"x" * 300)

    monkeypatch.setattr(page_module, "_collect_visual_rects", lambda page: [page_module.fitz.Rect(40, 60, 300, 260)])

    class _VisualConverter(_DummyConverter):
        def _extract_page_figure_caption_candidates(self, page):
            return []

        def _split_visual_rects_by_internal_captions(self, **kwargs):
            return kwargs["visual_rects"]

        def _collect_page_text_line_boxes(self, page):
            return []

        def _expanded_visual_crop_rect(self, **kwargs):
            return kwargs["rect"]

        def _match_figure_entries_with_captions(self, **kwargs):
            return kwargs["figure_entries"]

        def _persist_page_figure_metadata(self, **kwargs):
            entries = kwargs["figure_entries"]
            return {entry["asset_name"]: dict(entry) for entry in entries}

    def _unexpected_get_pixmap(*args, **kwargs):
        raise AssertionError("existing asset should have been reused")

    page.get_pixmap = _unexpected_get_pixmap

    image_names, figure_meta_by_asset, visual_rects = page_module._extract_page_visual_assets(
        _VisualConverter(),
        page=page,
        page_index=0,
        assets_dir=tmp_path,
        dpi=220,
    )

    assert image_names == ["page_1_fig_1.png"]
    assert "page_1_fig_1.png" in figure_meta_by_asset
    assert len(visual_rects) == 1


def _patch_page_pipeline(monkeypatch):
    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: False)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, *, page, page_index, is_references_page: [])
    monkeypatch.setattr(
        page_module,
        "_extract_page_visual_assets",
        lambda converter, *, page, page_index, assets_dir, dpi: (["page_1_fig_1.png"], {"page_1_fig_1.png": {"fig_no": 1}}, []),
    )
    monkeypatch.setattr(page_module, "_compress_png_bytes", lambda png_bytes, *, speed_config: png_bytes)
    monkeypatch.setattr(page_module, "_build_page_hint", lambda converter, *, page_index, is_references_page, image_names, figure_meta_by_asset: "hint")
    monkeypatch.setattr(
        page_module,
        "_apply_formula_overlay",
        lambda converter, *, png_bytes, page, page_index, page_w, page_h, dpi, is_references_page, page_hint: (png_bytes, page_hint, {"EQ1": "x+y"}),
    )


def test_process_vision_direct_page_omits_stage_timing_by_default(tmp_path, monkeypatch, capsys):
    _patch_page_pipeline(monkeypatch)
    monkeypatch.delenv("KB_PDF_STAGE_TIMINGS", raising=False)

    out = page_module.process_vision_direct_page(
        _DummyConverter(),
        page=_DummyPage(),
        page_index=0,
        total_pages=1,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        speed_mode="normal",
        speed_config={"compress": 3},
        dpi=220,
        mat=(1, 1),
    )

    captured = capsys.readouterr().out
    assert out == "raw-md-post"
    assert "Step 1 (refs check)" not in captured
    assert "TOTAL:" not in captured
    assert "Finished page 1/1" in captured


def test_process_vision_direct_page_logs_stage_timing_when_enabled(tmp_path, monkeypatch, capsys):
    _patch_page_pipeline(monkeypatch)
    monkeypatch.setenv("KB_PDF_STAGE_TIMINGS", "1")

    out = page_module.process_vision_direct_page(
        _DummyConverter(),
        page=_DummyPage(),
        page_index=0,
        total_pages=1,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        speed_mode="normal",
        speed_config={"compress": 3},
        dpi=220,
        mat=(1, 1),
    )

    captured = capsys.readouterr().out
    assert out == "raw-md-post"
    assert "Step 1 (refs check):" in captured
    assert "Step 3 (assets):" in captured
    assert "Step 5 (hints/overlay):" in captured
    assert "max_tokens=" in captured
    assert "Step 7 (postprocess):" in captured
    assert "TOTAL:" in captured


def test_process_vision_direct_page_skips_visual_assets_on_references_pages(tmp_path, monkeypatch):
    called = {"assets": 0}
    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: True)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, *, page, page_index, is_references_page: [])
    monkeypatch.setattr(
        page_module,
        "_extract_page_visual_assets",
        lambda converter, *, page, page_index, assets_dir, dpi: called.__setitem__("assets", called["assets"] + 1),
    )
    monkeypatch.setattr(page_module, "_compress_png_bytes", lambda png_bytes, *, speed_config: png_bytes)
    monkeypatch.setattr(page_module, "_apply_formula_overlay", lambda converter, **kwargs: (kwargs["png_bytes"], kwargs["page_hint"], {}))

    out = page_module.process_vision_direct_page(
        _DummyConverter(),
        page=_DummyPage(),
        page_index=0,
        total_pages=1,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        speed_mode="normal",
        speed_config={"compress": 3},
        dpi=220,
        mat=(1, 1),
    )

    assert out == "raw-md-post"
    assert called["assets"] == 0


def test_process_vision_direct_page_uses_references_fastpath_before_full_render(tmp_path, monkeypatch):
    page = _DummyPage()

    class _RefsFastpathConverter(_DummyConverter):
        def _vision_references_prefer_local_enabled(self):
            return False

        def _vision_references_column_mode_enabled(self):
            return True

        def _convert_references_page_with_column_vl(self, **kwargs):
            return "refs-md"

    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: True)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, *, page, page_index, is_references_page: [])
    monkeypatch.setattr(page_module, "_compress_png_bytes", lambda png_bytes, *, speed_config: png_bytes)
    monkeypatch.setattr(page_module, "_apply_formula_overlay", lambda converter, **kwargs: (kwargs["png_bytes"], kwargs["page_hint"], {}))

    out = page_module.process_vision_direct_page(
        _RefsFastpathConverter(),
        page=page,
        page_index=2,
        total_pages=4,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        speed_mode="normal",
        speed_config={"compress": 3},
        dpi=220,
        mat=(1, 1),
    )

    assert out == "refs-md-post"
    assert page.render_calls == []


def test_process_vision_direct_page_prefers_local_references_pipeline_before_vl(tmp_path, monkeypatch):
    page = _DummyPage()
    called = {"local": 0, "refs_vl": 0}

    class _LocalRefsConverter(_DummyConverter):
        def _vision_references_prefer_local_enabled(self):
            return True

        def _vision_references_column_mode_enabled(self):
            return True

        def _process_page(self, page, *, page_index, pdf_path, assets_dir):
            called["local"] += 1
            return "# References\n\n[1] Local ref."

        def _convert_references_page_with_column_vl(self, **kwargs):
            called["refs_vl"] += 1
            return "refs-md"

    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: True)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, *, page, page_index, is_references_page: [])

    out = page_module.process_vision_direct_page(
        _LocalRefsConverter(),
        page=page,
        page_index=2,
        total_pages=4,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        speed_mode="normal",
        speed_config={"compress": 3},
        dpi=220,
        mat=(1, 1),
    )

    assert out == "# References\n\n[1] Local ref."
    assert called["local"] == 1
    assert called["refs_vl"] == 0
    assert page.render_calls == []


def test_process_vision_direct_page_falls_back_to_vl_when_local_references_pipeline_is_empty(tmp_path, monkeypatch):
    page = _DummyPage()
    called = {"local": 0, "refs_vl": 0}

    class _FallbackRefsConverter(_DummyConverter):
        def _vision_references_prefer_local_enabled(self):
            return True

        def _vision_references_column_mode_enabled(self):
            return True

        def _process_page(self, page, *, page_index, pdf_path, assets_dir):
            called["local"] += 1
            return ""

        def _convert_references_page_with_column_vl(self, **kwargs):
            called["refs_vl"] += 1
            return "refs-md"

    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: True)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, *, page, page_index, is_references_page: [])
    monkeypatch.setattr(page_module, "_compress_png_bytes", lambda png_bytes, *, speed_config: png_bytes)
    monkeypatch.setattr(page_module, "_apply_formula_overlay", lambda converter, **kwargs: (kwargs["png_bytes"], kwargs["page_hint"], {}))

    out = page_module.process_vision_direct_page(
        _FallbackRefsConverter(),
        page=page,
        page_index=2,
        total_pages=4,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        speed_mode="normal",
        speed_config={"compress": 3},
        dpi=220,
        mat=(1, 1),
    )

    assert out == "refs-md-post"
    assert called["local"] == 1
    assert called["refs_vl"] == 1
    assert page.render_calls == []


def test_process_vision_direct_page_prefers_local_pipeline_for_large_top_figure_pages(tmp_path, monkeypatch):
    class _FigureHeavyPage(_DummyPage):
        def get_text(self, mode: str):
            if mode == "text":
                return ""
            if mode == "dict":
                return {
                    "blocks": [
                        {
                            "bbox": (60, 540, 280, 590),
                            "lines": [{"spans": [{"text": "absence of labels. Notably, structures exhibit positive and negative interference contrast."}]}],
                        },
                        {
                            "bbox": (300, 545, 560, 600),
                            "lines": [{"spans": [{"text": "To benchmark the performance against conventional confocal iSCAT, we compared the same region."}]}],
                        },
                    ]
                }
            raise AssertionError(mode)

    called = {"local": 0, "vision": 0}

    class _LocalRouteConverter(_DummyConverter):
        def _process_page(self, page, *, page_index, pdf_path, assets_dir):
            called["local"] += 1
            return "local-md"

        def _convert_page_with_vision_guardrails(self, **kwargs):
            called["vision"] += 1
            return "raw-md"

    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: False)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, *, page, page_index, is_references_page: [])
    monkeypatch.setattr(
        page_module,
        "_extract_page_visual_assets",
        lambda converter, *, page, page_index, assets_dir, dpi: (
            ["page_6_fig_1.png"],
            {"page_6_fig_1.png": {"fig_no": 3}},
            [page_module.fitz.Rect(71, 96, 531.2, 512.4)],
        ),
    )

    out = page_module.process_vision_direct_page(
        _LocalRouteConverter(),
        page=_FigureHeavyPage(),
        page_index=5,
        total_pages=8,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        speed_mode="normal",
        speed_config={"compress": 3},
        dpi=220,
        mat=(1, 1),
    )

    assert out == "local-md"
    assert called["local"] == 1
    assert called["vision"] == 0


def test_process_vision_direct_page_keeps_whole_page_vl_for_large_top_figure_pages_when_math_follows(tmp_path, monkeypatch):
    class _FigureWithMathBelowPage(_DummyPage):
        def get_text(self, mode: str):
            if mode == "text":
                return ""
            if mode == "dict":
                return {
                    "blocks": [
                        {
                            "bbox": (60, 540, 280, 590),
                            "lines": [{"spans": [{"text": "absence of labels. Notably, structures exhibit positive and negative interference contrast."}]}],
                        },
                        {
                            "bbox": (300, 545, 560, 600),
                            "lines": [{"spans": [{"text": "I = |obj * (h_det * h_ill)|^2 = 4pi/lambda + Delta phi"}]}],
                        },
                    ]
                }
            raise AssertionError(mode)

    called = {"local": 0, "vision": 0}

    class _MathPageConverter(_DummyConverter):
        def _process_page(self, page, *, page_index, pdf_path, assets_dir):
            called["local"] += 1
            return "local-md"

        def _convert_page_with_vision_guardrails(self, **kwargs):
            called["vision"] += 1
            return "raw-md"

    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: False)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, *, page, page_index, is_references_page: [])
    monkeypatch.setattr(
        page_module,
        "_extract_page_visual_assets",
        lambda converter, *, page, page_index, assets_dir, dpi: (
            ["page_3_fig_1.png"],
            {"page_3_fig_1.png": {"fig_no": 1}},
            [page_module.fitz.Rect(71, 96, 531.2, 512.4)],
        ),
    )
    monkeypatch.setattr(page_module, "_compress_png_bytes", lambda png_bytes, *, speed_config: png_bytes)
    monkeypatch.setattr(page_module, "_build_page_hint", lambda converter, *, page_index, is_references_page, image_names, figure_meta_by_asset: "hint")
    monkeypatch.setattr(
        page_module,
        "_apply_formula_overlay",
        lambda converter, *, png_bytes, page, page_index, page_w, page_h, dpi, is_references_page, page_hint: (png_bytes, page_hint, {}),
    )

    out = page_module.process_vision_direct_page(
        _MathPageConverter(),
        page=_FigureWithMathBelowPage(),
        page_index=2,
        total_pages=8,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        speed_mode="normal",
        speed_config={"compress": 3},
        dpi=220,
        mat=(1, 1),
    )

    assert out == "raw-md-post"
    assert called["local"] == 0
    assert called["vision"] == 1


def test_extract_page_visual_assets_skips_expensive_analysis_when_no_visual_rects(tmp_path, monkeypatch):
    class _AssetConverter:
        def __init__(self):
            self.caption_calls = 0
            self.line_box_calls = 0

        def _extract_page_figure_caption_candidates(self, page):
            self.caption_calls += 1
            return [{"bbox": (0, 0, 10, 10)}]

        def _collect_page_text_line_boxes(self, page):
            self.line_box_calls += 1
            return [(0, 0, 10, 10)]

        def _split_visual_rects_by_internal_captions(self, *, page, visual_rects, caption_candidates):
            return visual_rects

        def _match_figure_entries_with_captions(self, *, page, figure_entries, caption_candidates):
            return figure_entries

        def _persist_page_figure_metadata(self, *, assets_dir, page_index, figure_entries):
            return {}

        def _expanded_visual_crop_rect(self, *, rect, page_w, page_h, is_full_width, line_boxes):
            return rect

    converter = _AssetConverter()
    monkeypatch.setattr(page_module, "_collect_visual_rects", lambda page: [])

    image_names, figure_meta_by_asset, visual_rects = page_module._extract_page_visual_assets(
        converter,
        page=_DummyPage(),
        page_index=0,
        assets_dir=tmp_path,
        dpi=220,
    )

    assert image_names == []
    assert figure_meta_by_asset == {}
    assert visual_rects == []
    assert converter.caption_calls == 0
    assert converter.line_box_calls == 0


def test_process_vision_direct_page_caps_plain_middle_page_token_budget(tmp_path, monkeypatch):
    captured = {}

    class _TokenCaptureConverter(_DummyConverter):
        def _convert_page_with_vision_guardrails(self, **kwargs):
            captured.update(kwargs)
            return "raw-md"

    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: False)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, *, page, page_index, is_references_page: [])
    monkeypatch.setattr(page_module, "_extract_page_visual_assets", lambda converter, *, page, page_index, assets_dir, dpi: ([], {}, []))
    monkeypatch.setattr(page_module, "_compress_png_bytes", lambda png_bytes, *, speed_config: png_bytes)
    monkeypatch.setattr(page_module, "_build_page_hint", lambda converter, *, page_index, is_references_page, image_names, figure_meta_by_asset: "")
    monkeypatch.setattr(
        page_module,
        "_apply_formula_overlay",
        lambda converter, *, png_bytes, page, page_index, page_w, page_h, dpi, is_references_page, page_hint: (png_bytes, page_hint, {}),
    )

    out = page_module.process_vision_direct_page(
        _TokenCaptureConverter(),
        page=_DummyPage(),
        page_index=1,
        total_pages=4,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        speed_mode="normal",
        speed_config={"compress": 3},
        dpi=220,
        mat=(1, 1),
    )

    assert out == "raw-md-post"
    assert captured["max_tokens_override"] == 2816


def test_process_vision_direct_page_caps_deeper_plain_body_page_more_aggressively(tmp_path, monkeypatch):
    captured = {}

    class _TokenCaptureConverter(_DummyConverter):
        def _convert_page_with_vision_guardrails(self, **kwargs):
            captured.update(kwargs)
            return "raw-md"

    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: False)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, *, page, page_index, is_references_page: [])
    monkeypatch.setattr(page_module, "_extract_page_visual_assets", lambda converter, *, page, page_index, assets_dir, dpi: ([], {}, []))
    monkeypatch.setattr(page_module, "_compress_png_bytes", lambda png_bytes, *, speed_config: png_bytes)
    monkeypatch.setattr(page_module, "_build_page_hint", lambda converter, *, page_index, is_references_page, image_names, figure_meta_by_asset: "")
    monkeypatch.setattr(
        page_module,
        "_apply_formula_overlay",
        lambda converter, *, png_bytes, page, page_index, page_w, page_h, dpi, is_references_page, page_hint: (png_bytes, page_hint, {}),
    )

    out = page_module.process_vision_direct_page(
        _TokenCaptureConverter(),
        page=_DummyPage(),
        page_index=2,
        total_pages=5,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        speed_mode="normal",
        speed_config={"compress": 3},
        dpi=220,
        mat=(1, 1),
    )

    assert out == "raw-md-post"
    assert captured["max_tokens_override"] == 2560


class _TwoColumnFigurePage(_DummyPage):
    def get_text(self, mode: str):
        if mode == "text":
            return ""
        if mode == "dict":
            return {
                "blocks": [
                    {
                        "bbox": (56.7, 88.9, 290.7, 732.2),
                        "lines": [{"spans": [{"text": "left column continuation paragraph " * 5}]}],
                    },
                    {
                        "bbox": (304.7, 85.7, 538.8, 493.2),
                        "lines": [{"spans": [{"text": "Results Principle of interferometric ISM (iISM) " * 4}]}],
                    },
                    {
                        "bbox": (339.8, 517.4, 501.6, 533.1),
                        "lines": [{"spans": [{"text": "equation block"}]}],
                    },
                ]
            }
        raise AssertionError(mode)


def test_build_layout_page_hint_warns_about_two_column_and_midpage_full_width_figure():
    page = _TwoColumnFigurePage()
    visual_rects = [page_module.fitz.Rect(70, 200, 540, 500)]

    hint = page_module._build_layout_page_hint(page=page, visual_rects=visual_rects)

    assert "two-column" in hint
    assert "left column completely before starting the right column" in hint
    assert "figure-internal panel letters" in hint
    assert "figure and its caption as one unit" in hint


def test_process_vision_direct_page_caps_light_plain_page_more_aggressively(tmp_path, monkeypatch):
    captured = {}

    class _LightTextPage(_DummyPage):
        def get_text(self, mode: str):
            assert mode == "text"
            return "Short body text. " * 40

    class _TokenCaptureConverter(_DummyConverter):
        def _convert_page_with_vision_guardrails(self, **kwargs):
            captured.update(kwargs)
            return "raw-md"

    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: False)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, *, page, page_index, is_references_page: [])
    monkeypatch.setattr(page_module, "_extract_page_visual_assets", lambda converter, *, page, page_index, assets_dir, dpi: ([], {}, []))
    monkeypatch.setattr(page_module, "_compress_png_bytes", lambda png_bytes, *, speed_config: png_bytes)
    monkeypatch.setattr(page_module, "_build_page_hint", lambda converter, *, page_index, is_references_page, image_names, figure_meta_by_asset: "")
    monkeypatch.setattr(
        page_module,
        "_apply_formula_overlay",
        lambda converter, *, png_bytes, page, page_index, page_w, page_h, dpi, is_references_page, page_hint: (png_bytes, page_hint, {}),
    )

    out = page_module.process_vision_direct_page(
        _TokenCaptureConverter(),
        page=_LightTextPage(),
        page_index=2,
        total_pages=5,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        speed_mode="normal",
        speed_config={"compress": 3},
        dpi=220,
        mat=(1, 1),
    )

    assert out == "raw-md-post"
    assert captured["max_tokens_override"] == 2048


def test_classify_plain_body_text_density_distinguishes_light_and_dense_pages():
    class _Page:
        def __init__(self, text: str):
            self._text = text

        def get_text(self, mode: str):
            assert mode == "text"
            return self._text

    assert page_module._classify_plain_body_text_density(_Page("short " * 50)) == ("light", len(("short " * 50).strip()))
    density, count = page_module._classify_plain_body_text_density(_Page("longer body text " * 400))
    assert density == "dense"
    assert count > 4200


def test_choose_page_render_dpi_lowers_plain_middle_body_pages(monkeypatch):
    converter = SimpleNamespace(_vision_formula_overlay_enabled=lambda: False)
    monkeypatch.delenv("KB_PDF_VISION_DPI", raising=False)
    monkeypatch.delenv("KB_PDF_VISION_PLAIN_PAGE_DPI", raising=False)

    dpi, profile = page_module._choose_page_render_dpi(
        converter,
        speed_mode="normal",
        page_index=2,
        is_references_page=False,
        image_names=[],
        visual_rects=[],
        base_dpi=220,
    )

    assert dpi == 200
    assert profile == "plain_body"


def test_process_vision_direct_page_uses_lighter_render_dpi_for_plain_body_pages(tmp_path, monkeypatch):
    page = _DummyPage()
    captured = {}

    def _capture_overlay(converter, *, png_bytes, page, page_index, page_w, page_h, dpi, is_references_page, page_hint):
        captured["overlay_dpi"] = dpi
        return png_bytes, page_hint, {}

    class _DpiCaptureConverter(_DummyConverter):
        def _vision_references_column_mode_enabled(self):
            return False

        def _convert_page_with_vision_guardrails(self, **kwargs):
            captured.update(kwargs)
            return "raw-md"

        def _vision_formula_overlay_enabled(self):
            return False

    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: False)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, *, page, page_index, is_references_page: [])
    monkeypatch.setattr(page_module, "_extract_page_visual_assets", lambda converter, *, page, page_index, assets_dir, dpi: ([], {}, []))
    monkeypatch.setattr(page_module, "_compress_png_bytes", lambda png_bytes, *, speed_config: png_bytes)
    monkeypatch.setattr(page_module, "_build_page_hint", lambda converter, *, page_index, is_references_page, image_names, figure_meta_by_asset: "")
    monkeypatch.setattr(page_module, "_apply_formula_overlay", _capture_overlay)
    monkeypatch.setattr(page_module, "fitz", SimpleNamespace(Matrix=lambda x, y: ("M", round(x, 6), round(y, 6))))
    monkeypatch.delenv("KB_PDF_VISION_DPI", raising=False)
    monkeypatch.delenv("KB_PDF_VISION_PLAIN_PAGE_DPI", raising=False)

    out = page_module.process_vision_direct_page(
        _DpiCaptureConverter(),
        page=page,
        page_index=2,
        total_pages=5,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        speed_mode="normal",
        speed_config={"compress": 3},
        dpi=220,
        mat=("BASE", 1, 1),
    )

    assert out == "raw-md-post"
    assert page.render_calls[0]["matrix"] == ("M", round(200 / 72.0, 6), round(200 / 72.0, 6))
    assert captured["overlay_dpi"] == 200
