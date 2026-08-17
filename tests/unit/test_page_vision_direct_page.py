from __future__ import annotations

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


class _TextPage(_DummyPage):
    def __init__(self, text: str, *, references_heading_y: float | None = None):
        super().__init__()
        self._text = text
        self._references_heading_y = references_heading_y

    def get_text(self, mode: str):
        if mode == "text":
            return self._text
        if mode == "dict":
            if self._references_heading_y is None:
                return {"blocks": []}
            y0 = self._references_heading_y
            return {
                "blocks": [
                    {
                        "bbox": (40.0, y0, 180.0, y0 + 14.0),
                        "lines": [
                            {
                                "bbox": (40.0, y0, 180.0, y0 + 14.0),
                                "spans": [{"text": "REFERENCES"}],
                            }
                        ],
                    }
                ]
            }
        raise AssertionError(mode)


class _DummyConverter:
    def _mask_rects_on_png(self, png_bytes, rects, page_width, page_height):
        return png_bytes

    def _convert_page_with_vision_guardrails(self, **kwargs):
        return "raw-md"

    def _postprocess_vision_page_markdown(self, md, **kwargs):
        return md + "-post"


def test_detect_references_page_accepts_near_full_references_page():
    page = _TextPage(
        "\n".join(
            [
                "Journal running header",
                "REFERENCES",
                *[
                    f"[{idx}] A. Author and B. Writer, Journal of Imaging {idx}, 1-10 ({2013 + idx})."
                    for idx in range(1, 11)
                ],
            ]
        ),
        references_heading_y=72.0,
    )

    assert page_module._detect_references_page(page) is True


def test_detect_references_page_rejects_ssp_style_body_references_mixed_page():
    body = [
        "The reconstruction remains stable when the sensing operator satisfies the stated assumptions.",
        "The following proof uses the restricted isometry property and shows that the error is bounded.",
        "Case 1: the measurement residual is smaller than the noise term and the desired claim follows.",
        "Case 2: the remaining terms are controlled using Lemma 1 and the triangle inequality.",
        "ACKNOWLEDGMENTS",
        "The authors thank the anonymous reviewers for their useful comments and suggestions.",
        "APPENDIX",
        "Proof of Lemma 1. Let the support be partitioned into disjoint subsets of equal size.",
    ]
    references = [
        f"[{idx}] A. Author and B. Writer, Journal of Signal Processing {idx}, 1-10 (20{idx:02d})."
        for idx in range(1, 16)
    ]
    page = _TextPage(
        "\n".join([*body, "REFERENCES", *references]),
        references_heading_y=342.0,
    )

    assert page_module._detect_references_page(page) is False


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
        lambda converter, *, png_bytes, page, page_index, page_w, page_h, dpi, is_references_page, page_hint, formula_candidates=None: (png_bytes, page_hint, {"EQ1": "x+y"}),
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
    assert "[VISION_DIRECT][BUDGET] page 1:" in captured
    assert "class=figure_or_visual_heavy enabled=0" in captured
    assert "Step 5 (hints/overlay):" in captured
    assert "max_tokens=" in captured
    assert "Step 7 (postprocess):" in captured
    assert "TOTAL:" in captured


def test_ultra_fast_page_failure_rerenders_once_with_normal_profile(tmp_path, monkeypatch):
    _patch_page_pipeline(monkeypatch)
    calls: list[tuple[str, int]] = []

    class _FastFallbackConverter(_DummyConverter):
        def _convert_page_with_vision_guardrails(self, **kwargs):
            calls.append((str(kwargs["speed_mode"]), len(kwargs["png_bytes"])))
            return None if kwargs["speed_mode"] == "ultra_fast" else "normal-md"

        def _get_speed_mode_config(self, speed_mode, total_pages):
            assert speed_mode == "normal"
            return {"compress": 2, "dpi": 220, "max_tokens": 4096}

    out = page_module.process_vision_direct_page(
        _FastFallbackConverter(),
        page=_DummyPage(),
        page_index=0,
        total_pages=1,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        speed_mode="ultra_fast",
        speed_config={"compress": 5, "dpi": 150, "max_tokens": 2048},
        dpi=150,
        mat=(1, 1),
    )

    assert out == "normal-md-post"
    assert [mode for mode, _ in calls] == ["ultra_fast", "normal"]


def test_ultra_fast_successful_local_fallback_does_not_rerender_normal(tmp_path, monkeypatch):
    _patch_page_pipeline(monkeypatch)
    calls: list[str] = []

    class _LocalFallbackConverter(_DummyConverter):
        def _convert_page_with_vision_guardrails(self, **kwargs):
            calls.append(str(kwargs["speed_mode"]))
            return "local-fallback-md"

        def _get_speed_mode_config(self, speed_mode, total_pages):
            raise AssertionError("normal rerender must not run after a successful local fallback")

    out = page_module.process_vision_direct_page(
        _LocalFallbackConverter(),
        page=_DummyPage(),
        page_index=0,
        total_pages=1,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        speed_mode="ultra_fast",
        speed_config={"compress": 5, "dpi": 150, "max_tokens": 2048},
        dpi=150,
        mat=(1, 1),
    )

    assert out == "local-fallback-md-post"
    assert calls == ["ultra_fast"]


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
            return "# References\n\n[1] Local ref. Journal, 2024.\n[2] Another local ref. Journal, 2025."

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

    assert out == "# References\n\n[1] Local ref. Journal, 2024.\n[2] Another local ref. Journal, 2025."
    assert called["local"] == 1
    assert called["refs_vl"] == 0
    assert page.render_calls == []


def test_process_vision_direct_page_falls_back_when_local_references_pipeline_is_sparse(tmp_path, monkeypatch):
    page = _DummyPage()
    called = {"local": 0, "refs_vl": 0}

    class _SparseRefsConverter(_DummyConverter):
        def _vision_references_prefer_local_enabled(self):
            return True

        def _vision_references_column_mode_enabled(self):
            return True

        def _process_page(self, page, *, page_index, pdf_path, assets_dir):
            called["local"] += 1
            return "# References\n\n[1] Too short."

        def _convert_references_page_with_column_vl(self, **kwargs):
            called["refs_vl"] += 1
            return "# References\n\n[1] VL ref. Journal, 2024.\n[2] Another VL ref. Journal, 2025."

    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: True)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, *, page, page_index, is_references_page: [])
    monkeypatch.setattr(page_module, "_compress_png_bytes", lambda png_bytes, *, speed_config: png_bytes)
    monkeypatch.setattr(page_module, "_apply_formula_overlay", lambda converter, **kwargs: (kwargs["png_bytes"], kwargs["page_hint"], {}))

    out = page_module.process_vision_direct_page(
        _SparseRefsConverter(),
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

    assert out.startswith("# References")
    assert "VL ref" in out
    assert called["local"] == 1
    assert called["refs_vl"] == 1
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
        lambda converter, *, png_bytes, page, page_index, page_w, page_h, dpi, is_references_page, page_hint, formula_candidates=None: (png_bytes, page_hint, {}),
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


def test_process_vision_direct_page_skips_local_figure_route_for_source_detected_fragmented_formula(tmp_path, monkeypatch):
    class _FigureWithShortFormulaFragments(_DummyPage):
        def get_text(self, mode: str):
            if mode == "text":
                return ""
            if mode == "dict":
                return {
                    "blocks": [
                        {
                            "bbox": (60, 540, 280, 620),
                            "lines": [{"spans": [{"text": "A substantial left-column paragraph continues below the figure."}]}],
                        },
                        {
                            "bbox": (310, 600, 370, 620),
                            "lines": [{"spans": [{"text": "O l k = sigma"}]}],
                        },
                        {
                            "bbox": (380, 600, 450, 620),
                            "lines": [{"spans": [{"text": "sum w O + b"}]}],
                        },
                    ]
                }
            raise AssertionError(mode)

    called = {"local": 0, "vision": 0, "formula": 0}

    class _FormulaAwareConverter(_DummyConverter):
        def _collect_display_math_candidates(self, page, *, page_index, is_references_page):
            called["formula"] += 1
            return [(310.0, 600.0, 450.0, 620.0)]

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
            ["page_5_fig_1.png"],
            {"page_5_fig_1.png": {"fig_no": 3}},
            [page_module.fitz.Rect(52, 78, 545, 480)],
        ),
    )
    monkeypatch.setattr(page_module, "_compress_png_bytes", lambda png_bytes, *, speed_config: png_bytes)
    monkeypatch.setattr(page_module, "_build_page_hint", lambda converter, **kwargs: "hint")
    monkeypatch.setattr(
        page_module,
        "_apply_formula_overlay",
        lambda converter, **kwargs: (kwargs["png_bytes"], kwargs["page_hint"], {}),
    )

    out = page_module.process_vision_direct_page(
        _FormulaAwareConverter(),
        page=_FigureWithShortFormulaFragments(),
        page_index=4,
        total_pages=21,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        speed_mode="normal",
        speed_config={"compress": 3},
        dpi=220,
        mat=(1, 1),
    )

    assert out == "raw-md-post"
    assert called == {"local": 0, "vision": 1, "formula": 1}


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
        lambda converter, *, png_bytes, page, page_index, page_w, page_h, dpi, is_references_page, page_hint, formula_candidates=None: (png_bytes, page_hint, {}),
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
        lambda converter, *, png_bytes, page, page_index, page_w, page_h, dpi, is_references_page, page_hint, formula_candidates=None: (png_bytes, page_hint, {}),
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
        lambda converter, *, png_bytes, page, page_index, page_w, page_h, dpi, is_references_page, page_hint, formula_candidates=None: (png_bytes, page_hint, {}),
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


def test_classify_page_budget_prioritizes_source_risk_signals():
    common = {
        "page_index": 3,
        "is_references_page": False,
        "image_names": [],
        "visual_rects": [],
        "formula_candidate_count": 0,
        "plain_text_chars": 1800,
    }

    assert page_module._classify_page_budget(**{**common, "is_references_page": True}) == "references"
    assert page_module._classify_page_budget(**{**common, "formula_candidate_count": 1}) == "formula_sensitive"
    assert page_module._classify_page_budget(**{**common, "image_names": ["figure.png"]}) == "figure_or_visual_heavy"
    assert page_module._classify_page_budget(**common) == "text_dense_body"
    assert page_module._classify_page_budget(**{**common, "page_index": 0}) == "unknown"
    assert page_module._classify_page_budget(**{**common, "plain_text_chars": 0}) == "unknown"


def test_adaptive_render_policy_only_lowers_reviewed_text_body_class(monkeypatch):
    converter = SimpleNamespace(_vision_formula_overlay_enabled=lambda: True)
    monkeypatch.delenv("KB_PDF_VISION_DPI", raising=False)
    monkeypatch.delenv("KB_PDF_VISION_PLAIN_PAGE_DPI", raising=False)

    text_budget = page_module._choose_page_render_dpi(
        converter,
        speed_mode="normal",
        page_index=3,
        is_references_page=False,
        image_names=[],
        visual_rects=[],
        base_dpi=220,
        page_class="text_dense_body",
        adaptive_enabled=True,
    )
    formula_budget = page_module._choose_page_render_dpi(
        converter,
        speed_mode="normal",
        page_index=3,
        is_references_page=False,
        image_names=[],
        visual_rects=[],
        base_dpi=220,
        page_class="formula_sensitive",
        adaptive_enabled=True,
    )

    assert text_budget == (200, "adaptive_text_dense_body")
    assert formula_budget == (220, "adaptive_formula_sensitive")


def test_adaptive_page_budget_flag_defaults_off_and_accepts_explicit_enable(monkeypatch):
    monkeypatch.delenv("KB_PDF_VISION_ADAPTIVE_PAGE_BUDGETS", raising=False)
    assert page_module._adaptive_page_budgets_enabled() is False

    monkeypatch.setenv("KB_PDF_VISION_ADAPTIVE_PAGE_BUDGETS", "yes")
    assert page_module._adaptive_page_budgets_enabled() is True


def test_text_local_fastpath_flag_defaults_off_and_accepts_explicit_enable(monkeypatch):
    monkeypatch.delenv("KB_PDF_VISION_TEXT_LOCAL_FASTPATH", raising=False)
    assert page_module._vision_text_local_fastpath_enabled() is False

    monkeypatch.setenv("KB_PDF_VISION_TEXT_LOCAL_FASTPATH", "true")
    assert page_module._vision_text_local_fastpath_enabled() is True


def test_validate_local_text_markdown_accepts_source_faithful_markdown():
    source = " ".join(
        f"Observation {idx} reports measurement token{idx} with stable reconstruction evidence."
        for idx in range(1, 45)
    )

    accepted, reason, metrics = page_module._validate_local_text_markdown(
        source,
        f"## Results\n\n{source}",
    )

    assert accepted is True
    assert reason == "accepted"
    assert metrics["coverage"] > 0.99
    assert metrics["bigram_coverage"] > 0.99
    assert metrics["order_ratio"] > 0.99


def test_validate_local_text_markdown_rejects_missing_source_content():
    source = " ".join(
        f"Observation {idx} reports measurement token{idx} with stable reconstruction evidence."
        for idx in range(1, 45)
    )

    accepted, reason, metrics = page_module._validate_local_text_markdown(
        source,
        "## Results\n\nOnly a small fragment was retained from the source page.",
    )

    assert accepted is False
    assert reason == "output_too_short"
    assert metrics["output_tokens"] < 100


def test_source_heading_prefix_is_promoted_and_required_for_local_acceptance():
    page = SimpleNamespace(
        get_text=lambda mode: {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {"text": "Fabrication of DBRs", "font": "Journal-Semibold", "flags": 16},
                                {"text": "The deposited layers form the reflector.", "font": "Journal-Regular", "flags": 0},
                            ]
                        }
                    ],
                }
            ]
        }
    )
    headings = page_module._source_bold_prefix_headings(page)
    source = " ".join(
        ["Fabrication of DBRs The deposited layers form the reflector."]
        + [f"Measurement {idx} records stable optical response token{idx}." for idx in range(1, 40)]
    )
    markdown, promoted = page_module._promote_source_headings(source, headings)

    accepted, reason, _ = page_module._validate_local_text_markdown(
        source,
        markdown,
        required_headings=headings,
    )

    assert headings == ["Fabrication of DBRs"]
    assert promoted == 1
    assert markdown.startswith("### Fabrication of DBRs\n\n")
    assert accepted is True
    assert reason == "accepted"


def test_source_heading_candidates_exclude_repeated_noise():
    page = SimpleNamespace(
        get_text=lambda mode: {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {"text": "Nature Photonics", "font": "Journal-Semibold", "flags": 16},
                            ]
                        }
                    ],
                }
            ]
        }
    )

    headings = page_module._source_bold_prefix_headings(
        page,
        noise_texts={"NATure PHoTonIcs"},
    )

    assert headings == []


def test_reference_continuation_guard_requires_ordered_line_start_entries():
    references_page = _TextPage(
        "\n".join(
            [
                "[51] Xin Yuan et al. Plug-and-play algorithms for snapshot imaging.",
                "[52] Yang Liu et al. Video snapshot compressive imaging.",
                "[53] Yu-Jie Yuan et al. Neural radiance field editing.",
                "[54] Richard Zhang et al. Perceptual image metrics.",
            ]
        )
    )
    body_page = _TextPage(
        "The method follows prior work [51] and compares three settings.\n"
        "[1] First ablation setting\n[2] Second ablation setting\n"
        "The remaining body contains ordinary prose."
    )

    assert page_module._page_looks_like_reference_continuation(references_page) is True
    assert page_module._page_looks_like_reference_continuation(body_page) is False


def test_text_local_eligibility_rejects_quality_risks(monkeypatch):
    converter = SimpleNamespace(_process_page_local_only=lambda *args, **kwargs: "markdown")
    page = _TextPage("body " * 400)
    monkeypatch.setattr(page_module, "_page_has_table_risk", lambda page: False)

    assert page_module._local_text_fastpath_eligibility(
        converter,
        page=page,
        speed_mode="full_llm",
        page_class="text_dense_body",
        plain_text_chars=2000,
    ) == (False, "speed_mode")
    assert page_module._local_text_fastpath_eligibility(
        converter,
        page=page,
        speed_mode="normal",
        page_class="formula_sensitive",
        plain_text_chars=2000,
    ) == (False, "page_class")
    assert page_module._local_text_fastpath_eligibility(
        converter,
        page=page,
        speed_mode="normal",
        page_class="text_dense_body",
        plain_text_chars=500,
    ) == (False, "source_too_short")

    references_page = _TextPage(
        "\n".join(
            f"[{idx}] Author {idx}. Complete reference title and publication details."
            for idx in range(51, 55)
        )
    )
    assert page_module._local_text_fastpath_eligibility(
        converter,
        page=references_page,
        speed_mode="normal",
        page_class="text_dense_body",
        plain_text_chars=2000,
    ) == (False, "references_continuation")

    monkeypatch.setattr(page_module, "_page_has_table_risk", lambda page: True)
    assert page_module._local_text_fastpath_eligibility(
        converter,
        page=page,
        speed_mode="normal",
        page_class="text_dense_body",
        plain_text_chars=2000,
    ) == (False, "table_risk")


def test_adaptive_text_page_allows_token_cap_with_layout_hint():
    common = {
        "speed_mode": "normal",
        "page_index": 3,
        "is_references_page": False,
        "page_hint": "The page uses a two-column reading order.",
        "image_names": [],
        "visual_rects": [],
        "formula_placeholders": {},
        "plain_text_density": "medium",
        "page_class": "text_dense_body",
    }

    assert page_module._choose_page_max_tokens_override(**common, adaptive_enabled=False) is None
    assert page_module._choose_page_max_tokens_override(**common, adaptive_enabled=True) == 2560


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


def test_full_llm_keeps_quality_first_plain_page_budget(monkeypatch):
    converter = SimpleNamespace(_vision_formula_overlay_enabled=lambda: False)
    monkeypatch.delenv("KB_PDF_VISION_DPI", raising=False)
    monkeypatch.delenv("KB_PDF_VISION_PLAIN_PAGE_DPI", raising=False)

    dpi, profile = page_module._choose_page_render_dpi(
        converter,
        speed_mode="full_llm",
        page_index=2,
        is_references_page=False,
        image_names=[],
        visual_rects=[],
        base_dpi=220,
    )
    max_tokens = page_module._choose_page_max_tokens_override(
        speed_mode="full_llm",
        page_index=2,
        is_references_page=False,
        page_hint="",
        image_names=[],
        visual_rects=[],
        formula_placeholders={},
        plain_text_density="light",
    )

    assert (dpi, profile) == (220, "base")
    assert max_tokens is None


def test_process_vision_direct_page_uses_lighter_render_dpi_for_plain_body_pages(tmp_path, monkeypatch):
    page = _DummyPage()
    captured = {}

    def _capture_overlay(
        converter,
        *,
        png_bytes,
        page,
        page_index,
        page_w,
        page_h,
        dpi,
        is_references_page,
        page_hint,
        formula_candidates=None,
    ):
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


def test_process_vision_direct_page_adaptive_policy_uses_source_formula_scan(tmp_path, monkeypatch, capsys):
    page = _TextPage("A plain middle-body paragraph with enough source text. " * 80)
    captured = {}

    class _AdaptiveConverter(_DummyConverter):
        def _vision_formula_overlay_enabled(self):
            return True

        def _collect_display_math_candidates(self, page, *, page_index, is_references_page):
            return []

        def _convert_page_with_vision_guardrails(self, **kwargs):
            captured.update(kwargs)
            return "raw-md"

    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: False)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, **kwargs: [])
    monkeypatch.setattr(page_module, "_extract_page_visual_assets", lambda converter, **kwargs: ([], {}, []))
    monkeypatch.setattr(page_module, "_compress_png_bytes", lambda png_bytes, **kwargs: png_bytes)
    monkeypatch.setattr(page_module, "_build_page_hint", lambda converter, **kwargs: "")
    monkeypatch.setattr(
        page_module,
        "_apply_formula_overlay",
        lambda converter, **kwargs: (kwargs["png_bytes"], kwargs["page_hint"], {}),
    )
    monkeypatch.setattr(page_module, "fitz", SimpleNamespace(Matrix=lambda x, y: ("M", round(x, 6), round(y, 6))))
    monkeypatch.setenv("KB_PDF_VISION_ADAPTIVE_PAGE_BUDGETS", "1")
    monkeypatch.setenv("KB_PDF_STAGE_TIMINGS", "1")
    monkeypatch.delenv("KB_PDF_VISION_DPI", raising=False)
    monkeypatch.delenv("KB_PDF_VISION_PLAIN_PAGE_DPI", raising=False)

    out = page_module.process_vision_direct_page(
        _AdaptiveConverter(),
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
    assert captured["max_tokens_override"] == 2560
    output = capsys.readouterr().out
    assert "class=text_dense_body enabled=1 dpi=200 base_dpi=220" in output


def test_process_vision_direct_page_adaptive_policy_keeps_unknown_formula_signal_at_base_budget(
    tmp_path,
    monkeypatch,
):
    page = _TextPage("A plain middle-body paragraph with enough source text. " * 80)
    captured = {}

    class _NoFormulaSignalConverter(_DummyConverter):
        def _vision_formula_overlay_enabled(self):
            return False

        def _convert_page_with_vision_guardrails(self, **kwargs):
            captured.update(kwargs)
            return "raw-md"

    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: False)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, **kwargs: [])
    monkeypatch.setattr(page_module, "_extract_page_visual_assets", lambda converter, **kwargs: ([], {}, []))
    monkeypatch.setattr(page_module, "_compress_png_bytes", lambda png_bytes, **kwargs: png_bytes)
    monkeypatch.setattr(page_module, "_build_page_hint", lambda converter, **kwargs: "")
    monkeypatch.setattr(
        page_module,
        "_apply_formula_overlay",
        lambda converter, **kwargs: (kwargs["png_bytes"], kwargs["page_hint"], {}),
    )
    monkeypatch.setattr(page_module, "fitz", SimpleNamespace(Matrix=lambda x, y: ("M", round(x, 6), round(y, 6))))
    monkeypatch.setenv("KB_PDF_VISION_ADAPTIVE_PAGE_BUDGETS", "1")
    monkeypatch.delenv("KB_PDF_VISION_DPI", raising=False)
    monkeypatch.delenv("KB_PDF_VISION_PLAIN_PAGE_DPI", raising=False)

    out = page_module.process_vision_direct_page(
        _NoFormulaSignalConverter(),
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
    assert page.render_calls[0]["matrix"] == ("BASE", 1, 1)
    assert captured["max_tokens_override"] is None


def test_process_vision_direct_page_accepts_verified_text_local_fastpath(tmp_path, monkeypatch, capsys):
    source = " ".join(
        f"Observation {idx} reports measurement token{idx} with stable reconstruction evidence."
        for idx in range(1, 55)
    )
    page = _TextPage(source)
    calls = {"local": 0, "vision": 0}

    class _TextLocalConverter(_DummyConverter):
        def _vision_formula_overlay_enabled(self):
            return False

        def _collect_display_math_candidates(self, page, *, page_index, is_references_page):
            return []

        def _process_page_local_only(self, page, *, page_index, pdf_path, assets_dir):
            calls["local"] += 1
            return f"## Results\n\n{source}"

        def _convert_page_with_vision_guardrails(self, **kwargs):
            calls["vision"] += 1
            raise AssertionError("accepted local page must not call vision")

    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: False)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, **kwargs: [])
    monkeypatch.setattr(page_module, "_extract_page_visual_assets", lambda converter, **kwargs: ([], {}, []))
    monkeypatch.setattr(page_module, "_page_has_table_risk", lambda page: False)
    monkeypatch.setenv("KB_PDF_VISION_TEXT_LOCAL_FASTPATH", "1")
    monkeypatch.setenv("KB_PDF_STAGE_TIMINGS", "1")

    out = page_module.process_vision_direct_page(
        _TextLocalConverter(),
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

    assert out == f"## Results\n\n{source}"
    assert calls == {"local": 1, "vision": 0}
    assert page.render_calls == []
    output = capsys.readouterr().out
    assert "[VISION_DIRECT][TEXT_LOCAL] page 3: accepted=1 reason=accepted" in output
    assert "dpi=0 base_dpi=220" in output


def test_process_vision_direct_page_rejects_sparse_local_output_and_uses_vision(tmp_path, monkeypatch, capsys):
    source = " ".join(
        f"Observation {idx} reports measurement token{idx} with stable reconstruction evidence."
        for idx in range(1, 55)
    )
    page = _TextPage(source)
    calls = {"local": 0, "vision": 0}

    class _SparseLocalConverter(_DummyConverter):
        def _vision_formula_overlay_enabled(self):
            return False

        def _collect_display_math_candidates(self, page, *, page_index, is_references_page):
            return []

        def _process_page_local_only(self, page, *, page_index, pdf_path, assets_dir):
            calls["local"] += 1
            return "## Results\n\nA sparse fragment."

        def _convert_page_with_vision_guardrails(self, **kwargs):
            calls["vision"] += 1
            return "vision-md"

    monkeypatch.setattr(page_module, "_detect_references_page", lambda page: False)
    monkeypatch.setattr(page_module, "_collect_metadata_rects", lambda converter, **kwargs: [])
    monkeypatch.setattr(page_module, "_extract_page_visual_assets", lambda converter, **kwargs: ([], {}, []))
    monkeypatch.setattr(page_module, "_page_has_table_risk", lambda page: False)
    monkeypatch.setattr(page_module, "_compress_png_bytes", lambda png_bytes, **kwargs: png_bytes)
    monkeypatch.setattr(page_module, "_build_page_hint", lambda converter, **kwargs: "")
    monkeypatch.setattr(
        page_module,
        "_apply_formula_overlay",
        lambda converter, **kwargs: (kwargs["png_bytes"], kwargs["page_hint"], {}),
    )
    monkeypatch.setattr(page_module, "fitz", SimpleNamespace(Matrix=lambda x, y: ("M", x, y)))
    monkeypatch.setenv("KB_PDF_VISION_TEXT_LOCAL_FASTPATH", "1")

    out = page_module.process_vision_direct_page(
        _SparseLocalConverter(),
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

    assert out == "vision-md-post"
    assert calls == {"local": 1, "vision": 1}
    assert len(page.render_calls) == 1
    assert "accepted=0 reason=output_too_short" in capsys.readouterr().out
