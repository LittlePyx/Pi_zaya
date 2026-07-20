from pathlib import Path

import pytest

from kb.converter.config import ConvertConfig
from kb.converter.pipeline import PDFConverter
import kb.converter.page_local_pipeline as page_local_pipeline


class _GeometryPage:
    def __init__(self, lines):
        import fitz

        self.rect = fitz.Rect(0, 0, 600, 800)
        self._lines = lines

    def get_text(self, mode: str):
        assert mode == "dict"
        return {
            "blocks": [
                {
                    "lines": [
                        {
                            "bbox": bbox,
                            "spans": [{"text": text}],
                        }
                    ]
                }
                for bbox, text in self._lines
            ]
        }


def test_copyright_footer_is_not_formula_evidence():
    footer = "© 2024 Wiley-VCH GmbH 2401397 (17 of 21)"

    assert PDFConverter._looks_like_overlay_math_line(footer) is False
    assert PDFConverter._is_display_math_candidate_text(footer) is False


def test_formula_candidates_at_same_height_stay_in_their_pdf_columns(tmp_path):
    converter = _make_converter(tmp_path)
    page = _GeometryPage(
        [
            ((60, 280, 230, 305), "L = sum_i x_i"),
            ((260, 282, 286, 304), "(12)"),
            ((330, 280, 500, 305), "E = sum_i y_i"),
            ((550, 282, 578, 304), "(15)"),
        ]
    )

    candidates = converter._collect_display_math_candidates(
        page,
        page_index=5,
        is_references_page=False,
    )

    assert len(candidates) == 2
    assert [candidate["column_lane"] for candidate in candidates] == ["left", "right"]
    assert float(candidates[0]["rect"].x1) < 300
    assert float(candidates[1]["rect"].x0) > 300


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
    def __init__(self, outputs):
        self._outputs = list(outputs)
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
        self.calls.append(
            {
                "page_number": page_number,
                "total_pages": total_pages,
                "hint": hint,
                "speed_mode": speed_mode,
                "is_references_page": is_references_page,
            }
        )
        if self._outputs:
            return self._outputs.pop(0)
        return None


def test_fragmented_math_detector_flags_split_equation():
    broken = """
$$
\\frac{N}{T}
$$

N

$$
\\sum^{N}
$$

( DNN ( u n
), u n
)(16)
"""
    assert PDFConverter._looks_fragmented_math_output(broken) is True


def test_fragmented_math_detector_flags_mixed_complete_plus_shards():
    mixed = """
$$
L = \\text{loss\\_func} \\sum_{n=1}^{M}\\left(DNN(u_n^*,v_n^*)\\right)
$$

M

(DNN(u *), v *)

)(15)

$$
n = 1
$$
"""
    assert PDFConverter._looks_fragmented_math_output(mixed) is True


def test_fragmented_math_detector_accepts_coherent_equation():
    clean = "$$L = \\text{loss\\_func} \\sum_{n=1}^{N}(DNN(u_n),u_n) \\tag{16}$$"
    assert PDFConverter._looks_fragmented_math_output(clean) is False


def test_fragmented_math_detector_rejects_prose_or_citations_inside_display_math():
    citation = """
Before text.

$$
olution of reconstructed images from 128\\times128 to 256\\times256.[66]
$$

After text.
"""
    explanation = """
Before text.

$$
O^l refers to the output of the kth unit and denotes the previous layer.
$$

After text.
"""

    assert PDFConverter._looks_fragmented_math_output(citation) is True
    assert PDFConverter._looks_fragmented_math_output(explanation) is True


def test_fragmented_math_detector_rejects_lpr_variable_definition_fragments():
    broken = r"""
$$
(\sum O_l = \sigma w_{kj}^{l-1}O_j^{l-1}
$$

$$
)O^{l-1} + b(11)_j
$$

$O_l$

$$
O^{l}_{k} \text{ refers to the output of the kth unit in the lth layer}, O^{l-1}_{j}
$$

j
"""

    assert PDFConverter._looks_fragmented_math_output(broken) is True


def test_guardrails_retry_then_fallback_when_still_fragmented(tmp_path, monkeypatch):
    broken = """
$$
\\frac{N}{T}
$$

N

$$
\\sum^{N}
$$

( DNN ( u n
), u n
)(16)
"""
    converter = _make_converter(tmp_path)
    dummy = _DummyLLMWorker([broken, broken])
    converter.llm_worker = dummy
    monkeypatch.setenv("KB_PDF_VISION_FRAGMENT_FALLBACK", "1")

    monkeypatch.setattr(
        converter,
        "_process_page",
        lambda page, page_index, pdf_path, assets_dir: "FALLBACK_OK",
    )

    out = converter._convert_page_with_vision_guardrails(
        png_bytes=b"fake",
        page=object(),
        page_index=5,
        total_pages=12,
        page_hint="",
        speed_mode="normal",
        is_references_page=False,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        formula_placeholders={"[[EQ_1]]": "$$x=y$$"},
    )

    assert out == "FALLBACK_OK"
    assert len(dummy.calls) == 2


def test_guardrails_accepts_retry_result_when_fixed(tmp_path, monkeypatch):
    broken = """
$$
\\frac{N}{T}
$$

N

$$
\\sum^{N}
$$

( DNN ( u n
), u n
)(16)
"""
    clean = "$$L = \\text{loss\\_func} \\sum_{n=1}^{N}(DNN(u_n),u_n) \\tag{16}$$"
    converter = _make_converter(tmp_path)
    dummy = _DummyLLMWorker([broken, clean])
    converter.llm_worker = dummy

    monkeypatch.setattr(
        converter,
        "_process_page",
        lambda page, page_index, pdf_path, assets_dir: (_ for _ in ()).throw(
            AssertionError("fallback should not run when retry succeeded")
        ),
    )

    out = converter._convert_page_with_vision_guardrails(
        png_bytes=b"fake",
        page=object(),
        page_index=2,
        total_pages=8,
        page_hint="",
        speed_mode="normal",
        is_references_page=False,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        formula_placeholders={"[[EQ_1]]": "$$x=y$$"},
    )

    assert out == clean
    assert len(dummy.calls) == 2


def test_guardrails_rejects_fragmented_layout_crop_before_returning_it(tmp_path, monkeypatch):
    broken = """
$$
\\frac{N}{T}
$$

N

$$
\\sum^{N}
$$

( DNN ( u n
), u n
)(16)
"""
    clean = "$$L = \\sum_{n=1}^{N} x_n \\tag{15}$$"
    converter = _make_converter(tmp_path)
    dummy = _DummyLLMWorker([clean])
    converter.llm_worker = dummy
    monkeypatch.setattr(converter, "_convert_page_with_layout_crops", lambda **kwargs: broken)

    out = converter._convert_page_with_vision_guardrails(
        png_bytes=b"fake",
        page=object(),
        page_index=5,
        total_pages=12,
        page_hint="",
        speed_mode="normal",
        is_references_page=False,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        image_names=["figure.png"],
        visual_rects=[],
        formula_placeholders={"[[EQ_1]]": "$$x=y$$"},
    )

    assert out == clean
    assert len(dummy.calls) == 1


def test_guardrails_skip_math_fragment_check_for_references_page(tmp_path, monkeypatch):
    broken = """
$$
\\sum^{N}
$$

N
"""
    converter = _make_converter(tmp_path)
    dummy = _DummyLLMWorker([broken])
    converter.llm_worker = dummy

    monkeypatch.setattr(
        converter,
        "_process_page",
        lambda page, page_index, pdf_path, assets_dir: (_ for _ in ()).throw(
            AssertionError("references page should not enter fallback in this test")
        ),
    )

    out = converter._convert_page_with_vision_guardrails(
        png_bytes=b"fake",
        page=object(),
        page_index=1,
        total_pages=6,
        page_hint="references page",
        speed_mode="normal",
        is_references_page=True,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        formula_placeholders={"[[EQ_1]]": "$$x=y$$"},
    )

    assert out == broken


def test_guardrails_do_not_retry_fragmented_output_without_source_formula_evidence(tmp_path, monkeypatch):
    broken = "$$N$$\n\nN\n\n$$\\sum^N$$\n\n)(15)"
    converter = _make_converter(tmp_path)
    dummy = _DummyLLMWorker([broken, "SHOULD_NOT_BE_USED"])
    converter.llm_worker = dummy
    monkeypatch.setattr(converter, "_collect_display_math_candidates", lambda *args, **kwargs: [])

    out = converter._convert_page_with_vision_guardrails(
        png_bytes=b"fake",
        page=object(),
        page_index=3,
        total_pages=8,
        page_hint="",
        speed_mode="normal",
        is_references_page=False,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
    )

    assert out == broken
    assert len(dummy.calls) == 1


def test_ultra_fast_empty_output_is_capped_to_one_retry(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    dummy = _DummyLLMWorker([None, None, "late-result-must-not-be-used"])
    converter.llm_worker = dummy
    monkeypatch.setenv("KB_PDF_VISION_EMPTY_RETRY", "5")
    monkeypatch.setenv("KB_PDF_VISION_EMPTY_RETRY_BACKOFF_S", "0")
    monkeypatch.setattr(
        converter,
        "_process_page",
        lambda page, page_index, pdf_path, assets_dir: "LOCAL_FALLBACK",
    )

    out = converter._convert_page_with_vision_guardrails(
        png_bytes=b"fake",
        page=object(),
        page_index=3,
        total_pages=8,
        page_hint="",
        speed_mode="ultra_fast",
        is_references_page=False,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
    )

    assert out == "LOCAL_FALLBACK"
    assert len(dummy.calls) == 2


def test_ultra_fast_timeout_uses_local_only_extraction_fallback(tmp_path, monkeypatch):
    converter = _make_converter(tmp_path)
    dummy = _DummyLLMWorker([None])
    dummy.get_last_vl_error_code = lambda: "timeout"
    converter.llm_worker = dummy
    calls = {"local_only": 0, "regular": 0}

    def _local_only(converter_arg, page, *, page_index, pdf_path, assets_dir, allow_llm_enhance, safe_complex_fallback):
        assert converter_arg is converter
        assert allow_llm_enhance is False
        assert safe_complex_fallback is True
        calls["local_only"] += 1
        return "LOCAL_ONLY_MD"

    def _regular(page, *, page_index, pdf_path, assets_dir):
        calls["regular"] += 1
        raise AssertionError("timeout fallback must not enter LLM-enhanced extraction")

    monkeypatch.setattr(page_local_pipeline, "process_page", _local_only)
    monkeypatch.setattr(converter, "_process_page", _regular)

    out = converter._convert_page_with_vision_guardrails(
        png_bytes=b"fake",
        page=object(),
        page_index=0,
        total_pages=4,
        page_hint="",
        speed_mode="ultra_fast",
        is_references_page=False,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
    )

    assert out == "LOCAL_ONLY_MD"
    assert calls == {"local_only": 1, "regular": 0}


@pytest.mark.parametrize("speed_mode", ["ultra_fast", "normal"])
@pytest.mark.parametrize("error_code", ["timeout", "rate_limited", "circuit_open"])
def test_vision_health_failures_use_safe_local_extraction_without_retry(
    tmp_path,
    monkeypatch,
    speed_mode,
    error_code,
):
    converter = _make_converter(tmp_path)
    dummy = _DummyLLMWorker([None, "SHOULD_NOT_RETRY"])
    dummy.get_last_vl_error_code = lambda: error_code
    converter.llm_worker = dummy
    calls = {"local_only": 0, "regular": 0}

    def _local_only(converter_arg, page, *, page_index, pdf_path, assets_dir, allow_llm_enhance, safe_complex_fallback):
        assert converter_arg is converter
        assert allow_llm_enhance is False
        assert safe_complex_fallback is True
        calls["local_only"] += 1
        return "LOCAL_ONLY_MD"

    def _regular(page, *, page_index, pdf_path, assets_dir):
        calls["regular"] += 1
        raise AssertionError("vision health failures must not enter LLM-enhanced extraction")

    monkeypatch.setattr(page_local_pipeline, "process_page", _local_only)
    monkeypatch.setattr(converter, "_process_page", _regular)

    out = converter._convert_page_with_vision_guardrails(
        png_bytes=b"fake",
        page=object(),
        page_index=0,
        total_pages=4,
        page_hint="",
        speed_mode=speed_mode,
        is_references_page=False,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
    )

    assert out == "LOCAL_ONLY_MD"
    assert len(dummy.calls) == 1
    assert calls == {"local_only": 1, "regular": 0}


def test_ultra_fast_math_quality_retry_uses_normal_token_policy(tmp_path, monkeypatch):
    broken = """
$$
\\frac{N}{T}
$$

N

$$
\\sum^{N}
$$

( DNN ( u n
), u n
)(16)
"""
    clean = "$$N = \\sum_{i=1}^{m} x_i \\tag{15}$$"
    converter = _make_converter(tmp_path)
    dummy = _DummyLLMWorker([broken, clean])
    converter.llm_worker = dummy

    out = converter._convert_page_with_vision_guardrails(
        png_bytes=b"fake",
        page=object(),
        page_index=3,
        total_pages=8,
        page_hint="",
        speed_mode="ultra_fast",
        is_references_page=False,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
        formula_placeholders={"[[EQ_1]]": "$$x=y$$"},
    )

    assert out == clean
    assert [call["speed_mode"] for call in dummy.calls] == ["ultra_fast", "normal"]
    assert len(dummy.calls) == 2


def test_restore_formula_placeholders_exact_and_fuzzy():
    md = "Before [[EQ_1]] middle [ EQ_2 ] after."
    mapping = {
        "[[EQ_1]]": "$$\nA=B\n$$",
        "[[EQ_2]]": "$$\nC=D\n$$",
    }
    out = PDFConverter._restore_formula_placeholders(md, mapping)
    assert "[[EQ_1]]" not in out
    # Single-bracket variant is intentionally not matched now.
    assert "[ EQ_2 ]" in out
    assert "$$\nA=B\n$$" in out
    assert "$$\nC=D\n$$" not in out


def test_restore_formula_placeholders_leaves_missing_unmodified():
    md = "Only one token: [[EQ_1]]"
    mapping = {
        "[[EQ_1]]": "$$\nA=B\n$$",
        "[[EQ_2]]": "$$\nC=D\n$$",
    }
    out = PDFConverter._restore_formula_placeholders(md, mapping)
    assert "$$\nA=B\n$$" in out
    assert "$$\nC=D\n$$" not in out


def test_restore_formula_placeholders_backslash_safe():
    md = "Math token [[EQ_1]] done."
    mapping = {
        "[[EQ_1]]": "$$\nL=\\sum_{n=1}^{N}\\mu_n\\text{ok}\n$$",
    }
    out = PDFConverter._restore_formula_placeholders(md, mapping)
    assert "\\sum_{n=1}^{N}" in out
    assert "\\mu_n" in out
    assert "\\text{ok}" in out


def test_guardrails_default_keeps_vl_output_when_fragmented(tmp_path, monkeypatch):
    broken = """
$$
\\frac{N}{T}
$$

N

$$
\\sum^{N}
$$

( DNN ( u n
), u n
)(16)
"""
    converter = _make_converter(tmp_path)
    dummy = _DummyLLMWorker([broken, broken])
    converter.llm_worker = dummy

    monkeypatch.setattr(
        converter,
        "_process_page",
        lambda page, page_index, pdf_path, assets_dir: (_ for _ in ()).throw(
            AssertionError("fallback should be disabled by default")
        ),
    )

    out = converter._convert_page_with_vision_guardrails(
        png_bytes=b"fake",
        page=object(),
        page_index=4,
        total_pages=10,
        page_hint="",
        speed_mode="normal",
        is_references_page=False,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
    )

    assert out == broken


def test_legacy_extra_cleanup_toggle(monkeypatch, tmp_path):
    converter = _make_converter(tmp_path)

    monkeypatch.delenv("KB_PDF_LEGACY_EXTRA_CLEANUP", raising=False)
    assert converter._legacy_extra_cleanup_enabled() is False

    monkeypatch.setenv("KB_PDF_LEGACY_EXTRA_CLEANUP", "1")
    assert converter._legacy_extra_cleanup_enabled() is True


def test_inject_missing_page_image_links_for_figure_caption():
    md = """
Some paragraph.
Figure 5. Example caption text.
More paragraph.
""".strip()
    out = PDFConverter._inject_missing_page_image_links(
        md,
        page_index=4,
        image_names=["page_5_fig_1.png"],
        is_references_page=False,
    )
    assert "![Figure 5](./assets/page_5_fig_1.png)" in out
    assert out.index("![Figure 5](./assets/page_5_fig_1.png)") < out.index("Figure 5. Example caption text.")


def test_inject_missing_page_image_links_skips_references_page():
    md = "Figure 20. This line should stay unchanged."
    out = PDFConverter._inject_missing_page_image_links(
        md,
        page_index=19,
        image_names=["page_20_fig_1.png"],
        is_references_page=True,
    )
    assert out == md
