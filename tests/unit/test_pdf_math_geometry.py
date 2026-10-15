from __future__ import annotations

import pytest


def _span(text, x, baseline=30, size=10, font="CMMI10", width=6):
    return {"text": text, "font": font, "size": size,
            "origin": (x, baseline), "bbox": (x, baseline - size, x + width, baseline + size / 3)}


@pytest.mark.parametrize("font, expected", [
    ("CMSY10", r"\mathcal{J}"), ("ABCDEF+CMSY7", r"\mathcal{J}"),
    ("CMMI10", "J"), ("Times-Italic", "J"), ("unknownCMSY10", "J"),
])
def test_partition_style_comes_from_known_pdf_font_only(font, expected):
    from kb.converter.pdf_math_geometry import styled_span_text
    assert styled_span_text(_span("J", 10, font=font)) == expected


def test_math_clip_excludes_adjacent_prose_but_keeps_complement_and_partition():
    from kb.converter.pdf_math_geometry import extract_math_raw
    spans = [_span("f", 10), _span("∗", 15, 26, 7, "CMSY7"),
             _span("J", 15, 32.5, 7, "CMSY7"), _span("(x)", 22, width=15),
             _span("J", 38, 31.5, 7), _span("=E[y|x", 44, width=36),
             _span("J", 81, 31.5, 7), _span("c", 87, 29.5, 5), _span("]", 91),
             # A tall next-row symbol overlaps the crop by one point.
             {**_span("∈J", 30, 47, 10, "CMSY10"), "bbox": (30, 37, 42, 55)},
             _span("other", 140)]
    class Page:
        def get_text(self, mode):
            return {"blocks": [{"lines": [{"spans": spans}]}]}
    result = extract_math_raw(Page(), (9, 18, 100, 38))
    assert r"f^{\ast}_{\mathcal{J}}" in result
    assert r"x_{J^{c}}" in result
    assert r"\in" not in result and "other" not in result


def test_only_same_baseline_neighbor_fragments_are_joined():
    from kb.converter.pdf_math_geometry import coalesce_baseline_lines
    lines = [{"bbox": (10, 20, 70, 35), "spans": [_span("we choose J", 10, width=60)]},
             {"bbox": (75, 20, 81, 35), "spans": [_span("=", 75)]},
             {"bbox": (10, 40, 16, 55), "spans": [_span("=", 10, 50)]},
             {"bbox": (310, 20, 316, 35), "spans": [_span("=", 310)]}]
    result = coalesce_baseline_lines(lines)
    assert len(result) == 3
    assert [s["text"] for s in result[0]["spans"]] == ["we choose J", "="]
    assert len(result[1]["spans"]) == len(result[2]["spans"]) == 1


def test_inline_partition_quantifier_is_prose_in_source_index():
    from kb.source_blocks import is_display_equation_block
    assert not is_display_equation_block(r"for each subset J ∈ $\mathcal{J}$.")
    assert is_display_equation_block(r"$$J \in \mathcal{J}$$")


def test_multiline_base_glyphs_do_not_get_invented_scripts():
    from kb.converter.pdf_math_geometry import _linear_script_math
    assert _linear_script_math([_span("x", 10), _span("y", 10, 45)]) == ""
    assert _linear_script_math([_span("x", 10), _span("a", 16, 30, 7)]) == ""


def test_linear_script_decoder_leaves_mixed_prose_to_existing_fallback():
    from kb.converter.pdf_math_geometry import _linear_script_math
    assert _linear_script_math([_span("x", 10), _span("i", 16, 32, 7),
                                _span(" where a is positive", 22, font="Times-Roman", width=100)]) == ""


def test_real_math_rows_and_membership_are_retained():
    from kb.converter.pdf_math_geometry import extract_math_raw
    class Page:
        def get_text(self, mode):
            return {"blocks": [{"lines": [{"spans": [
                _span("J", 10), _span("∈", 17, font="CMSY10"),
                _span("J", 24, font="CMSY10")]}]}]}
    assert extract_math_raw(Page(), (9, 19, 35, 40)) == r"J\in\mathcal{J}"
