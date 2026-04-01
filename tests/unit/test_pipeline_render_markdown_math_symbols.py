from kb.converter.pipeline_render_markdown import (
    _count_mathish_symbols,
    _has_complex_math_token,
    _has_mathish_symbol,
    _looks_like_short_prose_math_fragment,
)


def test_math_symbol_helpers_handle_unicode_and_latex_tokens():
    src = r"x \in \mathbb{R}, \sum_i a_i \approx \sqrt{b} + c"
    assert _has_mathish_symbol(src)
    assert _count_mathish_symbols(src) >= 4
    assert _has_complex_math_token(src)


def test_math_symbol_helpers_do_not_flag_plain_prose():
    src = "This paragraph describes the method in plain language only."
    assert not _has_mathish_symbol(src)
    assert _count_mathish_symbols(src) == 0
    assert not _has_complex_math_token(src)


def test_short_prose_math_fragment_detection():
    assert _looks_like_short_prose_math_fragment("ments(see eq.(2)).")
    assert _looks_like_short_prose_math_fragment("see eq.~(2)")
    assert not _looks_like_short_prose_math_fragment(r"\frac{a}{b}")
