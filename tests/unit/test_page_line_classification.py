from __future__ import annotations

from kb.converter.page_line_classification import looks_like_mixed_math_prose_line


def test_inline_equation_inside_proposition_stays_prose() -> None:
    line = "E[x|y] = y, and the noise in each subset J ∈ J is independent from its complement."
    assert looks_like_mixed_math_prose_line(line)


def test_pure_display_equation_is_not_prose() -> None:
    assert not looks_like_mixed_math_prose_line("E ||f(x)-x||^2 = E ||f(x)-y||^2 + E ||x-y||^2")
    assert not looks_like_mixed_math_prose_line("f_J^*(x_Jc) = E[y_J | x_Jc]")
