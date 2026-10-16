from __future__ import annotations

import pytest

from kb.evidence_text import clean_display_text


@pytest.mark.parametrize("literal", [
    "input-conditioned p ( y | x ) has the same conditional expected values.",
    "The objective is ||f(x) - x||_2^2 and |x| is bounded.",
    r"The loss $L_2 = \|f(x)-x\|^2$ retains $\mathcal{J}$ and $x_i^2$.",
    r"Use \(P(Y=y \mid X=x)\) and \[\mathbb{E}[x_i^2]\].",
    "P(Y=y | X=x) and E[y | x] retain conditioning.",
    "||x||",
    r"\gamma = \frac{1}{nm} \sum_{ij} |W_{ij}|. \tag{3}",
])
def test_card_display_preserves_literal_math(literal):
    assert clean_display_text(literal) == literal
    assert clean_display_text(clean_display_text(literal)) == literal


@pytest.mark.parametrize(("raw", "expected"), [
    ("| Method | Score |\n| :--- | ---: |\n| A | 2 |", "Method Score A 2"),
    ("|Method|Score|\n|---|---|\n|A|2|", "Method Score A 2"),
    ("Method | Score", "Method Score"),
    ("$DMD$ modulation uses $p$ frequencies.", "DMD modulation uses p frequencies."),
    ("Options (red | blue) are alternatives.", "Options (red blue) are alternatives."),
    ("| loss | formula |\n| --- | --- |\n| L2 | $||x||_2^2$ |", "loss formula L2 $||x||_2^2$"),
    ("**Bound** p(y | x) [[CITE:ref-1:2]]", "Bound p(y | x)"),
])
def test_card_math_does_not_preserve_markdown_separators(raw, expected):
    assert clean_display_text(raw) == expected


def test_reference_overlay_keeps_conditional_math():
    from api.routers.references import _clean_answer_citation_overlay_quote

    literal = r"Conditional p(y | x), ||x|| and $L_2$ retain \mathcal{J}."
    assert _clean_answer_citation_overlay_quote(literal) == literal
