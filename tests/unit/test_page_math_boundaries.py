from __future__ import annotations

from kb.converter.page_math_boundaries import (
    repair_page_display_math_boundaries,
    unclosed_display_math_pages,
)
from kb.converter.post_processing import postprocess_markdown


def test_unclosed_display_math_is_detected_per_physical_page() -> None:
    source = "\n".join(
        [
            "<!-- kb_page: 3 -->",
            "$$",
            r"x = y",
            "<!-- kb_page: 4 -->",
            r"z = 1",
            "$$",
        ]
    )

    assert unclosed_display_math_pages(source) == [3, 4]


def test_page_boundary_repair_closes_formula_before_following_prose() -> None:
    source = "\n".join(
        [
            "<!-- kb_page: 5 -->",
            "The transition follows: $$",
            r"q_t(x) = \int p_t(x \mid z)\,\mathrm{d}z. \tag{7}",
            "",
            "This paragraph must remain ordinary prose.",
            "<!-- kb_page: 6 -->",
            "The next page is independent.",
        ]
    )

    repaired, repaired_pages, unresolved_pages = repair_page_display_math_boundaries(source)

    assert repaired_pages == [5]
    assert unresolved_pages == []
    assert unclosed_display_math_pages(repaired) == []
    assert "$$\n<!-- kb:page_math_boundary_repair page=5 -->\n\nThis paragraph" in repaired
    assert repaired.count("kb:page_math_boundary_repair page=5") == 1


def test_empty_unclosed_formula_is_contained_and_marked_for_retry() -> None:
    source = "<!-- kb_page: 8 -->\n\nAn equation should follow.\n$$"

    repaired, repaired_pages, unresolved_pages = repair_page_display_math_boundaries(source)

    assert repaired_pages == []
    assert unresolved_pages == [8]
    assert unclosed_display_math_pages(repaired) == []
    assert "kb:conversion_retry kind=math_text page=8 reason=unclosed_display_math" in repaired


def test_postprocess_keeps_subscripts_inside_display_math_after_inline_opener() -> None:
    source = "\n".join(
        [
            "<!-- kb_page: 3 -->",
            "Assume the process is absorbed on the boundary, $$",
            r"\mathbb{Q} \sim \mathrm{Ito}_\Omega(b, \sigma). \tag{3}",
            "$$",
            "",
            "The conditioned law is",
            "$$",
            r"\mathbb{Q}(\cdot \mid Z_\tau = x) = q_\Omega(x \mid Z_t). \tag{4}",
            "$$",
        ]
    )

    repaired = postprocess_markdown(source)

    assert unclosed_display_math_pages(repaired) == []
    assert r"Z_\tau = x" in repaired
    assert "Z_$" not in repaired

