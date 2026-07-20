from __future__ import annotations

from kb.converter.quality_compare import (
    compare_markdown_quality,
    render_quality_comparison_report,
    summarize_markdown_quality,
)


def test_summarize_markdown_quality_extracts_core_metrics():
    md = "\n".join(
        [
            "# Title",
            "",
            "## Abstract",
            "",
            "Body cites [11-13] and [15].",
            "",
            "$$",
            "a = b",
            "$$",
            "",
            "| A | B |",
            "| --- | --- |",
            "| 1 | 2 |",
            "",
            "![fig](./assets/a.png)",
            "**Figure 1.** Example caption.",
            "",
            "## References",
            "[11] Ref eleven.",
            "[12] Ref twelve.",
        ]
    )

    summary = summarize_markdown_quality(md)

    assert summary.has_abstract_heading is True
    assert summary.heading_count == 3
    assert summary.display_math_block_count == 1
    assert summary.table_block_count == 1
    assert summary.image_count == 1
    assert summary.caption_count == 1
    assert summary.reference_line_count == 2
    assert summary.max_reference_index == 12
    assert summary.body_citation_marker_count == 2
    assert summary.body_citation_range_count == 1
    assert summary.body_citation_expanded_index_count == 4


def test_summarize_markdown_quality_does_not_count_plural_algorithm_prose_as_caption():
    summary = summarize_markdown_quality(
        "# Title\n\nThese algorithms to enhance security are evaluated in the next section."
    )

    assert summary.caption_count == 0


def test_compare_markdown_quality_allows_equation_image_to_become_display_math():
    base = "# Title\n\n![Equation](./assets/page_4_eq_1.png)"
    candidate = "# Title\n\n$$\na = b\n$$"

    comparison = compare_markdown_quality(base, candidate)

    assert summarize_markdown_quality(base).image_count == 0
    assert comparison["regression_flags"]["images_dropped"] is False
    assert comparison["regression_flags"]["display_math_dropped"] is False


def test_compare_markdown_quality_flags_obvious_regressions():
    base = "\n".join(
        [
            "# Title",
            "",
            "## Abstract",
            "",
            "Body cites [1-3].",
            "",
            "## References",
            "[1] Ref one.",
            "[2] Ref two.",
            "[3] Ref three.",
        ]
    )
    candidate = "\n".join(
        [
            "# Title",
            "",
            "Body cites [1].",
            "",
            "## References",
            "[1] Ref one.",
        ]
    )

    comparison = compare_markdown_quality(base, candidate)

    assert comparison["regression_flags"]["missing_abstract_heading"] is True
    assert comparison["regression_flags"]["reference_lines_dropped"] is False
    assert comparison["regression_flags"]["reference_index_regressed"] is True
    assert comparison["delta"]["body_citation_expanded_index_count"] == -2


def test_render_quality_comparison_report_contains_key_sections():
    comparison = compare_markdown_quality("# Title\n", "# Title\n\n## References\n[1] Ref\n")

    report = render_quality_comparison_report(
        base_label="base.md",
        candidate_label="candidate.md",
        comparison=comparison,
    )

    assert "# Markdown Quality Comparison" in report
    assert "## Key Metrics" in report
    assert "## Regression Flags" in report
    assert "candidate.md" in report
