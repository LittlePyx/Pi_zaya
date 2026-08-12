from kb.converter.md_analyzer import MarkdownAnalyzer


def test_md_analyzer_does_not_flag_numbered_section_heading_as_formula():
    md = "\n".join(
        [
            "# Title",
            "",
            "## 3. Method",
            "",
            "Body text.",
        ]
    )
    issues = MarkdownAnalyzer().analyze(md)
    assert not any(
        issue.category == "heading" and "looks like a formula" in issue.message
        for issue in issues
    )


def test_md_analyzer_does_not_treat_caption_with_pipe_as_table():
    md = "\n".join(
        [
            "# Title",
            "",
            "**Figure 1.** | Illustration of the reported network.",
            "",
            "Body text.",
        ]
    )
    issues = MarkdownAnalyzer().analyze(md)
    assert not any(issue.category == "table" for issue in issues)


def test_md_analyzer_does_not_count_escaped_math_pipes_as_columns():
    md = "\n".join(
        [
            "# Title",
            "",
            "| Method | Norm | Score |",
            "| --- | --- | --- |",
            r"| Baseline | $\|\Theta\|$ | 0.91 |",
        ]
    )

    issues = MarkdownAnalyzer().analyze(md)

    assert not any(issue.category == "table" and issue.severity == "error" for issue in issues)


def test_md_analyzer_does_not_treat_display_math_pipes_as_tables():
    md = "\n".join(
        [
            "# Title",
            "",
            "$$",
            r"|\Theta| = L \times d_{model}",
            "$$",
        ]
    )

    issues = MarkdownAnalyzer().analyze(md)

    assert not any(issue.category == "table" for issue in issues)


def test_md_analyzer_does_not_treat_isolated_equation_row_as_table():
    md = "\n".join(
        [
            "# Title",
            "",
            "The conditional objective is",
            "| y | X",
            "X",
            "$$",
            r"\sum_t \log P(y_t \mid x, y_{<t})",
            "$$",
        ]
    )

    issues = MarkdownAnalyzer().analyze(md)

    assert not any(issue.category == "table" for issue in issues)


def test_md_analyzer_checks_table_at_end_of_document():
    md = "\n".join(
        [
            "# Title",
            "",
            "| Method | Score |",
            "| --- | --- |",
            "| Baseline | 0.91 | extra |",
        ]
    )

    issues = MarkdownAnalyzer().analyze(md)

    assert any(issue.category == "table" and issue.severity == "error" for issue in issues)


def test_md_analyzer_accepts_caption_before_image():
    md = "\n".join(
        [
            "# Title",
            "",
            "**Figure 3.** Reconstruction pipeline for the proposed imaging system.",
            "![Figure 3](assets/page_1_fig_1.png)",
        ]
    )

    issues = MarkdownAnalyzer().analyze(md)

    assert not any(
        issue.category == "caption" and "has no nearby caption" in issue.message
        for issue in issues
    )


def test_md_analyzer_accepts_source_table_evidence_caption():
    md = "\n".join(
        [
            "# Title",
            "",
            "**Table evidence.** Original table preserved from source PDF page 4.",
            "![Source PDF page 4 containing the recovered table](assets/page_4_table_recovery.png)",
        ]
    )

    issues = MarkdownAnalyzer().analyze(md)

    assert not any(issue.category == "caption" for issue in issues)


def test_md_analyzer_accepts_caption_after_page_anchor_gap():
    md = "\n".join(
        [
            "# Title",
            "",
            "![Figure 2](assets/page_4_fig_1.png)",
            "",
            "<!-- kb_page: 4 -->",
            "",
            "**Figure 2.** The pipeline of the proposed snapshot reconstruction method.",
        ]
    )

    issues = MarkdownAnalyzer().analyze(md)

    assert not any(
        issue.category == "caption" and "has no nearby caption" in issue.message
        for issue in issues
    )


def test_md_analyzer_accepts_descriptive_image_alt_caption():
    md = "\n".join(
        [
            "# Title",
            "",
            "![Fig. 5 Single photon detection principle of superconducting nanowire](assets/page_7_fig_1.png)",
        ]
    )

    issues = MarkdownAnalyzer().analyze(md)

    assert not any(
        issue.category == "caption" and issue.severity == "warning"
        for issue in issues
    )


def test_md_analyzer_still_warns_for_label_only_image_alt():
    md = "\n".join(
        [
            "# Title",
            "",
            "![Figure 2](assets/page_4_fig_1.png)",
        ]
    )

    issues = MarkdownAnalyzer().analyze(md)

    assert any(
        issue.category == "caption" and "has no nearby caption" in issue.message
        for issue in issues
    )


def test_md_analyzer_does_not_require_figure_captions_for_equation_assets():
    md = "\n".join(
        [
            "# Title",
            "",
            "![Equation](./assets/page_4_eq_1.png)",
            "<!-- kb:conversion_retry kind=equation page=4 -->",
            "",
            "![Formula](./assets/page_4_formula_2.png)",
        ]
    )

    issues = MarkdownAnalyzer().analyze(md)

    assert not any(issue.category == "caption" and issue.severity == "warning" for issue in issues)


def test_md_analyzer_does_not_parse_plural_algorithm_prose_as_caption():
    md = "# Title\n\nThese algorithms to enhance security are evaluated in the next section."

    issues = MarkdownAnalyzer().analyze(md)

    assert not any(issue.category == "caption" for issue in issues)


def test_md_analyzer_does_not_report_reference_list_as_long_paragraph():
    refs = "\n".join(
        f"[{idx}] Author {idx}. Reference title {idx}. Journal of Tests, 2024."
        for idx in range(1, 31)
    )
    md = f"# Title\n\n## References\n\n{refs}"

    issues = MarkdownAnalyzer().analyze(md)

    assert not any(
        issue.category == "structure" and "Very long paragraph" in issue.message
        for issue in issues
    )


def test_md_analyzer_recognizes_common_reference_heading_variants():
    refs = "\n".join(
        f"[{idx}] Author {idx}. Reference title {idx}. Journal of Tests, 2024."
        for idx in range(1, 31)
    )

    for heading in ("References and Notes", "Reference List", "Literature Cited", "Works Cited"):
        issues = MarkdownAnalyzer().analyze(f"# Title\n\n## {heading}\n\n{refs}")
        assert not any(
            issue.category == "structure" and "Very long paragraph" in issue.message
            for issue in issues
        )
