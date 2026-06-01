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
