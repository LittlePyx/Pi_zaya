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
