from kb.converter.post_processing import postprocess_markdown


def test_postprocess_moves_lowercase_fragment_before_intrusive_heading():
    src = """# Title

Furthermore, the detector has to be fast enough to measure the signals as the

<!-- box:end id=1 -->

## Applications and future potential for single-pixel imaging

In our discussions thus far we have assumed that this mode creates subtle

pattern masks are changed, large enough to collect the transmitted light.
"""

    out = postprocess_markdown(src)

    assert out.index("pattern masks are changed, large enough to collect the transmitted light.") < out.index(
        "## Applications and future potential for single-pixel imaging"
    )


def test_postprocess_keeps_heading_when_no_dangling_connector_before_it():
    src = """# Title

This paragraph is complete.

## Applications

in practice we can also do this.
"""
    out = postprocess_markdown(src)
    assert out.index("## Applications") < out.index("in practice we can also do this.")

