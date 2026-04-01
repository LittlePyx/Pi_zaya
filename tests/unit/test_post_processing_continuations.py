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


def test_postprocess_repairs_safe_short_prefix_word_splits_in_body_text():
    src = """# Title

For example, when the overlapping rate be- comes 1, the image re- construction problem becomes harder.
"""
    out = postprocess_markdown(src)
    assert "becomes 1" in out
    assert "reconstruction problem" in out
    assert "be- comes" not in out
    assert "re- construction" not in out


def test_postprocess_merges_vocab_continuation_line_after_connector():
    src = """# Title

We further employed off-the-shelf public high-resolution images (collected from the PASCAL VOC2007 [31] and
VOC2012 [32] datasets) to synthesize training data.
"""
    out = postprocess_markdown(src)
    assert "VOC2007 [31] and VOC2012 [32] datasets" in out


def test_postprocess_does_not_merge_italic_caption_tail_into_body_continuation():
    src = """# Title

We further employed off-the-shelf public high-resolution images (collected from the PASCAL VOC2007 [31] and
*model, a cat toy and different printed shapes) using the reported technique.*
VOC2012 [32] datasets) to digitally synthesize a large-scale realistic single-photon image dataset.
"""
    out = postprocess_markdown(src)
    assert "VOC2007 [31] and *model, a cat toy" not in out
    assert "*model, a cat toy and different printed shapes) using the reported technique.*" in out
    assert "VOC2012 [32] datasets" in out
