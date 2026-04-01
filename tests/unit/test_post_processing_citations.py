from kb.converter.post_processing import postprocess_markdown


def test_body_citations_are_normalized_to_plain_brackets():
    src = """
# 1. Intro
This is a sentence [ 188 ] and another [190-195] with ranges.

## References
[188] A. Author, Journal 2024, 1, 1.
    [190] B. Author, Journal 2024, 1, 2.
"""
    out = postprocess_markdown(src)
    assert "[188]" in out
    assert "[190-195]" in out
    assert "$^{[188]}$" not in out
    assert "$^{[190-195]}$" not in out
    # References list itself should stay plain.
    assert "\n[188] A. Author" in out


def test_existing_superscript_citations_are_canonicalized_to_plain_brackets():
    src = """
Text with existing cite $^{[199, 200]}$ and bare form ^{[201–203]}.
"""
    out = postprocess_markdown(src)
    assert "[199,200]" in out
    assert "[201-203]" in out
    assert "$^{[199,200]}$" not in out
    assert "$^{[201-203]}$" not in out


def test_does_not_convert_reference_entries_to_superscript():
    src = """
## References
[8] L. Pan, Y. Shen, J. Qi, Opt. Express 2023, 31, 13943.
"""
    out = postprocess_markdown(src)
    assert "\n[8] L. Pan" in out
    assert "$^{[8]}$" not in out


def test_html_and_plain_superscript_citations_are_canonicalized_to_plain_brackets():
    src = """
Camera acquisition is managed through *cam control*<sup>26</sup>.
The method follows prior work $^{25}$ and $^{17, 27}$, with a range ^{11–13}.
"""
    out = postprocess_markdown(src)
    assert "<sup>" not in out
    assert "[26]" in out
    assert "[25]" in out
    assert "[17,27]" in out
    assert "[11-13]" in out


def test_latex_textsuperscript_and_unicode_superscript_citations_are_canonicalized():
    src = """
Data acquisition used pyLabLib\\textsuperscript{26}.
This agrees with prior work²⁵ and a later refinement³⁴.
"""
    out = postprocess_markdown(src)
    assert "\\textsuperscript{26}" not in out
    assert "²⁵" not in out
    assert "³⁴" not in out
    assert "pyLabLib [26]" in out
    assert "[25]" in out
    assert "[34]" in out


def test_bare_inline_voc_citations_are_wrapped_back_into_brackets():
    src = """
We further employed off-the-shelf public high-resolution images (collected from the PASCAL VOC2007 31 and VOC2012 32 datasets) to synthesize training data.
"""
    out = postprocess_markdown(src)
    assert "VOC2007 [31] and VOC2012 [32] datasets" in out


def test_bare_inline_framework_citations_are_wrapped_back_into_brackets():
    src = """
The transformer framework 33 , 34 has recently attracted increasing attention and produced an impressive performance on multiple vision tasks 34-36 in image restoration.
"""
    out = postprocess_markdown(src)
    assert "framework [33,34] has" in out
    assert "tasks [34-36] in image restoration" in out


def test_bare_inline_task_range_with_punctuation_is_wrapped_back_into_brackets():
    src = """
The transformer framework 33 , 34 has recently attracted attention and produced strong performance on multiple vision tasks 34 – 36 . As presented below, the network uses three modules.
"""
    out = postprocess_markdown(src)
    assert "framework [33,34] has" in out
    assert "tasks [34-36] . As presented below" in out
