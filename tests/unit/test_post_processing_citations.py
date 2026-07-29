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


def test_bracketed_html_and_latex_superscript_citations_are_canonicalized():
    src = r"""
The first method\textsuperscript{[43]} agrees with the later result<sup>[180]</sup>.
The grouped evidence\textsuperscript{[44, 45]} remains traceable.
"""
    out = postprocess_markdown(src)

    assert r"\textsuperscript" not in out
    assert "<sup>" not in out
    assert "method [43]" in out
    assert "result [180]" in out
    assert "evidence [44,45]" in out


def test_numeric_superscript_powers_are_not_rewritten_as_citations():
    src = r"""
The area is m<sup>2</sup>, the volume is cm<sup>3</sup>, and the scale is 10\textsuperscript{6}.
The normalized quantities x², kg<sup>2</sup>, px², NA\textsuperscript{2}, σ<sup>2</sup>, β\textsuperscript{2}, and Δ² remain powers.
"""
    out = postprocess_markdown(src)

    assert "m²" in out
    assert "cm³" in out
    assert "10⁶" in out
    assert "x²" in out
    assert "kg²" in out
    assert "px²" in out
    assert "NA²" in out
    assert "σ²" in out
    assert "β²" in out
    assert "Δ²" in out
    assert "[2]" not in out
    assert "[3]" not in out
    assert "[6]" not in out


def test_unbracketed_superscripts_after_words_and_acronyms_remain_citations():
    src = (
        r"Prior CNN<sup>43</sup>, SPI<sup>180</sup>, DL\textsuperscript{12}, "
        r"work<sup>2</sup>, and method\textsuperscript{3} studies agree."
    )
    out = postprocess_markdown(src)

    assert "CNN [43]" in out
    assert "SPI [180]" in out
    assert "DL [12]" in out
    assert "work [2]" in out
    assert "method [3]" in out
    assert "CNN⁴³" not in out
    assert "SPI¹⁸⁰" not in out


def test_superscript_citation_normalization_skips_inline_code_spans():
    src = r"""
Literal `Method<sup>[43]</sup>` and ``Result\textsuperscript{[180]}`` stay unchanged.
Outside code, Method<sup>[43]</sup> and Result\textsuperscript{[180]} become citations.
"""
    out = postprocess_markdown(src)

    assert "`Method<sup>[43]</sup>`" in out
    assert r"``Result\textsuperscript{[180]}``" in out
    assert "Outside code, Method [43] and Result [180]" in out


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
