from kb.converter.post_processing import postprocess_markdown


def test_body_citations_are_normalized_to_superscript():
    src = """
# 1. Intro
This is a sentence [ 188 ] and another [190-195] with ranges.

## References
[188] A. Author, Journal 2024, 1, 1.
[190] B. Author, Journal 2024, 1, 2.
"""
    out = postprocess_markdown(src)
    assert "$^{[188]}$" in out
    assert "$^{[190-195]}$" in out
    # References list itself should stay plain.
    assert "\n[188] A. Author" in out


def test_existing_superscript_citations_are_canonicalized_once():
    src = """
Text with existing cite $^{[199, 200]}$ and bare form ^{[201–203]}.
"""
    out = postprocess_markdown(src)
    assert "$^{[199,200]}$" in out
    assert "$^{[201-203]}$" in out
    assert out.count("$^{[199,200]}$") == 1
    assert out.count("$^{[201-203]}$") == 1


def test_does_not_convert_reference_entries_to_superscript():
    src = """
## References
[8] L. Pan, Y. Shen, J. Qi, Opt. Express 2023, 31, 13943.
"""
    out = postprocess_markdown(src)
    assert "\n[8] L. Pan" in out
    assert "$^{[8]}$" not in out
