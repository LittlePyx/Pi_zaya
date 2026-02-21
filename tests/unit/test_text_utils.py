from kb.converter.text_utils import _join_lines_preserving_words, _normalize_text


def test_normalize_text_basic():
    text = "Hello   World\u00A0"
    assert _normalize_text(text) == "Hello World"


def test_smart_quotes_replacement():
    text = "\u201cHello\u201d"
    assert _normalize_text(text) == '"Hello"'


def test_ligature_replacement():
    text = "ﬁeld"
    assert _normalize_text(text) == "field"


def test_join_lines_hyphenation():
    lines = ["This is a sen-", "tence with hyphen."]
    assert _join_lines_preserving_words(lines) == "This is a sentence with hyphen."


def test_join_lines_standard():
    lines = ["This is a", "sentence."]
    assert _join_lines_preserving_words(lines) == "This is a sentence."

