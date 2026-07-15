from kb.converter.text_utils import (
    _join_lines_preserving_words,
    _normalize_text,
    contains_mojibake,
    count_mojibake,
)


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


def test_normalize_text_reattaches_detached_acute_to_following_letter():
    assert _normalize_text("Husz\u00b4ar") == "Huszár"
    assert _normalize_text("Husz \u00b4ar") == "Huszár"
    assert _normalize_text("Husz \u0301ar") == "Huszár"


def test_mojibake_helpers_detect_encoding_and_detached_accent_sequences():
    text = "Husz \u0301ar and FranÃ§ois"

    assert contains_mojibake(text) is True
    assert count_mojibake(text) == 2
    assert contains_mojibake("Huszár and François") is False
    assert count_mojibake("") == 0

