from kb.converter.text_utils import (
    _join_lines_preserving_words,
    _normalize_text,
    contains_only_detached_accent_mojibake,
    contains_mojibake,
    count_mojibake,
    normalize_detached_accents,
    normalize_known_superscript_tokens,
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


def test_detached_accent_normalizer_repairs_dotless_i_without_touching_markdown():
    text = "**Roberto Ram \u0301\u0131rez** and H \u0301ector"

    assert contains_only_detached_accent_mojibake(text) is True
    assert normalize_detached_accents(text) == "**Roberto Ram\u00edrez** and H\u00e9ctor"
    assert contains_only_detached_accent_mojibake("Fran脙搂ois") is False


def test_known_superscript_algorithm_tokens_are_repaired_without_touching_citations():
    text = (
        "s?ISM, s [2]ISM, s$^2$FLISM and citation [2]. "
        "https://github.com/VicidominiLab/s?ISM"
    )

    assert normalize_known_superscript_tokens(text) == (
        "s²ISM, s²ISM, s²FLISM and citation [2]. "
        "https://github.com/VicidominiLab/s2ISM"
    )

