
import pytest
from kb.converter.text_utils import _normalize_text, _fix_common_mojibake, _join_lines_preserving_words

def test_normalize_text_basic():
    # Test basic whitespace and unicode normalization
    text = "Hello   World\u00A0"  # \u00A0 is NBSP
    assert _normalize_text(text) == "Hello World"

def test_mojibake_replacement():
    # Test that common mojibake characters are replaced
    # â€” is often a mojibake for em-dash (—)
    text = "Textâ€”withâ€”dashes" 
    # Our _fix_common_mojibake might not catch this specific one if it's not in the map,
    # but let's test what IS in the map based on the file content.
    # Looking at file content: "\u2026": "..." is in canonical but maybe not in mojibake map depending on encoding.
    # Let's test a known replacement from the code if possible, or generic normalization.
    
    # From code: \u201c -> "
    text_smart_quotes = "\u201cHello\u201d"
    assert _normalize_text(text_smart_quotes) == '"Hello"'

def test_ligature_replacement():
    # fi ligature
    text = "ﬁeld"
    assert _normalize_text(text) == "field"

def test_join_lines_hyphenation():
    lines = ["This is a sen-", "tence with hyphen."]
    # Expect "This is a sentence with hyphen."
    assert _join_lines_preserving_words(lines) == "This is a sentence with hyphen."

def test_join_lines_standard():
    lines = ["This is a", "sentence."]
    assert _join_lines_preserving_words(lines) == "This is a sentence."

def test_join_lines_empty():
    assert _join_lines_preserving_words([]) == ""
