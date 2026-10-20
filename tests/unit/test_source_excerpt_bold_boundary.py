from __future__ import annotations

from kb.source_excerpt import literal_excerpt


def test_literal_excerpt_recovers_the_opening_of_an_exact_paired_source_label():
    raw = '**Comparison to single-image-based methods.** Several representative methods with published codes are selected for comparison.'
    assert literal_excerpt({'raw_text': raw, 'kind': 'paragraph'}, raw[2:]) == raw
    assert not literal_excerpt({'raw_text': raw, 'kind': 'paragraph'}, raw[2:].replace('published codes', 'clinical trials'))


def test_bold_like_formula_and_code_content_is_never_presentation_repaired():
    for raw in ('$**Arithmetic symbols are mathematical source data.**$', '`**These are literal characters in the source code.**`'):
        quote = raw[3:-1]
        assert literal_excerpt({'raw_text': raw, 'kind': 'paragraph'}, quote) == quote
