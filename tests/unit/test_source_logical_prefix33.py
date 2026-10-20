from __future__ import annotations

import pytest

from api.source_card_contract import _restore_leading_qualifier, refresh_source_citation_cards
from kb.source_blocks import load_source_blocks_cached


@pytest.mark.parametrize('prefix', ['Note that without ', 'Only if we apply ', 'Unless we omit '])
def test_excerpt_keeps_its_own_sentences_leading_condition(prefix):
    quote = 'our correction step, the raw output of the network is still noisy.'
    raw = '(bottom right). ' + prefix + quote + ' Indeed, the standard deviation of'
    assert _restore_leading_qualifier({'raw_text': raw}, quote) == prefix + quote


def test_unrelated_sentence_and_ambiguous_occurrence_do_not_expand_a_quote():
    quote = 'The raw output is noisy.'
    assert _restore_leading_qualifier({'raw_text': 'Without training, nothing works. ' + quote}, quote) == quote
    assert _restore_leading_qualifier({'raw_text': 'Without correction, ' + quote + quote}, quote) == quote
    assert _restore_leading_qualifier({'raw_text': quote}, quote) == quote


def test_literal_condition_is_identical_in_every_card_projection_and_refresh(tmp_path):
    quote = 'our correction step, the raw output of the network is still noisy.'
    full = 'Note that without ' + quote
    path = tmp_path / 'method.md'
    path.write_text('<!-- kb_page: 4 -->\n## Correction\n\n(bottom right). ' + full + ' Indeed, the standard deviation of', encoding='utf8')
    block = next(b for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph')
    detail = {**block, 'source_path': str(path), 'citation_route': 'system_a', 'evidence_quote': quote, 'evidence_source': 'verified_source_excerpt', 'answer_claim': 'Without correction, the raw output of the network is still noisy.'}
    for locale in ('en', 'zh', 'en'):
        detail = refresh_source_citation_cards([detail], render_locale=locale)[0]
        assert detail['evidence_quote'] == detail['card_evidence'] == detail['raw'] == full
        assert next(s['text'] for s in detail['card_view']['sections'] if s['id'] == 'evidence') == full
        assert all(detail[k] == block[k] for k in ('block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end'))
