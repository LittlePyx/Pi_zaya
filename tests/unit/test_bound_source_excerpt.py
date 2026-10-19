from __future__ import annotations

import copy

import pytest

from api.source_card_contract import _verified_bound_reader_excerpt, refresh_source_citation_cards
from kb.source_blocks import load_source_blocks_cached

CLAIM = 'The method requires a known noise distribution and can sample from the noise distribution.'
SHORT = 'N follows a known noise distribution.'
FULL = SHORT + ' Because it is known, we can draw an additional synthetic noise sample and sample from the noise distribution.'


def _record(tmp_path):
    path = tmp_path / 'method.md'
    path.write_text('<!-- kb_page: 3 -->\n## Method\n\n' + FULL, encoding='utf8')
    block = next(b for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph')
    return {**block, 'source_path': str(path), 'citation_route': 'system_a', 'binding_status': 'grounded', 'answer_claim': CLAIM, 'passage_bound_claims': [CLAIM + ' [1]'], 'evidence_quote': SHORT, 'reader_evidence_quote': FULL, 'evidence_source': 'citation_plan_claim_window', 'reader_evidence_source': 'citation_plan_located_block'}


def test_claim_specific_reader_passage_survives_display_shortening(tmp_path):
    record = _record(tmp_path); before = copy.deepcopy(record)
    assert _verified_bound_reader_excerpt(record)[1] == FULL
    result = refresh_source_citation_cards([record], render_locale='en')[0]
    assert record == before
    assert result['evidence_quote'] == result['card_evidence'] == FULL
    assert next(s['text'] for s in result['card_view']['sections'] if s['id'] == 'evidence') == FULL
    assert refresh_source_citation_cards([result], render_locale='en')[0]['evidence_quote'] == FULL


@pytest.mark.parametrize('failure', ['page', 'block', 'anchor', 'heading', 'source', 'text', 'unbound', 'wrong_claim', 'ambiguous_preview'])
def test_bound_reader_context_cannot_repair_stale_identity_or_missing_support(tmp_path, failure):
    record = _record(tmp_path)
    if failure == 'page': record['page_start'] = 99
    elif failure == 'block': record['block_id'] = 'stale'
    elif failure == 'anchor': record['anchor_id'] = 'stale'
    elif failure == 'heading': record['heading_path'] = 'Other'
    elif failure == 'source': record['source_path'] = str(tmp_path / 'missing.md')
    elif failure == 'text': record['reader_evidence_quote'] = FULL.replace('known', 'unknown')
    elif failure == 'unbound': record['passage_bound_claims'] = []
    elif failure == 'wrong_claim': record['answer_claim'] = 'The method reports 900 clinical trial participants.'
    elif failure == 'ambiguous_preview':
        record['answer_claim'] = CLAIM[:60] + '...'
        record['passage_bound_claims'] = [CLAIM, CLAIM + ' It also has an unsupported property.']
    assert _verified_bound_reader_excerpt(record) is None
