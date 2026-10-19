from __future__ import annotations

import pytest

from api.source_card_contract import refresh_source_citation_cards
from kb.source_blocks import load_source_blocks_cached


FIRST = r'N ∼$\mathcal{D}$ and $\mathcal{D}$ is a known noise distribution.'
CONTEXT = r'Because $\mathcal{D}$ is known, we can draw an additional synthetic noise sample'


def _fixture(tmp_path, raw=None, first=FIRST):
    raw = raw or first + ' ' + CONTEXT
    path = tmp_path / 'noise.md'
    path.write_text('<!-- kb_page: 3 -->\n## Method\n\n' + raw, encoding='utf8')
    block = next(b for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph')
    card = {**block, 'source_path': str(path), 'source_name': 'noise.pdf',
            'citation_route': 'system_a', 'evidence_source': 'citation_plan_claim_window',
            'evidence_quote': first, 'answer_claim': 'The noise distribution is known.',
            'citation_plan_slot': True, 'selection_reason': 'prompt_contract_block'}
    return card, raw


def test_short_variable_statement_keeps_its_verified_same_paragraph_context(tmp_path):
    card, raw = _fixture(tmp_path)
    result = refresh_source_citation_cards([card], render_locale='en')[0]
    assert result['evidence_quote'] == raw
    assert result['card_evidence'] == raw
    assert next(s['text'] for s in result['card_view']['sections'] if s['id'] == 'evidence') == raw
    assert all(result[k] == card[k] for k in ('source_path', 'block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end'))
    assert refresh_source_citation_cards([result], render_locale='en')[0]['evidence_quote'] == raw


@pytest.mark.parametrize('failure', ['block', 'anchor', 'page', 'heading', 'unrelated', 'too_large', 'lost_continuation', 'because', 'requires'])
def test_variable_context_cannot_relax_source_identity_support_or_size(tmp_path, failure):
    raw = FIRST + ' ' + CONTEXT
    if failure == 'too_large':
        raw += ' Additional source context.' * 100
    if failure == 'lost_continuation':
        raw += ' and therefore and thus'
    if failure in {'because', 'requires'}:
        raw += ' ' + failure
    card, _ = _fixture(tmp_path, raw)
    fields = {'block': 'block_id', 'anchor': 'anchor_id', 'page': 'page_start', 'heading': 'heading_path'}
    if failure in fields:
        card[fields[failure]] = 99 if failure == 'page' else 'wrong-identity'
    if failure == 'unrelated':
        card['answer_claim'] = 'A clinical trial enrolled 300 patients.'
    result = refresh_source_citation_cards([card], render_locale='en')[0]
    assert result['evidence_quote'] != raw


def test_plain_short_assertions_do_not_expand_into_unneeded_context(tmp_path):
    first = 'The noise distribution is known and fully specified in the source.'
    card, _ = _fixture(tmp_path, first=first)
    assert refresh_source_citation_cards([card], render_locale='en')[0]['evidence_quote'] == first
