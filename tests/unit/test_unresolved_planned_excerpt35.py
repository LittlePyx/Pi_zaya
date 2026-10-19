from __future__ import annotations

import copy

from api.source_card_contract import refresh_source_citation_cards
from kb.source_blocks import load_source_blocks_cached


def test_unresolved_planned_preview_cannot_leave_an_extra_nonliteral_card(tmp_path):
    first = 'The estimator is unchanged when the replacement targets have matching expectations.'
    middle = 'This requires the stated conditional expectation to be preserved.'
    last = 'These conditions apply to the distribution of training targets, not arbitrary inputs.'
    raw = ' '.join((first, middle, last))
    path = tmp_path / 'source.en.md'
    path.write_text('<!-- kb_page: 2 -->\n## Condition\n\n' + raw, encoding='utf8')
    block = next(b for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph')
    base = {**block, 'source_path': str(path), 'citation_route': 'system_a', 'binding_status': 'grounded', 'strict_locate': True,
            'evidence_source': 'citation_plan_claim_window', 'reader_evidence_source': 'citation_plan_located_block', 'reader_evidence_quote': raw}
    good = {**base, 'anchor': 'good', 'evidence_quote': first, 'answer_claim': first}
    unused = {**base, 'anchor': 'unused', 'evidence_quote': first + ' ' + last, 'answer_claim': ''}
    before = copy.deepcopy([good, unused])
    result = refresh_source_citation_cards([good, unused], render_locale='en')
    assert [good, unused] == before
    assert [row['anchor'] for row in result] == ['good']
    assert result[0]['evidence_quote'] == first
    assert refresh_source_citation_cards(result, render_locale='en') == result


def test_an_invalid_planned_preview_does_not_create_a_location_in_another_block(tmp_path):
    path = tmp_path / 'source.en.md'
    path.write_text('<!-- kb_page: 2 -->\n## Condition\n\nThe source only states the conditional expectation premise for targets.', encoding='utf8')
    block = next(b for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph')
    detail = {**block, 'source_path': str(path), 'citation_route': 'system_a', 'evidence_source': 'citation_plan_claim_window',
              'reader_evidence_source': 'citation_plan_located_block', 'reader_evidence_quote': block['raw_text'],
              'evidence_quote': 'An invented clinical study evaluated nine hundred patients.', 'answer_claim': ''}
    assert refresh_source_citation_cards([detail]) == []
