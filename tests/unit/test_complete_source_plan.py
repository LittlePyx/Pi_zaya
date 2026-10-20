from __future__ import annotations

import copy

import pytest

from api.claim_passage_plan import message_citation_plan, prepare_claim_passage_plan
from api.source_card_contract import prepare_source_card_message
from kb.source_blocks import load_source_blocks_cached

RAW = 'Our method uses only the input noisy image itself for training. The network is trained with dropout on pairs of Bernoulli-sampled instances of the input image.'
CLAIM = 'The network is trained with dropout on pairs of Bernoulli-sampled instances of the input noisy image [1].'


def _message(tmp_path):
    path = tmp_path / 'method.md'
    path.write_text('<!-- kb_page: 1 -->\n## Abstract\n\n' + RAW, encoding='utf8')
    block = next(b for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph')
    slot = {'source_path': str(path), 'heading_path': block['heading_path'], 'page_start': 1, 'page_end': 1, 'block_id': '', 'anchor_id': '', 'evidence_quote': RAW, 'candidate_hits': [1], 'preferred_system': 'system_a'}
    return {'role': 'assistant', 'content': CLAIM, 'meta': {'canonical_hit_paths': [str(path)], 'answer_quality': {'minimum_ok': True, 'has_hits': True, 'claim_evidence': {'minimum_ok': True}, 'citation_plan': {'budget': {'system_a': 1}, 'slots': [slot]}}}}, block


def test_complete_page_only_plan_obtains_a_literal_identity_before_claim_binding(tmp_path):
    message, block = _message(tmp_path); before = copy.deepcopy(message)
    located = prepare_source_card_message(message)
    result = prepare_claim_passage_plan(located, {'hits': []}, CLAIM)
    slot = message_citation_plan(result)['slots'][0]
    assert message == before and result['meta']['answer_quality']['minimum_ok'] is True
    assert all(slot[k] == block[k] for k in ('block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end'))
    assert slot['evidence_quote'] == RAW and slot['passage_bound_claims'] == [CLAIM]
    assert message_citation_plan(result)['budget'] == {'system_a': 1}
    assert prepare_source_card_message(result) == result


@pytest.mark.parametrize('failure', ['page', 'heading', 'block', 'anchor', 'source', 'text', 'ambiguous', 'failed_quality'])
def test_complete_answers_cannot_repair_conflicting_or_unsupported_source_identity(tmp_path, failure):
    message, _block = _message(tmp_path)
    slot = message_citation_plan(message)['slots'][0]
    if failure == 'page': slot.update(page_start=3, page_end=3)
    elif failure == 'heading': slot['heading_path'] = 'Other'
    elif failure == 'block': slot['block_id'] = 'stale'
    elif failure == 'anchor': slot['anchor_id'] = 'stale'
    elif failure == 'source': message['meta']['canonical_hit_paths'] = [str(tmp_path / 'other.md')]
    elif failure == 'text': slot['evidence_quote'] = RAW.replace('only', 'never')
    elif failure == 'ambiguous':
        path = tmp_path / 'method.md';path.write_text(path.read_text(encoding='utf8') + '\n\n' + RAW, encoding='utf8')
    elif failure == 'failed_quality': message['meta']['answer_quality']['claim_evidence']['minimum_ok'] = False
    before = copy.deepcopy(message)
    assert prepare_source_card_message(message) == before and message == before
