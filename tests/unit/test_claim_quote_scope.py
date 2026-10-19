from __future__ import annotations

import copy

from api.claim_passage_plan import message_citation_plan, prepare_claim_passage_plan
from kb.source_blocks import load_source_blocks_cached


def test_a_short_excerpt_cannot_borrow_a_claim_from_its_unshown_parent(tmp_path):
    short = 'We can sample from the noise distribution to obtain additional synthetic noise.'
    raw = 'Our method requires a known noise distribution. ' + short
    claim = 'Our method requires a known noise distribution and can sample from the noise distribution [1].'
    path = tmp_path / 'method.en.md'
    path.write_text('<!-- kb_page: 3 -->\n## Method\n\n' + raw, encoding='utf8')
    block = next(b for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph')
    slot = {**block, 'source_path': str(path), 'source_name': 'method.en.md', 'evidence_quote': short, 'preferred_system': 'system_a', 'candidate_hits': [1]}
    rec = {'role': 'assistant', 'content': claim, 'meta': {'canonical_hit_paths': [str(path)], 'answer_quality': {'citation_plan': {'slots': [slot], 'budget': {'system_a': 1}}}}}
    pack = {'hits': [{'text': raw, 'meta': {'source_path': str(path)}}]}
    before = copy.deepcopy((rec, pack))
    result = prepare_claim_passage_plan(rec, pack, claim)
    slots = message_citation_plan(result)['slots']
    assert (rec, pack) == before
    assert len(slots) == 2
    full = next(s for s in slots if s.get('passage_bound_claims'))
    assert full['evidence_quote'] == raw and full['block_id'] == block['block_id']
    assert full['passage_bound_claims'] == [claim]
    assert message_citation_plan(result)['budget'] == {'system_a': 1}
    assert prepare_claim_passage_plan(result, pack, claim) == result
    rec['meta']['answer_quality']['citation_plan']['slots'][0]['page_start'] = 99
    assert len(message_citation_plan(prepare_claim_passage_plan(rec, pack, claim))['slots']) == 1
