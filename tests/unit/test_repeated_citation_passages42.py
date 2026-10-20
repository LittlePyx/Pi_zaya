from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

import pytest

from api.claim_passage_plan import message_citation_plan, prepare_claim_passage_render
from api import chat_render
from kb.source_blocks import load_source_blocks_cached

CORRECTION = 'We can therefore recover an estimate of the clean image by doubling our network output and subtracting its input.'
RAW = 'Note that without our correction step, the raw output of the network is still noisy.'
MIDDLE = 'It also states explicitly that without the correction step, the raw output of the network is still noisy.'
ANSWER = ('The paper states that the clean image is recovered by doubling the network output and subtracting its input [1]. '
          + MIDDLE + ' So removing the correction would leave a noisy estimate, contradicting the claim that the network directly outputs the clean image [1].')


def _fixture(tmp_path):
    path = tmp_path / 'noisier2noise.md'
    path.write_text('# Noisier2Noise\n<!-- kb_page: 3 -->\n## Method\n\n' + CORRECTION + '\n\n<!-- kb_page: 4 -->\n' + RAW, encoding='utf8')
    rows = [b for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph']
    first = rows[0]
    slot = {**first, 'preferred_system': 'system_a', 'source_path': str(path),
            'source_name': 'noisier2noise.pdf', 'candidate_hits': [1], 'evidence_quote': first['text'],
            'source_passage_bundle': True, 'direct_source_evidence': [dict(b) for b in rows]}
    plan = {'budget': {'system_a': 2, 'system_b': 0}, 'per_paragraph_budget': {'system_a': 2, 'system_b': 0}, 'slots': [slot]}
    rec = {'id': 2, 'role': 'assistant', 'content': ANSWER, 'meta': {'canonical_hit_paths': [str(path)],
           'answer_quality': {'citation_plan': plan}, 'paper_guide_contracts': {'citation_plan': plan}}}
    pack = {'hits': [{'text': first['text'], 'meta': {**first, 'source_path': str(path), 'source_passages': rows}}], 'render_locale': 'en'}
    return rec, pack, rows


def test_repeated_same_source_markers_distribute_to_the_uncited_middle_sentence(tmp_path):
    rec, pack, rows = _fixture(tmp_path)
    before = deepcopy((rec, pack))
    result, answer = prepare_claim_passage_render(rec, pack, rec['content'])
    assert MIDDLE + ' [1]' in answer
    assert answer.replace(MIDDLE + ' [1]', MIDDLE) == ANSWER
    assert answer.count('[1]') == 3
    slots = message_citation_plan(result)['slots']
    target = next(s for s in slots if s['block_id'] == rows[1]['block_id'])
    assert target['evidence_quote'] == RAW and target['page_start'] == 4
    assert {c.replace(' [1]', '') for c in target['passage_bound_claims']} == {MIDDLE}
    assert message_citation_plan(result)['per_paragraph_budget'] == {'system_a': 2, 'system_b': 0}
    assert (rec, pack) == before


def test_full_render_and_repeat_read_keep_the_direct_raw_output_card(tmp_path, monkeypatch):
    rec, pack, rows = _fixture(tmp_path)
    before = deepcopy((rec, pack))
    monkeypatch.setattr('ui.refs_renderer._is_temp_source_path', lambda path: False)
    monkeypatch.setattr(chat_render, '_effective_reference_render_pack', lambda raw: raw)
    monkeypatch.setattr(chat_render, '_answer_aligned_reference_render_pack', lambda raw, *args: raw)
    monkeypatch.setattr(chat_render, '_effective_citation_render_locale', lambda raw: 'en')
    user = {'id': 1, 'role': 'user', 'content': 'Can the output correction be removed?'}
    first = chat_render.enrich_messages_with_reference_render([user, rec], {1: pack}, conv_id='repeated-source')[-1]
    second = chat_render.enrich_messages_with_reference_render([user, first], {1: pack}, conv_id='repeated-source')[-1]
    for rendered in (first, second):
        assert rendered['content'] == ANSWER
        cards = rendered['cite_details']
        assert {c['block_id'] for c in cards} == {b['block_id'] for b in rows}
        raw = next(c for c in cards if c['block_id'] == rows[1]['block_id'])
        assert raw['evidence_quote'] == RAW and raw['page_start'] == 4
        assert re.search(re.escape(MIDDLE) + r'\s*\[1\]\(#' + re.escape(raw['anchor']), rendered['rendered_body'])
    assert (rec, pack) == before


@pytest.mark.parametrize('bad', ['mixed', 'missing', 'anchor', 'page', 'namesake', 'ambiguous', 'uncited', 'nonterminal', 'code', 'math'])
def test_repeated_source_distribution_keeps_original_verification_boundaries(tmp_path, bad):
    rec, pack, rows = _fixture(tmp_path)
    slot = message_citation_plan(rec)['slots'][0]
    child = slot['direct_source_evidence'][1]
    answer = ANSWER
    if bad == 'mixed': answer = ANSWER.replace('input [1]', 'input [2]')
    elif bad == 'missing': slot['direct_source_evidence'] = [slot['direct_source_evidence'][0]]
    elif bad == 'anchor': child['anchor_id'] = 'stale'
    elif bad == 'page': child['page_start'] = 9
    elif bad == 'namesake': child['source_path'] = str(tmp_path / 'other' / 'noisier2noise.md')
    elif bad == 'ambiguous':
        path = Path(slot['source_path'])
        path.write_text(path.read_text('utf8') + '\n\n' + RAW, encoding='utf8')
        duplicate = [b for b in load_source_blocks_cached(path) if b['text'] == RAW][-1]
        slot['direct_source_evidence'].append(duplicate)
    elif bad == 'uncited': answer = ANSWER.replace('[1]', '')
    elif bad == 'nonterminal': answer = ANSWER + ' An uncited ending.'
    elif bad == 'code': answer = '```text\n' + ANSWER + '\n```'
    elif bad == 'math': answer = '$$\n' + ANSWER + '\n$$'
    before = deepcopy((rec, pack))
    result, surface = prepare_claim_passage_render(rec, pack, answer)
    assert surface == answer
    assert not any(s.get('block_id') == rows[1]['block_id'] for s in message_citation_plan(result)['slots'])
    assert (rec, pack) == before
