from __future__ import annotations

import copy
import re

import pytest

from api import chat_render
from api.claim_passage_plan import message_citation_plan
from tests.unit.test_n2n_target_condition import _chain


ANSWER = (
    'No — Noise2Noise does not require an explicit model of the corruption process [1].\n\n'
    'The statistical condition that remains necessary is that the corrupted targets must have the correct conditional expected value. '
    'The paper states that L2 minimization is unchanged "if we replace the targets with random numbers whose expectations match the targets," '
    'and that the optimal parameters remain unchanged "if input-conditioned target distributions p(y|x) are replaced with arbitrary distributions '
    'that have the same conditional expected values." In practice this means zero-mean corruption for L2 training (recovering the mean), '
    'with the appropriate summary (e.g., median) for other losses [1].\n\n'
    'So the necessary condition is a matching statistical summary — not a full likelihood or corruption model [1].'
)


def _fixture(tmp_path):
    path, _, hits, plan = _chain(tmp_path)
    rec = {'id': 2, 'role': 'assistant', 'content': ANSWER, 'meta': {
        'canonical_hit_paths': [str(path)], 'answer_quality': {'citation_plan': plan},
        'paper_guide_contracts': {'citation_plan': plan},
    }}
    return rec, {'hits': hits, 'render_locale': 'en'}


def test_sentence_context_keeps_abbreviation_and_quoted_sentence():
    from api.citation_display_registry import _citation_claim_context

    for sentence in (
        'Use the appropriate summary (e.g., median) for other losses [1].',
        'The paper states "the same conditional expected values." [1] Next sentence.',
        'The paper states "the same conditional expected values." [1](#a "source: paper.pdf | 3. Practical experiments") Next sentence.',
    ):
        marker = re.search(r'\[1\]', sentence)
        context = _citation_claim_context(sentence, *marker.span())
        assert context.startswith('Use the appropriate' if sentence.startswith('Use') else 'The paper states')
        assert 'Next sentence' not in context


def test_rendered_link_context_does_not_expand_over_other_paragraphs():
    from api.citation_display_registry import _citation_claim_context

    text = 'No explicit corruption model is required [1](#a "source: paper.pdf | 4. Discussion").\n\nThe targets must preserve conditional expectations [1](#b "source: paper.pdf | 3. Experiments").'
    marker = re.search(r'\[1\]\([^\n]*?\)', text)
    claim = _citation_claim_context(text, *marker.span())
    assert 'conditional expectations' not in claim


def test_full_render_distributes_paragraph_end_citation_to_verified_condition(tmp_path, monkeypatch):
    rec, pack = _fixture(tmp_path)
    before = copy.deepcopy((rec, pack))
    monkeypatch.setattr('ui.refs_renderer._is_temp_source_path', lambda path: False)
    monkeypatch.setattr(chat_render, '_effective_reference_render_pack', lambda raw: raw)
    monkeypatch.setattr(chat_render, '_answer_aligned_reference_render_pack', lambda raw, *args: raw)
    monkeypatch.setattr(chat_render, '_effective_citation_render_locale', lambda raw: 'en')
    user = {'id': 1, 'role': 'user', 'content': 'What statistical condition remains necessary?'}
    result = chat_render.enrich_messages_with_reference_render([user, rec], {1: pack}, conv_id='paragraph-passages')[-1]
    assert (rec, pack) == before
    for rendered in (result, chat_render.enrich_messages_with_reference_render([user, result], {1: pack}, conv_id='paragraph-passages')[-1]):
        cards = rendered['cite_details']
        assert {c['page_start'] for c in cards} == {2, 8}
        target = next(c for c in cards if c['page_start'] == 2)
        assert 'same conditional expected values' in target['evidence_quote']
        assert target['block_id'] and target['anchor_id']
        body = rendered['rendered_body']
        assert re.search(r'same conditional expected values\."\s*\[1\]\(#'+re.escape(target['anchor']), body)
        assert re.search(r'recovering the mean\),\s*\[1\]\(#'+re.escape(target['anchor']), body)
        discussion = next(c for c in cards if c['page_start'] == 8)
        assert re.search(r'other losses\s*\[1\]\(#'+re.escape(discussion['anchor']), body)
        assert re.search(r'corruption process\s*\[1\]\(#'+re.escape(discussion['anchor']), body)
        assert 'e.g., median' in body
        assert 'same conditional expected values' in rendered['copy_markdown']


@pytest.mark.parametrize('bad', ['page', 'anchor', 'block', 'heading', 'namesake', 'missing', 'different', 'unconditional', 'input', 'L1', 'number', 'negation', 'code', 'math', 'uncited', 'mixed'])
def test_paragraph_cannot_mint_condition_card_without_verified_support(tmp_path, bad):
    from api.claim_passage_plan import prepare_claim_passage_render

    rec, pack = _fixture(tmp_path)
    plan = message_citation_plan(rec)
    target = next(s for s in plan['slots'] if s['page_start'] == 2)
    if bad in ('page', 'anchor', 'block', 'heading'):
        target[{'page': 'page_start', 'anchor': 'anchor_id', 'block': 'block_id', 'heading': 'heading_path'}[bad]] = 99 if bad == 'page' else 'wrong'
    elif bad == 'namesake':
        target['source_path'] = str(tmp_path / 'other' / 'noise2noise.en.md')
    elif bad == 'missing':
        plan['slots'].remove(target)
    else:
        replacement = {
            'different': ('same conditional expected values', 'different conditional expected values'),
            'unconditional': ('conditional expected', 'unconditional expected'),
            'input': ('replace the targets', 'replace the inputs'),
            'L1': ('L2 minimization', 'L1 minimization'),
            'number': ('L2 minimization', 'L2 minimization with noise mean 7'),
            'negation': ('parameters remain unchanged', 'parameters change'),
        }.get(bad)
        if replacement:
            rec['content'] = ANSWER.replace(*replacement)
        elif bad == 'code':
            rec['content'] = '```text\n'+ANSWER+'\n```'
        elif bad == 'math':
            rec['content'] = '$$\n'+ANSWER+'\n$$'
        elif bad == 'uncited':
            rec['content'] = ANSWER.replace('[1]', '')
        elif bad == 'mixed':
            rec['content'] = ANSWER.replace('for other losses [1]', 'for other losses [1][2]')
    rec['meta']['answer_quality']['citation_plan'] = plan
    rec['meta']['paper_guide_contracts']['citation_plan'] = plan
    _, surface = prepare_claim_passage_render(rec, pack, rec['content'])
    if bad in ('different', 'unconditional', 'input', 'L1', 'number', 'negation'):
        assert not re.search(r'expected values\."\s*\[1\]', surface)
    else:
        assert surface == rec['content']


def test_no_reference_pack_cannot_enable_local_citation_actions(tmp_path):
    from api.claim_passage_plan import prepare_claim_passage_render

    rec, _ = _fixture(tmp_path)
    result, surface = prepare_claim_passage_render(rec, None, ANSWER)
    assert result == rec and surface == ANSWER


def test_distinct_verified_condition_blocks_are_ambiguous(tmp_path):
    from api.claim_passage_plan import prepare_claim_passage_render
    from kb.source_blocks import load_source_blocks_cached

    rec, pack = _fixture(tmp_path)
    plan = message_citation_plan(rec)
    target = next(s for s in plan['slots'] if s['page_start'] == 2)
    path = tmp_path / 'noise2noise.en.md'
    condition = next(b['text'] for b in load_source_blocks_cached(path) if b.get('page_start') == 2 and b['kind'] == 'paragraph')
    path.write_text(path.read_text(encoding='utf8')+'\n\n<!-- kb_page: 6 -->\n'+condition, encoding='utf8')
    duplicate = next(b for b in load_source_blocks_cached(path) if b.get('page_start') == 6 and b['kind'] == 'paragraph')
    plan['slots'].append({**target, **{k: duplicate[k] for k in ('block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end')}})
    rec['meta']['answer_quality']['citation_plan'] = plan
    rec['meta']['paper_guide_contracts']['citation_plan'] = plan
    _, surface = prepare_claim_passage_render(rec, pack, ANSWER)
    assert surface == ANSWER


def test_math_and_quoted_commas_are_not_clause_boundaries():
    from kb.citation_sentences import citation_sentence_spans

    for text in ('Use (mean, with fixed variance) [1].', 'The source says "mean, with fixed variance" [1].', '$f(x), with x=1$ [1].'):
        assert citation_sentence_spans(text) == [(0, len(text))]


@pytest.mark.parametrize('old_derived_slot', [False, True])
def test_already_cited_condition_sentences_keep_complete_direct_card(tmp_path, monkeypatch, old_derived_slot):
    from kb.source_blocks import load_source_blocks_cached

    rec, pack = _fixture(tmp_path)
    rec['content'] = (
        'No explicit corruption model is required [1].\n\n'
        'The paper states that L2 minimization is unchanged if the input-conditioned target distributions '
        'are replaced with arbitrary distributions that have the same conditional expected values [1]. '
        'This means training targets can be corrupted with zero-mean noise without changing what the network learns [1].\n\n'
        'The necessary statistical condition is a matching conditional expectation, not a full corruption model [1].'
    )
    if old_derived_slot:
        plan = message_citation_plan(rec)
        target = next(s for s in plan['slots'] if s['page_start'] == 2)
        block = next(b for b in load_source_blocks_cached(tmp_path/'noise2noise.en.md') if b['block_id'] == target['block_id'])
        plan['slots'].append({**target, 'evidence_quote': block['text'], 'evidence_selection_reason': 'claim_source_passage',
                              'passage_bound_claims': [rec['content'].split('\n\n')[1].split(' [1].')[0]+' [1].']})
        for key in ('answer_quality', 'paper_guide_contracts'):
            rec['meta'][key]['citation_plan'] = plan
    monkeypatch.setattr('ui.refs_renderer._is_temp_source_path', lambda path: False)
    monkeypatch.setattr(chat_render, '_effective_reference_render_pack', lambda raw: raw)
    monkeypatch.setattr(chat_render, '_answer_aligned_reference_render_pack', lambda raw, *args: raw)
    monkeypatch.setattr(chat_render, '_effective_citation_render_locale', lambda raw: 'en')
    rendered = chat_render.enrich_messages_with_reference_render([
        {'id': 1, 'role': 'user', 'content': 'What statistical condition remains necessary?'}, rec], {1: pack}, conv_id='existing-markers')[-1]
    cards = rendered['cite_details']
    pages = {c['anchor']: c['page_start'] for c in cards}
    markers = list(re.finditer(r'\[1\]\(#([^\s)]+)', rendered['rendered_body']))
    assert [pages[m[1]] for m in markers] == [8, 2, 2]
    # This synthetic summary scores only 3 against the source. Keep the
    # original threshold of 5: no local link is safer than borrowing page 8.
    assert '[1]' not in rendered['rendered_body'].split('\n\n')[-1]
    target_cards = [c for c in cards if c['page_start'] == 2]
    assert len(target_cards) == 1
    assert 'same conditional expected values' in target_cards[0]['evidence_quote']
    assert 'zero-mean noise without changing' in target_cards[0]['evidence_quote']


def test_full_audit_context_is_reread_from_each_exact_block(tmp_path):
    from api.claim_passage_plan import prepare_claim_passage_plan

    rec, pack = _fixture(tmp_path)
    before = copy.deepcopy((rec, pack))
    for slot in message_citation_plan(rec)['slots']:
        slot['citation_plan_full_evidence_quote'] = 'FAKE L1 target noise with mean 7.'
    original = copy.deepcopy(rec)
    result = prepare_claim_passage_plan(rec, pack, ANSWER)
    assert rec == original
    slots = message_citation_plan(result)['slots']
    assert all('FAKE' not in s.get('citation_plan_full_evidence_quote','') for s in slots)
    p2 = next(s for s in slots if s['page_start'] == 2)
    p8 = next(s for s in slots if s['page_start'] == 8)
    assert 'L 2 minimization' in p2['citation_plan_full_evidence_quote']
    assert 'conditional expected values' not in p8['citation_plan_full_evidence_quote']
    assert prepare_claim_passage_plan(result, pack, ANSWER) == result
    assert pack == before[1]


@pytest.mark.parametrize('condition', ['zero-mean corruption in training targets', 'matching conditional expectation'])
def test_general_summary_cannot_lend_a_statistical_condition(condition):
    from kb.source_passage_evidence import claim_passage_concepts_covered
    from tests.unit.test_n2n_target_condition import CONTRAST

    assert not claim_passage_concepts_covered('The requirement is '+condition, CONTRAST)
