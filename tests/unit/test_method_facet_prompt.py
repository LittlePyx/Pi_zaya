from __future__ import annotations

import pytest

from kb.direct_question_evidence import attach_direct_source_evidence, direct_question_evidence_hits, prioritize_direct_question_hits
from kb.method_facet_prompt import method_facet_prompt_instructions, method_facet_prompt_quote
from kb.paper_guide_answer_selection import _merge_same_source_answer_hits
from kb.source_question_facets import source_facet_quote


def _plan(tmp_path):
    path=tmp_path/'method.md'
    irrelevant='Previous work uses unrelated measurements and benchmark designs. '*14
    owned='We also allow spatially correlated noise models. On the other hand, we require the ability to sample from the noise distribution.'
    path.write_text('<!-- kb_page: 2 -->\n## Method\n\nOur background review is as follows. '+irrelevant+owned+'\n<!-- kb_page: 3 -->\nRecall that U and V are i.i.d. and therefore have exchangeable roles in our noise sampling construction.\n',encoding='utf8')
    question='Does this method allow spatially correlated noise? Explain the relationship between additional noise samples.'
    hits=direct_question_evidence_hits(path,question)
    lead=hits[0]
    slot={**lead['meta'],'preferred_system':'system_a','candidate_hits':[1],'evidence_quote':lead['text']}
    return attach_direct_source_evidence({'slots':[slot]},_merge_same_source_answer_hits(hits))['slots'][0],owned


def test_long_background_does_not_hide_complete_method_premises_or_extra_source(tmp_path):
    slot,owned=_plan(tmp_path)
    assert slot['evidence_quote']==owned
    prompt=method_facet_prompt_quote(slot)
    assert owned in prompt and 'U and V are i.i.d.' in prompt
    assert len(prompt)<=720 and 'Previous work' not in prompt
    assert 'p. 3' in prompt
    assert method_facet_prompt_instructions([slot])


@pytest.mark.parametrize('bad', ['page_start','anchor_id','block_id','heading_path','source_path'])
def test_prompt_cannot_expose_a_supplementary_passage_with_a_forged_locator(tmp_path,bad):
    slot,_=_plan(tmp_path)
    slot['direct_source_evidence']=[dict(row) for row in slot['direct_source_evidence']]
    row=next(r for r in slot['direct_source_evidence'] if r['page_start']==3)
    row[bad]=99 if bad=='page_start' else 'different'
    assert 'U and V' not in method_facet_prompt_quote(slot)
    assert not method_facet_prompt_instructions([slot])


def test_long_extra_passage_is_omitted_whole_within_the_original_budget(tmp_path):
    slot,_=_plan(tmp_path)
    prompt=method_facet_prompt_quote(slot,max_len=len(slot['evidence_quote'])+10)
    assert prompt==slot['evidence_quote'] and 'U and V' not in prompt


def test_clause_selection_retains_negated_requirement_and_source_bytes():
    text='Our method also supports several other inputs. We do not require the ability to sample from the noise distribution.'
    quote=source_facet_quote('Does this method require a noise model?',text)
    assert quote in text and 'do not require' in quote


def test_sample_identity_premise_is_kept_with_its_original_observation_roles():
    text='The previous result concerns a different task. In the context of image denoising, this means that we can recover an estimate of a clean image given a noisy one by an analogous process in which X + N is the original noisy image, and M is a synthetic sample from the same noise distribution.'
    quote=source_facet_quote('Explain spatially correlated noise and the relationship between additional noise samples.',text)
    assert quote in text and quote.startswith('In the context')
    assert 'X + N is the original noisy image' in quote and 'M is a synthetic sample' in quote


def test_comparison_recovers_complementary_scoped_facets_within_the_existing_passage_budget(tmp_path):
    path=tmp_path/'method.md'
    path.write_text('<!-- kb_page: 2 -->\nWe also allow spatially correlated noise models, but require the ability to sample from the noise distribution.\n<!-- kb_page: 3 -->\nIn image denoising, we take X + N as the original noisy image, and M is a synthetic sample from the same noise distribution.\n\nRecall that M and N are i.i.d. and therefore and thus\n',encoding='utf8')
    other=tmp_path/'first.md'
    other.write_text('<!-- kb_page: 4 -->\n**Proposition 1.** Assume the noise components are independent and of zero mean.\n\nThe remaining variance term is constant for the minimization.\n\nThe following explanation concerns an optional proof detail.\n',encoding='utf8')
    prompt='Compare Proposition 1 and spatially correlated noise. Distinguish additional noise samples from components within an image.'
    candidates=direct_question_evidence_hits(path,prompt,limit=2)+direct_question_evidence_hits(other,prompt,limit=2)
    assert not any('i.i.d.' in h['text'] for h in candidates)
    selected=prioritize_direct_question_hits(candidates,candidates,limit=4,prompt=prompt)
    assert len(selected)==4
    assert all(any(term in h['text'] for h in selected) for term in ('original noisy image','i.i.d.','spatially correlated','Proposition 1'))


def test_complete_statistical_assumptions_survive_real_source_bundle_truncation(tmp_path):
    path=tmp_path/'assumptions.md'
    premise='In this work we make two simple statistical assumptions: (i) the signal is not pixel-wise independent, (ii) the noise is conditionally pixel-wise independent given the signal.'
    path.write_text('<!-- kb_page: 2 -->\n'+('Earlier methods need unrelated training data. '*40)+premise+' '+('We evaluate unrelated benchmark details. '*40),encoding='utf8')
    hits=direct_question_evidence_hits(path,'What statistical assumptions are made about the signal and noise?')
    assert len(hits)==1 and hits[0]['text']==premise
    merged=_merge_same_source_answer_hits(hits*2)
    assert premise in merged[0]['text'] and 'Earlier methods' not in merged[0]['text']
    slot={**hits[0]['meta'],'preferred_system':'system_a','candidate_hits':[1],'evidence_quote':hits[0]['text']}
    bound=attach_direct_source_evidence({'slots':[slot]},merged)['slots'][0]
    assert bound['evidence_quote']==premise and bound['page_start']==2
    path.write_text('<!-- kb_page: 2 -->\n'+premise.replace('conditionally pixel-wise independent given the signal','spatially correlated'),encoding='utf8')
    assert not direct_question_evidence_hits(path,'What statistical assumptions are made about the signal and noise?')


def test_unclassified_slot_cannot_gain_supplementary_prompt_evidence(tmp_path):
    slot,_=_plan(tmp_path)
    slot.pop('direct_method_facets')
    assert method_facet_prompt_quote(slot)==slot['evidence_quote']
    assert not method_facet_prompt_instructions([slot])
