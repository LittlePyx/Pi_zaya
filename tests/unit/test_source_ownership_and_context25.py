from __future__ import annotations

import copy

import pytest

from api.claim_passage_plan import _slot_supports
from api.citation_occurrence_support import verified_occurrence_candidates
from kb.claim_source_scope import owned_source_body_claim
from kb.evidence_binding import _system_a_domain_terms
from kb.refusal_context_excerpt import bind_refusal_context_excerpt
from kb.source_blocks import load_source_blocks


def test_pixel_correlation_does_not_cross_noise_and_signal_roles():
    assert 'pixel patch correlation' not in _system_a_domain_terms('噪声跨像素独立 + 信号相关')
    for text in ('像素之间存在空间相关性','图像块间的相关性','correlation between pixels'):
        assert 'pixel patch correlation' in _system_a_domain_terms(text)


def test_source_alias_is_removed_only_from_its_own_body_gate():
    assert owned_source_body_claim('N2V assumes independent noise.', 'noise2void.pdf').strip()=='assumes independent noise.'
    assert owned_source_body_claim('N2N uses noisy image pairs.', 'noise2void.pdf').startswith('N2N')
    assert 'N2VNet' in owned_source_body_claim('N2VNet is a different method.', 'noise2void.pdf')


def test_noise_and_signal_roles_and_exclusivity_still_require_explicit_evidence():
    from kb.explicit_noise_relations import signal_dependency_covered, white_noise_restriction_covered

    claim='N2V依赖噪声跨像素独立 + 信号相关这一对条件。'
    assert signal_dependency_covered(claim,'The signal however is assumed to contain statistical dependencies.')
    assert signal_dependency_covered(claim,'The underlying true signal exhibits some correlation.')
    assert not signal_dependency_covered(claim,'Noise has statistical dependencies. The signal is independent.')
    assert not signal_dependency_covered(claim,'The signal is assumed not to contain statistical dependencies.')
    assert not signal_dependency_covered(claim,'It is not true that the signal has statistical dependencies.')
    only='The method is restricted to independent white noise.'
    assert not white_noise_restriction_covered(only,'Our method allows spatially correlated noise.')
    assert not white_noise_restriction_covered(only,'The method is not restricted to independent white noise.')
    assert white_noise_restriction_covered(only,'The method is restricted to independent white noise.')


def test_actual_noise2void_assumptions_survive_an_own_alias_and_other_method_mentions(tmp_path):
    md=tmp_path/'noise2void.en.md'
    raw=('A blind-spot network can also use N2N training. Since we assume the noise to be pixel-wise independent given the signal, '
         'neighboring pixels carry no information about the noise. The signal however is assumed to contain statistical dependencies. '
         'We can extract the input and target from the same noisy training image.')
    md.write_text('<!-- kb_page: 3 -->\n# Noise2Void - Learning Denoising\n## Method\n\n'+raw,encoding='utf8')
    b=next(b for b in load_source_blocks(md) if b['kind']=='paragraph')
    row={**{k:b[k] for k in ('block_id','anchor_id','heading_path','page_start','page_end')},'source_path':str(md),'source_name':'noise2void.pdf',
         'evidence_quote':raw,'evidence_source':'verified_source_excerpt','citation_plan_slot':True,
         'selection_reason':'prompt_contract_block','evidence_selection_reason':'prompt_contract_block'}
    claim='N2V 依赖的是噪声跨像素独立 + 信号相关这一对条件。'
    assert _slot_supports(claim,row)
    assert verified_occurrence_candidates(row,[row],claim)
    assert not verified_occurrence_candidates(row,[{**row,'anchor_id':'wrong'}],claim)
    assert not verified_occurrence_candidates(row,[row],'N2V证明了信号之间的Pearson相关系数为0.9。')


def test_one_literal_reading_quote_can_bind_alongside_an_unrelated_old_citation(tmp_path):
    quote='It is important to note that these methods make different assumptions about the task, and thus direct comparison may be misleading.'
    md=tmp_path/'paper.md';md.write_text('<!-- kb_page: 4 -->\n## Experiments\n\n'+quote,encoding='utf8')
    b=next(b for b in load_source_blocks(md) if b['kind']=='paragraph')
    slot={**b,'source_path':str(md),'preferred_system':'system_a','candidate_hits':[1],'evidence_quote':quote}
    plan={'slots':[slot]};prior=copy.deepcopy(plan)
    answer='The current retrieved evidence cannot establish a universal ranking.\n\nSome old unsupported fact [1].\n\nChecked source excerpt (reading context)\n\n> '+quote+' [1]'
    bound=bind_refusal_context_excerpt(answer,plan,[str(md)])
    assert plan==prior and len(bound['slots'])==2
    context=bound['slots'][-1]
    assert context['evidence_quote']==quote and context['passage_bound_claims']==[quote]
    assert context['block_id']==b['block_id'] and context['page_start']==4
    assert bind_refusal_context_excerpt(answer,bound,[str(md)])==bound
    assert bind_refusal_context_excerpt(answer.replace('different assumptions','identical assumptions'),plan,[str(md)])==plan
    assert bind_refusal_context_excerpt(answer+'\n\n> '+quote+' [1]',plan,[str(md)])==plan
