from __future__ import annotations

import copy

import pytest

from api.claim_passage_plan import with_verified_slot_origin
from api.citation_occurrence_support import verified_occurrence_candidates
from api.citation_occurrence_support import expand_verified_occurrence_quotes
from api.citation_display_registry import _ANSWER_CITATION_LINK_RE, _citation_claim_context
from kb.explicit_source_attribution import source_attribution_covered
from kb.noise_model_prerequisites import noise_model_prerequisites_covered
from kb.source_blocks import load_source_blocks
from kb.source_question_facets import source_facet_score, source_facet_quote
from kb.source_domain_terms import normalize_verified_noise_sampling_terms
from kb.source_fact_scope import scope_explicit_source_facts
from kb.direct_question_evidence import direct_question_evidence_hits


QUOTE = 'We consider image denoising under a fixed, known noise model. We are able to capture a single noisy realization Y := X + N. Because the noise distribution is known, we can draw an additional synthetic noise sample M.'
CLAIM = '噪声分布是已知的，并且能够从中额外抽取一个合成噪声样本。'


def fixture(tmp_path):
    md = tmp_path/'noise.en.md'; md.write_text('<!-- kb_page: 3 -->\n# Noise\n## Method\n\n'+QUOTE,encoding='utf8')
    b = next(b for b in load_source_blocks(md) if b['kind']=='paragraph')
    slot = {**{k:b[k] for k in ('block_id','anchor_id','heading_path','page_start','page_end')},
            'source_path':str(md),'source_name':'noise.pdf','evidence_quote':QUOTE,
            'evidence_selection_reason':'prompt_contract_block','preferred_system':'system_a'}
    detail = {**slot,'anchor':'noise-1','evidence_source':'verified_source_excerpt','citation_plan_slot':True,
              'selection_reason':'citation_plan_evidence_overlay'}
    detail.pop('evidence_selection_reason')
    return slot,detail


def test_existing_verified_origin_survives_an_overlay_without_granting_a_fact_flag(tmp_path):
    slot,detail=fixture(tmp_path);before=copy.deepcopy((slot,detail))
    assert verified_occurrence_candidates(detail,[detail],CLAIM)==[]
    derived=with_verified_slot_origin(detail,slot)
    assert derived['selection_reason']=='citation_plan_evidence_overlay'
    assert derived['citation_plan_evidence_selection_reason']=='prompt_contract_block'
    assert verified_occurrence_candidates(derived,[derived],CLAIM)==[derived]
    assert (slot,detail)==before
    assert not any('grounded' in k or k.startswith('canonical') for k in derived)


@pytest.mark.parametrize('field,value',[
    ('source_path','other.md'),('block_id','wrong'),('anchor_id','wrong'),('heading_path','Wrong'),
    ('page_start',4),('page_end',4),('evidence_quote',QUOTE.replace('known','unknown')),
    ('evidence_quote',QUOTE.replace('known','KNOWN')),
])
def test_conflicting_display_or_slot_cannot_inherit_origin(tmp_path,field,value):
    slot,detail=fixture(tmp_path)
    for current,planned in (({**detail,field:value},slot),(detail,{**slot,field:value})):
        assert with_verified_slot_origin(current,planned)==current


def test_training_procedure_inside_its_own_paragraph_outranks_a_caption():
    prompt='In the standard additive procedure, what is the input and training target?'
    method='Our training procedure is as follows. Given Y, we create Z := Y + M. We then train the network to predict Y given Z as input, using a pixel-wise loss.'
    caption='Figure 2. A singly-noisy realization is our training target, and a doubly-noisy realization is the input to our network.'
    assert source_facet_score(prompt,method)>source_facet_score(prompt,caption)
    assert source_facet_quote(prompt,method)==method
    assert source_facet_score(prompt,method.replace('We then train','We do not train'))<20


@pytest.mark.parametrize('bad',[
    'This is cast as inpainting randomly missing pixels; only un-corrupted pixels are used for training.',
    r'The notation writes the instances as $\boldsymbol{b}_1,\dots,\boldsymbol{b}_M$.',
    '噪声模型从“可估计的未知量”变成了“必须给定的已知先验”。',
    '因此，Noisier2Noise 省掉的是干净/配对数据，而不是噪声模型：噪声模型从“可估计的未知量”变成了“必须给定的已知先验”。',
])
def test_an_abstract_or_known_model_does_not_cover_unstated_specialized_facts(bad):
    abstract='We train with dropout on pairs of Bernoulli-sampled instances of a single input noisy image and average the predictions.'
    assert not source_attribution_covered(bad,abstract+' '+QUOTE,'paper.pdf')
    assert not source_attribution_covered(bad,abstract+' Our noise model is known, and our network learns to denoise.','paper.pdf')
    assert source_attribution_covered(bad,bad,'paper.pdf')


def test_single_noisy_realization_alias_preserves_quantity_and_noise_state():
    claim='每张训练图像只有一次含噪实现。'
    assert noise_model_prerequisites_covered(claim,'We capture a single noisy realization Y.')
    assert not noise_model_prerequisites_covered(claim,'We capture multiple noisy realizations.')
    assert not noise_model_prerequisites_covered(claim,'We capture one clean realization.')


def test_noise_estimation_does_not_borrow_a_negative_or_prior_work_statement():
    claim='Our method estimates the noise model.'
    assert source_attribution_covered(claim,'We estimate the noise model.','paper')
    assert not source_attribution_covered(claim,'We do not estimate the noise model.','paper')
    assert not source_attribution_covered(claim,'Other methods estimate the noise model. Our method uses a known noise distribution.','paper')


def test_rejected_explicit_fact_is_omitted_without_removing_a_valid_claim_or_raw_message(tmp_path):
    slot,_=fixture(tmp_path);slot['candidate_hits']=[1]
    hits=[{'text':QUOTE,'meta':{'source_path':slot['source_path']}}]
    link='[1]'
    good='The noise model is known. '+link
    bad='噪声模型从“可估计的未知量”变成了“必须给定的已知先验”。 '+link
    original=good+'\n\n'+bad
    scope=lambda value,plan:scope_explicit_source_facts(value,citation_plan=plan,answer_hits=hits,slot_hit_numbers=lambda s,h:s.get('candidate_hits',[]))
    assert scope(original,{'slots':[slot]})==good
    assert bad in original
    assert scope(original,{'slots':[{**slot,'anchor_id':'wrong'}]})==original
    assert scope(original,{})==original
    assert scope(original,{'slots':[{**slot,'block_id':'','anchor_id':''}]})==good
    assert scope(original,{'slots':[{**slot,'evidence_quote':QUOTE.replace('known','KNOWN')}]})==original
    quoted='> '+bad+'\n\n```text\n'+bad+'\n```'
    assert scope(quoted,{'slots':[slot]})==quoted
    fenced='```text\n\n'+bad+'\n\n```'
    assert scope(fenced,{'slots':[slot]})==fenced


def test_verified_noise_distribution_draw_uses_the_technical_verb_without_changing_literals(tmp_path):
    slot,_=fixture(tmp_path);plan={'slots':[slot]};prompt='需要已知的噪声模型和什么额外先验？'
    answer='能够从中额外抽取一个合成噪声样本 $M$。\n\n“抽取一个合成噪声样本”\n\n> 抽取一个合成噪声样本。\n\n```text\n抽取一个合成噪声样本\n```'
    normalized=normalize_verified_noise_sampling_terms(answer,prompt=prompt,citation_plan=plan)
    assert normalized.startswith('能够从中额外采样得到一个合成噪声样本 $M$。')
    assert normalized.split('\n\n')[1:]==answer.split('\n\n')[1:]
    assert normalize_verified_noise_sampling_terms(normalized,prompt=prompt,citation_plan=plan)==normalized
    assert normalize_verified_noise_sampling_terms(answer,prompt='Explain a random data example.',citation_plan=plan)==answer
    assert normalize_verified_noise_sampling_terms(answer,prompt=prompt,citation_plan={'slots':[{**slot,'anchor_id':'wrong'}]})==answer


def test_explicit_indexed_notation_keeps_the_actual_index_expression_and_font():
    assert source_attribution_covered(r'The notation is $\boldsymbol{b}_{M+1}$.',r'Use $\boldsymbol{b}_{M + 1}$.','paper')
    assert not source_attribution_covered(r'The notation is $\boldsymbol{b}_{M+2}$.',r'Use $\boldsymbol{b}_{M+1}$.','paper')
    assert not source_attribution_covered(r'The notation is $\mathcal{b}_{M+1}$.',r'Use $\boldsymbol{b}_{M+1}$.','paper')
    text=r'The notation is $\boldsymbol{b}_{M+1}$ [1](#test).'
    match=next(_ANSWER_CITATION_LINK_RE.finditer(text))
    claim=_citation_claim_context(text,match.start(),match.end())
    assert not source_attribution_covered(claim,'We train on Bernoulli-sampled instances.','paper')
    assert source_attribution_covered(claim,r'Use $\boldsymbol{b}_{M+1}$.','paper')


def test_occurrence_can_show_its_actual_full_paragraph_without_borrowing_prior_work_as_our_data(tmp_path):
    prior='Other dataset-based denoisers train over an external dataset of noisy/clean image pairs.'
    own='Our method uses the input noisy image itself for training, with dropout on pairs of Bernoulli-sampled instances.'
    raw='*'+prior+' '+own+'*'
    md=tmp_path/'self2self.en.md';md.write_text('<!-- kb_page: 1 -->\n# Study\n## Abstract\n\n'+raw,encoding='utf8')
    b=next(b for b in load_source_blocks(md) if b['kind']=='paragraph')
    row={**b,'source_path':str(md),'source_name':'self2self.pdf','evidence_quote':own,'card_evidence':own,
         'anchor':'self-1','evidence_source':'verified_source_excerpt','citation_plan_slot':True,'citation_route':'system_a',
         'selection_reason':'prompt_contract_block','render_locale':'en'}
    text='Our method uses the input noisy image itself for training [1](#self-1).\n\nOther dataset-based denoisers train over an external dataset of noisy/clean image pairs [1](#self-1).'
    before=copy.deepcopy(row)
    expanded=expand_verified_occurrence_quotes(text,[row],_ANSWER_CITATION_LINK_RE,_citation_claim_context)[0]
    assert expanded['evidence_quote']==raw and expanded['card_evidence']==raw
    assert expanded['block_id']==row['block_id'] and row==before
    from api.chat_render import _refine_system_a_cite_evidence_from_citation_plan
    from api.citation_display_registry import remap_system_a_citations_for_display
    actual=_refine_system_a_cite_evidence_from_citation_plan([expanded],{'slots':[{**row,'preferred_system':'system_a','candidate_hits':[1]}]},render_locale='en')[0]
    _,final,_=remap_system_a_citations_for_display(text,[{**actual,'num':1,'linked_nums':[1]}])
    assert final[0]['evidence_quote']==raw
    assert not source_attribution_covered('Our method trains on an external dataset of noisy images.',raw,'self2self.pdf')
    assert source_attribution_covered('Other dataset-based denoisers train over an external dataset.',raw,'self2self.pdf')
    wrong='Our method trains on an external dataset of noisy images [1](#self-1).'
    assert expand_verified_occurrence_quotes(wrong,[row],_ANSWER_CITATION_LINK_RE,_citation_claim_context)==[row]


def test_noise_independence_comparison_selects_both_actual_statistical_premises(tmp_path):
    caption='Figure 1. Our training procedure requires neither noisy image pairs nor clean target images.'
    premises='We make two simple statistical assumptions. The signal is not pixel-wise independent, but the noise is conditionally pixel-wise independent given the signal.'
    path=tmp_path/'noise2void.en.md'
    path.write_text('<!-- kb_page: 2 -->\n# Study\n\n'+caption+'\n\n'+premises,encoding='utf8')
    hits=direct_question_evidence_hits(path,'比较 Noise2Void 和 Noise2Self 对噪声的独立性要求。')
    assert hits[0]['text']==premises
    assert caption not in hits[0]['text']
