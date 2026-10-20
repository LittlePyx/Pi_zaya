from __future__ import annotations

import pytest

from kb.citation_sentences import citation_sentence_spans, split_definition_bridges
from kb.explicit_source_attribution import source_attribution_covered
from kb.explicit_noise_relations import iid_relation_covered, signal_dependency_covered
from kb.waived_data_requirements import waived_data_requirements_covered
from kb.direct_question_evidence import direct_question_evidence_hits


def test_two_loss_statistics_are_checked_separately_without_rewriting():
    text = 'The L2 loss is minimized by the mean (average), while the L1 loss is minimized by the median.'
    parts = split_definition_bridges(text)
    assert len(parts) == 2
    assert ''.join(text[a:b] for a,b in citation_sentence_spans(text)) == text
    mean = 'With the L2 loss the network learns to output the average of all plausible explanations.'
    median = 'The L1 loss recovers the median of the targets.'
    assert not source_attribution_covered(text, mean, 'paper.pdf')
    assert not source_attribution_covered(text, median, 'paper.pdf')
    assert source_attribution_covered(parts[0], mean, 'paper.pdf')
    assert source_attribution_covered(parts[1], median, 'paper.pdf')


def test_named_paper_contrast_and_individually_marked_facts_split():
    text = 'Self2Self 要求分量独立、零均值，而 Noisier2Noise 要求原始噪声与合成样本独立同分布。'
    parts = split_definition_bridges(text)
    assert len(parts) == 2
    assert '零均值' in parts[0] and 'Noisier2Noise' not in parts[0]
    assert 'Noisier2Noise' in parts[1] and 'Self2Self' not in parts[1]
    marked = '该方法支持空间相关噪声 [10002]，并指出需要从噪声分布采样 [10002]。'
    assert len(split_definition_bridges(marked)) == 2
    assert ''.join(marked[a:b] for a,b in citation_sentence_spans(marked)) == marked


@pytest.mark.parametrize('text', [
    'It is false that the L2 loss learns the mean, while the L1 loss learns the median.',
    'If the L2 loss learns the mean, while the L1 loss learns the median, consider this case.',
    'The caption says "Self2Self requires independence, whereas Noisier2Noise handles correlation".',
    '`Self2Self requires independence, whereas Noisier2Noise handles correlation`',
    '$g(Self2Self, whereas Noisier2Noise)$',
    'The methods (Self2Self requires independence, whereas Noisier2Noise handles correlation) differ.',
])
def test_clause_boundaries_do_not_turn_scoped_content_into_assertions(text):
    assert len(split_definition_bridges(text)) == 1


def test_quality_question_does_not_select_noise_assumptions(tmp_path):
    paper=tmp_path/'blind.md'
    quality='We introduce N2V. It does not require noisy image pairs, nor clean target images. Intuitively, N2V cannot be expected to outperform methods that have more information available during training. Still, performance drops in moderation.'
    stats='In this work we make two simple statistical assumptions: the signal is not pixel-wise independent, the noise is conditionally pixel-wise independent given the signal.'
    paper.write_text('# Blind denoiser\n\n<!-- kb_page: 1 -->\n\n'+quality+'\n\n<!-- kb_page: 2 -->\n\n'+stats,encoding='utf8')
    question='本文是否声称这种训练方式应当胜过拥有干净目标或独立噪声对的训练？作者怎样解释这个性能边界？'
    hits=direct_question_evidence_hits(paper,question)
    assert hits and hits[0]['text']==quality and hits[0]['meta']['page_start']==1
    assumption_hits=direct_question_evidence_hits(paper,'信号和噪声分别作了哪两条统计假设？')
    assert assumption_hits and stats in assumption_hits[0]['text']


def test_measurement_question_keeps_modality_and_independent_events(tmp_path):
    paper=tmp_path/'measurements.md'
    intro='We use groups of features whose noise is independent conditional on the true signal to predict one another.'
    actual='We demonstrate this on natural image and microscopy data, where we exploit noise independence between pixels, and on single-cell gene expression data, where we exploit independence between detections of individual molecules.'
    paper.write_text('# Measurements\n\n<!-- kb_page: 1 -->\n\n'+actual+'\n\n<!-- kb_page: 2 -->\n\n'+intro,encoding='utf8')
    hits=direct_question_evidence_hits(paper,'Besides natural and microscopy images, what measurement modality is demonstrated, and what independent events does the method exploit there?')
    assert hits and hits[0]['text']==actual
    assert not source_attribution_covered('The additional modality is single-cell gene expression data.',intro,'paper.pdf')
    assert source_attribution_covered('The additional modality is single-cell gene expression data.',actual,'paper.pdf')


def test_selection_proof_does_not_waive_clean_training_data():
    actual='That is, the self-supervised loss is the sum of the ordinary supervised loss and the variance of the noise. By minimizing the self-supervised loss over a class of J-invariant functions, one may find the optimal denoiser for a given dataset.'
    claim='因为这一项与去噪器 $f$ 无关，所以最小化自监督损失等价于最小化有监督损失，从而无需干净参照 $y$ 就能选出最优去噪器。'
    assert waived_data_requirements_covered(claim,actual)
    assert not waived_data_requirements_covered(claim,'The noise model is known and we can draw synthetic samples.')
    assert not waived_data_requirements_covered('因此无需干净训练数据即可训练去噪器。',actual)


def test_training_output_roles_cannot_borrow_the_correction_operation():
    correction='We recover an estimate of the clean image by doubling our network output and subtracting its input.'
    recipe='Our training procedure is as follows. Given Y, we create a noisier version Z = Y + M. We then train the network to predict Y given Z as input.'
    for claim in ('The network output is an estimate of Y, not X.', 'The network is trained to map doubly-noisy input to singly-noisy output.'):
        assert not source_attribution_covered(claim,correction,'paper.pdf')
        assert source_attribution_covered(claim,recipe,'paper.pdf')
    assert not source_attribution_covered('Inspect how M enters the training target.',recipe,'paper.pdf')
    assert source_attribution_covered('The training target is Y.',recipe,'paper.pdf')


def test_synthetic_label_after_variable_and_anaphoric_noise_negation():
    claim='其方法把 $X+N$ 视为原始含噪图、$M$ 视为来自同一噪声分布的合成样本。'
    wrong='Our method is applicable to white noise and spatially correlated noise.'
    actual='X + N is the original noisy image, and M is a synthetic sample from the same noise distribution.'
    assert not source_attribution_covered(claim,wrong,'paper.pdf')
    assert source_attribution_covered(claim,actual,'paper.pdf')
    assert not iid_relation_covered('前者不要求噪声在空间上不相关，后者也不要求同一幅图内的分量独立。','Assume the noise components are independent and of zero mean.')


def test_expectation_signal_and_loss_gap_need_their_actual_relations():
    boundary='We stick to denoising because multiple noisy observations help recover the true signal. This is not the case for blur.'
    for claim in ("The noisy input's expected value equals the clean signal.",'The scheme depends on the expected value of the noisy input equaling the clean signal.'):
        assert not source_attribution_covered(claim,boundary,'paper.pdf')
        assert source_attribution_covered(claim,'The expectation of the corrupted input data is the clean target that we seek to restore.','paper.pdf')
    signal='The underlying signal is assumed to be statistically predictable from neighbors.'
    assert not signal_dependency_covered(signal,'The receptive field excludes the pixel itself.')
    assert signal_dependency_covered(signal,'The signal however is assumed to contain statistical dependencies.')
    gap='The second term does not depend on f.'
    assert not source_attribution_covered(gap,'The function f(x)J does not depend on xJ.','paper.pdf')
    assert source_attribution_covered(gap,'The self-supervised loss is the sum of the ordinary supervised loss and the variance of the noise.','paper.pdf')


def test_verbatim_quotes_need_the_actual_passage_in_both_languages():
    literal='a wide variety of noise models, including spatially structured noise'
    wrong='Our method is applicable to both pixel-wise iid white noise and spatially correlated noise.'
    for claim in (f'论文称“{literal}”。',f'The approach covers "{literal}".'):
        assert not source_attribution_covered(claim,wrong,'paper.pdf')
        assert source_attribution_covered(claim,literal,'paper.pdf')


def test_clause_boundaries_remain_stable_after_marker_insertion():
    text='Self2Self 要求噪声分量独立、零均值，而 Noisier2Noise 要求两个噪声样本独立同分布。'
    for marker in ('[1]', '[[SUPPORT:DOC-1]]', '[1](#a "source: J minimizing (1)")'):
        marked=text.replace('，而', '， '+marker+'而')
        parts=split_definition_bridges(marked)
        assert len(parts)==2 and '零均值' in parts[0] and marker in parts[0]
        assert parts[1].startswith('而 Noisier2Noise')
        assert ''.join(marked[a:b] for a,b in citation_sentence_spans(marked))==marked
        coordinated='该方法支持空间相关噪声， '+marker+'并指出需要从噪声分布采样 [2]。'
        assert len(split_definition_bridges(coordinated))==2
    numeric='SCIGS obtains 30.2 dB [1], while SCINeRF obtains 31.5 dB [2].'
    assert len(split_definition_bridges(numeric))==1


def test_literal_single_variable_typesetting_preserves_variable_identity():
    claim='Proposition 1 opens with "Suppose $x$ is an unbiased estimator of $y$, i.e.".'
    source='Proposition 1. Suppose x is an unbiased estimator of y, i.e.'
    assert source_attribution_covered(claim,source,'paper.pdf')
    assert not source_attribution_covered(claim.replace('$y$','$z$'),source,'paper.pdf')


def test_j_invariance_cannot_reverse_the_excluded_subset():
    source='The function f(x)J does not depend on xJ.'
    wrong='$f$ 是 $J$-不变的，即 $f(x)$ 的第 $J$ 个分量不依赖 $x$ 在补集 $J^c$ 上的噪声。'
    assert not source_attribution_covered(wrong,source,'paper.pdf')
    assert source_attribution_covered(wrong.replace('在补集 $J^c$ 上的噪声','在 $J$ 上的值'),source,'paper.pdf')


def test_other_sentences_numbers_do_not_merge_comparison_citations():
    from api.citation_display_registry import _citation_claim_context
    text='Self2Self 要求分量独立、零均值， [1]而 Noisier2Noise 允许空间相关噪声 [2]。\n\nThe loss uses equations (7) and (8).'
    for marker,own,other in (('[1]','Self2Self','Noisier2Noise'),('[2]','Noisier2Noise','Self2Self')):
        start=text.index(marker);claim=_citation_claim_context(text,start,start+len(marker))
        assert own in claim and other not in claim and '(7)' not in claim
    quoted='该文明确声称方法“applicable to both pixel-wise i.i.d. white noise and spatially correlated noise” [2]，并指出它需要从噪声分布采样 [2]。\n\nEquation (8).'
    first=quoted.index('[2]');claim=_citation_claim_context(quoted,first,first+3)
    assert 'spatially correlated' in claim and '采样' not in claim


def test_link_titles_do_not_change_the_claims_at_each_citation():
    from api.citation_display_registry import _citation_claim_context
    first='[2](#a "source: noisier2noise.pdf | ref 2 | 2. Related Work")'
    second='[2](#b "source: noisier2noise.pdf | ref 2 | 3. Method / 3.2. Noisier2Noise")'
    text='该文明确声称方法“applicable to both pixel-wise i.i.d. white noise and spatially correlated noise” '+first+'，并指出需要能够从噪声分布中采样 '+second+'。'
    start=text.index(first);claim=_citation_claim_context(text,start,start+len(first))
    assert 'spatially correlated' in claim and '采样' not in claim
    start=text.index(second);claim=_citation_claim_context(text,start,start+len(second))
    assert '采样' in claim and 'spatially correlated' not in claim
    assert ''.join(text[a:b] for a,b in citation_sentence_spans(text))==text


def test_short_exact_bound_claim_is_identity_not_substring_support():
    from kb.source_passage_evidence import claim_occurrence_matches,claim_bound_candidates
    claim='且 $M$ 与 $N$ 是 i.i.d.、在给定 $Z$ 时具有相同期望 [2]。'
    assert claim_occurrence_matches(claim,[claim])
    assert not claim_occurrence_matches('并非'+claim,[claim])
    assert not claim_occurrence_matches('', [''])
    wrong={'block_id':'white','passage_bound_claims':['该方法支持空间相关噪声。']}
    actual={'block_id':'sample-pair','passage_bound_claims':[claim]}
    assert claim_bound_candidates([wrong,actual],claim)==[actual]


def test_minimal_comparison_keeps_both_papers_and_sample_premise_with_original_budget(tmp_path,monkeypatch):
    from api import chat_render
    from kb.source_blocks import load_source_blocks_cached
    s2s=tmp_path/'self2self.en.md';noisier=tmp_path/'noisier2noise.en.md'
    components='Proposition 1. Assume the noise components are independent and of zero mean.'
    spatial='Our method is applicable to both pixel-wise i.i.d. white noise and spatially correlated noise.'
    samples='Recall that M and N are i.i.d. and therefore and thus have the same expectation given Z,'
    s2s.write_text('# Self2Self\n\n<!-- kb_page: 4 -->\n\n'+components,encoding='utf8')
    noisier.write_text('# Noisier2Noise\n\n<!-- kb_page: 2 -->\n\n'+spatial+'\n\n<!-- kb_page: 3 -->\n\n'+samples,encoding='utf8')
    a=next(b for b in load_source_blocks_cached(s2s) if b['kind']=='paragraph')
    b,c=[b for b in load_source_blocks_cached(noisier) if b['kind']=='paragraph']
    def slot(path,block,number):
        return {'preferred_system':'system_a','source_path':str(path),'source_name':path.stem,
            'candidate_hits':[number],'evidence_quote':block['raw_text'],'source_passage_bundle':True,
            'evidence_selection_reason':'prompt_contract_block',
            **{k:block[k] for k in ('block_id','anchor_id','heading_path','page_start','page_end')}}
    first=slot(s2s,a,1);second=slot(noisier,b,2)
    second['direct_source_evidence']=[{**c,'source_path':str(noisier),'text':c['raw_text']}]
    plan={'budget':{'system_a':2,'system_b':0},'per_paragraph_budget':{'system_a':2,'system_b':0},'slots':[first,second]}
    answer='Self2Self 命题1要求噪声分量独立且零均值 [1]。\n\n该文明确声称方法“applicable to both pixel-wise i.i.d. white noise and spatially correlated noise” [2]。且 $M$ 与 $N$ 是 i.i.d.、在给定 $Z$ 时具有相同期望 [2]。'
    rec={'id':2,'role':'assistant','content':answer,'meta':{'canonical_hit_paths':[str(s2s),str(noisier)],
        'answer_quality':{'has_hits':True,'minimum_ok':True,'citation_plan':plan},'paper_guide_contracts':{'citation_plan':plan}}}
    pack={'render_locale':'en','hits':[{'text':block['raw_text'],'meta':{'source_path':str(path),'source_name':path.stem,'ref_answer_citation_num':number}}
        for path,block,number in ((s2s,a,1),(noisier,b,2))]}
    monkeypatch.setattr('ui.refs_renderer._is_temp_source_path',lambda _:False)
    monkeypatch.setattr(chat_render,'_effective_reference_render_pack',lambda value:value)
    monkeypatch.setattr(chat_render,'_answer_aligned_reference_render_pack',lambda value,*args:value)
    monkeypatch.setattr(chat_render,'_effective_citation_render_locale',lambda _:'en')
    user={'id':1,'role':'user','content':'Compare within-image noise independence and between-sample independence.'}
    for _ in range(2):
        rec=chat_render.enrich_messages_with_reference_render([user,rec],{1:pack},conv_id='comparison-original-budget')[-1]
        assert rec['content']==answer
        assert {d['block_id'] for d in rec['cite_details']}=={a['block_id'],b['block_id'],c['block_id']}
        assert next(d for d in rec['cite_details'] if d['block_id']==c['block_id'])['evidence_quote']==samples
        assert rec['meta']['answer_quality']['citation_plan']['per_paragraph_budget']['system_a']==2
