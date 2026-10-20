from __future__ import annotations

import pytest

from api.claim_passage_plan import _supports
from kb.claim_evidence_runtime import _support_score
from kb.explicit_noise_relations import iid_relation_covered
from kb.evidence_binding import assess_system_a_hit_binding
from kb.source_passage_evidence import claim_passage_concepts_covered


def test_iid_abbreviation_covers_the_complete_relation_at_the_unchanged_support_threshold():
    claim='原始噪声与额外生成噪声之间（U,V i.i.d.）'
    source='Recall that U and V are i.i.d. and therefore have exchangeable roles in our noise sampling construction.'
    assert _support_score(claim,source)>=5
    assert _supports(claim,source,'','unseen.pdf')
    binding=assess_system_a_hit_binding(answer_claim=claim,hit={'text':source},meta={},heading='',evidence_quote=source,source_name='unseen.pdf')
    assert not binding['suppress_link']


def test_bare_iid_source_states_the_complete_relation_without_extra_keyword_padding():
    claim='原始噪声与额外生成噪声之间（$M,N$ i.i.d.）'
    source='Recall that M and N are i.i.d. and therefore and thus'
    assert _supports(claim,source,'','unseen.pdf')


def test_other_methods_negation_in_previous_sentence_cannot_negate_iid_premise():
    claim = '给定两个 i.i.d. 噪声样本之和，对其中一个的最佳估计是二者之和的一半。'
    relation = 'Given the sum of two i.i.d. noise samples, the best estimate of one of them is half of their sum.'
    source = 'Other methods do not require sampling from this distribution. ' + relation
    assert iid_relation_covered(claim, source)
    assert _support_score(claim, source) >= 5
    assert not iid_relation_covered(claim, 'It is false that ' + relation)


def test_iid_scope_does_not_equate_tex_variables_or_samples_with_pixel_components():
    source = 'Recall that M and N are i.i.d. and therefore and thus'
    assert iid_relation_covered(r'额外噪声 $\boldsymbol{M},\boldsymbol{N}$ 独立同分布', source)
    assert not iid_relation_covered(r'额外噪声 $\boldsymbol{P},\boldsymbol{Q}$ 独立同分布', source)
    assert not iid_relation_covered('单幅图像内噪声分量独立同分布', 'Given the sum of two i.i.d. noise samples, the best estimate is half their sum.')


@pytest.mark.parametrize('source',[
    'We allow spatially correlated noise models and require the ability to sample noise.',
    'Our method also handles pixel-wise i.i.d. white noise.',
    'Recall that A and B are i.i.d. samples of a noise model.',
    'Recall that U and V are independent but have different distributions.',
    'Recall that U and V have the same marginal distribution but can be dependent.',
    'Recall that U and V are not necessarily i.i.d.',
])
def test_sample_relation_cannot_borrow_a_partial_different_or_pixel_only_statement(source):
    claim='U 和 V 为独立同分布的两份噪声（i.i.d.）'
    assert not iid_relation_covered(claim,source)
    assert not _supports(claim,source,'','unseen.pdf')
    binding=assess_system_a_hit_binding(answer_claim=claim,hit={'text':source},meta={},heading='',evidence_quote=source,source_name='unseen.pdf')
    assert binding['suppress_link']


def test_raw_output_claim_needs_its_explicit_noisiness_not_a_related_correction_operation():
    claim='Note that without our correction step, the raw output of the network is still noisy.'
    source='We recover the clean estimate by doubling our network output and subtracting its input.'
    assert not claim_passage_concepts_covered(claim,source)
    assert not _supports(claim,source,'','unseen.pdf')
    assert _supports(claim,claim,'','unseen.pdf')


def test_clean_output_or_negated_noisiness_cannot_support_a_still_noisy_claim():
    claim='The raw output is still noisy.'
    for source in ('The raw output is not noisy.','The raw output is a clean image.','The raw output is not a clean image.'):
        assert not claim_passage_concepts_covered(claim,source)
    assert not claim_passage_concepts_covered('The raw output is not still noisy.',claim)


def test_not_required_is_not_evidence_of_missing_sampling_ability():
    claim = '该方法允许空间相关噪声，代价是需要能从噪声分布中采样，而 Noise2Void、Noise2Self 等方法不具备这一能力。'
    source = 'We also allow spatially correlated noise models, which are problematic for Noise2Void and Noise2Self. On the other hand, we require the ability to sample from the noise distribution, which the three aforementioned methods do not.'
    assert not _supports(claim, source, '', 'method')


def test_pronominal_inability_cannot_be_inferred_from_not_requiring_prior():
    from kb.explicit_noise_relations import sampling_ability_covered
    claim = '论文还把自己与相关工作区分开：它要求具备从噪声分布采样的能力，而所比较的三种方法不具备这一点。'
    evidence = 'Our method requires the ability to sample from the noise distribution, which the three aforementioned methods do not.'
    assert not sampling_ability_covered(claim, evidence)


def test_rejected_sampling_inability_is_removed_even_without_a_remaining_marker():
    from kb.claim_evidence_runtime import audit_and_repair_claim_evidence
    claim = '论文在相关工作对比中也强调，本方法“需要能够从噪声分布中采样”，而它对比的三种方法不具备这一点。'
    source = 'Our method requires the ability to sample from the noise distribution, which the three aforementioned methods do not.'
    answer, meta = audit_and_repair_claim_evidence(claim+' [1]', [{'text':source,'meta':{'source_name':'method'}}], allowed_citation_numbers={1}, drop_unsupported_high_risk_claims=True, enforce_user_visible_binding=True)
    assert '不具备这一点' not in answer
    assert meta['dropped_unsupported_unplanned_claims']
