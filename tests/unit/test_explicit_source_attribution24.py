from __future__ import annotations

import pytest

from api.claim_passage_plan import _supports
from kb.explicit_source_attribution import source_attribution_covered
from kb.source_question_facets import source_facet_quote, source_facet_score


@pytest.mark.parametrize('claim,quote,expected',[
    ('Figure 2 states this explicitly: the singly-noisy image is the target.', 'We train the network to predict Y given Z as input.',False),
    ('Figure 2 states this explicitly: the singly-noisy image is the target.', 'Figure 2. The singly-noisy image is our target.',True),
    ('Figure 2 shows the target.', 'Figure 3. The singly-noisy image is our target.',False),
    ('图2明确显示目标。', 'Figure 2. The singly-noisy image is our target.',True),
    ('See Figure 2 for the training target.', 'We train the network to predict Y given Z.',True),
    ('Recent works train on the set of external noisy images only.', 'Our method only uses the input noisy image itself for training.',False),
    ('Recent works train on the set of external noisy images only.', 'Recent works train on the set of external noisy images only.',True),
    ('Training requires an external dataset.', 'Our method does not require an external dataset.',False),
    ('Training does not require an external dataset.', 'Our method only uses the input noisy image itself for training.',True),
    ("Self2Self's reported comparisons are against baselines (e.g., BM3D, TWSC, CDnCNN, DIP).", 'We compare to KSVD, BM3D and DIP.',False),
    ("Self2Self's reported comparisons are against baselines (e.g., BM3D, TWSC, CDnCNN, DIP).", 'We compare to BM3D, TWSC, CDnCNN and DIP.',True),
    ('Noise2Void is compared to the N2N baseline.', 'Noise2Void is compared to Noise2Noise.',True),
])
def test_named_attribution_is_a_body_obligation(claim,quote,expected):
    assert source_attribution_covered(claim,quote,'self2self.pdf') is expected


def test_metadata_cannot_authorize_a_missing_baseline_or_named_figure():
    proof={'citation_plan_evidence_selection_reason':'prompt_contract_block','primary_block_id':'a','page_start':3}
    assert not _supports('Figure 2 states that Y is the target.', 'We train the network to predict Y given Z as input.', '', 'noisier2noise',binding_meta=proof)
    assert not _supports('Self2Self compares against BM3D, TWSC and DIP baselines.', 'We compare against BM3D and DIP baselines.', '', 'self2self',binding_meta=proof)


def test_caption_mapping_is_retrieved_whole_with_its_own_identity():
    prompt='In the standard additive procedure, what is the input and training target?'
    quote='Figure 2. The steps of our method. Top Row: a singly-noisy realization (which is our training target), a doubly-noisy realization (which is the input to our network).'
    assert source_facet_score(prompt,quote)==18
    assert source_facet_quote(prompt,quote)==quote
    assert source_facet_score(prompt,quote.replace('training target','test image'))==0


def test_partition_tradeoff_is_not_an_abstract_independence_claim():
    claim='划分 J 的结构必须反映信号依赖与噪声独立，且子集大小带来偏差—方差权衡。'
    assert not source_attribution_covered(claim,'We use independent groups of features to predict one another.','noise2self')
    assert source_attribution_covered(claim,'The structure of the partition J has a bias-variance tradeoff.','noise2self')


def test_named_noise_samples_and_half_sum_require_their_own_assertions():
    from kb.explicit_noise_relations import iid_relation_covered

    claim='Noisier2Noise：独立性作用于原始噪声 N 与额外合成噪声 M 两个样本，用于和的一半估计。'
    assert not iid_relation_covered(claim,'Given two i.i.d. noise samples, the best estimate of one is half their sum.')
    assert not iid_relation_covered(claim,'Recall that M and N are i.i.d. and have the same expectation given Z.')
    assert iid_relation_covered(claim,'M and N are i.i.d. noise samples. Each has expectation half of their sum.')


def test_performance_boundary_uses_comparison_evidence_and_signed_copy():
    from api.method_relation_copy import method_relation_copy
    from api.reference_source_projection import reference_question_facet_priority
    from api.source_relation_copy import source_relation_copy

    boundary='While it cannot be expected that our approach outperforms methods that have additional information available during training, we observe that the denoising performance of our results only drops moderately.'
    data='This method does not require noisy image pairs, nor clean target im- ages.'
    claim='取消干净目标或独立噪声对扩大了可训练数据的范围，并不意味着去噪质量一定更高。'
    assert not source_attribution_covered(claim,data,'noise2void')
    assert source_attribution_covered(claim,boundary,'noise2void')
    copy=source_relation_copy(boundary)
    assert copy and '不预期' in copy[0] and '额外信息' in copy[1]
    assert not method_relation_copy('It is not true that '+boundary)
    assert source_relation_copy(data)
    prompt='本文是否声称这种训练方式应当胜过拥有干净目标或独立噪声对的训练？作者怎样解释这个性能边界？'
    assert reference_question_facet_priority(prompt,boundary)==1
    assert reference_question_facet_priority(prompt,data)==0
