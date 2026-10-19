from __future__ import annotations

from api.claim_passage_plan import _supports
from kb.explicit_noise_relations import iid_relation_covered

COMPONENT = 'Assume the noise components are independent and of zero mean.'
SAMPLES = 'Recall that M and N are i.i.d.'
APPLICABILITY = 'Our method is applicable to both pixel-wise i.i.d. white noise and spatially correlated noise.'


def test_component_independence_and_between_sample_independence_are_distinct():
    components = 'The noise components are independent.'
    samples = '额外合成噪声样本与原始噪声之间相互独立。'
    assert iid_relation_covered(components, COMPONENT)
    assert not iid_relation_covered(components, SAMPLES)
    assert iid_relation_covered(samples, SAMPLES)
    assert not iid_relation_covered(samples, COMPONENT)
    assert not iid_relation_covered(samples, APPLICABILITY)
    assert not iid_relation_covered('给定两个 i.i.d. 噪声样本之和，对其中一个的最佳估计是二者之和的一半。', APPLICABILITY)
    assert not iid_relation_covered(samples, 'M is a synthetic sample from the same noise distribution.')
    assert not iid_relation_covered(samples, 'M and N are not independent.')
    relation = 'Noisier2Noise 一侧：约束的是额外生成噪声与原始噪声的关系。'
    assert not iid_relation_covered(relation, APPLICABILITY)
    assert iid_relation_covered(relation, 'X + N is the original noisy image, and M is a synthetic sample from the same noise distribution.')
    assert not iid_relation_covered('The relationship between additional synthetic noise and original noise.', APPLICABILITY)
    assert iid_relation_covered('向原始噪声图像叠加合成噪声来训练网络。', 'We create a noisier version for training.')


def test_cross_paper_independence_comparison_cannot_borrow_noise_applicability():
    comparison = '两篇论文的独立性假设作用在不同对象上：Self2Self 约束噪声分量之间的关系，Noisier2Noise 依赖额外合成噪声样本与原始噪声之间的关系。'
    inference = '前者要求分量独立，后者要求两个噪声样本独立，二者可以同时成立。'
    for claim in (comparison, inference):
        assert not iid_relation_covered(claim, APPLICABILITY)
        assert not _supports(claim, APPLICABILITY, 'Independent noise samples', 'noisier2noise.en.md')
    assert not iid_relation_covered(inference, COMPONENT + ' M and N are independent noise samples.')
    assert iid_relation_covered(inference, COMPONENT + ' M and N are independent noise samples. These conditions can both hold simultaneously.')


def test_signal_negation_does_not_negate_the_separate_noise_premise():
    assert iid_relation_covered('The noise components are independent given the signal.', 'The signal is not pixel-wise independent, while noise is pixel-wise independent given the signal.')
    assert not iid_relation_covered('The noise components are independent.', 'The noise components are not independent.')
    directive = '再对照方法节中 M、N、Z 的定义，确认 i.i.d. 指的是哪两个量。'
    assert not iid_relation_covered(directive, APPLICABILITY)
    assert iid_relation_covered(directive, SAMPLES + ' They have the same expectation given Z.')
