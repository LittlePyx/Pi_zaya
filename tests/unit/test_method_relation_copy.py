from __future__ import annotations

import pytest

from api.source_relation_copy import source_relation_copy


PHASE = 'This paper shows that using dropout in both training and test is very effective on boosting the performance when training a denoising NN on only an input noisy image. The main reason is that it can effectively reduce the variance of the prediction.'
CORRELATION = 'We also allow spatially correlated noise models, which are problematic for other methods. On the other hand, we require the ability to sample from the noise distribution, which the other methods do not.'
CORRECTION = "We can therefore recover an estimate of the clean image by doubling our network's output and subtracting its input."


@pytest.mark.parametrize('evidence',[PHASE,CORRELATION,CORRECTION])
def test_complete_method_relations_get_specific_source_backed_copy(evidence):
    copy=source_relation_copy(evidence)
    assert copy and len(copy[0])>=15 and len(copy[1])>=25
    assert '置于同一段' not in ''.join(copy)


@pytest.mark.parametrize('evidence',[
    PHASE.replace('both training and test','training'),
    PHASE.replace('reduce the variance','increase the variance'),
    PHASE.replace('This paper shows that','It is false that'),
    CORRELATION.replace('We also allow','We do not allow'),
    CORRELATION.replace('we require the ability to sample from the noise distribution','we do not require the ability to sample from the noise distribution'),
    CORRECTION.replace('doubling our network\'s output and subtracting its input','doubling its input and subtracting the output'),
    'Dropout, ensemble, Self2Self, variance. A mechanism section names these terms.',
])
def test_incomplete_negated_or_reversed_relations_do_not_get_positive_copy(evidence):
    assert source_relation_copy(evidence) is None


def test_explicit_noise_assumptions_get_premise_copy_without_inferring_a_converse():
    copy=source_relation_copy('Theorem 9. Assume the noise components are independent and of zero mean. The loss has a fixed variance term.')
    assert copy and '前提' in copy[0] and '不能作为' in copy[1]


@pytest.mark.parametrize('evidence',[
    'We do not assume the noise components are independent and of zero mean.',
    'Assume the noise components are independent but may have nonzero mean.',
    'Assume the noise components have zero mean but are spatially correlated.',
])
def test_partial_or_negated_assumptions_cannot_get_the_complete_premise_copy(evidence):
    assert source_relation_copy(evidence) is None


def test_original_and_synthetic_noise_copy_requires_the_same_distribution_clause():
    evidence='In our construction X + N is the original noisy image, and M is a synthetic sample from the same noise distribution.'
    copy=source_relation_copy(evidence)
    assert copy and '同一噪声分布' in copy[0]
    assert source_relation_copy(evidence.replace('same noise distribution','different noise distribution')) is None
    assert source_relation_copy('It is false that '+evidence) is None
