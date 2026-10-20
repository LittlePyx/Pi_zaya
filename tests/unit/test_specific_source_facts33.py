from __future__ import annotations

import pytest

from kb.explicit_source_attribution import source_attribution_covered
from kb.explicit_training_assignments import explicit_training_assignments_covered
from kb.explicit_noise_relations import named_synthetic_sample_covered
from kb.source_question_facets import sample_relation_priority


PROCEDURE = 'Our training procedure is as follows. Given Y, we create Z := Y + M = X + N + M. We then train the network to predict Y given Z as input, using a pixel-wise L2 loss.'
PRIOR = 'We consider image denoising under a fixed, known noise model. We capture a single noisy realization Y := X + N. Because the distribution is known, we can draw an additional synthetic noise sample M.'
ABSTRACT = 'This paper proposes a self-supervised method which only uses the input noisy image itself for training. The network is trained with dropout on pairs of Bernoulli-sampled instances of the input image.'
MASK = 'The loss of each pair is measured only on those pixels that are masked by the Bernoulli vector. The summation of the loss over all pairs measures the difference over all image pixels.'
SPATIAL = 'Our method is applicable to both pixel-wise i.i.d. white noise and spatially correlated noise.'


@pytest.mark.parametrize('claim,wrong,right',[
    ('The relation to ground truth is formalized in Proposition 1.',MASK,'Proposition 1. The expected loss equals the clean-image loss plus a variance term.'),
    ('命题1指出噪声分量独立且零均值。',MASK,'Proposition 1. Assume the noise components are independent and of zero mean.'),
    ('Figure 2 likewise labels the singly-noisy realization as the training target.',PROCEDURE,'Figure 2. A singly-noisy realization is our training target.'),
    ('图2标注原始干净图像为不可见。',PRIOR,'Figure 2. Original clean image (unseen by our method).'),
    ('Self2Self needs no noise-model knowledge.',ABSTRACT,'Self2Self does not require knowledge of the noise model.'),
    ('Self2Self 不需要噪声模型先验。',ABSTRACT,'Self2Self 不需要噪声模型先验。'),
    ('The clean estimate is handled separately by a correction step.',PRIOR,'We recover the clean estimate by doubling the network output and subtracting its input.'),
    ('The motivation is explicitly to remove the "prerequisite on a large amount of training data relevant to the target images".',ABSTRACT,'We remove the prerequisite on a large amount of training data relevant to the target images.'),
    ('The network uses partial convolutions to renormalize the sampled pixels.',ABSTRACT,'The network uses partial convolutions to renormalize the sampled pixels.'),
    ('Model uncertainty gives the predictions statistical independence.',ABSTRACT,'Model uncertainty gives the predictions statistical independence.'),
    ('Residual learning and batch normalization speed up training.', 'The architecture incorporates batch normalization to improve denoising performance.', 'Residual learning and batch normalization speed up training.'),
    ('Self2Self constructs Bernoulli instances, requiring no noise-model knowledge.',ABSTRACT,'Self2Self does not require knowledge of the noise model.'),
])
def test_explicit_paper_facts_do_not_borrow_a_topically_related_excerpt(claim,wrong,right):
    assert not source_attribution_covered(claim,wrong,'self2self.pdf')
    assert source_attribution_covered(claim,right,'self2self.pdf')


def test_model_requirement_cannot_borrow_a_prior_work_negation():
    claim='Self2Self needs no noise-model knowledge.'
    quote='Other methods do not require knowledge of the noise model. Self2Self uses a known noise model.'
    assert not source_attribution_covered(claim,quote,'self2self.pdf')


def test_noise_model_requirement_is_not_sampling_inability():
    assert source_attribution_covered('Noisier2Noise needs a known noise model.',PRIOR,'noisier2noise.pdf')
    assert source_attribution_covered('The snippets do not establish a universal ranking.',SPATIAL,'paper.pdf')


def test_named_training_map_uses_actual_input_and_target_roles():
    claim=r'The network learns the mapping $Z \mapsto Y$.'
    assert not explicit_training_assignments_covered(claim,PRIOR)
    assert explicit_training_assignments_covered(claim,PROCEDURE)
    assert not explicit_training_assignments_covered(r'The network learns the mapping $Y \mapsto Z$.',PROCEDURE)


def test_named_synthetic_sample_cannot_borrow_a_spatial_noise_bullet():
    claim='把 M 视为来自同一噪声分布的合成样本。'
    assert not named_synthetic_sample_covered(claim,SPATIAL)
    assert named_synthetic_sample_covered(claim,PRIOR)
    assert not named_synthetic_sample_covered('N is the synthetic noise sample.',PRIOR)


def test_actual_method_sample_relation_precedes_the_unnamed_toy_insight():
    prompt='Noisier2Noise 可处理空间相关噪声，额外生成噪声与原始噪声的独立关系是什么？'
    toy='Given the sum of two i.i.d. noise samples, the best estimate of one is half their sum.'
    actual='Recall that M and N are i.i.d. and therefore have the same expectation given Z.'
    assert sample_relation_priority(prompt,actual)>sample_relation_priority(prompt,toy)
    assert sample_relation_priority(prompt,actual.replace('are i.i.d.','are not i.i.d.'))<sample_relation_priority(prompt,actual)
    assert sample_relation_priority(prompt,actual.replace('M and N','M and X'))<=sample_relation_priority(prompt,toy)
