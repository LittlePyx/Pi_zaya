from __future__ import annotations

import pytest

from api.claim_passage_plan import _supports
from kb.explicit_source_attribution import source_attribution_covered
from kb.source_question_facets import source_facet_quote, source_facet_score
from kb.explicit_noise_relations import named_synthetic_sample_covered, signal_dependency_covered


ASSUMPTION = 'If we extract a patch as input and use its center pixel as target, our network would just learn the identity (see Figure 2a). Let us assume a network architecture with a special receptive field.'
BLIND = '**Figure 2.** (a) A conventional network learns the identity. (b) In our proposed blind-spot network the receptive field of each pixel excludes the pixel itself, preventing it from learning the identity. We train it using the same noisy images as input and target.'
ABSTRACT = 'Our DnCNN uses residual learning and batch normalization to speed up training. It implicitly removes the latent clean image in the hidden layers.'
RECIPE = 'The input of our DnCNN is a noisy observation y = x + v. For DnCNN, we train a residual mapping R(y) ≈ v and recover x = y − R(y). The loss compares R(yi) to yi − xi. Here (yi, xi) are noisy-clean training image pairs.'


@pytest.mark.parametrize('claim',[
    'The receptive field excludes the pixel itself, so the center value cannot be copied to the output.',
    'The figure caption summarizes the consequence: "the receptive field of each pixel excludes the pixel itself, preventing it from learning the identity".',
    'Figure 2(b) states that the receptive field excludes the pixel itself.',
])
def test_exclusion_and_subfigure_do_not_borrow_an_architecture_assumption(claim):
    assert not source_attribution_covered(claim, ASSUMPTION, 'noise2void.pdf')
    assert source_attribution_covered(claim, BLIND, 'noise2void.pdf')
    assert not source_attribution_covered('The receptive field excludes the pixel itself.', 'The receptive field does not exclude the pixel itself.', 'paper.pdf')


def test_training_supervision_is_not_established_by_residual_learning_in_an_abstract():
    for claim in ('DnCNN predicts a residual, supervised against clean targets.', 'DnCNN training supervision comes from clean ground-truth data.'):
        assert not source_attribution_covered(claim, ABSTRACT, 'dncnn.pdf')
        assert source_attribution_covered(claim, RECIPE, 'dncnn.pdf')
    prior = 'Traditional methods are trained on pairs of noisy inputs and clean targets.'
    assert source_attribution_covered(prior, prior, 'review.pdf')
    assert not source_attribution_covered('Our network is supervised against clean targets.', 'Our method does not require noisy-clean training image pairs.', 'paper.pdf')


def test_second_noisy_observation_is_a_training_relation_not_a_blur_boundary():
    claim = 'The network is trained on a noisy input patch and a noisy target taken from a second noisy observation at the same position.'
    wrong = 'We restrict the task to denoising because multiple noisy observations can retrieve the true signal. This is not the case for blur.'
    right = 'The network is trained using independent pairs of noisy images as input and target.'
    assert not source_attribution_covered(claim, wrong, 'noise2void.pdf')
    assert source_attribution_covered(claim, right, 'noise2noise.pdf')


def test_a_loss_difference_does_not_imply_independence_from_the_clean_reference():
    wrong = 'By minimizing the self-supervised loss over a class of J-invariant functions, one may find the optimal denoiser.'
    claim = '由于这一差值不依赖于干净参照 $y$，最小化自监督损失即可选出最优去噪器。'
    assert not source_attribution_covered(claim, wrong, 'noise2self.pdf')
    identity = 'The self-supervised loss is the sum of the ordinary supervised loss and the variance of the noise. ' + wrong
    assert source_attribution_covered('The noise variance term does not depend on the denoising function f.', identity, 'noise2self.pdf')


def test_requested_comparison_retrieves_the_own_recipe_and_exclusion_statement():
    prompt = 'Compare the predicted quantity and training supervision of the two methods. Which predicts a residual, and which excludes the target pixel from its context?'
    assert source_facet_score(prompt, RECIPE) > source_facet_score(prompt, ABSTRACT)
    assert source_facet_score(prompt, BLIND) > source_facet_score(prompt, ASSUMPTION)
    assert source_facet_quote(prompt, BLIND) == BLIND
    assert source_facet_score('How does the blind-spot prevent the identity predictor?', BLIND) > source_facet_score('How does the blind-spot prevent the identity predictor?', ASSUMPTION)


def test_sample_draw_is_not_construction_of_the_training_target():
    claim = '正因为分布已知，才能额外采样得到合成噪声样本 M 来构造训练目标。'
    prior = 'Because the distribution is known, we can draw an additional synthetic noise sample M.'
    assert not source_attribution_covered(claim, prior, 'noisier2noise.pdf')
    assert source_attribution_covered('Synthetic noise is used to construct training targets.', 'We corrupt the training targets with zero-mean noise without changing what the network learns.', 'noise2noise.pdf')


def test_named_generated_sample_is_not_a_spatial_applicability_bullet():
    claim = '从同一噪声分布中采样一个合成噪声 M。'
    assert not named_synthetic_sample_covered(claim, 'Our method handles white and spatially correlated noise.')
    assert named_synthetic_sample_covered(claim, 'M is a synthetic noise sample from the known distribution.')
    assert named_synthetic_sample_covered('原始噪声 N 与合成噪声 M 独立同分布。', 'Recall that M and N are i.i.d.')


def test_signal_correlation_qualifier_does_not_borrow_a_noise_only_premise():
    claim = '噪声在给定信号下独立，而真实信号存在某种相关性。'
    assert not signal_dependency_covered(claim, 'We use features whose noise is independent conditional on the true signal.')
    assert signal_dependency_covered(claim, 'The signal exhibits some correlation, while noise is independent.')


def test_late_own_noise_premise_survives_long_prior_work_context():
    prompt = 'Noise2Void 和 Noise2Self 是否处理任意相关噪声？分别指出独立性要求。'
    prior = 'Prior work uses pairs of noisy images. ' * 70
    own = 'In this work we make two simple statistical assumptions: the signal is not pixel-wise independent, and the noise is conditionally pixel-wise independent given the signal.'
    method = 'Our blind-spot network cannot learn the identity. Since we assume the noise to be pixel-wise independent given the signal, neighbors carry no information about its noise. The signal however is assumed to contain statistical dependencies.'
    assert source_facet_score(prompt, own) > source_facet_score(prompt, prior)
    assert source_facet_quote(prompt, prior + own) == own
    assert source_facet_score(prompt, method) > 0
    assert source_facet_quote(prompt, method) == method
    assert source_facet_score(prompt, method.replace('we assume the noise', 'we do not assume the noise')) == 0
