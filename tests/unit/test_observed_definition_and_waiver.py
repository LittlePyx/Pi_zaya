from __future__ import annotations

import pytest

from api.claim_passage_plan import _supports
from kb.claim_source_eligibility import omitted_derivation_covered
from kb.explicit_training_assignments import explicit_training_assignments_covered
from kb.waived_data_requirements import waived_data_requirements_covered


@pytest.mark.parametrize('claim', [r'We observe $Y \\triangleq X+N$.', r'We observe $Y=X+N$.', 'The observation Y := X + N.'])
def test_observation_definition_requires_the_actual_additive_relation(claim):
    claim = claim.replace('\\\\', '\\')
    observation = 'We have an unobserved clean image X, and we observe a single noisy realization Y of that clean image.'
    assert not explicit_training_assignments_covered(claim, observation)
    assert not _supports(claim, observation, '', 'paper.pdf')
    assert explicit_training_assignments_covered(claim, 'We observe Y ≜ X + N.')
    for wrong in ('Y ≜ X - N.', 'Y ≜ X + M.', 'y ≜ X + N.'):
        assert not explicit_training_assignments_covered(claim, wrong)


@pytest.mark.parametrize('claim', [
    '方法不需要干净图像或配对噪声图像，但需要已知噪声分布。',
    'The method does not require clean images or paired noisy images, but needs a known noise model.',
])
def test_noise_model_knowledge_is_not_evidence_of_waived_data_requirements(claim):
    model = 'The noise distribution is known, so we can draw an additional synthetic noise sample.'
    assert not waived_data_requirements_covered(claim, model)
    assert not _supports(claim, model, '', 'paper.pdf')
    assert waived_data_requirements_covered(claim, 'The method does not require clean target images or noisy image pairs, but requires a known noise distribution.')
    assert not waived_data_requirements_covered(claim, 'The method does not require clean target images, but requires noisy image pairs.')


def test_waivers_keep_the_scope_of_but_and_clean_target_disposal():
    assert not waived_data_requirements_covered('No clean targets are needed.', 'Only a known noise model is needed.')
    assert not waived_data_requirements_covered('Clean targets are not needed.', 'Only a known noise model is needed.')
    assert waived_data_requirements_covered('No clean targets are needed.', 'We dispose of the clean targets entirely.')
    assert not waived_data_requirements_covered('We need no noisy image pairs.', 'We can dispense with clean targets but need noisy image pairs.')
    assert waived_data_requirements_covered('不需要干净图像，但需要配对噪声图像。', 'We dispose of the clean targets entirely, but require noisy image pairs.')
    assert not waived_data_requirements_covered('We need no clean targets.', 'We cannot do without clean targets.')


def test_missing_proof_cannot_be_inferred_from_applicability_or_scattered_words():
    claim = 'Noisier2Noise 未展示空间相关噪声下的完整推导。'
    assert not omitted_derivation_covered(claim, 'Our method is applicable to both pixel-wise i.i.d. white noise and spatially correlated noise.')
    assert not _supports(claim, 'Our method is applicable to both pixel-wise i.i.d. white noise and spatially correlated noise.', '', 'noisier2noise.pdf')
    for wrong in ('We do not omit the proof.', 'The proof is not omitted.', 'We do not provide training data. The derivation follows below.'):
        assert not omitted_derivation_covered(claim, wrong)
    assert omitted_derivation_covered(claim, 'We omit the full derivation for spatially correlated noise.')
    assert omitted_derivation_covered('The method permits spatially correlated noise.', 'Our method is applicable to spatially correlated noise.')


@pytest.mark.parametrize('claim', [
    r'Because the noise operator $\mathcal{A}$ is known, an additional synthetic noise sample $M$ can be drawn, giving $Z \triangleq Y+M=X+N+M$.',
    'The noise distribution is known and we can draw an additional synthetic noise sample.',
])
def test_draw_and_known_premises_cannot_borrow_an_assignment(claim):
    equation = 'a noisier version, Z ≜ Y + M = X + N + M.'
    assert not _supports(claim, equation, '', 'paper.pdf')
    assert _supports(claim, 'The noise operator A is known, so we can draw an additional synthetic noise sample M, giving Z ≜ Y + M = X + N + M.', '', 'paper.pdf')


def test_data_and_statistical_model_requirements_cannot_borrow_applicability():
    from kb.noise_model_prerequisites import noise_model_prerequisites_covered
    claim = '方法只需要每个训练样本的单个含噪实现加一个噪声分布的统计模型。'
    complete = 'Our method requires only a single noisy realization of each training example and a statistical model of the noise distribution.'
    assert noise_model_prerequisites_covered(claim, complete)
    # This helper is a veto, not a substitute for the existing cross-language binding gate.
    assert _supports(complete, complete, '', 'paper.pdf')
    for wrong in ('Our method is applicable to spatially correlated noise.', 'We augment training images with additional synthetic noise.', 'Only a statistical model of the noise distribution is needed.'):
        assert not noise_model_prerequisites_covered(claim, wrong)
        assert not _supports(claim, wrong, '', 'paper.pdf')


def test_a_standard_deviation_definition_is_not_the_loss_objective():
    claim = r'噪声标准差 $\sigma(i)$ 作为常数项写进等价损失里。'
    definition = r'where $\sigma(i)$ denotes the standard deviation of $n(i)$.'
    assert not _supports(claim, definition, '', 'paper.pdf')
    from kb.noise_model_prerequisites import objective_term_covered
    complete = definition + ' The equivalent loss includes the noise standard deviation as a constant term.'
    assert objective_term_covered(claim, complete)
    assert _supports('The noise standard deviation is a constant term in the equivalent loss.', complete, '', 'paper.pdf')


def test_feeding_sampled_inputs_is_not_the_same_as_using_pairs_for_training():
    claim = 'Each newly-formed network is fed a Bernoulli-sampled instance of y, and the recovered images are averaged.'
    training = 'The network is trained on pairs of Bernoulli-sampled instances and the predictions are averaged.'
    testing = 'Recovered images are generated by feeding a Bernoulli sampled instance of y to each newly-formed network. The recovered images are averaged.'
    assert not _supports(claim, training, '', 'paper.pdf')
    assert _supports(claim, testing, '', 'paper.pdf')


@pytest.mark.parametrize('claim', [
    'The selected excerpts do not contain a head-to-head comparison of the two methods at all.',
    'This is a boundary of the retrieved snippets, not proof that neither paper contains such a comparison.',
])
def test_window_absence_is_not_a_claim_a_comparison_paragraph_can_prove(claim):
    from kb.claim_source_eligibility import retrieval_limit_claim
    assert retrieval_limit_claim(claim)
    assert not _supports(claim, 'Several representative single-image denoising methods are selected for comparison.', '', 'paper.pdf')
