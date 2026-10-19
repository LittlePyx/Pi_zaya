from __future__ import annotations

from api.claim_passage_plan import _supports
from kb.explicit_noise_relations import known_noise_model_covered
from kb.source_question_facets import source_facet_quote, source_facet_score

CLAIM = '该方法需要已知的噪声分布，并且能从该分布中采样。'
SAMPLING = 'On the other hand, we require the ability to sample from the noise distribution.'
DEFINITION = 'We consider a scenario in which we wish to train a neural network under a fixed, known noise model. N is drawn from a known noise distribution. Because it is known, we can draw an additional synthetic noise sample M.'


def test_sampling_alone_cannot_supply_explicit_model_knowledge():
    assert not known_noise_model_covered(CLAIM, SAMPLING)
    assert not _supports(CLAIM, SAMPLING, 'Known noise model', 'paper.pdf')
    assert known_noise_model_covered(CLAIM, DEFINITION)
    assert _supports(CLAIM, DEFINITION, '', 'paper.pdf')
    assert not known_noise_model_covered(CLAIM, 'We do not require a known noise model.')
    assert not known_noise_model_covered(CLAIM, 'A known noise distribution is not required.')
    assert not known_noise_model_covered(CLAIM, 'We use an unknown noise distribution.')
    assert known_noise_model_covered('The method does not require a known noise model.', 'We denoise without a model.')


def test_noise_model_question_keeps_the_continuous_known_sampling_premise():
    prompt = '是否不需要噪声模型？请说明需要的观测和额外先验。'
    assert source_facet_score(prompt, DEFINITION) > source_facet_score(prompt, SAMPLING)
    assert source_facet_quote(prompt, DEFINITION) == DEFINITION
    assert not source_facet_score('Which pixels does the loss measure?', DEFINITION)


def test_inserted_sampling_clause_still_requires_the_full_known_distribution_premise():
    claim = 'Noisier2Noise 需要一个已知、且能从中采样的噪声分布。'
    assert _supports(claim, DEFINITION, '', 'noisier2noise.en.md')
    assert not _supports(claim, 'N is drawn from a known noise distribution.', '', 'noisier2noise.en.md')
    assert not _supports(claim, SAMPLING, '', 'noisier2noise.en.md')
