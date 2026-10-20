from __future__ import annotations

import pytest

from api.claim_passage_plan import _supports
from kb.explicit_training_assignments import explicit_training_assignments_covered, named_equation_covered

CLAIM = r'In Equation (6), the input is $\widehat{\boldsymbol{y}}_m:= \boldsymbol{b}_m \odot \boldsymbol{y}$ and the target is $\overline{\boldsymbol{y}}_m:= (\mathbf{1} - \boldsymbol{b}_m) \odot \boldsymbol{y}$.'
MASK = r'where $\|\cdot\|_{\boldsymbol{b}}^2 = \|(1 - \boldsymbol{b}) \odot \cdot\|_2^2$. It can be seen that the loss of each pair is measured only on those pixels that are masked by $\boldsymbol{b}_m$. As the masked pixels are randomly selected using a Bernoulli process, the summation of the loss over all pairs measures the difference over all image pixels.'
DEFINITIONS = r'where $\odot$ denotes the element-wise multiplication and $\boldsymbol{b}$ denotes one instance of binary Bernoulli vector. Then, a set of image pairs is defined as: for each $m$, $$ \widehat{\boldsymbol{y}}_m := \boldsymbol{b}_m \odot \boldsymbol{y}; \quad \overline{\boldsymbol{y}}_m := (\mathbf{1} - \boldsymbol{b}_m) \odot \boldsymbol{y}. \tag{6} $$'


def test_loss_mask_is_not_direct_evidence_for_explicit_training_pair_definitions():
    assert not _supports(CLAIM, MASK, '3.2. Training', 'self2self.en.md')
    assert not explicit_training_assignments_covered(CLAIM, MASK)
    assert _supports(CLAIM, DEFINITIONS, '3.2. Training', 'self2self.en.md')
    assert explicit_training_assignments_covered(CLAIM, DEFINITIONS)


@pytest.mark.parametrize('evidence', [DEFINITIONS.replace('\\widehat', '\\overline'), DEFINITIONS.replace('_m', '_n'), DEFINITIONS.replace('\\odot', '+'), DEFINITIONS.split(';')[0], MASK + ' Equation (6) defines the input and target.'])
def test_titles_numbers_and_wrong_definitions_cannot_substitute(evidence):
    assert not explicit_training_assignments_covered(CLAIM, evidence)


def test_visible_and_held_out_split_requires_the_named_equation_and_complementary_masks():
    claim = r'In Equation (6), the Bernoulli vector $\boldsymbol{b}_m$ splits the noisy image $\boldsymbol{y}$ into a visible part and a held-out part.'
    assert explicit_training_assignments_covered(claim, DEFINITIONS)
    assert _supports(claim, DEFINITIONS, '', 'paper.pdf')
    for invalid in (MASK, DEFINITIONS.replace('tag{6}', 'tag{7}'), DEFINITIONS.replace('(\\mathbf{1} - \\boldsymbol{b}_m)', '\\boldsymbol{b}_m'), DEFINITIONS.replace('_m', '_n')):
        assert not explicit_training_assignments_covered(claim, invalid)
        assert not _supports(claim, invalid, '', 'paper.pdf')


def test_training_dropout_alone_does_not_supply_prediction_averaging():
    claim = 'The trained model generates predictions with dropout and averages those predictions as the final result.'
    training = 'This paper uses dropout during both training and test to reduce prediction variance.'
    averaging = 'Dropout provides a computationally efficient way to train multiple models. The predictions from these models are averaged as the final result.'
    assert not _supports(claim, training, '', 'paper.pdf')
    assert _supports(claim, averaging, '', 'paper.pdf')


def test_training_loss_equation_cannot_point_to_an_unrelated_training_paragraph():
    claim = 'The network is trained on the loss in Equation (7).'
    equation = r'Train the network by minimizing the following loss: $$\min_{\theta} L(f_{\theta}(y), t). \tag{7}$$'
    assert named_equation_covered(claim, equation)
    assert not _supports(claim, 'Training the network with pairs of masked instances avoids an identity mapping.', 'Equation (7)', 'paper.pdf')
    assert not _supports(claim, equation.replace('tag{7}', 'tag{8}'), 'Equation (7)', 'paper.pdf')


def test_explicit_loss_mask_factor_cannot_borrow_a_bernoulli_identity_statement():
    claim = r'Because of the $(1-\boldsymbol{b})$ factor, the loss is measured only on the masked pixels.'
    identity = 'Bernoulli sampling avoids convergence of the network to an identity mapping. Training uses pairs of Bernoulli sampled instances.'
    assert _supports(claim, MASK, '', 'paper.pdf')
    assert not _supports(claim, identity, '', 'paper.pdf')


def test_input_definition_cannot_borrow_the_observation_paragraph():
    claim = r'The input Z is a noisier version of the observation: $Z \triangleq Y + M = X + N + M$.'
    observation = 'We have an unobserved clean image X, and we observe a single noisy realization Y of that clean image.'
    definition = 'A noisier version, Z ≜ Y + M = X + N + M.'
    assert explicit_training_assignments_covered(claim, definition)
    assert not explicit_training_assignments_covered(claim, observation)
    assert not _supports(claim, observation, '', 'paper.pdf')
    assert not explicit_training_assignments_covered(claim, definition.replace('+ M', '- M'))
