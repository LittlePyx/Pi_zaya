from __future__ import annotations

import pytest

from api.claim_passage_plan import _supports
from kb.explicit_training_assignments import explicit_training_assignments_covered

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
