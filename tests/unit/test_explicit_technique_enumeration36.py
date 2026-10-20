from __future__ import annotations

import pytest

from kb.evidence_binding import assess_system_a_hit_binding
from kb.explicit_technique_enumeration import technique_pair_quantities

SOURCE = 'It turns out that residual learning and batch normalization can benefit from each other, and their integration is effective in speeding up the training and boosting the denoising performance.'
CLAIM = 'The paper reports that these two techniques benefit from each other, and that their integration speeds up training while boosting denoising performance.'


def test_source_names_both_members_of_the_claimed_pair():
    # The final 'and' joins the consequence, not another enumeration member.
    assert technique_pair_quantities(CLAIM, SOURCE) == {('2', '', '')}
    binding = assess_system_a_hit_binding(answer_claim=CLAIM, hit={'text': SOURCE}, meta={}, evidence_quote=SOURCE, heading='', source_name='paper.pdf')
    assert not binding['suppress_link']


@pytest.mark.parametrize('source', [
    SOURCE.replace('can benefit', 'cannot benefit'),
    SOURCE.replace('residual learning and ', ''),
    SOURCE.replace('residual learning and batch normalization', 'residual learning and masking and batch normalization'),
    'Two images are used for training.',
    'The model uses batch normalization and has two hidden layers.',
    SOURCE.replace('residual learning', 'batch normalization'),
    'It is false that ' + SOURCE,
])
def test_numbers_and_incomplete_or_longer_lists_do_not_establish_the_pair(source):
    assert not technique_pair_quantities(CLAIM, source)


@pytest.mark.parametrize('claim', [
    CLAIM.replace('two', 'three'), CLAIM.replace('techniques', 'images'),
    'Two detectors accelerate training.', 'The network has two layers.',
])
def test_other_counts_are_not_covered_by_a_technique_pair(claim):
    assert not technique_pair_quantities(claim, SOURCE)
