from __future__ import annotations

import pytest

from api.claim_passage_plan import _supports
from kb.waived_data_requirements import waived_data_requirements_covered


@pytest.mark.parametrize('evidence', [
    'Paired noisy images are required, but clean images are not required.',
    'Paired noisy images are required and clean images are not required.',
    'Paired noisy images are required, clean images are not required.',
    'We need paired noisy images, whereas clean images are not needed.',
])
def test_a_clean_data_waiver_cannot_waive_a_positive_pair_requirement(evidence):
    assert not waived_data_requirements_covered('The method does not require paired noisy images.', evidence)
    assert not _supports('The method does not require paired noisy images.', evidence, '', 'paper.pdf')
    assert waived_data_requirements_covered('The method does not require clean images.', evidence)


def test_explicit_training_example_wording_preserves_both_waivers():
    claim = 'The method does not require clean images or paired noisy images.'
    source = 'We present a method without access to clean training examples or access to paired noisy training examples.'
    assert waived_data_requirements_covered(claim, source)
    assert not waived_data_requirements_covered(claim, 'The method does not require clean training examples, but paired noisy training examples are required.')
    assert not waived_data_requirements_covered(claim, 'Only a known noise distribution is required.')


def test_unnegated_object_lists_keep_both_categories():
    claim = 'Neither clean images nor paired noisy images are needed.'
    assert waived_data_requirements_covered(claim, 'No clean images and paired noisy images are needed.')
    assert not waived_data_requirements_covered(claim, 'Clean images are needed, but no paired noisy images are needed.')
    assert waived_data_requirements_covered(claim, 'Clean images are not needed, and paired noisy images are not needed.')


@pytest.mark.parametrize('source', [
    'The method does not require unpaired noisy images.',
    'The method does not require clean images and uses unpaired noisy training examples.',
    '方法不需要非配对的噪声图像。',
    '方法不需要干净图像，使用非配对的噪声图像训练。',
])
def test_unpaired_is_not_an_alias_for_paired(source):
    assert not waived_data_requirements_covered('The method does not require paired noisy images.', source)


def test_counterfactual_loss_prose_is_not_a_model_selection_proof():
    claim = '不用干净参照来选择去噪器，是因为噪声方差项与函数无关，不影响最优解。'
    loss = 'The self-supervised loss is the sum of the ordinary supervised loss and the variance of the noise.'
    choice = 'By minimizing the self-supervised loss over a class of J-invariant functions, one may find the optimal denoiser.'
    assert waived_data_requirements_covered(claim, loss + ' ' + choice)
    assert not waived_data_requirements_covered(claim, 'It is not true that ' + loss.lower() + ' ' + choice)
    assert not waived_data_requirements_covered(claim, loss + ' Not ' + choice.lower())
