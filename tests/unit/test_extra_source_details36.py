from __future__ import annotations

import pytest

from kb.explicit_source_attribution import source_attribution_covered


def test_each_loss_statistic_needs_its_own_relation():
    mean = 'With the L2 loss the network learns the average of all plausible explanations.'
    median = 'The L1 loss recovers the median of the targets.'
    combined = 'The L2 loss is minimized by the mean and the L1 loss by the median.'
    assert not source_attribution_covered(combined, median, 'paper.pdf')
    assert not source_attribution_covered(combined, mean, 'paper.pdf')
    assert source_attribution_covered(combined, mean + median, 'paper.pdf')
    assert not source_attribution_covered('The necessary condition is mean for L2, median for L1.', 'Targets have the same conditional expected values.', 'paper.pdf')


def test_expansion_per_layer_cannot_conflict_with_the_cited_depth_formula():
    source = 'The receptive field at depth d is (2d+1)×(2d+1).'
    claim = '每增加一层，感受野每边扩大 1 个像素。'
    assert not source_attribution_covered(claim, source, 'paper.pdf')
    assert source_attribution_covered(claim.replace('1', '2'), source, 'paper.pdf')


def test_mask_training_recipe_cannot_borrow_a_partial_caption():
    claim = '训练时随机选一个像素，把其强度复制到该位置形成盲点，用修改后的图作输入、原图作目标，损失只在被掩蔽的盲点像素上计算。'
    assert not source_attribution_covered(claim, 'Blind-spot masking scheme used during training. (a) A noisy training image.', 'paper.pdf')
    actual = 'A center pixel is replaced by a value copied from a neighboring pixel. The loss is measured only on the masked pixels.'
    assert source_attribution_covered('A center pixel is replaced by a value copied from a neighboring pixel.', actual, 'paper.pdf')
    assert source_attribution_covered('The loss is measured only on the masked pixels.', actual, 'paper.pdf')


@pytest.mark.parametrize('claim', [
    'The blind spot removes the one input pixel that would let the network copy the answer.',
    'The receptive field excludes the center pixel.',
    'Because the center value is never visible to the network, it cannot copy that value.',
])
def test_clipped_exclusion_sentence_is_not_a_complete_exclusion_statement(claim):
    clipped = 'We assume the receptive field to have a blind-spot in its center. The CNN prediction is affected by all input pixels in a square neighborhood except for the Now let us consider N2N.'
    full = 'In our blind-spot network, the receptive field of each pixel excludes the pixel itself, preventing it from learning the identity.'
    assert not source_attribution_covered(claim, clipped, 'paper.pdf')
    assert source_attribution_covered(claim, full, 'paper.pdf')


def test_a_blur_boundary_is_not_the_expectation_equality():
    claim = 'The scheme works because the expected value of the noisy input equals the clean signal.'
    boundary = 'We restrict the task to denoising: multiple noisy observations can retrieve the true signal, unlike blur.'
    assert not source_attribution_covered(claim, boundary, 'paper.pdf')
    assert source_attribution_covered(claim, 'The expectation of the corrupted input data is the clean target that we seek to restore.', 'paper.pdf')
