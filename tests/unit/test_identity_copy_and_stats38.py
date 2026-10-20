from __future__ import annotations

from kb.claim_evidence_runtime import _is_high_risk_claim
from kb.explicit_source_attribution import source_attribution_covered
from kb.explicit_source_details import explicit_source_details_covered


def test_blocked_copy_explanation_is_not_a_pixel_replacement_recipe():
    quote = 'In our blind-spot network, the receptive field of each pixel excludes the pixel itself, preventing it from learning the identity.'
    claim = 'The receptive field excludes the center pixel, so the network cannot copy the target pixel from its input.'
    assert source_attribution_covered(claim, quote, 'paper.pdf')
    assert source_attribution_covered('盲点使中心像素值不能被复制，因此网络不能学会恒等映射。', quote, 'paper.pdf')
    wrong = '训练时随机选一个像素，把其强度复制到该位置形成盲点，损失只在被掩蔽的像素上计算。'
    assert not source_attribution_covered(wrong, 'Blind-spot masking scheme used during training. (a) A noisy training image.', 'paper.pdf')
    assert not explicit_source_details_covered('无需干净目标，把像素值复制到该位置形成盲点。', 'A blind-spot masking scheme is used during training.')


def test_a_negated_pixel_block_is_not_exclusion_evidence():
    assert not source_attribution_covered('The receptive field excludes the center pixel.', 'The center pixel is not blocked.', 'paper.pdf')


def test_literal_iid_statement_is_a_statistical_fact_even_without_a_spelled_out_alias():
    assert _is_high_risk_claim('The noise samples M and N are i.i.d.')
    assert not _is_high_risk_claim('Read the i.i.d. noise assumption for further detail.')
