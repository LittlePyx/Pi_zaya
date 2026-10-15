from __future__ import annotations

from api.reference_card_quality import attach_ref_card_polish_contract
from api.source_relation_copy import source_relation_copy


def test_specific_copy_comes_from_relation_not_keywords():
    assert source_relation_copy("We measure loss, noise, variance, and supervised performance.") is None
    copy = source_relation_copy("The self-supervised loss is the sum of the ordinary supervised loss and the variance of the noise.")
    assert copy and "之和" in copy[0] and "约束" in copy[1]
    assert source_relation_copy("We cannot calibrate traditional image denoising methods such as median filters and non-local means.") is None


def test_dncnn_residual_copy_requires_the_output_relation():
    copy = source_relation_copy(
        "Rather than directly outputting the denoised image \\hat{x}, the proposed DnCNN is designed "
        "to predict the residual image \\hat{v}, i.e., the difference between the noisy observation and the latent clean image."
    )
    assert copy and "\u9884\u6d4b\u7684\u662f" in copy[0] and "\u8f93\u51fa\u6b8b\u5dee" in copy[1]
    assert source_relation_copy(
        "DnCNN predicts the clean image and defines a residual as the difference between the latent clean image and noisy observation."
    ) is None


def test_missing_copy_can_use_actual_primary_evidence():
    evidence = "We can calibrate traditional image denoising methods such as median filters and non-local means."
    ui = attach_ref_card_polish_contract({"render_locale": "zh", "primary_evidence": {"snippet": evidence}, "summary_line": "", "why_line": ""})
    assert "中值" in ui["summary_line"] and "传统算法" in ui["why_line"]


def test_calibration_copy_wins_when_block_also_contains_noise_assumption():
    evidence = (
        "We use groups of features whose noise is independent conditional on the true signal to predict one another. "
        "The same approach can also be used to calibrate traditional image denoising methods such as median filters "
        "and non-local means."
    )
    copy = source_relation_copy(evidence)
    assert copy and "中值滤波" in copy[0] and "校准范围" in copy[1]


def test_paper_title_cannot_supply_card_relation():
    evidence = "calibrate traditional image denoising methods such as median filters and non-local means"
    ui = attach_ref_card_polish_contract({"render_locale": "zh", "display_name": evidence, "primary_evidence": {"snippet": "Different subject."}})
    assert "中值" not in ui.get("summary_line", "")


def test_noise2void_pixel_independence_gets_specific_copy():
    copy = source_relation_copy(
        "We assume the noise to be pixel-wise independent given the signal; neighboring pixels carry no information about n_i."
    )
    assert copy and "逐像素" in copy[0] and "盲点网络" in copy[1]


def test_architecture_copy_requires_pooling_filter_and_formula_relation():
    copy = source_relation_copy(
        "We set filters to 3 × 3 and remove all pooling layers. "
        "The receptive field with depth of d should be (2d+1)×(2d+1)."
    )
    assert copy and "移除全部池化层" in copy[0] and "感受野公式" in copy[1]
    assert source_relation_copy(
        "We set filters to 3 × 3 and remove all pooling layers. The receptive field grows."
    ) is None


def test_task_enumeration_and_conditional_target_copy_are_evidence_bound():
    tasks = source_relation_copy(
        "We consider blind Gaussian denoising, single image super-resolution, and JPEG image deblocking."
    )
    assert tasks and "超分辨率" in tasks[0] and "三类恢复任务" in tasks[1]

    targets = source_relation_copy(
        "Input-conditioned target distributions may be replaced with arbitrary distributions that have "
        "the same conditional expected values. This implies that targets can be corrupted with zero-mean noise."
    )
    assert targets and "条件期望" in targets[0] and "有偏" in targets[1]
    assert source_relation_copy(
        "Input-conditioned target distributions have different conditional expected values and non-zero-mean noise."
    ) is None


def test_training_pair_and_two_assumption_copy_require_complete_relations():
    pairs = source_relation_copy(
        "Our N2V scheme does not require noisy image pairs, nor clean target images."
    )
    assert pairs and "独立带噪图像对" in pairs[0] and "两类配对数据" in pairs[1]
    assert source_relation_copy("Our N2V scheme does not require noisy image pairs.") is None

    assumptions = source_relation_copy(
        "The signal is not pixel-wise independent, and the noise is conditionally pixel-wise independent given the signal."
    )
    assert assumptions and "信号不是逐像素独立" in assumptions[0]
    assert "两条假设" in assumptions[1]
