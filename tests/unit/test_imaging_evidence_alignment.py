from __future__ import annotations

from kb.evidence_binding import assess_system_a_hit_binding
from kb.imaging_equation_binding import has_masked_frame_sum
from kb.claim_evidence_runtime import _support_score


_SOURCE = r"""The whole imaging process is Y = \sum_i X_i \odot M_i + Z.
Y and X_i are the captured compressed image and the virtual image within exposure time.
The binary mask uses element-wise multiplication; Z is measurement noise."""


def _binding(claim: str):
    return assess_system_a_hit_binding(
        answer_claim=claim, hit={"text": _SOURCE},
        meta={"citation_plan_evidence_authoritative": True, "citation_plan_full_evidence_quote": _SOURCE, "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence"},
        heading="Image Formation", evidence_quote=_SOURCE, source_name="paper.pdf",
    )


def test_chinese_sum_of_masked_frames_binds_to_explicit_imaging_equation():
    assert _binding("传感器把每个虚拟帧经对应掩模调制后的结果在曝光时间内求和，再叠加噪声，得到一张压缩测量。 ")["status"] == "grounded"


def test_reversed_mask_order_and_multiplicative_noise_are_not_supported():
    assert _binding("先把虚拟帧求和后再用掩模调制，得到压缩图像。")["status"] == "mismatch"
    assert _binding("虚拟帧与测量噪声相乘，得到压缩图像。")["status"] == "mismatch"


def test_symbolic_masked_sum_counts_as_explicit_relation_and_not_keyword_only():
    source = r"$$Y = \sum_{i=1}^{N} X_i \odot M_i + Z$$ Binary masks encode the virtual images during exposure time; Z is measurement noise."
    claim = "传感器把每个虚拟帧经对应掩模调制后的结果在曝光时间内求和，再叠加噪声，得到一张压缩测量。"
    assert has_masked_frame_sum(source)
    assert _support_score(claim, source) >= 5
    assert not has_masked_frame_sum(r"$$Y = (\sum_i X_i) \odot M + Z$$ A binary mask is applied after summation.")
