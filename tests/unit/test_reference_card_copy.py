from __future__ import annotations

from api.reference_card_copy import (
    build_grounded_ref_why_line,
    finalize_ref_card_copy,
    looks_generic_ref_why_line,
    looks_templated_ref_why_line,
)


def test_generic_why_line_detector_catches_prompt_echo_template() -> None:
    assert looks_generic_ref_why_line(
        'This hit is directly relevant because it answers "Which paper in my library..."'
    )
    assert looks_generic_ref_why_line("这条命中直接回应用户查询，适合作为定位入口。")
    assert looks_generic_ref_why_line("该段落适合作为定位切口，因为属于当前命中证据的保守说明。")
    assert looks_generic_ref_why_line(
        "该文在“Abstract”处直接讨论了“single-pixel、imaging”，与问题的关注点直接对应。请只依据原文回答。"
    )


def test_templated_why_line_detector_does_not_reject_specific_evidence() -> None:
    assert not looks_templated_ref_why_line(
        "The Related Work section names alternating direction method of multipliers as the reconstruction baseline."
    )
    assert not looks_generic_ref_why_line(
        "Related Work 中明确提及 Snapshot Compressive Imaging（SCI），直接回应用户查询。"
    )


def test_build_grounded_ref_why_line_personalizes_chinese_copy() -> None:
    out = build_grounded_ref_why_line(
        prefer_zh=True,
        focus_terms=["ADMM"],
        heading_path="SCINeRF / 2. Related Work",
        action="define",
    )

    assert out == ""
    assert "这条命中" not in out


def test_finalize_ref_card_copy_replaces_template_why_line() -> None:
    summary, why, changed = finalize_ref_card_copy(
        summary_line="Most existing methods employ ADMM for iterative optimization.",
        why_line="This hit is directly relevant because it matches the user question.",
        prefer_zh=False,
        focus_terms=["ADMM"],
        heading_path="2. Related Work",
        action="define",
    )

    assert summary.startswith("Most existing methods")
    assert changed is True
    assert "This hit is directly relevant" not in why
    assert why == ""


def test_grounded_ref_why_line_uses_summary_when_terms_are_missing() -> None:
    out = build_grounded_ref_why_line(
        prefer_zh=True,
        focus_terms=[],
        heading_path="SCINeRF / 1. Introduction",
        summary_line="The paper targets recovering 3D scenes from a single coded snapshot.",
    )

    assert out == ""
    assert "这条命中" not in out
    assert "定位入口" not in out


def test_finalize_ref_card_copy_explains_spi_advantage_instead_of_echoing_prompt() -> None:
    summary, why, changed = finalize_ref_card_copy(
        summary_line=(
            "However, the limited image quality and lengthy computational times for iterative "
            "reconstruction still hinder its practical application."
        ),
        why_line=(
            "该文在“Abstract”处直接讨论了“single-pixel、imaging”，"
            "与问题的关注点直接对应。请只依据本文回答。"
        ),
        prefer_zh=True,
        focus_terms=["single-pixel imaging", "深度学习"],
        heading_path="Abstract",
        action="answer",
    )

    assert summary.startswith("However, the limited image quality")
    assert changed is True
    assert why == "摘要明确指出迭代重建同时受图像质量和计算耗时限制，这是判断深度学习为何能改善单像素成像实用性的直接依据。"
    assert "请只依据" not in why


def test_finalize_ref_card_copy_explains_self_supervised_method_evidence() -> None:
    _summary, why, changed = finalize_ref_card_copy(
        summary_line=(
            "The self-supervised image-loop neural network improves image details for "
            "single-pixel imaging without ground-truth images."
        ),
        why_line="与当前问题的关注点直接对应。",
        prefer_zh=True,
        focus_terms=["single-pixel imaging"],
        heading_path="2. Method and experiment setup",
    )

    assert changed is True
    assert "自监督网络" in why
    assert "无需真值图像" in why


def test_grounded_ref_why_line_preserves_negated_improvement_semantics() -> None:
    _summary, why, changed = finalize_ref_card_copy(
        summary_line="Deep learning does not improve image quality under severe noise.",
        why_line="与当前问题的关注点直接对应。",
        prefer_zh=True,
        focus_terms=["deep learning", "single-pixel imaging"],
        heading_path="Limitations",
    )

    assert changed is True
    assert "并未带来" in why
    assert "方法局限" in why
    assert "实际收益" not in why


def test_grounded_ref_why_line_explains_physical_degradation_generalization() -> None:
    _summary, why, changed = finalize_ref_card_copy(
        summary_line=(
            "This generalization performance is attributed to the incorporation of the physical "
            "degradation model of single-pixel imaging, which enables the network to learn "
            "degradation-robust representations."
        ),
        why_line="可用来核对‘Introduction’里怎样讨论‘single pixel imaging’。",
        prefer_zh=True,
        focus_terms=["single pixel imaging"],
        heading_path="Results / Simulation results",
    )

    assert changed is True
    assert "物理退化模型" in why
    assert "鲁棒性" in why
    assert "可用来核对" not in why
def test_finalize_ref_card_copy_replaces_location_shell_with_metric_specific_why() -> None:
    _summary, why, changed = finalize_ref_card_copy(
        summary_line="Choosing 333 patterns yields a reconstruction frame rate of 30 Hz.",
        why_line="\u539f\u6587\u5728\u2018Results\u2019\u8868\u660e\uff1aChoosing 333 patterns yields 30 Hz.",
        prefer_zh=True,
        focus_terms=["real-time single-pixel imaging"],
        heading_path="Results",
    )

    assert changed is True
    assert "333" in why
    assert "30 Hz" in why
    assert not why.startswith("\u539f\u6587\u5728")


def test_metric_specific_why_does_not_invent_missing_sampling_count() -> None:
    _summary, why, changed = finalize_ref_card_copy(
        summary_line="The reconstruction frame rate reaches 30 Hz in the experiment.",
        why_line="\u4e0e\u5f53\u524d\u95ee\u9898\u7684\u5173\u6ce8\u70b9\u76f4\u63a5\u5bf9\u5e94\u3002",
        prefer_zh=True,
        focus_terms=["real-time single-pixel imaging"],
        heading_path="Results",
    )

    assert changed is True
    assert "30 Hz" in why
    assert "333" not in why


def test_generic_relevance_template_is_removed_when_no_specific_relation_exists() -> None:
    summary, why, changed = finalize_ref_card_copy(
        summary_line="The paper describes a reconstruction method for single-pixel imaging.",
        why_line="Use this evidence to check how the paper discusses reconstruction.",
        prefer_zh=False,
        focus_terms=["reconstruction"],
        heading_path="Methods",
    )

    assert summary.startswith("The paper describes")
    assert why == ""
    assert changed is True
