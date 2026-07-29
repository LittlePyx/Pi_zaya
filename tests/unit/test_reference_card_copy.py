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
    assert looks_generic_ref_why_line(
        "“Abstract”提供回答该问题所需的原文定位，卡片中的结论可在这里逐项核对。"
    )
    assert looks_generic_ref_why_line(
        "该引用复用生成回答时实际提供的原文证据，且与答案中的关键词一致。"
    )
    assert looks_generic_ref_why_line(
        "“Introduction”给出该方法的定义或结果，是与另一方法逐项对照时的原文依据。"
    )
    assert looks_generic_ref_why_line(
        "'Introduction' provides the method definition or result needed for a point-by-point comparison."
    )


def test_templated_why_line_detector_does_not_reject_specific_evidence() -> None:
    assert not looks_templated_ref_why_line(
        "The Related Work section names alternating direction method of multipliers as the reconstruction baseline."
    )
    assert not looks_generic_ref_why_line(
        "Related Work 中明确提及 Snapshot Compressive Imaging（SCI），直接回应用户查询。"
    )


def test_english_grounded_copy_uses_evidence_instead_of_a_template_shell() -> None:
    why = build_grounded_ref_why_line(
        prefer_zh=False,
        focus_terms=[],
        heading_path="SCINeRF / Abstract",
        summary_line=(
            "SCINeRF formulates the physical imaging process of SCI as part "
            "of the training of NeRF."
        ),
    )

    assert "lineage evidence" in why
    assert "neural scene representation" in why
    assert not looks_generic_ref_why_line(why)


def test_build_grounded_ref_why_line_personalizes_chinese_copy() -> None:
    out = build_grounded_ref_why_line(
        prefer_zh=True,
        focus_terms=["ADMM"],
        heading_path="SCINeRF / 2. Related Work",
        action="define",
    )

    assert out == ""
    assert "这条命中" not in out


def test_grounded_ref_why_line_explains_admm_prior_work_origin() -> None:
    zh = build_grounded_ref_why_line(
        prefer_zh=True,
        focus_terms=[],
        heading_path="SCINeRF / 2. Related Work",
        summary_line="Most existing methods employ ADMM for iterative optimization.",
    )
    en = build_grounded_ref_why_line(
        prefer_zh=False,
        focus_terms=[],
        heading_path="SCINeRF / 2. Related Work",
        summary_line="Most existing methods employ ADMM for iterative optimization.",
    )

    assert all(term in zh for term in ("ADMM", "已有方法", "不是本文新提出"))
    assert "existing methods" in en
    assert "new contributions" in en
    assert not looks_generic_ref_why_line(zh)
    assert not looks_generic_ref_why_line(en)


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
    assert "existing methods" in why
    assert "new contributions" in why


def test_finalize_ref_card_copy_separates_detector_guide_and_relevance() -> None:
    summary = (
        "Single-photon detections can detect individual photons at very low light levels. "
        "Mainstream SPDs include PMTs, SPADs, SNSPDs, and TES."
    )
    summary_out, why, changed = finalize_ref_card_copy(
        summary_line=summary,
        why_line="该引用复用生成回答时实际提供的原文证据，且与答案中的关键词一致。",
        prefer_zh=True,
        focus_terms=["single-photon", "detector"],
        heading_path="Abstract",
        action="compare",
    )

    assert summary_out == summary
    assert changed is True
    assert why != summary_out
    assert all(term in why for term in ("SPAD", "硬件", "基线"))


def test_detector_metric_table_gets_specific_relevance_copy() -> None:
    summary = (
        "Si-SPAD working parameter (wavelength) = 400-1000 nm; "
        "quantum efficiency = 50%-92%; operating temperature = 200-300 K."
    )

    why = build_grounded_ref_why_line(
        prefer_zh=True,
        focus_terms=["single-photon detector"],
        heading_path="2.3 Superconducting",
        summary_line=summary,
    )

    assert all(term in why for term in ("Si-SPAD", "工作波段", "量子效率", "温控", "硬件基线"))
    assert not looks_generic_ref_why_line(why)


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


def test_grounded_ref_why_line_explains_microscopy_method_roles() -> None:
    cases = [
        (
            "Since super-resolution and optical sectioning are achieved simultaneously, "
            "we named our technique s²ISM.",
            ("s²ISM", "超分辨率", "光学切片"),
        ),
        (
            "This technique combines interferometric detection with image scanning microscopy "
            "to achieve about 120 nm lateral resolution.",
            ("干涉检测", "120 nm", "分辨率"),
        ),
        (
            "Light-field imaging captures both position and angular information for volumetric reconstruction.",
            ("位置", "角度", "体积"),
        ),
    ]

    for summary, required in cases:
        why = build_grounded_ref_why_line(
            prefer_zh=True,
            focus_terms=[],
            heading_path="Abstract",
            summary_line=summary,
        )
        assert all(term in why for term in required)
        assert not looks_generic_ref_why_line(why)


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


def test_grounded_ref_why_line_explains_cassi_measurement_chain() -> None:
    _summary, why, changed = finalize_ref_card_copy(
        summary_line=(
            "A dual-disperser architecture with a binary-valued aperture compresses "
            "the spectral data cube into a single detector measurement."
        ),
        why_line="与当前问题的关注点直接对应。",
        prefer_zh=True,
        focus_terms=["CASSI"],
        heading_path="System architecture",
    )

    assert changed is True
    assert all(term in why for term in ("CASSI", "双色散器", "二值编码孔径", "光谱"))


def test_grounded_ref_why_line_explains_scinerf_bridge() -> None:
    _summary, why, changed = finalize_ref_card_copy(
        summary_line=(
            "SCINeRF formulates the physical imaging process of SCI as part of "
            "the training of NeRF."
        ),
        why_line="Use this evidence to check the answer.",
        prefer_zh=True,
        focus_terms=["SCINeRF"],
        heading_path="Introduction",
    )

    assert changed is True
    assert all(term in why for term in ("SCI", "NeRF", "物理成像", "三维"))


def test_grounded_ref_why_line_explains_scigs_dynamic_3d_stage() -> None:
    _summary, why, changed = finalize_ref_card_copy(
        summary_line=(
            "SCIGS reconstructs a 3D explicit scene from a single compressed image "
            "and extends the application to dynamic 3D scenes."
        ),
        why_line="Use this evidence to check the answer.",
        prefer_zh=True,
        focus_terms=["SCIGS"],
        heading_path="Abstract",
    )

    assert changed is True
    assert all(term in why for term in ("SCIGS", "单幅压缩图像", "三维", "动态"))


def test_grounded_ref_why_line_explains_s2ism_tradeoff() -> None:
    _summary, why, changed = finalize_ref_card_copy(
        summary_line=(
            "Reducing the pinhole size improves spatial resolution and optical sectioning "
            "but decreases the signal-to-noise ratio (SNR)."
        ),
        why_line="与当前问题的关注点直接对应。",
        prefer_zh=True,
        focus_terms=["s2ISM"],
        heading_path="Discussion",
    )

    assert changed is True
    assert all(term in why for term in ("空间分辨率", "光学切片", "信噪比", "s2ISM"))


def test_grounded_ref_why_line_explains_s2ism_thick_sample_limitation() -> None:
    _summary, why, changed = finalize_ref_card_copy(
        summary_line=(
            "Current image scanning microscopy approaches do not provide optical "
            "sectioning and fail with thick samples unless the detector size is limited, "
            "introducing a trade-off with signal-to-noise ratio."
        ),
        why_line="与当前问题相关。",
        prefer_zh=True,
        focus_terms=["s2ISM"],
        heading_path="Abstract",
    )

    assert changed is True
    assert all(term in why for term in ("厚样本", "光学切片", "信噪比", "s2ISM"))
    assert len(why) >= 45


def test_grounded_ref_why_line_handles_truncated_s2ism_abstract() -> None:
    _summary, why, changed = finalize_ref_card_copy(
        summary_line=(
            "Fast detector arrays enable image scanning microscopy, which overcomes the "
            "trade-off between spatial resolution and signal-to-noise ratio. However, "
            "current approaches do not provide optical sectioning."
        ),
        why_line="",
        prefer_zh=True,
        focus_terms=["s2ISM", "三方权衡"],
        heading_path="Abstract",
    )

    assert changed is True
    assert all(term in why for term in ("空间分辨率", "光学切片", "信噪比", "s2ISM"))


def test_grounded_ref_why_line_explains_fdm_parallel_channels() -> None:
    _summary, why, changed = finalize_ref_card_copy(
        summary_line=(
            "Different detector channels are encoded at subcarrier frequencies; "
            "the measured crosstalk is 2.3% and 13.0%."
        ),
        why_line="与当前问题的关注点直接对应。",
        prefer_zh=True,
        focus_terms=["FDM"],
        heading_path="Results",
    )

    assert changed is True
    assert all(term in why for term in ("子载波", "并行", "2.3%", "13.0%"))


def test_comparison_evidence_gets_specific_relevance_copy() -> None:
    fdm = build_grounded_ref_why_line(
        prefer_zh=True,
        focus_terms=[],
        heading_path="Abstract",
        summary_line=(
            "We implement frequency-division methods to parallelize the single-pixel "
            "imaging process and improve acquisition speed without altering detector "
            "integration time."
        ),
    )
    fdm_encoding = build_grounded_ref_why_line(
        prefer_zh=True,
        focus_terms=[],
        heading_path="B. Encoding",
        summary_line=(
            "Each pixel is modulated on p frequencies simultaneously. The light is "
            "multiplexed into a single-pixel detector and the signal is then demodulated "
            "by p lock-in amplifiers."
        ),
    )
    video = build_grounded_ref_why_line(
        prefer_zh=True,
        focus_terms=[],
        heading_path="Abstract",
        summary_line=(
            "Photometric stereo uses four spatially-separated single-pixel detectors "
            "to reconstruct 3D video at 8 frames per second."
        ),
    )
    foveated = build_grounded_ref_why_line(
        prefer_zh=True,
        focus_terms=[],
        heading_path="Abstract",
        summary_line=(
            "A high-resolution foveal region tracks motion while every frame delivers "
            "new information from the entire field of view and slower regions accumulate "
            "detail over consecutive frames."
        ),
    )
    bases = build_grounded_ref_why_line(
        prefer_zh=True,
        focus_terms=[],
        heading_path="Introduction",
        summary_line=(
            "HSI uses Hadamard basis patterns while FSI uses Fourier basis patterns "
            "and the paper compares imaging efficiency and noise robustness."
        ),
    )

    assert "频分复用" in fdm and "积分时间" in fdm
    assert "p 个频率通道" in fdm_encoding and "并行解调" in fdm_encoding
    assert "四个" in video and "8 帧/秒" in video
    assert "中央凹" in foveated and "跨帧" in foveated
    assert "Hadamard" in bases and "Fourier" in bases
