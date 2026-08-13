import subprocess
import sys

import pytest

from kb.claim_evidence_runtime import (
    _mentioned_source_hit_indexes,
    _split_claim_segments,
    audit_and_repair_claim_evidence,
    claim_evidence_audit,
)


def test_claim_splitter_keeps_semicolon_inside_inline_math() -> None:
    line = (
        "For DCD, $Y = [Y^c;Y^p]$ and "
        "$\\boldsymbol{\\Phi} = [\\boldsymbol{\\Phi}^c;\\boldsymbol{\\Phi}^p]$."
    )

    assert _split_claim_segments(line) == [line]


def test_display_equation_inherits_adjacent_cited_variable_definition() -> None:
    answer = (
        "其相位关系为：\n\n"
        "$$\n"
        "\\Delta\\varphi = \\frac{4\\pi}{\\lambda} n z + "
        "\\varphi_{\\text{Gouy}} \\tag{2}\n"
        "$$\n\n"
        "其中 $n$ 为折射率，$z$ 为轴向位置，$\\lambda$ 为波长 [2](#source)。"
    )

    audit = claim_evidence_audit(answer)

    assert audit["uncited_high_risk_claims"] == 0
    assert not audit["unresolved_claims"]
    assert not audit["unresolved_claims"]


def test_display_equation_inherits_adjacent_clickable_source_note() -> None:
    answer = (
        "The quantizer is:\n\n"
        "$$\n"
        r"\widetilde{W}=\text{RoundClip}(W/\gamma,-1,1), \tag{1}" "\n"
        "$$\n"
        "*（式(1) 对应命中的库内文献：`bitnet.pdf`）*\n"
        "[1](#source)"
    )

    audit = claim_evidence_audit(answer)

    assert audit["uncited_high_risk_claims"] == 0
    assert not audit["unresolved_claims"]


def test_display_equation_inherits_citation_on_same_source_note_line() -> None:
    answer = (
        "The quantizer is:\n\n"
        "$$\n"
        r"\widetilde{W}=\text{RoundClip}(W/\gamma,-1,1), \tag{1}" "\n"
        "$$\n"
        "*（式(1) 对应命中的库内文献：`bitnet.pdf`）* [1](#source)"
    )

    audit = claim_evidence_audit(answer)

    assert audit["uncited_high_risk_claims"] == 0
    assert not audit["unresolved_claims"]


def test_display_equation_inherits_citation_from_explicit_intro_line() -> None:
    answer = (
        "γ 是权重矩阵所有元素的平均绝对值： [1](#source)\n\n"
        "$$\n"
        r"\gamma = \frac{1}{nm} \sum_{ij} |W_{ij}|, \tag{3}" "\n"
        "$$\n\n"
        "将原始权重除以 $\\gamma + \\epsilon$，得到缩放后的权重。 "
        "[1](#source)\n\n"
        "$$\n"
        r"\widetilde{W}=\text{RoundClip}(W/\gamma,-1,1), \tag{1}" "\n"
        "$$"
    )

    audit = claim_evidence_audit(answer)

    assert audit["uncited_high_risk_claims"] == 0
    assert not audit["unresolved_claims"]


def test_strict_claim_repair_never_inserts_a_citation_inside_display_math() -> None:
    answer = (
        "训练中最小化的点预测损失为 [1]：\n\n"
        "$$\n"
        r"\text{TrainLoss} = \frac{1}{N} \sum_{j=1}^{N} "
        r"\text{MSE}\left(\hat{y}_{pj+1:pj+h}, y_{pj+1:pj+h}\right)"
        "\n$$"
    )
    hit = {
        "text": (
            "We focus on point forecasting and use a point forecasting loss during "
            "training like Mean Squared Error (MSE)."
        )
    }

    repaired, _meta = audit_and_repair_claim_evidence(
        answer,
        [hit],
        prompt="What loss does TimesFM train with?",
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    math_body = repaired.split("$$", 2)[1]
    assert "[1]" not in math_body
    assert r"y_{pj+1:pj+h}\right)" in math_body


def test_display_equation_does_not_inherit_unrelated_preceding_citation() -> None:
    answer = (
        "This paper also reports an ablation result [1](#source).\n\n"
        "$$\n"
        r"\widetilde{W}=\text{RoundClip}(W/\gamma,-1,1), \tag{1}" "\n"
        "$$"
    )

    audit = claim_evidence_audit(answer)

    assert audit["uncited_high_risk_claims"] == 1


def test_final_gate_restores_complete_iism_depth_phase_relation() -> None:
    evidence = (
        "In a confocal geometry, the relative phase between reflected and scattered "
        "electric fields is Delta phi = 4\\pi n z / \\lambda + phi_Gouy, with n "
        "the refractive index of the medium, z the axial position of the scatterer, "
        "lambda the illumination wavelength, and phi_Gouy the Gouy phase."
    )
    repaired, meta = audit_and_repair_claim_evidence(
        "相位公式给出了深度项。",
        [
            {
                "text": evidence,
                "meta": {"citation_plan_evidence_quotes": [evidence]},
            }
        ],
        prompt="iISM 的相位为何携带深度？z、n、λ 和 Gouy phase 分别是什么？",
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert all(term in repaired for term in ("iISM", "4\\pi", "反射光", "散射光", "轴向位置", "折射率", "照明波长", "Gouy"))
    assert "[1]" in repaired
    assert meta["restored_source_facts"] == 1


def test_final_gate_restores_qclfm_refocus_steps_after_claim_removal() -> None:
    evidence = (
        "The operation for digital refocusing can be achieved using two steps. First, "
        "using the position and angular information of each photon, the trajectory can "
        "be reconstructed through a ray tracing operation. The second step is to reverse "
        "this diffraction by applying a wave propagation of distance -z."
    )
    repaired, meta = audit_and_repair_claim_evidence(
        "iISM 的相位携带深度信息 [[CITE:source=QCLFM.pdf]]。",
        [{"text": evidence, "meta": {"citation_plan_evidence_quotes": [evidence]}}],
        prompt="比较 QCLFM 怎样用位置与角度信息做两步数字重聚焦与 iISM 的 Gouy phase。",
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert all(term in repaired for term in ("QCLFM", "光线追踪", "波传播", "-z", "[1]"))
    assert meta["restored_source_facts"] == 1
    assert meta["minimum_ok"] is True


def test_named_source_detection_does_not_treat_generic_transformer_as_a_paper() -> None:
    hits = [
        {
            "text": "Autoformer uses Auto-Correlation for long-term forecasting.",
            "meta": {"source_name": "Autoformer.pdf"},
        },
        {
            "text": "FEDformer is a frequency enhanced Transformer.",
            "meta": {"source_name": "FEDformer.pdf"},
        },
        {
            "text": "TimesNet transforms one-dimensional series into 2D space.",
            "meta": {"source_name": "TimesNet.pdf"},
        },
        {
            "text": "PatchTST uses patches as Transformer input tokens.",
            "meta": {"source_name": "A Time Series is Worth 64 Words.pdf"},
        },
    ]

    assert _mentioned_source_hit_indexes(
        "Autoformer uses an input length of 96 in the experiment.",
        hits,
    ) == [1]
    assert _mentioned_source_hit_indexes(
        "PatchTST uses patches as Transformer tokens.",
        hits,
    ) == [4]


def test_claim_evidence_runtime_does_not_load_legacy_ui_renderer() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import kb.claim_evidence_runtime; "
                "assert 'ui.refs_renderer' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr


def test_final_gate_keeps_supported_review_coverage_and_drops_unsupported_suffix() -> None:
    review_quote = (
        "This review summarizes the principles and technical challenges of several "
        "single-photon detectors. Their complexity and high manufacturing cost pose "
        "challenges to wider adoption."
    )
    method_quote = (
        "We introduce deep learning into SPAD for improved single-photon imaging."
    )
    hits = [
        {
            "text": review_quote,
            "meta": {
                "source_name": "single-photon detector review.pdf",
                "citation_plan_evidence_quotes": [review_quote],
            },
        },
        {
            "text": method_quote,
            "meta": {
                "source_name": "physics-informed deep learning.pdf",
                "citation_plan_evidence_quotes": [method_quote],
            },
        },
    ]
    answer = (
        "读这篇综述可以了解各类探测器的物理原理、制造难度和适用场景，"
        "为理解暗计数和死时间提供背景。\n\n"
        "该方法论文使用 deep learning 改进 SPAD 成像 [2]。"
    )

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        prompt="这两篇文献应该如何搭配阅读？",
        allowed_citation_numbers={1, 2},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert "各类探测器的物理原理、制造难度和适用场景 [1]" in repaired
    assert "暗计数和死时间提供背景" not in repaired
    assert "SPAD 成像 [2]" in repaired
    assert meta["minimum_ok"] is True
    assert any(
        item.get("reason") == "supported_paper_coverage_prefix"
        for item in meta.get("repairs") or []
    )


def test_chinese_fdm_claim_is_repaired_from_unique_english_evidence() -> None:
    answer = "频分复用可并行采集，并在不改变探测器积分时间时权衡信噪比与采集速度。"
    hits = [
        {
            "text": (
                "Frequency-division multiplexed single-pixel imaging parallelizes acquisition "
                "and offers a trade-off between signal-to-noise ratio and acquisition speed "
                "without altering detector integration time."
            )
        },
        {"text": "A figure visualizes illumination pattern state characterization."},
    ]

    repaired, meta = audit_and_repair_claim_evidence(answer, hits)

    assert "[1]" in repaired
    assert meta["repaired_citations"] == 1
    assert meta["uncited_high_risk_claims"] == 0


def test_source_blockquote_is_not_rewritten_as_an_answer_claim() -> None:
    answer = "> Frequency-division multiplexing parallelizes acquisition."

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        [{"text": "Frequency-division multiplexing parallelizes acquisition."}],
    )

    assert repaired == answer
    assert meta["total_claims"] == 0
    assert meta["repaired_citations"] == 0


def test_unique_matching_hit_repairs_missing_claim_citation():
    answer = "该方法把 SPAD 串扰和暗计数纳入物理噪声模型。"
    hits = [
        {
            "text": (
                "We introduce a multi-source physical noise model for SPAD imaging, "
                "including crosstalk and dark count rate."
            )
        },
        {"text": "Single-pixel imaging uses structured illumination and a bucket detector."},
    ]

    repaired, meta = audit_and_repair_claim_evidence(answer, hits)

    assert repaired == "该方法把 SPAD 串扰和暗计数纳入物理噪声模型 [1]。"
    assert meta["repaired_citations"] == 1
    assert meta["uncited_high_risk_claims"] == 0


def test_ambiguous_hit_does_not_guess_a_citation():
    answer = "该方法使用 SPAD 物理模型提升重建质量。"
    hits = [
        {"text": "A SPAD physical model improves reconstruction quality."},
        {"text": "The SPAD physical model is used for reconstruction quality."},
    ]

    repaired, meta = audit_and_repair_claim_evidence(answer, hits)

    assert repaired == answer
    assert meta["repaired_citations"] == 0
    assert meta["uncited_high_risk_claims"] == 1


def test_absolute_negative_is_scoped_to_current_cited_evidence():
    answer = "这篇论文未验证真实 SPAD 数据。"

    repaired, meta = audit_and_repair_claim_evidence(answer, [])

    assert repaired == "当前引用证据未直接验证真实 SPAD 数据。"
    assert meta["scoped_negative_claims"] == 1
    assert meta["uncited_high_risk_claims"] == 0


def test_existing_evidence_boundary_does_not_bind_an_unrelated_citation():
    answer = (
        "需要指出的是，现有证据并未明确说明该方法在训练数据极度匮乏时的鲁棒性边界，"
        "这些属于超出当前证据范围的推断 [3]。"
    )
    hits = [
        {"text": "A calibrated SPAD noise model is used to synthesize training pairs."},
        {"text": "A transformer reconstructs high-resolution single-photon images."},
        {"text": "Single-pixel imaging can improve reconstruction speed."},
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        drop_unsupported_high_risk_claims=True,
    )

    assert "当前引用证据未直接说明" in repaired
    assert "[3]" not in repaired
    assert meta["rebound_citations"] == 0
    assert meta["uncited_high_risk_claims"] == 0


def test_false_single_pixel_single_photon_equivalence_is_repaired():
    answer = "单像素成像（一种单光子成像的变体）可以加速重建 [1]。"

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        [{"text": "Single-pixel imaging with deep learning provides fast reconstruction speed."}],
    )

    assert "一种单光子成像的变体" not in repaired
    assert "与单光子成像不同" in repaired
    assert meta["repaired_modality_boundaries"] == 1


def test_hard_numeric_citation_mismatch_is_removed():
    answer = "核心结论由真实 SPAD 噪声支持 [1]。采集效率只有 7%，需要数百秒 [2]。"
    hits = [
        {"text": "The physical multi-source SPAD noise model improves fidelity."},
        {"text": "Light-field microscopy captures position and angular information."},
    ]

    repaired, meta = audit_and_repair_claim_evidence(answer, hits)

    assert "真实 SPAD 噪声" in repaired
    assert "7%" not in repaired
    assert meta["dropped_hard_mismatch_claims"] == 1


def test_digit_fold_claim_keeps_english_number_word_multiplier_evidence() -> None:
    answer = (
        "\u5165\u5c04\u7167\u660e\u529f\u7387\u964d\u4f4e\u7ea6 10 \u500d\uff0c"
        "\u53ef\u663e\u8457\u51cf\u5c11\u5149\u635f\u4f24 [1]\u3002"
    )
    evidence = (
        "iISM can operate with tenfold lower incident illumination power while "
        "reducing photodamage."
    )

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        [
            {
                "text": evidence,
                "meta": {"citation_plan_evidence_quotes": [evidence]},
            }
        ],
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert repaired == answer
    assert meta["dropped_hard_mismatch_claims"] == 0
    assert meta["renderer_rejected_citations"] == 0
    assert meta["citation_mismatch_claims"] == 0


def test_fold_equivalence_rejects_same_value_and_direction_for_different_metric() -> None:
    answer = (
        "\u5149\u573a\u4f4d\u7f6e\u5206\u8fa8\u7387\u901a\u5e38\u964d\u4f4e\u7ea6 10 \u500d [1]\u3002"
    )
    evidence = (
        "iISM can operate with tenfold lower incident illumination power while "
        "reducing photodamage."
    )

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        [
            {
                "text": evidence,
                "meta": {"citation_plan_evidence_quotes": [evidence]},
            }
        ],
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert repaired == ""
    assert meta["dropped_hard_mismatch_claims"] == 1


@pytest.mark.parametrize(
    "evidence",
    [
        (
            "iISM can operate with twelvefold lower incident illumination power while "
            "reducing photodamage."
        ),
        (
            "iISM can operate with tenfold higher incident illumination power while "
            "reducing photodamage."
        ),
        "iISM uses 10 Hz illumination while reducing photodamage.",
    ],
)
def test_fold_equivalence_rejects_magnitude_direction_and_unit_conflicts(
    evidence: str,
) -> None:
    answer = (
        "\u5165\u5c04\u7167\u660e\u529f\u7387\u964d\u4f4e\u7ea6 10 \u500d\uff0c"
        "\u53ef\u663e\u8457\u51cf\u5c11\u5149\u635f\u4f24 [1]\u3002"
    )

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        [
            {
                "text": evidence,
                "meta": {"citation_plan_evidence_quotes": [evidence]},
            }
        ],
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert repaired == ""
    assert meta["dropped_hard_mismatch_claims"] == 1


def test_prompt_spad_term_is_restored_from_grounded_evidence():
    answer = "物理信息深度学习改善单光子成像质量 [1]。"
    hits = [{"text": "Physics-informed deep learning improves SPAD single-photon imaging."}]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        prompt="physics-informed deep learning 在 SPAD 成像里帮了什么？",
    )

    assert "SPAD 单光子成像" in repaired
    assert meta["restored_prompt_terms"] == 1


def test_claim_audit_keeps_advice_out_of_high_risk_count():
    audit = claim_evidence_audit(
        "建议先阅读方法部分。\n\n该方法使用 SPAD 多源噪声模型 [1]。"
    )

    assert audit["high_risk_claims"] == 1
    assert audit["uncited_high_risk_claims"] == 0


def test_midphrase_chinese_citations_move_to_supported_clause_end() -> None:
    answer = (
        "搭配建议：先读探测器综述 [1] 的第 2.3 节和表 1，了解制造难点；"
        "再读方法论文 [2] 的物理噪声模型，理解它如何指导训练。"
    )

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        [
            {"text": "Section 2.3 and Table 1 compare detector manufacturing challenges."},
            {"text": "The physical noise model guides deep network training."},
        ],
        allow_citation_repairs=False,
    )

    assert "综述的第 2.3 节和表 1，了解制造难点 [1]；" in repaired
    assert "论文的物理噪声模型，理解它如何指导训练 [2]。" in repaired
    assert "[1] 的" not in repaired
    assert "[2] 的" not in repaired
    assert meta["relocated_midphrase_citations"] == 2


def test_anaphoric_core_claim_inherits_previous_grounded_source() -> None:
    answer = (
        "Foveated dynamic supersampling 让高分辨率中央凹区域跟踪运动，"
        "同时每一帧仍从整个视场获取新的空间信息 [1]。"
        "其核心是快速记录快速变化的特征，同时通过多帧累积增强慢变区域细节。"
    )
    hits = [
        {
            "text": (
                "A high-resolution foveal region tracks motion while every frame delivers "
                "new spatial information across the entire field of view. This strategy "
                "rapidly records quickly changing features while accumulating detail of "
                "more slowly evolving regions over consecutive frames."
            )
        }
    ]

    repaired, meta = audit_and_repair_claim_evidence(answer, hits)

    assert "其核心是快速记录快速变化的特征，同时通过多帧累积增强慢变区域细节 [1]。" in repaired
    assert meta["uncited_high_risk_claims"] == 0


def test_final_gate_drops_unsupported_uncited_high_risk_claim_without_strict_plan() -> None:
    answer = (
        "该论文报告中央凹区域会跟踪运动 [1]。"
        "该系统还能把所有场景的重建速度提高十倍。"
    )

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        [{"text": "A high-resolution foveal region tracks motion within the scene."}],
        drop_unsupported_high_risk_claims=True,
    )

    assert "中央凹区域会跟踪运动 [1]" in repaired
    assert "提高十倍" not in repaired
    assert meta["dropped_unsupported_unplanned_claims"] == 1


def test_final_gate_drops_high_risk_claim_with_wrong_numeric_source() -> None:
    answer = (
        "\u7cfb\u7edf\u8ddf\u8e2a\u4e2d\u592e\u51f9\u533a\u57df\u7684\u8fd0\u52a8 [1]\u3002"
        "\u901a\u8fc7\u727a\u7272\u9759\u6b62\u533a\u57df\u5355\u5e27\u7684\u91c7\u6837\u6570\uff0c"
        "\u6362\u53d6\u8fd0\u52a8\u533a\u57df\u7684\u9ad8\u65f6\u95f4\u5206\u8fa8\u7387 [1]\u3002"
    )
    hits = [
        {
            "text": (
                "A high-resolution foveal region tracks motion while every frame "
                "delivers new spatial information across the entire field of view."
            )
        }
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1},
        drop_unsupported_high_risk_claims=True,
    )

    assert "\u4e2d\u592e\u51f9\u533a\u57df\u7684\u8fd0\u52a8 [1]" in repaired
    assert "\u727a\u7272\u9759\u6b62\u533a\u57df" not in repaired
    assert meta["stripped_weak_citations"] == 1
    assert meta["dropped_unsupported_unplanned_claims"] == 1
    assert meta["uncited_high_risk_claims"] == 0


def test_final_gate_rejects_specific_relations_missing_from_source() -> None:
    answer = (
        "\u4e2d\u592e\u51f9\u533a\u57df\u8ddf\u8e2a\u8fd0\u52a8 [1]\u3002"
        "\u8fd9\u4e00\u9009\u62e9\u51b3\u5b9a\u4e86\u5355\u6b21\u6d4b\u91cf\u7684\u4fe1\u566a\u6bd4\u548c\u7b97\u6cd5\u590d\u6742\u5ea6 [1]\u3002"
        "\u7cfb\u7edf\u5229\u7528\u6709\u9650\u6d4b\u91cf\u9884\u7b97\u548c\u603b\u5e27\u6570\u8fdb\u884c\u52a8\u6001\u5206\u914d [1]\u3002"
    )
    hits = [
        {
            "text": (
                "A high-resolution foveal region tracks motion while every frame "
                "delivers new spatial information across the entire field of view."
            )
        }
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1},
        drop_unsupported_high_risk_claims=True,
    )

    assert "\u4e2d\u592e\u51f9\u533a\u57df\u8ddf\u8e2a\u8fd0\u52a8 [1]" in repaired
    assert "\u4fe1\u566a\u6bd4" not in repaired
    assert "\u6709\u9650\u6d4b\u91cf\u9884\u7b97" not in repaired
    assert meta["stripped_weak_citations"] == 2
    assert meta["dropped_unsupported_unplanned_claims"] == 2


def test_final_gate_removes_only_the_wrong_citation_from_multi_source_claim() -> None:
    answer = (
        "Hadamard \u5355\u50cf\u7d20\u6210\u50cf\u4f7f\u7528 Hadamard \u57fa\u56fe\u6848\uff0c"
        "Fourier \u5355\u50cf\u7d20\u6210\u50cf\u4f7f\u7528 Fourier \u57fa\u56fe\u6848 [2] [1]\u3002"
    )
    hits = [
        {"text": "A high-resolution foveal region tracks motion over consecutive frames."},
        {"text": "HSI uses Hadamard basis patterns while FSI uses Fourier basis patterns."},
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1, 2},
        drop_unsupported_high_risk_claims=True,
    )

    assert "[2]" in repaired
    assert "[1]" not in repaired
    assert meta["stripped_weak_citations"] == 1


def test_final_gate_renumbers_list_after_dropping_first_item() -> None:
    answer = (
        "**\u8bbe\u8ba1\u51b3\u7b56**\uff1a\n"
        "1. \u8fd9\u4e00\u9009\u62e9\u51b3\u5b9a\u4e86\u672a\u62a5\u544a\u7684\u4fe1\u566a\u6bd4 [1]\u3002\n"
        "2. Foveated supersampling \u8d1f\u8d23\u8d44\u6e90\u5206\u914d [1]\u3002"
    )
    hits = [
        {"text": "A high-resolution foveal region tracks motion over consecutive frames."}
    ]

    repaired, _meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1},
        drop_unsupported_high_risk_claims=True,
    )

    assert "1. Foveated supersampling" in repaired
    assert "2. Foveated supersampling" not in repaired


def test_final_gate_drops_citation_the_renderer_would_hide() -> None:
    answer = (
        "\u8fd9\u4e2a\u9009\u62e9\u51b3\u5b9a\u4e86\u4e0d\u540c\u91c7\u6837\u7387\u4e0b\u7684\u91cd\u5efa\u8d28\u91cf\u548c\u566a\u58f0\u9c81\u68d2\u6027 [1]\u3002"
    )
    quote = (
        "HSI uses Hadamard basis patterns while FSI uses Fourier basis patterns. "
        "We compare imaging efficiency and noise robustness."
    )
    hits = [
        {
            "text": quote,
            "meta": {
                "source_name": "Hadamard versus Fourier single-pixel imaging.pdf",
                "heading_path": "Introduction",
                "page_start": 3,
                "citation_plan_evidence_quotes": [quote],
            },
        }
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1},
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert repaired == ""
    assert meta["renderer_rejected_citations"] == 1
    assert meta["dropped_unsupported_unplanned_claims"] == 1


def test_final_gate_accepts_later_same_source_plan_quote_for_compound_claim() -> None:
    first_quote = (
        "CLIP jointly trains an image encoder and a text encoder to predict the "
        "correct pairings in a batch."
    )
    exact_quote = (
        "The pre-training task predicts which caption goes with which image on a "
        "dataset of 400 million image text pairs collected from the internet."
    )
    answer = (
        "CLIP predicts which caption goes with which image using 400 million "
        "image-text pairs [1]."
    )
    hits = [
        {
            "text": f"{first_quote}\n{exact_quote}",
            "meta": {
                "source_name": "clip.pdf",
                "heading_path": "Abstract",
                "page_start": 1,
                "citation_plan_evidence_quotes": [first_quote, exact_quote],
            },
        }
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1},
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert "400 million" in repaired
    assert "[1]" in repaired
    assert meta["renderer_rejected_citations"] == 0


def test_repairs_each_uncited_claim_in_a_multi_sentence_line() -> None:
    answer = (
        "该方法使用真实 SPAD 噪声模型指导训练。"
        "它解决了光子受限场景中的低分辨率和高噪声问题。"
    )
    hits = [
        {
            "text": (
                "The real physical noise model of SPAD arrays guides network training "
                "for photon-limited inputs with low resolution and heavy noise."
            )
        },
        {"text": "Single-pixel imaging uses a bucket detector."},
    ]

    repaired, meta = audit_and_repair_claim_evidence(answer, hits)

    assert repaired.count("[1]") == 2
    assert meta["repaired_citations"] == 2
    assert meta["uncited_high_risk_claims"] == 0


def test_drops_empty_and_retrieval_placeholder_sections() -> None:
    answer = (
        "结论：SPAD 多源噪声模型支持重建 [1]。\n\n"
        "### 关键证据\n\n"
        "### 局限性\n"
        "当前检索到的证据未提及其他探测器，现有片段中未详细说明。\n\n"
        "### 阅读建议\n"
        "建议继续阅读方法部分。"
    )

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        [{"text": "The SPAD multi-source noise model supports reconstruction."}],
    )

    assert "### 关键证据" not in repaired
    assert "### 局限性" not in repaired
    assert "当前检索" not in repaired
    assert "### 阅读建议" in repaired
    assert meta["dropped_placeholder_sections"] == 2


def test_bibliographic_abbreviation_and_year_do_not_split_or_drop_claim() -> None:
    answer = (
        "该工作（Nat. Commun. 2023）收集了2790张真实SPAD图像，"
        "覆盖90个场景 [1]。然后使用物理模型生成训练数据 [1]。"
    )
    hits = [
        {
            "text": (
                "We collected 2790 real SPAD images over 90 scenes. "
                "The calibrated physical model generated the training data."
            )
        }
    ]

    repaired, meta = audit_and_repair_claim_evidence(answer, hits)

    assert "Nat. Commun. 2023" in repaired
    assert "2790张" in repaired
    assert "然后使用" in repaired
    assert meta["dropped_hard_mismatch_claims"] == 0


def test_rebinds_single_pixel_number_to_unique_spad_source() -> None:
    answer = (
        "该方法利用校准后的物理噪声模型，从低分辨率高噪声的 SPAD 数据中恢复高分辨率图像 [2]。"
    )
    hits = [
        {
            "text": (
                "The calibrated real-world physical noise model enables high-resolution "
                "SPAD single-photon reconstruction from low-resolution noisy inputs."
            ),
            "meta": {"source_name": "PIDL single-photon"},
        },
        {
            "text": "A transformer improves large-scale single-pixel imaging reconstruction.",
            "meta": {"source_name": "single-pixel transformer"},
        },
    ]

    repaired, meta = audit_and_repair_claim_evidence(answer, hits)

    assert "[1]" in repaired
    assert "[2]" not in repaired
    assert meta["rebound_citations"] == 1


def test_adds_unique_physical_noise_support_to_followup_sentence() -> None:
    answer = "这使网络学习到真实物理噪声的分布，而不是简单的合成噪声。"
    hits = [
        {
            "text": (
                "We established a real-world physical noise model of SPAD arrays for "
                "network training."
            ),
            "meta": {"source_name": "PIDL single-photon"},
        },
        {
            "text": "A detector review compares sensitivity and response speed.",
            "meta": {"source_name": "detector review"},
        },
    ]

    repaired, meta = audit_and_repair_claim_evidence(answer, hits)

    assert "[1]" in repaired
    assert meta["uncited_high_risk_claims"] == 0


def test_reading_questions_and_navigation_are_not_factual_claims() -> None:
    answer = (
        "这两篇文献分别从硬件探测器和算法重建两个维度解决问题，建议按此顺序搭配阅读。\n\n"
        "阅读目的：学习如何把物理噪声特性转化为可训练的数学模型。\n\n"
        "你可以思考：该物理建模方法是否可能推广到其他类型的单光子探测器？\n\n"
        "算法是否利用了综述中的波导集成？\n\n"
        "快速浏览综述：重点看 2.3 节和 4.2 节，了解探测器性能指标。"
    )

    audit = claim_evidence_audit(answer)

    assert audit["high_risk_claims"] == 0
    assert audit["uncited_high_risk_claims"] == 0


def test_anaphoric_continuation_inherits_previous_unique_citation() -> None:
    answer = (
        "该方法建立了真实硬件的多源物理噪声模型 [1]。"
        "这使得网络能够学习并补偿实际硬件退化，而不只处理理想化噪声。"
    )
    hits = [
        {
            "text": (
                "A real-world multi-source physical noise model of SPAD arrays is used "
                "to train the network and compensate hardware degradation."
            )
        }
    ]

    repaired, meta = audit_and_repair_claim_evidence(answer, hits)

    assert "实际硬件退化，而不只处理理想化噪声 [1]" in repaired
    assert meta["uncited_high_risk_claims"] == 0
    assert any(item.get("reason") == "anaphoric_continuation" for item in meta["repairs"])


def test_domain_specific_anaphoric_continuations_inherit_previous_citation() -> None:
    hits = [
        {
            "text": (
                "The calibrated model synthesizes a large dataset for super-resolution and bit-depth enhancement. "
                "A foveated strategy records fast-changing regions and accumulates slower regions over frames."
            )
        }
    ]

    for answer, expected in (
        (
            "论文先标定了真实物理噪声模型 [1]。基于该模型，论文合成大规模数据集并实现超分辨率。",
            "实现超分辨率 [1]",
        ),
        (
            "系统对中央凹区域进行动态高分辨率采样 [1]。其核心思想是对快速变化区域用高采样率记录、对缓慢变化区域跨帧累积并提升分辨率。",
            "提升分辨率 [1]",
        ),
    ):
        repaired, meta = audit_and_repair_claim_evidence(answer, hits)

        assert expected in repaired
        assert meta["uncited_high_risk_claims"] == 0
        assert any(item.get("reason") == "anaphoric_continuation" for item in meta["repairs"])


def test_mismatch_repair_keeps_anaphoric_sentence_with_previous_paper() -> None:
    answer = (
        "探测器综述列出了 Si-SPAD、PMT、SNSPD 和 TES [1]。"
        "它也说明制造复杂、成本高和低温条件限制了这些探测器的普及 [2]。"
    )
    hits = [
        {
            "text": (
                "The review covers Si-SPAD, PMT, SNSPD and TES. Their complex and costly "
                "manufacturing and special low-temperature conditions limit adoption."
            )
        },
        {"text": "Deep learning reconstructs noisy SPAD images."},
    ]

    repaired, meta = audit_and_repair_claim_evidence(answer, hits)

    assert "低温条件限制了这些探测器的普及 [1]" in repaired
    assert any(
        item.get("reason") == "anaphoric_continuation"
        for item in meta.get("rebound_repairs", [])
    )


def test_reading_section_heading_supplies_source_to_uncited_bullets_then_loses_own_marker() -> None:
    answer = (
        "### 1. 先读探测器综述（, [1]）\n\n"
        "- **搭配价值**：这篇综述说明制造复杂、成本高和低温条件限制了探测器普及。\n\n"
        "### 2. 再读物理噪声模型 [2]\n\n"
        "- **核心方法**：论文建立 SPAD 物理噪声模型并用 2790 张真实数据标定。"
    )
    hits = [
        {"text": "Complex manufacturing cost and low-temperature conditions limit detector adoption."},
        {"text": "A real SPAD physical noise model is calibrated with 2790 real images."},
    ]

    repaired, meta = audit_and_repair_claim_evidence(answer, hits)

    assert "限制了探测器普及 [1]" in repaired
    assert "2790 张真实数据标定 [2]" in repaired
    assert "### 1. 先读探测器综述" in repaired
    assert "### 1. 先读探测器综述（" not in repaired
    assert "### 2. 再读物理噪声模型 [2]" not in repaired
    assert meta["removed_heading_citations"] == 2


def test_reading_advice_and_scoped_missing_details_are_not_factual_citation_gaps() -> None:
    answer = (
        "这样在读第二篇时，你就能分清哪些噪声是硬件带来的。\n\n"
        "**读法建议**：重点关注 Fig. 1a 和 Fig. 1b。\n\n"
        "两篇文献均未提供具体的训练超参数或推理速度数据。"
    )

    repaired, meta = audit_and_repair_claim_evidence(answer, [])

    assert "当前引用证据未显示" in repaired
    assert meta["uncited_high_risk_claims"] == 0


def test_strict_comparison_synthesis_receives_both_planned_source_citations() -> None:
    answer = (
        "核心区别：前者在单个探测器上通过频率复用并行化空间调制，"
        "后者通过多个探测器并行化多方向照明下的信号采集。"
    )
    hits = [
        {
            "text": (
                "Each SLM pixel is modulated on p frequencies simultaneously and the "
                "multiplexed signal enters a single-pixel detector."
            )
        },
        {},
        {},
        {
            "text": (
                "Four spatially-separated single-pixel detectors sense reflected light "
                "under different illumination directions for photometric stereo."
            )
        },
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1, 4},
        drop_unsupported_unplanned_claims=True,
    )

    assert "[1] [4]" in repaired
    assert meta["uncited_high_risk_claims"] == 0


def test_partly_cited_comparison_sentence_receives_both_planned_sources() -> None:
    answer = (
        "\u5b83\u4eec\u5206\u522b\u51b3\u5b9a\u4e86\u7528\u4ec0\u4e48\u6a21\u5f0f\u53bb\u6d4b\u91cf\uff08Hadamard/Fourier \u7a7a\u95f4\u7f16\u7801\u57fa\uff09\uff0c"
        "\u4ee5\u53ca\u5982\u4f55\u7528 foveated supersampling \u52a8\u6001\u5206\u914d\u6d4b\u91cf\u8d44\u6e90 [1]\u3002"
    )
    hits = [
        {"text": "A foveated region tracks motion and allocates sampling over consecutive frames."},
        {"text": "HSI uses Hadamard basis patterns while FSI uses Fourier basis patterns."},
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1, 2},
    )

    assert "[1]" in repaired
    assert "[2]" in repaired
    assert any(
        item.get("reason") == "compound_comparison_synthesis"
        for item in meta.get("repairs", [])
    )
    assert any(
        item.get("reason") == "compound_comparison_synthesis"
        for item in meta.get("repairs", [])
    )


def test_strict_comparison_completes_a_partially_cited_two_source_claim() -> None:
    answer = (
        "核心区别：前者在单个探测器上通过频率复用并行化空间调制，"
        "后者通过多个探测器并行化多方向照明下的信号采集 [1]。"
    )
    hits = [
        {
            "text": (
                "Each SLM pixel is modulated on p frequencies simultaneously and the "
                "multiplexed signal enters a single-pixel detector."
            )
        },
        {},
        {},
        {
            "text": (
                "Four spatially-separated single-pixel detectors sense reflected light "
                "under different illumination directions for photometric stereo."
            )
        },
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1, 4},
        drop_unsupported_unplanned_claims=True,
    )

    assert "[1] [4]" in repaired
    assert meta["uncited_high_risk_claims"] == 0
    assert any(
        item.get("reason") == "compound_comparison_synthesis"
        for item in meta.get("repairs", [])
    )


def test_this_is_continuation_inherits_the_preceding_source_citation() -> None:
    answer = (
        "快速变化的区域用高分辨率快速记录，慢变区域通过多帧累积细节 [1]。"
        "这是多帧时序层面的采样策略，与单帧使用什么基函数无关。"
    )
    hits = [
        {
            "text": (
                "The strategy rapidly records quickly changing features while accumulating "
                "details of slowly evolving regions over consecutive frames."
            )
        }
    ]

    repaired, meta = audit_and_repair_claim_evidence(answer, hits)

    assert "这是多帧时序层面的采样策略，与单帧使用什么基函数无关 [1]。" in repaired
    assert meta["uncited_high_risk_claims"] == 0


def test_final_gate_keeps_supported_suffix_after_unsupported_contrast() -> None:
    answer = (
        "它不改变基函数本身，而是通过让高分辨率的中央凹区域跟踪场景中的运动，"
        "同时让每一帧从整个视场中传递新的空间信息，从而快速记录快速变化特征的细节，"
        "并跨多帧累积缓慢变化区域的细节 [1]。"
    )
    hits = [
        {
            "text": (
                "A high-resolution foveal region tracks motion in the scene while every "
                "frame delivers new spatial information from across the entire field of "
                "view. It rapidly records the detail of quickly changing features while "
                "accumulating detail of slowly evolving regions over consecutive frames."
            )
        }
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
    )

    assert "不改变基函数" not in repaired
    assert "整个视场" in repaired
    assert "跨多帧累积" in repaired
    assert "[1]" in repaired
    assert any(
        item.get("reason") == "supported_contrast_suffix"
        for item in meta.get("rebound_repairs", [])
    )


def test_trims_speculation_after_an_explicit_evidence_boundary() -> None:
    answer = (
        "当前检索到的内容未明确提及波导集成，"
        "但波导集成提高的探测效率可以直接转化为更高的有效光子计数 [2]。"
    )

    repaired, meta = audit_and_repair_claim_evidence(answer, [])

    assert repaired == "当前检索到的内容未明确提及波导集成。"
    assert meta["trimmed_unsupported_inferences"] == 1
    assert meta["uncited_high_risk_claims"] == 0


def test_strict_plan_rebinds_supported_claim_and_drops_unplanned_claim() -> None:
    answer = (
        "The foveated method uses changing pixel boundaries across consecutive frames [3]. "
        "SPH uses heterodyne holography [4]."
    )
    hits = [
        {"text": "The foveated method uses changing pixel boundaries from one frame to the next."},
        {"text": "HSI uses Hadamard basis patterns and FSI uses Fourier basis patterns."},
        {"text": "A generic single-pixel imaging review."},
        {"text": "SPH uses heterodyne holography."},
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1, 2},
        drop_unsupported_unplanned_claims=True,
    )

    assert "pixel boundaries across consecutive frames [1]" in repaired
    assert "heterodyne holography" not in repaired
    assert meta["removed_unplanned_citations"] == 2
    assert meta["dropped_unsupported_unplanned_claims"] == 1


def test_strict_plan_rebinds_hadamard_claim_from_foveated_to_basis_source() -> None:
    answer = (
        "\u9009\u62e9 Hadamard/Fourier \u662f\u5728\u51b3\u5b9a\u6210\u50cf\u7684\u6570\u5b66\u57fa\u7840\uff0c"
        "\u5373\u4f7f\u7528\u54ea\u4e00\u7ec4\u6b63\u4ea4\u57fa\u6765\u5206\u89e3\u548c\u91cd\u5efa\u56fe\u50cf [1]\u3002"
    )
    hits = [
        {
            "text": (
                "A high-resolution foveal region tracks motion while the system "
                "accumulates detail over consecutive frames."
            ),
            "meta": {"source_name": "Adaptive foveated single-pixel imaging.pdf"},
        },
        {
            "text": (
                "HSI uses Hadamard basis patterns for illumination while FSI uses "
                "Fourier basis patterns."
            ),
            "meta": {"source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf"},
        },
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1, 2},
    )

    assert "[2]" in repaired
    assert "[1]" not in repaired
    assert meta["rebound_citations"] == 1


def test_strict_plan_rebinds_scigs_claim_from_scinerf_to_scigs_source() -> None:
    answer = (
        "SCIGS uses 3DGS to reconstruct an explicit dynamic 3D scene from one compressed snapshot [1]."
    )
    hits = [
        {
            "text": (
                "SCINeRF recovers a 3D scene representation from a single temporal "
                "compressed image by incorporating the physical SCI process into NeRF training."
            ),
            "meta": {"source_name": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image.pdf"},
        },
        {
            "text": (
                "SCIGS is the first method to reconstruct an explicit 3D scene from a single "
                "compressed image and extends the reconstruction to dynamic 3D scenes."
            ),
            "meta": {"source_name": "SCIGS: 3D Gaussians Splatting from a Snapshot Compressive Image.pdf"},
        },
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1, 2},
    )

    assert "[2]" in repaired
    assert "[1]" not in repaired
    assert meta["rebound_citations"] == 1


def test_strict_plan_adds_source_to_uncited_numeric_markdown_table_row() -> None:
    answer = "\n".join(
        [
            "| Method | Dynamic-scene result |",
            "| --- | --- |",
            "| SCIGS vs SCINeRF | SSIM 0.9137 vs 0.7974 |",
        ]
    )
    hits = [
        {
            "text": "SCINeRF incorporates the physical SCI process into NeRF training.",
            "meta": {"source_name": "SCINeRF.pdf"},
        },
        {
            "text": (
                "On the dynamic dataset, SCIGS obtains SSIM 0.9137 while "
                "SCINeRF obtains SSIM 0.7974."
            ),
            "meta": {"source_name": "SCIGS.pdf"},
        },
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1, 2},
    )

    assert "SSIM 0.9137 vs 0.7974 [2] |" in repaired
    assert any(
        item.get("reason") == "markdown_table_fact"
        for item in meta.get("repairs", [])
    )


def test_restores_reported_3d_video_frame_rate_from_eligible_evidence() -> None:
    answer = (
        "四个空间分离的单像素探测器并行采集不同照明方向，"
        "从而实现连续实时的 3D 视频重建 [4]。"
    )
    hits = [
        {},
        {},
        {},
        {
            "text": (
                "Four spatially-separated single-pixel detectors reconstruct "
                "continuous real-time 3D video at approximately 8 frames per second."
            )
        },
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        prompt="这套 3D single-pixel video 为什么需要多个探测器并行采集？",
        allowed_citation_numbers={4},
    )

    assert "约为 8 帧/秒 [4]" in repaired
    assert meta["restored_evidence_numbers"] == 1


def test_restores_dmd_pattern_pair_and_frame_budget_as_one_relation() -> None:
    answer = (
        "每个测量值使用一对正负图案，因此 333 个测量需要 666 次切换 [1]。"
    )
    hits = [
        {
            "text": (
                "We chose to operate the DMD at 20 kHz. Since our detector senses "
                "only one output port, after each pattern we display the corresponding "
                "negative pattern, and take the difference of the two signals. Motivated "
                "to achieve 30 fps or 15 fps this allowed for a maximum of 333 or 666 "
                "patterns respectively."
            )
        }
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        prompt=(
            "为什么能做到 128×128、30 fps？20 kHz DMD、正负图案和 "
            "333 个测量之间是什么关系？"
        ),
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert "DMD 实际运行在 20 kHz [1]" in repaired
    assert "每帧测量上限分别为 333 和 666 组 [1]" in repaired
    assert meta["restored_evidence_numbers"] == 1


def test_restores_empty_part_based_step_only_from_complete_method_evidence() -> None:
    answer = (
        "按三个环节说明：\n\n"
        "1. Part-based 特征\n\n"
        "2. I_N(out) 与 I_N(real) 损失\n"
        "探测信号差用于训练 ILNet [1]。"
    )
    hits = [
        {
            "text": (
                "The ILNet first uses the part-based model to divide image features into "
                "different parts to facilitate fine-grained learning. The difference between "
                "I_N(out) and I_N(real) is used as a loss function."
            )
        }
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        prompt=(
            "ILNet 如何按 part-based 特征、I_N(out) 与 I_N(real) 损失形成自监督闭环？"
        ),
        allowed_citation_numbers={1},
    )

    assert "image features（图像特征）划分为 different parts（不同部分）" in repaired
    assert "更细粒度学习并改善重建细节 [1]" in repaired
    assert meta["restored_source_facts"] == 1


def test_restores_sequential_cs_full_name_from_source_definition() -> None:
    repaired, meta = audit_and_repair_claim_evidence(
        "SCS uses two stages [1].",
        [
            {
                "text": (
                    "Our procedure is referred to as Sequential Compressed Sensing "
                    "(SCS), and the algorithm consists of two stages."
                )
            }
        ],
        prompt="What does Sequential Compressed Sensing do in two stages?",
        allowed_citation_numbers={1},
    )

    assert repaired.startswith("Sequential Compressed Sensing（SCS）")
    assert meta["restored_source_facts"] == 1


def test_restores_sequential_cs_name_as_heading_from_cited_source_identity() -> None:
    repaired, meta = audit_and_repair_claim_evidence(
        "第一阶段逐步筛除零分量，第二阶段完成支持集恢复 [1]。",
        [
            {
                "text": "The algorithm consists of two stages.",
                "meta": {
                    "source_name": "Sequentially designed compressed sensing",
                    "citation_plan_evidence_quotes": [
                        "The algorithm consists of two stages."
                    ],
                },
            }
        ],
        prompt="Sequential Compressed Sensing 的两阶段过程分别做什么？",
        allowed_citation_numbers={1},
    )

    assert repaired.startswith("### Sequential Compressed Sensing\n\n")
    assert meta["restored_source_facts"] == 1


def test_restores_fdm_non_awg_optimum_from_complete_source_relation() -> None:
    evidence = (
        "Our FDM scheme increases acquisition speed without lowering integration time. "
        "If system noise is not AWG, there may exist a characteristic time for optimal "
        "SNR. FDM decreases acquisition time without deviation from such an optimal "
        "integration time."
    )
    repaired, meta = audit_and_repair_claim_evidence(
        "FDM is not fully equivalent to reducing integration time [1].",
        [{"text": evidence}],
        prompt="Why is FDM more useful for non-AWG noise?",
        allowed_citation_numbers={1},
    )

    assert "characteristic time may provide optimal SNR" in repaired
    assert "without moving away from that optimal integration time [1]" in repaired
    assert meta["restored_source_facts"] == 1


def test_restores_complete_sph_sampling_budget_from_one_source_block() -> None:
    evidence = (
        "Thus, the beat frequency of these two beams is 62,500 Hz. The signal was "
        "digitized with a sampling rate of 1.25 Ms/s. Considering the 48-μs refresh "
        "time, three beating cycles last for each Hadamard pattern and 20 data points "
        "were acquired within one cycle."
    )
    repaired, meta = audit_and_repair_claim_evidence(
        "The Nyquist sampling criterion must be followed [1].",
        [{"text": evidence}],
        prompt=(
            "How do 62.5 kHz, 1.25 Ms/s, and the 48 μs pattern period fit together?"
        ),
        allowed_citation_numbers={1},
    )

    assert repaired.startswith("The experiment uses a 62.5 kHz beat frequency")
    assert "20 data points per beat cycle" in repaired
    assert "three beating cycles per pattern [1]" in repaired
    assert meta["restored_source_facts"] == 1


def test_restores_both_sph_frequency_change_conditions_from_exact_source() -> None:
    evidence = (
        "Thus, the beat frequency of these two beams is 62,500 Hz and the signal was "
        "digitized at a sampling rate of 1.25 Ms/s. Considering the 48-μs refresh "
        "time, three beating cycles last for each Hadamard pattern and 20 data points "
        "were acquired within one cycle. Reconstruction quality is not sensitive to "
        "the beat frequency provided the Nyquist sampling criterion was followed. An "
        "integer number of beating cycles for each displayed pattern is also desired."
    )
    repaired, meta = audit_and_repair_claim_evidence(
        (
            "62.5 kHz and 1.25 Ms/s give 20 data points per cycle, while the 48 μs "
            "pattern contains three beating cycles [1]. Nyquist sampling is required [1]."
        ),
        [{"text": evidence}],
        prompt=(
            "How do 62.5 kHz, 1.25 Ms/s, and 48 μs fit together, and what two "
            "conditions preserve reconstruction quality when changing the beat frequency?"
        ),
        allowed_citation_numbers={1},
    )

    assert "integer number of beating cycles per displayed pattern [1]" in repaired
    assert meta["restored_source_facts"] == 1


def test_restored_sph_chinese_parameter_clauses_are_individually_cited() -> None:
    evidence = (
        "Thus, the beat frequency of these two beams is 62,500 Hz and the signal was "
        "digitized with a sampling rate of 1.25 Ms/s. Considering the 48-μs refresh "
        "time of the DMD and the 62,500-Hz beating frequency, three beating cycles "
        "last for each Hadamard pattern and 20 data points were acquired within one "
        "cycle. The Nyquist sampling criterion must be followed, and an integer number "
        "of beating cycles for each displayed pattern is desired."
    )
    repaired, meta = audit_and_repair_claim_evidence(
        "更换拍频要满足奈奎斯特采样准则和整数个拍频周期 [1]。",
        [{"text": evidence, "meta": {"citation_plan_evidence_quotes": [evidence]}}],
        prompt=(
            "SPH 实验里 62.5 kHz 拍频、1.25 Ms/s 采样和 DMD 的 48 μs 图案周期"
            "怎样配合？请说明更换拍频的两个条件。"
        ),
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert "DMD 图案周期 48 μs [1]。" in repaired
    assert "拍频 62,500 Hz" in repaired
    assert "62.5 kHz" not in repaired
    assert "每个图案包含 3 个拍频周期 [1]" in repaired
    assert not any("62.5 kHz" in claim for claim in meta.get("unresolved_claims", []))


def test_restores_three_d_video_per_pattern_sample_budget() -> None:
    evidence = (
        "The DAQ has a maximum acquisition rate of 250 kHz for all channels. As there "
        "are four channels employed, sampling rate for each channel is set to 62.5 kHz. "
        "Given that each pattern is displayed for 50 μs, there are approximately three "
        "samples acquired for each pattern."
    )
    repaired, meta = audit_and_repair_claim_evidence(
        "3D single-pixel video 的 DAQ 总采样率为 250 kHz，四路各为 62.5 kHz [1]。",
        [{"text": evidence, "meta": {"citation_plan_evidence_quotes": [evidence]}}],
        prompt=(
            "3D single-pixel video 的 DAQ 总采样率怎样分给四路探测器？"
            "在 50 μs 图案显示时间下，每个图案实际得到多少个样本？"
        ),
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert "每个图案显示 50 μs" in repaired
    assert "每个图案约采集 3 个样本" in repaired
    assert "approximately three samples acquired for each pattern" in repaired
    assert meta["restored_evidence_numbers"] >= 1


def test_strict_gate_keeps_perovskite_threshold_and_coupling_bundle() -> None:
    evidence = (
        "The dual-cavity device shows a minimum lasing threshold of 92 A cm$^{-2}$. "
        "The PeLED delivers directional emission into the single-crystal perovskite "
        "microcavity at a coupling efficiency of about 82.7% to establish lasing."
    )
    answer = "器件的最低激光阈值是 92 A cm⁻²，腔间耦合效率约为 82.7% [1]。"

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        [{"text": evidence, "meta": {"citation_plan_evidence_quotes": [evidence]}}],
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert repaired == answer
    assert meta["stripped_weak_citations"] == 0
    assert meta["minimum_ok"] is True


def test_strict_gate_keeps_source_exact_scinerf_motion_assumption() -> None:
    evidence = (
        "Since the compressed multi-view images are taken within a relatively-short "
        "exposure time, we assume that the camera trajectory during the imaging process "
        "is linear and obtain poses using linear interpolation. For more complex motions, "
        "we can exploit higher-order spline or directly optimize individual poses."
    )
    answer = (
        "During the compressed exposure, SCINeRF assumes that the camera trajectory "
        "is linear during the imaging process and obtains poses using linear interpolation "
        "[1]. For more complex motion, it suggests a higher-order spline or directly "
        "optimizing individual poses [1]."
    )

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        [{"text": evidence, "meta": {"citation_plan_evidence_quotes": [evidence]}}],
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert "camera trajectory is linear during the imaging process" in repaired
    assert "linear interpolation [1]" in repaired
    assert meta["dropped_hard_mismatch_claims"] == 0


def test_restores_sequential_cs_first_stage_step_count_from_main_result() -> None:
    evidence = (
        "The algorithm consists of two stages. The first stage involves "
        "$\\log_2 \\log n$ steps. The second stage uses $k \\log n$ additional "
        "measurements, and support is recovered at much lower SNRs."
    )
    repaired, meta = audit_and_repair_claim_evidence(
        "Sequential Compressed Sensing（SCS）分为两个阶段 [1]。",
        [{"text": evidence, "meta": {"citation_plan_evidence_quotes": [evidence]}}],
        prompt=(
            "Sequential Compressed Sensing 的两阶段过程分别做什么？"
            "请给出第一阶段步数。"
        ),
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert "first stage involves $\\log_2 \\log n$ steps [1]" in repaired
    assert meta["restored_source_facts"] >= 1


def test_restores_sequential_cs_second_stage_measurement_budget() -> None:
    evidence = (
        "The algorithm consists of two stages. The first stage leaves at most "
        "$n / \\log n + k$ components. In the second stage, we reliably remove "
        "all remaining zero components using $k \\log n$ additional measurements."
    )
    repaired, meta = audit_and_repair_claim_evidence(
        "第一阶段留下至多 $n / \\log n + k$ 个候选分量 [1]。",
        [{"text": evidence, "meta": {"citation_plan_evidence_quotes": [evidence]}}],
        prompt=(
            "Sequential Compressed Sensing 的第二阶段做什么？"
            "请给出第二阶段额外测量数 k log n。"
        ),
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert "第二阶段" in repaired
    assert "$k \\log n$ 次额外测量" in repaired
    assert "移除剩余零分量 [1]" in repaired
    assert meta["restored_source_facts"] >= 1


def test_restores_single_prompt_source_identifier_as_nonassertive_heading() -> None:
    evidence = (
        "Position and angular information are measured on separate cameras, so position "
        "resolution need not be sacrificed for angular resolution. The DOF is 2–5 times "
        "larger at 5 μm resolution."
    )
    repaired, meta = audit_and_repair_claim_evidence(
        "位置和角度信息由不同相机测量，因此不必牺牲两种分辨率 [1]。",
        [{"text": evidence, "meta": {"citation_plan_evidence_quotes": [evidence]}}],
        prompt="QCLFM 为什么能同时保住位置和角度分辨率？",
        allowed_citation_numbers={1},
    )

    assert repaired.startswith("### QCLFM\n\n")
    assert meta["restored_prompt_terms"] == 1


def test_restores_qclfm_separate_camera_source_wording() -> None:
    evidence = (
        "Since each degree of freedom can potentially be measured on separate cameras, "
        "one does not need to sacrifice position resolution for angular resolution or "
        "vice versa as in conventional LFM designs."
    )
    repaired, meta = audit_and_repair_claim_evidence(
        "QCLFM 把两个自由度放在两台相机上独立测量，因此不必做分辨率取舍 [1]。",
        [{"text": evidence, "meta": {"citation_plan_evidence_quotes": [evidence]}}],
        prompt="QCLFM 为什么能同时保住位置和角度分辨率？",
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert "不同相机（separate cameras）" in repaired
    assert "无需牺牲位置分辨率来换取角度分辨率 [1]" in repaired
    assert meta["restored_source_facts"] >= 1


def test_drops_distilled_energy_explanation_not_present_in_planned_evidence() -> None:
    answer = (
        "SCS removes half the zero components [1].\n"
        "Its distilled sensing concentrates sensing energy at likely signal locations [1]."
    )
    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        [
            {
                "text": (
                    "The procedure is based on distilled sensing and uses sparse sensing "
                    "matrices to remove half the zero components and identify irrelevant "
                    "signal components."
                )
            }
        ],
        prompt="Explain Sequential Compressed Sensing.",
        allowed_citation_numbers={1},
    )

    assert "removes half" in repaired
    assert "concentrates sensing energy" not in repaired
    assert meta["dropped_unsupported_inferences"] == 1


def test_multi_source_numeric_comparison_is_kept_only_after_union_coverage() -> None:
    answer = "SCIGS obtains 30.2 dB [1], while SCINeRF obtains 31.5 dB [2]."
    hits = [
        {
            "text": "SCIGS obtains 30.2 dB on the benchmark.",
            "meta": {"source_name": "SCIGS.pdf"},
        },
        {
            "text": "SCINeRF obtains 31.5 dB on the benchmark.",
            "meta": {"source_name": "SCINeRF.pdf"},
        },
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1, 2},
        drop_unsupported_unplanned_claims=True,
    )

    assert repaired == answer
    assert meta["citation_mismatch_claims"] == 0
    assert meta["dropped_hard_mismatch_claims"] == 0


def test_incomplete_multi_source_numeric_union_does_not_false_bind_cards() -> None:
    answer = "SCIGS obtains 30.2 dB [1], while SCINeRF obtains 31.5 dB [2]."
    hits = [
        {
            "text": "SCIGS obtains 30.2 dB on the benchmark.",
            "meta": {"source_name": "SCIGS.pdf"},
        },
        {
            "text": "SCINeRF is a related reconstruction method.",
            "meta": {"source_name": "SCINeRF.pdf"},
        },
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1, 2},
        drop_unsupported_unplanned_claims=True,
    )

    assert repaired == ""
    assert meta["dropped_hard_mismatch_claims"] == 1


def test_plain_comma_multi_source_comparison_uses_complete_union() -> None:
    answer = "Alpha reaches PSNR 30 dB [1]，Beta reaches PSNR 40 dB [2]."
    hits = [
        {
            "text": "Alpha reaches PSNR 30 dB on the benchmark.",
            "meta": {"source_name": "Alpha.pdf"},
        },
        {
            "text": "Beta reaches PSNR 40 dB on the benchmark.",
            "meta": {"source_name": "Beta.pdf"},
        },
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1, 2},
        drop_unsupported_unplanned_claims=True,
    )

    assert repaired == answer
    assert meta["citation_mismatch_claims"] == 0
    assert meta["dropped_hard_mismatch_claims"] == 0


def test_plain_comma_single_source_does_not_claim_unsupported_half() -> None:
    answer = "Alpha reaches PSNR 30 dB [1], Beta reaches PSNR 40 dB."
    hits = [
        {
            "text": "Alpha reaches PSNR 30 dB on the benchmark.",
            "meta": {"source_name": "Alpha.pdf"},
        }
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
    )

    assert repaired == ""
    assert meta["dropped_hard_mismatch_claims"] == 1


@pytest.mark.parametrize(
    ("answer", "evidence"),
    [
        ("The model reaches SSIM 0.91 [1].", "The model reaches LPIPS 0.91."),
        ("The reconstruction reaches PSNR 40 dB [1].", "The measured SNR is 40 dB."),
    ],
)
def test_runtime_rejects_same_quantity_with_incompatible_metric(
    answer: str,
    evidence: str,
) -> None:
    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        [{"text": evidence, "meta": {"source_name": "Benchmark.pdf"}}],
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
    )

    assert repaired == ""
    assert meta["dropped_hard_mismatch_claims"] == 1


def test_sidd_table_best_tie_is_cited_instead_of_dropped() -> None:
    answer = (
        "在《ECCV-2022-Simple Baselines for Image Restoration》的表 6 中，"
        "SIDD PSNR 的最高值为 40.30，由 Baseline ours 和 NAFNet ours 并列取得。"
    )
    evidence = (
        "Table 6. Simple Baselines for Image Restoration / 5 Experiments / "
        "5.2 Applications. Table 6. Image Denoising Results on SIDD [1]. "
        "SIDD PSNR: MPRNet [37] = 39.71; MIRNet [40] = 39.72; "
        "NBNet [6] = 39.75; UFormer [36] = 39.89; MAXIM [32] = 39.96; "
        "HINet [5] = 39.99; Restormer [39] = 40.02; "
        "Baseline ours = 40.30; NAFNet ours = 40.30"
    )
    hits = [
        {
            "text": evidence,
            "meta": {
                "source_name": "ECCV-2022-Simple Baselines for Image Restoration",
                "source_path": "simple-baselines.en.md",
                "heading_path": "5 Experiments / 5.2 Applications",
            },
        }
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
    )

    assert "Baseline ours" in repaired
    assert "NAFNet ours" in repaired
    assert "40.30" in repaired
    assert "[1]" in repaired
    assert meta["repaired_citations"] == 1
    assert meta["dropped_unsupported_unplanned_claims"] == 0


def test_cross_language_causal_mask_claim_is_repaired_and_kept_grounded() -> None:
    evidence = (
        "The Ag mask exhibits superior reflectivity and consequently lower "
        "transmissivity, resulting in better contrast for the complementary "
        "patterns than the Cr master plate. Therefore, the Ag mask was selected."
    )
    answer = (
        "最终选择银掩模而不是铬掩模，是因为银的反射率更高、透射率更低，"
        "从而为互补图案提供更好的对比度。"
    )
    hits = [
        {
            "text": evidence,
            "meta": {
                "source_name": "RT-SPI supplement",
                "source_path": "rt-spi-supp.en.md",
                "heading_path": "1. FABRICATION OF THE SPINNING MASK",
                "block_id": "blk-mask",
                "anchor_id": "p-mask",
                "page_start": 2,
                "citation_plan_evidence_quotes": [evidence],
            },
        }
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert repaired.endswith("[1]。")
    assert meta["repaired_citations"] == 1
    assert meta["renderer_rejected_citations"] == 0


def test_reader_visible_micro_unit_remains_grounded_against_latex_source() -> None:
    answer = (
        "The reported DOF is 2–5 times larger at 5 μm resolution [1]."
    )
    evidence = (
        r"This allowed a DOF between 2–5 times larger at the "
        r"$5\,\mu\mathrm{m}$ resolution."
    )

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        [{"text": evidence, "meta": {"source_name": "QCLFM.pdf"}}],
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert repaired == answer
    assert meta["dropped_hard_mismatch_claims"] == 0


def test_qclfm_tradeoff_claim_rebinds_from_title_neighbor_to_exact_mechanism() -> None:
    source_name = "Quantum correlation light-field microscope with extreme depth of field"
    limitation = (
        "A major limitation is the slow data acquisition speed of the event camera, "
        "which is limited by detection efficiency and timing resolution."
    )
    mechanism = (
        "The design uses the inherent position and angular/momentum correlation of "
        "entangled photon pairs. Since each degree of freedom can be measured on "
        "separate cameras, position resolution need not be sacrificed for angular resolution."
    )
    answer = (
        "QCLFM 利用位置和角度/动量关联，让两个自由度在不同相机上测量，"
        "从而不必牺牲位置或角度分辨率 [1]。"
    )
    hits = [
        {"text": limitation, "meta": {"source_name": source_name}},
        {"text": mechanism, "meta": {"source_name": source_name}},
    ]

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        allow_citation_repairs=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert repaired.endswith("[2]。")
    assert meta["rebound_citations"] == 1


def test_strict_plan_prefers_uniquely_stronger_qclfm_occurrence() -> None:
    broad = (
        "The design uses the inherent position and angular/momentum correlation of "
        "entangled photon pairs. Each degree of freedom can be measured on separate cameras."
    )
    exact = (
        "Type II SPDC generates signal and idler photon pairs correlated in time, position "
        "and momentum. The signal event camera captures position information, while a "
        "different idler event camera records momentum/angular information. Time correlation "
        "identifies photon pairs, so the signal photon's angular information is inferred from "
        "its time-correlated partner."
    )
    answer = (
        "QCLFM 利用 II 型 SPDC 纠缠光子对的位置与动量关联：信号光子的位置信息和"
        "闲频光子的动量信息分别由两台事件相机记录，再通过时间关联识别光子对并推断"
        "信号光子的角信息 [1]。"
    )

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        [
            {"text": broad, "meta": {"source_name": "QCLFM"}},
            {"text": exact, "meta": {"source_name": "QCLFM"}},
        ],
        allowed_citation_numbers={1, 2},
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert repaired.endswith("[2]。")
    assert any(
        item.get("reason") == "uniquely_stronger_evidence"
        for item in meta.get("rebound_repairs", [])
    )


def test_explicit_per_paper_sections_cannot_borrow_another_papers_citation() -> None:
    hits = [
        {
            "text": (
                "PatchTST uses channel-independence, where each channel is a single "
                "univariate time series that shares the same embedding and Transformer "
                "weights across all the series."
            ),
            "meta": {
                "source_name": "ICLR-A TIME SERIES IS WORTH 64 WORDS.pdf",
                "heading_path": "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers / Abstract",
            },
        },
        {
            "text": (
                "Autoformer is a decomposition architecture with an Auto-Correlation "
                "mechanism that discovers dependencies and aggregates similar sub-series."
            ),
            "meta": {
                "source_name": "arXiv-Autoformer-Decomposition.pdf",
                "heading_path": "Autoformer: Decomposition Transformers with Auto-Correlation / Abstract",
            },
        },
        {
            "text": (
                "iTransformer regards independent time series as the variate tokens "
                "and uses self-attention to capture multivariate correlations."
            ),
            "meta": {
                "source_name": "ICLR-ITRANSFORMER-INVERTED.pdf",
                "heading_path": "iTransformer: Inverted Transformers / Abstract",
            },
        },
        {
            "text": (
                "FEDformer combines Transformer with seasonal-trend decomposition. "
                "The decomposition method captures the global profile while Transformer "
                "captures more detailed structures."
            ),
            "meta": {
                "source_name": "FEDformer.pdf",
                "heading_path": "FEDformer: Frequency Enhanced Decomposed Transformer / Abstract",
            },
        },
        {
            "text": (
                "TimesNet uses TimesBlock to transform 1D time series into 2D tensors "
                "and model intraperiod and interperiod variations."
            ),
            "meta": {
                "source_name": "ICLR-TIMESNET-TEMPORAL-2D.pdf",
                "heading_path": "TimesNet: Temporal 2D-Variation Modeling / Abstract",
            },
        },
        {
            "text": (
                "Informer introduces ProbSparse self-attention, which achieves "
                "O(L log L) time complexity and memory usage for long sequence "
                "time-series forecasting."
            ),
            "meta": {
                "source_name": "arXiv-Informer-Beyond-Efficient.pdf",
                "heading_path": "Informer: Beyond Efficient Transformer / Abstract",
            },
        },
    ]
    answer = """## Informer

**Core modeling unit**: timestamp token in an encoder-decoder Transformer [3].

**Core modeling unit**: \u65f6\u95f4\u6233\u7ea7token\uff0c\u6bcf\u4e2atoken\u878d\u5408\u540c\u4e00\u65f6\u95f4\u6233\u7684\u591a\u4e2a\u53d8\u91cf [3].

**Long dependency mechanism**: Informer introduces ProbSparse self-attention with O(L log L) complexity.

## FEDformer

The decomposition method captures the global profile while Transformer captures detailed structures.

## PatchTST

**Core modeling unit**: channel-independent univariate series.

**Experiment task**: multivariate long-term forecasting.

**Limitation**: 论文未在检索到的摘要片段中明确陈述局限。

## iTransformer

**Core modeling unit**: independent time series as variate tokens [3].

## Cross-paper comparison

Informer and iTransformer both use temporal tokens, but iTransformer inverts them to variate tokens [3].

Informer uses ProbSparse self-attention for long dependencies [3]."""

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        prompt=(
            "Compare Informer, Autoformer, FEDformer, PatchTST, TimesNet and "
            "iTransformer core modeling unit, long dependency mechanism, "
            "experimental task, and limitation. Each paper must provide "
            "locatable evidence."
        ),
        allowed_citation_numbers={1, 2, 3, 4, 5, 6},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    informer_section = repaired.split("## FEDformer", 1)[0]
    assert "timestamp token" not in informer_section
    assert "\u65f6\u95f4\u6233\u7ea7token" not in informer_section
    assert "ProbSparse self-attention" in informer_section
    assert "[6]" in informer_section
    assert "global profile" in repaired and "[4]" in repaired
    assert "channel-independent" in repaired and "[1]" in repaired
    assert "multivariate long-term forecasting [1]" in repaired
    assert "未在检索到的摘要片段中明确陈述局限" in repaired
    assert "variate tokens [3]" in repaired
    assert "both use temporal tokens" not in repaired
    assert "Informer uses ProbSparse self-attention for long dependencies [6]" in repaired
    assert "The current evidence does not directly provide this facet" in repaired
    assert meta["added_requested_facet_boundaries"] > 0
    assert meta["section_source_rebound_citations"] >= 2
    assert meta["dropped_cross_source_claims"] == 3
    assert meta["minimum_ok"] is True


def test_two_paper_comparison_rebinds_named_source_and_drops_mixed_training_claim() -> None:
    hits = [
        {
            "text": (
                "SAM 2 uses a streaming memory that stores previous prompts and predictions "
                "across video frames. Its data engine uses the model in the loop."
            ),
            "meta": {
                "source_name": "sam2.pdf",
                "source_path": "sam2/sam2.en.md",
                "heading_path": "Figure 1",
            },
        },
        {
            "text": (
                "MedSAM follows the SAM architecture. The prompt encoder transforms "
                "user-drawn bounding boxes via positional encoding, and the mask decoder "
                "fuses image and prompt features using cross-attention."
            ),
            "meta": {
                "source_name": "medsam.pdf",
                "source_path": "medsam/medsam.en.md",
                "heading_path": "Fig. 2b",
            },
        },
    ]
    answer = """## MedSAM

MedSAM 的提示编码器通过位置编码表示边界框，并以交叉注意力融合图像与提示特征 [1]。

训练时，边界框会转换为二值掩码并与图像拼接作为模型输入 [1]。
"""

    repaired, meta = audit_and_repair_claim_evidence(
        answer,
        hits,
        prompt="对比 SAM 2 与 MedSAM，不要把二者的训练数据或提示机制混为一谈。",
        allowed_citation_numbers={1, 2},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert "提示编码器通过位置编码" in repaired
    assert "特征 [2]" in repaired
    assert "二值掩码" not in repaired
    assert meta["rebound_citations"] >= 1
    assert meta["dropped_cross_source_claims"] >= 1
