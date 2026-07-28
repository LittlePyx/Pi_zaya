from kb.claim_evidence_runtime import audit_and_repair_claim_evidence, claim_evidence_audit


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
