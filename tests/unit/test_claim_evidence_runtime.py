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
