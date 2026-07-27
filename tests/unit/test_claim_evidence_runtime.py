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
