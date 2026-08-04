import pytest

import kb.evidence_binding as evidence_binding


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        (
            "SPAD inputs have low bit depth, low resolution, and heavy noise.",
            "photon-limited SPAD degradation",
        ),
        (
            "The system performs super-resolution with enhanced bit depth and imaging quality.",
            "super-resolution bit-depth outcome",
        ),
        (
            "Frequency channels are multiplexed for parallel acquisition.",
            "frequency-division multiplexing",
        ),
        (
            "Spatial resolution, SNR, and optical sectioning form a three-way tradeoff.",
            "s2ism tradeoff",
        ),
        (
            "A SPAD operates in Geiger mode above breakdown voltage and uses a quenching circuit.",
            "spad geiger quenching",
        ),
        (
            "SCINeRF models the physical imaging formation process.",
            "scinerf physical formation",
        ),
        (
            "单光子 SPAD 阵列受到低分辨率和严重噪声影响。",
            "photon-limited SPAD degradation",
        ),
    ],
)
def test_compound_domain_requirements_preserve_binding_terms(
    surface: str,
    expected: str,
) -> None:
    evidence_binding._system_a_domain_terms.cache_clear()

    assert expected in evidence_binding._system_a_domain_terms(surface)


def test_compound_domain_matching_skips_legacy_full_surface_lookaheads(
    monkeypatch,
) -> None:
    class _UnexpectedSearch:
        def search(self, _surface: str):
            raise AssertionError("legacy compound lookahead should not run")

    patched_patterns = tuple(
        (
            (name, _UnexpectedSearch())
            if name in evidence_binding._SYSTEM_A_COMPOUND_DOMAIN_NAMES
            else (name, pattern)
        )
        for name, pattern in evidence_binding._SYSTEM_A_DOMAIN_PATTERNS
    )
    monkeypatch.setattr(
        evidence_binding,
        "_SYSTEM_A_DOMAIN_PATTERNS",
        patched_patterns,
    )
    evidence_binding._system_a_domain_terms.cache_clear()

    terms = evidence_binding._system_a_domain_terms(
        "SPAD inputs have low bit depth, low resolution, and heavy noise."
    )

    assert "photon-limited SPAD degradation" in terms
    evidence_binding._system_a_domain_terms.cache_clear()


def test_review_claim_rejects_method_paper_evidence() -> None:
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim="先读综述了解探测器类型与性能瓶颈。",
        hit={"text": "We introduce deep learning into a SPAD imaging system."},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Abstract",
        evidence_quote="We introduce deep learning into a SPAD imaging system.",
        source_name="High-resolution single-photon imaging.pdf",
    )

    assert binding["status"] == "mismatch"
    assert binding["suppress_link"] is True
    assert binding["missing_terms"] == ["review identity"]


def test_authoritative_evidence_rejects_missing_claim_value() -> None:
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim="At 1% sampling, HSI reaches PSNR 30.2 dB and SSIM 0.91.",
        hit={"text": "At 1% sampling, HSI reaches PSNR 30.2 dB."},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Results",
        evidence_quote="At 1% sampling, HSI reaches PSNR 30.2 dB.",
        source_name="Hadamard single-pixel imaging.pdf",
    )

    assert binding["status"] == "mismatch"
    assert binding["suppress_link"] is True
    assert binding["missing_terms"] == ["0.91"]


@pytest.mark.parametrize(
    "claim",
    [
        "在表 6 中，Baseline 与 NAFNet 的 SIDD PSNR 都达到 40.30 dB。",
        "Table 6 shows that Baseline and NAFNet both reach 40.30 dB PSNR on SIDD.",
        "图 6 表明 Baseline 与 NAFNet 的 SIDD PSNR 都达到 40.30 dB。",
        "As shown in Figure 6, Baseline and NAFNet both reach 40.30 dB PSNR on SIDD.",
        "公式（6）给出的结果是 40.30 dB。",
        "Section 6 reports a result of 40.30 dB.",
    ],
)
def test_structure_locator_number_is_not_required_in_card_evidence(claim: str) -> None:
    evidence = (
        "SIDD PSNR: Baseline = 40.30 dB; NAFNet = 40.30 dB; "
        "Restormer = 40.02 dB."
    )
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim=claim,
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Results",
        evidence_quote=evidence,
        source_name="Simple Baselines for Image Restoration.pdf",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False


def test_structure_locator_exemption_does_not_hide_missing_metric_value() -> None:
    evidence = "SIDD PSNR: Baseline = 40.20 dB; NAFNet = 40.20 dB."
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim="表 6 显示 Baseline 与 NAFNet 的 SIDD PSNR 都达到 40.30 dB。",
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Results",
        evidence_quote=evidence,
        source_name="Simple Baselines for Image Restoration.pdf",
    )

    assert binding["status"] == "mismatch"
    assert binding["suppress_link"] is True
    assert binding["missing_terms"] == ["40.3 db"]


def test_authoritative_evidence_accepts_digit_and_number_word_equivalence() -> None:
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim="The system uses 4 detectors for parallel acquisition.",
        hit={"text": "The system uses four detectors for parallel acquisition."},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Method",
        evidence_quote="The system uses four detectors for parallel acquisition.",
        source_name="Real-time 3D single-pixel video.pdf",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False


def test_system_a_rejects_neighboring_paper_with_different_method_identity() -> None:
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim="SCIGS reconstructs an explicit dynamic 3D scene from one compressed image.",
        hit={
            "text": (
                "SCINeRF recovers a 3D scene from a single compressed image by incorporating "
                "the physical SCI process into NeRF training."
            )
        },
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Abstract",
        evidence_quote=(
            "SCINeRF recovers a 3D scene from a single compressed image by incorporating "
            "the physical SCI process into NeRF training."
        ),
        source_name="SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image.pdf",
    )

    assert binding["status"] == "mismatch"
    assert binding["suppress_link"] is True
    assert binding["missing_terms"] == ["method identity"]


def test_single_card_must_cover_every_quantity_in_comparison_claim() -> None:
    evidence = "SCIGS obtains 30.2 dB on the benchmark."
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim="SCIGS obtains 30.2 dB while SCINeRF obtains 31.5 dB.",
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Results",
        evidence_quote=evidence,
        source_name="SCIGS.pdf",
    )

    assert binding["status"] == "mismatch"
    assert binding["suppress_link"] is True
    assert binding["missing_terms"] == ["31.5 db"]


def test_verified_multi_card_union_can_bind_each_comparison_clause() -> None:
    claim = "SCIGS obtains 30.2 dB while SCINeRF obtains 31.5 dB."
    first = "SCIGS obtains 30.2 dB on the benchmark."
    second = "SCINeRF obtains 31.5 dB on the benchmark."
    common_meta = {
        "citation_plan_evidence_authoritative": True,
        "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        "citation_group_evidence_quotes": [first, second],
    }

    first_binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim=claim,
        hit={"text": first},
        meta=common_meta,
        heading="Results",
        evidence_quote=first,
        source_name="SCIGS.pdf",
    )
    second_binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim=claim,
        hit={"text": second},
        meta=common_meta,
        heading="Results",
        evidence_quote=second,
        source_name="SCINeRF.pdf",
    )

    assert first_binding["status"] == "grounded"
    assert second_binding["status"] == "grounded"


@pytest.mark.parametrize(
    ("claim", "evidence", "missing"),
    [
        ("The wavelength is 40 nm.", "The measured SNR is 40 dB.", "40 nm"),
        ("The reconstruction reaches 8 fps.", "The modulation rate is 8 Hz.", "8 fps"),
    ],
)
def test_same_number_with_incompatible_unit_is_rejected(
    claim: str,
    evidence: str,
    missing: str,
) -> None:
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim=claim,
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Results",
        evidence_quote=evidence,
        source_name="Benchmark.pdf",
    )

    assert binding["status"] == "mismatch"
    assert binding["suppress_link"] is True
    assert binding["missing_terms"] == [missing]


@pytest.mark.parametrize(
    ("claim", "evidence", "expected_missing"),
    [
        ("The wavelength is 2000 nm.", "The wavelength is 1550 nm.", "2000 nm"),
        ("The system reaches 2020 fps.", "The system reaches 2000 fps.", "2020 fps"),
        ("The image contains 2048 pixels.", "The image contains 1024 pixels.", "2048 pixel"),
    ],
)
def test_four_digit_quantity_with_explicit_unit_is_not_treated_as_year(
    claim: str,
    evidence: str,
    expected_missing: str,
) -> None:
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim=claim,
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Results",
        evidence_quote=evidence,
        source_name="Benchmark.pdf",
    )

    assert binding["status"] == "mismatch"
    assert binding["missing_terms"] == [expected_missing]


def test_unqualified_four_digit_publication_year_is_not_a_required_fact() -> None:
    evidence = "The method reports a result of 40 dB."
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim="Published in 2020, the method reports a result of 40 dB.",
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Results",
        evidence_quote=evidence,
        source_name="Benchmark.pdf",
    )

    assert binding["status"] == "grounded"


@pytest.mark.parametrize(
    "claim",
    [
        "Tables 5 and 6 report a result of 40.30 dB.",
        "Figures 2-4 report a result of 40.30 dB.",
        "Table six reports a result of 40.30 dB.",
    ],
)
def test_structure_locator_lists_ranges_and_number_words_are_excluded(claim: str) -> None:
    evidence = "The reported result is 40.30 dB."
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim=claim,
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Results",
        evidence_quote=evidence,
        source_name="Benchmark.pdf",
    )

    assert binding["status"] == "grounded"


@pytest.mark.parametrize(
    "claim",
    [
        "Alpha reaches PSNR 30 dB, Beta reaches PSNR 40 dB.",
        "Alpha 达到 PSNR 30 dB，Beta 达到 PSNR 40 dB。",
    ],
)
def test_verified_group_union_scopes_plain_comma_comparison(claim: str) -> None:
    first = "Alpha reaches PSNR 30 dB."
    second = "Beta reaches PSNR 40 dB."
    common_meta = {
        "citation_plan_evidence_authoritative": True,
        "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        "citation_group_evidence_quotes": [first, second],
    }

    bindings = [
        evidence_binding.assess_system_a_hit_binding(
            answer_claim=claim,
            hit={"text": evidence},
            meta=common_meta,
            heading="Results",
            evidence_quote=evidence,
            source_name="Benchmark.pdf",
        )
        for evidence in (first, second)
    ]

    assert [binding["status"] for binding in bindings] == ["grounded", "grounded"]


def test_plain_comma_scope_is_not_enabled_for_a_single_card() -> None:
    evidence = "Alpha reaches PSNR 30 dB."
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim="Alpha reaches PSNR 30 dB, Beta reaches PSNR 40 dB.",
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Results",
        evidence_quote=evidence,
        source_name="Alpha.pdf",
    )

    assert binding["status"] == "mismatch"
    assert binding["missing_terms"] == ["40 db"]


@pytest.mark.parametrize(
    ("claim", "evidence"),
    [
        ("The model reaches SSIM 0.91.", "The model reaches LPIPS 0.91."),
        ("The reconstruction reaches PSNR 40 dB.", "The measured SNR is 40 dB."),
    ],
)
def test_same_value_and_unit_with_different_metric_is_rejected(
    claim: str,
    evidence: str,
) -> None:
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim=claim,
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Results",
        evidence_quote=evidence,
        source_name="Benchmark.pdf",
    )

    assert binding["status"] == "mismatch"
    assert binding["suppress_link"] is True


def test_year_in_hyphenated_paper_title_is_not_inferred_as_image_count() -> None:
    evidence = "The paper reports SIDD PSNR of 40.30."
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim=(
            "ECCV-2022-Simple Baselines for Image Restoration reports "
            "SIDD PSNR of 40.30."
        ),
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Results",
        evidence_quote=evidence,
        source_name="ECCV-2022-Simple Baselines for Image Restoration.pdf",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False


def test_compact_table_metric_header_applies_to_following_method_cells() -> None:
    evidence = (
        "SIDD PSNR: Restormer = 40.02; Baseline ours = 40.30; "
        "NAFNet ours = 40.30"
    )
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim=(
            "SIDD PSNR reaches 40.30 for both Baseline ours and NAFNet ours."
        ),
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="5.2 Applications",
        evidence_quote=evidence,
        source_name="Simple Baselines for Image Restoration.pdf",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False


def test_psnr_table_value_without_repeated_unit_supports_db_claim() -> None:
    evidence = "SIDD PSNR: Restormer = 40.02; Baseline ours = 40.30."
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim="Restormer PSNR is 40.02 dB.",
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="5.2 Applications",
        evidence_quote=evidence,
        source_name="Simple Baselines for Image Restoration.pdf",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False


def test_generic_unitless_value_does_not_support_db_claim() -> None:
    evidence = "Restormer reports a benchmark value of 40.02."
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim="Restormer reports a benchmark value of 40.02 dB.",
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Results",
        evidence_quote=evidence,
        source_name="Benchmark.pdf",
    )

    assert binding["status"] == "mismatch"
    assert binding["suppress_link"] is True


def test_ssim_results_header_applies_across_comma_separated_cells() -> None:
    evidence = "SSIM results: A = 0.50, B = 0.60, Ours = 0.76."
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim="SSIM is 0.50 for A, 0.60 for B, and 0.76 for Ours.",
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Results",
        evidence_quote=evidence,
        source_name="Benchmark.pdf",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False


def test_number_does_not_borrow_count_unit_from_a_later_quantity() -> None:
    quantities = evidence_binding._system_a_fact_quantities(
        "The model reaches PSNR 30.5 after training on 100 images."
    )

    assert ("30.5", "db", "psnr") in quantities
    assert ("30.5", "image", "psnr") not in quantities
    assert ("100", "image", "") in quantities

    reversed_quantities = evidence_binding._system_a_fact_quantities(
        "The model is trained on 100 images and reaches PSNR 30.5 dB."
    )
    assert ("100", "image", "") in reversed_quantities
    assert ("100", "image", "psnr") not in reversed_quantities
    assert ("30.5", "db", "psnr") in reversed_quantities


def test_adjacent_postfix_metric_remains_attached_to_value() -> None:
    quantities = evidence_binding._system_a_fact_quantities(
        "The reported score is 0.9 SSIM."
    )

    assert ("0.9", "", "ssim") in quantities


def test_latex_micro_unit_matches_reader_visible_unicode_unit() -> None:
    latex_quantities = evidence_binding._system_a_fact_quantities(
        r"The reported resolution is $5\,\mu\mathrm{m}$."
    )
    reader_quantities = evidence_binding._system_a_fact_quantities(
        "The reported resolution is 5 μm."
    )

    assert ("5", "um", "") in latex_quantities
    assert latex_quantities == reader_quantities


def test_metric_inherits_across_explicit_numeric_range_bridge_only() -> None:
    claim = "PSNR improves from 30 dB to 31 dB."
    evidence = "PSNR is 30 dB while SNR is 31 dB."

    claim_quantities = evidence_binding._system_a_fact_quantities(claim)
    evidence_quantities = evidence_binding._system_a_fact_quantities(evidence)

    assert ("30", "db", "psnr") in claim_quantities
    assert ("31", "db", "psnr") in claim_quantities
    assert ("30", "db", "psnr") in evidence_quantities
    assert ("31", "db", "snr") in evidence_quantities

    for surface in (
        "PSNR values are 30 dB, 31 dB.",
        "PSNR improves from 30 to 31.",
    ):
        quantities = evidence_binding._system_a_fact_quantities(surface)
        assert ("30", "db", "psnr") in quantities
        assert ("31", "db", "psnr") in quantities

    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim=claim,
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Results",
        evidence_quote=evidence,
        source_name="Benchmark.pdf",
    )

    assert binding["status"] == "mismatch"
    assert binding["suppress_link"] is True


@pytest.mark.parametrize(
    "surface",
    [
        "PSNR is 30 dB and 31 images.",
        "PSNR: 30 dB / 31 images.",
    ],
)
def test_count_quantity_never_inherits_quality_metric(surface: str) -> None:
    quantities = evidence_binding._system_a_fact_quantities(surface)

    assert ("30", "db", "psnr") in quantities
    assert ("31", "image", "") in quantities
    assert ("31", "image", "psnr") not in quantities


def test_db_and_unitless_psnr_ranges_still_inherit_metric() -> None:
    db_quantities = evidence_binding._system_a_fact_quantities(
        "PSNR is 30 dB and 31 dB."
    )
    unitless_quantities = evidence_binding._system_a_fact_quantities(
        "PSNR ranges from 30 to 31."
    )

    assert {("30", "db", "psnr"), ("31", "db", "psnr")} <= db_quantities
    assert {("30", "db", "psnr"), ("31", "db", "psnr")} <= unitless_quantities


def test_multiplier_quantities_normalize_language_magnitude_and_direction() -> None:
    chinese = evidence_binding._system_a_fact_quantities("入射照明功率降低约 10 倍。")
    english = evidence_binding._system_a_fact_quantities(
        "Incident illumination power is tenfold lower."
    )
    opposite = evidence_binding._system_a_fact_quantities(
        "Incident illumination power is tenfold higher."
    )

    assert ("10", "fold", "decrease") in chinese
    assert ("10", "fold", "decrease") in english
    assert ("10", "fold", "increase") in opposite
    assert evidence_binding._quantity_is_covered(
        ("10", "fold", "decrease"), english
    )
    assert not evidence_binding._quantity_is_covered(
        ("10", "fold", "decrease"), opposite
    )


def test_multiplier_binding_accepts_cross_language_equivalence_but_not_units() -> None:
    evidence = (
        "iISM reaches about 120 nm lateral resolution at tenfold lower incident "
        "illumination power while significantly reducing photodamage."
    )
    grounded = evidence_binding.assess_system_a_hit_binding(
        answer_claim="在约 120 nm 横向分辨率下，入射照明功率降低约 10 倍，可显著减少光损伤。",
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Abstract",
        evidence_quote=evidence,
        source_name="iISM.pdf",
    )
    unit_conflict = evidence_binding.assess_system_a_hit_binding(
        answer_claim="The acquisition frequency is 10 Hz.",
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Abstract",
        evidence_quote=evidence,
        source_name="iISM.pdf",
    )

    assert grounded["status"] == "grounded"
    assert grounded["suppress_link"] is False
    assert unit_conflict["status"] == "mismatch"
    assert unit_conflict["suppress_link"] is True


def test_fact_quantities_normalize_chinese_word_multiplier() -> None:
    quantities = evidence_binding._system_a_fact_quantities(
        "缩小针孔可把横向分辨率提高到衍射极限的两倍。"
    )

    assert ("2", "fold", "increase") in quantities


def test_latex_sim_does_not_hide_reported_frame_rate() -> None:
    evidence = (
        "The system reconstructs continuous real-time 3D video at $\\sim$8 "
        "frames per second for image resolutions of $64 \\times 64$ pixels."
    )
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim="原文报告该三维视频系统的重建速度约为 8 帧/秒。",
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Abstract",
        evidence_quote=evidence,
        source_name="3D single-pixel video.pdf",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False


def test_authoritative_plan_does_not_bypass_missing_piln_identity() -> None:
    evidence = (
        "Single-pixel imaging combines compressive sensing with computational "
        "reconstruction to recover images from bucket-detector measurements."
    )
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim=(
            "PILN 将物理成像模型作为网络迭代指导，并通过可学习模块完成重建。"
        ),
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Acquisition and image reconstruction strategies",
        evidence_quote=evidence,
        source_name="Principles and prospects for single-pixel imaging.pdf",
    )

    assert binding["status"] == "mismatch"
    assert binding["suppress_link"] is True
    assert "piln" in binding["missing_terms"]


def test_physical_noise_evidence_does_not_support_unstated_black_box_or_robustness_claims() -> None:
    evidence = (
        "We established a real-world physical noise model of SPAD arrays and calibrated "
        "it with real-shot images. The calibrated model was used to synthesize image "
        "pairs for network training."
    )
    claims = (
        "该方法用物理噪声模型替代纯数据驱动的黑箱学习。",
        "该方法在训练数据有限或场景变化时仍能保持鲁棒性。",
        "传统方法失效时，该网络仍能恢复清晰图像。",
        "该网络从物理噪声中解耦出真实信号。",
    )

    for claim in claims:
        binding = evidence_binding.assess_system_a_hit_binding(
            answer_claim=claim,
            hit={"text": evidence},
            meta={
                "citation_plan_evidence_authoritative": True,
                "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
            },
            heading="Introduction",
            evidence_quote=evidence,
            source_name="Physics-informed SPAD imaging.pdf",
        )

        assert binding["status"] == "mismatch"
        assert binding["suppress_link"] is True
        assert binding["missing_terms"] == ["explicit relation"]


def test_authoritative_full_plan_evidence_is_distinct_from_short_locator_snippet() -> None:
    locator_snippet = "The abstract introduces the overall network architecture."
    full_plan_evidence = (
        "1D signals collected by the single-pixel detector are used as labels "
        "for adaptively optimizing and reconstructing the image."
    )
    claim = (
        "单像素探测器采集的 1D 信号作为监督标签，"
        "用于自适应优化和图像重建。"
    )
    common = {
        "answer_claim": claim,
        "hit": {"text": locator_snippet},
        "heading": "Abstract",
        "evidence_quote": locator_snippet,
        "source_name": "Method paper.pdf",
    }
    binding = evidence_binding.assess_system_a_hit_binding(
        hit={"text": locator_snippet},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
            "citation_plan_full_evidence_quote": full_plan_evidence,
        },
        answer_claim=claim,
        heading=common["heading"],
        evidence_quote=common["evidence_quote"],
        source_name=common["source_name"],
    )
    ignored_untrusted_full_plan = evidence_binding.assess_system_a_hit_binding(
        **common,
        meta={
            "citation_plan_evidence_authoritative": False,
            "citation_plan_full_evidence_quote": full_plan_evidence,
        },
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False
    assert ignored_untrusted_full_plan["suppress_link"] is True


def test_cassi_prompt_contract_ignores_stale_bibliography_acronyms() -> None:
    evidence = (
        "The primary features of the system design are two dispersive elements, "
        "arranged in opposition and surrounding a binary-valued aperture code."
    )
    claim = (
        "CASSI（编码孔径快照光谱成像）由两个相向布置的色散元件"
        "围绕一个二值编码孔径组成。"
    )

    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim=claim,
        hit={"text": "LWIR coded aperture spectrometer. Proceedings of SPIE."},
        meta={
            "canonical_answer_evidence": True,
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_contract_block",
            "citation_plan_full_evidence_quote": evidence,
        },
        heading="Abstract",
        evidence_quote=evidence,
        source_name=(
            "Single-shot compressive spectral imaging with a dual-disperser architecture.pdf"
        ),
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False


def test_english_application_gloss_is_not_treated_as_another_paper_title() -> None:
    evidence = (
        "As the approach suits a wide variety of detector technologies, images can be "
        "collected at wavelengths outside the reach of FPA technology or at high frame "
        "rates or in three dimensions. Promising applications include the visualization "
        "of hazardous gas leaks and 3D situation awareness for autonomous vehicles."
    )
    claim = (
        "代表性应用包括危险气体泄漏的可视化和自动驾驶的三维态势感知"
        "（3D situation awareness for autonomous vehicles）。"
    )

    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim=claim,
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
            "citation_plan_full_evidence_quote": evidence,
        },
        heading="Abstract",
        evidence_quote=evidence,
        source_name="Principles and prospects for single-pixel imaging.pdf",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False
    assert "适用场景" in binding["reason"]


def test_spad_physics_informed_binding_explains_training_mechanism() -> None:
    evidence = (
        "We first established a real-world physical noise model of SPAD arrays and "
        "calibrated it with a real-shot SPAD image dataset. With the calibrated physical "
        "noise model, we synthesized a realistic single-photon image dataset containing "
        "image pairs for network training."
    )
    claim = (
        "physics-informed deep learning 用真实物理噪声模型生成训练数据，"
        "让网络学习真实 SPAD 成像退化。"
    )

    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim=claim,
        hit={"text": evidence},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
            "citation_plan_full_evidence_quote": evidence,
        },
        heading="Introduction",
        evidence_quote=evidence,
        source_name="High-resolution single-photon imaging with physics-informed deep learning.pdf",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False
    assert "标定真实物理噪声模型" in binding["reason"]
    assert "训练数据" in binding["reason"]
