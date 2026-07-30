from kb.evidence_term_mapping import evidence_alignment_tokens, method_identity_conflicts


def test_chinese_3d_video_mechanism_aligns_with_english_source_terms() -> None:
    answer_tokens = evidence_alignment_tokens(
        "四个空间分离的单像素探测器用于光度立体，速度约为 8 帧/秒。"
    )
    evidence_tokens = evidence_alignment_tokens(
        "Four spatially-separated single-pixel detectors perform photometric stereo "
        "at approximately 8 frames per second."
    )

    assert {
        "four",
        "spatially",
        "separated",
        "detectors",
        "photometric",
        "stereo",
        "frames",
        "second",
    } <= answer_tokens & evidence_tokens


def test_fdm_chinese_question_aligns_with_english_mechanism_terms() -> None:
    question_tokens = evidence_alignment_tokens(
        "频分复用为什么能并行采集，又如何在不改变探测器积分时间时权衡信噪比和采集速度？"
    )
    evidence_tokens = evidence_alignment_tokens(
        "Frequency-division multiplexing can parallelize acquisition and offers a trade-off "
        "between signal-to-noise ratio and acquisition speed without altering detector integration time."
    )

    assert {
        "frequency",
        "multiplexed",
        "parallelize",
        "signal",
        "noise",
        "speed",
        "detector",
        "integration",
        "time",
    } <= question_tokens
    assert len(question_tokens & evidence_tokens) >= 8


def test_iism_chinese_question_aligns_with_english_benefit_terms() -> None:
    question_tokens = evidence_alignment_tokens(
        "iISM 对活细胞成像有什么好处，是否能把照明功率降低约十倍并减少光损伤？"
    )
    evidence_tokens = evidence_alignment_tokens(
        "iISM delivers optical sectioning at tenfold lower incident illumination power, "
        "thereby reducing photodamage in live-cell imaging."
    )

    assert {"tenfold", "lower", "illumination", "photodamage"} <= (
        question_tokens & evidence_tokens
    )


def test_scigs_question_prefers_the_variant_claim_in_the_abstract() -> None:
    question_tokens = evidence_alignment_tokens(
        "SCIGS 的核心新意是 3DGS 本身吗？它对单张压缩图和动态场景声称了什么？"
    )
    evidence_tokens = evidence_alignment_tokens(
        "We propose SCIGS, a variant of 3DGS, as the first method to reconstruct "
        "an explicit dynamic 3D scene from a single compressed image."
    )

    assert {"variant", "single", "compressed", "dynamic", "3d"} <= (
        question_tokens & evidence_tokens
    )


def test_method_identity_conflict_distinguishes_neighboring_3d_reconstruction_papers() -> None:
    assert method_identity_conflicts(
        "SCIGS reconstructs a dynamic 3D scene from one compressed image.",
        "SCINeRF incorporates the physical SCI process into NeRF training.",
    )
    assert not method_identity_conflicts(
        "SCIGS differs from SCINeRF in its scene representation.",
        "SCINeRF incorporates the physical SCI process into NeRF training.",
    )
    assert not method_identity_conflicts(
        "PILN is self-supervised.",
        "ILNet uses a self-supervised image loop.",
    )


def test_iism_acronym_matches_its_spelled_out_method_name() -> None:
    assert not method_identity_conflicts(
        "iISM reaches about 120 nm lateral resolution.",
        "Interferometric Image Scanning Microscopy achieves about 120 nm lateral resolution.",
    )
    assert not method_identity_conflicts(
        "iISM reduces the incident illumination power by about tenfold.",
        "LSA-2026-Interferometric Image Scanning Microscopy operates at tenfold lower incident illumination power.",
    )


def test_method_identity_conflict_ignores_bibliographic_filename_prefixes() -> None:
    assert not method_identity_conflicts(
        "CASSI uses two opposing dispersive elements and a binary-valued aperture.",
        (
            "The primary features are two dispersive elements surrounding a binary-valued "
            "aperture. OE-2007-Single-shot compressive spectral imaging.pdf"
        ),
    )


def test_cnr_metric_is_not_a_method_conflict_with_lsa_filename() -> None:
    assert not method_identity_conflicts(
        "CNR improves under the reported acquisition setting.",
        (
            "The contrast-to-noise ratio improves under the reported acquisition "
            "setting. LSA-2025-Comprehensive compensation.pdf"
        ),
    )

    assert method_identity_conflicts(
        "SCIGS reconstructs a dynamic 3D scene.",
        "SCINeRF reconstructs a neural radiance field.",
    )


def test_foveated_question_aligns_with_not_simple_zoom_claim() -> None:
    question_tokens = evidence_alignment_tokens(
        "动态超采样是不是只对运动区域做局部放大？它和普通 zoom 的关键区别是什么？"
    )
    evidence_tokens = evidence_alignment_tokens(
        "Unlike a simple zoom, every frame delivers new spatial information "
        "from across the entire field of view while a high-resolution foveal region tracks motion."
    )

    assert {"unlike", "simple", "zoom", "every", "frame", "foveal"} <= (
        question_tokens & evidence_tokens
    )


def test_foveated_fast_and_slow_detail_claim_aligns_with_english_source() -> None:
    answer_tokens = evidence_alignment_tokens(
        "\u5feb\u901f\u53d8\u5316\u7684\u7279\u5f81\u88ab\u5feb\u901f\u8bb0\u5f55\uff0c"
        "\u800c\u7f13\u6162\u6f14\u53d8\u7684\u533a\u57df\u901a\u8fc7\u591a\u5e27\u7d2f\u79ef\u7ec6\u8282\u3002"
    )
    evidence_tokens = evidence_alignment_tokens(
        "This strategy rapidly records the detail of quickly changing features while "
        "simultaneously accumulating detail of more slowly evolving regions over several "
        "consecutive frames."
    )

    assert {
        "rapidly",
        "records",
        "changing",
        "features",
        "accumulating",
        "detail",
        "slowly",
        "evolving",
        "consecutive",
        "frames",
    } <= (answer_tokens & evidence_tokens)


def test_imaging_efficiency_and_noise_robustness_align_with_english_source() -> None:
    answer_tokens = evidence_alignment_tokens(
        "\u6210\u50cf\u6548\u7387\u548c\u566a\u58f0\u9c81\u68d2\u6027"
    )
    evidence_tokens = evidence_alignment_tokens(
        "We compare the methods in terms of imaging efficiency and noise robustness."
    )

    assert {"imaging", "efficiency", "noise", "robustness"} <= (
        answer_tokens & evidence_tokens
    )


def test_detector_manufacturing_challenges_align_with_english_source() -> None:
    answer_tokens = evidence_alignment_tokens(
        "\u786c\u4ef6\u5236\u9020\u590d\u6742\u3001\u6210\u672c\u9ad8\uff0c\u4e14\u9700\u8981\u4f4e\u6e29\u7b49\u7279\u6b8a\u5de5\u4f5c\u6761\u4ef6"
    )
    evidence_tokens = evidence_alignment_tokens(
        "The complexity and high manufacturing cost, coupled with special conditions "
        "such as a low-temperature environment, limit adoption."
    )

    assert {
        "complexity",
        "manufacturing",
        "high",
        "cost",
        "low",
        "temperature",
        "special",
        "conditions",
    } <= (answer_tokens & evidence_tokens)


def test_transposed_pnsr_table_header_aligns_with_psnr_claim() -> None:
    assert "psnr" in evidence_alignment_tokens(
        "Table 2: PNSR (dB), Hadamard = 8.01; Fourier = 8.08; SSIM = 11.1."
    )


def test_hsi_fsi_acronyms_match_spelled_out_single_pixel_basis_names() -> None:
    assert not method_identity_conflicts(
        "FSI has better reconstruction quality than HSI under undersampling.",
        (
            "Hadamard single-pixel imaging versus Fourier single-pixel imaging. "
            "Table 2 reports PNSR and SSIM for both bases."
        ),
    )


def test_pnsr_typo_is_a_metric_not_a_method_identity() -> None:
    assert not method_identity_conflicts(
        "HSI and FSI are compared under undersampling.",
        "The same HSI/FSI comparison reports PNSR values in Table 2.",
    )


def test_spi_application_claims_align_with_abstract_terms() -> None:
    answer_tokens = evidence_alignment_tokens(
        "\u5371\u9669\u6c14\u4f53\u6cc4\u6f0f\u53ef\u89c6\u5316\uff0c\u4ee5\u53ca\u81ea\u52a8\u9a7e\u9a76\u7684\u4e09\u7ef4\u6001\u52bf\u611f\u77e5\uff0c"
        "\u9700\u8981\u8986\u76d6\u9762\u9635\u63a2\u6d4b\u5668\u4e4b\u5916\u7684\u6ce2\u957f\u548c\u9ad8\u5e27\u7387\u3002"
    )
    evidence_tokens = evidence_alignment_tokens(
        "Images can be collected at wavelengths outside the reach of FPA technology "
        "or at high frame rates or in three dimensions. Applications include hazardous "
        "gas leaks and 3D situation awareness for autonomous vehicles."
    )

    assert {
        "wavelengths",
        "fpa",
        "high",
        "frame",
        "rates",
        "hazardous",
        "gas",
        "leaks",
        "autonomous",
        "vehicles",
    } <= (answer_tokens & evidence_tokens)


def test_detector_review_guide_aligns_with_english_abstract() -> None:
    answer_tokens = evidence_alignment_tokens(
        "这篇综述介绍各类探测器的物理原理、制造难度和适用场景。"
    )
    evidence_tokens = evidence_alignment_tokens(
        "This review summarizes detector principles, manufacturing complexity, "
        "technical challenges, and barriers to wider adoption."
    )

    assert {
        "detector",
        "principles",
        "manufacturing",
        "complexity",
        "adoption",
    }.issubset(answer_tokens & evidence_tokens)


def test_real_degradation_chain_terms_align_with_english_source() -> None:
    claim = (
        "投影端散射和非理想聚焦使照明图案模糊；空间下采样之后，机械抖动造成"
        "相对错位和乘性波动；探测路径的散射缺陷再引入模糊，并叠加光子散粒噪声"
        "与电子噪声。整个场景光强的积分会使读出噪声传播到整幅图像。"
    )
    tokens = evidence_alignment_tokens(claim)

    assert {
        "illumination",
        "scattering",
        "downsampling",
        "jitters",
        "misalignment",
        "multiplicative",
        "detection",
        "photon",
        "electronic",
        "integrates",
        "propagate",
        "entire",
        "image",
    } <= tokens


def test_piln_measurement_label_claim_aligns_with_english_source_sentence() -> None:
    answer_tokens = evidence_alignment_tokens(
        "无需真实图像标签：ILNet 不需要成对的高质量图像作为训练标签，"
        "而是利用物理采集的 1D 信号作为监督信号。"
    )
    evidence_tokens = evidence_alignment_tokens(
        "1D signals collected by the single-pixel detector are used as labels for "
        "adaptively optimizing and reconstructing the image."
    )

    assert {
        "signals",
        "collected",
        "single",
        "pixel",
        "detector",
        "used",
        "labels",
        "adaptively",
        "optimizing",
        "reconstructing",
    } <= (answer_tokens & evidence_tokens)
