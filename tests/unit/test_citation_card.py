from __future__ import annotations

from kb.citation_card import compose_citation_card


def test_explicit_citation_route_wins_over_legacy_inpaper_flag() -> None:
    route_only_system_b = compose_citation_card(
        {
            "citation_route": "system_b",
            "is_inpaper": False,
            "title": "Upstream reconstruction method",
            "citation_context": "The current paper cites this method as prior work.",
            "raw": "[7] Upstream reconstruction method.",
        }
    )
    explicit_system_a = compose_citation_card(
        {
            "citation_route": "system_a",
            "is_inpaper": True,
            "source_name": "Current paper.pdf",
            "evidence_quote": "The current paper directly reports this result.",
        }
    )

    assert route_only_system_b["card_kind"] == "upstream_reference"
    assert route_only_system_b["card_view"]["route"] == "system_b"
    assert explicit_system_a["card_kind"] == "answer_evidence"
    assert explicit_system_a["card_view"]["route"] == "system_a"


def test_system_a_card_composer_builds_quality_fields() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "SCIGS.pdf",
            "heading_path": "Abstract",
            "answer_claim": "SCIGS is a 3D Gaussian Splatting variant for SCI.",
            "evidence_quote": "SCIGS is a variant of 3DGS for snapshot compressive imaging.",
            "location_label": "SCIGS / Abstract · p. 1",
            "support_relation": "答案句和原文都说明 SCIGS 面向 SCI 的 3DGS 变体。",
            "binding_status": "grounded",
            "binding_confidence": 0.86,
        }
    )

    assert detail["card_kind"] == "answer_evidence"
    assert detail["card_title"] == "SCIGS.pdf"
    assert detail["card_subtitle"] == "Abstract · p. 1"
    assert detail["card_locator"] == "Abstract · p. 1"
    assert detail["card_evidence"].startswith("SCIGS is a variant")
    assert detail["card_takeaway"] == ""
    assert detail["card_support_explanation"] == ""
    assert detail["card_flow"] == []
    assert detail["card_quality_label"] == "证据匹配"
    assert detail["card_quality_score"] >= 0.8
    assert detail["card_warning"] == ""


def test_system_a_card_describes_training_and_generalization_risk_specifically() -> None:
    detail = compose_citation_card(
        {
            "citation_route": "system_a",
            "source_name": "DL-SPI review.pdf",
            "heading_path": "4. Strategy and Advantages / Data-Driven Strategy",
            "answer_claim": (
                "数据驱动策略存在 prolonged training duration（训练时间长）和 "
                "limited generalization（泛化能力有限）。"
            ),
            "evidence_quote": (
                "Data-driven strategies have prolonged training duration and limited "
                "generalization when adapting to diverse imaging scenes."
            ),
            "render_locale": "zh",
            "binding_status": "grounded",
            "binding_confidence": 0.9,
        }
    )

    assert detail["card_support_explanation"] == (
        "原文明确把训练时间长和泛化能力有限列为数据驱动策略的局限。"
    )


def test_system_a_card_centers_long_evidence_on_claim_aligned_sentence() -> None:
    target_sentence = (
        "Specifically, we formulate the physical imaging process of SCI as part "
        "of the training of NeRF, allowing us to exploit its impressive performance "
        "in capturing complex scene structures."
    )
    long_abstract = (
        "In this paper, we explore the potential of Snapshot Compressive Imaging (SCI) "
        "technique for recovering the underlying 3D scene representation from a single "
        "temporal compressed image. "
        "SCI is a cost-effective method that enables the recording of high-dimensional "
        "data into a single image using low-cost 2D imaging sensors. "
        "To achieve this, a series of specially designed 2D masks are usually employed, "
        "which reduces storage requirements and offers potential privacy protection. "
        "Inspired by this, our approach builds upon the powerful 3D scene representation "
        "capabilities of neural radiance fields (NeRF). "
        f"{target_sentence} "
        "To assess the method, we conduct extensive evaluations using synthetic and real data."
    )

    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "SCINeRF.pdf",
            "heading_path": "Abstract",
            "answer_claim": (
                "\u5b83\u5c06 SCI \u7684\u7269\u7406\u6210\u50cf\u8fc7\u7a0b\u516c\u5f0f\u5316\u4e3a NeRF \u8bad\u7ec3\u7684\u4e00\u90e8\u5206\uff0c"
                "\u5229\u7528 NeRF \u6355\u6349\u590d\u6742\u573a\u666f\u7ed3\u6784\u3002"
            ),
            "evidence_quote": long_abstract,
            "location_label": "Abstract / p. 1",
            "selection_reason": "prompt_aligned_source_sentence",
            "strict_locate": True,
            "page_start": 1,
            "binding_status": "grounded",
            "binding_confidence": 0.95,
        }
    )

    assert target_sentence in detail["card_evidence"]
    assert not detail["card_evidence"].startswith("In this paper")
    assert not detail["card_evidence"].endswith("...")


def test_system_a_card_preserves_strict_prompt_contract_evidence() -> None:
    evidence = (
        "Single photon avalanche diode (SPAD) operates in Geiger mode. "
        "… Its bias voltage is higher than the reverse bias breakdown voltage. "
        "… The avalanche diode must be supported by a quenching circuit."
    )
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "SPAD.pdf",
            "heading_path": "Principle of SPAD",
            "answer_claim": "SPAD 的雪崩发生后需要淬灭电路。",
            "evidence_quote": evidence,
            "location_label": "Principle of SPAD · p. 2",
            "selection_reason": "prompt_contract_block",
            "strict_locate": True,
            "page_start": 2,
            "binding_status": "grounded",
            "binding_confidence": 0.9,
        }
    )

    assert detail["card_evidence"] == evidence
    assert "Geiger mode" in detail["card_evidence"]
    assert "breakdown voltage" in detail["card_evidence"]
    assert "quenching circuit" in detail["card_evidence"]


def test_system_a_card_preserves_strict_lineage_evidence() -> None:
    evidence = (
        "In this paper, we explore Snapshot Compressive Imaging for recovering the "
        "underlying 3D scene representation from a single temporal compressed image. … "
        "Specifically, we formulate the physical imaging process of SCI as part of the "
        "training of NeRF."
    )
    detail = compose_citation_card(
        {
            "citation_route": "system_a",
            "source_name": "SCINeRF.pdf",
            "heading_path": "Abstract",
            "answer_claim": "SCINeRF 把 SCI 物理成像过程嵌入 NeRF 训练。",
            "evidence_quote": evidence,
            "location_label": "Abstract · p. 1",
            "selection_reason": "lineage_exact_source_block",
            "strict_locate": True,
            "page_start": 1,
            "binding_status": "grounded",
            "binding_confidence": 0.9,
        }
    )

    assert detail["card_evidence"] == evidence
    assert "evidence_quote_filtered" not in detail["card_quality_flags"]
    assert "missing_evidence_quote" not in detail["card_quality_flags"]


def test_system_a_card_preserves_verified_structured_metric_evidence() -> None:
    evidence = (
        "Table 6. SIDD PSNR: Baseline ours = 40.30; NAFNet ours = 40.30; "
        "Restormer ours = 40.02."
    )
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "Simple Baselines.pdf",
            "heading_path": "5.2 Applications",
            "answer_claim": "Baseline and NAFNet tie for the best SIDD PSNR at 40.30 dB.",
            "evidence_quote": evidence,
            "location_label": "5.2 Applications · p. 13",
            "strict_locate": True,
            "page_start": 13,
            "binding_status": "grounded",
            "binding_confidence": 0.9,
        }
    )

    assert detail["card_evidence"] == evidence
    assert "Table 6" in detail["card_evidence"]
    assert detail["card_evidence"].count("=") == 3


def test_system_a_card_composer_strips_markdown_source_markup() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "Foveated.pdf",
            "heading_path": "INTRODUCTION",
            "evidence_quote": (
                "## Foveated single-pixel imaging\n"
                "Single-pixel imaging is based on structured illumination."
            ),
            "location_label": "INTRODUCTION",
        }
    )

    assert detail["card_evidence"].startswith("Single-pixel imaging is based")
    assert "##" not in detail["card_evidence"]


def test_system_a_card_composer_trims_mid_word_ellipsis() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "PIDL.pdf",
            "heading_path": "Abstract",
            "answer_claim": "The paper introduces deep learning into SPAD.",
            "evidence_quote": "we introduce deep learning into SPAD, enabling super-resolution single-photon ima...",
            "location_label": "Abstract",
        }
    )

    assert detail["card_evidence"].endswith("single-photon...")
    assert "ima..." not in detail["card_evidence"]


def test_system_a_card_uses_grounded_evidence_takeaway_when_claim_is_too_short_for_shelf() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "Principles and prospects for single-pixel imaging.pdf",
            "heading_path": "Acquisition and image reconstruction strategies",
            "answer_claim": "压缩感知如何让测量次数少于像素总数成为可能。",
            "evidence_quote": (
                "Their pioneering work has laid the foundations for recovering images from a "
                "single-pixel camera when the number of measurements is fewer than the total number "
                "of unknown pixels in the image, also known as under-sampling or sub-sampling."
            ),
            "location_label": "Acquisition and image reconstruction strategies",
            "binding_status": "grounded",
            "binding_confidence": 0.86,
        }
    )

    summary = str(detail["card_view"]["summary"])
    assert summary.startswith("压缩感知让单像素相机")
    assert len(summary) >= 24
    assert summary != detail["answer_claim"]


def test_card_composer_scrubs_markdown_table_and_structured_tokens() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "Fixture.pdf",
            "heading_path": "Method",
            "answer_claim": "Single-pixel imaging uses structured illumination.",
            "evidence_quote": (
                "## Evidence\n"
                "| field | value |\n"
                "| --- | --- |\n"
                "| note | **Single-pixel imaging** uses `structured illumination` [[CITE:abc12345:3]] "
                "and $DMD$ modulation to collect measurements. |\n"
            ),
            "location_label": "Method",
            "binding_status": "grounded",
            "binding_confidence": 0.86,
        }
    )

    assert "Single-pixel imaging" in detail["card_evidence"]
    assert "structured illumination" in detail["card_evidence"]
    for bad in ("##", "|", "**", "`", "CITE", "$"):
        assert bad not in detail["card_evidence"]


def test_system_a_card_composer_strips_title_author_prefix_from_evidence() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf",
            "heading_path": "5. Single-Pixel Imaging Realizations with Deep Learning",
            "answer_claim": "Deep learning improves single-pixel imaging quality and speed.",
            "evidence_quote": (
                "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning "
                "Kai Song, Yaoxing Bian,\\ Dong Wang, Runrui Li, Ku Wu, Hongrui Liu, "
                "Chengbing Qin, Jianyong Hu,\\ and Liantuan Xiao* "
                "Single-pixel imaging technology can capture images at wavelengths outside "
                "the reach of conventional focal plane array detectors. However, the limited "
                "image quality and lengthy computational times still hinder practical application."
            ),
            "location_label": "5. Single-Pixel Imaging Realizations with Deep Learning",
        }
    )

    assert detail["card_evidence"].startswith("Single-pixel imaging technology can capture")
    assert "limited image quality" in detail["card_evidence"]
    assert "Kai Song" not in detail["card_evidence"]
    assert "Yaoxing" not in detail["card_evidence"]
    assert "\\" not in detail["card_evidence"]
    assert "Advances and Challenges" not in detail["card_evidence"]
    assert "单像素成像可以覆盖" in detail["card_takeaway"]


def test_system_a_card_composer_skips_fragmentary_chunk_lead() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
            "heading_path": "Abstract / Understanding compressed sensing",
            "answer_claim": "The review explains single-pixel camera configurations using a DMD.",
            "evidence_quote": (
                "rson can be described uniquely with a few targeted questions—a property "
                "closely related to sparsity that is key to many measurement problems and "
                "gives rise to the fields of both data compression and Figure 1. "
                "Computational imaging configurations. A DMD can be used to spatially filter "
                "light by selectively redirecting parts of an incident light beam at ±24° to "
                "the normal. a, Single-pixel camera configuration."
            ),
            "location_label": "Abstract / Understanding compressed sensing",
        }
    )

    assert detail["card_evidence"].startswith("A DMD can be used to spatially filter light")
    assert "DMD 可以作为单像素相机" in detail["card_takeaway"]
    assert "rson can be described" not in detail["card_evidence"]
    assert "targeted questions" not in detail["card_evidence"]
    assert "Computational imaging configurations" not in detail["card_evidence"]
    assert "a, Single-pixel camera configuration" not in detail["card_evidence"]


def test_system_a_card_composer_keeps_readable_evidence_window() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "Deep-SPI.pdf",
            "heading_path": "Deep learning / Reconstruction",
            "answer_claim": "深度学习能在低采样率下提升单像素成像质量。",
            "evidence_quote": (
                "Early single-pixel imaging methods rely on iterative reconstruction. "
                "Deep learning models map low-dimensional measurements to target images. "
                "This reduces the required sampling ratio while preserving reconstruction quality. "
                "Implementation details are discussed later."
            ),
            "location_label": "Deep learning / Reconstruction",
        }
    )

    assert "Deep learning models map low-dimensional measurements" in detail["card_evidence"]
    assert "This reduces the required sampling ratio" in detail["card_evidence"]
    assert "Implementation details are discussed later" not in detail["card_evidence"]


def test_system_a_card_composer_trims_incomplete_evidence_tail() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "Deep-SPI.pdf",
            "heading_path": "Abstract",
            "answer_claim": "Deep learning improves single-pixel imaging quality and speed.",
            "evidence_quote": (
                "Single-pixel imaging technology can capture images at wavelengths outside "
                "the reach of conventional focal plane array detectors. However, the limited "
                "image quality and lengthy computational times for iterative reconstruction "
                "still hinder its practical application. Recently, single-pixel imaging based "
                "on deep learning has attrac"
            ),
            "location_label": "Abstract",
        }
    )

    assert "attrac" not in detail["card_evidence"]
    assert detail["card_evidence"].endswith("application.")
    assert "Recently" not in detail["card_evidence"]


def test_system_a_card_composer_removes_dangling_bracket_tail() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "Foveated.pdf",
            "heading_path": "INTRODUCTION",
            "answer_claim": "Single-pixel imaging measures correlations with projected patterns.",
            "evidence_quote": (
                "Single-pixel imaging is based on the measurement of the level of correlation "
                "between the scene and a series of patterns. The patterns can either be "
                "projected onto the scene [known as structured illumination (1)"
            ),
            "location_label": "INTRODUCTION",
        }
    )

    assert detail["card_evidence"] == (
        "Single-pixel imaging is based on the measurement of the level of correlation "
        "between the scene and a series of patterns."
    )
    assert "[" not in detail["card_evidence"]


def test_system_a_card_composer_drops_mismatched_location_leaf() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "Deep-SPI.pdf",
            "heading_path": "5. Single-Pixel Imaging Realizations with Deep Learning / 5.4. Optical Encryption",
            "answer_claim": "Deep learning improves single-pixel imaging quality and speed.",
            "evidence_quote": (
                "Deep learning models can improve single-pixel imaging reconstruction quality "
                "and speed."
            ),
            "location_label": "5. Single-Pixel Imaging Realizations with Deep Learning / 5.4. Optical Encryption",
        }
    )

    assert detail["card_locator"] == "5. Single-Pixel Imaging Realizations with Deep Learning"
    assert "Optical Encryption" not in detail["card_locator"]


def test_system_a_card_composer_labels_document_front_evidence_as_abstract() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "Deep-SPI.pdf",
            "heading_path": "5. Single-Pixel Imaging Realizations with Deep Learning / 5.4. Optical Encryption",
            "answer_claim": "Deep learning improves single-pixel imaging quality and speed.",
            "evidence_quote": (
                "# Advances and Challenges of Single-Pixel Imaging Based on Deep Learning\n"
                "Kai Song, Yaoxing Bian, Dong Wang\n"
                "Single-pixel imaging technology can capture images at wavelengths outside "
                "the reach of conventional focal plane array detectors. However, the limited "
                "image quality and lengthy computational times still hinder practical application."
            ),
            "location_label": "5. Single-Pixel Imaging Realizations with Deep Learning / 5.4. Optical Encryption",
        }
    )

    assert detail["card_locator"] == "Abstract"
    assert detail["card_evidence"].startswith("Single-pixel imaging technology can capture")


def test_system_a_card_composer_avoids_duplicate_takeaway() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "Deep-SPI.pdf",
            "heading_path": "Deep learning / Reconstruction",
            "answer_claim": "深度学习能在低采样率下提升单像素成像质量。",
            "evidence_quote": "Deep learning can improve single-pixel imaging reconstruction quality at lower sampling ratios.",
            "location_label": "Deep learning / Reconstruction",
            "binding_status": "grounded",
            "binding_confidence": 0.86,
        }
    )

    assert detail["card_evidence"].startswith("Deep learning can improve")
    assert detail["card_takeaway"] != detail["card_evidence"]


def test_system_a_card_composer_suppresses_low_value_answer_label() -> None:
    detail = compose_citation_card(
        {
            "source_name": "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf",
            "heading_path": "5. Single-Pixel Imaging Realizations with Deep Learning",
            "answer_claim": "Deep learning review 1",
            "evidence_quote": (
                "Deep learning models map low-dimensional measurements to target images. "
                "This reduces the required sampling ratio while preserving reconstruction quality."
            ),
            "location_label": "5. Single-Pixel Imaging Realizations with Deep Learning",
        }
    )

    assert detail["card_claim"] == ""
    assert detail["card_claim_label"] == "答案要点"
    assert detail["card_locator_label"] == "原文位置"
    assert "low_value_answer_claim" in detail["card_quality_flags"]
    assert detail["card_evidence"].startswith("Deep learning models")


def test_system_a_card_composer_suppresses_bibliographic_answer_label() -> None:
    detail = compose_citation_card(
        {
            "source_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
            "answer_claim": "文献：Hadamard single-pixel imaging versus Fourier single-pixel imaging (Optics Express, 2017)",
            "evidence_quote": (
                "Hadamard basis patterns are binary, which makes HSI naturally suitable for "
                "DMD-based implementations."
            ),
            "location_label": "2. Comparison of theory / 2.2 Basis patterns generation",
        }
    )

    assert detail["card_claim"] == ""
    assert "low_value_answer_claim" in detail["card_quality_flags"]
    assert detail["card_evidence"].startswith("Hadamard basis patterns")


def test_system_a_card_composer_suppresses_reading_guide_title_claims() -> None:
    claims = (
        (
            "\u300aAdvances and Challenges of Single-Pixel Imaging Based on Deep Learning\u300b"
            "\uff08Laser & Photonics Reviews, 2025\uff09"
        ),
        (
            "\u91cd\u70b9\u9605\u8bfb \u201cAcquisition and image reconstruction strategies\u201d "
            "\u90e8\u5206\uff0c\u7406\u89e3\u4e3a\u4ec0\u4e48 SPI \u80fd\u5728\u7ea2\u5916\u6ce2\u6bb5\u5de5\u4f5c\u3002"
        ),
        (
            "\u82e5\u5bf9\u592a\u8d6b\u5179\u6ce2\u6bb5\u611f\u5174\u8da3\uff0c\u53ef\u770b "
            "\u300aFrequency-division-multiplexed single-pixel imaging with metamaterials\u300b"
            "\uff08Optica, 2016\uff09\u3002"
        ),
        "\u65b9\u6cd5\u5bf9\u6bd4\uff1aHadamard single-pixel imaging versus Fourier single-pixel imaging (Optics Express, 2017)",
        "\u524d\u6cbf\u8fdb\u5c55\uff1aAdvances and Challenges of Single-Pixel Imaging Based on Deep Learning (Laser & Photonics Reviews, 2025)",
        "\u7efc\u8ff0\u5165\u95e8\uff1aPrinciples and prospects for single-pixel imaging (Nature Photonics, 2019)",
        (
            "\u4e0b\u4e00\u6b65\uff1a\u5728\u8bfb\u5b8c\u4e0a\u8ff0\u4e09\u7bc7\u540e\uff0c"
            "\u53ef\u9488\u5bf9\u4f60\u611f\u5174\u8da3\u7684\u5e94\u7528\u65b9\u5411\u67e5\u9605\u4e13\u95e8\u8bba\u6587\u3002"
            "\u4f8b\u5982\uff0cImaging biological tissue with single-pixel compressive holography "
            "(Nature Communications, 2021)"
        ),
    )
    for claim in claims:
        detail = compose_citation_card(
            {
                "source_name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
                "heading_path": "Acquisition and image reconstruction strategies",
                "answer_claim": claim,
                "evidence_quote": (
                    "Single-pixel imaging can operate at wavelengths where focal plane arrays "
                    "are expensive or unavailable."
                ),
                "location_label": "Acquisition and image reconstruction strategies",
            }
        )

        assert detail["card_claim"] == ""
        assert "low_value_answer_claim" in detail["card_quality_flags"]
        assert detail["card_evidence"].startswith("Single-pixel imaging can operate")


def test_system_b_card_composer_marks_answer_context_only() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "SCINeRF.pdf",
            "title": "Distributed Optimization and Statistical Learning via ADMM",
            "authors": "Boyd S",
            "year": "2011",
            "answer_claim": "ADMM is borrowed optimization background.",
            "citation_context": "ADMM is borrowed optimization background.",
            "citation_context_source": "answer_context",
            "user_question_relation": "用户问的是这个想法从哪里来；这条参考是当前论文给出的上游来源。",
        }
    )

    assert detail["card_kind"] == "upstream_reference"
    assert detail["card_evidence_label"] == "回答里的线索"
    assert detail["card_evidence"] == ""
    assert detail["card_takeaway_label"] == "上游作用"
    assert "ADMM 优化框架背景" in detail["card_takeaway"]
    assert detail["card_support_label"] == ""
    assert "answer_context_only" in detail["card_quality_flags"]
    assert "answer_context_hidden_from_card" in detail["card_quality_flags"]
    assert "完整引用语境" in detail["card_warning"]
    assert detail["system_b_trace_complete"] is False
    assert "answer_context_only" in detail["system_b_trace_flags"]
    assert detail["system_b_trace_steps"] == ["答案句", "引用语境待核对", "上游文献"]


def test_system_b_card_keeps_precomputed_cassi_citation_context() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "CVPR-2024-SCINeRF.pdf",
            "title": "Snapshot Compressive Imaging: Theory, Algorithms, and Applications",
            "authors": "Xin Yuan, David J. Brady, and Aggelos K. Katsaggelos",
            "venue": "IEEE Signal Processing Magazine",
            "year": "2021",
            "raw": (
                "Xin Yuan, David J. Brady, and Aggelos K. Katsaggelos. Snapshot compressive "
                "imaging: Theory, algorithms, and applications. IEEE Signal Processing "
                "Magazine, 38(2):65-88, 2021."
            ),
            "answer_claim": (
                "SCINeRF cites video SCI prior work when tracing the transition from "
                "compressed video imaging to 3D reconstruction."
            ),
            "citation_context": (
                "...Drawing inspiration from Compressed Sensing (CS) [5,8], video Snapshot "
                "Compressive Imaging (SCI) [50] system has emerged to address these limitations...."
            ),
            "citation_context_source": "structured_reference_index",
            "heading_path": "SCINeRF / 1. Introduction",
            "location_label": "SCINeRF / 1. Introduction / p. 1",
        }
    )

    assert detail["card_evidence"].startswith("Drawing inspiration from Compressed Sensing")
    assert "video Snapshot Compressive Imaging (SCI) [50]" in detail["card_evidence"]
    assert "missing_citation_context" not in detail["card_quality_flags"]
    assert "reference_entry_only" not in detail["card_quality_flags"]
    assert "视频快照压缩成像路线" in detail["card_takeaway"]
    assert "missing_takeaway" not in detail["card_quality_flags"]
    assert detail["system_b_trace_complete"] is True
    assert detail["system_b_trace_context"] == detail["card_evidence"]
    assert detail["system_b_trace_source"] == "structured_reference_index"


def test_system_b_card_composer_distills_generic_english_role() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "Fixture Paper.pdf",
            "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
            "raw": "[1] Gehm M, Brady D. Single-shot compressive spectral imaging with a dual-disperser architecture. Optics Express, 2007.",
            "answer_claim": "Equation (1) uses this reference as the upstream source for single-shot compressive spectral imaging.",
            "citation_context": "The current paper cites this work when tracing the single-shot compressive spectral imaging background.",
            "upstream_work_role": "Cited prior work or background source used to trace the upstream origin of the answer.",
            "user_question_relation": "The user is asking about the evidence behind the answer; this reference is the upstream paper to open next.",
        }
    )

    assert detail["card_kind"] == "upstream_reference"
    assert "单次压缩光谱成像" in detail["card_takeaway"]
    assert "The user is asking" not in detail["card_takeaway"]
    assert detail["card_flow"] == []


def test_system_b_card_composer_distills_classic_spi_compressive_sampling_role() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "3D single-pixel video.pdf",
            "title": "Single-pixel imaging via compressive sampling",
            "authors": "Duarte M, Davenport M, Takhar D, et al",
            "venue": "IEEE Signal Processing Magazine",
            "year": "2008",
            "raw": (
                "Duarte M F, Davenport M A, Takhar D, Laska J N, Kelly K E and "
                "Baraniuk R G 2008 Single-pixel imaging via compressive sampling "
                "IEEE Signal Process. Mag."
            ),
            "answer_claim": "This upstream paper is useful for following the citation chain behind SPI.",
            "citation_context": (
                "Single-pixel imaging is a computational imaging technique that allows "
                "a single-pixel detector to be used as an imaging device."
            ),
            "location_label": "3D single-pixel video / Introduction",
        }
    )

    assert detail["card_kind"] == "upstream_reference"
    assert "单像素压缩采样路线" in detail["card_takeaway"]
    assert "missing_takeaway" not in detail["card_quality_flags"]
    assert detail["system_b_trace_complete"] is True
    assert detail["system_b_trace_steps"] == ["答案句", "当前论文引用处", "上游文献"]
    assert "answer_context_only" not in detail["system_b_trace_flags"]


def test_system_b_card_composer_suppresses_duplicate_claim_and_support_copy() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "SCINeRF.pdf",
            "title": "Distributed Optimization and Statistical Learning via ADMM",
            "authors": "Boyd S",
            "year": "2011",
            "answer_claim": "ADMM is borrowed optimization background.",
            "citation_context": "ADMM is borrowed optimization background.",
            "citation_context_source": "answer_context",
            "support_relation": "The user is asking about the evidence behind the answer; this reference is the upstream paper to open next.",
        }
    )

    assert detail["card_claim"] == ""
    assert detail["card_support_explanation"] == ""
    assert "ADMM" in detail["card_takeaway"]


def test_system_b_card_composer_strips_tex_citation_markup() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf",
            "title": "Optical imaging by means of two-photon quantum entanglement",
            "authors": "Pittman T, Shih Y, Strekalov D",
            "venue": "Physical Review A",
            "year": "1995",
            "answer_claim": "SPI has advantages in sensitivity and spectral response.",
            "citation_context": (
                "...$^{[4-6]}$ Unlike traditional focal plane array detectors, SPI only adopts "
                "a single-pixel detector to collect echo signals, offering significant advantages "
                "in detection sensitivity, spectral response range, and imaging cost."
            ),
            "citation_context_source": "source_markdown",
            "location_label": "Advances and Challenges / 1. Introduction",
        }
    )

    assert detail["card_evidence"].startswith("Unlike traditional focal plane array detectors")
    assert "$" not in detail["card_evidence"]
    assert "^{" not in detail["card_evidence"]
    assert "[4-6]" not in detail["card_evidence"]
    assert "weak_citation_context" not in detail["card_quality_flags"]
    assert "单像素探测" in detail["card_context_summary"]
    assert "context_summary" in detail["card_visible_sections"]


def test_system_b_card_composer_suppresses_author_list_context() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "NatPhoton-2025-Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy.pdf",
            "title": "Missing Cone Of Frequencies And Low-Pass Distortion In Three-Dimensional Microscopic Images",
            "authors": "Macias-Garza",
            "venue": "IEEE Trans. Acoust., Speech, Signal Process.",
            "year": "1988",
            "answer_claim": "This citation should support the discussion of three-dimensional microscopic imaging limits.",
            "citation_context": (
                "Alessandro Zunino [1,4], Giacomo Garre [1,2,4], Eleonora Perego [1,3], "
                "Sabrina Zappone [1,2], Mattia Donato [1], Nadine Vastenhouw [3] "
                "& Giuseppe Vicidomini [1]"
            ),
            "citation_context_source": "source_markdown",
            "location_label": "Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy",
        }
    )

    assert detail["card_evidence"] == ""
    assert detail["card_locator_label"] == "当前论文引用处"
    assert "weak_citation_context" in detail["card_quality_flags"]
    assert "missing_citation_context" in detail["card_quality_flags"]


def test_system_b_card_composer_summarizes_missing_cone_context() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "NatPhoton-2025-Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy.pdf",
            "title": "Missing Cone Of Frequencies And Low-Pass Distortion In Three-Dimensional Microscopic Images",
            "raw": "Macias-Garza F. The missing cone problem and low-pass distortion in optical serial sectioning microscopy.",
            "answer_claim": "结构检测可以缓解三维显微成像中的频率缺失问题。",
            "citation_context": (
                "The current paper cites missing cone and low-pass distortion work when "
                "discussing limitations in three-dimensional microscopic images."
            ),
            "citation_context_source": "source_markdown",
            "location_label": "Introduction",
        }
    )

    assert "频率缺失" in detail["card_context_summary"]
    assert "低通失真" in detail["card_context_summary"]
    assert "The current paper cites" not in detail["card_context_summary"]


def test_system_b_card_composer_parses_title_from_raw_reference_without_promoting_raw_entry() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "Fixture Paper.pdf",
            "raw": (
                "[1] Gehm M, Brady D. Single-shot compressive spectral imaging with a "
                "dual-disperser architecture. Optics Express, 2007. doi:10.1364/OE.15.014013"
            ),
            "answer_claim": "The answer traces single-shot compressive spectral imaging.",
            "citation_context": (
                "The current paper cites this work when tracing the single-shot "
                "compressive spectral imaging background."
            ),
        }
    )

    assert detail["card_title"] == "Single-shot compressive spectral imaging with a dual-disperser architecture"
    assert not detail["card_title"].startswith("[1] Gehm")
    assert "missing_reference_title" not in detail["card_quality_flags"]


def test_system_b_card_composer_keeps_raw_reference_out_of_missing_title_header() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "Fixture Paper.pdf",
            "raw": "Unparseable upstream item. No stable title.",
            "citation_context": (
                "Alessandro Zunino [1,4], Giacomo Garre [1,2,4], Eleonora Perego [1,3], "
                "Sabrina Zappone [1,2], Mattia Donato [1], Nadine Vastenhouw [3] "
                "& Giuseppe Vicidomini [1]"
            ),
            "citation_context_source": "source_markdown",
        }
    )

    assert detail["card_title"] == "上游参考文献"
    assert detail["card_evidence"] == ""
    assert "missing_reference_title" in detail["card_quality_flags"]
    assert "weak_citation_context" in detail["card_quality_flags"]


def test_system_b_card_composer_keeps_upstream_reference_entry_separate() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "NatPhoton-2025-Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy.pdf",
            "title": "Missing Cone Of Frequencies And Low-Pass Distortion In Three-Dimensional Microscopic Images",
            "raw": (
                "[3] Macias-Garza, F., Bovik, A. C., Diller, K. R., Aggarwal, S. J. "
                "& Aggarwal, J. K. The missing cone problem and low-pass distortion in "
                "optical serial sectioning microscopy. IEEE Trans. Acoust., Speech, "
                "Signal Process. 2, 890-893 (1988)."
            ),
            "answer_claim": "This citation should support the discussion of three-dimensional microscopic imaging limits.",
            "citation_context": (
                "Alessandro Zunino [1,4], Giacomo Garre [1,2,4], Eleonora Perego [1,3], "
                "Sabrina Zappone [1,2], Mattia Donato [1], Nadine Vastenhouw [3] "
                "& Giuseppe Vicidomini [1]"
            ),
            "citation_context_source": "source_markdown",
            "location_label": "Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy",
        }
    )

    assert detail["card_evidence"] == ""
    assert detail["card_evidence_label"] == "引用语境"
    assert detail["card_reference_label"] == "上游文献条目"
    assert "The missing cone problem" in detail["card_reference_entry"]
    assert "reference_entry_only" in detail["card_quality_flags"]
    assert "上游论文正文证据" in detail["card_warning"]


def test_card_composer_exposes_compact_display_contract() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "Fixture.pdf",
            "heading_path": "Method",
            "answer_claim": "深度学习用于提高单像素重建质量。",
            "evidence_quote": "## Method\nDeep learning models improve single-pixel reconstruction quality.",
            "location_label": "Method",
            "summary_line": "No summary available",
            "binding_status": "grounded",
            "binding_confidence": 0.86,
        }
    )

    assert detail["card_display_contract_version"] == 2
    assert "locator" in detail["card_visible_sections"]
    assert "evidence" in detail["card_visible_sections"]
    assert "card_evidence_markup_cleaned" in detail["card_quality_flags"]
    assert detail["card_evidence"]
    assert "##" not in detail["card_evidence"]


def test_system_b_card_contract_keeps_reference_entry_out_of_main_evidence() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "Paper.pdf",
            "title": "Missing Cone Of Frequencies And Low-Pass Distortion In Three-Dimensional Microscopic Images",
            "raw": (
                "[3] Macias-Garza, F., Bovik, A. C., Diller, K. R., Aggarwal, S. J. "
                "& Aggarwal, J. K. The missing cone problem and low-pass distortion in "
                "optical serial sectioning microscopy. IEEE Trans. Acoust., Speech, "
                "Signal Process. 2, 890-893 (1988)."
            ),
            "answer_claim": "This citation is relevant to missing-cone limits in 3D microscopy.",
            "citation_context": (
                "Macias-Garza, F., Bovik, A. C., Diller, K. R., Aggarwal, S. J. "
                "& Aggarwal, J. K. The missing cone problem and low-pass distortion in "
                "optical serial sectioning microscopy. IEEE Trans. Acoust., Speech, "
                "Signal Process. 2, 890-893 (1988)."
            ),
            "citation_context_source": "source_markdown",
            "location_label": "Related Work",
        }
    )

    assert detail["card_evidence"] == ""
    assert detail["card_reference_entry"]
    assert "reference" not in detail["card_visible_sections"]
    assert "reference_entry_only" in detail["card_quality_flags"]


def test_card_composer_strips_repeated_source_title_from_locator() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf",
            "heading_path": "5. Single-Pixel Imaging Realizations with Deep Learning",
            "answer_claim": "深度学习综述适合先读方法收益和限制。",
            "evidence_quote": "Deep learning methods improve single-pixel imaging quality and speed.",
            "location_label": (
                "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning / "
                "5. Single-Pixel Imaging Realizations with Deep Learning"
            ),
            "binding_status": "grounded",
            "binding_confidence": 0.86,
        }
    )

    assert detail["card_locator"] == "5. Single-Pixel Imaging Realizations with Deep Learning"
    assert "Advances and Challenges" not in detail["card_locator"]


def test_system_b_card_composer_strips_current_paper_title_from_locator() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "NatPhoton-2025-Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy.pdf",
            "title": "Missing Cone Of Frequencies And Low-Pass Distortion In Three-Dimensional Microscopic Images",
            "raw": "[3] Macias-Garza F. The missing cone problem. IEEE Trans., 1988.",
            "answer_claim": "This upstream paper explains missing-cone limits.",
            "citation_context": "The current paper cites this missing-cone work when discussing three-dimensional microscopy limits.",
            "citation_context_source": "source_markdown",
            "location_label": (
                "Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy / "
                "Introduction"
            ),
        }
    )

    assert detail["card_locator"] == "Introduction"
    assert "Structured detection for simultaneous" not in detail["card_locator"]


def test_system_a_card_view_contract_exposes_renderable_sections() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "Deep-SPI.pdf",
            "heading_path": "Abstract",
            "answer_claim": "深度学习可以降低单像素成像的采样压力。",
            "evidence_quote": (
                "Deep learning models map low-dimensional measurements to target images. "
                "This reduces the required sampling ratio while preserving reconstruction quality."
            ),
            "location_label": "Abstract",
            "binding_status": "grounded",
            "binding_confidence": 0.9,
        }
    )

    view = detail["card_view"]
    section_ids = [section["id"] for section in view["sections"]]

    assert view["version"] == 2
    assert view["route"] == "system_a"
    assert view["header"]["kicker"] == "答案依据"
    assert view["header"]["title"] == "Deep-SPI.pdf"
    assert "locator" in section_ids
    assert "evidence" in section_ids
    assert "claim" not in section_ids
    assert "support" not in section_ids
    assert section_ids.index("evidence") < section_ids.index("locator")
    assert view["summary"]
    assert all("##" not in section["text"] for section in view["sections"])


def test_grounded_system_a_card_view_exposes_distinct_relevance_copy() -> None:
    relevance = (
        "This passage connects the answer's image-loop mechanism to the paper's "
        "iterative reuse of the reconstructed image as the next network input."
    )
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "PILN.pdf",
            "heading_path": "Abstract",
            "answer_claim": "ILNet feeds its reconstructed image into the next iteration.",
            "evidence_quote": (
                "The 2D image generated by ILNet can serve as input for the subsequent "
                "iteration to continuously incorporate prior information."
            ),
            "location_label": "Abstract / p. 2",
            "support_relation": relevance,
            "binding_status": "grounded",
            "binding_confidence": 0.9,
        },
        locale="en",
    )

    support = next(
        section for section in detail["card_view"]["sections"]
        if section["id"] == "support"
    )
    assert support["label"] == "Why it supports the answer"
    assert support["text"] == relevance
    assert detail["card_support_explanation"] == relevance
    assert "support" in detail["card_visible_sections"]


def test_grounded_system_a_card_never_exposes_template_binding_reason() -> None:
    template_reason = (
        "This citation reuses the source evidence actually supplied during answer "
        "generation and matches the claim terms ILNet and iteration."
    )
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "PILN.pdf",
            "heading_path": "Abstract",
            "answer_claim": "ILNet feeds its reconstructed image into the next iteration.",
            "evidence_quote": (
                "The 2D image generated by ILNet can serve as input for the subsequent "
                "iteration to continuously incorporate prior information."
            ),
            "location_label": "Abstract / p. 2",
            "binding_status": "grounded",
            "binding_confidence": 0.9,
            "binding_reason": template_reason,
        },
        locale="en",
    )

    assert detail["card_support_explanation"] == ""
    assert "support" not in detail["card_visible_sections"]
    assert "support" not in {
        section["id"] for section in detail["card_view"]["sections"]
    }
    assert template_reason not in str(detail["card_view"])


def test_model_driven_system_a_takeaway_replaces_fragmented_contradictory_claim() -> None:
    detail = compose_citation_card(
        {
            "citation_route": "system_a",
            "source_name": "Advances and Challenges of DL-SPI.pdf",
            "heading_path": "4.1.2. Model-Driven Strategy",
            "answer_claim": "泛化能力 受限于训练数据分布 泛化性优异",
            "evidence_quote": (
                "Model-driven strategy is an unsupervised learning mode that exhibits "
                "exceptional generalization. This strategy integrates the physical "
                "process of SPI with neural networks and leverages the discrepancy "
                "between real and estimated measurements to guide network optimization."
            ),
            "location_label": "4.1.2. Model-Driven Strategy / p. 8",
        },
        locale="zh",
    )

    takeaway = detail["card_takeaway"]
    assert all(term in takeaway for term in ("SPI", "物理过程", "神经网络", "测量", "无监督"))
    assert "受限于训练数据分布 泛化性优异" not in takeaway
    assert len(takeaway) >= 36


def test_system_a_card_unwraps_chinese_source_excerpt_prefix() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "SPD-review.pdf",
            "heading_path": "3 Single photon detection parameter",
            "answer_claim": "Detector review builds SPAD and single-photon detector background.",
            "evidence_quote": (
                "\u539f\u6587\u7247\u6bb5\u5199\u5230\uff1a\u201c"
                "Single-photon detections represent a highly sensitive light detection technique."
                "\u201d"
            ),
            "location_label": "3 Single photon detection parameter",
            "binding_status": "grounded",
            "binding_confidence": 0.8,
        }
    )

    assert "missing_evidence_quote" not in detail["card_quality_flags"]
    assert detail["card_evidence"].startswith("Single-photon detections")
    assert "\u539f\u6587\u7247\u6bb5\u5199\u5230" not in detail["card_evidence"]


def test_system_a_card_keeps_complete_sentence_with_coordinating_lead() -> None:
    evidence = (
        "And the aforementioned semiconductive SPDs (Si-SPAD, InSb, HgCdTe, etc.), "
        "the superconducting TES detector has higher detection efficiency, faster response "
        "speed, lower dark count, and higher energy resolution."
    )
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "SPD-review.pdf",
            "heading_path": "3 Single photon detection parameter",
            "answer_claim": (
                "TES detectors have higher detection efficiency and lower dark count "
                "than the previously discussed semiconductor SPDs."
            ),
            "evidence_quote": evidence,
            "location_label": "3 Single photon detection parameter",
            "binding_status": "grounded",
            "binding_confidence": 0.9,
        }
    )

    assert detail["evidence_quote"] == evidence
    assert detail["card_evidence"] == evidence
    assert "evidence_quote_filtered" not in detail["card_quality_flags"]
    assert "missing_evidence_quote" not in detail["card_quality_flags"]


def test_system_a_card_still_filters_detached_coordinating_fragment() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "SPD-review.pdf",
            "heading_path": "Detector comparison",
            "answer_claim": "The detector has lower dark count.",
            "evidence_quote": "And lower dark count and higher energy resolution.",
            "location_label": "Detector comparison",
            "binding_status": "grounded",
            "binding_confidence": 0.9,
        }
    )

    assert detail["card_evidence"] == ""
    assert "missing_evidence_quote" in detail["card_quality_flags"]


def test_system_a_card_suppresses_reading_roadmap_bibliographic_claim() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
            "heading_path": "Experiment design / Coding choice",
            "answer_claim": (
                "\u518d\u8bfb\u65b9\u6cd5\u5bf9\u6bd4\uff1a"
                "\u300aHadamard single-pixel imaging versus Fourier single-pixel imaging\u300b "
                "(Optics Express, 2017)"
            ),
            "evidence_quote": (
                "Hadamard basis patterns are binary, which makes HSI naturally suitable "
                "for single-pixel imaging systems based on digital micromirror devices."
            ),
            "location_label": "Experiment design / Coding choice",
            "binding_status": "grounded",
            "binding_confidence": 0.82,
        }
    )

    assert detail["answer_claim"] == ""
    assert detail["card_claim"] == ""
    assert detail["evidence_quote"] == detail["card_evidence"]
    assert "low_value_answer_claim" in detail["card_quality_flags"]
    assert "claim" not in {section["id"] for section in detail["card_view"]["sections"]}


def test_system_b_card_view_contract_separates_context_from_reference_entry() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "Current.pdf",
            "title": "Missing Cone Of Frequencies And Low-Pass Distortion In Three-Dimensional Microscopic Images",
            "raw": (
                "[3] Macias-Garza, F., Bovik, A. C., Diller, K. R., Aggarwal, S. J. "
                "& Aggarwal, J. K. The missing cone problem and low-pass distortion in "
                "optical serial sectioning microscopy. IEEE Trans. Acoust., Speech, "
                "Signal Process. 2, 890-893 (1988)."
            ),
            "answer_claim": "这篇上游论文解释三维显微成像里的 missing-cone 限制。",
            "citation_context": (
                "The current paper cites the missing-cone problem when explaining why "
                "three-dimensional microscopy can suffer low-pass distortion."
            ),
            "citation_context_source": "source_markdown",
            "location_label": "Introduction",
        }
    )

    view = detail["card_view"]
    section_ids = {section["id"] for section in view["sections"]}
    section_text = "\n".join(section["text"] for section in view["sections"])

    assert view["version"] == 2
    assert view["route"] == "system_b"
    assert view["header"]["kicker"] == "上游引用"
    assert view["header"]["title"].startswith("Missing Cone")
    assert "locator" in section_ids
    assert "evidence" in section_ids
    assert "reference" not in section_ids
    assert "The current paper cites" in section_text
    assert "Macias-Garza, F." not in view["summary"]


def test_citation_card_contract_uses_english_locale_for_system_a_review_copy() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "PILN.pdf",
            "heading_path": "Abstract",
            "answer_claim": "PILN targets low-sampling single-pixel imaging.",
            "evidence_quote": "The part-based image-loop network reconstructs single-pixel images at low sampling rates.",
            "location_label": "Abstract",
            "binding_status": "candidate",
            "binding_confidence": 0.56,
        },
        locale="en",
    )

    assert detail["render_locale"] == "en"
    assert detail["card_view"]["header"]["kicker"] == "Answer evidence"
    assert detail["card_takeaway_label"] == "Evidence focus"
    assert detail["card_claim_label"] == "Answer point"
    assert detail["card_locator_label"] == "Source location"
    assert detail["card_evidence_label"] == "Source evidence"
    assert detail["card_support_label"] == "Evidence reliability"
    assert detail["card_quality_label"] == "Candidate evidence"
    assert detail["card_warning"] == "This link is candidate evidence. Open the source to confirm the context."


def test_compound_plan_evidence_keeps_all_verified_source_clauses() -> None:
    evidence = (
        "The mask values require phase-sensitive detection. … "
        "Each SLM pixel is modulated on p frequencies simultaneously. … "
        "The light is multiplexed into a single-pixel detector. … "
        "The signal is demodulated by p LIAs."
    )
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "FDM.pdf",
            "heading_path": "B. Encoding",
            "answer_claim": "FDM parallelizes the encoding channels.",
            "evidence_quote": evidence,
            "evidence_source": "retrieval_hit",
            "compound_plan_evidence": True,
            "binding_status": "grounded",
            "binding_confidence": 1.0,
        },
        locale="en",
    )

    assert "phase-sensitive detection" in detail["evidence_quote"]
    assert "p frequencies simultaneously" in detail["evidence_quote"]
    assert "multiplexed into a single-pixel detector" in detail["evidence_quote"]
    assert "demodulated by p LIAs" in detail["evidence_quote"]


def test_formula_card_derives_visible_equation_locators_from_source_tags() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "paper.pdf",
            "heading_path": "2 Quantization Function",
            "page_start": 2,
            "location_label": "2 Quantization Function · p. 2",
            "answer_claim": "RoundClip maps scaled weights to ternary values.",
            "evidence_quote": (
                r"\widetilde{W}=\text{RoundClip}(W/\gamma,-1,1), \tag{1} "
                r"\gamma=\frac{1}{nm}\sum_{ij}|W_{ij}|. \tag{3}"
            ),
            "compound_plan_evidence": True,
            "binding_status": "grounded",
            "binding_confidence": 1.0,
        },
        locale="en",
    )

    assert "Equation (1)" in detail["card_locator"]
    assert "Equation (3)" in detail["card_locator"]
    assert "Equation (1)" in detail["card_view"]["sections"][-1]["text"]


def test_citation_card_contract_uses_english_locale_for_system_b_trace_copy() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "Current.pdf",
            "raw": "[12] Doe, J. Compressive sampling for single-pixel imaging. Optics Letters (2010).",
            "answer_claim": "The answer traces compressive sampling background through the current paper.",
            "location_label": "Related Work",
        },
        locale="en",
    )

    assert detail["render_locale"] == "en"
    assert detail["card_view"]["header"]["kicker"] == "Upstream citation"
    assert detail["card_title"] != "上游参考文献"
    assert detail["card_takeaway_label"] == "Upstream role"
    assert detail["card_claim_label"] == "Answer sentence"
    assert detail["card_locator_label"] == "Where current paper cites it"
    assert detail["card_evidence_label"] == "Citation context"
    assert detail["card_reference_label"] == "Upstream reference entry"
    assert detail["system_b_trace_steps"] == ["Answer sentence", "Citation context to check", "Upstream reference"]
    assert "citation context" in detail["system_b_trace_reason"].lower()


def test_structured_metric_card_recovers_verified_reader_evidence() -> None:
    reader_evidence = (
        "Table 6. Image Denoising Results on SIDD. "
        "SIDD PSNR: MPRNet = 39.71; Restormer = 40.02; "
        "Baseline ours = 40.30; NAFNet ours = 40.30."
    )

    detail = compose_citation_card(
        {
            "is_inpaper": False,
            "source_name": "Simple Baselines.pdf",
            "heading_path": "5 Experiments / 5.2 Applications",
            "answer_claim": "Baseline and NAFNet tie at the highest SIDD PSNR of 40.30.",
            "evidence_quote": "",
            "reader_evidence_quote": reader_evidence,
            "selection_reason": "structured_table_metric_hit",
            "strict_locate": True,
            "page_start": 13,
            "anchor_kind": "table",
        },
        locale="en",
    )

    assert detail["card_evidence"] == reader_evidence
    assert "Baseline ours = 40.30" in detail["summary_line"]
    assert "evidence_quote_filtered" not in detail["card_quality_flags"]
    assert "missing_evidence_quote" not in detail["card_quality_flags"]
