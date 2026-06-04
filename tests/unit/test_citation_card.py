from __future__ import annotations

from kb.citation_card import compose_citation_card


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
