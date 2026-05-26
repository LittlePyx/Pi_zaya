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
    assert detail["card_subtitle"] == "SCIGS / Abstract · p. 1"
    assert detail["card_locator"] == "SCIGS / Abstract · p. 1"
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
    assert detail["card_claim_label"] == "答案中的话"
    assert detail["card_locator_label"] == "原文位置"
    assert "low_value_answer_claim" in detail["card_quality_flags"]
    assert detail["card_evidence"].startswith("Deep learning models")


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
    assert detail["card_takeaway_label"] == "上游作用"
    assert "ADMM 优化框架背景" in detail["card_takeaway"]
    assert detail["card_support_label"] == ""
    assert "answer_context_only" in detail["card_quality_flags"]
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
    assert detail["card_locator_label"] == "引用出现位置"
    assert "weak_citation_context" in detail["card_quality_flags"]
    assert "missing_citation_context" in detail["card_quality_flags"]


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
