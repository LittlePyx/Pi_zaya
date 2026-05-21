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
