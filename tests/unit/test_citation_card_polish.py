from __future__ import annotations

from kb.citation_card_polish import (
    citation_card_polish_cache_key,
    polish_citation_card_detail,
)


def test_polish_citation_card_accepts_grounded_short_fields() -> None:
    detail = {
        "source_name": "Single pixel imaging review.pdf",
        "heading_path": "Methods",
        "answer_claim": "DMD modulation is the sampling hardware behind this method.",
        "evidence_quote": "A DMD can spatially filter light and redirect the incident beam during measurement.",
        "location_label": "Methods",
    }

    def fake_llm(**_kwargs: object) -> str:
        return (
            '{"card_takeaway":"DMD is the hardware mechanism that makes the sampling strategy concrete.",'
            '"card_claim":"The answer is tying the method to DMD-based optical modulation.",'
            '"card_support_explanation":"The quoted sentence names the DMD action rather than only describing an outcome."}'
        )

    out = polish_citation_card_detail(detail, llm_fn=fake_llm)

    assert out["citation_card_polish_status"] == "full"
    assert out["citation_card_polish_source"] == "llm"
    assert out["citation_card_polish_route"] == "system_a"
    assert out["citation_card_polish_fields"] == [
        "card_takeaway",
        "card_claim",
        "card_support_explanation",
    ]
    assert out["citation_card_polish_quality_score"] > 0.7
    assert out["card_takeaway"].startswith("DMD is the hardware")
    assert "card_evidence" not in out
    assert out["citation_card_view_patch_version"] == 1
    assert out["card_view"]["route"] == "system_a"
    sections = {section["id"]: section for section in out["card_view"]["sections"]}
    assert sections["takeaway"]["text"].startswith("DMD is the hardware")
    assert "locator" in sections
    assert "evidence" in sections


def test_polish_citation_card_rejects_markdown_and_generic_output() -> None:
    detail = {
        "source_name": "Fixture.pdf",
        "heading_path": "Abstract",
        "answer_claim": "The work improves single-pixel imaging.",
        "evidence_quote": "Deep learning reduces the sampling ratio while preserving reconstruction quality.",
    }

    def fake_llm(**_kwargs: object) -> str:
        return (
            '{"card_takeaway":"| field | value |",'
            '"card_claim":"This evidence supports the answer.",'
            '"card_support_explanation":"```bad```"}'
        )

    out = polish_citation_card_detail(detail, llm_fn=fake_llm)

    assert out == {
        "citation_card_polish_status": "empty",
        "citation_card_polish_source": "llm_empty",
        "citation_card_polish_checked": True,
    }


def test_polish_citation_card_rejects_repeated_baseline_text() -> None:
    detail = {
        "source_name": "Fixture.pdf",
        "title": "Adaptive foveated single-pixel imaging with dynamic supersampling",
        "heading_path": "INTRODUCTION / Foveated single-pixel imaging",
        "answer_claim": "这篇文章展示了自适应采样策略。",
        "evidence_quote": "自适应采样在感兴趣区域使用高分辨率采样，边缘用低分辨率采样，从而提升帧率。",
        "location_label": "INTRODUCTION / Foveated single-pixel imaging",
    }

    def fake_llm(**_kwargs: object) -> str:
        return (
            '{"card_takeaway":"自适应采样在感兴趣区域使用高分辨率采样，边缘用低分辨率采样，从而提升帧率。",'
            '"card_claim":"INTRODUCTION / Foveated single-pixel imaging",'
            '"card_support_explanation":"Adaptive foveated single-pixel imaging with dynamic supersampling"}'
        )

    out = polish_citation_card_detail(detail, llm_fn=fake_llm)

    assert out == {
        "citation_card_polish_status": "empty",
        "citation_card_polish_source": "llm_empty",
        "citation_card_polish_checked": True,
    }


def test_polish_citation_card_system_b_payload_and_quality_metadata() -> None:
    detail = {
        "is_inpaper": True,
        "source_name": "Current paper.pdf",
        "title": "Optical imaging by means of two-photon quantum entanglement",
        "authors": "Pittman T, Shih Y",
        "venue": "Physical Review A",
        "year": "1995",
        "raw": "Pittman T, Shih Y. Optical imaging by means of two-photon quantum entanglement. Physical Review A, 1995.",
        "answer_claim": "单像素成像可以降低成像成本。",
        "citation_context": "Unlike traditional focal plane array detectors, SPI only adopts a SPD to collect echo signals.",
        "heading_path": "1. Introduction",
    }
    captured: dict[str, object] = {}

    def fake_llm(**kwargs: object) -> str:
        captured.update(kwargs)
        return (
            '{"card_takeaway":"这篇上游文献被用来说明双光子量子纠缠也能服务于光学成像路线。",'
            '"card_claim":"单像素成像路线可以借由不同探测机制降低硬件复杂度。",'
            '"card_context_summary":"当前论文在介绍单像素探测器成本优势时，把这篇早期量子成像工作作为光学成像来源线索。",'
            '"card_support_explanation":"当前论文是在介绍 SPI 探测器成本优势时把它列为早期光学成像来源之一。"}'
        )

    out = polish_citation_card_detail(detail, llm_fn=fake_llm)

    assert out["citation_card_polish_status"] == "full"
    assert out["citation_card_polish_route"] == "system_b"
    assert out["citation_card_polish_fields"] == [
        "card_takeaway",
        "card_claim",
        "card_context_summary",
        "card_support_explanation",
    ]
    assert out["citation_card_polish_quality_score"] > 0.7
    assert "早期量子成像工作" in out["card_context_summary"]
    assert out["citation_card_view_patch_version"] == 1
    assert out["card_view"]["route"] == "system_b"
    sections = {section["id"]: section for section in out["card_view"]["sections"]}
    assert "takeaway" in sections
    assert "context_summary" in sections
    assert sections["context_summary"]["text"] == out["card_context_summary"]
    assert "Physical Review A" not in "\n".join(section["text"] for section in sections.values())
    payload = str(captured["candidate_payload"])
    assert "Reference entry:" in payload
    assert "Citation context:" in payload
    assert "Physical Review A" in payload


def test_polish_citation_card_system_b_rejects_context_summary_copies() -> None:
    detail = {
        "is_inpaper": True,
        "source_name": "Current paper.pdf",
        "title": "Missing Cone Of Frequencies And Low-Pass Distortion In Three-Dimensional Microscopic Images",
        "raw": "Macias-Garza F. The missing cone problem and low-pass distortion in optical serial sectioning microscopy.",
        "answer_claim": "结构检测可以缓解三维显微图像中的频率缺失问题。",
        "citation_context": "The missing cone problem and low-pass distortion in optical serial sectioning microscopy.",
        "heading_path": "Related work",
    }

    def fake_llm(**_kwargs: object) -> str:
        return (
            '{"card_takeaway":"这篇上游工作被当前论文用于说明三维显微成像中的缺失锥与低通失真问题。",'
            '"card_context_summary":"The missing cone problem and low-pass distortion in optical serial sectioning microscopy.",'
            '"card_support_explanation":"这篇上游工作被当前论文用于说明三维显微成像中的缺失锥与低通失真问题。"}'
        )

    out = polish_citation_card_detail(detail, llm_fn=fake_llm)

    assert out["citation_card_polish_status"] == "full"
    assert out["citation_card_polish_fields"] == ["card_takeaway"]
    assert "card_context_summary" not in out
    assert "card_support_explanation" not in out
    assert any("card_context_summary:duplicates_" in item for item in out["citation_card_polish_rejected"])


def test_polish_citation_card_rejects_metadata_repetition_in_narrative_fields() -> None:
    detail = {
        "is_inpaper": True,
        "source_name": "Current paper.pdf",
        "title": "Optical imaging by means of two-photon quantum entanglement",
        "authors": "Pittman T, Shih Y",
        "venue": "Physical Review A",
        "year": "1995",
        "doi": "10.1103/physreva.52.r3429",
        "raw": "Pittman T, Shih Y. Optical imaging by means of two-photon quantum entanglement. Physical Review A, 1995.",
        "answer_claim": "单像素成像可以降低成像成本。",
        "citation_context": "Unlike traditional focal plane array detectors, SPI only adopts a SPD to collect echo signals.",
        "heading_path": "1. Introduction",
    }

    def fake_llm(**_kwargs: object) -> str:
        return (
            '{"card_takeaway":"这篇发表于 Physical Review A 1995 的论文 Optical imaging by means of two-photon quantum entanglement 值得打开。",'
            '"card_context_summary":"当前论文引用它时强调 SPI 用单点探测器替代传统焦平面阵列，线索落在探测结构和成本优势。",'
            '"card_support_explanation":"DOI 是 10.1103/physreva.52.r3429。"}'
        )

    out = polish_citation_card_detail(detail, llm_fn=fake_llm)

    assert out["citation_card_polish_status"] == "full"
    assert out["citation_card_polish_fields"] == ["card_context_summary"]
    assert "card_takeaway" not in out
    assert "card_support_explanation" not in out
    assert out["card_context_summary"].startswith("当前论文引用它时强调 SPI")
    assert any("card_takeaway:metadata_repeated" in item for item in out["citation_card_polish_rejected"])
    assert any("card_support_explanation:metadata_repeated" in item for item in out["citation_card_polish_rejected"])


def test_v2_system_a_polish_keeps_authoritative_compound_card_evidence() -> None:
    first_step = (
        "The operation for digital refocusing of a sample placed out of focus by a distance z "
        "can be achieved using two steps. First, using the position and angular information of "
        "each photon, and knowing the optical elements used between them, the trajectory of the "
        "photons can be reconstructed through a ray tracing operation."
    )
    second_step = (
        "Thus, the second step is to reverse this diffraction by applying a wave propagation of "
        "distance -z to the image obtained after step one in order to bring the sample back into "
        "focus."
    )
    compound = f"{first_step} {second_step}"
    detail = {
        "card_display_contract_version": 2,
        "is_inpaper": False,
        "source_name": "qCLFM.pdf",
        "heading_path": "A. Concept",
        "answer_claim": (
            "Digital refocusing first reconstructs photon trajectories with ray tracing, then "
            "reverses diffraction with wave propagation."
        ),
        "evidence_quote": first_step,
        "card_evidence": compound,
    }
    captured: dict[str, object] = {}

    def fake_llm(**kwargs: object) -> str:
        captured.update(kwargs)
        return (
            '{"card_takeaway":"The cited passage preserves both physical stages of the '
            'refocusing mechanism."}'
        )

    out = polish_citation_card_detail(detail, llm_fn=fake_llm)

    assert captured["evidence"] == compound
    assert second_step in str(captured["candidate_payload"])
    evidence_section = next(
        section for section in out["card_view"]["sections"] if section["id"] == "evidence"
    )
    assert evidence_section["text"] == compound
    assert "ray tracing" in evidence_section["text"]
    assert "wave propagation" in evidence_section["text"]


def test_v2_system_a_polish_cache_key_uses_authoritative_card_evidence() -> None:
    first_step = "First, photon trajectories are reconstructed through ray tracing."
    compound = (
        f"{first_step} Second, wave propagation of distance -z reverses diffraction and restores "
        "focus."
    )
    base = {
        "card_display_contract_version": 2,
        "is_inpaper": False,
        "source_name": "qCLFM.pdf",
        "heading_path": "A. Concept",
        "answer_claim": "Digital refocusing uses ray tracing and wave propagation.",
        "evidence_quote": first_step,
        "card_evidence": compound,
    }

    key = citation_card_polish_cache_key(base)

    assert key == citation_card_polish_cache_key(
        {**base, "evidence_quote": "A stale and unrelated shorter quote."}
    )
    assert key == citation_card_polish_cache_key(
        {
            "cardDisplayContractVersion": 2,
            "isInpaper": False,
            "sourceName": "qCLFM.pdf",
            "headingPath": "A. Concept",
            "answerClaim": "Digital refocusing uses ray tracing and wave propagation.",
            "evidenceQuote": first_step,
            "cardEvidence": compound,
        }
    )
    assert key != citation_card_polish_cache_key(
        {**base, "card_evidence": first_step}
    )


def test_citation_card_polish_cache_key_normalizes_frontend_aliases() -> None:
    snake = {
        "source_name": "Fixture.pdf",
        "heading_path": "Abstract",
        "answer_claim": "The paper uses structured illumination.",
        "evidence_quote": "Structured illumination patterns are projected onto the scene.",
        "location_label": "Abstract",
    }
    camel = {
        "sourceName": "Fixture.pdf",
        "headingPath": "Abstract",
        "answerClaim": "The paper uses structured illumination.",
        "evidenceQuote": "Structured illumination patterns are projected onto the scene.",
        "locationLabel": "Abstract",
    }

    assert citation_card_polish_cache_key(snake) == citation_card_polish_cache_key(camel)


def test_citation_card_polish_cache_key_separates_render_languages() -> None:
    detail = {
        "source_name": "Fixture.pdf",
        "heading_path": "Abstract",
        "answer_claim": "SCINeRF uses the SCI physical model during NeRF training.",
        "evidence_quote": "The physical imaging process is part of NeRF training.",
    }

    assert citation_card_polish_cache_key(
        {**detail, "render_locale": "zh"}
    ) != citation_card_polish_cache_key({**detail, "render_locale": "en"})


def test_polish_citation_card_rejects_output_in_the_wrong_language() -> None:
    detail = {
        "render_locale": "en",
        "source_name": "Fixture.pdf",
        "heading_path": "Abstract",
        "answer_claim": "SCINeRF uses the SCI physical model during NeRF training.",
        "evidence_quote": "The physical imaging process is part of NeRF training.",
    }

    def fake_llm(**kwargs: object) -> str:
        assert kwargs["render_locale"] == "en"
        return '{"card_takeaway":"原文把物理成像过程纳入神经辐射场训练。"}'

    out = polish_citation_card_detail(detail, llm_fn=fake_llm)

    assert out["citation_card_polish_status"] == "empty"
    assert out["citation_card_polish_source"] == "llm_empty"
