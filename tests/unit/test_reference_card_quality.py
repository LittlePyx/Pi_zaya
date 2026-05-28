from __future__ import annotations

from api.reference_card_quality import (
    attach_ref_card_polish_contract,
    attach_refs_pack_polish_contract,
    citation_detail_quality,
    citation_shelf_item_quality,
    ref_card_hit_quality,
    refs_pack_has_full_llm_copy,
    summarize_citation_detail_quality,
    summarize_citation_shelf_quality,
    summarize_ref_card_hit_quality,
)
from api.reference_card_payload import build_ref_card_ui_payload


def test_ref_card_polish_contract_marks_full_llm_card():
    ui = attach_ref_card_polish_contract(
        {
            "summary_kind": "guide",
            "summary_generation": "llm_grounded",
            "why_generation": "llm_grounded",
        }
    )

    assert ui["polish_status"] == "full"
    assert ui["summary_polish_status"] == "full"
    assert ui["why_polish_status"] == "full"
    assert ui["polish_source"] == "llm"


def test_ref_card_polish_contract_marks_pending_before_llm_copy():
    ui = attach_ref_card_polish_contract(
        {
            "summary_kind": "guide",
            "summary_generation": "pending_section_seed",
            "why_generation": "pending_focus_seed",
            "score_pending": True,
        },
        hit_meta={"ref_pack_state": "pending"},
    )

    assert ui["polish_status"] == "pending"
    assert ui["summary_polish_status"] == "pending"
    assert ui["why_polish_status"] == "pending"


def test_refs_pack_polish_contract_counts_mixed_cards():
    pack = attach_refs_pack_polish_contract(
        {
            "display_state": "ready",
            "hits": [
                {
                    "ui_meta": {
                        "summary_kind": "guide",
                        "summary_generation": "llm_grounded",
                        "why_generation": "llm_grounded",
                    }
                },
                {
                    "ui_meta": {
                        "summary_kind": "guide",
                        "summary_generation": "deterministic_grounded",
                        "why_generation": "deterministic_grounded",
                    }
                },
            ],
        }
    )

    assert pack["polish_status"] == "heuristic"
    assert pack["polish_counts"]["full"] == 1
    assert pack["polish_counts"]["heuristic"] == 1
    assert refs_pack_has_full_llm_copy(pack) is False


def test_refs_pack_full_llm_copy_requires_all_visible_cards_full():
    assert refs_pack_has_full_llm_copy(
        {
            "hits": [
                {
                    "ui_meta": {
                        "summary_kind": "guide",
                        "summary_generation": "llm_grounded",
                        "why_generation": "llm_grounded",
                    }
                }
            ]
        }
    )


def test_ref_card_payload_builder_attaches_polish_contract():
    payload = build_ref_card_ui_payload(
        display_name="Demo.pdf",
        heading_path="2. Method",
        section_label="2. Method",
        subsection_label="",
        page_start=0,
        page_end=0,
        score=9.2,
        score_pending=False,
        score_tier="high",
        summary_line="A concise LLM-grounded summary.",
        summary_kind="guide",
        summary_surface={},
        summary_generation="llm_grounded",
        summary_basis_meta={},
        summary_source="prompt_aligned",
        primary_evidence_heading_path="2. Method",
        primary_evidence={},
        why_line="A concise LLM-grounded relevance note.",
        why_generation="llm_grounded",
        why_basis_meta={},
        anchor_target_kind="",
        anchor_target_number=0,
        anchor_match_score=0.0,
        explicit_doc_match_score=0.0,
        semantic_badges=[],
        can_open=True,
        citation_meta={},
        source_path="demo.en.md",
        reader_open={},
    )

    assert payload["polish_status"] == "full"
    assert payload["polish_contract_version"] == 1
    assert payload["card_view"]["quality"]["label"] == "full"
    assert payload["card_view"]["quality"]["source"] == "llm"
    sections = {section["id"]: section for section in payload["card_view"]["sections"]}
    assert sections["summary"]["text"] == "A concise LLM-grounded summary."
    assert sections["why"]["text"] == "A concise LLM-grounded relevance note."


def test_ref_card_polish_contract_unwraps_source_excerpt_summary():
    ui = attach_ref_card_polish_contract(
        {
            "display_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
            "source_path": "dl-spi-review.en.md",
            "summary_line": (
                "\u539f\u6587\u7247\u6bb5\u5199\u5230\uff1a\u201c"
                "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning: "
                "However, the limited image quality still hinders practical application."
                "\u201d"
            ),
            "why_line": "This card explains a concrete limitation of deep-learning single-pixel imaging.",
            "summary_generation": "deterministic_grounded",
            "why_generation": "deterministic_grounded",
        }
    )

    assert ui["summary_line"].startswith("However, the limited image quality")
    assert "\u539f\u6587\u7247\u6bb5\u5199\u5230" not in ui["summary_line"]
    assert "Advances and Challenges" not in ui["summary_line"]
    assert ui["card_view"]["summary"].startswith("However, the limited image quality")


def test_ref_card_hit_quality_accepts_grounded_openable_card():
    quality = ref_card_hit_quality(
        {
            "text": "The method maps low-dimensional measurements back to target images.",
            "meta": {"source_path": "demo.en.md", "ref_pack_state": "ready"},
            "ui_meta": {
                "display_name": "Demo SPI paper",
                "source_path": "demo.en.md",
                "heading_path": "4. Strategy and Advantages / Data-driven strategy",
                "summary_line": "This card explains how the model maps compressed measurements back to images.",
                "why_line": "It directly supports the user's question about reconstruction quality under low sampling.",
                "polish_status": "full",
                "can_open": True,
                "reader_open": {
                    "sourcePath": "demo.en.md",
                    "headingPath": "4. Strategy and Advantages",
                    "blockId": "blk-1",
                    "anchorId": "p-1",
                    "snippet": "The encoder samples the image into low-dimensional measurements.",
                },
            },
        }
    )

    assert quality["ok"] is True
    assert quality["score"] == 1.0


def test_citation_detail_quality_accepts_complete_has_attracted_evidence():
    quality = citation_detail_quality(
        {
            "num": 2,
            "anchor": "dl-a2",
            "source_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
            "source_path": "dl-spi-review.en.md",
            "heading_path": "Abstract",
            "card_evidence": (
                "Single-pixel imaging technology can capture images outside conventional focal plane arrays. "
                "Recently, single-pixel imaging based on deep learning has attracted a lot of attention "
                "due to its exceptional reconstruction quality and fast reconstruction speed."
            ),
            "card_claim": "The answer uses this review as the deep-learning SPI overview.",
        }
    )

    assert quality["ok"] is True


def test_ref_card_hit_quality_rejects_template_duplicate_and_broken_copy():
    quality = ref_card_hit_quality(
        {
            "text": "## Foveated single-pixel imaging has attrac...",
            "ui_meta": {
                "display_name": "Foveated SPI",
                "summary_line": "This hit is directly relevant to the user question.",
                "why_line": "This hit is directly relevant to the user question.",
                "polish_status": "sparkly",
                "reader_open": {"sourcePath": "demo.en.md", "snippet": "## Foveated single-pixel imaging has attrac..."},
            },
        },
        forbidden_phrases=["directly relevant"],
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is False
    assert "ref_card_template_phrase_visible" in names
    assert "ref_card_duplicate_summary_why" in names
    assert "ref_card_forbidden_phrase" in names
    assert "ref_card_unknown_polish_status" in names
    assert "ref_card_raw_markdown_visible" in names
    assert "ref_card_broken_evidence" in names


def test_summarize_ref_card_hit_quality_indexes_failures():
    summary = summarize_ref_card_hit_quality(
        [
            {
                "ui_meta": {
                    "display_name": "Good",
                    "summary_line": "This card has a focused summary for the answer.",
                    "why_line": "It explains why this source belongs in the answer.",
                }
            },
            {"ui_meta": {"summary_line": "short", "why_line": "short"}},
        ]
    )

    assert summary["ok"] is False
    assert summary["count"] == 2
    assert summary["ok_count"] == 1
    assert any(item["index"] == 2 and item["name"] == "ref_card_summary_too_short" for item in summary["failures"])


def test_citation_detail_quality_accepts_grounded_system_a_card():
    quality = citation_detail_quality(
        {
            "num": 1,
            "anchor": "a1",
            "source_name": "Demo.pdf",
            "source_path": "demo.en.md",
            "heading_path": "2. Method / Reconstruction",
            "evidence_quote": "The method maps low-dimensional measurements back to target images with a learned decoder.",
            "answer_claim": "The method improves reconstruction from fewer measurements.",
            "support_relation": "The quoted sentence explains the encoder-decoder measurement mapping.",
        }
    )

    assert quality["ok"] is True
    assert quality["route"] == "system_a"


def test_citation_detail_quality_rejects_visible_weak_system_a_binding():
    quality = citation_detail_quality(
        {
            "num": 1,
            "anchor": "a1",
            "source_name": "3D single-pixel video.pdf",
            "source_path": "demo.en.md",
            "heading_path": "Methods / Photometric stereo",
            "evidence_quote": "Photometric stereo estimates surface orientation from different illumination directions.",
            "answer_claim": "Hadamard subsampling is useful for real-time low-sampling imaging.",
            "binding_status": "candidate",
            "binding_confidence": 0.35,
            "card_quality_flags": ["candidate_binding"],
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is False
    assert "system_a_weak_binding_visible" in names


def test_citation_detail_quality_rejects_raw_markdown_and_fragmented_evidence():
    quality = citation_detail_quality(
        {
            "num": 2,
            "anchor": "a2",
            "source_name": "Foveated SPI.pdf",
            "heading_path": "INTRODUCTION",
            "evidence_quote": "## Foveated single-pixel imaging has attrac...",
            "answer_claim": "This is a method card.",
            "support_relation": "This quote supports the answer.",
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is False
    assert "raw_markdown_visible" in names
    assert "system_a_broken_evidence" in names


def test_citation_detail_quality_rejects_duplicate_visible_card_text():
    quality = citation_detail_quality(
        {
            "num": 1,
            "anchor": "a1",
            "source_name": "Demo.pdf",
            "heading_path": "2. Method",
            "card_takeaway": "深度学习模型把低维测量映射回目标图像，从而提升重建质量。",
            "card_evidence": "深度学习模型把低维测量映射回目标图像，从而提升重建质量。",
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is False
    assert "duplicate_visible_card_text" in names


def test_citation_detail_quality_rejects_metadata_repeated_in_card_copy():
    quality = citation_detail_quality(
        {
            "num": 2,
            "anchor": "r2",
            "is_inpaper": True,
            "source_name": "Current paper.pdf",
            "title": "Optical imaging by means of two-photon quantum entanglement",
            "venue": "Physical Review A",
            "year": "1995",
            "raw": "Pittman T, Shih Y. Optical imaging by means of two-photon quantum entanglement. Physical Review A, 1995.",
            "heading_path": "1. Introduction",
            "answer_claim": "单像素成像可以降低成像成本。",
            "citation_context": "Unlike traditional focal plane array detectors, SPI only adopts a SPD to collect echo signals.",
            "card_takeaway": "这篇发表于 Physical Review A 1995 的论文值得打开。",
            "system_b_trace_complete": True,
            "system_b_trace_score": 0.8,
            "system_b_trace_reference": "Pittman T, Shih Y. Optical imaging by means of two-photon quantum entanglement.",
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is False
    assert "narrative_metadata_repeated" in names


def test_citation_detail_quality_accepts_system_b_support_relation_language():
    quality = citation_detail_quality(
        {
            "num": 4,
            "anchor": "r4",
            "is_inpaper": True,
            "source_name": "SCINeRF Neural Radiance Fields from a Snapshot Compressive Image.pdf",
            "title": "Distributed Optimization and Statistical Learning via ADMM",
            "raw": "Boyd et al. Distributed Optimization and Statistical Learning via ADMM.",
            "heading_path": "SCINeRF / 2. Related Work",
            "answer_claim": "ADMM is prior optimization background, not a new SCINeRF invention.",
            "citation_context": "Most existing methods employ ADMM-based optimization for snapshot compressive imaging.",
            "card_takeaway": "This upstream work provides the optimization framework behind the cited ADMM method.",
            "card_support_explanation": "It maps the answer claim back to a reference cited by the current paper.",
            "system_b_trace_complete": True,
            "system_b_trace_score": 0.82,
            "system_b_trace_reference": "Boyd et al. Distributed Optimization and Statistical Learning via ADMM.",
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is True
    assert "narrative_metadata_repeated" not in names


def test_citation_detail_quality_accepts_grounded_system_b_card():
    quality = citation_detail_quality(
        {
            "num": 4,
            "anchor": "r4",
            "is_inpaper": True,
            "source_name": "SCINeRF Neural Radiance Fields from a Snapshot Compressive Image.pdf",
            "title": "Distributed Optimization and Statistical Learning via ADMM",
            "raw": "Boyd et al. Distributed Optimization and Statistical Learning via ADMM.",
            "heading_path": "SCINeRF / 2. Related Work / Snapshot Compressive Imaging",
            "answer_claim": "ADMM is prior optimization background, not a new SCINeRF invention.",
            "citation_context": "Most existing methods employ ADMM-based optimization for snapshot compressive imaging.",
            "upstream_work_role": "This upstream work provides the optimization framework behind the cited ADMM method.",
            "user_question_relation": "The citation shows ADMM is prior work rather than a new SCINeRF contribution.",
            "system_b_trace_complete": True,
            "system_b_trace_score": 0.82,
            "system_b_trace_steps": ["答案句", "当前论文引用处", "上游文献"],
            "system_b_trace_answer": "ADMM is prior optimization background, not a new SCINeRF invention.",
            "system_b_trace_context": "Most existing methods employ ADMM-based optimization for snapshot compressive imaging.",
            "system_b_trace_reference": "Boyd et al. Distributed Optimization and Statistical Learning via ADMM.",
        }
    )

    assert quality["ok"] is True
    assert quality["route"] == "system_b"


def test_citation_detail_quality_rejects_weak_system_b_card():
    quality = citation_detail_quality(
        {
            "ref_num": 3,
            "is_inpaper": True,
            "source_name": "Paper.pdf",
            "raw": "Missing cone problem and low-pass distortion.",
            "heading_path": "Unknown location",
            "citation_context": "Missing cone problem and low-pass distortion.",
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is False
    assert "missing_click_anchor" in names
    assert "system_b_missing_takeaway" in names
    assert "system_b_missing_locator" in names
    assert "system_b_missing_answer_claim" in names


def test_summarize_citation_detail_quality_counts_routes_and_failures():
    summary = summarize_citation_detail_quality(
        [
            {
                "num": 1,
                "anchor": "a1",
                "source_name": "Demo.pdf",
                "heading_path": "Abstract",
                "evidence_quote": "A complete evidence sentence explains the answer in enough detail.",
            },
            {
                "is_inpaper": True,
                "source": "inline_marker",
                "ref_num": "4",
                "raw": "inline marker only",
            },
        ]
    )

    assert summary["ok"] is False
    assert summary["route_counts"] == {"system_a": 1, "system_b": 1}
    assert summary["ok_route_counts"]["system_a"] == 1
    assert any(item["name"] == "inline_marker_not_rendered" for item in summary["failures"])
    assert summary["system_b_audit"]["system_b_total"] == 1
    assert summary["system_b_audit"]["needs_review_count"] == 1
    assert summary["system_b_audit"]["review_examples"]


def test_summarize_citation_detail_quality_audits_system_b_sources():
    summary = summarize_citation_detail_quality(
        [
            {
                "num": 4,
                "anchor": "r4",
                "is_inpaper": True,
                "citation_route": "system_b",
                "routing_reason": "structured_cite",
                "source_name": "SCINeRF.pdf",
                "title": "Distributed Optimization and Statistical Learning via ADMM",
                "raw": "Boyd et al. Distributed Optimization and Statistical Learning via ADMM.",
                "answer_claim": "ADMM is prior optimization background.",
                "citation_context": "The current paper cites ADMM while discussing optimization background.",
                "citation_context_source": "source_markdown",
                "location_label": "Related Work",
                "card_takeaway": "这篇上游文献提供 ADMM 优化背景。",
                "system_b_trace_complete": True,
                "system_b_trace_score": 0.82,
                "system_b_trace_source": "source_markdown",
                "system_b_trace_flags": [],
            },
            {
                "num": 24,
                "anchor": "r24",
                "is_inpaper": True,
                "citation_route": "system_b",
                "routing_reason": "reference_index_fallback",
                "source_name": "SCI.pdf",
                "title": "Single-shot compressive spectral imaging",
                "raw": "Gehm et al. Single-shot compressive spectral imaging.",
                "answer_claim": "This is an upstream source.",
                "citation_context": "This is an upstream source.",
                "citation_context_source": "answer_context",
                "location_label": "SCI.pdf",
                "card_takeaway": "这篇文献提供单次压缩光谱成像背景。",
                "system_b_trace_complete": False,
                "system_b_trace_score": 0.32,
                "system_b_trace_source": "answer_context",
                "system_b_trace_flags": ["answer_context_only"],
            },
        ]
    )

    audit = summary["system_b_audit"]
    assert audit["system_b_total"] == 2
    assert audit["structured_cite_count"] == 1
    assert audit["reference_index_fallback_count"] == 1
    assert audit["source_markdown_count"] == 1
    assert audit["answer_context_only_count"] == 1
    assert audit["trace_complete_count"] == 1
    assert audit["needs_review_count"] == 1


def test_citation_shelf_item_quality_accepts_exportable_system_b_item():
    quality = citation_shelf_item_quality(
        {
            "num": 4,
            "anchor": "r4",
            "is_inpaper": True,
            "source_name": "SCINeRF.pdf",
            "title": "Distributed Optimization and Statistical Learning via ADMM",
            "authors": "Boyd et al.",
            "venue": "Foundations and Trends in Machine Learning",
            "year": "2011",
            "doi": "10.1561/2200000016",
            "raw": "Boyd et al. Distributed Optimization and Statistical Learning via ADMM.",
            "card_view": {
                "header": {
                    "title": "Distributed Optimization and Statistical Learning via ADMM",
                    "subtitle": "SCINeRF / Related Work",
                },
                "sections": [
                    {
                        "id": "takeaway",
                        "text": "This upstream paper provides the ADMM optimization framework used by earlier SCI reconstruction methods.",
                    }
                ],
                "summary": "This upstream paper provides the ADMM optimization framework used by earlier SCI reconstruction methods.",
                "quality": {"flags": []},
            },
        }
    )

    assert quality["ok"] is True
    assert quality["route"] == "system_b"
    assert quality["title"].startswith("Distributed Optimization")


def test_citation_shelf_item_quality_does_not_show_candidate_review_when_identity_is_complete():
    quality = citation_shelf_item_quality(
        {
            "num": 4,
            "anchor": "r4",
            "is_inpaper": True,
            "source_name": "Distributed Optimization and Statistical Learning via ADMM.pdf",
            "title": "Distributed Optimization and Statistical Learning via ADMM",
            "authors": "Boyd et al.",
            "venue": "Foundations and Trends in Machine Learning",
            "year": "2011",
            "doi": "10.1561/2200000016",
            "raw": "Boyd et al. Distributed Optimization and Statistical Learning via ADMM.",
            "summary_line": "This upstream paper provides the ADMM optimization framework used by earlier SCI reconstruction methods.",
            "external_metadata_status": "candidate",
            "external_metadata_reason": "Low-similarity metrics candidate kept as a clue.",
            "external_doi": "10.1561/2200000016",
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is True
    assert quality["metadata"]["metadata_ready"] is True
    assert quality["metadata"]["review_needed"] is False
    assert "shelf_untrusted_external_metadata_visible" not in names


def test_citation_shelf_item_quality_treats_system_a_raw_as_evidence_not_bibliography():
    quality = citation_shelf_item_quality(
        {
            "num": 1,
            "anchor": "a1",
            "source_name": "SCINeRF Neural Radiance Fields from a Snapshot Compressive Image.pdf",
            "source_path": "scinerf.en.md",
            "title": "3.2 Image Formation Model of Video SCI",
            "raw": "Due to the mask modulation, the image restoration problem is not ill-posed anymore.",
            "evidence_quote": "Due to the mask modulation, the image restoration problem is not ill-posed anymore.",
            "summary_line": "This evidence explains why the SCI forward model can recover virtual frames from a compressed image.",
        }
    )

    assert quality["ok"] is True
    assert quality["metadata"]["bibliographic"] is False
    assert quality["metadata"]["metadata_ready"] is True
    assert quality["metadata"]["review_needed"] is False


def test_citation_shelf_item_quality_rejects_placeholder_summary_and_markdown():
    quality = citation_shelf_item_quality(
        {
            "num": 1,
            "anchor": "a1",
            "source_path": "demo.en.md",
            "title": "INTRODUCTION",
            "summary_line": "No summary available",
            "evidence_quote": "## Broken evidence has attrac...",
        }
    )

    names = {item["name"] for item in quality["failures"]}
    assert quality["ok"] is False
    assert "shelf_template_phrase_visible" in names
    assert "shelf_summary_too_short" in names


def test_summarize_citation_shelf_quality_indexes_failures():
    summary = summarize_citation_shelf_quality(
        [
            {
                "num": 1,
                "anchor": "a1",
                "source_name": "Demo citation shelf paper.pdf",
                "source_path": "demo.en.md",
                "summary_line": "This shelf note explains why the cited evidence is useful for the answer.",
            },
            {"num": 2, "anchor": "a2", "summary_line": "short"},
        ]
    )

    assert summary["ok"] is False
    assert summary["count"] == 2
    assert summary["ok_count"] == 1
    assert any(item["index"] == 2 and item["name"] == "shelf_missing_source_identity" for item in summary["failures"])
