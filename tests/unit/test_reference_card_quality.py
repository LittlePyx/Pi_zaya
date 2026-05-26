from __future__ import annotations

from api.reference_card_quality import (
    attach_ref_card_polish_contract,
    attach_refs_pack_polish_contract,
    citation_detail_quality,
    refs_pack_has_full_llm_copy,
    summarize_citation_detail_quality,
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


def test_citation_detail_quality_accepts_grounded_system_b_card():
    quality = citation_detail_quality(
        {
            "num": 4,
            "anchor": "r4",
            "is_inpaper": True,
            "source_name": "SCINeRF.pdf",
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
