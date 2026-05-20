from __future__ import annotations

from api.reference_card_quality import (
    attach_ref_card_polish_contract,
    attach_refs_pack_polish_contract,
    refs_pack_has_full_llm_copy,
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
