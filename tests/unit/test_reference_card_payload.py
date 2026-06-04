from __future__ import annotations

from api.reference_card_payload import build_ref_card_ui_payload


def test_build_ref_card_ui_payload_preserves_frontend_contract() -> None:
    payload = build_ref_card_ui_payload(
        display_name="Paper A",
        heading_path="Paper A / Related Work",
        section_label="Related Work",
        subsection_label="",
        page_start=2,
        page_end=3,
        score=8.5,
        score_pending=False,
        score_tier="high",
        summary_line="This section discusses ADMM.",
        summary_kind="section",
        summary_surface={
            "summary_kind": "abstract",
            "summary_label": "Abstract",
            "summary_title": "What this says",
        },
        summary_generation="section_grounded",
        summary_basis_meta={
            "summary_generation": "translated_abstract",
            "summary_basis": "metadata",
        },
        summary_source="navigation",
        primary_evidence_heading_path="Paper A / Related Work",
        primary_evidence={"snippet": "ADMM baseline"},
        why_line="Related Work names ADMM.",
        why_generation="deterministic_grounded",
        why_basis_meta={
            "why_generation": "llm_polished",
            "why_basis": "heading and snippet",
        },
        anchor_target_kind="section",
        anchor_target_number=2,
        anchor_match_score=7.0,
        explicit_doc_match_score=1.0,
        semantic_badges=["ADMM"],
        can_open=True,
        citation_meta={"doi": "10.0000/example"},
        source_path="db/paper.md",
        reader_open={"snippet": "ADMM baseline", "page": 2},
    )

    assert payload["display_name"] == "Paper A"
    assert payload["heading_path"] == "Paper A / Related Work"
    assert payload["summary_kind"] == "abstract"
    assert payload["summary_generation"] == "translated_abstract"
    assert payload["why_generation"] == "llm_polished"
    assert payload["primary_evidence"] == {"snippet": "ADMM baseline"}
    assert payload["reader_open"] == {"snippet": "ADMM baseline", "page": 2}
    assert payload["semantic_badges"] == ["ADMM"]
    assert payload["can_open"] is True
    assert payload["card_view"]["version"] == 1
    assert payload["card_view"]["route"] == "references"
    assert payload["card_view"]["header"]["title"] == "Paper A"
    sections = {section["id"]: section for section in payload["card_view"]["sections"]}
    assert sections["summary"]["label"] == "Abstract"
    assert sections["summary"]["text"] == "This section discusses ADMM."
    assert sections["why"]["text"] == "Related Work names ADMM."
    assert sections["location"]["text"] == "Paper A / Related Work · pp. 2-3"


def test_build_ref_card_ui_payload_normalizes_optional_mappings() -> None:
    payload = build_ref_card_ui_payload(
        display_name="Paper B",
        heading_path="",
        section_label="",
        subsection_label="",
        page_start=0,
        page_end=0,
        score=None,
        score_pending=True,
        score_tier="",
        summary_line="",
        summary_kind="section",
        summary_surface=None,
        summary_generation="section_grounded",
        summary_basis_meta=None,
        summary_source="fallback",
        primary_evidence_heading_path="",
        primary_evidence=None,
        why_line="",
        why_generation="fallback",
        why_basis_meta=None,
        anchor_target_kind="",
        anchor_target_number=0,
        anchor_match_score=0.0,
        explicit_doc_match_score=0.0,
        semantic_badges=(),
        can_open=False,
        citation_meta=None,
        source_path="",
        reader_open=None,
    )

    assert payload["summary_kind"] == "section"
    assert payload["summary_generation"] == "section_grounded"
    assert payload["primary_evidence"] == {}
    assert payload["citation_meta"] == {}
    assert payload["reader_open"] == {}
    assert payload["semantic_badges"] == []
    assert payload["card_view"]["sections"] == []


def test_build_ref_card_ui_payload_uses_english_render_locale_defaults() -> None:
    payload = build_ref_card_ui_payload(
        display_name="Paper C",
        heading_path="Paper C / Abstract",
        section_label="Abstract",
        subsection_label="",
        page_start=1,
        page_end=1,
        score=6.0,
        score_pending=False,
        score_tier="medium",
        summary_line="This abstract introduces a part-based image-loop network.",
        summary_kind="section",
        summary_surface={},
        summary_generation="section_grounded",
        summary_basis_meta={},
        summary_source="navigation",
        primary_evidence_heading_path="Paper C / Abstract",
        primary_evidence={"snippet": "part-based image-loop network"},
        why_line="The abstract matches the question about low-sampling reconstruction.",
        why_generation="deterministic_grounded",
        why_basis_meta={},
        anchor_target_kind="section",
        anchor_target_number=1,
        anchor_match_score=6.0,
        explicit_doc_match_score=1.0,
        semantic_badges=[],
        can_open=True,
        citation_meta={},
        source_path="db/paper-c.md",
        reader_open={},
        render_locale="en",
    )

    assert payload["render_locale"] == "en"
    sections = {section["id"]: section for section in payload["card_view"]["sections"]}
    assert sections["summary"]["label"] == "Guide"
    assert sections["summary"]["title"] == "What This Evidence Shows"
    assert sections["why"]["label"] == "Relevance"
    assert sections["location"]["label"] == "Location"
    assert sections["location"]["title"] == "Source location"
    assert sections["location"]["text"] == "Paper C / Abstract / p. 1"
