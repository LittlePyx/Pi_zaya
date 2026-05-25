from __future__ import annotations

import re

from ui.refs_renderer import _annotate_inpaper_citations_with_hover_meta, _should_route_numeric_to_system_b


def test_system_a_citation_detail_carries_reader_card_fields() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "The method is explained in the retrieved paper [1].",
        [
            {
                "text": "Most existing methods employ alternating direction method of multipliers (ADMM) [4].",
                "meta": {
                    "source_path": "db/demo/paper.en.md",
                    "heading_path": "2. Related Work",
                    "evidence_quote": "Most existing methods employ alternating direction method of multipliers (ADMM) [4].",
                    "primary_block_id": "blk_001",
                    "primary_anchor_id": "p_001",
                    "anchor_kind": "sentence",
                    "page_start": 2,
                    "page_end": 3,
                    "ref_rank": {"display_score": 8.75, "why": "Related Work names ADMM as prior optimization machinery."},
                },
            }
        ],
        anchor_ns="test",
    )

    assert "#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["is_inpaper"] is False
    assert detail["heading_path"] == "2. Related Work"
    assert "method is explained" in detail["answer_claim"]
    assert detail["evidence_quote"].startswith("Most existing methods employ")
    assert detail["evidence_source"] == "retrieval_hit"
    assert "2. Related Work" in detail["location_label"]
    assert "pp. 2-3" in detail["location_label"]
    assert "Related Work names ADMM" in detail["support_relation"]
    assert detail["summary_source"] == "retrieval_hit"
    assert detail["block_id"] == "blk_001"
    assert detail["anchor_id"] == "p_001"
    assert detail["anchor_kind"] == "sentence"
    assert detail["page_start"] == 2
    assert detail["page_end"] == 3
    assert detail["score"] == 8.75
    assert "ADMM" in detail["why_line"]
    assert detail["card_kind"] == "answer_evidence"
    assert detail["card_title"] == "paper.pdf"
    assert detail["card_subtitle"].startswith("2. Related Work")
    assert detail["card_locator"].startswith("2. Related Work")
    assert detail["card_evidence"].startswith("Most existing methods employ")
    assert detail["card_quality_label"] in {"候选依据", "证据匹配"}


def test_system_a_suppresses_link_when_answer_claim_conflicts_with_hit_topic() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "Interferometric (iSCAT) microscopy detects unlabeled proteins through "
            "scattering contrast [2]."
        ),
        [
            {
                "text": "Adaptive foveated single-pixel imaging uses dynamic supersampling.",
                "meta": {
                    "source_path": "db/demo/foveated.en.md",
                    "heading_path": "INTRODUCTION / Foveated single-pixel imaging",
                },
            },
            {
                "text": (
                    "Structured detection for simultaneous super-resolution and optical "
                    "sectioning in laser scanning microscopy."
                ),
                "meta": {
                    "source_path": "db/demo/structured-detection.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": (
                        "This work proposes structured detection for optical sectioning "
                        "in laser scanning microscopy."
                    ),
                },
            },
        ],
        anchor_ns="test",
    )

    assert "[2](#kb-cite-" not in rendered
    assert "Interferometric (iSCAT) microscopy detects unlabeled proteins" in rendered
    assert "Structured detection" not in rendered
    assert details == []


def test_system_a_marks_grounded_binding_with_shared_domain_terms() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "Structured detection improves optical sectioning in laser scanning microscopy [1].",
        [
            {
                "text": (
                    "Structured detection enables simultaneous super-resolution and "
                    "optical sectioning in laser scanning microscopy."
                ),
                "meta": {
                    "source_path": "db/demo/structured-detection.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": (
                        "Structured detection enables simultaneous super-resolution "
                        "and optical sectioning in laser scanning microscopy."
                    ),
                },
            },
        ],
        anchor_ns="test",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["binding_status"] == "grounded"
    assert detail["binding_confidence"] >= 0.8
    assert "structured detection" in detail["binding_overlap_terms"]
    assert detail["card_quality_label"] == "证据匹配"
    assert detail["card_warning"] == ""
    assert "答案句" in detail["support_relation"] or "answer sentence" in detail["support_relation"]


def test_system_a_reuses_one_card_for_duplicate_evidence_hits() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "Structured detection improves optical sectioning in laser scanning "
            "microscopy [1] and is the same evidence when mentioned again [2]."
        ),
        [
            {
                "text": "Structured detection improves optical sectioning in laser scanning microscopy.",
                "meta": {
                    "source_path": "db/demo/structured-detection.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": "Structured detection improves optical sectioning in laser scanning microscopy.",
                    "primary_block_id": "abstract-1",
                    "primary_anchor_id": "sent-1",
                    "anchor_kind": "sentence",
                },
            },
            {
                "text": "Structured detection improves optical sectioning in laser scanning microscopy.",
                "meta": {
                    "source_path": "db/demo/structured-detection.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": "Structured detection improves optical sectioning in laser scanning microscopy.",
                    "primary_block_id": "abstract-1",
                    "primary_anchor_id": "sent-1",
                    "anchor_kind": "sentence",
                },
            },
        ],
        anchor_ns="test",
    )

    anchors = re.findall(r"\[(?:1|2)\]\(#([^) \"\n]+)", rendered)
    assert len(anchors) == 1
    assert len(set(anchors)) == 1
    assert len(details) == 1
    assert details[0]["linked_nums"] == [1, 2]
    assert details[0]["evidence_fingerprint"]


def test_system_a_reuses_repeated_same_number_for_same_evidence() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "Adaptive sampling puts high resolution near the fovea [1].\n"
            "Hardware construction uses a DMD with projection and detection paths [1]."
        ),
        [
            {
                "text": (
                    "## Foveated single-pixel imaging\n"
                    "Single-pixel imaging can use dynamic supersampling with a DMD."
                ),
                "meta": {
                    "source_path": "db/demo/foveated.en.md",
                    "heading_path": "INTRODUCTION",
                    "evidence_quote": (
                        "## Foveated single-pixel imaging\n"
                        "Single-pixel imaging can use dynamic supersampling with a DMD."
                    ),
                    "primary_block_id": "intro-1",
                    "primary_anchor_id": "sent-1",
                    "anchor_kind": "paragraph",
                },
            },
        ],
        anchor_ns="test",
    )

    anchors = re.findall(r"\[1\]\(#([^) \"\n]+)", rendered)
    assert len(anchors) == 2
    assert len(set(anchors)) == 1
    assert len(details) == 1
    assert "occurrence_specific_claim" not in details[0]["card_quality_flags"]
    assert "Hardware construction uses a DMD" in details[0]["answer_claim"]
    assert "##" not in details[0]["card_evidence"]


def test_system_a_prefers_primary_evidence_location_from_hit_ui_meta() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "Deep learning helps with difficult scattering cases [1].",
        [
            {
                "text": "# Paper title\nAuthor A\nSingle-pixel imaging based on deep learning has attrac",
                "meta": {
                    "source_path": "db/demo/lpr.en.md",
                    "ref_best_heading_path": "5. Single-Pixel Imaging Realizations with Deep Learning / 5.4. Optical Encryption",
                },
                "ui_meta": {
                    "heading_path": "5.2. Imaging Through Scattering Media",
                    "primary_evidence": {
                        "heading_path": "5.2. Imaging Through Scattering Media",
                        "snippet": (
                            "Turbulence-immune imaging is a classical challenge in the field of imaging "
                            "through scattering weak media. DL has exhibited remarkable efficacy in addressing this problem"
                        ),
                        "block_id": "blk_scattering",
                        "anchor_id": "p_42",
                        "anchor_kind": "paragraph",
                    },
                },
            }
        ],
        anchor_ns="test",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["heading_path"] == "5.2. Imaging Through Scattering Media"
    assert detail["card_locator"].startswith("5.2. Imaging Through Scattering Media")
    assert detail["block_id"] == "blk_scattering"
    assert detail["anchor_id"] == "p_42"
    assert "Optical Encryption" not in detail["card_locator"]
    assert "attrac" not in detail["card_evidence"]


def test_system_a_prefers_reader_open_primary_evidence_when_direct_primary_missing() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "Light-field microscopy reconstructs three-dimensional information [1].",
        [
            {
                "text": (
                    "# Quantum correlation light-field microscope with extreme depth of field\n"
                    "Yingwen Zhang, Yuhang Qin, Wenhao Li"
                ),
                "meta": {
                    "source_path": "db/demo/qclfm.en.md",
                    "heading_path": "I. INTRODUCTION",
                },
                "ui_meta": {
                    "reader_open": {
                        "primaryEvidence": {
                            "headingPath": "I. INTRODUCTION / Light-field microscope",
                            "highlightSnippet": (
                                "Conventional light-field microscope designs typically make use "
                                "of a microlens array to record spatial and angular information."
                            ),
                            "blockId": "intro-light-field",
                            "anchorId": "sent-light-field",
                            "anchorKind": "sentence",
                        }
                    }
                },
            }
        ],
        anchor_ns="test",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["heading_path"] == "I. INTRODUCTION / Light-field microscope"
    assert detail["card_locator"].startswith("I. INTRODUCTION / Light-field microscope")
    assert detail["block_id"] == "intro-light-field"
    assert detail["anchor_id"] == "sent-light-field"
    assert "microlens array" in detail["card_evidence"]
    assert "Yingwen Zhang" not in detail["card_evidence"]
    assert "##" not in detail["card_evidence"]


def test_system_a_does_not_route_to_system_b_from_reference_title_words() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "This paper is useful for learning deep-learning single-pixel imaging [1].",
        [
            {
                "text": "Deep learning improves single-pixel imaging reconstruction quality.",
                "meta": {
                    "source_path": "db/demo/deep-spi.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": "Deep learning improves single-pixel imaging reconstruction quality.",
                },
            }
        ],
        anchor_ns="test",
    )

    assert "[1](#kb-cite-" in rendered
    assert len(details) == 1
    assert details[0]["citation_route"] == "system_a"
    assert details[0]["card_kind"] == "answer_evidence"
    assert not _should_route_numeric_to_system_b(
        "This paper is useful for learning deep-learning single-pixel imaging [1].",
        {
            "title": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
            "raw": "A review with references and prior work.",
        },
    )


def test_system_a_keeps_distinct_cards_for_distinct_evidence_locations() -> None:
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        (
            "Structured detection improves optical sectioning [1], while dynamic "
            "supersampling changes the sampling pattern [2]."
        ),
        [
            {
                "text": "Structured detection improves optical sectioning in laser scanning microscopy.",
                "meta": {
                    "source_path": "db/demo/microscopy.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": "Structured detection improves optical sectioning in laser scanning microscopy.",
                    "primary_block_id": "abstract-1",
                    "primary_anchor_id": "sent-1",
                    "anchor_kind": "sentence",
                },
            },
            {
                "text": "Dynamic supersampling allocates more samples to important image regions.",
                "meta": {
                    "source_path": "db/demo/microscopy.en.md",
                    "heading_path": "Method",
                    "evidence_quote": "Dynamic supersampling allocates more samples to important image regions.",
                    "primary_block_id": "method-2",
                    "primary_anchor_id": "sent-2",
                    "anchor_kind": "sentence",
                },
            },
        ],
        anchor_ns="test",
    )

    anchors = re.findall(r"\[(?:1|2)\]\(#([^) \"\n]+)", rendered)
    assert len(anchors) == 2
    assert len(set(anchors)) == 2
    assert len(details) == 2
    assert [d["linked_nums"] for d in details] == [[1], [2]]
