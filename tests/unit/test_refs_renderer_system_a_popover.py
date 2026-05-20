from __future__ import annotations

from ui.refs_renderer import _annotate_inpaper_citations_with_hover_meta


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
