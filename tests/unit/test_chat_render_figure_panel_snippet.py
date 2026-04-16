from __future__ import annotations

from pathlib import Path

from api.chat_render import _enrich_provenance_segments_for_display


def test_enrich_provenance_segments_for_display_prefers_panel_clause_snippet(tmp_path: Path):
    md = tmp_path / "Demo.en.md"
    md.write_text("# Demo\n", encoding="utf-8")
    provenance = {
        "md_path": str(md),
        "source_path": str(md),
        "source_name": "Demo.pdf",
        "block_map": {
            "blk_cap": {
                "block_id": "blk_cap",
                "anchor_id": "p_00001",
                "kind": "paragraph",
                "heading_path": "Demo / Figure 1",
                "text": "Figure 1. (a) Cat. (b) Dog. (c) Bird.",
            }
        },
        "segments": [
            {
                "segment_id": "seg_001",
                "segment_index": 1,
                "kind": "paragraph",
                "segment_type": "prose",
                "text": "Figure 1 panel (b) shows the dog.",
                "raw_markdown": "Figure 1 panel (b) shows the dog.",
                "evidence_mode": "direct",
                "claim_type": "figure_panel",
                "must_locate": True,
                "anchor_kind": "figure",
                "anchor_text": "Figure 1 panel (b)",
                "support_slot_figure_number": 1,
                "support_slot_panel_letters": ["b"],
                "primary_block_id": "blk_cap",
                "primary_anchor_id": "p_00001",
                "primary_heading_path": "Demo / Figure 1",
                "evidence_block_ids": ["blk_cap"],
                "evidence_quote": "Figure 1. (a) Cat. (b) Dog. (c) Bird.",
                "hit_level": "exact",
                "locate_policy": "required",
                "locate_surface_policy": "primary",
            }
        ],
    }

    enriched = _enrich_provenance_segments_for_display(provenance, hits=[], anchor_ns="t")
    seg = (enriched.get("segments") or [None])[0] or {}
    lt = seg.get("locate_target") or {}
    assert "Dog" in str(lt.get("snippet") or "")
    assert "(b)" in str(lt.get("snippet") or "")

