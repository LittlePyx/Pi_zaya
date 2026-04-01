from __future__ import annotations

from pathlib import Path

from kb.paper_guide_grounding_runtime import (
    _build_paper_guide_support_slots,
    _resolve_paper_guide_support_markers,
)
from kb.source_blocks import load_source_blocks


def _find_block_text(blocks: list[dict], *, block_id: str) -> str:
    bid = str(block_id or "").strip()
    for block in list(blocks or []):
        if not isinstance(block, dict):
            continue
        if str(block.get("block_id") or "").strip() != bid:
            continue
        return str(block.get("raw_text") or block.get("text") or "").strip()
    return ""


def test_paper_guide_multi_panel_support_marker_resolution_points_to_exact_caption(tmp_path: Path):
    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")

    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Results\n\n"
            "**Figure 1.** d Open pinhole confocal iSCAT. "
            "e Closed pinhole confocal iSCAT. "
            "f Resulting iPSF from iISM after adaptive pixel-reassignment (APR), with same incident illumination power and number of detected photons. "
            "g Line profiles of the iPSF in the three configurations as indicated in d-f.\n"
        ),
        encoding="utf-8",
    )

    cards = [
        {
            "doc_idx": 1,
            "sid": "s1",
            "source_path": str(source_pdf),
            "heading": "Results / Figure 1",
            "snippet": (
                "Figure 1. f Resulting iPSF from iISM after adaptive pixel-reassignment (APR). "
                "g Line profiles of the iPSF in the three configurations."
            ),
            "candidate_refs": [],
            "deepread_texts": [],
        }
    ]

    slots = _build_paper_guide_support_slots(
        cards,
        prompt="Walk me through Figure 1 panels f and g.",
        prompt_family="figure_walkthrough",
        db_dir=tmp_path,
    )

    # Practical assertions: we expect per-panel slots, and their locate anchors must be exact substrings
    # of the resolved caption block so reader locate highlight jumps precisely.
    assert len(slots) == 2
    for slot in slots:
        letters = [str(ch or "").strip().lower() for ch in list(slot.get("panel_letters") or []) if str(ch or "").strip()]
        assert len(letters) == 1
        assert letters[0] in {"f", "g"}
        assert str(slot.get("anchor_id") or "").strip()
        assert str(slot.get("block_id") or "").strip()
        assert str(slot.get("locate_anchor") or "").strip().lower().startswith(f"{letters[0]} ")

    # Build an answer that explicitly requests both support ids and resolve them.
    slots_sorted = sorted(slots, key=lambda s: str((s.get("panel_letters") or [""])[0]).lower())
    answer_in = "\n".join(
        [
            f"Panel f walkthrough. {slots_sorted[0]['support_example']}",
            f"Panel g walkthrough. {slots_sorted[1]['support_example']}",
        ]
    )
    answer_out, resolutions = _resolve_paper_guide_support_markers(
        answer_in,
        support_slots=slots,
        prompt_family="figure_walkthrough",
        db_dir=tmp_path,
    )

    assert "[[SUPPORT:" not in answer_out
    assert len(resolutions) == 2

    blocks = load_source_blocks(md_main)
    for rec in resolutions:
        block_id = str(rec.get("block_id") or "").strip()
        locate_anchor = str(rec.get("locate_anchor") or "").strip()
        assert block_id
        assert locate_anchor
        block_text = _find_block_text(blocks, block_id=block_id)
        assert block_text
        assert locate_anchor in block_text


def test_paper_guide_method_support_resolution_never_invents_reference_numbers(tmp_path: Path):
    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")

    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Methods\n\n"
            "We apply the adaptive pixel-reassignment (APR) algorithm as described in [12] to compute the iPSF.\n\n"
            "## References\n\n"
            "[12] A. Author. A Great Method Paper. 2020.\n"
        ),
        encoding="utf-8",
    )

    cards = [
        {
            "doc_idx": 1,
            "sid": "m1",
            "source_path": str(source_pdf),
            "heading": "Methods",
            "snippet": "We apply the adaptive pixel-reassignment (APR) algorithm as described in [12] to compute the iPSF.",
            "candidate_refs": [12],
            "deepread_texts": [],
        }
    ]

    slots = _build_paper_guide_support_slots(
        cards,
        prompt="How is APR applied in the method?",
        prompt_family="method",
        db_dir=tmp_path,
    )
    assert slots
    slot = slots[0]
    assert str(slot.get("cite_policy") or "").strip().lower() == "prefer_ref"
    assert 12 in [int(n) for n in list(slot.get("candidate_refs") or []) if int(n) > 0]

    answer_in = f"APR is applied as follows. {slot['support_example']}"
    answer_out, resolutions = _resolve_paper_guide_support_markers(
        answer_in,
        support_slots=slots,
        prompt_family="method",
        db_dir=tmp_path,
    )

    assert "[[SUPPORT:" not in answer_out
    assert "[[CITE:m1:12]]" in answer_out
    assert resolutions
    assert int(resolutions[0].get("resolved_ref_num") or 0) == 12
