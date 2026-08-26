from pathlib import Path

import pytest

from kb.research_note_evidence import (
    build_research_note_evidence_context,
    repair_research_note_source_state,
    source_content_fingerprint,
)
from kb.source_blocks import load_source_blocks


def _write_source(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "paper.md"
    path.write_text(body, encoding="utf-8")
    return path


def _fixture_source(tmp_path: Path) -> tuple[Path, dict]:
    path = _write_source(
        tmp_path,
        """# Example paper

<!-- kb_page: 2 -->

## Methods

The detector was calibrated before every acquisition.

Photon arrival is a stochastic process governed by Poisson statistics.

Dark counts were measured with the shutter closed.
""",
    )
    blocks = load_source_blocks(path)
    target = next(block for block in blocks if "Photon arrival" in str(block.get("text") or ""))
    return path, target


def test_exact_block_returns_surrounding_context_and_canonical_reader_target(tmp_path: Path) -> None:
    path, target = _fixture_source(tmp_path)

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/example.pdf",
        source_name="Example paper",
        heading_path="Old heading",
        evidence_quote="Photon arrival is a stochastic process",
        block_id=str(target["block_id"]),
    )

    assert result["status"] == "exact"
    assert result["reason"] == "exact_block"
    assert result["can_open_exact"] is True
    assert "Photon arrival" in result["target"]["text"]
    assert "calibrated" in result["before"][0]["text"]
    assert "Dark counts" in result["after"][0]["text"]
    assert result["page_start"] == 2
    assert result["reader_open"]["blockId"] == target["block_id"]
    assert result["reader_open"]["strictLocate"] is True


def test_unique_quote_can_recover_an_exact_location_after_ids_change(tmp_path: Path) -> None:
    path, target = _fixture_source(tmp_path)

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/example.pdf",
        source_name="Example paper",
        evidence_quote="Photon arrival is a stochastic process governed by Poisson statistics.",
        block_id="stale-block-id",
    )

    assert result["status"] == "exact"
    assert result["reason"] == "quote_relocated"
    assert result["reader_open"]["blockId"] == target["block_id"]


def test_positional_block_id_is_not_trusted_when_its_text_no_longer_matches(tmp_path: Path) -> None:
    path, target = _fixture_source(tmp_path)
    wrong_block = next(
        block
        for block in load_source_blocks(path)
        if "calibrated" in str(block.get("text") or "")
    )

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/example.pdf",
        evidence_quote="Photon arrival is a stochastic process governed by Poisson statistics.",
        block_id=str(wrong_block["block_id"]),
    )

    assert result["status"] == "exact"
    assert result["reason"] == "quote_relocated"
    assert result["target"]["block_id"] == target["block_id"]


def test_fuzzy_match_is_only_a_review_candidate(tmp_path: Path) -> None:
    path, _target = _fixture_source(tmp_path)

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/example.pdf",
        source_name="Example paper",
        evidence_quote="stochastic photon arrivals follow Poisson behavior",
        block_id="stale-block-id",
    )

    assert result["status"] == "candidate"
    assert result["reason"] == "saved_target_missing"
    assert result["can_open_exact"] is False
    assert result["reader_open"] is None
    assert result["candidates"]
    assert result["candidates"][0]["reader_open"]["strictLocate"] is True


def test_ambiguous_exact_quote_requires_candidate_review(tmp_path: Path) -> None:
    path = _write_source(
        tmp_path,
        """# Repeated evidence

## First experiment

The same calibration sentence appears here verbatim.

## Second experiment

The same calibration sentence appears here verbatim.
""",
    )

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/repeated.pdf",
        evidence_quote="The same calibration sentence appears here verbatim.",
        block_id="stale-block-id",
    )

    assert result["status"] == "candidate"
    assert result["reason"] == "ambiguous_quote"
    assert len(result["candidates"]) == 2
    assert result["reader_open"] is None


def test_unrelated_quote_does_not_create_a_false_location(tmp_path: Path) -> None:
    path, _target = _fixture_source(tmp_path)

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/example.pdf",
        evidence_quote="xylophone zeppelin kumquat",
        block_id="stale-block-id",
    )

    assert result["status"] == "unavailable"
    assert result["reason"] == "no_match"
    assert result["candidates"] == []


def test_surrounding_context_does_not_cross_a_section_heading(tmp_path: Path) -> None:
    path = _write_source(
        tmp_path,
        """# Paper

## Earlier section

This paragraph belongs to the earlier section.

## Target section

This is the first evidence paragraph in the target section.
""",
    )
    blocks = load_source_blocks(path)
    target = next(block for block in blocks if "first evidence" in str(block.get("text") or ""))

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/sections.pdf",
        evidence_quote="This is the first evidence paragraph in the target section.",
        block_id=str(target["block_id"]),
    )

    assert result["status"] == "exact"
    assert result["before"] == []


def test_source_fingerprint_marks_a_changed_conversion(tmp_path: Path) -> None:
    path, target = _fixture_source(tmp_path)
    fingerprint = source_content_fingerprint(path)

    unchanged = build_research_note_evidence_context(
        md_path=path,
        source_path="library/example.pdf",
        evidence_quote="Photon arrival is a stochastic process governed by Poisson statistics.",
        block_id=str(target["block_id"]),
        source_fingerprint=fingerprint,
    )
    path.write_text(path.read_text(encoding="utf-8") + "\nA new appendix line.\n", encoding="utf-8")
    changed = build_research_note_evidence_context(
        md_path=path,
        source_path="library/example.pdf",
        evidence_quote="Photon arrival is a stochastic process governed by Poisson statistics.",
        block_id=str(target["block_id"]),
        source_fingerprint=fingerprint,
    )

    assert unchanged["source_fingerprint"] == fingerprint
    assert unchanged["source_changed"] is False
    assert changed["source_fingerprint"] != fingerprint
    assert changed["source_changed"] is True


def test_confirmed_candidate_repairs_one_locator_and_keeps_an_audit_trail() -> None:
    source_state = {
        "version": 1,
        "links": [
            {
                "kind": "source",
                "label": "Detector evidence",
                "source_path": "library/example.pdf",
                "capture_id": "capture-1",
                "capture_kind": "selection",
                "block_id": "old-block",
                "anchor_id": "old-anchor",
                "anchor_kind": "paragraph",
                "heading_path": "Old section",
                "location_label": "Old section · page 1",
                "page_start": 1,
                "evidence_quote": "Old detector sentence.",
                "start_offset": 3,
                "end_offset": 25,
                "occurrence": 2,
                "asset_src": "/kept.png",
            }
        ],
    }
    repaired = repair_research_note_source_state(
        source_state,
        source_index=0,
        source_path="LIBRARY\\example.pdf",
        capture_id="capture-1",
        old_block_id="old-block",
        old_anchor_id="old-anchor",
        old_evidence_quote="Old detector sentence.",
        target={
            "block_id": "new-block",
            "anchor_id": "new-anchor",
            "kind": "paragraph",
            "heading_path": "Results",
            "page_start": 4,
            "page_end": 4,
            "line_start": 22,
            "line_end": 24,
            "text": "The repaired detector sentence is exact.",
        },
        source_fingerprint="sha256:abc123",
        repaired_at=1_800_000_000,
    )

    assert source_state["links"][0]["block_id"] == "old-block"
    link = repaired["links"][0]
    assert link["capture_id"] == "capture-1"
    assert link["capture_kind"] == "selection"
    assert link["asset_src"] == "/kept.png"
    assert link["block_id"] == "new-block"
    assert link["anchor_id"] == "new-anchor"
    assert link["heading_path"] == "Results"
    assert link["page_start"] == 4
    assert link["evidence_quote"] == "The repaired detector sentence is exact."
    assert "location_label" not in link
    assert "start_offset" not in link
    assert "end_offset" not in link
    assert "occurrence" not in link
    assert link["evidence_verification"] == {
        "status": "exact",
        "reason": "confirmed_candidate",
        "verified_at": 1_800_000_000.0,
        "source_fingerprint": "sha256:abc123",
    }
    audit = link["locator_repair_history"][-1]
    assert audit["from"]["block_id"] == "old-block"
    assert audit["to"]["block_id"] == "new-block"
    assert audit["repaired_at"] == 1_800_000_000.0


def test_locator_repair_rejects_a_source_that_changed_after_reader_opened() -> None:
    state = {
        "links": [
            {
                "source_path": "library/example.pdf",
                "capture_id": "newer-capture",
                "block_id": "newer-block",
                "evidence_quote": "Newer evidence.",
            }
        ]
    }

    with pytest.raises(ValueError, match="source changed"):
        repair_research_note_source_state(
            state,
            source_index=0,
            source_path="library/example.pdf",
            capture_id="older-capture",
            old_block_id="older-block",
            old_evidence_quote="Older evidence.",
            target={"block_id": "candidate", "text": "Candidate evidence."},
            source_fingerprint="sha256:abc123",
        )
