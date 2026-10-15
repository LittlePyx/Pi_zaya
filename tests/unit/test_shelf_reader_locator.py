from __future__ import annotations

from kb.shelf_reader_locator import maintain_shelf_reader_locator


def _block(
    index: int,
    text: str,
    *,
    heading: str = "Paper / Methods",
    kind: str = "paragraph",
) -> dict:
    return {
        "doc_id": "doc123",
        "block_id": f"blk_doc123_{index:05d}",
        "anchor_id": f"p_{index:05d}",
        "kind": kind,
        "heading_path": heading,
        "page_start": index,
        "page_end": index,
        "text": text,
    }


def test_keeps_supported_direct_locator() -> None:
    blocks = [_block(1, "The proposed reconstruction uses a physics-informed forward model.")]
    result = maintain_shelf_reader_locator(blocks, {
        "key": "shelf-1",
        "block_id": "blk_doc123_00001",
        "anchor_id": "p_00001",
        "anchor_kind": "paragraph",
        "heading_path": "Paper / Methods",
        "evidence_quote": "uses a physics-informed forward model",
        "page_start": 1,
        "page_end": 1,
    })

    assert result["ok"] is True
    assert result["changed"] is False
    assert result["status"] == "ready"
    assert result["reason"] == "identity_valid"


def test_recovers_stale_locator_from_unique_evidence_quote() -> None:
    blocks = [
        _block(1, "General background text."),
        _block(7, "The proposed reconstruction uses a physics-informed forward model."),
    ]
    result = maintain_shelf_reader_locator(blocks, {
        "key": "shelf-2",
        "block_id": "blk_doc123_00001",
        "anchor_id": "p_00001",
        "anchor_kind": "paragraph",
        "heading_path": "Paper / Methods",
        "evidence_quote": "uses a physics-informed forward model",
    })

    assert result["ok"] is True
    assert result["changed"] is True
    assert result["status"] == "recovered"
    assert result["reason"] == "unique_evidence_quote"
    assert result["locator"]["block_id"] == "blk_doc123_00007"
    assert result["locator"]["anchor_id"] == "p_00007"


def test_strict_reader_rejects_surviving_stale_identity_and_relocates_quote() -> None:
    evidence = "The coded aperture preserves spectral detail across the reconstructed frame."
    blocks = [
        _block(2, "The coded aperture is calibrated before capture and the detector is synchronized."),
        _block(7, evidence, heading="Paper / Results"),
    ]
    result = maintain_shelf_reader_locator(blocks, {
        "key": "reader-open",
        "block_id": "blk_doc123_00002",
        "anchor_id": "p_00002",
        "anchor_kind": "paragraph",
        "heading_path": "Paper / Methods",
        "evidence_quote": evidence,
        "strict_evidence": True,
    })

    assert result["ok"] is True
    assert result["changed"] is True
    assert result["status"] == "recovered"
    assert result["reason"] == "unique_evidence_quote"
    assert result["locator"]["block_id"] == "blk_doc123_00007"


def test_disambiguates_repeated_quote_with_saved_heading() -> None:
    quote = "The signal is reconstructed from the coded measurement."
    blocks = [
        _block(2, quote, heading="Paper / Introduction"),
        _block(9, quote, heading="Paper / Reconstruction method"),
    ]
    result = maintain_shelf_reader_locator(blocks, {
        "key": "shelf-3",
        "block_id": "stale-block",
        "anchor_kind": "paragraph",
        "heading_path": "Paper / Reconstruction method",
        "evidence_quote": quote,
    })

    assert result["ok"] is True
    assert result["reason"] == "context_disambiguated_quote"
    assert result["locator"]["block_id"] == "blk_doc123_00009"


def test_does_not_auto_persist_ambiguous_repeated_quote() -> None:
    quote = "The signal is reconstructed from the coded measurement."
    blocks = [
        _block(2, quote, heading="Paper / Results"),
        _block(9, quote, heading="Paper / Results"),
    ]
    result = maintain_shelf_reader_locator(blocks, {
        "key": "shelf-4",
        "block_id": "stale-block",
        "anchor_kind": "paragraph",
        "heading_path": "Paper / Results",
        "evidence_quote": quote,
    })

    assert result["ok"] is False
    assert result["changed"] is False
    assert result["status"] == "unresolved"
    assert result["reason"] == "ambiguous_quote"


def test_does_not_replace_locator_for_weak_match() -> None:
    blocks = [
        _block(1, "This paragraph discusses an unrelated imaging system."),
        _block(2, "Another paragraph reports a different experiment."),
    ]
    result = maintain_shelf_reader_locator(blocks, {
        "key": "shelf-5",
        "block_id": "missing-block",
        "heading_path": "Paper / Methods",
        "evidence_quote": "quantum detector calibration with transfer learning",
    })

    assert result["ok"] is False
    assert result["changed"] is False
    assert result["reason"] == "ambiguous_or_weak_match"
