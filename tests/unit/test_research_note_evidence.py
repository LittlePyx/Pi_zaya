from pathlib import Path

from kb.research_note_evidence import build_research_note_evidence_context
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
