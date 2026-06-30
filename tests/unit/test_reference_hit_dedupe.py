from __future__ import annotations

from api import reference_hit_dedupe as dedupe


def _hit(
    *,
    source: str = "paper-a.en.md",
    heading: str = "2. Related Work",
    text: str = "Most existing methods employ ADMM-based optimization for reconstruction.",
    reader_open: dict | None = None,
    score: float = 1.0,
) -> dict:
    ui_meta = {
        "source_path": source,
        "display_name": source.replace(".en.md", ".pdf"),
        "heading_path": heading,
        "summary_line": text,
        "why_line": "This hit mentions ADMM reconstruction.",
        "score": score,
    }
    if reader_open is not None:
        ui_meta["reader_open"] = reader_open
    return {
        "text": text,
        "meta": {"source_path": source, "ref_best_heading_path": heading},
        "ui_meta": ui_meta,
    }


def test_refs_hit_keys_prefer_ui_meta_then_reader_open() -> None:
    hit = {
        "reader_open": {"sourcePath": r"db\PaperB\PaperB.en.md", "headingPath": "3. Methods"},
        "ui_meta": {"display_name": "PaperB.pdf"},
    }

    assert dedupe._refs_hit_source_key(hit) == "db paperb paperb en md"
    assert dedupe._refs_hit_heading_key(hit) == "3 methods"


def test_refs_hit_exact_locate_score_counts_strict_anchor_and_primary_evidence() -> None:
    hit = _hit(
        reader_open={
            "strictLocate": True,
            "blockId": "blk-1",
            "anchorId": "sent-1",
            "anchorKind": "sentence",
            "primaryEvidence": {"snippet": "ADMM reconstruction evidence."},
        }
    )

    assert dedupe._refs_hit_locate_key(hit) == "loc:blk-1|sent-1|sentence|"
    assert dedupe._refs_hit_exact_locate_score(hit) == 1.30


def test_refs_hits_are_near_duplicates_for_same_source_and_locate_key() -> None:
    left = _hit(
        reader_open={
            "sourcePath": "paper-a.en.md",
            "headingPath": "2. Related Work",
            "blockId": "blk-1",
            "anchorId": "sent-1",
        }
    )
    right = _hit(
        reader_open={
            "sourcePath": "paper-a.en.md",
            "headingPath": "2. Related Work",
            "blockId": "blk-1",
            "anchorId": "sent-1",
        }
    )

    assert dedupe._refs_hits_are_near_duplicates(left, right) is True


def test_refs_hits_are_not_duplicates_across_sources_or_headings() -> None:
    left = _hit(source="paper-a.en.md", heading="2. Related Work")
    other_source = _hit(source="paper-b.en.md", heading="2. Related Work")
    other_heading = _hit(source="paper-a.en.md", heading="4. Experiments")

    assert dedupe._refs_hits_are_near_duplicates(left, other_source) is False
    assert dedupe._refs_hits_are_near_duplicates(left, other_heading) is False


def test_dedupe_refs_hits_prefers_precise_locate_over_higher_display_score() -> None:
    loose = _hit(score=9.5)
    precise = _hit(
        score=4.0,
        reader_open={
            "sourcePath": "paper-a.en.md",
            "headingPath": "2. Related Work",
            "strictLocate": True,
            "blockId": "blk-1",
            "anchorId": "sent-1",
        },
    )

    hits, removed = dedupe._dedupe_refs_hits_for_display(
        prompt="Which paper explains ADMM reconstruction?",
        hits=[loose, precise],
        focus_match_count=lambda prompt, hit: 1,
        section_intent_score=lambda prompt, hit: 1.0,
        display_score=lambda hit: float((hit.get("ui_meta") or {}).get("score") or 0.0),
    )

    assert removed == 1
    assert len(hits) == 1
    ui_meta = dict(hits[0].get("ui_meta") or {})
    assert (ui_meta.get("reader_open") or {}).get("blockId") == "blk-1"
    assert ui_meta.get("merged_duplicate_count") == 1
    assert ui_meta.get("merged_duplicate_headings") == ["2. Related Work"]
