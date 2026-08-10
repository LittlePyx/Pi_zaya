from __future__ import annotations

from pathlib import Path

from kb.research_gap import build_project_research_gaps, find_research_gap_candidates, research_gap_summary


def _matrix(source_path: str) -> dict:
    return {
        "id": "matrix-1",
        "project_id": "project-1",
        "title": "Imaging comparison",
        "objective": "Compare dynamic imaging reconstruction methods and quantitative results.",
        "revision": 3,
        "quality_status": "needs_review",
        "rows": [
            {
                "id": "row-a",
                "paper": "Paper A",
                "source_name": "Paper A",
                "source_path": source_path,
                "cells": {},
            },
            {
                "id": "row-b",
                "paper": "Paper B",
                "source_name": "Paper B",
                "source_path": "b.md",
                "cells": {},
            },
        ],
        "quality": {
            "missing_cells": [{"row_id": "row-a", "field": "limitation"}],
            "unsupported_cells": [{"row_id": "row-b", "field": "metric"}],
            "reasons": ["unsupported_cells", "unexpected_sources"],
        },
        "comparison_audits": [
            {
                "id": "comparison-1",
                "status": "not_comparable",
                "left_row_id": "row-a",
                "right_row_id": "row-b",
                "reasons": ["dataset_mismatch"],
                "input": {
                    "dimensions": [
                        {"left_value": "DAVIS", "right_value": "Synthetic"},
                    ],
                    "left_target": "Method A",
                    "right_target": "Method B",
                },
            }
        ],
    }


def test_project_gap_queue_uses_deterministic_quality_lineage_and_change_signals(tmp_path: Path) -> None:
    source = tmp_path / "a.md"
    source.write_text("A", encoding="utf-8")
    matrix = _matrix(str(source))
    brief = {
        "id": "brief-1",
        "title": "Living brief",
        "revision": 2,
        "quality": {"source_matrix_id": "matrix-1"},
        "evidence": [{"source_path": str(source), "citation_number": 3}],
        "lineage": {
            "status": "matrix_updated",
            "source_matrix_id": "matrix-1",
            "reasons": ["source_matrix_updated"],
            "impact": {"affected_citation_numbers": [3]},
        },
    }
    change = {
        "id": "change-1",
        "event_key": "content-a",
        "matrix_id": "matrix-1",
        "kind": "source_content_changed",
        "severity": "error",
        "source_name": "Paper A",
        "impact": {
            "affected_brief_count": 1,
            "affected_citation_count": 1,
            "affected_comparison_ids": ["comparison-1"],
        },
    }

    gaps = build_project_research_gaps(
        project_id="project-1",
        matrices=[matrix],
        briefs=[brief],
        evidence_changes=[change],
    )

    kinds = [item["kind"] for item in gaps]
    assert set(kinds) == {
        "source_change",
        "unsupported_cell",
        "brief_stale",
        "matrix_needs_review",
        "comparison_not_comparable",
        "missing_cell",
    }
    missing = next(item for item in gaps if item["kind"] == "missing_cell")
    assert missing["candidate_searchable"] is True
    assert missing["impact"]["affected_brief_count"] == 1
    assert missing["impact"]["affected_citation_count"] == 1
    assert missing["impact"]["affected_comparison_count"] == 1
    matrix_gap = next(item for item in gaps if item["kind"] == "matrix_needs_review")
    assert matrix_gap["reasons"] == ["unexpected_sources"]
    scores = [int(item["priority_score"]) for item in gaps]
    assert scores == sorted(scores, reverse=True)
    summary = research_gap_summary([{**item, "status": "open"} for item in gaps])
    assert summary["total"] == 6
    assert summary["affected_matrix_count"] == 1
    assert summary["affected_brief_count"] == 1


def test_candidate_search_excludes_matrix_sources_and_keeps_exact_locator(tmp_path: Path) -> None:
    existing = tmp_path / "existing.md"
    candidate = tmp_path / "candidate.md"
    existing.write_text("existing", encoding="utf-8")
    candidate.write_text("candidate", encoding="utf-8")
    gap = next(
        item
        for item in build_project_research_gaps(
            project_id="project-1",
            matrices=[_matrix(str(existing))],
        )
        if item["kind"] == "missing_cell"
    )
    exact_quote = "However, the dynamic reconstruction remains limited by motion and calibration errors."
    chunks = [
        {
            "id": "existing:1",
            "text": exact_quote,
            "meta": {"source_path": str(existing), "heading_path": "Limitations", "evidence_ready": True},
        },
        {
            "id": "candidate:1",
            "text": exact_quote,
            "meta": {
                "source_path": str(candidate),
                "heading_path": "Discussion / Limitations",
                "page_start": 7,
                "page_end": 7,
                "block_id": "blk-7",
                "evidence_ready": True,
            },
        },
    ]

    candidates = find_research_gap_candidates(
        gap,
        db_dir=tmp_path,
        excluded_source_paths=[str(existing)],
        chunks=chunks,
    )

    assert len(candidates) == 1
    assert candidates[0]["source_path"] == str(candidate)
    assert candidates[0]["evidence_quote"] == exact_quote
    assert candidates[0]["heading_path"] == "Discussion / Limitations"
    assert candidates[0]["page_start"] == 7
    assert candidates[0]["block_id"] == "blk-7"
