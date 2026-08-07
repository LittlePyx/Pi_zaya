from __future__ import annotations

from copy import deepcopy

from kb.research_brief_lineage import (
    matrix_change_impact,
    matrix_contract_fingerprint,
    research_brief_lineage,
)


def _matrix(*, revision: int = 1, result: str = "Result A") -> dict:
    evidence_id = "ev-a" if result == "Result A" else "ev-b"
    return {
        "id": "matrix-1",
        "title": "Verified comparison",
        "revision": revision,
        "quality_status": "verified",
        "rows": [
            {
                "id": "row-a",
                "paper": "Paper A",
                "source_name": "Paper A",
                "source_path": "F:/papers/a.md",
                "source_status": "active",
                "notes": "",
                "cells": {
                    "key_result": {
                        "field": "key_result",
                        "value": result,
                        "support_status": "grounded",
                        "evidence_ids": [evidence_id],
                        "manual_override": False,
                    }
                },
            }
        ],
        "evidence": [
            {
                "id": evidence_id,
                "field": "key_result",
                "source_path": "F:/papers/a.md",
                "source_name": "Paper A",
                "evidence_quote": result,
            }
        ],
        "source_items": [
            {
                "key": "paper-a",
                "title": "Paper A",
                "sourcePath": "F:/papers/a.md",
                "year": "2025",
            }
        ],
        "comparison_audits": [],
    }


def _brief(source_matrix: dict) -> dict:
    return {
        "id": "brief-1",
        "quality_status": "verified",
        "quality": {
            "source_matrix_id": source_matrix["id"],
            "source_matrix_title": source_matrix["title"],
            "source_matrix_revision": source_matrix["revision"],
            "source_matrix_quality_status": "verified",
            "source_matrix_fingerprint": matrix_contract_fingerprint(source_matrix),
        },
        "evidence": [
            {
                "citation_number": 1,
                "source_path": "F:/papers/a.md",
                "evidence_quote": "Result A",
                "matrix_field": "key_result",
            }
        ],
    }


def test_lineage_reports_changed_fields_and_affected_citations() -> None:
    historical = _matrix()
    current = _matrix(revision=2, result="Result B")
    brief = _brief(historical)

    lineage = research_brief_lineage(
        brief,
        current_matrix=current,
        historical_matrix=historical,
        include_impact=True,
    )

    assert lineage["status"] == "matrix_updated"
    assert lineage["historical_verified"] is True
    assert lineage["latest_verified"] is False
    assert lineage["refresh_available"] is True
    assert lineage["export_allowed"] is True
    assert lineage["export_mode"] == "historical"
    assert lineage["impact"]["changed_row_count"] == 1
    assert lineage["impact"]["changed_field_count"] == 1
    assert lineage["impact"]["rows"][0]["fields"] == ["key_result"]
    assert lineage["impact"]["affected_citation_numbers"] == [1]


def test_lineage_treats_revision_only_change_as_current_equivalent() -> None:
    historical = _matrix()
    current = deepcopy(historical)
    current["revision"] = 3
    brief = _brief(historical)

    lineage = research_brief_lineage(
        brief,
        current_matrix=current,
        historical_matrix=historical,
        include_impact=True,
    )

    assert lineage["status"] == "current_equivalent"
    assert lineage["latest_verified"] is True
    assert lineage["impact"] == {}

    summary = research_brief_lineage(
        brief,
        current_matrix=current,
        summary_only=True,
    )
    assert summary["status"] == "current_equivalent"
    assert summary["latest_verified"] is True


def test_lineage_blocks_missing_or_mutated_source_revision() -> None:
    historical = _matrix()
    brief = _brief(historical)

    missing_matrix = research_brief_lineage(brief, current_matrix=None)
    assert missing_matrix["status"] == "matrix_missing"
    assert missing_matrix["export_allowed"] is False

    current = _matrix(revision=2, result="Result B")
    missing_revision = research_brief_lineage(brief, current_matrix=current)
    assert missing_revision["status"] == "source_revision_missing"
    assert missing_revision["export_allowed"] is False

    mutated = _matrix(result="Result B")
    same_revision = research_brief_lineage(brief, current_matrix=mutated)
    assert same_revision["status"] == "integrity_mismatch"
    assert same_revision["export_allowed"] is False


def test_matrix_change_impact_tracks_comparison_and_source_metadata() -> None:
    historical = _matrix()
    historical["comparison_audits"] = [
        {
            "id": "comparison-1",
            "status": "verified",
            "left_source_name": "Paper A",
            "right_source_name": "Paper B",
            "conclusion": "A is lower.",
        }
    ]
    current = deepcopy(historical)
    current["comparison_audits"] = []
    current["source_items"][0]["year"] = "2026"

    impact = matrix_change_impact(historical, current)

    assert impact["changed_comparison_count"] == 1
    assert impact["comparisons"][0]["change"] == "removed"
    assert impact["changed_source_count"] == 1
    assert impact["sources"][0]["change"] == "changed"
