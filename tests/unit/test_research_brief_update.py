from __future__ import annotations

from copy import deepcopy

from kb.research_brief import research_brief_evidence
from kb.research_brief_update import (
    apply_research_brief_update_decisions,
    build_research_brief_update_plan,
    stable_matrix_hits,
)


def _matrix(*, revision: int, result: str, evidence_id: str) -> dict:
    return {
        "id": "matrix-1",
        "title": "Verified matrix",
        "revision": revision,
        "quality_status": "verified",
        "rows": [
            {
                "id": "row-a",
                "paper": "Paper A",
                "source_name": "Paper A",
                "source_path": "F:/papers/a.md",
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
            {"key": "paper-a", "title": "Paper A", "sourcePath": "F:/papers/a.md"}
        ],
        "comparison_audits": [],
    }


def _brief(matrix: dict) -> dict:
    hits = stable_matrix_hits([], matrix)
    return {
        "id": "brief-1",
        "revision": 4,
        "content_markdown": (
            "## Findings\n\n"
            "- Paper A reports the original measured result [1].\n\n"
            "## Researcher note\n\n"
            "Keep this manually edited interpretation exactly as written."
        ),
        "evidence": research_brief_evidence(hits),
    }


def test_incremental_plan_changes_only_affected_claim_and_preserves_manual_content() -> None:
    historical = _matrix(revision=1, result="The measured result is 1.0 dB", evidence_id="ev-a")
    current = _matrix(revision=2, result="The measured result is 1.4 dB", evidence_id="ev-b")
    brief = _brief(historical)

    plan = build_research_brief_update_plan(
        brief,
        historical_matrix=historical,
        current_matrix=current,
        impact={"affected_citation_numbers": [1], "changed_field_count": 1},
        locale="en",
        model_generator=None,
    )

    assert len(plan["items"]) == 1
    item = plan["items"][0]
    assert item["action"] == "replace"
    assert item["citation_numbers_before"] == [1]
    assert item["citation_numbers_after"] == [1]
    assert "1.4 dB" in item["proposed_markdown"]
    assert "1.0 dB" not in item["proposed_markdown"]
    assert "Keep this manually edited interpretation exactly as written." in plan["preview_content_markdown"]
    assert plan["preservation"]["unaffected_character_count"] > 0
    assert plan["preservation"]["unaffected_preservation_ratio"] > 0.5


def test_rejected_change_keeps_original_bytes_and_is_reported() -> None:
    content = "Before\n\n- Old claim [1].\n\nAfter"
    items = [
        {
            "id": "change-1",
            "start": content.index("- Old"),
            "end": content.index("- Old") + len("- Old claim [1]."),
            "proposed_markdown": "- New claim [1].",
        }
    ]

    rejected = apply_research_brief_update_decisions(
        content,
        items,
        {"change-1": "reject"},
    )
    accepted = apply_research_brief_update_decisions(
        content,
        items,
        {"change-1": "accept"},
    )

    assert rejected["content_markdown"] == content
    assert rejected["rejected_item_ids"] == ["change-1"]
    assert accepted["content_markdown"] == "Before\n\n- New claim [1].\n\nAfter"
    assert accepted["all_accepted"] is True


def test_stable_hits_keep_unaffected_higher_citation_slot_after_removal() -> None:
    historical = _matrix(revision=1, result="Result A", evidence_id="ev-a")
    historical["rows"].append(deepcopy(historical["rows"][0]))
    historical["rows"][1].update(
        {"id": "row-b", "paper": "Paper B", "source_name": "Paper B", "source_path": "F:/papers/b.md"}
    )
    historical["rows"][1]["cells"]["key_result"]["evidence_ids"] = ["ev-b"]
    historical["evidence"].append(
        {
            "id": "ev-b",
            "field": "key_result",
            "source_path": "F:/papers/b.md",
            "source_name": "Paper B",
            "evidence_quote": "Result B",
        }
    )
    old_evidence = research_brief_evidence(stable_matrix_hits([], historical))
    current = deepcopy(historical)
    current["revision"] = 2
    current["rows"] = [current["rows"][1]]
    current["evidence"] = [current["evidence"][1]]

    slots = stable_matrix_hits(old_evidence, current)
    evidence = research_brief_evidence(slots)

    assert len(evidence) == 2
    assert evidence[1]["source_name"] == "Paper B"
    assert evidence[1]["evidence_quote"] == "Result B"
    assert evidence[0]["citation_slot_filler"] is True
