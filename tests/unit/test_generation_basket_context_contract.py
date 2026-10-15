from __future__ import annotations

from api.routers.generate import _sanitize_prompt_context
from kb.task_runtime import _selected_research_context_filter_terms, _selected_research_context_source_paths


def test_sanitized_reference_retains_full_paper_scope_and_library_identity():
    item = {"kind": "reference", "title": "Frequency-division multiplexed single pixel imaging", "sourcePath": "db/citing-paper.md", "libraryMatchPath": "db/matched-paper.md", "libraryMatchStatus": "matched", "libraryMatchTitle": "Frequency-division multiplexed single pixel imaging"}
    normalized = _sanitize_prompt_context({"items": [item]})["items"]
    assert normalized[0]["libraryMatchPath"] == item["libraryMatchPath"]
    assert normalized[0]["libraryMatchStatus"] == "matched"
    assert _selected_research_context_filter_terms(normalized)["source_paths"] == ["db/matched-paper.md"]
    assert "db/matched-paper.md" in _selected_research_context_source_paths(normalized)


def test_sanitized_excerpt_retains_its_exact_location_boundary():
    item = {"kind": "excerpt", "title": "Selected evidence", "sourcePath": "db/paper.md", "excerpt": "The method trains the language model while freezing its vision encoder.", "headingPath": "Method / Training", "blockId": "blk_training", "anchorId": "p_training", "anchorKind": "paragraph"}
    normalized = _sanitize_prompt_context({"items": [item]})["items"]
    boundary = _selected_research_context_filter_terms(normalized)["item_constraints"][0]
    assert boundary["block_id"] == "blk_training"
    assert boundary["anchor_id"] == "p_training"
    assert boundary["heading_path"] == "Method / Training"
    assert boundary["excerpt"] == item["excerpt"]
