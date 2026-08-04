from __future__ import annotations

from pathlib import Path

import kb.task_runtime as task_runtime
from kb.task_runtime import (
    _effective_query_scope,
    _filter_hits_for_selected_research_context,
    _format_selected_research_context_block,
    _merge_selected_research_context_evidence_hits,
    _normalize_query_scope,
    _query_scope_prompt_block,
    _selected_research_context_evidence_contract,
    _selected_research_context_evidence_hits,
)


def test_normalize_query_scope_aliases() -> None:
    assert _normalize_query_scope("current-paper") == "current_paper"
    assert _normalize_query_scope("citation_shelf") == "basket"
    assert _normalize_query_scope("full_library") == "library"
    assert _normalize_query_scope("unknown") == ""


def test_source_only_selected_context_scans_exact_paper_blocks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "paper.en.md"
    source.write_text("# Paper\n\n## Method\n\nExact source evidence.\n", encoding="utf-8")
    calls: list[dict] = []

    def fake_scan(**kwargs):
        calls.append(dict(kwargs))
        return [
            {
                "text": "Exact source evidence.",
                "score": 88.0,
                "meta": {
                    "source_path": str(source),
                    "heading_path": "Paper / Method",
                    "block_id": "blk-method",
                    "anchor_id": "p-method",
                    "paper_guide_targeted_block": True,
                },
            }
        ]

    monkeypatch.setattr(task_runtime, "_paper_guide_targeted_source_block_hits", fake_scan)
    hits = _selected_research_context_evidence_hits(
        [
            {
                "kind": "source",
                "sourcePath": str(source),
                "sourceName": "Paper",
                "title": "Paper",
            }
        ],
        max_hits=2,
        prompt="How does the exact method work?",
        db_dir=tmp_path,
    )

    assert calls and calls[0]["bound_source_path"] == str(source)
    assert hits[0]["text"] == "Exact source evidence."
    assert hits[0]["meta"]["block_id"] == "blk-method"
    assert hits[0]["meta"]["citation_context_source"] == "research_basket_source_scan"


def test_effective_query_scope_respects_available_context() -> None:
    assert _effective_query_scope(
        requested="current_paper",
        paper_guide_mode=True,
        has_current_paper=True,
        has_basket=False,
    ) == "current_paper"
    assert _effective_query_scope(
        requested="current_paper",
        paper_guide_mode=True,
        has_current_paper=False,
        has_basket=False,
    ) == "library"
    assert _effective_query_scope(
        requested="basket",
        paper_guide_mode=True,
        has_current_paper=True,
        has_basket=True,
    ) == "basket"
    assert _effective_query_scope(
        requested="",
        paper_guide_mode=True,
        has_current_paper=True,
        has_basket=False,
    ) == "current_paper"


def test_query_scope_prompt_block_names_full_library_mode() -> None:
    block = _query_scope_prompt_block(
        scope="library",
        selected_count=0,
        current_source_name="reader.pdf",
        current_source_path="",
    )

    assert "QUERY SCOPE: Full library." in block
    assert "whole indexed literature library" in block
    assert "organize the answer by paper" in block


def test_basket_scope_filters_reader_selection_to_selected_block() -> None:
    hits = [
        {
            "text": "Selected source explains sparse filtering.",
            "meta": {"source_path": "F:/papers/selected-paper.en.md", "block_id": "blk-1"},
            "score": 2.0,
        },
        {
            "text": "Same source mentions Sparse 3-D transform-domain filtering but is a different paragraph.",
            "meta": {"source_path": "F:/papers/selected-paper.en.md", "block_id": "blk-2"},
            "score": 2.5,
        },
        {
            "text": "Unrelated library paper.",
            "meta": {"source_path": "F:/papers/unrelated-paper.en.md"},
            "score": 3.0,
        },
    ]
    selected_items = [
        {
            "kind": "reader_selection",
            "title": "Sparse 3-D transform-domain filtering",
            "sourcePath": "F:/papers/selected-paper.en.md",
            "blockId": "blk-1",
            "excerpt": "Selected source explains sparse filtering.",
        }
    ]

    filtered, trace = _filter_hits_for_selected_research_context(hits, selected_items)

    assert filtered == [hits[0]]
    assert trace["mode"] == "matched_library_hits"
    assert trace["before"] == 3
    assert trace["after"] == 1
    assert trace["item_constraint_count"] == 1


def test_basket_scope_filters_hits_by_doi_or_title_without_source_path() -> None:
    hits = [
        {
            "text": "Dabov et al. introduced Sparse 3-D transform-domain filtering. DOI 10.1109/TIP.2007.901238.",
            "meta": {"source_path": "F:/papers/references.en.md"},
            "score": 2.0,
        },
        {
            "text": "A different paragraph from the current paper should not pass only because the reference came from it.",
            "meta": {"source_path": "F:/papers/current-paper.en.md"},
            "score": 1.8,
        },
        {
            "text": "Another denoising paper without the selected bibliographic identity.",
            "meta": {"source_path": "F:/papers/other.en.md"},
            "score": 1.5,
        },
    ]
    selected_items = [
        {
            "kind": "reference",
            "title": "Sparse 3-D transform-domain filtering",
            "doi": "https://doi.org/10.1109/tip.2007.901238",
            "sourcePath": "F:/papers/current-paper.en.md",
            "excerpt": "baseline method",
        }
    ]

    filtered, trace = _filter_hits_for_selected_research_context(hits, selected_items)

    assert filtered == [hits[0]]
    assert trace["doi_count"] == 1
    assert trace["title_count"] == 1


def test_basket_scope_reference_library_match_path_allows_matched_paper_body() -> None:
    hits = [
        {
            "text": "The matched upstream paper body discusses collaborative filtering in detail.",
            "meta": {"source_path": "F:/papers/upstream-matched.en.md", "block_id": "body-1"},
            "score": 2.0,
        },
        {
            "text": "The current paper has a bibliography row but should not be opened as the upstream paper body.",
            "meta": {"source_path": "F:/papers/current-paper.en.md", "block_id": "refs-12"},
            "score": 1.8,
        },
    ]
    selected_items = [
        {
            "kind": "reference",
            "title": "Collaborative filtering for implicit feedback datasets",
            "sourcePath": "F:/papers/current-paper.en.md",
            "libraryMatchPath": "F:/papers/upstream-matched.en.md",
            "libraryMatchStatus": "ready",
            "excerpt": "Reference entry in the current paper.",
        }
    ]

    filtered, trace = _filter_hits_for_selected_research_context(hits, selected_items)

    assert filtered == [hits[0]]
    assert trace["source_path_count"] == 1


def test_selected_research_context_prompt_uses_library_match_label_not_local_path() -> None:
    block = _format_selected_research_context_block(
        {
            "items": [
                {
                    "kind": "reference",
                    "title": "Collaborative filtering for implicit feedback datasets",
                    "sourceName": "current-paper.pdf",
                    "libraryMatchPath": r"F:\private\papers\upstream-matched.en.md",
                    "libraryMatchStatus": "ready",
                    "excerpt": "Hu et al. 2008 reference row.",
                }
            ]
        }
    )

    assert "library_match=upstream-matched.en.md (ready)" in block
    assert r"F:\private\papers" not in block


def test_basket_scope_drops_library_hits_when_no_selected_identity_matches() -> None:
    hits = [
        {
            "text": "A full-library hit that should not leak into basket-only mode.",
            "meta": {"source_path": "F:/papers/unrelated.en.md"},
            "score": 4.0,
        }
    ]
    selected_items = [{"summary": "User note without a source, DOI, or title."}]

    filtered, trace = _filter_hits_for_selected_research_context(hits, selected_items)

    assert filtered == []
    assert trace["mode"] == "selected_context_only"
    assert trace["after"] == 0


def test_selected_research_context_builds_reader_evidence_hit() -> None:
    selected_items = [
        {
            "kind": "reader_selection",
            "title": "Sparse 3-D transform-domain filtering",
            "sourcePath": "F:/papers/selected-paper.en.md",
            "sourceName": "selected-paper.pdf",
            "headingPath": "Methods / Filtering",
            "blockId": "blk-1",
            "anchorId": "a-1",
            "excerpt": "Selected source explains sparse filtering.",
        }
    ]

    hits = _selected_research_context_evidence_hits(selected_items)

    assert len(hits) == 1
    hit = hits[0]
    meta = hit["meta"]
    assert "Selected excerpt: Selected source explains sparse filtering." in hit["text"]
    assert meta["source_path"] == "F:/papers/selected-paper.en.md"
    assert meta["source_name"] == "selected-paper.pdf"
    assert meta["block_id"] == "blk-1"
    assert meta["anchor_id"] == "a-1"
    assert meta["research_basket_evidence"] is True
    assert meta["ref_pack_state"] == "ready"


def test_selected_research_context_reference_prefers_library_match_path() -> None:
    selected_items = [
        {
            "kind": "reference",
            "title": "Collaborative filtering for implicit feedback datasets",
            "sourcePath": "F:/papers/current-paper.en.md",
            "libraryMatchPath": "F:/papers/upstream-matched.en.md",
            "libraryMatchStatus": "ready",
            "libraryMatchTitle": "Collaborative filtering for implicit feedback datasets",
            "libraryMatchDoi": "10.1109/ICDM.2008.22",
            "libraryMatchYear": "2008",
            "excerpt": "Hu et al. 2008 reference row.",
        }
    ]

    hits = _selected_research_context_evidence_hits(selected_items)

    assert len(hits) == 1
    meta = hits[0]["meta"]
    assert meta["source_path"] == "F:/papers/upstream-matched.en.md"
    assert meta["selected_context_source_path"] == "F:/papers/current-paper.en.md"
    assert meta["basket_source_role"] == "matched_library_paper"
    assert meta["doi"] == "10.1109/ICDM.2008.22"


def test_selected_research_context_unmatched_reference_uses_synthetic_source() -> None:
    selected_items = [
        {
            "kind": "reference",
            "title": "A hard to find preprint",
            "doi": "10.1234/example.1",
            "summary": "Only bibliographic metadata is available.",
        }
    ]

    hits = _selected_research_context_evidence_hits(selected_items)

    assert len(hits) == 1
    meta = hits[0]["meta"]
    assert meta["source_path"].startswith("__research_basket__/")
    assert meta["basket_source_role"] == "synthetic_basket_item"
    assert meta["source_name"].startswith("Research basket:")


def test_selected_research_context_evidence_merges_before_retrieval_hits() -> None:
    basket_hits = _selected_research_context_evidence_hits(
        [
            {
                "kind": "reader_selection",
                "title": "Selected paragraph",
                "sourcePath": "F:/papers/current.en.md",
                "blockId": "blk-selected",
                "excerpt": "The exact selected paragraph.",
            }
        ]
    )
    retrieval_hits = [
        {
            "text": "The retrieved duplicate paragraph.",
            "meta": {"source_path": "F:/papers/current.en.md", "block_id": "blk-selected"},
            "score": 4.0,
        },
        {
            "text": "A different supporting paragraph.",
            "meta": {"source_path": "F:/papers/current.en.md", "block_id": "blk-other"},
            "score": 3.0,
        },
    ]

    merged = _merge_selected_research_context_evidence_hits(retrieval_hits, basket_hits, limit=4)

    assert merged[0]["meta"]["research_basket_evidence"] is True
    assert len(merged) == 2
    assert merged[1]["meta"]["block_id"] == "blk-other"


def test_selected_research_context_evidence_contract_is_compact() -> None:
    hits = _selected_research_context_evidence_hits(
        [
            {
                "kind": "reader_selection",
                "title": "Selected paragraph",
                "sourcePath": "F:/papers/current.en.md",
                "sourceName": "current.pdf",
                "blockId": "blk-selected",
                "excerpt": "The exact selected paragraph.",
            }
        ]
    )

    contract = _selected_research_context_evidence_contract(hits)

    assert contract["version"] == 1
    assert contract["count"] == 1
    assert contract["items"][0]["source_name"] == "current.pdf"
    assert contract["items"][0]["block_id"] == "blk-selected"
