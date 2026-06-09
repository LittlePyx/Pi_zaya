from __future__ import annotations

from kb.task_runtime import (
    _effective_query_scope,
    _normalize_query_scope,
    _query_scope_prompt_block,
)


def test_normalize_query_scope_aliases() -> None:
    assert _normalize_query_scope("current-paper") == "current_paper"
    assert _normalize_query_scope("citation_shelf") == "basket"
    assert _normalize_query_scope("full_library") == "library"
    assert _normalize_query_scope("unknown") == ""


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
