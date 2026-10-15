from __future__ import annotations

from kb.parent_collapsed_dense_retriever import (
    ParentCollapsedDenseRetriever,
    focused_child_representations,
)


def test_focused_children_use_exact_original_slices_and_frozen_windows() -> None:
    text = "".join(chr(0x4E00 + (index % 100)) for index in range(500))
    children = focused_child_representations(
        [{"id": "doc:7", "text": text}],
        token_offsets=lambda value: [(index, index + 1) for index in range(len(value))],
    )

    assert [(row.token_start, row.token_end) for row in children] == [
        (0, 256),
        (192, 448),
        (384, 500),
    ]
    assert all(row.text == text[row.token_start : row.token_end] for row in children)
    assert {row.parent_chunk_id for row in children} == {"doc:7"}


def test_short_parent_does_not_create_redundant_children() -> None:
    text = "x" * 320
    assert focused_child_representations(
        [{"id": "doc:1", "text": text}],
        token_offsets=lambda value: [(index, index + 1) for index in range(len(value))],
    ) == []


def test_child_score_returns_unchanged_parent_identity_text_and_locator() -> None:
    parents = [
        {"id": "doc-a:1", "text": "full parent A", "meta": {"page_start": 4}},
        {"id": "doc-b:2", "text": "full parent B", "meta": {"page_start": 9}},
    ]
    retriever = ParentCollapsedDenseRetriever(
        parents,
        parent_embeddings=[(0.0, 1.0), (0.6, 0.8)],
        child_embeddings={"doc-a:1": [(0, (1.0, 0.0))]},
        embed_query=lambda _query: (1.0, 0.0),
    )

    hits = retriever.search("semantic paraphrase", top_k=2)

    assert [hit["id"] for hit in hits] == ["doc-a:1", "doc-b:2"]
    assert hits[0]["text"] == "full parent A"
    assert hits[0]["meta"] == {"page_start": 4}
    assert hits[0]["_dense_representation_kind"] == "child"
    assert hits[0]["_dense_representation_ordinal"] == 0


def test_scoped_parent_search_preserves_locator_and_excludes_other_sources() -> None:
    parents = [
        *[{"id": f"outside:{i}", "text": "outside", "meta": {"source_path": "outside.md"}} for i in range(45)],
        {"id": "paper:1", "text": "full original", "meta": {"source_path": "paper.md", "page_start": 9, "block_id": "p9"}},
        {"id": "paper:2", "text": "bad conversion", "meta": {"source_path": "paper.md", "evidence_ready": False}},
    ]
    retriever = ParentCollapsedDenseRetriever(
        parents, [[1.0, 0.0]] * 45 + [[0.0, 1.0], [1.0, 0.0]],
        {"paper:1": [(1, [0.6, 0.8])]}, lambda _query: [1.0, 0.0],
    )
    hits = retriever.search_scoped("question", source_paths=["paper.md"], top_k=1)
    assert [hit["id"] for hit in hits] == ["paper:1"]
    assert hits[0]["text"] == "full original"
    assert hits[0]["meta"]["block_id"] == "p9"
    assert hits[0]["_dense_representation_kind"] == "child"
    assert retriever.search_scoped("question", source_paths=[], top_k=1) == []
