from __future__ import annotations

import math
import random

import pytest

import kb.dense_retriever as dense_module
from kb.dense_retriever import DenseRetriever


def _normalized(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def _corpus(count: int, dimension: int) -> tuple[list[dict], list[list[float]]]:
    randomizer = random.Random(20260903)
    chunks = [
        {
            "id": f"doc-{index // 8}:chunk-{index}",
            "text": f"evidence {index}",
            "meta": {"evidence_ready": index % 17 != 0},
        }
        for index in range(count)
    ]
    vectors = [
        _normalized([randomizer.uniform(-1.0, 1.0) for _ in range(dimension)])
        for _ in chunks
    ]
    return chunks, vectors


def test_vectorized_dense_ranking_matches_scalar_for_seeded_queries(
    monkeypatch,
) -> None:
    pytest.importorskip("numpy")
    chunks, vectors = _corpus(256, 64)
    randomizer = random.Random(42)
    queries = {
        f"query-{index}": _normalized(
            [randomizer.uniform(-1.0, 1.0) for _ in range(64)]
        )
        for index in range(40)
    }
    vectorized = DenseRetriever(chunks, vectors, queries.__getitem__)
    monkeypatch.setattr(dense_module, "_optional_numpy", lambda: None)
    scalar = DenseRetriever(chunks, vectors, queries.__getitem__)

    assert vectorized.scoring_backend == "numpy_float32_exact_shortlist"
    assert scalar.scoring_backend == "python_scalar"
    for query in queries:
        vector_hits = vectorized.search(query, top_k=40)
        scalar_hits = scalar.search(query, top_k=40)
        assert [hit["id"] for hit in vector_hits] == [hit["id"] for hit in scalar_hits]
        assert [hit["score"] for hit in vector_hits] == pytest.approx(
            [hit["score"] for hit in scalar_hits], abs=2e-7
        )


def test_vectorized_dense_ranking_keeps_input_order_for_exact_ties() -> None:
    pytest.importorskip("numpy")
    chunks = [
        {"id": f"doc:chunk-{index}", "text": str(index), "meta": {}}
        for index in range(5)
    ]
    retriever = DenseRetriever(
        chunks,
        [[1.0, 0.0] for _ in chunks],
        lambda _query: [1.0, 0.0],
    )

    assert [hit["id"] for hit in retriever.search("tie", top_k=5)] == [
        f"doc:chunk-{index}" for index in range(5)
    ]


def test_dense_retriever_falls_back_when_numpy_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(dense_module, "_optional_numpy", lambda: None)
    retriever = DenseRetriever(
        [{"id": "a", "text": "a", "meta": {}}],
        [[1.0, 0.0]],
        lambda _query: [1.0, 0.0],
    )

    assert retriever.scoring_backend == "python_scalar"
    assert retriever.search("query", top_k=1)[0]["id"] == "a"
