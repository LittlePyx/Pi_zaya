from __future__ import annotations

import pytest

from kb.protected_dense_tail_retriever import ProtectedDenseTailRetriever


def _hit(chunk_id: str, score: float = 0.0) -> dict:
    return {"text": chunk_id, "score": score, "meta": {"chunk_id": chunk_id}}


class _RankedRetriever:
    def __init__(self, identities: list[str], *, fail: bool = False) -> None:
        self._hits = [
            _hit(identity, 1.0 / rank) for rank, identity in enumerate(identities, 1)
        ]
        self._fail = fail

    @property
    def is_empty(self) -> bool:
        return not self._hits

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        del query
        if self._fail:
            raise RuntimeError("synthetic dense failure")
        return self._hits[:top_k]


def _ids(hits: list[dict]) -> list[str]:
    return [str(hit["meta"]["chunk_id"]) for hit in hits]


def test_one_dense_tail_slot_preserves_prefix_and_supported_sparse_tail() -> None:
    sparse = _RankedRetriever([f"s{rank}" for rank in range(1, 13)])
    dense = _RankedRetriever(["n1", "s10", "s7", "n2"])
    retriever = ProtectedDenseTailRetriever(sparse, dense, dense_novelty_quota=1)

    hits = retriever.search("label-agnostic query", top_k=10)

    assert _ids(hits) == ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s10", "n1"]
    assert [hit["_fusion_slot"] for hit in hits[:5]] == ["protected_sparse_prefix"] * 5
    assert hits[-1]["_fusion_slot"] == "dense_novelty_tail"
    assert hits[-1]["_dense_rank"] == 1
    assert "_sparse_rank" not in hits[-1]
    assert hits[8]["_sparse_rank"] == 10
    assert hits[8]["_dense_rank"] == 2


def test_two_dense_tail_slots_are_novel_deduplicated_and_dense_ranked() -> None:
    sparse = _RankedRetriever([f"s{rank}" for rank in range(1, 11)])
    dense = _RankedRetriever(["s1", "n1", "n1", "n2", "s8", "n3"])
    retriever = ProtectedDenseTailRetriever(sparse, dense, dense_novelty_quota=2)

    hits = retriever.search("query", top_k=10)

    assert _ids(hits[:5]) == ["s1", "s2", "s3", "s4", "s5"]
    assert _ids(hits[-2:]) == ["n1", "n2"]
    assert len(_ids(hits)) == len(set(_ids(hits))) == 10
    assert [hit["_dense_rank"] for hit in hits[-2:]] == [2, 4]


def test_dense_failure_returns_sparse_order_without_raising() -> None:
    sparse = _RankedRetriever([f"s{rank}" for rank in range(1, 9)])
    dense = _RankedRetriever([], fail=True)
    retriever = ProtectedDenseTailRetriever(sparse, dense, dense_novelty_quota=1)

    hits = retriever.search("query", top_k=6)

    assert _ids(hits) == ["s1", "s2", "s3", "s4", "s5", "s6"]
    assert all(hit["_fusion_slot"] == "dense_failure_sparse_fallback" for hit in hits)
    assert all(hit["_retrieval_channels"] == ["sparse"] for hit in hits)


def test_short_sparse_list_fills_from_dense_without_duplicates() -> None:
    sparse = _RankedRetriever(["s1", "s2"])
    dense = _RankedRetriever(["s1", "n1", "n2", "n3"])
    retriever = ProtectedDenseTailRetriever(sparse, dense, dense_novelty_quota=1)

    hits = retriever.search("query", top_k=5)

    assert _ids(hits) == ["s1", "s2", "n1", "n2", "n3"]
    assert len(_ids(hits)) == len(set(_ids(hits)))


@pytest.mark.parametrize(
    ("quota", "prefix", "window"),
    [(0, 5, 5), (6, 5, 5), (1, -1, 5), (1, 5, -1)],
)
def test_invalid_safety_contract_is_rejected(
    quota: int, prefix: int, window: int
) -> None:
    with pytest.raises(ValueError):
        ProtectedDenseTailRetriever(
            _RankedRetriever([]),
            _RankedRetriever([]),
            dense_novelty_quota=quota,
            protected_sparse_prefix=prefix,
            sparse_tail_window=window,
        )
