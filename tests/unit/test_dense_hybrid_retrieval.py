from __future__ import annotations

from types import SimpleNamespace

from kb.cross_encoder_reranker import CrossEncoderReranker, CrossEncoderRerankingRetriever
from kb.dense_retriever import DenseRetriever
from kb.hybrid_retriever import HybridCandidateRetriever
from kb.retrieval_engine import _search_hits_with_fallback
from kb.retriever import BM25Retriever


def _chunks() -> list[dict]:
    return [
        {
            "id": "exact",
            "text": "ADMM reconstruction uses an explicit optimization loop",
            "meta": {"source_path": "exact.pdf"},
        },
        {
            "id": "semantic",
            "text": "A compact adaptation updates only a small parameter subspace",
            "meta": {"source_path": "semantic.pdf"},
        },
        {
            "id": "unsafe",
            "text": "unreliable semantic evidence",
            "meta": {"source_path": "unsafe.pdf", "evidence_ready": False},
        },
        {
            "id": "filler",
            "text": "quantum detector calibration",
            "meta": {"source_path": "filler.pdf"},
        },
    ]


def test_dense_retriever_finds_semantic_hit_without_lexical_overlap() -> None:
    chunks = _chunks()
    query_vectors = {"avoid full fine tuning": [0.0, 1.0, 0.0]}
    retriever = DenseRetriever(
        chunks,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        lambda query: query_vectors[query],
    )

    hits = retriever.search("avoid full fine tuning", top_k=3)

    assert hits[0]["id"] == "semantic"
    assert all(hit["id"] != "unsafe" for hit in hits)
    hits[0]["meta"]["request_only"] = True
    assert "request_only" not in retriever.search("avoid full fine tuning", top_k=1)[0]["meta"]


def test_hybrid_retriever_rrf_preserves_both_candidate_channels() -> None:
    chunks = _chunks()
    sparse = BM25Retriever(chunks)
    dense = DenseRetriever(
        chunks,
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.2, 0.8]],
        lambda _query: [0.0, 1.0],
    )
    hybrid = HybridCandidateRetriever(sparse, dense)

    hits = hybrid.search("ADMM reconstruction", top_k=3)

    assert {hit["id"] for hit in hits} >= {"exact", "semantic"}
    exact = next(hit for hit in hits if hit["id"] == "exact")
    assert "sparse" in exact["_retrieval_channels"]
    semantic = next(hit for hit in hits if hit["id"] == "semantic")
    assert "dense" in semantic["_retrieval_channels"]


def test_hybrid_retriever_falls_back_to_sparse_when_dense_fails() -> None:
    class BrokenDense:
        is_empty = False

        def search(self, _query: str, top_k: int = 6) -> list[dict]:
            raise RuntimeError("index unavailable")

    sparse = BM25Retriever(_chunks())
    hybrid = HybridCandidateRetriever(sparse, BrokenDense())

    assert hybrid.search("ADMM", top_k=1)[0]["id"] == "exact"


def test_cross_encoder_reranker_reorders_and_fails_open() -> None:
    hits = [
        {"id": "a", "score": 3.0, "text": "broad context", "meta": {}},
        {"id": "b", "score": 2.0, "text": "direct evidence", "meta": {}},
    ]
    reranker = CrossEncoderReranker(
        lambda pairs: [0.1 if "broad" in evidence else 0.9 for _query, evidence in pairs]
    )
    assert [hit["id"] for hit in reranker.rerank("question", hits)] == ["b", "a"]

    broken = CrossEncoderReranker(lambda _pairs: (_ for _ in ()).throw(RuntimeError("offline")))
    assert [hit["id"] for hit in broken.rerank("question", hits)] == ["a", "b"]


def test_hybrid_and_cross_encoder_adapters_fit_existing_retrieval_pipeline() -> None:
    chunks = _chunks()
    sparse = BM25Retriever(chunks)
    dense = DenseRetriever(
        chunks,
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.2, 0.8]],
        lambda _query: [0.0, 1.0],
    )
    hybrid = HybridCandidateRetriever(sparse, dense)
    retriever = CrossEncoderRerankingRetriever(
        hybrid,
        CrossEncoderReranker(
            lambda pairs: [1.0 if "parameter subspace" in evidence else 0.0 for _query, evidence in pairs]
        ),
    )

    hits, _scores, _query, _translated, _variants = _search_hits_with_fallback(
        "avoid full fine tuning",
        retriever,
        top_k=2,
        settings=SimpleNamespace(api_key=None, query_expansion_enabled=False),
        allow_translate=False,
    )

    assert hits[0]["id"] == "semantic"
