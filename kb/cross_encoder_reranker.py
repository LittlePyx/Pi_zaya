from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence

from .candidate_retriever import CandidateRetriever


Pair = tuple[str, str]
PairScorer = Callable[[Sequence[Pair]], Sequence[float]]


class CrossEncoderReranker:
    """Model-agnostic cross-encoder adapter with deterministic fail-open behavior.

    The injected scorer receives ``(query, evidence_text)`` pairs.  Keeping the
    model adapter outside this module prevents model downloads, credentials, and
    network policy from leaking into retrieval ranking code.
    """

    def __init__(
        self,
        score_pairs: PairScorer,
        *,
        candidate_limit: int = 40,
        protected_prefix: int = 0,
    ) -> None:
        self._score_pairs = score_pairs
        self._candidate_limit = max(1, int(candidate_limit))
        self._protected_prefix = max(0, int(protected_prefix))

    @property
    def candidate_limit(self) -> int:
        return self._candidate_limit

    def rerank(self, query: str, hits: list[dict], *, top_k: int | None = None) -> list[dict]:
        source_hits = [copy.deepcopy(hit) for hit in list(hits or []) if isinstance(hit, dict)]
        if not str(query or "").strip() or not source_hits:
            return source_hits[: max(0, int(top_k))] if top_k is not None else source_hits

        protected_count = min(len(source_hits), self._protected_prefix)
        protected = source_hits[:protected_count]
        rerankable = source_hits[protected_count : self._candidate_limit]
        if not rerankable:
            return source_hits[: max(0, int(top_k))] if top_k is not None else source_hits
        pairs = [(str(query), str(hit.get("text") or "")) for hit in rerankable]
        try:
            raw_scores = list(self._score_pairs(pairs))
            if len(raw_scores) != len(rerankable):
                raise ValueError(
                    "cross-encoder score count mismatch: "
                    f"expected {len(rerankable)}, got {len(raw_scores)}"
                )
            scores = [float(score) for score in raw_scores]
            if not all(math.isfinite(score) for score in scores):
                raise ValueError("cross-encoder scores must be finite")
        except Exception:
            return source_hits[: max(0, int(top_k))] if top_k is not None else source_hits

        ranked: list[tuple[float, int, dict]] = []
        for index, (score, hit) in enumerate(zip(scores, rerankable)):
            hit["_pre_rerank_score"] = float(hit.get("score", 0.0) or 0.0)
            hit["cross_encoder_score"] = score
            hit["score"] = score
            ranked.append((score, index, hit))
        ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        output = protected + [hit for _score, _index, hit in ranked]
        output.extend(source_hits[self._candidate_limit :])
        if top_k is not None:
            return output[: max(0, int(top_k))]
        return output


class CrossEncoderRerankingRetriever:
    """Candidate-retriever adapter that reranks a bounded upstream window."""

    def __init__(self, retriever: CandidateRetriever, reranker: CrossEncoderReranker) -> None:
        self._retriever = retriever
        self._reranker = reranker

    @property
    def is_empty(self) -> bool:
        return self._retriever.is_empty

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        limit = max(1, int(top_k))
        candidates = self._retriever.search(
            query,
            top_k=max(limit, self._reranker.candidate_limit),
        )
        return self._reranker.rerank(query, candidates, top_k=limit)
