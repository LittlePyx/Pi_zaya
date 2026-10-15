from __future__ import annotations

import copy
from collections.abc import Sequence

from .dense_retriever import DenseRetriever
from .embedding_vectors import normalize_embedding
from .retrieval_source_scope import RetrievalSourceScope


class SourceScopedDenseRetriever(DenseRetriever):
    """Keep the frozen full-index kernel; score only explicitly selected chunks."""

    def search_scoped(
        self, query: str, *, source_paths: Sequence[str], top_k: int = 6,
    ) -> list[dict]:
        scope = RetrievalSourceScope.from_paths(source_paths)
        indices = tuple(idx for idx in self._eligible_indices if scope.contains(self._chunks[idx]))
        if not str(query or "").strip() or not indices:
            return []
        vector = normalize_embedding(self._embed_query(query.strip()), expected_dim=self._dimension)
        scores = sorted(
            (item for item in self._scalar_scores(vector, indices) if item[0] >= self._min_score),
            key=lambda item: (item[0], -item[1]), reverse=True,
        )
        return [
            {
                "id": self._chunks[idx].get("id", str(idx)),
                "text": self._chunks[idx].get("text", ""),
                "meta": copy.deepcopy(self._chunks[idx].get("meta", {})),
                "score": score,
                "dense_score": score,
            }
            for score, idx in scores[:max(1, int(top_k))]
        ]
