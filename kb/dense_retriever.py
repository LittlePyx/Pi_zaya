from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any

from .embedding_vectors import Embedding, normalize_embedding


QueryEmbedder = Callable[[str], Embedding]


def _optional_numpy() -> Any | None:
    try:
        import numpy
    except ImportError:
        return None
    return numpy


class DenseRetriever:
    """Exact cosine retrieval over caller-supplied local chunk embeddings.

    This class intentionally owns no model, credential, network, or persistence
    policy.  Those concerns must be implemented by a separate index/provider
    layer before dense retrieval is enabled in the product.
    """

    def __init__(
        self,
        chunks: list[dict],
        chunk_embeddings: Sequence[Embedding],
        embed_query: QueryEmbedder,
        *,
        min_score: float = 0.0,
    ) -> None:
        self._chunks = list(chunks or [])
        if len(self._chunks) != len(chunk_embeddings):
            raise ValueError(
                "chunk and embedding counts must match: "
                f"{len(self._chunks)} chunks, {len(chunk_embeddings)} embeddings"
            )
        self._embed_query = embed_query
        self._min_score = float(min_score)
        self._dimension = 0
        self._embeddings: list[tuple[float, ...]] = []
        for raw_vector in chunk_embeddings:
            normalized = normalize_embedding(
                raw_vector,
                expected_dim=self._dimension or None,
            )
            if not self._dimension:
                self._dimension = len(normalized)
            self._embeddings.append(normalized)
        self._eligible_indices = tuple(
            index
            for index, chunk in enumerate(self._chunks)
            if (chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}).get(
                "evidence_ready"
            )
            is not False
        )
        self._numpy = _optional_numpy()
        self._vector_matrix: Any | None = None
        if self._numpy is not None and self._embeddings:
            matrix = self._numpy.asarray(self._embeddings, dtype=self._numpy.float32)
            matrix.setflags(write=False)
            self._vector_matrix = matrix

    @property
    def is_empty(self) -> bool:
        return not self._chunks

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def scoring_backend(self) -> str:
        return (
            "numpy_float32_exact_shortlist"
            if self._vector_matrix is not None
            else "python_scalar"
        )

    def _scalar_scores(
        self,
        query_vector: tuple[float, ...],
        indices: Sequence[int],
    ) -> list[tuple[float, int]]:
        return [
            (
                float(
                    sum(
                        left * right
                        for left, right in zip(query_vector, self._embeddings[idx])
                    )
                ),
                idx,
            )
            for idx in indices
        ]

    def _vectorized_exact_scores(
        self,
        query_vector: tuple[float, ...],
        *,
        limit: int,
    ) -> list[tuple[float, int]]:
        vector = self._numpy.asarray(query_vector, dtype=self._numpy.float32)
        approximate = self._vector_matrix @ vector
        eligible_approximate = sorted(
            (
                (float(approximate[idx]), idx)
                for idx in self._eligible_indices
                if float(approximate[idx]) >= self._min_score
            ),
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        if len(eligible_approximate) >= limit:
            boundary = eligible_approximate[limit - 1][0]
        else:
            boundary = self._min_score
        # A normalized dot product's float32 accumulation/conversion error is
        # bounded conservatively by a small multiple of dimension * epsilon.
        # Only this boundary band can change top-k membership, so recomputing
        # it with the original Python formula preserves exact scores/order.
        margin = max(
            1e-7,
            8.0 * self._dimension * float(self._numpy.finfo(self._numpy.float32).eps),
        )
        shortlist = [
            idx
            for idx in self._eligible_indices
            if float(approximate[idx]) >= boundary - margin
        ]
        exact = [
            item
            for item in self._scalar_scores(query_vector, shortlist)
            if item[0] >= self._min_score
        ]
        if len(exact) < min(limit, len(self._eligible_indices)):
            return [
                item
                for item in self._scalar_scores(query_vector, self._eligible_indices)
                if item[0] >= self._min_score
            ]
        return exact

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        clean_query = str(query or "").strip()
        if not clean_query or not self._chunks:
            return []
        limit = max(1, int(top_k))
        query_vector = normalize_embedding(
            self._embed_query(clean_query),
            expected_dim=self._dimension,
        )
        scored: list[tuple[float, int]] = []
        if self._vector_matrix is not None:
            scored = self._vectorized_exact_scores(query_vector, limit=limit)
        else:
            scored = [
                item
                for item in self._scalar_scores(query_vector, self._eligible_indices)
                if item[0] >= self._min_score
            ]
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        hits: list[dict] = []
        for score, idx in scored[:limit]:
            chunk = self._chunks[idx]
            hits.append(
                {
                    "score": score,
                    "dense_score": score,
                    "id": chunk.get("id", str(idx)),
                    "text": chunk.get("text", ""),
                    "meta": copy.deepcopy(chunk.get("meta", {})),
                }
            )
        return hits
