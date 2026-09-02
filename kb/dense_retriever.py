from __future__ import annotations

import copy
from collections.abc import Callable, Sequence

from .embedding_vectors import Embedding, normalize_embedding


QueryEmbedder = Callable[[str], Embedding]


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

    @property
    def is_empty(self) -> bool:
        return not self._chunks

    @property
    def dimension(self) -> int:
        return self._dimension

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        clean_query = str(query or "").strip()
        if not clean_query or not self._chunks:
            return []
        query_vector = normalize_embedding(
            self._embed_query(clean_query),
            expected_dim=self._dimension,
        )
        scored: list[tuple[float, int]] = []
        for idx, (chunk, chunk_vector) in enumerate(zip(self._chunks, self._embeddings)):
            meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
            if meta.get("evidence_ready") is False:
                continue
            score = float(sum(left * right for left, right in zip(query_vector, chunk_vector)))
            if score < self._min_score:
                continue
            scored.append((score, idx))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        hits: list[dict] = []
        for score, idx in scored[: max(1, int(top_k))]:
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
