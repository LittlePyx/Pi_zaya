from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kb.embedding_vectors import Embedding, normalize_embedding
from kb.retrieval_source_scope import RetrievalSourceScope


TokenOffsets = Callable[[str], Sequence[tuple[int, int]]]
QueryEmbedder = Callable[[str], Embedding]


@dataclass(frozen=True)
class FocusedChildRepresentation:
    parent_chunk_id: str
    ordinal: int
    token_start: int
    token_end: int
    text: str


def focused_child_representations(
    chunks: Sequence[dict],
    *,
    token_offsets: TokenOffsets,
    window_tokens: int = 256,
    stride_tokens: int = 192,
    minimum_parent_tokens: int = 321,
) -> list[FocusedChildRepresentation]:
    window = max(8, int(window_tokens))
    stride = max(1, int(stride_tokens))
    minimum = max(window + 1, int(minimum_parent_tokens))
    if stride > window:
        raise ValueError("focused child stride cannot exceed its window")
    children: list[FocusedChildRepresentation] = []
    for chunk in chunks:
        parent_id = str(chunk.get("id") or "").strip()
        text = str(chunk.get("text") or "")
        if not parent_id or not text:
            continue
        offsets = [
            (int(start), int(end))
            for start, end in token_offsets(text)
            if int(start) >= 0 and int(end) > int(start) and int(end) <= len(text)
        ]
        if len(offsets) < minimum:
            continue
        ordinal = 0
        token_start = 0
        while token_start < len(offsets):
            token_end = min(len(offsets), token_start + window)
            char_start = offsets[token_start][0]
            char_end = offsets[token_end - 1][1]
            child_text = text[char_start:char_end]
            if child_text:
                children.append(
                    FocusedChildRepresentation(
                        parent_chunk_id=parent_id,
                        ordinal=ordinal,
                        token_start=token_start,
                        token_end=token_end,
                        text=child_text,
                    )
                )
                ordinal += 1
            if token_end >= len(offsets):
                break
            token_start += stride
    return children


class ParentCollapsedDenseRetriever:
    """Score full and focused representations, then return unchanged parents."""

    def __init__(
        self,
        parent_chunks: Sequence[dict],
        parent_embeddings: Sequence[Embedding],
        child_embeddings: dict[str, Sequence[tuple[int, Embedding]]],
        embed_query: QueryEmbedder,
        *,
        min_score: float = -1.0,
    ) -> None:
        self._parents = list(parent_chunks)
        if len(self._parents) != len(parent_embeddings):
            raise ValueError("parent chunk and embedding counts must match")
        self._embed_query = embed_query
        self._min_score = float(min_score)
        self._dimension = 0
        self._parent_embeddings: list[tuple[float, ...]] = []
        for vector in parent_embeddings:
            normalized = normalize_embedding(vector, expected_dim=self._dimension or None)
            if not self._dimension:
                self._dimension = len(normalized)
            self._parent_embeddings.append(normalized)
        self._child_embeddings: dict[str, list[tuple[int, tuple[float, ...]]]] = {}
        for parent_id, rows in child_embeddings.items():
            self._child_embeddings[str(parent_id)] = [
                (
                    int(ordinal),
                    normalize_embedding(vector, expected_dim=self._dimension),
                )
                for ordinal, vector in rows
            ]

    @property
    def is_empty(self) -> bool:
        return not self._parents

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        return self._search(query, top_k=top_k, scope=None)

    def search_scoped(
        self, query: str, *, source_paths: Sequence[str], top_k: int = 6,
    ) -> list[dict]:
        scope = RetrievalSourceScope.from_paths(source_paths)
        if not scope.paths:
            return []
        return self._search(query, top_k=top_k, scope=scope)

    def _search(
        self, query: str, *, top_k: int, scope: RetrievalSourceScope | None,
    ) -> list[dict]:
        clean_query = str(query or "").strip()
        if not clean_query or not self._parents:
            return []
        query_vector = normalize_embedding(
            self._embed_query(clean_query),
            expected_dim=self._dimension,
        )
        scored: list[tuple[float, int, str, int]] = []
        for index, (parent, parent_vector) in enumerate(
            zip(self._parents, self._parent_embeddings)
        ):
            meta = parent.get("meta") if isinstance(parent.get("meta"), dict) else {}
            if meta.get("evidence_ready") is False:
                continue
            if scope is not None and not scope.contains(parent):
                continue
            best_score = float(
                sum(left * right for left, right in zip(query_vector, parent_vector))
            )
            best_kind = "parent"
            best_ordinal = -1
            parent_id = str(parent.get("id") or "")
            for ordinal, child_vector in self._child_embeddings.get(parent_id, []):
                child_score = float(
                    sum(left * right for left, right in zip(query_vector, child_vector))
                )
                if child_score > best_score:
                    best_score = child_score
                    best_kind = "child"
                    best_ordinal = ordinal
            if best_score >= self._min_score:
                scored.append((best_score, index, best_kind, best_ordinal))
        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        hits: list[dict] = []
        for score, parent_index, representation_kind, ordinal in scored[
            : max(1, int(top_k))
        ]:
            parent = self._parents[parent_index]
            hits.append(
                {
                    "score": score,
                    "dense_score": score,
                    "id": parent.get("id", str(parent_index)),
                    "text": parent.get("text", ""),
                    "meta": copy.deepcopy(parent.get("meta", {})),
                    "_dense_representation_kind": representation_kind,
                    "_dense_representation_ordinal": ordinal,
                }
            )
        return hits
