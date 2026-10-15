from __future__ import annotations

import copy
from typing import Any

from .candidate_retriever import CandidateRetriever
from .hybrid_retriever import _hit_identity


def _unique_ranked(hits: list[dict]) -> list[tuple[str, int, dict]]:
    ranked: list[tuple[str, int, dict]] = []
    seen: set[str] = set()
    for rank, hit in enumerate(hits, start=1):
        identity = _hit_identity(hit)
        if identity in seen:
            continue
        seen.add(identity)
        ranked.append((identity, rank, hit))
    return ranked


class ProtectedDenseTailRetriever:
    """Keep a sparse prefix and reserve bounded tail space for dense novelty.

    Sparse tail survivors are selected by dense support and then returned in
    their original sparse order. The retriever is intentionally label-agnostic:
    it only sees the two ranked hit lists.
    """

    def __init__(
        self,
        sparse: CandidateRetriever,
        dense: CandidateRetriever,
        *,
        dense_novelty_quota: int,
        candidate_multiplier: int = 4,
        protected_sparse_prefix: int = 5,
        sparse_tail_window: int = 5,
    ) -> None:
        if dense_novelty_quota <= 0:
            raise ValueError("dense_novelty_quota must be positive")
        if protected_sparse_prefix < 0 or sparse_tail_window < 0:
            raise ValueError("sparse prefix and tail window cannot be negative")
        if dense_novelty_quota > sparse_tail_window:
            raise ValueError("dense novelty quota cannot exceed sparse tail window")
        self._sparse = sparse
        self._dense = dense
        self._dense_novelty_quota = int(dense_novelty_quota)
        self._candidate_multiplier = max(1, int(candidate_multiplier))
        self._protected_sparse_prefix = int(protected_sparse_prefix)
        self._sparse_tail_window = int(sparse_tail_window)

    @property
    def is_empty(self) -> bool:
        return self._sparse.is_empty and self._dense.is_empty

    @staticmethod
    def _decorated_hit(
        raw_hit: dict,
        *,
        identity: str,
        sparse_ranks: dict[str, int],
        dense_ranks: dict[str, int],
        slot: str,
    ) -> dict:
        hit = copy.deepcopy(raw_hit)
        channels: list[str] = []
        if identity in sparse_ranks:
            hit["_sparse_rank"] = sparse_ranks[identity]
            channels.append("sparse")
        if identity in dense_ranks:
            hit["_dense_rank"] = dense_ranks[identity]
            channels.append("dense")
        hit["_retrieval_channels"] = channels
        hit["_fusion_strategy"] = "protected_dense_tail"
        hit["_fusion_slot"] = slot
        return hit

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        limit = max(1, int(top_k))
        candidate_limit = max(limit, limit * self._candidate_multiplier)
        sparse_hits = list(self._sparse.search(query, top_k=candidate_limit) or [])
        sparse_ranked = _unique_ranked(sparse_hits)
        sparse_ranks = {identity: rank for identity, rank, _ in sparse_ranked}
        try:
            dense_hits = list(self._dense.search(query, top_k=candidate_limit) or [])
        except Exception:
            fallback: list[dict] = []
            for identity, _, raw_hit in sparse_ranked[:limit]:
                fallback.append(
                    self._decorated_hit(
                        raw_hit,
                        identity=identity,
                        sparse_ranks=sparse_ranks,
                        dense_ranks={},
                        slot="dense_failure_sparse_fallback",
                    )
                )
            return fallback

        dense_ranked = _unique_ranked(dense_hits)
        dense_ranks = {identity: rank for identity, rank, _ in dense_ranked}
        sparse_by_id = {identity: hit for identity, _, hit in sparse_ranked}
        dense_by_id = {identity: hit for identity, _, hit in dense_ranked}

        protected_count = min(limit, self._protected_sparse_prefix, len(sparse_ranked))
        protected = sparse_ranked[:protected_count]
        sparse_tail = sparse_ranked[
            protected_count : protected_count + self._sparse_tail_window
        ]
        sparse_exclusion_ids = {
            identity
            for identity, _, _ in sparse_ranked[
                : self._protected_sparse_prefix + self._sparse_tail_window
            ]
        }
        novelty = [row for row in dense_ranked if row[0] not in sparse_exclusion_ids][
            : min(self._dense_novelty_quota, max(0, limit - protected_count))
        ]

        survivor_count = min(
            len(sparse_tail),
            max(0, limit - protected_count - len(novelty)),
        )
        supported_survivors = sorted(
            sparse_tail,
            key=lambda row: (dense_ranks.get(row[0], candidate_limit + 1), row[1]),
        )[:survivor_count]
        supported_survivors.sort(key=lambda row: row[1])

        selected: list[tuple[str, dict, str]] = [
            (identity, raw_hit, "protected_sparse_prefix")
            for identity, _, raw_hit in protected
        ]
        selected.extend(
            (identity, raw_hit, "sparse_tail_survivor")
            for identity, _, raw_hit in supported_survivors
        )
        selected.extend(
            (identity, raw_hit, "dense_novelty_tail")
            for identity, _, raw_hit in novelty
        )

        emitted = {identity for identity, _, _ in selected}
        for identity, _, raw_hit in sparse_ranked:
            if len(selected) >= limit:
                break
            if identity not in emitted:
                selected.append((identity, raw_hit, "sparse_fill"))
                emitted.add(identity)
        for identity, _, raw_hit in dense_ranked:
            if len(selected) >= limit:
                break
            if identity not in emitted:
                selected.append((identity, raw_hit, "dense_fill"))
                emitted.add(identity)

        return [
            self._decorated_hit(
                raw_hit,
                identity=identity,
                sparse_ranks=sparse_ranks,
                dense_ranks=dense_ranks,
                slot=slot,
            )
            for identity, raw_hit, slot in selected[:limit]
        ]
