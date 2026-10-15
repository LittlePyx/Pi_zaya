from __future__ import annotations

import copy

from .candidate_retriever import CandidateRetriever


def _hit_identity(hit: dict) -> str:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    return str(
        meta.get("chunk_id")
        or hit.get("chunk_id")
        or hit.get("id")
        or f"text:{str(hit.get('text') or '')[:160]}"
    )


def _source_identity(hit: dict) -> str:
    identity = _hit_identity(hit)
    if identity and not identity.startswith("text:"):
        return identity.split(":", 1)[0]
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    return str(meta.get("source_path") or meta.get("doc_id") or identity)


class HybridCandidateRetriever:
    """Fuse sparse and dense local candidates while keeping sparse fail-safe."""

    def __init__(
        self,
        sparse: CandidateRetriever,
        dense: CandidateRetriever,
        *,
        sparse_weight: float = 2.0,
        dense_weight: float = 1.0,
        rrf_k: float = 20.0,
        candidate_multiplier: int = 4,
        protected_sparse_prefix: int = 0,
        dense_source_gate_depth: int = 0,
    ) -> None:
        if sparse_weight <= 0.0 or dense_weight <= 0.0:
            raise ValueError("retrieval channel weights must be positive")
        if rrf_k <= 0.0:
            raise ValueError("rrf_k must be positive")
        self._sparse = sparse
        self._dense = dense
        self._sparse_weight = float(sparse_weight)
        self._dense_weight = float(dense_weight)
        self._rrf_k = float(rrf_k)
        self._candidate_multiplier = max(1, int(candidate_multiplier))
        self._protected_sparse_prefix = max(0, int(protected_sparse_prefix))
        self._dense_source_gate_depth = max(0, int(dense_source_gate_depth))

    @property
    def is_empty(self) -> bool:
        return self._sparse.is_empty and self._dense.is_empty

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        limit = max(1, int(top_k))
        candidate_limit = max(limit, limit * self._candidate_multiplier)
        sparse_hits = list(self._sparse.search(query, top_k=candidate_limit) or [])
        try:
            dense_hits = list(self._dense.search(query, top_k=candidate_limit) or [])
        except Exception:
            # Dense retrieval is optional. Provider/index failures must never
            # take down the established local BM25 path.
            return [copy.deepcopy(hit) for hit in sparse_hits[:limit]]

        if self._dense_source_gate_depth:
            allowed_sources = {
                _source_identity(hit)
                for hit in sparse_hits[: self._dense_source_gate_depth]
            }
            dense_hits = [
                hit for hit in dense_hits if _source_identity(hit) in allowed_sources
            ]

        protected = [
            copy.deepcopy(hit)
            for hit in sparse_hits[: min(limit, self._protected_sparse_prefix)]
        ]
        protected_ids = {_hit_identity(hit) for hit in protected}

        hit_map: dict[str, dict] = {}
        raw_scores: dict[str, dict[str, float]] = {}
        fused_scores: dict[str, float] = {}
        channels: dict[str, set[str]] = {}
        first_seen: dict[str, int] = {}
        seen_order = 0

        for channel, weight, hits in (
            ("sparse", self._sparse_weight, sparse_hits),
            ("dense", self._dense_weight, dense_hits),
        ):
            seen_in_channel: set[str] = set()
            for rank, raw_hit in enumerate(hits, start=1):
                identity = _hit_identity(raw_hit)
                if identity in seen_in_channel:
                    continue
                seen_in_channel.add(identity)
                if identity not in hit_map:
                    hit_map[identity] = copy.deepcopy(raw_hit)
                    first_seen[identity] = seen_order
                    seen_order += 1
                channels.setdefault(identity, set()).add(channel)
                raw_scores.setdefault(identity, {})[channel] = float(raw_hit.get("score", 0.0) or 0.0)
                fused_scores[identity] = fused_scores.get(identity, 0.0) + weight / (self._rrf_k + rank)

        ordered = sorted(
            hit_map,
            key=lambda identity: (
                fused_scores.get(identity, 0.0),
                -first_seen.get(identity, 0),
            ),
            reverse=True,
        )
        fused_hits: list[dict] = []
        for identity in ordered[: limit + len(protected)]:
            if identity in protected_ids:
                continue
            hit = hit_map[identity]
            scores = raw_scores.get(identity, {})
            hit["score"] = float(fused_scores.get(identity, 0.0))
            if "sparse" in scores:
                hit["_sparse_score"] = scores["sparse"]
            if "dense" in scores:
                hit["_dense_score"] = scores["dense"]
            hit["_retrieval_channels"] = sorted(channels.get(identity, set()))
            fused_hits.append(hit)
            if len(fused_hits) >= limit - len(protected):
                break
        return protected + fused_hits
