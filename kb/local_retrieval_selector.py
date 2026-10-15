from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .candidate_retriever import CandidateRetriever
from .hybrid_retriever import _hit_identity
from .local_dense_runtime import normalize_retrieval_mode
from .local_semantic_component_lifecycle import (
    LOCAL_SEMANTIC_COMPONENT_PRODUCT_ENABLED,
)
from .local_semantic_component_service import load_local_semantic_component_runtime
from .local_semantic_runtime import local_semantic_runtime_failure_reason
from .portable_local_semantic_runtime import PortableLocalSemanticRuntimeError
from .protected_dense_tail_retriever import ProtectedDenseTailRetriever
from .retrieval_source_scope import RetrievalSourceScope
from .local_semantic_runtime import LocalSemanticUnavailable


FUSION_OUTPUT_DEPTH = 10
FUSION_CANDIDATE_DEPTH = 40


class _StaticCandidateRetriever:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = copy.deepcopy(list(hits or []))

    @property
    def is_empty(self) -> bool:
        return not self._hits

    def search(self, _query: str, top_k: int = 6) -> list[dict]:
        return copy.deepcopy(self._hits[: max(1, int(top_k))])


class _SourceFilteredRetriever:
    def __init__(self, delegate: CandidateRetriever, scope: RetrievalSourceScope) -> None:
        self._delegate = delegate
        self._scope = scope

    @property
    def is_empty(self) -> bool:
        return self._delegate.is_empty

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        search = getattr(self._delegate, "search_scoped", None)
        if not callable(search):
            raise LocalSemanticUnavailable("scoped_dense_search_unavailable")
        return [
            hit
            for hit in search(
                query,
                source_paths=tuple(self._scope.paths),
                top_k=top_k,
            )
            if self._scope.contains(hit)
        ][: max(1, int(top_k))]


@dataclass(frozen=True)
class ProductRetrievalSelection:
    hits: list[dict]
    requested_mode: str
    used_mode: str
    fallback_reason_id: str = ""
    protected_prefix_preserved: bool = True
    dense_novelty_count: int = 0
    cold_start: bool = False
    index_cache_hit: bool = False

    def trace_payload(self) -> dict[str, object]:
        return {
            "requested_mode": self.requested_mode,
            "used_mode": self.used_mode,
            "fallback_reason_id": self.fallback_reason_id,
            "protected_prefix_preserved": self.protected_prefix_preserved,
            "dense_novelty_count": self.dense_novelty_count,
            "cold_start": self.cold_start,
            "index_cache_hit": self.index_cache_hit,
        }


def _fallback(
    original: list[dict],
    *,
    requested_mode: str,
    reason_id: str = "",
) -> ProductRetrievalSelection:
    return ProductRetrievalSelection(
        hits=copy.deepcopy(original),
        requested_mode=requested_mode,
        used_mode="bm25",
        fallback_reason_id=reason_id,
    )


def preserve_product_bm25(
    sparse_hits: list[dict],
    *,
    retrieval_mode: object,
    reason_id: str = "",
) -> ProductRetrievalSelection:
    return _fallback(
        list(sparse_hits or []),
        requested_mode=normalize_retrieval_mode(retrieval_mode),
        reason_id=str(reason_id or ""),
    )


def _append_sparse_remainder(fused: list[dict], sparse: list[dict]) -> list[dict]:
    target_count = max(len(sparse), min(FUSION_OUTPUT_DEPTH, len(fused)))
    out = copy.deepcopy(fused)
    emitted = {_hit_identity(hit) for hit in out}
    for hit in sparse:
        if len(out) >= target_count:
            break
        identity = _hit_identity(hit)
        if identity in emitted:
            continue
        out.append(copy.deepcopy(hit))
        emitted.add(identity)
    return out


def select_product_retrieval(
    *,
    query: str,
    sparse_hits: list[dict],
    db_dir: Path,
    retrieval_mode: object,
    source_scope: str,
    source_paths: Sequence[str] | None = None,
    runtime_service: object | None = None,
    product_enabled: bool = LOCAL_SEMANTIC_COMPONENT_PRODUCT_ENABLED,
) -> ProductRetrievalSelection:
    """Apply the accepted local hybrid candidate or return exact BM25 values."""

    original = copy.deepcopy(list(sparse_hits or []))
    requested_mode = normalize_retrieval_mode(retrieval_mode)
    if requested_mode == "bm25":
        return _fallback(original, requested_mode=requested_mode)
    if not product_enabled:
        return _fallback(
            original,
            requested_mode=requested_mode,
            reason_id="local_semantic_not_promoted",
        )
    scoped = str(source_scope or "").strip().lower() not in {"library", "full_library"}
    scope = RetrievalSourceScope.from_paths(source_paths)
    if scoped and not scope.paths:
        return _fallback(original, requested_mode=requested_mode, reason_id="local_semantic_scope_sources_missing")
    if scoped and any(not scope.contains(hit) for hit in original):
        return _fallback(original, requested_mode=requested_mode, reason_id="source_scope_violation")
    try:
        runtime = (
            runtime_service.load_ready(Path(db_dir))
            if runtime_service is not None
            else load_local_semantic_component_runtime(Path(db_dir))
        )
        dense: CandidateRetriever = runtime.retriever
        if scoped:
            dense = _SourceFilteredRetriever(dense, scope)
        candidate = ProtectedDenseTailRetriever(
            _StaticCandidateRetriever(original),
            dense,
            dense_novelty_quota=2,
            candidate_multiplier=4,
            protected_sparse_prefix=5,
            sparse_tail_window=5,
        )
        fused = candidate.search(str(query or ""), top_k=FUSION_OUTPUT_DEPTH)
    except Exception as exc:
        reason_id = (
            exc.code
            if isinstance(exc, PortableLocalSemanticRuntimeError)
            else local_semantic_runtime_failure_reason(exc)
        )
        return _fallback(
            original,
            requested_mode=requested_mode,
            reason_id=reason_id,
        )

    if any(
        str(hit.get("_fusion_slot") or "") == "dense_failure_sparse_fallback"
        for hit in fused
    ):
        return _fallback(
            original,
            requested_mode=requested_mode,
            reason_id="dense_query_failed",
        )
    prefix_depth = min(5, len(original))
    prefix_preserved = [
        _hit_identity(hit) for hit in fused[:prefix_depth]
    ] == [_hit_identity(hit) for hit in original[:prefix_depth]]
    if not prefix_preserved:
        return _fallback(
            original,
            requested_mode=requested_mode,
            reason_id="protected_prefix_violation",
        )
    if scoped and any(not scope.contains(hit) for hit in fused):
        return _fallback(
            original,
            requested_mode=requested_mode,
            reason_id="source_scope_violation",
        )
    selected = _append_sparse_remainder(fused, original)
    return ProductRetrievalSelection(
        hits=selected,
        requested_mode=requested_mode,
        used_mode="hybrid_local",
        protected_prefix_preserved=True,
        dense_novelty_count=sum(
            1
            for hit in fused
            if str(hit.get("_fusion_slot") or "")
            in {"dense_novelty_tail", "dense_fill"}
        ),
        cold_start=runtime.cold_start,
        index_cache_hit=runtime.index_cache_hit,
    )
