from __future__ import annotations

import copy
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .candidate_retriever import CandidateRetriever
from .embedding_index import EmbeddingIndexError, EmbeddingIndexStore
from .hybrid_retriever import _hit_identity, _source_identity
from .protected_dense_tail_retriever import ProtectedDenseTailRetriever
from .retrieval_observability import record_retrieval_shadow_observation


QWEN_PROVIDER_IDENTITY = (
    "55229348d5bbe4b1e1935aeb8ffe57cc03c527685b1d08d44f11a56a54525cb8"
)
SHADOW_OUTPUT_DEPTH = 10
SHADOW_CANDIDATE_DEPTH = 40


@dataclass(frozen=True)
class DenseShadowRuntime:
    retriever: CandidateRetriever
    index_cache_hit: bool
    cold_start: bool


@dataclass(frozen=True)
class ShadowSubmission:
    status: str
    future: Future[dict] | None = None


class _StaticCandidateRetriever:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = copy.deepcopy(list(hits or []))

    @property
    def is_empty(self) -> bool:
        return not self._hits

    def search(self, _query: str, top_k: int = 6) -> list[dict]:
        return copy.deepcopy(self._hits[: max(1, int(top_k))])


class _SourceFilteredRetriever:
    def __init__(self, delegate: CandidateRetriever, allowed_sources: set[str]) -> None:
        self._delegate = delegate
        self._allowed_sources = set(allowed_sources)

    @property
    def is_empty(self) -> bool:
        return self._delegate.is_empty

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        candidate_depth = max(SHADOW_CANDIDATE_DEPTH, int(top_k))
        return [
            hit
            for hit in self._delegate.search(query, top_k=candidate_depth)
            if _source_identity(hit) in self._allowed_sources
        ][: max(1, int(top_k))]


def _sequence_ids(hits: list[dict], depth: int) -> list[str]:
    return [_hit_identity(hit) for hit in list(hits or [])[: max(0, int(depth))]]


def _positional_changes(left: list[str], right: list[str], depth: int) -> int:
    return sum(
        1
        for index in range(max(0, int(depth)))
        if (left[index] if index < len(left) else None)
        != (right[index] if index < len(right) else None)
    )


def compare_shadow_rankings(
    *,
    product_hits: list[dict],
    shadow_hits: list[dict],
    scoped: bool,
) -> dict:
    product_ids = _sequence_ids(product_hits, SHADOW_OUTPUT_DEPTH)
    shadow_ids = _sequence_ids(shadow_hits, SHADOW_OUTPUT_DEPTH)
    prefix_depth = min(5, len(product_ids))
    allowed_sources = {_source_identity(hit) for hit in product_hits}
    source_escapes = (
        sum(1 for hit in shadow_hits if _source_identity(hit) not in allowed_sources)
        if scoped
        else 0
    )
    return {
        "product_hit_count": len(product_hits),
        "shadow_hit_count": len(shadow_hits),
        "overlap_count_at_5": len(set(product_ids[:5]) & set(shadow_ids[:5])),
        "overlap_count_at_10": len(set(product_ids[:10]) & set(shadow_ids[:10])),
        "changed_count_at_5": _positional_changes(product_ids, shadow_ids, 5),
        "changed_count_at_10": _positional_changes(product_ids, shadow_ids, 10),
        "dense_novelty_count": sum(
            1
            for hit in shadow_hits
            if str(hit.get("_fusion_slot") or "")
            in {"dense_novelty_tail", "dense_fill"}
        ),
        "source_escape_count": source_escapes,
        "protected_prefix_preserved": (
            len(shadow_ids) >= prefix_depth
            and shadow_ids[:prefix_depth] == product_ids[:prefix_depth]
        ),
    }


def _default_runtime_loader(db_dir: Path, index_dir: Path) -> DenseShadowRuntime:
    from .transformers_embedding_provider import (
        TransformersPoolingEmbeddingProvider,
    )

    provider = TransformersPoolingEmbeddingProvider(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        dimension=1024,
        query_prefix=(
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            "Query:"
        ),
        passage_prefix="",
        max_length=8192,
        batch_size=1,
        device="cuda",
        dtype="float16",
        allow_download=False,
        pooling="last_token",
        padding_side="left",
    )
    if provider.descriptor.identity != QWEN_PROVIDER_IDENTITY:
        raise RuntimeError("dense shadow provider identity mismatch")
    loaded = EmbeddingIndexStore(db_dir, index_dir=index_dir).load(provider)
    return DenseShadowRuntime(
        retriever=loaded.as_retriever(provider),
        index_cache_hit=False,
        cold_start=True,
    )


RuntimeLoader = Callable[[Path, Path], DenseShadowRuntime]


class RetrievalShadowRunner:
    """Run one content-free, fail-open Dense comparison outside the answer path."""

    def __init__(self, runtime_loader: RuntimeLoader = _default_runtime_loader) -> None:
        self._runtime_loader = runtime_loader
        self._state_lock = threading.Lock()
        self._in_flight = False
        self._runtime_key: tuple[str, str, tuple[int, int], tuple[int, int]] | None = (
            None
        )
        self._runtime: DenseShadowRuntime | None = None

    @staticmethod
    def _signature(path: Path) -> tuple[int, int]:
        try:
            stat = path.stat()
            return int(stat.st_mtime_ns), int(stat.st_size)
        except OSError:
            return 0, 0

    def _get_runtime(self, db_dir: Path, index_dir: Path) -> DenseShadowRuntime:
        key = (
            str(db_dir.resolve()).casefold(),
            str(index_dir.resolve()).casefold(),
            self._signature(db_dir / "docs.json"),
            self._signature(index_dir / "current.json"),
        )
        with self._state_lock:
            if self._runtime is not None and self._runtime_key == key:
                return DenseShadowRuntime(
                    retriever=self._runtime.retriever,
                    index_cache_hit=True,
                    cold_start=False,
                )
        loaded = self._runtime_loader(db_dir, index_dir)
        with self._state_lock:
            self._runtime_key = key
            self._runtime = loaded
        return loaded

    def _run(
        self,
        *,
        query: str,
        product_hits: list[dict],
        db_dir: Path,
        index_dir: Path,
        channel: str,
        source_scope: str,
        scoped: bool,
    ) -> dict:
        started_at = time.perf_counter()
        original_copy = copy.deepcopy(product_hits)
        runtime: DenseShadowRuntime | None = None
        try:
            runtime = self._get_runtime(db_dir, index_dir)
            dense: CandidateRetriever = runtime.retriever
            if scoped:
                dense = _SourceFilteredRetriever(
                    dense,
                    {_source_identity(hit) for hit in product_hits},
                )
            candidate = ProtectedDenseTailRetriever(
                _StaticCandidateRetriever(product_hits),
                dense,
                dense_novelty_quota=2,
                candidate_multiplier=4,
                protected_sparse_prefix=5,
                sparse_tail_window=5,
            )
            shadow_hits = candidate.search(query, top_k=SHADOW_OUTPUT_DEPTH)
            comparison = compare_shadow_rankings(
                product_hits=original_copy,
                shadow_hits=shadow_hits,
                scoped=scoped,
            )
            product_unchanged = product_hits == original_copy
            record_retrieval_shadow_observation(
                started_at=started_at,
                channel=channel,
                source_scope=source_scope,
                status="done",
                product_output_unchanged=product_unchanged,
                cold_start=runtime.cold_start,
                index_cache_hit=runtime.index_cache_hit,
                **comparison,
            )
            return {
                "status": "done",
                "duration_ms": (time.perf_counter() - started_at) * 1000.0,
                "product_output_unchanged": product_unchanged,
                "cold_start": runtime.cold_start,
                "index_cache_hit": runtime.index_cache_hit,
                **comparison,
            }
        except Exception as exc:
            reason = (
                f"dense_index_{exc.code}"
                if isinstance(exc, EmbeddingIndexError)
                else "dense_runtime_unavailable"
            )
            product_unchanged = product_hits == original_copy
            record_retrieval_shadow_observation(
                started_at=started_at,
                channel=channel,
                source_scope=source_scope,
                status="error",
                product_hit_count=len(product_hits),
                product_output_unchanged=product_unchanged,
                cold_start=bool(runtime.cold_start) if runtime else True,
                index_cache_hit=bool(runtime.index_cache_hit) if runtime else False,
                fallback_reason=reason,
            )
            return {
                "status": "error",
                "fallback_reason_id": reason,
                "duration_ms": (time.perf_counter() - started_at) * 1000.0,
                "product_output_unchanged": product_unchanged,
            }

    def schedule(
        self,
        *,
        query: str,
        product_hits: list[dict],
        db_dir: Path,
        index_dir: Path,
        channel: str,
        source_scope: str,
        scoped: bool,
    ) -> ShadowSubmission:
        with self._state_lock:
            if self._in_flight:
                record_retrieval_shadow_observation(
                    started_at=time.perf_counter(),
                    channel=channel,
                    source_scope=source_scope,
                    status="cancelled",
                    product_hit_count=len(product_hits),
                    fallback_reason="shadow_busy",
                )
                return ShadowSubmission("busy")
            self._in_flight = True

        future: Future[dict] = Future()
        isolated_hits = copy.deepcopy(list(product_hits or []))

        def target() -> None:
            try:
                future.set_result(
                    self._run(
                        query=str(query or ""),
                        product_hits=isolated_hits,
                        db_dir=db_dir,
                        index_dir=index_dir,
                        channel=channel,
                        source_scope=source_scope,
                        scoped=scoped,
                    )
                )
            except BaseException as exc:  # pragma: no cover - defensive Future boundary
                future.set_exception(exc)
            finally:
                with self._state_lock:
                    self._in_flight = False

        threading.Thread(
            target=target,
            name="pi-zaya-dense-shadow",
            daemon=True,
        ).start()
        return ShadowSubmission("submitted", future)


_DEFAULT_RUNNER = RetrievalShadowRunner()


def schedule_product_retrieval_shadow(
    *,
    query: str,
    product_hits: list[dict],
    db_dir: Path,
    source_scope: str,
) -> ShadowSubmission:
    if str(os.environ.get("KB_DENSE_SHADOW_ENABLED") or "").strip() != "1":
        return ShadowSubmission("disabled")
    if not str(os.environ.get("KB_LOCAL_OBSERVABILITY_PATH") or "").strip():
        return ShadowSubmission("observability_disabled")
    raw_index = str(os.environ.get("KB_DENSE_SHADOW_INDEX_DIR") or "").strip()
    if not raw_index:
        return ShadowSubmission("index_not_configured")
    index_dir = Path(raw_index).expanduser().resolve(strict=False)
    if not index_dir.is_dir():
        record_retrieval_shadow_observation(
            started_at=time.perf_counter(),
            channel="product_chat_shadow",
            source_scope=source_scope,
            status="error",
            product_hit_count=len(product_hits),
            fallback_reason="dense_index_missing",
        )
        return ShadowSubmission("index_missing")
    clean_scope = str(source_scope or "").strip().lower()
    try:
        return _DEFAULT_RUNNER.schedule(
            query=str(query or ""),
            product_hits=list(product_hits or []),
            db_dir=Path(db_dir).expanduser().resolve(strict=False),
            index_dir=index_dir,
            channel="product_chat_shadow",
            source_scope=clean_scope,
            scoped=clean_scope not in {"library", "full_library"},
        )
    except Exception:
        record_retrieval_shadow_observation(
            started_at=time.perf_counter(),
            channel="product_chat_shadow",
            source_scope=clean_scope,
            status="error",
            product_hit_count=len(product_hits),
            fallback_reason="shadow_schedule_failed",
        )
        return ShadowSubmission("schedule_failed")
