from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from kb.local_dense_runtime import normalize_retrieval_mode
from kb.local_retrieval_selector import preserve_product_bm25, select_product_retrieval
from kb.retrieval_observability import record_retrieval_observation
from kb.retrieval_shadow import schedule_product_retrieval_shadow


TaskPublisher = Callable[..., None]
_LOCAL_SEMANTIC_SCOPES = {
    "basket",
    "current_paper",
    "research_basket",
    "scoped",
    "selected_sources",
}


def _local_semantic_scope_is_eligible(scope: object) -> bool:
    normalized = str(scope or "").strip().lower().replace("-", "_")
    return normalized in _LOCAL_SEMANTIC_SCOPES


@dataclass
class TaskRetrievalObservation:
    started_at: float | None = None
    scope: str = "unknown_scope"
    channel: str = "bm25_product"
    recorded: bool = False
    local_trace: dict[str, object] = field(default_factory=dict)

    def begin(self, started_at: float, scope: object, channel: str = "bm25_product") -> None:
        self.started_at = float(started_at)
        self.scope = str(scope or "unknown_scope")
        self.channel = str(channel or "bm25_product")

    def select_and_publish(
        self,
        *,
        query: str,
        sparse_hits: Sequence[dict],
        db_dir: Path,
        retrieval_mode: str,
        publish: TaskPublisher,
        source_paths: Sequence[str] | None = None,
        filter_selected_hits: Callable[[list[dict]], list[dict]] | None = None,
    ) -> tuple[list[dict], list[float]]:
        requested_mode = normalize_retrieval_mode(retrieval_mode)
        if (
            self.channel == "bm25_product"
            and requested_mode == "hybrid_local"
            and not _local_semantic_scope_is_eligible(self.scope)
        ):
            selection = preserve_product_bm25(
                sparse_hits,
                retrieval_mode=requested_mode,
                reason_id="local_semantic_scope_not_eligible",
            )
        elif self.channel == "bm25_product":
            selection = select_product_retrieval(
                query=query,
                sparse_hits=list(sparse_hits),
                db_dir=db_dir,
                retrieval_mode=requested_mode,
                source_scope=self.scope,
                source_paths=source_paths,
            )
        else:
            reason = f"{self.channel}_preserved"
            selection = preserve_product_bm25(
                sparse_hits,
                retrieval_mode=requested_mode,
                reason_id=reason if requested_mode == "hybrid_local" else "",
            )
        if selection.used_mode == "hybrid_local":
            self.channel = "hybrid_local_product"
        self.local_trace = selection.trace_payload()
        hits = list(selection.hits)
        if filter_selected_hits is not None:
            hits = filter_selected_hits(hits)
            self.local_trace["dense_novelty_count"] = sum(
                hit.get("_fusion_slot") in {"dense_novelty_tail", "dense_fill"} for hit in hits
            )
        publish(local_retrieval=self.local_trace)
        return hits, [float(hit.get("score", 0.0) or 0.0) for hit in hits]

    def record(
        self,
        *,
        hits: Sequence[dict],
        query_variants: Sequence[object],
        translation_used: bool = False,
        query: str = "",
        bm25_hits_for_shadow: Sequence[dict] | None = None,
        db_dir: Path | None = None,
    ) -> None:
        fallback_reason = str(self.local_trace.get("fallback_reason_id") or "") or None
        if fallback_reason is None and self.channel == "structured_section":
            fallback_reason = "structured_section_match"
        elif fallback_reason is None and translation_used:
            fallback_reason = "query_translation"
        elif fallback_reason is None and len(query_variants) > 1:
            fallback_reason = "query_expansion"
        record_retrieval_observation(
            started_at=self.started_at,
            channel=self.channel,
            source_scope=self.scope,
            hit_count=len(hits),
            query_variant_count=len(query_variants),
            translation_used=translation_used,
            fallback_reason=fallback_reason,
            requested_mode=str(self.local_trace.get("requested_mode") or ""),
            used_mode=str(self.local_trace.get("used_mode") or ""),
            automatic_fallback=bool(
                self.local_trace.get("requested_mode") == "hybrid_local"
                and self.local_trace.get("used_mode") == "bm25"
                and fallback_reason
            ),
            cold_start=bool(self.local_trace.get("cold_start")),
            index_cache_hit=bool(self.local_trace.get("index_cache_hit")),
        )
        self.recorded = True
        if bm25_hits_for_shadow is not None and db_dir is not None:
            schedule_product_retrieval_shadow(
                query=query,
                product_hits=list(bm25_hits_for_shadow),
                db_dir=db_dir,
                source_scope=self.scope,
            )

    def record_failure(self, *, error: Exception, task_snapshot: dict, task_id: str) -> None:
        if self.started_at is None or self.recorded:
            return
        cancelled = str(error) == "canceled" or (
            str(task_snapshot.get("id") or "") == str(task_id or "")
            and bool(task_snapshot.get("cancel") or False)
        )
        record_retrieval_observation(
            started_at=self.started_at,
            channel=self.channel,
            source_scope=self.scope,
            fallback_reason="task_cancelled" if cancelled else "retrieval_error",
            status="cancelled" if cancelled else "error",
        )
        self.recorded = True
