from __future__ import annotations

from pathlib import Path

import kb.task_retrieval_observation as observation_module
from kb.local_retrieval_selector import ProductRetrievalSelection
from kb.task_retrieval_observation import (
    TaskRetrievalObservation,
    _local_semantic_scope_is_eligible,
)


def _hits() -> list[dict]:
    return [
        {
            "id": "doc-a:1",
            "score": 4.5,
            "text": "evidence",
            "meta": {"source_path": "doc-a.md"},
        }
    ]


def test_local_semantic_scope_requires_explicit_source_boundary() -> None:
    assert _local_semantic_scope_is_eligible("current_paper") is True
    assert _local_semantic_scope_is_eligible("basket") is True
    assert _local_semantic_scope_is_eligible("selected-sources") is True
    assert _local_semantic_scope_is_eligible("library") is False
    assert _local_semantic_scope_is_eligible("unknown_scope") is False


def test_full_library_hybrid_fails_closed_before_runtime_load(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def unexpected_selector(**_kwargs):
        raise AssertionError("full-library request reached local semantic runtime")

    monkeypatch.setattr(observation_module, "select_product_retrieval", unexpected_selector)
    published: list[dict] = []
    observation = TaskRetrievalObservation(scope="library")

    hits, scores = observation.select_and_publish(
        query="find evidence across the library",
        sparse_hits=_hits(),
        db_dir=tmp_path,
        retrieval_mode="hybrid_local",
        publish=lambda **payload: published.append(payload),
    )

    assert hits == _hits()
    assert scores == [4.5]
    assert observation.channel == "bm25_product"
    assert observation.local_trace["fallback_reason_id"] == (
        "local_semantic_scope_not_eligible"
    )
    assert published == [{"local_retrieval": observation.local_trace}]


def test_explicit_paper_scope_can_reach_hidden_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[dict] = []

    def candidate_selector(**kwargs):
        calls.append(kwargs)
        return ProductRetrievalSelection(
            hits=list(kwargs["sparse_hits"]),
            requested_mode="hybrid_local",
            used_mode="hybrid_local",
        )

    monkeypatch.setattr(observation_module, "select_product_retrieval", candidate_selector)
    observation = TaskRetrievalObservation(scope="current_paper")

    hits, _scores = observation.select_and_publish(
        query="find the mechanism",
        sparse_hits=_hits(),
        db_dir=tmp_path,
        retrieval_mode="hybrid-local",
        source_paths=["doc-a.md", "doc-b.md"],
        publish=lambda **_payload: None,
    )

    assert hits == _hits()
    assert len(calls) == 1
    assert calls[0]["source_scope"] == "current_paper"
    assert calls[0]["source_paths"] == ["doc-a.md", "doc-b.md"]
    assert observation.channel == "hybrid_local_product"


def test_bm25_full_library_still_uses_normal_selector_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[dict] = []

    def bm25_selector(**kwargs):
        calls.append(kwargs)
        return ProductRetrievalSelection(
            hits=list(kwargs["sparse_hits"]),
            requested_mode="bm25",
            used_mode="bm25",
        )

    monkeypatch.setattr(observation_module, "select_product_retrieval", bm25_selector)
    observation = TaskRetrievalObservation(scope="library")

    observation.select_and_publish(
        query="ordinary search",
        sparse_hits=_hits(),
        db_dir=tmp_path,
        retrieval_mode="bm25",
        publish=lambda **_payload: None,
    )

    assert len(calls) == 1
    assert calls[0]["retrieval_mode"] == "bm25"


def test_record_joins_requested_and_actual_modes_after_automatic_fallback(
    monkeypatch,
) -> None:
    recorded: list[dict] = []
    monkeypatch.setattr(
        observation_module,
        "record_retrieval_observation",
        lambda **payload: recorded.append(payload) or True,
    )
    observation = TaskRetrievalObservation(scope="current_paper")
    observation.local_trace = {
        "requested_mode": "hybrid_local",
        "used_mode": "bm25",
        "fallback_reason_id": "local_semantic_component_not_ready",
        "cold_start": False,
        "index_cache_hit": False,
    }

    observation.record(hits=_hits(), query_variants=["private query"])

    assert recorded[0]["requested_mode"] == "hybrid_local"
    assert recorded[0]["used_mode"] == "bm25"
    assert recorded[0]["automatic_fallback"] is True
    assert recorded[0]["cold_start"] is False
    assert recorded[0]["index_cache_hit"] is False


def test_basket_excerpt_boundary_is_applied_after_dense_fusion(monkeypatch, tmp_path):
    sparse = _hits()
    outside_excerpt = {"id": "doc-a:2", "score": 9, "text": "same paper, outside selected excerpt", "meta": {"source_path": "doc-a.md"}, "_fusion_slot": "dense_novelty_tail"}
    monkeypatch.setattr(observation_module, "select_product_retrieval", lambda **_kwargs: ProductRetrievalSelection(
        hits=sparse + [outside_excerpt], requested_mode="hybrid_local", used_mode="hybrid_local", dense_novelty_count=1,
    ))
    seen = []

    def excerpt_filter(hits):
        seen.extend(hit["id"] for hit in hits)
        return [hit for hit in hits if hit["id"] == "doc-a:1"]

    observation = TaskRetrievalObservation(scope="basket")
    hits, scores = observation.select_and_publish(
        query="selected excerpt", sparse_hits=sparse, db_dir=tmp_path,
        retrieval_mode="hybrid_local", source_paths=["doc-a.md"],
        filter_selected_hits=excerpt_filter, publish=lambda **_payload: None,
    )
    assert seen == ["doc-a:1", "doc-a:2"]
    assert hits == sparse and scores == [4.5]
    assert observation.local_trace["dense_novelty_count"] == 0
