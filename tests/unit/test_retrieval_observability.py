from __future__ import annotations

import json
import time

import pytest

from kb.agent import tools as agent_tools
from kb.retrieval_observability import record_retrieval_observation


def test_retrieval_observation_contains_only_redacted_operation_data(monkeypatch, tmp_path) -> None:
    path = tmp_path / "retrieval.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))

    assert record_retrieval_observation(
        started_at=time.perf_counter(),
        channel="bm25_product",
        source_scope="basket",
        hit_count=4,
        query_variant_count=2,
        translation_used=True,
        fallback_reason="query_translation",
    )

    attributes = json.loads(path.read_text(encoding="utf-8"))["attributes"]
    assert attributes["outcome_id"] == "retrieval.ready"
    assert attributes["retrieval_channel_id"] == "bm25_product"
    assert attributes["source_scope_id"] == "research_basket"
    assert attributes["retrieval_hit_count"] == 4
    assert attributes["query_variant_count"] == 2
    assert attributes["translation_used"] is True
    assert attributes["fallback_reason_id"] == "query_translation"


def test_hybrid_product_channel_is_recorded_without_content(monkeypatch, tmp_path) -> None:
    path = tmp_path / "hybrid.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))

    assert record_retrieval_observation(
        started_at=time.perf_counter(),
        channel="hybrid_local_product",
        source_scope="library",
        hit_count=10,
        requested_mode="hybrid_local",
        used_mode="hybrid_local",
        automatic_fallback=False,
        cold_start=False,
        index_cache_hit=True,
    )

    stored = path.read_text(encoding="utf-8")
    attributes = json.loads(stored)["attributes"]
    assert attributes["retrieval_channel_id"] == "hybrid_local_product"
    assert attributes["requested_retrieval_mode_id"] == "hybrid_local"
    assert attributes["used_retrieval_mode_id"] == "hybrid_local"
    assert attributes["automatic_fallback"] is False
    assert attributes["cold_start"] is False
    assert attributes["index_cache_hit"] is True
    assert "query" not in attributes
    assert "prompt" not in attributes
    assert "evidence" not in attributes


def test_hybrid_fallback_records_requested_and_actual_modes(monkeypatch, tmp_path) -> None:
    path = tmp_path / "hybrid-fallback.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))

    assert record_retrieval_observation(
        started_at=time.perf_counter(),
        channel="bm25_product",
        source_scope="current_paper",
        hit_count=5,
        requested_mode="hybrid-local",
        used_mode="bm25",
        automatic_fallback=True,
        fallback_reason="local_semantic_component_not_ready",
    )

    attributes = json.loads(path.read_text(encoding="utf-8"))["attributes"]
    assert attributes["requested_retrieval_mode_id"] == "hybrid_local"
    assert attributes["used_retrieval_mode_id"] == "bm25"
    assert attributes["automatic_fallback"] is True


def test_agent_retrieval_wrapper_does_not_persist_query_hits_or_paths(monkeypatch, tmp_path) -> None:
    path = tmp_path / "agent-retrieval.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))
    monkeypatch.setattr(
        agent_tools,
        "_retrieve_evidence_impl",
        lambda *args, **kwargs: {
            "hits": [
                {
                    "text": "private evidence text",
                    "meta": {"source_path": "C:/private/paper.pdf"},
                }
            ],
            "used_translation": True,
            "query_variants": ["private question", "private translated question"],
        },
    )

    result = agent_tools.retrieve_evidence(
        "private question",
        db_dir=tmp_path,
        source_paths=["C:/private/paper.pdf"],
    )

    assert len(result["hits"]) == 1
    stored_text = path.read_text(encoding="utf-8")
    attributes = json.loads(stored_text)["attributes"]
    assert attributes["retrieval_channel_id"] == "agent_bm25"
    assert attributes["source_scope_id"] == "selected_sources"
    assert attributes["retrieval_hit_count"] == 1
    assert "private question" not in stored_text
    assert "private evidence text" not in stored_text
    assert "paper.pdf" not in stored_text


def test_agent_retrieval_error_is_redacted_and_propagated(monkeypatch, tmp_path) -> None:
    path = tmp_path / "agent-error.jsonl"
    monkeypatch.setenv("KB_LOCAL_OBSERVABILITY_PATH", str(path))

    def fail(*_args, **_kwargs):
        raise RuntimeError("private query and C:/private/paper.pdf")

    monkeypatch.setattr(agent_tools, "_retrieve_evidence_impl", fail)

    with pytest.raises(RuntimeError, match="private query"):
        agent_tools.retrieve_evidence("private query", db_dir=tmp_path)

    stored_text = path.read_text(encoding="utf-8")
    attributes = json.loads(stored_text)["attributes"]
    assert attributes["status"] == "error"
    assert attributes["fallback_reason_id"] == "retrieval_error"
    assert "private query" not in stored_text
    assert "paper.pdf" not in stored_text
