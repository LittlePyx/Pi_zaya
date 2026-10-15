from __future__ import annotations

import copy
import json

import pytest

from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from tools.retrieval_eval.run_user_journey_fusion_v3_1_semantic_acceptance import (
    DEFAULT_FIXTURE,
    DEFAULT_PREDECESSOR,
    DEFAULT_PROTOCOL,
    DEFAULT_SELECTION,
    FUSION_CHANNEL,
    PRODUCT_CHANNEL,
    _result_fingerprint,
    evaluate_acceptance,
    semantic_corpus_chunks,
    validate_report,
)
from tools.retrieval_eval.run_user_journey_fusion_v3_selection import GATE_METRICS
from tools.retrieval_eval.validate_user_journey_fusion_v3_1_semantic_protocol_v2 import (
    effective_protocol,
)


def _channel(recall5: float, recall10: float, wrong5: float) -> dict:
    return {
        "modes": {
            mode: {
                "metrics": {
                    "recall@5": recall5,
                    "recall@10": recall10,
                    "wrong_source_rate@5": wrong5,
                }
            }
            for mode in ("scoped", "full_library")
        }
    }


def test_semantic_acceptance_passes_exact_frozen_threshold() -> None:
    product = _channel(0.2, 0.3, 0.1)
    candidate = _channel(0.2, 0.45, 0.1)
    assert evaluate_acceptance(product, candidate)["passes_all"] is True


def test_semantic_acceptance_rejects_prefix_or_source_regression() -> None:
    product = _channel(0.2, 0.3, 0.1)
    assert evaluate_acceptance(product, _channel(0.1, 0.5, 0.1))["passes_all"] is False
    assert evaluate_acceptance(product, _channel(0.2, 0.5, 0.2))["passes_all"] is False


def test_semantic_corpus_filters_existing_index_without_rebuild() -> None:
    fixture = {
        "corpus": {
            "chunk_count": 2,
            "documents": [{"index_doc_id": "a"}, {"index_doc_id": "b"}],
        }
    }
    chunks = [{"id": "a:1"}, {"id": "extra:1"}, {"id": "b:1"}]
    assert [row["id"] for row in semantic_corpus_chunks(fixture, chunks)] == [
        "a:1",
        "b:1",
    ]


def test_semantic_corpus_rejects_missing_frozen_chunks() -> None:
    fixture = {"corpus": {"chunk_count": 2, "documents": [{"index_doc_id": "a"}]}}
    with pytest.raises(ValueError, match="expected 2"):
        semantic_corpus_chunks(fixture, [{"id": "a:1"}])


def test_semantic_synthetic_rejection_report_validation_path() -> None:
    protocol = effective_protocol(
        json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    )
    fixture = load_retrieval_benchmark_fixture(
        DEFAULT_FIXTURE, repo_root=DEFAULT_FIXTURE.parents[1]
    )
    predecessor = json.loads(DEFAULT_PREDECESSOR.read_text(encoding="utf-8"))
    case_ids = [case["id"] for case in fixture["cases"]]
    product_modes = {}
    for mode_name in ("scoped", "full_library"):
        previous = predecessor["channels"]["product_bm25"][mode_name]
        metrics = {key: 0.0 for key in GATE_METRICS}
        for key in ("recall@5", "recall@10", "wrong_source_rate@5"):
            metrics[key] = previous[key]
        product_modes[mode_name] = {
            "case_count": 12,
            "metrics": metrics,
            "latency_ms": {"p50": 0.0, "p95": 0.0, "max": 0.0},
            "ks": [5, 10],
            "cases": [
                {"id": case_id, "first_relevant_rank": None, "metrics": dict(metrics)}
                for case_id in case_ids
            ],
        }
    product = {"channel": PRODUCT_CHANNEL, "modes": product_modes}
    fusion = copy.deepcopy(product)
    fusion["channel"] = FUSION_CHANNEL
    channels = {PRODUCT_CHANNEL: product, FUSION_CHANNEL: fusion}
    acceptance = evaluate_acceptance(product, fusion)
    report = {
        "schema_version": 1,
        "report_kind": "retrieval_user_journey_fusion_v3_1_semantic_acceptance_v2",
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "fixture_sha256": sha256_file(DEFAULT_FIXTURE),
        "predecessor_sha256": sha256_file(DEFAULT_PREDECESSOR),
        "selection_sha256": sha256_file(DEFAULT_SELECTION),
        "case_ids": case_ids,
        "candidate": protocol["unchanged_candidate"],
        "index": {
            "relative_path": protocol["dedicated_index"]["relative_path"],
            "provider_identity": protocol["dedicated_index"]["provider_identity"],
            "document_count": fixture["corpus"]["document_count"],
            "chunk_count": fixture["corpus"]["chunk_count"],
            "embedded_chunk_count": fixture["corpus"]["chunk_count"],
            "no_op": False,
            "build_performed": True,
            "state": "ready",
        },
        "product_index": {
            "before_sha256": "a" * 64,
            "after_sha256": "a" * 64,
            "unchanged": True,
        },
        "query_embedding_cache": {
            "enabled": True,
            "unique_query_count": 12,
            "shared_across_scoped_and_full_library": True,
        },
        "channels": channels,
        "acceptance": acceptance,
        "result_fingerprint": _result_fingerprint(channels),
        "decision": {
            "status": "rejected_for_product_shadow",
            "product_shadow_allowed": False,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
    }
    assert (
        validate_report(
            report,
            protocol=protocol,
            fixture=fixture,
            predecessor=predecessor,
            expected_result_fingerprint=None,
            expected_index=None,
        )
        == []
    )
