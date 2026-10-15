from __future__ import annotations

import copy
import json

from tools.retrieval_eval.run_scoped_fresh_e5 import (
    CANDIDATE_CHANNEL,
    PRODUCT_CHANNEL,
    _decision,
    evaluate_quality_gate,
    evaluate_runtime_gate,
    result_fingerprint,
)
from tools.retrieval_eval.validate_scoped_fresh_e5_protocol import DEFAULT_PROTOCOL


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def _metrics(*, recall_10: float, mrr: float = 0.5) -> dict[str, float]:
    return {
        "recall@5": 0.5,
        "recall@10": recall_10,
        "mrr": mrr,
        "ndcg@10": 0.5,
        "contract_recall@10": recall_10,
    }


def _channels() -> dict:
    product_cases = [
        {"id": "a", "metrics": _metrics(recall_10=0.5)},
        {"id": "b", "metrics": _metrics(recall_10=0.5)},
    ]
    candidate_cases = [
        {"id": "a", "metrics": _metrics(recall_10=1.0)},
        {"id": "b", "metrics": _metrics(recall_10=0.5)},
    ]
    return {
        PRODUCT_CHANNEL: {
            "metrics": _metrics(recall_10=0.5),
            "cases": product_cases,
        },
        CANDIDATE_CHANNEL: {
            "metrics": _metrics(recall_10=0.75),
            "cases": candidate_cases,
        },
    }


def _worker_and_integrity() -> tuple[dict, dict]:
    protocol = _protocol()
    existing = protocol["existing_corpus_and_index"]
    candidate = protocol["candidate"]
    trace = [
        {
            "mode": "scoped",
            "duration_ms": 20.0,
            "fallback_reason_id": "",
            "source_escape_count": 0,
            "protected_prefix_preserved": True,
            "product_sparse_input_unchanged": True,
            "cold_start": index == 0,
            "index_cache_hit": index > 0,
            "dense_novelty_count": 1,
        }
        for index in range(12)
    ]
    worker = {
        "runtime": {
            "subprocess_wall_ms": 1000.0,
            "peak_process_rss_mib": 1000.0,
            "python_version": "3.10.11",
        },
        "index": {
            "build_id": existing["build_id"],
            "document_count": existing["document_count"],
            "parent_count": existing["parent_count"],
            "window_count": existing["window_count"],
        },
        "selector_trace": trace,
        "execution": {
            "index_build_call_count": 0,
            "full_library_call_count": 0,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
            "network_used": False,
        },
    }
    integrity = {
        "working_db_before_sha256": existing["working_db_retrieval_snapshot_sha256"],
        "working_db_after_sha256": existing["working_db_retrieval_snapshot_sha256"],
        "index_before_sha256": existing["index_tree_sha256"],
        "index_after_sha256": existing["index_tree_sha256"],
        "artifact_before_sha256": candidate["artifact_tree_sha256"],
        "artifact_after_sha256": candidate["artifact_tree_sha256"],
        "model_after_sha256": candidate["model_sha256"],
        "tokenizer_after_sha256": candidate["tokenizer_sha256"],
    }
    return worker, integrity


def test_scoped_fresh_quality_gate_passes_gain_without_harm() -> None:
    gate = evaluate_quality_gate(_channels(), _protocol())
    assert gate["passes_all"] is True
    assert gate["recall_at_10_gain"] == 0.25


def test_scoped_fresh_quality_gate_rejects_case_harm_hidden_by_aggregate() -> None:
    channels = _channels()
    channels[CANDIDATE_CHANNEL]["cases"][1]["metrics"]["mrr"] = 0.0
    gate = evaluate_quality_gate(channels, _protocol())
    assert gate["passes_all"] is False
    assert gate["case_regressions"]["mrr"] == ["b"]


def test_scoped_fresh_runtime_gate_requires_read_only_scoped_reuse() -> None:
    worker, integrity = _worker_and_integrity()
    gate = evaluate_runtime_gate(worker, _protocol(), integrity)
    assert gate["passes_all"] is True
    changed = copy.deepcopy(worker)
    changed["execution"]["index_build_call_count"] = 1
    assert evaluate_runtime_gate(changed, _protocol(), integrity)["passes_all"] is False


def test_scoped_fresh_decision_never_enables_product() -> None:
    decision = _decision(
        _protocol(),
        {"passes_all": True},
        {"passes_all": True},
    )
    assert decision["status"] == "fresh_scoped_suite_passed_authorize_packaging_experiment"
    assert decision["product_enabled"] is False
    assert decision["user_mode_control_visible"] is False


def test_scoped_fresh_result_fingerprint_ignores_timing() -> None:
    report = {
        "index": {},
        "selector_trace": [{"case_id": "a", "duration_ms": 1.0}],
        "channels": {
            PRODUCT_CHANNEL: {
                "channel": PRODUCT_CHANNEL,
                "mode": "scoped",
                "case_count": 1,
                "metrics": {},
                "latency_ms": {"p95": 1.0},
                "cases": [
                    {
                        "id": "a",
                        "first_relevant_rank": 1,
                        "ranked_chunk_ids": ["x"],
                        "metrics": {},
                        "latency_ms": 1.0,
                    }
                ],
            }
        },
        "runtime_gate": {"checks": {"ok": True}},
        "quality_gate": {},
        "decision": {},
    }
    changed = copy.deepcopy(report)
    changed["selector_trace"][0]["duration_ms"] = 99.0
    changed["channels"][PRODUCT_CHANNEL]["latency_ms"]["p95"] = 99.0
    changed["channels"][PRODUCT_CHANNEL]["cases"][0]["latency_ms"] = 99.0
    assert result_fingerprint(report) == result_fingerprint(changed)
