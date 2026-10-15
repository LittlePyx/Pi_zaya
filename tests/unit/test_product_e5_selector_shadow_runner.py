from __future__ import annotations

import copy
import json

from tools.retrieval_eval.run_product_e5_selector_shadow import (
    CANDIDATE_CHANNEL,
    DEFAULT_PROTOCOL,
    DEFAULT_REPORT,
    PRODUCT_CHANNEL,
    _decision,
    evaluate_quality_gate,
    evaluate_runtime_gate,
    summarize_selector_trace,
    validate_report,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def _trace() -> list[dict]:
    return [
        {
            "ordinal": index + 1,
            "mode": "scoped" if index < 12 else "full_library",
            "duration_ms": 10.0 + index,
            "requested_mode": "hybrid_local",
            "used_mode": "hybrid_local",
            "fallback_reason_id": "",
            "protected_prefix_preserved": True,
            "dense_novelty_count": 1,
            "source_escape_count": 0,
            "product_sparse_input_unchanged": True,
            "cold_start": index == 0,
            "index_cache_hit": index > 0,
        }
        for index in range(24)
    ]


def _channels() -> dict:
    product_metrics = {
        "recall@5": 0.3,
        "recall@10": 0.3,
        "wrong_source_rate@5": 0.2,
        "mrr": 0.2,
        "ndcg@10": 0.2,
    }
    candidate_metrics = {
        **product_metrics,
        "recall@10": 0.5,
        "mrr": 0.3,
        "ndcg@10": 0.3,
    }
    return {
        PRODUCT_CHANNEL: {
            "modes": {
                mode: {"metrics": dict(product_metrics)}
                for mode in ("scoped", "full_library")
            }
        },
        CANDIDATE_CHANNEL: {
            "modes": {
                mode: {"metrics": dict(candidate_metrics)}
                for mode in ("scoped", "full_library")
            }
        },
    }


def _worker() -> dict:
    return {
        "runtime": {"full_build_ms": 100000.0, "peak_process_rss_mib": 1200.0},
        "index": {
            "parent_count": 1673,
            "window_count": 2016,
            "index_bytes": 3300000,
            "no_op": False,
            "atomic_readback_passed": True,
        },
        "selector_trace": _trace(),
        "channels": _channels(),
        "execution": {
            "full_build_count": 1,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
            "network_used": False,
        },
    }


def _integrity(protocol: dict) -> dict:
    source_sha = protocol["source_corpus"]["source_tree_sha256"]
    model_sha = protocol["candidate"]["model_sha256"]
    tokenizer_sha = protocol["candidate"]["tokenizer_sha256"]
    return {
        "source_before_sha256": source_sha,
        "source_after_sha256": source_sha,
        "working_db_before_sha256": source_sha,
        "working_db_after_sha256": source_sha,
        "artifact_tree_before_sha256": "artifact-tree",
        "artifact_tree_after_sha256": "artifact-tree",
        "model_before_sha256": model_sha,
        "model_after_sha256": model_sha,
        "tokenizer_before_sha256": tokenizer_sha,
        "tokenizer_after_sha256": tokenizer_sha,
    }


def test_product_e5_selector_shadow_runtime_gate_passes_complete_trace() -> None:
    protocol = _protocol()
    gate = evaluate_runtime_gate(_worker(), protocol, _integrity(protocol))

    assert gate["passes_all"] is True
    assert gate["summary"]["cold_load_count"] == 1
    assert gate["summary"]["warm_cache_hit_count"] == 23


def test_product_e5_selector_shadow_runtime_gate_rejects_fallback() -> None:
    protocol = _protocol()
    worker = _worker()
    worker["selector_trace"][5]["fallback_reason_id"] = "dense_query_failed"

    gate = evaluate_runtime_gate(worker, protocol, _integrity(protocol))

    assert gate["checks"]["fallback_count"] is False
    assert gate["passes_all"] is False


def test_product_e5_selector_shadow_quality_gate_rejects_scoped_regression() -> None:
    protocol = _protocol()
    channels = _channels()
    channels[CANDIDATE_CHANNEL]["modes"]["scoped"]["metrics"]["mrr"] = 0.1

    gate = evaluate_quality_gate(channels, protocol)

    assert gate["modes"]["scoped"]["checks"]["mrr_no_regression"] is False
    assert gate["passes_all"] is False


def test_product_e5_selector_shadow_decision_never_enables_product() -> None:
    protocol = _protocol()
    worker = _worker()
    runtime = evaluate_runtime_gate(worker, protocol, _integrity(protocol))
    quality = evaluate_quality_gate(worker["channels"], protocol)

    decision = _decision(protocol, runtime, quality)
    failed = _decision(protocol, {**runtime, "passes_all": False}, quality)

    assert decision["status"] == "product_selector_shadow_passed_expand_judged_cases"
    assert decision["product_enabled"] is False
    assert failed["status"] == "product_selector_runtime_rejected_keep_bm25"


def test_selector_summary_does_not_treat_cold_load_as_warm_cache_hit() -> None:
    rows = copy.deepcopy(_trace())
    rows[0]["index_cache_hit"] = True

    summary = summarize_selector_trace(rows)

    assert summary["cold_load_count"] == 1
    assert summary["warm_cache_hit_count"] == 23


def test_product_e5_selector_shadow_report_is_pinned_and_valid() -> None:
    report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))

    assert validate_report(report, protocol=_protocol()) == []


def test_product_e5_selector_shadow_report_rejects_result_tamper() -> None:
    report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    report["selector_trace"][0]["fallback_reason_id"] = "dense_query_failed"

    errors = validate_report(report, protocol=_protocol())

    assert "product E5 selector runtime gate is inconsistent" in errors
    assert "product E5 selector result fingerprint is inconsistent" in errors
