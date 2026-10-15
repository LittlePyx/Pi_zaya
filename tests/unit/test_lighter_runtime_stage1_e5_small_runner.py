from __future__ import annotations

import copy
import json

from tools.retrieval_eval.run_lighter_runtime_stage1_e5_small import (
    _evaluate_gate,
    _result_fingerprint,
)
from tools.retrieval_eval.validate_lighter_runtime_stage1_e5_small_protocol import (
    DEFAULT_PROTOCOL,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def _worker() -> dict:
    vector_summary_12 = {
        "count": 12,
        "dimension": 384,
        "minimum_norm": 1.0,
        "maximum_norm": 1.0,
        "matrix_sha256": "1" * 64,
    }
    vector_summary_74 = {**vector_summary_12, "count": 74, "matrix_sha256": "2" * 64}
    return {
        "runtime": {
            "provider_load_ms": 500.0,
            "peak_process_rss_mib": 1000.0,
            "session_providers": ["CPUExecutionProvider"],
        },
        "query": {
            "timings": [{"duration_ms": 10.0} for _ in range(12)],
            "truncated_count": 0,
            "vectors": vector_summary_12,
        },
        "documents": {
            "parent_count": 1673,
            "full_window_count": 2016,
            "stress_parent_count": 64,
            "stress_window_count": 74,
            "truncated_parent_count": 0,
            "uncovered_content_token_count": 0,
            "maximum_input_tokens": 453,
            "projected_full_build_ms": 100000.0,
            "projected_parent_chunks_per_second": 16.73,
            "vectors": vector_summary_74,
        },
        "execution": {
            "full_corpus_windows_embedded": 0,
            "quality_evaluations": 0,
            "product_enabled": False,
        },
    }


def test_e5_stage1_gate_accepts_bounded_complete_worker() -> None:
    gate = _evaluate_gate(
        protocol=_protocol(),
        worker=_worker(),
        subprocess_wall_ms=120000.0,
        source_unchanged=True,
        artifact_unchanged=True,
        tokenizer_unchanged=True,
    )
    assert gate["passes_all"] is True


def test_e5_stage1_gate_rejects_incomplete_or_slow_work() -> None:
    worker = copy.deepcopy(_worker())
    worker["documents"]["uncovered_content_token_count"] = 1
    worker["documents"]["projected_full_build_ms"] = 600001.0
    gate = _evaluate_gate(
        protocol=_protocol(),
        worker=worker,
        subprocess_wall_ms=120000.0,
        source_unchanged=True,
        artifact_unchanged=True,
        tokenizer_unchanged=True,
    )
    assert gate["passes_all"] is False
    assert gate["checks"]["content_coverage"] is False
    assert gate["checks"]["projected_full_build"] is False


def test_e5_stage1_fingerprint_ignores_latency_but_pins_vectors_and_decision() -> None:
    report = {
        "query": {"vectors": _worker()["query"]["vectors"]},
        "document_projection": {
            "vectors": _worker()["documents"]["vectors"],
            "stress_work": {"attention_work_proxy": 1},
            "full_corpus_work": {"attention_work_proxy": 2},
        },
        "gate": {"passes_all": True},
        "decision": {"status": "passed"},
        "runtime": {"provider_load_ms": 1.0},
    }
    changed = copy.deepcopy(report)
    changed["runtime"]["provider_load_ms"] = 999.0
    assert _result_fingerprint(report) == _result_fingerprint(changed)
    changed["decision"]["status"] = "rejected"
    assert _result_fingerprint(report) != _result_fingerprint(changed)
