from __future__ import annotations

import copy
import json

import pytest

from tools.retrieval_eval import run_cpu_full_index_feasibility as runner
from tools.retrieval_eval.validate_cpu_full_index_feasibility_protocol import (
    DEFAULT_PROTOCOL,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def _channel(metrics: dict) -> dict:
    return {
        "channel": "test",
        "modes": {
            "scoped": {"metrics": dict(metrics)},
            "full_library": {"metrics": dict(metrics)},
        },
    }


def _metrics() -> dict:
    return {
        "recall@5": 0.25,
        "recall@10": 0.666667,
        "mrr": 0.2,
        "ndcg@10": 0.3,
        "wrong_source_rate@5": 0.0,
    }


def _gate_inputs() -> dict:
    protocol = _protocol()
    metrics = _metrics()
    return {
        "product": _channel(metrics),
        "cpu_query_parent": _channel(metrics),
        "candidate": _channel(metrics),
        "semantic_acceptance": {"passes_all": True},
        "provider_load_ms": 20000.0,
        "build": {
            "duration_ms": 300000.0,
            "embedded_chunk_count": 1673,
            "reused_document_count": 0,
            "reused_chunk_count": 0,
            "no_op": False,
        },
        "index_status": {
            "state": "ready",
            "document_count": 10,
            "chunk_count": 1673,
        },
        "index_size_bytes": 8_000_000,
        "peak_process_rss_mib": 4096.0,
        "query_durations_ms": [400.0] * 12,
        "expected_query_count": 12,
        "runtime_info": {"truncated_document_count": 0},
        "source_corpus_unchanged": True,
        "quality_gate": protocol["quality_gate"],
        "operational_gate": protocol["operational_gate"],
    }


def test_full_index_gate_passes_only_complete_bounded_result() -> None:
    result = runner.evaluate_gate(**_gate_inputs())
    assert result["passes_all"] is True
    assert all(result["operational_checks"].values())


def test_full_index_gate_rejects_slow_or_truncated_build() -> None:
    inputs = _gate_inputs()
    inputs["build"]["duration_ms"] = 700000.0
    inputs["runtime_info"]["truncated_document_count"] = 1
    result = runner.evaluate_gate(**inputs)
    assert result["passes_all"] is False
    assert result["operational_checks"]["full_build_ms"] is False
    assert result["operational_checks"]["document_truncation"] is False


def test_attempt_receipt_is_written_before_work_and_blocks_second_run(
    tmp_path, monkeypatch
) -> None:
    protocol = _protocol()
    attempt = tmp_path / "attempt.json"
    monkeypatch.setattr(runner, "DEFAULT_ATTEMPT", attempt)
    monkeypatch.setattr(runner, "DEFAULT_REPORT", tmp_path / "report.json")
    monkeypatch.setattr(runner, "DEFAULT_INDEX", tmp_path / "index")
    monkeypatch.setattr(
        runner,
        "_tree_fingerprint",
        lambda _path: protocol["isolated_corpus"]["tree_sha256"],
    )
    receipt = runner._begin_attempt(semantic_db_root=tmp_path, protocol=protocol)
    assert json.loads(attempt.read_text(encoding="utf-8"))["status"] == "started"
    assert receipt["maximum_runs"] == 1
    with pytest.raises(ValueError, match="receipt already exists"):
        runner._begin_attempt(semantic_db_root=tmp_path, protocol=protocol)


def test_result_fingerprint_ignores_latency_but_not_rankings() -> None:
    report = {
        "channels": {
            "candidate": {
                "channel": "candidate",
                "modes": {
                    "scoped": {
                        "latency_ms": {"p95": 100.0},
                        "cases": [{"id": "case-1", "first_relevant_rank": 2}],
                    }
                },
            }
        },
        "semantic_acceptance": {"passes_all": True},
        "gate": {"passes_all": True},
        "decision": {"status": "pass"},
    }
    changed_latency = copy.deepcopy(report)
    changed_latency["channels"]["candidate"]["modes"]["scoped"]["latency_ms"][
        "p95"
    ] = 500.0
    assert runner._result_fingerprint(report) == runner._result_fingerprint(
        changed_latency
    )
    changed_rank = copy.deepcopy(report)
    changed_rank["channels"]["candidate"]["modes"]["scoped"]["cases"][0][
        "first_relevant_rank"
    ] = 3
    assert runner._result_fingerprint(report) != runner._result_fingerprint(changed_rank)


def test_peak_process_rss_is_available() -> None:
    assert runner._peak_process_rss_mib() > 0.0


def test_cpu_full_index_outcome_is_valid_rejected_and_pinned() -> None:
    protocol, fixture, parent = runner._load_contracts()
    report = json.loads(runner.DEFAULT_REPORT.read_text(encoding="utf-8"))
    assert runner.validate_report(
        report,
        protocol=protocol,
        fixture=fixture,
        cpu_query_parent=parent,
    ) == []
    assert report["result_fingerprint"] == runner.EXPECTED_RESULT_FINGERPRINT
    assert runner.sha256_file(runner.DEFAULT_REPORT) == runner.EXPECTED_REPORT_SHA256
    assert report["decision"] == {
        "status": "rejected_cpu_full_index_runtime",
        "cpu_optional_runtime_footprint_study_allowed": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }
