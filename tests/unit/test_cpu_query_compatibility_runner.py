from __future__ import annotations

import json

from tools.retrieval_eval.run_cpu_query_compatibility import (
    DEFAULT_REPORT,
    EXPECTED_REPORT_SHA256,
    EXPECTED_RESULT_FINGERPRINT,
    _load_contracts,
    _persist_then_validate,
    _stable_channel_projection,
    evaluate_gate,
    validate_report,
)
from kb.retrieval_eval_fixture import sha256_file


def _channel(metrics: dict) -> dict:
    return {
        "modes": {
            "scoped": {"metrics": dict(metrics)},
            "full_library": {"metrics": dict(metrics)},
        }
    }


def _gate() -> dict:
    return {
        "require_no_regression_vs_parent_fusion_metrics": [
            "recall@5",
            "recall@10",
            "mrr",
            "ndcg@10",
        ],
        "maximum_provider_load_ms": 30000.0,
        "maximum_warm_query_embedding_p95_ms": 1000.0,
        "required_query_completion_rate": 1.0,
    }


def test_cpu_gate_passes_only_complete_fast_non_regressing_result() -> None:
    metrics = {
        "recall@5": 0.4,
        "recall@10": 0.7,
        "mrr": 0.5,
        "ndcg@10": 0.6,
        "wrong_source_rate@5": 0.0,
    }
    result = evaluate_gate(
        product=_channel(metrics),
        parent_fusion=_channel(metrics),
        cpu_fusion=_channel(metrics),
        semantic_acceptance={"passes_all": True},
        provider_load_ms=2000.0,
        query_durations_ms=[100.0] * 12,
        expected_query_count=12,
        gate=_gate(),
        index_unchanged=True,
    )
    assert result["passes_all"] is True


def test_cpu_gate_rejects_latency_and_quality_regression() -> None:
    parent = {
        "recall@5": 0.4,
        "recall@10": 0.7,
        "mrr": 0.5,
        "ndcg@10": 0.6,
        "wrong_source_rate@5": 0.0,
    }
    cpu = {**parent, "recall@10": 0.6}
    result = evaluate_gate(
        product=_channel(parent),
        parent_fusion=_channel(parent),
        cpu_fusion=_channel(cpu),
        semantic_acceptance={"passes_all": True},
        provider_load_ms=2000.0,
        query_durations_ms=[100.0] * 11 + [1100.0],
        expected_query_count=12,
        gate=_gate(),
        index_unchanged=True,
    )
    assert result["passes_all"] is False
    assert result["mode_checks"]["scoped"]["no_regression_recall@10"] is False
    assert result["operational_checks"]["warm_query_embedding_p95_ms"] is False


def test_product_control_projection_ignores_latency_but_not_ranking() -> None:
    left = _channel({"recall@5": 0.4})
    right = _channel({"recall@5": 0.4})
    for channel, latency in ((left, 10.0), (right, 999.0)):
        for mode in channel["modes"].values():
            mode.update(
                {
                    "latency_ms": {"p95": latency},
                    "cases": [{"id": "case-1", "first_relevant_rank": 2}],
                }
            )
    assert _stable_channel_projection(left) == _stable_channel_projection(right)
    right["modes"]["scoped"]["cases"][0]["first_relevant_rank"] = 3
    assert _stable_channel_projection(left) != _stable_channel_projection(right)


def test_raw_report_is_persisted_before_validation(tmp_path) -> None:
    path = tmp_path / "raw-report.json"
    report = {"schema_version": 1, "value": "retained"}

    def validator(value: dict) -> list[str]:
        assert path.is_file()
        assert json.loads(path.read_text(encoding="utf-8")) == report
        assert value == report
        return ["synthetic validation failure"]

    assert _persist_then_validate(
        report,
        report_path=path,
        validator=validator,
    ) == ["synthetic validation failure"]
    assert path.is_file()


def test_cpu_query_outcome_is_valid_and_pinned() -> None:
    protocol, fixture, parent = _load_contracts()
    report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    assert validate_report(
        report,
        protocol=protocol,
        fixture=fixture,
        parent=parent,
    ) == []
    assert report["result_fingerprint"] == EXPECTED_RESULT_FINGERPRINT
    assert sha256_file(DEFAULT_REPORT) == EXPECTED_REPORT_SHA256
    assert report["decision"] == {
        "status": "passed_for_cpu_full_index_feasibility",
        "cpu_full_index_study_allowed": True,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }
