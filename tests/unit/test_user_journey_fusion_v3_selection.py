from __future__ import annotations

import copy
import json

from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_metrics import mean_metrics
from tools.retrieval_eval.run_user_journey_fusion_v3_selection import (
    DEFAULT_BASELINE,
    DEFAULT_DENSE_PROTOCOL,
    DEFAULT_FIXTURE,
    DEFAULT_PROTOCOL,
    FUSION_CHANNEL,
    GATE_METRICS,
    PRODUCT_CHANNEL,
    _selection_projection,
    evaluate_combined_gate,
    evaluate_partition_gate,
    partition_fixture,
    selection_decision,
    validate_selection_report,
)
from tools.retrieval_eval.validate_user_journey_fusion_v3_1_protocol import (
    effective_protocol,
)


def _channel(metrics: dict[str, float]) -> dict:
    return {
        "modes": {
            mode: {"metrics": dict(metrics)} for mode in ("scoped", "full_library")
        }
    }


def _metrics() -> dict[str, float]:
    return {
        "mrr": 0.2,
        "recall@5": 0.2,
        "recall@10": 0.3,
        "ndcg@10": 0.2,
        "contract_recall@10": 0.3,
        "wrong_source_rate@5": 0.1,
        "wrong_source_rate@10": 0.2,
    }


def test_fusion_v3_gate_passes_only_with_safe_strict_gain() -> None:
    product = _channel(_metrics())
    candidate_metrics = _metrics()
    candidate_metrics["recall@10"] = 0.4
    candidate_metrics["mrr"] = 0.21
    candidate = _channel(candidate_metrics)
    gate = evaluate_partition_gate(product, candidate)
    assert gate["passes_all"] is True


def test_fusion_v3_gate_rejects_wrong_source_regression() -> None:
    product = _channel(_metrics())
    candidate_metrics = _metrics()
    candidate_metrics["recall@10"] = 0.4
    candidate_metrics["wrong_source_rate@10"] = 0.3
    gate = evaluate_partition_gate(product, _channel(candidate_metrics))
    assert gate["passes_all"] is False


def test_fusion_v3_combined_gate_requires_two_point_gain() -> None:
    product = _channel(_metrics())
    candidate_metrics = _metrics()
    candidate_metrics["recall@10"] = 0.31
    assert (
        evaluate_combined_gate(product, _channel(candidate_metrics))["passes_all"]
        is False
    )
    candidate_metrics["recall@10"] = 0.32
    assert (
        evaluate_combined_gate(product, _channel(candidate_metrics))["passes_all"]
        is True
    )


def test_fusion_v3_partition_fixture_never_mixes_holdout() -> None:
    fixture = {
        "split": {
            "calibration_case_ids": ["c1", "c2"],
            "holdout_case_ids": ["h1"],
        },
        "cases": [{"id": "c1"}, {"id": "h1"}, {"id": "c2"}],
    }
    selected = partition_fixture(fixture, partition="calibration")
    assert [row["id"] for row in selected["cases"]] == ["c1", "c2"]
    assert selected["ks"] == [5, 10]


def test_fusion_v3_decision_stops_before_holdout_on_calibration_failure() -> None:
    failed = {"passes_all": False}
    decision = selection_decision(failed, None, None)
    assert decision == {
        "outcome": "rejected_calibration",
        "semantic_acceptance_allowed": False,
        "product_enabled": False,
    }


def test_fusion_v3_decision_requires_all_three_selection_stages() -> None:
    passed = {"passes_all": True}
    assert (
        selection_decision(passed, passed, passed)["outcome"]
        == "eligible_for_semantic_acceptance"
    )
    assert (
        selection_decision(passed, copy.deepcopy({"passes_all": False}), None)[
            "outcome"
        ]
        == "rejected_holdout"
    )


def test_fusion_v3_synthetic_rejection_report_validation_path() -> None:
    protocol = effective_protocol(
        json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    )
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    dense_protocol = json.loads(DEFAULT_DENSE_PROTOCOL.read_text(encoding="utf-8"))
    allowed = set(fixture["split"]["calibration_case_ids"])
    product_modes = {}
    for mode_name in ("scoped", "full_library"):
        source_rows = [
            row
            for row in baseline["channels"]["bm25_product_queries"]["modes"][mode_name][
                "cases"
            ]
            if row["id"] in allowed
        ]
        rows = []
        for source in source_rows:
            metrics = {key: source["metrics"].get(key, 0.0) for key in GATE_METRICS}
            rank = source["first_relevant_rank"]
            projected_rank = rank if rank is not None and rank <= 10 else None
            if projected_rank is None:
                metrics["mrr"] = 0.0
            rows.append(
                {
                    "id": source["id"],
                    "first_relevant_rank": projected_rank,
                    "metrics": metrics,
                }
            )
        aggregate = mean_metrics([row["metrics"] for row in rows])
        product_modes[mode_name] = {
            "case_count": len(rows),
            "metrics": {key: round(aggregate[key], 6) for key in GATE_METRICS},
            "latency_ms": {"p50": 0.0, "p95": 0.0, "max": 0.0},
            "ks": [5, 10],
            "cases": rows,
        }
    product = {"channel": PRODUCT_CHANNEL, "modes": product_modes}
    fusion = copy.deepcopy(product)
    fusion["channel"] = FUSION_CHANNEL
    calibration = {
        "channels": {PRODUCT_CHANNEL: product, FUSION_CHANNEL: fusion},
        "gate": evaluate_partition_gate(product, fusion),
    }
    selection = {"calibration": calibration, "holdout": None, "combined": None}
    report = {
        "schema_version": 1,
        "report_kind": "retrieval_user_journey_fusion_v3_1_selection",
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "fixture_sha256": sha256_file(DEFAULT_FIXTURE),
        "baseline_sha256": sha256_file(DEFAULT_BASELINE),
        "candidate": protocol["frozen_candidate"],
        "index": {**dense_protocol["existing_index"], "rebuilt": False},
        "query_embedding_cache": {
            "enabled": True,
            "unique_query_count": 12,
            "shared_across_partitions_and_modes": True,
        },
        "selection": selection,
        "decision": selection_decision(calibration["gate"], None, None),
        "result_fingerprint": _selection_projection(selection)["sha256"],
        "holdout_accessed": False,
        "semantic_acceptance_accessed": False,
        "product_enabled": False,
    }
    assert (
        validate_selection_report(
            report,
            protocol=protocol,
            fixture=fixture,
            baseline=baseline,
            expected_result_fingerprint=None,
        )
        == []
    )
