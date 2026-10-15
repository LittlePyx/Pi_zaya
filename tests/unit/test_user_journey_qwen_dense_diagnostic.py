from __future__ import annotations

import copy
import json

import pytest

from tools.retrieval_eval.run_user_journey_qwen_dense_diagnostic import (
    DEFAULT_BASELINE,
    DEFAULT_FIXTURE,
    DENSE_CHANNEL,
    PRODUCT_CHANNEL,
    _result_fingerprint,
    calibration_fixture,
    evaluate_complementarity,
    frozen_product_control,
    validate_report,
)
from tools.retrieval_eval.validate_user_journey_qwen_dense_protocol import (
    DEFAULT_PROTOCOL,
)


def _channel(name: str, scoped: list[float], full: list[float]) -> dict:
    def mode(values: list[float]) -> dict:
        rows = [
            {
                "id": f"case-{index}",
                "first_relevant_rank": 1 if value else None,
                "metrics": {"recall@10": value},
            }
            for index, value in enumerate(values)
        ]
        return {
            "metrics": {"recall@10": round(sum(values) / len(values), 6)},
            "cases": rows,
        }

    return {
        "channel": name,
        "modes": {"scoped": mode(scoped), "full_library": mode(full)},
    }


def test_dense_complementarity_requires_unique_wins_and_aggregate_gain() -> None:
    product = _channel("product", [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    dense = _channel("dense", [1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0])
    result = evaluate_complementarity(product, dense)
    assert result["modes"]["scoped"]["class_counts_at_10"]["dense_only"] == 2
    assert result["fusion_v3_design_condition"]["passes_all"] is True


def test_dense_complementarity_rejects_gain_without_two_unique_wins() -> None:
    product = _channel("product", [0.0, 0.0], [0.0, 0.0])
    dense = _channel("dense", [1.0, 0.0], [1.0, 0.0])
    assert (
        evaluate_complementarity(product, dense)["fusion_v3_design_condition"][
            "passes_all"
        ]
        is False
    )


def test_calibration_fixture_excludes_every_holdout_case() -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    fixture = {
        "split": {
            "calibration_case_ids": ["a", "b"],
            "holdout_case_ids": ["h"],
        },
        "cases": [{"id": "a"}, {"id": "h"}, {"id": "b"}],
    }
    changed = copy.deepcopy(protocol)
    changed["partition"]["case_count"] = 2
    selected = calibration_fixture(fixture, changed)
    assert [case["id"] for case in selected["cases"]] == ["a", "b"]


def test_calibration_fixture_fails_when_partition_count_drifts() -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    fixture = {
        "split": {"calibration_case_ids": ["a"], "holdout_case_ids": ["h"]},
        "cases": [{"id": "a"}, {"id": "h"}],
    }
    with pytest.raises(ValueError, match="calibration partition"):
        calibration_fixture(fixture, protocol)


def test_synthetic_dense_report_validation_path() -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    calibration_ids = list(fixture["split"]["calibration_case_ids"])
    product = frozen_product_control(baseline, calibration_ids)
    dense = copy.deepcopy(product)
    dense["channel"] = DENSE_CHANNEL
    channels = {PRODUCT_CHANNEL: product, DENSE_CHANNEL: dense}
    from kb.retrieval_eval_fixture import sha256_file

    report = {
        "schema_version": 1,
        "report_kind": "retrieval_user_journey_qwen_dense_diagnostic_v1",
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "fixture_sha256": sha256_file(DEFAULT_FIXTURE),
        "baseline_sha256": sha256_file(DEFAULT_BASELINE),
        "calibration_case_ids": calibration_ids,
        "frozen_representation": protocol["frozen_representation"],
        "index": {**protocol["existing_index"], "rebuilt": False},
        "query_embedding_cache": {
            "enabled": True,
            "unique_query_count": 12,
            "shared_across_scoped_and_full_library": True,
        },
        "channels": channels,
        "analysis": evaluate_complementarity(product, dense),
        "result_fingerprint": _result_fingerprint(channels),
        "holdout_accessed": False,
        "fusion_or_reranker_executed": False,
        "product_enabled": False,
    }
    assert (
        validate_report(
            report,
            protocol=protocol,
            fixture=fixture,
            baseline=baseline,
            expected_result_fingerprint=None,
        )
        == []
    )
