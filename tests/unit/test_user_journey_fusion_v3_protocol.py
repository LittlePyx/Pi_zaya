from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_user_journey_fusion_v3_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_fusion_v3_protocol_is_frozen_before_runner_and_measurement() -> None:
    assert validate_protocol(_protocol()) == []


def test_fusion_v3_protocol_rejects_a_second_candidate() -> None:
    payload = copy.deepcopy(_protocol())
    payload["frozen_candidate"]["candidate_count"] = 2
    assert any("candidate changed" in error for error in validate_protocol(payload))


def test_fusion_v3_protocol_rejects_top_k_contract_coupling() -> None:
    payload = copy.deepcopy(_protocol())
    payload["depth_contract"]["selection_reported_metric_depths"].append(20)
    assert any("depth contract" in error for error in validate_protocol(payload))


def test_fusion_v3_protocol_rejects_early_holdout_access() -> None:
    payload = copy.deepcopy(_protocol())
    payload["partition"]["holdout_hidden_until_calibration_passes"] = False
    assert any("partition boundary" in error for error in validate_protocol(payload))


def test_fusion_v3_protocol_cannot_relax_source_safety() -> None:
    payload = copy.deepcopy(_protocol())
    payload["calibration_gate"]["required_in_both_modes"].remove(
        "wrong_source_rate@10_no_regression"
    )
    assert any("calibration gate" in error for error in validate_protocol(payload))


def test_fusion_v3_protocol_cannot_enable_product() -> None:
    payload = copy.deepcopy(_protocol())
    payload["safety_boundary"]["product_enabled"] = True
    assert any("safety boundary" in error for error in validate_protocol(payload))
