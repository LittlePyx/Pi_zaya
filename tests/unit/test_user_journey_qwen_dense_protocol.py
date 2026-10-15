from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_user_journey_qwen_dense_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_qwen_dense_diagnostic_protocol_is_frozen_before_measurement() -> None:
    assert validate_protocol(_protocol()) == []


def test_qwen_dense_diagnostic_protocol_rejects_holdout_access() -> None:
    payload = copy.deepcopy(_protocol())
    payload["partition"]["holdout_access_allowed"] = True
    assert any("calibration/holdout" in error for error in validate_protocol(payload))


def test_qwen_dense_diagnostic_protocol_separates_ranking_and_reporting_depths() -> (
    None
):
    payload = copy.deepcopy(_protocol())
    payload["diagnostic_contract"]["retrieval_ranking_depth"] = 10
    assert any("depth or execution" in error for error in validate_protocol(payload))


def test_qwen_dense_diagnostic_protocol_rejects_representation_changes() -> None:
    payload = copy.deepcopy(_protocol())
    payload["frozen_representation"]["query_prefix"] = "changed"
    assert any("representation" in error for error in validate_protocol(payload))


def test_qwen_dense_diagnostic_protocol_cannot_enable_product() -> None:
    payload = copy.deepcopy(_protocol())
    payload["safety_boundary"]["product_enabled"] = True
    assert any("safety boundary" in error for error in validate_protocol(payload))
