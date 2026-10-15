from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_user_journey_fusion_v3_1_semantic_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_semantic_acceptance_protocol_is_frozen_before_measurement() -> None:
    assert validate_protocol(_protocol()) == []


def test_semantic_acceptance_protocol_rejects_candidate_change() -> None:
    payload = copy.deepcopy(_protocol())
    payload["unchanged_candidate"]["dense_novelty_quota"] = 1
    assert any("candidate changed" in error for error in validate_protocol(payload))


def test_semantic_acceptance_protocol_rejects_threshold_relaxation() -> None:
    payload = copy.deepcopy(_protocol())
    payload["acceptance_gate"]["minimum_gain_vs_product_in_each_mode"] = 0.1
    assert any("acceptance gate" in error for error in validate_protocol(payload))


def test_semantic_acceptance_protocol_cannot_enable_shadow_on_failure() -> None:
    payload = copy.deepcopy(_protocol())
    payload["failure_policy"]["product_shadow_allowed_on_failure"] = True
    assert any("failure policy" in error for error in validate_protocol(payload))
