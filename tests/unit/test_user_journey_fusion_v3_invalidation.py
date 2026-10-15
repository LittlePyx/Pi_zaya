from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_user_journey_fusion_v3_invalidation import (
    DEFAULT_INVALIDATION,
    validate_invalidation,
)


def _invalidation() -> dict:
    return json.loads(DEFAULT_INVALIDATION.read_text(encoding="utf-8"))


def test_fusion_v3_failed_execution_is_closed_without_result() -> None:
    assert validate_invalidation(_invalidation()) == []


def test_fusion_v3_invalidation_cannot_claim_candidate_result() -> None:
    payload = copy.deepcopy(_invalidation())
    payload["execution_boundary"]["calibration_candidate_result_observed"] = True
    assert any(
        "execution boundary" in error for error in validate_invalidation(payload)
    )


def test_fusion_v3_invalidation_cannot_open_holdout() -> None:
    payload = copy.deepcopy(_invalidation())
    payload["effective_decision"]["holdout_allowed_under_v3"] = True
    assert any("no-promotion" in error for error in validate_invalidation(payload))


def test_fusion_v3_replacement_cannot_change_candidate() -> None:
    payload = copy.deepcopy(_invalidation())
    payload["effective_decision"]["candidate_configuration_change_allowed"] = True
    assert any("no-promotion" in error for error in validate_invalidation(payload))
