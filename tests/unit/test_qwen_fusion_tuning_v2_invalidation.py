from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_qwen_fusion_tuning_v2_invalidation import (
    DEFAULT_INVALIDATION,
    validate_invalidation,
)


def _invalidation() -> dict:
    return json.loads(DEFAULT_INVALIDATION.read_text(encoding="utf-8"))


def test_invalid_fusion_tuning_run_is_pinned_and_cannot_promote() -> None:
    assert validate_invalidation(_invalidation()) == []


def test_invalid_fusion_tuning_run_cannot_enable_confirmation() -> None:
    payload = copy.deepcopy(_invalidation())
    payload["effective_decision"]["confirmation_allowed"] = True
    assert any("no-promotion" in error for error in validate_invalidation(payload))


def test_invalid_fusion_tuning_run_cannot_be_remeasured() -> None:
    payload = copy.deepcopy(_invalidation())
    payload["effective_decision"]["remeasurement_allowed_under_v2"] = True
    assert any("no-promotion" in error for error in validate_invalidation(payload))
