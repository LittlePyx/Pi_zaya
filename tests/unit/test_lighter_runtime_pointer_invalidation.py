from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_lighter_runtime_pointer_invalidation import (
    DEFAULT_INVALIDATION,
    validate_invalidation,
)


def _invalidation() -> dict:
    return json.loads(DEFAULT_INVALIDATION.read_text(encoding="utf-8"))


def test_lighter_runtime_pointer_invalidation_is_valid() -> None:
    assert validate_invalidation(_invalidation()) == []


def test_lighter_runtime_pointer_invalidation_cannot_authorize_old_protocol() -> None:
    changed = copy.deepcopy(_invalidation())
    changed["scope"]["stage_1_execution_under_invalidated_protocol_allowed"] = True
    assert "lighter runtime pointer invalidation scope changed" in validate_invalidation(changed)


def test_lighter_runtime_pointer_invalidation_requires_corrected_recovery() -> None:
    changed = copy.deepcopy(_invalidation())
    changed["recovery"]["must_pin_retained_index_v1"] = False
    assert "lighter runtime pointer recovery changed" in validate_invalidation(changed)
