from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_lighter_runtime_stage2_e5_small_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_e5_stage2_protocol_passes() -> None:
    assert validate_protocol(_protocol()) == []


def test_e5_stage2_protocol_rejects_second_full_build() -> None:
    changed = copy.deepcopy(_protocol())
    changed["authorization"]["maximum_full_builds_across_batch"] = 2
    assert "E5 Stage-2 authorization changed" in validate_protocol(changed)


def test_e5_stage2_protocol_rejects_non_atomic_index() -> None:
    changed = copy.deepcopy(_protocol())
    changed["index_format"]["atomic_current_pointer_replace"] = False
    assert "E5 Stage-2 index format changed" in validate_protocol(changed)


def test_e5_stage2_protocol_rejects_weaker_quality_gate() -> None:
    changed = copy.deepcopy(_protocol())
    changed["semantic_evaluation"]["protected_mrr_no_regression_vs_product_each_mode"] = False
    assert "E5 Stage-2 semantic gates changed" in validate_protocol(changed)


def test_e5_stage2_protocol_rejects_user_exposure() -> None:
    changed = copy.deepcopy(_protocol())
    changed["product_boundary"]["ordinary_user_surface_exposed"] = True
    assert "E5 Stage-2 product boundary changed" in validate_protocol(changed)
