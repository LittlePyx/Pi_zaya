from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_lighter_runtime_stage1_e5_small_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_e5_small_stage1_protocol_passes() -> None:
    assert validate_protocol(_protocol()) == []


def test_e5_small_stage1_protocol_rejects_partial_evidence_windows() -> None:
    changed = copy.deepcopy(_protocol())
    changed["encoding"]["silent_tail_loss_allowed"] = True
    assert "E5-small Stage-1 encoding changed" in validate_protocol(changed)


def test_e5_small_stage1_protocol_rejects_local_quality_shortcut() -> None:
    changed = copy.deepcopy(_protocol())
    changed["measurement"]["quality_evaluation_allowed"] = True
    assert "E5-small Stage-1 measurement changed" in validate_protocol(changed)


def test_e5_small_stage1_protocol_rejects_weaker_stage2_quality() -> None:
    changed = copy.deepcopy(_protocol())
    changed["stage_2_if_authorized"][
        "protected_wrong_source_at_5_equal_to_product_each_mode"
    ] = False
    assert "E5-small Stage-2 authorization boundary changed" in validate_protocol(changed)


def test_e5_small_stage1_protocol_rejects_product_enablement() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["product_enabled"] = True
    assert "E5-small Stage-1 execution changed" in validate_protocol(changed)
