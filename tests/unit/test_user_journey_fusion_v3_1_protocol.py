from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_user_journey_fusion_v3_1_protocol import (
    DEFAULT_PROTOCOL,
    effective_protocol,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_fusion_v3_1_protocol_freezes_only_depth_projection_correction() -> None:
    payload = _protocol()
    assert validate_protocol(payload) == []
    assert effective_protocol(payload)["frozen_candidate"]["candidate_count"] == 1


def test_fusion_v3_1_protocol_rejects_candidate_contract_change() -> None:
    payload = copy.deepcopy(_protocol())
    payload["unchanged_contract_section_sha256"]["frozen_candidate"] = "0" * 64
    assert any("inherited candidate" in error for error in validate_protocol(payload))


def test_fusion_v3_1_protocol_rejects_another_projection_change() -> None:
    payload = copy.deepcopy(_protocol())
    payload["sole_correction"]["selection_ranking_depth"] = 9
    assert any("sole depth-projection" in error for error in validate_protocol(payload))


def test_fusion_v3_1_protocol_allows_only_one_replacement_run() -> None:
    payload = copy.deepcopy(_protocol())
    payload["replacement_execution"]["maximum_runs"] = 2
    assert any("execution boundary" in error for error in validate_protocol(payload))
