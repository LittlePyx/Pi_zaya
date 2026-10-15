from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_user_journey_fusion_v3_1_semantic_protocol_v2 import (
    DEFAULT_PROTOCOL,
    effective_protocol,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_semantic_v2_freezes_dedicated_index_before_measurement() -> None:
    payload = _protocol()
    assert validate_protocol(payload) == []
    assert (
        effective_protocol(payload)["unchanged_candidate"]["dense_novelty_quota"] == 2
    )


def test_semantic_v2_rejects_inherited_candidate_change() -> None:
    payload = copy.deepcopy(_protocol())
    payload["inherited_contract_sha256"]["unchanged_candidate"] = "0" * 64
    assert any("candidate, gate" in error for error in validate_protocol(payload))


def test_semantic_v2_allows_only_one_initial_index_build() -> None:
    payload = copy.deepcopy(_protocol())
    payload["dedicated_index"]["maximum_initial_builds"] = 2
    assert any("index contract" in error for error in validate_protocol(payload))


def test_semantic_v2_rejects_query_or_gate_variants() -> None:
    payload = copy.deepcopy(_protocol())
    payload["execution_contract"]["model_or_prompt_variants"] = 1
    assert any("execution contract" in error for error in validate_protocol(payload))
