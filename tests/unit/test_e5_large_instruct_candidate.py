from __future__ import annotations

import json

from tools.retrieval_eval.validate_e5_large_instruct_candidate import (
    DEFAULT_REGISTRY,
    validate_candidate,
)


def _candidate() -> dict:
    return json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))


def test_e5_large_instruct_candidate_is_frozen_before_measurement() -> None:
    assert validate_candidate(_candidate()) == []


def test_e5_large_validator_rejects_prompt_supply_chain_and_product_drift() -> None:
    changed = _candidate()
    changed["selected_candidate"]["query_prefix"] = "Query: "
    changed["selected_candidate"]["trust_remote_code"] = True
    changed["selected_candidate"]["product_enabled"] = True
    changed["comparison_baseline"]["model_safetensors_sha256"] = "0" * 64
    errors = validate_candidate(changed)
    assert any("query_prefix" in error for error in errors)
    assert any("trust_remote_code" in error for error in errors)
    assert any("product_enabled" in error for error in errors)
    assert any("comparison baseline model_safetensors_sha256" in error for error in errors)
