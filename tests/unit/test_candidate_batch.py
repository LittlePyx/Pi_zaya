from __future__ import annotations

import json

from tools.retrieval_eval.validate_candidate_batch import DEFAULT_BATCH, validate_batch


def _batch() -> dict:
    return json.loads(DEFAULT_BATCH.read_text(encoding="utf-8"))


def test_candidate_batch_is_frozen_before_screening() -> None:
    assert validate_batch(_batch()) == []


def test_candidate_batch_rejects_encoding_supply_chain_and_product_drift() -> None:
    changed = _batch()
    changed["candidates"][0]["query_prefix"] = "Query: "
    changed["candidates"][1]["trust_remote_code"] = True
    changed["benchmark_runtime"]["transformers_version"] = "latest"
    changed["execution_boundary"]["product_enabled"] = True
    errors = validate_batch(changed)
    assert any("query_prefix changed" in error for error in errors)
    assert any("encoding contract hash changed" in error for error in errors)
    assert any("execution boundary changed" in error for error in errors)
    assert any("benchmark runtime contract changed" in error for error in errors)
    assert any("execution boundary product_enabled" in error for error in errors)
