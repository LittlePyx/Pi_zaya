from __future__ import annotations

import json
from pathlib import Path

from tools.retrieval_eval.validate_parent_collapsed_multivector_candidate import (
    validate_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "docs" / "retrieval_parent_collapsed_multivector_candidate_v1.json"


def _registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_parent_collapsed_multivector_candidate_is_frozen_before_measurement() -> None:
    assert validate_candidate(_registry()) == []


def test_candidate_validator_rejects_locator_or_product_activation_drift() -> None:
    changed = _registry()
    changed["representation_contract"]["returned_id"] = "child id"
    changed["embedding_candidate"]["product_enabled"] = True
    errors = validate_candidate(changed)
    assert any("returned_id" in error for error in errors)
    assert any("product_enabled" in error for error in errors)
