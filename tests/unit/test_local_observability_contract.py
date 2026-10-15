from __future__ import annotations

from kb.local_observability import local_observability_contract
from tools.observability.validate_local_observability_contract import validate_contract


def test_frozen_local_observation_contract_passes_privacy_replay() -> None:
    assert validate_contract(local_observability_contract()) == []


def test_contract_validator_rejects_allowlist_drift() -> None:
    changed = local_observability_contract()
    changed["events"]["pi_zaya.retrieval"].append("query")
    errors = validate_contract(changed)
    assert any("does not match" in error for error in errors)
