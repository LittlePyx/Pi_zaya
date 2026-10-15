from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from tools.research_qa.validate_claim_nli_candidate import validate_registry


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "docs" / "claim_entailment_nli_candidate_v1.json"


def _registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_claim_nli_candidate_is_frozen_before_measurement() -> None:
    payload = _registry()
    assert validate_registry(payload) == []
    assert payload["candidate"]["product_enabled"] is False
    assert payload["privacy_and_supply_chain"]["inference_location"] == "local_only"
    assert payload["measurement_policy"]["run_count_on_frozen_fixture"] == 1


def test_claim_nli_candidate_rejects_threshold_or_privacy_drift() -> None:
    changed = deepcopy(_registry())
    changed["acceptance"]["contradiction_false_support_rate_max"] = 0.5
    assert any("thresholds changed" in error for error in validate_registry(changed))

    changed = deepcopy(_registry())
    changed["privacy_and_supply_chain"]["paper_text_uploaded"] = True
    assert any("privacy contract changed" in error for error in validate_registry(changed))
