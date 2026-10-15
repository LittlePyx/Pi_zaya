from __future__ import annotations

import json

from tools.retrieval_eval.validate_candidate_comparison_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_candidate_comparison_protocol_is_frozen_before_shortlisting() -> None:
    assert validate_protocol(_protocol()) == []


def test_candidate_protocol_rejects_holdout_budget_and_product_drift() -> None:
    changed = _protocol()
    changed["inner_split"]["confirmation_case_ids"].pop()
    changed["candidate_budget"]["maximum_new_candidates"] = 20
    changed["failure_policy"]["product_enabled"] = True
    errors = validate_protocol(changed)
    assert any("confirmation membership" in error for error in errors)
    assert any("candidate safety or resource budget" in error for error in errors)
    assert any("failure or product-exposure policy" in error for error in errors)
