from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_user_journey_fusion_v3_1_semantic_invalidation import (
    DEFAULT_INVALIDATION,
    validate_invalidation,
)


def _payload() -> dict:
    return json.loads(DEFAULT_INVALIDATION.read_text(encoding="utf-8"))


def test_semantic_preflight_is_closed_without_measurement() -> None:
    assert validate_invalidation(_payload()) == []


def test_semantic_preflight_cannot_claim_query_embedding() -> None:
    payload = copy.deepcopy(_payload())
    payload["execution_boundary"]["query_embeddings_created"] = 1
    assert any(
        "execution boundary" in error for error in validate_invalidation(payload)
    )


def test_semantic_preflight_cannot_change_candidate_or_gate() -> None:
    payload = copy.deepcopy(_payload())
    payload["effective_decision"]["candidate_or_gate_change_allowed"] = True
    assert any(
        "no-result decision" in error for error in validate_invalidation(payload)
    )
