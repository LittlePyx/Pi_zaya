from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_product_shadow_vectorized_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_vectorized_shadow_protocol_is_valid() -> None:
    assert validate_protocol(_protocol()) == []


def test_vectorized_shadow_protocol_rejects_threshold_relaxation() -> None:
    changed = copy.deepcopy(_protocol())
    changed["inherited_unchanged"]["maximum_warm_shadow_p95_ms"] = 275.0
    assert "vectorized shadow inherited experiment changed" in validate_protocol(
        changed
    )


def test_vectorized_shadow_protocol_rejects_approximate_only_ranking() -> None:
    changed = copy.deepcopy(_protocol())
    changed["allowed_implementation_changes"]["exact_shortlist_rule"] = "float32_only"
    assert "vectorized shadow scoring contract changed" in validate_protocol(changed)


def test_vectorized_shadow_protocol_rejects_second_replacement_run() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["maximum_replacement_runs"] = 2
    assert "vectorized shadow execution boundary changed" in validate_protocol(changed)
