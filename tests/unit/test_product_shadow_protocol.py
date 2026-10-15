from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_product_shadow_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_product_shadow_protocol_is_frozen_and_valid() -> None:
    assert validate_protocol(_protocol()) == []


def test_product_shadow_protocol_rejects_product_enablement() -> None:
    changed = copy.deepcopy(_protocol())
    changed["product_boundary"]["changes_product_ranking"] = True
    assert "product shadow safety boundary changed" in validate_protocol(changed)


def test_product_shadow_protocol_rejects_candidate_tuning() -> None:
    changed = copy.deepcopy(_protocol())
    changed["candidate"]["fusion"]["dense_novelty_quota"] = 3
    assert "product shadow accepted candidate changed" in validate_protocol(changed)


def test_product_shadow_protocol_rejects_raw_content_telemetry() -> None:
    changed = copy.deepcopy(_protocol())
    changed["telemetry"]["fields"].append("query")
    assert "product shadow telemetry admits raw content" in validate_protocol(changed)


def test_product_shadow_protocol_rejects_relaxed_operational_gate() -> None:
    changed = copy.deepcopy(_protocol())
    changed["controlled_operational_gate"]["maximum_source_escape_count"] = 1
    assert "product shadow operational gate changed" in validate_protocol(changed)
