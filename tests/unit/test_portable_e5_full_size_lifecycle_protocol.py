from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_portable_e5_full_size_lifecycle_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_full_size_lifecycle_protocol_passes() -> None:
    assert validate_protocol(_protocol()) == []


def test_protocol_rejects_component_payload_drift() -> None:
    changed = copy.deepcopy(_protocol())
    changed["component"]["payload_sha256"] = "0" * 64
    assert "full-size lifecycle component changed" in validate_protocol(changed)


def test_protocol_rejects_network_enablement() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["allow_external_network"] = True
    assert "full-size lifecycle execution boundary changed" in validate_protocol(changed)


def test_protocol_rejects_ranking_rewrite() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["expected_queries"][0]["ranked_chunk_ids"].reverse()
    assert "full-size lifecycle expected rankings changed" in validate_protocol(changed)


def test_protocol_rejects_longer_recovery_threshold() -> None:
    changed = copy.deepcopy(_protocol())
    changed["gates"]["maximum_stale_repair_ms"] = 120000.0
    assert "full-size lifecycle gates changed" in validate_protocol(changed)


def test_protocol_rejects_product_enablement() -> None:
    changed = copy.deepcopy(_protocol())
    changed["product_boundary"]["product_enabled"] = True
    assert "full-size lifecycle product boundary changed" in validate_protocol(changed)
