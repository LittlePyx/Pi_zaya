from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_portable_e5_full_size_lifecycle_protocol_v2 import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_short_path_lifecycle_protocol_passes() -> None:
    assert validate_protocol(_protocol()) == []


def test_protocol_rejects_v1_rejection_rewrite() -> None:
    changed = copy.deepcopy(_protocol())
    changed["authorization"]["rejected_v1_result_fingerprint"] = "0" * 64
    assert "short-path lifecycle authorization changed" in validate_protocol(changed)


def test_protocol_rejects_path_budget_weakening() -> None:
    changed = copy.deepcopy(_protocol())
    changed["path_repair"]["maximum_windows_target_path_characters"] = 275
    assert "short-path lifecycle repair boundary changed" in validate_protocol(changed)


def test_protocol_rejects_long_component_store_path() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["component_store_relative_path"] = "component_store"
    assert "short-path lifecycle execution boundary changed" in validate_protocol(
        changed
    )


def test_protocol_rejects_ranking_rewrite() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["expected_queries"][0]["ranked_chunk_ids"].reverse()
    assert "short-path lifecycle expected rankings changed" in validate_protocol(changed)


def test_protocol_rejects_timing_threshold_relaxation() -> None:
    changed = copy.deepcopy(_protocol())
    changed["gates"]["maximum_install_ready_ms"] = 180000.0
    assert "short-path lifecycle gates changed" in validate_protocol(changed)


def test_protocol_rejects_product_enablement() -> None:
    changed = copy.deepcopy(_protocol())
    changed["product_boundary"]["product_enabled"] = True
    assert "short-path lifecycle product boundary changed" in validate_protocol(changed)
