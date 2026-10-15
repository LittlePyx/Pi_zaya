from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_portable_e5_full_size_lifecycle_protocol_v3 import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_release_component_lifecycle_protocol_passes() -> None:
    assert validate_protocol(_protocol()) == []


def test_protocol_rejects_v2_lineage_rewrite() -> None:
    changed = copy.deepcopy(_protocol())
    changed["authorization"]["full_size_v2_result_fingerprint"] = "0" * 64
    assert "release-component lifecycle authorization changed" in validate_protocol(
        changed
    )


def test_protocol_rejects_release_payload_rewrite() -> None:
    changed = copy.deepcopy(_protocol())
    changed["component"]["payload_file_count"] = 680
    assert "release-component lifecycle component changed" in validate_protocol(changed)


def test_protocol_rejects_restored_bytecode() -> None:
    changed = copy.deepcopy(_protocol())
    changed["release_handoff"]["cached_bytecode_entries_removed"] = 0
    assert "release-component lifecycle handoff changed" in validate_protocol(changed)


def test_protocol_rejects_path_layout_rewrite() -> None:
    changed = copy.deepcopy(_protocol())
    changed["gates"]["expected_component_layout_max_path_chars"] = 239
    assert "release-component lifecycle gates changed" in validate_protocol(changed)


def test_protocol_rejects_ranking_rewrite() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["expected_queries"][0]["ranked_chunk_ids"].reverse()
    assert "release-component lifecycle expected rankings changed" in validate_protocol(
        changed
    )


def test_protocol_rejects_product_enablement() -> None:
    changed = copy.deepcopy(_protocol())
    changed["product_boundary"]["product_enabled"] = True
    assert "release-component lifecycle product boundary changed" in validate_protocol(
        changed
    )
