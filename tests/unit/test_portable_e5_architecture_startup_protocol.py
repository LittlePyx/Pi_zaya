from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_portable_e5_architecture_startup_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_architecture_startup_protocol_passes_historical_validation() -> None:
    assert validate_protocol(_protocol()) == []


def test_architecture_startup_protocol_passes_strict_preflight() -> None:
    assert validate_protocol(_protocol(), inspect_runtime_files=True) == []


def test_architecture_startup_protocol_preserves_v2_rejection() -> None:
    changed = copy.deepcopy(_protocol())
    changed["source_v2"]["verdict_may_be_changed"] = True
    assert "portable E5 architecture-startup v2 rejection lineage changed" in validate_protocol(
        changed
    )


def test_architecture_startup_protocol_rejects_candidate_relabel() -> None:
    changed = copy.deepcopy(_protocol())
    changed["architecture_candidate"]["relevance_rerun_allowed"] = True
    assert "portable E5 architecture-startup candidate changed" in validate_protocol(changed)


def test_architecture_startup_protocol_rejects_timing_relaxation() -> None:
    changed = copy.deepcopy(_protocol())
    changed["gates"]["maximum_product_import_ms"] = 10_000.0
    assert "portable E5 architecture-startup gates changed" in validate_protocol(changed)


def test_architecture_startup_protocol_rejects_ranking_change() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["expected_queries"][0]["ranked_chunk_ids"].reverse()
    assert "portable E5 architecture-startup expected ranking changed" in validate_protocol(
        changed
    )


def test_architecture_startup_protocol_rejects_payload_identity_change() -> None:
    changed = copy.deepcopy(_protocol())
    changed["component"]["payload_sha256"] = "0" * 64
    assert "portable E5 architecture-startup component changed" in validate_protocol(changed)


def test_architecture_startup_protocol_never_enables_product() -> None:
    changed = copy.deepcopy(_protocol())
    changed["decision"]["product_enablement_allowed"] = True
    assert "portable E5 architecture-startup decision changed" in validate_protocol(changed)
