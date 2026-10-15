from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_local_semantic_preparation_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_local_semantic_preparation_protocol_passes() -> None:
    assert validate_protocol(_protocol()) == []


def test_protocol_rejects_automatic_query_download() -> None:
    changed = copy.deepcopy(_protocol())
    changed["artifacts"]["automatic_query_time_download_allowed"] = True
    assert "local semantic artifact integrity contract changed" in validate_protocol(changed)


def test_protocol_rejects_artifact_hash_drift() -> None:
    changed = copy.deepcopy(_protocol())
    changed["artifacts"]["files"][0]["sha256"] = "0" * 64
    assert "local semantic artifact integrity contract changed" in validate_protocol(changed)


def test_protocol_rejects_index_overwrite_before_commit() -> None:
    changed = copy.deepcopy(_protocol())
    changed["library_index"]["previous_ready_generation_preserved_until_commit"] = False
    assert "local semantic index atomicity contract changed" in validate_protocol(changed)


def test_protocol_rejects_missing_user_remediation() -> None:
    changed = copy.deepcopy(_protocol())
    changed["public_contract"]["remediation_by_state"].pop("stale")
    assert "local semantic public remediation contract changed" in validate_protocol(changed)


def test_protocol_rejects_ordinary_user_exposure() -> None:
    changed = copy.deepcopy(_protocol())
    changed["product_boundary"]["ordinary_user_surface_exposed"] = True
    assert "local semantic promotion or product boundary changed" in validate_protocol(changed)
