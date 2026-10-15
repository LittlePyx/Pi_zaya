from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_portable_optional_e5_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_portable_optional_e5_protocol_passes() -> None:
    assert validate_protocol(_protocol()) == []


def test_portable_optional_e5_protocol_rejects_base_archive_mutation() -> None:
    changed = copy.deepcopy(_protocol())
    changed["base_portable"]["archive_bytes"] += 1
    assert "portable optional E5 base archive contract changed" in validate_protocol(
        changed
    )


def test_portable_optional_e5_protocol_rejects_default_bundling() -> None:
    changed = copy.deepcopy(_protocol())
    changed["decision"]["bundling_into_every_install_allowed"] = True
    assert "portable optional E5 decision changed" in validate_protocol(changed)


def test_portable_optional_e5_protocol_rejects_product_enablement() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["product_enabled"] = True
    assert "portable optional E5 execution boundary changed" in validate_protocol(
        changed
    )


def test_portable_optional_e5_protocol_rejects_external_knowledge() -> None:
    changed = copy.deepcopy(_protocol())
    changed["product_boundary"]["external_web_or_model_knowledge_used"] = True
    assert "portable optional E5 product boundary changed" in validate_protocol(changed)


def test_portable_optional_e5_protocol_rejects_relaxed_size_gate() -> None:
    changed = copy.deepcopy(_protocol())
    changed["gates"]["maximum_component_archive_bytes"] += 1
    assert "portable optional E5 gates changed" in validate_protocol(changed)
