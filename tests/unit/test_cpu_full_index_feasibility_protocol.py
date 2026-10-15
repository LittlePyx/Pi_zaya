from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_cpu_full_index_feasibility_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_cpu_full_index_protocol_is_valid() -> None:
    assert validate_protocol(_protocol()) == []


def test_cpu_full_index_protocol_rejects_longer_build_budget() -> None:
    changed = copy.deepcopy(_protocol())
    changed["operational_gate"]["maximum_full_build_ms"] = 900000.0
    assert "CPU full-index operational gate changed" in validate_protocol(changed)


def test_cpu_full_index_protocol_rejects_smaller_document_batch() -> None:
    changed = copy.deepcopy(_protocol())
    changed["cpu_provider"]["document_batch_size"] = 4
    assert "CPU full-index provider changed" in validate_protocol(changed)


def test_cpu_full_index_protocol_rejects_product_index_write() -> None:
    changed = copy.deepcopy(_protocol())
    changed["dedicated_index"]["product_index_writes_allowed"] = True
    assert "CPU full-index destination boundary changed" in validate_protocol(changed)


def test_cpu_full_index_protocol_rejects_product_enablement() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["product_enabled"] = True
    assert "CPU full-index execution boundary changed" in validate_protocol(changed)
