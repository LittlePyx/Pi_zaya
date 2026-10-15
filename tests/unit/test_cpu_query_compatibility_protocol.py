from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_cpu_query_compatibility_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_cpu_query_protocol_is_valid() -> None:
    assert validate_protocol(_protocol()) == []


def test_cpu_query_protocol_rejects_latency_relaxation() -> None:
    changed = copy.deepcopy(_protocol())
    changed["predeclared_gate"]["maximum_warm_query_embedding_p95_ms"] = 1500.0
    assert "CPU query gate changed" in validate_protocol(changed)


def test_cpu_query_protocol_rejects_zero_query_lineage_change() -> None:
    changed = copy.deepcopy(_protocol())
    changed["predecessor"]["queries_encoded_before_successor_freeze"] = 1
    assert "CPU query predecessor or zero-query invalidation changed" in validate_protocol(
        changed
    )


def test_cpu_query_protocol_rejects_index_write() -> None:
    changed = copy.deepcopy(_protocol())
    changed["read_only_index"]["writes_allowed"] = True
    assert "CPU query read-only index boundary changed" in validate_protocol(changed)


def test_cpu_query_protocol_rejects_product_enablement() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["product_enabled"] = True
    assert "CPU query execution boundary changed" in validate_protocol(changed)
