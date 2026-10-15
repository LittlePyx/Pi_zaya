from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_cpu_padding_diagnostic_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_cpu_padding_protocol_is_valid() -> None:
    assert validate_protocol(_protocol()) == []


def test_cpu_padding_protocol_rejects_model_load() -> None:
    changed = copy.deepcopy(_protocol())
    changed["tokenizer"]["model_load_allowed"] = True
    assert "CPU padding tokenizer boundary changed" in validate_protocol(changed)


def test_cpu_padding_protocol_rejects_branch_relaxation() -> None:
    changed = copy.deepcopy(_protocol())
    changed["branch_gate"]["minimum_attention_work_proxy_speedup"] = 2.0
    assert "CPU padding branch gate changed" in validate_protocol(changed)


def test_cpu_padding_protocol_rejects_embedding() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["documents_embedded"] = 1
    assert "CPU padding execution boundary changed" in validate_protocol(changed)
