from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_qwen_fusion_protocol_v2 import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_qwen_fusion_protocol_v2_is_frozen_and_valid() -> None:
    assert validate_protocol(_protocol()) == []


def test_qwen_fusion_protocol_rejects_hidden_evidence_access() -> None:
    payload = copy.deepcopy(_protocol())
    payload["evidence_boundary"]["confirmation_hidden_until_one_tuning_nominee"] = False
    assert any("hidden evidence boundary" in row for row in validate_protocol(payload))


def test_qwen_fusion_protocol_rejects_post_result_candidate_change() -> None:
    payload = copy.deepcopy(_protocol())
    payload["frozen_candidates"][1]["dense_novelty_quota"] = 3
    assert any("candidate set changed" in row for row in validate_protocol(payload))


def test_qwen_fusion_protocol_rejects_product_exposure() -> None:
    payload = copy.deepcopy(_protocol())
    payload["failure_policy"]["product_enabled"] = True
    assert any("product exposure" in row for row in validate_protocol(payload))
