from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_user_journey_expansion_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_user_journey_expansion_protocol_is_frozen_and_valid() -> None:
    assert validate_protocol(_protocol()) == []


def test_user_journey_protocol_rejects_writing_scope() -> None:
    payload = copy.deepcopy(_protocol())
    payload["product_boundary"]["answer_generation_evaluated"] = True
    assert any("reading-only" in error for error in validate_protocol(payload))


def test_user_journey_protocol_rejects_external_label_knowledge() -> None:
    payload = copy.deepcopy(_protocol())
    payload["product_boundary"]["external_web_or_model_knowledge_for_labels"] = True
    assert any("reading-only" in error for error in validate_protocol(payload))


def test_user_journey_protocol_rejects_holdout_weakening() -> None:
    payload = copy.deepcopy(_protocol())
    payload["deterministic_split"]["holdout_allowed_for_tuning"] = True
    assert any("holdout" in error for error in validate_protocol(payload))
