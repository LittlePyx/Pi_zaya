from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_e5_small_window_census_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_e5_small_window_census_protocol_passes() -> None:
    assert validate_protocol(_protocol()) == []


def test_e5_small_window_census_protocol_rejects_tail_loss() -> None:
    changed = copy.deepcopy(_protocol())
    changed["representation"]["silent_tail_loss_allowed"] = True
    assert "E5-small window-census representation changed" in validate_protocol(changed)


def test_e5_small_window_census_protocol_rejects_model_inference() -> None:
    changed = copy.deepcopy(_protocol())
    changed["measurement"]["model_inference_allowed"] = True
    assert "E5-small window-census execution boundary changed" in validate_protocol(changed)


def test_e5_small_window_census_protocol_rejects_window_tuning() -> None:
    changed = copy.deepcopy(_protocol())
    changed["representation"]["content_window_tokens"] = 510
    assert "E5-small window-census representation changed" in validate_protocol(changed)
