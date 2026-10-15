from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_lighter_runtime_screen_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_lighter_runtime_protocol_is_frozen_before_download() -> None:
    assert validate_protocol(_protocol()) == []


def test_lighter_runtime_protocol_rejects_candidate_or_supply_chain_drift() -> None:
    changed = copy.deepcopy(_protocol())
    changed["candidates"][0]["primary_artifact"]["sha256"] = "0" * 64
    changed["candidate_budget"]["maximum_candidates"] = 3
    errors = validate_protocol(changed)
    assert any("frozen identity changed" in error for error in errors)
    assert "lighter runtime candidate budget changed" in errors


def test_lighter_runtime_protocol_rejects_dependency_expansion() -> None:
    changed = copy.deepcopy(_protocol())
    changed["runtime_contract"]["torch_allowed"] = True
    assert "lighter runtime dependency contract changed" in validate_protocol(changed)


def test_lighter_runtime_protocol_rejects_unbounded_or_repeated_builds() -> None:
    changed = copy.deepcopy(_protocol())
    changed["stage_1_bounded_screen"]["maximum_wall_clock_seconds_per_candidate"] = 3600
    changed["stage_2_single_winner"]["maximum_full_builds_across_batch"] = 2
    errors = validate_protocol(changed)
    assert "lighter runtime stage 1 gates changed" in errors
    assert "lighter runtime stage 2 gates changed" in errors


def test_lighter_runtime_protocol_rejects_product_surface_exposure() -> None:
    changed = copy.deepcopy(_protocol())
    changed["product_boundary"]["product_enabled"] = True
    changed["execution"]["ordinary_user_surface_exposed"] = True
    errors = validate_protocol(changed)
    assert "lighter runtime product boundary changed" in errors
    assert "lighter runtime execution boundary changed" in errors
