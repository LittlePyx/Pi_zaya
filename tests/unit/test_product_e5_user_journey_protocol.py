from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_product_e5_user_journey_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_product_e5_user_journey_protocol_passes() -> None:
    assert validate_protocol(_protocol()) == []


def test_product_e5_user_journey_rejects_second_build() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["maximum_completed_full_builds"] = 2
    assert "product E5 user-journey execution boundary changed" in validate_protocol(changed)


def test_product_e5_user_journey_rejects_new_holdout_claim() -> None:
    changed = copy.deepcopy(_protocol())
    changed["evaluation_status"]["new_untouched_holdout_claim_allowed"] = True
    assert "product E5 user-journey evaluation status changed" in validate_protocol(changed)


def test_product_e5_user_journey_rejects_candidate_tuning() -> None:
    changed = copy.deepcopy(_protocol())
    changed["candidate"]["configuration_changes_after_selector_shadow"] = 1
    assert "product E5 user-journey candidate changed" in validate_protocol(changed)


def test_product_e5_user_journey_rejects_wrong_source_regression() -> None:
    changed = copy.deepcopy(_protocol())
    changed["quality_gates"]["wrong_source_rate_no_regression_each_mode"].pop()
    assert "product E5 user-journey quality gates changed" in validate_protocol(changed)


def test_product_e5_user_journey_rejects_case_harm_allowance() -> None:
    changed = copy.deepcopy(_protocol())
    changed["quality_gates"]["maximum_case_regressions_each_mode"]["recall@10"] = 1
    assert "product E5 user-journey quality gates changed" in validate_protocol(changed)


def test_product_e5_user_journey_rejects_user_surface() -> None:
    changed = copy.deepcopy(_protocol())
    changed["product_boundary"]["ordinary_user_surface_exposed"] = True
    assert "product E5 user-journey product boundary changed" in validate_protocol(changed)
