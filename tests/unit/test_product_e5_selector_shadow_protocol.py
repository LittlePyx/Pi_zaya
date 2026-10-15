from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_product_e5_selector_shadow_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_product_e5_selector_shadow_protocol_passes() -> None:
    assert validate_protocol(_protocol()) == []


def test_product_e5_selector_shadow_rejects_second_full_build() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["maximum_completed_full_builds"] = 2
    assert "product E5 selector execution boundary changed" in validate_protocol(changed)


def test_product_e5_selector_shadow_rejects_user_exposure() -> None:
    changed = copy.deepcopy(_protocol())
    changed["product_boundary"]["ordinary_user_surface_exposed"] = True
    assert "product E5 selector product boundary changed" in validate_protocol(changed)


def test_product_e5_selector_shadow_rejects_selector_tuning() -> None:
    changed = copy.deepcopy(_protocol())
    changed["selector_contract"]["dense_novelty_quota"] = 3
    assert "product E5 selector contract changed" in validate_protocol(changed)


def test_product_e5_selector_shadow_rejects_relaxed_quality_gate() -> None:
    changed = copy.deepcopy(_protocol())
    changed["quality_gates"]["mrr_no_regression_vs_product_each_mode"] = False
    assert "product E5 selector quality gates changed" in validate_protocol(changed)


def test_product_e5_selector_shadow_rejects_source_writes() -> None:
    changed = copy.deepcopy(_protocol())
    changed["source_corpus"]["source_writes_allowed"] = True
    assert "product E5 selector source corpus changed" in validate_protocol(changed)


def test_product_e5_selector_shadow_rejects_dependency_overlay_drift() -> None:
    changed = copy.deepcopy(_protocol())
    changed["runtime_environment"]["rank_bm25_version"] = "latest"
    assert "product E5 selector runtime environment changed" in validate_protocol(changed)
