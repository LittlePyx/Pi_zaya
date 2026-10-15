from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_portable_e5_product_startup_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_portable_e5_product_startup_protocol_passes() -> None:
    assert validate_protocol(_protocol()) == []


def test_portable_e5_product_startup_preserves_v1_rejection() -> None:
    changed = copy.deepcopy(_protocol())
    changed["source_v1"]["verdict_may_be_changed"] = True
    assert "portable E5 product-startup v1 rejection lineage changed" in validate_protocol(
        changed
    )


def test_portable_e5_product_startup_rejects_timing_relaxation() -> None:
    changed = copy.deepcopy(_protocol())
    changed["gates"]["maximum_product_import_ms"] = 15_500.0
    assert "portable E5 product-startup gates changed" in validate_protocol(changed)


def test_portable_e5_product_startup_rejects_ranking_change() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["expected_queries"][0]["ranked_chunk_ids"].reverse()
    assert "portable E5 product-startup expected ranking changed" in validate_protocol(
        changed
    )


def test_portable_e5_product_startup_rejects_evaluation_import_in_timer() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["evaluation_helper_imported_before_startup_timing"] = True
    assert "portable E5 product-startup execution boundary changed" in validate_protocol(
        changed
    )


def test_portable_e5_product_startup_never_enables_product() -> None:
    changed = copy.deepcopy(_protocol())
    changed["decision"]["product_enablement_allowed"] = True
    assert "portable E5 product-startup decision changed" in validate_protocol(changed)
