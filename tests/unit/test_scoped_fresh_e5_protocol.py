from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_scoped_fresh_e5_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_scoped_fresh_e5_protocol_passes() -> None:
    assert validate_protocol(_protocol()) == []


def test_scoped_fresh_e5_protocol_rejects_full_library_mode() -> None:
    changed = copy.deepcopy(_protocol())
    changed["execution"]["modes"].append("full_library")
    assert "scoped fresh E5 execution boundary changed" in validate_protocol(
        changed,
        inspect_files=False,
    )


def test_scoped_fresh_e5_protocol_rejects_index_rebuild() -> None:
    changed = copy.deepcopy(_protocol())
    changed["existing_corpus_and_index"]["index_rebuild_allowed"] = True
    assert "scoped fresh existing corpus/index contract changed" in validate_protocol(
        changed,
        inspect_files=False,
    )


def test_scoped_fresh_e5_protocol_rejects_candidate_change() -> None:
    changed = copy.deepcopy(_protocol())
    changed["candidate"]["selector_configuration"]["protected_sparse_prefix"] = 4
    assert "scoped fresh E5 candidate changed" in validate_protocol(
        changed,
        inspect_files=False,
    )


def test_scoped_fresh_e5_protocol_rejects_quality_relaxation() -> None:
    changed = copy.deepcopy(_protocol())
    changed["quality_gates"]["maximum_case_regressions"]["mrr"] = 1
    assert "scoped fresh E5 quality gates changed" in validate_protocol(
        changed,
        inspect_files=False,
    )


def test_scoped_fresh_e5_protocol_never_enables_product() -> None:
    changed = copy.deepcopy(_protocol())
    changed["decision"]["product_enablement_allowed"] = True
    assert "scoped fresh E5 decision changed" in validate_protocol(
        changed,
        inspect_files=False,
    )
