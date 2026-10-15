from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_scoped_fresh_authoring_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_scoped_fresh_authoring_protocol_passes() -> None:
    assert validate_protocol(_protocol()) == []


def test_scoped_fresh_authoring_rejects_seen_document() -> None:
    changed = copy.deepcopy(_protocol())
    changed["document_selection"]["document_ids"][0] = "perovskite-laser"
    assert "scoped fresh authoring document selection changed" in validate_protocol(changed)


def test_scoped_fresh_authoring_rejects_retriever_use() -> None:
    changed = copy.deepcopy(_protocol())
    changed["authoring_boundary"]["bm25_or_dense_retrieval_during_authoring_allowed"] = True
    assert "scoped fresh authoring boundary changed" in validate_protocol(changed)


def test_scoped_fresh_authoring_rejects_gate_relaxation() -> None:
    changed = copy.deepcopy(_protocol())
    changed["future_evaluation"]["maximum_case_recall_at_10_regressions"] = 1
    assert "scoped fresh authoring future gate changed" in validate_protocol(changed)


def test_scoped_fresh_authoring_never_enables_product() -> None:
    changed = copy.deepcopy(_protocol())
    changed["product_boundary"]["product_enabled"] = True
    assert "scoped fresh authoring product boundary changed" in validate_protocol(changed)
