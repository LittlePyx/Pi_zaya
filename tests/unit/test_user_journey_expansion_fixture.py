from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_user_journey_expansion_fixture import (
    DEFAULT_OUTPUT,
    validate_fixture,
)


def _fixture() -> dict:
    return json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))


def test_user_journey_qrels_are_frozen_before_retrieval() -> None:
    assert validate_fixture(_fixture()) == []


def test_user_journey_qrels_reject_modified_evidence_hash() -> None:
    payload = copy.deepcopy(_fixture())
    payload["cases"][0]["labels"][0]["text_sha256"] = "0" * 64
    errors = validate_fixture(payload)
    assert any("local-source projection" in error for error in errors)


def test_user_journey_qrels_reject_retrieval_rank_leakage() -> None:
    payload = copy.deepcopy(_fixture())
    payload["cases"][0]["labels"][0]["rank"] = 1
    errors = validate_fixture(payload)
    assert any("local-source projection" in error for error in errors)


def test_user_journey_qrels_keep_holdout_out_of_tuning() -> None:
    payload = copy.deepcopy(_fixture())
    payload["split"]["holdout_allowed_for_tuning"] = True
    errors = validate_fixture(payload)
    assert any("local-source projection" in error for error in errors)


def test_user_journey_cross_paper_qrels_cover_each_scoped_document() -> None:
    payload = copy.deepcopy(_fixture())
    cross = next(
        case
        for case in payload["cases"]
        if case["evaluation_focus"] == "cross_paper_evidence_comparison"
    )
    removed_doc = cross["labels"][0]["doc_id"]
    cross["labels"] = [
        label for label in cross["labels"] if label["doc_id"] != removed_doc
    ]
    errors = validate_fixture(payload)
    assert any("local-source projection" in error for error in errors)
