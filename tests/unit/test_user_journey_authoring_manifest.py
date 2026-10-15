from __future__ import annotations

import copy
import json

from tools.retrieval_eval.validate_user_journey_authoring_manifest import (
    DEFAULT_MANIFEST,
    validate_manifest,
)


def _manifest() -> dict:
    return json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))


def test_user_journey_questions_are_frozen_and_balanced() -> None:
    assert validate_manifest(_manifest()) == []


def test_user_journey_manifest_rejects_retriever_assisted_authoring() -> None:
    payload = copy.deepcopy(_manifest())
    payload["retriever_executed"] = True
    assert any("rank-independent" in error for error in validate_manifest(payload))


def test_user_journey_manifest_rejects_writing_requests() -> None:
    payload = copy.deepcopy(_manifest())
    payload["cases"][0]["question"] = "Draft a manuscript paragraph from this paper."
    assert any("writing-assistance" in error for error in validate_manifest(payload))


def test_user_journey_manifest_rejects_cross_paper_scope_collapse() -> None:
    payload = copy.deepcopy(_manifest())
    cross = next(
        case
        for case in payload["cases"]
        if case["journey_type"] == "cross_paper_evidence_comparison"
    )
    cross["scope_doc_ids"] = [cross["primary_doc_id"]]
    assert any("journey type" in error for error in validate_manifest(payload))
