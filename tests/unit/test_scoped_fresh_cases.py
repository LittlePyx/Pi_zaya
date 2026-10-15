from __future__ import annotations

import copy
import json

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.validate_scoped_fresh_cases import (
    DEFAULT_CHUNK_PROJECTION,
    DEFAULT_FIXTURE,
    EXPECTED_FIXTURE_SHA256,
    validate_fixture,
)


def _fixture() -> dict:
    return json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))


def _projected_chunks() -> list[dict]:
    payload = json.loads(DEFAULT_CHUNK_PROJECTION.read_text(encoding="utf-8"))
    return list(payload["chunks"])


def test_scoped_fresh_fixture_is_sealed_and_valid() -> None:
    assert sha256_file(DEFAULT_FIXTURE) == EXPECTED_FIXTURE_SHA256
    assert validate_fixture(_fixture()) == []


def test_scoped_fresh_fixture_accepts_public_source_projection() -> None:
    assert validate_fixture(_fixture(), chunks=_projected_chunks()) == []


def test_scoped_fresh_fixture_rejects_projection_text_drift() -> None:
    chunks = copy.deepcopy(_projected_chunks())
    chunks[0]["text_sha256"] = "0" * 64
    errors = validate_fixture(_fixture(), chunks=chunks)
    assert any("corpus fingerprint" in error for error in errors)


def test_scoped_fresh_fixture_rejects_locale_drift() -> None:
    changed = copy.deepcopy(_fixture())
    changed["cases"][0]["locale"] = "en"
    assert "fixture locale quota changed" in validate_fixture(changed)


def test_scoped_fresh_fixture_rejects_previous_evidence_overlap() -> None:
    changed = copy.deepcopy(_fixture())
    changed["cases"][0]["labels"][0]["chunk_id"] = "12d3dafe5e149b77:23"
    errors = validate_fixture(changed)
    assert any("evidence overlaps the previous journey suite" in error for error in errors)


def test_scoped_fresh_fixture_rejects_source_page_drift() -> None:
    changed = copy.deepcopy(_fixture())
    changed["cases"][0]["labels"][0]["page_start"] = 5
    errors = validate_fixture(changed)
    assert any("evidence page or source-text identity changed" in error for error in errors)


def test_scoped_fresh_fixture_rejects_retriever_authorship() -> None:
    changed = copy.deepcopy(_fixture())
    changed["retriever_executed_during_authoring_or_labeling"] = True
    assert (
        "retriever execution was introduced into source-only authoring"
        in validate_fixture(changed)
    )


def test_scoped_fresh_fixture_rejects_product_enablement() -> None:
    changed = copy.deepcopy(_fixture())
    changed["product_boundary"]["product_enabled"] = True
    assert "scoped fresh product boundary changed" in validate_fixture(changed)


def test_scoped_fresh_fixture_rejects_result_leakage() -> None:
    changed = copy.deepcopy(_fixture())
    changed["cases"][0]["metrics"] = {"recall@10": 1.0}
    assert (
        "retrieval result data leaked into the sealed authoring fixture"
        in validate_fixture(changed)
    )
