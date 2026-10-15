from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from kb.claim_entailment_eval_fixture import (
    claim_entailment_fixture_summary,
    validate_claim_entailment_fixture,
)
from kb.retrieval_eval_fixture import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "docs" / "claim_entailment_eval_v1.json"
QRELS_PATH = REPO_ROOT / "docs" / "retrieval_eval_unseen_v2.json"


def _payloads() -> tuple[dict, dict]:
    return (
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8")),
        json.loads(QRELS_PATH.read_text(encoding="utf-8")),
    )


def _errors(payload: dict, qrels: dict) -> list[str]:
    return validate_claim_entailment_fixture(
        payload,
        qrels,
        qrels_sha256=sha256_file(QRELS_PATH),
    )


def test_claim_entailment_fixture_is_balanced_traceable_and_model_independent() -> None:
    payload, qrels = _payloads()
    assert _errors(payload, qrels) == []
    summary = claim_entailment_fixture_summary(payload)
    assert summary["case_count"] == 24
    assert summary["label_counts"] == {
        "entailment": 8,
        "contradiction": 8,
        "insufficient": 8,
    }
    assert summary["locale_counts"] == {"en": 12, "zh": 12}
    assert summary["lexical_adversary_count"] == 4
    assert "model" not in payload
    assert "scores" not in payload


def test_claim_entailment_fixture_rejects_label_and_qrels_drift() -> None:
    payload, qrels = _payloads()
    changed_label = deepcopy(payload)
    changed_label["cases"][0]["label"] = "insufficient"
    assert "semantic label composition changed" in _errors(changed_label, qrels)

    changed_qrels = deepcopy(payload)
    changed_qrels["source_qrels"]["sha256"] = "0" * 64
    assert "source qrels SHA-256 does not match" in _errors(changed_qrels, qrels)


def test_claim_entailment_fixture_rejects_weak_lexical_adversary() -> None:
    payload, qrels = _payloads()
    weak = deepcopy(payload)
    row = next(case for case in weak["cases"] if case["adversary_type"].startswith("lexical_"))
    row["claim"] = "This unrelated claim has no support in the quoted evidence at all."
    row["claim_token_coverage"] = 0.0
    errors = _errors(weak, qrels)
    assert any("lexical adversary coverage is below" in error for error in errors)
