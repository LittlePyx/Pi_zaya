from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from tools.research_qa.run_claim_nli_benchmark import validate_report


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "docs" / "claim_entailment_eval_v1.json"
REGISTRY_PATH = REPO_ROOT / "docs" / "claim_entailment_nli_candidate_v1.json"
REPORT_PATH = REPO_ROOT / "docs" / "claim_entailment_nli_baseline_v1.json"


def _inputs() -> tuple[dict, dict, dict]:
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (FIXTURE_PATH, REGISTRY_PATH, REPORT_PATH)
    )


def test_frozen_nli_report_passes_only_for_shadow_design() -> None:
    fixture, registry, report = _inputs()
    assert validate_report(report, FIXTURE_PATH, REGISTRY_PATH, fixture, registry) == []
    assert report["result"]["metrics"] == {
        "three_way_accuracy": 0.791667,
        "binary_support_accuracy": 0.916667,
        "entailment_recall": 0.875,
        "non_entailment_specificity": 0.9375,
        "contradiction_false_support_rate": 0.125,
        "lexical_adversary_false_support_rate": 0.0,
        "insufficient_false_support_rate": 0.0,
        "cross_lingual_accuracy": 0.916667,
    }
    assert report["acceptance"]["passes_all"] is True
    assert report["decision"] == {
        "status": "eligible_for_shadow_design",
        "product_enabled": False,
        "reason": "All pre-registered gates passed; this permits only a shadow arbitration design.",
    }
    false_support = [
        row["id"]
        for row in report["result"]["cases"]
        if not row["expected_supported"] and row["observed_supported"]
    ]
    assert false_support == ["timesfm-contradicts-zh"]


def test_nli_report_rejects_score_or_activation_drift() -> None:
    fixture, registry, report = _inputs()
    changed = deepcopy(report)
    changed["result"]["cases"][0]["scores"] = {
        "entailment": 0.0,
        "neutral": 1.0,
        "contradiction": 0.0,
    }
    assert any(
        "metrics or case decisions are stale" in error
        for error in validate_report(changed, FIXTURE_PATH, REGISTRY_PATH, fixture, registry)
    )

    enabled = deepcopy(report)
    enabled["candidate"]["product_enabled"] = True
    assert any(
        "product-disabled" in error
        for error in validate_report(enabled, FIXTURE_PATH, REGISTRY_PATH, fixture, registry)
    )
