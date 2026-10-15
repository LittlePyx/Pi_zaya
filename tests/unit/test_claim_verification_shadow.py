from __future__ import annotations

import json
from pathlib import Path

from kb.claim_verification_shadow import arbitrate_shadow_claim, evaluate_support_strategy
from tools.research_qa.build_claim_shadow_report import build_report


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "docs" / "claim_verification_shadow_design_v1.json"


def test_shadow_arbitration_requires_structure_and_entailment() -> None:
    bound = {"structurally_bound": True}
    assert arbitrate_shadow_claim(citation_structure=bound, nli_label="entailment") == {
        "outcome": "semantic_support",
        "shadow_supported": True,
    }
    assert arbitrate_shadow_claim(citation_structure=bound, nli_label="neutral")["outcome"] == (
        "semantic_withheld"
    )
    assert arbitrate_shadow_claim(
        citation_structure={"structurally_bound": False}, nli_label="entailment"
    )["outcome"] == "structural_reject"


def test_support_strategy_reports_false_support_separately() -> None:
    fixture = {
        "cases": [
            {"id": "e", "label": "entailment"},
            {"id": "c", "label": "contradiction"},
            {"id": "i", "label": "insufficient"},
        ]
    }
    result = evaluate_support_strategy(fixture, {"e": True, "c": True, "i": False})
    assert result["metrics"]["binary_support_accuracy"] == 0.666667
    assert result["metrics"]["contradiction_false_support_rate"] == 1.0
    assert result["false_supported_ids"] == ["c"]


def test_committed_shadow_report_is_deterministic_and_non_product() -> None:
    expected = build_report()
    assert json.loads(REPORT.read_text(encoding="utf-8")) == expected
    selected = expected["strategies"]["selected_structural_binding_plus_nli"]
    naive = expected["strategies"]["naive_lexical_and_nli"]
    assert selected["metrics"]["entailment_recall"] == 0.875
    assert selected["false_supported_ids"] == ["timesfm-contradicts-zh"]
    assert naive["metrics"]["entailment_recall"] == 0.375
    assert expected["decision"]["production_effect"] is False
    assert expected["decision"]["ordinary_user_status_exposed"] is False
