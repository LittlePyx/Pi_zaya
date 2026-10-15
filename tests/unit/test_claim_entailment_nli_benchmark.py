from __future__ import annotations

from kb.claim_entailment_nli_benchmark import (
    evaluate_nli_acceptance,
    evaluate_nli_predictions,
)


def _fixture() -> dict:
    return {
        "cases": [
            {"id": "e", "label": "entailment", "locale": "en", "adversary_type": "none"},
            {
                "id": "c",
                "label": "contradiction",
                "locale": "zh",
                "adversary_type": "lexical_negation",
            },
            {"id": "i", "label": "insufficient", "locale": "zh", "adversary_type": "none"},
        ]
    }


def test_nli_benchmark_keeps_three_way_and_binary_support_distinct() -> None:
    result = evaluate_nli_predictions(
        _fixture(),
        [
            {"id": "e", "scores": {"entailment": 0.8, "neutral": 0.1, "contradiction": 0.1}},
            {"id": "c", "scores": {"entailment": 0.1, "neutral": 0.2, "contradiction": 0.7}},
            {"id": "i", "scores": {"entailment": 0.1, "neutral": 0.2, "contradiction": 0.7}},
        ],
    )
    assert result["metrics"]["three_way_accuracy"] == 0.666667
    assert result["metrics"]["binary_support_accuracy"] == 1.0
    assert result["metrics"]["contradiction_false_support_rate"] == 0.0


def test_nli_acceptance_requires_every_pre_registered_check() -> None:
    result = {
        "metrics": {
            "three_way_accuracy": 0.8,
            "entailment_recall": 0.75,
            "non_entailment_specificity": 0.9,
            "contradiction_false_support_rate": 0.0,
            "lexical_adversary_false_support_rate": 0.0,
            "insufficient_false_support_rate": 0.1,
            "cross_lingual_accuracy": 0.8,
        }
    }
    thresholds = {
        "three_way_accuracy_min": 0.75,
        "entailment_recall_min": 0.75,
        "non_entailment_specificity_min": 0.8125,
        "contradiction_false_support_rate_max": 0.125,
        "lexical_adversary_false_support_rate_max": 0.0,
        "insufficient_false_support_rate_max": 0.25,
        "cross_lingual_accuracy_min": 0.7,
        "truncated_pair_count_max": 0,
        "all_required": True,
    }
    assert evaluate_nli_acceptance(result, thresholds, truncated_pair_count=0)["passes_all"]
    assert not evaluate_nli_acceptance(result, thresholds, truncated_pair_count=1)["passes_all"]
