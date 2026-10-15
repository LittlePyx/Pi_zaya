from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any


NLI_LABELS = ("entailment", "neutral", "contradiction")


def evaluate_nli_predictions(
    fixture: Mapping[str, Any],
    predictions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate fixed three-way NLI predictions and the stricter support gate."""

    by_id = {str(row.get("id") or ""): row for row in predictions}
    rows: list[dict[str, Any]] = []
    for case in fixture.get("cases") or []:
        case_id = str(case.get("id") or "")
        prediction = by_id.get(case_id)
        if prediction is None:
            raise ValueError(f"missing NLI prediction for {case_id}")
        scores = {
            label: float((prediction.get("scores") or {}).get(label) or 0.0)
            for label in NLI_LABELS
        }
        observed = max(NLI_LABELS, key=lambda label: scores[label])
        fixture_label = str(case.get("label") or "")
        expected = "neutral" if fixture_label == "insufficient" else fixture_label
        expected_supported = expected == "entailment"
        observed_supported = observed == "entailment"
        rows.append(
            {
                "id": case_id,
                "fixture_label": fixture_label,
                "expected_nli_label": expected,
                "observed_nli_label": observed,
                "expected_supported": expected_supported,
                "observed_supported": observed_supported,
                "correct_three_way": observed == expected,
                "correct_binary_support": observed_supported == expected_supported,
                "locale": str(case.get("locale") or ""),
                "adversary_type": str(case.get("adversary_type") or ""),
                "scores": {label: round(scores[label], 6) for label in NLI_LABELS},
            }
        )

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 6) if denominator else 0.0

    total = len(rows)
    entailments = [row for row in rows if row["fixture_label"] == "entailment"]
    non_entailments = [row for row in rows if row["fixture_label"] != "entailment"]
    contradictions = [row for row in rows if row["fixture_label"] == "contradiction"]
    insufficient = [row for row in rows if row["fixture_label"] == "insufficient"]
    lexical = [
        row
        for row in contradictions
        if str(row["adversary_type"]).startswith("lexical_")
    ]
    cross_lingual = [row for row in rows if row["locale"] == "zh"]
    label_confusion = Counter(
        (str(row["expected_nli_label"]), str(row["observed_nli_label"])) for row in rows
    )
    return {
        "case_count": total,
        "metrics": {
            "three_way_accuracy": ratio(
                sum(1 for row in rows if row["correct_three_way"]), total
            ),
            "binary_support_accuracy": ratio(
                sum(1 for row in rows if row["correct_binary_support"]), total
            ),
            "entailment_recall": ratio(
                sum(1 for row in entailments if row["observed_supported"]),
                len(entailments),
            ),
            "non_entailment_specificity": ratio(
                sum(1 for row in non_entailments if not row["observed_supported"]),
                len(non_entailments),
            ),
            "contradiction_false_support_rate": ratio(
                sum(1 for row in contradictions if row["observed_supported"]),
                len(contradictions),
            ),
            "lexical_adversary_false_support_rate": ratio(
                sum(1 for row in lexical if row["observed_supported"]),
                len(lexical),
            ),
            "insufficient_false_support_rate": ratio(
                sum(1 for row in insufficient if row["observed_supported"]),
                len(insufficient),
            ),
            "cross_lingual_accuracy": ratio(
                sum(1 for row in cross_lingual if row["correct_three_way"]),
                len(cross_lingual),
            ),
        },
        "label_confusion": {
            expected: {
                observed: label_confusion[(expected, observed)] for observed in NLI_LABELS
            }
            for expected in NLI_LABELS
        },
        "cases": rows,
    }


def evaluate_nli_acceptance(
    result: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    truncated_pair_count: int,
) -> dict[str, Any]:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
    checks = {
        "three_way_accuracy": float(metrics.get("three_way_accuracy") or 0.0)
        >= float(thresholds.get("three_way_accuracy_min") or 0.0),
        "entailment_recall": float(metrics.get("entailment_recall") or 0.0)
        >= float(thresholds.get("entailment_recall_min") or 0.0),
        "non_entailment_specificity": float(
            metrics.get("non_entailment_specificity") or 0.0
        )
        >= float(thresholds.get("non_entailment_specificity_min") or 0.0),
        "contradiction_false_support_rate": float(
            metrics.get("contradiction_false_support_rate") or 0.0
        )
        <= float(thresholds.get("contradiction_false_support_rate_max") or 0.0),
        "lexical_adversary_false_support_rate": float(
            metrics.get("lexical_adversary_false_support_rate") or 0.0
        )
        <= float(thresholds.get("lexical_adversary_false_support_rate_max") or 0.0),
        "insufficient_false_support_rate": float(
            metrics.get("insufficient_false_support_rate") or 0.0
        )
        <= float(thresholds.get("insufficient_false_support_rate_max") or 0.0),
        "cross_lingual_accuracy": float(metrics.get("cross_lingual_accuracy") or 0.0)
        >= float(thresholds.get("cross_lingual_accuracy_min") or 0.0),
        "truncated_pair_count": int(truncated_pair_count)
        <= int(thresholds.get("truncated_pair_count_max") or 0),
    }
    return {"thresholds": dict(thresholds), "checks": checks, "passes_all": all(checks.values())}
