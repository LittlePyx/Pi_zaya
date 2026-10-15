from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def arbitrate_shadow_claim(
    *,
    citation_structure: Mapping[str, Any],
    nli_label: str,
) -> dict[str, Any]:
    """Return a non-product shadow outcome without exposing model diagnostics."""

    if not bool(citation_structure.get("structurally_bound")):
        return {"outcome": "structural_reject", "shadow_supported": False}
    if nli_label == "entailment":
        return {"outcome": "semantic_support", "shadow_supported": True}
    if nli_label == "contradiction":
        return {"outcome": "semantic_contradiction", "shadow_supported": False}
    if nli_label == "neutral":
        return {"outcome": "semantic_withheld", "shadow_supported": False}
    return {"outcome": "semantic_unavailable", "shadow_supported": False}


def evaluate_support_strategy(
    fixture: Mapping[str, Any],
    observed_support: Mapping[str, bool],
) -> dict[str, Any]:
    rows = []
    for case in fixture.get("cases") or []:
        case_id = str(case.get("id") or "")
        label = str(case.get("label") or "")
        expected = label == "entailment"
        observed = bool(observed_support.get(case_id, False))
        rows.append(
            {
                "id": case_id,
                "label": label,
                "expected_supported": expected,
                "observed_supported": observed,
                "correct": expected == observed,
            }
        )

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 6) if denominator else 0.0

    entailments = [row for row in rows if row["label"] == "entailment"]
    non_entailments = [row for row in rows if row["label"] != "entailment"]
    contradictions = [row for row in rows if row["label"] == "contradiction"]
    insufficient = [row for row in rows if row["label"] == "insufficient"]
    return {
        "case_count": len(rows),
        "metrics": {
            "binary_support_accuracy": ratio(
                sum(1 for row in rows if row["correct"]), len(rows)
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
            "insufficient_false_support_rate": ratio(
                sum(1 for row in insufficient if row["observed_supported"]),
                len(insufficient),
            ),
        },
        "false_supported_ids": [
            row["id"]
            for row in rows
            if not row["expected_supported"] and row["observed_supported"]
        ],
        "false_rejected_ids": [
            row["id"]
            for row in rows
            if row["expected_supported"] and not row["observed_supported"]
        ],
    }
