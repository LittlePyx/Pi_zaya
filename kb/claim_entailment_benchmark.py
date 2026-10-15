from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any


Verifier = Callable[[str, list[dict[str, Any]]], Any]


def evaluate_binary_claim_support(
    fixture: Mapping[str, Any],
    verifier: Verifier,
) -> dict[str, Any]:
    """Measure whether a verifier accepts only entailed claims.

    This binary contract is intentionally stricter than the current verifier's
    own status names: contradiction and insufficient evidence must both be
    rejected. A semantic model may later report the three-way label as an
    additional metric, but it cannot hide false support behind aggregation.
    """
    evidence = {
        str(row.get("id") or ""): " ".join(
            str(segment.get("quote") or "").strip()
            for segment in list(row.get("segments") or [])
            if isinstance(segment, Mapping)
        )
        for row in list(fixture.get("evidence") or [])
        if isinstance(row, Mapping)
    }
    rows: list[dict[str, Any]] = []
    confusion = Counter()
    expected_counts = Counter()
    for raw_case in list(fixture.get("cases") or []):
        if not isinstance(raw_case, Mapping):
            continue
        case_id = str(raw_case.get("id") or "")
        expected_label = str(raw_case.get("label") or "")
        expected_supported = expected_label == "entailment"
        claim = str(raw_case.get("claim") or "").strip()
        evidence_text = evidence.get(str(raw_case.get("evidence_id") or ""), "")
        cited_claim = re.sub(r"([.!?。！？])$", r" [1]\1", claim) if claim else ""
        if "[1]" not in cited_claim:
            cited_claim = f"{claim} [1]"
        result = verifier(
            cited_claim,
            [{"id": "fixture-evidence", "text": evidence_text, "score": 1.0, "meta": {}}],
        )
        result_dict = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
        claims = [
            row for row in list(result_dict.get("claims") or []) if isinstance(row, Mapping)
        ]
        observed_supported = bool(claims and claims[0].get("supported"))
        key = (
            "expected_supported" if expected_supported else "expected_rejected",
            "observed_supported" if observed_supported else "observed_rejected",
        )
        confusion[key] += 1
        expected_counts[expected_label] += 1
        rows.append(
            {
                "id": case_id,
                "expected_label": expected_label,
                "expected_supported": expected_supported,
                "observed_supported": observed_supported,
                "correct": observed_supported == expected_supported,
                "matched_evidence_count": int(
                    claims[0].get("matched_evidence_count") or 0 if claims else 0
                ),
                "unsupported_reason": str(
                    claims[0].get("unsupported_reason") or "" if claims else "no_claim"
                ),
            }
        )

    total = len(rows)
    true_positive = confusion[("expected_supported", "observed_supported")]
    false_negative = confusion[("expected_supported", "observed_rejected")]
    false_positive = confusion[("expected_rejected", "observed_supported")]
    true_negative = confusion[("expected_rejected", "observed_rejected")]
    entailment_count = expected_counts["entailment"]
    contradiction_rows = [row for row in rows if row["expected_label"] == "contradiction"]
    insufficient_rows = [row for row in rows if row["expected_label"] == "insufficient"]

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 6) if denominator else 0.0

    return {
        "case_count": total,
        "expected_label_counts": dict(expected_counts),
        "binary_metrics": {
            "accuracy": ratio(true_positive + true_negative, total),
            "entailment_recall": ratio(true_positive, entailment_count),
            "non_entailment_specificity": ratio(
                true_negative,
                false_positive + true_negative,
            ),
            "contradiction_false_support_rate": ratio(
                sum(1 for row in contradiction_rows if row["observed_supported"]),
                len(contradiction_rows),
            ),
            "insufficient_false_support_rate": ratio(
                sum(1 for row in insufficient_rows if row["observed_supported"]),
                len(insufficient_rows),
            ),
        },
        "confusion": {
            "true_supported": true_positive,
            "false_rejected": false_negative,
            "false_supported": false_positive,
            "true_rejected": true_negative,
        },
        "cases": rows,
    }
