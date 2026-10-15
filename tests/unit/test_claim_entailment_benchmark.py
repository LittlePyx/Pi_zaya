from __future__ import annotations

from kb.claim_entailment_benchmark import evaluate_binary_claim_support


class _Result:
    def __init__(self, supported: bool) -> None:
        self._supported = supported

    def to_dict(self) -> dict:
        return {
            "claims": [{
                "supported": self._supported,
                "matched_evidence_count": int(self._supported),
                "unsupported_reason": "" if self._supported else "citation_evidence_mismatch",
            }]
        }


def _fixture() -> dict:
    return {
        "evidence": [{"id": "ev", "segments": [{"quote": "The system stores prior state."}]}],
        "cases": [
            {"id": "yes", "evidence_id": "ev", "claim": "It stores state.", "label": "entailment"},
            {"id": "no", "evidence_id": "ev", "claim": "It deletes state.", "label": "contradiction"},
            {"id": "unknown", "evidence_id": "ev", "claim": "It uses 12 layers.", "label": "insufficient"},
        ],
    }


def test_binary_claim_support_scores_false_support_by_expected_semantics() -> None:
    result = evaluate_binary_claim_support(
        _fixture(),
        lambda answer, _hits: _Result("deletes" not in answer),
    )
    assert result["binary_metrics"] == {
        "accuracy": 0.666667,
        "entailment_recall": 1.0,
        "non_entailment_specificity": 0.5,
        "contradiction_false_support_rate": 0.0,
        "insufficient_false_support_rate": 1.0,
    }
    assert result["confusion"] == {
        "true_supported": 1,
        "false_rejected": 0,
        "false_supported": 1,
        "true_rejected": 1,
    }
