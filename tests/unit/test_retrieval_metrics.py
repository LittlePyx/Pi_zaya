from __future__ import annotations

import pytest

from kb.retrieval_metrics import evaluate_ranking, mean_metrics, ndcg_at_k, recall_at_k, reciprocal_rank


def test_retrieval_metrics_cover_binary_and_graded_relevance() -> None:
    ranked = ["noise", "relevant-low", "relevant-high"]
    relevance = {"relevant-high": 3.0, "relevant-low": 1.0}

    assert recall_at_k(ranked, relevance, 1) == 0.0
    assert recall_at_k(ranked, relevance, 3) == 1.0
    assert reciprocal_rank(ranked, relevance) == 0.5
    assert 0.0 < ndcg_at_k(ranked, relevance, 3) < 1.0
    metrics = evaluate_ranking(ranked, relevance, ks=(1, 3))
    assert metrics["mrr"] == 0.5
    assert metrics["recall@3"] == 1.0


def test_mean_metrics_uses_zero_for_missing_case_metric() -> None:
    result = mean_metrics([{"mrr": 1.0, "recall@5": 1.0}, {"mrr": 0.5}])

    assert result["mrr"] == pytest.approx(0.75)
    assert result["recall@5"] == pytest.approx(0.5)
