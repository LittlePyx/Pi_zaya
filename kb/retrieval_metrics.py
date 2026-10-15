from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = {str(item) for item in relevant_ids}
    if not relevant:
        return 0.0
    retrieved = {str(item) for item in ranked_ids[: max(0, int(k))]}
    return len(relevant & retrieved) / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = {str(item) for item in relevant_ids}
    for rank, item in enumerate(ranked_ids, start=1):
        if str(item) in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    limit = max(0, int(k))
    if limit <= 0 or not relevance:
        return 0.0

    def gain(score: float, rank: int) -> float:
        return (math.pow(2.0, max(0.0, float(score))) - 1.0) / math.log2(rank + 1.0)

    dcg = sum(
        gain(float(relevance.get(str(item), 0.0) or 0.0), rank)
        for rank, item in enumerate(ranked_ids[:limit], start=1)
    )
    ideal_scores = sorted((max(0.0, float(score)) for score in relevance.values()), reverse=True)[:limit]
    ideal_dcg = sum(gain(score, rank) for rank, score in enumerate(ideal_scores, start=1))
    return dcg / ideal_dcg if ideal_dcg > 0.0 else 0.0


def evaluate_ranking(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, float],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    relevant_ids = {str(item) for item, score in relevance.items() if float(score) > 0.0}
    metrics: dict[str, float] = {"mrr": reciprocal_rank(ranked_ids, relevant_ids)}
    for raw_k in ks:
        k = max(1, int(raw_k))
        metrics[f"recall@{k}"] = recall_at_k(ranked_ids, relevant_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, relevance, k)
    return metrics


def mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({str(key) for row in rows for key in row})
    if not rows:
        return {key: 0.0 for key in keys}
    return {
        key: sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows)
        for key in keys
    }
