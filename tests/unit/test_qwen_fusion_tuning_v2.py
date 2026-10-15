from __future__ import annotations

import json

from tools.retrieval_eval.run_qwen_fusion_tuning_v2 import (
    PRODUCT_CHANNEL,
    evaluate_tuning_gate,
    select_nominee,
)
from tools.retrieval_eval.validate_qwen_fusion_protocol_v2 import DEFAULT_PROTOCOL


def _channel(
    recall5: float, recall10: float, mrr: float, ndcg: float, wrong5: float
) -> dict:
    mode = {
        "metrics": {
            "recall@5": recall5,
            "recall@10": recall10,
            "mrr": mrr,
            "ndcg@10": ndcg,
            "wrong_source_rate@5": wrong5,
        }
    }
    return {"modes": {"scoped": mode, "full_library": mode}}


def test_tuning_gate_requires_prefix_safety_no_regression_and_strict_gain() -> None:
    product = _channel(0.6, 0.7, 0.4, 0.5, 0.1)
    passing = evaluate_tuning_gate(
        product=product,
        candidate=_channel(0.6, 0.72, 0.41, 0.51, 0.1),
    )
    assert passing["passes_all"] is True

    equal_but_not_better = evaluate_tuning_gate(product=product, candidate=product)
    assert equal_but_not_better["checks"]["strict_recall_at_10_gain_any_mode"] is False
    assert equal_but_not_better["passes_all"] is False

    regression = evaluate_tuning_gate(
        product=product,
        candidate=_channel(0.6, 0.72, 0.39, 0.51, 0.1),
    )
    assert regression["checks"]["mrr_no_regression_both_modes"] is False
    assert regression["passes_all"] is False


def test_tuning_nomination_is_eligible_only_and_prefers_safer_quota_tie() -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    product = _channel(0.6, 0.7, 0.4, 0.5, 0.1)
    candidate = _channel(0.6, 0.72, 0.41, 0.51, 0.1)
    channels = {
        PRODUCT_CHANNEL: product,
        "ungated_rrf_2_1_k20_v2": candidate,
        "protected_tail_dense_quota_1_v2": candidate,
        "protected_tail_dense_quota_2_v2": candidate,
    }
    gates = {
        "ungated_rrf_2_1_k20_v2": {"passes_all": False},
        "protected_tail_dense_quota_1_v2": {"passes_all": True},
        "protected_tail_dense_quota_2_v2": {"passes_all": True},
    }

    decision = select_nominee(channels=channels, gates=gates, protocol=protocol)

    assert decision["eligible_candidate_ids"] == [
        "protected_tail_dense_quota_1_v2",
        "protected_tail_dense_quota_2_v2",
    ]
    assert decision["nominee_candidate_id"] == "protected_tail_dense_quota_1_v2"
    assert decision["confirmation_allowed"] is True
