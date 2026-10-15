from __future__ import annotations

from tools.retrieval_eval.run_parent_collapsed_multivector_benchmark import (
    DEFAULT_FIXTURE,
    DEFAULT_REPORT,
    _validate_report,
    evaluate_development_gates,
)


def _channel(recall5, recall10, mrr, ndcg10, wrong5, p95):
    mode = {
        "metrics": {
            "recall@5": recall5,
            "recall@10": recall10,
            "mrr": mrr,
            "ndcg@10": ndcg10,
            "wrong_source_rate@5": wrong5,
        },
        "latency_ms": {"p95": p95},
    }
    return {"modes": {"scoped": mode, "full_library": mode}}


def test_development_gates_require_quality_identity_cost_and_latency() -> None:
    result = evaluate_development_gates(
        parent_dense=_channel(0.5, 0.5, 0.3, 0.35, 0.0, 20.0),
        multivector=_channel(0.55, 0.55, 0.32, 0.37, 0.0, 30.0),
        safe_rrf=_channel(0.6, 0.66, 0.4, 0.45, 0.0, 25.0),
        product_baseline=_channel(0.6, 0.62, 0.39, 0.44, 0.0, 2.0),
        representation_count_multiplier=1.8,
    )
    assert result["passes_all"] is True

    too_large = evaluate_development_gates(
        parent_dense=_channel(0.5, 0.5, 0.3, 0.35, 0.0, 20.0),
        multivector=_channel(0.55, 0.55, 0.32, 0.37, 0.0, 30.0),
        safe_rrf=_channel(0.6, 0.66, 0.4, 0.45, 0.0, 25.0),
        product_baseline=_channel(0.6, 0.62, 0.39, 0.44, 0.0, 2.0),
        representation_count_multiplier=2.01,
    )
    assert too_large["passes_all"] is False
    assert too_large["checks"]["representation_count_multiplier_pass"] is False


def test_committed_multivector_development_report_is_frozen_and_rejected() -> None:
    assert _validate_report(DEFAULT_REPORT, DEFAULT_FIXTURE) == []
