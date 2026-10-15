from __future__ import annotations

import json

from kb.retrieval_eval_fixture import sha256_file
from tools.retrieval_eval.analyze_cpu_padding import (
    DEFAULT_REPORT,
    EXPECTED_REPORT_SHA256,
    EXPECTED_RESULT_FINGERPRINT,
    _load_protocol,
    compare_strategies,
    evaluate_branch,
    make_batches,
    summarize_batches,
    validate_report,
)


def _strategy(*, budget: int | None) -> dict:
    return {
        "id": "test",
        "sort_within_document": True,
        "maximum_batch_size": 8,
        "maximum_padded_tokens_per_batch": budget,
        "oversize_singleton": budget is not None,
    }


def test_token_budget_isolates_oversize_long_sequence() -> None:
    rows = [(f"chunk-{index}", 100) for index in range(7)] + [("long", 4712)]
    batches = make_batches(rows, _strategy(budget=4096))
    assert [len(batch) for batch in batches] == [7, 1]
    assert batches[-1] == [("long", 4712)]


def test_batch_summary_counts_padding_and_attention_work() -> None:
    summary = summarize_batches(
        {"doc": [("short", 100), ("long", 1000)]},
        {
            "id": "fixed",
            "sort_within_document": False,
            "maximum_batch_size": 8,
            "maximum_padded_tokens_per_batch": None,
            "oversize_singleton": False,
        },
    )
    assert summary["effective_token_count"] == 1100
    assert summary["padded_token_count"] == 2000
    assert summary["attention_work_proxy"] == 2_000_000
    assert summary["peak_batch_padded_tokens"] == 2000


def test_comparison_and_branch_use_predeclared_absolute_projection() -> None:
    baseline = {
        "strategy": {"id": "baseline"},
        "padded_token_count": 1000,
        "attention_work_proxy": 10000,
        "peak_batch_padded_tokens": 8000,
    }
    candidate = {
        "strategy": {"id": "candidate"},
        "padded_token_count": 400,
        "attention_work_proxy": 2000,
        "peak_batch_padded_tokens": 3000,
    }
    comparison = compare_strategies(baseline, candidate, parent_build_ms=4_000_000.0)
    gate = {
        "minimum_attention_work_proxy_speedup": 4.0,
        "minimum_peak_padded_tokens_reduction": 0.5,
        "maximum_projected_build_ms_for_bounded_followup": 900000.0,
        "pass_decision": "sample",
        "reject_decision": "lighter",
    }
    assert comparison["attention_work_proxy_speedup"] == 5.0
    assert comparison["projected_build_ms_from_attention_proxy"] == 800000.0
    assert evaluate_branch(comparison, gate)["status"] == "sample"


def test_cpu_padding_outcome_is_valid_rejected_and_pinned() -> None:
    protocol = _load_protocol()
    report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    assert validate_report(report, protocol=protocol) == []
    assert report["result_fingerprint"] == EXPECTED_RESULT_FINGERPRINT
    assert sha256_file(DEFAULT_REPORT) == EXPECTED_REPORT_SHA256
    assert report["decision"] == {
        "checks": {
            "attention_work_proxy_speedup": False,
            "peak_padded_tokens_reduction": True,
            "projected_build_ms": False,
        },
        "passes_all": False,
        "status": "skip_native_qwen_and_screen_lighter_runtime",
        "full_build_authorized": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }
