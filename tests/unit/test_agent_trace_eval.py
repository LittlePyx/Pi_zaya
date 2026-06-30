from tools.research_qa.run_agent_trace_eval import build_eval_report, evaluate_cases, evaluate_quality_cases


def test_agent_trace_eval_runs_on_golden_dataset():
    summary = evaluate_cases()

    assert summary["ok"] is True
    assert summary["case_count"] > 0
    assert summary["scope_context_present"] == summary["case_count"]


def test_agent_trace_eval_report_marks_unmeasured_metrics_null():
    summary = evaluate_cases()
    report = build_eval_report(summary, commit="test-commit", date="2026-01-01T00:00:00+00:00")

    assert report["commit"] == "test-commit"
    assert report["num_cases"] == summary["case_count"]
    assert report["planner_validation_ok"] is True
    assert report["retrieval_recall_at_5"] is None
    assert report["citation_precision"] is None
    assert report["claim_support_rate"] is None
    assert report["unsupported_claim_rate"] is None
    assert report["no_evidence_refusal_accuracy"] is None


def test_agent_quality_eval_runs_on_recorded_fixture():
    quality = evaluate_quality_cases()

    assert quality["ok"] is True, quality["errors"]
    assert quality["case_count"] >= 5
    assert quality["retrieval_hit_rate"] == 1.0
    assert quality["expected_source_hit_rate"] == 1.0
    assert quality["expected_answer_point_coverage"] == 1.0
    assert quality["no_evidence_refusal_accuracy"] == 1.0
    assert quality["external_fallback_disclosure_accuracy"] == 1.0
    assert quality["trace_clutter_free_rate"] == 1.0


def test_agent_trace_eval_report_includes_quality_metrics():
    summary = evaluate_cases()
    quality = evaluate_quality_cases()
    report = build_eval_report(
        summary,
        quality_summary=quality,
        commit="test-commit",
        date="2026-01-01T00:00:00+00:00",
    )

    assert report["quality_eval_ok"] is True
    assert report["num_quality_cases"] == quality["case_count"]
    assert report["retrieval_recall_at_5"] == 1.0
    assert report["claim_support_rate"] == 1.0
    assert report["unsupported_claim_rate"] == 0.0
    assert report["external_fallback_disclosure_accuracy"] == 1.0
    assert report["trace_clutter_free_rate"] == 1.0
