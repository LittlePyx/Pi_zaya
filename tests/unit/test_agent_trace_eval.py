from tools.research_qa.run_agent_trace_eval import build_eval_report, evaluate_cases


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
