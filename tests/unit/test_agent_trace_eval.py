from tools.research_qa.run_agent_trace_eval import evaluate_cases


def test_agent_trace_eval_runs_on_golden_dataset():
    summary = evaluate_cases()

    assert summary["ok"] is True
    assert summary["case_count"] > 0
    assert summary["scope_context_present"] == summary["case_count"]
