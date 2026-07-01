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
    assert report["source_blend_accuracy"] is None
    assert report["unnecessary_notice_rate"] is None
    assert report["required_notice_accuracy"] is None


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
    assert quality["source_blend_accuracy"] == 1.0
    assert quality["source_blend_expected_count"] == quality["case_count"]
    assert quality["unnecessary_notice_rate"] == 0.0
    assert quality["required_notice_accuracy"] == 1.0
    assert quality["quality_gate_observed_count"] == 0
    assert quality["quality_gate_passed_rate"] is None
    assert quality["quality_gate_repaired_rate"] is None
    assert quality["quality_gate_fallback_rate"] is None


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
    assert report["source_blend_accuracy"] == 1.0
    assert report["source_blend_expected_count"] == quality["case_count"]
    assert report["unnecessary_notice_rate"] == 0.0
    assert report["required_notice_accuracy"] == 1.0
    assert report["quality_gate_observed_count"] == 0
    assert report["quality_gate_passed_rate"] is None


def test_agent_trace_eval_counts_recorded_quality_gate_statuses(tmp_path):
    fixture = tmp_path / "quality.jsonl"
    fixture.write_text(
        "\n".join(
            [
                '{"id":"passed","query":"What method?","answer_mode":"evidence_grounded","source_blend":"local_grounded","expected_source_blend":"local_grounded","answer":"Paper A uses retrieval [1].","evidence_hits":[{"text":"Paper A uses retrieval.","score":3,"meta":{"source_path":"paper-a.md"}}],"expected_retrieval_hit":true,"should_use_local_evidence":true,"external_fallback_allowed":false,"expected_answer_points":["retrieval"],"expected_user_notice":"none","quality_gate_status":"passed"}',
                '{"id":"fallback","query":"What limitation?","answer_mode":"evidence_grounded","source_blend":"local_grounded","expected_source_blend":"local_grounded","answer":"Paper A reports a limitation [1].","evidence_hits":[{"text":"Paper A reports a limitation.","score":3,"meta":{"source_path":"paper-a.md"}}],"expected_retrieval_hit":true,"should_use_local_evidence":true,"external_fallback_allowed":false,"expected_answer_points":["limitation"],"expected_user_notice":"none","agent_trace":{"summary":{"quality_gate_status":"fallback"}}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    quality = evaluate_quality_cases(fixture)

    assert quality["ok"] is True, quality["errors"]
    assert quality["quality_gate_observed_count"] == 2
    assert quality["quality_gate_passed_rate"] == 0.5
    assert quality["quality_gate_fallback_rate"] == 0.5
    assert quality["source_blend_accuracy"] == 1.0


def test_agent_trace_eval_detects_source_blend_and_notice_regressions(tmp_path):
    fixture = tmp_path / "quality.jsonl"
    fixture.write_text(
        "\n".join(
            [
                '{"id":"general-noise","query":"Compare Python lists and tuples.","answer_mode":"general_llm","source_blend":"general_llm","expected_source_blend":"general_llm","answer":"Note: no matching local knowledge-base evidence was found. Python lists are mutable.","evidence_hits":[],"expected_retrieval_hit":false,"should_use_local_evidence":false,"external_fallback_allowed":true,"expected_answer_points":["mutable"],"expected_user_notice":"none"}',
                '{"id":"wrong-source","query":"Why do diffusion models work?","answer_mode":"general_llm","source_blend":"general_llm","expected_source_blend":"external_academic","answer":"Diffusion models learn denoising steps.","evidence_hits":[],"expected_retrieval_hit":false,"should_use_local_evidence":false,"external_fallback_allowed":true,"expected_answer_points":["denoising"],"expected_user_notice":"external_not_local"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    quality = evaluate_quality_cases(fixture)

    assert quality["ok"] is False
    assert quality["source_blend_accuracy"] == 0.5
    assert quality["unnecessary_notice_rate"] == 1.0
    assert quality["required_notice_accuracy"] == 0.0
    assert any("unnecessary knowledge-base miss notice" in error for error in quality["errors"])
    assert any("did not match expected external_academic" in error for error in quality["errors"])
