from tools.research_qa.validate_research_agent_golden import validate_cases


def test_research_agent_golden_dataset_matches_planner_and_tool_plan():
    summary = validate_cases()

    assert summary["ok"], summary["errors"]
    assert summary["case_count"] >= 8
    for question_type in [
        "single_paper_qa",
        "multi_paper_comparison",
        "reading_guide",
        "reference_followup",
        "unknown",
    ]:
        assert summary["question_types"].get(question_type, 0) >= 1
