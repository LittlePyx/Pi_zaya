from kb.agent.planner import classify_question_type, plan_research_question


def test_planner_classifies_common_question_types():
    assert classify_question_type("Compare these two papers") == "multi_paper_comparison"
    assert classify_question_type("How to read this paper first?") == "reading_guide"
    assert classify_question_type("Which upstream references does it cite?") == "reference_followup"
    assert classify_question_type("What is the main method?") == "single_paper_qa"
    assert classify_question_type("") == "unknown"


def test_planner_builds_tool_plan_for_comparison():
    question_type, plan = plan_research_question("compare A versus B")

    assert question_type == "multi_paper_comparison"
    assert [step.tool for step in plan] == [
        "retrieve_evidence",
        "compare_papers",
        "generate_grounded_answer",
        "verify_answer_citations",
    ]
    assert all(step.status == "pending" for step in plan)
