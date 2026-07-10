from kb.agent.planner import classify_question_type, plan_research_intent, plan_research_question


def test_planner_classifies_common_question_types():
    assert classify_question_type("Compare these two papers") == "multi_paper_comparison"
    assert classify_question_type("How to read this paper first?") == "reading_guide"
    assert classify_question_type("Which upstream references does it cite?") == "reference_followup"
    assert classify_question_type("What is the main method?") == "single_paper_qa"
    assert classify_question_type("") == "unknown"


def test_answer_audit_takes_precedence_over_negated_reading_route_phrase():
    prompt = "审查上一条回答，逐条核对标题与依据，不要重新生成阅读路线。"

    intent = plan_research_intent(prompt)

    assert intent.task_type == "multi_paper_comparison"
    assert intent.routing_signals[0] == "answer_audit"
    assert "compare_papers" in intent.required_tools


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


def test_planner_intent_exposes_tools_confidence_and_evidence_need():
    intent = plan_research_intent('What limitations does "Paper A" report?')

    assert intent.task_type == "single_paper_qa"
    assert intent.target_papers == ["Paper A"]
    assert intent.required_tools == [
        "retrieve_evidence",
        "generate_grounded_answer",
        "verify_answer_citations",
    ]
    assert intent.evidence_need == "high"
    assert "limitation_analysis" in intent.routing_signals
    assert 0.0 <= intent.confidence <= 1.0


def test_planner_intent_for_comparison_selects_comparison_tool():
    intent = plan_research_intent("Compare these papers by experiment design.")

    assert intent.task_type == "multi_paper_comparison"
    assert "compare_papers" in intent.required_tools
    assert intent.evidence_need == "high"
    assert "comparison_keyword" in intent.routing_signals
