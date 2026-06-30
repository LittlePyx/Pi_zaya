from kb.research_answer_plan import infer_research_answer_plan


def test_infer_research_answer_plan_prefers_family_for_exact_method():
    plan = infer_research_answer_plan(
        prompt="Explain equation (3) and the variables.",
        paper_guide_prompt_family="equation",
        answer_intent="reading",
        answer_output_mode="fact_answer",
        paper_guide_mode=True,
    )

    assert plan.kind == "method_explain"
    assert "inputs/outputs" in plan.evidence_need


def test_infer_research_answer_plan_detects_compare_prompt():
    plan = infer_research_answer_plan(
        prompt="Compare the trade-offs between Hadamard and Fourier single-pixel imaging.",
        answer_intent="reading",
    )

    assert plan.kind == "compare"
    assert "comparison axes" in plan.evidence_need
    assert "side-by-side" in plan.answer_shape


def test_infer_research_answer_plan_detects_chinese_experiment_prompt():
    plan = infer_research_answer_plan(
        prompt="这个实验怎么复现？需要哪些对照组和评价指标？",
        answer_intent="reading",
    )

    assert plan.kind == "experiment_design"
    assert "variables" in plan.evidence_need


def test_infer_research_answer_plan_detects_literature_positioning():
    plan = infer_research_answer_plan(
        prompt="这篇文章引用了哪些先前工作，我接下来应该读什么？",
        answer_intent="reading",
    )

    assert plan.kind == "literature_positioning"
    assert "upstream references" in plan.evidence_need


def test_infer_research_answer_plan_maps_paper_guide_panel_families():
    assert (
        infer_research_answer_plan(
            prompt="Walk me through Figure 3.",
            paper_guide_prompt_family="figure_walkthrough",
            paper_guide_mode=True,
        ).kind
        == "method_explain"
    )
    assert (
        infer_research_answer_plan(
            prompt="Explain this boxed framework.",
            paper_guide_prompt_family="box_only",
            paper_guide_mode=True,
        ).kind
        == "paper_summary"
    )
    assert (
        infer_research_answer_plan(
            prompt="What does the discussion imply about limitations?",
            paper_guide_prompt_family="discussion_only",
            paper_guide_mode=True,
        ).kind
        == "critical_review"
    )


def test_infer_research_answer_plan_maps_research_workflow_intents():
    assert (
        infer_research_answer_plan(
            prompt="Help me write a related-work paragraph around these papers.",
            answer_intent="writing",
        ).kind
        == "literature_positioning"
    )
    assert (
        infer_research_answer_plan(
            prompt="Debug why this result fails to reproduce.",
            answer_intent="troubleshoot",
        ).kind
        == "critical_review"
    )


def test_research_answer_plan_prompt_block_is_compact():
    plan = infer_research_answer_plan(
        prompt="What are the main limitations and missing controls?",
        answer_output_mode="critical_review",
    )
    block = plan.to_prompt_block()

    assert plan.kind == "critical_review"
    assert block.count("\n") <= 5
    assert "Research answer plan:" in block
