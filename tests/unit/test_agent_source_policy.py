from kb.agent.source_policy import decide_answer_source


def test_source_policy_routes_general_no_hit_to_plain_llm():
    decision = decide_answer_source(
        hit_count=0,
        candidate_hit_count=0,
        academic_question=False,
        local_grounding_requested=False,
    )

    assert decision.source_blend == "general_llm"
    assert decision.answer_mode == "general_llm"
    assert decision.source_notice == "none"
    assert decision.evidence_status == "not_applicable"
    assert "general_question_no_indexed_evidence_required" in decision.to_evidence_gate()["reasons"]


def test_source_policy_routes_academic_no_hit_to_external_with_notice():
    decision = decide_answer_source(
        hit_count=0,
        candidate_hit_count=2,
        retrieval_confidence="low",
        retrieval_reasons=["weak_query_overlap"],
        academic_question=True,
        local_grounding_requested=True,
    )

    gate = decision.to_evidence_gate()
    assert decision.source_blend == "external_academic"
    assert decision.answer_mode == "external_academic_llm"
    assert decision.source_policy == "external_allowed_with_notice"
    assert decision.source_notice == "external"
    assert gate["retrieval_confidence"] == "low"
    assert "not_based_on_local_knowledge_base" in gate["reasons"]
    assert "local_grounding_requested" in gate["reasons"]


def test_source_policy_routes_thin_local_evidence_to_hybrid():
    decision = decide_answer_source(
        hit_count=1,
        candidate_hit_count=1,
        academic_question=True,
        local_grounding_requested=True,
    )

    assert decision.source_blend == "hybrid_local_external"
    assert decision.answer_mode == "hybrid_local_external"
    assert decision.source_policy == "local_plus_external_background"
    assert decision.evidence_status == "needs_review"


def test_source_policy_routes_strong_nonacademic_hits_to_local_grounded():
    decision = decide_answer_source(
        hit_count=3,
        candidate_hit_count=3,
        academic_question=False,
        local_grounding_requested=False,
    )

    assert decision.source_blend == "local_grounded"
    assert decision.answer_mode == "evidence_grounded"
    assert decision.source_policy == "local_only"
    assert decision.evidence_status == "grounded"
