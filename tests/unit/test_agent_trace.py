from kb.agent.runner import build_agent_trace_for_completed_answer, build_generation_agent_notes


def test_agent_trace_serializes_completed_rag_answer():
    trace = build_agent_trace_for_completed_answer(
        "What does the paper claim?",
        "The paper claims retrieval improves answer grounding [1].",
        evidence_hits=[
            {
                "text": "retrieval improves answer grounding in the system",
                "score": 3.2,
                "meta": {
                    "source_name": "demo.md",
                    "source_path": "demo.md",
                    "heading_path": "Results",
                },
            }
        ],
        scope_context={"query_scope": "library", "scope_source": "test"},
    )

    assert trace["mode"] == "research_agent"
    assert trace["question_type"] == "single_paper_qa"
    assert trace["context"]["query_scope"] == "library"
    assert trace["verification"]["total_claims"] == 1
    assert trace["verification"]["supported_claims"] == 1
    assert trace["summary"]["supported_claims"] == 1
    assert trace["summary"]["total_claims"] == 1
    assert trace["verification"]["evidence_status"] == "needs_review"
    assert trace["summary"]["evidence_status"] == "needs_review"
    assert trace["summary"]["evidence_hit_count"] == 1
    assert trace["summary"]["tool_call_count"] == 3
    assert trace["summary"]["query_scope"] == "library"
    assert [step["tool"] for step in trace["steps"]] == [
        "retrieve_evidence",
        "generate_grounded_answer",
        "verify_answer_citations",
    ]


def test_agent_trace_can_mark_error_status():
    trace = build_agent_trace_for_completed_answer("x", "Generation failed.", status="error")

    assert trace["status"] == "error"
    assert all(step["status"] == "error" for step in trace["plan"])


def test_completed_external_trace_skips_local_verification():
    trace = build_agent_trace_for_completed_answer(
        "Compare Python lists and tuples.",
        "Lists are mutable; tuples are immutable.",
        answer_mode="general_llm",
        status="canceled",
    )

    assert trace["status"] == "canceled"
    assert trace["plan"][-1]["status"] == "skipped"
    assert trace["steps"][-1]["status"] == "skipped"
    assert trace["research_run"]["status"] == "failed"


def test_agent_trace_marks_no_evidence_as_insufficient():
    trace = build_agent_trace_for_completed_answer("x", "Answer without evidence [1].", evidence_hits=[])

    assert trace["verification"]["evidence_status"] == "insufficient"
    assert trace["summary"]["evidence_status"] == "insufficient"
    assert trace["summary"]["evidence_hit_count"] == 0


def test_generation_agent_notes_allow_external_academic_fallback_when_no_hits():
    bridge = build_generation_agent_notes(
        "In the literature, how does retrieval augmented generation improve academic question answering?",
        evidence_hits=[],
        candidate_hits=[],
        scope_context={"query_scope": "library", "scope_source": "test"},
    )

    gate = bridge["agent_notes"]["evidence_gate"]
    assert gate["answer_mode"] == "external_academic_llm"
    assert gate["source_policy"] == "external_allowed_with_notice"
    assert gate["evidence_status"] == "not_applicable"
    assert bridge["context"]["answer_source_blend"] == "external_academic"


def test_generation_agent_notes_recommend_hybrid_for_thin_local_evidence():
    bridge = build_generation_agent_notes(
        "How does the paper use retrieval augmented generation?",
        evidence_hits=[
            {
                "text": "The paper uses retrieval before generation to improve grounding.",
                "score": 3.0,
                "meta": {
                    "source_name": "Paper A",
                    "source_path": "paper-a.md",
                    "heading_path": "Method",
                },
            }
        ],
        candidate_hits=[],
        scope_context={"query_scope": "library", "scope_source": "test"},
    )

    gate = bridge["agent_notes"]["evidence_gate"]
    assert gate["answer_mode"] == "hybrid_local_external"
    assert gate["source_policy"] == "local_plus_external_background"
    assert bridge["hybrid_generation_recommended"] is True
    assert bridge["context"]["hybrid_generation_recommended"] is True
    assert bridge["agent_notes"]["research_run"]["source_policy"] == "local_plus_external_background"
    assert len(bridge["agent_notes"]["evidence_matrix"]) == 1


def test_completed_trace_marks_external_answer_verification_not_applicable():
    query = "In the literature, how does retrieval augmented generation improve academic question answering?"
    bridge = build_generation_agent_notes(
        query,
        evidence_hits=[],
        candidate_hits=[],
        scope_context={"query_scope": "library", "scope_source": "test"},
    )

    trace = build_agent_trace_for_completed_answer(
        query,
        (
            "Note: no matching local knowledge-base evidence was found; this is an external model answer, "
            "not a knowledge-base-grounded answer.\n\n"
            "RAG can improve answer grounding by retrieving relevant context before generation."
        ),
        evidence_hits=[],
        scope_context=bridge["context"],
        agent_notes=bridge["agent_notes"],
        answer_mode=bridge["context"]["answer_mode"],
    )

    assert trace["verification"]["evidence_status"] == "not_applicable"
    assert trace["summary"]["evidence_status"] == "not_applicable"
    assert trace["research_run"]["source_policy"] == "external_allowed_with_notice"


def test_completed_trace_preserves_hybrid_generation_quality_gate():
    query = "How does the paper use retrieval augmented generation?"
    bridge = build_generation_agent_notes(
        query,
        evidence_hits=[
            {
                "text": "The paper uses retrieval before generation to improve grounding.",
                "score": 3.0,
                "meta": {
                    "source_name": "Paper A",
                    "source_path": "paper-a.md",
                    "heading_path": "Method",
                },
            }
        ],
        candidate_hits=[],
        scope_context={"query_scope": "library", "scope_source": "test"},
    )

    trace = build_agent_trace_for_completed_answer(
        query,
        (
            "Note: local citations [n] come from the knowledge base; uncited background may use external model context.\n\n"
            "Local evidence: the paper uses retrieval before generation [1].\n"
            "External context: RAG commonly uses retrieved context to reduce unsupported generation."
        ),
        evidence_hits=[
            {
                "text": "The paper uses retrieval before generation to improve grounding.",
                "score": 3.0,
                "meta": {"source_name": "Paper A", "source_path": "paper-a.md"},
            }
        ],
        scope_context=bridge["context"],
        agent_notes=bridge["agent_notes"],
        answer_mode="hybrid_local_external",
        generation_output={
            "answer_mode": "hybrid_local_external",
            "source_blend": "hybrid_local_external",
            "quality_gate": {"status": "passed", "reasons": [], "warnings": []},
            "observation": "Generated a hybrid answer from local evidence plus external model context.",
        },
    )

    assert trace["research_run"]["source_policy"] == "local_plus_external_background"
    assert trace["summary"]["answer_source_blend"] == "hybrid_local_external"
    assert trace["summary"]["quality_gate_status"] == "passed"
    assert trace["steps"][1]["output"]["quality_gate"]["status"] == "passed"
