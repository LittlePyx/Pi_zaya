from kb.agent.runner import build_agent_trace_for_completed_answer


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
    )

    assert trace["mode"] == "research_agent"
    assert trace["question_type"] == "single_paper_qa"
    assert trace["verification"]["total_claims"] == 1
    assert trace["verification"]["supported_claims"] == 1
    assert [step["tool"] for step in trace["steps"]] == [
        "retrieve_evidence",
        "generate_grounded_answer",
        "verify_answer_citations",
    ]


def test_agent_trace_can_mark_error_status():
    trace = build_agent_trace_for_completed_answer("x", "Generation failed.", status="error")

    assert trace["status"] == "error"
    assert all(step["status"] == "error" for step in trace["plan"])
