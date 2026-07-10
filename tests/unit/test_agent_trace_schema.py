from kb.agent.runner import build_agent_trace_for_completed_answer
from kb.agent.schema import validate_agent_trace


def test_research_agent_contract_models_keep_public_shape():
    from api.contracts.research_agent import ResearchAgentRequest, ResearchAgentResponse

    request = ResearchAgentRequest(query="What does the paper show?", prompt_context={"scope": "library"})
    response = ResearchAgentResponse(answer="A concise answer.", agent_trace={}, hits=[])

    assert request.query == "What does the paper show?"
    assert request.prompt_context == {"scope": "library"}
    assert response.model_dump() == {"answer": "A concise answer.", "agent_trace": {}, "hits": []}


def test_agent_trace_schema_accepts_completed_trace():
    trace = build_agent_trace_for_completed_answer(
        "How does the method work?",
        "The method retrieves grounded evidence [1].",
        evidence_hits=[
            {
                "text": "The method retrieves grounded evidence before answering.",
                "score": 2.0,
                "meta": {"source_name": "paper.md", "source_path": "paper.md"},
            }
        ],
        scope_context={"query_scope": "library", "scope_source": "test"},
    )

    validation = validate_agent_trace(trace)

    assert validation["ok"] is True
    assert validation["summary"]["has_context"] is True
    assert validation["summary"]["tool_call_count"] == 3
    assert trace["summary"]["plan_step_count"] == len(trace["plan"])
    assert trace["context"]["planner_intent"]["task_type"] == "single_paper_qa"
    assert 0.0 <= trace["summary"]["planner_confidence"] <= 1.0
    assert trace["research_run"]["status"] == "verified"
    assert trace["summary"]["research_run_status"] == "verified"
    assert trace["summary"]["source_policy"] == "local_only"
    assert trace["summary"]["evidence_matrix_rows"] == 1
    assert trace["summary"]["answer_source_blend"] == ""
    assert trace["summary"]["quality_gate_status"] == ""
    assert trace["summary"]["quality_gate_reasons"] == []


def test_agent_trace_schema_accepts_quality_gate_observability_fields():
    trace = build_agent_trace_for_completed_answer(
        "Compare the two methods.",
        "The method retrieves grounded evidence [1].",
        evidence_hits=[
            {
                "text": "The method retrieves grounded evidence before answering.",
                "score": 2.0,
                "meta": {"source_name": "paper.md", "source_path": "paper.md"},
            }
        ],
        scope_context={"query_scope": "library", "scope_source": "test"},
    )
    trace["steps"][1]["output"]["quality_gate"] = {
        "status": "fallback",
        "reasons": ["missing_local_citation"],
        "warnings": ["trace_clutter"],
    }
    trace["summary"].update(
        {
            "quality_gate_status": "fallback",
            "quality_gate_reasons": ["missing_local_citation"],
            "quality_gate_warnings": ["trace_clutter"],
        }
    )

    validation = validate_agent_trace(trace)

    assert validation["ok"] is True
    assert validation["summary"]["quality_gate_status"] == "fallback"
    assert validation["summary"]["quality_gate_reason_count"] == 1


def test_agent_trace_schema_reports_invalid_fields():
    validation = validate_agent_trace(
        {
            "mode": "other",
            "question_type": "single_paper_qa",
            "context": [],
            "plan": [{"goal": "", "tool": "missing", "status": "done"}],
            "steps": [
                {"tool": "retrieve_evidence", "status": "bad", "output": []},
                {
                    "tool": "generate_grounded_answer",
                    "status": "done",
                    "output": {"quality_gate": {"status": "maybe", "reasons": "missing"}},
                },
            ],
            "verification": {"total_claims": "x", "supported_claims": 0, "unsupported_claims": 0, "evidence_status": "mystery"},
            "summary": {
                "tool_call_count": "many",
                "evidence_status": "mystery",
                "answer_source_blend": "mystery",
                "quality_gate_status": "maybe",
            },
            "status": "done",
            "errors": [],
        }
    )

    assert validation["ok"] is False
    assert any("mode" in error for error in validation["errors"])
    assert any("context" in error for error in validation["errors"])
    assert any("tool" in error for error in validation["errors"])
    assert any("evidence_status" in error for error in validation["errors"])
    assert any("summary.tool_call_count" in error for error in validation["errors"])
    assert any("summary.answer_source_blend" in error for error in validation["errors"])
    assert any("quality_gate.status" in error for error in validation["errors"])
    assert any("summary.quality_gate_status" in error for error in validation["errors"])
