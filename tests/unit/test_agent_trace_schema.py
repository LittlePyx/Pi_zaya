from kb.agent.runner import build_agent_trace_for_completed_answer
from kb.agent.schema import validate_agent_trace


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


def test_agent_trace_schema_reports_invalid_fields():
    validation = validate_agent_trace(
        {
            "mode": "other",
            "question_type": "single_paper_qa",
            "context": [],
            "plan": [{"goal": "", "tool": "missing", "status": "done"}],
            "steps": [{"tool": "retrieve_evidence", "status": "bad", "output": []}],
            "verification": {"total_claims": "x", "supported_claims": 0, "unsupported_claims": 0},
            "status": "done",
            "errors": [],
        }
    )

    assert validation["ok"] is False
    assert any("mode" in error for error in validation["errors"])
    assert any("context" in error for error in validation["errors"])
    assert any("tool" in error for error in validation["errors"])
