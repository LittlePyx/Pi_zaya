from kb.agent.runner import build_agent_trace_for_completed_answer, build_generation_agent_notes
from kb.agent.source_summary import build_agent_source_summary


def test_agent_source_summary_marks_local_kb_without_trace_details():
    trace = build_agent_trace_for_completed_answer(
        "What does Paper A say?",
        "Paper A uses retrieval before generation [1].",
        evidence_hits=[
            {
                "text": "Paper A uses retrieval before generation.",
                "score": 4.0,
                "meta": {"source_name": "Paper A", "source_path": "paper-a.md"},
            }
        ],
        scope_context={"query_scope": "library", "answer_source_blend": "local_grounded"},
        answer_mode="evidence_grounded",
    )

    summary = build_agent_source_summary(trace)

    assert summary["kind"] == "local_kb"
    assert summary["label_key"] == "agent_trace_source_local_only"
    assert summary["should_show"] is True
    assert "plan" not in summary
    assert "steps" not in summary
    assert "claims" not in summary


def test_agent_source_summary_marks_hybrid_local_external():
    query = "How should I interpret sparse evidence?"
    bridge = build_generation_agent_notes(
        query,
        evidence_hits=[
            {
                "text": "Paper A reports sparse citation coverage.",
                "score": 3.0,
                "meta": {"source_name": "Paper A", "source_path": "paper-a.md"},
            }
        ],
        candidate_hits=[],
        scope_context={"query_scope": "library"},
    )
    trace = build_agent_trace_for_completed_answer(
        query,
        (
            "Note: local citations [n] come from the knowledge base; uncited background may use external model context.\n\n"
            "Local evidence: Paper A reports sparse citation coverage [1].\n"
            "External context: sparse coverage means conclusions should stay provisional."
        ),
        evidence_hits=[
            {
                "text": "Paper A reports sparse citation coverage.",
                "score": 3.0,
                "meta": {"source_name": "Paper A", "source_path": "paper-a.md"},
            }
        ],
        scope_context=bridge["context"],
        agent_notes=bridge["agent_notes"],
        answer_mode="hybrid_local_external",
    )

    summary = build_agent_source_summary(trace)

    assert summary["kind"] == "local_plus_external"
    assert summary["label_key"] == "agent_trace_source_local_external"
    assert summary["source_policy"] == "local_plus_external_background"


def test_agent_source_summary_marks_external_answer_as_not_from_kb():
    query = "Why do diffusion models denoise?"
    bridge = build_generation_agent_notes(
        query,
        evidence_hits=[],
        candidate_hits=[],
        scope_context={"query_scope": "library"},
    )
    trace = build_agent_trace_for_completed_answer(
        query,
        (
            "Note: no matching local knowledge-base evidence was found; this is an external model answer, "
            "not a knowledge-base-grounded answer.\n\n"
            "Diffusion models learn a reverse denoising process."
        ),
        evidence_hits=[],
        scope_context=bridge["context"],
        agent_notes=bridge["agent_notes"],
        answer_mode="external_academic_llm",
    )

    summary = build_agent_source_summary(trace)

    assert summary["kind"] == "external_not_kb"
    assert summary["label_key"] == "agent_trace_evidence_not_from_kb"
    assert summary["confidence"] == "external"
    assert summary["evidence_hit_count"] == 0


def test_agent_source_summary_handles_missing_trace():
    assert build_agent_source_summary(None) == {}
