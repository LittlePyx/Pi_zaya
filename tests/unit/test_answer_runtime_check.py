from __future__ import annotations

from kb.answer_runtime_check import build_answer_runtime_check


def test_answer_runtime_check_passes_local_grounded_answer():
    check = build_answer_runtime_check(
        answer="Paper A uses retrieval before generation [1].",
        answer_quality={"answer_profile": "local_evidence_grounded"},
        answer_mode="evidence_grounded",
        agent_source_summary={
            "kind": "local_kb",
            "label": "Local KB",
            "should_show": True,
            "source_blend": "local_grounded",
            "source_notice_count": 0,
        },
    )

    assert check["status"] == "passed"
    assert check["summary"]["failed"] == []
    assert check["checks"]["main_answer_clutter"]["ok"] is True
    assert "answer" not in check
    assert "agent_trace" not in check


def test_answer_runtime_check_flags_unnecessary_kb_notice_for_general_api():
    check = build_answer_runtime_check(
        answer="Note: no matching local knowledge-base evidence was found.\n\nThis is a general explanation.",
        answer_quality={"answer_profile": "general_api"},
        answer_mode="general_llm",
        agent_source_summary={
            "kind": "general_api",
            "label": "Not from KB",
            "should_show": True,
            "source_blend": "general_llm",
            "source_notice_count": 1,
        },
    )

    assert check["status"] == "needs_review"
    assert "notice_shape" in check["summary"]["failed"]
    assert "unnecessary_source_notice" in check["checks"]["notice_shape"]["reasons"]


def test_answer_runtime_check_flags_source_summary_mismatch():
    check = build_answer_runtime_check(
        answer=(
            "Note: local citations [n] come from the knowledge base; uncited background may use external model context.\n\n"
            "Local evidence says retrieval was used [1]. External context: this can reduce hallucination."
        ),
        answer_quality={"answer_profile": "hybrid_synthesis"},
        answer_mode="hybrid_local_external",
        agent_source_summary={
            "kind": "local_kb",
            "label": "Local KB",
            "should_show": True,
            "source_blend": "hybrid_local_external",
            "source_notice_count": 1,
        },
    )

    assert check["status"] == "needs_review"
    assert "source_summary" in check["summary"]["failed"]
    assert "source_summary_kind_mismatch" in check["checks"]["source_summary"]["reasons"]


def test_answer_runtime_check_flags_trace_clutter_without_storing_snippets():
    check = build_answer_runtime_check(
        answer='Research Agent Trace\n{"mode": "research_agent"}\nPlan steps: retrieve then answer.',
        answer_quality={"answer_profile": "external_academic"},
        answer_mode="external_academic_llm",
        agent_source_summary={
            "kind": "external_not_kb",
            "label": "Not from KB",
            "should_show": True,
            "source_blend": "external_academic",
            "source_notice_count": 0,
        },
    )

    assert check["status"] == "needs_review"
    assert "main_answer_clutter" in check["summary"]["failed"]
    assert "trace_panel_leak" in check["checks"]["main_answer_clutter"]["reasons"]
    assert "checks" in check
    assert "Research Agent Trace" not in str(check)
