from __future__ import annotations

from kb.answer_runtime_check import build_answer_runtime_check, repair_answer_for_runtime_contract


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


def test_runtime_repair_removes_general_api_kb_miss_notice():
    result = repair_answer_for_runtime_contract(
        answer="Note: no matching local knowledge-base evidence was found. Python lists are mutable.",
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

    assert result["changed"] is True
    assert result["answer"] == "Python lists are mutable."
    assert "unnecessary_source_notice_removed" in result["reasons"]
    assert result["after"]["status"] == "passed"
    assert "Note:" not in result["answer"]
    assert "original" not in result


def test_runtime_repair_adds_hybrid_notice_once():
    result = repair_answer_for_runtime_contract(
        answer="Local evidence says retrieval was used [1]. External context: this can reduce hallucination.",
        answer_quality={"answer_profile": "hybrid_synthesis"},
        answer_mode="hybrid_local_external",
        agent_source_summary={
            "kind": "local_plus_external",
            "label": "Local + external",
            "should_show": True,
            "source_blend": "hybrid_local_external",
            "source_notice_count": 0,
        },
    )

    assert result["changed"] is True
    assert result["answer"].count("local citations [n] come from the knowledge base") == 1
    assert "missing_source_notice_added" in result["reasons"]
    assert result["after"]["status"] == "passed"


def test_runtime_repair_collapses_duplicate_external_notices():
    answer = (
        "Note: no matching local knowledge-base evidence was found; this is an external model answer, "
        "not a knowledge-base-grounded answer.\n"
        "Note: no matching local knowledge-base evidence was found.\n\n"
        "Diffusion models learn a reverse denoising process."
    )

    result = repair_answer_for_runtime_contract(
        answer=answer,
        answer_quality={"answer_profile": "external_academic"},
        answer_mode="external_academic_llm",
        agent_source_summary={
            "kind": "external_not_kb",
            "label": "Not from KB",
            "should_show": True,
            "source_blend": "external_academic",
            "source_notice_count": 2,
        },
    )

    assert result["changed"] is True
    assert result["answer"].count("no matching local knowledge-base evidence was found") == 1
    assert "source_notice_normalized" in result["reasons"]
    assert result["after"]["status"] == "passed"


def test_runtime_repair_removes_trace_suffix_before_storage():
    result = repair_answer_for_runtime_contract(
        answer="The useful answer stays.\n\nResearch Agent Trace\nPlan\n- retrieve_evidence debug",
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

    assert result["changed"] is True
    assert result["answer"] == "The useful answer stays."
    assert "debug_content_removed" in result["reasons"]


def test_runtime_repair_keeps_non_notice_phrase_inside_body():
    result = repair_answer_for_runtime_contract(
        answer="The phrase no matching local knowledge-base evidence is only an example here.",
        answer_quality={"answer_profile": "general_api"},
        answer_mode="general_llm",
        agent_source_summary={
            "kind": "general_api",
            "label": "Not from KB",
            "should_show": True,
            "source_blend": "general_llm",
            "source_notice_count": 0,
        },
    )

    assert result["changed"] is False
    assert "only an example" in result["answer"]
