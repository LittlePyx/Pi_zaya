from __future__ import annotations

import re

from kb.agent.runner import build_agent_trace_for_completed_answer
from kb.agent.source_summary import build_agent_source_summary
from kb.answer_runtime_check import build_answer_runtime_check, repair_answer_for_runtime_contract


_FORBIDDEN_VISIBLE_TERMS = (
    "Research Agent Trace",
    "agent_trace",
    "tool calls",
    "verification statistics",
)


def _notice_line_count(answer: str) -> int:
    count = 0
    for raw in str(answer or "").splitlines():
        line = " ".join(str(raw or "").lower().split())
        if not line:
            continue
        if line.startswith("note:") and (
            "knowledge base" in line
            or "local citations" in line
            or "external model" in line
        ):
            count += 1
            continue
        if (
            "no matching local knowledge-base evidence" in line
            or "not a knowledge-base-grounded answer" in line
        ):
            count += 1
    return count


def _has_local_citation(answer: str) -> bool:
    return bool(re.search(r"\[[0-9][0-9,\-\s]*\]", str(answer or "")))


def _assert_no_visible_debug(answer: str) -> None:
    for term in _FORBIDDEN_VISIBLE_TERMS:
        assert term not in answer


def _final_visible_answer(
    *,
    query: str,
    raw_answer: str,
    answer_profile: str,
    answer_mode: str,
    source_blend: str,
    hits: list[dict] | None = None,
) -> dict:
    source_policy = {
        "local_grounded": "local_only",
        "hybrid_local_external": "local_plus_external_background",
        "external_academic": "external_allowed_with_notice",
        "general_llm": "external_allowed_without_notice",
    }.get(source_blend, "")
    scope_context = {
        "query_scope": "library",
        "answer_source_blend": source_blend,
        "answer_mode": answer_mode,
        "source_policy": source_policy,
    }
    trace = build_agent_trace_for_completed_answer(
        query,
        raw_answer,
        evidence_hits=list(hits or []),
        scope_context=scope_context,
        answer_mode=answer_mode,
    )
    source_summary = build_agent_source_summary(trace)
    answer_quality = {"answer_profile": answer_profile}
    repair = repair_answer_for_runtime_contract(
        answer=raw_answer,
        query=query,
        answer_quality=answer_quality,
        agent_trace=trace,
        agent_source_summary=source_summary,
        answer_mode=answer_mode,
        source_blend=source_blend,
    )
    final_answer = str(repair.get("answer") or "")
    final_trace = build_agent_trace_for_completed_answer(
        query,
        final_answer,
        evidence_hits=list(hits or []),
        scope_context=scope_context,
        answer_mode=answer_mode,
    )
    final_source_summary = build_agent_source_summary(final_trace)
    final_check = build_answer_runtime_check(
        answer=final_answer,
        answer_quality=answer_quality,
        agent_trace=final_trace,
        agent_source_summary=final_source_summary,
        answer_mode=answer_mode,
        source_blend=source_blend,
    )
    return {
        "answer": final_answer,
        "trace": final_trace,
        "source_summary": final_source_summary,
        "repair": repair,
        "runtime_check": final_check,
    }


def test_visible_local_kb_answer_keeps_citation_and_removes_trace_debug():
    hit = {
        "text": "Paper A uses retrieval before generation to ground its answer.",
        "score": 4.0,
        "meta": {"source_name": "Paper A", "source_path": "paper-a.md"},
    }

    out = _final_visible_answer(
        query="What does Paper A say about retrieval?",
        raw_answer=(
            "Paper A uses retrieval before generation to ground its answer [1].\n\n"
            "Research Agent Trace\nPlan\n- retrieve_evidence debug"
        ),
        answer_profile="local_evidence_grounded",
        answer_mode="evidence_grounded",
        source_blend="local_grounded",
        hits=[hit],
    )

    assert out["runtime_check"]["status"] == "passed"
    assert out["repair"]["changed"] is True
    assert _has_local_citation(out["answer"])
    assert out["source_summary"]["kind"] == "local_kb"
    assert _notice_line_count(out["answer"]) == 0
    _assert_no_visible_debug(out["answer"])


def test_visible_external_academic_no_hit_discloses_not_from_kb_once():
    out = _final_visible_answer(
        query="Why do diffusion models work?",
        raw_answer="Diffusion models learn a reverse denoising process.",
        answer_profile="external_academic",
        answer_mode="external_academic_llm",
        source_blend="external_academic",
        hits=[],
    )

    assert out["runtime_check"]["status"] == "passed"
    assert out["repair"]["changed"] is True
    assert out["source_summary"]["kind"] == "external_not_kb"
    assert "not a knowledge-base-grounded answer" in out["answer"]
    assert _notice_line_count(out["answer"]) == 1
    _assert_no_visible_debug(out["answer"])


def test_visible_general_api_answer_removes_unnecessary_kb_miss_notice():
    out = _final_visible_answer(
        query="Compare Python lists and tuples.",
        raw_answer="Note: no matching local knowledge-base evidence was found. Python lists are mutable; tuples are immutable.",
        answer_profile="general_api",
        answer_mode="general_llm",
        source_blend="general_llm",
        hits=[],
    )

    assert out["runtime_check"]["status"] == "passed"
    assert out["repair"]["changed"] is True
    assert out["source_summary"]["kind"] == "general_api"
    assert "Python lists are mutable" in out["answer"]
    assert "no matching local knowledge-base evidence" not in out["answer"].lower()
    assert _notice_line_count(out["answer"]) == 0
    _assert_no_visible_debug(out["answer"])


def test_visible_hybrid_answer_preserves_local_citation_and_one_source_notice():
    hit = {
        "text": "The paper uses retrieval before generation to reduce unsupported claims.",
        "score": 3.0,
        "meta": {"source_name": "Paper A", "source_path": "paper-a.md"},
    }
    raw_answer = (
        "Note: local citations [n] come from the knowledge base; uncited background may use external model context.\n"
        "Note: local citations [n] come from the knowledge base.\n\n"
        "Local evidence says the paper uses retrieval before generation [1]. "
        "External context: retrieval can reduce unsupported generation."
    )

    out = _final_visible_answer(
        query="How does the paper use retrieval, and what is the broader context?",
        raw_answer=raw_answer,
        answer_profile="hybrid_synthesis",
        answer_mode="hybrid_local_external",
        source_blend="hybrid_local_external",
        hits=[hit],
    )

    assert out["runtime_check"]["status"] == "passed"
    assert out["source_summary"]["kind"] == "local_plus_external"
    assert _has_local_citation(out["answer"])
    assert "External context:" in out["answer"]
    assert _notice_line_count(out["answer"]) == 1
    assert out["answer"].count("local citations [n] come from the knowledge base") == 1
    _assert_no_visible_debug(out["answer"])


def test_visible_answer_strips_accidental_trace_payload_before_final_check():
    out = _final_visible_answer(
        query="Give a concise general explanation.",
        raw_answer='This is the useful answer.\n\n```json\n{"agent_trace": {"mode": "research_agent"}}\n```',
        answer_profile="general_api",
        answer_mode="general_llm",
        source_blend="general_llm",
        hits=[],
    )

    assert out["runtime_check"]["status"] == "passed"
    assert out["repair"]["changed"] is True
    assert out["answer"] == "This is the useful answer."
    assert out["repair"]["after"]["status"] == "passed"
    _assert_no_visible_debug(out["answer"])
