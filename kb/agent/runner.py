from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .planner import plan_research_question
from .tools import (
    build_reading_guide,
    compare_papers,
    generate_grounded_answer,
    retrieve_evidence,
    retrieve_references,
    verify_answer_citations,
)
from .types import AgentExecutionStep, AgentTrace
from .verifier import verify_answer_citations as verify_completed_answer


def _summarize_hits(hits: list[dict[str, Any]], *, limit: int = 6) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for hit in list(hits or [])[:limit]:
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        out.append(
            {
                "source_name": str(meta.get("source_name") or meta.get("title") or "").strip(),
                "source_path": str(meta.get("source_path") or "").strip(),
                "heading_path": str(meta.get("heading_path") or meta.get("top_heading") or "").strip(),
                "score": float(hit.get("score") or 0.0),
                "text_preview": str(hit.get("text") or "").strip()[:220],
            }
        )
    return out


def _run_step(trace: AgentTrace, index: int, tool_fn, *args, **kwargs) -> dict[str, Any]:
    step = trace.plan[index]
    step.status = "running"
    started = time.perf_counter()
    try:
        result = tool_fn(*args, **kwargs)
        step.status = "done"
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))
        trace.steps.append(
            AgentExecutionStep(
                tool=step.tool,
                status="done",
                observation=str(result.get("observation") or ""),
                output={k: v for k, v in result.items() if k not in {"hits", "answer", "observation"}},
                elapsed_ms=elapsed_ms,
            )
        )
        return result
    except Exception as exc:
        step.status = "error"
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))
        trace.errors.append(f"{step.tool}: {str(exc)[:240]}")
        trace.steps.append(
            AgentExecutionStep(
                tool=step.tool,
                status="error",
                observation="Tool failed; continuing with partial trace where possible.",
                error=str(exc)[:240],
                elapsed_ms=elapsed_ms,
            )
        )
        return {}


def run_research_agent(
    query: str,
    *,
    db_dir: str | Path,
    settings: Any = None,
    history: list[dict[str, Any]] | None = None,
    top_k: int = 6,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    max_steps: int = 6,
) -> dict[str, Any]:
    question_type, plan = plan_research_question(query)
    trace = AgentTrace(question_type=question_type, plan=plan, status="running")
    context: dict[str, Any] = {"hits": [], "answer": ""}
    for idx, plan_step in enumerate(list(trace.plan)[: max(1, int(max_steps or 6))]):
        if plan_step.tool == "retrieve_evidence":
            result = _run_step(trace, idx, retrieve_evidence, query, db_dir=db_dir, settings=settings, top_k=top_k)
            context["hits"] = list(result.get("hits") or [])
            if trace.steps:
                trace.steps[-1].output["hits"] = _summarize_hits(context["hits"])
        elif plan_step.tool == "retrieve_references":
            _run_step(trace, idx, retrieve_references, query, context["hits"], settings=settings, top_k=top_k)
        elif plan_step.tool == "build_reading_guide":
            _run_step(trace, idx, build_reading_guide, query, context["hits"])
        elif plan_step.tool == "compare_papers":
            _run_step(trace, idx, compare_papers, query, context["hits"])
        elif plan_step.tool == "generate_grounded_answer":
            result = _run_step(
                trace,
                idx,
                generate_grounded_answer,
                query,
                context["hits"],
                settings=settings,
                history=history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            context["answer"] = str(result.get("answer") or "")
        elif plan_step.tool == "verify_answer_citations":
            result = _run_step(trace, idx, verify_answer_citations, context["answer"], context["hits"])
            trace.verification = verify_completed_answer(context["answer"], context["hits"])
            if isinstance(result.get("verification"), dict):
                # Keep the dataclass path authoritative while preserving tool output.
                pass
    if not context["answer"]:
        context["answer"] = "Research Agent Mode could not generate an answer. See agent_trace.errors for details."
    trace.status = "error" if trace.errors and not context["answer"] else "done"
    return {"answer": context["answer"], "agent_trace": trace.to_dict(), "hits": context["hits"]}


def build_agent_trace_for_completed_answer(
    query: str,
    answer: str,
    *,
    evidence_hits: list[dict[str, Any]] | None = None,
    status: str = "done",
) -> dict[str, Any]:
    question_type, plan = plan_research_question(query)
    hits = [h for h in list(evidence_hits or []) if isinstance(h, dict)]
    verification = verify_completed_answer(answer, hits)
    final_status = str(status or "done").strip().lower()
    if final_status not in {"done", "error", "canceled"}:
        final_status = "done"
    trace = AgentTrace(question_type=question_type, plan=plan, verification=verification, status=final_status)
    for step in trace.plan:
        step.status = "done" if final_status == "done" else "error"
    trace.steps.append(
        AgentExecutionStep(
            tool="retrieve_evidence",
            status="done",
            observation=f"Existing RAG retrieval supplied {len(hits)} answer evidence hit(s).",
            output={"hits": _summarize_hits(hits)},
        )
    )
    trace.steps.append(
        AgentExecutionStep(
            tool="generate_grounded_answer",
            status="done" if str(status or "done") == "done" else "error",
            observation="Existing RAG generation produced the answer; agent trace was attached as an explicit wrapper.",
            output={"answer_chars": len(str(answer or ""))},
        )
    )
    trace.steps.append(
        AgentExecutionStep(
            tool="verify_answer_citations",
            status="done",
            observation=(
                f"Verified {verification.supported_claims}/{verification.total_claims} claim(s) "
                "with citation/evidence support."
            ),
            output=verification.to_dict(),
        )
    )
    return trace.to_dict()
