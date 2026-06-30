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


def _normalize_query_scope(value: object) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in {"current", "paper", "current_paper", "source", "reader"}:
        return "current_paper"
    if raw in {"basket", "shelf", "citation_shelf", "selected"}:
        return "basket"
    if raw in {"library", "all", "all_library", "full_library"}:
        return "library"
    return ""


def _clip_context_value(value: object, *, max_chars: int = 700) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = " ".join(text.split())
    return text[:max_chars]


def _selected_context_items(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    out: list[dict[str, Any]] = []
    for raw in list(value.get("items") or []):
        if isinstance(raw, dict):
            out.append(raw)
    return out


def _source_key(value: object) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    try:
        return str(Path(raw).expanduser().resolve(strict=False)).replace("\\", "/").lower()
    except Exception:
        return raw.lower()


def _source_variants(value: object) -> set[str]:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return set()
    variants = {raw.lower(), _source_key(raw)}
    try:
        path = Path(raw)
        if path.name:
            variants.add(path.name.lower())
        if path.stem:
            variants.add(path.stem.lower())
    except Exception:
        pass
    return {item for item in variants if item}


def _hit_source_blob(hit: dict[str, Any]) -> str:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    fields = [
        meta.get("source_path"),
        meta.get("source_name"),
        meta.get("title"),
        meta.get("doi"),
        meta.get("source_sha1"),
    ]
    return " ".join(str(x or "").replace("\\", "/").lower() for x in fields if str(x or "").strip())


def _selected_context_terms(items: list[dict[str, Any]]) -> set[str]:
    terms: set[str] = set()
    for item in list(items or []):
        for field in (
            "sourcePath",
            "source_path",
            "sourceName",
            "source_name",
            "title",
            "doi",
            "key",
        ):
            value = _clip_context_value(item.get(field), max_chars=500)
            if not value:
                continue
            terms.update(_source_variants(value))
            terms.add(value.lower())
    return {term for term in terms if term}


def _build_scope_context(
    *,
    query_scope: object = "",
    selected_research_context: dict[str, Any] | None = None,
    current_source_path: object = "",
    current_source_name: object = "",
    source: str = "agent_runner",
) -> dict[str, Any]:
    selected_items = _selected_context_items(selected_research_context or {})
    requested = _normalize_query_scope(query_scope)
    has_current = bool(str(current_source_path or "").strip() or str(current_source_name or "").strip())
    has_basket = bool(selected_items)
    effective = requested or ("current_paper" if has_current else "library")
    if effective == "current_paper" and not has_current:
        effective = "library"
    if effective == "basket" and not has_basket:
        effective = "library"
    return {
        "query_scope": effective,
        "requested_query_scope": requested,
        "current_source_path": _clip_context_value(current_source_path, max_chars=1200),
        "current_source_name": _clip_context_value(current_source_name, max_chars=500),
        "selected_research_context_count": len(selected_items),
        "scope_source": source,
    }


def _filter_hits_by_scope(
    hits: list[dict[str, Any]],
    *,
    scope_context: dict[str, Any],
    selected_research_context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scope = str(scope_context.get("query_scope") or "").strip()
    before = len(hits)
    if scope not in {"current_paper", "basket"}:
        return hits, {"active": False, "query_scope": scope or "library", "before": before, "after": before}

    if scope == "current_paper":
        terms: set[str] = set()
        terms.update(_source_variants(scope_context.get("current_source_path")))
        terms.update(_source_variants(scope_context.get("current_source_name")))
    else:
        terms = _selected_context_terms(_selected_context_items(selected_research_context or {}))

    filtered: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        blob = _hit_source_blob(hit)
        if any(term and term in blob for term in terms):
            filtered.append(hit)
    return filtered, {
        "active": True,
        "query_scope": scope,
        "before": before,
        "after": len(filtered),
        "term_count": len(terms),
    }


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
    query_scope: str = "",
    selected_research_context: dict[str, Any] | None = None,
    current_source_path: str = "",
    current_source_name: str = "",
) -> dict[str, Any]:
    question_type, plan = plan_research_question(query)
    scope_context = _build_scope_context(
        query_scope=query_scope,
        selected_research_context=selected_research_context,
        current_source_path=current_source_path,
        current_source_name=current_source_name,
        source="direct_research_agent",
    )
    trace = AgentTrace(question_type=question_type, context=scope_context, plan=plan, status="running")
    context: dict[str, Any] = {"hits": [], "answer": "", "agent_notes": {}}
    for idx, plan_step in enumerate(list(trace.plan)[: max(1, int(max_steps or 6))]):
        if plan_step.tool == "retrieve_evidence":
            result = _run_step(trace, idx, retrieve_evidence, query, db_dir=db_dir, settings=settings, top_k=top_k)
            raw_hits = [hit for hit in list(result.get("hits") or []) if isinstance(hit, dict)]
            scoped_hits, scope_filter = _filter_hits_by_scope(
                raw_hits,
                scope_context=scope_context,
                selected_research_context=selected_research_context,
            )
            context["hits"] = scoped_hits
            trace.context["retrieved_hit_count"] = len(raw_hits)
            trace.context["scoped_hit_count"] = len(scoped_hits)
            if trace.steps:
                trace.steps[-1].output["hits"] = _summarize_hits(context["hits"])
                if scope_filter.get("active"):
                    trace.steps[-1].output["scope_filter"] = scope_filter
        elif plan_step.tool == "retrieve_references":
            result = _run_step(
                trace,
                idx,
                retrieve_references,
                query,
                context["hits"],
                db_dir=db_dir,
                settings=settings,
                top_k=top_k,
            )
            if isinstance(result.get("references"), list):
                context["agent_notes"]["references"] = result["references"]
        elif plan_step.tool == "build_reading_guide":
            result = _run_step(trace, idx, build_reading_guide, query, context["hits"])
            if isinstance(result.get("guide"), list):
                context["agent_notes"]["guide"] = result["guide"]
        elif plan_step.tool == "compare_papers":
            result = _run_step(trace, idx, compare_papers, query, context["hits"])
            if isinstance(result.get("comparisons"), list):
                context["agent_notes"]["comparisons"] = result["comparisons"]
        elif plan_step.tool == "generate_grounded_answer":
            result = _run_step(
                trace,
                idx,
                generate_grounded_answer,
                query,
                context["hits"],
                settings=settings,
                history=history,
                agent_notes=context["agent_notes"],
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
    scope_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    question_type, plan = plan_research_question(query)
    hits = [h for h in list(evidence_hits or []) if isinstance(h, dict)]
    verification = verify_completed_answer(answer, hits)
    final_status = str(status or "done").strip().lower()
    if final_status not in {"done", "error", "canceled"}:
        final_status = "done"
    trace = AgentTrace(
        question_type=question_type,
        context=dict(scope_context or {}),
        plan=plan,
        verification=verification,
        status=final_status,
    )
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
