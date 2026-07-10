from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any

from .planner import plan_research_intent, plan_research_question
from .research_run import build_research_run
from .source_policy import decide_answer_source
from .tools import (
    build_reading_guide,
    compare_papers,
    generate_grounded_answer,
    retrieve_evidence,
    retrieve_references,
    verify_answer_citations,
)
from .types import AgentExecutionStep, AgentTrace, AgentVerification, EvidenceStatus, QuestionType
from .verifier import verify_answer_citations as verify_completed_answer


_RESEARCH_OBJECT_RE = re.compile(
    r"\b(papers?|articles?|stud(?:y|ies)|literature|citations?|references?|bibliography|doi|arxiv|abstract|authors?)\b"
    r"|(?:\u8bba\u6587|\u6587\u732e|\u6587\u7ae0|\u8fd9\u7bc7|\u8be5\u6587|\u5f15\u7528|\u53c2\u8003\u6587\u732e|\u4f5c\u8005|\u6458\u8981|\u77e5\u8bc6\u5e93)",
    flags=re.IGNORECASE,
)
_RESEARCH_TASK_RE = re.compile(
    r"\b(methods?|approaches?|models?|contributions?|limitations?|experiments?|results?|datasets?|ablation|evaluation|architecture|section|figure|table|claim|prove|show|finding|findings|upstream|prior\s+work|summari[sz]e|explain)\b"
    r"|(?:\u65b9\u6cd5|\u8d21\u732e|\u5c40\u9650|\u5b9e\u9a8c|\u7ed3\u679c|\u6570\u636e\u96c6|\u6d88\u878d|\u8bc4\u4f30|\u7ae0\u8282|\u8bc1\u660e|\u4e0a\u6e38|\u603b\u7ed3|\u89e3\u91ca)",
    flags=re.IGNORECASE,
)
_ACADEMIC_DOMAIN_RE = re.compile(
    r"\b(transformers?|attention|diffusion|rag|retrieval|embeddings?|neural\s+networks?|deep\s+learning|machine\s+learning|llms?|large\s+language\s+models?|contrastive|reinforcement\s+learning|bayesian|causal|statistical|optimization)\b"
    r"|(?:\u6269\u6563\u6a21\u578b|\u6ce8\u610f\u529b|\u68c0\u7d22\u589e\u5f3a|\u5411\u91cf|\u795e\u7ecf\u7f51\u7edc|\u6df1\u5ea6\u5b66\u4e60|\u673a\u5668\u5b66\u4e60|\u5927\u8bed\u8a00\u6a21\u578b|\u56e0\u679c|\u8d1d\u53f6\u65af|\u4f18\u5316)",
    flags=re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}")
_CONFIDENCE_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "answer",
    "based",
    "before",
    "between",
    "compare",
    "does",
    "doing",
    "from",
    "give",
    "have",
    "into",
    "just",
    "know",
    "like",
    "main",
    "make",
    "method",
    "methods",
    "paper",
    "papers",
    "please",
    "read",
    "really",
    "show",
    "shows",
    "study",
    "tell",
    "that",
    "their",
    "there",
    "these",
    "this",
    "what",
    "where",
    "which",
    "while",
    "with",
    "work",
    "works",
}


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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value or 0.0)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _confidence_token(token: str) -> str:
    text = str(token or "").strip().lower()
    if not text:
        return ""
    if re.fullmatch(r"[a-z][a-z0-9_-]+", text):
        if len(text) > 4 and text.endswith("ies"):
            text = f"{text[:-3]}y"
        elif len(text) > 4 and text.endswith("s"):
            text = text[:-1]
    return text


def _confidence_terms(text: object) -> set[str]:
    terms: set[str] = set()
    for raw in _TOKEN_RE.findall(str(text or "")):
        token = _confidence_token(raw)
        if not token or token in _CONFIDENCE_STOPWORDS:
            continue
        if len(token) < 3 and not re.search(r"[\u4e00-\u9fff]", token):
            continue
        terms.add(token)
    return terms


def _nested_numeric(meta: dict[str, Any], *path: str) -> float:
    current: Any = meta
    for key in path:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    return _safe_float(current)


def _hit_score(hit: dict[str, Any]) -> float:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    scores = [
        _safe_float(hit.get("score")),
        _safe_float(meta.get("score")),
        _safe_float(meta.get("display_score")),
        _nested_numeric(meta, "ref_rank", "display_score"),
        _nested_numeric(meta, "ref_rank", "score"),
    ]
    return max(scores or [0.0])


def _meta_confidence_signal(meta: dict[str, Any]) -> bool:
    numeric_signals = [
        _safe_float(meta.get("direct_prompt_match_score")),
        _safe_float(meta.get("explicit_doc_match_score")),
        _safe_float(meta.get("anchor_match_score")),
        _safe_float(meta.get("evidence_confidence")),
    ]
    if max(numeric_signals or [0.0]) >= 4.0:
        return True
    ref_rank = meta.get("ref_rank") if isinstance(meta.get("ref_rank"), dict) else {}
    rank_signals = [
        _safe_float(ref_rank.get("display_score")),
        _safe_float(ref_rank.get("score")),
        _safe_float(ref_rank.get("llm")),
    ]
    if max(rank_signals or [0.0]) >= 70.0:
        return True
    if str(meta.get("ref_loc_quality") or "").strip().lower() == "high":
        return True
    return False


def _hit_confidence_blob(hit: dict[str, Any]) -> str:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    fields = [
        hit.get("text"),
        meta.get("heading_path"),
        meta.get("top_heading"),
        meta.get("ref_best_heading_path"),
        meta.get("source_name"),
        meta.get("title"),
    ]
    return " ".join(str(x or "") for x in fields if str(x or "").strip())


def _hit_confidence_row(
    query_terms: set[str],
    hit: dict[str, Any],
    *,
    scoped_source_requested: bool = False,
) -> dict[str, Any]:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    hit_terms = _confidence_terms(_hit_confidence_blob(hit))
    overlap = sorted(query_terms & hit_terms)
    score = _hit_score(hit)
    meta_signal = _meta_confidence_signal(meta)
    scope_signal = bool(scoped_source_requested and score >= 0.5)
    useful = bool(
        len(overlap) >= 2
        or (len(overlap) >= 1 and score >= 1.0)
        or score >= 8.0
        or meta_signal
        or scope_signal
    )
    return {
        "source_name": str(meta.get("source_name") or meta.get("title") or "").strip(),
        "source_path": str(meta.get("source_path") or "").strip(),
        "heading_path": str(meta.get("heading_path") or meta.get("top_heading") or "").strip(),
        "score": round(score, 4),
        "query_overlap_count": len(overlap),
        "query_overlap_terms": overlap[:8],
        "metadata_signal": meta_signal,
        "scope_signal": scope_signal,
        "useful": useful,
    }


def _assess_retrieval_confidence(
    query: str,
    hits: list[dict[str, Any]],
    *,
    scope_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_hits = [hit for hit in list(hits or []) if isinstance(hit, dict)]
    query_terms = _confidence_terms(query)
    if not candidate_hits:
        return {
            "level": "none",
            "candidate_hit_count": 0,
            "usable_hit_count": 0,
            "low_confidence_hit_count": 0,
            "query_term_count": len(query_terms),
            "top_score": 0.0,
            "max_query_overlap": 0,
            "reasons": ["no_candidate_hits"],
            "signals": [],
            "usable_hits": [],
        }

    scope = str((scope_context or {}).get("query_scope") or "").strip()
    scoped_source_requested = scope in {"current_paper", "basket"}
    rows = [
        _hit_confidence_row(
            query_terms,
            hit,
            scoped_source_requested=scoped_source_requested,
        )
        for hit in candidate_hits
    ]
    usable_pairs = [(hit, row) for hit, row in zip(candidate_hits, rows) if row.get("useful")]
    usable_hits = [hit for hit, _row in usable_pairs]
    top_score = max((_safe_float(row.get("score")) for row in rows), default=0.0)
    max_overlap = max((int(row.get("query_overlap_count") or 0) for row in rows), default=0)
    reasons: list[str] = []
    if len(usable_hits) < len(candidate_hits):
        reasons.append("filtered_low_confidence_hits")
    if not query_terms:
        reasons.append("no_query_terms")
    if not usable_hits:
        reasons.append("low_retrieval_confidence")
        if max_overlap <= 0:
            reasons.append("no_query_overlap")
        if top_score < 1.0:
            reasons.append("low_retrieval_score")
        return {
            "level": "low",
            "candidate_hit_count": len(candidate_hits),
            "usable_hit_count": 0,
            "low_confidence_hit_count": len(candidate_hits),
            "query_term_count": len(query_terms),
            "top_score": round(top_score, 4),
            "max_query_overlap": max_overlap,
            "reasons": reasons,
            "signals": rows[:6],
            "usable_hits": [],
        }

    if len(usable_hits) >= 2 and (top_score >= 4.0 or max_overlap >= 2 or any(row.get("metadata_signal") for row in rows)):
        level = "high"
    else:
        level = "medium"
    return {
        "level": level,
        "candidate_hit_count": len(candidate_hits),
        "usable_hit_count": len(usable_hits),
        "low_confidence_hit_count": max(0, len(candidate_hits) - len(usable_hits)),
        "query_term_count": len(query_terms),
        "top_score": round(top_score, 4),
        "max_query_overlap": max_overlap,
        "reasons": reasons,
        "signals": rows[:6],
        "usable_hits": usable_hits,
    }


def _public_retrieval_confidence(confidence: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in dict(confidence or {}).items() if k != "usable_hits"}


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


def _looks_like_research_grounding_request(
    query: str,
    *,
    scope_context: dict[str, Any],
    question_type: QuestionType,
) -> bool:
    text = str(query or "").strip()
    if not text:
        return False
    has_research_object = bool(_RESEARCH_OBJECT_RE.search(text))
    if has_research_object:
        return True
    has_research_task = bool(_RESEARCH_TASK_RE.search(text))
    scope = str(scope_context.get("query_scope") or "").strip()
    requested_scope = str(scope_context.get("requested_query_scope") or "").strip()
    has_scoped_source = bool(
        str(scope_context.get("current_source_path") or "").strip()
        or str(scope_context.get("current_source_name") or "").strip()
        or int(scope_context.get("selected_research_context_count") or 0) > 0
    )
    if scope in {"current_paper", "basket"} and has_scoped_source and has_research_task:
        return True
    if requested_scope in {"current_paper", "basket", "library"} and has_research_task:
        return True
    if question_type in {"reference_followup", "reading_guide"} and has_research_task:
        return True
    return False


def _looks_like_academic_question(query: str, *, question_type: QuestionType) -> bool:
    text = str(query or "").strip()
    if not text:
        return False
    if question_type in {"reading_guide", "reference_followup"}:
        return True
    if _RESEARCH_OBJECT_RE.search(text) or _RESEARCH_TASK_RE.search(text) or _ACADEMIC_DOMAIN_RE.search(text):
        return True
    return False


def _pre_generation_evidence_gate(
    hits: list[dict[str, Any]],
    *,
    query: str = "",
    scope_context: dict[str, Any] | None = None,
    question_type: QuestionType = "unknown",
    retrieval_confidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hit_count = len([hit for hit in list(hits or []) if isinstance(hit, dict)])
    confidence = _public_retrieval_confidence(dict(retrieval_confidence or {}))
    confidence_level = str(confidence.get("level") or "").strip()
    confidence_reasons = [
        str(reason or "").strip()
        for reason in list(confidence.get("reasons") or [])
        if str(reason or "").strip()
    ]
    candidate_hit_count = int(confidence.get("candidate_hit_count") or hit_count or 0)
    decision = decide_answer_source(
        hit_count=hit_count,
        candidate_hit_count=candidate_hit_count,
        retrieval_confidence=confidence_level,
        retrieval_reasons=confidence_reasons,
        academic_question=_looks_like_academic_question(query, question_type=question_type),
        local_grounding_requested=_looks_like_research_grounding_request(
            query,
            scope_context=scope_context or {},
            question_type=question_type,
        ),
    )
    return decision.to_evidence_gate()


def _is_general_answer_mode(agent_notes: dict[str, Any] | None) -> bool:
    if not isinstance(agent_notes, dict):
        return False
    gate = agent_notes.get("evidence_gate")
    return isinstance(gate, dict) and str(gate.get("answer_mode") or "") in {
        "general_llm",
        "external_academic_llm",
    }


def _answer_mode(agent_notes: dict[str, Any] | None) -> str:
    if not isinstance(agent_notes, dict):
        return ""
    gate = agent_notes.get("evidence_gate")
    if not isinstance(gate, dict):
        return ""
    return str(gate.get("answer_mode") or "").strip()


def _pre_answer_evidence_status(agent_notes: dict[str, Any] | None, hits: list[dict[str, Any]]) -> EvidenceStatus:
    gate = agent_notes.get("evidence_gate") if isinstance(agent_notes, dict) else {}
    status = str(gate.get("evidence_status") or "").strip() if isinstance(gate, dict) else ""
    if status in {"grounded", "needs_review", "insufficient", "not_applicable"}:
        return status  # type: ignore[return-value]
    return "needs_review" if hits else "not_applicable"


def _attach_pre_answer_research_context(
    context: dict[str, Any],
    *,
    query: str,
    question_type: QuestionType,
    scope_context: dict[str, Any],
) -> None:
    notes = context.get("agent_notes")
    if not isinstance(notes, dict):
        notes = {}
        context["agent_notes"] = notes
    hits = [hit for hit in list(context.get("hits") or []) if isinstance(hit, dict)]
    pre_status = _pre_answer_evidence_status(notes, hits)
    run = build_research_run(
        query,
        question_type=question_type,
        hits=hits,
        agent_notes=notes,
        scope_context=scope_context,
        verification_status=pre_status,
        status="synthesizing",
    )
    payload = run.to_dict()
    notes["evidence_matrix"] = list(payload.get("evidence_matrix") or [])
    notes["research_run"] = {
        "run_id": str(payload.get("run_id") or ""),
        "status": str(payload.get("status") or ""),
        "source_policy": str(payload.get("source_policy") or ""),
        "query_scope": str(payload.get("query_scope") or ""),
        "metrics": dict(payload.get("metrics") or {}),
    }
    scope_context["pre_answer_evidence_matrix_rows"] = len(notes["evidence_matrix"])
    scope_context["pre_answer_source_policy"] = notes["research_run"]["source_policy"]


def _general_answer_verification() -> AgentVerification:
    return AgentVerification(
        evidence_status="not_applicable",
        evidence_hit_count=0,
        evidence_status_reasons=["not_based_on_local_knowledge_base"],
    )


def _hybrid_generation_recommended(
    gate: dict[str, Any],
    confidence: dict[str, Any],
    hits: list[dict[str, Any]],
) -> bool:
    if str(gate.get("answer_mode") or "").strip() != "hybrid_local_external":
        return False
    if str(gate.get("source_policy") or "").strip() != "local_plus_external_background":
        return False
    reasons = {
        str(item or "").strip()
        for item in [*list(gate.get("reasons") or []), *list(confidence.get("reasons") or [])]
        if str(item or "").strip()
    }
    if len([hit for hit in list(hits or []) if isinstance(hit, dict)]) < 2:
        return True
    if str(confidence.get("level") or "").strip() in {"low", "none"}:
        return True
    return bool(
        reasons
        & {
            "low_evidence_count",
            "low_retrieval_confidence",
            "filtered_low_confidence_hits",
            "no_usable_local_evidence",
        }
    )


def build_generation_agent_notes(
    query: str,
    *,
    evidence_hits: list[dict[str, Any]] | None = None,
    candidate_hits: list[dict[str, Any]] | None = None,
    scope_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the agent evidence gate used by the legacy streaming generator.

    This keeps `/api/generate` on the existing RAG path while giving it the
    same source-policy decision that the standalone research-agent endpoint
    uses before answer generation.
    """
    intent = plan_research_intent(query)
    question_type, _plan = plan_research_question(query)
    scoped_context = dict(scope_context or {})
    hits = [hit for hit in list(evidence_hits or []) if isinstance(hit, dict)]
    candidates = [hit for hit in list(candidate_hits or []) if isinstance(hit, dict)]
    confidence_source = hits if hits else candidates
    retrieval_confidence = _assess_retrieval_confidence(
        query,
        confidence_source,
        scope_context=scoped_context,
    )
    gate = _pre_generation_evidence_gate(
        hits,
        query=query,
        scope_context=scoped_context,
        question_type=question_type,
        retrieval_confidence=retrieval_confidence,
    )
    public_confidence = _public_retrieval_confidence(retrieval_confidence)
    agent_notes: dict[str, Any] = {"evidence_gate": gate}
    research_run = build_research_run(
        query,
        question_type=question_type,
        hits=hits,
        agent_notes=agent_notes,
        scope_context=scoped_context,
        verification_status=_pre_answer_evidence_status(agent_notes, hits),
        status="synthesizing",
    )
    research_run_payload = research_run.to_dict()
    agent_notes["evidence_matrix"] = list(research_run_payload.get("evidence_matrix") or [])
    agent_notes["research_run"] = {
        "run_id": str(research_run_payload.get("run_id") or ""),
        "status": str(research_run_payload.get("status") or ""),
        "source_policy": str(research_run_payload.get("source_policy") or ""),
        "query_scope": str(research_run_payload.get("query_scope") or ""),
        "metrics": dict(research_run_payload.get("metrics") or {}),
    }
    hybrid_generation = _hybrid_generation_recommended(gate, public_confidence, hits)
    return {
        "question_type": question_type,
        "planner_intent": intent.to_dict(),
        "agent_notes": agent_notes,
        "retrieval_confidence": public_confidence,
        "hybrid_generation_recommended": bool(hybrid_generation),
        "context": {
            "planner_intent": intent.to_dict(),
            "planner_confidence": intent.confidence,
            "evidence_need": intent.evidence_need,
            "retrieved_hit_count": int(len(candidates)),
            "usable_hit_count": int(len(hits)),
            "retrieval_confidence": str(public_confidence.get("level") or ""),
            "answer_source_blend": str(gate.get("source_blend") or ""),
            "answer_mode": str(gate.get("answer_mode") or ""),
            "source_policy": str(gate.get("source_policy") or ""),
            "hybrid_generation_recommended": bool(hybrid_generation),
            "pre_answer_evidence_matrix_rows": len(agent_notes["evidence_matrix"]),
            "pre_answer_source_policy": agent_notes["research_run"]["source_policy"],
        },
    }


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


_FINAL_TRACE_STATUSES = {"done", "error", "canceled"}


def _normalize_trace_status(value: object, *, fallback: str = "done") -> str:
    status = str(value or "").strip().lower()
    return status if status in _FINAL_TRACE_STATUSES else fallback


def _finalize_agent_trace(
    trace: AgentTrace,
    query: str,
    *,
    hits: list[dict[str, Any]],
    agent_notes: dict[str, Any] | None,
    scope_context: dict[str, Any] | None,
    status: str = "done",
) -> None:
    context = trace.context
    if isinstance(scope_context, dict):
        for key, value in scope_context.items():
            context.setdefault(key, value)
    gate = agent_notes.get("evidence_gate") if isinstance(agent_notes, dict) else None
    if isinstance(gate, dict):
        for context_key, gate_key in (
            ("answer_source_blend", "source_blend"),
            ("answer_mode", "answer_mode"),
            ("source_policy", "source_policy"),
            ("retrieval_confidence", "retrieval_confidence"),
        ):
            value = str(gate.get(gate_key) or "").strip()
            if value:
                context.setdefault(context_key, value)
    final_status = _normalize_trace_status(status)
    if final_status == "done" and trace.errors:
        final_status = "error"
    trace.status = final_status
    trace.research_run = build_research_run(
        query,
        question_type=trace.question_type,
        hits=hits,
        agent_notes=agent_notes or {},
        scope_context=context,
        verification_status=trace.verification.evidence_status,
        failed=final_status in {"error", "canceled"},
    )


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
    intent = plan_research_intent(query)
    question_type, plan = plan_research_question(query)
    scope_context = _build_scope_context(
        query_scope=query_scope,
        selected_research_context=selected_research_context,
        current_source_path=current_source_path,
        current_source_name=current_source_name,
        source="direct_research_agent",
    )
    scope_context["planner_intent"] = intent.to_dict()
    scope_context["planner_confidence"] = intent.confidence
    scope_context["evidence_need"] = intent.evidence_need
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
            retrieval_confidence = _assess_retrieval_confidence(
                query,
                scoped_hits,
                scope_context=scope_context,
            )
            context["hits"] = [
                hit for hit in list(retrieval_confidence.get("usable_hits") or []) if isinstance(hit, dict)
            ]
            trace.context["retrieved_hit_count"] = len(raw_hits)
            trace.context["scoped_hit_count"] = len(scoped_hits)
            trace.context["usable_hit_count"] = len(context["hits"])
            trace.context["retrieval_confidence"] = str(retrieval_confidence.get("level") or "")
            if trace.steps:
                trace.steps[-1].output["hits"] = _summarize_hits(context["hits"])
                trace.steps[-1].output["retrieval_confidence"] = _public_retrieval_confidence(retrieval_confidence)
                if len(context["hits"]) < len(scoped_hits):
                    trace.steps[-1].output["candidate_hits"] = _summarize_hits(scoped_hits)
                if scope_filter.get("active"):
                    trace.steps[-1].output["scope_filter"] = scope_filter
            context["agent_notes"]["evidence_gate"] = _pre_generation_evidence_gate(
                context["hits"],
                query=query,
                scope_context=scope_context,
                question_type=question_type,
                retrieval_confidence=retrieval_confidence,
            )
            gate = context["agent_notes"]["evidence_gate"]
            if isinstance(gate, dict):
                trace.context["answer_source_blend"] = str(gate.get("source_blend") or "")
                trace.context["answer_mode"] = str(gate.get("answer_mode") or "")
                trace.context["source_policy"] = str(gate.get("source_policy") or "")
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
            _attach_pre_answer_research_context(
                context,
                query=query,
                question_type=question_type,
                scope_context=scope_context,
            )
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
            if _is_general_answer_mode(context.get("agent_notes")):
                plan_step.status = "skipped"
                trace.verification = _general_answer_verification()
                trace.steps.append(
                    AgentExecutionStep(
                        tool=plan_step.tool,
                        status="skipped",
                        observation="Skipped local citation verification because this answer was not grounded in local knowledge-base evidence.",
                        output=trace.verification.to_dict(),
                    )
                )
                continue
            answer_mode = _answer_mode(context.get("agent_notes"))
            result = _run_step(
                trace,
                idx,
                verify_answer_citations,
                context["answer"],
                context["hits"],
                answer_mode=answer_mode,
            )
            trace.verification = verify_completed_answer(
                context["answer"],
                context["hits"],
                answer_mode=answer_mode,
            )
            if isinstance(result.get("verification"), dict):
                # Keep the dataclass path authoritative while preserving tool output.
                pass
    if not context["answer"]:
        context["answer"] = "Research Agent Mode could not generate an answer. See agent_trace.errors for details."
    _finalize_agent_trace(
        trace,
        query,
        hits=context["hits"],
        agent_notes=context.get("agent_notes"),
        scope_context=scope_context,
        status="error" if trace.errors else "done",
    )
    return {"answer": context["answer"], "agent_trace": trace.to_dict(), "hits": context["hits"]}


def build_agent_trace_for_completed_answer(
    query: str,
    answer: str,
    *,
    evidence_hits: list[dict[str, Any]] | None = None,
    status: str = "done",
    scope_context: dict[str, Any] | None = None,
    agent_notes: dict[str, Any] | None = None,
    answer_mode: str = "",
    generation_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    intent = plan_research_intent(query)
    question_type, plan = plan_research_question(query)
    hits = [h for h in list(evidence_hits or []) if isinstance(h, dict)]
    resolved_answer_mode = str(answer_mode or _answer_mode(agent_notes) or "").strip()
    if resolved_answer_mode in {"general_llm", "external_academic_llm"}:
        verification = _general_answer_verification()
    else:
        verification = verify_completed_answer(answer, hits, answer_mode=resolved_answer_mode)
    final_status = _normalize_trace_status(status)
    context = dict(scope_context or {})
    context.setdefault("planner_intent", intent.to_dict())
    context.setdefault("planner_confidence", intent.confidence)
    context.setdefault("evidence_need", intent.evidence_need)
    trace = AgentTrace(
        question_type=question_type,
        context=context,
        plan=plan,
        verification=verification,
        status=final_status,
    )
    for step in trace.plan:
        if step.tool == "verify_answer_citations" and resolved_answer_mode in {"general_llm", "external_academic_llm"}:
            step.status = "skipped"
        else:
            step.status = "done" if final_status == "done" else final_status
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
            observation=str(
                (generation_output or {}).get("observation")
                or "Existing RAG generation produced the answer; agent trace was attached as an explicit wrapper."
            ),
            output={
                "answer_chars": len(str(answer or "")),
                **{
                    key: value
                    for key, value in dict(generation_output or {}).items()
                    if key not in {"answer", "hits"}
                },
            },
        )
    )
    trace.steps.append(
        AgentExecutionStep(
            tool="verify_answer_citations",
            status="skipped" if resolved_answer_mode in {"general_llm", "external_academic_llm"} else final_status,
            observation=(
                "Skipped local citation verification because this answer was not grounded in local knowledge-base evidence."
                if resolved_answer_mode in {"general_llm", "external_academic_llm"}
                else (
                    f"Verified {verification.supported_claims}/{verification.total_claims} claim(s) "
                    "with citation/evidence support."
                )
            ),
            output=verification.to_dict(),
        )
    )
    _finalize_agent_trace(
        trace,
        query,
        hits=hits,
        agent_notes=agent_notes,
        scope_context=context,
        status=final_status,
    )
    return trace.to_dict()
