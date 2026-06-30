from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from kb.llm import DeepSeekChat
from kb.rag import build_messages
from kb.retrieval_engine import _group_hits_by_doc_for_refs, _search_hits_with_fallback
from kb.retriever import BM25Retriever
from kb.store import load_all_chunks

from .verifier import verify_answer_citations as _verify_answer_citations


_METHOD_RE = re.compile(
    r"\b(method|approach|algorithm|model|architecture|training|pipeline|framework|network|implementation)\b"
    r"|(?:方法|算法|模型|架构|训练|流程|框架|网络|实现)",
    flags=re.IGNORECASE,
)
_LIMITATION_RE = re.compile(
    r"\b(limit|limitation|challenge|failure|weakness|future work|remain|however|although|but)\b"
    r"|(?:局限|限制|挑战|失败|不足|未来|然而|但是)",
    flags=re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,}|[\u4e00-\u9fff]{2,}")


def _clip(text: Any, limit: int = 180) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    return clean[:limit]


def _source_name(hit: dict[str, Any]) -> str:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    return (
        str(meta.get("source_name") or meta.get("title") or meta.get("source_path") or "").strip()
        or str(hit.get("id") or "").strip()
    )


def _hit_meta(hit: dict[str, Any]) -> dict[str, Any]:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    return meta if isinstance(meta, dict) else {}


def _heading(hit: dict[str, Any]) -> str:
    meta = _hit_meta(hit)
    return str(meta.get("heading_path") or meta.get("top_heading") or "").strip()


def _source_path(hit: dict[str, Any]) -> str:
    return str(_hit_meta(hit).get("source_path") or "").strip()


def _tokens(text: Any) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(str(text or ""))}


def _score(hit: dict[str, Any]) -> float:
    try:
        return float(hit.get("score") or 0.0)
    except Exception:
        return 0.0


def _evidence_row(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "heading_path": _heading(hit),
        "text_preview": _clip(hit.get("text"), 280),
        "score": _score(hit),
    }


def _first_matching_snippet(hits: list[dict[str, Any]], pattern: re.Pattern[str]) -> str:
    for hit in hits:
        text = " ".join([_heading(hit), str(hit.get("text") or "")])
        if pattern.search(text):
            snippet = _clip(hit.get("text"), 260)
            if snippet:
                return snippet
    return ""


def _relation_to_question(query: str, hits: list[dict[str, Any]]) -> str:
    query_terms = _tokens(query)
    if not query_terms:
        return "Retrieved as a candidate source for this comparison; review the evidence snippets."
    doc_terms: set[str] = set()
    for hit in hits[:6]:
        doc_terms |= _tokens(_heading(hit))
        doc_terms |= _tokens(hit.get("text"))
    overlap = sorted(query_terms & doc_terms)[:8]
    if overlap:
        return f"Matches the query through: {', '.join(overlap)}."
    return "Retrieved as a candidate source for this comparison; no strong lexical overlap was isolated."


def _format_agent_notes(agent_notes: dict[str, Any] | None) -> str:
    if not isinstance(agent_notes, dict) or not agent_notes:
        return ""
    compact: dict[str, Any] = {}
    if isinstance(agent_notes.get("comparisons"), list):
        compact["comparisons"] = list(agent_notes.get("comparisons") or [])[:6]
    if isinstance(agent_notes.get("guide"), list):
        compact["reading_guide"] = list(agent_notes.get("guide") or [])[:8]
    if isinstance(agent_notes.get("references"), list):
        compact["references"] = list(agent_notes.get("references") or [])[:8]
    if not compact:
        return ""
    try:
        text = json.dumps(compact, ensure_ascii=False, indent=2)
    except Exception:
        text = str(compact)
    return text[:5000]


def retrieve_evidence(query: str, *, db_dir: str | Path, settings: Any = None, top_k: int = 6) -> dict[str, Any]:
    chunks = load_all_chunks(Path(db_dir))
    retriever = BM25Retriever(chunks)
    hits, scores, used_query, used_translation, query_variants = _search_hits_with_fallback(
        query,
        retriever,
        max(1, min(20, int(top_k or 6))),
        settings,
        allow_translate=True,
        allow_expand=False,
    )
    hits = list(hits or [])[: max(1, min(20, int(top_k or 6)))]
    return {
        "hits": hits,
        "scores": list(scores or [])[: len(hits)],
        "used_query": used_query,
        "used_translation": bool(used_translation),
        "query_variants": list(query_variants or []),
        "observation": f"Retrieved {len(hits)} evidence snippet(s).",
    }


def retrieve_references(query: str, hits: list[dict[str, Any]], *, settings: Any = None, top_k: int = 6) -> dict[str, Any]:
    docs = _group_hits_by_doc_for_refs(
        list(hits or []),
        prompt_text=query,
        top_k_docs=max(1, min(10, int(top_k or 6))),
        deep_query=query,
        deep_read=False,
        llm_rerank=False,
        settings=settings,
    )
    references: list[dict[str, Any]] = []
    for doc in docs[: max(1, min(10, int(top_k or 6)))]:
        meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
        references.append(
            {
                "source_name": str(meta.get("source_name") or meta.get("title") or "").strip(),
                "source_path": str(meta.get("source_path") or "").strip(),
                "heading_path": str(meta.get("heading_path") or meta.get("ref_best_heading_path") or "").strip(),
                "score": float(doc.get("score") or meta.get("score") or 0.0),
            }
        )
    return {
        "references": references,
        "observation": f"Grouped evidence into {len(references)} reference source(s).",
    }


def build_reading_guide(query: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in list(hits or [])[:8]:
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        heading = str(meta.get("heading_path") or meta.get("top_heading") or "").strip()
        source = _source_name(hit)
        key = f"{source}|{heading}"
        if key in seen:
            continue
        seen.add(key)
        sections.append(
            {
                "source_name": source,
                "heading_path": heading,
                "why": _clip(hit.get("text"), 220),
            }
        )
    return {
        "guide": sections,
        "observation": f"Prepared {len(sections)} reading waypoint(s).",
    }


def compare_papers(query: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for hit in list(hits or [])[:12]:
        source = _source_name(hit)
        if not source:
            continue
        by_source.setdefault(source, []).append(hit)
    comparisons: list[dict[str, Any]] = []
    for source, source_hits in list(by_source.items())[:6]:
        ordered = sorted(source_hits, key=_score, reverse=True)
        method = _first_matching_snippet(ordered, _METHOD_RE)
        limitation = _first_matching_snippet(ordered, _LIMITATION_RE)
        headings: list[str] = []
        for hit in ordered[:4]:
            heading = _heading(hit)
            if heading and heading not in headings:
                headings.append(heading)
        comparisons.append(
            {
                "paper": source,
                "source_name": source,
                "source_path": _source_path(ordered[0]) if ordered else "",
                "method": method or "Not identified in the retrieved snippets.",
                "evidence": [_evidence_row(hit) for hit in ordered[:3]],
                "limitation": limitation or "Not identified in the retrieved snippets.",
                "relation_to_question": _relation_to_question(query, ordered),
                "supporting_headings": headings,
                "hit_count": len(source_hits),
            }
        )
    return {
        "comparisons": comparisons,
        "observation": f"Prepared structured comparison notes for {len(comparisons)} source(s).",
    }


def _fallback_grounded_answer(
    query: str,
    hits: list[dict[str, Any]],
    *,
    reason: str = "",
    agent_notes: dict[str, Any] | None = None,
) -> str:
    if not hits:
        suffix = f" Reason: {reason}" if reason else ""
        return (
            "No relevant indexed evidence was retrieved, so Research Agent Mode cannot produce "
            f"a paper-grounded answer yet.{suffix}"
        )
    lines = [
        "Research Agent Mode ran in degraded mode because no text LLM is configured.",
        "",
        "Evidence-backed notes:",
    ]
    notes_text = _format_agent_notes(agent_notes)
    if notes_text:
        lines.extend(["", "Structured agent notes:", notes_text])
    for idx, hit in enumerate(hits[:4], start=1):
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        source = _source_name(hit)
        heading = str(meta.get("heading_path") or meta.get("top_heading") or "").strip()
        location = f" ({heading})" if heading else ""
        lines.append(f"- [{idx}] {source}{location}: {_clip(hit.get('text'), 260)}")
    lines.append("")
    lines.append("Limits: this is a retrieved-evidence summary, not a synthesized LLM answer.")
    return "\n".join(lines).strip()


def generate_grounded_answer(
    query: str,
    hits: list[dict[str, Any]],
    *,
    settings: Any = None,
    history: list[dict[str, Any]] | None = None,
    agent_notes: dict[str, Any] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> dict[str, Any]:
    if not getattr(settings, "text_api_key", None):
        answer = _fallback_grounded_answer(query, hits, reason="missing text API key", agent_notes=agent_notes)
        return {"answer": answer, "llm_used": False, "observation": "Generated degraded-mode answer without an LLM."}
    try:
        notes_text = _format_agent_notes(agent_notes)
        answer_query = query
        if notes_text:
            answer_query = (
                f"{query}\n\n"
                "Research Agent structured notes. Use these as an evidence map for the answer; "
                "do not add claims that are not supported by the retrieved snippets:\n"
                f"{notes_text}"
            )
        messages = build_messages(answer_query, list(history or []), list(hits or []))
        answer = DeepSeekChat(settings).chat(
            messages=messages,
            temperature=float(temperature),
            max_tokens=max(256, min(4096, int(max_tokens or 1200))),
        )
        if not str(answer or "").strip():
            answer = _fallback_grounded_answer(query, hits, reason="empty LLM response", agent_notes=agent_notes)
            return {"answer": answer, "llm_used": False, "observation": "LLM returned empty text; used fallback answer."}
        return {"answer": str(answer).strip(), "llm_used": True, "observation": "Generated answer with existing RAG prompt."}
    except Exception as exc:
        answer = _fallback_grounded_answer(query, hits, reason=str(exc)[:160], agent_notes=agent_notes)
        return {
            "answer": answer,
            "llm_used": False,
            "error": str(exc)[:240],
            "observation": "LLM generation failed; used fallback answer.",
        }


def verify_answer_citations(answer: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    verification = _verify_answer_citations(answer, hits)
    return {
        "verification": verification.to_dict(),
        "observation": (
            f"Verified {verification.supported_claims}/{verification.total_claims} claim(s) "
            "with citation/evidence support."
        ),
    }
