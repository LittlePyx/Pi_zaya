from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from kb.llm import DeepSeekChat
from kb.rag import build_messages
from kb.retrieval_engine import _group_hits_by_doc_for_refs, _search_hits_with_fallback
from kb.retriever import BM25Retriever
from kb.store import load_all_chunks

from .verifier import verify_answer_citations as _verify_answer_citations


def _clip(text: Any, limit: int = 180) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    return clean[:limit]


def _source_name(hit: dict[str, Any]) -> str:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    return (
        str(meta.get("source_name") or meta.get("title") or meta.get("source_path") or "").strip()
        or str(hit.get("id") or "").strip()
    )


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
    by_source: dict[str, dict[str, Any]] = {}
    for hit in list(hits or [])[:12]:
        source = _source_name(hit)
        if not source:
            continue
        rec = by_source.setdefault(source, {"source_name": source, "hit_count": 0, "evidence": []})
        rec["hit_count"] = int(rec["hit_count"]) + 1
        if len(rec["evidence"]) < 2:
            rec["evidence"].append(_clip(hit.get("text"), 180))
    comparisons = list(by_source.values())[:6]
    return {
        "comparisons": comparisons,
        "observation": f"Prepared comparison notes for {len(comparisons)} source(s).",
    }


def _fallback_grounded_answer(query: str, hits: list[dict[str, Any]], *, reason: str = "") -> str:
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
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> dict[str, Any]:
    if not getattr(settings, "text_api_key", None):
        answer = _fallback_grounded_answer(query, hits, reason="missing text API key")
        return {"answer": answer, "llm_used": False, "observation": "Generated degraded-mode answer without an LLM."}
    try:
        messages = build_messages(query, list(history or []), list(hits or []))
        answer = DeepSeekChat(settings).chat(
            messages=messages,
            temperature=float(temperature),
            max_tokens=max(256, min(4096, int(max_tokens or 1200))),
        )
        if not str(answer or "").strip():
            answer = _fallback_grounded_answer(query, hits, reason="empty LLM response")
            return {"answer": answer, "llm_used": False, "observation": "LLM returned empty text; used fallback answer."}
        return {"answer": str(answer).strip(), "llm_used": True, "observation": "Generated answer with existing RAG prompt."}
    except Exception as exc:
        answer = _fallback_grounded_answer(query, hits, reason=str(exc)[:160])
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
