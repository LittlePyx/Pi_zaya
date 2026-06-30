from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from kb.answer_presentation import clean_assistant_answer_presentation_text
from kb.inpaper_citation_grounding import extract_candidate_ref_nums_from_hits, parse_ref_num_set
from kb.llm import DeepSeekChat
from kb.rag import build_messages
from kb.reference_index import load_reference_index, resolve_reference_entry
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


def _positive_int(value: Any) -> int:
    try:
        n = int(value)
    except Exception:
        return 0
    return n if n > 0 else 0


def _append_ref_num(out: list[int], seen: set[int], value: Any, *, limit: int) -> None:
    if len(out) >= max(1, int(limit)):
        return
    if isinstance(value, dict):
        for key in ("ref_num", "reference_num", "resolved_ref_num", "top_ref_num", "num"):
            _append_ref_num(out, seen, value.get(key), limit=limit)
            if len(out) >= max(1, int(limit)):
                return
        for key in ("candidate_refs", "support_ref_candidates", "ref_nums"):
            _append_ref_num(out, seen, value.get(key), limit=limit)
            if len(out) >= max(1, int(limit)):
                return
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _append_ref_num(out, seen, item, limit=limit)
            if len(out) >= max(1, int(limit)):
                return
        return
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return
        parsed = parse_ref_num_set(text.strip("[](){} "), max_items=max(1, int(limit)))
        if not parsed:
            parsed = [_positive_int(m.group(1)) for m in re.finditer(r"\b(\d{1,4})\b", text)]
        for n in parsed:
            _append_ref_num(out, seen, n, limit=limit)
            if len(out) >= max(1, int(limit)):
                return
        return
    n = _positive_int(value)
    if n <= 0 or n in seen:
        return
    seen.add(n)
    out.append(n)


def _candidate_ref_nums_for_source(
    hits: list[dict[str, Any]],
    *,
    source_path: str,
    limit: int,
) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for n in extract_candidate_ref_nums_from_hits(
        list(hits or []),
        source_path=source_path,
        max_candidates=max(1, int(limit)),
    ):
        _append_ref_num(out, seen, n, limit=limit)
        if len(out) >= max(1, int(limit)):
            return out
    for hit in list(hits or []):
        meta = _hit_meta(hit)
        src = str(meta.get("source_path") or "").strip()
        if source_path and src and src != source_path:
            continue
        for key in (
            "resolved_ref_num",
            "top_ref_num",
            "ref_num",
            "reference_num",
            "candidate_refs",
            "support_ref_candidates",
            "ref_nums",
        ):
            _append_ref_num(out, seen, meta.get(key), limit=limit)
            if len(out) >= max(1, int(limit)):
                return out
    return out


def _doc_ref_nums_from_index(
    index_data: dict[str, Any],
    *,
    source_path: str,
    source_sha1: str = "",
    limit: int = 6,
) -> list[int]:
    docs = index_data.get("docs") if isinstance(index_data, dict) else {}
    if not isinstance(docs, dict):
        return []
    want_path = str(source_path or "").strip().lower()
    want_name = Path(str(source_path or "")).name.lower()
    want_stem = Path(str(source_path or "")).stem.lower()
    want_sha1 = str(source_sha1 or "").strip().lower()
    candidates: list[tuple[int, dict[str, Any]]] = []
    for raw_doc in docs.values():
        if not isinstance(raw_doc, dict):
            continue
        score = 0
        doc_path = str(raw_doc.get("path") or "").strip().lower()
        doc_name = str(raw_doc.get("name") or "").strip().lower()
        doc_stem = str(raw_doc.get("stem") or "").strip().lower()
        doc_sha1 = str(raw_doc.get("sha1") or "").strip().lower()
        if want_sha1 and doc_sha1 and want_sha1 == doc_sha1:
            score = max(score, 5)
        if want_path and doc_path and want_path == doc_path:
            score = max(score, 4)
        if want_name and doc_name and want_name == doc_name:
            score = max(score, 3)
        if want_stem and doc_stem and want_stem == doc_stem:
            score = max(score, 2)
        if score <= 0:
            continue
        candidates.append((score, raw_doc))
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _score_value, doc in candidates[:1]:
        refs = doc.get("refs") if isinstance(doc, dict) else {}
        if not isinstance(refs, dict):
            continue
        nums = sorted((_positive_int(k) for k in refs.keys()), key=int)
        return [n for n in nums if n > 0][: max(1, int(limit))]
    return []


def _load_reference_index_safely(db_dir: str | Path | None, settings: Any = None) -> dict[str, Any]:
    raw_dir = db_dir if db_dir is not None else getattr(settings, "db_dir", None)
    if not raw_dir:
        return {}
    try:
        return load_reference_index(Path(raw_dir).expanduser())
    except Exception:
        return {}


def _source_summary_row(doc: dict[str, Any]) -> dict[str, Any]:
    meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    source_path = str(meta.get("source_path") or "").strip()
    return {
        "source_name": str(meta.get("source_name") or meta.get("title") or Path(source_path).name).strip(),
        "source_path": source_path,
        "heading_path": str(meta.get("heading_path") or meta.get("ref_best_heading_path") or meta.get("top_heading") or "").strip(),
        "evidence_preview": _clip(doc.get("text") or " ".join(list(meta.get("ref_show_snippets") or [])[:1]), 320),
        "score": float(doc.get("score") or meta.get("score") or 0.0),
        "reference_index_available": False,
    }


def _reference_relation_to_question(query: str, ref: dict[str, Any], evidence_preview: str) -> str:
    query_terms = _tokens(query)
    ref_text = " ".join(
        str(ref.get(key) or "")
        for key in ("title", "authors", "venue", "year", "doi", "raw")
    )
    overlap = sorted(query_terms & (_tokens(ref_text) | _tokens(evidence_preview)))[:8]
    if overlap:
        return f"Matches the query or citing context through: {', '.join(overlap)}."
    if evidence_preview:
        return "Selected from retrieved citing-paper evidence near an in-text reference marker."
    return "Selected from the citing paper's local reference index."


def _resolved_reference_row(
    resolved: dict[str, Any],
    *,
    query: str,
    doc: dict[str, Any],
    evidence_preview: str,
) -> dict[str, Any]:
    meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    ref = resolved.get("ref") if isinstance(resolved.get("ref"), dict) else {}
    ref_num = _positive_int(resolved.get("ref_num") or ref.get("num"))
    title = str(ref.get("title") or "").strip()
    raw = str(ref.get("raw") or "").strip()
    source_path = str(resolved.get("source_path") or meta.get("source_path") or "").strip()
    source_name = str(resolved.get("source_name") or meta.get("source_name") or Path(source_path).name).strip()
    heading = str(meta.get("heading_path") or meta.get("ref_best_heading_path") or meta.get("top_heading") or "").strip()
    page_start = _positive_int(meta.get("page_start"))
    page_end = _positive_int(meta.get("page_end"))
    anchor_seed = re.sub(r"[^A-Za-z0-9_-]+", "-", f"{source_path}-{ref_num}")[:80].strip("-")
    anchor = f"agent-ref-{anchor_seed or ref_num}"
    relation = _reference_relation_to_question(query, ref, evidence_preview)
    return {
        "anchor": anchor,
        "num": ref_num,
        "is_inpaper": True,
        "source_name": source_name,
        "source_path": source_path,
        "source_paper": source_name,
        "heading_path": heading,
        "ref_num": ref_num,
        "title": title,
        "authors": str(ref.get("authors") or "").strip(),
        "year": str(ref.get("year") or "").strip(),
        "venue": str(ref.get("venue") or "").strip(),
        "doi": str(ref.get("doi") or "").strip(),
        "doi_url": str(ref.get("doi_url") or "").strip(),
        "raw": _clip(raw, 520),
        "cite_fmt": _clip(raw, 520),
        "evidence_preview": _clip(evidence_preview, 320),
        "evidence_quote": _clip(evidence_preview, 320),
        "why_relevant": relation,
        "citation_context": _clip(evidence_preview, 320),
        "citation_context_source": "agent_trace",
        "upstream_work_role": relation,
        "user_question_relation": relation,
        "why_line": relation,
        "support_relation": relation,
        "location_label": heading,
        "shelf_item_kind": "reference",
        "shelf_origin": "agent_trace",
        "shelf_excerpt": _clip(raw or title, 520),
        "shelf_excerpt_label": "Reference entry",
        "card_kind": "reference",
        "card_title": title,
        "card_subtitle": " / ".join([x for x in [str(ref.get("authors") or "").strip(), str(ref.get("year") or "").strip(), str(ref.get("venue") or "").strip()] if x]),
        "card_reference_entry": _clip(raw, 520),
        "card_context_summary": relation,
        "card_evidence": _clip(evidence_preview, 320),
        "card_locator": heading,
        "card_support_explanation": relation,
        "page_start": page_start,
        "page_end": page_end,
        "score": float(doc.get("score") or meta.get("score") or 0.0),
        "reference_index_available": True,
        "metadata_status": str(ref.get("metadata_status") or "").strip(),
    }


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
    if isinstance(agent_notes.get("evidence_gate"), dict):
        gate = agent_notes.get("evidence_gate") or {}
        compact["evidence_gate"] = {
            "evidence_status": str(gate.get("evidence_status") or "").strip(),
            "evidence_hit_count": _positive_int(gate.get("evidence_hit_count")),
            "reasons": list(gate.get("reasons") or [])[:4] if isinstance(gate.get("reasons"), list) else [],
            "instruction": _clip(gate.get("instruction"), 220),
        }
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


def retrieve_references(
    query: str,
    hits: list[dict[str, Any]],
    *,
    db_dir: str | Path | None = None,
    settings: Any = None,
    top_k: int = 6,
) -> dict[str, Any]:
    limit = max(1, min(10, int(top_k or 6)))
    docs = _group_hits_by_doc_for_refs(
        list(hits or []),
        prompt_text=query,
        top_k_docs=limit,
        deep_query=query,
        deep_read=False,
        llm_rerank=False,
        settings=settings,
    )
    index_data = _load_reference_index_safely(db_dir, settings=settings)
    index_available = bool(isinstance(index_data.get("docs"), dict) and index_data.get("docs"))
    references: list[dict[str, Any]] = []
    fallback_sources: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for doc in docs[:limit]:
        meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
        source_path = str(meta.get("source_path") or "").strip()
        source_sha1 = str(meta.get("source_sha1") or "").strip()
        fallback_sources.append(_source_summary_row(doc))
        if not source_path or not index_available:
            continue
        ref_limit = max(limit * 2, 6)
        ref_nums = _candidate_ref_nums_for_source(list(hits or []), source_path=source_path, limit=ref_limit)
        if not ref_nums:
            ref_nums = _doc_ref_nums_from_index(
                index_data,
                source_path=source_path,
                source_sha1=source_sha1,
                limit=min(3, limit),
            )
        evidence_preview = _clip(doc.get("text") or " ".join(list(meta.get("ref_show_snippets") or [])[:1]), 360)
        for ref_num in ref_nums:
            resolved = resolve_reference_entry(index_data, source_path, int(ref_num), source_sha1=source_sha1)
            if not isinstance(resolved, dict):
                continue
            resolved_source_path = str(resolved.get("source_path") or source_path).strip().lower()
            key = (resolved_source_path, _positive_int(resolved.get("ref_num") or ref_num))
            if key in seen or key[1] <= 0:
                continue
            seen.add(key)
            references.append(
                _resolved_reference_row(
                    resolved,
                    query=query,
                    doc=doc,
                    evidence_preview=evidence_preview,
                )
            )
            if len(references) >= limit:
                break
        if len(references) >= limit:
            break
    if references:
        observation = f"Resolved {len(references)} upstream reference(s) from {len(docs)} citing source paper(s)."
    elif index_available:
        references = fallback_sources[:limit]
        observation = "Reference index was available, but no concrete upstream reference matched the retrieved evidence."
    else:
        references = fallback_sources[:limit]
        observation = f"Grouped evidence into {len(references)} reference source(s); reference index was unavailable."
    return {
        "references": references,
        "reference_index_available": index_available,
        "source_count": len(docs),
        "resolved_reference_count": len([r for r in references if r.get("reference_index_available") is True]),
        "observation": observation,
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
            "No relevant indexed evidence was retrieved, so I cannot produce "
            f"a paper-grounded answer yet.{suffix}"
        )
    lines = [
        "No text LLM is configured; showing retrieved evidence notes.",
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
    if not hits:
        answer = _fallback_grounded_answer(query, hits, reason="no retrieved evidence", agent_notes=agent_notes)
        return {"answer": answer, "llm_used": False, "observation": "Skipped LLM answer because no indexed evidence was retrieved."}
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
                "do not add claims that are not supported by the retrieved snippets. "
                "Return only the user-facing answer; do not include Research Agent Trace, "
                "plan steps, tool calls, verification statistics, or JSON:\n"
                f"{notes_text}"
            )
        messages = build_messages(answer_query, list(history or []), list(hits or []))
        answer = DeepSeekChat(settings).chat(
            messages=messages,
            temperature=float(temperature),
            max_tokens=max(256, min(4096, int(max_tokens or 1200))),
        )
        answer = clean_assistant_answer_presentation_text(answer).strip()
        if not answer:
            answer = _fallback_grounded_answer(query, hits, reason="empty LLM response", agent_notes=agent_notes)
            return {"answer": answer, "llm_used": False, "observation": "LLM returned empty text; used fallback answer."}
        return {"answer": answer, "llm_used": True, "observation": "Generated answer with existing RAG prompt."}
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
