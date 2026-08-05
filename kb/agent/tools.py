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
_ANSWER_DEBUG_RE = re.compile(
    r"\b(?:agent_trace|agentTrace|Research Agent Trace|retrieve_evidence|retrieve_references|"
    r"build_reading_guide|compare_papers|generate_grounded_answer|verify_answer_citations|"
    r"supported_claims|unsupported_claims|total_claims|question_type)\b",
    flags=re.IGNORECASE,
)
def _clip(text: Any, limit: int = 180) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    return clean[:limit]


def _source_name(hit: dict[str, Any]) -> str:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    source = (
        str(meta.get("source_name") or meta.get("title") or meta.get("source_path") or "").strip()
        or str(hit.get("id") or "").strip()
    )
    if not source:
        return ""
    if "\\" in source or "/" in source:
        source = Path(source).name or source
    for suffix in (".en.md", ".md", ".pdf"):
        if source.lower().endswith(suffix):
            return source[: -len(suffix)]
    return source


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


def _compact_evidence_matrix_rows(rows: Any, *, limit: int = 8) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows[: max(1, int(limit or 8))]:
        if not isinstance(row, dict):
            continue
        compact = {
            "paper": _clip(row.get("paper") or row.get("source_name"), 140),
            "source_name": _clip(row.get("source_name"), 120),
            "source_path": _clip(row.get("source_path"), 180),
            "method": _clip(row.get("method"), 220),
            "dataset_or_experiment": _clip(row.get("dataset_or_experiment"), 180),
            "key_result": _clip(row.get("key_result"), 220),
            "limitation": _clip(row.get("limitation"), 220),
            "evidence_quote": _clip(row.get("evidence_quote"), 260),
            "citation": _clip(row.get("citation"), 24),
            "heading_path": _clip(row.get("heading_path"), 180),
            "support_status": _clip(row.get("support_status"), 40),
        }
        out.append({key: value for key, value in compact.items() if value})
    return out


def _format_agent_notes(agent_notes: dict[str, Any] | None) -> str:
    if not isinstance(agent_notes, dict) or not agent_notes:
        return ""
    compact: dict[str, Any] = {}
    if isinstance(agent_notes.get("research_run"), dict):
        run = agent_notes.get("research_run") or {}
        metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
        compact["research_run"] = {
            "status": str(run.get("status") or "").strip(),
            "source_policy": str(run.get("source_policy") or "").strip(),
            "query_scope": str(run.get("query_scope") or "").strip(),
            "evidence_matrix_rows": _positive_int(metrics.get("evidence_matrix_rows")),
            "local_evidence_hit_count": _positive_int(metrics.get("local_evidence_hit_count")),
        }
    matrix_rows = _compact_evidence_matrix_rows(agent_notes.get("evidence_matrix"))
    if matrix_rows:
        compact["evidence_matrix"] = matrix_rows
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
            "answer_mode": str(gate.get("answer_mode") or "").strip(),
            "source_blend": str(gate.get("source_blend") or "").strip(),
            "source_policy": str(gate.get("source_policy") or "").strip(),
            "source_notice": str(gate.get("source_notice") or "").strip(),
            "evidence_hit_count": _positive_int(gate.get("evidence_hit_count")),
            "candidate_hit_count": _positive_int(gate.get("candidate_hit_count")),
            "retrieval_confidence": str(gate.get("retrieval_confidence") or "").strip(),
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


def _retrieval_source_variants(value: object) -> set[str]:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return set()
    variants = {raw.lower()}
    try:
        path = Path(raw)
        variants.add(path.name.lower())
        variants.add(str(path.expanduser().resolve(strict=False)).replace("\\", "/").lower())
    except Exception:
        pass
    return {item for item in variants if item}


def _balanced_scoped_hits(
    chunks: list[dict[str, Any]],
    *,
    source_paths: list[str],
    query_variants: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    for source_path in source_paths:
        expected = _retrieval_source_variants(source_path)
        group = [
            chunk
            for chunk in chunks
            if expected
            & _retrieval_source_variants(
                (chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}).get("source_path")
            )
        ]
        if group:
            groups.append(group)
    if not groups:
        return []
    per_source_limit = max(2, (max(1, int(top_k)) + len(groups) - 1) // len(groups))
    queries = list(dict.fromkeys(str(item or "").strip() for item in query_variants if str(item or "").strip()))
    ranked_groups: list[list[dict[str, Any]]] = []
    for group in groups:
        retriever = BM25Retriever(group)
        best_by_id: dict[str, dict[str, Any]] = {}
        for query in queries:
            for hit in retriever.search(query, top_k=max(per_source_limit * 4, per_source_limit)):
                identity = str(hit.get("id") or "").strip() or (
                    str((hit.get("meta") or {}).get("source_path") or "")
                    + "\n"
                    + str(hit.get("text") or "")
                )
                previous = best_by_id.get(identity)
                if previous is None or float(hit.get("score") or 0.0) > float(previous.get("score") or 0.0):
                    best_by_id[identity] = hit
        ranked_groups.append(
            sorted(
                best_by_id.values(),
                key=lambda hit: float(hit.get("score") or 0.0),
                reverse=True,
            )[:per_source_limit]
        )
    balanced: list[dict[str, Any]] = []
    for rank in range(per_source_limit):
        for group in ranked_groups:
            if rank < len(group):
                balanced.append(group[rank])
                if len(balanced) >= max(1, int(top_k)):
                    return balanced
    return balanced


def retrieve_evidence(
    query: str,
    *,
    db_dir: str | Path,
    settings: Any = None,
    top_k: int = 6,
    source_paths: list[str] | None = None,
) -> dict[str, Any]:
    chunks = load_all_chunks(Path(db_dir))
    requested_paths = list(dict.fromkeys(str(item or "").strip() for item in list(source_paths or []) if str(item or "").strip()))
    if requested_paths:
        allowed = set().union(*(_retrieval_source_variants(item) for item in requested_paths))
        chunks = [
            chunk
            for chunk in chunks
            if allowed
            & _retrieval_source_variants(
                (chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}).get("source_path")
            )
        ]
    retriever = BM25Retriever(chunks)
    hits, scores, used_query, used_translation, query_variants = _search_hits_with_fallback(
        query,
        retriever,
        max(1, min(20, int(top_k or 6))),
        settings,
        allow_translate=True,
        allow_expand=False,
    )
    limit = max(1, min(20, int(top_k or 6)))
    if requested_paths:
        balanced = _balanced_scoped_hits(
            chunks,
            source_paths=requested_paths,
            query_variants=[*list(query_variants or []), used_query, query],
            top_k=limit,
        )
        if balanced:
            hits = balanced
            scores = [float(hit.get("score") or 0.0) for hit in hits]
    hits = list(hits or [])[:limit]
    return {
        "hits": hits,
        "scores": list(scores or [])[: len(hits)],
        "used_query": used_query,
        "used_translation": bool(used_translation),
        "query_variants": list(query_variants or []),
        "requested_source_count": len(requested_paths),
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


def _is_general_llm_mode(agent_notes: dict[str, Any] | None) -> bool:
    if not isinstance(agent_notes, dict):
        return False
    gate = agent_notes.get("evidence_gate")
    return isinstance(gate, dict) and str(gate.get("answer_mode") or "").strip() in {
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


def _answer_contract(agent_notes: dict[str, Any] | None) -> str:
    if not isinstance(agent_notes, dict):
        return ""
    value = str(agent_notes.get("answer_contract") or "").strip().lower()
    return value if value == "research_brief" else ""


def _source_blend(agent_notes: dict[str, Any] | None) -> str:
    if not isinstance(agent_notes, dict):
        return ""
    gate = agent_notes.get("evidence_gate")
    if not isinstance(gate, dict):
        return ""
    blend = str(gate.get("source_blend") or "").strip()
    if blend:
        return blend
    mode = str(gate.get("answer_mode") or "").strip()
    if mode == "evidence_grounded":
        return "local_grounded"
    if mode == "external_academic_llm":
        return "external_academic"
    if mode in {"hybrid_local_external", "general_llm"}:
        return mode
    return ""


def _is_hybrid_answer_mode(agent_notes: dict[str, Any] | None) -> bool:
    return _answer_mode(agent_notes) == "hybrid_local_external"


def _has_cjk(text: str) -> bool:
    return sum(1 for c in str(text or "") if "\u3400" <= c <= "\u9fff" or "\uf900" <= c <= "\ufaff") >= 4


def _general_llm_messages(
    query: str,
    history: list[dict[str, Any]] | None = None,
    *,
    academic: bool = False,
    prefer_web: bool = False,
) -> list[dict[str, Any]]:
    prefer_zh = _has_cjk(query)
    if academic:
        system = (
            "You are a concise academic assistant. The local indexed knowledge base returned no relevant evidence. "
            "Answer from general academic knowledge"
            + (" and available web search results" if prefer_web else "")
            + ". Do not claim that the answer is grounded in the user's local papers. "
            "If the question asks about a specific local/current paper, explain that the local evidence was not found "
            "and limit the rest of the answer to general background, likely interpretations, or how to verify it. "
            "Do not invent local citation cards, local snippet ids, or tool traces."
        )
    else:
        system = (
            "You are a concise, helpful assistant. Answer the user's general question directly. "
            "Do not invent local knowledge-base citations, paper evidence, tool traces, or JSON. "
            "If the user asks for paper-specific evidence, say that the indexed evidence is insufficient."
        )
    if prefer_zh:
        system += " Reply in Chinese unless the user asks otherwise."
    trimmed = [m for m in list(history or []) if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    return [{"role": "system", "content": system}, *trimmed, {"role": "user", "content": str(query or "").strip()}]


def _prepend_external_notice(answer: str, query: str, *, web_used: bool = False) -> str:
    clean = str(answer or "").strip()
    notice = _external_answer_notice(query, web_used=web_used)
    if not clean:
        return notice
    if clean.startswith(notice):
        return clean
    return f"{notice}\n\n{clean}"


def _prepend_hybrid_notice(answer: str, query: str, *, web_used: bool = False) -> str:
    clean = str(answer or "").strip()
    notice = _hybrid_answer_notice(query, web_used=web_used)
    if not clean:
        return notice
    if clean.startswith(notice):
        return clean
    return f"{notice}\n\n{clean}"


def _web_search_configured(settings: Any) -> bool:
    return bool(
        getattr(settings, "agent_web_search_enabled", False)
        and getattr(settings, "agent_web_search_api_key", None)
        and str(getattr(settings, "agent_web_search_model", "") or "").strip()
    )


def _external_answer_notice(query: str, *, web_used: bool = False) -> str:
    if _has_cjk(query):
        if web_used:
            return (
                "\u6ce8\u610f\uff1a\u672c\u5730\u77e5\u8bc6\u5e93\u6ca1\u6709\u547d\u4e2d\u76f8\u5173\u8bc1\u636e\uff0c"
                "\u4ee5\u4e0b\u662f\u5916\u90e8\u6a21\u578b\u7ed3\u5408\u8054\u7f51\u7684\u56de\u7b54\uff0c"
                "\u4e0d\u4ee3\u8868\u5f53\u524d\u77e5\u8bc6\u5e93\u7ed3\u8bba\u3002"
            )
        return (
            "\u6ce8\u610f\uff1a\u672c\u5730\u77e5\u8bc6\u5e93\u6ca1\u6709\u547d\u4e2d\u76f8\u5173\u8bc1\u636e\uff0c"
            "\u4ee5\u4e0b\u662f\u5916\u90e8\u6a21\u578b\u7684\u901a\u7528\u56de\u7b54\uff0c"
            "\u4e0d\u4ee3\u8868\u5f53\u524d\u77e5\u8bc6\u5e93\u7ed3\u8bba\u3002"
        )
    if web_used:
        return "Note: no matching local knowledge-base evidence was found; this is an external model answer with web search, not a knowledge-base-grounded answer."
    return "Note: no matching local knowledge-base evidence was found; this is an external model answer, not a knowledge-base-grounded answer."


def _hybrid_answer_notice(query: str, *, web_used: bool = False) -> str:
    if _has_cjk(query):
        if web_used:
            return (
                "\u6ce8\u610f\uff1a\u5e26 [n] \u7684\u5185\u5bb9\u6765\u81ea\u672c\u5730\u77e5\u8bc6\u5e93\uff1b"
                "\u672a\u5e26\u672c\u5730\u5f15\u7528\u7684\u80cc\u666f\u89e3\u91ca\u53ef\u80fd\u6765\u81ea\u5916\u90e8\u6a21\u578b\u8054\u7f51\u8865\u5145\u3002"
            )
        return (
            "\u6ce8\u610f\uff1a\u5e26 [n] \u7684\u5185\u5bb9\u6765\u81ea\u672c\u5730\u77e5\u8bc6\u5e93\uff1b"
            "\u672a\u5e26\u672c\u5730\u5f15\u7528\u7684\u80cc\u666f\u89e3\u91ca\u53ef\u80fd\u6765\u81ea\u5916\u90e8\u6a21\u578b\u8865\u5145\u3002"
        )
    if web_used:
        return "Note: local citations [n] come from the knowledge base; uncited background may use external model and web context."
    return "Note: local citations [n] come from the knowledge base; uncited background may use external model context."


def _fallback_general_answer(query: str, *, reason: str = "", academic: bool = False) -> str:
    if _has_cjk(query):
        suffix = f" \u539f\u56e0\uff1a{reason}" if reason else ""
        if academic:
            return (
                "\u672c\u5730\u77e5\u8bc6\u5e93\u6ca1\u6709\u547d\u4e2d\u76f8\u5173\u8bc1\u636e\uff0c"
                "\u4e14\u5f53\u524d\u6ca1\u6709\u53ef\u7528\u7684\u6587\u672c\u6a21\u578b API\uff0c"
                f"\u65e0\u6cd5\u751f\u6210\u5916\u90e8\u5b66\u672f\u56de\u7b54\u3002{suffix}"
            ).strip()
        return (
            "\u8fd9\u4e2a\u95ee\u9898\u4e0d\u9700\u8981\u77e5\u8bc6\u5e93\u8bc1\u636e\uff0c"
            "\u4f46\u5f53\u524d\u6ca1\u6709\u53ef\u7528\u7684\u6587\u672c\u6a21\u578b API\uff0c"
            f"\u65e0\u6cd5\u751f\u6210\u666e\u901a\u56de\u7b54\u3002{suffix}"
        ).strip()
    suffix = f" Reason: {reason}" if reason else ""
    if academic:
        return f"No matching local knowledge-base evidence was found, and no text model API is available to generate an external academic answer.{suffix}"
    return f"This question does not require indexed evidence, but no text model API is available to generate a general answer.{suffix}"


def _hybrid_answer_query(query: str, notes_text: str, *, prefer_web: bool = False) -> str:
    policy = (
        "Hybrid answer source policy:\n"
        "- Treat retrieved knowledge-base snippets as authoritative for paper-specific claims.\n"
        "- Use `evidence_matrix` as the synthesis scaffold: cover paper, method, result, limitation, and evidence cells when present.\n"
        "- Cite every local paper-specific claim with the retrieved snippet marker like [1] or [2].\n"
        "- Put external academic background only under a short 'External context'/'Background' line when it helps.\n"
        "- Do not cite external background with local snippet markers, and do not claim it came from the knowledge base.\n"
        "- If local evidence conflicts with external background, say so and prioritize local evidence.\n"
        "- Use the user's language and keep the answer concise; do not output trace, plan, tool calls, verification statistics, or JSON.\n\n"
        "Compact answer shape:\n"
        "1. Start with one direct answer sentence.\n"
        "2. Add 'Local evidence' bullets for paper-specific evidence, each with local citations.\n"
        "3. Add one optional 'External context' line only for useful uncited background.\n"
        "4. Add one optional 'Limit' sentence only when local evidence is thin, conflicting, or incomplete."
    )
    if prefer_web:
        policy += "\n- When web search contributes background, keep it secondary to the local snippets."
    if notes_text:
        return (
            f"{query}\n\n"
            f"{policy}\n\n"
            "Research Agent structured notes:\n"
            f"{notes_text}"
        )
    return f"{query}\n\n{policy}"


def _source_notice_required(answer_mode: str) -> bool:
    return str(answer_mode or "").strip() in {"hybrid_local_external", "external_academic_llm"}


def _local_quality_gate_required(answer_mode: str, hits: list[dict[str, Any]]) -> bool:
    mode = str(answer_mode or "").strip()
    return bool(hits) and mode not in {"external_academic_llm", "general_llm"}


def _answer_quality_gate(
    answer: str,
    hits: list[dict[str, Any]],
    *,
    agent_notes: dict[str, Any] | None = None,
    answer_mode: str = "",
    raw_answer: str = "",
) -> dict[str, Any]:
    clean = clean_assistant_answer_presentation_text(answer).strip()
    raw = str(raw_answer if raw_answer else answer or "")
    verification = _verify_answer_citations(clean, hits, answer_mode=answer_mode)
    hard_reasons: list[str] = []
    warnings: list[str] = []
    if raw.strip() and raw.strip() != clean:
        warnings.append("trace_debug_removed")
    if _ANSWER_DEBUG_RE.search(clean):
        hard_reasons.append("debug_content_in_answer")
    if _source_notice_required(answer_mode) and verification.source_notice_count <= 0:
        hard_reasons.append("missing_source_notice")
    if _local_quality_gate_required(answer_mode, hits):
        if verification.total_claims <= 0:
            hard_reasons.append("missing_local_evidence_claim")
        for claim in verification.claims:
            if not isinstance(claim, dict) or str(claim.get("claim_kind") or "") != "local_claim":
                continue
            if not bool(claim.get("citation_present") or claim.get("has_citation")):
                hard_reasons.append("missing_local_citation")
                continue
            if not bool(claim.get("supported")):
                hard_reasons.append("unsupported_local_claim")
    deduped = list(dict.fromkeys(hard_reasons))
    return {
        "ok": not deduped,
        "answer": clean,
        "reasons": deduped,
        "warnings": list(dict.fromkeys(warnings)),
        "verification": verification.to_dict(),
    }


def _answer_repair_query(
    query: str,
    *,
    candidate_answer: str,
    gate: dict[str, Any],
    notes_text: str,
    answer_mode: str,
) -> str:
    reasons = ", ".join(str(item) for item in list(gate.get("reasons") or []) if str(item or "").strip()) or "quality gate failed"
    policy = (
        "Revise the candidate answer so it passes the Research Agent answer quality gate.\n"
        "Return only the user-facing answer. Do not include trace JSON, tool calls, plan steps, verification statistics, or analysis.\n"
        "Every local paper-specific claim must use local citation markers like [1] or [2].\n"
        "Do not keep any local claim that is not supported by the retrieved snippets or evidence matrix.\n"
        "If evidence is thin, say the limitation briefly instead of inventing details.\n"
    )
    if answer_mode == "hybrid_local_external":
        policy += (
            "Keep local evidence separate from external/background context. "
            "External background must stay short and must not use local citation markers.\n"
        )
    return (
        f"User question:\n{query}\n\n"
        f"Quality gate reasons:\n{reasons}\n\n"
        f"{policy}\n"
        f"Candidate answer:\n{candidate_answer}\n\n"
        f"Structured evidence notes:\n{notes_text or '(none)'}"
    )


def _repair_answer_once(
    query: str,
    answer: str,
    hits: list[dict[str, Any]],
    *,
    settings: Any,
    history: list[dict[str, Any]] | None,
    agent_notes: dict[str, Any] | None,
    answer_mode: str,
    gate: dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> str:
    if not getattr(settings, "text_api_key", None):
        return ""
    notes_text = _format_agent_notes(agent_notes)
    repair_query = _answer_repair_query(
        query,
        candidate_answer=answer,
        gate=gate,
        notes_text=notes_text,
        answer_mode=answer_mode,
    )
    messages = build_messages(
        repair_query,
        list(history or []),
        list(hits or []),
        answer_contract=_answer_contract(agent_notes),
    )
    repaired = DeepSeekChat(settings).chat(
        messages=messages,
        temperature=min(float(temperature or 0.2), 0.2),
        max_tokens=max(256, min(4096, int(max_tokens or 1200))),
    )
    return clean_assistant_answer_presentation_text(repaired).strip()


def _fallback_quality_gate_answer(
    query: str,
    hits: list[dict[str, Any]],
    *,
    reason: str = "",
    agent_notes: dict[str, Any] | None = None,
) -> str:
    if not hits:
        return _fallback_general_answer(query, reason=reason, academic=True)
    prefer_zh = _has_cjk(query)
    lines = ["## \u8bc1\u636e" if prefer_zh else "## Evidence"]
    selected_hits: list[tuple[int, dict[str, Any]]] = []
    source_limit = 8 if _answer_contract(agent_notes) == "research_brief" else 4
    seen_sources: set[str] = set()
    indexed_hits = [
        (index, hit)
        for index, hit in enumerate(list(hits or []), start=1)
        if isinstance(hit, dict)
    ]
    for index, hit in indexed_hits:
        source_key = (_source_path(hit) or _source_name(hit)).replace("\\", "/").strip().lower()
        if source_key and source_key not in seen_sources:
            seen_sources.add(source_key)
            selected_hits.append((index, hit))
        if len(selected_hits) >= source_limit:
            break
    for index, hit in indexed_hits:
        if len(selected_hits) >= source_limit:
            break
        if any(existing_index == index for existing_index, _ in selected_hits):
            continue
        selected_hits.append((index, hit))
    for index, hit in selected_hits:
        text = re.sub(r"\[[0-9][0-9,\-\s]*\]", " ", str(hit.get("text") or ""))
        text = re.sub(r"(?:^|\s)#{1,6}\s*", " ", text)
        text = re.sub(r"[*_`|]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        candidates = [
            part.strip(" -:;,.\u3002\uff1b\uff1a")
            for part in re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+", text)[:8]
        ]
        candidates = [part for part in candidates if len(part) >= 32]
        sentence = max(candidates, key=len, default=text)
        sentence = _clip(sentence, 360).rstrip(" -:;,.\u3002\uff1b\uff1a")
        if sentence:
            lines.append(f"- {sentence} [{index}].")
    return "\n".join(lines).strip()


def _finalize_grounded_answer(
    query: str,
    answer: str,
    hits: list[dict[str, Any]],
    *,
    settings: Any,
    history: list[dict[str, Any]] | None,
    agent_notes: dict[str, Any] | None,
    answer_mode: str,
    web_used: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    defer_quality_gate_repair: bool = False,
) -> dict[str, Any]:
    raw_clean = clean_assistant_answer_presentation_text(answer).strip()
    debug_removed = bool(str(answer or "").strip() and str(answer or "").strip() != raw_clean)
    clean = raw_clean
    if answer_mode == "hybrid_local_external":
        clean = _prepend_hybrid_notice(clean, query, web_used=web_used)
    gate = _answer_quality_gate(clean, hits, agent_notes=agent_notes, answer_mode=answer_mode, raw_answer=clean)
    if debug_removed:
        gate["warnings"] = list(dict.fromkeys(list(gate.get("warnings") or []) + ["trace_debug_removed"]))
    if gate["ok"]:
        return {"answer": gate["answer"], "quality_gate": {"status": "passed", **{k: gate[k] for k in ("reasons", "warnings")}}}
    if defer_quality_gate_repair:
        return {
            "answer": gate["answer"],
            "quality_gate": {
                "status": "failed",
                "reasons": list(gate.get("reasons") or []),
                "warnings": list(gate.get("warnings") or []),
                "verification": gate.get("verification") if isinstance(gate.get("verification"), dict) else {},
            },
        }
    repaired = ""
    repair_error = ""
    try:
        repaired = _repair_answer_once(
            query,
            gate["answer"],
            hits,
            settings=settings,
            history=history,
            agent_notes=agent_notes,
            answer_mode=answer_mode,
            gate=gate,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        repair_error = str(exc)[:240]
    if repaired:
        if answer_mode == "hybrid_local_external":
            repaired = _prepend_hybrid_notice(repaired, query, web_used=web_used)
        repaired_gate = _answer_quality_gate(repaired, hits, agent_notes=agent_notes, answer_mode=answer_mode, raw_answer=repaired)
        if repaired_gate["ok"]:
            return {
                "answer": repaired_gate["answer"],
                "quality_gate": {
                    "status": "repaired",
                    "reasons": list(gate.get("reasons") or []),
                    "warnings": list(dict.fromkeys(list(gate.get("warnings") or []) + list(repaired_gate.get("warnings") or []))),
                },
            }
        gate = repaired_gate
    fallback = _fallback_quality_gate_answer(
        query,
        hits,
        reason=", ".join(str(item) for item in list(gate.get("reasons") or [])[:3]),
        agent_notes=agent_notes,
    )
    if answer_mode == "hybrid_local_external":
        fallback = _prepend_hybrid_notice(fallback, query, web_used=web_used)
    verification = gate.get("verification") if isinstance(gate.get("verification"), dict) else {}
    unsupported_claim_previews = [
        {
            "text": _clip(claim.get("claim_text") or claim.get("text"), 180),
            "reason": str(claim.get("unsupported_reason") or "").strip(),
        }
        for claim in list(verification.get("claims") or [])
        if isinstance(claim, dict) and not bool(claim.get("supported"))
    ][:4]
    return {
        "answer": fallback,
        "quality_gate": {
            "status": "fallback",
            "reasons": list(gate.get("reasons") or []),
            "warnings": list(gate.get("warnings") or []),
            "repair_error": repair_error,
            "unsupported_claim_previews": unsupported_claim_previews,
        },
    }


def generate_grounded_answer(
    query: str,
    hits: list[dict[str, Any]],
    *,
    settings: Any = None,
    history: list[dict[str, Any]] | None = None,
    agent_notes: dict[str, Any] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    defer_quality_gate_repair: bool = False,
) -> dict[str, Any]:
    source_blend = _source_blend(agent_notes)
    if not hits:
        if _is_general_llm_mode(agent_notes):
            mode = _answer_mode(agent_notes)
            academic = mode == "external_academic_llm"
            if academic and _web_search_configured(settings):
                try:
                    chat = DeepSeekChat(settings)
                    web_result = chat.chat_with_web_search(
                        messages=_general_llm_messages(query, history, academic=True, prefer_web=True),
                        temperature=float(temperature),
                        max_tokens=max(256, min(4096, int(max_tokens or 1200))),
                    )
                    answer = clean_assistant_answer_presentation_text(str(web_result.get("content") or "")).strip()
                    if answer:
                        answer = _prepend_external_notice(answer, query, web_used=True)
                        return {
                            "answer": answer,
                            "llm_used": True,
                            "answer_mode": mode,
                            "source_blend": source_blend or "external_academic",
                            "web_search_used": True,
                            "web_citations": list(web_result.get("annotations") or [])[:12],
                            "web_search_model": str(web_result.get("model") or ""),
                            "observation": "Generated an external academic answer with API web search because no local evidence was retrieved.",
                        }
                except Exception:
                    pass
            if not getattr(settings, "text_api_key", None):
                answer = _fallback_general_answer(query, reason="missing text API key", academic=academic)
                return {
                    "answer": answer,
                    "llm_used": False,
                    "answer_mode": mode or "general_llm",
                    "source_blend": source_blend or ("external_academic" if academic else "general_llm"),
                    "web_search_used": False,
                    "observation": "Could not call an external LLM answer because no text API key is configured.",
                }
            try:
                answer = DeepSeekChat(settings).chat(
                    messages=_general_llm_messages(query, history, academic=academic),
                    temperature=float(temperature),
                    max_tokens=max(256, min(4096, int(max_tokens or 1200))),
                )
                answer = clean_assistant_answer_presentation_text(answer).strip()
                if not answer:
                    answer = _fallback_general_answer(query, reason="empty LLM response", academic=academic)
                    return {
                        "answer": answer,
                        "llm_used": False,
                        "answer_mode": mode or "general_llm",
                        "source_blend": source_blend or ("external_academic" if academic else "general_llm"),
                        "web_search_used": False,
                        "observation": "External LLM returned empty text; used fallback answer.",
                    }
                if academic:
                    answer = _prepend_external_notice(answer, query, web_used=False)
                return {
                    "answer": answer,
                    "llm_used": True,
                    "answer_mode": mode or "general_llm",
                    "source_blend": source_blend or ("external_academic" if academic else "general_llm"),
                    "web_search_used": False,
                    "observation": (
                        "Generated an external academic LLM answer because no local evidence was retrieved."
                        if academic
                        else "Generated a general LLM answer without using local evidence."
                    ),
                }
            except Exception as exc:
                answer = _fallback_general_answer(query, reason=str(exc)[:160], academic=academic)
                return {
                    "answer": answer,
                    "llm_used": False,
                    "answer_mode": mode or "general_llm",
                    "source_blend": source_blend or ("external_academic" if academic else "general_llm"),
                    "web_search_used": False,
                    "error": str(exc)[:240],
                    "observation": "External LLM generation failed; used fallback answer.",
                }
        answer = _fallback_grounded_answer(query, hits, reason="no retrieved evidence", agent_notes=agent_notes)
        return {
            "answer": answer,
            "llm_used": False,
            "answer_mode": "evidence_grounded",
            "source_blend": source_blend or "local_grounded",
            "web_search_used": False,
            "observation": "Skipped LLM answer because no indexed evidence was retrieved.",
        }
    hybrid = _is_hybrid_answer_mode(agent_notes)
    if hybrid and _web_search_configured(settings):
        try:
            notes_text = _format_agent_notes(agent_notes)
            answer_query = _hybrid_answer_query(query, notes_text, prefer_web=True)
            messages = build_messages(
                answer_query,
                list(history or []),
                list(hits or []),
                answer_contract=_answer_contract(agent_notes),
            )
            web_result = DeepSeekChat(settings).chat_with_web_search(
                messages=messages,
                temperature=float(temperature),
                max_tokens=max(256, min(4096, int(max_tokens or 1200))),
            )
            answer = clean_assistant_answer_presentation_text(str(web_result.get("content") or "")).strip()
            if answer:
                finalized = _finalize_grounded_answer(
                    query,
                    answer,
                    hits,
                    settings=settings,
                    history=history,
                    agent_notes=agent_notes,
                    answer_mode="hybrid_local_external",
                    web_used=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    defer_quality_gate_repair=defer_quality_gate_repair,
                )
                return {
                    "answer": finalized["answer"],
                    "llm_used": True,
                    "answer_mode": "hybrid_local_external",
                    "source_blend": source_blend or "hybrid_local_external",
                    "web_search_used": True,
                    "web_citations": list(web_result.get("annotations") or [])[:12],
                    "web_search_model": str(web_result.get("model") or ""),
                    "quality_gate": finalized["quality_gate"],
                    "observation": "Generated a hybrid answer from local evidence plus external API web context.",
                }
        except Exception:
            pass
    if not getattr(settings, "text_api_key", None):
        answer = _fallback_grounded_answer(query, hits, reason="missing text API key", agent_notes=agent_notes)
        return {
            "answer": answer,
            "llm_used": False,
            "answer_mode": "hybrid_local_external" if hybrid else "evidence_grounded",
            "source_blend": source_blend or ("hybrid_local_external" if hybrid else "local_grounded"),
            "web_search_used": False,
            "observation": "Generated degraded-mode answer without an LLM.",
        }
    try:
        notes_text = _format_agent_notes(agent_notes)
        answer_query = query
        if hybrid:
            answer_query = _hybrid_answer_query(query, notes_text, prefer_web=False)
        elif notes_text:
            answer_query = (
                f"{query}\n\n"
                "Research Agent structured notes. Use these as an evidence map for the answer; "
                "when `evidence_matrix` is present, synthesize from its paper/method/result/limitation/evidence cells first. "
                "Do not add claims that are not supported by the retrieved snippets. "
                "Return only the user-facing answer; do not include Research Agent Trace, "
                "plan steps, tool calls, verification statistics, or JSON:\n"
                f"{notes_text}"
            )
        messages = build_messages(
            answer_query,
            list(history or []),
            list(hits or []),
            answer_contract=_answer_contract(agent_notes),
        )
        answer = DeepSeekChat(settings).chat(
            messages=messages,
            temperature=float(temperature),
            max_tokens=max(256, min(4096, int(max_tokens or 1200))),
        )
        answer = clean_assistant_answer_presentation_text(answer).strip()
        if not answer:
            answer = _fallback_grounded_answer(query, hits, reason="empty LLM response", agent_notes=agent_notes)
            return {
                "answer": answer,
                "llm_used": False,
                "answer_mode": "hybrid_local_external" if hybrid else "evidence_grounded",
                "source_blend": source_blend or ("hybrid_local_external" if hybrid else "local_grounded"),
                "web_search_used": False,
                "observation": "LLM returned empty text; used fallback answer.",
            }
        answer_mode = "hybrid_local_external" if hybrid else "evidence_grounded"
        finalized = _finalize_grounded_answer(
            query,
            answer,
            hits,
            settings=settings,
            history=history,
            agent_notes=agent_notes,
            answer_mode=answer_mode,
            web_used=False,
            temperature=temperature,
            max_tokens=max_tokens,
            defer_quality_gate_repair=defer_quality_gate_repair,
        )
        return {
            "answer": finalized["answer"],
            "llm_used": True,
            "answer_mode": answer_mode,
            "source_blend": source_blend or ("hybrid_local_external" if hybrid else "local_grounded"),
            "web_search_used": False,
            "quality_gate": finalized["quality_gate"],
            "observation": (
                "Generated a hybrid answer from local evidence plus external model context."
                if hybrid
                else "Generated answer with existing RAG prompt."
            ),
        }
    except Exception as exc:
        answer = _fallback_grounded_answer(query, hits, reason=str(exc)[:160], agent_notes=agent_notes)
        return {
            "answer": answer,
            "llm_used": False,
            "answer_mode": "hybrid_local_external" if hybrid else "evidence_grounded",
            "source_blend": source_blend or ("hybrid_local_external" if hybrid else "local_grounded"),
            "web_search_used": False,
            "error": str(exc)[:240],
            "observation": "LLM generation failed; used fallback answer.",
        }


def verify_answer_citations(answer: str, hits: list[dict[str, Any]], *, answer_mode: str = "") -> dict[str, Any]:
    verification = _verify_answer_citations(answer, hits, answer_mode=answer_mode)
    return {
        "verification": verification.to_dict(),
        "observation": (
            f"Verified {verification.supported_claims}/{verification.total_claims} claim(s) "
            "with citation/evidence support."
        ),
    }
