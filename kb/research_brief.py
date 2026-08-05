from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any


_MAX_BRIEF_SOURCES = 8
_MAX_EVIDENCE_TEXT = 1_800
_CITATION_RE = re.compile(r"(?<!\[)\[(\d+(?:\s*(?:,|-)\s*\d+)*)\](?!\])")


def _text(value: object, *, limit: int = 2_000) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ").strip()
    return text[: max(0, int(limit))]


def _first_text(item: dict[str, Any], *keys: str, limit: int = 2_000) -> str:
    for key in keys:
        value = _text(item.get(key), limit=limit)
        if value:
            return value
    return ""


def _safe_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if number == number and abs(number) != float("inf") else float(default)


def _citation_numbers(answer: object, *, limit: int = 100) -> list[int]:
    numbers: list[int] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(str(answer or "")):
        for part in re.split(r"\s*,\s*", match.group(1)):
            raw = part.strip()
            if "-" in raw:
                start_raw, end_raw = [item.strip() for item in raw.split("-", 1)]
                start = _safe_int(start_raw)
                end = _safe_int(end_raw)
                if start <= 0 or end < start or end - start >= limit:
                    continue
                candidates = range(start, end + 1)
            else:
                candidates = (_safe_int(raw),)
            for number in candidates:
                if number <= 0 or number in seen:
                    continue
                seen.add(number)
                numbers.append(number)
                if len(numbers) >= limit:
                    return numbers
    return numbers


def _source_path(item: dict[str, Any]) -> str:
    library_path = _first_text(
        item,
        "libraryMatchPath",
        "library_match_path",
        limit=1_200,
    )
    if library_path:
        return library_path
    kind = _first_text(item, "shelfItemKind", "shelf_item_kind", limit=80).lower()
    route = _first_text(item, "citationRoute", "citation_route", limit=80).lower()
    if kind in {"reference", "inpaper", "reader_reference", "reader_references"}:
        return ""
    if route == "system_b" or item.get("isInpaper") is True or item.get("is_inpaper") is True:
        return ""
    return _first_text(
        item,
        "sourcePath",
        "source_path",
        limit=1_200,
    )


def _source_name(item: dict[str, Any]) -> str:
    return _first_text(
        item,
        "libraryMatchTitle",
        "library_match_title",
        "title",
        "cardTitle",
        "main",
        "sourceName",
        "source_name",
        limit=500,
    )


def _source_variants(value: object) -> set[str]:
    raw = _text(value, limit=1_200).replace("\\", "/")
    if not raw:
        return set()
    variants = {raw.lower()}
    try:
        path = Path(raw)
        variants.add(path.name.lower())
        variants.add(path.stem.lower())
        variants.add(str(path.expanduser().resolve(strict=False)).replace("\\", "/").lower())
    except Exception:
        pass
    return {item for item in variants if item}


def select_research_brief_sources(
    shelf_items: list[dict[str, Any]],
    *,
    item_keys: list[str] | None = None,
    limit: int = _MAX_BRIEF_SOURCES,
) -> list[dict[str, Any]]:
    requested = {_text(key, limit=500) for key in list(item_keys or []) if _text(key, limit=500)}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in list(shelf_items or []):
        if not isinstance(raw, dict):
            continue
        key = _first_text(raw, "key", "id", limit=500)
        if requested and key not in requested:
            continue
        source_path = _source_path(raw)
        source_name = _source_name(raw)
        if not source_path or not source_name:
            continue
        identity = source_path.replace("\\", "/").lower()
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(dict(raw))
        if len(selected) >= max(1, min(_MAX_BRIEF_SOURCES, int(limit or _MAX_BRIEF_SOURCES))):
            break
    return selected


def research_brief_context(
    selected_items: list[dict[str, Any]],
    *,
    conversation_id: str = "",
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for index, item in enumerate(list(selected_items or []), start=1):
        source_path = _source_path(item)
        source_name = _source_name(item)
        title = _first_text(item, "title", "cardTitle", "main", limit=500) or source_name
        excerpt = _first_text(
            item,
            "shelfExcerpt",
            "shelf_excerpt",
            "evidenceQuote",
            "evidence_quote",
            "cardEvidence",
            "card_evidence",
            "citationContext",
            "citation_context",
            limit=1_600,
        )
        summary = _first_text(
            item,
            "summaryLine",
            "summary_line",
            "cardTakeaway",
            "card_takeaway",
            "whyLine",
            "why_line",
            limit=1_000,
        )
        items.append(
            {
                "key": _first_text(item, "key", "id", limit=500) or f"brief-source-{index}",
                "title": title,
                "sourceName": source_name,
                "sourcePath": source_path,
                "locationLabel": _first_text(item, "locationLabel", "location_label", limit=500),
                "headingPath": _first_text(item, "headingPath", "heading_path", limit=800),
                "blockId": _first_text(item, "blockId", "block_id", limit=500),
                "anchorId": _first_text(item, "anchorId", "anchor_id", "anchor", limit=500),
                "doi": _first_text(item, "doi", "doiUrl", "doi_url", limit=400),
                "authors": _first_text(item, "authors", limit=500),
                "year": _first_text(item, "year", limit=40),
                "summary": summary,
                "excerpt": excerpt,
                "note": _first_text(item, "note", limit=1_000),
            }
        )
    return {
        "version": 1,
        "id": f"research-brief:{_text(conversation_id, limit=120) or 'project'}",
        "source": "citation_shelf",
        "conversationId": _text(conversation_id, limit=120),
        "itemCount": len(items),
        "items": items,
    }


def research_brief_prompt(objective: str, *, locale: str = "zh") -> str:
    goal = _text(objective, limit=4_000)
    if str(locale or "").strip().lower().startswith("en"):
        return (
            "Create an evidence-grounded research brief using only the selected literature-basket sources. "
            f"Research objective: {goal or 'Synthesize the selected papers into an actionable research brief.'}\n\n"
            "Use these sections: Executive findings; Methods and experimental comparison; Quantitative evidence; "
            "Disagreements or boundary conditions; Research gaps and next questions. Write no more than eight substantive "
            "bullet sentences in total. Each bullet must be exactly one sentence and must end with its matching numeric "
            "citation such as [1] or [1, 2]. Headings must contain no factual claims. Use only citation numbers supplied "
            "with the evidence. Keep each paper's conditions attached to that paper; do not merge conditions from different "
            "papers. Omit any unsupported section or state a concise evidence gap without inventing background knowledge."
        )
    return (
        "仅使用已选文献篮中的本地论文证据，生成一份可继续编辑的研究简报。"
        f"研究目标：{goal or '综合所选论文，形成可执行的研究简报。'}\n\n"
        "请包含：核心结论；方法与实验条件对比；定量证据；分歧或适用边界；研究空白与下一步问题。"
        "全文最多写八条实质性要点；每条只能有一个完整句子，句末必须带匹配的数字引用，例如 [1] 或 [1, 2]。"
        "标题中不得写事实主张，只能使用证据中已有的引用编号。不同论文的条件必须分别绑定到原论文，不得拼接。"
        "没有证据支持的章节应省略，或只简洁说明证据缺口，不得用常识或外部知识补齐。"
    )


def research_brief_evidence(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, hit in enumerate(list(hits or []), start=1):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        source_path = _first_text(meta, "source_path", limit=1_200)
        source_name = _first_text(meta, "source_name", "title", limit=500)
        if not source_name and source_path:
            source_name = Path(source_path).stem
        out.append(
            {
                "citation_number": index,
                "source_path": source_path,
                "source_name": source_name,
                "title": _first_text(meta, "title", "source_name", limit=500),
                "heading_path": _first_text(meta, "heading_path", "top_heading", limit=800),
                "location_label": _first_text(meta, "location_label", "heading_path", limit=500),
                "page_start": meta.get("page_start") or meta.get("page") or None,
                "page_end": meta.get("page_end") or meta.get("page") or None,
                "block_id": _first_text(meta, "block_id", limit=500),
                "anchor_id": _first_text(meta, "anchor_id", "anchor", limit=500),
                "evidence_quote": _text(hit.get("text"), limit=_MAX_EVIDENCE_TEXT),
                "score": _safe_float(hit.get("score")),
            }
        )
    return out


def _item_matches_evidence(item: dict[str, Any], evidence: dict[str, Any]) -> bool:
    item_terms = _source_variants(_source_path(item)) | _source_variants(_source_name(item))
    evidence_terms = _source_variants(evidence.get("source_path")) | _source_variants(evidence.get("source_name"))
    return bool(item_terms & evidence_terms)


def research_brief_bibliography(
    selected_items: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for evidence_item in list(evidence or []):
        match = next(
            (item for item in selected_items if _item_matches_evidence(item, evidence_item)),
            {},
        )
        doi = _first_text(match, "doi", "doiUrl", "doi_url", limit=400)
        source_path = _first_text(evidence_item, "source_path", limit=1_200) or _source_path(match)
        title = (
            _first_text(match, "title", "cardTitle", "main", limit=800)
            or _first_text(evidence_item, "title", "source_name", limit=800)
        )
        identity = (doi.lower() if doi else source_path.replace("\\", "/").lower()) or title.lower()
        citation_number = _safe_int(evidence_item.get("citation_number"))
        if identity in seen:
            row = out[seen[identity]]
            numbers = row.setdefault("citation_numbers", [])
            if citation_number > 0 and citation_number not in numbers:
                numbers.append(citation_number)
            continue
        seen[identity] = len(out)
        out.append(
            {
                "title": title,
                "authors": _first_text(match, "authors", limit=800),
                "year": _first_text(match, "year", limit=40),
                "venue": _first_text(match, "venue", limit=500),
                "doi": doi,
                "source_path": source_path,
                "source_name": _first_text(evidence_item, "source_name", limit=500) or _source_name(match),
                "citation_numbers": [citation_number] if citation_number > 0 else [],
            }
        )
    return out


def research_brief_quality(
    *,
    answer: str,
    agent_trace: dict[str, Any],
    selected_items: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    trace = agent_trace if isinstance(agent_trace, dict) else {}
    verification = trace.get("verification") if isinstance(trace.get("verification"), dict) else {}
    summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
    generation_quality_gate = str(summary.get("quality_gate_status") or "").strip().lower()
    generation_mode = (
        "extractive_fallback" if generation_quality_gate == "fallback" else "model_synthesis"
    )
    total_claims = _safe_int(verification.get("total_claims") or summary.get("total_claims"))
    supported_claims = _safe_int(verification.get("supported_claims") or summary.get("supported_claims"))
    unsupported_claims = _safe_int(
        verification.get("unsupported_claims") or summary.get("unsupported_claims")
    )
    support_ratio = _safe_float(verification.get("support_ratio") or summary.get("support_ratio"))
    if total_claims > 0 and support_ratio <= 0:
        support_ratio = supported_claims / total_claims
    cited_numbers = sorted(_citation_numbers(answer))
    cited_evidence = [
        row for row in evidence if _safe_int(row.get("citation_number")) in cited_numbers
    ]
    expected_terms: set[str] = set()
    for item in selected_items:
        expected_terms.update(_source_variants(_source_path(item)))
        expected_terms.update(_source_variants(_source_name(item)))
    unexpected_sources: list[str] = []
    for row in evidence:
        variants = _source_variants(row.get("source_path")) | _source_variants(row.get("source_name"))
        if not variants or not (variants & expected_terms):
            unexpected_sources.append(_text(row.get("source_name") or row.get("source_path"), limit=500))
    selected_sources_without_evidence = [
        _source_name(item) or _source_path(item)
        for item in selected_items
        if not any(_item_matches_evidence(item, row) for row in cited_evidence)
    ]
    reasons: list[str] = []
    query_scope = str(summary.get("query_scope") or "").strip().lower()
    if query_scope != "basket":
        reasons.append("query_scope_not_basket")
    if generation_quality_gate not in {"passed", "repaired", "fallback"}:
        reasons.append("generation_quality_gate_missing")
    if str(trace.get("status") or "").strip().lower() != "done":
        reasons.append("agent_not_done")
    if list(trace.get("errors") or []):
        reasons.append("agent_errors")
    if str(verification.get("evidence_status") or summary.get("evidence_status") or "").strip().lower() != "grounded":
        reasons.append("evidence_not_grounded")
    if total_claims <= 0:
        reasons.append("no_audited_claims")
    if unsupported_claims > 0 or support_ratio < 0.999:
        reasons.append("unsupported_claims")
    if not evidence:
        reasons.append("no_evidence_hits")
    if unexpected_sources:
        reasons.append("unexpected_sources")
    if selected_sources_without_evidence:
        reasons.append("selected_sources_without_evidence")
    evidence_numbers = {_safe_int(row.get("citation_number")) for row in evidence}
    unresolved_citations = [number for number in cited_numbers if number not in evidence_numbers]
    if not cited_numbers:
        reasons.append("no_visible_citations")
    if unresolved_citations:
        reasons.append("unresolved_citations")
    quality = {
        "contract_version": 1,
        "query_scope": query_scope,
        "evidence_status": str(verification.get("evidence_status") or summary.get("evidence_status") or ""),
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "support_ratio": round(max(0.0, min(1.0, support_ratio)), 4),
        "evidence_hit_count": len(evidence),
        "selected_source_count": len(selected_items),
        "citation_numbers": cited_numbers,
        "unresolved_citations": unresolved_citations,
        "unexpected_sources": [item for item in unexpected_sources if item],
        "selected_sources_without_evidence": [
            item for item in selected_sources_without_evidence if item
        ],
        "generation_quality_gate": generation_quality_gate,
        "generation_mode": generation_mode,
        "warnings": ["extractive_fallback"] if generation_mode == "extractive_fallback" else [],
        "reasons": reasons,
        "edited_after_verification": False,
    }
    return ("verified" if not reasons else "needs_review"), quality


def _reference_line(item: dict[str, Any], index: int) -> str:
    authors = _text(item.get("authors"), limit=800)
    year = _text(item.get("year"), limit=40)
    title = _text(item.get("title"), limit=800) or f"Reference {index}"
    venue = _text(item.get("venue"), limit=500)
    doi = _text(item.get("doi"), limit=400)
    prefix = f"{authors}. " if authors else ""
    suffix = ". ".join(part for part in (year, venue) if part)
    doi_part = f" DOI: {doi}." if doi else ""
    return f"{prefix}{title}. {suffix}{'.' if suffix else ''}{doi_part}".strip()


def research_brief_markdown(record: dict[str, Any]) -> str:
    title = _text(record.get("title"), limit=240) or "Research brief"
    objective = _text(record.get("objective"), limit=4_000)
    content = _text(record.get("content_markdown"), limit=160_000)
    quality_status = _text(record.get("quality_status"), limit=40) or "draft"
    revision = max(1, int(record.get("revision") or 1))
    lines = [f"# {title}", "", f"> Evidence status: {quality_status}; revision: {revision}."]
    if objective:
        lines.extend(["", "## Research objective", "", objective])
    if content:
        lines.extend(["", content])
    evidence = [item for item in list(record.get("evidence") or []) if isinstance(item, dict)]
    if evidence:
        lines.extend(["", "## Evidence appendix"])
        for item in evidence:
            number = int(item.get("citation_number") or 0)
            source = _text(item.get("source_name") or item.get("source_path"), limit=500) or "Source"
            locator = _text(item.get("heading_path") or item.get("location_label"), limit=800)
            quote = _text(item.get("evidence_quote"), limit=_MAX_EVIDENCE_TEXT)
            label = f"[{number}] {source}" if number > 0 else source
            lines.extend(["", f"### {label}{f' — {locator}' if locator else ''}"])
            if quote:
                lines.extend(["", f"> {quote.replace(chr(10), chr(10) + '> ')}"])
    bibliography = [item for item in list(record.get("bibliography") or []) if isinstance(item, dict)]
    if bibliography:
        lines.extend(["", "## References", ""])
        lines.extend(f"{index}. {_reference_line(item, index)}" for index, item in enumerate(bibliography, start=1))
    return "\n".join(lines).strip() + "\n"


def _bibtex_key(item: dict[str, Any], index: int) -> str:
    authors = _text(item.get("authors"), limit=200)
    author = re.sub(r"[^A-Za-z0-9]+", "", authors.split(",")[0].split()[-1] if authors else "")
    year = re.sub(r"\D+", "", _text(item.get("year"), limit=20))
    title = re.sub(r"[^A-Za-z0-9]+", "", _text(item.get("title"), limit=120).split(" ")[0])
    return f"{author or 'source'}{year or 'nd'}{title or index}"[:80]


def _bibtex_value(value: object) -> str:
    return _text(value, limit=2_000).replace("{", "\\{").replace("}", "\\}")


def research_brief_bibtex(record: dict[str, Any]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(list(record.get("bibliography") or []), start=1):
        if not isinstance(item, dict):
            continue
        fields = {
            "title": item.get("title"),
            "author": item.get("authors"),
            "year": item.get("year"),
            "journal": item.get("venue"),
            "doi": item.get("doi"),
        }
        body = ",\n".join(
            f"  {key} = {{{_bibtex_value(value)}}}"
            for key, value in fields.items()
            if _text(value, limit=2_000)
        )
        blocks.append(f"@article{{{_bibtex_key(item, index)},\n{body}\n}}")
    return "\n\n".join(blocks).strip() + ("\n" if blocks else "")


def research_brief_ris(record: dict[str, Any]) -> str:
    blocks: list[str] = []
    for item in list(record.get("bibliography") or []):
        if not isinstance(item, dict):
            continue
        lines = ["TY  - JOUR"]
        authors = _text(item.get("authors"), limit=800)
        for author in [part.strip() for part in re.split(r";|\band\b", authors) if part.strip()]:
            lines.append(f"AU  - {author}")
        for tag, key in (("TI", "title"), ("PY", "year"), ("JO", "venue"), ("DO", "doi")):
            value = _text(item.get(key), limit=1_000)
            if value:
                lines.append(f"{tag}  - {value}")
        lines.append("ER  -")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks).strip() + ("\n" if blocks else "")


def research_brief_docx(record: dict[str, Any]) -> bytes:
    from docx import Document

    document = Document()
    document.add_heading(_text(record.get("title"), limit=240) or "Research brief", level=0)
    objective = _text(record.get("objective"), limit=4_000)
    if objective:
        document.add_heading("Research objective", level=1)
        document.add_paragraph(objective)
    content = _text(record.get("content_markdown"), limit=160_000)
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            document.add_heading(heading.group(2).strip(), level=min(4, len(heading.group(1))))
            continue
        if re.match(r"^[-*+]\s+", line):
            document.add_paragraph(re.sub(r"^[-*+]\s+", "", line), style="List Bullet")
            continue
        if re.match(r"^\d+[.)]\s+", line):
            document.add_paragraph(re.sub(r"^\d+[.)]\s+", "", line), style="List Number")
            continue
        document.add_paragraph(line.lstrip("> "))
    evidence = [item for item in list(record.get("evidence") or []) if isinstance(item, dict)]
    if evidence:
        document.add_heading("Evidence appendix", level=1)
        for item in evidence:
            number = int(item.get("citation_number") or 0)
            source = _text(item.get("source_name") or item.get("source_path"), limit=500) or "Source"
            locator = _text(item.get("heading_path") or item.get("location_label"), limit=800)
            document.add_heading(f"[{number}] {source}" if number > 0 else source, level=2)
            if locator:
                document.add_paragraph(locator)
            quote = _text(item.get("evidence_quote"), limit=_MAX_EVIDENCE_TEXT)
            if quote:
                document.add_paragraph(quote)
    bibliography = [item for item in list(record.get("bibliography") or []) if isinstance(item, dict)]
    if bibliography:
        document.add_heading("References", level=1)
        for index, item in enumerate(bibliography, start=1):
            document.add_paragraph(_reference_line(item, index), style="List Number")
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()
