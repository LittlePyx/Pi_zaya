from __future__ import annotations

from pathlib import PurePath, PureWindowsPath
from typing import Any


def _top_heading(heading_path: str) -> str:
    hp = (heading_path or "").strip()
    if not hp:
        return ""
    # Our heading_path uses " / " as a stack separator.
    return hp.split(" / ", 1)[0].strip()


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return "." * max_chars
    return text[: max_chars - 3].rstrip() + "..."


def _compact_heading_path(heading_path: str, max_chars: int = 180) -> str:
    hp = " / ".join(part.strip() for part in (heading_path or "").split(" / ") if part.strip())
    if len(hp) <= max_chars:
        return hp

    parts = hp.split(" / ")
    if len(parts) >= 2:
        compact = f"{parts[0]} / ... / {parts[-1]}"
        if len(compact) <= max_chars:
            return compact
    return _truncate_text(hp, max_chars)


def _path_name(path_text: str) -> str:
    path_text = (path_text or "").strip()
    if not path_text:
        return ""
    try:
        if "\\" in path_text:
            return PureWindowsPath(path_text).name or path_text
        return PurePath(path_text).name or path_text
    except Exception:
        return path_text


def _source_label(meta: dict[str, Any]) -> str:
    for key in ("source_name", "display_name", "title", "doc_title", "paper_title"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value

    source_path = str(meta.get("source_path") or meta.get("md_path") or "").strip()
    name = _path_name(source_path)
    if name:
        return name
    doc_id = str(meta.get("doc_id") or meta.get("source_sha1") or "").strip()
    return doc_id


def _source_path_prompt_label(path_text: str) -> str:
    return _path_name(path_text) or str(path_text or "").strip()


def _format_score(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.2f}".rstrip("0").rstrip(".")
    except Exception:
        text = str(value).strip()
        return text[:32]


def _page_label(meta: dict[str, Any]) -> str:
    p0 = meta.get("page_start", None)
    p1 = meta.get("page_end", None)
    try:
        if p0 is None:
            return ""
        page0 = int(p0)
        if p1 is not None and int(p1) != page0:
            return f"pages: {page0}-{int(p1)}"
        return f"page: {page0}"
    except Exception:
        return ""


def _format_context(hits: list[dict], max_chars: int = 12000) -> str:
    parts: list[str] = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        meta = hit.get("meta", {}) or {}
        if not isinstance(meta, dict):
            meta = {}

        body = str(hit.get("text", "") or "").strip()
        if not body:
            continue

        header_parts = [f"[{i}]"]
        doc_label = _source_label(meta)
        if doc_label:
            header_parts.append(f"doc: {doc_label}")

        source_path = str(meta.get("source_path") or meta.get("md_path") or "").strip()
        if source_path:
            header_parts.append(f"source: {_source_path_prompt_label(source_path)}")

        heading = _compact_heading_path(str(meta.get("heading_path") or meta.get("top_heading") or ""))
        if heading:
            header_parts.append(f"section: {heading}")

        page = _page_label(meta)
        if page:
            header_parts.append(page)

        score = _format_score(hit.get("score", meta.get("score", meta.get("bm25_score"))))
        if score:
            header_parts.append(f"score: {score}")

        chunk_id = str(
            meta.get("chunk_id")
            or meta.get("block_id")
            or meta.get("anchor_id")
            or hit.get("id")
            or ""
        ).strip()
        if chunk_id:
            header_parts.append(f"id: {chunk_id}")

        chunk = " | ".join(header_parts) + "\n" + body
        if used and used + len(chunk) > max_chars:
            break
        if not parts and len(chunk) > max_chars:
            chunk = _truncate_text(chunk, max_chars)
        parts.append(chunk)
        used += len(chunk)
    return "\n\n---\n\n".join(parts)


def _has_cjk(text: str) -> bool:
    """Return True when the text is mostly asking in Chinese/Japanese/Korean."""
    if not text:
        return False
    cjk_count = sum(1 for c in text if "\u3400" <= c <= "\u9fff" or "\uf900" <= c <= "\ufaff")
    return cjk_count >= 4


def _labels(prefer_zh: bool) -> dict[str, str]:
    if prefer_zh:
        return {
            "answer_lang": "中文",
            "query": "问题",
            "snippets": "检索片段",
            "conclusion": "结论",
            "evidence": "证据",
            "limits": "限制",
            "next_steps": "下一步",
            "refs": "可参考定位",
            "no_hits": "未命中知识库片段",
            "no_refs": "（本次未命中）",
        }
    return {
        "answer_lang": "English",
        "query": "Question",
        "snippets": "Retrieved snippets",
        "conclusion": "Conclusion",
        "evidence": "Evidence",
        "limits": "Limits",
        "next_steps": "Next Steps",
        "refs": "Referenced Sources",
        "no_hits": "(No relevant snippets found in the knowledge base)",
        "no_refs": "(No hits this time)",
    }


def _system_prompt(prefer_zh: bool) -> str:
    labels = _labels(prefer_zh)
    return (
        "You are the user's academic knowledge-base assistant. "
        f"Read the retrieved snippets first, then answer in {labels['answer_lang']}.\n\n"
        "Grounding contract:\n"
        "1. Answer the exact research question first; do not give generic reading advice.\n"
        "2. Use retrieved snippets as the authority for paper-specific claims.\n"
        "3. Every paper-specific claim should carry a citation marker like [1] or [2].\n"
        "4. If evidence is weak or absent, say what is missing instead of inventing papers, formulas, data, or conclusions.\n"
        "5. For multi-paper questions, compare the papers directly instead of summarizing them one by one.\n"
        "6. Cite only snippets that support the sentence where the citation appears.\n\n"
        "Required answer shape:\n"
        f"{labels['conclusion']}: direct answer in 1-3 sentences, with citations when snippets exist.\n"
        f"{labels['evidence']}: compact bullets; each bullet pairs one claim with the snippet id(s) that support it.\n"
        f"{labels['limits']}: what the retrieved evidence cannot prove, or why confidence is limited.\n"
        f"{labels['next_steps']}: one practical follow-up, such as the section/figure/table to inspect next.\n"
        f"{labels['refs']}: list only the sources actually cited, with doc, section, page if available, and a short reason.\n\n"
        "No-hit behavior:\n"
        f"- If there are no retrieved snippets, start with \"{labels['no_hits']}\".\n"
        f"- Then give a clearly labeled general answer, and write \"{labels['no_refs']}\" under {labels['refs']}.\n"
    )


def build_messages(
    user_query: str,
    history: list[dict],
    hits: list[dict],
) -> list[dict]:
    prefer_zh = _has_cjk(user_query)
    labels = _labels(prefer_zh)
    ctx = _format_context(hits)
    user = (
        f"{labels['query']}:\n"
        f"{user_query}\n\n"
        f"{labels['snippets']}:\n"
        f"{ctx if ctx else labels['no_refs']}\n"
    )

    # Only keep user/assistant roles in history.
    trimmed = [m for m in history if m.get("role") in ("user", "assistant")]
    return [{"role": "system", "content": _system_prompt(prefer_zh)}, *trimmed, {"role": "user", "content": user}]
