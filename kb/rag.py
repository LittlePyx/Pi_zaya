from __future__ import annotations


def _top_heading(heading_path: str) -> str:
    hp = (heading_path or "").strip()
    if not hp:
        return ""
    # Our heading_path uses " / " as a stack separator.
    return hp.split(" / ", 1)[0].strip()


def _format_context(hits: list[dict], max_chars: int = 12000) -> str:
    parts: list[str] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        meta = h.get("meta", {}) or {}
        header = f"[{i}] source: {meta.get('source_path', '')}"
        top = _top_heading(meta.get("heading_path", ""))
        if top:
            header += f" | section: {top}"
        p0 = meta.get("page_start", None)
        p1 = meta.get("page_end", None)
        try:
            if p0 is not None:
                if p1 is not None and int(p1) != int(p0):
                    header += f" | pages: {int(p0)}-{int(p1)}"
                else:
                    header += f" | page: {int(p0)}"
        except Exception:
            pass
        body = h.get("text", "")
        chunk = header + "\n" + body
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n\n---\n\n".join(parts)


def _has_cjk(text: str) -> bool:
    “””Check if text contains CJK (Chinese/Japanese/Korean) characters.”””
    if not text:
        return False
    cjk_count = sum(1 for c in text if '一' <= c <= '鿿')
    return cjk_count >= 4


def build_messages(
    user_query: str,
    history: list[dict],
    hits: list[dict],
) -> list[dict]:
    # Keep system prompt simple and strict: use context first, cite [n], don't hallucinate.
    prefer_zh = _has_cjk(user_query)
    answer_lang = “中文” if prefer_zh else “English”
    source_hint = “检索片段” if prefer_zh else “retrieved snippets”
    no_hit_notice = “未命中知识库片段” if prefer_zh else “(No relevant snippets found in the knowledge base)”
    ref_title = “可参考定位” if prefer_zh else “Referenced Sources”
    ref_miss = “（本次未命中）” if prefer_zh else “(No hits this time)”
    ref_suggest = “建议去你认为最相关的论文的 REFERENCES/Related Work 追溯” if prefer_zh else “check the REFERENCES/Related Work sections of the most relevant paper”
    system = (
        “You are my personal knowledge base assistant. Read the retrieved snippets first, “
        f”then answer in {answer_lang}.\n”
        “Rules:\n”
        f”1) If {source_hint} exist: answer based on them first; when citing, use [1] [2] markers.\n”
        f”2) If {source_hint} are empty: still give a general answer, but start with \”{no_hit_notice}\”.\n”
        “3) Do not fabricate papers, formulas, data, or conclusions.\n”
        “4) Output format:\n”
        f”   - First give the answer.\n”
        f”   - Then list \”{ref_title}:\” with the sources you actually cited, each with source + section.\n”
        f”   - If no hits: write \”{ref_miss}\” under \”{ref_title}\”, and {ref_suggest}.\n”
    )

    ctx = _format_context(hits)
    user = (
        "问题：\n"
        f"{user_query}\n\n"
        "检索片段：\n"
        f"{ctx if ctx else '(无)'}\n"
    )

    # Only keep user/assistant roles in history
    trimmed = [m for m in history if m.get("role") in ("user", "assistant")]
    return [{"role": "system", "content": system}, *trimmed, {"role": "user", "content": user}]
