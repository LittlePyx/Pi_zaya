from __future__ import annotations

import re

from api.reference_summary_text import _clean_summary_line, _summary_excerpt


def _metadata_summary_line(meta: dict) -> str:
    title = _clean_summary_line(str((meta or {}).get("title") or ""))
    venue = _clean_summary_line(str((meta or {}).get("venue") or ""))
    year = str((meta or {}).get("year") or "").strip()
    authors = _clean_summary_line(str((meta or {}).get("authors") or ""))
    author_head = ""
    if authors:
        author_head = re.split(r"[,;&]| and ", authors, maxsplit=1, flags=re.I)[0].strip()
    loc = ""
    if venue and year:
        loc = f"{venue}（{year}）"
    elif venue:
        loc = venue
    elif year:
        loc = year
    if author_head and loc:
        return (
            f"当前仅检索到文献元数据：{author_head} 的相关研究发表于 {loc}。"
            "由于缺少可用摘要文本，暂无法可靠提炼其方法细节与实验结论，建议通过 DOI 查看原文摘要与正文。"
        )
    if loc:
        return (
            f"当前仅检索到文献元数据：该工作发表于 {loc}。"
            "由于缺少可用摘要文本，暂无法可靠提炼其方法细节与实验结论，建议通过 DOI 查看原文摘要与正文。"
        )
    if title:
        return (
            "当前仅检索到题名与基础元数据，尚未获取可用摘要文本。"
            "为保证学术准确性，建议通过 DOI 查看原文摘要与正文后再进行方法和结论层面的判断。"
        )
    return (
        "当前仅检索到有限元数据，尚未获取可用摘要文本。"
        "为保证学术准确性，建议通过 DOI 查看原文摘要与正文后再进行方法和结论层面的判断。"
    )


def _contextual_summary_line(meta: dict) -> str:
    context = _summary_excerpt(
        str(
            (meta or {}).get("citation_context")
            or (meta or {}).get("card_evidence")
            or (meta or {}).get("evidence_quote")
            or ""
        ),
        max_sentences=2,
        max_len=280,
    )
    if not context:
        return ""
    claim = _summary_excerpt(
        str((meta or {}).get("answer_claim") or (meta or {}).get("card_claim") or ""),
        max_sentences=1,
        max_len=160,
    )
    location = _clean_summary_line(
        str((meta or {}).get("location_label") or (meta or {}).get("card_locator") or (meta or {}).get("heading_path") or "")
    )
    parts: list[str] = []
    if claim:
        parts.append(f"暂无可用摘要；当前回答主要借它支撑：{claim}")
    else:
        parts.append("暂无可用摘要；可先根据当前论文里的引用语境判断它在回答中的作用。")
    if location:
        parts.append(f"引用位置：{location}。")
    parts.append(f"引用语境：{context}")
    return _summary_excerpt(" ".join(parts), max_sentences=3, max_len=420)
