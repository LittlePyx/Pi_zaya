from __future__ import annotations

import re


GENERIC_REF_WHY_PATTERNS: tuple[str, ...] = (
    "this hit is directly relevant",
    "directly relevant because",
    "good entry point",
    "directly responds to the user's question",
    "aligns with the user's question",
    "matched section",
    "matched passage",
    "source section",
    "available evidence",
    "matched evidence",
    "current evidence",
    "这条命中",
    "本条命中",
    "直接相关",
    "直接回应",
    "定位入口",
    "定位切口",
    "导读入口",
    "定义、方法或结果信息",
    "关键证据来源",
    "命中章节讲什么",
    "提供什么",
    "当前命中证据",
    "保守说明",
)


def normalize_ref_card_copy(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", s)
    s = re.sub(r"([（(])\s+", r"\1", s)
    s = re.sub(r"\s+([）)])", r"\1", s)
    return s.strip()


def looks_generic_ref_why_line(text: str) -> bool:
    s = normalize_ref_card_copy(text)
    if not s:
        return True
    low = s.lower()
    starts_with_hit_shell = bool(
        re.match(r"^(?:this hit|this match|this card)\b", low)
        or re.match(r"^(?:这条命中|本条命中|该命中|这条卡片|该卡片)", s)
    )
    if starts_with_hit_shell:
        return True
    if "..." in s and re.search(r"\b(which|what|where|how|why)\b", low):
        return True
    if re.search(r"\b(which paper|in my library|point me to|source section)\b", low):
        return True
    has_specific_signal = bool(
        re.search(r"[“\"'][^“\"']{3,120}[”\"']", s)
        or re.search(r"\b(?:section|related work|method|experiment|figure|table)\b", low)
        or re.search(r"\b[A-Z][A-Z0-9-]{2,}\b", s)
        or re.search(r"(明确提及|定义|解释|比较|讨论|指出|提到)", s)
        or re.search(r"\b(?:defines?|explains?|compares?|mentions?|states?|discusses?)\b", low)
    )
    prompt_echo = bool(
        re.search(r"\b(?:directly responds? to|user(?:'s)? question|query)\b", low)
        or re.search(r"(直接回应|用户查询|当前问题)", s)
    )
    if prompt_echo and not has_specific_signal:
        return True
    generic_patterns = tuple(
        pattern
        for pattern in GENERIC_REF_WHY_PATTERNS
        if pattern not in {"直接回应", "directly responds to the user's question"}
    )
    return any(pattern.lower() in low for pattern in generic_patterns)


def looks_templated_ref_why_line(text: str) -> bool:
    s = normalize_ref_card_copy(text)
    if not s:
        return False
    low = s.lower()
    return any(pattern.lower() in low for pattern in GENERIC_REF_WHY_PATTERNS)


def _compact_heading_leaf(heading_path: str) -> str:
    parts = [part.strip() for part in str(heading_path or "").split(" / ") if part.strip()]
    if not parts:
        return ""
    leaf = parts[-1]
    if len(leaf) > 90:
        leaf = leaf[:87].rstrip() + "..."
    return leaf


def _compact_terms(focus_terms: list[str] | tuple[str, ...], *, max_terms: int = 2) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(focus_terms or []):
        term = normalize_ref_card_copy(str(raw or ""))
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= max(1, int(max_terms or 2)):
            break
    return out


def _compact_summary_fragment(summary_line: str, *, prefer_zh: bool) -> str:
    s = normalize_ref_card_copy(summary_line)
    if not s:
        return ""
    parts = re.split(r"(?<=[。！？.!?])\s+|[；;]\s*", s)
    fragment = next((part.strip() for part in parts if part.strip()), s)
    limit = 88 if prefer_zh else 128
    if len(fragment) > limit:
        fragment = fragment[: limit - 3].rstrip(" ,，。.;；:：") + "..."
    return fragment


def build_grounded_ref_why_line(
    *,
    prefer_zh: bool,
    focus_terms: list[str] | tuple[str, ...],
    heading_path: str,
    summary_line: str = "",
    action: str = "",
) -> str:
    terms = _compact_terms(focus_terms)
    loc = _compact_heading_leaf(heading_path)
    summary = _compact_summary_fragment(summary_line, prefer_zh=prefer_zh)
    action_norm = str(action or "").strip().lower()
    if prefer_zh:
        if terms and loc:
            verb = "比较" if action_norm == "compare" else ("解释" if action_norm == "define" else "讨论")
            return f"可用来核对“{loc}”里怎样{verb}“{' / '.join(terms)}”。"
        if terms:
            return f"可用来核对论文怎样使用“{' / '.join(terms)}”这条线索。"
        if loc and summary:
            return f"可用来核对“{loc}”中的原文表述：{summary}"
        if loc:
            return f"可用来核对“{loc}”中的具体原文表述。"
        if summary:
            return f"可用来核对原文中的这句证据：{summary}"
        return ""

    if terms and loc:
        verb = "compares" if action_norm == "compare" else ("defines or explains" if action_norm == "define" else "discusses")
        return f'Use "{loc}" to check how the paper {verb} "{ " / ".join(terms) }".'
    if terms:
        return f'Use this evidence to check how the paper uses "{ " / ".join(terms) }".'
    if loc and summary:
        return f'Use "{loc}" to check this source wording: {summary}'
    if loc:
        return f'Use "{loc}" to check the paper\'s source wording.'
    if summary:
        return f"Use this source wording as the evidence anchor: {summary}"
    return ""


def finalize_ref_card_copy(
    *,
    summary_line: str,
    why_line: str,
    prefer_zh: bool,
    focus_terms: list[str] | tuple[str, ...],
    heading_path: str,
    action: str = "",
) -> tuple[str, str, bool]:
    summary = normalize_ref_card_copy(summary_line)
    why = normalize_ref_card_copy(why_line)
    changed = why != str(why_line or "").strip()
    if looks_generic_ref_why_line(why) or looks_templated_ref_why_line(why):
        grounded = build_grounded_ref_why_line(
            prefer_zh=prefer_zh,
            focus_terms=focus_terms,
            heading_path=heading_path,
            summary_line=summary,
            action=action,
        )
        if grounded:
            why = grounded
            changed = True
    return summary, why, changed
