from __future__ import annotations

import re
from typing import Any

from kb.evidence_text import clean_display_text, looks_low_value_citation_context, source_title_candidate


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s，。；;,)）]+", re.IGNORECASE)
_NARRATIVE_METADATA_RE = re.compile(
    r"\b(?:doi|jcr|impact\s*factor|if\s*[:：]?\s*\d|published\s+(?:in|by)|"
    r"journal|conference|venue|citation\s+count|cited\s+by)\b|"
    r"(?:发表于|发表在|期刊|会议|年份|被引|影响因子|分区|出处|来源论文|论文标题|标题是|作者是)",
    re.IGNORECASE,
)
_GENERIC_SUMMARY_RE = re.compile(
    r"\b(?:this reference is relevant|upstream paper to open next|good entry point|"
    r"this evidence supports|directly relevant)\b|"
    r"(?:这条|这篇|该)(?:引用|文献|论文|证据).{0,14}(?:相关|有用|值得打开|值得阅读|可以作为入口)",
    re.IGNORECASE,
)
_INLINE_REF_MARKER_RE = re.compile(r"\[(?:R)?\d{1,4}(?:\s*[-,;]\s*(?:R)?\d{1,4})*\]", re.IGNORECASE)


def _tokens(value: str) -> set[str]:
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if not text:
        return set()
    out = set(re.findall(r"[a-z0-9]{2,}", text))
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if len(cjk_chars) >= 2:
        out.update("".join(cjk_chars[idx : idx + 2]) for idx in range(len(cjk_chars) - 1))
    elif cjk_chars:
        out.update(cjk_chars)
    return {token for token in out if token}


def _sameish(left: str, right: str) -> bool:
    a = clean_display_text(left, max_len=620).lower()
    b = clean_display_text(right, max_len=620).lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 32 and a in b:
        return True
    if len(b) >= 32 and b in a:
        return True
    at = _tokens(a)
    bt = _tokens(b)
    if len(at) < 5 or len(bt) < 5:
        return False
    return len(at & bt) / max(1, min(len(at), len(bt))) >= 0.82


def _compact_identity(value: str) -> str:
    text = source_title_candidate(value)
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text.lower()).strip()


def _contains_identity_text(text: str, candidate: str, *, min_len: int = 22) -> bool:
    body = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(text or "").lower()).strip()
    ident = _compact_identity(candidate)
    if not body or len(ident) < min_len:
        return False
    return ident in body


def _first_sentence(value: str, *, max_len: int = 120) -> str:
    text = clean_display_text(value, max_len=max_len + 80)
    if not text:
        return ""
    parts = re.split(r"(?<=[\u3002\uff01\uff1f;!?\.])\s+", text)
    out = next((part.strip() for part in parts if part.strip()), text)
    if len(out) > max_len:
        out = out[: max(0, max_len - 1)].rstrip() + "..."
    return out


def reject_system_b_context_summary(
    summary: Any,
    *,
    context: Any = "",
    claim: Any = "",
    title: Any = "",
    source: Any = "",
    reference_entry: Any = "",
    locator: Any = "",
    takeaway: Any = "",
) -> str:
    text = clean_display_text(summary, max_len=260)
    if not text:
        return "empty"
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    if "[[CITE:" in str(summary or "") or "```" in str(summary or ""):
        return "raw_markup"
    if _INLINE_REF_MARKER_RE.search(text):
        return "raw_reference_marker"
    if _DOI_RE.search(text) or _NARRATIVE_METADATA_RE.search(text):
        return "metadata_repeated"
    if _GENERIC_SUMMARY_RE.search(text):
        return "generic"
    if (not has_cjk) and looks_low_value_citation_context(text):
        return "low_value"
    if _sameish(text, clean_display_text(context, max_len=620)):
        return "duplicates_context"
    if _sameish(text, clean_display_text(claim, max_len=420)):
        return "duplicates_claim"
    if _sameish(text, clean_display_text(reference_entry, max_len=620)):
        return "duplicates_reference"
    if _sameish(text, clean_display_text(takeaway, max_len=180)):
        return "duplicates_takeaway"
    if _contains_identity_text(text, str(title or "")) or _contains_identity_text(text, str(source or "")):
        return "metadata_repeated"
    if _contains_identity_text(text, str(locator or ""), min_len=18):
        return "duplicates_locator"
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if cjk_chars and len(cjk_chars) < 12:
        return "too_short"
    if not cjk_chars and len(_tokens(text)) < 6:
        return "too_short"
    return ""


def accept_system_b_context_summary(summary: Any, **kwargs: Any) -> str:
    text = clean_display_text(summary, max_len=220)
    return "" if reject_system_b_context_summary(text, **kwargs) else text


def build_system_b_context_summary(
    *,
    context: Any,
    claim: Any = "",
    role: Any = "",
    relation: Any = "",
    title: Any = "",
    source: Any = "",
    reference_entry: Any = "",
    locator: Any = "",
    locale: str = "",
) -> str:
    text = clean_display_text(context, max_len=520)
    if not text or looks_low_value_citation_context(text):
        return ""
    low = " ".join(str(part or "") for part in (text, claim, role, relation, title)).lower()
    if "answer_context" in low:
        return ""

    prefer_en = str(locale or "").strip().lower() == "en"
    if "missing cone" in low or "low-pass distortion" in low:
        candidate = (
            "The current paper cites it while discussing frequency loss or low-pass distortion limits in 3D microscopy."
            if prefer_en
            else "当前论文在讨论三维显微成像的频率缺失或低通失真限制时引用它，用来追溯这一成像瓶颈的来源。"
        )
    elif "admm" in low or "alternating direction method" in low:
        candidate = (
            "The current paper cites it while discussing reconstruction or optimization methods, linking the idea back to ADMM-style optimization."
            if prefer_en
            else "当前论文在讨论重建或优化方法时引用它，用来说明相关思路来自既有 ADMM 优化框架。"
        )
    elif "single-shot compressive spectral imaging" in low:
        candidate = (
            "The current paper cites it as upstream background for single-shot compressive spectral imaging."
            if prefer_en
            else "当前论文在追溯单次压缩光谱成像背景时引用它，用来补上这一成像路线的上游来源。"
        )
    elif (
        ("single-pixel" in low or "spi" in low)
        and ("focal plane" in low or "single-pixel detector" in low or "spd" in low or "cost" in low)
    ):
        candidate = (
            "The current paper cites it when contrasting single-pixel detection with focal-plane arrays in hardware or cost."
            if prefer_en
            else "当前论文在说明单像素探测相对焦平面阵列的硬件或成本差异时引用它。"
        )
    elif "structured detection" in low or ("super-resolution" in low and "sectioning" in low):
        candidate = (
            "The current paper cites it when discussing how structured detection balances resolution, SNR, and optical sectioning."
            if prefer_en
            else "当前论文在讨论结构化检测如何兼顾分辨率、信噪比和光学切片时引用它。"
        )
    elif "detector-array" in low or "calibrated detector" in low:
        candidate = (
            "The current paper cites it when discussing detector-array or calibration design, connecting the method to prior hardware approaches."
            if prefer_en
            else "当前论文在说明探测器阵列或标定设计时引用它，用来连接当前方法与已有硬件方案。"
        )
    elif re.search(r"\bcites?\b.+\bwhen\b", low):
        focus = _first_sentence(text, max_len=120)
        candidate = (
            f"The current paper cites it in this context: {focus}"
            if prefer_en and focus
            else (f"当前论文在这一语境中引用它：{focus}" if focus else "")
        )
    else:
        candidate = ""

    return accept_system_b_context_summary(
        candidate,
        context=text,
        claim=claim,
        title=title,
        source=source,
        reference_entry=reference_entry,
        locator=locator,
    )
