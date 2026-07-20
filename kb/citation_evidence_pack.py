from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from kb.citation_context_summary import build_system_b_context_summary
from kb.evidence_text import (
    clean_display_text,
    looks_low_value_citation_context,
    pick_readable_evidence_text,
)


_SENTENCE_CUE_RE = re.compile(r"[\u3002\uff01\uff1f\uff1b\uff0c\uff1a;:,.!?]")
_LOW_VALUE_LABEL_RE = re.compile(
    r"^(?:[A-Za-z][A-Za-z\s-]{2,56}\s+\d{1,3}|"
    r"(?:deep learning|review|method|baseline|principles?|introduction)\s*\d{0,3})$",
    re.IGNORECASE,
)
_READING_ROADMAP_BIBLIO_LABEL_RE = re.compile(
    r"^(?:(?:\u5148|\u518d|\u540e)(?:\u8bfb|\u770b|\u9605\u8bfb)|"
    r"\u6700\u540e(?:\u8bfb|\u770b|\u9605\u8bfb)|"
    r"first\s+read|next\s+read|then\s+read|read\s+next|finally\s+read)"
    r".{0,44}(?:\u300a[^\u300b]{8,}\u300b|[\"“][^\"”]{8,}[\"”])"
    r"(?:\s*[\(\uff08][^\)\uff09]{2,80}[\)\uff09])?\s*$",
    re.IGNORECASE,
)
_BIBLIO_LABEL_ONLY_RE = re.compile(
    r"^(?:\u6587\u732e|\u8bba\u6587|\u6587\u7ae0|\u53c2\u8003\u6587\u732e|paper|source|reference)\s*[:\uff1a]\s*"
    r".{8,240}(?:[\(\uff08][^\)\uff09]*(?:18|19|20)\d{2}[^\)\uff09]*[\)\uff09])?\s*$",
    re.IGNORECASE,
)
_BIBLIO_TITLE_ONLY_RE = re.compile(
    r"^(?:[*_`]{0,2})?(?:\u300a[^\u300b]{8,220}\u300b|[\"'\u201c\u201d][^\"'\u201c\u201d]{8,220}[\"'\u201c\u201d])"
    r"(?:[*_`]{0,2})?(?:\s*[\(\uff08][^\)\uff09]{2,120}(?:(?:18|19|20)\d{2}|review|express|"
    r"photonics|optica|nature|science)[^\)\uff09]*[\)\uff09])?\s*$",
    re.IGNORECASE,
)
_BIBLIO_ROLE_TITLE_RE = re.compile(
    r"^(?:[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9\s_-]{1,28})[:\uff1a]\s*"
    r"(?:\u300a)?[A-Za-z][^\n]{10,220}(?:\u300b)?\s*"
    r"[\(\uff08][^\)\uff09]*(?:18|19|20)\d{2}[^\)\uff09]*[\)\uff09]\s*$",
    re.IGNORECASE,
)
_BIBLIO_NARRATIVE_TITLE_RE = re.compile(
    r"^(?:.{0,140})(?:"
    r"\u4e0b\u4e00\u6b65|\u4f8b\u5982|\u67e5\u9605|\u8bfb\u5b8c|\u5e94\u7528\u65b9\u5411|"
    r"next\s+step|for\s+example|e\.g\.|consult|read\s+after"
    r").{0,260}[\(\uff08][^\)\uff09]*(?:18|19|20)\d{2}[^\)\uff09]*[\)\uff09].*$",
    re.IGNORECASE,
)
_READING_INSTRUCTION_CLAIM_RE = re.compile(
    r"^(?:.{0,28})?(?:"
    r"\u91cd\u70b9(?:\u9605\u8bfb|\u770b)|"
    r"\u5173\u952e(?:\u9605\u8bfb|\u770b)|"
    r"\u5efa\u8bae(?:\u9605\u8bfb|\u770b)|"
    r"\u53ef(?:\u8fdb\u4e00\u6b65)?(?:\u9605\u8bfb|\u770b)|"
    r"\u82e5.{0,40}\u611f\u5174\u8da3.{0,16}\u53ef(?:\u770b|\u9605\u8bfb)|"
    r"read\s+(?:the\s+)?(?:section|part|chapter|paper|review)|"
    r"focus\s+on|look\s+at"
    r").{0,320}(?:"
    r"\u300a[^\u300b]{4,}\u300b|"
    r"[\"'\u201c\u201d][^\"'\u201c\u201d]{4,}[\"'\u201c\u201d]|"
    r"\u90e8\u5206|\u8282|section|chapter"
    r").*$",
    re.IGNORECASE,
)
_REFERENCE_MARKER_RE = re.compile(r"\[\s*[Rr]?\d{1,4}(?:\s*[-,;]\s*[Rr]?\d{1,4})*\s*\]")
_DOC_FRONT_RE = re.compile(r"^\s{0,3}#\s+.+?\n.{0,420}?\b(?:abstract|single[-\s]?pixel|deep learning|this paper|in this review)\b", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class CitationEvidencePack:
    route: str
    answer_claim: str = ""
    evidence_quote: str = ""
    evidence_label: str = ""
    evidence_focus: str = ""
    support_explanation: str = ""
    location_label: str = ""
    location_label_name: str = ""
    reference_entry: str = ""
    reference_label: str = ""
    warning: str = ""
    flags: tuple[str, ...] = field(default_factory=tuple)
    score_delta: float = 0.0


def _tokens(value: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", str(value or ""))]


def _has_cjk(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(value or "")))


def _first_sentence(value: str, *, max_len: int = 120) -> str:
    text = clean_display_text(value, max_len=max_len + 80)
    if not text:
        return ""
    parts = re.split(r"(?<=[\u3002\uff01\uff1f;!?\.])\s+", text)
    out = next((part.strip() for part in parts if part.strip()), text)
    if len(out) > max_len:
        out = out[: max(0, max_len - 1)].rstrip() + "..."
    return out


def _finish_answer_claim(value: Any, *, max_len: int = 220) -> str:
    text = clean_display_text(value, max_len=max_len * 2)
    if not text:
        return ""
    text = _REFERENCE_MARKER_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n,，;；:：")
    text = re.sub(r"^\s*(?:\d{1,3}[.)、．]|[-*•])\s*", "", text).strip()
    if len(text) <= max_len:
        return text
    head = text[:max_len].rstrip()
    cut = max(
        head.rfind("。"),
        head.rfind("！"),
        head.rfind("？"),
        head.rfind("；"),
        head.rfind(";"),
        head.rfind(". "),
    )
    if cut >= 40:
        return head[: cut + 1].strip()
    soft_cut = max(head.rfind("，"), head.rfind(","), head.rfind("："), head.rfind(":"))
    if soft_cut >= 60:
        return f"{head[:soft_cut].strip()}..."
    return f"{head[: max(0, max_len - 1)].strip()}..."


def _is_low_value_answer_claim(value: str) -> bool:
    text = clean_display_text(value, max_len=420)
    if not text:
        return True
    stripped = _REFERENCE_MARKER_RE.sub("", text).strip()
    if not stripped:
        return True
    if _LOW_VALUE_LABEL_RE.fullmatch(stripped):
        return True
    if _READING_ROADMAP_BIBLIO_LABEL_RE.match(stripped):
        return True
    if _BIBLIO_LABEL_ONLY_RE.match(stripped):
        return True
    if _BIBLIO_TITLE_ONLY_RE.match(stripped):
        return True
    if _BIBLIO_ROLE_TITLE_RE.match(stripped):
        return True
    if _BIBLIO_NARRATIVE_TITLE_RE.match(stripped):
        return True
    if _READING_INSTRUCTION_CLAIM_RE.match(stripped):
        return True
    tokens = _tokens(stripped)
    if _has_cjk(stripped):
        return len(stripped) < 18 and not _SENTENCE_CUE_RE.search(stripped)
    return len(tokens) <= 4 and not _SENTENCE_CUE_RE.search(stripped)


def meaningful_answer_claim(value: Any, *, max_len: int = 420) -> tuple[str, bool]:
    claim = _finish_answer_claim(value, max_len=max_len)
    if not claim:
        return "", False
    if _is_low_value_answer_claim(claim):
        return "", True
    return claim, False


def _reference_entry(value: Any, *, max_len: int = 900) -> str:
    text = clean_display_text(value, max_len=max_len)
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _meaningful_token_set(value: str) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "paper",
        "study",
        "work",
        "section",
        "introduction",
        "abstract",
        "method",
        "methods",
    }
    return {token for token in _tokens(value) if len(token) >= 3 and token not in stop}


def _sanitize_location_label_for_evidence(*, location_label: str, evidence_raw: Any, evidence: str, claim: str) -> str:
    loc = clean_display_text(location_label, max_len=260)
    if not loc:
        return ""
    raw = str(evidence_raw or "").strip()
    if raw and _DOC_FRONT_RE.search(raw) and not re.search(r"\b(?:abstract|introduction)\b", loc, re.IGNORECASE):
        return "Abstract"

    parts = [part.strip() for part in re.split(r"\s*/\s*", loc) if part.strip()]
    if len(parts) < 2:
        return loc
    leaf = parts[-1]
    leaf_tokens = _meaningful_token_set(leaf)
    if not leaf_tokens:
        return loc
    context_tokens = _meaningful_token_set(f"{evidence} {claim}")
    if not context_tokens:
        return loc
    leaf_overlap = len(leaf_tokens & context_tokens)
    parent_tokens = _meaningful_token_set(" / ".join(parts[:-1]))
    parent_overlap = len(parent_tokens & context_tokens)
    if leaf_overlap == 0 and parent_overlap >= 2:
        return " / ".join(parts[:-1])
    return loc


def _prefer_en_locale(locale: str) -> bool:
    return str(locale or "").strip().lower() == "en"


def _claim_support_sentence(*, claim: str, evidence: str, route: str, locale: str = "") -> str:
    if not claim or not evidence:
        return ""
    claim_tokens = set(_tokens(claim))
    evidence_tokens = set(_tokens(evidence))
    overlap = [
        token
        for token in claim_tokens & evidence_tokens
        if len(token) >= 3 and token not in {"the", "and", "this", "that", "with", "from", "paper", "method"}
    ]
    if len(overlap) < 2:
        return ""
    claim_low = str(claim or "").lower()
    evidence_low = str(evidence or "").lower()
    prefer_en = _prefer_en_locale(locale)
    if (
        re.search(r"\b(?:real[- ]?time|imaging speed|frame rate|faster)\b|实时|帧率", claim_low)
        and re.search(r"\b(?:real[- ]?time|imaging speed|frame rate|faster|\d+\s*(?:fps|hz))\b|实时|帧率", evidence_low)
    ):
        return (
            "The source reports the speed or real-time result stated in the answer."
            if prefer_en
            else "原文直接报告了回答所述的成像速度或实时性能。"
        )
    if (
        re.search(r"\b(?:degradation[- ]?robust|domain shift|generalization|robustness)\b|退化鲁棒|域偏移|泛化", claim_low)
        and re.search(r"\b(?:degradation[- ]?robust|domain shift|generalization|robustness)\b|退化鲁棒|域偏移|泛化", evidence_low)
    ):
        return (
            "The source directly reports the robustness or cross-domain generalization claimed in the answer."
            if prefer_en
            else "原文直接报告了回答所述的退化鲁棒性或跨域泛化结果。"
        )
    if (
        re.search(r"\b(?:resolution|image quality|psnr|ssim|low[- ]?light)\b|分辨率|图像质量|低照度", claim_low)
        and re.search(r"\b(?:resolution|image quality|psnr|ssim|low[- ]?light)\b|分辨率|图像质量|低照度", evidence_low)
    ):
        return (
            "The source provides the image-quality or resolution evidence used by the answer."
            if prefer_en
            else "原文给出了回答所依据的图像质量或分辨率证据。"
        )
    # Shared keywords alone are useful for ranking, but not meaningful enough
    # to expose as a support explanation to the user.
    return ""


def _evidence_focus(*, claim: str, evidence: str, route: str) -> str:
    if route == "system_b":
        return ""
    sentence = _first_sentence(evidence, max_len=118)
    if not sentence:
        return ""
    if _has_cjk(sentence):
        return sentence
    return ""


def build_system_a_evidence_pack(
    *,
    answer_claim: Any,
    evidence_raw: Any,
    source: str = "",
    title: str = "",
    heading: str = "",
    location_label: str = "",
    support_hint: Any = "",
    locale: str = "",
) -> CitationEvidencePack:
    claim, low_claim = meaningful_answer_claim(answer_claim, max_len=220)
    evidence = pick_readable_evidence_text(
        evidence_raw,
        source=source,
        title=title,
        claim=claim,
        heading=heading,
        max_len=460,
    )
    flags: list[str] = []
    score_delta = 0.0
    if low_claim:
        flags.append("low_value_answer_claim")
        score_delta -= 0.04
    if clean_display_text(evidence_raw, max_len=1200) and not evidence:
        flags.append("evidence_quote_filtered")
        score_delta -= 0.06
    support = clean_display_text(support_hint, max_len=420)
    if not support:
        support = _claim_support_sentence(claim=claim, evidence=evidence, route="system_a", locale=locale)
    safe_location = _sanitize_location_label_for_evidence(
        location_label=location_label,
        evidence_raw=evidence_raw,
        evidence=evidence,
        claim=claim,
    )
    return CitationEvidencePack(
        route="system_a",
        answer_claim=claim,
        evidence_quote=evidence,
        evidence_label="Source evidence" if _prefer_en_locale(locale) else "原文证据",
        evidence_focus=_evidence_focus(claim=claim, evidence=evidence, route="system_a"),
        support_explanation=support,
        location_label=safe_location,
        location_label_name="Source location" if _prefer_en_locale(locale) else "原文位置",
        flags=tuple(flags),
        score_delta=score_delta,
    )


def build_system_b_evidence_pack(
    *,
    answer_claim: Any,
    citation_context_raw: Any,
    citation_context_source: str = "",
    source: str = "",
    title: str = "",
    heading: str = "",
    location_label: str = "",
    raw_reference: Any = "",
    role_hint: Any = "",
    relation_hint: Any = "",
    locale: str = "",
) -> CitationEvidencePack:
    claim, low_claim = meaningful_answer_claim(answer_claim, max_len=220)
    context = pick_readable_evidence_text(
        citation_context_raw,
        source=source,
        title=title,
        claim=claim,
        heading=heading,
        max_len=520,
    )
    weak_context = bool(clean_display_text(citation_context_raw, max_len=1200) and not context)
    if context and looks_low_value_citation_context(context):
        context = ""
        weak_context = True

    source_key = str(citation_context_source or "").strip().lower()
    answer_context_only = source_key in {"answer_context", "answer_reference_mention"}
    reference_entry = _reference_entry(raw_reference, max_len=900)
    support = clean_display_text(relation_hint, max_len=420) or clean_display_text(role_hint, max_len=420)
    if not support and context and claim:
        support = _claim_support_sentence(claim=claim, evidence=context, route="system_b", locale=locale)
    context_summary = ""

    flags: list[str] = []
    score_delta = 0.0
    if low_claim:
        flags.append("low_value_answer_claim")
        score_delta -= 0.04
    if answer_context_only:
        flags.append("answer_context_only")
        score_delta -= 0.08
    else:
        context_summary = build_system_b_context_summary(
            context=context,
            claim=claim,
            title=title,
            source=source,
            reference_entry=reference_entry,
            locator=location_label,
            role=clean_display_text(role_hint, max_len=260),
            relation=clean_display_text(relation_hint, max_len=260),
            locale=locale,
        )
    if weak_context:
        flags.append("weak_citation_context")
        score_delta -= 0.08
    if not context:
        flags.append("missing_citation_context")
        score_delta -= 0.06
        if reference_entry:
            flags.append("reference_entry_only")
            support = (
                "The upstream paper text is not available yet; this card shows where the citation appears and the bibliography entry."
                if _prefer_en_locale(locale)
                else "暂未拿到上游论文正文；这张卡展示的是引用出现位置和参考文献条目。"
            )

    evidence_label = (
        ("Answer cue" if answer_context_only else "Citation context")
        if _prefer_en_locale(locale)
        else ("回答里的线索" if answer_context_only else "引用语境")
    )
    warning = ""
    if answer_context_only:
        warning = (
            "Only the answer cue or bibliography entry is available, so the full citation context should still be checked in the source."
            if _prefer_en_locale(locale)
            else "目前只有回答线索或参考条目，完整引用语境仍建议打开原文核对。"
        )
    elif not context and reference_entry:
        warning = (
            "No upstream full-text evidence is available yet; use the citing location and bibliography entry as source clues."
            if _prefer_en_locale(locale)
            else "目前没有上游论文正文证据，只能先依据引用出现位置和参考文献条目判断来源。"
        )
    elif weak_context:
        warning = (
            "The extracted citation context is weak, so low-value text was hidden. Open the source to verify it."
            if _prefer_en_locale(locale)
            else "当前自动抽取的引用语境质量较弱，已隐藏低价值片段；建议打开原文核对。"
        )

    return CitationEvidencePack(
        route="system_b",
        answer_claim=claim,
        evidence_quote=context,
        evidence_label=evidence_label,
        evidence_focus=context_summary,
        support_explanation=support,
        location_label=clean_display_text(location_label, max_len=260),
        location_label_name="Where current paper cites it" if _prefer_en_locale(locale) else "当前论文引用处",
        reference_entry=reference_entry,
        reference_label="Upstream reference entry" if _prefer_en_locale(locale) else "上游文献条目",
        warning=warning,
        flags=tuple(flags),
        score_delta=score_delta,
    )
