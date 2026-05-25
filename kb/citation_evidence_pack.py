from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

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
_REFERENCE_MARKER_RE = re.compile(r"\[\s*\d{1,4}(?:\s*[-,;]\s*\d{1,4})*\s*\]")


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


def _is_low_value_answer_claim(value: str) -> bool:
    text = clean_display_text(value, max_len=420)
    if not text:
        return True
    stripped = _REFERENCE_MARKER_RE.sub("", text).strip()
    if not stripped:
        return True
    if _LOW_VALUE_LABEL_RE.fullmatch(stripped):
        return True
    tokens = _tokens(stripped)
    if _has_cjk(stripped):
        return len(stripped) < 18 and not _SENTENCE_CUE_RE.search(stripped)
    return len(tokens) <= 4 and not _SENTENCE_CUE_RE.search(stripped)


def meaningful_answer_claim(value: Any, *, max_len: int = 420) -> tuple[str, bool]:
    claim = clean_display_text(value, max_len=max_len)
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


def _claim_support_sentence(*, claim: str, evidence: str, route: str) -> str:
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
    sample = " / ".join(overlap[:4])
    if route == "system_b":
        return f"这条上游引用和回答中的说法共享关键线索：{sample}。"
    return f"回答句和原文证据共享关键线索：{sample}。"


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
) -> CitationEvidencePack:
    claim, low_claim = meaningful_answer_claim(answer_claim, max_len=420)
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
        support = _claim_support_sentence(claim=claim, evidence=evidence, route="system_a")
    return CitationEvidencePack(
        route="system_a",
        answer_claim=claim,
        evidence_quote=evidence,
        evidence_label="原文证据",
        evidence_focus=_evidence_focus(claim=claim, evidence=evidence, route="system_a"),
        support_explanation=support,
        location_label=clean_display_text(location_label, max_len=260),
        location_label_name="原文位置",
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
) -> CitationEvidencePack:
    claim, low_claim = meaningful_answer_claim(answer_claim, max_len=420)
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
    answer_context_only = bool(context and source_key == "answer_context")
    reference_entry = _reference_entry(raw_reference, max_len=900)
    support = clean_display_text(relation_hint, max_len=420) or clean_display_text(role_hint, max_len=420)
    if not support and context and claim:
        support = _claim_support_sentence(claim=claim, evidence=context, route="system_b")

    flags: list[str] = []
    score_delta = 0.0
    if low_claim:
        flags.append("low_value_answer_claim")
        score_delta -= 0.04
    if answer_context_only:
        flags.append("answer_context_only")
        score_delta -= 0.08
    if weak_context:
        flags.append("weak_citation_context")
        score_delta -= 0.08
    if not context:
        flags.append("missing_citation_context")
        score_delta -= 0.06
        if reference_entry:
            flags.append("reference_entry_only")
            support = "暂未拿到上游论文正文；这张卡展示的是当前论文中的引用位置和参考文献条目。"

    evidence_label = "回答里的线索" if answer_context_only else "当前论文引用语境"
    warning = ""
    if answer_context_only:
        warning = "目前只有回答线索或参考条目，完整引用语境仍建议打开原文核对。"
    elif not context and reference_entry:
        warning = "目前没有上游论文正文证据，只能先依据当前论文引用处和参考文献条目判断来源。"
    elif weak_context:
        warning = "当前自动抽取的引用语境质量较弱，已隐藏低价值片段；建议打开原文核对。"

    return CitationEvidencePack(
        route="system_b",
        answer_claim=claim,
        evidence_quote=context,
        evidence_label=evidence_label,
        support_explanation=support,
        location_label=clean_display_text(location_label, max_len=260),
        location_label_name="当前论文引用处",
        reference_entry=reference_entry,
        reference_label="上游文献条目",
        warning=warning,
        flags=tuple(flags),
        score_delta=score_delta,
    )
