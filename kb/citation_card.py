from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from kb.citation_evidence_pack import build_system_a_evidence_pack, build_system_b_evidence_pack
from kb.evidence_text import (
    CITATION_CARD_EVIDENCE_MAX_LEN,
    clean_display_text,
    finish_evidence_text,
    source_title_candidate,
)

CITATION_CARD_DISPLAY_CONTRACT_VERSION = 2
CITATION_CARD_VIEW_CONTRACT_VERSION = 2
_CARD_LABELS: dict[str, dict[str, str]] = {
    "zh": {
        "support_relevance_system_a": "\u4e0e\u56de\u7b54\u7684\u5173\u7cfb",
        "warning": "提醒",
        "takeaway_system_a": "证据重点",
        "takeaway_system_b": "上游作用",
        "evidence": "原文证据",
        "source_location": "原文位置",
        "current_citation_location": "当前论文引用处",
        "context_summary": "语境摘要",
        "citation_context": "引用语境",
        "reference_entry": "上游文献条目",
        "support_system_a": "可靠度",
        "support_system_b": "说明",
        "header_system_a": "答案依据",
        "header_system_b": "上游引用",
        "title_system_a": "答案依据",
        "title_system_b": "上游参考文献",
        "answer_point": "答案要点",
        "answer_sentence": "答案里的这句话",
        "citation_location": "引用出现位置",
        "quality_system_b_high": "上游来源清楚",
        "quality_system_b_mid": "可追溯来源",
        "quality_system_b_low": "需要核对来源",
        "quality_system_a_high": "证据匹配",
        "quality_system_a_mid": "候选依据",
        "quality_system_a_low": "需要核对",
        "support_label_review": "这条依据的可靠度",
        "candidate_support": "这条引用只能作为候选依据；请打开原文核对答案句和命中片段是否真正对应。",
        "warning_mismatch": "答案句和命中片段术语冲突，已尽量抑制链接；如果仍看到这张卡，请优先打开原文核对。",
        "warning_candidate": "这条链接只是候选依据，建议打开原文确认语境。",
        "warning_upstream_incomplete": "这条上游参考信息不完整，建议打开引用语境确认。",
        "trace_complete": "答案句命中了当前论文的引用语境，该语境再指向这篇上游文献。",
        "trace_answer_context_only": "目前只拿到了答案句里的引用线索，还没有定位到当前论文正文中的引用语境。",
        "trace_missing_context": "缺少当前论文里围绕该引用的正文语境，需要打开引用语境核对。",
        "trace_reference_entry": "当前语境看起来像参考文献条目本身，不足以说明答案句如何使用了它。",
        "trace_review": "这条上游引用链还需要结合当前论文位置和参考条目核对。",
        "trace_step_answer": "答案句",
        "trace_step_context": "当前论文引用处",
        "trace_step_review": "引用语境待核对",
        "trace_step_reference": "上游文献",
    },
    "en": {
        "warning": "Note",
        "takeaway_system_a": "Evidence focus",
        "takeaway_system_b": "Upstream role",
        "evidence": "Source evidence",
        "source_location": "Source location",
        "current_citation_location": "Where current paper cites it",
        "context_summary": "Context summary",
        "citation_context": "Citation context",
        "reference_entry": "Upstream reference entry",
        "support_system_a": "Reliability",
        "support_relevance_system_a": "Why it supports the answer",
        "support_system_b": "Note",
        "header_system_a": "Answer evidence",
        "header_system_b": "Upstream citation",
        "title_system_a": "Answer evidence",
        "title_system_b": "Upstream reference",
        "answer_point": "Answer point",
        "answer_sentence": "Answer sentence",
        "citation_location": "Citation location",
        "quality_system_b_high": "Clear upstream source",
        "quality_system_b_mid": "Traceable source",
        "quality_system_b_low": "Needs source check",
        "quality_system_a_high": "Evidence matched",
        "quality_system_a_mid": "Candidate evidence",
        "quality_system_a_low": "Needs review",
        "support_label_review": "Evidence reliability",
        "candidate_support": "This citation is only candidate evidence. Open the source to confirm the answer sentence and matched passage actually correspond.",
        "warning_mismatch": "The answer sentence and matched passage use conflicting terms. The link was suppressed as much as possible; if this card still appears, check the source first.",
        "warning_candidate": "This link is candidate evidence. Open the source to confirm the context.",
        "warning_upstream_incomplete": "This upstream reference is incomplete. Open the citation context to verify it.",
        "trace_complete": "The answer sentence hits citation context in the current paper, and that context points to this upstream reference.",
        "trace_answer_context_only": "Only the citation cue in the answer sentence is available; the citing context in the current paper has not been located yet.",
        "trace_missing_context": "The surrounding citing context in the current paper is missing, so open the citation context to verify it.",
        "trace_reference_entry": "The current context looks like a bibliography entry, not enough to show how the answer sentence uses it.",
        "trace_review": "This upstream citation chain still needs checking against the current paper location and bibliography entry.",
        "trace_step_answer": "Answer sentence",
        "trace_step_context": "Current paper citation",
        "trace_step_review": "Citation context to check",
        "trace_step_reference": "Upstream reference",
    },
}


def _card_locale(value: Any = "") -> str:
    return "en" if str(value or "").strip().lower() == "en" else "zh"


def _card_label(locale: str, key: str) -> str:
    labels = _CARD_LABELS.get(_card_locale(locale), _CARD_LABELS["zh"])
    return labels.get(key) or _CARD_LABELS["zh"].get(key) or key


def _clean_text(value: Any, *, max_len: int = 520) -> str:
    return clean_display_text(value, max_len=max_len)


def _loose_tokens(value: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", str(value or ""))]


def _first_text(rec: Mapping[str, Any], *keys: str, max_len: int = 520) -> str:
    for key in keys:
        value = _clean_text(rec.get(key), max_len=max_len)
        if value:
            return value
    return ""


def _first_raw_value(rec: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = rec.get(key)
        if str(value or "").strip():
            return value
    return ""


_CARD_TEXT_LIMITS = {
    "card_title": 220,
    "card_subtitle": 220,
    "card_takeaway_label": 80,
    "card_takeaway": 140,
    "card_claim_label": 80,
    "card_claim": 220,
    "card_locator_label": 80,
    "card_locator": 260,
    "card_evidence_label": 80,
    "card_evidence": CITATION_CARD_EVIDENCE_MAX_LEN,
    "card_context_summary": 220,
    "card_reference_label": 80,
    "card_reference_entry": 520,
    "card_support_label": 80,
    "card_support_explanation": 420,
    "card_quality_label": 80,
    "card_warning": 360,
    "system_b_trace_reason": 360,
    "system_b_trace_answer": 220,
    "system_b_trace_context": 520,
    "system_b_trace_reference": 360,
    "system_b_trace_locator": 260,
}


def _is_placeholder_card_text(value: str) -> bool:
    text = _clean_text(value, max_len=160).strip().lower()
    if not text:
        return False
    return text in {
        "no summary available",
        "no notes",
        "none",
        "n/a",
        "na",
        "unknown",
        "unknown location",
        "not located",
        "source paper",
        "current paper",
    }


def _text_has_visible_markup_artifact(value: str) -> bool:
    text = str(value or "")
    if not text:
        return False
    if "[[CITE:" in text or "```" in text:
        return True
    if re.search(r"(?m)^\s{0,3}#{1,6}\s+\S", text):
        return True
    if re.search(r"(?m)^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$", text):
        return True
    if re.search(r"\$\s*\^\{\s*\[[\d,\-\s;]+\]\s*\}\s*\$", text):
        return True
    return False


def _clean_quality_flags(value: Any) -> list[str]:
    raw_values = value if isinstance(value, (list, tuple)) else []
    return _dedup_strings([str(item or "") for item in raw_values])


def _add_flag(flags: list[str], name: str) -> None:
    if name and name not in flags:
        flags.append(name)


def _card_visible_sections(out: Mapping[str, Any], *, route: str) -> list[str]:
    sections: list[str] = []
    flags = set(str(item or "") for item in (out.get("card_quality_flags") or []))
    warning = str(out.get("card_warning") or "").strip()
    takeaway = str(out.get("card_takeaway") or "").strip()
    locator = str(out.get("card_locator") or "").strip()
    evidence = str(out.get("card_evidence") or "").strip()
    support = str(out.get("card_support_explanation") or "").strip()
    if warning:
        sections.append("warning")
    if takeaway:
        sections.append("takeaway")

    if route == "system_a":
        if evidence:
            sections.append("evidence")
        if locator:
            sections.append("locator")
        if support:
            sections.append("support")
        return sections

    if locator:
        sections.append("locator")
    if evidence:
        sections.append("evidence")
    if route == "system_b" and str(out.get("card_context_summary") or "").strip():
        sections.append("context_summary")
    if (
        route == "system_b"
        and str(out.get("card_reference_entry") or "").strip()
        and (
            "missing_reference_title" in flags
            or str(out.get("card_title") or "").strip() in {"", "上游参考文献", "Upstream reference"}
        )
    ):
        sections.append("reference")
    if support:
        sections.append("support")
    return sections


def _card_view_section(
    *,
    section_id: str,
    label: str,
    text: str,
    kind: str,
    hint: str = "",
    tone: str = "",
) -> dict[str, Any]:
    return {
        "id": section_id,
        "label": _clean_text(label, max_len=80),
        "text": _clean_text(text, max_len=620),
        "kind": kind,
        "hint": _clean_text(hint, max_len=80),
        "tone": _clean_text(tone, max_len=40),
    }


def _append_card_view_section(
    sections: list[dict[str, Any]],
    *,
    section_id: str,
    label: str,
    text: str,
    kind: str,
    hint: str = "",
    tone: str = "",
) -> None:
    clean_text = _clean_text(text, max_len=620)
    if not clean_text:
        return
    for existing in sections:
        if existing.get("id") == section_id:
            return
        if _sameish(str(existing.get("text") or ""), clean_text):
            return
    sections.append(
        _card_view_section(
            section_id=section_id,
            label=label,
            text=clean_text,
            kind=kind,
            hint=hint,
            tone=tone,
        )
    )


def _build_card_view(out: Mapping[str, Any], *, route: str, locale: str = "") -> dict[str, Any]:
    is_system_b = route == "system_b"
    title = _clean_text(out.get("card_title"), max_len=220)
    subtitle = _clean_text(out.get("card_subtitle"), max_len=220)
    sections: list[dict[str, Any]] = []
    flags = _clean_quality_flags(out.get("card_quality_flags"))

    if _clean_text(out.get("card_warning"), max_len=360):
        _append_card_view_section(
            sections,
            section_id="warning",
            label=_card_label(locale, "warning"),
            text=str(out.get("card_warning") or ""),
            kind="warning",
            tone="warning",
        )
    _append_card_view_section(
        sections,
        section_id="takeaway",
        label=str(out.get("card_takeaway_label") or _card_label(locale, "takeaway_system_b" if is_system_b else "takeaway_system_a")),
        text=str(out.get("card_takeaway") or ""),
        kind="insight",
        tone="primary",
    )
    if not is_system_b:
        _append_card_view_section(
            sections,
            section_id="evidence",
            label=str(out.get("card_evidence_label") or _card_label(locale, "evidence")),
            text=str(out.get("card_evidence") or ""),
            kind="quote",
        )
        _append_card_view_section(
            sections,
            section_id="locator",
            label=str(out.get("card_locator_label") or _card_label(locale, "source_location")),
            text=str(out.get("card_locator") or ""),
            kind="locator",
        )
    else:
        _append_card_view_section(
            sections,
            section_id="locator",
            label=str(out.get("card_locator_label") or _card_label(locale, "current_citation_location")),
            text=str(out.get("card_locator") or ""),
            kind="locator",
        )
        _append_card_view_section(
            sections,
            section_id="context_summary",
            label=_card_label(locale, "context_summary"),
            text=str(out.get("card_context_summary") or ""),
            kind="summary",
        )
        _append_card_view_section(
            sections,
            section_id="evidence",
            label=str(out.get("card_evidence_label") or _card_label(locale, "citation_context")),
            text=str(out.get("card_evidence") or ""),
            kind="quote",
        )
        if "missing_reference_title" in flags or "reference_entry_only" in flags or not title:
            _append_card_view_section(
                sections,
                section_id="reference",
                label=str(out.get("card_reference_label") or _card_label(locale, "reference_entry")),
                text=str(out.get("card_reference_entry") or ""),
                kind="reference",
            )
    show_support = is_system_b or bool(
        _clean_text(out.get("card_support_explanation"), max_len=420)
    )
    if show_support:
        _append_card_view_section(
            sections,
            section_id="support",
            label=str(out.get("card_support_label") or _card_label(locale, "support_system_b" if is_system_b else "support_system_a")),
            text=str(out.get("card_support_explanation") or ""),
            kind="support",
        )

    summary = ""
    for preferred in ("takeaway", "context_summary", "evidence", "reference"):
        match = next((item for item in sections if item.get("id") == preferred), None)
        if match and str(match.get("text") or "").strip():
            summary = _clean_text(match.get("text"), max_len=260)
            break

    return {
        "version": CITATION_CARD_VIEW_CONTRACT_VERSION,
        "route": route,
        "kind": _clean_text(out.get("card_kind"), max_len=80),
        "header": {
            "kicker": _card_label(locale, "header_system_b" if is_system_b else "header_system_a"),
            "title": title,
            "subtitle": subtitle,
        },
        "sections": sections,
        "summary": summary,
        "quality": {
            "label": _clean_text(out.get("card_quality_label"), max_len=80),
            "score": _safe_float(out.get("card_quality_score"), 0.0),
            "flags": flags,
            "warning": _clean_text(out.get("card_warning"), max_len=360),
        },
    }


def _finalize_card_output(card: dict[str, Any], *, route: str, locale: str = "") -> dict[str, Any]:
    out = dict(card)
    preserve_evidence_boundary = bool(
        out.pop("_preserve_card_evidence_boundary", False)
    )
    flags = _clean_quality_flags(out.get("card_quality_flags"))
    for key, limit in _CARD_TEXT_LIMITS.items():
        before = str(out.get(key) or "")
        if key == "card_evidence" and preserve_evidence_boundary:
            out[key] = clean_display_text(out.get(key), max_len=limit)
        elif key == "card_evidence":
            out[key] = finish_evidence_text(out.get(key), max_len=limit)
        else:
            out[key] = _clean_text(out.get(key), max_len=limit)
        if _is_placeholder_card_text(str(out.get(key) or "")):
            out[key] = ""
            _add_flag(flags, f"{key}_placeholder_removed")
        if before and _text_has_visible_markup_artifact(before) and not _text_has_visible_markup_artifact(str(out.get(key) or "")):
            _add_flag(flags, f"{key}_markup_cleaned")

    evidence = str(out.get("card_evidence") or "").strip()
    claim = str(out.get("card_claim") or "").strip()
    takeaway = str(out.get("card_takeaway") or "").strip()
    context_summary = str(out.get("card_context_summary") or "").strip()
    support = str(out.get("card_support_explanation") or "").strip()
    answer_context_only = route == "system_b" and "answer_context_only" in set(flags)

    if answer_context_only and evidence:
        # The answer sentence is useful for tracing, but it is not original paper evidence.
        out["card_evidence"] = ""
        evidence = ""
        _add_flag(flags, "answer_context_hidden_from_card")

    if takeaway and (
        _sameish(takeaway, evidence)
        or _sameish(takeaway, claim)
        or _looks_low_value_takeaway(takeaway)
    ):
        out["card_takeaway"] = ""
        takeaway = ""

    if route == "system_b":
        if context_summary and (
            _sameish(context_summary, evidence)
            or _sameish(context_summary, takeaway)
            or _sameish(context_summary, claim)
            or _looks_generic_system_b_text(context_summary)
        ):
            out["card_context_summary"] = ""
            context_summary = ""
        if claim and evidence and _sameish(claim, evidence):
            out["card_claim"] = ""
            claim = ""
        if support and (
            _sameish(support, takeaway)
            or _sameish(support, evidence)
            or _sameish(support, claim)
            or _looks_generic_system_b_text(support)
        ):
            out["card_support_explanation"] = ""
        if claim and answer_context_only:
            out["card_claim"] = ""
            claim = ""
    elif support and (
        _sameish(support, evidence)
        or _sameish(support, claim)
        or _sameish(support, takeaway)
    ):
        out["card_support_explanation"] = ""

    if not str(out.get("card_evidence") or "").strip():
        out["card_evidence_label"] = _clean_text(out.get("card_evidence_label"), max_len=80)
    out["card_quality_flags"] = _dedup_strings(flags)
    out["render_locale"] = _card_locale(locale)
    out["card_display_contract_version"] = CITATION_CARD_DISPLAY_CONTRACT_VERSION
    out["card_visible_sections"] = _card_visible_sections(out, route=route)
    out["card_view"] = _build_card_view(out, route=route, locale=locale)
    return out


def _clean_reference_entry(value: Any, *, max_len: int = 900) -> str:
    text = _clean_text(value, max_len=max_len)
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _looks_reference_author_segment(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    comma_count = text.count(",")
    amp_or_and = bool(re.search(r"\b(?:and|et al)\b|&", text, re.IGNORECASE))
    initials = len(re.findall(r"\b[A-Z]\.?\b", text))
    surnames = len(re.findall(r"\b[A-Z][A-Za-z'`-]{2,}\b", text))
    if comma_count >= 1 and initials >= 2 and surnames >= 2:
        return True
    if comma_count >= 2 and (initials >= 2 or amp_or_and):
        return True
    return comma_count >= 3 and surnames >= 3


def _looks_reference_venue_segment(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lower = text.lower()
    if re.search(r"\b(?:18|19|20)\d{2}\b", lower):
        return True
    if re.search(r"\b\d{1,4}\s*,\s*\d{1,6}(?:[-–]\d{1,6})?\b", lower):
        return True
    venue_tokens = (
        "journal",
        "transactions",
        "proceedings",
        "conference",
        "letters",
        "express",
        "optics",
        "photonics",
        "physical review",
        "phys. rev",
        "ieee",
        "acm",
        "springer",
        "elsevier",
        "nature",
        "science",
        "arxiv",
    )
    return len(lower.split()) <= 12 and any(token in lower for token in venue_tokens)


def _looks_reference_title_segment(value: str) -> bool:
    text = str(value or "").strip(" .;:,")
    if not text:
        return False
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'`-]*|[\u4e00-\u9fff]+", text)
    if len(text) < 8 or len(text) > 260:
        return False
    if len(words) < 3 or len(words) > 32:
        return False
    if re.search(r"\b(?:doi|arxiv)\b", text, re.IGNORECASE):
        return False
    if _looks_reference_author_segment(text) or _looks_reference_venue_segment(text):
        return False
    if len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", text)) < 2:
        return False
    return True


def _fallback_system_b_title_from_raw_reference(raw: str) -> str:
    text = _clean_reference_entry(raw, max_len=900)
    if not text:
        return ""
    has_reference_shape = bool(
        re.search(r"^\s*(?:\[\s*\d{1,4}\s*\]|\d{1,4}\s*[.)])\s+", text)
        or re.search(r"\b(?:18|19|20)\d{2}\b", text)
        or re.search(r"\bdoi\s*:?\s*10\.", text, re.IGNORECASE)
        or text.count(",") >= 2
    )
    if not has_reference_shape:
        return ""
    text = re.sub(r"^\s*(?:\[\s*\d{1,4}\s*\]|\d{1,4}\s*[.)])\s*", "", text)
    text = re.sub(r"https?://\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdoi\s*:?\s*10\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\barxiv\s*:?\s*\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .;:,")
    if not text:
        return ""

    quoted = re.search(r"[\"“”]([^\"“”]{8,260})[\"“”]", text)
    if quoted:
        title = str(quoted.group(1) or "").strip(" .;:,")
        if _looks_reference_title_segment(title):
            return _clean_text(title, max_len=220)

    year_match = re.search(r"\((?:18|19|20)\d{2}\)\s*([^.]{8,260})\.", text)
    if year_match:
        title = str(year_match.group(1) or "").strip(" .;:,")
        if _looks_reference_title_segment(title):
            return _clean_text(title, max_len=220)

    segments = [
        part.strip(" .;:,")
        for part in re.split(r"\.\s+(?=[A-Z][A-Za-z0-9])", text)
        if part.strip(" .;:,")
    ]
    if not segments:
        return ""

    for idx, segment in enumerate(segments):
        if idx == 0 and _looks_reference_author_segment(segment):
            continue
        if _looks_reference_title_segment(segment):
            return _clean_text(segment, max_len=220)
    return ""


def _source_name(source_path: str) -> str:
    text = str(source_path or "").strip()
    if not text:
        return ""
    name = Path(text).name or text
    low = name.lower()
    if low.endswith(".en.md"):
        return name[:-6] + ".pdf"
    if low.endswith(".md"):
        return name[:-3] + ".pdf"
    return name


def _identity_label_candidates(*values: str) -> list[str]:
    out: list[str] = []
    for value in values:
        raw = _clean_text(value, max_len=260)
        if not raw:
            continue
        base = Path(raw.replace("\\", "/")).name or raw
        for candidate in (raw, base, source_title_candidate(base), source_title_candidate(raw)):
            text = _clean_text(candidate, max_len=260).strip(" /·")
            if not text:
                continue
            key = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text.lower()).strip()
            if len(key) < 4:
                continue
            if key not in {re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", item.lower()).strip() for item in out}:
                out.append(text)
    return out


def _same_location_identity(left: str, right: str) -> bool:
    a = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", _clean_text(left, max_len=260).lower()).strip()
    b = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", _clean_text(right, max_len=260).lower()).strip()
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 16 and len(b) >= 16 and (a in b or b in a):
        return True
    at = set(a.split())
    bt = set(b.split())
    if len(at) < 3 or len(bt) < 3:
        return False
    return len(at & bt) / max(1, min(len(at), len(bt))) >= 0.82


def _strip_redundant_locator_prefix(locator: str, *, source: str = "", title: str = "") -> str:
    text = _clean_text(locator, max_len=260).strip(" /·")
    if not text:
        return ""
    identities = _identity_label_candidates(source, title)
    if not identities:
        return text

    parts = [part.strip() for part in re.split(r"\s*/\s*", text) if part.strip()]
    changed = False
    while len(parts) > 1 and any(_same_location_identity(parts[0], item) for item in identities):
        parts = parts[1:]
        changed = True
    if changed:
        return " / ".join(parts).strip()

    if any(_same_location_identity(text, item) for item in identities):
        return ""
    for item in sorted(identities, key=len, reverse=True):
        if len(item) < 10:
            continue
        pattern = re.compile(rf"^\s*{re.escape(item)}\s*(?:/|·|-|—|:|：)\s*", re.IGNORECASE)
        stripped = pattern.sub("", text).strip(" /·")
        if stripped != text:
            return stripped
    return text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _page_label(start: Any, end: Any) -> str:
    p0 = _safe_int(start)
    p1 = _safe_int(end)
    if p0 <= 0:
        return ""
    if p1 <= 0 or p1 == p0:
        return f"p. {p0}"
    return f"pp. {min(p0, p1)}-{max(p0, p1)}"


def _anchor_kind_label(value: str) -> str:
    key = str(value or "").strip().lower()
    return {
        "sentence": "句子",
        "paragraph": "段落",
        "equation": "公式",
        "figure": "图",
        "table": "表",
    }.get(key, str(value or "").strip())


def _sameish(left: str, right: str) -> bool:
    a = re.sub(r"\s+", " ", str(left or "")).strip().lower()
    b = re.sub(r"\s+", " ", str(right or "")).strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 36 and a in b:
        return True
    if len(b) >= 36 and b in a:
        return True
    at = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", a))
    bt = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", b))
    if len(at) < 5 or len(bt) < 5:
        return False
    return len(at & bt) / max(1, min(len(at), len(bt))) >= 0.82


def _has_cjk(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(value or "")))


def _looks_low_value_takeaway(value: str) -> bool:
    text = _clean_text(value, max_len=360)
    if not text:
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z\s-]{2,48}\s+\d{1,3}", text):
        return True
    if re.search(r"(?:这条证据|该证据|this evidence|the evidence).{0,12}(?:支持|支撑|supports?)", text, re.IGNORECASE):
        return True
    if (
        re.search(
            r"(?:泛化能力|泛化性).{0,24}(?:受限|有限).{0,24}(?:优异|优秀|良好)",
            text,
        )
        or re.search(
            r"(?:泛化能力|泛化性).{0,24}(?:优异|优秀|良好).{0,24}(?:受限|有限)",
            text,
        )
    ):
        return True
    tokens = _loose_tokens(text)
    if _has_cjk(text):
        return len(text) < 12 and not re.search(r"[：:，,。；;]", text)
    return len(tokens) <= 6


_SYSTEM_A_SUPPORT_TEMPLATE_RE = re.compile(
    r"(?i)(?:"
    r"^this\s+citation\s+(?:reuses|uses|is\s+only)|"
    r"^this\s+(?:answer\s+sentence|claim)\s+is\s+supported\s+by|"
    r"^the\s+(?:same\s+)?source\s+(?:passage\s+)?(?:directly\s+)?(?:contains|reports|provides)|"
    r"^the\s+answer\s+and\s+source\s+align|"
    r"^the\s+answer\s+(?:sentence\s+)?and\s+(?:the\s+)?source\s+both\s+(?:say|state|show)|"
    r"^open\s+the\s+source\s+to\s+confirm|"
    r"^\u8be5?\u5f15\u7528\u590d\u7528\u751f\u6210\u56de\u7b54\u65f6|"
    r"^\u8be5?\u5f15\u7528\u4f7f\u7528\u4e86\u5df2\u6838\u5bf9|"
    r"^\u8fd9\u6761\u5f15\u7528\u53ea\u80fd\u4f5c\u4e3a\u5019\u9009|"
    r"^\u539f\u6587\u5728\u8be5\u5b9a\u4f4d\u5904\u7ed9\u51fa\u7684\u5177\u4f53\u9648\u8ff0|"
    r"^\u7b54\u6848\u4e0e\u82f1\u6587\u539f\u6587\u5728.*\u591a\u4e2a\u5177\u4f53\u52a8\u4f5c|"
    r"^\u7b54\u6848\u53e5\u548c\u539f\u6587\u90fd\u8bf4\u660e|"
    r"^\u8bf7\u6253\u5f00\u539f\u6587\u6838\u5bf9"
    r")"
)


def _meaningful_system_a_support(value: Any) -> str:
    """Return user-facing relevance copy, never an internal binding template."""

    text = _clean_text(value, max_len=420)
    if not text or _SYSTEM_A_SUPPORT_TEMPLATE_RE.search(text):
        return ""
    tokens = _loose_tokens(text)
    if _has_cjk(text):
        return text if len(re.sub(r"\s+", "", text)) >= 18 else ""
    return text if len(tokens) >= 7 and len(text) >= 40 else ""


def _system_a_support_hint(rec: Mapping[str, Any]) -> str:
    # ``binding_reason`` is an internal verifier explanation.  Even when it is
    # accurate, its stock phrasing is not relevance copy for an end user.
    for key in ("support_relation", "why_line"):
        text = _meaningful_system_a_support(rec.get(key))
        if text:
            return text
    return ""


def _trim_takeaway(value: str, *, max_len: int = 96) -> str:
    text = _clean_text(value, max_len=max_len + 20)
    text = re.sub(r"^\s*(?:这条证据说明|证据说明|它说明|说明)[:：]\s*", "", text)
    text = text.strip(" \t\r\n。；;")
    if len(text) > max_len:
        text = text[: max(0, max_len - 1)].rstrip(" ，,；;:：") + "..."
    if text and _has_cjk(text) and not text.endswith(("。", "！", "？", "...")):
        text += "。"
    return text


def _takeaway_from_english_evidence(evidence: str) -> str:
    text = str(evidence or "")
    low = text.lower()
    if (
        "single-pixel camera" in low
        and "number of measurements is fewer" in low
        and ("unknown pixels" in low or "under-sampling" in low or "sub-sampling" in low)
    ):
        return "压缩感知让单像素相机能在测量次数少于图像未知像素总数时，通过欠采样恢复图像。"
    if "dmd" in low and ("spatially filter" in low or "single-pixel camera configuration" in low):
        return "DMD 可以作为单像素相机中的空间调制器，通过选择性重定向光束来完成采样和成像配置。"
    if "single-pixel imaging technology can capture images at wavelengths outside" in low:
        return "单像素成像可以覆盖传统焦平面阵列探测器难以触达的波段，但实用性仍受图像质量和计算时间限制。"
    if "structured detection" in low and "optical sectioning" in low:
        return "结构化检测用于在激光扫描显微中同时改善层切、分辨率和信噪比。"
    if (
        "model-driven strategy" in low
        and "physical process of spi" in low
        and "neural network" in low
        and "discrepancy between real and estimated measurements" in low
    ):
        return (
            "模型驱动策略将 SPI 物理过程嵌入神经网络，并以真实与估计测量之间的差异约束优化；"
            "原文将其描述为具有较强泛化能力的无监督模式。"
        )
    if "deep learning" in low and "single-pixel" in low and re.search(r"\b(?:quality|speed|reconstruction)\b", low):
        return "深度学习方法主要用于提升单像素成像的重建质量、速度或采样效率。"
    if "snapshot compressive imaging" in low and ("recover" in low or "reconstruct" in low):
        return "快照压缩成像通过一次压缩观测恢复场景信息，是该回答所说成像任务的直接背景。"
    return ""


def _system_a_takeaway(*, claim: str, evidence: str, heading: str, locale: str = "") -> str:
    claim_clean = _trim_takeaway(claim, max_len=110)
    evidence_takeaway = _trim_takeaway(_takeaway_from_english_evidence(evidence), max_len=110)
    claim_is_usable = bool(
        _card_locale(locale) != "en"
        and claim_clean
        and _has_cjk(claim_clean)
        and not _looks_low_value_takeaway(claim_clean)
    )
    if claim_is_usable and (len(claim_clean) >= 24 or not evidence_takeaway):
        return claim_clean
    if evidence_takeaway and not _looks_low_value_takeaway(evidence_takeaway):
        return evidence_takeaway
    return claim_clean if claim_is_usable else ""


def _looks_generic_system_b_text(value: str) -> bool:
    text = _clean_text(value, max_len=360).lower()
    if not text:
        return True
    generic_patterns = [
        r"这条链接把回答中的说法追溯到",
        r"这条参考是当前论文给出的上游来源",
        r"这篇上游文献条目",
        r"the user is asking about the evidence",
        r"upstream paper to open next",
        r"cited prior work or background source",
        r"trace the upstream origin",
        r"this reference is the cited prior work",
    ]
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in generic_patterns):
        return True
    tokens = _loose_tokens(text)
    if re.search(r"[\u4e00-\u9fff]", text):
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
        return len(cjk_chars) <= 10 and len(tokens) <= 3
    return len(tokens) <= 5


def _system_b_explicit_takeaway(*, role: str, relation: str, locale: str = "") -> str:
    for value in (role, relation):
        text = _trim_takeaway(value, max_len=118)
        if not text or _looks_generic_system_b_text(text):
            continue
        if _card_locale(locale) == "en":
            if _has_cjk(text):
                continue
            return _trim_takeaway(text, max_len=118)
        if not _has_cjk(text):
            continue
        text = re.sub(r"^用户问[“\"].+?[”\"，,；;]\s*", "", text)
        text = re.sub(r"^这条参考(?:正好)?说明", "这篇上游文献说明", text)
        text = re.sub(r"^它说明", "这篇上游文献说明", text)
        return _trim_takeaway(text, max_len=118)
    return ""


def _system_b_takeaway(*, title: str, claim: str, context: str, role: str, relation: str, locale: str = "") -> str:
    explicit = _system_b_explicit_takeaway(role=role, relation=relation, locale=locale)
    if explicit:
        return explicit

    combined = " ".join(str(part or "") for part in (title, claim, context, role, relation)).lower()
    prefer_en = _card_locale(locale) == "en"
    if "admm-net" in combined or "unfold" in combined or "unrolled" in combined:
        return "This upstream work links iterative optimization ideas to trainable network designs." if prefer_en else "这篇上游文献提供把迭代优化思想展开成可训练网络的前人线索。"
    if "admm" in combined or "alternating direction method" in combined:
        return "This upstream work provides ADMM optimization background for checking how the current paper builds on prior methods." if prefer_en else "这篇上游文献提供 ADMM 优化框架背景，用来判断当前论文是在借鉴既有方法。"
    if "single-shot compressive spectral imaging" in combined:
        return "This upstream work provides background for single-shot compressive spectral imaging." if prefer_en else "这篇上游文献提供单次压缩光谱成像的前人背景，是回答中相关概念的来源线索。"
    if "snapshot compressive imaging" in combined or re.search(
        r"\bvideo\s+SCI\b", combined, flags=re.I
    ):
        return (
            "The current paper cites this upstream review when introducing video snapshot compressive imaging; it is a direct entry point to that route's theory, algorithms, and applications."
            if prefer_en
            else "当前论文在引出视频快照压缩成像路线时引用这篇上游综述；它可作为继续核对该路线理论、算法与应用的直接入口。"
        )
    if "single-pixel imaging via compressive sampling" in combined or (
        "single-pixel" in combined and "compressive sampling" in combined
    ):
        return "This upstream work is a classic source for the compressive-sampling route in single-pixel imaging." if prefer_en else "这篇上游文献是单像素压缩采样路线的经典来源，适合用来补上“单个探测器如何靠调制与重建成像”的基础背景。"
    if re.search(r"\b(?:baseline|compare|compared|comparison|against)\b", combined):
        return "This upstream work is mainly used as a comparison baseline or related-method reference." if prefer_en else "这篇上游文献在当前论文中主要作为对比基线或相关方法参照。"
    if re.search(r"\b(?:dataset|benchmark|evaluation|experiment)\b", combined):
        return "This upstream work provides dataset, evaluation, or benchmark context." if prefer_en else "这篇上游文献提供实验数据、评测场景或 benchmark 线索。"
    if re.search(r"\b(?:architecture|network|model|module)\b", combined):
        return "This upstream work provides prior context for model architecture or method design." if prefer_en else "这篇上游文献提供模型结构或方法设计上的前人参考。"
    if re.search(r"\b(?:background|prior work|related work|origin|source)\b", combined):
        return "This upstream work provides related-work background and source context for the claim." if prefer_en else "这篇上游文献提供当前说法的相关工作背景和来源线索。"
    return ""


def _quality_label(score: float, *, route: str, locale: str = "") -> str:
    if route == "system_b":
        if score >= 0.78:
            return _card_label(locale, "quality_system_b_high")
        if score >= 0.58:
            return _card_label(locale, "quality_system_b_mid")
        return _card_label(locale, "quality_system_b_low")
    if score >= 0.78:
        return _card_label(locale, "quality_system_a_high")
    if score >= 0.52:
        return _card_label(locale, "quality_system_a_mid")
    return _card_label(locale, "quality_system_a_low")


def _dedup_strings(values: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value, max_len=120)
        if not text:
            continue
        key = re.sub(r"\s+", " ", text).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _is_paper_only_locator(locator: str, source: str) -> bool:
    loc = _clean_text(locator, max_len=260).lower()
    src = _clean_text(source, max_len=260).lower()
    if not loc:
        return True
    if src and loc == src:
        return True
    return loc in {"unknown location", "not located", "source paper", "current paper"}


def _compose_system_b_trace(
    *,
    rec: Mapping[str, Any],
    pack_flags: list[str],
    claim: str,
    context: str,
    source: str,
    locator: str,
    title: str,
    raw_reference: str,
    reference_entry: str,
    score: float,
    locale: str = "",
) -> dict[str, Any]:
    reference = reference_entry or raw_reference or title
    context_source = str(rec.get("citation_context_source") or rec.get("evidence_source") or "").strip().lower()
    routing_reason = _first_text(rec, "routing_reason", max_len=160) or "structured_cite"
    routing_confidence = _safe_float(rec.get("routing_confidence"), 0.0)

    flags: list[str] = []
    if not claim:
        flags.append("missing_answer_claim")
    if not context:
        flags.append("missing_citation_context")
    if context_source in {"answer_context", "answer_reference_mention"}:
        flags.append("answer_context_only")
    if not reference:
        flags.append("missing_reference_entry")
    if not locator:
        flags.append("missing_citing_location")
    elif _is_paper_only_locator(locator, source):
        flags.append("paper_only_citing_location")
    if context and reference and _sameish(context, reference):
        flags.append("context_is_reference_entry")
    if routing_confidence and routing_confidence < 0.5:
        flags.append("low_routing_confidence")
    flags.extend(str(item or "").strip() for item in pack_flags if str(item or "").strip())
    flags = _dedup_strings(flags)

    hard_flags = {
        "missing_answer_claim",
        "missing_citation_context",
        "answer_context_only",
        "missing_reference_entry",
        "missing_citing_location",
        "context_is_reference_entry",
        "low_routing_confidence",
        "weak_citation_context",
        "reference_entry_only",
    }
    trace_complete = not any(flag in hard_flags for flag in flags)
    trace_score = max(0.0, min(1.0, score))
    for flag in flags:
        if flag in {"missing_citation_context", "missing_reference_entry"}:
            trace_score -= 0.22
        elif flag in {"answer_context_only", "context_is_reference_entry"}:
            trace_score -= 0.18
        elif flag in {"missing_answer_claim", "missing_citing_location", "low_routing_confidence"}:
            trace_score -= 0.14
        elif flag == "paper_only_citing_location":
            trace_score -= 0.06
        elif flag in {"weak_citation_context", "reference_entry_only"}:
            trace_score -= 0.1
    trace_score = max(0.0, min(1.0, trace_score))

    if trace_complete:
        reason = _card_label(locale, "trace_complete")
    elif "answer_context_only" in flags:
        reason = _card_label(locale, "trace_answer_context_only")
    elif "missing_citation_context" in flags:
        reason = _card_label(locale, "trace_missing_context")
    elif "context_is_reference_entry" in flags:
        reason = _card_label(locale, "trace_reference_entry")
    else:
        reason = _card_label(locale, "trace_review")

    steps = (
        [_card_label(locale, "trace_step_answer"), _card_label(locale, "trace_step_context"), _card_label(locale, "trace_step_reference")]
        if trace_complete
        else [_card_label(locale, "trace_step_answer"), _card_label(locale, "trace_step_review"), _card_label(locale, "trace_step_reference")]
    )
    return {
        "system_b_trace_complete": trace_complete,
        "system_b_trace_score": round(trace_score, 3),
        "system_b_trace_reason": reason,
        "system_b_trace_flags": flags,
        "system_b_trace_steps": steps,
        "system_b_trace_answer": claim,
        "system_b_trace_context": context,
        "system_b_trace_reference": reference,
        "system_b_trace_locator": locator,
        "system_b_trace_source": context_source or routing_reason,
    }


def _locator(rec: Mapping[str, Any]) -> str:
    loc = _first_text(rec, "location_label", max_len=260)
    if not loc:
        heading = _first_text(rec, "heading_path", max_len=180)
        page = _page_label(rec.get("page_start"), rec.get("page_end"))
        kind = _anchor_kind_label(str(rec.get("anchor_kind") or ""))
        loc = " · ".join(part for part in (heading, page, kind) if part)

    evidence = _first_raw_value(
        rec,
        "evidence_quote",
        "reader_evidence_quote",
        "citation_plan_reader_evidence_quote",
        "summary_line",
        "raw",
    )
    equation_numbers = list(
        dict.fromkeys(
            match.group(1)
            for match in re.finditer(
                r"\\tag\s*\{\s*(\d{1,4})\s*\}",
                str(evidence or ""),
            )
        )
    )[:4]
    equation_labels = [
        f"Equation ({number})"
        for number in equation_numbers
        if f"Equation ({number})".casefold() not in loc.casefold()
    ]
    if equation_labels:
        # Equation tags are visible source locators, not generated metadata.
        # Keep them with the page/section so a formula card can be audited down
        # to the exact numbered expressions it displays.
        loc = " · ".join([loc, *equation_labels]) if loc else " · ".join(equation_labels)
    return loc


def _compose_system_a(rec: dict[str, Any], *, locale: str = "") -> dict[str, Any]:
    source = _first_text(rec, "source_name", max_len=180) or _source_name(str(rec.get("source_path") or ""))
    heading = _first_text(rec, "heading_path", "title", max_len=180)
    title = source or heading or _card_label(locale, "title_system_a")
    claim_raw = _first_text(rec, "answer_claim", max_len=420)
    evidence_raw = _first_text(
        rec,
        "evidence_quote",
        "reader_evidence_quote",
        "citation_plan_reader_evidence_quote",
        "summary_line",
        "raw",
        "cite_fmt",
        max_len=1400,
    )
    evidence_raw_for_pack = _first_raw_value(
        rec,
        "evidence_quote",
        "reader_evidence_quote",
        "citation_plan_reader_evidence_quote",
        "summary_line",
        "raw",
        "cite_fmt",
    ) or evidence_raw
    locator = _strip_redundant_locator_prefix(_locator(rec), source=source, title=title)
    if not locator and source:
        locator = f"Document-level match: {source}"
    support_hint = _system_a_support_hint(rec)
    pack = build_system_a_evidence_pack(
        answer_claim=claim_raw,
        evidence_raw=evidence_raw_for_pack,
        source=source,
        title=_first_text(rec, "title", max_len=240),
        heading=heading,
        location_label=locator,
        support_hint=support_hint,
        locale=locale,
    )
    claim = pack.answer_claim
    evidence = pack.evidence_quote
    exact_support_locked = bool(
        (
            str(rec.get("routing_reason") or "").strip().lower()
            == "exact_support_preflight"
            or str(rec.get("evidence_source") or "").strip().lower()
            == "exact_support_preflight"
            or str(
                rec.get("selection_reason")
                or rec.get("selectionReason")
                or ""
            ).strip().lower()
            in {
                "exact_support_preflight",
                "prompt_contract_block",
                "spad_noise_model_exact_source",
                "lineage_exact_source_block",
            }
        )
        and bool(rec.get("strict_locate") or rec.get("strictLocate"))
        and int(rec.get("page_start") or rec.get("pageStart") or 0) > 0
    )
    structured_metric_evidence_locked = bool(
        bool(rec.get("strict_locate") or rec.get("strictLocate"))
        and int(rec.get("page_start") or rec.get("pageStart") or 0) > 0
        and re.search(
            r"(?i)\b(?:PSNR|SSIM|LPIPS|FID|FPS)\b",
            str(evidence_raw_for_pack or ""),
        )
        and len(
            re.findall(
                r"(?i)(?:^|[;,:])\s*[A-Za-z][A-Za-z0-9 +()_-]{0,48}\s*=\s*-?\d+(?:\.\d+)?",
                str(evidence_raw_for_pack or ""),
            )
        )
        >= 2
    )
    compound_plan_evidence_locked = bool(
        rec.get("compound_plan_evidence")
        and str(rec.get("evidence_source") or "").strip().lower()
        != "answer_context_only"
    )
    exact_evidence = (
        clean_display_text(evidence_raw_for_pack, max_len=900)
        if exact_support_locked or structured_metric_evidence_locked or compound_plan_evidence_locked
        else ""
    )
    if exact_evidence:
        # This text was selected from a verified page occurrence before
        # general retrieval.  The generic readability filter intentionally
        # rejects some long/list-like excerpts; that is inappropriate for an
        # authoritative exact-support or prompt-contract passage because it
        # can drop the final clause that made the passage satisfy the prompt.
        evidence = exact_evidence
    takeaway = _system_a_takeaway(claim=claim, evidence=evidence, heading=heading, locale=locale)
    if not takeaway:
        takeaway = pack.evidence_focus
    if takeaway and (_sameish(takeaway, evidence) or _sameish(takeaway, claim)):
        takeaway = ""
    subtitle = locator or _strip_redundant_locator_prefix(heading, source=source, title=title)
    binding_status = str(rec.get("binding_status") or "").strip().lower()
    binding_confidence = _safe_float(rec.get("binding_confidence"), 0.0)
    support = pack.support_explanation or support_hint

    ranked_score = min(0.76, max(0.42, _safe_float(rec.get("score"), 0.0) / 10.0))
    score = max(binding_confidence, ranked_score) if binding_confidence else ranked_score
    score += pack.score_delta
    flags: list[str] = [
        flag
        for flag in pack.flags
        if not (
            exact_evidence
            and flag in {"evidence_quote_filtered", "missing_evidence_quote"}
        )
    ]
    if _text_has_visible_markup_artifact(str(evidence_raw_for_pack or "")):
        flags.append("card_evidence_markup_cleaned")
    if not claim:
        flags.append("missing_answer_claim")
        score -= 0.08
    if not evidence:
        flags.append("missing_evidence_quote")
        score -= 0.16
    if not locator:
        flags.append("missing_precise_location")
        score -= 0.08
    if binding_status == "mismatch":
        flags.append("binding_mismatch")
        score = min(score, 0.25)
    elif binding_status == "candidate":
        flags.append("candidate_binding")
        score = min(score, 0.58)
    if claim and evidence and _sameish(claim, evidence):
        flags.append("claim_duplicates_evidence")
    if bool(rec.get("occurrence_specific")):
        flags.append("occurrence_specific_claim")
    score = max(0.0, min(1.0, score))

    needs_review = bool(binding_status in {"candidate", "mismatch"} or score < 0.55)
    support_label = ""
    support_text = ""
    if needs_review:
        support_label = _card_label(locale, "support_label_review")
        support_text = support or _card_label(locale, "candidate_support")
    elif (
        binding_status == "grounded"
        and binding_confidence >= 0.7
        and support
        and not _sameish(support, claim)
        and not _sameish(support, evidence)
        and not _sameish(support, takeaway)
    ):
        support_label = _card_label(locale, "support_relevance_system_a")
        support_text = support
    warning = ""
    if "binding_mismatch" in flags:
        warning = _card_label(locale, "warning_mismatch")
    elif "candidate_binding" in flags or score < 0.55:
        warning = _card_label(locale, "warning_candidate")

    return _finalize_card_output({
        "card_kind": "answer_evidence",
        "card_title": title,
        "card_subtitle": subtitle,
        "answer_claim": claim,
        "evidence_quote": evidence,
        "summary_line": evidence,
        "card_takeaway_label": _card_label(locale, "takeaway_system_a"),
        "card_takeaway": takeaway,
        "card_claim_label": _card_label(locale, "answer_point"),
        "card_claim": claim,
        "card_locator_label": pack.location_label_name or _card_label(locale, "source_location"),
        "card_locator": pack.location_label or locator,
        "card_evidence_label": pack.evidence_label or _card_label(locale, "evidence"),
        "card_evidence": evidence,
        "card_support_label": support_label,
        "card_support_explanation": support_text,
        "card_quality_label": _quality_label(score, route="system_a", locale=locale),
        "card_quality_score": round(score, 3),
        "card_quality_flags": flags,
        "card_warning": warning,
        "card_flow": [],
        "_preserve_card_evidence_boundary": bool(
            structured_metric_evidence_locked or compound_plan_evidence_locked
        ),
    }, route="system_a", locale=locale)


def _compose_system_b(rec: dict[str, Any], *, locale: str = "") -> dict[str, Any]:
    source = _first_text(rec, "source_name", max_len=180) or _source_name(str(rec.get("source_path") or ""))
    raw_reference = _clean_reference_entry(rec.get("raw") or rec.get("cite_fmt"), max_len=900)
    explicit_title = _first_text(rec, "title", max_len=220)
    parsed_title = _fallback_system_b_title_from_raw_reference(raw_reference)
    title = explicit_title or parsed_title or _card_label(locale, "title_system_b")
    subtitle = " · ".join(
        part
        for part in (
            _first_text(rec, "authors", max_len=160),
            _first_text(rec, "venue", max_len=80),
            _first_text(rec, "year", max_len=16),
        )
        if part
    )
    claim_raw = _first_text(rec, "answer_claim", max_len=420)
    context_raw = _first_text(rec, "citation_context", "evidence_quote", "summary_line", max_len=1400)
    reference_locator = _strip_redundant_locator_prefix(
        _locator(rec),
        source=source,
        title=source,
    ) or source
    context_locator = _first_text(rec, "citation_context_location_label", max_len=260)
    if not context_locator:
        context_heading = _first_text(rec, "citation_context_heading_path", max_len=180)
        context_page = _page_label(
            rec.get("citation_context_page_start"),
            rec.get("citation_context_page_end"),
        )
        context_locator = " / ".join(
            part for part in (context_heading, context_page) if part
        )
    locator = _strip_redundant_locator_prefix(
        context_locator or reference_locator,
        source=source,
        title=source,
    ) or source
    role = _first_text(rec, "upstream_work_role", "why_line", max_len=420)
    relation = _first_text(rec, "user_question_relation", "support_relation", max_len=420)
    pack = build_system_b_evidence_pack(
        answer_claim=claim_raw,
        citation_context_raw=context_raw,
        citation_context_source=str(rec.get("citation_context_source") or rec.get("evidence_source") or ""),
        source=source,
        title=title,
        heading=_first_text(
            rec,
            "citation_context_heading_path",
            "heading_path",
            "location_label",
            max_len=180,
        ),
        location_label=locator,
        raw_reference=raw_reference,
        role_hint=role,
        relation_hint=relation,
        locale=locale,
    )
    claim = pack.answer_claim
    context = pack.evidence_quote
    takeaway = _system_b_takeaway(title=title, claim=claim, context=context, role=role, relation=relation, locale=locale)
    support = pack.support_explanation

    score = 0.72 + pack.score_delta
    flags: list[str] = list(pack.flags)
    if _text_has_visible_markup_artifact(str(context_raw or "")):
        flags.append("card_evidence_markup_cleaned")
    if not explicit_title and not parsed_title:
        flags.append("missing_reference_title")
        score -= 0.16
    if not source:
        flags.append("missing_citing_source")
        score -= 0.12
    if not locator:
        flags.append("missing_citing_location")
        score -= 0.1
    if not takeaway:
        flags.append("missing_takeaway")
        score -= 0.08
    score = max(0.0, min(1.0, score))
    trace = _compose_system_b_trace(
        rec=rec,
        pack_flags=flags,
        claim=claim,
        context=context,
        source=source,
        locator=pack.location_label or locator,
        title=title,
        raw_reference=raw_reference,
        reference_entry=pack.reference_entry,
        score=score,
        locale=locale,
    )

    evidence_label = pack.evidence_label or _card_label(locale, "citation_context")
    warning = pack.warning
    if not warning and score < 0.58:
        warning = _card_label(locale, "warning_upstream_incomplete")

    summary_quality = rec.get("summary_quality")
    return _finalize_card_output({
        "card_kind": "upstream_reference",
        "card_title": title,
        "card_subtitle": subtitle,
        "card_takeaway_label": _card_label(locale, "takeaway_system_b"),
        "card_takeaway": takeaway,
        "card_claim_label": _card_label(locale, "answer_sentence"),
        "card_claim": claim,
        "card_locator_label": pack.location_label_name or _card_label(locale, "citation_location"),
        "card_locator": pack.location_label or locator,
        "card_evidence_label": evidence_label,
        "card_evidence": context,
        "card_context_summary": pack.evidence_focus,
        "card_reference_label": pack.reference_label,
        "card_reference_entry": pack.reference_entry,
        "card_support_label": "",
        "card_support_explanation": support,
        "card_quality_label": _quality_label(score, route="system_b", locale=locale),
        "card_quality_score": round(score, 3),
        "card_quality_flags": flags,
        "card_warning": warning,
        "card_flow": [],
        "summary_line": _first_text(rec, "summary_line", max_len=900),
        "summary_source": _first_text(rec, "summary_source", max_len=80),
        "summary_provider": _first_text(rec, "summary_provider", max_len=80),
        "summary_quality": dict(summary_quality) if isinstance(summary_quality, Mapping) else {},
        **trace,
    }, route="system_b", locale=locale)


def compose_citation_card(detail: Mapping[str, Any] | None, *, locale: str = "") -> dict[str, Any]:
    rec = dict(detail or {}) if isinstance(detail, Mapping) else {}
    if not rec:
        return {}
    render_locale = _card_locale(locale or rec.get("render_locale") or rec.get("locale"))
    rec["render_locale"] = render_locale
    explicit_route = _first_text(rec, "citation_route", "citationRoute", max_len=40).lower()
    is_system_b = (
        explicit_route == "system_b"
        or (
            explicit_route != "system_a"
            and bool(rec.get("is_inpaper") or rec.get("isInpaper"))
        )
    )
    card = _compose_system_b(rec, locale=render_locale) if is_system_b else _compose_system_a(rec, locale=render_locale)
    rec.update(card)
    return rec


def refresh_citation_card_contract(detail: Mapping[str, Any] | None, *, locale: str = "") -> dict[str, Any]:
    """Rebuild display contract fields after trusted text fields are patched."""
    rec = dict(detail or {}) if isinstance(detail, Mapping) else {}
    if not rec:
        return {}
    render_locale = _card_locale(locale or rec.get("render_locale") or rec.get("locale"))
    rec["render_locale"] = render_locale
    explicit_route = _first_text(rec, "citation_route", "citationRoute", max_len=40).lower()
    route = "system_b" if (
        explicit_route == "system_b"
        or (
            explicit_route != "system_a"
            and (
                bool(rec.get("is_inpaper") or rec.get("isInpaper"))
                or str(rec.get("card_kind") or "") == "upstream_reference"
            )
        )
    ) else "system_a"
    return _finalize_card_output(rec, route=route, locale=render_locale)
