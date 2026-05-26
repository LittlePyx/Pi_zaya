from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import replace
from functools import lru_cache
from typing import Any, Callable, Mapping

from kb.citation_card import compose_citation_card, refresh_citation_card_contract
from kb.citation_context_summary import reject_system_b_context_summary
from kb.config import load_settings
from kb.evidence_text import clean_display_text, source_title_candidate
from kb.llm import DeepSeekChat


_ALIAS_TO_SNAKE = {
    "isInpaper": "is_inpaper",
    "sourceName": "source_name",
    "sourcePath": "source_path",
    "citeFmt": "cite_fmt",
    "doiUrl": "doi_url",
    "linkedNums": "linked_nums",
    "summaryLine": "summary_line",
    "summarySource": "summary_source",
    "answerClaim": "answer_claim",
    "headingPath": "heading_path",
    "evidenceQuote": "evidence_quote",
    "evidenceSource": "evidence_source",
    "citationContext": "citation_context",
    "citationContextSource": "citation_context_source",
    "upstreamWorkRole": "upstream_work_role",
    "userQuestionRelation": "user_question_relation",
    "locationLabel": "location_label",
    "supportRelation": "support_relation",
    "whyLine": "why_line",
    "anchorKind": "anchor_kind",
    "pageStart": "page_start",
    "pageEnd": "page_end",
    "cardKind": "card_kind",
    "cardTitle": "card_title",
    "cardSubtitle": "card_subtitle",
    "cardTakeawayLabel": "card_takeaway_label",
    "cardTakeaway": "card_takeaway",
    "cardClaimLabel": "card_claim_label",
    "cardClaim": "card_claim",
    "cardLocatorLabel": "card_locator_label",
    "cardLocator": "card_locator",
    "cardEvidenceLabel": "card_evidence_label",
    "cardEvidence": "card_evidence",
    "cardContextSummary": "card_context_summary",
    "cardReferenceLabel": "card_reference_label",
    "cardReferenceEntry": "card_reference_entry",
    "cardSupportLabel": "card_support_label",
    "cardSupportExplanation": "card_support_explanation",
    "cardQualityScore": "card_quality_score",
    "cardQualityFlags": "card_quality_flags",
    "cardWarning": "card_warning",
}
_TEXT_PATCH_KEYS = ("card_takeaway", "card_claim", "card_context_summary", "card_support_explanation")
_VIEW_PATCH_VERSION = 1
_BAD_MARKUP_RE = re.compile(r"\[\[?\s*CITE\s*:|```|<[^>]+>|!\[[^\]]*]\(|\|")
_INLINE_REF_MARKER_RE = re.compile(r"\[(?:R)?\d{1,4}(?:\s*[-,;]\s*(?:R)?\d{1,4})*\]", re.IGNORECASE)
_PROMPT_LABEL_RE = re.compile(
    r"^(?:"
    r"card[_\s-]*(?:takeaway|claim|context|summary|support)|"
    r"证据重点|答案中的话|答案里的这句话|对应答案|原文证据|上游作用|引用语境|"
    r"语境摘要|引用语境摘要|为什么引用它|为什么值得打开|"
    r"引用出现位置|引用所在位置|当前位置|当前论文位置|当前论文引用处|上游文献条目|"
    r"为什么能支撑这句话|这条依据的可靠度"
    r")\s*[:：\-]\s*",
    re.IGNORECASE,
)
_GENERIC_RE = re.compile(
    r"\b(?:this reference is relevant|this evidence supports|good entry point|"
    r"directly relevant|upstream paper to open next|"
    r"no summary available|loading citation metrics|"
    r"这条(?:引用|证据|参考).{0,12}(?:相关|有用|值得打开|可以作为入口)|"
    r"该(?:文献|论文|证据).{0,12}(?:相关|有用|值得阅读|可以作为入口)|"
    r"可(?:以)?(?:作为|用于).{0,10}(?:候选读物|入口|定位切口))\b",
    re.IGNORECASE,
)
_LOW_INFORMATION_RE = re.compile(
    r"^(?:"
    r"abstract|introduction|methods?|related work|references?|"
    r"当前论文位置|引用出现位置|上游文献条目|对应答案|原文证据|引用语境|"
    r"打开(?:答案依据|引用语境)|start reading guide|open citation shelf|add to shelf"
    r")$",
    re.IGNORECASE,
)
_NARRATIVE_METADATA_RE = re.compile(
    r"\b(?:doi|jcr|impact\s*factor|if\s*[:：]?\s*\d|published\s+(?:in|by)|"
    r"journal|conference|venue|citation\s+count|cited\s+by)\b|"
    r"(?:发表于|发表在|期刊|会议|年份|被引|影响因子|分区|出处|来源论文|论文标题|标题是|作者是)",
    re.IGNORECASE,
)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s，。；;,)）]+", re.IGNORECASE)
_AUTHOR_LIST_RE = re.compile(
    r"(?:[A-Z][a-zA-Z'`-]+,\s*(?:[A-Z]\.?\s*){1,4}|[A-Z][a-zA-Z'`-]+\s+[A-Z]\.?)"
    r"(?:\s*(?:,|;|and|&)\s*(?:[A-Z][a-zA-Z'`-]+,\s*(?:[A-Z]\.?\s*){1,4}|[A-Z][a-zA-Z'`-]+\s+[A-Z]\.?)){2,}",
)


def normalize_citation_card_detail(detail: Mapping[str, Any] | None) -> dict[str, Any]:
    rec = dict(detail or {}) if isinstance(detail, Mapping) else {}
    for camel, snake in _ALIAS_TO_SNAKE.items():
        if camel in rec and snake not in rec:
            rec[snake] = rec.get(camel)
    if "is_inpaper" not in rec:
        rec["is_inpaper"] = bool(rec.get("isInpaper"))
    return rec


def citation_card_polish_enabled() -> bool:
    raw = str(os.environ.get("KB_CITATION_CARD_POLISH_USE_LLM", "") or "").strip().lower()
    if raw:
        return raw in {"1", "true", "on", "yes"}
    raw_ref = str(os.environ.get("KB_REFS_CARD_POLISH_USE_LLM", "1") or "1").strip().lower()
    return raw_ref not in {"0", "false", "off", "no"}


def _citation_card_polish_timeout_s() -> float:
    try:
        raw = float(str(os.environ.get("KB_CITATION_CARD_POLISH_TIMEOUT_S", "12") or "12"))
    except Exception:
        raw = 12.0
    return max(2.0, min(45.0, raw))


def _citation_card_polish_max_retries() -> int:
    try:
        raw = int(str(os.environ.get("KB_CITATION_CARD_POLISH_MAX_RETRIES", "1") or "1"))
    except Exception:
        raw = 1
    return max(0, min(2, raw))


def _text(value: Any, *, max_len: int = 900) -> str:
    return clean_display_text(value, max_len=max_len)


def _semantic_tokens(value: str) -> set[str]:
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if not text:
        return set()
    tokens = set(re.findall(r"[a-z0-9]{2,}", text))
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if len(cjk_chars) >= 2:
        tokens.update("".join(cjk_chars[idx : idx + 2]) for idx in range(len(cjk_chars) - 1))
    elif cjk_chars:
        tokens.update(cjk_chars)
    return {token for token in tokens if token}


def _sameish(left: str, right: str) -> bool:
    a = re.sub(r"\s+", " ", str(left or "")).strip().lower()
    b = re.sub(r"\s+", " ", str(right or "")).strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 32 and a in b:
        return True
    if len(b) >= 32 and b in a:
        return True
    at = _semantic_tokens(a)
    bt = _semantic_tokens(b)
    if len(at) < 5 or len(bt) < 5:
        return False
    return len(at & bt) / max(1, min(len(at), len(bt))) >= 0.82


def _compact_identity(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(value or "").lower()).strip()


def _contains_identity_text(text: str, candidate: str, *, min_len: int = 22) -> bool:
    body = _compact_identity(text)
    ident = _compact_identity(candidate)
    if not body or len(ident) < min_len:
        return False
    return ident in body


def _looks_redundant_metadata_text(text: str, baseline: Mapping[str, str]) -> bool:
    value = _clean_polish_candidate(text, max_len=320)
    if not value:
        return False
    if _DOI_RE.search(value) or _NARRATIVE_METADATA_RE.search(value):
        return True
    if _AUTHOR_LIST_RE.search(value):
        return True

    title = source_title_candidate(baseline.get("title", ""))
    source = source_title_candidate(baseline.get("source", ""))
    if _contains_identity_text(value, title) or _contains_identity_text(value, source):
        return True

    venue = str(baseline.get("venue") or "").strip()
    year = str(baseline.get("year") or "").strip()
    if venue and _contains_identity_text(value, venue, min_len=7):
        return True
    if year and re.search(r"\b(?:18|19|20)\d{2}\b", value) and _NARRATIVE_METADATA_RE.search(value):
        return True
    return False


def _clean_polish_candidate(value: Any, *, max_len: int) -> str:
    text = _text(value, max_len=max_len)
    for _ in range(2):
        cleaned = _PROMPT_LABEL_RE.sub("", text).strip()
        if cleaned == text:
            break
        text = cleaned
    return text


def _looks_bad_polish_text(value: str) -> bool:
    text = _clean_polish_candidate(value, max_len=260)
    if not text:
        return True
    if _BAD_MARKUP_RE.search(text):
        return True
    if _INLINE_REF_MARKER_RE.search(text):
        return True
    if _GENERIC_RE.search(text):
        return True
    if _LOW_INFORMATION_RE.match(text.strip()):
        return True
    if re.search(r"\.(?:pdf|md)\b", text, re.IGNORECASE):
        return True
    tokens = _semantic_tokens(text)
    if len(tokens) < 4 and len(text) < 16:
        return True
    return False


def _candidate_payload(base: Mapping[str, Any]) -> str:
    rows: list[str] = []
    metadata = " · ".join(
        part
        for part in (
            _text(base.get("authors"), max_len=180),
            _text(base.get("venue"), max_len=80),
            _text(base.get("year"), max_len=16),
        )
        if part
    )
    for label, key in (
        ("Reference title", "card_title"),
        ("Reference metadata", "__metadata__"),
        ("Answer sentence", "card_claim"),
        ("Evidence or citation context", "card_evidence"),
        ("Current takeaway", "card_takeaway"),
        ("Upstream role", "upstream_work_role"),
        ("Question relation", "user_question_relation"),
        ("Support relation", "support_relation"),
        ("Citation context", "citation_context"),
        ("Reference entry", "card_reference_entry"),
        ("Raw reference", "raw"),
        ("Location", "card_locator"),
        ("Quality warning", "card_warning"),
    ):
        value = metadata if key == "__metadata__" else _text(base.get(key), max_len=520)
        if value:
            rows.append(f"{label}: {value}")
    return "\n".join(rows[:11])


def citation_card_polish_cache_key(detail: Mapping[str, Any] | None) -> str:
    rec = normalize_citation_card_detail(detail)
    base = compose_citation_card(rec)
    selected = {
        "version": 4,
        "is_inpaper": bool(base.get("is_inpaper")),
        "num": str(base.get("num") or ""),
        "source_name": _text(base.get("source_name"), max_len=220),
        "title": _text(base.get("title") or base.get("card_title"), max_len=260),
        "card_title": _text(base.get("card_title"), max_len=260),
        "card_claim": _text(base.get("card_claim") or base.get("answer_claim"), max_len=420),
        "card_evidence": _text(base.get("card_evidence") or base.get("evidence_quote"), max_len=620),
        "citation_context": _text(base.get("citation_context"), max_len=620),
        "upstream_work_role": _text(base.get("upstream_work_role") or base.get("why_line"), max_len=420),
        "user_question_relation": _text(base.get("user_question_relation") or base.get("support_relation"), max_len=420),
        "card_reference_entry": _text(base.get("card_reference_entry") or base.get("raw"), max_len=620),
        "card_locator": _text(base.get("card_locator") or base.get("location_label"), max_len=260),
        "venue": _text(base.get("venue"), max_len=120),
        "year": _text(base.get("year"), max_len=16),
        "doi": _text(base.get("doi") or base.get("doi_url"), max_len=140),
    }
    blob = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8", "ignore")).hexdigest()


@lru_cache(maxsize=512)
def _llm_polish_citation_card_json(
    *,
    is_inpaper: bool,
    title: str,
    source_name: str,
    heading_path: str,
    current_takeaway: str,
    answer_claim: str,
    evidence: str,
    candidate_payload: str,
) -> str:
    if not citation_card_polish_enabled() or not candidate_payload:
        return ""
    try:
        settings = load_settings()
    except Exception:
        return ""
    if not getattr(settings, "api_key", None):
        return ""
    try:
        fast_settings = replace(
            settings,
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), _citation_card_polish_timeout_s()),
            max_retries=_citation_card_polish_max_retries(),
        )
    except Exception:
        fast_settings = settings
    if is_inpaper:
        route_hint = (
            "SystemB upstream bibliography reference: the current paper cites this upstream work near "
            "the answer sentence. Explain why that cited work is worth opening, without pretending it is "
            "direct evidence unless the provided citation context says so."
        )
    else:
        route_hint = (
            "SystemA answer evidence card: this source text is direct evidence for a sentence in the answer. "
            "Explain the concrete point supported by the quote."
        )
    try:
        out = DeepSeekChat(fast_settings).chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You polish a small citation popover card for a research reading assistant. "
                        "Return JSON only: {\"card_takeaway\":\"...\",\"card_claim\":\"...\",\"card_context_summary\":\"...\",\"card_support_explanation\":\"...\"}. "
                        "Write concise Chinese. card_takeaway is the primary polished line: name the specific mechanism, claim, limitation, or upstream role. "
                        "It must not be a title, location, bibliography entry, copied evidence, or generic relevance sentence. "
                        "card_claim is optional and should only be filled if it is a short answer-side statement that is not duplicated by the evidence. "
                        "card_context_summary is only for SystemB: summarize why the current paper cites this upstream work in the provided context. "
                        "Leave card_context_summary empty for SystemA or when the context is too weak. "
                        "card_support_explanation is optional and only for a non-obvious reliability or tracing note. Leave optional fields empty when they add no new information. "
                        "Do not mention authors, venue, year, DOI, citation count, IF/JCR, or repeat the paper/source title; those are shown elsewhere in the UI. "
                        "Use only the supplied fields. Do not invent facts. Do not rewrite the evidence quote itself. "
                        "Do not output markdown, bullets, table bars, formulas, DOC/SID/CITE/reference markers, UI labels, or generic phrases."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Card route: {route_hint}\n"
                        f"Reference/paper title: {title}\n"
                        f"Source paper: {source_name}\n"
                        f"Heading/location: {heading_path}\n"
                        f"Current takeaway: {current_takeaway}\n"
                        f"Answer sentence: {answer_claim}\n"
                        f"Evidence/context: {evidence}\n\n"
                        f"Available grounded fields:\n{candidate_payload}\n"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=300,
        )
    except Exception:
        return ""
    return str(out or "").strip()


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return dict(parsed) if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return dict(parsed) if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _baseline_texts(base: Mapping[str, Any]) -> dict[str, str]:
    return {
        "evidence": _text(base.get("card_evidence") or base.get("evidence_quote") or base.get("citation_context"), max_len=620),
        "context": _text(base.get("citation_context") or base.get("card_evidence") or base.get("evidence_quote"), max_len=620),
        "claim": _text(base.get("card_claim") or base.get("answer_claim"), max_len=420),
        "takeaway": _text(base.get("card_takeaway"), max_len=180),
        "title": _text(base.get("title") or base.get("card_title"), max_len=260),
        "source": _text(base.get("source_name"), max_len=220),
        "locator": _text(base.get("card_locator") or base.get("heading_path") or base.get("location_label"), max_len=260),
        "reference": _text(base.get("card_reference_entry") or base.get("raw") or base.get("cite_fmt"), max_len=620),
        "warning": _text(base.get("card_warning"), max_len=360),
        "venue": _text(base.get("venue"), max_len=120),
        "year": _text(base.get("year"), max_len=16),
        "authors": _text(base.get("authors"), max_len=220),
        "doi": _text(base.get("doi") or base.get("doi_url"), max_len=140),
    }


def _reject_polish_reason(
    *,
    key: str,
    text: str,
    route: str,
    baseline: Mapping[str, str],
    accepted: Mapping[str, Any],
) -> str:
    if _looks_bad_polish_text(text):
        return "bad_or_generic"
    if key in _TEXT_PATCH_KEYS and _looks_redundant_metadata_text(text, baseline):
        return "metadata_repeated"
    comparable = {
        label: value
        for label, value in baseline.items()
        if value and label not in {"warning"}
    }
    if key == "card_takeaway":
        for label, value in comparable.items():
            if label == "takeaway":
                continue
            if _sameish(text, value):
                return f"duplicates_{label}"
        if route == "system_b" and _sameish(text, baseline.get("reference", "")):
            return "duplicates_reference"
    elif key == "card_claim":
        for label in ("evidence", "takeaway", "title", "locator", "reference"):
            if _sameish(text, baseline.get(label, "")):
                return f"duplicates_{label}"
        if _sameish(text, str(accepted.get("card_takeaway") or "")):
            return "duplicates_takeaway"
    elif key == "card_context_summary":
        if route != "system_b":
            return "wrong_route"
        summary_reason = reject_system_b_context_summary(
            text,
            context=baseline.get("context") or baseline.get("evidence", ""),
            claim=baseline.get("claim", ""),
            title=baseline.get("title", ""),
            source=baseline.get("source", ""),
            reference_entry=baseline.get("reference", ""),
            locator=baseline.get("locator", ""),
            takeaway=str(accepted.get("card_takeaway") or baseline.get("takeaway", "")),
        )
        if summary_reason:
            return summary_reason
        for label in ("evidence", "claim", "takeaway", "title", "locator", "reference"):
            if _sameish(text, baseline.get(label, "")):
                return f"duplicates_{label}"
        for accepted_key in ("card_takeaway", "card_claim"):
            if _sameish(text, str(accepted.get(accepted_key) or "")):
                return f"duplicates_{accepted_key}"
    elif key == "card_support_explanation":
        for label in ("evidence", "claim", "takeaway", "title", "locator", "reference"):
            if _sameish(text, baseline.get(label, "")):
                return f"duplicates_{label}"
        for accepted_key in ("card_takeaway", "card_claim", "card_context_summary"):
            if _sameish(text, str(accepted.get(accepted_key) or "")):
                return f"duplicates_{accepted_key}"
    return ""


def _quality_score_for_patch(patch: Mapping[str, Any], *, rejected_count: int) -> float:
    score = 0.35
    if str(patch.get("card_takeaway") or "").strip():
        score += 0.34
    if str(patch.get("card_claim") or "").strip():
        score += 0.14
    if str(patch.get("card_context_summary") or "").strip():
        score += 0.16
    if str(patch.get("card_support_explanation") or "").strip():
        score += 0.1
    if rejected_count:
        score -= min(0.18, 0.04 * rejected_count)
    return round(max(0.0, min(1.0, score)), 3)


def _attach_card_view_patch(
    *,
    base: Mapping[str, Any],
    patch: dict[str, Any],
    accepted_keys: list[str],
    rejected: list[str],
) -> list[str]:
    refreshed = refresh_citation_card_contract({**dict(base), **patch})
    final_keys: list[str] = []
    for key in accepted_keys:
        text = _text(
            refreshed.get(key),
            max_len=220 if key == "card_context_summary" else (420 if key != "card_takeaway" else 160),
        )
        if not text:
            patch.pop(key, None)
            rejected.append(f"{key}:finalized_empty")
            continue
        patch[key] = text
        final_keys.append(key)
    if final_keys:
        patch["card_display_contract_version"] = refreshed.get("card_display_contract_version")
        patch["card_visible_sections"] = refreshed.get("card_visible_sections") or []
        patch["card_view"] = refreshed.get("card_view") or {}
        patch["citation_card_view_patch_version"] = _VIEW_PATCH_VERSION
    return final_keys


def polish_citation_card_detail(
    detail: Mapping[str, Any] | None,
    *,
    llm_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    rec = normalize_citation_card_detail(detail)
    base = compose_citation_card(rec)
    payload = _candidate_payload(base)
    if not payload:
        return {
            "citation_card_polish_status": "empty",
            "citation_card_polish_source": "none",
            "citation_card_polish_checked": True,
        }
    if not citation_card_polish_enabled() and llm_fn is None:
        return {
            "citation_card_polish_status": "disabled",
            "citation_card_polish_source": "disabled",
            "citation_card_polish_checked": True,
        }

    call = llm_fn or _llm_polish_citation_card_json
    raw = call(
        is_inpaper=bool(base.get("is_inpaper")),
        title=_text(base.get("title") or base.get("card_title"), max_len=260),
        source_name=_text(base.get("source_name"), max_len=220),
        heading_path=_text(base.get("heading_path") or base.get("card_locator"), max_len=260),
        current_takeaway=_text(base.get("card_takeaway"), max_len=180),
        answer_claim=_text(base.get("card_claim") or base.get("answer_claim"), max_len=420),
        evidence=_text(base.get("card_evidence") or base.get("evidence_quote") or base.get("citation_context"), max_len=620),
        candidate_payload=payload,
    )
    parsed = _parse_json_object(raw)
    route = "system_b" if bool(base.get("is_inpaper")) else "system_a"
    patch: dict[str, Any] = {
        "citation_card_polish_status": "full",
        "citation_card_polish_source": "llm",
        "citation_card_polish_checked": True,
        "citation_card_polish_route": route,
    }
    baseline = _baseline_texts(base)
    accepted_keys: list[str] = []
    rejected: list[str] = []
    for key in _TEXT_PATCH_KEYS:
        text = _clean_polish_candidate(
            parsed.get(key),
            max_len=220 if key == "card_context_summary" else (420 if key != "card_takeaway" else 160),
        )
        reason = _reject_polish_reason(key=key, text=text, route=route, baseline=baseline, accepted=patch)
        if reason:
            rejected.append(f"{key}:{reason}")
            continue
        patch[key] = text
        accepted_keys.append(key)
    if not accepted_keys:
        return {
            "citation_card_polish_status": "empty",
            "citation_card_polish_source": "llm_empty",
            "citation_card_polish_checked": True,
        }
    accepted_keys = _attach_card_view_patch(base=base, patch=patch, accepted_keys=accepted_keys, rejected=rejected)
    if not accepted_keys:
        return {
            "citation_card_polish_status": "empty",
            "citation_card_polish_source": "llm_empty",
            "citation_card_polish_checked": True,
        }
    patch["citation_card_polish_fields"] = accepted_keys
    patch["citation_card_polish_quality_score"] = _quality_score_for_patch(patch, rejected_count=len(rejected))
    if rejected:
        patch["citation_card_polish_rejected"] = rejected[:6]
    return patch
