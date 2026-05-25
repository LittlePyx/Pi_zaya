from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import replace
from functools import lru_cache
from typing import Any, Callable, Mapping

from kb.citation_card import compose_citation_card
from kb.config import load_settings
from kb.evidence_text import clean_display_text
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
    "cardSupportLabel": "card_support_label",
    "cardSupportExplanation": "card_support_explanation",
    "cardQualityFlags": "card_quality_flags",
}
_TEXT_PATCH_KEYS = ("card_takeaway", "card_claim", "card_support_explanation")
_BAD_MARKUP_RE = re.compile(r"\[\[?\s*CITE\s*:|```|<[^>]+>|!\[[^\]]*]\(|\|")
_GENERIC_RE = re.compile(
    r"\b(?:this reference is relevant|this evidence supports|good entry point|"
    r"directly relevant|upstream paper to open next|"
    r"这条(?:引用|证据|参考).{0,12}(?:相关|有用|值得打开|可以作为入口))\b",
    re.IGNORECASE,
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
    at = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", a))
    bt = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", b))
    if len(at) < 5 or len(bt) < 5:
        return False
    return len(at & bt) / max(1, min(len(at), len(bt))) >= 0.82


def _looks_bad_polish_text(value: str) -> bool:
    text = _text(value, max_len=260)
    if not text:
        return True
    if _BAD_MARKUP_RE.search(text):
        return True
    if _GENERIC_RE.search(text):
        return True
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text)
    if len(tokens) < 4 and len(text) < 16:
        return True
    return False


def _candidate_payload(base: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for label, key in (
        ("Answer sentence", "card_claim"),
        ("Evidence quote", "card_evidence"),
        ("Current takeaway", "card_takeaway"),
        ("Citation context", "citation_context"),
        ("Raw reference", "raw"),
        ("Location", "card_locator"),
    ):
        value = _text(base.get(key), max_len=520)
        if value:
            rows.append(f"{label}: {value}")
    return "\n".join(rows[:6])


def citation_card_polish_cache_key(detail: Mapping[str, Any] | None) -> str:
    rec = normalize_citation_card_detail(detail)
    base = compose_citation_card(rec)
    selected = {
        "version": 1,
        "is_inpaper": bool(base.get("is_inpaper")),
        "num": str(base.get("num") or ""),
        "source_name": _text(base.get("source_name"), max_len=220),
        "title": _text(base.get("title") or base.get("card_title"), max_len=260),
        "card_claim": _text(base.get("card_claim") or base.get("answer_claim"), max_len=420),
        "card_evidence": _text(base.get("card_evidence") or base.get("evidence_quote"), max_len=620),
        "citation_context": _text(base.get("citation_context"), max_len=620),
        "card_locator": _text(base.get("card_locator") or base.get("location_label"), max_len=260),
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
    route_hint = "SystemB upstream bibliography reference" if is_inpaper else "SystemA answer evidence card"
    try:
        out = DeepSeekChat(fast_settings).chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You polish a small citation popover card for a research reading assistant. "
                        "Return JSON only: {\"card_takeaway\":\"...\",\"card_claim\":\"...\",\"card_support_explanation\":\"...\"}. "
                        "Write concise Chinese. card_takeaway is the primary polished line: explain what the evidence/reference contributes. "
                        "card_claim is optional and should only be filled if it is a short answer-side statement that is not duplicated by the evidence. "
                        "card_support_explanation is optional and only for a non-obvious reliability or tracing note. "
                        "Use only the supplied fields. Do not invent facts. Do not rewrite the evidence quote itself. "
                        "Do not output markdown, bullets, table bars, formulas, DOC/SID/CITE markers, or generic phrases."
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
            max_tokens=220,
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
    patch: dict[str, Any] = {
        "citation_card_polish_status": "full",
        "citation_card_polish_source": "llm",
        "citation_card_polish_checked": True,
    }
    evidence = _text(base.get("card_evidence") or base.get("evidence_quote") or base.get("citation_context"), max_len=620)
    claim_seed = _text(base.get("card_claim") or base.get("answer_claim"), max_len=420)
    accepted_any = False
    for key in _TEXT_PATCH_KEYS:
        text = _text(parsed.get(key), max_len=420 if key != "card_takeaway" else 160)
        if _looks_bad_polish_text(text):
            continue
        if key == "card_claim" and (not text or _sameish(text, evidence)):
            continue
        if key == "card_support_explanation" and (
            _sameish(text, evidence) or _sameish(text, claim_seed) or _sameish(text, str(patch.get("card_takeaway") or ""))
        ):
            continue
        patch[key] = text
        accepted_any = True
    if not accepted_any:
        return {
            "citation_card_polish_status": "empty",
            "citation_card_polish_source": "llm_empty",
            "citation_card_polish_checked": True,
        }
    return patch
