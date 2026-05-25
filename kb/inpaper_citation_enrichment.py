from __future__ import annotations

import re
from typing import Any, Callable, Mapping, MutableMapping

from kb.citation_context import extract_inpaper_reference_context
from kb.source_blocks import normalize_inline_markdown


_STRUCT_CITE_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\]\]",
    re.IGNORECASE,
)
_STRUCT_CITE_SINGLE_RE = re.compile(
    r"(?<!\[)\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})(?:\s*:\s*(\d{1,4}))?\s*\](?!\])",
    re.IGNORECASE,
)
_STRUCT_CITE_SID_ONLY_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*\]\]",
    re.IGNORECASE,
)
_STRUCT_CITE_GARBAGE_RE = re.compile(r"\[\[?\s*CITE\s*:[^\]\n]*\]?\]", re.IGNORECASE)


def strip_structured_cite_tokens(text: str) -> str:
    """Remove internal structured citation markers while keeping surrounding prose."""

    out = str(text or "")
    out = _STRUCT_CITE_RE.sub("", out)
    out = _STRUCT_CITE_SINGLE_RE.sub("", out)
    out = _STRUCT_CITE_SID_ONLY_RE.sub("", out)
    out = _STRUCT_CITE_GARBAGE_RE.sub("", out)
    return out


def extract_structured_cite_answer_context_line(
    text: str,
    token_start: int,
    token_end: int,
    *,
    normalizer: Callable[[str], str] | None = None,
    max_len: int = 420,
) -> str:
    """Return the answer line around a structured citation token without the token."""

    try:
        start = max(0, int(token_start))
        end = max(start, int(token_end))
    except Exception:
        return ""
    source = str(text or "")
    left = source.rfind("\n", 0, start)
    left = 0 if left < 0 else left + 1
    right = source.find("\n", end)
    if right < 0:
        right = len(source)
    raw = strip_structured_cite_tokens(str(source[left:right] or "").strip())
    clean_fn = normalizer or normalize_inline_markdown
    try:
        raw = clean_fn(raw)
    except Exception:
        raw = normalize_inline_markdown(raw)
    raw = re.sub(r"\s+", " ", str(raw or "")).strip()
    raw = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", raw)
    return raw[: max(0, int(max_len or 420))]


def apply_answer_context_to_inpaper_detail(
    detail: MutableMapping[str, Any],
    answer_context: str,
    *,
    max_claim_chars: int = 420,
) -> bool:
    context = re.sub(r"\s+", " ", str(answer_context or "")).strip()
    if not context:
        return False
    if not str(detail.get("answer_claim") or "").strip():
        detail["answer_claim"] = context[:max_claim_chars]
    if not str(detail.get("citation_context") or "").strip():
        detail["citation_context"] = context[:max_claim_chars]
        detail["citation_context_source"] = "answer_context"
    if not str(detail.get("evidence_quote") or "").strip():
        detail["evidence_quote"] = context[:max_claim_chars]
        detail["evidence_source"] = "answer_context"
    if not str(detail.get("summary_line") or "").strip():
        detail["summary_line"] = context[:360]
        detail["summary_source"] = "answer_context"
    return True


def apply_source_context_to_inpaper_detail(
    detail: MutableMapping[str, Any],
    source_context: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(source_context, Mapping):
        return False
    context = str(source_context.get("citation_context") or "").strip()
    if not context:
        return False
    source_kind = str(source_context.get("citation_context_source") or "source_markdown").strip() or "source_markdown"
    detail["citation_context"] = context[:520]
    detail["citation_context_source"] = source_kind
    detail["evidence_quote"] = context[:520]
    detail["evidence_source"] = source_kind
    detail["summary_line"] = context[:360]
    detail["summary_source"] = source_kind
    for key in ("heading_path", "location_label", "anchor_kind"):
        value = str(source_context.get(key) or "").strip()
        if value:
            detail[key] = value
    for key in ("page_start", "page_end", "line_start", "line_end"):
        try:
            value_i = int(source_context.get(key) or 0)
        except Exception:
            value_i = 0
        if value_i > 0:
            detail[key] = value_i
    for key in ("block_id", "anchor_id"):
        value = str(source_context.get(key) or "").strip()
        if value:
            detail[key] = value
    detail["citation_context_quality"] = str(source_context.get("citation_context_quality") or "").strip()
    try:
        detail["citation_context_score"] = float(source_context.get("citation_context_score") or 0.0)
    except Exception:
        detail["citation_context_score"] = 0.0
    return True


def enrich_inpaper_detail_context(
    detail: MutableMapping[str, Any],
    *,
    source_path: str,
    ref_num: int,
    answer_context: str = "",
    source_answer_context: str | None = None,
    fallback_answer_context: bool = True,
    extract_context_fn: Callable[..., Mapping[str, Any]] = extract_inpaper_reference_context,
) -> MutableMapping[str, Any]:
    """Attach answer-line and citing-paper context to a SystemB citation detail."""

    local_answer_context = re.sub(r"\s+", " ", str(answer_context or "")).strip()
    if fallback_answer_context:
        apply_answer_context_to_inpaper_detail(detail, local_answer_context)
    elif local_answer_context and not str(detail.get("answer_claim") or "").strip():
        detail["answer_claim"] = local_answer_context[:420]

    try:
        ref_n = int(ref_num)
    except Exception:
        ref_n = 0
    query_context = source_answer_context if source_answer_context is not None else local_answer_context
    source_context: Mapping[str, Any] | None = None
    if str(source_path or "").strip() and ref_n > 0:
        try:
            source_context = extract_context_fn(
                str(source_path or "").strip(),
                ref_n,
                answer_context=str(query_context or ""),
            )
        except Exception:
            source_context = None
    apply_source_context_to_inpaper_detail(detail, source_context)
    return detail
