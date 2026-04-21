from __future__ import annotations

import os
import re
from pathlib import Path

from kb.answer_contract import (
    _apply_answer_contract_v1,
    _build_answer_quality_probe,
    _enhance_kb_miss_fallback,
    _reconcile_kb_notice,
)
from kb.paper_guide_contracts import (
    _build_paper_guide_render_packet_model,
    _build_paper_guide_retrieval_bundle_model,
    _build_paper_guide_support_pack_model,
    _paper_guide_grounding_trace_segment_model_from_raw,
    _paper_guide_model_dump,
)
from kb.paper_guide.router import _resolve_paper_guide_intent
from kb.paper_guide_postprocess import (
    _sanitize_paper_guide_answer_for_user,
    _sanitize_structured_cite_tokens,
    _strip_model_ref_section,
)
from kb.reference_query_family import (
    extract_multi_paper_topic as _shared_extract_multi_paper_topic,
    prompt_explicitly_requests_multi_paper_list,
    prompt_prefers_zh,
    prompt_requires_reference_focus_match as _shared_prompt_requires_reference_focus_match,
    prompt_targets_sci_topic as _shared_prompt_targets_sci_topic,
)
from kb.source_blocks import normalize_inline_markdown
from ui.chat_widgets import _normalize_math_markdown

_CITE_CANON_RE = re.compile(
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
_SID_INLINE_RE = re.compile(r"\[\s*SID\s*:\s*[A-Za-z0-9_-]{4,24}\s*\]", re.IGNORECASE)
_SID_RE = re.compile(r"^[A-Za-z0-9_-]{4,24}$")
_INLINE_REF_NUM_RE = re.compile(r"\[(\d{1,4})\]")
_FREEFORM_NUMERIC_CITE_RE = re.compile(
    r"(?<![!\\])\[(\d{1,4}(?:\s*(?:-|–|—|,)\s*\d{1,4})*)\](?!\()"
)
_DOC_HEADING_LINE_RE = re.compile(r"(?im)^\s*DOC-\d{1,3}(?:-S\d{1,3})?\s*[:：]\s*$")
_DOC_TITLE_LINE_RE = re.compile(r"(?im)^\s*(?:title|标题)\s*[:：]\s*(.+?)\s*$")
_DOC_DIAGNOSTIC_LINE_RE = re.compile(
    r"(?im)^\s*(?:note|注意|说明)\s*[:：]?\s*DOC-\d{1,3}(?:-S\d{1,3})?[^\n]*$"
)
_DOC_RESULT_PREAMBLE_RE = re.compile(
    r"(?im)^\s*(?:based on the retrieved results|according to the retrieved results|根据提供的检索结果|根据检索结果)[^:：\n]*[:：]?\s*$"
)
_PAPER_GUIDE_NEGATIVE_SHELL_RE = re.compile(
    r"(?i)\b(?:not stated|does not state|do not state|does not specify|do not specify|"
    r"does not discuss|do not discuss|does not mention|do not mention|makes no statement|"
    r"cannot be determined from the retrieved)\b"
)
_PAPER_GUIDE_SUPPLEMENT_BLOCK_MARKER_RE = re.compile(
    r"(?i)(补充说明（通用知识，非检索片段内容|supplementary note \(generic knowledge, non-retrieved content\))"
)
_PAPER_GUIDE_SUPPLEMENT_OPTOUT_RE = re.compile(
    r"(?i)(只基于原文|仅基于原文|不要补充|不要扩展|不要通用知识|only from the paper|paper-only|no supplement|no general knowledge)"
)
_PAPER_GUIDE_SUPPLEMENT_DISCLAIMER_RE = re.compile(
    r"(?i)(以下内容是\s*AI\s*基于通用知识的补充|"
    r"不代表论文原文明确陈述|"
    r"the notes below are ai supplemental context|"
    r"not explicit claims from the paper)"
)
_STRUCTURED_ANSWER_SECTION_RE = re.compile(
    r"(?im)^\s*(Conclusion|Evidence|Limits|Next Steps|结论|依据|证据|边界|限制|局限|下一步建议|下一步)\s*[:：]"
)
_CROSS_PAPER_QUERY_RE = re.compile(
    r"(\bwhich other papers?\b|\bother papers?\b|\bbesides this paper\b|\banother paper\b|"
    r"除此之外|除(?:了)?这篇|其他论文|别的论文|还有哪些论文|另一篇论文)",
    flags=re.IGNORECASE,
)


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _as_positive_int(value: object) -> int:
    try:
        n = int(value)
    except Exception:
        return 0
    return n if n > 0 else 0


def _collect_low_confidence_candidate_refs(
    *,
    support_resolution: list[dict] | None,
    candidate_refs_by_source: dict[str, list[int]] | None,
    retrieval_confidence_hint: dict[str, object] | None,
    max_items: int = 6,
) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()

    def _add(value: object) -> None:
        n = _as_positive_int(value)
        if n <= 0 or n in seen:
            return
        seen.add(n)
        out.append(n)

    for rec in list(support_resolution or []):
        if not isinstance(rec, dict):
            continue
        _add(rec.get("resolved_ref_num"))
        for key in ("candidate_refs", "support_ref_candidates", "ref_nums"):
            for item in list(rec.get(key) or []):
                _add(item)

    for refs in list((candidate_refs_by_source or {}).values()):
        for item in list(refs or []):
            _add(item)

    hint = dict(retrieval_confidence_hint or {})
    for item in list(hint.get("candidate_refs") or []):
        _add(item)
    for key in ("resolved_ref_num", "top_ref_num"):
        _add(hint.get(key))

    return [int(n) for n in out[: max(1, int(max_items or 6))] if int(n) > 0]


def _has_structured_cite_marker(text: str) -> bool:
    return bool(_CITE_CANON_RE.search(str(text or "")))


def _collect_inline_reference_numbers(text: str, *, max_items: int = 6) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for m in _INLINE_REF_NUM_RE.finditer(str(text or "")):
        n = _as_positive_int(m.group(1))
        if n <= 0 or n in seen:
            continue
        seen.add(n)
        out.append(n)
        if len(out) >= max(1, int(max_items or 6)):
            break
    return out


def _prompt_explicitly_requests_citation_lookup(prompt: str) -> bool:
    text = str(prompt or "").strip().lower()
    if not text:
        return False
    patterns = (
        "citation",
        "cited",
        "cite",
        "reference number",
        "reference numbers",
        "which reference",
        "which references",
        "what in-paper citation",
        "prior work is",
        "attributed to",
        "引用",
        "引文",
        "参考文献",
        "编号",
    )
    return any(pattern in text for pattern in patterns)


def _should_preserve_final_answer_numeric_citations(
    *,
    prompt: str,
    answer_output_mode: str,
    paper_guide_mode: bool,
    prompt_family: str,
) -> bool:
    if str(prompt_family or "").strip().lower() == "citation_lookup":
        return True
    if "citation" in str(answer_output_mode or "").strip().lower():
        return True
    if paper_guide_mode and _prompt_explicitly_requests_citation_lookup(prompt):
        return True
    return False


def _strip_final_answer_citation_markers(answer: str, *, preserve_numeric_markers: bool) -> str:
    text = str(answer or "")
    if not text:
        return text
    out = _sanitize_structured_cite_tokens(text)
    out = _CITE_CANON_RE.sub("", out)
    out = _STRUCT_CITE_SINGLE_RE.sub("", out)
    out = _STRUCT_CITE_SID_ONLY_RE.sub("", out)
    out = _STRUCT_CITE_GARBAGE_RE.sub("", out)
    out = _SID_INLINE_RE.sub("", out)
    if not preserve_numeric_markers:
        out = _FREEFORM_NUMERIC_CITE_RE.sub("", out)
    out = re.sub(r"[ \t]+([,.;:!?])", r"\1", out)
    out = re.sub(r"(?m)[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _sanitize_internal_doc_label_blocks(answer: str) -> str:
    text = str(answer or "").strip()
    if not text or ("DOC-" not in text.upper()):
        return text

    lines = [str(line or "").rstrip() for line in text.splitlines()]
    out: list[str] = []
    idx = 0
    converted = False

    def _push_block(value: str) -> None:
        block = str(value or "").strip()
        if block:
            out.append(block)

    while idx < len(lines):
        line = lines[idx].strip()
        if _DOC_RESULT_PREAMBLE_RE.match(line):
            idx += 1
            continue
        if _DOC_DIAGNOSTIC_LINE_RE.match(line):
            converted = True
            idx += 1
            continue
        if not _DOC_HEADING_LINE_RE.match(line):
            _push_block(lines[idx])
            idx += 1
            continue

        converted = True
        idx += 1
        title = ""
        body_lines: list[str] = []
        while idx < len(lines):
            current = lines[idx].strip()
            if _DOC_HEADING_LINE_RE.match(current):
                break
            if _DOC_DIAGNOSTIC_LINE_RE.match(current):
                idx += 1
                continue
            title_match = _DOC_TITLE_LINE_RE.match(current)
            if title_match and not title:
                title = str(title_match.group(1) or "").strip()
                idx += 1
                continue
            if current:
                body_lines.append(current)
            idx += 1

        body = re.sub(r"\s+", " ", " ".join(body_lines)).strip()
        if title and body:
            _push_block(f"- {title}: {body}")
        elif title:
            _push_block(f"- {title}")
        elif body:
            _push_block(f"- {body}")

    if not converted:
        return text

    out_text = "\n\n".join(part for part in out if str(part or "").strip())
    out_text = re.sub(r"\n{3,}", "\n\n", out_text).strip()
    return out_text or text


def _source_name_from_path_like(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    name = Path(raw).name
    for suffix in (".en.md", ".zh.md", ".md"):
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".pdf"
    return name


def _normalize_topic_identity(text: str) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return ""
    raw = raw.replace(".en.md", " ").replace(".md", " ").replace(".pdf", " ")
    raw = re.sub(r"[_/\\]+", " ", raw)
    raw = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def _single_line_summary(text: str, *, source_name: str = "", max_chars: int = 180) -> str:
    cleaned = _normalize_math_markdown(normalize_inline_markdown(str(text or "").strip()))
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*(?:abstract|introduction|related work|conclusion|conclusions)\s*[:.-]?\s*", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*[.)-]?\s*(?:abstract|introduction|related work|conclusion|conclusions)\s*[:.-]?\s*", "", cleaned)
    cleaned = re.sub(r"\$[^$\n]{1,60}\$", " ", cleaned)
    cleaned = cleaned.replace("\\sim", "~").replace("\\mum", "um").replace("\\mu", "u")
    cleaned = re.sub(r"\\[A-Za-z]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -\n\t")
    source_display = str(source_name or "").strip()
    source_stem = re.sub(r"(?i)\.pdf$", "", source_display).strip()
    if source_stem:
        cleaned = re.sub(rf"^\s*{re.escape(source_stem)}\s*", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"^[A-Z][A-Za-z.\-\s,]{24,220}(?=\bAbstract\b)", "", cleaned).strip()
    cleaned = re.sub(r"^(?:figure|table)\s+\d+\s*[:.-]?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^(?:abstract)\s*[:.-]?\s*", "", cleaned, flags=re.I)
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    trimmed = cleaned[: max_chars - 1].rstrip()
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0].rstrip()
    return trimmed + "…"



def _sanitize_multi_paper_doc_list_entry_for_scoring(*, prompt: str, raw_item: dict) -> dict:
    entry = {k: v for k, v in dict(raw_item or {}).items() if v not in ("", None, [], {})}
    summary = str(entry.get("summary_line") or "").strip()
    if _looks_generic_multi_paper_support_text(summary, prompt=prompt):
        entry.pop("summary_line", None)
    primary = dict(entry.get("primary_evidence") or {}) if isinstance(entry.get("primary_evidence"), dict) else {}
    primary_snippet_was_generic = False
    if primary:
        snippet = str(primary.get("highlight_snippet") or primary.get("snippet") or "").strip()
        if _looks_generic_multi_paper_support_text(snippet, prompt=prompt):
            primary_snippet_was_generic = True
            primary.pop("snippet", None)
            primary.pop("highlight_snippet", None)
        if primary:
            entry["primary_evidence"] = primary
        else:
            entry.pop("primary_evidence", None)
    summary = str(entry.get("summary_line") or "").strip()
    topic = _extract_multi_paper_topic(prompt)
    topic_norm = _normalize_topic_identity(topic)
    summary_norm = _normalize_topic_identity(summary)
    if summary and topic_norm and summary_norm and _surface_has_token_sequence(summary_norm, topic_norm.split()):
        support_surface = _multi_paper_entry_surface(
            source_name=str(entry.get("source_name") or "").strip(),
            heading_path=str(entry.get("heading_path") or "").strip(),
            summary_line="",
            primary_evidence=entry.get("primary_evidence") if isinstance(entry.get("primary_evidence"), dict) else {},
        )
        support_surface_norm = _normalize_topic_identity(support_surface)
        support_has_topic = _multi_paper_segment_matches(
            segment=topic_norm,
            surface_norm=support_surface_norm,
            surface_tokens=support_surface_norm.split(),
            raw_low=str(support_surface or "").lower(),
        )
        if primary_snippet_was_generic and (not support_has_topic):
            entry.pop("summary_line", None)
    return entry


def _multi_paper_topic_segments(topic: str) -> list[str]:
    norm = _normalize_topic_identity(topic)
    if not norm:
        return []
    pieces = re.split(
        r"\b(?:for|via|using|through|with|without|about|regarding|based on|based)\b",
        norm,
        flags=re.I,
    )
    out: list[str] = []
    for piece in pieces:
        seg = re.sub(r"\s+", " ", str(piece or "").strip())
        if seg:
            out.append(seg)
    return out


def _surface_has_token_sequence(surface_norm: str, token_seq: list[str]) -> bool:
    tokens = [str(tok or "").strip() for tok in list(token_seq or []) if str(tok or "").strip()]
    if not surface_norm or not tokens:
        return False
    phrase = " ".join(tokens).strip()
    if not phrase:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", surface_norm, flags=re.I))


def _is_informative_multi_paper_focus_token(token: str) -> bool:
    low = str(token or "").strip().lower()
    if not low:
        return False
    generic_tokens = {
        "single",
        "pixel",
        "imaging",
        "image",
        "images",
        "paper",
        "papers",
        "library",
        "libraries",
    }
    return low not in generic_tokens


def _multi_paper_segment_matches(
    *,
    segment: str,
    surface_norm: str,
    surface_tokens: list[str],
    raw_low: str,
) -> bool:
    seg_norm = _normalize_topic_identity(segment)
    if not seg_norm:
        return False
    seg_tokens = [tok for tok in seg_norm.split() if tok and len(tok) >= 4]
    if not seg_tokens:
        return False
    surface_token_set = set(surface_tokens)
    if len(seg_tokens) == 1:
        token = str(seg_tokens[0] or "")
        return bool(token in surface_token_set) and (not _multi_paper_focus_term_only_negated(token, raw_low))
    if _surface_has_token_sequence(surface_norm, seg_tokens):
        return not _multi_paper_focus_term_only_negated(" ".join(seg_tokens), raw_low)
    non_negated_tokens = [
        tok for tok in seg_tokens
        if (tok in surface_token_set) and (not _multi_paper_focus_term_only_negated(tok, raw_low))
    ]
    for width in range(min(3, len(seg_tokens)), 1, -1):
        for idx in range(0, len(seg_tokens) - width + 1):
            phrase_tokens = seg_tokens[idx : idx + width]
            if not any(_is_informative_multi_paper_focus_token(tok) for tok in phrase_tokens):
                continue
            if _surface_has_token_sequence(surface_norm, phrase_tokens):
                return True
    if len(seg_tokens) == 2:
        return len(non_negated_tokens) >= 2
    return len(non_negated_tokens) >= len(seg_tokens)


def _multi_paper_focus_match(
    *,
    prompt: str,
    source_name: str,
    heading_path: str,
    summary_line: str,
    primary_evidence: dict | None,
) -> bool:
    topic = _extract_multi_paper_topic(prompt)
    if not topic:
        return False
    surface = _multi_paper_entry_surface(
        source_name=source_name,
        heading_path=heading_path,
        summary_line=summary_line,
        primary_evidence=primary_evidence,
    )
    surface_norm = _normalize_topic_identity(surface)
    if not surface_norm:
        return False
    raw_low = str(surface or "").lower()
    surface_tokens = [tok for tok in surface_norm.split() if tok]
    segments = _multi_paper_topic_segments(topic)
    if not segments:
        return False
    for segment in segments:
        if not _multi_paper_segment_matches(
            segment=segment,
            surface_norm=surface_norm,
            surface_tokens=surface_tokens,
            raw_low=raw_low,
        ):
            return False
    return True



def _multi_paper_entry_surface(
    *,
    source_name: str,
    heading_path: str,
    summary_line: str,
    primary_evidence: dict | None,
) -> str:
    primary = dict(primary_evidence or {}) if isinstance(primary_evidence, dict) else {}
    parts = [
        str(source_name or "").strip(),
        str(heading_path or "").strip(),
        str(summary_line or "").strip(),
        str(primary.get("snippet") or "").strip(),
        str(primary.get("highlight_snippet") or "").strip(),
        str(primary.get("selection_reason") or "").strip(),
    ]
    return " ".join(part for part in parts if part)


def _multi_paper_topic_score(
    *,
    prompt: str,
    source_name: str,
    heading_path: str,
    summary_line: str,
    primary_evidence: dict | None,
) -> float:
    surface = _multi_paper_entry_surface(
        source_name=source_name,
        heading_path=heading_path,
        summary_line=summary_line,
        primary_evidence=primary_evidence,
    )
    surface_norm = _normalize_topic_identity(surface)
    raw_low = str(surface or "").lower()
    if not surface_norm:
        return 0.0

    score = 0.0
    topic = _extract_multi_paper_topic(prompt)
    focus_matched = _multi_paper_focus_match(
        prompt=prompt,
        source_name=source_name,
        heading_path=heading_path,
        summary_line=summary_line,
        primary_evidence=primary_evidence,
    )
    prompt_requires_focus = _multi_paper_prompt_requires_explicit_focus_match(prompt)
    generic_topic_stop = {
        "which", "papers", "paper", "other", "library", "libraries",
        "mention", "mentions", "mentioned", "discuss", "discusses", "discussed",
        "image", "images", "imaging", "technique", "techniques",
        "single", "pixel",
    }
    topic_tokens = [
        tok for tok in _normalize_topic_identity(topic).split()
        if tok and len(tok) >= 4 and tok not in generic_topic_stop
    ]
    if topic_tokens:
        surface_token_set = set(surface_norm.split())
        overlap_tokens = [tok for tok in topic_tokens if tok in surface_token_set]
        overlap = len(overlap_tokens)
        non_negated_overlap = [
            tok for tok in overlap_tokens
            if not _multi_paper_focus_term_only_negated(tok, raw_low)
        ]
        overlap = len(non_negated_overlap)
        if overlap >= 2:
            score += 1.2 * float(overlap)
        elif overlap == 1:
            token = str(non_negated_overlap[0] or "")
            min_len = 4 if len(topic_tokens) <= 1 else 6
            if len(token) >= min_len:
                score += 1.4 if len(topic_tokens) <= 1 else 1.2
    if focus_matched:
        score += 2.6
    elif prompt_requires_focus and topic and (not _prompt_targets_sci_topic(prompt)):
        return 0.0

    if _prompt_targets_sci_topic(prompt):
        sci_positive_norm = (
            "snapshot compressive imaging",
            "snapshot compressive image",
            "single shot compressive spectral imaging",
        )
        sci_positive_raw = (
            "scinerf",
            "scigs",
            "snapshot compressive imaging",
            "snapshot compressive image",
            "single-shot compressive spectral imaging",
            "single shot compressive spectral imaging",
        )
        if re.search(r"\bsci\b", raw_low):
            score += 3.5
        if any(alias in surface_norm for alias in sci_positive_norm):
            score += 3.5
        if any(alias in raw_low for alias in sci_positive_raw):
            score += 2.5
        if ("single pixel imaging" in surface_norm) and (score <= 0.0):
            score -= 2.5
        if ("single pixel compressive holography" in surface_norm) and (score <= 0.0):
            score -= 3.0
        if ("compressive sensing" in surface_norm) and (score <= 0.0):
            score -= 1.2
    return score


def _classify_multi_paper_topic_match(
    *,
    prompt: str,
    source_name: str,
    heading_path: str,
    summary_line: str,
    primary_evidence: dict | None,
) -> str:
    surface = _multi_paper_entry_surface(
        source_name=source_name,
        heading_path=heading_path,
        summary_line=summary_line,
        primary_evidence=primary_evidence,
    )
    surface_norm = _normalize_topic_identity(surface)
    raw_low = str(surface or "").lower()
    if not surface_norm:
        return ""
    topic_score = _multi_paper_topic_score(
        prompt=prompt,
        source_name=source_name,
        heading_path=heading_path,
        summary_line=summary_line,
        primary_evidence=primary_evidence,
    )
    if _prompt_targets_sci_topic(prompt):
        if re.search(r"\bsci\b", raw_low) or ("snapshot compressive imaging" in raw_low):
            return "explicit_sci_mention"
        if (
            ("single-shot compressive spectral imaging" in raw_low)
            or ("single shot compressive spectral imaging" in raw_low)
        ):
            return "sci_related_predecessor"
    return "topic_aligned" if topic_score > 0.0 else ""


def _multi_paper_topic_match_rank(match_kind: str) -> int:
    kind = str(match_kind or "").strip().lower()
    if kind == "explicit_sci_mention":
        return 2
    if kind == "sci_related_predecessor":
        return 1
    if kind:
        return 1
    return 0


def _multi_paper_topic_match_note(*, prompt: str, match_kind: str) -> str:
    kind = str(match_kind or "").strip().lower()
    if not kind:
        return ""
    prefer_zh = bool(prompt_prefers_zh(prompt))
    if kind == "explicit_sci_mention":
        if prefer_zh:
            return "\u6587\u4e2d\u660e\u786e\u63d0\u5230 Snapshot Compressive Imaging (SCI)\u3002"
        return "The paper explicitly mentions Snapshot Compressive Imaging (SCI)."
    if kind == "sci_related_predecessor":
        if prefer_zh:
            return "\u8fd9\u7bc7\u66f4\u9002\u5408\u89c6\u4e3a\u4e0e SCI \u76f8\u5173\u7684\u65e9\u671f\u524d\u8eab\u5de5\u4f5c\uff1a\u8ba8\u8bba\u7684\u662f single-shot compressive spectral imaging\uff0c\u4e0e SCI \u6982\u5ff5\u76f8\u5173\uff0c\u4f46\u4e0d\u662f\u4e25\u683c\u7684 SCI \u672f\u8bed\u547d\u4e2d\u3002"
        return "This is better treated as an early related predecessor: it discusses single-shot compressive spectral imaging, which is SCI-adjacent rather than an exact SCI term match."
    return ""


def _filter_multi_paper_doc_list_contract(*, prompt: str, doc_list: list[dict] | None) -> list[dict]:
    rows: list[dict] = []
    for idx, raw_item in enumerate(list(doc_list or [])):
        if not isinstance(raw_item, dict):
            continue
        entry = _sanitize_multi_paper_doc_list_entry_for_scoring(
            prompt=prompt,
            raw_item=raw_item,
        )
        entry["_topic_score"] = _multi_paper_topic_score(
            prompt=prompt,
            source_name=str(entry.get("source_name") or "").strip(),
            heading_path=str(entry.get("heading_path") or "").strip(),
            summary_line=str(entry.get("summary_line") or "").strip(),
            primary_evidence=entry.get("primary_evidence") if isinstance(entry.get("primary_evidence"), dict) else {},
        )
        entry["topic_match_kind"] = _classify_multi_paper_topic_match(
            prompt=prompt,
            source_name=str(entry.get("source_name") or "").strip(),
            heading_path=str(entry.get("heading_path") or "").strip(),
            summary_line=str(entry.get("summary_line") or "").strip(),
            primary_evidence=entry.get("primary_evidence") if isinstance(entry.get("primary_evidence"), dict) else {},
        )
        entry["_topic_match_rank"] = _multi_paper_topic_match_rank(str(entry.get("topic_match_kind") or ""))
        entry["_order"] = idx
        rows.append(entry)

    positive_rows = [row for row in rows if float(row.get("_topic_score") or 0.0) > 0.0]
    if positive_rows:
        rows = positive_rows
    elif _multi_paper_prompt_requires_explicit_focus_match(prompt):
        return []

    rows.sort(
        key=lambda item: (
            -int(item.get("_topic_match_rank") or 0),
            -float(item.get("_topic_score") or 0.0),
            int(item.get("_order") or 0),
        )
    )
    return [
        {k: v for k, v in row.items() if not str(k).startswith("_")}
        for row in rows
    ]


def _doc_list_entry_matches_bound_source(
    entry: dict,
    *,
    bound_source_path: str,
    bound_source_name: str,
) -> bool:
    if not isinstance(entry, dict):
        return False
    target_tokens = {
        token
        for token in (
            _normalize_topic_identity(bound_source_path),
            _normalize_topic_identity(bound_source_name),
            _normalize_topic_identity(_source_name_from_path_like(bound_source_path)),
        )
        if token
    }
    if not target_tokens:
        return False
    candidate_tokens = {
        token
        for token in (
            _normalize_topic_identity(str(entry.get("source_path") or "")),
            _normalize_topic_identity(str(entry.get("source_name") or "")),
            _normalize_topic_identity(_source_name_from_path_like(str(entry.get("source_path") or ""))),
        )
        if token
    }
    if not candidate_tokens:
        return False
    if candidate_tokens.intersection(target_tokens):
        return True
    for left in candidate_tokens:
        for right in target_tokens:
            if (len(left) >= 20 and left in right) or (len(right) >= 20 and right in left):
                return True
    return False


def _exclude_bound_source_from_multi_paper_doc_list_contract(
    *,
    doc_list: list[dict] | None,
    bound_source_path: str,
    bound_source_name: str,
) -> list[dict]:
    rows = [dict(item) for item in list(doc_list or []) if isinstance(item, dict)]
    if not rows:
        return []
    out: list[dict] = []
    for item in rows:
        if _doc_list_entry_matches_bound_source(
            item,
            bound_source_path=bound_source_path,
            bound_source_name=bound_source_name,
        ):
            continue
        out.append(item)
    return out


def _multi_paper_primary_precision_score(primary_evidence: dict | None) -> tuple[int, int, int, int, int, int]:
    primary = dict(primary_evidence or {}) if isinstance(primary_evidence, dict) else {}
    if not primary:
        return (0, 0, 0, 0, 0, 0)
    reason = str(primary.get("selection_reason") or primary.get("selectionReason") or "").strip().lower()
    reason_rank = {
        "prompt_aligned_block": 8,
        "prompt_aligned": 7,
        "reader_open": 5,
        "strict_locate": 5,
        "provenance_segment": 5,
        "shared_refs_pack": 5,
        "pending_section_seed": 2,
        "shared_contract_seed": 1,
        "answer_hit_top": 0,
    }.get(reason, 3 if reason else 0)
    strict_locate = primary.get("strict_locate")
    if strict_locate is None:
        strict_locate = primary.get("strictLocate")
    return (
        1 if bool(strict_locate) else 0,
        1 if str(primary.get("block_id") or primary.get("blockId") or "").strip() else 0,
        1 if str(primary.get("anchor_id") or primary.get("anchorId") or "").strip() else 0,
        1 if str(primary.get("heading_path") or primary.get("headingPath") or "").strip() else 0,
        1
        if str(primary.get("highlight_snippet") or primary.get("snippet") or "").strip()
        else 0,
        reason_rank,
    )


def _multi_paper_primary_is_weak(primary_evidence: dict | None) -> bool:
    primary = dict(primary_evidence or {}) if isinstance(primary_evidence, dict) else {}
    if not primary:
        return True
    strict_locate = primary.get("strict_locate")
    if strict_locate is None:
        strict_locate = primary.get("strictLocate")
    if bool(strict_locate):
        return False
    if str(primary.get("block_id") or primary.get("blockId") or "").strip():
        return False
    if str(primary.get("anchor_id") or primary.get("anchorId") or "").strip():
        return False
    reason = str(primary.get("selection_reason") or primary.get("selectionReason") or "").strip().lower()
    return reason in {"", "answer_hit_top", "pending_section_seed"}


def _looks_like_multi_paper_section_heading(heading: str) -> bool:
    text = re.sub(r"\s+", " ", str(heading or "").strip())
    if not text:
        return False
    low = text.lower()
    if re.match(r"^(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*[.)-]?\s+[a-z]", low, flags=re.I):
        return True
    return bool(
        re.match(
            r"(?i)^(?:abstract|introduction|related work|background|preliminar(?:y|ies)|"
            r"method(?:s)?|approach|framework|experiments?|results?|discussion|"
            r"conclusion(?:s)?|applications?|appendix|supplementary)\b",
            text,
        )
    )


def _extract_multi_paper_surface_seed(raw_text: str) -> tuple[str, str]:
    raw = str(raw_text or "").strip()
    if not raw:
        return "", ""

    abstract_match = re.search(
        r"(?is)(?:^|\n)\s*\*\*Abstract\*\*\s*[:：]\s*(.+?)(?=(?:\n\s*#{1,6}\s+\S)|\Z)",
        raw,
    )
    if abstract_match:
        return "Abstract", str(abstract_match.group(1) or "").strip()

    heading_matches = list(re.finditer(r"(?m)^\s{0,3}#{1,6}\s*([^\n#]{1,140})\s*$", raw))
    for idx, match in enumerate(heading_matches):
        heading = re.sub(r"\s+", " ", str(match.group(1) or "").strip())
        if not _looks_like_multi_paper_section_heading(heading):
            continue
        next_match = heading_matches[idx + 1] if (idx + 1) < len(heading_matches) else None
        excerpt = raw[match.end() : (next_match.start() if next_match else len(raw))].strip()
        return heading, excerpt
    return "", raw


def _normalize_multi_paper_surface_seed(
    *,
    source_name: str,
    heading_path: str,
    raw_text: str,
) -> tuple[str, str]:
    normalized_heading = str(heading_path or "").strip()
    inferred_heading, excerpt_text = _extract_multi_paper_surface_seed(raw_text)
    if inferred_heading:
        normalized_heading = inferred_heading
    normalized_summary = _single_line_summary(
        str(excerpt_text or raw_text or "").strip(),
        source_name=source_name,
    )
    return normalized_heading, normalized_summary


def _normalize_multi_paper_contract_primary_evidence(
    *,
    source_path: str,
    source_name: str,
    heading_path: str,
    raw_text: str,
    primary_evidence: dict | None,
    selection_reason: str,
) -> dict:
    primary = dict(primary_evidence or {}) if isinstance(primary_evidence, dict) else {}
    weak_primary = _multi_paper_primary_is_weak(primary)
    normalized_heading, normalized_summary = _normalize_multi_paper_surface_seed(
        source_name=source_name,
        heading_path=heading_path,
        raw_text=raw_text,
    )
    out = {
        key: value
        for key, value in primary.items()
        if value not in ("", None, [], {})
    }
    if source_path and (not str(out.get("source_path") or "").strip()):
        out["source_path"] = source_path
    if source_name and (not str(out.get("source_name") or "").strip()):
        out["source_name"] = source_name
    if normalized_heading and (weak_primary or (not str(out.get("heading_path") or "").strip())):
        out["heading_path"] = normalized_heading
    if normalized_summary and (
        weak_primary
        or (
            not str(out.get("highlight_snippet") or out.get("snippet") or "").strip()
        )
    ):
        out["snippet"] = normalized_summary
        out["highlight_snippet"] = normalized_summary
    if selection_reason and (not str(out.get("selection_reason") or "").strip()):
        out["selection_reason"] = str(selection_reason or "").strip()
    return {
        key: value
        for key, value in out.items()
        if value not in ("", None, [], {})
    }


def _pick_multi_paper_card_raw_summary(
    *,
    prompt: str,
    card: dict,
    primary_evidence: dict | None,
) -> str:
    primary = dict(primary_evidence or {}) if isinstance(primary_evidence, dict) else {}
    primary_summary = str(primary.get("highlight_snippet") or primary.get("snippet") or "").strip()
    if primary_summary and (not _looks_generic_multi_paper_support_text(primary_summary, prompt=prompt)):
        return primary_summary

    card_summary = str(card.get("snippet") or "").strip()
    deepread_candidates = [
        str(item or "").strip()
        for item in list(card.get("deepread_texts") or [])
        if str(item or "").strip()
    ]
    deepread_summary = str(deepread_candidates[0] or "").strip() if deepread_candidates else ""

    if card_summary and (not _looks_generic_multi_paper_support_text(card_summary, prompt=prompt)):
        return card_summary
    if deepread_summary and (not _looks_generic_multi_paper_support_text(deepread_summary, prompt=prompt)):
        return deepread_summary
    return primary_summary or card_summary or deepread_summary


def _build_multi_paper_doc_list_contract(
    *,
    prompt: str,
    seed_docs: list[dict] | None = None,
    answer_hits: list[dict] | None,
    evidence_cards: list[dict] | None,
) -> list[dict]:
    entries: list[dict] = []
    entry_by_source: dict[str, dict] = {}

    def _merge_entry(
        *,
        source_path: str,
        source_name: str,
        heading_path: str,
        summary: str,
        primary_evidence: dict | None,
        rank: int,
    ) -> None:
        src = str(source_path or "").strip()
        if not src:
            return
        entry = entry_by_source.get(src)
        if entry is None:
            entry = {
                "source_path": src,
                "source_name": str(source_name or "").strip() or _source_name_from_path_like(src),
                "heading_path": "",
                "summary_line": "",
                "_source_rank": int(rank),
            }
            entry_by_source[src] = entry
            entries.append(entry)
        else:
            entry["_source_rank"] = min(int(entry.get("_source_rank") or rank), int(rank))

        source_name_norm = str(source_name or "").strip() or _source_name_from_path_like(src)
        if source_name_norm and (not str(entry.get("source_name") or "").strip()):
            entry["source_name"] = source_name_norm

        current_primary_score = _multi_paper_primary_precision_score(
            entry.get("primary_evidence") if isinstance(entry.get("primary_evidence"), dict) else {}
        )
        incoming_primary_score = _multi_paper_primary_precision_score(primary_evidence)

        new_heading = str(heading_path or "").strip()
        cur_heading = str(entry.get("heading_path") or "").strip()
        if new_heading and (
            (not cur_heading)
            or (
                int(rank) >= 2
                and (
                    current_primary_score <= (0, 0, 0, 0, 0, 0)
                    or incoming_primary_score >= current_primary_score
                )
            )
        ):
            entry["heading_path"] = new_heading

        new_summary = str(summary or "").strip()
        cur_summary = str(entry.get("summary_line") or "").strip()
        if new_summary and (
            (not cur_summary)
            or (
                int(rank) >= 2
                and (
                    current_primary_score <= (0, 0, 0, 0, 0, 0)
                    or incoming_primary_score >= current_primary_score
                )
                and len(new_summary) >= max(24, len(cur_summary))
            )
        ):
            entry["summary_line"] = new_summary

        if isinstance(primary_evidence, dict) and primary_evidence:
            norm_primary = {k: v for k, v in dict(primary_evidence).items() if v not in ("", None, [], {})}
            if norm_primary:
                current_primary = (
                    dict(entry.get("primary_evidence") or {})
                    if isinstance(entry.get("primary_evidence"), dict)
                    else {}
                )
                current_primary_score = _multi_paper_primary_precision_score(current_primary)
                norm_primary_score = _multi_paper_primary_precision_score(norm_primary)
                if (not current_primary) or norm_primary_score >= current_primary_score:
                    entry["primary_evidence"] = norm_primary
                    if str(norm_primary.get("heading_path") or "").strip():
                        entry["heading_path"] = str(norm_primary.get("heading_path") or "").strip()
                    snippet = _single_line_summary(
                        str(norm_primary.get("highlight_snippet") or norm_primary.get("snippet") or "").strip(),
                        source_name=str(entry.get("source_name") or ""),
                    )
                    if snippet and (
                        (not str(entry.get("summary_line") or "").strip())
                        or norm_primary_score >= current_primary_score
                    ):
                        entry["summary_line"] = snippet

    for doc in list(seed_docs or []):
        if not isinstance(doc, dict):
            continue
        meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or "").strip()
        source_name = _source_name_from_path_like(source_path)
        raw_summary = str((((meta or {}).get("ref_show_snippets") or [None])[0]) or doc.get("text") or "").strip()
        heading_path_raw = (
            str((meta or {}).get("ref_best_heading_path") or "").strip()
            or str((meta or {}).get("heading_path") or "").strip()
            or str((meta or {}).get("top_heading") or "").strip()
        )
        heading_path, summary = _normalize_multi_paper_surface_seed(
            source_name=source_name,
            heading_path=heading_path_raw,
            raw_text=raw_summary,
        )
        primary_evidence = _normalize_multi_paper_contract_primary_evidence(
            source_path=source_path,
            source_name=source_name,
            heading_path=heading_path,
            raw_text=raw_summary,
            primary_evidence=None,
            selection_reason="pending_section_seed",
        )
        _merge_entry(
            source_path=source_path,
            source_name=source_name,
            heading_path=heading_path,
            summary=summary,
            primary_evidence=primary_evidence,
            rank=1,
        )

    for card in list(evidence_cards or []):
        if not isinstance(card, dict):
            continue
        primary = dict(card.get("primary_evidence") or {}) if isinstance(card.get("primary_evidence"), dict) else {}
        source_path = str(card.get("source_path") or primary.get("source_path") or "").strip()
        source_name = str(primary.get("source_name") or "").strip() or _source_name_from_path_like(source_path)
        raw_summary = _pick_multi_paper_card_raw_summary(
            prompt=prompt,
            card=card,
            primary_evidence=primary,
        )
        heading_path_raw = str(primary.get("heading_path") or "").strip() or str(card.get("heading") or "").strip()
        heading_path, summary = _normalize_multi_paper_surface_seed(
            source_name=source_name,
            heading_path=heading_path_raw,
            raw_text=raw_summary,
        )
        normalized_primary = _normalize_multi_paper_contract_primary_evidence(
            source_path=source_path,
            source_name=source_name,
            heading_path=heading_path,
            raw_text=raw_summary,
            primary_evidence=primary,
            selection_reason=str(primary.get("selection_reason") or "answer_hit_top").strip(),
        )
        _merge_entry(
            source_path=source_path,
            source_name=source_name,
            heading_path=heading_path,
            summary=summary,
            primary_evidence=normalized_primary,
            rank=3,
        )

    for hit in list(answer_hits or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or "").strip()
        source_name = _source_name_from_path_like(source_path)
        raw_summary = str((((meta or {}).get("ref_show_snippets") or [None])[0]) or hit.get("text") or "").strip()
        heading_path_raw = (
            str((meta or {}).get("ref_best_heading_path") or "").strip()
            or str((meta or {}).get("heading_path") or "").strip()
            or str((meta or {}).get("top_heading") or "").strip()
        )
        heading_path, summary = _normalize_multi_paper_surface_seed(
            source_name=source_name,
            heading_path=heading_path_raw,
            raw_text=raw_summary,
        )
        primary_evidence = _normalize_multi_paper_contract_primary_evidence(
            source_path=source_path,
            source_name=source_name,
            heading_path=heading_path,
            raw_text=raw_summary,
            primary_evidence=None,
            selection_reason="answer_hit_top",
        )
        _merge_entry(
            source_path=source_path,
            source_name=source_name,
            heading_path=heading_path,
            summary=summary,
            primary_evidence=primary_evidence,
            rank=2,
        )

    normalized_entries = [
        {
            k: v
            for k, v in dict(raw_entry or {}).items()
            if k not in {"_source_rank"} and v not in ("", None, [], {})
        }
        for raw_entry in entries
    ]
    return _filter_multi_paper_doc_list_contract(
        prompt=prompt,
        doc_list=normalized_entries,
    )



def _format_multi_paper_list_answer_v2(*, prompt: str, docs: list[dict]) -> str:
    rows = [dict(item) for item in list(docs or []) if isinstance(item, dict)]
    if not rows:
        return ""
    prefer_zh = bool(prompt_prefers_zh(prompt))
    topic = _extract_multi_paper_topic(prompt)
    paper_count = len(rows)
    if prefer_zh:
        intro = (
            f"\u6839\u636e\u547d\u4e2d\u7684\u5e93\u5185\u6587\u732e\uff0c\u4ee5\u4e0b {paper_count} \u7bc7\u6587\u7ae0\u4e0e\u201c{topic}\u201d\u76f4\u63a5\u76f8\u5173\uff1a"
            if topic
            else f"\u6839\u636e\u547d\u4e2d\u7684\u5e93\u5185\u6587\u732e\uff0c\u4ee5\u4e0b {paper_count} \u7bc7\u6587\u7ae0\u4e0e\u5f53\u524d\u95ee\u9898\u76f4\u63a5\u76f8\u5173\uff1a"
        )
        lines = [intro, ""]
        for idx, item in enumerate(rows, start=1):
            name = str(item.get("source_name") or _source_name_from_path_like(item.get("source_path") or "")).strip() or f"\u6587\u732e {idx}"
            heading = str(item.get("heading_path") or "").strip()
            summary = str(item.get("summary_line") or "").strip()
            match_note = _multi_paper_topic_match_note(
                prompt=prompt,
                match_kind=str(item.get("topic_match_kind") or ""),
            )
            lines.append(f"{idx}. **{name}**")
            if heading:
                lines.append(f"   - \u5b9a\u4f4d\uff1a{heading}")
            if summary:
                lines.append(f"   - \u4f9d\u636e\uff1a{summary}")
            if match_note:
                lines.append(f"   - \u76f8\u5173\u6027\uff1a{match_note}")
            lines.append("")
        return "\n".join(lines).strip()

    intro = (
        f"The following library paper directly relates to '{topic}':"
        if topic and paper_count == 1
        else f"The following library paper directly relates to the current query:"
        if paper_count == 1
        else f"The following {paper_count} library papers directly relate to '{topic}':"
        if topic
        else f"The following {paper_count} library papers directly relate to the current query:"
    )
    lines = [intro, ""]
    for idx, item in enumerate(rows, start=1):
        name = str(item.get("source_name") or _source_name_from_path_like(item.get("source_path") or "")).strip() or f"Paper {idx}"
        heading = str(item.get("heading_path") or "").strip()
        summary = str(item.get("summary_line") or "").strip()
        match_note = _multi_paper_topic_match_note(
            prompt=prompt,
            match_kind=str(item.get("topic_match_kind") or ""),
        )
        lines.append(f"{idx}. **{name}**")
        if heading:
            lines.append(f"   - Locate: {heading}")
        if summary:
            lines.append(f"   - Evidence: {summary}")
        if match_note:
            lines.append(f"   - Match: {match_note}")
        lines.append("")
    return "\n".join(lines).strip()


def _extract_multi_paper_topic(prompt: str) -> str:
    return _shared_extract_multi_paper_topic(prompt)


def _multi_paper_prompt_requires_explicit_focus_match(prompt: str) -> bool:
    return _shared_prompt_requires_reference_focus_match(prompt)


def _looks_generic_multi_paper_support_text(text: str, *, prompt: str) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    patterns = (
        "directly related to the current query",
        "directly relevant to the current query",
        "directly relevant to the current question",
        "directly responds to the user",
        "can serve as the current question",
        "matched section",
        "besides this paper, what other",
        "what other...",
        "\u4e0e\u5f53\u524d\u95ee\u9898\u76f4\u63a5\u76f8\u5173",
        "\u4e0e\u7528\u6237\u67e5\u8be2",
        "\u76f4\u63a5\u56de\u5e94\u7528\u6237",
        "\u5e93\u5185\u660e\u786e\u547d\u4e2d",
        "\u547d\u4e2d\u7ae0\u8282",
        "\u4e3b\u9898\u4e00\u81f4",
        "\u540c\u7c7b\u6280\u672f\u6587\u732e",
        "\u53ef\u4f5c\u4e3a\u5f53\u524d\u95ee\u9898",
    )
    if any(pattern in low for pattern in patterns):
        return True
    prompt_echo = str(prompt or "").strip().lower()
    if prompt_echo:
        prompt_echo = re.sub(r"\s+", " ", prompt_echo)
        if len(prompt_echo) >= 18 and prompt_echo[:32] in low:
            return True
    return False


def _multi_paper_focus_term_only_negated(term: str, surface: str) -> bool:
    token = str(term or "").strip().lower()
    normalized_surface = str(surface or "").strip().lower()
    if not token or not normalized_surface:
        return False
    escaped = re.escape(token)
    all_count = len(re.findall(rf"\b{escaped}\b", normalized_surface, flags=re.I))
    if all_count <= 0:
        return False
    neg_patterns = (
        rf"\b(?:without|not|no|lack(?:s|ing)?|avoid(?:s|ed|ing)?|rather than|instead of|does not mention|doesn't mention|does not discuss|doesn't discuss)\b[^.!?;\n]{{0,32}}\b{escaped}\b",
        rf"\b(?:\u672a\u63d0\u53ca|\u4e0d\u6d89\u53ca|\u6ca1\u6709|\u5e76\u672a|\u4e0d\u662f)\b[^\u3002\uff01\uff1f\uff1b\n]{{0,20}}{escaped}\b",
        rf"\b{escaped}\b[^.!?;\n]{{0,24}}\b(?:not|absent|omitted)\b",
    )
    negated_count = sum(
        len(re.findall(pattern, normalized_surface, flags=re.I))
        for pattern in neg_patterns
    )
    return negated_count >= all_count


def _prompt_targets_sci_topic(prompt: str) -> bool:
    return _shared_prompt_targets_sci_topic(prompt)


def _format_multi_paper_list_answer(*, prompt: str, docs: list[dict]) -> str:
    return _format_multi_paper_list_answer_v2(prompt=prompt, docs=docs)


def _select_minimum_paper_guide_ref_num(
    *,
    answer: str,
    support_resolution: list[dict] | None,
    candidate_refs_by_source: dict[str, list[int]] | None,
    retrieval_confidence_hint: dict[str, object] | None,
) -> int:
    inline_refs = _collect_inline_reference_numbers(answer, max_items=6)
    if inline_refs:
        return int(inline_refs[0])
    refs = _collect_low_confidence_candidate_refs(
        support_resolution=support_resolution,
        candidate_refs_by_source=candidate_refs_by_source,
        retrieval_confidence_hint=retrieval_confidence_hint,
        max_items=6,
    )
    return int(refs[0]) if refs else 0


def _select_minimum_paper_guide_sid(
    *,
    support_resolution: list[dict] | None,
    locked_citation_source: dict | None,
) -> str:
    for rec in list(support_resolution or []):
        if not isinstance(rec, dict):
            continue
        sid = str(rec.get("sid") or "").strip()
        if sid and _SID_RE.match(sid):
            return sid
    locked_sid = str((locked_citation_source or {}).get("sid") or "").strip()
    if locked_sid and _SID_RE.match(locked_sid):
        return locked_sid
    return ""


def _maybe_ensure_minimum_paper_guide_citation(
    answer: str,
    *,
    paper_guide_mode: bool,
    prompt_family: str = "",
    has_hits: bool,
    support_resolution: list[dict] | None = None,
    candidate_refs_by_source: dict[str, list[int]] | None = None,
    retrieval_confidence_hint: dict[str, object] | None = None,
    locked_citation_source: dict | None = None,
) -> str:
    text = str(answer or "").strip()
    family = str(prompt_family or "").strip().lower()
    if not text:
        return text
    if not paper_guide_mode or not has_hits:
        return text
    if family and family not in {"citation_lookup"}:
        return text
    if _has_structured_cite_marker(text):
        return text
    # Keep negative shells citation-free to avoid implying unsupported absence claims.
    if _PAPER_GUIDE_NEGATIVE_SHELL_RE.search(text):
        return text
    sid = _select_minimum_paper_guide_sid(
        support_resolution=support_resolution,
        locked_citation_source=locked_citation_source,
    )
    if not sid:
        return text
    ref_num = _select_minimum_paper_guide_ref_num(
        answer=text,
        support_resolution=support_resolution,
        candidate_refs_by_source=candidate_refs_by_source,
        retrieval_confidence_hint=retrieval_confidence_hint,
    )
    if ref_num <= 0:
        return text
    return f"{text} [[CITE:{sid}:{int(ref_num)}]]"


def _maybe_prepend_paper_guide_low_confidence_notice(
    answer: str,
    *,
    paper_guide_mode: bool,
    prompt_text: str,
    prompt_family: str,
    retrieval_confidence_hint: dict[str, object] | None,
    support_resolution: list[dict] | None = None,
    candidate_refs_by_source: dict[str, list[int]] | None = None,
) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    if not paper_guide_mode:
        return text
    hint = dict(retrieval_confidence_hint or {})
    if not hint:
        return text
    if not bool(hint.get("low_confidence")):
        return text
    try:
        enabled = bool(int(str(os.environ.get("KB_PAPER_GUIDE_LOW_CONF_NOTICE", "1") or "1")))
    except Exception:
        enabled = True
    if not enabled:
        return text
    lowered = text.lower()
    if ("low confidence" in lowered) or ("低置信" in text):
        return text
    reason = str(hint.get("low_confidence_reason") or hint.get("force_rescue_reason") or "").strip()
    if not reason:
        reason = "weak_evidence_alignment"
    reason_map_en = {
        "empty_hits": "no scoped evidence was retrieved",
        "target_miss": "the requested target section was not matched directly",
        "reference_only_hits": "retrieval mostly returned reference-like snippets",
        "weak_signal": "retrieval signal is weak for the requested claim",
        "strict_family_without_targeted_support": "strict question type lacks targeted support",
        "strict_family_weak_overlap": "strict question type has weak lexical overlap",
        "strict_family_sparse_hits": "strict question type has sparse evidence hits",
        "broad_family_weak_overlap": "broad summary question has weak evidence overlap",
    }
    reason_map_zh = {
        "empty_hits": "未检索到同文证据片段",
        "target_miss": "未直接命中你指定的目标段落",
        "reference_only_hits": "检索结果主要是参考文献样式片段",
        "weak_signal": "针对该问题的证据信号偏弱",
        "strict_family_without_targeted_support": "严格问题类型缺少定向证据支撑",
        "strict_family_weak_overlap": "严格问题类型与证据词重叠较弱",
        "strict_family_sparse_hits": "严格问题类型命中证据过少",
        "broad_family_weak_overlap": "概览类问题与证据重叠较弱",
    }
    family = str(prompt_family or "").strip().lower()
    if family in {"abstract"}:
        return text
    is_zh = _contains_cjk(prompt_text)
    if is_zh:
        reason_msg = reason_map_zh.get(reason, reason)
        notice = f"提示：当前回答基于低置信证据匹配（{reason_msg}）。建议点击“定位到原文证据”核对关键句。"
    else:
        reason_msg = reason_map_en.get(reason, reason.replace("_", " "))
        notice = (
            f"Note: this answer is based on lower-confidence evidence matching ({reason_msg}). "
            f"Please verify key claims via locate-to-source evidence."
        )
    candidate_refs = _collect_low_confidence_candidate_refs(
        support_resolution=support_resolution,
        candidate_refs_by_source=candidate_refs_by_source,
        retrieval_confidence_hint=hint,
        max_items=6,
    )
    if candidate_refs:
        refs_text = ", ".join(f"[{int(n)}]" for n in candidate_refs if int(n) > 0)
        if refs_text:
            if is_zh:
                notice += f" 候选参考文献：{refs_text}（供交叉核对）。"
            else:
                notice += f" Candidate refs for cross-check: {refs_text}."
    return f"{notice}\n\n{text}"


def _build_paper_guide_supplement_lines(*, prompt_family: str, prefer_zh: bool) -> list[str]:
    family = str(prompt_family or "").strip().lower()
    if prefer_zh:
        if family == "citation_lookup":
            return [
                "引用问题应以文内编号与参考文献列表为准，通用背景不能替代原始引用链。",
                "若仍不稳定，建议继续追问“具体术语 + 句子位置”以触发更窄范围定位。",
            ]
        if family in {"method", "reproduce"}:
            return [
                "方法理解通常要把“输入/输出、关键模块、训练设定、适用边界”分开核对。",
                "用于实验前，建议把本段补充与可定位原文逐条对照后再采用。",
            ]
        if family in {"equation", "figure_walkthrough", "box_only"}:
            return [
                "公式/图示解读常依赖上下文定义，单句解释可能遗漏符号约束与实验条件。",
                "若要用于结论，请优先以可定位的原文片段为准。",
            ]
        return [
            "以下内容用于帮助理解领域背景，不等同于论文原文已明确陈述。",
            "需要用于结论时，请以可定位的原文证据为准。",
        ]
    if family == "citation_lookup":
        return [
            "Reference questions should be decided by in-paper numbering and the reference list, not by generic background.",
            "If grounding is still weak, ask with exact terms plus sentence scope to trigger narrower locate matching.",
        ]
    if family in {"method", "reproduce"}:
        return [
            "Method understanding is more reliable when input/output, key modules, training setup, and failure boundaries are checked separately.",
            "Before applying this in experiments, map each supplemental point to a locate-able source sentence.",
        ]
    if family in {"equation", "figure_walkthrough", "box_only"}:
        return [
            "Equation/figure interpretation often depends on nearby definitions; a single sentence can miss constraints.",
            "Use locate-able paper evidence as the final authority for decisions.",
        ]
    return [
        "The notes below are general background to aid understanding, not explicit paper-verified claims.",
        "For final conclusions, prioritize locate-able source evidence.",
    ]


def _normalize_paper_guide_supplement_lines(
    raw_lines: object,
    *,
    max_items: int = 3,
) -> list[str]:
    if isinstance(raw_lines, (list, tuple)):
        text = "\n".join(str(item or "") for item in raw_lines)
    else:
        text = str(raw_lines or "")
    text = str(text or "").strip()
    if not text:
        return []

    text = re.sub(r"```(?:markdown|md|text)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = _PAPER_GUIDE_SUPPLEMENT_BLOCK_MARKER_RE.sub("", text)
    text = _PAPER_GUIDE_SUPPLEMENT_DISCLAIMER_RE.sub("", text)

    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        s = str(line or "").strip()
        if not s:
            continue
        s = re.sub(r"^\s*>\s*", "", s)
        s = re.sub(r"^\s*#{1,6}\s*", "", s)
        s = re.sub(r"^\s*\*\*(.*?)\*\*\s*$", r"\1", s)
        s = re.sub(r"^\s*\d+[.)]\s*", "- ", s)
        if re.match(r"^\s*[*-]\s+", s):
            s = "- " + re.sub(r"^\s*[*-]\s+", "", s).strip()
        s = _CITE_CANON_RE.sub("", s)
        s = re.sub(r"\[(\d{1,4})\]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        if (not s) or _PAPER_GUIDE_SUPPLEMENT_DISCLAIMER_RE.search(s):
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max(1, int(max_items or 3)):
            break

    if out:
        return out

    flat = re.sub(r"\s+", " ", text).strip()
    if not flat:
        return []
    flat = _CITE_CANON_RE.sub("", flat)
    flat = re.sub(r"\[(\d{1,4})\]", "", flat)
    flat = re.sub(r"\s+", " ", flat).strip()
    if not flat:
        return []
    return [flat[:280].rstrip()]


def _count_paper_guide_supportive_segments(support_resolution: list[dict] | None) -> int:
    count = 0
    for rec in list(support_resolution or []):
        if not isinstance(rec, dict):
            continue
        if any(
            str(rec.get(key) or "").strip()
            for key in ("locate_anchor", "evidence_quote", "segment_text", "anchor_text", "primary_block_id")
        ) or _as_positive_int(rec.get("resolved_ref_num")) > 0:
            count += 1
    return count


def _should_append_paper_guide_supplement(
    *,
    answer: str,
    prompt_family: str,
    retrieval_confidence_hint: dict[str, object] | None,
    support_resolution: list[dict] | None,
) -> bool:
    hint = dict(retrieval_confidence_hint or {})
    if bool(hint.get("low_confidence")):
        return True
    family = str(prompt_family or "").strip().lower()
    support_count = _count_paper_guide_supportive_segments(support_resolution)
    explanation_family = family in {
        "method",
        "reproduce",
        "equation",
        "figure_walkthrough",
        "overview",
        "compare",
        "strength_limits",
        "box_only",
        "discussion_only",
    }
    if explanation_family and support_count <= 1 and _PAPER_GUIDE_NEGATIVE_SHELL_RE.search(str(answer or "")):
        return True
    return False


def _maybe_append_paper_guide_supplement_block(
    answer: str,
    *,
    paper_guide_mode: bool,
    has_hits: bool,
    prompt_text: str,
    prompt_family: str,
    retrieval_confidence_hint: dict[str, object] | None,
    grounded_answer: str = "",
    support_resolution: list[dict] | None = None,
    build_paper_guide_supplement_lines=None,
) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    if not paper_guide_mode:
        return text
    if not has_hits:
        return text
    try:
        enabled = bool(int(str(os.environ.get("KB_PAPER_GUIDE_SUPPLEMENT_BLOCK", "1") or "1")))
    except Exception:
        enabled = True
    if not enabled:
        return text
    if _PAPER_GUIDE_SUPPLEMENT_BLOCK_MARKER_RE.search(text):
        return text
    if _PAPER_GUIDE_SUPPLEMENT_OPTOUT_RE.search(str(prompt_text or "")):
        return text
    if _STRUCTURED_ANSWER_SECTION_RE.search(text):
        return text
    if _CROSS_PAPER_QUERY_RE.search(str(prompt_text or "")):
        return text
    # When the grounded answer is explicitly a "not stated / does not specify" response,
    # avoid adding generic supplement blocks. Users asking for a concrete paper detail
    # are better served by a short negative answer plus actionable paper-only next steps.
    grounded_norm = normalize_inline_markdown(str(grounded_answer or text)).lower()
    if re.search(r"(?i)\b(?:does not specify|does not mention|not stated|cannot be determined)\b", grounded_norm):
        q = str(prompt_text or "").strip().lower()
        # Skip for "hardware/compute spec" questions where generic supplement is usually noise.
        # Apply regardless of family inference because intent classifiers can vary.
        if any(
            tok in q
            for tok in (
                "gpu",
                "cuda",
                "nvidia",
                "rtx",
                "a100",
                "v100",
                "3090",
                "4090",
                "hardware",
                "compute",
                "device",
            )
        ):
            return text
    hint = dict(retrieval_confidence_hint or {})
    if not _should_append_paper_guide_supplement(
        answer=str(grounded_answer or text),
        prompt_family=str(prompt_family or ""),
        retrieval_confidence_hint=hint,
        support_resolution=list(support_resolution or []),
    ):
        return text
    prefer_zh = _contains_cjk(prompt_text)
    lines: list[str] = []
    if callable(build_paper_guide_supplement_lines):
        try:
            lines = _normalize_paper_guide_supplement_lines(
                build_paper_guide_supplement_lines(
                    prompt_text=str(prompt_text or ""),
                    grounded_answer=str(grounded_answer or text),
                    prompt_family=str(prompt_family or ""),
                    prefer_zh=bool(prefer_zh),
                    retrieval_confidence_hint=dict(hint),
                    support_resolution=list(support_resolution or []),
                ),
                max_items=3,
            )
        except Exception:
            lines = []
    if not lines:
        lines = _build_paper_guide_supplement_lines(prompt_family=prompt_family, prefer_zh=prefer_zh)
    if not lines:
        return text
    if prefer_zh:
        header = "> 补充说明（通用知识，非检索片段内容 / Supplementary note (generic knowledge, non-retrieved content)）："
        disclaimer = "> 以下内容是 AI 基于通用知识的补充，不代表论文原文明确陈述。"
    else:
        header = "> Supplementary note (generic knowledge, non-retrieved content / 补充说明（通用知识，非检索片段内容）):"
        disclaimer = "> The notes below are AI supplemental context and are not explicit claims from the paper."
    block = [header, disclaimer]
    block.extend(f"> - {line}" for line in lines[:3] if str(line or "").strip())
    return f"{text}\n\n" + "\n".join(block).strip()


def _build_paper_guide_contract_snapshot(
    *,
    paper_guide_mode: bool,
    intent_model,
    answer_markdown: str,
    final_answer_markdown: str,
    evidence_cards: list[dict] | None,
    candidate_refs_by_source: dict[str, list[int]] | None,
    support_slots: list[dict] | None,
    support_resolution: list[dict] | None,
    needs_supplement: bool,
    citation_validation: dict | None,
    doc_list_contract: list[dict] | None = None,
    paper_guide_contracts_seed: dict | None = None,
) -> dict:
    seed = dict(paper_guide_contracts_seed or {})
    doc_list = [dict(item) for item in list(doc_list_contract or []) if isinstance(item, dict)]
    primary_evidence = _pick_shared_primary_evidence(
        paper_guide_contracts_seed=paper_guide_contracts_seed,
        evidence_cards=evidence_cards,
    )
    render_packet_seed = seed.get("render_packet") if isinstance(seed.get("render_packet"), dict) else {}
    if (not paper_guide_mode) and (not primary_evidence) and (not render_packet_seed) and (not doc_list):
        return {}

    snapshot = {"version": 1}
    if not paper_guide_mode:
        render_packet_model = _build_paper_guide_render_packet_model(
            answer_markdown=str(render_packet_seed.get("answer_markdown") or final_answer_markdown or "").strip(),
            notice=str(render_packet_seed.get("notice") or "").strip(),
            rendered_body=str(render_packet_seed.get("rendered_body") or "").strip(),
            rendered_content=str(render_packet_seed.get("rendered_content") or "").strip(),
            copy_markdown=str(render_packet_seed.get("copy_markdown") or "").strip(),
            copy_text=str(render_packet_seed.get("copy_text") or "").strip(),
            cite_details=list(render_packet_seed.get("cite_details") or []),
            citation_validation=(
                render_packet_seed.get("citation_validation")
                if isinstance(render_packet_seed.get("citation_validation"), dict)
                else citation_validation
            ),
            locate_target=render_packet_seed.get("locate_target") if isinstance(render_packet_seed.get("locate_target"), dict) else {},
            reader_open=render_packet_seed.get("reader_open") if isinstance(render_packet_seed.get("reader_open"), dict) else {},
            provenance_segments=list(render_packet_seed.get("provenance_segments") or []),
            primary_evidence=primary_evidence,
        )
        render_packet_dump = _paper_guide_model_dump(render_packet_model)
        if any(render_packet_dump.values()):
            snapshot["render_packet"] = render_packet_dump
        if primary_evidence:
            snapshot["primary_evidence"] = dict(primary_evidence)
        if doc_list:
            snapshot["doc_list"] = doc_list
        return {
            key: value
            for key, value in snapshot.items()
            if value not in (None, "", [], {})
        }

    pack_records = list(support_resolution or []) or list(support_slots or [])
    support_pack_model = _build_paper_guide_support_pack_model(
        family=str(getattr(intent_model, "family", "") or "").strip(),
        answer_markdown=str(answer_markdown or "").strip(),
        support_records=pack_records,
        needs_supplement=bool(needs_supplement),
    )
    grounding_trace = [
        _paper_guide_model_dump(_paper_guide_grounding_trace_segment_model_from_raw(item))
        for item in list(support_resolution or [])
        if isinstance(item, dict)
    ]
    snapshot = {
        "version": 1,
        "intent": _paper_guide_model_dump(intent_model),
        "support_pack": _paper_guide_model_dump(support_pack_model),
        "grounding_trace": grounding_trace,
    }
    retrieval_bundle = seed.get("retrieval_bundle") if isinstance(seed.get("retrieval_bundle"), dict) else {}
    if retrieval_bundle:
        snapshot["retrieval_bundle"] = dict(retrieval_bundle)
    else:
        prompt_context_seed = seed.get("prompt_context") if isinstance(seed.get("prompt_context"), dict) else {}
        retrieval_bundle_model = _build_paper_guide_retrieval_bundle_model(
            prompt_family=str(getattr(intent_model, "family", "") or "").strip(),
            target_scope=prompt_context_seed.get("target_scope") if isinstance(prompt_context_seed.get("target_scope"), dict) else {},
            evidence_cards=list(evidence_cards or []),
            candidate_refs_by_source=dict(candidate_refs_by_source or {}),
            direct_source_path=str(prompt_context_seed.get("direct_source_path") or "").strip(),
            focus_source_path=str(prompt_context_seed.get("focus_source_path") or "").strip(),
            bound_source_path=str(prompt_context_seed.get("bound_source_path") or "").strip(),
        )
        retrieval_bundle_dump = _paper_guide_model_dump(retrieval_bundle_model)
        if any(retrieval_bundle_dump.values()):
            snapshot["retrieval_bundle"] = retrieval_bundle_dump
    prompt_context = seed.get("prompt_context") if isinstance(seed.get("prompt_context"), dict) else {}
    if prompt_context:
        snapshot["prompt_context"] = dict(prompt_context)
    render_packet_model = _build_paper_guide_render_packet_model(
        answer_markdown=str(render_packet_seed.get("answer_markdown") or final_answer_markdown or "").strip(),
        notice=str(render_packet_seed.get("notice") or "").strip(),
        rendered_body=str(render_packet_seed.get("rendered_body") or "").strip(),
        rendered_content=str(render_packet_seed.get("rendered_content") or "").strip(),
        copy_markdown=str(render_packet_seed.get("copy_markdown") or "").strip(),
        copy_text=str(render_packet_seed.get("copy_text") or "").strip(),
        cite_details=list(render_packet_seed.get("cite_details") or []),
        citation_validation=(
            render_packet_seed.get("citation_validation")
            if isinstance(render_packet_seed.get("citation_validation"), dict)
            else citation_validation
        ),
        locate_target=render_packet_seed.get("locate_target") if isinstance(render_packet_seed.get("locate_target"), dict) else {},
        reader_open=render_packet_seed.get("reader_open") if isinstance(render_packet_seed.get("reader_open"), dict) else {},
        provenance_segments=list(render_packet_seed.get("provenance_segments") or []),
        primary_evidence=primary_evidence,
    )
    render_packet_dump = _paper_guide_model_dump(render_packet_model)
    if any(render_packet_dump.values()):
        snapshot["render_packet"] = render_packet_dump
    if primary_evidence:
        snapshot["primary_evidence"] = dict(primary_evidence)
    if doc_list:
        snapshot["doc_list"] = doc_list
    return {
        key: value
        for key, value in snapshot.items()
        if value not in (None, "", [], {})
    }


def _pick_shared_primary_evidence(
    *,
    paper_guide_contracts_seed: dict | None,
    evidence_cards: list[dict] | None,
) -> dict:
    def _primary_precision_score(primary: dict | None) -> tuple[int, int, int, int, int, int]:
        if not isinstance(primary, dict) or not primary:
            return (0, 0, 0, 0, 0, 0)
        reason = str(primary.get("selection_reason") or primary.get("selectionReason") or "").strip().lower()
        reason_rank = {
            "prompt_aligned": 6,
            "reader_open": 5,
            "strict_locate": 5,
            "provenance_segment": 5,
            "shared_refs_pack": 5,
            "pending_section_seed": 2,
            "shared_contract_seed": 1,
            "answer_hit_top": 0,
        }.get(reason, 3 if reason else 0)
        return (
            reason_rank,
            1 if str(primary.get("block_id") or primary.get("blockId") or "").strip() else 0,
            1 if str(primary.get("anchor_id") or primary.get("anchorId") or "").strip() else 0,
            1 if str(primary.get("heading_path") or primary.get("headingPath") or "").strip() else 0,
            1 if str(primary.get("snippet") or "").strip() else 0,
            1
            if str(primary.get("source_path") or primary.get("sourcePath") or primary.get("source_name") or primary.get("sourceName") or "").strip()
            else 0,
        )

    best: dict = {}
    best_score = (0, 0, 0, 0, 0, 0)

    seed = dict(paper_guide_contracts_seed or {})
    candidates: list[dict] = []
    primary = seed.get("primary_evidence")
    if isinstance(primary, dict) and primary:
        candidates.append(dict(primary))
    for card in list(evidence_cards or []):
        if not isinstance(card, dict):
            continue
        primary = card.get("primary_evidence")
        if isinstance(primary, dict) and primary:
            candidates.append(dict(primary))

    for candidate in candidates:
        score = _primary_precision_score(candidate)
        if (not best) or score > best_score:
            best = dict(candidate)
            best_score = score
    return best


def _finalize_generation_answer(
    partial: str,
    *,
    prompt: str,
    prompt_for_user: str,
    answer_hits: list[dict],
    db_dir: Path | None,
    locked_citation_source: dict | None,
    answer_intent: str,
    answer_depth: str,
    answer_output_mode: str,
    paper_guide_mode: bool,
    paper_guide_contract_enabled: bool,
    paper_guide_prompt_family: str,
    paper_guide_special_focus_block: str,
    paper_guide_focus_source_path: str,
    paper_guide_direct_source_path: str,
    paper_guide_bound_source_path: str,
    paper_guide_candidate_refs_by_source: dict[str, list[int]] | None,
    paper_guide_support_slots: list[dict] | None,
    paper_guide_evidence_cards: list[dict] | None,
    paper_guide_contracts_seed: dict | None = None,
    paper_guide_retrieval_confidence_hint: dict[str, object] | None = None,
    apply_paper_guide_answer_postprocess,
    maybe_append_library_figure_markdown,
    validate_structured_citations,
    build_paper_guide_supplement_lines=None,
) -> dict:
    resolved_paper_guide_intent = _resolve_paper_guide_intent(
        prompt_for_user or prompt,
        prompt_family=paper_guide_prompt_family,
    )
    effective_paper_guide_family = str(getattr(resolved_paper_guide_intent, "family", "") or "").strip()
    sanitize_paper_guide_family = effective_paper_guide_family or "overview"
    multi_paper_list_prompt = bool(prompt_explicitly_requests_multi_paper_list(prompt_for_user or prompt))
    multi_paper_doc_list = (
        _build_multi_paper_doc_list_contract(
            prompt=prompt or prompt_for_user,
            seed_docs=list((paper_guide_contracts_seed or {}).get("doc_list_seed") or []),
            answer_hits=list(answer_hits or []),
            evidence_cards=list(paper_guide_evidence_cards or []),
        )
        if multi_paper_list_prompt
        else []
    )
    answer = _normalize_math_markdown(
        _strip_model_ref_section(_sanitize_structured_cite_tokens(partial or ""))
    ).strip() or "(No text returned)"
    answer = _reconcile_kb_notice(answer, has_hits=bool(answer_hits))
    shared_primary_evidence = _pick_shared_primary_evidence(
        paper_guide_contracts_seed=dict(paper_guide_contracts_seed or {}),
        evidence_cards=list(paper_guide_evidence_cards or []),
    )
    if paper_guide_contract_enabled:
        answer = _apply_answer_contract_v1(
            answer,
            prompt=prompt,
            has_hits=bool(answer_hits),
            answer_hits=answer_hits,
            primary_evidence=shared_primary_evidence,
            intent=answer_intent,
            depth=answer_depth,
            output_mode=answer_output_mode,
        )
    answer = _enhance_kb_miss_fallback(
        answer,
        has_hits=bool(answer_hits),
        intent=answer_intent,
        depth=answer_depth,
        contract_enabled=bool(paper_guide_contract_enabled),
        output_mode=answer_output_mode,
    )
    answer, paper_guide_support_resolution = apply_paper_guide_answer_postprocess(
        answer,
        paper_guide_mode=paper_guide_mode,
        prompt=prompt,
        prompt_for_user=prompt_for_user,
        prompt_family=paper_guide_prompt_family,
        special_focus_block=paper_guide_special_focus_block,
        focus_source_path=paper_guide_focus_source_path,
        direct_source_path=paper_guide_direct_source_path,
        bound_source_path=paper_guide_bound_source_path,
        db_dir=db_dir,
        answer_hits=answer_hits,
        support_slots=list(paper_guide_support_slots or []),
        cards=list(paper_guide_evidence_cards or []),
        locked_citation_source=locked_citation_source,
    )
    answer = maybe_append_library_figure_markdown(
        answer,
        prompt=prompt,
        answer_hits=answer_hits,
        bound_source_path=paper_guide_bound_source_path,
    )
    answer, citation_validation = validate_structured_citations(
        answer,
        answer_hits=answer_hits,
        db_dir=db_dir,
        locked_source=locked_citation_source,
        paper_guide_mode=bool(paper_guide_mode),
        paper_guide_candidate_refs_by_source=dict(paper_guide_candidate_refs_by_source or {}),
        paper_guide_support_slots=list(paper_guide_support_slots or []),
        paper_guide_support_resolution=list(paper_guide_support_resolution or []),
    )
    # Citation validation may legitimately rewrite or inject structured cite markers for grounding,
    # but the final user-facing paper-guide answer still needs the same family-aware sanitization pass.
    if paper_guide_mode:
        answer = _sanitize_paper_guide_answer_for_user(
            answer,
            has_hits=bool(answer_hits),
            prompt=prompt_for_user or prompt,
            prompt_family=sanitize_paper_guide_family,
        )
        answer = _maybe_ensure_minimum_paper_guide_citation(
            answer,
            paper_guide_mode=bool(paper_guide_mode),
            prompt_family=sanitize_paper_guide_family,
            has_hits=bool(answer_hits),
            support_resolution=list(paper_guide_support_resolution or []),
            candidate_refs_by_source=dict(paper_guide_candidate_refs_by_source or {}),
            retrieval_confidence_hint=dict(paper_guide_retrieval_confidence_hint or {}),
            locked_citation_source=locked_citation_source,
        )
        # The minimum-citation helper may append structured markers for internal grounding.
        # Always re-sanitize for the final user-facing string so raw tokens never leak.
        answer = _sanitize_paper_guide_answer_for_user(
            answer,
            has_hits=bool(answer_hits),
            prompt=prompt_for_user or prompt,
            prompt_family=sanitize_paper_guide_family,
        )
    answer = _sanitize_internal_doc_label_blocks(answer)
    preserve_numeric_citations = _should_preserve_final_answer_numeric_citations(
        prompt=prompt_for_user or prompt,
        answer_output_mode=answer_output_mode,
        paper_guide_mode=bool(paper_guide_mode),
        prompt_family=sanitize_paper_guide_family,
    )
    answer = _strip_final_answer_citation_markers(
        answer,
        preserve_numeric_markers=preserve_numeric_citations,
    )
    if multi_paper_list_prompt and multi_paper_doc_list:
        formatted_multi_paper_answer = _format_multi_paper_list_answer_v2(
            prompt=prompt_for_user or prompt,
            docs=multi_paper_doc_list,
        )
        if formatted_multi_paper_answer:
            answer = formatted_multi_paper_answer
    grounded_answer = str(answer or "")
    answer = _maybe_prepend_paper_guide_low_confidence_notice(
        answer,
        paper_guide_mode=bool(paper_guide_mode),
        prompt_text=prompt_for_user or prompt,
        prompt_family=sanitize_paper_guide_family,
        retrieval_confidence_hint=dict(paper_guide_retrieval_confidence_hint or {}),
        support_resolution=list(paper_guide_support_resolution or []),
        candidate_refs_by_source=dict(paper_guide_candidate_refs_by_source or {}),
    )
    answer = _maybe_append_paper_guide_supplement_block(
        answer,
        paper_guide_mode=bool(paper_guide_mode),
        has_hits=bool(answer_hits),
        prompt_text=prompt_for_user or prompt,
        prompt_family=sanitize_paper_guide_family,
        retrieval_confidence_hint=dict(paper_guide_retrieval_confidence_hint or {}),
        grounded_answer=grounded_answer,
        support_resolution=list(paper_guide_support_resolution or []),
        build_paper_guide_supplement_lines=build_paper_guide_supplement_lines,
    )
    paper_guide_contracts = _build_paper_guide_contract_snapshot(
        paper_guide_mode=bool(paper_guide_mode),
        intent_model=resolved_paper_guide_intent,
        answer_markdown=grounded_answer,
        final_answer_markdown=answer,
        evidence_cards=list(paper_guide_evidence_cards or []),
        candidate_refs_by_source=dict(paper_guide_candidate_refs_by_source or {}),
        support_slots=list(paper_guide_support_slots or []),
        support_resolution=list(paper_guide_support_resolution or []),
        needs_supplement=bool(_PAPER_GUIDE_SUPPLEMENT_BLOCK_MARKER_RE.search(answer)),
        citation_validation=dict(citation_validation or {}),
        doc_list_contract=list(multi_paper_doc_list or []),
        paper_guide_contracts_seed=dict(paper_guide_contracts_seed or {}),
    )
    answer_quality = _build_answer_quality_probe(
        answer,
        has_hits=bool(answer_hits),
        contract_enabled=bool(paper_guide_contract_enabled),
        intent=answer_intent,
        depth=answer_depth,
        output_mode=answer_output_mode,
        paper_guide_mode=bool(paper_guide_mode),
        prompt_family=sanitize_paper_guide_family,
    )
    retrieval_confidence = dict(paper_guide_retrieval_confidence_hint or {})
    if bool(retrieval_confidence.get("low_confidence")):
        refs_for_notice = _collect_low_confidence_candidate_refs(
            support_resolution=list(paper_guide_support_resolution or []),
            candidate_refs_by_source=dict(paper_guide_candidate_refs_by_source or {}),
            retrieval_confidence_hint=retrieval_confidence,
            max_items=6,
        )
        if refs_for_notice:
            retrieval_confidence["candidate_refs_for_notice"] = list(refs_for_notice)
    answer_quality["retrieval_confidence"] = retrieval_confidence
    return {
        "answer": answer,
        "paper_guide_support_resolution": list(paper_guide_support_resolution or []),
        "paper_guide_contracts": paper_guide_contracts,
        "citation_validation": citation_validation,
        "answer_quality": answer_quality,
    }
