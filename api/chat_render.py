from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from kb import task_runtime
from kb.paper_guide_contracts import (
    _build_paper_guide_render_packet_model,
    _paper_guide_model_dump,
)
from kb.paper_guide.grounder import (
    _build_paper_guide_segment_locate_target,
    _build_paper_guide_segment_reader_open,
    _resolve_paper_guide_panel_clause_snippet,
)
from kb.paper_guide_provenance import (
    _annotate_provenance_hit_levels,
    _backfill_segment_primary_blocks_from_anchor_lookup,
    _build_anchor_provenance_lookup,
    _canonicalize_support_segment_heading,
)
from kb.paper_guide_structured_index_runtime import (
    load_paper_guide_anchor_index,
    load_paper_guide_equation_index,
    load_paper_guide_figure_index,
)
from kb.citation_meta import extract_first_doi
from kb.config import load_settings
from kb.reference_index import extract_references_map_from_md, load_reference_index, resolve_reference_entry
from ui.chat_widgets import _md_to_plain_text, _normalize_copy_citation_links, _normalize_math_markdown
from ui.refs_renderer import (
    _annotate_equation_tags_with_sources,
    _annotate_inpaper_citations_with_hover_meta,
    _normalize_reference_for_popup,
    _source_cite_id,
)

_STRUCT_CITE_RE = re.compile(r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\]\]", re.IGNORECASE)
_STRUCT_CITE_SINGLE_RE = re.compile(r"(?<!\[)\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})(?:\s*:\s*(\d{1,4}))?\s*\](?!\])", re.IGNORECASE)
_STRUCT_CITE_SID_ONLY_RE = re.compile(r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*\]\]", re.IGNORECASE)
_STRUCT_CITE_GARBAGE_RE = re.compile(r"\[\[?\s*CITE\s*:[^\]\n]*\]?\]", re.IGNORECASE)
_STRUCT_SID_INLINE_RE = re.compile(r"\[\s*SID\s*:\s*[A-Za-z0-9_-]{4,24}\s*\]", re.IGNORECASE)
_STRUCT_SID_HEADER_LINE_RE = re.compile(
    r"(?im)^\s*\[\d{1,3}\]\s*\[\s*SID\s*:\s*[A-Za-z0-9_-]{4,24}\s*\][^\n]*\n?",
    re.IGNORECASE,
)
_VISIBLE_NUMERIC_CITE_RE = re.compile(r"\[\d{1,4}(?:\s*(?:-|–|—|,)\s*\d{1,4})*\]")
_EQ_SOURCE_NOTE_RE = re.compile(
    r"\*\s*.*?\((\d{1,4})\).*?`([^`]+)`.*?(?:Open/Page)?[^\n]*\*",
    re.IGNORECASE,
)
_REF_MAP_CACHE: dict[str, dict[int, str]] = {}
_RENDER_CACHE_SCHEMA_VERSION = 3


def _env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.environ.get(str(name or "").strip(), default) or "").strip()
    if not raw:
        return False
    if raw.lower() in {"1", "true", "yes", "on"}:
        return True
    try:
        return bool(int(raw))
    except Exception:
        return False


def _extract_box_number_for_display(seg: dict) -> int:
    try:
        box_num = int((seg or {}).get("support_slot_box_number") or 0)
    except Exception:
        box_num = 0
    if box_num > 0:
        return box_num
    heading = str((seg or {}).get("primary_heading_path") or "").strip()
    m = re.search(r"(?i)\bbox\s*(\d+)\b", heading)
    if m:
        try:
            return int(m.group(1) or 0)
        except Exception:
            return 0
    text = str((seg or {}).get("text") or "").strip()
    m = re.search(r"(?i)^\s*from\s+box\s*(\d+)\b", text)
    if m:
        try:
            return int(m.group(1) or 0)
        except Exception:
            return 0
    return 0


def _propagate_box_scope_for_display(segments: list[dict]) -> list[dict]:
    out = [dict(seg) if isinstance(seg, dict) else seg for seg in list(segments or [])]
    visible_direct_indices = [
        idx
        for idx, seg in enumerate(out)
        if isinstance(seg, dict)
        and str(seg.get("evidence_mode") or "").strip().lower() == "direct"
        and str(seg.get("locate_policy") or "").strip().lower() != "hidden"
    ]
    if not visible_direct_indices:
        return out
    explicit_boxes: dict[int, int] = {}
    for idx in visible_direct_indices:
        seg = out[idx]
        box_num = _extract_box_number_for_display(seg)
        if box_num <= 0:
            continue
        explicit_boxes[idx] = box_num
        seg["primary_heading_path"] = f"Box {int(box_num)}"
        seg["support_slot_box_number"] = int(box_num)
    if not explicit_boxes:
        return out
    for pos, idx in enumerate(visible_direct_indices):
        if idx in explicit_boxes:
            continue
        seg = out[idx]
        heading = str(seg.get("primary_heading_path") or "").strip()
        if re.search(r"(?i)\bfig(?:ure)?\b", heading):
            continue
        prev_box = 0
        next_box = 0
        for prev_idx in reversed(visible_direct_indices[:pos]):
            prev_box = int(explicit_boxes.get(prev_idx) or 0)
            if prev_box > 0:
                break
        for next_idx in visible_direct_indices[pos + 1 :]:
            next_box = int(explicit_boxes.get(next_idx) or 0)
            if next_box > 0:
                break
        if prev_box > 0 and prev_box == next_box:
            seg["primary_heading_path"] = f"Box {int(prev_box)}"
            seg["support_slot_box_number"] = int(prev_box)
    return out


@lru_cache(maxsize=1)
def _load_reference_index_cached() -> dict:
    try:
        return load_reference_index(load_settings().db_dir)
    except Exception:
        return {}


def _split_kb_miss_notice(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    s = text.lstrip()
    prefix = "未命中知识库片段"
    if not s.startswith(prefix):
        return "", text

    nl = s.find("\n")
    if nl != -1:
        return s[:nl].strip(), s[nl + 1 :].lstrip("\n")

    for sep in ("。", ".", "！", "!", "？", "?", ";", "；"):
        idx = s.find(sep)
        if 0 <= idx <= 80:
            return s[: idx + 1].strip(), s[idx + 1 :].lstrip()

    return prefix, s[len(prefix) :].lstrip("：: \t")


def _normalize_equation_source_notes(md: str) -> str:
    def _clean_label(raw: str) -> str:
        text = str(raw or "").strip().replace("\\", "/")
        if not text:
            return ""
        if "/" in text:
            text = text.rsplit("/", 1)[-1].strip()
        pdf_names = re.findall(r"([A-Za-z][A-Za-z0-9 _().,+:-]{3,220}\.pdf)", text, re.IGNORECASE)
        if pdf_names:
            return str(pdf_names[-1] or "").strip()
        return text.strip("`*#：:;；，,()（）[] ")

    def _replace(m: re.Match[str]) -> str:
        eq_num = str(m.group(1) or "").strip()
        label = _clean_label(str(m.group(2) or ""))
        if not eq_num or not label:
            return m.group(0)
        return f"*（式({eq_num}) 对应命中的库内文献：`{label}`）*"

    out = _EQ_SOURCE_NOTE_RE.sub(_replace, str(md or ""))
    # Fallback for legacy/mojibake variants that still contain "Open/Page".
    out = re.sub(
        r"(?im)^\*\s*.*?\((\d{1,4})\).*?`([^`]+)`.*?Open/Page[^\n]*$",
        _replace,
        out,
    )
    lines: list[str] = []
    for ln in str(out).splitlines():
        l = str(ln or "")
        ll = l.lower()
        if l.lstrip().startswith("*"):
            m_eq = re.search(r"\((\d{1,4})\)", l)
            m_label = re.search(r"([^\n`]{0,260}\.pdf)", l, re.IGNORECASE)
            if m_eq and m_label and (
                ("open/page" in ll)
                or ("参考定位" in l)
                or ("鍙傝€冨畾浣" in l)
                or ("#1" in l)
            ):
                label = _clean_label(m_label.group(1))
                if label:
                    l = f"*（式({m_eq.group(1)}) 对应命中的库内文献：`{label}`）*"
        lines.append(l)
    return "\n".join(lines).replace("Open/Page", "")


def _strip_structured_cite_tokens_for_display(md: str) -> str:
    s = str(md or "")
    if not s:
        return s
    out = s
    if "CITE" in s.upper():
        out = _STRUCT_CITE_RE.sub("", out)
        out = _STRUCT_CITE_SINGLE_RE.sub("", out)
        out = _STRUCT_CITE_SID_ONLY_RE.sub("", out)
        out = _STRUCT_CITE_GARBAGE_RE.sub("", out)
    out = _STRUCT_SID_HEADER_LINE_RE.sub("", out)
    out = _STRUCT_SID_INLINE_RE.sub("", out)
    return out


def _normalize_chat_markdown_for_display(md: str) -> str:
    return _normalize_math_markdown(_strip_structured_cite_tokens_for_display(md))


_FREEFORM_NUMERIC_CITE_RE = re.compile(
    r"(?<![!\\])\[(\d{1,4}(?:\s*(?:-|–|—|,)\s*\d{1,4})*)\](?!\()"
)


def _message_intent_family(rec: dict | None) -> str:
    if not isinstance(rec, dict):
        return ""
    meta = dict(rec.get("meta") or {}) if isinstance(rec.get("meta"), dict) else {}
    contracts = dict(meta.get("paper_guide_contracts") or {}) if isinstance(meta.get("paper_guide_contracts"), dict) else {}
    intent = dict(contracts.get("intent") or {}) if isinstance(contracts.get("intent"), dict) else {}
    return str(intent.get("family") or "").strip().lower()


def _message_answer_prompt_family(rec: dict | None) -> str:
    if not isinstance(rec, dict):
        return ""
    meta = dict(rec.get("meta") or {}) if isinstance(rec.get("meta"), dict) else {}
    answer_quality = dict(meta.get("answer_quality") or {}) if isinstance(meta.get("answer_quality"), dict) else {}
    return str(answer_quality.get("prompt_family") or "").strip().lower()


def _message_answer_output_mode(rec: dict | None) -> str:
    if not isinstance(rec, dict):
        return ""
    meta = dict(rec.get("meta") or {}) if isinstance(rec.get("meta"), dict) else {}
    answer_quality = dict(meta.get("answer_quality") or {}) if isinstance(meta.get("answer_quality"), dict) else {}
    return str(answer_quality.get("output_mode") or "").strip().lower()


def _should_link_inpaper_citations_for_message(*, rec: dict | None, content: str) -> bool:
    raw = str(content or "")
    if not raw:
        return False
    if _message_intent_family(rec) == "citation_lookup":
        return True
    if _message_answer_prompt_family(rec) == "citation_lookup":
        return True
    return "citation" in _message_answer_output_mode(rec)


def _strip_freeform_numeric_citation_markers(md: str) -> str:
    text = str(md or "")
    if (not text) or ("[" not in text):
        return text
    out = _FREEFORM_NUMERIC_CITE_RE.sub("", text)
    out = re.sub(r"[ \t]+([,.;:!?])", r"\1", out)
    out = re.sub(r"(?m)[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    return out.strip()


def _should_retry_structured_cite_fallback(*, raw_body: str, rendered_body: str, cite_details: list[dict]) -> bool:
    if cite_details:
        return False
    raw = str(raw_body or "")
    rendered = str(rendered_body or "")
    had_structured = bool(
        _STRUCT_CITE_RE.search(raw)
        or _STRUCT_CITE_SINGLE_RE.search(raw)
        or _STRUCT_CITE_SID_ONLY_RE.search(raw)
    )
    if not had_structured:
        return False
    # If the primary annotator already preserved visible numeric markers as a
    # safety downgrade, keep them and avoid re-linking.
    if _VISIBLE_NUMERIC_CITE_RE.search(rendered):
        return False
    return True


def _build_render_texts(*, rendered_full: str, rendered_body: str, notice: str, cite_details: list[dict]) -> tuple[str, str, str, str]:
    rendered_content = _normalize_chat_markdown_for_display(rendered_full)
    body_norm = _normalize_chat_markdown_for_display(rendered_body) if rendered_body else ""
    if (not body_norm) and (not notice):
        body_norm = rendered_content
    copy_markdown = _normalize_copy_citation_links(rendered_content, cite_details)
    copy_text = _md_to_plain_text(copy_markdown)
    return rendered_content, body_norm, copy_markdown, copy_text


def _stable_json_hash(payload: object) -> str:
    try:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        raw = repr(payload)
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()


def _build_message_render_cache_key(
    *,
    conv_id: str,
    msg_id: int,
    role: str,
    content: str,
    refs_user_msg_id: int,
    ref_pack: dict | None,
    provenance: dict | None,
) -> str:
    base = {
        "schema": _RENDER_CACHE_SCHEMA_VERSION,
        "conv_id": str(conv_id or ""),
        "msg_id": int(msg_id or 0),
        "role": str(role or ""),
        "content": str(content or ""),
        "refs_user_msg_id": int(refs_user_msg_id or 0),
        "ref_sig": _stable_json_hash(ref_pack or {}),
        "provenance_sig": _stable_json_hash(provenance or {}),
    }
    return _stable_json_hash(base)


def _extract_render_cache(meta: dict | None, *, expected_key: str) -> dict | None:
    if not isinstance(meta, dict):
        return None
    cache = meta.get("render_cache")
    if not isinstance(cache, dict):
        return None
    if int(cache.get("schema") or 0) != _RENDER_CACHE_SCHEMA_VERSION:
        return None
    if str(cache.get("cache_key") or "").strip() != str(expected_key or "").strip():
        return None
    cite_details = cache.get("cite_details")
    if not isinstance(cite_details, list):
        cite_details = []
    render_packet = cache.get("render_packet")
    if not isinstance(render_packet, dict):
        render_packet = {}
    return {
        "notice": str(cache.get("notice") or ""),
        "rendered_body": str(cache.get("rendered_body") or ""),
        "rendered_content": str(cache.get("rendered_content") or ""),
        "copy_markdown": str(cache.get("copy_markdown") or ""),
        "copy_text": str(cache.get("copy_text") or ""),
        "cite_details": [dict(item) for item in cite_details if isinstance(item, dict)],
        "refs_user_msg_id": int(cache.get("refs_user_msg_id") or 0),
        "render_packet": dict(render_packet),
    }


def _build_render_cache_payload(
    *,
    cache_key: str,
    notice: str,
    rendered_body: str,
    rendered_content: str,
    copy_markdown: str,
    copy_text: str,
    cite_details: list[dict],
    refs_user_msg_id: int,
    render_packet: dict | None = None,
) -> dict:
    return {
        "schema": _RENDER_CACHE_SCHEMA_VERSION,
        "cache_key": str(cache_key or ""),
        "notice": str(notice or ""),
        "rendered_body": str(rendered_body or ""),
        "rendered_content": str(rendered_content or ""),
        "copy_markdown": str(copy_markdown or ""),
        "copy_text": str(copy_text or ""),
        "cite_details": [dict(item) for item in (cite_details or []) if isinstance(item, dict)],
        "refs_user_msg_id": int(refs_user_msg_id or 0),
        "render_packet": dict(render_packet or {}) if isinstance(render_packet, dict) else {},
    }


def _merge_render_packet_contract_meta(
    *,
    rec: dict,
    msg_id: int,
    enriched_provenance: dict | None,
    chat_store=None,
) -> None:
    meta = dict(rec.get("meta") or {}) if isinstance(rec.get("meta"), dict) else {}
    contracts = dict(meta.get("paper_guide_contracts") or {}) if isinstance(meta.get("paper_guide_contracts"), dict) else {}
    if not contracts:
        return
    existing_packet = dict(contracts.get("render_packet") or {}) if isinstance(contracts.get("render_packet"), dict) else {}
    existing_cite_details = [
        dict(item)
        for item in list(existing_packet.get("cite_details") or [])
        if isinstance(item, dict)
    ]
    current_cite_details = [
        dict(item)
        for item in list(rec.get("cite_details") or [])
        if isinstance(item, dict)
    ]
    allow_inpaper_citation_linking = _should_link_inpaper_citations_for_message(
        rec=rec,
        content=str(rec.get("content") or ""),
    )
    preserve_existing_render = bool(allow_inpaper_citation_linking and existing_cite_details and (not current_cite_details))
    rendered_body = (
        str(existing_packet.get("rendered_body") or "").strip()
        if preserve_existing_render
        else str(rec.get("rendered_body") or "").strip()
    )
    rendered_content = (
        str(existing_packet.get("rendered_content") or "").strip()
        if preserve_existing_render
        else str(rec.get("rendered_content") or "").strip()
    )
    copy_markdown = (
        str(existing_packet.get("copy_markdown") or "").strip()
        if preserve_existing_render
        else str(rec.get("copy_markdown") or "").strip()
    )
    copy_text = (
        str(existing_packet.get("copy_text") or "").strip()
        if preserve_existing_render
        else str(rec.get("copy_text") or "").strip()
    )
    existing_notice = str(existing_packet.get("notice") or "").strip()
    current_notice = str(rec.get("notice") or "").strip()
    provenance_segments = list((enriched_provenance or {}).get("segments") or [])
    provenance_primary_evidence = (
        dict((enriched_provenance or {}).get("primary_evidence") or {})
        if isinstance((enriched_provenance or {}).get("primary_evidence"), dict)
        else {}
    )
    if not provenance_primary_evidence:
        provenance_primary_evidence = (
            dict(existing_packet.get("primary_evidence") or {})
            if isinstance(existing_packet.get("primary_evidence"), dict)
            else {}
        )
    if not provenance_primary_evidence:
        provenance_primary_evidence = (
            dict(contracts.get("primary_evidence") or {})
            if isinstance(contracts.get("primary_evidence"), dict)
            else {}
        )
    has_current_locate_identity = any(
        isinstance(item, dict)
        and (
            isinstance(item.get("locate_target"), dict)
            or isinstance(item.get("reader_open"), dict)
        )
        for item in provenance_segments
    )
    # When we preserve existing cite details/rendered output (because the current
    # render degraded), do not accidentally drop a newly-detected notice (e.g.
    # KB-miss) just because the existing packet had no notice.
    notice = existing_notice if (preserve_existing_render and existing_notice) else current_notice
    render_packet_model = _build_paper_guide_render_packet_model(
        answer_markdown=str(rec.get("content") or "").strip(),
        notice=notice,
        rendered_body=rendered_body,
        rendered_content=rendered_content,
        copy_markdown=copy_markdown,
        copy_text=copy_text,
        cite_details=existing_cite_details if preserve_existing_render else current_cite_details,
        citation_validation=(
            existing_packet.get("citation_validation")
            if isinstance(existing_packet.get("citation_validation"), dict)
            else {}
        ),
        locate_target=(
            existing_packet.get("locate_target")
            if ((not has_current_locate_identity) and isinstance(existing_packet.get("locate_target"), dict))
            else {}
        ),
        reader_open=(
            existing_packet.get("reader_open")
            if ((not has_current_locate_identity) and isinstance(existing_packet.get("reader_open"), dict))
            else {}
        ),
        provenance_segments=provenance_segments,
        primary_evidence=provenance_primary_evidence,
    )
    render_packet = _paper_guide_model_dump(render_packet_model)
    if existing_packet == render_packet:
        rec["meta"] = meta
        return
    contracts["render_packet"] = render_packet
    meta["paper_guide_contracts"] = contracts
    rec["meta"] = meta
    if chat_store is not None and msg_id > 0:
        try:
            chat_store.merge_message_meta(msg_id, {"paper_guide_contracts": contracts})
        except Exception:
            pass


def _project_render_packet_compat_fields(rec: dict) -> None:
    meta = dict(rec.get("meta") or {}) if isinstance(rec.get("meta"), dict) else {}
    contracts = dict(meta.get("paper_guide_contracts") or {}) if isinstance(meta.get("paper_guide_contracts"), dict) else {}
    packet = dict(contracts.get("render_packet") or {}) if isinstance(contracts.get("render_packet"), dict) else {}
    if not packet:
        return
    rec["notice"] = str(packet.get("notice") or "")
    rec["rendered_body"] = str(packet.get("rendered_body") or "")
    rec["rendered_content"] = str(packet.get("rendered_content") or "")
    rec["copy_markdown"] = str(packet.get("copy_markdown") or "")
    rec["copy_text"] = str(packet.get("copy_text") or "")
    rec["cite_details"] = [
        dict(item)
        for item in list(packet.get("cite_details") or [])
        if isinstance(item, dict)
    ]
    rec["meta"] = meta


def _maybe_strip_legacy_render_fields(rec: dict, *, enabled: bool) -> None:
    if not enabled and not _env_flag("KB_CHAT_RENDER_PACKET_ONLY", "0"):
        return
    # Keep core identity fields; strip legacy render projections from response payload.
    for key in (
        "notice",
        "rendered_body",
        "rendered_content",
        "copy_markdown",
        "copy_text",
        "cite_details",
    ):
        rec.pop(key, None)


def _restore_render_packet_contract_from_cache(rec: dict, cached: dict | None) -> None:
    if not isinstance(cached, dict):
        return
    render_packet = cached.get("render_packet")
    if not isinstance(render_packet, dict) or not render_packet:
        return
    meta = dict(rec.get("meta") or {}) if isinstance(rec.get("meta"), dict) else {}
    contracts = dict(meta.get("paper_guide_contracts") or {}) if isinstance(meta.get("paper_guide_contracts"), dict) else {}
    contracts["render_packet"] = dict(render_packet)
    meta["paper_guide_contracts"] = contracts
    rec["meta"] = meta


def _reader_open_candidate_key(candidate: dict | None) -> str:
    cand = dict(candidate or {})
    return "::".join(
        [
            str(cand.get("blockId") or "").strip().lower(),
            str(cand.get("anchorId") or "").strip().lower(),
            str(cand.get("anchorKind") or "").strip().lower(),
            str(cand.get("anchorNumber") or "").strip().lower(),
            str(cand.get("headingPath") or "").strip().lower(),
            str(cand.get("highlightSnippet") or "").strip().lower()[:180],
            str(cand.get("snippet") or "").strip().lower()[:180],
        ]
    )


def _build_reader_open_alternative_candidates(
    seg: dict,
    *,
    block_lookup: dict[str, dict],
    locate_target: dict,
    anchor_number: int,
) -> list[dict]:
    primary_block_id = str(seg.get("primary_block_id") or locate_target.get("blockId") or "").strip()
    primary_anchor_id = str(seg.get("primary_anchor_id") or locate_target.get("anchorId") or "").strip()
    block_id_order: list[str] = []
    for raw_block_id in (
        [primary_block_id]
        + list(seg.get("evidence_block_ids") or [])
        + list(seg.get("support_block_ids") or [])
        + list(seg.get("related_block_ids") or [])
    ):
        block_id = str(raw_block_id or "").strip()
        if block_id:
            block_id_order.append(block_id)
    if not block_id_order:
        return []
    candidates: list[dict] = []
    seen: set[str] = set()
    primary_key = ""
    for block_id in block_id_order[:8]:
        block = block_lookup.get(block_id)
        if not isinstance(block, dict):
            continue
        block_text = str(block.get("text") or "").strip()
        heading_path = str(block.get("heading_path") or "").strip()
        anchor_id = str(block.get("anchor_id") or "").strip()
        block_kind = str(block.get("kind") or "").strip().lower()
        is_primary = bool(primary_block_id and block_id == primary_block_id) or bool(primary_anchor_id and anchor_id and anchor_id == primary_anchor_id)
        anchor_kind = (
            str(locate_target.get("anchorKind") or seg.get("anchor_kind") or "").strip()
            if is_primary
            else ("equation" if block_kind == "equation" else "figure" if block_kind == "figure" else block_kind)
        )
        try:
            block_number = int(block.get("number") or 0)
        except Exception:
            block_number = 0
        candidate_anchor_number = block_number if block_number > 0 else (int(anchor_number or 0) if is_primary else 0)
        snippet = (
            str(locate_target.get("snippet") or "").strip()
            if is_primary
            else block_text
        ) or block_text or heading_path or str(seg.get("text") or "").strip()
        highlight_snippet = (
            str(locate_target.get("highlightSnippet") or "").strip()
            if is_primary
            else block_text
        ) or snippet
        candidate = {
            "headingPath": heading_path or None,
            "snippet": snippet or None,
            "highlightSnippet": highlight_snippet or None,
            "blockId": block_id or None,
            "anchorId": anchor_id or None,
            "anchorKind": anchor_kind or None,
            "anchorNumber": candidate_anchor_number or None,
        }
        key = _reader_open_candidate_key(candidate)
        if (not key) or (key in seen):
            continue
        seen.add(key)
        candidates.append(candidate)
        if is_primary and not primary_key:
            primary_key = key
    if not candidates:
        return []
    return [
        candidate
        for candidate in candidates
        if _reader_open_candidate_key(candidate) != primary_key
    ][:4]


def _enrich_provenance_segments_for_display(
    provenance: dict | None,
    hits: list[dict],
    *,
    anchor_ns: str,
) -> dict | None:
    if not isinstance(provenance, dict):
        return provenance
    block_map_raw = provenance.get("block_map")
    block_map = dict(block_map_raw) if isinstance(block_map_raw, dict) else {}
    lookup: dict[str, dict] = {
        str(block_id): dict(block)
        for block_id, block in block_map.items()
        if str(block_id or "").strip() and isinstance(block, dict)
    }
    md_path_raw = str(provenance.get("md_path") or "").strip()
    anchor_lookup_by_anchor_id: dict[str, dict] = {}
    equation_index_rows: list[dict] = []
    figure_index_rows: list[dict] = []
    if md_path_raw:
        try:
            for block in task_runtime.load_source_blocks(Path(md_path_raw)):
                if not isinstance(block, dict):
                    continue
                block_id = str(block.get("block_id") or "").strip()
                if not block_id:
                    continue
                lookup[block_id] = dict(block)
        except Exception:
            pass
        try:
            anchor_index_rows = load_paper_guide_anchor_index(Path(md_path_raw))
        except Exception:
            anchor_index_rows = []
        try:
            anchor_block_lookup, anchor_lookup_by_anchor_id = _build_anchor_provenance_lookup(anchor_index_rows)
        except Exception:
            anchor_block_lookup, anchor_lookup_by_anchor_id = {}, {}
        for block_id, block in dict(anchor_block_lookup or {}).items():
            block_id_str = str(block_id or "").strip()
            if block_id_str and isinstance(block, dict):
                lookup[block_id_str] = dict(block)
        try:
            equation_index_rows = load_paper_guide_equation_index(Path(md_path_raw))
        except Exception:
            equation_index_rows = []
        try:
            figure_index_rows = load_paper_guide_figure_index(Path(md_path_raw))
        except Exception:
            figure_index_rows = []
    if lookup:
        try:
            hardened_segments = task_runtime._apply_provenance_required_coverage_contract(
                provenance.get("segments"),
                block_lookup=lookup,
                equation_index_rows=equation_index_rows,
                figure_index_rows=figure_index_rows,
            )
            hardened_segments = _backfill_segment_primary_blocks_from_anchor_lookup(
                hardened_segments,
                block_lookup=lookup,
                anchor_lookup_by_anchor_id=anchor_lookup_by_anchor_id,
            )
            hardened_segments, contract_meta = task_runtime._apply_provenance_strict_identity_contract(hardened_segments)
            provenance = dict(provenance)
            provenance["segments"] = hardened_segments
            referenced_block_ids: set[str] = set()
            for seg in hardened_segments:
                if not isinstance(seg, dict):
                    continue
                primary_block_id = str(seg.get("primary_block_id") or "").strip()
                if primary_block_id:
                    referenced_block_ids.add(primary_block_id)
                for block_id_raw in list(seg.get("support_block_ids") or []) + list(seg.get("evidence_block_ids") or []):
                    block_id = str(block_id_raw or "").strip()
                    if block_id:
                        referenced_block_ids.add(block_id)
            merged_block_map = dict(block_map)
            for block_id in referenced_block_ids:
                block = lookup.get(block_id)
                if isinstance(block, dict):
                    merged_block_map[block_id] = dict(block)
            provenance["block_map"] = merged_block_map
            for key, value in dict(contract_meta or {}).items():
                provenance[key] = value
        except Exception:
            provenance = dict(provenance)
    if isinstance(provenance.get("segments"), list):
        provenance = dict(provenance)
        provenance["segments"] = [
            _canonicalize_support_segment_heading(seg)
            if isinstance(seg, dict)
            else seg
            for seg in list(provenance.get("segments") or [])
        ]
        provenance["segments"] = _propagate_box_scope_for_display(provenance.get("segments") or [])
        provenance["segments"] = _annotate_provenance_hit_levels(provenance.get("segments") or [])
    segments_raw = provenance.get("segments")
    if not isinstance(segments_raw, list):
        return provenance
    display_block_map_raw = provenance.get("block_map")
    display_block_map = {
        str(block_id): dict(block)
        for block_id, block in dict(display_block_map_raw or {}).items()
        if str(block_id or "").strip() and isinstance(block, dict)
    }
    source_path = str(provenance.get("source_path") or "").strip()
    source_name = str(provenance.get("source_name") or "").strip()
    if (not source_name) and source_path:
        source_name = _source_name_from_path(source_path)
    segments_out: list[dict] = []
    for idx, seg0 in enumerate(segments_raw, start=1):
        if not isinstance(seg0, dict):
            continue
        seg = dict(seg0)
        raw_markdown = str(seg.get("raw_markdown") or seg.get("raw_text") or seg.get("text") or "").strip()
        rendered_segment = raw_markdown
        cite_details: list[dict] = []
        if rendered_segment:
            rendered_segment = _annotate_equation_tags_with_sources(rendered_segment, hits)
            rendered_segment = _normalize_equation_source_notes(rendered_segment)
            rendered_segment, cite_details = _annotate_inpaper_citations_with_hover_meta(
                rendered_segment,
                hits,
                anchor_ns=f"{anchor_ns}:seg:{idx}",
            )
            if _should_retry_structured_cite_fallback(
                raw_body=raw_markdown,
                rendered_body=rendered_segment,
                cite_details=cite_details,
            ):
                rendered_segment, cite_details = _fallback_render_structured_citations(
                    raw_markdown,
                    hits,
                    anchor_ns=f"{anchor_ns}:seg:{idx}",
                )
        seg["display_markdown"] = _normalize_chat_markdown_for_display(rendered_segment or raw_markdown or str(seg.get("text") or ""))
        seg["cite_details"] = cite_details
        panel_clause_snippet = _resolve_paper_guide_panel_clause_snippet(
            seg,
            block_lookup=display_block_map,
            md_path=str(provenance.get("md_path") or "").strip(),
        )
        locate_target = _build_paper_guide_segment_locate_target(
            seg,
            panel_clause_snippet=panel_clause_snippet,
        )
        if locate_target:
            seg["locate_target"] = locate_target
        try:
            claim_group_distance = int(seg.get("claim_group_target_distance") or 0)
        except Exception:
            claim_group_distance = 0
        claim_group = {
            "id": str(seg.get("claim_group_id") or "").strip() or None,
            "kind": str(seg.get("claim_group_kind") or "").strip() or None,
            "leadText": str(seg.get("claim_group_lead_text") or "").strip() or None,
            "distance": claim_group_distance or None,
        }
        alternative_candidates = _build_reader_open_alternative_candidates(
            seg,
            block_lookup=display_block_map,
            locate_target=locate_target,
            anchor_number=int(locate_target.get("anchorNumber") or 0),
        )
        reader_open = _build_paper_guide_segment_reader_open(
            seg,
            source_path=source_path,
            source_name=source_name,
            locate_target=locate_target,
            alternative_candidates=alternative_candidates,
            claim_group=claim_group,
        )
        if reader_open:
            seg["reader_open"] = reader_open
        segments_out.append(seg)
    out = dict(provenance)
    out["segments"] = segments_out
    return out


def _source_name_from_path(source_path: str) -> str:
    name = Path(str(source_path or "")).name or str(source_path or "")
    low = name.lower()
    if low.endswith(".en.md"):
        return name[:-6] + ".pdf"
    if low.endswith(".md"):
        return name[:-3] + ".pdf"
    return name or "unknown.pdf"


def _load_ref_map(source_path: str) -> dict[int, str]:
    key = str(source_path or "").strip().lower()
    if not key:
        return {}
    cached = _REF_MAP_CACHE.get(key)
    if isinstance(cached, dict):
        return cached
    path = Path(source_path)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        ref_map = extract_references_map_from_md(text)
    except Exception:
        ref_map = {}
    _REF_MAP_CACHE[key] = ref_map
    return ref_map


def _build_anchor(anchor_ns: str, sid: str, ref_num: int, source_name: str) -> str:
    base = f"{anchor_ns}|{sid}|{int(ref_num)}|{source_name.lower()}"
    sig = hashlib.sha1(base.encode("utf-8", "ignore")).hexdigest()[:10]
    return f"kb-cite-{sig}-{int(ref_num)}"


def _fallback_render_structured_citations(md: str, hits: list[dict], *, anchor_ns: str) -> tuple[str, list[dict]]:
    src_by_sid: dict[str, str] = {}
    sha_by_source: dict[str, str] = {}
    for hit in hits or []:
        meta = (hit or {}).get("meta", {}) or {}
        source_path = str(meta.get("source_path") or "").strip()
        if not source_path:
            continue
        src_by_sid.setdefault(_source_cite_id(source_path).lower(), source_path)
        source_sha1 = str(meta.get("source_sha1") or "").strip().lower()
        if source_sha1:
            sha_by_source.setdefault(source_path, source_sha1)

    details_by_key: dict[str, dict] = {}
    index_data = _load_reference_index_cached()

    def _mk_detail(sid: str, ref_num: int) -> dict | None:
        source_path = src_by_sid.get(str(sid or "").strip().lower())
        if not source_path:
            return None
        key = f"{sid.lower()}|{int(ref_num)}"
        rec = details_by_key.get(key)
        if isinstance(rec, dict):
            return rec

        source_name = _source_name_from_path(source_path)
        anchor = _build_anchor(anchor_ns, sid, int(ref_num), source_name)

        ref_rec: dict | None = None
        try:
            resolved = resolve_reference_entry(
                index_data,
                source_path,
                int(ref_num),
                source_sha1=sha_by_source.get(source_path, ""),
            )
        except Exception:
            resolved = None
        if isinstance(resolved, dict):
            ref0 = resolved.get("ref")
            if isinstance(ref0, dict):
                ref_rec = dict(ref0)

        if not isinstance(ref_rec, dict):
            ref_map = _load_ref_map(source_path)
            raw = str(ref_map.get(int(ref_num)) or "").strip()
            if not raw:
                return None
            ref_rec = {
                "raw": raw,
                "doi": str(extract_first_doi(raw) or "").strip(),
            }

        ref2 = _normalize_reference_for_popup(
            ref_rec
        ) or {}
        raw = str(ref2.get("raw") or ref_rec.get("raw") or "").strip()
        doi = str(ref2.get("doi") or ref_rec.get("doi") or extract_first_doi(raw) or "").strip()
        doi_url = str(ref2.get("doi_url") or "").strip()
        if (not doi_url) and doi:
            doi_url = f"https://doi.org/{doi}"
        rec = {
            "num": int(ref_num),
            "anchor": anchor,
            "source_name": source_name,
            "source_path": source_path,
            "raw": str(ref2.get("raw") or raw).strip(),
            "title": str(ref2.get("title") or "").strip(),
            "authors": str(ref2.get("authors") or "").strip(),
            "venue": str(ref2.get("venue") or "").strip(),
            "year": str(ref2.get("year") or "").strip(),
            "volume": str(ref2.get("volume") or "").strip(),
            "issue": str(ref2.get("issue") or "").strip(),
            "pages": str(ref2.get("pages") or "").strip(),
            "doi": str(ref2.get("doi") or doi).strip(),
            "doi_url": doi_url,
            "cite_fmt": str(ref2.get("cite_fmt") or raw).strip(),
        }
        details_by_key[key] = rec
        return rec

    def _replace(m: re.Match) -> str:
        sid = str(m.group(1) or "").strip()
        n_txt = str(m.group(2) or "").strip()
        if not n_txt:
            return ""
        try:
            n = int(n_txt)
        except Exception:
            return ""
        detail = _mk_detail(sid, n)
        if not detail:
            return ""
        return f"[{n}](#{detail['anchor']})"

    out = _STRUCT_CITE_RE.sub(_replace, str(md or ""))
    out = _STRUCT_CITE_SINGLE_RE.sub(_replace, out)
    out = _STRUCT_CITE_SID_ONLY_RE.sub("", out)
    out = _STRUCT_CITE_GARBAGE_RE.sub("", out)
    details = sorted(details_by_key.values(), key=lambda item: (int(item.get("num") or 0), str(item.get("source_name") or "")))
    return out, details


def enrich_messages_with_reference_render(
    messages: list[dict],
    refs_by_user: dict[int, dict],
    *,
    conv_id: str,
    chat_store=None,
    render_packet_only: bool = False,
) -> list[dict]:
    out: list[dict] = []
    last_user_msg_id = 0
    for idx, msg in enumerate(messages or []):
        rec = dict(msg or {})
        role = str(rec.get("role") or "")
        content = str(rec.get("content") or "")
        try:
            msg_id = int(rec.get("id") or 0)
        except Exception:
            msg_id = 0

        if role == "user":
            if msg_id > 0:
                last_user_msg_id = msg_id
            out.append(rec)
            continue

        ref_pack = refs_by_user.get(last_user_msg_id) if isinstance(refs_by_user, dict) else None
        hits = list((ref_pack or {}).get("hits") or []) if isinstance(ref_pack, dict) else []
        provenance_raw = rec.get("provenance") if isinstance(rec.get("provenance"), dict) else None
        render_cache_key = _build_message_render_cache_key(
            conv_id=conv_id,
            msg_id=msg_id,
            role=role,
            content=content,
            refs_user_msg_id=int(last_user_msg_id or 0),
            ref_pack=ref_pack if isinstance(ref_pack, dict) else None,
            provenance=provenance_raw if isinstance(provenance_raw, dict) else None,
        )
        cached = _extract_render_cache(
            rec.get("meta") if isinstance(rec.get("meta"), dict) else None,
            expected_key=render_cache_key,
        )
        if cached:
            _restore_render_packet_contract_from_cache(rec, cached)
            rec["cite_details"] = list(cached.get("cite_details") or [])
            rec["copy_markdown"] = str(cached.get("copy_markdown") or "")
            rec["copy_text"] = str(cached.get("copy_text") or "")
            rec["rendered_content"] = str(cached.get("rendered_content") or "")
            rec["notice"] = str(cached.get("notice") or "")
            rec["rendered_body"] = str(cached.get("rendered_body") or "")
            rec["refs_user_msg_id"] = int(cached.get("refs_user_msg_id") or last_user_msg_id or 0)
        else:
            notice, body = _split_kb_miss_notice(content)
            if notice and hits:
                notice = ""
                body = content
            cite_details: list[dict] = []
            rendered_body = str(body or "")
            raw_body = rendered_body
            allow_inpaper_citation_linking = _should_link_inpaper_citations_for_message(
                rec=rec,
                content=content,
            )
            if rendered_body.strip():
                rendered_body = _annotate_equation_tags_with_sources(rendered_body, hits)
                rendered_body = _normalize_equation_source_notes(rendered_body)
                if allow_inpaper_citation_linking:
                    rendered_body, cite_details = _annotate_inpaper_citations_with_hover_meta(
                        rendered_body,
                        hits,
                        anchor_ns=f"{conv_id}:{idx}:{msg_id}:api",
                    )
                    if _should_retry_structured_cite_fallback(
                        raw_body=raw_body,
                        rendered_body=rendered_body,
                        cite_details=cite_details,
                    ):
                        rendered_body, cite_details = _fallback_render_structured_citations(
                            raw_body,
                            hits,
                            anchor_ns=f"{conv_id}:{idx}:{msg_id}:api",
                        )
                else:
                    rendered_body = _strip_structured_cite_tokens_for_display(rendered_body)
                    rendered_body = _strip_freeform_numeric_citation_markers(rendered_body)

            rendered_full = ""
            if notice and rendered_body:
                rendered_full = f"{notice}\n\n{rendered_body}"
            elif notice:
                rendered_full = notice
            elif rendered_body:
                rendered_full = rendered_body
            else:
                rendered_full = content

            rendered_markdown, rendered_body_norm, copy_markdown, copy_text = _build_render_texts(
                rendered_full=rendered_full,
                rendered_body=str(rendered_body or ""),
                notice=notice,
                cite_details=cite_details,
            )
            rec["cite_details"] = cite_details
            rec["copy_markdown"] = copy_markdown
            rec["copy_text"] = copy_text
            rec["rendered_content"] = rendered_markdown
            rec["notice"] = notice
            rec["rendered_body"] = rendered_body_norm
            rec["refs_user_msg_id"] = int(last_user_msg_id or 0)
        enriched_provenance = _enrich_provenance_segments_for_display(
            provenance_raw if isinstance(provenance_raw, dict) else None,
            hits,
            anchor_ns=f"{conv_id}:{idx}:{msg_id}:api",
        )
        if isinstance(enriched_provenance, dict):
            rec["provenance"] = enriched_provenance
            if isinstance(rec.get("meta"), dict):
                rec["meta"] = dict(rec.get("meta") or {})
                rec["meta"]["provenance"] = enriched_provenance
        _merge_render_packet_contract_meta(
            rec=rec,
            msg_id=msg_id,
            enriched_provenance=enriched_provenance if isinstance(enriched_provenance, dict) else None,
            chat_store=chat_store,
        )
        _project_render_packet_compat_fields(rec)
        _maybe_strip_legacy_render_fields(rec, enabled=bool(render_packet_only))
        if chat_store is not None and msg_id > 0 and not cached:
            try:
                meta = dict(rec.get("meta") or {}) if isinstance(rec.get("meta"), dict) else {}
                contracts = dict(meta.get("paper_guide_contracts") or {}) if isinstance(meta.get("paper_guide_contracts"), dict) else {}
                render_packet = dict(contracts.get("render_packet") or {}) if isinstance(contracts.get("render_packet"), dict) else {}
                chat_store.set_message_render_cache(
                    msg_id,
                    _build_render_cache_payload(
                        cache_key=render_cache_key,
                        notice=str(rec.get("notice") or ""),
                        rendered_body=str(rec.get("rendered_body") or ""),
                        rendered_content=str(rec.get("rendered_content") or ""),
                        copy_markdown=str(rec.get("copy_markdown") or ""),
                        copy_text=str(rec.get("copy_text") or ""),
                        cite_details=[
                            dict(item)
                            for item in list(rec.get("cite_details") or [])
                            if isinstance(item, dict)
                        ],
                        refs_user_msg_id=int(rec.get("refs_user_msg_id") or last_user_msg_id or 0),
                        render_packet=render_packet,
                    ),
                )
            except Exception:
                pass
        rec["render_cache_key"] = str(render_cache_key or "")[:12]
        out.append(rec)

    return out

