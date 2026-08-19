from __future__ import annotations

import sqlite3
import time
import uuid
import json
import math
import re
from pathlib import Path

from kb.path_safety import clean_file_source_path_input

DEFAULT_ACTIVE_CONVERSATION_LIMIT = 400
MAX_CITATION_SHELF_ITEMS = 120
MAX_PROJECT_NAME_LEN = 120
MAX_RESEARCH_BRIEF_TITLE_LEN = 240
MAX_RESEARCH_BRIEF_OBJECTIVE_LEN = 4_000
MAX_RESEARCH_BRIEF_CONTENT_LEN = 160_000
MAX_EVIDENCE_MATRIX_TITLE_LEN = 240
MAX_EVIDENCE_MATRIX_OBJECTIVE_LEN = 4_000
_SHELF_MAX_ITEM_KEYS = 80
_SHELF_MAX_DICT_KEYS = 32
_SHELF_MAX_LIST_ITEMS = 32
_SHELF_MAX_VALUE_DEPTH = 3
_SHELF_DEFAULT_STRING_LIMIT = 700
_SHELF_TEXT_LIMIT_BY_KEY = {
    "abstract": 1600,
    "anchor": 800,
    "anchorid": 800,
    "blockid": 800,
    "cardcontextsummary": 1600,
    "cardevidence": 1600,
    "cardreferenceentry": 2000,
    "cardtakeaway": 1000,
    "cardtitle": 500,
    "citationcontext": 1600,
    "citefmt": 1600,
    "doi": 400,
    "doiurl": 400,
    "evidencequote": 1600,
    "evidencesource": 120,
    "headingpath": 800,
    "key": 500,
    "locationlabel": 500,
    "main": 500,
    "note": 4000,
    "raw": 1600,
    "shelfexcerpt": 1600,
    "shelfexcerptlabel": 120,
    "shelforigin": 120,
    "source": 500,
    "sourcename": 500,
    "sourcepath": 800,
    "summaryline": 1000,
    "title": 800,
    "venue": 500,
    "whyline": 1000,
    "year": 40,
}
_STATE_MAX_TOP_LEVEL_KEYS = 80
_STATE_MAX_DICT_KEYS = 120
_STATE_MAX_LIST_ITEMS = 500
_STATE_MAX_VALUE_DEPTH = 6
_STATE_MAX_STRING_LIMIT = 4000
_STATE_MAX_KEY_LIMIT = 120
_MESSAGE_REFS_NESTED_PAYLOAD_REPAIR_KEY = "message_refs_strip_nested_rendered_payload_v1"


_DEFAULT_CONVERSATION_TITLE_RE = re.compile(
    r"^(?:新对话|新会话|New Chat|New conversation|(?:研究问答|新会话|Research QA|New conversation)\s*[·:：-]\s*[\d:/\-\s]+)$",
    flags=re.IGNORECASE,
)


def _shelf_text(value: object, limit: int = 2000) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def _normalize_reader_source_path_key(value: object) -> str:
    text = clean_file_source_path_input(value)
    if not text:
        return ""
    text = text.replace("\\", "/")
    unc_prefix = text.startswith("//")
    absolute_prefix = (not unc_prefix) and text.startswith("/")
    parts: list[str] = []
    for raw_part in text.split("/"):
        part = raw_part.strip()
        if not part or part == ".":
            continue
        if part == "..":
            prev = parts[-1] if parts else ""
            if prev and prev != ".." and not re.match(r"^[A-Za-z]:$", prev):
                parts.pop()
            else:
                parts.append(part)
            continue
        parts.append(part)
    out = "/".join(parts)
    if unc_prefix and out:
        out = f"//{out}"
    elif absolute_prefix and out:
        out = f"/{out}"
    out = re.sub(r"^/([A-Za-z]:)(/|$)", r"\1\2", out)
    return out.rstrip("/").lower()


def _shelf_key_norm(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key or "").lower())


def _shelf_string_limit(key: object) -> int:
    return _SHELF_TEXT_LIMIT_BY_KEY.get(_shelf_key_norm(key), _SHELF_DEFAULT_STRING_LIMIT)


def _shelf_value_is_empty(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _sanitize_citation_shelf_value(value: object, *, key: str = "", depth: int = _SHELF_MAX_VALUE_DEPTH) -> object:
    if value is None:
        return None
    if isinstance(value, str):
        return _shelf_text(value, limit=_shelf_string_limit(key))
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if depth <= 0:
        return _shelf_text(value, limit=_SHELF_DEFAULT_STRING_LIMIT)
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            clean_key = _shelf_text(raw_key, limit=120)
            if not clean_key or clean_key in out:
                continue
            clean_value = _sanitize_citation_shelf_value(raw_value, key=clean_key, depth=depth - 1)
            if _shelf_value_is_empty(clean_value):
                continue
            out[clean_key] = clean_value
            if len(out) >= _SHELF_MAX_DICT_KEYS:
                break
        return out
    if isinstance(value, (list, tuple)):
        out_list: list[object] = []
        for entry in value:
            clean_value = _sanitize_citation_shelf_value(entry, key=key, depth=depth - 1)
            if _shelf_value_is_empty(clean_value):
                continue
            out_list.append(clean_value)
            if len(out_list) >= _SHELF_MAX_LIST_ITEMS:
                break
        return out_list
    return _shelf_text(value, limit=_shelf_string_limit(key))


def _sanitize_citation_shelf_item_payload(item: dict) -> dict:
    out: dict[str, object] = {}
    for raw_key, raw_value in item.items():
        clean_key = _shelf_text(raw_key, limit=120)
        if not clean_key or clean_key in out:
            continue
        clean_value = _sanitize_citation_shelf_value(raw_value, key=clean_key)
        if _shelf_value_is_empty(clean_value):
            continue
        out[clean_key] = clean_value
        if len(out) >= _SHELF_MAX_ITEM_KEYS:
            break
    return out


def _state_text(value: object, limit: int = _STATE_MAX_STRING_LIMIT) -> str:
    return str(value).replace("\x00", " ")[:limit]


def _sanitize_json_state_value(value: object, *, depth: int = _STATE_MAX_VALUE_DEPTH) -> object:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _state_text(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if depth <= 0:
        return _state_text(value)
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            clean_key = _state_text(raw_key, limit=_STATE_MAX_KEY_LIMIT).strip()
            if not clean_key or clean_key in out:
                continue
            out[clean_key] = _sanitize_json_state_value(raw_value, depth=depth - 1)
            if len(out) >= _STATE_MAX_DICT_KEYS:
                break
        return out
    if isinstance(value, (list, tuple)):
        out_list: list[object] = []
        for entry in value:
            out_list.append(_sanitize_json_state_value(entry, depth=depth - 1))
            if len(out_list) >= _STATE_MAX_LIST_ITEMS:
                break
        return out_list
    return _state_text(value)


def _sanitize_json_state_dict(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    out: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        clean_key = _state_text(raw_key, limit=_STATE_MAX_KEY_LIMIT).strip()
        if not clean_key or clean_key in out:
            continue
        out[clean_key] = _sanitize_json_state_value(raw_value)
        if len(out) >= _STATE_MAX_TOP_LEVEL_KEYS:
            break
    return out


def _apply_json_state_patch(current: object, patch: object) -> dict:
    state = _sanitize_json_state_dict(current)
    patch_dict = _sanitize_json_state_dict(patch)
    for key, value in patch_dict.items():
        clean_key = str(key or "").strip()
        if not clean_key:
            continue
        if value is None:
            state.pop(clean_key, None)
        else:
            state[clean_key] = value
    return _sanitize_json_state_dict(state)


def _shelf_first_text(item: dict, *keys: str, limit: int = 2000) -> str:
    for key in keys:
        text = _shelf_text(item.get(key), limit=limit)
        if text:
            return text
    return ""


def _normalize_shelf_doi(value: object) -> str:
    text = _shelf_text(value, limit=400).lower()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    return text.strip(" \t\r\n'\"`([{<.,;:)]}>")


def _normalize_shelf_title(value: object) -> str:
    text = _shelf_text(value, limit=800).lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _shelf_item_kind(item: dict) -> str:
    raw = _shelf_first_text(item, "shelfItemKind", "shelf_item_kind", "cardKind", "card_kind", limit=80)
    key = re.sub(r"[\s-]+", "_", raw.lower())
    if key in {"reference", "inpaper", "reader_reference", "reader_references"}:
        return "reference"
    if key in {"reader_selection", "selection", "reader_excerpt"}:
        return "reader_selection"
    if key in {"excerpt", "note"}:
        return "excerpt"
    if item.get("isInpaper") is True or item.get("is_inpaper") is True:
        return "reference"
    return "citation"


def _shelf_item_identity(item: dict) -> str:
    kind = _shelf_item_kind(item)
    if kind == "reader_selection":
        source = _shelf_first_text(item, "sourcePath", "source_path", "sourceName", "source_name", limit=800).lower()
        anchor = _shelf_first_text(item, "blockId", "block_id", "anchorId", "anchor_id", "anchor", limit=800).lower()
        start = _shelf_first_text(item, "startOffset", "start_offset", "startReadableIndex", "start_readable_index", limit=80)
        end = _shelf_first_text(item, "endOffset", "end_offset", "endReadableIndex", "end_readable_index", limit=80)
        excerpt = _normalize_shelf_title(_shelf_first_text(
            item,
            "shelfExcerpt",
            "shelf_excerpt",
            "evidenceQuote",
            "evidence_quote",
            "raw",
            "citeFmt",
            "cite_fmt",
            limit=500,
        ))
        return f"reader-selection:{source}|{anchor}|{start}|{end}|{excerpt[:160]}"
    doi = _normalize_shelf_doi(_shelf_first_text(item, "doi", "doiUrl", "doi_url", limit=400))
    if doi:
        return f"doi:{doi}"
    title = _normalize_shelf_title(_shelf_first_text(item, "title", "main", "cardTitle", "card_title", limit=800))
    year = _shelf_first_text(item, "year", limit=20)
    year = year if re.fullmatch(r"\d{4}", year) else ""
    if title:
        return f"title:{title}|{year}"
    source = _shelf_first_text(item, "sourcePath", "source_path", "sourceName", "source_name", limit=800).lower()
    anchor = _shelf_first_text(item, "anchor", "anchorId", "anchor_id", limit=800).lower()
    if source or anchor:
        return f"source:{source}|{anchor}"
    key = _shelf_first_text(item, "key", limit=400)
    return f"key:{key}" if key else ""


def _shelf_item_stable_key(item: dict) -> str:
    return _shelf_first_text(item, "key", limit=500)


def _shelf_text_is_weak(value: object) -> bool:
    text = _shelf_text(value, limit=500).lower()
    return text in {
        "untitled",
        "untitled excerpt",
        "reference entry",
        "selected text",
        "excerpt",
        "unknown",
    }


def _shelf_quality_status_rank(value: object) -> int:
    status = _shelf_text(value, limit=120).lower().replace("-", "_")
    if status in {"ready", "repaired", "verified", "ok", "complete"}:
        return 5
    if status in {"enriched", "crossref", "matched", "external_ready"}:
        return 4
    if status in {"pending", "syncing", "checking", "queued"}:
        return 3
    if status in {"needs_review", "partial", "incomplete"}:
        return 2
    if status in {"missing", "not_found", "failed", "error", "invalid"}:
        return 1
    return 0


def _shelf_metadata_quality_rank(value: object) -> int:
    if not isinstance(value, dict):
        return 0
    if value.get("ok") is True:
        return 5
    return _shelf_quality_status_rank(value.get("status") or value.get("repair_status") or value.get("state"))


def _merge_shelf_metadata_quality(current: object, incoming: object) -> dict:
    current_dict = dict(current) if isinstance(current, dict) else {}
    incoming_dict = dict(incoming) if isinstance(incoming, dict) else {}
    if not current_dict:
        return incoming_dict
    if not incoming_dict:
        return current_dict
    current_rank = _shelf_metadata_quality_rank(current_dict)
    incoming_rank = _shelf_metadata_quality_rank(incoming_dict)
    if incoming_rank >= current_rank:
        merged = dict(current_dict)
        merged.update(incoming_dict)
        return merged
    merged = dict(incoming_dict)
    merged.update(current_dict)
    return merged


def _normalize_citation_shelf_item(item: dict) -> dict:
    out = _sanitize_citation_shelf_item_payload(item)
    kind = _shelf_item_kind(out)
    origin = _shelf_first_text(out, "shelfOrigin", "shelf_origin", "evidenceSource", "evidence_source", limit=120)
    if not origin:
        origin = "reader_selection" if kind == "reader_selection" else "reader_references" if kind == "reference" else "chat_answer"
    excerpt = _shelf_first_text(
        out,
        "shelfExcerpt",
        "shelf_excerpt",
        "evidenceQuote",
        "evidence_quote",
        "citationContext",
        "citation_context",
        "cardEvidence",
        "card_evidence",
        "raw",
        "citeFmt",
        "cite_fmt",
        limit=1600,
    )
    label = _shelf_first_text(out, "shelfExcerptLabel", "shelf_excerpt_label", limit=120)
    if not label:
        label = "Reference entry" if kind == "reference" else "Selected text" if kind == "reader_selection" else "Excerpt"

    out["shelfItemKind"] = kind
    out["shelf_item_kind"] = kind
    out["shelfOrigin"] = origin
    out["shelf_origin"] = origin
    out["shelfExcerpt"] = excerpt
    out["shelf_excerpt"] = excerpt
    out["shelfExcerptLabel"] = label
    out["shelf_excerpt_label"] = label
    out["key"] = _shelf_first_text(out, "key", limit=500) or _shelf_item_identity(out)
    out["main"] = _shelf_first_text(out, "main", "title", "cardTitle", "card_title", "raw", limit=500)
    tags = out.get("tags")
    out["tags"] = [_shelf_text(tag, limit=60) for tag in tags if _shelf_text(tag, limit=60)] if isinstance(tags, list) else []
    out["note"] = _shelf_text(out.get("note"), limit=4000)
    return out


def _merge_citation_shelf_item(existing: dict, incoming: dict) -> dict:
    base = _normalize_citation_shelf_item(existing)
    new = _normalize_citation_shelf_item(incoming)
    out = dict(base)
    rich_text_keys = {
        "shelfExcerpt",
        "shelf_excerpt",
        "evidenceQuote",
        "evidence_quote",
        "citationContext",
        "citation_context",
        "cardEvidence",
        "card_evidence",
        "cardReferenceEntry",
        "card_reference_entry",
        "summaryLine",
        "summary_line",
        "cardTakeaway",
        "card_takeaway",
        "cardContextSummary",
        "card_context_summary",
        "whyLine",
        "why_line",
        "headingPath",
        "heading_path",
        "locationLabel",
        "location_label",
    }
    status_keys = {
        "metadataRepairStatus",
        "metadata_repair_status",
        "externalMetadataStatus",
        "external_metadata_status",
        "libraryMatchStatus",
        "library_match_status",
    }
    metadata_quality_keys = {"metadataQuality", "metadata_quality"}
    for key, value in new.items():
        if key in {"tags", "note"}:
            continue
        if isinstance(value, str):
            incoming_text = _shelf_text(value, limit=4000)
            if not incoming_text:
                continue
            current_text = _shelf_text(out.get(key), limit=4000)
            if key in status_keys:
                if (not current_text) or _shelf_quality_status_rank(incoming_text) >= _shelf_quality_status_rank(current_text):
                    out[key] = value
                continue
            should_replace = (
                (not current_text)
                or _shelf_text_is_weak(current_text)
                or (
                    key in rich_text_keys
                    and len(incoming_text) > len(current_text)
                    and (len(current_text) < 120 or current_text.lower() in incoming_text.lower())
                )
            )
            if should_replace:
                out[key] = value
        elif isinstance(value, dict):
            if key in metadata_quality_keys:
                out[key] = _merge_shelf_metadata_quality(out.get(key), value)
                continue
            current = out.get(key) if isinstance(out.get(key), dict) else {}
            merged = dict(current or {})
            for sub_key, sub_value in value.items():
                if sub_value in (None, "", [], {}):
                    continue
                merged[sub_key] = sub_value
            if merged:
                out[key] = merged
        elif isinstance(value, list):
            if not isinstance(out.get(key), list) or not out.get(key):
                out[key] = value
    existing_note = _shelf_text(base.get("note"), limit=4000)
    incoming_note = _shelf_text(new.get("note"), limit=4000)
    out["note"] = existing_note or incoming_note
    merged_tags: list[str] = []
    seen_tags: set[str] = set()
    for tag in list(base.get("tags") or []) + list(new.get("tags") or []):
        clean = _shelf_text(tag, limit=60)
        tag_key = clean.lower()
        if not clean or tag_key in seen_tags:
            continue
        seen_tags.add(tag_key)
        merged_tags.append(clean)
    out["tags"] = merged_tags

    # A shelf item is deduplicated by DOI/title and may have been saved before
    # a later answer produced an exact evidence quote and locator.  Do not let
    # the old hard-failure flags survive that richer merge merely because the
    # incoming card omitted a redundant ``quality.flags: []`` field.
    incoming_evidence = _shelf_first_text(
        new,
        "evidenceQuote",
        "evidence_quote",
        "cardEvidence",
        "card_evidence",
        "citationContext",
        "citation_context",
        "shelfExcerpt",
        "shelf_excerpt",
        limit=1600,
    )
    try:
        incoming_page_start = int(
            new.get("pageStart") or new.get("page_start") or 0
        )
    except (TypeError, ValueError):
        incoming_page_start = 0
    incoming_has_locator = bool(
        _shelf_first_text(
            new,
            "blockId",
            "block_id",
            "anchorId",
            "anchor_id",
            "headingPath",
            "heading_path",
            "locationLabel",
            "location_label",
            limit=800,
        )
        or incoming_page_start > 0
    )
    resolved_flags: set[str] = set()
    if incoming_evidence:
        resolved_flags.update({"missing_evidence_quote", "evidence_quote_filtered"})
    if incoming_has_locator:
        resolved_flags.add("missing_precise_location")
    if resolved_flags:
        for key in ("card_quality_flags", "cardQualityFlags"):
            if isinstance(out.get(key), list):
                out[key] = [
                    flag
                    for flag in out[key]
                    if str(flag or "").strip().lower() not in resolved_flags
                ]
        for key in ("cardView", "card_view"):
            card_view = dict(out.get(key) or {}) if isinstance(out.get(key), dict) else {}
            quality = (
                dict(card_view.get("quality") or {})
                if isinstance(card_view.get("quality"), dict)
                else {}
            )
            if isinstance(quality.get("flags"), list):
                quality["flags"] = [
                    flag
                    for flag in quality["flags"]
                    if str(flag or "").strip().lower() not in resolved_flags
                ]
                card_view["quality"] = quality
                out[key] = card_view
    return _normalize_citation_shelf_item(out)


def _normalize_citation_shelf_items(items: list[dict]) -> list[dict]:
    seen = set()
    out: list[dict] = []
    for raw in list(items or []):
        if not isinstance(raw, dict):
            continue
        item = _normalize_citation_shelf_item(raw)
        identity = _shelf_item_identity(item)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        out.append(item)
        if len(out) >= MAX_CITATION_SHELF_ITEMS:
            break
    return out


def _normalize_conversation_mode(mode: str) -> str:
    m = str(mode or "").strip().lower()
    if m in {"paper_guide", "normal"}:
        return m
    return "normal"


def _clean_project_name(name: object, *, default: str = "") -> str:
    text = str(name or "").replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text and default:
        text = default
    return text[:MAX_PROJECT_NAME_LEN]


def _project_record(row: sqlite3.Row | dict) -> dict:
    rec = dict(row)
    rec["name"] = _clean_project_name(rec.get("name"), default="未命名项目")
    return rec


def _research_brief_text(value: object, *, limit: int, multiline: bool = False) -> str:
    text = str(value or "").replace("\x00", " ")
    if multiline:
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    else:
        text = re.sub(r"\s+", " ", text).strip()
    return text[: max(0, int(limit))]


def _research_brief_json(value: object, *, default: object) -> object:
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "")
        except Exception:
            return default
        return parsed
    return value if value is not None else default


def _research_brief_record(row: sqlite3.Row | dict, *, include_content: bool = True) -> dict:
    rec = dict(row)
    evidence = _research_brief_json(rec.pop("evidence_json", "[]"), default=[])
    bibliography = _research_brief_json(rec.pop("bibliography_json", "[]"), default=[])
    agent_trace = _research_brief_json(rec.pop("agent_trace_json", "{}"), default={})
    quality = _research_brief_json(rec.pop("quality_json", "{}"), default={})
    rec["title"] = _research_brief_text(
        rec.get("title"),
        limit=MAX_RESEARCH_BRIEF_TITLE_LEN,
    )
    rec["objective"] = _research_brief_text(
        rec.get("objective"),
        limit=MAX_RESEARCH_BRIEF_OBJECTIVE_LEN,
        multiline=True,
    )
    rec["content_markdown"] = _research_brief_text(
        rec.get("content_markdown"),
        limit=MAX_RESEARCH_BRIEF_CONTENT_LEN,
        multiline=True,
    ) if include_content else ""
    rec["evidence"] = evidence if isinstance(evidence, list) else []
    rec["bibliography"] = bibliography if isinstance(bibliography, list) else []
    rec["agent_trace"] = agent_trace if isinstance(agent_trace, dict) else {}
    rec["quality"] = quality if isinstance(quality, dict) else {}
    rec["revision"] = max(1, int(rec.get("revision") or 1))
    rec["quality_status"] = str(rec.get("quality_status") or "draft").strip() or "draft"
    return rec


def _research_brief_update_plan_record(row: sqlite3.Row | dict) -> dict:
    rec = dict(row)
    payload = _research_brief_json(rec.pop("payload_json", "{}"), default={})
    out = dict(payload) if isinstance(payload, dict) else {}
    out.update(
        {
            "id": str(rec.get("id") or out.get("id") or ""),
            "brief_id": str(rec.get("brief_id") or out.get("brief_id") or ""),
            "base_brief_revision": max(1, int(rec.get("base_revision") or out.get("base_brief_revision") or 1)),
            "matrix_id": str(rec.get("matrix_id") or out.get("matrix_id") or ""),
            "target_matrix_revision": max(1, int(rec.get("matrix_revision") or out.get("target_matrix_revision") or 1)),
            "matrix_fingerprint": str(rec.get("matrix_fingerprint") or out.get("matrix_fingerprint") or ""),
            "status": str(rec.get("status") or out.get("status") or "open"),
            "created_at": float(rec.get("created_at") or out.get("created_at") or 0.0),
            "updated_at": float(rec.get("updated_at") or out.get("updated_at") or 0.0),
        }
    )
    return out


def _evidence_matrix_record(row: sqlite3.Row | dict, *, include_content: bool = True) -> dict:
    rec = dict(row)
    rows = _research_brief_json(rec.pop("rows_json", "[]"), default=[])
    evidence = _research_brief_json(rec.pop("evidence_json", "[]"), default=[])
    source_items = _research_brief_json(rec.pop("source_items_json", "[]"), default=[])
    comparison_flags = _research_brief_json(rec.pop("comparison_flags_json", "[]"), default=[])
    comparison_audits = _research_brief_json(rec.pop("comparison_audits_json", "[]"), default=[])
    quality = _research_brief_json(rec.pop("quality_json", "{}"), default={})
    rec["title"] = _research_brief_text(
        rec.get("title"),
        limit=MAX_EVIDENCE_MATRIX_TITLE_LEN,
    )
    rec["objective"] = _research_brief_text(
        rec.get("objective"),
        limit=MAX_EVIDENCE_MATRIX_OBJECTIVE_LEN,
        multiline=True,
    )
    rec["rows"] = rows if include_content and isinstance(rows, list) else []
    rec["evidence"] = evidence if include_content and isinstance(evidence, list) else []
    rec["source_items"] = source_items if include_content and isinstance(source_items, list) else []
    rec["comparison_flags"] = comparison_flags if include_content and isinstance(comparison_flags, list) else []
    rec["comparison_audits"] = comparison_audits if include_content and isinstance(comparison_audits, list) else []
    rec["quality"] = quality if isinstance(quality, dict) else {}
    rec["revision"] = max(1, int(rec.get("revision") or 1))
    rec["quality_status"] = str(rec.get("quality_status") or "draft").strip() or "draft"
    return rec


def _evidence_watch_event_record(row: sqlite3.Row | dict) -> dict:
    rec = dict(row)
    payload = _research_brief_json(rec.pop("payload_json", "{}"), default={})
    out = dict(payload) if isinstance(payload, dict) else {}
    out.update(
        {
            "id": str(rec.get("id") or out.get("id") or ""),
            "event_key": str(rec.get("event_key") or out.get("event_key") or ""),
            "project_id": str(rec.get("project_id") or out.get("project_id") or ""),
            "matrix_id": str(rec.get("matrix_id") or out.get("matrix_id") or ""),
            "matrix_revision": max(1, int(rec.get("matrix_revision") or out.get("matrix_revision") or 1)),
            "kind": str(rec.get("kind") or out.get("kind") or ""),
            "status": str(rec.get("status") or out.get("status") or "open"),
            "created_at": float(rec.get("created_at") or out.get("created_at") or 0.0),
            "updated_at": float(rec.get("updated_at") or out.get("updated_at") or 0.0),
        }
    )
    return out


def _research_gap_record(row: sqlite3.Row | dict) -> dict:
    rec = dict(row)
    payload = _research_brief_json(rec.pop("payload_json", "{}"), default={})
    action = _research_brief_json(rec.pop("action_json", "{}"), default={})
    out = dict(payload) if isinstance(payload, dict) else {}
    out.update(
        {
            "id": str(rec.get("id") or out.get("id") or ""),
            "gap_key": str(rec.get("gap_key") or out.get("gap_key") or ""),
            "project_id": str(rec.get("project_id") or out.get("project_id") or ""),
            "matrix_id": str(rec.get("matrix_id") or out.get("matrix_id") or ""),
            "brief_id": str(rec.get("brief_id") or out.get("brief_id") or ""),
            "kind": str(rec.get("kind") or out.get("kind") or ""),
            "status": str(rec.get("status") or out.get("status") or "open"),
            "action": action if isinstance(action, dict) else {},
            "created_at": float(rec.get("created_at") or out.get("created_at") or 0.0),
            "updated_at": float(rec.get("updated_at") or out.get("updated_at") or 0.0),
        }
    )
    return out


def _is_default_conversation_title(title: str) -> bool:
    text = str(title or "").strip()
    if not text:
        return True
    return bool(_DEFAULT_CONVERSATION_TITLE_RE.match(text))


class ChatStore:
    """
    A tiny local chat persistence layer.
    - One sqlite file
    - Multiple conversations
    - Append-only messages
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self, *, timeout_s: float | None = None) -> sqlite3.Connection:
        # WAL helps concurrent reads while API and background tasks overlap.
        try:
            timeout_final = float(timeout_s if timeout_s is not None else 30.0)
        except Exception:
            timeout_final = 30.0
        timeout_final = max(0.05, timeout_final)
        conn = sqlite3.connect(str(self._db_path), timeout=timeout_final, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            conn.execute(f"PRAGMA busy_timeout={int(timeout_final * 1000)};")
        except Exception:
            pass
        return conn

    def _begin_immediate(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")

    def _project_exists(self, conn: sqlite3.Connection, project_id: str | None) -> bool:
        pid = str(project_id or "").strip()
        if not pid:
            return False
        row = conn.execute("SELECT 1 FROM projects WHERE id = ? LIMIT 1", (pid,)).fetchone()
        return row is not None

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                  id TEXT PRIMARY KEY,
                  title TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conv_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  content TEXT NOT NULL,
                  attachments_json TEXT NOT NULL DEFAULT '[]',
                  meta_json TEXT NOT NULL DEFAULT '{}',
                  created_at REAL NOT NULL,
                  FOREIGN KEY(conv_id) REFERENCES conversations(id)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conv_id);")
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN attachments_json TEXT NOT NULL DEFAULT '[]'")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN meta_json TEXT NOT NULL DEFAULT '{}'")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS message_refs (
                  user_msg_id INTEGER PRIMARY KEY,
                  conv_id TEXT NOT NULL,
                  prompt TEXT NOT NULL,
                  prompt_sig TEXT NOT NULL,
                  hits_json TEXT NOT NULL,
                  scores_json TEXT NOT NULL,
                  rendered_payload_json TEXT NOT NULL DEFAULT '',
                  rendered_payload_sig TEXT NOT NULL DEFAULT '',
                  render_status TEXT NOT NULL DEFAULT '',
                  render_error TEXT NOT NULL DEFAULT '',
                  render_error_detail TEXT NOT NULL DEFAULT '',
                  render_built_at REAL NOT NULL DEFAULT 0,
                  render_attempts INTEGER NOT NULL DEFAULT 0,
                  render_evidence_sig TEXT NOT NULL DEFAULT '',
                  render_locale TEXT NOT NULL DEFAULT '',
                  used_query TEXT NOT NULL,
                  used_translation INTEGER NOT NULL DEFAULT 0,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL
                );
                """
            )
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN rendered_payload_json TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN rendered_payload_sig TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN render_status TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN render_error TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN render_error_detail TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN render_built_at REAL NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN render_attempts INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN render_evidence_sig TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN render_locale TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE message_refs ADD COLUMN query_variants_json TEXT NOT NULL DEFAULT '[]'")
            except sqlite3.OperationalError:
                pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_message_refs_conv_id ON message_refs(conv_id);")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_store_repairs (
                  repair_key TEXT PRIMARY KEY,
                  repaired_at REAL NOT NULL
                );
                """
            )
            repair_done = conn.execute(
                "SELECT 1 FROM chat_store_repairs WHERE repair_key = ? LIMIT 1",
                (_MESSAGE_REFS_NESTED_PAYLOAD_REPAIR_KEY,),
            ).fetchone()
            if repair_done is None:
                try:
                    # Older render writes could recursively embed the previous
                    # payload under ``rendered_payload``.  One observed row was
                    # 67 MB although its useful top-level packet was ~56 KB.
                    # JSON1 removes that redundant subtree inside SQLite, so
                    # startup does not deserialize hundreds of megabytes in
                    # Python.  The repair is idempotent and recorded once.
                    conn.execute(
                        """
                        UPDATE message_refs
                        SET rendered_payload_json = json_remove(
                            rendered_payload_json,
                            '$.rendered_payload'
                        )
                        WHERE rendered_payload_json <> ''
                          AND json_valid(rendered_payload_json)
                          AND json_type(
                              rendered_payload_json,
                              '$.rendered_payload'
                          ) IS NOT NULL
                        """
                    )
                except sqlite3.OperationalError:
                    # Very old SQLite builds may not include JSON1.  Keep the
                    # data untouched and retry after the runtime is upgraded.
                    pass
                else:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO chat_store_repairs
                            (repair_key, repaired_at)
                        VALUES (?, ?)
                        """,
                        (_MESSAGE_REFS_NESTED_PAYLOAD_REPAIR_KEY, time.time()),
                    )
            # Projects (ChatGPT-style): optional grouping for conversations
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                  id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL
                );
                """
            )
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN project_id TEXT;")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN archived INTEGER NOT NULL DEFAULT 0;")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN archived_at REAL;")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN mode TEXT NOT NULL DEFAULT 'normal';")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN bound_source_path TEXT NOT NULL DEFAULT '';")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN bound_source_name TEXT NOT NULL DEFAULT '';")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN bound_source_ready INTEGER NOT NULL DEFAULT 0;")
            except sqlite3.OperationalError:
                pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_project_id ON conversations(project_id);")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_scope_archived_updated "
                "ON conversations(project_id, archived, updated_at DESC);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_sources (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conv_id TEXT NOT NULL,
                  source_path TEXT NOT NULL,
                  source_name TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  UNIQUE(conv_id, source_path),
                  FOREIGN KEY(conv_id) REFERENCES conversations(id)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation_sources_conv_id ON conversation_sources(conv_id);")
            conn.execute(
                "DELETE FROM conversation_sources "
                "WHERE conv_id NOT IN (SELECT id FROM conversations)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_reader_states (
                  conv_id TEXT NOT NULL,
                  source_path TEXT NOT NULL,
                  state_json TEXT NOT NULL DEFAULT '{}',
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  PRIMARY KEY(conv_id, source_path),
                  FOREIGN KEY(conv_id) REFERENCES conversations(id)
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_reader_states_conv_id "
                "ON conversation_reader_states(conv_id, updated_at DESC);"
            )
            conn.execute(
                "DELETE FROM conversation_reader_states "
                "WHERE conv_id NOT IN (SELECT id FROM conversations)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_research_states (
                  conv_id TEXT PRIMARY KEY,
                  state_json TEXT NOT NULL DEFAULT '{}',
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  FOREIGN KEY(conv_id) REFERENCES conversations(id)
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_research_states_updated "
                "ON conversation_research_states(updated_at DESC);"
            )
            conn.execute(
                "DELETE FROM conversation_research_states "
                "WHERE conv_id NOT IN (SELECT id FROM conversations)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS citation_shelves (
                  scope TEXT NOT NULL,
                  scope_id TEXT NOT NULL,
                  items_json TEXT NOT NULL DEFAULT '[]',
                  open INTEGER NOT NULL DEFAULT 0,
                  revision INTEGER NOT NULL DEFAULT 0,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  PRIMARY KEY(scope, scope_id)
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_citation_shelves_updated "
                "ON citation_shelves(updated_at DESC);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_briefs (
                  id TEXT PRIMARY KEY,
                  project_id TEXT NOT NULL,
                  source_conv_id TEXT,
                  title TEXT NOT NULL,
                  objective TEXT NOT NULL DEFAULT '',
                  content_markdown TEXT NOT NULL DEFAULT '',
                  evidence_json TEXT NOT NULL DEFAULT '[]',
                  bibliography_json TEXT NOT NULL DEFAULT '[]',
                  agent_trace_json TEXT NOT NULL DEFAULT '{}',
                  quality_status TEXT NOT NULL DEFAULT 'draft',
                  quality_json TEXT NOT NULL DEFAULT '{}',
                  revision INTEGER NOT NULL DEFAULT 1,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
                  FOREIGN KEY(source_conv_id) REFERENCES conversations(id) ON DELETE SET NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_briefs_project_updated "
                "ON research_briefs(project_id, updated_at DESC);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_brief_revisions (
                  brief_id TEXT NOT NULL,
                  revision INTEGER NOT NULL,
                  title TEXT NOT NULL,
                  objective TEXT NOT NULL DEFAULT '',
                  content_markdown TEXT NOT NULL DEFAULT '',
                  evidence_json TEXT NOT NULL DEFAULT '[]',
                  bibliography_json TEXT NOT NULL DEFAULT '[]',
                  agent_trace_json TEXT NOT NULL DEFAULT '{}',
                  quality_status TEXT NOT NULL DEFAULT 'draft',
                  quality_json TEXT NOT NULL DEFAULT '{}',
                  created_at REAL NOT NULL,
                  PRIMARY KEY(brief_id, revision),
                  FOREIGN KEY(brief_id) REFERENCES research_briefs(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_brief_revisions_created "
                "ON research_brief_revisions(brief_id, created_at DESC);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_brief_update_plans (
                  id TEXT PRIMARY KEY,
                  brief_id TEXT NOT NULL,
                  base_revision INTEGER NOT NULL,
                  matrix_id TEXT NOT NULL,
                  matrix_revision INTEGER NOT NULL,
                  matrix_fingerprint TEXT NOT NULL,
                  payload_json TEXT NOT NULL DEFAULT '{}',
                  status TEXT NOT NULL DEFAULT 'open',
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  FOREIGN KEY(brief_id) REFERENCES research_briefs(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_brief_update_plans_brief "
                "ON research_brief_update_plans(brief_id, status, updated_at DESC);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_evidence_matrices (
                  id TEXT PRIMARY KEY,
                  project_id TEXT NOT NULL,
                  source_conv_id TEXT,
                  title TEXT NOT NULL,
                  objective TEXT NOT NULL DEFAULT '',
                  rows_json TEXT NOT NULL DEFAULT '[]',
                  evidence_json TEXT NOT NULL DEFAULT '[]',
                  source_items_json TEXT NOT NULL DEFAULT '[]',
                  comparison_flags_json TEXT NOT NULL DEFAULT '[]',
                  comparison_audits_json TEXT NOT NULL DEFAULT '[]',
                  quality_status TEXT NOT NULL DEFAULT 'draft',
                  quality_json TEXT NOT NULL DEFAULT '{}',
                  revision INTEGER NOT NULL DEFAULT 1,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
                  FOREIGN KEY(source_conv_id) REFERENCES conversations(id) ON DELETE SET NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_evidence_matrices_project_updated "
                "ON research_evidence_matrices(project_id, updated_at DESC);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_evidence_matrix_revisions (
                  matrix_id TEXT NOT NULL,
                  revision INTEGER NOT NULL,
                  title TEXT NOT NULL,
                  objective TEXT NOT NULL DEFAULT '',
                  rows_json TEXT NOT NULL DEFAULT '[]',
                  evidence_json TEXT NOT NULL DEFAULT '[]',
                  source_items_json TEXT NOT NULL DEFAULT '[]',
                  comparison_flags_json TEXT NOT NULL DEFAULT '[]',
                  comparison_audits_json TEXT NOT NULL DEFAULT '[]',
                  quality_status TEXT NOT NULL DEFAULT 'draft',
                  quality_json TEXT NOT NULL DEFAULT '{}',
                  created_at REAL NOT NULL,
                  PRIMARY KEY(matrix_id, revision),
                  FOREIGN KEY(matrix_id) REFERENCES research_evidence_matrices(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_evidence_matrix_revisions_created "
                "ON research_evidence_matrix_revisions(matrix_id, created_at DESC);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_evidence_watch_baselines (
                  matrix_id TEXT PRIMARY KEY,
                  project_id TEXT NOT NULL,
                  matrix_revision INTEGER NOT NULL,
                  snapshot_json TEXT NOT NULL DEFAULT '{}',
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
                  FOREIGN KEY(matrix_id) REFERENCES research_evidence_matrices(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_evidence_watch_events (
                  id TEXT PRIMARY KEY,
                  event_key TEXT NOT NULL UNIQUE,
                  project_id TEXT NOT NULL,
                  matrix_id TEXT NOT NULL,
                  matrix_revision INTEGER NOT NULL,
                  kind TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'open',
                  payload_json TEXT NOT NULL DEFAULT '{}',
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
                  FOREIGN KEY(matrix_id) REFERENCES research_evidence_matrices(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_evidence_watch_project_status "
                "ON research_evidence_watch_events(project_id, status, updated_at DESC);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_evidence_watch_matrix_status "
                "ON research_evidence_watch_events(matrix_id, status, updated_at DESC);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_gap_items (
                  id TEXT PRIMARY KEY,
                  gap_key TEXT NOT NULL UNIQUE,
                  project_id TEXT NOT NULL,
                  matrix_id TEXT,
                  brief_id TEXT,
                  kind TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'open',
                  payload_json TEXT NOT NULL DEFAULT '{}',
                  action_json TEXT NOT NULL DEFAULT '{}',
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_gap_project_status "
                "ON research_gap_items(project_id, status, updated_at DESC);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_gap_matrix_status "
                "ON research_gap_items(matrix_id, status, updated_at DESC);"
            )
            for table_name in ("research_evidence_matrices", "research_evidence_matrix_revisions"):
                try:
                    conn.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN comparison_audits_json TEXT NOT NULL DEFAULT '[]'"
                    )
                except sqlite3.OperationalError:
                    pass
            conn.execute(
                "DELETE FROM citation_shelves "
                "WHERE scope = 'conversation' AND scope_id NOT IN (SELECT id FROM conversations)"
            )
            conn.execute(
                "DELETE FROM message_refs "
                "WHERE NOT EXISTS (SELECT 1 FROM conversations WHERE conversations.id = message_refs.conv_id) "
                "OR NOT EXISTS ("
                "  SELECT 1 FROM messages "
                "  WHERE messages.id = message_refs.user_msg_id "
                "    AND messages.conv_id = message_refs.conv_id "
                "    AND messages.role = 'user'"
                ")"
            )

    def _archive_excess_conversations(
        self,
        conn: sqlite3.Connection,
        *,
        project_id: str | None,
        active_limit: int = DEFAULT_ACTIVE_CONVERSATION_LIMIT,
    ) -> int:
        keep_n = max(1, int(active_limit))
        if project_id is None:
            rows = conn.execute(
                """
                SELECT id
                FROM conversations
                WHERE project_id IS NULL AND COALESCE(archived, 0) = 0
                ORDER BY updated_at DESC, created_at DESC, id DESC
                LIMIT -1 OFFSET ?
                """,
                (keep_n,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id
                FROM conversations
                WHERE project_id = ? AND COALESCE(archived, 0) = 0
                ORDER BY updated_at DESC, created_at DESC, id DESC
                LIMIT -1 OFFSET ?
                """,
                (project_id, keep_n),
            ).fetchall()
        ids = [str(r["id"] or "").strip() for r in rows if str(r["id"] or "").strip()]
        if not ids:
            return 0
        now = time.time()
        conn.executemany(
            "UPDATE conversations SET archived = 1, archived_at = ? WHERE id = ?",
            [(now, cid) for cid in ids],
        )
        return len(ids)

    def _archive_excess_conversations_all_scopes(
        self,
        conn: sqlite3.Connection,
        *,
        active_limit: int = DEFAULT_ACTIVE_CONVERSATION_LIMIT,
    ) -> int:
        keep_n = max(1, int(active_limit))
        rows = conn.execute(
            """
            WITH ranked AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        PARTITION BY project_id
                        ORDER BY updated_at DESC, created_at DESC, id DESC
                    ) AS rn
                FROM conversations
                WHERE COALESCE(archived, 0) = 0
            )
            SELECT id
            FROM ranked
            WHERE rn > ?
            """,
            (keep_n,),
        ).fetchall()
        ids = [str(r["id"] or "").strip() for r in rows if str(r["id"] or "").strip()]
        if not ids:
            return 0
        now = time.time()
        conn.executemany(
            "UPDATE conversations SET archived = 1, archived_at = ? WHERE id = ?",
            [(now, cid) for cid in ids],
        )
        return len(ids)

    def _touch_conversation_active(self, conn: sqlite3.Connection, conv_id: str, now: float) -> str | None:
        row = conn.execute("SELECT project_id FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        if not row:
            return None
        project_id = row["project_id"]
        conn.execute(
            "UPDATE conversations SET updated_at = ?, archived = 0, archived_at = NULL WHERE id = ?",
            (now, conv_id),
        )
        return str(project_id).strip() if isinstance(project_id, str) and project_id.strip() else None

    def create_project(self, name: str) -> str:
        pid = uuid.uuid4().hex
        now = time.time()
        name = _clean_project_name(name, default="未命名项目")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO projects (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (pid, name, now, now),
            )
        return pid

    def list_projects(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, created_at, updated_at FROM projects ORDER BY updated_at DESC"
            ).fetchall()
        return [_project_record(r) for r in rows]

    def get_project(self, project_id: str) -> dict | None:
        if not (project_id or "").strip():
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, created_at, updated_at FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        return _project_record(row) if row else None

    def rename_project(self, project_id: str, name: str) -> bool:
        if not (project_id or "").strip():
            return False
        name = _clean_project_name(name)
        if not name:
            return False
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE projects SET name = ?, updated_at = ? WHERE id = ?",
                (name, now, project_id),
            )
        return cur.rowcount > 0

    def delete_project(self, project_id: str) -> bool:
        if not (project_id or "").strip():
            return False
        with self._connect() as conn:
            self._begin_immediate(conn)
            conn.execute("UPDATE conversations SET project_id = NULL WHERE project_id = ?", (project_id,))
            self._archive_excess_conversations(conn, project_id=None)
            conn.execute("DELETE FROM citation_shelves WHERE scope = 'project' AND scope_id = ?", (project_id,))
            conn.execute("DELETE FROM research_briefs WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM research_evidence_matrices WHERE project_id = ?", (project_id,))
            cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return cur.rowcount > 0

    @staticmethod
    def _research_brief_json_text(value: object, *, fallback: object) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)
        except Exception:
            return json.dumps(fallback, ensure_ascii=False)

    @staticmethod
    def _insert_research_brief_revision(conn: sqlite3.Connection, record: dict) -> None:
        conn.execute(
            """
            INSERT INTO research_brief_revisions (
              brief_id, revision, title, objective, content_markdown,
              evidence_json, bibliography_json, agent_trace_json,
              quality_status, quality_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.get("id") or ""),
                int(record.get("revision") or 1),
                str(record.get("title") or ""),
                str(record.get("objective") or ""),
                str(record.get("content_markdown") or ""),
                str(record.get("evidence_json") or "[]"),
                str(record.get("bibliography_json") or "[]"),
                str(record.get("agent_trace_json") or "{}"),
                str(record.get("quality_status") or "draft"),
                str(record.get("quality_json") or "{}"),
                float(record.get("updated_at") or record.get("created_at") or time.time()),
            ),
        )

    def create_research_brief(
        self,
        *,
        project_id: str,
        title: str,
        objective: str = "",
        content_markdown: str = "",
        source_conv_id: str | None = None,
        evidence: list[dict] | None = None,
        bibliography: list[dict] | None = None,
        agent_trace: dict | None = None,
        quality_status: str = "draft",
        quality: dict | None = None,
    ) -> dict | None:
        pid = str(project_id or "").strip()
        if not pid:
            return None
        brief_id = uuid.uuid4().hex
        now = time.time()
        clean_title = _research_brief_text(
            title,
            limit=MAX_RESEARCH_BRIEF_TITLE_LEN,
        ) or "Untitled research brief"
        clean_objective = _research_brief_text(
            objective,
            limit=MAX_RESEARCH_BRIEF_OBJECTIVE_LEN,
            multiline=True,
        )
        clean_content = _research_brief_text(
            content_markdown,
            limit=MAX_RESEARCH_BRIEF_CONTENT_LEN,
            multiline=True,
        )
        evidence_json = self._research_brief_json_text(list(evidence or []), fallback=[])
        bibliography_json = self._research_brief_json_text(list(bibliography or []), fallback=[])
        agent_trace_json = self._research_brief_json_text(dict(agent_trace or {}), fallback={})
        quality_json = self._research_brief_json_text(dict(quality or {}), fallback={})
        status = str(quality_status or "draft").strip().lower() or "draft"
        source_cid = str(source_conv_id or "").strip() or None
        with self._connect() as conn:
            self._begin_immediate(conn)
            if not self._project_exists(conn, pid):
                return None
            if source_cid:
                source_row = conn.execute(
                    "SELECT project_id FROM conversations WHERE id = ?",
                    (source_cid,),
                ).fetchone()
                if not source_row or str(source_row["project_id"] or "").strip() != pid:
                    source_cid = None
            record = {
                "id": brief_id,
                "project_id": pid,
                "source_conv_id": source_cid,
                "title": clean_title,
                "objective": clean_objective,
                "content_markdown": clean_content,
                "evidence_json": evidence_json,
                "bibliography_json": bibliography_json,
                "agent_trace_json": agent_trace_json,
                "quality_status": status,
                "quality_json": quality_json,
                "revision": 1,
                "created_at": now,
                "updated_at": now,
            }
            conn.execute(
                """
                INSERT INTO research_briefs (
                  id, project_id, source_conv_id, title, objective,
                  content_markdown, evidence_json, bibliography_json,
                  agent_trace_json, quality_status, quality_json, revision,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    brief_id,
                    pid,
                    source_cid,
                    clean_title,
                    clean_objective,
                    clean_content,
                    evidence_json,
                    bibliography_json,
                    agent_trace_json,
                    status,
                    quality_json,
                    1,
                    now,
                    now,
                ),
            )
            self._insert_research_brief_revision(conn, record)
        return _research_brief_record(record)

    def list_research_briefs(self, project_id: str, *, limit: int = 80) -> list[dict]:
        pid = str(project_id or "").strip()
        if not pid:
            return []
        lim = max(1, min(300, int(limit or 80)))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, project_id, source_conv_id, title, objective,
                       '' AS content_markdown, '[]' AS evidence_json,
                       '[]' AS bibliography_json, '{}' AS agent_trace_json,
                       quality_status, quality_json, revision,
                       created_at, updated_at
                FROM research_briefs
                WHERE project_id = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                (pid, lim),
            ).fetchall()
        return [_research_brief_record(row, include_content=False) for row in rows]

    def get_research_brief(self, brief_id: str) -> dict | None:
        bid = str(brief_id or "").strip()
        if not bid:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, project_id, source_conv_id, title, objective,
                       content_markdown, evidence_json, bibliography_json,
                       agent_trace_json, quality_status, quality_json, revision,
                       created_at, updated_at
                FROM research_briefs
                WHERE id = ?
                """,
                (bid,),
            ).fetchone()
        return _research_brief_record(row) if row else None

    def update_research_brief(
        self,
        brief_id: str,
        *,
        expected_revision: int | None = None,
        title: str | None = None,
        objective: str | None = None,
        content_markdown: str | None = None,
        evidence: list[dict] | None = None,
        bibliography: list[dict] | None = None,
        agent_trace: dict | None = None,
        quality_status: str | None = None,
        quality: dict | None = None,
    ) -> tuple[dict | None, bool]:
        bid = str(brief_id or "").strip()
        if not bid:
            return None, False
        with self._connect() as conn:
            self._begin_immediate(conn)
            row = conn.execute(
                "SELECT * FROM research_briefs WHERE id = ?",
                (bid,),
            ).fetchone()
            if not row:
                return None, False
            current = dict(row)
            current_revision = max(1, int(current.get("revision") or 1))
            if expected_revision is not None and int(expected_revision) != current_revision:
                return _research_brief_record(current), True
            next_record = dict(current)
            if title is not None:
                next_record["title"] = _research_brief_text(
                    title,
                    limit=MAX_RESEARCH_BRIEF_TITLE_LEN,
                ) or str(current.get("title") or "Untitled research brief")
            if objective is not None:
                next_record["objective"] = _research_brief_text(
                    objective,
                    limit=MAX_RESEARCH_BRIEF_OBJECTIVE_LEN,
                    multiline=True,
                )
            if content_markdown is not None:
                next_record["content_markdown"] = _research_brief_text(
                    content_markdown,
                    limit=MAX_RESEARCH_BRIEF_CONTENT_LEN,
                    multiline=True,
                )
            if evidence is not None:
                next_record["evidence_json"] = self._research_brief_json_text(list(evidence), fallback=[])
            if bibliography is not None:
                next_record["bibliography_json"] = self._research_brief_json_text(list(bibliography), fallback=[])
            if agent_trace is not None:
                next_record["agent_trace_json"] = self._research_brief_json_text(dict(agent_trace), fallback={})
            if quality_status is not None:
                next_record["quality_status"] = str(quality_status or "draft").strip().lower() or "draft"
            if quality is not None:
                next_record["quality_json"] = self._research_brief_json_text(dict(quality), fallback={})
            next_record["revision"] = current_revision + 1
            next_record["updated_at"] = time.time()
            conn.execute(
                """
                UPDATE research_briefs
                SET title = ?, objective = ?, content_markdown = ?,
                    evidence_json = ?, bibliography_json = ?, agent_trace_json = ?,
                    quality_status = ?, quality_json = ?, revision = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    next_record["title"],
                    next_record["objective"],
                    next_record["content_markdown"],
                    next_record["evidence_json"],
                    next_record["bibliography_json"],
                    next_record["agent_trace_json"],
                    next_record["quality_status"],
                    next_record["quality_json"],
                    next_record["revision"],
                    next_record["updated_at"],
                    bid,
                ),
            )
            self._insert_research_brief_revision(conn, next_record)
        return _research_brief_record(next_record), False

    def create_research_brief_update_plan(
        self,
        brief_id: str,
        *,
        expected_revision: int,
        matrix_id: str,
        matrix_revision: int,
        matrix_fingerprint: str,
        payload: dict,
    ) -> tuple[dict | None, bool]:
        bid = str(brief_id or "").strip()
        mid = str(matrix_id or "").strip()
        if not bid or not mid:
            return None, False
        plan_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as conn:
            self._begin_immediate(conn)
            brief = conn.execute(
                "SELECT revision FROM research_briefs WHERE id = ?",
                (bid,),
            ).fetchone()
            if brief is None:
                return None, False
            current_revision = max(1, int(brief["revision"] or 1))
            if current_revision != int(expected_revision):
                return {"brief_id": bid, "base_brief_revision": current_revision}, True
            conn.execute(
                "UPDATE research_brief_update_plans SET status = 'superseded', updated_at = ? "
                "WHERE brief_id = ? AND status = 'open'",
                (now, bid),
            )
            clean_payload = dict(payload or {})
            clean_payload["id"] = plan_id
            clean_payload["brief_id"] = bid
            payload_json = self._research_brief_json_text(clean_payload, fallback={})
            conn.execute(
                """
                INSERT INTO research_brief_update_plans (
                  id, brief_id, base_revision, matrix_id, matrix_revision,
                  matrix_fingerprint, payload_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)
                """,
                (
                    plan_id,
                    bid,
                    current_revision,
                    mid,
                    max(1, int(matrix_revision or 1)),
                    str(matrix_fingerprint or ""),
                    payload_json,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM research_brief_update_plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
        return (_research_brief_update_plan_record(row) if row else None), False

    def get_research_brief_update_plan(self, brief_id: str, plan_id: str) -> dict | None:
        bid = str(brief_id or "").strip()
        pid = str(plan_id or "").strip()
        if not bid or not pid:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_brief_update_plans WHERE id = ? AND brief_id = ?",
                (pid, bid),
            ).fetchone()
        return _research_brief_update_plan_record(row) if row else None

    def get_open_research_brief_update_plan(self, brief_id: str) -> dict | None:
        bid = str(brief_id or "").strip()
        if not bid:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM research_brief_update_plans
                WHERE brief_id = ? AND status = 'open'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (bid,),
            ).fetchone()
        return _research_brief_update_plan_record(row) if row else None

    def set_research_brief_update_plan_status(
        self,
        brief_id: str,
        plan_id: str,
        *,
        status: str,
    ) -> bool:
        bid = str(brief_id or "").strip()
        pid = str(plan_id or "").strip()
        next_status = str(status or "").strip().lower()
        if not bid or not pid or next_status not in {"applied", "discarded", "superseded"}:
            return False
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE research_brief_update_plans SET status = ?, updated_at = ? "
                "WHERE id = ? AND brief_id = ? AND status = 'open'",
                (next_status, time.time(), pid, bid),
            )
        return int(cur.rowcount or 0) > 0

    def list_research_brief_revisions(self, brief_id: str, *, limit: int = 40) -> list[dict]:
        bid = str(brief_id or "").strip()
        if not bid:
            return []
        lim = max(1, min(200, int(limit or 40)))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT brief_id AS id, revision, title, objective,
                       '' AS content_markdown, '[]' AS evidence_json,
                       '[]' AS bibliography_json, '{}' AS agent_trace_json,
                       quality_status, quality_json, created_at, created_at AS updated_at
                FROM research_brief_revisions
                WHERE brief_id = ?
                ORDER BY revision DESC
                LIMIT ?
                """,
                (bid, lim),
            ).fetchall()
        return [_research_brief_record(row, include_content=False) for row in rows]

    def get_research_brief_revision(self, brief_id: str, revision: int) -> dict | None:
        bid = str(brief_id or "").strip()
        rev = max(1, int(revision or 1))
        if not bid:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT brief_id AS id, revision, title, objective, content_markdown,
                       evidence_json, bibliography_json, agent_trace_json,
                       quality_status, quality_json, created_at, created_at AS updated_at
                FROM research_brief_revisions
                WHERE brief_id = ? AND revision = ?
                """,
                (bid, rev),
            ).fetchone()
        return _research_brief_record(row) if row else None

    def restore_research_brief_revision(
        self,
        brief_id: str,
        revision: int,
        *,
        expected_revision: int | None = None,
    ) -> tuple[dict | None, bool]:
        historical = self.get_research_brief_revision(brief_id, revision)
        if historical is None:
            return None, False
        return self.update_research_brief(
            brief_id,
            expected_revision=expected_revision,
            title=str(historical.get("title") or ""),
            objective=str(historical.get("objective") or ""),
            content_markdown=str(historical.get("content_markdown") or ""),
            evidence=list(historical.get("evidence") or []),
            bibliography=list(historical.get("bibliography") or []),
            agent_trace=dict(historical.get("agent_trace") or {}),
            quality_status=str(historical.get("quality_status") or "draft"),
            quality=dict(historical.get("quality") or {}),
        )

    def delete_research_brief(self, brief_id: str) -> bool:
        bid = str(brief_id or "").strip()
        if not bid:
            return False
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM research_briefs WHERE id = ?", (bid,))
        return int(cur.rowcount or 0) > 0

    @staticmethod
    def _insert_evidence_matrix_revision(conn: sqlite3.Connection, record: dict) -> None:
        conn.execute(
            """
            INSERT INTO research_evidence_matrix_revisions (
              matrix_id, revision, title, objective, rows_json, evidence_json,
              source_items_json, comparison_flags_json, comparison_audits_json,
              quality_status, quality_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.get("id") or ""),
                int(record.get("revision") or 1),
                str(record.get("title") or ""),
                str(record.get("objective") or ""),
                str(record.get("rows_json") or "[]"),
                str(record.get("evidence_json") or "[]"),
                str(record.get("source_items_json") or "[]"),
                str(record.get("comparison_flags_json") or "[]"),
                str(record.get("comparison_audits_json") or "[]"),
                str(record.get("quality_status") or "draft"),
                str(record.get("quality_json") or "{}"),
                float(record.get("updated_at") or record.get("created_at") or time.time()),
            ),
        )

    def create_evidence_matrix(
        self,
        *,
        project_id: str,
        title: str,
        objective: str = "",
        source_conv_id: str | None = None,
        rows: list[dict] | None = None,
        evidence: list[dict] | None = None,
        source_items: list[dict] | None = None,
        comparison_flags: list[dict] | None = None,
        comparison_audits: list[dict] | None = None,
        quality_status: str = "draft",
        quality: dict | None = None,
    ) -> dict | None:
        pid = str(project_id or "").strip()
        if not pid:
            return None
        matrix_id = uuid.uuid4().hex
        now = time.time()
        clean_title = _research_brief_text(
            title,
            limit=MAX_EVIDENCE_MATRIX_TITLE_LEN,
        ) or "Untitled evidence matrix"
        clean_objective = _research_brief_text(
            objective,
            limit=MAX_EVIDENCE_MATRIX_OBJECTIVE_LEN,
            multiline=True,
        )
        source_cid = str(source_conv_id or "").strip() or None
        record = {
            "id": matrix_id,
            "project_id": pid,
            "source_conv_id": source_cid,
            "title": clean_title,
            "objective": clean_objective,
            "rows_json": self._research_brief_json_text(list(rows or []), fallback=[]),
            "evidence_json": self._research_brief_json_text(list(evidence or []), fallback=[]),
            "source_items_json": self._research_brief_json_text(list(source_items or []), fallback=[]),
            "comparison_flags_json": self._research_brief_json_text(list(comparison_flags or []), fallback=[]),
            "comparison_audits_json": self._research_brief_json_text(list(comparison_audits or []), fallback=[]),
            "quality_status": str(quality_status or "draft").strip().lower() or "draft",
            "quality_json": self._research_brief_json_text(dict(quality or {}), fallback={}),
            "revision": 1,
            "created_at": now,
            "updated_at": now,
        }
        with self._connect() as conn:
            self._begin_immediate(conn)
            if not self._project_exists(conn, pid):
                return None
            if source_cid:
                source_row = conn.execute(
                    "SELECT project_id FROM conversations WHERE id = ?",
                    (source_cid,),
                ).fetchone()
                if not source_row or str(source_row["project_id"] or "").strip() != pid:
                    record["source_conv_id"] = None
            conn.execute(
                """
                INSERT INTO research_evidence_matrices (
                  id, project_id, source_conv_id, title, objective, rows_json,
                  evidence_json, source_items_json, comparison_flags_json, comparison_audits_json,
                  quality_status, quality_json, revision, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["project_id"],
                    record["source_conv_id"],
                    record["title"],
                    record["objective"],
                    record["rows_json"],
                    record["evidence_json"],
                    record["source_items_json"],
                    record["comparison_flags_json"],
                    record["comparison_audits_json"],
                    record["quality_status"],
                    record["quality_json"],
                    record["revision"],
                    record["created_at"],
                    record["updated_at"],
                ),
            )
            self._insert_evidence_matrix_revision(conn, record)
        return _evidence_matrix_record(record)

    def list_evidence_matrices(self, project_id: str, *, limit: int = 80) -> list[dict]:
        pid = str(project_id or "").strip()
        if not pid:
            return []
        lim = max(1, min(300, int(limit or 80)))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, project_id, source_conv_id, title, objective,
                       '[]' AS rows_json, '[]' AS evidence_json,
                       '[]' AS source_items_json, '[]' AS comparison_flags_json,
                       '[]' AS comparison_audits_json,
                       quality_status, quality_json, revision, created_at, updated_at
                FROM research_evidence_matrices
                WHERE project_id = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                (pid, lim),
            ).fetchall()
        return [_evidence_matrix_record(row, include_content=False) for row in rows]

    def get_evidence_matrix(self, matrix_id: str) -> dict | None:
        mid = str(matrix_id or "").strip()
        if not mid:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_evidence_matrices WHERE id = ?",
                (mid,),
            ).fetchone()
        return _evidence_matrix_record(row) if row else None

    def update_evidence_matrix(
        self,
        matrix_id: str,
        *,
        expected_revision: int | None = None,
        title: str | None = None,
        objective: str | None = None,
        rows: list[dict] | None = None,
        evidence: list[dict] | None = None,
        source_items: list[dict] | None = None,
        comparison_flags: list[dict] | None = None,
        comparison_audits: list[dict] | None = None,
        quality_status: str | None = None,
        quality: dict | None = None,
    ) -> tuple[dict | None, bool]:
        mid = str(matrix_id or "").strip()
        if not mid:
            return None, False
        with self._connect() as conn:
            self._begin_immediate(conn)
            row = conn.execute(
                "SELECT * FROM research_evidence_matrices WHERE id = ?",
                (mid,),
            ).fetchone()
            if not row:
                return None, False
            current = dict(row)
            current_revision = max(1, int(current.get("revision") or 1))
            if expected_revision is not None and int(expected_revision) != current_revision:
                return _evidence_matrix_record(current), True
            next_record = dict(current)
            if title is not None:
                next_record["title"] = _research_brief_text(
                    title,
                    limit=MAX_EVIDENCE_MATRIX_TITLE_LEN,
                ) or str(current.get("title") or "Untitled evidence matrix")
            if objective is not None:
                next_record["objective"] = _research_brief_text(
                    objective,
                    limit=MAX_EVIDENCE_MATRIX_OBJECTIVE_LEN,
                    multiline=True,
                )
            for key, value, fallback in (
                ("rows_json", rows, []),
                ("evidence_json", evidence, []),
                ("source_items_json", source_items, []),
                ("comparison_flags_json", comparison_flags, []),
                ("comparison_audits_json", comparison_audits, []),
                ("quality_json", quality, {}),
            ):
                if value is not None:
                    next_record[key] = self._research_brief_json_text(value, fallback=fallback)
            if quality_status is not None:
                next_record["quality_status"] = str(quality_status or "draft").strip().lower() or "draft"
            next_record["revision"] = current_revision + 1
            next_record["updated_at"] = time.time()
            conn.execute(
                """
                UPDATE research_evidence_matrices
                SET title = ?, objective = ?, rows_json = ?, evidence_json = ?,
                    source_items_json = ?, comparison_flags_json = ?, comparison_audits_json = ?,
                    quality_status = ?, quality_json = ?, revision = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    next_record["title"],
                    next_record["objective"],
                    next_record["rows_json"],
                    next_record["evidence_json"],
                    next_record["source_items_json"],
                    next_record["comparison_flags_json"],
                    next_record["comparison_audits_json"],
                    next_record["quality_status"],
                    next_record["quality_json"],
                    next_record["revision"],
                    next_record["updated_at"],
                    mid,
                ),
            )
            self._insert_evidence_matrix_revision(conn, next_record)
        return _evidence_matrix_record(next_record), False

    def list_evidence_matrix_revisions(self, matrix_id: str, *, limit: int = 40) -> list[dict]:
        mid = str(matrix_id or "").strip()
        if not mid:
            return []
        lim = max(1, min(200, int(limit or 40)))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT matrix_id AS id, revision, title, objective,
                       '[]' AS rows_json, '[]' AS evidence_json,
                       '[]' AS source_items_json, '[]' AS comparison_flags_json,
                       '[]' AS comparison_audits_json,
                       quality_status, quality_json, created_at, created_at AS updated_at
                FROM research_evidence_matrix_revisions
                WHERE matrix_id = ?
                ORDER BY revision DESC
                LIMIT ?
                """,
                (mid, lim),
            ).fetchall()
        return [_evidence_matrix_record(row, include_content=False) for row in rows]

    def get_evidence_matrix_revision(self, matrix_id: str, revision: int) -> dict | None:
        mid = str(matrix_id or "").strip()
        rev = max(1, int(revision or 1))
        if not mid:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT matrix_id AS id, revision, title, objective, rows_json,
                       evidence_json, source_items_json, comparison_flags_json, comparison_audits_json,
                       quality_status, quality_json, created_at, created_at AS updated_at
                FROM research_evidence_matrix_revisions
                WHERE matrix_id = ? AND revision = ?
                """,
                (mid, rev),
            ).fetchone()
        return _evidence_matrix_record(row) if row else None

    def restore_evidence_matrix_revision(
        self,
        matrix_id: str,
        revision: int,
        *,
        expected_revision: int | None = None,
    ) -> tuple[dict | None, bool]:
        historical = self.get_evidence_matrix_revision(matrix_id, revision)
        if historical is None:
            return None, False
        return self.update_evidence_matrix(
            matrix_id,
            expected_revision=expected_revision,
            title=str(historical.get("title") or ""),
            objective=str(historical.get("objective") or ""),
            rows=list(historical.get("rows") or []),
            evidence=list(historical.get("evidence") or []),
            source_items=list(historical.get("source_items") or []),
            comparison_flags=list(historical.get("comparison_flags") or []),
            comparison_audits=list(historical.get("comparison_audits") or []),
            quality_status=str(historical.get("quality_status") or "draft"),
            quality=dict(historical.get("quality") or {}),
        )

    def delete_evidence_matrix(self, matrix_id: str) -> bool:
        mid = str(matrix_id or "").strip()
        if not mid:
            return False
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM research_evidence_matrices WHERE id = ?", (mid,))
        return int(cur.rowcount or 0) > 0

    def get_evidence_watch_baseline(self, matrix_id: str) -> dict | None:
        mid = str(matrix_id or "").strip()
        if not mid:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT matrix_id, project_id, matrix_revision, snapshot_json, created_at, updated_at "
                "FROM research_evidence_watch_baselines WHERE matrix_id = ?",
                (mid,),
            ).fetchone()
        if not row:
            return None
        snapshot = _research_brief_json(row["snapshot_json"], default={})
        return {
            "matrix_id": str(row["matrix_id"] or ""),
            "project_id": str(row["project_id"] or ""),
            "matrix_revision": max(1, int(row["matrix_revision"] or 1)),
            "snapshot": snapshot if isinstance(snapshot, dict) else {},
            "created_at": float(row["created_at"] or 0.0),
            "updated_at": float(row["updated_at"] or 0.0),
        }

    def set_evidence_watch_baseline(
        self,
        matrix_id: str,
        *,
        project_id: str,
        matrix_revision: int,
        snapshot: dict,
    ) -> dict | None:
        mid = str(matrix_id or "").strip()
        pid = str(project_id or "").strip()
        if not mid or not pid:
            return None
        now = time.time()
        snapshot_json = self._research_brief_json_text(dict(snapshot or {}), fallback={})
        with self._connect() as conn:
            self._begin_immediate(conn)
            matrix = conn.execute(
                "SELECT project_id FROM research_evidence_matrices WHERE id = ?",
                (mid,),
            ).fetchone()
            if not matrix or str(matrix["project_id"] or "") != pid:
                return None
            conn.execute(
                """
                INSERT INTO research_evidence_watch_baselines (
                  matrix_id, project_id, matrix_revision, snapshot_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(matrix_id) DO UPDATE SET
                  project_id = excluded.project_id,
                  matrix_revision = excluded.matrix_revision,
                  snapshot_json = excluded.snapshot_json,
                  updated_at = excluded.updated_at
                """,
                (mid, pid, max(1, int(matrix_revision or 1)), snapshot_json, now, now),
            )
        return self.get_evidence_watch_baseline(mid)

    def sync_evidence_watch_events(
        self,
        *,
        project_id: str,
        matrix_id: str,
        matrix_revision: int,
        events: list[dict],
    ) -> list[dict]:
        pid = str(project_id or "").strip()
        mid = str(matrix_id or "").strip()
        revision = max(1, int(matrix_revision or 1))
        if not pid or not mid:
            return []
        active = [item for item in list(events or []) if isinstance(item, dict) and str(item.get("event_key") or "")]
        active_keys = {str(item.get("event_key") or "") for item in active}
        now = time.time()
        with self._connect() as conn:
            self._begin_immediate(conn)
            if active_keys:
                placeholders = ",".join("?" for _ in active_keys)
                conn.execute(
                    f"UPDATE research_evidence_watch_events SET status = 'resolved', updated_at = ? "
                    f"WHERE matrix_id = ? AND status = 'open' AND event_key NOT IN ({placeholders})",
                    (now, mid, *sorted(active_keys)),
                )
            else:
                conn.execute(
                    "UPDATE research_evidence_watch_events SET status = 'resolved', updated_at = ? "
                    "WHERE matrix_id = ? AND status = 'open'",
                    (now, mid),
                )
            for item in active:
                event_key = str(item.get("event_key") or "")
                payload = dict(item)
                payload_json = self._research_brief_json_text(payload, fallback={})
                conn.execute(
                    """
                    INSERT INTO research_evidence_watch_events (
                      id, event_key, project_id, matrix_id, matrix_revision, kind,
                      status, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
                    ON CONFLICT(event_key) DO UPDATE SET
                      project_id = excluded.project_id,
                      matrix_id = excluded.matrix_id,
                      matrix_revision = excluded.matrix_revision,
                      kind = excluded.kind,
                      status = CASE
                        WHEN research_evidence_watch_events.status IN ('resolved', 'applied') THEN 'open'
                        ELSE research_evidence_watch_events.status
                      END,
                      payload_json = excluded.payload_json,
                      created_at = CASE
                        WHEN research_evidence_watch_events.status IN ('resolved', 'applied') THEN excluded.created_at
                        ELSE research_evidence_watch_events.created_at
                      END,
                      updated_at = excluded.updated_at
                    """,
                    (
                        uuid.uuid4().hex,
                        event_key,
                        pid,
                        mid,
                        revision,
                        str(item.get("kind") or ""),
                        payload_json,
                        now,
                        now,
                    ),
                )
            rows = conn.execute(
                "SELECT * FROM research_evidence_watch_events "
                "WHERE matrix_id = ? AND status = 'open' ORDER BY updated_at DESC, created_at DESC",
                (mid,),
            ).fetchall()
        return [_evidence_watch_event_record(row) for row in rows]

    def list_project_evidence_watch_events(
        self,
        project_id: str,
        *,
        status: str = "open",
        limit: int = 200,
    ) -> list[dict]:
        pid = str(project_id or "").strip()
        status_norm = str(status or "open").strip().lower()
        if not pid or status_norm not in {"open", "ignored", "applied", "resolved"}:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM research_evidence_watch_events "
                "WHERE project_id = ? AND status = ? ORDER BY updated_at DESC, created_at DESC LIMIT ?",
                (pid, status_norm, max(1, min(500, int(limit or 200)))),
            ).fetchall()
        return [_evidence_watch_event_record(row) for row in rows]

    def get_evidence_watch_event(self, event_id: str) -> dict | None:
        eid = str(event_id or "").strip()
        if not eid:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_evidence_watch_events WHERE id = ?",
                (eid,),
            ).fetchone()
        return _evidence_watch_event_record(row) if row else None

    def set_evidence_watch_event_status(
        self,
        event_id: str,
        *,
        project_id: str,
        status: str,
    ) -> dict | None:
        eid = str(event_id or "").strip()
        pid = str(project_id or "").strip()
        status_norm = str(status or "").strip().lower()
        if not eid or not pid or status_norm not in {"ignored", "applied", "resolved"}:
            return None
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE research_evidence_watch_events SET status = ?, updated_at = ? "
                "WHERE id = ? AND project_id = ?",
                (status_norm, time.time(), eid, pid),
            )
            if int(cur.rowcount or 0) <= 0:
                return None
        return self.get_evidence_watch_event(eid)

    def resolve_matrix_evidence_watch_events(self, matrix_id: str, *, status: str = "applied") -> int:
        mid = str(matrix_id or "").strip()
        status_norm = str(status or "applied").strip().lower()
        if not mid or status_norm not in {"applied", "resolved"}:
            return 0
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE research_evidence_watch_events SET status = ?, updated_at = ? "
                "WHERE matrix_id = ? AND status = 'open'",
                (status_norm, time.time(), mid),
            )
        return int(cur.rowcount or 0)

    def sync_research_gap_items(self, *, project_id: str, gaps: list[dict]) -> list[dict]:
        pid = str(project_id or "").strip()
        if not pid:
            return []
        active = [item for item in list(gaps or []) if isinstance(item, dict) and str(item.get("gap_key") or "")]
        active_keys = {str(item.get("gap_key") or "") for item in active}
        now = time.time()
        with self._connect() as conn:
            self._begin_immediate(conn)
            if not self._project_exists(conn, pid):
                return []
            if active_keys:
                placeholders = ",".join("?" for _ in active_keys)
                conn.execute(
                    f"UPDATE research_gap_items SET status = 'resolved', updated_at = ? "
                    f"WHERE project_id = ? AND status IN ('open', 'in_progress') AND gap_key NOT IN ({placeholders})",
                    (now, pid, *sorted(active_keys)),
                )
            else:
                conn.execute(
                    "UPDATE research_gap_items SET status = 'resolved', updated_at = ? "
                    "WHERE project_id = ? AND status IN ('open', 'in_progress')",
                    (now, pid),
                )
            for item in active:
                gap_key = str(item.get("gap_key") or "")
                payload_json = self._research_brief_json_text(dict(item), fallback={})
                conn.execute(
                    """
                    INSERT INTO research_gap_items (
                      id, gap_key, project_id, matrix_id, brief_id, kind,
                      status, payload_json, action_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'open', ?, '{}', ?, ?)
                    ON CONFLICT(gap_key) DO UPDATE SET
                      project_id = excluded.project_id,
                      matrix_id = excluded.matrix_id,
                      brief_id = excluded.brief_id,
                      kind = excluded.kind,
                      status = CASE
                        WHEN research_gap_items.status = 'resolved' THEN 'open'
                        ELSE research_gap_items.status
                      END,
                      payload_json = excluded.payload_json,
                      created_at = CASE
                        WHEN research_gap_items.status = 'resolved' THEN excluded.created_at
                        ELSE research_gap_items.created_at
                      END,
                      updated_at = excluded.updated_at
                    """,
                    (
                        uuid.uuid4().hex,
                        gap_key,
                        pid,
                        str(item.get("matrix_id") or "") or None,
                        str(item.get("brief_id") or "") or None,
                        str(item.get("kind") or ""),
                        payload_json,
                        now,
                        now,
                    ),
                )
            rows = conn.execute(
                "SELECT * FROM research_gap_items WHERE project_id = ? "
                "AND status IN ('open', 'in_progress') "
                "ORDER BY json_extract(payload_json, '$.priority_score') DESC, updated_at DESC, created_at DESC",
                (pid,),
            ).fetchall()
        return [_research_gap_record(row) for row in rows]

    def list_project_research_gaps(
        self,
        project_id: str,
        *,
        status: str = "active",
        limit: int = 300,
    ) -> list[dict]:
        pid = str(project_id or "").strip()
        status_norm = str(status or "active").strip().lower()
        if not pid or status_norm not in {"active", "open", "in_progress", "ignored", "resolved"}:
            return []
        where_status = "status IN ('open', 'in_progress')" if status_norm == "active" else "status = ?"
        params: tuple[object, ...] = (pid,) if status_norm == "active" else (pid, status_norm)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM research_gap_items WHERE project_id = ? AND "
                + where_status
                + " ORDER BY json_extract(payload_json, '$.priority_score') DESC, updated_at DESC, created_at DESC LIMIT ?",
                (*params, max(1, min(1000, int(limit or 300)))),
            ).fetchall()
        return [_research_gap_record(row) for row in rows]

    def get_research_gap(self, gap_id: str) -> dict | None:
        gid = str(gap_id or "").strip()
        if not gid:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM research_gap_items WHERE id = ?", (gid,)).fetchone()
        return _research_gap_record(row) if row else None

    def set_research_gap_status(
        self,
        gap_id: str,
        *,
        project_id: str,
        status: str,
        action: dict | None = None,
    ) -> dict | None:
        gid = str(gap_id or "").strip()
        pid = str(project_id or "").strip()
        status_norm = str(status or "").strip().lower()
        if not gid or not pid or status_norm not in {"open", "in_progress", "ignored", "resolved"}:
            return None
        with self._connect() as conn:
            self._begin_immediate(conn)
            row = conn.execute(
                "SELECT action_json FROM research_gap_items WHERE id = ? AND project_id = ?",
                (gid, pid),
            ).fetchone()
            if not row:
                return None
            current_action = _research_brief_json(row["action_json"], default={})
            merged_action = dict(current_action) if isinstance(current_action, dict) else {}
            merged_action.update(dict(action or {}))
            cur = conn.execute(
                "UPDATE research_gap_items SET status = ?, action_json = ?, updated_at = ? "
                "WHERE id = ? AND project_id = ?",
                (
                    status_norm,
                    self._research_brief_json_text(merged_action, fallback={}),
                    time.time(),
                    gid,
                    pid,
                ),
            )
            if int(cur.rowcount or 0) <= 0:
                return None
        return self.get_research_gap(gid)

    def create_conversation(
        self,
        title: str = "新会话",
        project_id: str | None = None,
        *,
        mode: str = "normal",
        bound_source_path: str = "",
        bound_source_name: str = "",
        bound_source_ready: bool = False,
    ) -> str:
        conv_id = uuid.uuid4().hex
        now = time.time()
        mode_norm = _normalize_conversation_mode(mode)
        source_path = str(bound_source_path or "").strip()
        source_name = str(bound_source_name or "").strip()
        source_ready = 1 if bool(bound_source_ready and source_path) else 0
        project_id_norm = str(project_id or "").strip() or None
        with self._connect() as conn:
            if project_id_norm and not self._project_exists(conn, project_id_norm):
                project_id_norm = None
            conn.execute(
                "INSERT INTO conversations ("
                "id, title, created_at, updated_at, project_id, archived, archived_at, "
                "mode, bound_source_path, bound_source_name, bound_source_ready"
                ") VALUES (?, ?, ?, ?, ?, 0, NULL, ?, ?, ?, ?)",
                (
                    conv_id,
                    title.strip() or "新会话",
                    now,
                    now,
                    project_id_norm,
                    mode_norm,
                    source_path,
                    source_name,
                    source_ready,
                ),
            )
            if source_path:
                conn.execute(
                    """
                    INSERT INTO conversation_sources (conv_id, source_path, source_name, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(conv_id, source_path)
                    DO UPDATE SET source_name = excluded.source_name, updated_at = excluded.updated_at
                    """,
                    (conv_id, source_path, source_name or Path(source_path).name, now, now),
                )
            self._archive_excess_conversations(conn, project_id=project_id_norm)
        return conv_id

    def list_conversations(
        self,
        project_id: str | None = None,
        limit: int = 50,
        *,
        include_archived: bool = False,
    ) -> list[dict]:
        with self._connect() as conn:
            if not bool(include_archived):
                self._archive_excess_conversations(conn, project_id=project_id)
            if project_id is None:
                rows = conn.execute(
                    "SELECT id, title, created_at, updated_at, project_id, "
                    "COALESCE(archived, 0) AS archived, archived_at, "
                    "COALESCE(mode, 'normal') AS mode, "
                    "COALESCE(bound_source_path, '') AS bound_source_path, "
                    "COALESCE(bound_source_name, '') AS bound_source_name, "
                    "COALESCE(bound_source_ready, 0) AS bound_source_ready "
                    "FROM conversations "
                    "WHERE project_id IS NULL "
                    + ("" if include_archived else "AND COALESCE(archived, 0) = 0 ")
                    + "ORDER BY updated_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, title, created_at, updated_at, project_id, "
                    "COALESCE(archived, 0) AS archived, archived_at, "
                    "COALESCE(mode, 'normal') AS mode, "
                    "COALESCE(bound_source_path, '') AS bound_source_path, "
                    "COALESCE(bound_source_name, '') AS bound_source_name, "
                    "COALESCE(bound_source_ready, 0) AS bound_source_ready "
                    "FROM conversations "
                    "WHERE project_id = ? "
                    + ("" if include_archived else "AND COALESCE(archived, 0) = 0 ")
                    + "ORDER BY updated_at DESC LIMIT ?",
                    (project_id, int(limit)),
                ).fetchall()
        return [dict(r) for r in rows]

    def sidebar_snapshot(self, limit: int = 80, *, include_archived: bool = False) -> dict:
        lim = max(1, min(300, int(limit or 80)))
        conversation_columns = (
            "id, title, created_at, updated_at, project_id, "
            "COALESCE(archived, 0) AS archived, archived_at, "
            "COALESCE(mode, 'normal') AS mode, "
            "COALESCE(bound_source_path, '') AS bound_source_path, "
            "COALESCE(bound_source_name, '') AS bound_source_name, "
            "COALESCE(bound_source_ready, 0) AS bound_source_ready "
        )
        archive_filter = "" if include_archived else "AND COALESCE(archived, 0) = 0 "
        with self._connect() as conn:
            project_rows = conn.execute(
                "SELECT id, name, created_at, updated_at FROM projects ORDER BY updated_at DESC"
            ).fetchall()
            projects = [_project_record(r) for r in project_rows]
            project_ids = [str(project.get("id") or "").strip() for project in projects if str(project.get("id") or "").strip()]
            if not bool(include_archived):
                self._archive_excess_conversations_all_scopes(conn)
            rows = conn.execute(
                "WITH ranked AS ("
                "SELECT "
                + conversation_columns
                + ", ROW_NUMBER() OVER ("
                "PARTITION BY project_id "
                "ORDER BY updated_at DESC, created_at DESC, id DESC"
                ") AS group_rank "
                "FROM conversations "
                "WHERE (project_id IS NULL OR project_id IN (SELECT id FROM projects)) "
                + archive_filter
                + ") "
                "SELECT id, title, created_at, updated_at, project_id, "
                "archived, archived_at, mode, bound_source_path, bound_source_name, bound_source_ready "
                "FROM ranked "
                "WHERE group_rank <= ? "
                "ORDER BY project_id IS NOT NULL, project_id, updated_at DESC, created_at DESC, id DESC",
                (lim,),
            ).fetchall()
            project_conversations: dict[str, list[dict]] = {project_id: [] for project_id in project_ids}
            root_conversations: list[dict] = []
            for row in rows:
                rec = dict(row)
                project_id = str(rec.get("project_id") or "").strip()
                if project_id and project_id in project_conversations:
                    project_conversations[project_id].append(rec)
                else:
                    root_conversations.append(rec)
        return {
            "projects": projects,
            "root_conversations": root_conversations,
            "project_conversations": project_conversations,
        }

    def get_conversation(self, conv_id: str, *, timeout_s: float | None = None) -> dict | None:
        conv_id = (conv_id or "").strip()
        if not conv_id:
            return None
        with self._connect(timeout_s=timeout_s) as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at, project_id, "
                "COALESCE(archived, 0) AS archived, archived_at, "
                "COALESCE(mode, 'normal') AS mode, "
                "COALESCE(bound_source_path, '') AS bound_source_path, "
                "COALESCE(bound_source_name, '') AS bound_source_name, "
                "COALESCE(bound_source_ready, 0) AS bound_source_ready "
                "FROM conversations WHERE id = ?",
                (conv_id,),
            ).fetchone()
        return dict(row) if row else None

    def set_conversation_guide(
        self,
        conv_id: str,
        *,
        mode: str | None = None,
        bound_source_path: str | None = None,
        bound_source_name: str | None = None,
        bound_source_ready: bool | None = None,
    ) -> bool:
        cid = str(conv_id or "").strip()
        if not cid:
            return False
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT mode, bound_source_path, bound_source_name, bound_source_ready, project_id "
                "FROM conversations WHERE id = ?",
                (cid,),
            ).fetchone()
            if not row:
                return False
            mode_cur = _normalize_conversation_mode(str(row["mode"] or "normal"))
            path_cur = str(row["bound_source_path"] or "").strip()
            name_cur = str(row["bound_source_name"] or "").strip()
            ready_cur = bool(int(row["bound_source_ready"] or 0))
            mode_next = mode_cur if mode is None else _normalize_conversation_mode(mode)
            path_next = path_cur if bound_source_path is None else str(bound_source_path or "").strip()
            name_next = name_cur if bound_source_name is None else str(bound_source_name or "").strip()
            if bound_source_ready is None:
                ready_next = ready_cur
            else:
                ready_next = bool(bound_source_ready and path_next)
            if not path_next:
                ready_next = False
            conn.execute(
                "UPDATE conversations "
                "SET mode = ?, bound_source_path = ?, bound_source_name = ?, bound_source_ready = ?, "
                "updated_at = ?, archived = 0, archived_at = NULL "
                "WHERE id = ?",
                (
                    mode_next,
                    path_next,
                    name_next,
                    1 if ready_next else 0,
                    now,
                    cid,
                ),
            )
            if path_next:
                conn.execute(
                    """
                    INSERT INTO conversation_sources (conv_id, source_path, source_name, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(conv_id, source_path)
                    DO UPDATE SET source_name = excluded.source_name, updated_at = excluded.updated_at
                    """,
                    (cid, path_next, name_next or Path(path_next).name, now, now),
                )
            project_id = str(row["project_id"] or "").strip() or None
            self._archive_excess_conversations(conn, project_id=project_id)
        return True

    def set_conversation_project(self, conv_id: str, project_id: str | None) -> bool:
        conv_id = (conv_id or "").strip()
        if not conv_id:
            return False
        project_id_norm = str(project_id or "").strip() or None
        now = time.time()
        with self._connect() as conn:
            self._begin_immediate(conn)
            old = conn.execute("SELECT project_id FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            if not old:
                return False
            if project_id_norm and not self._project_exists(conn, project_id_norm):
                return False
            old_project_id = old["project_id"]
            cur = conn.execute(
                "UPDATE conversations SET project_id = ?, updated_at = ?, archived = 0, archived_at = NULL WHERE id = ?",
                (project_id_norm, now, conv_id),
            )
            old_pid = str(old_project_id).strip() if isinstance(old_project_id, str) and old_project_id.strip() else None
            self._archive_excess_conversations(conn, project_id=project_id_norm)
            if old_pid != project_id_norm:
                self._archive_excess_conversations(conn, project_id=old_pid)
        return cur.rowcount > 0

    def delete_conversation(self, conv_id: str) -> bool:
        with self._connect() as conn:
            self._begin_immediate(conn)
            conn.execute("DELETE FROM conversation_reader_states WHERE conv_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversation_research_states WHERE conv_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversation_sources WHERE conv_id = ?", (conv_id,))
            conn.execute("DELETE FROM citation_shelves WHERE scope = 'conversation' AND scope_id = ?", (conv_id,))
            conn.execute("DELETE FROM message_refs WHERE conv_id = ?", (conv_id,))
            conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        return cur.rowcount > 0

    def bind_conversation_source(self, conv_id: str, source_path: str, source_name: str = "") -> bool:
        cid = str(conv_id or "").strip()
        src = str(source_path or "").strip()
        if (not cid) or (not src):
            return False
        name = str(source_name or "").strip() or Path(src).name
        now = time.time()
        with self._connect() as conn:
            self._begin_immediate(conn)
            if not conn.execute("SELECT 1 FROM conversations WHERE id = ?", (cid,)).fetchone():
                return False
            conn.execute(
                """
                INSERT INTO conversation_sources (conv_id, source_path, source_name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(conv_id, source_path)
                DO UPDATE SET source_name = excluded.source_name, updated_at = excluded.updated_at
                """,
                (cid, src, name, now, now),
            )
            project_id = self._touch_conversation_active(conn, cid, now)
            self._archive_excess_conversations(conn, project_id=project_id)
        return True

    def list_conversation_sources(self, conv_id: str, limit: int = 8) -> list[dict]:
        cid = str(conv_id or "").strip()
        if not cid:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_path, source_name, created_at, updated_at
                FROM conversation_sources
                WHERE conv_id = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (cid, max(1, int(limit))),
            ).fetchall()
        return [
            {
                "source_path": str(r["source_path"] or ""),
                "source_name": str(r["source_name"] or ""),
                "created_at": float(r["created_at"] or 0.0),
                "updated_at": float(r["updated_at"] or 0.0),
            }
            for r in rows
        ]

    def get_conversation_reader_state(self, conv_id: str, source_path: str) -> dict | None:
        cid = str(conv_id or "").strip()
        src = str(source_path or "").strip()
        src_key = _normalize_reader_source_path_key(src)
        if (not cid) or (not src_key):
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT conv_id, source_path, state_json, created_at, updated_at
                FROM conversation_reader_states
                WHERE conv_id = ? AND source_path = ?
                """,
                (cid, src_key),
            ).fetchone()
            if not row:
                candidates = conn.execute(
                    """
                    SELECT conv_id, source_path, state_json, created_at, updated_at
                    FROM conversation_reader_states
                    WHERE conv_id = ?
                    ORDER BY updated_at DESC
                    """,
                    (cid,),
                ).fetchall()
                for candidate in candidates:
                    if _normalize_reader_source_path_key(candidate["source_path"]) == src_key:
                        row = candidate
                        break
        if not row:
            return {
                "conv_id": cid,
                "source_path": src,
                "state": {},
                "created_at": 0.0,
                "updated_at": 0.0,
            }
        try:
            state = json.loads(row["state_json"] or "{}")
        except Exception:
            state = {}
        state = _sanitize_json_state_dict(state)
        return {
            "conv_id": str(row["conv_id"] or ""),
            "source_path": src,
            "state": state,
            "created_at": float(row["created_at"] or 0.0),
            "updated_at": float(row["updated_at"] or 0.0),
        }

    def patch_conversation_reader_state(self, conv_id: str, source_path: str, patch: dict) -> dict | None:
        cid = str(conv_id or "").strip()
        src = str(source_path or "").strip()
        src_key = _normalize_reader_source_path_key(src)
        if (not cid) or (not src_key):
            return None
        patch_dict = _sanitize_json_state_dict(patch)
        now = time.time()
        with self._connect() as conn:
            self._begin_immediate(conn)
            conv = conn.execute("SELECT id FROM conversations WHERE id = ?", (cid,)).fetchone()
            if not conv:
                return None
            row = conn.execute(
                """
                SELECT state_json, created_at
                FROM conversation_reader_states
                WHERE conv_id = ? AND source_path = ?
                """,
                (cid, src_key),
            ).fetchone()
            legacy_source_path = ""
            if not row:
                candidates = conn.execute(
                    """
                    SELECT source_path, state_json, created_at
                    FROM conversation_reader_states
                    WHERE conv_id = ?
                    ORDER BY updated_at DESC
                    """,
                    (cid,),
                ).fetchall()
                for candidate in candidates:
                    candidate_source = str(candidate["source_path"] or "")
                    if _normalize_reader_source_path_key(candidate_source) == src_key:
                        row = candidate
                        legacy_source_path = candidate_source
                        break
            if row:
                try:
                    current = json.loads(row["state_json"] or "{}")
                except Exception:
                    current = {}
                current = _sanitize_json_state_dict(current)
                created_at = float(row["created_at"] or now)
            else:
                current = {}
                created_at = now
            current = _apply_json_state_patch(current, patch_dict)
            try:
                state_json = json.dumps(current, ensure_ascii=False, allow_nan=False, default=str)
            except Exception:
                state_json = "{}"
                current = {}
            conn.execute(
                """
                INSERT INTO conversation_reader_states (conv_id, source_path, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(conv_id, source_path)
                DO UPDATE SET state_json = excluded.state_json, updated_at = excluded.updated_at
                """,
                (cid, src_key, state_json, created_at, now),
            )
            if legacy_source_path and legacy_source_path != src_key:
                conn.execute(
                    "DELETE FROM conversation_reader_states WHERE conv_id = ? AND source_path = ?",
                    (cid, legacy_source_path),
                )
        return {
            "conv_id": cid,
            "source_path": src,
            "state": current,
            "created_at": created_at,
            "updated_at": now,
        }

    def get_conversation_research_state(self, conv_id: str) -> dict | None:
        cid = str(conv_id or "").strip()
        if not cid:
            return None
        with self._connect() as conn:
            conv = conn.execute("SELECT id FROM conversations WHERE id = ?", (cid,)).fetchone()
            if not conv:
                return None
            row = conn.execute(
                """
                SELECT conv_id, state_json, created_at, updated_at
                FROM conversation_research_states
                WHERE conv_id = ?
                """,
                (cid,),
            ).fetchone()
        if not row:
            return {
                "conv_id": cid,
                "state": {},
                "created_at": 0.0,
                "updated_at": 0.0,
            }
        try:
            state = json.loads(row["state_json"] or "{}")
        except Exception:
            state = {}
        state = _sanitize_json_state_dict(state)
        return {
            "conv_id": str(row["conv_id"] or ""),
            "state": state,
            "created_at": float(row["created_at"] or 0.0),
            "updated_at": float(row["updated_at"] or 0.0),
        }

    def patch_conversation_research_state(self, conv_id: str, patch: dict) -> dict | None:
        cid = str(conv_id or "").strip()
        if not cid:
            return None
        patch_dict = _sanitize_json_state_dict(patch)
        now = time.time()
        with self._connect() as conn:
            self._begin_immediate(conn)
            conv = conn.execute("SELECT id FROM conversations WHERE id = ?", (cid,)).fetchone()
            if not conv:
                return None
            row = conn.execute(
                """
                SELECT state_json, created_at
                FROM conversation_research_states
                WHERE conv_id = ?
                """,
                (cid,),
            ).fetchone()
            if row:
                try:
                    current = json.loads(row["state_json"] or "{}")
                except Exception:
                    current = {}
                current = _sanitize_json_state_dict(current)
                created_at = float(row["created_at"] or now)
            else:
                current = {}
                created_at = now
            current = _apply_json_state_patch(current, patch_dict)
            try:
                state_json = json.dumps(current, ensure_ascii=False, allow_nan=False, default=str)
            except Exception:
                state_json = "{}"
                current = {}
            conn.execute(
                """
                INSERT INTO conversation_research_states (conv_id, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conv_id)
                DO UPDATE SET state_json = excluded.state_json, updated_at = excluded.updated_at
                """,
                (cid, state_json, created_at, now),
            )
        return {
            "conv_id": cid,
            "state": current,
            "created_at": created_at,
            "updated_at": now,
        }

    def _resolve_citation_shelf_scope(
        self,
        conn: sqlite3.Connection,
        *,
        conv_id: str | None = None,
        project_id: str | None = None,
        scope: str = "project",
    ) -> tuple[str, str, str | None] | None:
        scope_norm = str(scope or "").strip().lower()
        if scope_norm in {"conversation", "conv"}:
            scope_norm = "conversation"
        else:
            scope_norm = "project"

        cid = str(conv_id or "").strip()
        pid = str(project_id or "").strip() or None
        if cid:
            row = conn.execute("SELECT project_id FROM conversations WHERE id = ?", (cid,)).fetchone()
            if not row:
                return None
            if not pid:
                pid = str(row["project_id"] or "").strip() or None

        if scope_norm == "conversation":
            if not cid:
                return None
            return scope_norm, cid, pid
        return scope_norm, pid or "__default__", pid

    def _citation_shelf_empty_record(
        self,
        *,
        scope: str,
        scope_id: str,
        project_id: str | None = None,
        created_at: float = 0.0,
        updated_at: float = 0.0,
        revision: int = 0,
    ) -> dict:
        return {
            "version": 1,
            "scope": scope,
            "scope_id": scope_id,
            "project_id": project_id,
            "items": [],
            "open": False,
            "revision": int(revision or 0),
            "created_at": float(created_at or 0.0),
            "updated_at": float(updated_at or 0.0),
        }

    def _hydrate_citation_shelf_row(
        self,
        row: sqlite3.Row | None,
        *,
        scope: str,
        scope_id: str,
        project_id: str | None = None,
    ) -> dict:
        if not row:
            return self._citation_shelf_empty_record(scope=scope, scope_id=scope_id, project_id=project_id)
        try:
            items = json.loads(row["items_json"] or "[]")
        except Exception:
            items = []
        if not isinstance(items, list):
            items = []
        return {
            "version": 1,
            "scope": str(row["scope"] or scope),
            "scope_id": str(row["scope_id"] or scope_id),
            "project_id": project_id,
            "items": _normalize_citation_shelf_items([item for item in items if isinstance(item, dict)]),
            "open": bool(int(row["open"] or 0)),
            "revision": int(row["revision"] or 0),
            "created_at": float(row["created_at"] or 0.0),
            "updated_at": float(row["updated_at"] or 0.0),
        }

    def get_citation_shelf(
        self,
        *,
        conv_id: str | None = None,
        project_id: str | None = None,
        scope: str = "project",
    ) -> dict | None:
        with self._connect() as conn:
            resolved = self._resolve_citation_shelf_scope(
                conn,
                conv_id=conv_id,
                project_id=project_id,
                scope=scope,
            )
            if resolved is None:
                return None
            scope_norm, scope_id, resolved_project_id = resolved
            row = conn.execute(
                """
                SELECT scope, scope_id, items_json, open, revision, created_at, updated_at
                FROM citation_shelves
                WHERE scope = ? AND scope_id = ?
                """,
                (scope_norm, scope_id),
            ).fetchone()
        return self._hydrate_citation_shelf_row(
            row,
            scope=scope_norm,
            scope_id=scope_id,
            project_id=resolved_project_id,
        )

    def save_citation_shelf(
        self,
        *,
        items: list[dict],
        open: bool = False,
        conv_id: str | None = None,
        project_id: str | None = None,
        scope: str = "project",
        allow_empty_overwrite: bool = False,
    ) -> dict | None:
        normalized_items = _normalize_citation_shelf_items(items)
        try:
            items_json = json.dumps(normalized_items, ensure_ascii=False, default=str)
        except Exception:
            normalized_items = []
            items_json = "[]"
        open_int = 1 if bool(open) else 0
        now = time.time()
        with self._connect() as conn:
            self._begin_immediate(conn)
            resolved = self._resolve_citation_shelf_scope(
                conn,
                conv_id=conv_id,
                project_id=project_id,
                scope=scope,
            )
            if resolved is None:
                return None
            scope_norm, scope_id, resolved_project_id = resolved
            row = conn.execute(
                """
                SELECT scope, scope_id, items_json, open, revision, created_at, updated_at
                FROM citation_shelves
                WHERE scope = ? AND scope_id = ?
                """,
                (scope_norm, scope_id),
            ).fetchone()
            if row and not normalized_items and not open_int and not allow_empty_overwrite:
                try:
                    existing_items = json.loads(row["items_json"] or "[]")
                except Exception:
                    existing_items = []
                if isinstance(existing_items, list) and any(isinstance(item, dict) for item in existing_items):
                    return self._hydrate_citation_shelf_row(
                        row,
                        scope=scope_norm,
                        scope_id=scope_id,
                        project_id=resolved_project_id,
                    )
            if row and str(row["items_json"] or "[]") == items_json and int(row["open"] or 0) == open_int:
                return self._hydrate_citation_shelf_row(
                    row,
                    scope=scope_norm,
                    scope_id=scope_id,
                    project_id=resolved_project_id,
                )
            created_at = float(row["created_at"] or now) if row else now
            revision = int(row["revision"] or 0) + 1 if row else 1
            conn.execute(
                """
                INSERT INTO citation_shelves (scope, scope_id, items_json, open, revision, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope, scope_id)
                DO UPDATE SET
                  items_json = excluded.items_json,
                  open = excluded.open,
                  revision = excluded.revision,
                  updated_at = excluded.updated_at
                """,
                (scope_norm, scope_id, items_json, open_int, revision, created_at, now),
            )
        return {
            "version": 1,
            "scope": scope_norm,
            "scope_id": scope_id,
            "project_id": resolved_project_id,
            "items": normalized_items,
            "open": bool(open_int),
            "revision": revision,
            "created_at": created_at,
            "updated_at": now,
        }

    def append_citation_shelf_item(
        self,
        *,
        item: dict,
        open: bool = True,
        conv_id: str | None = None,
        project_id: str | None = None,
        scope: str = "project",
    ) -> dict | None:
        normalized_new = _normalize_citation_shelf_items([item])
        if not normalized_new:
            return self.get_citation_shelf(conv_id=conv_id, project_id=project_id, scope=scope)
        new_item = normalized_new[0]
        with self._connect() as conn:
            self._begin_immediate(conn)
            resolved = self._resolve_citation_shelf_scope(
                conn,
                conv_id=conv_id,
                project_id=project_id,
                scope=scope,
            )
            if resolved is None:
                return None
            scope_norm, scope_id, resolved_project_id = resolved
            row = conn.execute(
                """
                SELECT scope, scope_id, items_json, open, revision, created_at, updated_at
                FROM citation_shelves
                WHERE scope = ? AND scope_id = ?
                """,
                (scope_norm, scope_id),
            ).fetchone()
            current = self._hydrate_citation_shelf_row(
                row,
                scope=scope_norm,
                scope_id=scope_id,
                project_id=resolved_project_id,
            )
            current_items = [entry for entry in list(current.get("items") or []) if isinstance(entry, dict)]
            new_identity = _shelf_item_identity(new_item)
            new_stable_key = _shelf_item_stable_key(new_item)
            existing_index = next(
                (
                    idx
                    for idx, existing in enumerate(current_items)
                    if (
                        (new_identity and _shelf_item_identity(existing) == new_identity)
                        or (new_stable_key and _shelf_item_stable_key(existing) == new_stable_key)
                    )
                ),
                -1,
            )
            if existing_index >= 0:
                existing_item = _merge_citation_shelf_item(current_items[existing_index], new_item)
                next_items = _normalize_citation_shelf_items([
                    existing_item,
                    *current_items[:existing_index],
                    *current_items[existing_index + 1:],
                ])
            else:
                next_items = _normalize_citation_shelf_items([new_item, *current_items])
            next_open = bool(open or current.get("open"))
            if next_items == list(current.get("items") or []) and next_open == bool(current.get("open")):
                return current
            now = time.time()
            created_at = float(current.get("created_at") or now)
            revision = int(current.get("revision") or 0) + 1
            conn.execute(
                """
                INSERT INTO citation_shelves (scope, scope_id, items_json, open, revision, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope, scope_id)
                DO UPDATE SET
                  items_json = excluded.items_json,
                  open = excluded.open,
                  revision = excluded.revision,
                  updated_at = excluded.updated_at
                """,
                (
                    scope_norm,
                    scope_id,
                    json.dumps(next_items, ensure_ascii=False, default=str),
                    1 if next_open else 0,
                    revision,
                    created_at,
                    now,
                ),
            )
        return {
            "version": 1,
            "scope": scope_norm,
            "scope_id": scope_id,
            "project_id": resolved_project_id,
            "items": next_items,
            "open": next_open,
            "revision": revision,
            "created_at": created_at,
            "updated_at": now,
        }

    def delete_citation_shelf(
        self,
        *,
        conv_id: str | None = None,
        project_id: str | None = None,
        scope: str = "project",
    ) -> dict | None:
        with self._connect() as conn:
            self._begin_immediate(conn)
            resolved = self._resolve_citation_shelf_scope(
                conn,
                conv_id=conv_id,
                project_id=project_id,
                scope=scope,
            )
            if resolved is None:
                return None
            scope_norm, scope_id, resolved_project_id = resolved
            conn.execute(
                "DELETE FROM citation_shelves WHERE scope = ? AND scope_id = ?",
                (scope_norm, scope_id),
            )
        return self._citation_shelf_empty_record(
            scope=scope_norm,
            scope_id=scope_id,
            project_id=resolved_project_id,
        )

    def get_messages(self, conv_id: str, limit: int | None = None) -> list[dict]:
        sql = "SELECT id, role, content, attachments_json, meta_json, created_at FROM messages WHERE conv_id = ? ORDER BY id ASC"
        params: tuple = (conv_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (conv_id, int(limit))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return self._hydrate_message_rows(rows)

    def get_messages_upto_id(self, conv_id: str, max_id: int, limit: int | None = None) -> list[dict]:
        mid = int(max_id or 0)
        if mid <= 0:
            return self.get_messages(conv_id, limit=limit)
        sql = "SELECT id, role, content, attachments_json, meta_json, created_at FROM messages WHERE conv_id = ? AND id <= ? ORDER BY id ASC"
        params: tuple = (conv_id, mid)
        if limit is not None:
            sql += " LIMIT ?"
            params = (conv_id, mid, int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return self._hydrate_message_rows(rows)

    def get_message(self, message_id: int) -> dict | None:
        try:
            mid = int(message_id or 0)
        except Exception:
            return None
        if mid <= 0:
            return None
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, conv_id, role, content, attachments_json, meta_json, created_at FROM messages WHERE id = ?",
                (mid,),
            ).fetchall()
        messages = self._hydrate_message_rows(rows)
        return messages[0] if messages else None

    def get_messages_page(
        self,
        conv_id: str,
        limit: int = 24,
        before_id: int | None = None,
    ) -> tuple[list[dict], bool, int | None, int | None]:
        page_size = max(1, min(200, int(limit or 24)))
        before = int(before_id or 0)
        params: tuple
        sql = (
            "SELECT id, role, content, attachments_json, meta_json, created_at "
            "FROM messages WHERE conv_id = ?"
        )
        params = (conv_id,)
        if before > 0:
            sql += " AND id < ?"
            params = (conv_id, before)
        sql += " ORDER BY id DESC LIMIT ?"
        params = (*params, page_size + 1)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        has_more_before = len(rows) > page_size
        page_rows = list(rows[:page_size])
        page_rows.reverse()
        out = self._hydrate_message_rows(page_rows)
        oldest_loaded_id = int(out[0]["id"]) if out else None
        newest_loaded_id = int(out[-1]["id"]) if out else None
        return out, has_more_before, oldest_loaded_id, newest_loaded_id

    def message_exists(self, message_id: int) -> bool:
        try:
            mid = int(message_id or 0)
        except Exception:
            return False
        if mid <= 0:
            return False
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM messages WHERE id = ?", (mid,)).fetchone()
        return row is not None

    def _hydrate_message_rows(self, rows: list[sqlite3.Row]) -> list[dict]:
        out: list[dict] = []
        for row in rows:
            rec = dict(row)
            try:
                attachments = json.loads(rec.get("attachments_json") or "[]")
            except Exception:
                attachments = []
            if not isinstance(attachments, list):
                attachments = []
            try:
                meta = json.loads(rec.get("meta_json") or "{}")
            except Exception:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            rec["attachments"] = attachments
            rec["meta"] = meta
            if isinstance(meta.get("provenance"), dict):
                rec["provenance"] = dict(meta.get("provenance") or {})
            rec.pop("attachments_json", None)
            rec.pop("meta_json", None)
            out.append(rec)
        return out

    def append_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        attachments: list[dict] | None = None,
        meta: dict | None = None,
    ) -> int:
        role = (role or "").strip()
        if role not in ("user", "assistant", "system"):
            role = "user"
        content = (content or "").strip()
        try:
            attachments_json = json.dumps(list(attachments or []), ensure_ascii=False, default=str)
        except Exception:
            attachments_json = "[]"
        try:
            meta_json = json.dumps(dict(meta or {}), ensure_ascii=False, default=str)
        except Exception:
            meta_json = "{}"
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages (conv_id, role, content, attachments_json, meta_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (conv_id, role, content, attachments_json, meta_json, now),
            )
            project_id = self._touch_conversation_active(conn, conv_id, now)
            self._archive_excess_conversations(conn, project_id=project_id)
            try:
                return int(cur.lastrowid or 0)
            except Exception:
                return 0

    def grounded_answer_count(self) -> int:
        """Count assistant answers paired with a non-empty retrieval reference set."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS answer_count
                FROM messages AS assistant
                WHERE assistant.role = 'assistant'
                  AND length(trim(assistant.content)) > 0
                  AND EXISTS (
                    SELECT 1
                    FROM message_refs AS refs
                    WHERE refs.conv_id = assistant.conv_id
                      AND refs.user_msg_id = (
                        SELECT MAX(user_message.id)
                        FROM messages AS user_message
                        WHERE user_message.conv_id = assistant.conv_id
                          AND user_message.role = 'user'
                          AND user_message.id < assistant.id
                      )
                      AND trim(refs.hits_json) NOT IN ('', '[]', '{}', 'null')
                  )
                """
            ).fetchone()
        return int(row["answer_count"] or 0) if row else 0

    def update_message_content(self, message_id: int, content: str, *, touch_conversation: bool = False) -> bool:
        mid = int(message_id or 0)
        if mid <= 0:
            return False
        text = (content or "").strip()
        with self._connect() as conn:
            row = conn.execute("SELECT conv_id, meta_json FROM messages WHERE id = ?", (mid,)).fetchone()
            if not row:
                return False
            try:
                meta = json.loads(row["meta_json"] or "{}")
            except Exception:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            meta.pop("render_cache", None)
            try:
                meta_json = json.dumps(meta, ensure_ascii=False, default=str)
            except Exception:
                meta_json = "{}"
            conn.execute("UPDATE messages SET content = ?, meta_json = ? WHERE id = ?", (text, meta_json, mid))
            if touch_conversation:
                now = time.time()
                project_id = self._touch_conversation_active(conn, str(row["conv_id"] or ""), now)
                self._archive_excess_conversations(conn, project_id=project_id)
        return True

    def update_message_meta(self, message_id: int, meta: dict, *, touch_conversation: bool = False) -> bool:
        mid = int(message_id or 0)
        if mid <= 0:
            return False
        try:
            meta_json = json.dumps(dict(meta or {}), ensure_ascii=False, default=str)
        except Exception:
            meta_json = "{}"
        with self._connect() as conn:
            row = conn.execute("SELECT conv_id FROM messages WHERE id = ?", (mid,)).fetchone()
            if not row:
                return False
            conn.execute("UPDATE messages SET meta_json = ? WHERE id = ?", (meta_json, mid))
            if touch_conversation:
                now = time.time()
                project_id = self._touch_conversation_active(conn, str(row["conv_id"] or ""), now)
                self._archive_excess_conversations(conn, project_id=project_id)
        return True

    def merge_message_meta(self, message_id: int, patch: dict, *, touch_conversation: bool = False) -> bool:
        mid = int(message_id or 0)
        if mid <= 0:
            return False
        patch_dict = dict(patch or {})
        with self._connect() as conn:
            row = conn.execute("SELECT conv_id, meta_json FROM messages WHERE id = ?", (mid,)).fetchone()
            if not row:
                return False
            try:
                current = json.loads(row["meta_json"] or "{}")
            except Exception:
                current = {}
            if not isinstance(current, dict):
                current = {}
            current.update(patch_dict)
            try:
                meta_json = json.dumps(current, ensure_ascii=False, default=str)
            except Exception:
                meta_json = "{}"
            conn.execute("UPDATE messages SET meta_json = ? WHERE id = ?", (meta_json, mid))
            if touch_conversation:
                now = time.time()
                project_id = self._touch_conversation_active(conn, str(row["conv_id"] or ""), now)
                self._archive_excess_conversations(conn, project_id=project_id)
        return True

    def set_message_render_cache(self, message_id: int, cache_payload: dict | None) -> bool:
        mid = int(message_id or 0)
        if mid <= 0:
            return False
        with self._connect() as conn:
            row = conn.execute("SELECT meta_json FROM messages WHERE id = ?", (mid,)).fetchone()
            if not row:
                return False
            try:
                current = json.loads(row["meta_json"] or "{}")
            except Exception:
                current = {}
            if not isinstance(current, dict):
                current = {}
            next_meta = dict(current)
            if isinstance(cache_payload, dict) and cache_payload:
                next_meta["render_cache"] = dict(cache_payload)
            else:
                next_meta.pop("render_cache", None)
            if next_meta == current:
                return True
            try:
                meta_json = json.dumps(next_meta, ensure_ascii=False, default=str)
            except Exception:
                meta_json = "{}"
            conn.execute("UPDATE messages SET meta_json = ? WHERE id = ?", (meta_json, mid))
        return True

    def delete_message(self, message_id: int) -> bool:
        mid = int(message_id or 0)
        if mid <= 0:
            return False
        now = time.time()
        with self._connect() as conn:
            row = conn.execute("SELECT conv_id FROM messages WHERE id = ?", (mid,)).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM message_refs WHERE user_msg_id = ?", (mid,))
            conn.execute("DELETE FROM messages WHERE id = ?", (mid,))
            project_id = self._touch_conversation_active(conn, str(row["conv_id"] or ""), now)
            self._archive_excess_conversations(conn, project_id=project_id)
        return True

    def _message_refs_owner_exists(self, conn: sqlite3.Connection, *, user_msg_id: int, conv_id: str) -> bool:
        mid = int(user_msg_id or 0)
        cid = (conv_id or "").strip()
        if mid <= 0 or not cid:
            return False
        row = conn.execute(
            "SELECT 1 FROM messages WHERE id = ? AND conv_id = ? AND role = 'user'",
            (mid, cid),
        ).fetchone()
        return row is not None

    def _delete_message_refs_if_orphaned(self, conn: sqlite3.Connection, *, user_msg_id: int, conv_id: str) -> bool:
        if self._message_refs_owner_exists(conn, user_msg_id=user_msg_id, conv_id=conv_id):
            return False
        conn.execute("DELETE FROM message_refs WHERE user_msg_id = ?", (int(user_msg_id or 0),))
        return True

    def upsert_message_refs(
        self,
        *,
        user_msg_id: int,
        conv_id: str,
        prompt: str,
        prompt_sig: str,
        hits: list[dict],
        scores: list[float],
        used_query: str,
        used_translation: bool,
        rendered_payload: dict | None = None,
        rendered_payload_sig: str = "",
        render_status: str | None = None,
        render_error: str | None = None,
        render_error_detail: str | None = None,
        render_built_at: float | None = None,
        render_attempts: int | None = None,
        render_evidence_sig: str | None = None,
        render_locale: str | None = None,
        query_variants: list[str] | None = None,
        skip_if_rendered_full: bool = False,
    ) -> bool:
        mid = int(user_msg_id or 0)
        if mid <= 0:
            return False
        now = time.time()
        conv_id = (conv_id or "").strip()
        prompt = (prompt or "").strip()
        prompt_sig = (prompt_sig or "").strip()
        used_query = (used_query or "").strip()
        try:
            hits_json = json.dumps(list(hits or []), ensure_ascii=False, default=str)
        except Exception:
            hits_json = "[]"
        try:
            scores_json = json.dumps(list(scores or []), ensure_ascii=False, default=str)
        except Exception:
            scores_json = "[]"
        try:
            rendered_payload_json = (
                json.dumps(dict(rendered_payload or {}), ensure_ascii=False, default=str)
                if isinstance(rendered_payload, dict)
                else ""
            )
        except Exception:
            rendered_payload_json = ""
        try:
            query_variants_json = json.dumps(list(query_variants or []), ensure_ascii=False)
        except Exception:
            query_variants_json = "[]"
        rendered_sig = str(rendered_payload_sig or "").strip() if rendered_payload_json else ""
        with self._connect() as conn:
            if not self._message_refs_owner_exists(conn, user_msg_id=mid, conv_id=conv_id):
                conn.execute("DELETE FROM message_refs WHERE user_msg_id = ?", (mid,))
                return False
            row = conn.execute(
                """
                SELECT user_msg_id, created_at, render_status, render_error, render_error_detail,
                       render_built_at, render_attempts, render_evidence_sig, render_locale
                FROM message_refs
                WHERE user_msg_id = ?
                """,
                (mid,),
            ).fetchone()
            next_render_status = str(render_status).strip() if render_status is not None else str((row["render_status"] if row else "") or "").strip()
            next_render_error = str(render_error).strip() if render_error is not None else str((row["render_error"] if row else "") or "").strip()
            next_render_error_detail = (
                str(render_error_detail).strip()
                if render_error_detail is not None
                else str((row["render_error_detail"] if row else "") or "").strip()
            )
            try:
                next_render_built_at = float(render_built_at) if render_built_at is not None else float((row["render_built_at"] if row else 0.0) or 0.0)
            except Exception:
                next_render_built_at = 0.0
            try:
                next_render_attempts = int(render_attempts) if render_attempts is not None else int((row["render_attempts"] if row else 0) or 0)
            except Exception:
                next_render_attempts = 0
            next_render_attempts = max(0, next_render_attempts)
            next_render_evidence_sig = (
                str(render_evidence_sig).strip()
                if render_evidence_sig is not None
                else str((row["render_evidence_sig"] if row else "") or "").strip()
            )
            next_render_locale = (
                str(render_locale).strip()
                if render_locale is not None
                else str((row["render_locale"] if row else "") or "").strip()
            )
            if row:
                created_at = float(row["created_at"] or now)
                cursor = conn.execute(
                    """
                    UPDATE message_refs
                    SET conv_id = ?, prompt = ?, prompt_sig = ?, hits_json = ?, scores_json = ?,
                        rendered_payload_json = ?, rendered_payload_sig = ?,
                        render_status = ?, render_error = ?, render_error_detail = ?,
                        render_built_at = ?, render_attempts = ?, render_evidence_sig = ?, render_locale = ?,
                        used_query = ?, used_translation = ?, query_variants_json = ?, updated_at = ?
                    WHERE user_msg_id = ?
                      AND (? = 0 OR LOWER(TRIM(render_status)) <> 'full')
                    """,
                    (
                        conv_id,
                        prompt,
                        prompt_sig,
                        hits_json,
                        scores_json,
                        rendered_payload_json,
                        rendered_sig,
                        next_render_status,
                        next_render_error,
                        next_render_error_detail,
                        next_render_built_at,
                        next_render_attempts,
                        next_render_evidence_sig,
                        next_render_locale,
                        used_query,
                        1 if bool(used_translation) else 0,
                        query_variants_json,
                        now,
                        mid,
                        1 if bool(skip_if_rendered_full) else 0,
                    ),
                )
                if bool(skip_if_rendered_full) and int(cursor.rowcount or 0) <= 0:
                    return False
            else:
                created_at = now
                conn.execute(
                    """
                    INSERT INTO message_refs
                    (
                        user_msg_id, conv_id, prompt, prompt_sig, hits_json, scores_json,
                        rendered_payload_json, rendered_payload_sig,
                        render_status, render_error, render_error_detail,
                        render_built_at, render_attempts, render_evidence_sig, render_locale,
                        used_query, used_translation, query_variants_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mid,
                        conv_id,
                        prompt,
                        prompt_sig,
                        hits_json,
                        scores_json,
                        rendered_payload_json,
                        rendered_sig,
                        next_render_status,
                        next_render_error,
                        next_render_error_detail,
                        next_render_built_at,
                        next_render_attempts,
                        next_render_evidence_sig,
                        next_render_locale,
                        used_query,
                        1 if bool(used_translation) else 0,
                        query_variants_json,
                        created_at,
                        now,
                    ),
                )
        return True

    def set_message_refs_rendered_payload(
        self,
        *,
        user_msg_id: int,
        rendered_payload: dict | None,
        rendered_payload_sig: str = "",
        render_status: str | None = None,
        render_error: str | None = None,
        render_error_detail: str | None = None,
        render_built_at: float | None = None,
        render_attempts: int | None = None,
        render_evidence_sig: str | None = None,
        render_locale: str | None = None,
    ) -> bool:
        mid = int(user_msg_id or 0)
        if mid <= 0:
            return False
        try:
            rendered_payload_json = (
                json.dumps(dict(rendered_payload or {}), ensure_ascii=False, default=str)
                if isinstance(rendered_payload, dict)
                else ""
            )
        except Exception:
            rendered_payload_json = ""
        rendered_sig = str(rendered_payload_sig or "").strip() if rendered_payload_json else ""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT user_msg_id, conv_id, render_status, render_error, render_error_detail,
                       render_built_at, render_attempts, render_evidence_sig, render_locale
                FROM message_refs
                WHERE user_msg_id = ?
                """,
                (mid,),
            ).fetchone()
            if not row:
                return False
            if self._delete_message_refs_if_orphaned(conn, user_msg_id=mid, conv_id=str(row["conv_id"] or "")):
                return False
            next_render_status = str(render_status).strip() if render_status is not None else str(row["render_status"] or "").strip()
            next_render_error = str(render_error).strip() if render_error is not None else str(row["render_error"] or "").strip()
            next_render_error_detail = (
                str(render_error_detail).strip()
                if render_error_detail is not None
                else str(row["render_error_detail"] or "").strip()
            )
            try:
                next_render_built_at = float(render_built_at) if render_built_at is not None else float(row["render_built_at"] or 0.0)
            except Exception:
                next_render_built_at = 0.0
            try:
                next_render_attempts = int(render_attempts) if render_attempts is not None else int(row["render_attempts"] or 0)
            except Exception:
                next_render_attempts = 0
            next_render_attempts = max(0, next_render_attempts)
            next_render_evidence_sig = (
                str(render_evidence_sig).strip()
                if render_evidence_sig is not None
                else str(row["render_evidence_sig"] or "").strip()
            )
            next_render_locale = (
                str(render_locale).strip()
                if render_locale is not None
                else str(row["render_locale"] or "").strip()
            )
            conn.execute(
                """
                UPDATE message_refs
                SET rendered_payload_json = ?, rendered_payload_sig = ?,
                    render_status = ?, render_error = ?, render_error_detail = ?,
                    render_built_at = ?, render_attempts = ?, render_evidence_sig = ?, render_locale = ?
                WHERE user_msg_id = ?
                """,
                (
                    rendered_payload_json,
                    rendered_sig,
                    next_render_status,
                    next_render_error,
                    next_render_error_detail,
                    next_render_built_at,
                    next_render_attempts,
                    next_render_evidence_sig,
                    next_render_locale,
                    mid,
                ),
            )
        return True

    def set_message_refs_render_state(
        self,
        *,
        user_msg_id: int,
        render_status: str | None = None,
        render_error: str | None = None,
        render_error_detail: str | None = None,
        render_built_at: float | None = None,
        render_attempts: int | None = None,
        render_evidence_sig: str | None = None,
        render_locale: str | None = None,
    ) -> bool:
        mid = int(user_msg_id or 0)
        if mid <= 0:
            return False
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT user_msg_id, conv_id, render_status, render_error, render_error_detail,
                       render_built_at, render_attempts, render_evidence_sig, render_locale
                FROM message_refs
                WHERE user_msg_id = ?
                """,
                (mid,),
            ).fetchone()
            if not row:
                return False
            if self._delete_message_refs_if_orphaned(conn, user_msg_id=mid, conv_id=str(row["conv_id"] or "")):
                return False
            next_render_status = str(render_status).strip() if render_status is not None else str(row["render_status"] or "").strip()
            next_render_error = str(render_error).strip() if render_error is not None else str(row["render_error"] or "").strip()
            next_render_error_detail = (
                str(render_error_detail).strip()
                if render_error_detail is not None
                else str(row["render_error_detail"] or "").strip()
            )
            try:
                next_render_built_at = float(render_built_at) if render_built_at is not None else float(row["render_built_at"] or 0.0)
            except Exception:
                next_render_built_at = 0.0
            try:
                next_render_attempts = int(render_attempts) if render_attempts is not None else int(row["render_attempts"] or 0)
            except Exception:
                next_render_attempts = 0
            next_render_attempts = max(0, next_render_attempts)
            next_render_evidence_sig = (
                str(render_evidence_sig).strip()
                if render_evidence_sig is not None
                else str(row["render_evidence_sig"] or "").strip()
            )
            next_render_locale = (
                str(render_locale).strip()
                if render_locale is not None
                else str(row["render_locale"] or "").strip()
            )
            conn.execute(
                """
                UPDATE message_refs
                SET render_status = ?, render_error = ?, render_error_detail = ?,
                    render_built_at = ?, render_attempts = ?, render_evidence_sig = ?, render_locale = ?
                WHERE user_msg_id = ?
                """,
                (
                    next_render_status,
                    next_render_error,
                    next_render_error_detail,
                    next_render_built_at,
                    next_render_attempts,
                    next_render_evidence_sig,
                    next_render_locale,
                    mid,
                ),
            )
        return True

    def list_message_refs_state(
        self,
        conv_id: str,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, object]:
        """Return a small cache-validation snapshot without loading refs JSON."""

        with self._connect(timeout_s=timeout_s) as conn:
            rows = conn.execute(
                """
                SELECT user_msg_id, prompt_sig, rendered_payload_sig,
                       render_status, render_error, render_built_at,
                       render_attempts, render_evidence_sig, render_locale,
                       used_query, used_translation, created_at, updated_at,
                       LENGTH(hits_json) AS hits_json_chars,
                       LENGTH(scores_json) AS scores_json_chars,
                       LENGTH(rendered_payload_json) AS rendered_payload_json_chars,
                       LENGTH(query_variants_json) AS query_variants_json_chars
                FROM message_refs
                WHERE conv_id = ?
                ORDER BY user_msg_id ASC
                """,
                (conv_id,),
            ).fetchall()
            message_row = conn.execute(
                """
                SELECT COUNT(*) AS message_count,
                       COALESCE(MAX(id), 0) AS max_message_id,
                       COALESCE(SUM(LENGTH(content)), 0) AS content_chars,
                       COALESCE(SUM(LENGTH(meta_json)), 0) AS meta_chars
                FROM messages
                WHERE conv_id = ?
                """,
                (conv_id,),
            ).fetchone()

        state_rows: list[dict[str, object]] = []
        for row in rows:
            state_rows.append(
                {
                    "user_msg_id": int(row["user_msg_id"] or 0),
                    "prompt_sig": str(row["prompt_sig"] or ""),
                    "rendered_payload_sig": str(row["rendered_payload_sig"] or ""),
                    "render_status": str(row["render_status"] or ""),
                    "render_error": str(row["render_error"] or ""),
                    "render_built_at": float(row["render_built_at"] or 0.0),
                    "render_attempts": int(row["render_attempts"] or 0),
                    "render_evidence_sig": str(row["render_evidence_sig"] or ""),
                    "render_locale": str(row["render_locale"] or ""),
                    "used_query": str(row["used_query"] or ""),
                    "used_translation": bool(int(row["used_translation"] or 0)),
                    "created_at": float(row["created_at"] or 0.0),
                    "updated_at": float(row["updated_at"] or 0.0),
                    "hits_json_chars": int(row["hits_json_chars"] or 0),
                    "scores_json_chars": int(row["scores_json_chars"] or 0),
                    "rendered_payload_json_chars": int(
                        row["rendered_payload_json_chars"] or 0
                    ),
                    "query_variants_json_chars": int(
                        row["query_variants_json_chars"] or 0
                    ),
                }
            )
        message_state = {
            "message_count": int((message_row["message_count"] if message_row else 0) or 0),
            "max_message_id": int((message_row["max_message_id"] if message_row else 0) or 0),
            "content_chars": int((message_row["content_chars"] if message_row else 0) or 0),
            "meta_chars": int((message_row["meta_chars"] if message_row else 0) or 0),
        }
        return {"rows": state_rows, "messages": message_state}

    def list_message_refs(self, conv_id: str, *, timeout_s: float | None = None) -> dict[int, dict]:
        with self._connect(timeout_s=timeout_s) as conn:
            rows = conn.execute(
                """
                SELECT mr.user_msg_id, mr.conv_id, mr.prompt, mr.prompt_sig, mr.hits_json, mr.scores_json,
                       mr.rendered_payload_json, mr.rendered_payload_sig,
                       mr.render_status, mr.render_error, mr.render_error_detail,
                       mr.render_built_at, mr.render_attempts, mr.render_evidence_sig, mr.render_locale,
                       mr.used_query, mr.used_translation, mr.query_variants_json, mr.created_at, mr.updated_at
                FROM message_refs AS mr
                JOIN messages AS m
                  ON m.id = mr.user_msg_id
                 AND m.conv_id = mr.conv_id
                 AND m.role = 'user'
                WHERE mr.conv_id = ?
                ORDER BY mr.user_msg_id ASC
                """,
                (conv_id,),
            ).fetchall()
        out: dict[int, dict] = {}
        for r in rows:
            try:
                mid = int(r["user_msg_id"] or 0)
            except Exception:
                mid = 0
            if mid <= 0:
                continue
            try:
                hits = json.loads(r["hits_json"] or "[]")
            except Exception:
                hits = []
            if not isinstance(hits, list):
                hits = []
            try:
                scores = json.loads(r["scores_json"] or "[]")
            except Exception:
                scores = []
            if not isinstance(scores, list):
                scores = []
            try:
                rendered_payload = json.loads(r["rendered_payload_json"] or "{}")
            except Exception:
                rendered_payload = {}
            if not isinstance(rendered_payload, dict):
                rendered_payload = {}
            out[mid] = {
                "user_msg_id": mid,
                "conv_id": str(r["conv_id"] or ""),
                "prompt": str(r["prompt"] or ""),
                "prompt_sig": str(r["prompt_sig"] or ""),
                "hits": hits,
                "scores": scores,
                "rendered_payload": rendered_payload,
                "rendered_payload_sig": str(r["rendered_payload_sig"] or ""),
                "render_status": str(r["render_status"] or ""),
                "render_error": str(r["render_error"] or ""),
                "render_error_detail": str(r["render_error_detail"] or ""),
                "render_built_at": float(r["render_built_at"] or 0.0),
                "render_attempts": int(r["render_attempts"] or 0),
                "render_evidence_sig": str(r["render_evidence_sig"] or ""),
                "render_locale": str(r["render_locale"] or ""),
                "used_query": str(r["used_query"] or ""),
                "used_translation": bool(int(r["used_translation"] or 0)),
                "query_variants": json.loads(r["query_variants_json"] or "[]") if isinstance(r["query_variants_json"], str) and r["query_variants_json"].strip() else [],
                "created_at": float(r["created_at"] or 0.0),
                "updated_at": float(r["updated_at"] or 0.0),
            }
        return out

    def set_title_if_default(self, conv_id: str, new_title: str) -> bool:
        new_title = (new_title or "").strip()
        if not new_title:
            return False
        new_title = new_title.replace("\n", " ").strip()
        new_title = new_title[:80]

        with self._connect() as conn:
            row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            if not row:
                return False
            if not _is_default_conversation_title(str(row["title"] or "")):
                return False
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (new_title, conv_id),
            )
        return True

    def set_title(self, conv_id: str, new_title: str) -> bool:
        cid = (conv_id or "").strip()
        title = (new_title or "").replace("\n", " ").strip()[:80]
        if not cid or not title:
            return False
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM conversations WHERE id = ?", (cid,)).fetchone()
            if not row:
                return False
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, cid),
            )
        return True
