from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from api.message_render_contract import (
    build_render_cache_payload as _contract_build_render_cache_payload,
    content_has_linkable_answer_citations as _contract_content_has_linkable_answer_citations,
    count_linkable_source_hits as _contract_count_linkable_source_hits,
    iter_numeric_citation_numbers as _contract_iter_numeric_citation_numbers,
    normalize_render_cache_payload,
    project_render_packet_to_record,
    render_payload_has_citation_links,
    render_payload_is_degraded_for_citations,
    strip_legacy_render_fields,
)
from api.deps import load_prefs
from api.reference_card_quality import attach_refs_pack_polish_contract
from kb import task_runtime
from kb.paper_guide_contracts import (
    _build_paper_guide_render_packet_model,
    _paper_guide_model_dump,
    _paper_guide_promote_hidden_direct_segment_for_render,
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
from kb.citation_card import compose_citation_card
from kb.inpaper_citation_enrichment import (
    enrich_inpaper_detail_context,
    extract_structured_cite_answer_context_line,
)
from kb.evidence_text import (
    clean_display_text as _clean_evidence_display_text,
    evidence_sentence_quality as _evidence_sentence_quality,
    looks_low_value_citation_context as _looks_low_value_citation_context,
    pick_readable_evidence_text as _pick_readable_evidence_text,
)
from kb.config import load_settings
from kb.reference_index import extract_references_map_from_md, load_reference_index, resolve_reference_entry
from kb.markdown_rendering import _md_to_plain_text, _normalize_copy_citation_links, _normalize_math_markdown
from api.reference_rendering import (
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


def _call_with_optional_render_locale(func, *args, render_locale: str = "", **kwargs):
    call_kwargs = dict(kwargs)
    locale = str(render_locale or "").strip()
    if locale:
        try:
            params = inspect.signature(func).parameters
            accepts_locale = "render_locale" in params or any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
            )
        except (TypeError, ValueError):
            accepts_locale = True
        if accepts_locale:
            call_kwargs["render_locale"] = locale
    return func(*args, **call_kwargs)
_REF_MAP_CACHE: dict[str, dict[int, str]] = {}
# Bump whenever citation rendering/card contracts change in a way that should
# repair historical conversations on the next page load.
_RENDER_CACHE_SCHEMA_VERSION = 21


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


def _render_primary_source_identity(raw: dict | None) -> str:
    if not isinstance(raw, dict):
        return ""
    # The same local paper can have a storage filename and a richer metadata
    # display name. Canonical paths are stable across those presentation forms.
    for key in ("source_path", "sourcePath", "source_name", "sourceName"):
        text = str(raw.get(key) or "").strip().lower()
        if not text:
            continue
        name = Path(text).name or text
        for suffix in (".en.md", ".md", ".pdf"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return " ".join(name.replace("_", " ").replace("-", " ").split())
    return ""


def _render_primary_heading_identity(raw: dict | None) -> str:
    if not isinstance(raw, dict):
        return ""
    return str(raw.get("heading_path") or raw.get("headingPath") or "").strip().lower()


def _primary_evidence_is_compatible(base: dict | None, candidate: dict | None) -> bool:
    if not isinstance(base, dict) or not isinstance(candidate, dict):
        return False
    base_source = _render_primary_source_identity(base)
    cand_source = _render_primary_source_identity(candidate)
    if base_source and cand_source and base_source != cand_source:
        return False
    base_heading = _render_primary_heading_identity(base)
    cand_heading = _render_primary_heading_identity(candidate)
    if base_heading and cand_heading and base_heading != cand_heading:
        return False
    return True


def _primary_evidence_precision_score(raw: dict | None) -> tuple[int, int, int, int, int, int]:
    if not isinstance(raw, dict) or not raw:
        return (0, 0, 0, 0, 0, 0)
    reason = str(raw.get("selection_reason") or raw.get("selectionReason") or "").strip().lower()
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
    strict_locate = raw.get("strict_locate")
    if strict_locate is None:
        strict_locate = raw.get("strictLocate")
    return (
        reason_rank,
        1 if str(raw.get("block_id") or raw.get("blockId") or "").strip() else 0,
        1 if str(raw.get("anchor_id") or raw.get("anchorId") or "").strip() else 0,
        1 if _render_primary_heading_identity(raw) else 0,
        1 if bool(strict_locate) else 0,
        1 if str(raw.get("source_path") or raw.get("sourcePath") or raw.get("source_name") or raw.get("sourceName") or "").strip() else 0,
    )


def _should_adopt_refs_primary(contract_primary: dict | None, refs_primary: dict | None) -> bool:
    if not isinstance(refs_primary, dict) or not refs_primary:
        return False
    if not isinstance(contract_primary, dict) or not contract_primary:
        return True
    current_score = _primary_evidence_precision_score(contract_primary)
    refs_score = _primary_evidence_precision_score(refs_primary)
    if refs_score <= current_score:
        return False
    current_reason = str(contract_primary.get("selection_reason") or contract_primary.get("selectionReason") or "").strip().lower()
    if current_reason in {"answer_hit_top", "shared_contract_seed", "pending_section_seed"}:
        return True
    if not _primary_evidence_is_compatible(contract_primary, refs_primary):
        return True
    return refs_score > current_score


def _merge_render_packet_primary_evidence(
    *,
    contract_primary: dict | None,
    provenance_primary: dict | None,
    existing_primary: dict | None,
) -> dict:
    base = dict(contract_primary or {}) if isinstance(contract_primary, dict) and contract_primary else {}
    if not base:
        if isinstance(provenance_primary, dict) and provenance_primary:
            base = dict(provenance_primary)
        elif isinstance(existing_primary, dict) and existing_primary:
            base = dict(existing_primary)
    for candidate in (provenance_primary, existing_primary):
        if not isinstance(candidate, dict) or not candidate:
            continue
        if not base:
            base = dict(candidate)
            continue
        if not _primary_evidence_is_compatible(base, candidate):
            continue
        for key, value in dict(candidate).items():
            if base.get(key) in (None, "", [], {}):
                base[key] = value
    return base


def _primary_evidence_text(raw: dict | None) -> str:
    if not isinstance(raw, dict):
        return ""
    return str(
        raw.get("highlight_snippet")
        or raw.get("highlightSnippet")
        or raw.get("snippet")
        or ""
    ).strip()


_REF_PRIMARY_LOW_VALUE_RE = re.compile(
    r"(?i)(?:"
    r"no\s+summary\s+available|metadata\s+only|only\s+metadata|"
    r"source\s+excerpt\s+says\s*[:：].*(?:\.\.\.|\u2026)|"
    r"\u539f\u6587\u7247\u6bb5\u5199\u5230\s*[:：].*(?:\.\.\.|\u2026)|"
    r"\u8fd9\u7bc7\u6587\u732e\u5f53\u524d\u7f3a\u5c11\u53ef\u7528\u6458\u8981|"
    r"\u4ec5\u6839\u636e\u5143\u6570\u636e|"
    r"\u5f53\u524d\u4ec5\u68c0\u7d22\u5230\u6587\u732e\u5143\u6570\u636e"
    r")"
)


def _primary_evidence_candidates_from_ref_hit(hit: dict | None) -> list[dict]:
    if not isinstance(hit, dict):
        return []
    ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    reader_open = ui_meta.get("reader_open") if isinstance(ui_meta.get("reader_open"), dict) else {}
    out: list[dict] = []
    seen: set[str] = set()

    def push(raw: object, *, rank: int) -> None:
        if not isinstance(raw, dict):
            return
        text = _primary_evidence_text(raw)
        if not text:
            return
        key = re.sub(r"\s+", " ", f"{raw.get('heading_path') or raw.get('headingPath') or ''}|{text}").strip().lower()
        if key in seen:
            return
        seen.add(key)
        item = dict(raw)
        item["_rank"] = int(rank)
        out.append(item)

    push(ui_meta.get("primary_evidence"), rank=0)
    for key in ("primaryEvidence", "primary_evidence", "locateTarget", "locate_target"):
        push(reader_open.get(key), rank=1)
    if _primary_evidence_text(reader_open):
        push(reader_open, rank=2)
    for container, base_rank in ((reader_open, 3), (ui_meta, 10)):
        if not isinstance(container, dict):
            continue
        for list_key in ("evidenceAlternatives", "visibleAlternatives", "alternatives"):
            values = container.get(list_key)
            if not isinstance(values, list):
                continue
            for idx, item in enumerate(values[:8]):
                push(item, rank=base_rank + idx)
    hit_text = str(hit.get("text") or "").strip()
    if hit_text:
        push(
            {
                "heading_path": (
                    str(meta.get("heading_path") or "").strip()
                    or str(meta.get("ref_best_heading_path") or "").strip()
                    or str(ui_meta.get("heading_path") or "").strip()
                    or str(ui_meta.get("primary_evidence_heading_path") or "").strip()
                ),
                "snippet": hit_text,
                "block_id": str(meta.get("primary_block_id") or meta.get("block_id") or "").strip(),
                "anchor_id": str(meta.get("primary_anchor_id") or meta.get("anchor_id") or "").strip(),
                "anchor_kind": str(meta.get("anchor_kind") or "").strip(),
            },
            rank=30,
        )
    return out


def _primary_evidence_quality_score(raw: dict, *, claim: str = "", source: str = "") -> float:
    text = _primary_evidence_text(raw)
    heading = str(raw.get("heading_path") or raw.get("headingPath") or "").strip()
    clean = _clean_evidence_display_text(text, max_len=700)
    readable = _pick_readable_evidence_text(
        text,
        source=source,
        title=heading,
        claim=claim,
        heading=heading,
        max_len=520,
    )
    scoring_text = readable or clean
    score = _evidence_sentence_quality(scoring_text, claim=claim, heading=heading, title=source)
    if readable:
        score += 2.0
    else:
        score -= 2.0
    if _REF_PRIMARY_LOW_VALUE_RE.search(clean):
        score -= 6.0
    try:
        if _looks_low_value_citation_context(clean):
            score -= 3.0
    except Exception:
        pass
    if str(raw.get("block_id") or raw.get("blockId") or "").strip():
        score += 0.2
    if str(raw.get("anchor_id") or raw.get("anchorId") or "").strip():
        score += 0.2
    try:
        score -= min(0.8, max(0, int(raw.get("_rank") or 0)) * 0.04)
    except Exception:
        pass
    return float(score)


def _primary_evidence_claim_from_ref_hit(hit: dict | None) -> str:
    if not isinstance(hit, dict):
        return ""
    ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    ref_pack = meta.get("ref_pack") if isinstance(meta.get("ref_pack"), dict) else {}
    return str(
        ui_meta.get("summary_line")
        or ui_meta.get("card_summary")
        or ui_meta.get("why_line")
        or ref_pack.get("what")
        or ""
    ).strip()


def _primary_evidence_from_ref_hit(hit: dict | None) -> dict:
    candidates = _primary_evidence_candidates_from_ref_hit(hit)
    if not candidates:
        return {}
    meta = hit.get("meta") if isinstance(hit, dict) and isinstance(hit.get("meta"), dict) else {}
    source = str(meta.get("source_name") or meta.get("source_path") or "").strip()
    claim = _primary_evidence_claim_from_ref_hit(hit)
    candidates.sort(
        key=lambda item: (
            _primary_evidence_quality_score(item, claim=claim, source=source),
            -int(item.get("_rank") or 0),
        ),
        reverse=True,
    )
    best = dict(candidates[0])
    best.pop("_rank", None)
    return best


def _ref_pack_primary_evidence_by_source(ref_pack: dict | None) -> dict[str, dict]:
    if not isinstance(ref_pack, dict):
        return {}
    out: dict[str, dict] = {}
    pack_primary = ref_pack.get("primary_evidence") if isinstance(ref_pack.get("primary_evidence"), dict) else {}
    pack_primary_key = _render_primary_source_identity(pack_primary)
    if pack_primary_key and pack_primary:
        out[pack_primary_key] = dict(pack_primary)
    candidate_hits: list[dict] = []
    candidate_hits.extend([item for item in list(ref_pack.get("hits") or []) if isinstance(item, dict)])
    candidate_hits.extend([item for item in list(ref_pack.get("enriched_hits") or []) if isinstance(item, dict)])
    for hit in candidate_hits:
        if not isinstance(hit, dict):
            continue
        primary = _primary_evidence_from_ref_hit(hit)
        if not primary:
            continue
        source_key = (
            _render_primary_source_identity(hit.get("meta") if isinstance(hit.get("meta"), dict) else {})
            or _render_primary_source_identity(hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {})
            or _render_primary_source_identity(primary)
        )
        if not source_key:
            continue
        current = out.get(source_key)
        if current and _primary_evidence_precision_score(current) >= _primary_evidence_precision_score(primary):
            continue
        out[source_key] = primary
    return out


def _system_a_detail_needs_ref_primary_backfill(detail: dict) -> bool:
    if bool(detail.get("is_inpaper")):
        return False
    if str(detail.get("citation_route") or "").strip().lower() == "system_b":
        return False
    flags = {str(item or "").strip().lower() for item in list(detail.get("card_quality_flags") or [])}
    if flags & {"evidence_quote_filtered", "missing_evidence_quote", "missing_precise_location"}:
        return True
    if not str(detail.get("block_id") or detail.get("anchor_id") or "").strip():
        return True
    evidence = str(
        detail.get("card_evidence")
        or detail.get("evidence_quote")
        or detail.get("summary_line")
        or detail.get("raw")
        or ""
    ).strip()
    if not evidence:
        return True
    if re.match(r"^\s*#{1,6}\s+", evidence):
        return True
    if re.search(r"\$\^\{|\bdagger\b|\\dagger|\*[, ]", evidence[:360], re.IGNORECASE):
        return True
    return False


def _backfill_system_a_cite_details_from_ref_pack(cite_details: list[dict], ref_pack: dict | None, *, render_locale: str = "") -> list[dict]:
    if not cite_details or not isinstance(ref_pack, dict):
        return cite_details
    primary_by_source = _ref_pack_primary_evidence_by_source(ref_pack)
    if not primary_by_source:
        return cite_details
    out: list[dict] = []
    for raw in cite_details:
        detail = dict(raw or {}) if isinstance(raw, dict) else {}
        if not detail:
            out.append(detail)
            continue
        if bool(detail.get("is_inpaper")) or str(detail.get("citation_route") or "").strip().lower() == "system_b":
            out.append(detail)
            continue
        source_key = _render_primary_source_identity(detail)
        primary = primary_by_source.get(source_key) if source_key else None
        snippet = _primary_evidence_text(primary if isinstance(primary, dict) else {})
        if not isinstance(primary, dict) or not snippet:
            out.append(detail)
            continue
        authoritative_answer_alignment = bool(
            str(primary.get("selection_reason") or "").strip().lower() == "answer_aligned_block"
            and bool(primary.get("strict_locate"))
            and str(primary.get("block_id") or primary.get("anchor_id") or "").strip()
        )
        if (not _system_a_detail_needs_ref_primary_backfill(detail)) and (not authoritative_answer_alignment):
            out.append(detail)
            continue
        heading = str(primary.get("heading_path") or primary.get("headingPath") or "").strip()
        block_id = str(primary.get("block_id") or primary.get("blockId") or "").strip()
        anchor_id = str(primary.get("anchor_id") or primary.get("anchorId") or "").strip()
        anchor_kind = str(primary.get("anchor_kind") or primary.get("anchorKind") or detail.get("anchor_kind") or "").strip()
        detail["heading_path"] = heading or str(detail.get("heading_path") or "").strip()
        detail["title"] = detail["heading_path"] or str(detail.get("title") or "").strip()
        detail["summary_line"] = snippet
        detail["evidence_quote"] = snippet
        detail["raw"] = snippet
        detail["evidence_source"] = "reference_primary_evidence"
        detail["summary_source"] = "reference_primary_evidence"
        detail["block_id"] = block_id or str(detail.get("block_id") or "").strip()
        detail["anchor_id"] = anchor_id or str(detail.get("anchor_id") or "").strip()
        detail["anchor_kind"] = anchor_kind
        location_bits: list[str] = []
        if detail.get("heading_path"):
            location_bits.append(str(detail.get("heading_path") or "").strip())
        if detail.get("anchor_kind"):
            location_bits.append(str(detail.get("anchor_kind") or "").strip())
        if location_bits:
            detail["location_label"] = " · ".join(part for part in location_bits if part)
        out.append(compose_citation_card(detail, locale=render_locale))
    return out


def _effective_reference_render_pack(raw_pack: dict | None) -> dict:
    if not isinstance(raw_pack, dict):
        return {}
    pack = dict(raw_pack)
    rendered_payload = dict(raw_pack.get("rendered_payload") or {}) if isinstance(raw_pack.get("rendered_payload"), dict) else {}
    if not rendered_payload:
        return attach_refs_pack_polish_contract(pack)
    merged = dict(rendered_payload)
    # Always prefer original hits/scores (they reflect the full retrieval result
    # the LLM actually saw).  rendered_payload may hold a stale or partial subset.
    if pack.get("hits") not in (None, "", [], {}):
        merged["hits"] = pack["hits"]
    if pack.get("scores") not in (None, "", [], {}):
        merged["scores"] = pack["scores"]
    if rendered_payload.get("hits") not in (None, "", [], {}):
        merged["enriched_hits"] = rendered_payload["hits"]
    for key in (
        "user_msg_id",
        "conv_id",
        "prompt",
        "prompt_sig",
        "render_status",
        "render_error",
        "render_error_detail",
        "render_built_at",
        "render_attempts",
        "render_evidence_sig",
        "render_locale",
        "used_query",
        "used_translation",
        "created_at",
        "updated_at",
    ):
        if merged.get(key) in (None, "", [], {}):
            value = pack.get(key)
            if value not in (None, "", [], {}):
                merged[key] = value
    return attach_refs_pack_polish_contract(merged)


def _effective_citation_render_locale(ref_pack: dict | None = None) -> str:
    packs: list[dict] = []
    if isinstance(ref_pack, dict):
        packs.append(ref_pack)
        rendered_payload = ref_pack.get("rendered_payload")
        if isinstance(rendered_payload, dict):
            packs.append(rendered_payload)
    for pack in packs:
        raw = str(pack.get("render_locale") or "").strip().lower()
        if raw in {"zh", "en"}:
            return raw
    try:
        prefs = load_prefs()
    except Exception:
        prefs = {}
    raw_card = str((prefs or {}).get("refs_card_locale") or "").strip().lower()
    if raw_card in {"zh", "en"}:
        return raw_card
    raw_ui = str((prefs or {}).get("ui_locale") or "").strip().lower()
    if raw_ui in {"zh", "en"}:
        return raw_ui
    return "zh"


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


_EMPTY_EXAMPLE_CONNECTOR_RE = re.compile(
    r"(?P<open>[（(])\s*(?:如|for\s+example|e\.g\.)\s*(?:或|和|及|以及|or|and|、|,|，)\s*",
    re.IGNORECASE,
)
_BARE_EMPTY_EXAMPLE_CONNECTOR_RE = re.compile(
    r"(?<![\w\u4e00-\u9fff])(?:如|for\s+example|e\.g\.)\s*(?:或|和|及|以及|or|and|、|,|，)\s*",
    re.IGNORECASE,
)
_DUPLICATE_NEIGHBOR_TERM_RE = re.compile(
    r"(?P<term>[A-Za-z][A-Za-z0-9+.-]*(?:\s+[A-Za-z][A-Za-z0-9+.-]*){0,4}|[\u4e00-\u9fff]{2,12})"
    r"\s*(?:、|，|,|/)\s*(?P=term)(?=\s*(?:[，。,.;；、)）]|$))"
)


def _cleanup_answer_surface_artifacts(md: str) -> str:
    out = str(md or "")
    if not out:
        return out
    out = _EMPTY_EXAMPLE_CONNECTOR_RE.sub(lambda m: str(m.group("open") or ""), out)
    out = _BARE_EMPTY_EXAMPLE_CONNECTOR_RE.sub("", out)
    for _ in range(3):
        nxt = _DUPLICATE_NEIGHBOR_TERM_RE.sub(lambda m: str(m.group("term") or "").strip(), out)
        if nxt == out:
            break
        out = nxt
    out = re.sub(r"\s+([，。；：！？、])", r"\1", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([（(])\s+", r"\1", out)
    out = re.sub(r"\s+([）)])", r"\1", out)
    out = re.sub(r"[（(]\s*[）)]", "", out)
    return out.strip()


def _normalize_chat_markdown_for_display(md: str) -> str:
    return _normalize_math_markdown(_cleanup_answer_surface_artifacts(_strip_structured_cite_tokens_for_display(md)))


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


def _message_citation_plan(rec: dict | None) -> dict:
    if not isinstance(rec, dict):
        return {}
    meta = dict(rec.get("meta") or {}) if isinstance(rec.get("meta"), dict) else {}
    answer_quality = dict(meta.get("answer_quality") or {}) if isinstance(meta.get("answer_quality"), dict) else {}
    plan = answer_quality.get("citation_plan")
    if isinstance(plan, dict) and plan:
        return dict(plan)
    contracts = dict(meta.get("paper_guide_contracts") or {}) if isinstance(meta.get("paper_guide_contracts"), dict) else {}
    plan = contracts.get("citation_plan")
    if isinstance(plan, dict) and plan:
        return dict(plan)
    return {}


def _citation_plan_system_b_budget(plan: dict | None) -> int:
    if not isinstance(plan, dict):
        return 1
    budget = plan.get("budget") if isinstance(plan.get("budget"), dict) else {}
    try:
        return int((budget or {}).get("system_b") if "system_b" in (budget or {}) else 1)
    except Exception:
        return 1


_READING_COVERAGE_BRIDGES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(r"\b(?:single[-\s]?photon|spad|photodetectors?|detectors?|detection)\b", re.IGNORECASE),
        ("单光子", "探测器", "硬件", "spad", "暗计数", "死时间", "后脉冲", "串扰"),
    ),
    (
        re.compile(r"\b(?:physics[-\s]?informed|deep learning|transformer|neural|noise model)\b", re.IGNORECASE),
        ("physics-informed", "deep learning", "深度学习", "物理", "噪声", "噪声模型", "transformer"),
    ),
    (
        re.compile(r"\b(?:single[-\s]?pixel|spi|compressive|sampling|reconstruction)\b", re.IGNORECASE),
        ("单像素", "单像素成像", "压缩", "采样", "重建", "spi"),
    ),
    (
        re.compile(r"\b(?:dual[-\s]?disperser|spectral\s+imaging|cassi|coded\s+aperture|single[-\s]?shot)\b", re.IGNORECASE),
        ("dual-disperser", "single-shot", "spectral", "spectral imaging", "光谱", "光谱成像", "双色散", "2007"),
    ),
)


def _reading_source_surface(hit: dict | None, slot: dict | None = None) -> str:
    parts: list[str] = []
    if isinstance(slot, dict):
        parts.extend(
            str(slot.get(key) or "")
            for key in ("topic", "source_name", "heading_path", "evidence_quote")
        )
    if isinstance(hit, dict):
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        parts.extend(
            str(value or "")
            for value in (
                hit.get("text"),
                meta.get("source_name"),
                meta.get("source_path"),
                meta.get("heading_path"),
                ui_meta.get("display_name"),
                ui_meta.get("summary_line"),
                ui_meta.get("why_line"),
            )
        )
    return " ".join(part for part in parts if part)


def _reading_coverage_terms(surface: str) -> set[str]:
    raw = str(surface or "")
    terms = {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9+.-]{2,}", raw)}
    terms.update(re.findall(r"[\u4e00-\u9fff]{2,8}", raw))
    for pattern, bridged_terms in _READING_COVERAGE_BRIDGES:
        if pattern.search(raw):
            terms.update(term.lower() for term in bridged_terms)
    return {term for term in terms if term and term not in {"the", "and", "for", "with", "this", "that", "from", "into"}}


def _reading_paragraph_affinity(paragraph: str, terms: set[str], *, source_surface: str = "") -> float:
    text = str(paragraph or "")
    if not text or not terms:
        return 0.0
    low = text.lower()
    source_low = str(source_surface or "").lower()
    if (
        ("physics-informed" in source_low or "deep learning" in source_low or "noise model" in source_low)
        and not re.search(r"physics-informed|deep learning|深度学习|噪声|模型|算法|AI", text, re.IGNORECASE)
    ):
        return 0.0
    score = 0.0
    for term in terms:
        if len(term) < 2:
            continue
        if term in low or term in text:
            score += 1.0 if len(term) <= 4 else 1.4
    if re.match(r"^\s*(?:\d+[.)、]|[-*+])\s*", text):
        score += 0.4
    if re.search(r"先读|再读|搭配|顺序|综述|review|paper|论文", text, re.IGNORECASE):
        score += 0.4
    return score


def _append_numeric_citation_to_paragraph(paragraph: str, num: int) -> str:
    marker = f"[{int(num)}]"
    text = str(paragraph or "")
    if marker in text:
        return text
    match = re.search(r"([。！？.!?])(\s*)$", text)
    if match:
        return f"{text[:match.start(1)].rstrip()} {marker}{match.group(1)}{match.group(2)}"
    return f"{text.rstrip()} {marker}"


def _reading_slot_source_key(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").lower()


def _reading_slot_hit_nums(slot: dict, hits: list[dict], canonical_paths: list[str] | None = None) -> list[int]:
    nums: list[int] = []
    wanted_path = _reading_slot_source_key(slot.get("source_path") or slot.get("sourcePath"))
    wanted_name = _reading_slot_source_key(slot.get("source_name") or slot.get("sourceName"))
    hit_paths: set[str] = set()
    for hit in list(hits or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        hit_path = _reading_slot_source_key(
            (meta or {}).get("source_path")
            or (ui_meta or {}).get("source_path")
            or (ui_meta or {}).get("sourcePath")
        )
        if hit_path:
            hit_paths.add(hit_path)
    if wanted_path and isinstance(canonical_paths, list):
        for idx, raw_path in enumerate(canonical_paths, start=1):
            canon_path = _reading_slot_source_key(raw_path)
            if canon_path and canon_path == wanted_path and canon_path in hit_paths:
                return [int(idx)]
    if wanted_path or wanted_name:
        for idx, hit in enumerate(list(hits or []), start=1):
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
            hit_path = _reading_slot_source_key(
                (meta or {}).get("source_path")
                or (ui_meta or {}).get("source_path")
                or (ui_meta or {}).get("sourcePath")
            )
            hit_name = _reading_slot_source_key(
                (ui_meta or {}).get("display_name")
                or (ui_meta or {}).get("source_name")
                or (ui_meta or {}).get("sourceName")
                or (meta or {}).get("source_name")
                or Path(hit_path).name
            )
            if wanted_path and hit_path == wanted_path:
                nums.append(int(idx))
                break
            if wanted_name and hit_name and (wanted_name in hit_name or hit_name in wanted_name):
                nums.append(int(idx))
                break
        if wanted_path and not nums:
            return []
    for raw in list(slot.get("candidate_hits") or []):
        try:
            num = int(raw)
        except Exception:
            continue
        if 1 <= num <= len(hits) and num not in nums:
            nums.append(num)
    return nums


def _reading_hit_for_slot(slot: dict, hits: list[dict], num: int) -> dict | None:
    wanted_path = _reading_slot_source_key(slot.get("source_path") or slot.get("sourcePath"))
    if wanted_path:
        for hit in list(hits or []):
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
            hit_path = _reading_slot_source_key(
                (meta or {}).get("source_path")
                or (ui_meta or {}).get("source_path")
                or (ui_meta or {}).get("sourcePath")
            )
            if hit_path == wanted_path:
                return hit
    idx = int(num) - 1
    if 0 <= idx < len(hits):
        hit = hits[idx]
        return hit if isinstance(hit, dict) else None
    return None


def _reading_guide_numbered_sections_have_sources(text: str) -> bool:
    section_re = re.compile(
        r"(?m)^\s*(?:#{1,6}\s*)?(?:第\s*)?\d{1,2}(?:\s*步\s*[:：]|[.)]\s+)"
    )
    matches = list(section_re.finditer(str(text or "")))
    if len(matches) < 2:
        return False
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(str(text or ""))
        section = str(text or "")[match.start() : end]
        if not re.search(
            r"\[\d{1,5}(?:\s*(?:-|–|—|,|，|、)\s*\d{1,5})*\]|\[\[\s*CITE\s*:",
            section,
            flags=re.IGNORECASE,
        ):
            return False
    return True


def _reading_guide_repair_missing_system_a_citations(
    md: str,
    hits: list[dict],
    citation_plan: dict | None,
    *,
    output_mode: str,
    canonical_paths: list[str] | None = None,
) -> str:
    if "reading" not in str(output_mode or ""):
        return str(md or "")
    if not isinstance(citation_plan, dict) or not hits:
        return str(md or "")
    text = str(md or "")
    if not text.strip():
        return text
    if _reading_guide_numbered_sections_have_sources(text):
        return text
    candidates: list[tuple[int, dict]] = []
    for slot in list(citation_plan.get("slots") or []):
        if not isinstance(slot, dict):
            continue
        if str(slot.get("preferred_system") or "").strip().lower() == "system_b":
            continue
        nums = _reading_slot_hit_nums(slot, hits, canonical_paths=canonical_paths)
        for num in nums[:1]:
            candidates.append((num, slot))
    if not candidates:
        return text

    parts = re.split(r"(\n{2,})", text)
    used_part_indices: set[int] = set()
    for num, slot in candidates[:6]:
        surface = _reading_source_surface(_reading_hit_for_slot(slot, hits, num), slot)
        terms = _reading_coverage_terms(surface)
        best_idx = -1
        best_score = 0.0
        for idx in range(0, len(parts), 2):
            if idx in used_part_indices:
                continue
            paragraph = parts[idx]
            if not paragraph.strip() or f"[{num}]" in paragraph:
                continue
            score = _reading_paragraph_affinity(paragraph, terms, source_surface=surface)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0 and best_score >= 2.2:
            parts[best_idx] = _append_numeric_citation_to_paragraph(parts[best_idx], num)
            used_part_indices.add(best_idx)
    return "".join(parts)


def _render_norm_source_key(path_like: str | Path) -> str:
    raw = str(path_like or "").strip()
    if not raw:
        return ""
    try:
        return str(Path(raw).expanduser().resolve(strict=False)).strip().lower()
    except Exception:
        return str(Path(raw).expanduser()).strip().lower()


def _reference_index_doc_for_source(index_data: dict | None, source_path: str, *, source_sha1: str = "") -> dict | None:
    if not isinstance(index_data, dict):
        return None
    docs = index_data.get("docs")
    if not isinstance(docs, dict):
        return None
    src_key = _render_norm_source_key(source_path)
    if src_key and isinstance(docs.get(src_key), dict):
        return dict(docs[src_key])

    sha = str(source_sha1 or "").strip().lower()
    if sha:
        for raw_doc in docs.values():
            if not isinstance(raw_doc, dict):
                continue
            if str(raw_doc.get("sha1") or "").strip().lower() == sha:
                return dict(raw_doc)

    want = str(source_path or "").strip()
    if not want:
        return None
    want_name = Path(want).name.lower()
    want_stem = Path(want).stem.lower()
    for raw_doc in docs.values():
        if not isinstance(raw_doc, dict):
            continue
        doc_name = str(raw_doc.get("name") or "").strip().lower()
        doc_stem = str(raw_doc.get("stem") or "").strip().lower()
        doc_path = str(raw_doc.get("path") or "").strip()
        if _render_norm_source_key(doc_path) == src_key:
            return dict(raw_doc)
        if want_name and doc_name and want_name == doc_name:
            return dict(raw_doc)
        if want_stem and doc_stem and want_stem == doc_stem:
            return dict(raw_doc)
    return None


def _title_tokens_for_named_system_b(title: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", str(title or "").lower())


def _named_system_b_title_matches_current_source(title: str, hits: list[dict]) -> bool:
    title_key = " ".join(_title_tokens_for_named_system_b(title))
    if len(title_key) < 18:
        return False
    for hit in list(hits or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or "").strip()
        candidates = [
            str((meta or {}).get("title") or ""),
            str((meta or {}).get("source_name") or ""),
            str((ui_meta or {}).get("title") or ""),
            str((ui_meta or {}).get("display_name") or (ui_meta or {}).get("displayName") or ""),
            Path(source_path).stem if source_path else "",
        ]
        for candidate in candidates:
            candidate_key = " ".join(_title_tokens_for_named_system_b(candidate))
            if len(candidate_key) >= 18 and (title_key in candidate_key or candidate_key in title_key):
                return True
    return False


def _usable_named_system_b_title(title: str) -> bool:
    text = str(title or "").strip()
    if len(text) < 18 or len(text) > 220:
        return False
    tokens = _title_tokens_for_named_system_b(text)
    if len(tokens) < 4:
        return False
    low = text.lower().strip(" .;:,")
    if low in {"references", "bibliography", "abstract", "introduction"}:
        return False
    if re.fullmatch(r"(?:optics express|nature|science|ieee|acm|springer|elsevier|arxiv)", low):
        return False
    return True


def _named_system_b_title_pattern(title: str) -> re.Pattern[str] | None:
    text = str(title or "").strip()
    if not _usable_named_system_b_title(text):
        return None
    escaped = re.escape(text)
    escaped = escaped.replace(r"\ ", r"[\s\u00a0]+")
    escaped = escaped.replace(r"\-", r"[\-–—\s]+")
    return re.compile(rf"(?<![#/\w])({escaped})(?![/\w])", re.IGNORECASE)


def _title_match_has_nearby_cite_marker(text: str, start: int, end: int) -> bool:
    left = str(text or "")[max(0, int(start) - 12): int(start)]
    right = str(text or "")[int(end): min(len(str(text or "")), int(end) + 96)]
    if left.endswith("[") and right.lstrip().startswith("]("):
        return True
    if re.search(r"\]\([^)]{0,160}$", left):
        return True
    if re.match(r"\s*(?:\[[Rr]?\d{1,4}\]|\[\[?\s*CITE\s*:)", right, re.IGNORECASE):
        return True
    return False


def _repair_named_system_b_citation_markers(
    md: str,
    hits: list[dict],
    citation_plan: dict | None,
) -> tuple[str, bool]:
    """Attach System B markers when the answer already names an indexed upstream title.

    This is intentionally narrow: exact title mention from the current source
    document's reference index only. It avoids converting broad prose like
    "Optica 2024" into a possibly wrong bibliography link.
    """

    text = str(md or "")
    if not text.strip() or not hits:
        return text, False
    if "[[CITE:" in text.upper():
        return text, False

    plan = dict(citation_plan or {}) if isinstance(citation_plan, dict) else {}
    try:
        budget = int(((plan.get("budget") if isinstance(plan.get("budget"), dict) else {}) or {}).get("system_b", 2))
    except Exception:
        budget = 2
    budget = max(0, min(2, budget if plan else 1))
    if budget <= 0:
        return text, False

    index_data = _load_reference_index_cached()
    if not isinstance(index_data, dict) or not index_data:
        return text, False

    source_rows: list[tuple[str, str, dict]] = []
    seen_sources: set[str] = set()
    for hit in hits[:8]:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or "").strip()
        if not source_path:
            continue
        source_key = _render_norm_source_key(source_path)
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        doc = _reference_index_doc_for_source(
            index_data,
            source_path,
            source_sha1=str((meta or {}).get("source_sha1") or "").strip().lower(),
        )
        if isinstance(doc, dict):
            source_rows.append((source_path, str((meta or {}).get("source_sha1") or "").strip().lower(), doc))

    candidates: list[tuple[int, str, str, str]] = []
    seen_titles: set[str] = set()
    for source_path, _source_sha1, doc in source_rows:
        refs = doc.get("refs")
        if not isinstance(refs, dict):
            continue
        sid = _source_cite_id(source_path)
        for raw_n, raw_ref in sorted(refs.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9):
            if not isinstance(raw_ref, dict):
                continue
            try:
                ref_num = int(raw_n)
            except Exception:
                continue
            title = str(raw_ref.get("title") or "").strip()
            if not _usable_named_system_b_title(title):
                continue
            if _named_system_b_title_matches_current_source(title, hits):
                continue
            title_key = " ".join(_title_tokens_for_named_system_b(title))
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            candidates.append((ref_num, sid, title, source_path))

    insertions: list[tuple[int, str]] = []
    used_refs: set[tuple[str, int]] = set()
    for ref_num, sid, title, source_path in candidates:
        if len(insertions) >= budget:
            break
        pattern = _named_system_b_title_pattern(title)
        if pattern is None:
            continue
        match = pattern.search(text)
        if not match:
            continue
        if _title_match_has_nearby_cite_marker(text, int(match.start()), int(match.end())):
            continue
        ref_key = (source_path.lower(), int(ref_num))
        if ref_key in used_refs:
            continue
        used_refs.add(ref_key)
        insertions.append((int(match.end()), f" [[CITE:{sid}:{int(ref_num)}]]"))

    if not insertions:
        return text, False
    out = text
    for pos, marker in sorted(insertions, key=lambda item: item[0], reverse=True):
        out = f"{out[:pos]}{marker}{out[pos:]}"
    return out, True


_SUPP_REF_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


def _supp_ref_normalize_search_text(text: str) -> str:
    raw = str(text or "").lower()
    raw = re.sub(r"https?://(?:dx\.)?doi\.org/", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def _supp_ref_title_key(title: str) -> str:
    return " ".join(_title_tokens_for_named_system_b(str(title or "")))


def _supp_ref_clean_doi(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", raw, flags=re.IGNORECASE).strip()
    doi = extract_first_doi(raw) or raw
    doi = re.sub(r"^(?:doi\s*:\s*)", "", str(doi or "").strip(), flags=re.IGNORECASE)
    doi = doi.strip().rstrip(".,;)]}")
    return doi.lower()


def _supp_ref_doi_url(doi: str) -> str:
    d = _supp_ref_clean_doi(doi)
    return f"https://doi.org/{d}" if d else ""


def _supp_ref_answer_dois(*texts: str) -> set[str]:
    out: set[str] = set()
    for text in texts:
        for m in _SUPP_REF_DOI_RE.finditer(str(text or "")):
            doi = _supp_ref_clean_doi(m.group(0))
            if doi:
                out.add(doi)
    return out


def _supp_ref_add_source_identity(
    raw: dict | None,
    out: list[dict],
    seen: set[str],
    *,
    fallback_source_path: str = "",
    fallback_source_name: str = "",
) -> None:
    if not isinstance(raw, dict):
        return
    source_path = str(
        raw.get("source_path")
        or raw.get("sourcePath")
        or raw.get("md_path")
        or raw.get("mdPath")
        or fallback_source_path
        or ""
    ).strip()
    source_name = str(
        raw.get("source_name")
        or raw.get("sourceName")
        or raw.get("display_name")
        or raw.get("displayName")
        or fallback_source_name
        or ""
    ).strip()
    source_sha1 = str(raw.get("source_sha1") or raw.get("sourceSha1") or "").strip().lower()
    if not source_path:
        return
    key = _render_norm_source_key(source_path)
    if not key or key in seen:
        return
    seen.add(key)
    out.append({"source_path": source_path, "source_name": source_name, "source_sha1": source_sha1})


def _supp_ref_collect_source_identities(
    *,
    ref_pack: dict | None,
    cite_details: list[dict] | None,
    provenance_segments: list[dict] | None,
    limit: int = 12,
) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()

    def add(raw: dict | None, *, fallback_source_path: str = "", fallback_source_name: str = "") -> None:
        if len(out) >= limit:
            return
        _supp_ref_add_source_identity(
            raw,
            out,
            seen,
            fallback_source_path=fallback_source_path,
            fallback_source_name=fallback_source_name,
        )

    for detail in list(cite_details or []):
        if isinstance(detail, dict):
            add(detail)

    packs: list[dict] = []
    if isinstance(ref_pack, dict):
        packs.append(ref_pack)
        rendered_payload = ref_pack.get("rendered_payload")
        if isinstance(rendered_payload, dict):
            packs.append(rendered_payload)
    for pack in packs:
        add(pack.get("primary_evidence") if isinstance(pack.get("primary_evidence"), dict) else None)
        for key in ("hits", "enriched_hits"):
            for hit in list(pack.get(key) or []):
                if not isinstance(hit, dict):
                    continue
                meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
                ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
                add(meta if isinstance(meta, dict) else None)
                add(ui_meta if isinstance(ui_meta, dict) else None)
                add(ui_meta.get("reader_open") if isinstance(ui_meta.get("reader_open"), dict) else None)
                add(ui_meta.get("primary_evidence") if isinstance(ui_meta.get("primary_evidence"), dict) else None)
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break

    for seg in list(provenance_segments or []):
        if not isinstance(seg, dict):
            continue
        add(seg)
        add(seg.get("reader_open") if isinstance(seg.get("reader_open"), dict) else None)
        add(seg.get("locate_target") if isinstance(seg.get("locate_target"), dict) else None)
        if len(out) >= limit:
            break

    return out[:limit]


def _supp_ref_iter_index_rows(
    index_data: dict | None,
    source_identities: list[dict],
) -> list[dict]:
    rows: list[dict] = []
    seen_docs: set[str] = set()
    for source in source_identities:
        if not isinstance(source, dict):
            continue
        source_path = str(source.get("source_path") or "").strip()
        if not source_path:
            continue
        doc = _reference_index_doc_for_source(
            index_data,
            source_path,
            source_sha1=str(source.get("source_sha1") or "").strip().lower(),
        )
        if not isinstance(doc, dict):
            continue
        doc_key = _render_norm_source_key(str(doc.get("path") or source_path))
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)
        refs = doc.get("refs")
        if not isinstance(refs, dict):
            continue
        display_source_path = str(doc.get("path") or source_path).strip()
        display_source_name = str(doc.get("name") or source.get("source_name") or _source_name_from_path(display_source_path)).strip()
        for raw_num, raw_ref in sorted(refs.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9):
            if not isinstance(raw_ref, dict):
                continue
            try:
                ref_num = int(raw_num)
            except Exception:
                continue
            if ref_num <= 0:
                continue
            ref2 = _normalize_reference_for_popup(raw_ref) or {}
            raw_text = str(ref2.get("raw") or raw_ref.get("raw") or "").strip()
            doi = _supp_ref_clean_doi(str(ref2.get("doi") or raw_ref.get("doi") or extract_first_doi(raw_text) or ""))
            title = str(ref2.get("title") or raw_ref.get("title") or "").strip()
            rows.append(
                {
                    "source_path": display_source_path,
                    "source_name": display_source_name,
                    "ref_num": ref_num,
                    "ref": raw_ref,
                    "ref_norm": ref2,
                    "raw": raw_text,
                    "doi": doi,
                    "doi_url": str(ref2.get("doi_url") or raw_ref.get("doi_url") or _supp_ref_doi_url(doi)).strip(),
                    "title": title,
                    "authors": str(ref2.get("authors") or raw_ref.get("authors") or "").strip(),
                    "venue": str(ref2.get("venue") or raw_ref.get("venue") or "").strip(),
                    "year": str(ref2.get("year") or raw_ref.get("year") or "").strip(),
                    "volume": str(ref2.get("volume") or raw_ref.get("volume") or "").strip(),
                    "issue": str(ref2.get("issue") or raw_ref.get("issue") or "").strip(),
                    "pages": str(ref2.get("pages") or raw_ref.get("pages") or "").strip(),
                }
            )
    return rows


def _supp_ref_venue_year_phrase(row: dict) -> str:
    venue = str((row or {}).get("venue") or "").strip()
    year = str((row or {}).get("year") or "").strip()
    if not venue or not re.fullmatch(r"(?:19|20)\d{2}", year):
        return ""
    venue_norm = _supp_ref_normalize_search_text(venue)
    if len(venue_norm) < 5:
        return ""
    if venue_norm in {"ieee", "acm", "springer", "elsevier", "nature", "science"}:
        return ""
    return _supp_ref_normalize_search_text(f"{venue} {year}")


def _supp_ref_existing_identity(cite_details: list[dict] | None) -> tuple[set[tuple[str, int]], set[str], set[str]]:
    ref_keys: set[tuple[str, int]] = set()
    dois: set[str] = set()
    titles: set[str] = set()
    for detail in list(cite_details or []):
        if not isinstance(detail, dict):
            continue
        source_path = str(detail.get("source_path") or detail.get("sourcePath") or "").strip()
        try:
            ref_num = int(detail.get("num") or 0)
        except Exception:
            ref_num = 0
        if source_path and ref_num > 0:
            ref_keys.add((_render_norm_source_key(source_path), ref_num))
        doi = _supp_ref_clean_doi(str(detail.get("doi") or detail.get("doi_url") or detail.get("doiUrl") or ""))
        if doi:
            dois.add(doi)
        title_key = _supp_ref_title_key(str(detail.get("title") or detail.get("card_title") or detail.get("cardTitle") or ""))
        if title_key:
            titles.add(title_key)
    return ref_keys, dois, titles


def _supp_ref_context_line(text: str, start: int, end: int) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    try:
        return extract_structured_cite_answer_context_line(
            raw,
            int(max(0, start)),
            int(max(0, end)),
            normalizer=_md_to_plain_text,
        )
    except Exception:
        left = raw[: max(0, int(start))]
        right = raw[int(end) :]
        line_start = max(left.rfind("\n"), left.rfind("。"), left.rfind("."), left.rfind(";"), left.rfind("；"))
        line_end_candidates = [
            idx for idx in (right.find("\n"), right.find("。"), right.find("."), right.find(";"), right.find("；")) if idx >= 0
        ]
        line_end = min(line_end_candidates) if line_end_candidates else min(len(right), 220)
        return raw[line_start + 1 : int(end) + line_end].strip()


def _supp_ref_match_retrieved_document(row: dict, source_identities: list[dict] | None) -> dict:
    title_key = _supp_ref_title_key(str(row.get("title") or ""))
    if not title_key or len(title_key) < 18:
        return {}
    matches: list[dict] = []
    seen: set[str] = set()
    for raw in list(source_identities or []):
        if not isinstance(raw, dict):
            continue
        source_path = str(raw.get("source_path") or "").strip()
        if not source_path:
            continue
        source_name = str(raw.get("source_name") or "").strip() or _source_name_from_path(source_path)
        source_key = _supp_ref_title_key(source_name or source_path)
        if not source_key or not (title_key == source_key or title_key in source_key):
            continue
        identity = _render_norm_source_key(source_path)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        matches.append({
            "source_path": source_path,
            "source_name": source_name or _source_name_from_path(source_path),
            "source_sha1": str(raw.get("source_sha1") or "").strip(),
        })
    return matches[0] if len(matches) == 1 else {}


def _supp_ref_candidate_detail(
    *,
    row: dict,
    anchor_ns: str,
    answer_context: str,
    render_locale: str,
    match_method: str,
    confidence: float,
    retrieved_document: dict | None = None,
) -> dict:
    reference_source_path = str(row.get("source_path") or "").strip()
    reference_source_name = str(row.get("source_name") or _source_name_from_path(reference_source_path)).strip()
    ref_num = int(row.get("ref_num") or 0)
    title = str(row.get("title") or "").strip()
    raw = str(row.get("raw") or "").strip()
    local_doc = dict(retrieved_document or {}) if isinstance(retrieved_document, dict) else {}
    source_path = str(local_doc.get("source_path") or reference_source_path).strip()
    source_name = str(local_doc.get("source_name") or reference_source_name or _source_name_from_path(source_path)).strip()
    direct_library_match = bool(local_doc and source_path)
    anchor = _build_anchor(anchor_ns, _source_cite_id(source_path), 1 if direct_library_match else ref_num, source_name)
    locale = str(render_locale or "").strip().lower()
    if direct_library_match and locale == "en":
        support_relation = "This paper is available in the local library and can be opened directly."
        binding_reason = "Matched the mentioned title to a retrieved local-library document."
    elif direct_library_match:
        support_relation = "回答提到的论文已在本地文献库中，可直接打开全文。"
        binding_reason = "回答标题与本次检索到的库内论文一致。"
    elif locale == "en":
        support_relation = "The answer mentions this work, and it appears in the current paper's bibliography."
        binding_reason = f"Matched from local reference index by {match_method}."
    else:
        support_relation = "回答提到了这项工作，且它出现在当前论文的参考文献表中。"
        binding_reason = f"根据本地参考文献索引命中：{match_method}。"
    rec = {
        "num": 0 if direct_library_match else ref_num,
        "anchor": anchor,
        "source_name": source_name,
        "source_path": source_path,
        "is_inpaper": not direct_library_match,
        "citation_route": "system_a" if direct_library_match else "system_b",
        "routing_reason": (
            "unlinked_reference_candidate:retrieved_library_document"
            if direct_library_match
            else f"unlinked_reference_candidate:{match_method}"
        ),
        "routing_confidence": float(confidence),
        "raw": raw,
        "title": title,
        "authors": str(row.get("authors") or "").strip(),
        "venue": str(row.get("venue") or "").strip(),
        "year": str(row.get("year") or "").strip(),
        "volume": str(row.get("volume") or "").strip(),
        "issue": str(row.get("issue") or "").strip(),
        "pages": str(row.get("pages") or "").strip(),
        "doi": str(row.get("doi") or "").strip(),
        "doi_url": str(row.get("doi_url") or "").strip(),
        "cite_fmt": raw,
        "heading_path": "" if direct_library_match else "References",
        "location_label": source_name if direct_library_match else f"References / [{ref_num}]",
        "evidence_quote": str(answer_context or "").strip() if direct_library_match else raw,
        "evidence_source": "answer_reference_mention" if direct_library_match else "reference_index",
        "citation_context": str(answer_context or "").strip(),
        "citation_context_source": "answer_reference_mention",
        "summary_line": str(answer_context or "").strip(),
        "summary_source": "answer_reference_mention",
        "answer_claim": str(answer_context or "").strip(),
        "support_relation": support_relation,
        "binding_status": "library_match" if direct_library_match else "candidate",
        "binding_confidence": float(confidence),
        "binding_reason": binding_reason,
        "render_locale": locale,
    }
    if direct_library_match:
        rec.update({
            "library_match": {
                "matched": True,
                "status": "in_library",
                "confidence": 0.96,
                "method": "retrieved_title",
                "reason": "retrieved_document_title_exact",
                "path": source_path,
                "sha1": str(local_doc.get("source_sha1") or "").strip(),
                "title": title,
                "doi": str(row.get("doi") or "").strip(),
                "year": str(row.get("year") or "").strip(),
            },
            "library_match_status": "in_library",
            "library_match_confidence": 0.96,
            "library_match_method": "retrieved_title",
            "library_match_reason": "retrieved_document_title_exact",
            "library_match_path": source_path,
            "library_match_sha1": str(local_doc.get("source_sha1") or "").strip(),
            "library_match_title": title,
            "library_match_doi": str(row.get("doi") or "").strip(),
            "library_match_year": str(row.get("year") or "").strip(),
            "reference_source_path": reference_source_path,
            "reference_source_name": reference_source_name,
            "reference_ref_num": ref_num,
        })
    else:
        enrich_inpaper_detail_context(
            rec,
            source_path=source_path,
            ref_num=ref_num,
            answer_context=str(answer_context or "").strip(),
            source_answer_context=str(answer_context or "").strip(),
        )
    return compose_citation_card(rec, locale=render_locale)


def _build_unlinked_reference_candidates(
    *,
    answer_markdown: str,
    rendered_body: str,
    copy_text: str,
    cite_details: list[dict] | None,
    ref_pack: dict | None,
    provenance_segments: list[dict] | None,
    render_locale: str = "",
    anchor_ns: str = "",
    limit: int = 5,
) -> list[dict]:
    source_text = "\n\n".join([str(answer_markdown or ""), str(rendered_body or ""), str(copy_text or "")]).strip()
    if not source_text:
        return []
    index_data = _load_reference_index_cached()
    if not isinstance(index_data, dict) or not index_data:
        return []
    source_identities = _supp_ref_collect_source_identities(
        ref_pack=ref_pack,
        cite_details=cite_details,
        provenance_segments=provenance_segments,
    )
    if not source_identities:
        return []
    rows = _supp_ref_iter_index_rows(index_data, source_identities)
    if not rows:
        return []

    answer_dois = _supp_ref_answer_dois(source_text)
    answer_norm = _supp_ref_normalize_search_text(_md_to_plain_text(source_text) or source_text)
    existing_ref_keys, existing_dois, existing_title_keys = _supp_ref_existing_identity(cite_details)
    phrase_rows: dict[str, list[dict]] = {}
    for row in rows:
        phrase = _supp_ref_venue_year_phrase(row)
        if not phrase:
            continue
        if phrase and phrase in answer_norm:
            phrase_rows.setdefault(phrase, []).append(row)

    out: list[dict] = []
    seen: set[str] = set()
    seen_library_sources: set[str] = set()

    def add_candidate(row: dict, *, match_method: str, confidence: float, mention: str, start: int = -1, end: int = -1) -> None:
        if len(out) >= limit:
            return
        source_path = str(row.get("source_path") or "").strip()
        ref_num = int(row.get("ref_num") or 0)
        source_key = _render_norm_source_key(source_path)
        doi = _supp_ref_clean_doi(str(row.get("doi") or ""))
        title_key = _supp_ref_title_key(str(row.get("title") or ""))
        if (source_key, ref_num) in existing_ref_keys:
            return
        if doi and doi in existing_dois:
            return
        if title_key and title_key in existing_title_keys:
            return
        candidate_key = doi or f"{source_key}::{ref_num}" or title_key
        if not candidate_key or candidate_key in seen:
            return
        seen.add(candidate_key)
        context = _supp_ref_context_line(source_text, start, end) if start >= 0 and end >= start else str(mention or "").strip()
        detail = _supp_ref_candidate_detail(
            row=row,
            anchor_ns=anchor_ns or "unlinked-reference",
            answer_context=context,
            render_locale=render_locale,
            match_method=match_method,
            confidence=confidence,
            retrieved_document=_supp_ref_match_retrieved_document(row, source_identities),
        )
        detail_source_path = str(detail.get("source_path") or source_path).strip()
        detail_source_name = str(detail.get("source_name") or row.get("source_name") or "").strip()
        direct_library_source_key = (
            _render_norm_source_key(detail_source_path)
            if not bool(detail.get("is_inpaper", True))
            else ""
        )
        if direct_library_source_key:
            if direct_library_source_key in seen_library_sources:
                return
            seen_library_sources.add(direct_library_source_key)
        out.append(
            {
                "id": hashlib.sha1(f"{candidate_key}|{match_method}".encode("utf-8", "ignore")).hexdigest()[:12],
                "status": "reference_list_hit",
                "match_method": match_method,
                "confidence": round(float(confidence), 3),
                "mention": str(mention or "").strip(),
                "source_path": detail_source_path,
                "source_name": detail_source_name,
                "ref_num": int(detail.get("num") or 0),
                "reference_source_path": source_path if detail_source_path != source_path else "",
                "reference_ref_num": ref_num if detail_source_path != source_path else 0,
                "title": str(row.get("title") or "").strip(),
                "authors": str(row.get("authors") or "").strip(),
                "venue": str(row.get("venue") or "").strip(),
                "year": str(row.get("year") or "").strip(),
                "doi": doi,
                "doi_url": str(row.get("doi_url") or _supp_ref_doi_url(doi)).strip(),
                "raw": str(row.get("raw") or "").strip(),
                "cite_detail": detail,
            }
        )

    for row in rows:
        doi = _supp_ref_clean_doi(str(row.get("doi") or ""))
        if doi and doi in answer_dois:
            pos = source_text.lower().find(doi)
            add_candidate(
                row,
                match_method="doi_mention",
                confidence=0.96,
                mention=doi,
                start=pos,
                end=pos + len(doi) if pos >= 0 else -1,
            )
        if len(out) >= limit:
            return out

    for row in rows:
        title = str(row.get("title") or "").strip()
        pattern = _named_system_b_title_pattern(title)
        if pattern is None:
            continue
        match = pattern.search(source_text)
        if not match:
            continue
        if _title_match_has_nearby_cite_marker(source_text, int(match.start()), int(match.end())):
            continue
        add_candidate(
            row,
            match_method="title_mention",
            confidence=0.9,
            mention=match.group(1),
            start=int(match.start()),
            end=int(match.end()),
        )
        if len(out) >= limit:
            return out

    for phrase, matched_rows in phrase_rows.items():
        if len(matched_rows) != 1:
            continue
        row = matched_rows[0]
        add_candidate(
            row,
            match_method="unique_venue_year_mention",
            confidence=0.68,
            mention=" ".join([str(row.get("venue") or "").strip(), str(row.get("year") or "").strip()]).strip(),
        )
        if len(out) >= limit:
            return out

    return out


def _should_link_inpaper_citations_for_message(*, rec: dict | None, content: str, hits: list[dict] | None = None) -> bool:
    raw = str(content or "")
    if not raw:
        return False
    if _message_intent_family(rec) == "citation_lookup":
        return True
    if _message_answer_prompt_family(rec) == "citation_lookup":
        return True
    if hits and (
        _STRUCT_CITE_RE.search(raw)
        or _STRUCT_CITE_SINGLE_RE.search(raw)
        or _STRUCT_CITE_SID_ONLY_RE.search(raw)
    ):
        # System B is a typed citation protocol: only structured
        # [[CITE:<sid>:<ref_num>]] tokens point at the current paper's
        # bibliography.  The renderer validates the SID/ref against the
        # reference index and drops unresolved tokens, so we should not decide
        # System A vs System B from answer wording here.
        return True
    # Classic RAG with [n] markers and available hits → link citations.
    if hits and re.search(r"\[\d{1,4}\]", raw):
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
    # Count resolved System B entries — if the primary renderer handled all
    # [[CITE:...]] markers, we don't need the fallback.
    sysb_resolved = sum(1 for d in cite_details if d.get("is_inpaper") is True)
    raw_cite_count = len(_STRUCT_CITE_RE.findall(raw)) + len(_STRUCT_CITE_SINGLE_RE.findall(raw))
    if sysb_resolved >= raw_cite_count:
        return False
    # Some [[CITE:...]] markers were not resolved by the primary renderer
    # (typically because the SID wasn't in sid_to_source).  Fallback to the
    # standalone renderer which builds its own SID mapping from hits.
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
    render_locale: str = "",
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
        "render_locale": str(render_locale or "").strip().lower(),
    }
    return _stable_json_hash(base)


def _iter_numeric_citation_numbers(text: str) -> list[int]:
    return _contract_iter_numeric_citation_numbers(text)


def _count_linkable_source_hits(hits: list[dict] | None) -> int:
    return _contract_count_linkable_source_hits(hits)


def _content_has_linkable_answer_citations(content: str, hits: list[dict] | None) -> bool:
    return _contract_content_has_linkable_answer_citations(content, hits)


def _cache_has_rendered_citation_links(cache: dict) -> bool:
    return render_payload_has_citation_links(cache)


def _existing_render_packet_from_record(rec: dict | None) -> dict:
    if not isinstance(rec, dict):
        return {}
    meta = dict(rec.get("meta") or {}) if isinstance(rec.get("meta"), dict) else {}
    contracts = dict(meta.get("paper_guide_contracts") or {}) if isinstance(meta.get("paper_guide_contracts"), dict) else {}
    packet = contracts.get("render_packet")
    return dict(packet) if isinstance(packet, dict) else {}


def _message_render_source_markdown(rec: dict | None, content: str) -> str:
    raw_content = str(content or "").strip()
    if raw_content:
        return raw_content
    packet = _existing_render_packet_from_record(rec)
    for key in ("answer_markdown", "rendered_body", "rendered_content", "copy_markdown"):
        value = str(packet.get(key) or "").strip()
        if value:
            return value
    return ""


def _render_cache_is_degraded_for_citations(cache: dict, *, raw_content: str, hits: list[dict] | None) -> bool:
    return render_payload_is_degraded_for_citations(cache, raw_content=raw_content, hits=hits)


def _extract_render_cache(
    meta: dict | None,
    *,
    expected_key: str,
    raw_content: str = "",
    hits: list[dict] | None = None,
) -> dict | None:
    if not isinstance(meta, dict):
        return None
    payload = normalize_render_cache_payload(
        meta.get("render_cache"),
        schema=_RENDER_CACHE_SCHEMA_VERSION,
        expected_key=expected_key,
    )
    if payload is None:
        return None
    normalized = payload.as_dict()
    if str(raw_content or "").strip() and not (
        str(normalized.get("rendered_content") or "").strip()
        or str(normalized.get("rendered_body") or "").strip()
    ):
        return None
    if render_payload_is_degraded_for_citations(payload, raw_content=raw_content, hits=hits):
        return None
    return normalized


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
    return _contract_build_render_cache_payload(
        schema=_RENDER_CACHE_SCHEMA_VERSION,
        cache_key=cache_key,
        notice=notice,
        rendered_body=rendered_body,
        rendered_content=rendered_content,
        copy_markdown=copy_markdown,
        copy_text=copy_text,
        cite_details=cite_details,
        refs_user_msg_id=refs_user_msg_id,
        render_packet=render_packet,
    )


def _sync_render_cache_packet(meta: dict, render_packet: dict) -> bool:
    cache = meta.get("render_cache") if isinstance(meta, dict) else None
    if not isinstance(cache, dict) or not cache:
        return False
    next_cache = dict(cache)
    current_packet = (
        dict(next_cache.get("render_packet") or {})
        if isinstance(next_cache.get("render_packet"), dict)
        else {}
    )
    next_packet = dict(render_packet or {}) if isinstance(render_packet, dict) else {}
    if current_packet == next_packet:
        return False
    next_cache["render_packet"] = next_packet
    meta["render_cache"] = next_cache
    return True


def _merge_render_packet_contract_meta(
    *,
    rec: dict,
    msg_id: int,
    enriched_provenance: dict | None,
    ref_pack: dict | None = None,
    chat_store=None,
    render_locale: str = "",
) -> None:
    meta = dict(rec.get("meta") or {}) if isinstance(rec.get("meta"), dict) else {}
    contracts = dict(meta.get("paper_guide_contracts") or {}) if isinstance(meta.get("paper_guide_contracts"), dict) else {}
    if not contracts:
        return
    contracts_changed = False
    if isinstance(ref_pack, dict):
        refs_primary = dict(ref_pack.get("primary_evidence") or {}) if isinstance(ref_pack.get("primary_evidence"), dict) else {}
        if refs_primary:
            current_contract_primary = (
                dict(contracts.get("primary_evidence") or {})
                if isinstance(contracts.get("primary_evidence"), dict)
                else {}
            )
            if _should_adopt_refs_primary(current_contract_primary, refs_primary):
                merged_contract_primary = dict(refs_primary)
                if _primary_evidence_is_compatible(merged_contract_primary, current_contract_primary):
                    merged_contract_primary = _merge_render_packet_primary_evidence(
                        contract_primary=merged_contract_primary,
                        provenance_primary=current_contract_primary,
                        existing_primary={},
                    )
            else:
                merged_contract_primary = _merge_render_packet_primary_evidence(
                    contract_primary=refs_primary,
                    provenance_primary=current_contract_primary,
                    existing_primary={},
                )
            if merged_contract_primary:
                if dict(contracts.get("primary_evidence") or {}) != merged_contract_primary:
                    contracts["primary_evidence"] = merged_contract_primary
                    contracts_changed = True
    existing_packet = dict(contracts.get("render_packet") or {}) if isinstance(contracts.get("render_packet"), dict) else {}
    existing_cite_details = [
        compose_citation_card(item, locale=render_locale)
        for item in list(existing_packet.get("cite_details") or [])
        if isinstance(item, dict)
    ]
    current_cite_details = [
        compose_citation_card(item, locale=render_locale)
        for item in list(rec.get("cite_details") or [])
        if isinstance(item, dict)
    ]
    if isinstance(ref_pack, dict):
        existing_cite_details = _backfill_system_a_cite_details_from_ref_pack(existing_cite_details, ref_pack, render_locale=render_locale)
        current_cite_details = _backfill_system_a_cite_details_from_ref_pack(current_cite_details, ref_pack, render_locale=render_locale)
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
    answer_markdown = _message_render_source_markdown(rec, str(rec.get("content") or ""))
    provenance_segments = list((enriched_provenance or {}).get("segments") or [])
    provenance_primary_evidence = _merge_render_packet_primary_evidence(
        contract_primary=(
            dict(contracts.get("primary_evidence") or {})
            if isinstance(contracts.get("primary_evidence"), dict)
            else {}
        ),
        provenance_primary=(
            dict((enriched_provenance or {}).get("primary_evidence") or {})
            if isinstance((enriched_provenance or {}).get("primary_evidence"), dict)
            else {}
        ),
        existing_primary=(
            dict(existing_packet.get("primary_evidence") or {})
            if isinstance(existing_packet.get("primary_evidence"), dict)
            else {}
        ),
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
    selected_cite_details = existing_cite_details if preserve_existing_render else current_cite_details
    unlinked_reference_candidates = _build_unlinked_reference_candidates(
        answer_markdown=answer_markdown,
        rendered_body=rendered_body,
        copy_text=copy_text,
        cite_details=selected_cite_details,
        ref_pack=ref_pack if isinstance(ref_pack, dict) else None,
        provenance_segments=provenance_segments,
        render_locale=render_locale,
        anchor_ns=f"unlinked:{msg_id}",
    )
    render_packet_model = _build_paper_guide_render_packet_model(
        answer_markdown=answer_markdown,
        notice=notice,
        rendered_body=rendered_body,
        rendered_content=rendered_content,
        copy_markdown=copy_markdown,
        copy_text=copy_text,
        cite_details=selected_cite_details,
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
        unlinked_reference_candidates=unlinked_reference_candidates,
    )
    render_packet = _paper_guide_model_dump(render_packet_model)
    cache_changed = _sync_render_cache_packet(meta, render_packet)
    if existing_packet == render_packet:
        if contracts_changed or cache_changed:
            meta["paper_guide_contracts"] = contracts
            rec["meta"] = meta
            if chat_store is not None and msg_id > 0:
                try:
                    patch = {"paper_guide_contracts": contracts}
                    if cache_changed and isinstance(meta.get("render_cache"), dict):
                        patch["render_cache"] = dict(meta.get("render_cache") or {})
                    chat_store.merge_message_meta(msg_id, patch)
                except Exception:
                    pass
        else:
            rec["meta"] = meta
        return
    contracts["render_packet"] = render_packet
    meta["paper_guide_contracts"] = contracts
    rec["meta"] = meta
    if chat_store is not None and msg_id > 0:
        try:
            patch = {"paper_guide_contracts": contracts}
            if cache_changed and isinstance(meta.get("render_cache"), dict):
                patch["render_cache"] = dict(meta.get("render_cache") or {})
            chat_store.merge_message_meta(msg_id, patch)
        except Exception:
            pass


def _project_render_packet_compat_fields(rec: dict) -> None:
    meta = dict(rec.get("meta") or {}) if isinstance(rec.get("meta"), dict) else {}
    contracts = dict(meta.get("paper_guide_contracts") or {}) if isinstance(meta.get("paper_guide_contracts"), dict) else {}
    packet = dict(contracts.get("render_packet") or {}) if isinstance(contracts.get("render_packet"), dict) else {}
    if not project_render_packet_to_record(rec, packet):
        return
    rec["meta"] = meta


def _maybe_strip_legacy_render_fields(rec: dict, *, enabled: bool) -> None:
    if not enabled and not _env_flag("KB_CHAT_RENDER_PACKET_ONLY", "0"):
        return
    # Keep core identity fields; strip legacy render projections from response payload.
    strip_legacy_render_fields(rec)


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
    render_locale = _effective_citation_render_locale(None)
    has_visible_direct_segment = any(
        isinstance(item, dict)
        and str(item.get("evidence_mode") or "").strip().lower() == "direct"
        and str(item.get("locate_policy") or "").strip().lower() != "hidden"
        and (
            str(item.get("primary_block_id") or "").strip()
            or any(str(block_id or "").strip() for block_id in list(item.get("evidence_block_ids") or []))
        )
        for item in list(segments_raw or [])
    )
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
            rendered_segment, cite_details = _call_with_optional_render_locale(
                _annotate_inpaper_citations_with_hover_meta,
                rendered_segment,
                hits,
                anchor_ns=f"{anchor_ns}:seg:{idx}",
                render_locale=render_locale,
            )
            if _should_retry_structured_cite_fallback(
                raw_body=raw_markdown,
                rendered_body=rendered_segment,
                cite_details=cite_details,
            ):
                rendered_segment, cite_details = _call_with_optional_render_locale(
                    _fallback_render_structured_citations,
                    raw_markdown,
                    hits,
                    anchor_ns=f"{anchor_ns}:seg:{idx}",
                    render_locale=render_locale,
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
        promoted_seg = (
            {}
            if has_visible_direct_segment
            else _paper_guide_promote_hidden_direct_segment_for_render(seg)
        )
        if promoted_seg:
            seg = promoted_seg
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


def _fallback_render_structured_citations(md: str, hits: list[dict], *, anchor_ns: str, render_locale: str = "") -> tuple[str, list[dict]]:
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

    def _structured_cite_context_line(token_start: int, token_end: int) -> str:
        return extract_structured_cite_answer_context_line(
            str(md or ""),
            int(token_start),
            int(token_end),
            normalizer=_md_to_plain_text,
        )

    def _mk_detail(sid: str, ref_num: int, *, answer_context: str = "") -> dict | None:
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
            "is_inpaper": True,
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
            "render_locale": str(render_locale or "").strip().lower(),
        }
        local_answer_context = str(answer_context or "").strip()
        enrich_inpaper_detail_context(
            rec,
            source_path=source_path,
            ref_num=int(ref_num),
            answer_context=local_answer_context,
            source_answer_context=local_answer_context or str(md or "")[:4000],
        )
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
        context_line = _structured_cite_context_line(int(m.start()), int(m.end()))
        detail = _mk_detail(sid, n, answer_context=context_line)
        if not detail:
            return ""
        return f"[{n}](#{detail['anchor']})"

    out = _STRUCT_CITE_RE.sub(_replace, str(md or ""))
    out = _STRUCT_CITE_SINGLE_RE.sub(_replace, out)
    out = _STRUCT_CITE_SID_ONLY_RE.sub("", out)
    out = _STRUCT_CITE_GARBAGE_RE.sub("", out)
    details = [
        compose_citation_card(item, locale=render_locale)
        for item in sorted(details_by_key.values(), key=lambda item: (int(item.get("num") or 0), str(item.get("source_name") or "")))
    ]
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
        render_source = _message_render_source_markdown(rec, content)
        try:
            msg_id = int(rec.get("id") or 0)
        except Exception:
            msg_id = 0

        if role == "user":
            if msg_id > 0:
                last_user_msg_id = msg_id
            out.append(rec)
            continue

        raw_ref_pack = refs_by_user.get(last_user_msg_id) if isinstance(refs_by_user, dict) else None
        ref_pack = _effective_reference_render_pack(raw_ref_pack if isinstance(raw_ref_pack, dict) else None)
        render_locale = _effective_citation_render_locale(ref_pack if isinstance(ref_pack, dict) else None)
        if isinstance(raw_ref_pack, dict) and isinstance(ref_pack, dict) and ref_pack:
            raw_ref_pack.update(ref_pack)
        hits = list((ref_pack or {}).get("hits") or []) if isinstance(ref_pack, dict) else []
        provenance_raw = rec.get("provenance") if isinstance(rec.get("provenance"), dict) else None
        render_cache_key = _build_message_render_cache_key(
            conv_id=conv_id,
            msg_id=msg_id,
            role=role,
            content=render_source,
            refs_user_msg_id=int(last_user_msg_id or 0),
            ref_pack=ref_pack if isinstance(ref_pack, dict) else None,
            provenance=provenance_raw if isinstance(provenance_raw, dict) else None,
            render_locale=render_locale,
        )
        cached = _extract_render_cache(
            rec.get("meta") if isinstance(rec.get("meta"), dict) else None,
            expected_key=render_cache_key,
            raw_content=render_source,
            hits=hits,
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
            notice, body = _split_kb_miss_notice(render_source)
            if notice and hits:
                notice = ""
                body = render_source
            cite_details: list[dict] = []
            rendered_body = str(body or "")
            raw_body = rendered_body
            citation_plan = _message_citation_plan(rec)
            allow_inpaper_citation_linking = _should_link_inpaper_citations_for_message(
                rec=rec,
                content=render_source,
                hits=hits,
            )
            if rendered_body.strip():
                rendered_body = _annotate_equation_tags_with_sources(rendered_body, hits)
                rendered_body = _normalize_equation_source_notes(rendered_body)
                rendered_body, linked_named_system_b = _repair_named_system_b_citation_markers(
                    rendered_body,
                    hits,
                    citation_plan,
                )
                allow_inpaper_citation_linking = bool(allow_inpaper_citation_linking or linked_named_system_b)
                if allow_inpaper_citation_linking:
                    # Pass canonical hit ordering if available, so [n] resolves to
                    # the same source the LLM referenced during generation.
                    _rec_meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
                    _canon_paths = list(_rec_meta.get("canonical_hit_paths") or []) if isinstance(_rec_meta.get("canonical_hit_paths"), list) else []
                    annotate_kwargs = {
                        "anchor_ns": f"{conv_id}:{idx}:{msg_id}:api",
                        "canonical_paths": _canon_paths or None,
                    }
                    if citation_plan:
                        annotate_kwargs["citation_plan"] = citation_plan
                    rendered_body = _reading_guide_repair_missing_system_a_citations(
                        rendered_body,
                        hits,
                        citation_plan,
                        output_mode=_message_answer_output_mode(rec),
                        canonical_paths=_canon_paths or None,
                    )
                    rendered_body, cite_details = _call_with_optional_render_locale(
                        _annotate_inpaper_citations_with_hover_meta,
                        rendered_body,
                        hits,
                        render_locale=render_locale,
                        **annotate_kwargs,
                    )
                    if _should_retry_structured_cite_fallback(
                        raw_body=raw_body,
                        rendered_body=rendered_body,
                        cite_details=cite_details,
                    ) and _citation_plan_system_b_budget(citation_plan) > 0:
                        rendered_body, cite_details = _call_with_optional_render_locale(
                            _fallback_render_structured_citations,
                            raw_body,
                            hits,
                            anchor_ns=f"{conv_id}:{idx}:{msg_id}:api",
                            render_locale=render_locale,
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
                rendered_full = render_source or content

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
            ref_pack=ref_pack if isinstance(ref_pack, dict) else None,
            chat_store=chat_store,
            render_locale=render_locale,
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
