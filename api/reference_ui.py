from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import hashlib
import html
import math
import os
from pathlib import Path
from urllib.parse import quote
import re

from kb.config import load_settings
from kb.citation_meta import fetch_best_crossref_for_reference, fetch_best_crossref_meta, fetch_crossref_work_by_doi
from kb.file_naming import citation_meta_display_pdf_name
from kb.library_store import LibraryStore
from kb.llm import DeepSeekChat
from kb.source_filters import is_excluded_source_path
from ui.refs_renderer import (
    _enrich_bibliometrics,
    _fallback_fill_reference_meta_from_raw,
    _has_metrics_payload,
    _infer_title_from_source_text,
    _normalize_reference_for_popup,
    _parse_filename_meta,
    _build_ref_navigation,
    _fallback_why_line_ui,
    _is_non_navigational_heading_ui,
    _looks_like_doc_title_heading_ui,
    _open_pdf_at,
    _resolve_pdf_for_source,
    _safe_page_range,
    _sanitize_heading_path_ui,
    _score_tier,
    _split_section_subsection,
    _top_heading,
    _openalex_work_by_doi,
    fetch_crossref_meta,
)

_MIN_REF_UI_SCORE = 5.2
_MAX_REF_UI_GAP = 1.8


def _source_filename(source_path: str) -> str:
    s = str(source_path or "").strip()
    if not s:
        return ""
    parts = re.split(r"[\\/]+", s)
    return str(parts[-1] or "").strip() if parts else s


def _source_identity_keys(source_path: str) -> set[str]:
    raw = str(source_path or "").strip()
    if not raw:
        return set()
    out: set[str] = set()
    norm = raw.replace("\\", "/").strip().lower()
    if norm:
        out.add(norm)

    name = _source_filename(raw).strip().lower()
    if name:
        out.add(name)
        if name.endswith(".en.md"):
            pdf_name = name[:-6] + ".pdf"
            stem_name = name[:-6]
            out.add(pdf_name)
            out.add(stem_name)
        elif name.endswith(".md"):
            pdf_name = name[:-3] + ".pdf"
            stem_name = name[:-3]
            out.add(pdf_name)
            out.add(stem_name)
    return {item for item in out if item}


def _same_source_identity(source_path: str, bound_source_path: str) -> bool:
    left = _source_identity_keys(source_path)
    right = _source_identity_keys(bound_source_path)
    if not left or not right:
        return False
    return bool(left.intersection(right))


def _clamp_ui_score(score: float) -> float:
    try:
        v = float(score)
    except Exception:
        v = 0.0
    return max(0.0, min(10.0, v))


def _stable_score_micro_jitter(source_path: str) -> float:
    """Small deterministic jitter to avoid repeated identical decimals (e.g. *.76)."""
    s = str(source_path or "").strip()
    if not s:
        return 0.0
    try:
        h = hashlib.sha1(s.encode("utf-8", "ignore")).digest()
        u = int.from_bytes(h[:2], "big") / 65535.0  # 0..1
    except Exception:
        return 0.0
    return (u - 0.5) * 0.08  # about [-0.04, +0.04]


def _calibrated_ui_score(meta: dict, rank: dict) -> float | None:
    try:
        llm_score = float(rank.get("llm", 0.0) or 0.0)
    except Exception:
        llm_score = 0.0
    if llm_score <= 0:
        return None

    try:
        bm25 = float(rank.get("bm25", 0.0) or 0.0)
    except Exception:
        bm25 = 0.0
    try:
        deep = float(rank.get("deep", 0.0) or 0.0)
    except Exception:
        deep = 0.0
    try:
        term_bonus = float(rank.get("term_bonus", 0.0) or 0.0)
    except Exception:
        term_bonus = 0.0
    try:
        semantic_score = float(rank.get("semantic_score", 0.0) or 0.0)
    except Exception:
        semantic_score = 0.0

    llm_ui = llm_score / 10.0

    # Build an evidence-driven UI component from retrieval signals.
    # Use smooth transforms to keep score spread continuous and avoid repeated
    # fixed decimal tails from a single signal source.
    evidence_ui = 5.0
    evidence_ui += 1.8 * math.tanh((bm25 - 2.5) / 3.0)
    evidence_ui += 1.2 * math.tanh((deep - 1.5) / 4.0)
    evidence_ui += 0.9 * math.tanh(term_bonus / 1.8)
    if semantic_score > 0:
        evidence_ui = (0.82 * evidence_ui) + (0.18 * _clamp_ui_score(semantic_score))
    evidence_ui = _clamp_ui_score(evidence_ui)

    # Blend LLM relevance with retrieval evidence.
    ui = (0.64 * llm_ui) + (0.36 * evidence_ui)

    if term_bonus < 0:
        ui += 0.60 * term_bonus
    elif term_bonus > 0:
        ui += min(0.30, 0.12 * term_bonus)

    if bm25 < 1.0:
        ui -= 1.15
    elif bm25 < 2.0:
        ui -= 0.75
    elif bm25 < 3.5:
        ui -= 0.35

    if deep <= 0:
        ui -= 0.15

    section = str(
        meta.get("ref_section")
        or ((meta.get("ref_pack") or {}).get("section") if isinstance(meta.get("ref_pack"), dict) else "")
        or meta.get("ref_best_heading_path")
        or meta.get("heading_path")
        or ""
    ).strip()
    loc_quality = str(meta.get("ref_loc_quality") or "").strip().lower()
    if not section:
        ui -= 0.70
    elif loc_quality != "high":
        ui -= 0.25

    # Add continuous spread from evidence.
    try:
        bm25_spread = max(-1.0, min(1.0, math.tanh((bm25 - 3.0) / 4.0)))
    except Exception:
        bm25_spread = 0.0
    try:
        deep_spread = max(-1.0, min(1.0, math.tanh((deep - 2.0) / 6.0)))
    except Exception:
        deep_spread = 0.0
    ui += (0.14 * bm25_spread) + (0.12 * deep_spread)

    # Deterministic micro-jitter by source to break exact ties.
    ui += _stable_score_micro_jitter(str(meta.get("source_path") or ""))

    # Do not allow weak lexical evidence to surface as "high relevance"
    # just because the LLM was optimistic.
    if term_bonus <= 0.0 and bm25 < 2.0:
        ui = min(ui, 6.4)
    if term_bonus <= 0.0 and bm25 < 1.0:
        ui = min(ui, 5.8)
    if term_bonus <= 0.0 and (not section):
        ui = min(ui, 5.6)

    return _clamp_ui_score(ui)

def _effective_ui_score(hit: dict) -> tuple[float | None, bool]:
    meta = (hit or {}).get("meta", {}) or {}
    pack_state = str(meta.get("ref_pack_state") or "").strip().lower()
    rank = meta.get("ref_rank") if isinstance(meta.get("ref_rank"), dict) else {}
    if pack_state == "ready":
        calibrated = _calibrated_ui_score(meta, rank)
        if calibrated is not None:
            return calibrated, False
    return None, pack_state == "pending"


def _should_force_keep_ref_hit(hit: dict) -> bool:
    meta = (hit or {}).get("meta", {}) or {}
    if not isinstance(meta, dict):
        return False
    if str(meta.get("ref_pack_state") or "").strip().lower() == "pending":
        return True
    try:
        explicit_doc = float(meta.get("explicit_doc_match_score") or 0.0)
    except Exception:
        explicit_doc = 0.0
    if explicit_doc >= 6.0:
        return True
    if str(meta.get("anchor_target_kind") or "").strip():
        try:
            anchor_score = float(meta.get("anchor_match_score") or 0.0)
        except Exception:
            anchor_score = 0.0
        if anchor_score > 0.0:
            return True
    return False


def _display_source_name(source_path: str, pdf_path: Path | None, lib_store: LibraryStore | None) -> str:
    try:
        if pdf_path is not None and lib_store is not None:
            meta = lib_store.get_citation_meta(pdf_path)
            full_name = citation_meta_display_pdf_name(meta)
            if full_name:
                return full_name
    except Exception:
        pass

    name = _source_filename(source_path) or str(source_path or "")
    low = name.lower()
    if low.endswith(".en.md"):
        name = name[:-6] + ".pdf"
    elif low.endswith(".md"):
        name = name[:-3] + ".pdf"
    return name or "unknown.pdf"


def _positive_int(x) -> int:
    try:
        v = int(x)
    except Exception:
        return 0
    return v if v > 0 else 0


def _non_negative_float(x) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return v if v > 0.0 else 0.0


def _anchor_kind_prefix(kind: str) -> str:
    k = str(kind or "").strip().lower()
    if k == "figure":
        return "图示语义命中"
    if k == "equation":
        return "公式语义命中"
    if k == "table":
        return "表格语义命中"
    if k == "theorem":
        return "定理语义命中"
    if k == "lemma":
        return "引理语义命中"
    if k == "definition":
        return "定义语义命中"
    return "锚点语义命中"


def _anchor_kind_label(kind: str, number: int) -> str:
    k = str(kind or "").strip().lower()
    n = _positive_int(number)
    if (not k) or n <= 0:
        return ""
    if k == "figure":
        return f"图{n}"
    if k == "equation":
        return f"公式{n}"
    if k == "table":
        return f"表{n}"
    if k == "theorem":
        return f"定理{n}"
    if k == "lemma":
        return f"引理{n}"
    if k == "definition":
        return f"定义{n}"
    return f"{k} {n}"


def _build_semantic_badges(
    *,
    anchor_target_kind: str,
    anchor_target_number: int,
    anchor_match_score: float,
    explicit_doc_match_score: float,
) -> list[dict]:
    badges: list[dict] = []
    anchor_label = _anchor_kind_label(anchor_target_kind, anchor_target_number)
    if anchor_label:
        badges.append(
            {
                "text": f"{_anchor_kind_prefix(anchor_target_kind)} {anchor_label}",
                "score": _non_negative_float(anchor_match_score),
            }
        )
        return badges
    if _non_negative_float(explicit_doc_match_score) >= 6.0:
        badges.append({"text": "文档语义直连", "score": _non_negative_float(explicit_doc_match_score)})
    return badges


def _fallback_ref_ui_summary_line(
    meta: dict,
    *,
    prompt: str,
    citation_meta: dict | None = None,
) -> str:
    prompt_is_cjk = _has_cjk_text(prompt)
    title = str((citation_meta or {}).get("title") or (meta or {}).get("title") or "").strip()

    def _normalize_candidate(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        cand = _summary_excerpt(raw, max_sentences=2, max_len=360)
        if not cand:
            cand = _first_summary_sentence(raw, max_len=220)
        if not cand:
            return ""
        if _looks_like_title_echo(cand, title):
            return ""
        if prompt_is_cjk:
            cand = _translate_summary_to_zh(cand)
        cand = _summary_excerpt(cand, max_sentences=2, max_len=360)
        if not cand:
            return ""
        if _looks_like_title_echo(cand, title):
            return ""
        return cand

    for key in ("ref_show_snippets",):
        raw_arr = meta.get(key)
        if not isinstance(raw_arr, list):
            continue
        for item in raw_arr[:3]:
            cand = _normalize_candidate(str(item or ""))
            if cand:
                return cand

    citation_summary = _normalize_candidate(str((citation_meta or {}).get("summary_line") or ""))
    if citation_summary:
        return citation_summary

    for key in ("ref_overview_snippets",):
        raw_arr = meta.get(key)
        if not isinstance(raw_arr, list):
            continue
        for item in raw_arr[:3]:
            cand = _normalize_candidate(str(item or ""))
            if cand:
                return cand
    return ""


def build_hit_ui_meta(
    hit: dict,
    *,
    prompt: str,
    pdf_root: Path | None,
    lib_store: LibraryStore | None,
    preloaded_citation_meta: dict[str, dict] | None = None,
) -> dict:
    meta = (hit or {}).get("meta", {}) or {}
    source_path = str(meta.get("source_path") or "").strip()
    heading_path = _sanitize_heading_path_ui(
        str(meta.get("ref_best_heading_path") or meta.get("heading_path") or "").strip(),
        prompt=prompt,
        source_path=source_path,
    )
    heading = str(
        meta.get("top_heading")
        or _top_heading(heading_path)
        or _top_heading(str(meta.get("heading_path") or ""))
        or ""
    ).strip()
    if heading and _is_non_navigational_heading_ui(heading, prompt=prompt, source_path=source_path):
        heading = ""
    if heading and _looks_like_doc_title_heading_ui(heading, source_path):
        heading = ""

    section_label = str(meta.get("ref_section") or "").strip()
    subsection_label = str(meta.get("ref_subsection") or "").strip()
    if section_label and _is_non_navigational_heading_ui(section_label, prompt=prompt, source_path=source_path):
        section_label = ""
    if subsection_label and _is_non_navigational_heading_ui(subsection_label, prompt=prompt, source_path=source_path):
        subsection_label = ""
    if (not section_label) and heading_path:
        section_label, subsection_label = _split_section_subsection(heading_path)
    if section_label and _looks_like_doc_title_heading_ui(section_label, source_path):
        section_label = ""
        subsection_label = ""

    p0, p1 = _safe_page_range(meta)
    score, score_pending = _effective_ui_score(hit)
    anchor_target_kind = str(meta.get("anchor_target_kind") or "").strip().lower()
    anchor_target_number = _positive_int(meta.get("anchor_target_number"))
    anchor_match_score = _non_negative_float(meta.get("anchor_match_score"))
    explicit_doc_match_score = _non_negative_float(meta.get("explicit_doc_match_score"))
    semantic_badges = _build_semantic_badges(
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        anchor_match_score=anchor_match_score,
        explicit_doc_match_score=explicit_doc_match_score,
    )
    pdf_path = _resolve_pdf_for_source(pdf_root, source_path)
    display_name = _display_source_name(source_path, pdf_path, lib_store)
    citation_meta = {}
    preload_map = preloaded_citation_meta if isinstance(preloaded_citation_meta, dict) else {}
    preload_meta = preload_map.get(source_path) if source_path else None
    if isinstance(preload_meta, dict) and preload_meta:
        citation_meta = dict(preload_meta)
    if pdf_path is not None and lib_store is not None:
        try:
            if not citation_meta:
                citation_meta = lib_store.get_citation_meta(pdf_path) or {}
        except Exception:
            if not citation_meta:
                citation_meta = {}

    nav = _build_ref_navigation(meta, prompt=prompt, heading_fallback=heading)
    summary_line = str(nav.get("summary_line") or nav.get("what") or "").strip()
    if not summary_line:
        summary_line = _fallback_ref_ui_summary_line(meta, prompt=prompt, citation_meta=citation_meta)
    why_line = str(nav.get("why") or "").strip()
    if not why_line:
        why_line = _fallback_why_line_ui(
            prompt=prompt,
            heading_label=heading_path or heading,
            section_label=section_label,
            subsection_label=subsection_label,
            find_terms=list(nav.get("find") or []),
        )

    return {
        "display_name": display_name,
        "heading_path": heading_path or heading,
        "section_label": section_label,
        "subsection_label": subsection_label,
        "page_start": p0,
        "page_end": p1,
        "score": score,
        "score_pending": bool(score_pending),
        "score_tier": _score_tier(score or 0.0) if score is not None else "",
        "summary_line": summary_line,
        "why_line": why_line,
        "anchor_target_kind": anchor_target_kind,
        "anchor_target_number": anchor_target_number,
        "anchor_match_score": anchor_match_score,
        "explicit_doc_match_score": explicit_doc_match_score,
        "semantic_badges": semantic_badges,
        "can_open": bool(pdf_path),
        "citation_meta": citation_meta if isinstance(citation_meta, dict) else {},
        "source_path": source_path,
    }


def _should_prefetch_citation_meta(meta: dict | None) -> bool:
    if not isinstance(meta, dict) or (not meta):
        return True
    if not _has_metrics_payload(meta):
        return True
    if not str(meta.get("doi") or meta.get("doi_url") or "").strip():
        return True
    if not str(meta.get("venue") or meta.get("conference_name") or "").strip():
        return True
    if not str(meta.get("year") or "").strip():
        return True
    return False


def _prefetch_refs_citation_meta(
    hits: list[dict],
    *,
    pdf_root: Path | None,
    md_root: Path | None,
    lib_store: LibraryStore | None,
) -> dict[str, dict]:
    tasks: dict[str, tuple[str, dict]] = {}
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or (ui_meta or {}).get("source_path") or "").strip()
        if (not source_path) or is_excluded_source_path(source_path):
            continue
        existing = (ui_meta or {}).get("citation_meta")
        existing_meta = existing if isinstance(existing, dict) else {}
        if (source_path in tasks) or (not _should_prefetch_citation_meta(existing_meta)):
            continue
        tasks[source_path] = (source_path, existing_meta)
    if not tasks:
        return {}

    out: dict[str, dict] = {}

    def _one(source_path: str) -> tuple[str, dict]:
        meta = ensure_source_citation_meta(
            source_path=source_path,
            pdf_root=pdf_root,
            md_root=md_root,
            lib_store=lib_store,
        )
        return source_path, (meta if isinstance(meta, dict) else {})

    max_workers = max(1, min(4, len(tasks)))
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_one, source_path) for source_path in tasks.keys()]
            for fu in as_completed(futs):
                try:
                    source_path, meta = fu.result()
                except Exception:
                    continue
                out[source_path] = meta
    except Exception:
        for source_path in tasks.keys():
            try:
                source_path2, meta = _one(source_path)
            except Exception:
                continue
            out[source_path2] = meta
    return out


def enrich_refs_payload(
    refs_by_user: dict[int, dict],
    *,
    pdf_root: Path | None,
    md_root: Path | None,
    lib_store: LibraryStore | None,
    guide_mode: bool = False,
    guide_source_path: str = "",
    guide_source_name: str = "",
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    guide_source_path_norm = str(guide_source_path or "").strip()
    guide_source_name_norm = str(guide_source_name or "").strip()
    guide_active = bool(guide_mode and guide_source_path_norm)
    for user_msg_id, pack in (refs_by_user or {}).items():
        if not isinstance(pack, dict):
            continue
        prompt = str(pack.get("prompt") or "").strip()
        raw_hits = []
        scored_ready: list[float] = []
        filtered_self_hits = 0
        for hit in list(pack.get("hits") or []):
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            source_path = str((meta or {}).get("source_path") or "").strip()
            if is_excluded_source_path(source_path):
                continue
            if guide_active and _same_source_identity(source_path, guide_source_path_norm):
                filtered_self_hits += 1
                continue
            raw_hits.append(dict(hit))
            score, score_pending = _effective_ui_score(hit)
            if (not score_pending) and (score is not None):
                scored_ready.append(float(score))
        best_ready = max(scored_ready) if scored_ready else None
        dyn_keep_min = max(_MIN_REF_UI_SCORE, (best_ready - _MAX_REF_UI_GAP)) if best_ready is not None else _MIN_REF_UI_SCORE
        has_pending = any(
            str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("ref_pack_state") or "")).strip().lower() == "pending"
            for hit in raw_hits
        )

        hits = []
        for hit2 in raw_hits:
            score, score_pending = _effective_ui_score(hit2)
            force_keep = _should_force_keep_ref_hit(hit2)
            if has_pending:
                hits.append(hit2)
                continue
            if force_keep:
                hits.append(hit2)
                continue
            if (score is None) and (not score_pending):
                continue
            if (not score_pending) and (score is not None) and score < dyn_keep_min:
                continue
            hits.append(hit2)
        if (not hits) and raw_hits:
            fallback_hit = next((hit for hit in raw_hits if _should_force_keep_ref_hit(hit)), None)
            if fallback_hit is not None:
                hits = [fallback_hit]
        preloaded_citation_meta = _prefetch_refs_citation_meta(
            hits,
            pdf_root=pdf_root,
            md_root=md_root,
            lib_store=lib_store,
        ) if hits else {}
        for hit2 in hits:
            hit2["ui_meta"] = build_hit_ui_meta(
                hit2,
                prompt=prompt,
                pdf_root=pdf_root,
                lib_store=lib_store,
                preloaded_citation_meta=preloaded_citation_meta,
            )
        pack2 = dict(pack)
        pack2["hits"] = hits
        if guide_active:
            pack2["guide_filter"] = {
                "active": True,
                "hidden_self_source": bool(filtered_self_hits > 0),
                "filtered_hit_count": int(filtered_self_hits),
                "guide_source_path": guide_source_path_norm,
                "guide_source_name": guide_source_name_norm or _source_filename(guide_source_path_norm),
            }
        out[int(user_msg_id)] = pack2
    return out


def open_reference_source(*, source_path: str, pdf_root: Path | None, page: int | None = None) -> tuple[bool, str]:
    pdf_path = _resolve_pdf_for_source(pdf_root, source_path)
    if pdf_path is None:
        return False, "PDF not found"
    return _open_pdf_at(pdf_path, page=page)


def build_doi_url(doi_or_url: str) -> str:
    raw = str(doi_or_url or "").strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    return "https://doi.org/" + quote(raw, safe="/:;._-()")


def _is_weak_meta_value(key: str, value: str) -> bool:
    s = str(value or "").strip()
    if not s:
        return True
    if key == "title":
        if len(s) <= 4:
            return True
        if len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", s)) <= 1:
            return True
        if re.fullmatch(r"[A-Za-z][A-Za-z.\s&-]{1,40}\(\d{4}\)\.?", s):
            return True
        if re.fullmatch(r"[A-Za-z][A-Za-z.\s&-]{1,40}\d{4}\.?", s):
            return True
    if key == "authors":
        if len(s) <= 3:
            return True
        if len(re.findall(r"[A-Za-z\u4e00-\u9fff]+", s)) <= 1:
            return True
    if key == "venue":
        if len(s) <= 1:
            return True
    return False


def _normalize_doi_like(value: str) -> str:
    s = str(value or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s, flags=re.I)
    s = s.strip(" \t\r\n.,;:()[]{}<>")
    return s


def _clean_summary_line(text: str) -> str:
    s = html.unescape(str(text or ""))
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\[[0-9,\-\s]{1,24}\]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(?:abstract|摘要)\s*[:：-]?\s*", "", s, flags=re.I).strip()
    if len(s) < 20:
        return ""
    return s


def _first_summary_sentence(text: str, *, max_len: int = 220) -> str:
    s = _clean_summary_line(text)
    if not s:
        return ""
    parts = re.split(r"(?<=[。！？!?\.])\s+", s)
    for part in parts:
        cand = str(part or "").strip()
        if len(cand) < 20:
            continue
        if len(cand) > max_len:
            cand = cand[:max_len].rstrip(" ,;:") + "..."
        return cand
    if len(s) > max_len:
        return s[:max_len].rstrip(" ,;:") + "..."
    return s


def _summary_excerpt(text: str, *, max_sentences: int = 3, max_len: int = 520) -> str:
    s = _clean_summary_line(text)
    if not s:
        return ""
    parts = re.split(r"(?<=[。！？!?\.])\s+", s)
    picked: list[str] = []
    total = 0
    for part in parts:
        cand = str(part or "").strip()
        if len(cand) < 18:
            continue
        if (total + len(cand)) > max_len:
            remain = max_len - total
            if remain >= 30:
                picked.append(cand[:remain].rstrip(" ,;:") + "...")
            break
        picked.append(cand)
        total += len(cand)
        if len(picked) >= max_sentences:
            break
    if picked:
        return " ".join(picked).strip()
    if len(s) > max_len:
        return s[:max_len].rstrip(" ,;:") + "..."
    return s


def _metadata_summary_line(meta: dict) -> str:
    title = _clean_summary_line(str((meta or {}).get("title") or ""))
    venue = _clean_summary_line(str((meta or {}).get("venue") or ""))
    year = str((meta or {}).get("year") or "").strip()
    authors = _clean_summary_line(str((meta or {}).get("authors") or ""))
    author_head = ""
    if authors:
        author_head = re.split(r"[,;&]| and ", authors, maxsplit=1, flags=re.I)[0].strip()
    loc = ""
    if venue and year:
        loc = f"{venue}（{year}）"
    elif venue:
        loc = venue
    elif year:
        loc = year
    if author_head and loc:
        return (
            f"当前仅检索到文献元数据：{author_head} 的相关研究发表于 {loc}。"
            "由于缺少可用摘要文本，暂无法可靠提炼其方法细节与实验结论，建议通过 DOI 查看原文摘要与正文。"
        )
    if loc:
        return (
            f"当前仅检索到文献元数据：该工作发表于 {loc}。"
            "由于缺少可用摘要文本，暂无法可靠提炼其方法细节与实验结论，建议通过 DOI 查看原文摘要与正文。"
        )
    if title:
        return (
            "当前仅检索到题名与基础元数据，尚未获取可用摘要文本。"
            "为保证学术准确性，建议通过 DOI 查看原文摘要与正文后再进行方法和结论层面的判断。"
        )
    return (
        "当前仅检索到有限元数据，尚未获取可用摘要文本。"
        "为保证学术准确性，建议通过 DOI 查看原文摘要与正文后再进行方法和结论层面的判断。"
    )


def _summary_from_crossref_abstract(meta: dict) -> str:
    doi_like = str((meta or {}).get("doi") or (meta or {}).get("doi_url") or "").strip()
    doi = _normalize_doi_like(doi_like)
    if not doi:
        return ""
    try:
        work = fetch_crossref_work_by_doi(doi)
    except Exception:
        work = None
    if not isinstance(work, dict):
        return ""
    abstract = str(work.get("abstract") or "").strip()
    if not abstract:
        return ""
    return _summary_excerpt(abstract, max_sentences=3, max_len=520)


def _openalex_abstract_text(work: dict) -> str:
    if not isinstance(work, dict):
        return ""
    raw_abs = str(work.get("abstract") or "").strip()
    if raw_abs:
        return raw_abs
    inv = work.get("abstract_inverted_index")
    if not isinstance(inv, dict):
        return ""
    words: list[tuple[int, str]] = []
    for token, positions in inv.items():
        if not isinstance(token, str):
            continue
        if not isinstance(positions, list):
            continue
        for p in positions:
            try:
                pos = int(p)
            except Exception:
                continue
            if pos < 0:
                continue
            words.append((pos, token))
    if not words:
        return ""
    words.sort(key=lambda x: x[0])
    return " ".join(w for _, w in words).strip()


def _summary_from_openalex_abstract(meta: dict) -> str:
    doi_like = str((meta or {}).get("doi") or (meta or {}).get("doi_url") or "").strip()
    doi = _normalize_doi_like(doi_like)
    if not doi:
        return ""
    try:
        work = _openalex_work_by_doi(doi)
    except Exception:
        work = None
    abstract = _openalex_abstract_text(work if isinstance(work, dict) else {})
    if not abstract:
        return ""
    return _summary_excerpt(abstract, max_sentences=3, max_len=520)


def _looks_like_title_echo(summary_line: str, title: str) -> bool:
    s = _clean_summary_line(summary_line).lower()
    t = _clean_summary_line(title).lower()
    if (not s) or (not t):
        return False
    s_norm = "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]+", s))
    t_norm = "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]+", t))
    if (not s_norm) or (not t_norm):
        return False
    if (t_norm in s_norm) and (len(s_norm) <= len(t_norm) + 36):
        return True
    s_tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", s)
    t_tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", t)
    if (len(t_tokens) >= 4) and s_tokens:
        common = len(set(s_tokens) & set(t_tokens))
        if common >= max(3, int(0.85 * len(set(t_tokens)))) and len(set(s_tokens)) <= len(set(t_tokens)) + 3:
            return True
    return False


def _has_cjk_text(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _has_latin_text(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", str(text or "")))


def _has_summary_action_signal(text: str) -> bool:
    s = str(text or "")
    return bool(re.search(r"(提出|设计|构建|采用|引入|实现|develop|propose|introduce|present)", s, flags=re.I))


def _has_summary_result_signal(text: str) -> bool:
    s = str(text or "")
    return bool(re.search(r"(结果|显示|提升|降低|加速|优于|有效|性能|实验|result|show|improv|outperform|achiev)", s, flags=re.I))


def _is_summary_quality_ok(text: str) -> bool:
    s = _clean_summary_line(text)
    if len(s) < 50:
        return False
    if not _has_summary_action_signal(s):
        return False
    if not _has_summary_result_signal(s):
        return False
    return True


@lru_cache(maxsize=512)
def _llm_summarize_abstract_zh(title: str, abstract_text: str) -> str:
    abs_text = _summary_excerpt(abstract_text, max_sentences=5, max_len=900)
    title_text = _clean_summary_line(title)
    if not abs_text:
        return ""
    raw_flag = str(os.environ.get("KB_CITE_SUMMARY_USE_LLM", "1") or "").strip().lower()
    if raw_flag in {"0", "false", "off", "no"}:
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
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 10.0),
            max_retries=0,
        )
    except Exception:
        fast_settings = settings
    try:
        ds = DeepSeekChat(fast_settings)
        out = (
            ds.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是科研论文助手。请基于给定信息输出2-3句中文学术概括，要求："
                            "第1句说明研究问题或目标；"
                            "第2句说明核心方法或机制（作者具体做了什么）；"
                            "第3句说明关键结果、贡献或适用边界（若摘要未给量化指标需明确说明）。"
                            "严禁编造数据或结论，严禁只复述标题。只输出概括正文。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"论文标题：{title_text}\n"
                            f"摘要原文：{abs_text}\n\n"
                            "请给出中文学术概括："
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=360,
            )
            or ""
        ).strip()
    except Exception:
        return ""
    out = _summary_excerpt(out, max_sentences=3, max_len=360)
    if not _has_cjk_text(out):
        return ""
    if not _is_summary_quality_ok(out):
        return ""
    return out


@lru_cache(maxsize=512)
def _translate_summary_to_zh(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    src = _summary_excerpt(src, max_sentences=3, max_len=520)
    if not src:
        return ""
    if _has_cjk_text(src) and (not _has_latin_text(src)):
        return src
    raw_flag = str(os.environ.get("KB_CITE_SUMMARY_TRANSLATE_ZH", "1") or "").strip().lower()
    if raw_flag in {"0", "false", "off", "no"}:
        return src
    try:
        settings = load_settings()
    except Exception:
        return src
    if not getattr(settings, "api_key", None):
        return src
    try:
        fast_settings = replace(
            settings,
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 8.0),
            max_retries=0,
        )
    except Exception:
        fast_settings = settings
    try:
        ds = DeepSeekChat(fast_settings)
        out = (
            ds.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "将给定文献摘要改写为中文学术概括，输出 2-3 句。"
                            "要求："
                            "1) 尽量覆盖研究问题/方法/主要结果或贡献；"
                            "2) 术语准确、语气学术；"
                            "3) 不编造原文没有的信息；"
                            "4) 只输出概括正文，不要列表或前缀标签。"
                        ),
                    },
                    {"role": "user", "content": src},
                ],
                temperature=0.0,
                max_tokens=320,
            )
            or ""
        ).strip()
    except Exception:
        return src
    out = re.sub(r"\s+", " ", out).strip()
    if not out:
        return src
    if not _has_cjk_text(out):
        return src
    return _summary_excerpt(out, max_sentences=3, max_len=360)


def _ensure_summary_line(meta: dict, *, allow_crossref_abstract: bool) -> dict:
    out = dict(meta or {})
    existing_line = _summary_excerpt(str(out.get("summary_line") or ""), max_sentences=3, max_len=360)
    existing_source = str(out.get("summary_source") or "").strip().lower()
    title = str(out.get("title") or "").strip()
    if existing_line:
        if (existing_source == "metadata") and _looks_like_title_echo(existing_line, title):
            existing_line = ""
        else:
            out["summary_line"] = _translate_summary_to_zh(existing_line)
            out["summary_source"] = existing_source if existing_source in {"fulltext", "abstract", "metadata"} else "fulltext"
            return out

    if allow_crossref_abstract:
        abstract_line = _summary_from_crossref_abstract(out)
        if abstract_line:
            llm_summary = _llm_summarize_abstract_zh(title=title, abstract_text=abstract_line)
            out["summary_line"] = llm_summary or _translate_summary_to_zh(abstract_line)
            out["summary_source"] = "abstract"
            return out
        openalex_line = _summary_from_openalex_abstract(out)
        if openalex_line:
            llm_summary = _llm_summarize_abstract_zh(title=title, abstract_text=openalex_line)
            out["summary_line"] = llm_summary or _translate_summary_to_zh(openalex_line)
            out["summary_source"] = "abstract"
            return out

    fallback = _metadata_summary_line(out)
    if fallback:
        out["summary_line"] = fallback
        out["summary_source"] = "metadata"
    return out


def _merge_meta_prefer_richer(base: dict, incoming: dict) -> dict:
    out = dict(base or {})
    base_doi = _normalize_doi_like(str(out.get("doi") or out.get("doi_url") or ""))
    incoming_doi = _normalize_doi_like(str((incoming or {}).get("doi") or (incoming or {}).get("doi_url") or ""))
    doi_conflict = bool(base_doi and incoming_doi and (base_doi != incoming_doi))
    conflict_sensitive_keys = {
        "title",
        "authors",
        "venue",
        "year",
        "volume",
        "issue",
        "pages",
        "doi",
        "doi_url",
        "citation_count",
        "citation_source",
        "journal_if",
        "journal_quartile",
        "journal_if_source",
        "conference_tier",
        "conference_rank_source",
        "conference_ccf",
        "conference_ccf_source",
        "venue_kind",
        "openalex_venue",
        "conference_name",
        "conference_acronym",
        "bibliometrics_checked",
    }
    for key, raw_value in (incoming or {}).items():
        if raw_value in (None, "", [], {}):
            continue
        if doi_conflict and key in conflict_sensitive_keys:
            # Identity mismatch: keep current citation-level metadata.
            continue
        value = raw_value
        if not isinstance(value, str):
            out[key] = value
            continue
        cur = str(out.get(key) or "").strip()
        new = str(value or "").strip()
        if not cur:
            out[key] = new
            continue
        if key in {
            "doi",
            "doi_url",
            "citation_count",
            "citation_source",
            "journal_if",
            "journal_quartile",
            "journal_if_source",
            "conference_tier",
            "conference_rank_source",
            "conference_ccf",
            "conference_ccf_source",
            "venue_kind",
            "openalex_venue",
            "conference_name",
            "conference_acronym",
            "bibliometrics_checked",
        }:
            out[key] = value
            continue
        cur_weak = _is_weak_meta_value(key, cur)
        new_weak = _is_weak_meta_value(key, new)
        if cur_weak and (not new_weak):
            out[key] = new
            continue
        if (not cur_weak) and new_weak:
            continue
        if len(new) > len(cur) + 12:
            out[key] = new
    return out


def ensure_source_citation_meta(*, source_path: str, pdf_root: Path | None, md_root: Path | None, lib_store: LibraryStore | None) -> dict:
    pdf_path = _resolve_pdf_for_source(pdf_root, source_path)
    meta: dict = {}
    if pdf_path is not None and lib_store is not None:
        try:
            stored = lib_store.get_citation_meta(pdf_path)
            if isinstance(stored, dict):
                meta = dict(stored)
        except Exception:
            meta = {}

    if _has_metrics_payload(meta):
        return _ensure_summary_line(meta, allow_crossref_abstract=False)

    venue_hint, year_hint, _ = _parse_filename_meta(source_path)
    fallback_title = _source_filename(source_path) or str(source_path or "")
    if fallback_title.lower().endswith(".pdf"):
        fallback_title = fallback_title[:-4]
    fallback_title = re.sub(r"\.en\.md$", "", fallback_title, flags=re.I)
    fallback_title = re.sub(r"\.md$", "", fallback_title, flags=re.I)
    search_title = _infer_title_from_source_text(
        source_path,
        fallback_title,
        md_root_hint=str(md_root or ""),
    )
    if search_title:
        meta.setdefault("title", search_title)
    if venue_hint:
        meta.setdefault("venue", venue_hint)
    if year_hint:
        meta.setdefault("year", year_hint)

    fetched = fetch_crossref_meta(
        search_title,
        source_path=source_path,
        expected_venue=venue_hint,
        expected_year=year_hint,
        md_root_hint=str(md_root or ""),
    )
    if (
        (not isinstance(fetched, dict))
        and search_title
        and (not _is_weak_meta_value("title", search_title))
    ):
        try:
            fetched = fetch_best_crossref_meta(
                query_title=search_title,
                expected_year="",
                expected_venue="",
                doi_hint="",
                min_score=0.90,
                allow_title_only=True,
            )
        except Exception:
            fetched = None
    if isinstance(fetched, dict):
        meta = _merge_meta_prefer_richer(
            meta,
            {k: v for k, v in fetched.items() if v not in (None, "", [], {})},
        )

    enriched = _enrich_bibliometrics(meta or {})
    if isinstance(enriched, dict):
        meta = enriched
    if isinstance(meta, dict):
        meta = _ensure_summary_line(meta, allow_crossref_abstract=False)

    if pdf_path is not None and lib_store is not None and isinstance(meta, dict) and meta:
        try:
            lib_store.set_citation_meta(pdf_path, meta)
        except Exception:
            pass
    return meta if isinstance(meta, dict) else {}


def enrich_citation_detail_meta(detail: dict) -> dict:
    meta = _normalize_reference_for_popup(detail or {}) or dict(detail or {})
    raw0 = str(meta.get("cite_fmt") or meta.get("raw") or "").strip()

    def _fallback_parse_raw_reference(raw: str) -> dict:
        s = str(raw or "").strip()
        s = re.sub(r"^\s*(?:\[\s*\d+\s*\]\s*)+", "", s)
        s = s.replace("*", "")
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            return {}

        out: dict[str, str] = {}
        arxiv_m = re.search(r"\barXiv:(\d{4}\.\d{4,5})(?:v\d+)?\b", s, flags=re.I)
        if arxiv_m:
            out["doi_url"] = f"https://arxiv.org/abs/{arxiv_m.group(1)}"
            out["doi"] = f"arXiv:{arxiv_m.group(1)}"

        year_m = re.search(r"\((19|20)\d{2}\)", s)
        if year_m:
            out["year"] = year_m.group(0).strip("()")
        else:
            year2 = re.search(r"\b(19|20)\d{2}\b", s)
            if year2:
                out["year"] = year2.group(0)

        try:
            shared = _fallback_fill_reference_meta_from_raw(
                {
                    "raw": s,
                    "venue": str(meta.get("venue") or "").strip(),
                    "title": str(meta.get("title") or "").strip(),
                    "authors": str(meta.get("authors") or "").strip(),
                    "year": str(meta.get("year") or "").strip(),
                    "pages": str(meta.get("pages") or "").strip(),
                    "volume": str(meta.get("volume") or "").strip(),
                }
            )
        except Exception:
            shared = {}
        if isinstance(shared, dict):
            for key in ("authors", "title", "venue", "year", "volume", "issue", "pages"):
                value = str(shared.get(key) or "").strip()
                if value:
                    out.setdefault(key, value)

        etal_match = re.match(r"^(?P<authors>.+?\bet al\.)\s+(?P<title>.+?)\.\s+(?P<venue>.+)$", s, flags=re.I)
        if etal_match:
            out.setdefault("authors", etal_match.group("authors").strip(" ."))
            out.setdefault("title", etal_match.group("title").strip(" ."))
            out.setdefault("venue", etal_match.group("venue").strip(" ."))
            return out

        if not any(str(out.get(key) or "").strip() for key in ("authors", "title", "venue")):
            parts = [p.strip(" .") for p in re.split(r"\.\s+", s) if p.strip(" .")]
            if len(parts) >= 3:
                out.setdefault("authors", parts[0])
                out.setdefault("title", parts[1])
                out.setdefault("venue", parts[2])
            elif len(parts) == 2:
                out.setdefault("authors", parts[0])
                out.setdefault("title", parts[1])
        return out

    if raw0:
        parsed0 = _fallback_parse_raw_reference(raw0)
        for key, value in parsed0.items():
            if value and not str(meta.get(key) or "").strip():
                meta[key] = value

    title = str(meta.get("title") or "").strip()
    raw = str(meta.get("cite_fmt") or meta.get("raw") or "").strip()
    venue = str(meta.get("venue") or "").strip()
    year = str(meta.get("year") or "").strip()
    doi = str(meta.get("doi") or "").strip()
    doi_url = str(meta.get("doi_url") or "").strip()
    if doi and not doi_url:
        meta["doi_url"] = build_doi_url(doi)
    if doi:
        try:
            canonical = fetch_best_crossref_meta(
                query_title="" if _is_weak_meta_value("title", title) else title,
                doi_hint=doi,
                expected_year=year,
                expected_venue=venue,
                min_score=0.90,
                allow_title_only=False,
            )
        except Exception:
            canonical = None
        if isinstance(canonical, dict):
            meta_doi = _normalize_doi_like(str(meta.get("doi") or meta.get("doi_url") or doi))
            canonical_doi = _normalize_doi_like(str(canonical.get("doi") or canonical.get("doi_url") or ""))
            if meta_doi and canonical_doi and (meta_doi == canonical_doi):
                for key in ("title", "authors", "venue", "year", "volume", "issue", "pages", "doi", "doi_url"):
                    value = canonical.get(key)
                    if value not in (None, "", [], {}):
                        meta[key] = value
            else:
                meta = _merge_meta_prefer_richer(meta, canonical)
            if str(meta.get("doi") or "").strip() and not str(meta.get("doi_url") or "").strip():
                meta["doi_url"] = build_doi_url(str(meta.get("doi") or "").strip())
    if not doi:
        fetched_ref = None
        if raw:
            try:
                # Prefer "no enrichment" over wrong paper binding.
                fetched_ref = fetch_best_crossref_for_reference(reference_text=raw, min_score=0.74)
            except Exception:
                fetched_ref = None
        if isinstance(fetched_ref, dict):
            meta = _merge_meta_prefer_richer(
                meta,
                {k: v for k, v in fetched_ref.items() if v not in (None, "", [], {})},
            )
            doi = str(meta.get("doi") or "").strip()
            if doi and not str(meta.get("doi_url") or "").strip():
                meta["doi_url"] = build_doi_url(doi)
        if doi:
            try:
                canonical = fetch_best_crossref_meta(
                    query_title="" if _is_weak_meta_value("title", str(meta.get("title") or title).strip()) else str(meta.get("title") or title).strip(),
                    doi_hint=doi,
                    expected_year=str(meta.get("year") or year).strip(),
                    expected_venue=str(meta.get("venue") or venue).strip(),
                    min_score=0.90,
                    allow_title_only=False,
                )
            except Exception:
                canonical = None
            if isinstance(canonical, dict):
                meta = _merge_meta_prefer_richer(meta, canonical)
                if str(meta.get("doi") or "").strip() and not str(meta.get("doi_url") or "").strip():
                    meta["doi_url"] = build_doi_url(str(meta.get("doi") or "").strip())
            enriched = _enrich_bibliometrics(meta)
            if isinstance(enriched, dict):
                meta = enriched
            return _ensure_summary_line(meta, allow_crossref_abstract=True)

        search_title = title
        if not search_title:
            raw2 = re.sub(r"^\s*(?:\[\s*\d+\s*\]\s*)+", "", raw).strip()
            search_title = raw2[:220]
        fetched = fetch_crossref_meta(
            search_title,
            source_path="",
            expected_venue=venue,
            expected_year=year,
            md_root_hint="",
        )
        if (
            (not isinstance(fetched, dict))
            and search_title
            and (not _is_weak_meta_value("title", search_title))
        ):
            try:
                fetched = fetch_best_crossref_meta(
                    query_title=search_title,
                    expected_year="",
                    expected_venue="",
                    doi_hint="",
                    min_score=0.90,
                    allow_title_only=True,
                )
            except Exception:
                fetched = None
        if isinstance(fetched, dict):
            meta = _merge_meta_prefer_richer(
                meta,
                {k: v for k, v in fetched.items() if v not in (None, "", [], {})},
            )
            doi = str(meta.get("doi") or "").strip()
            if doi and not str(meta.get("doi_url") or "").strip():
                meta["doi_url"] = build_doi_url(doi)
    enriched = _enrich_bibliometrics(meta)
    if isinstance(enriched, dict):
        meta = enriched
    return _ensure_summary_line(meta, allow_crossref_abstract=True)
