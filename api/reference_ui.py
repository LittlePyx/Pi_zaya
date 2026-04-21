from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import difflib
import hashlib
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote
import re
import requests

from api.deps import load_prefs
from kb.answer_contract import _prefer_zh_locale
from kb.config import load_settings
from kb.citation_meta import fetch_best_crossref_for_reference, fetch_best_crossref_meta, fetch_crossref_work_by_doi
from kb.file_naming import citation_meta_display_pdf_name
from kb.library_store import LibraryStore
from kb.llm import DeepSeekChat
from kb.reference_query_family import (
    extract_multi_paper_topic as _shared_extract_multi_paper_topic,
    prompt_explicitly_requests_multi_paper_list as _shared_prompt_explicitly_requests_multi_paper_list,
    prompt_explicitly_requests_single_paper_pick as _shared_prompt_explicitly_requests_single_paper_pick,
    prompt_reference_focus_action as _shared_prompt_reference_focus_action,
    prompt_requests_reference_compare as _shared_prompt_requests_reference_compare,
    prompt_requests_reference_definition as _shared_prompt_requests_reference_definition,
    prompt_requires_reference_focus_match as _shared_prompt_requires_reference_focus_match,
    prompt_targets_sci_topic as _shared_prompt_targets_sci_topic,
)
from kb.source_blocks import load_source_blocks, match_source_blocks
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
_MIN_SINGLE_PAPER_DIRECT_HIT_SCORE = 4.25
_MIN_PENDING_SINGLE_PAPER_DIRECT_HIT_SCORE = 3.0
_MIN_COMPARE_DIRECT_HIT_SCORE = 5.0
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _refs_card_locale_pref() -> str:
    raw = str(os.environ.get("KB_REFS_CARD_LOCALE") or "").strip().lower()
    if raw in {"zh", "en", "auto"}:
        return raw
    try:
        prefs = load_prefs()
    except Exception:
        prefs = {}
    raw = str((prefs or {}).get("refs_card_locale") or "").strip().lower()
    if raw in {"zh", "en", "auto"}:
        return raw
    ui_locale = str((prefs or {}).get("ui_locale") or "").strip().lower()
    return ui_locale if ui_locale in {"zh", "en"} else "auto"


def _refs_card_ui_locale_pref() -> str:
    try:
        prefs = load_prefs()
    except Exception:
        prefs = {}
    raw = str((prefs or {}).get("ui_locale") or "").strip().lower()
    return raw if raw in {"zh", "en"} else ""


def _prefer_zh_ref_card_locale(*texts: str) -> bool:
    pref = _refs_card_locale_pref()
    if pref == "zh":
        return True
    if pref == "en":
        return False
    ui_pref = _refs_card_ui_locale_pref()
    if ui_pref == "zh":
        return True
    if ui_pref == "en":
        return False
    return _prefer_zh_locale(*texts)


def _prompt_strongly_prefers_english(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return cjk == 0 and latin >= 4


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


def _normalize_title_identity(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    if low.endswith(".en.md"):
        raw = raw[:-6]
    elif low.endswith(".md") or low.endswith(".pdf"):
        raw = raw[:-3] if low.endswith(".md") else raw[:-4]
    raw = re.sub(r"(19\d{2}|20\d{2})\s*-\s*", r"\1 - ", raw)
    raw = re.sub(r"[_/\\]+", " ", raw)
    raw = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip().lower()
    return raw


def _title_identity_keys(source_like: str) -> set[str]:
    raw = str(source_like or "").strip()
    if not raw:
        return set()
    out: set[str] = set()

    def _push(value: str):
        norm = _normalize_title_identity(value)
        if norm:
            out.add(norm)

    _push(raw)
    name = _source_filename(raw)
    if name:
        _push(name)
    _venue, _year, parsed_title = _parse_filename_meta(raw)
    if parsed_title:
        _push(parsed_title)
    base = name or raw
    m = re.search(r"(?:19\d{2}|20\d{2})\s*-\s*(.+)$", base)
    if m:
        _push(str(m.group(1) or "").strip())
    return {item for item in out if item}


def _same_source_title_identity(left_source: str, right_source: str) -> bool:
    left = _title_identity_keys(left_source)
    right = _title_identity_keys(right_source)
    if not left or not right:
        return False

    def _first_identity_token(value: str) -> str:
        stop = {
            "the", "and", "for", "with", "from", "into", "using", "based", "towards",
            "conference", "symposium", "workshop", "journal", "transactions", "letters",
            "ieee", "cvpr", "iccv", "eccv", "neurips", "iclr", "icml",
        }
        tokens = [tok for tok in str(value or "").split() if tok]
        for tok in tokens:
            if re.fullmatch(r"(19\d{2}|20\d{2})", tok):
                continue
            if tok in stop:
                continue
            if len(tok) < 3:
                continue
            return tok
        return tokens[0] if tokens else ""

    if left.intersection(right):
        return True
    for a in left:
        for b in right:
            if min(len(a), len(b)) < 20:
                continue
            if (a in b) or (b in a):
                return True
            a_tokens = set(a.split())
            b_tokens = set(b.split())
            if len(a_tokens) < 4 or len(b_tokens) < 4:
                continue
            overlap = len(a_tokens.intersection(b_tokens))
            smaller = min(len(a_tokens), len(b_tokens))
            if smaller <= 0:
                continue
            if (overlap / float(smaller)) >= 0.75 and _first_identity_token(a) == _first_identity_token(b):
                return True
    return False


def _hit_matches_guide_source(meta: dict, *, guide_source_path: str, guide_source_name: str) -> bool:
    if not isinstance(meta, dict):
        return False
    candidates = [
        str(meta.get("source_path") or "").strip(),
        str(meta.get("source_name") or "").strip(),
        str(meta.get("display_name") or "").strip(),
    ]
    candidates = [item for item in candidates if item]
    if not candidates:
        return False
    guide_path = str(guide_source_path or "").strip()
    guide_name = str(guide_source_name or "").strip()
    for candidate in candidates:
        if guide_path and _same_source_identity(candidate, guide_path):
            return True
        if guide_name and _same_source_title_identity(candidate, guide_name):
            return True
        if guide_path and _same_source_title_identity(candidate, guide_path):
            return True
    return False


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


def _failed_ref_fallback_ui_score(meta: dict, rank: dict) -> float | None:
    if not isinstance(meta, dict):
        return None
    rank_d = rank if isinstance(rank, dict) else {}
    try:
        bm25 = float(rank_d.get("bm25", 0.0) or 0.0)
    except Exception:
        bm25 = 0.0
    try:
        deep = float(rank_d.get("deep", 0.0) or 0.0)
    except Exception:
        deep = 0.0
    try:
        term_bonus = float(rank_d.get("term_bonus", 0.0) or 0.0)
    except Exception:
        term_bonus = 0.0
    try:
        semantic_score = float(rank_d.get("semantic_score", 0.0) or 0.0)
    except Exception:
        semantic_score = 0.0

    ui = 5.15
    ui += 1.55 * math.tanh((bm25 - 3.0) / 4.0)
    ui += 1.15 * math.tanh((deep - 8.0) / 16.0)
    ui += 0.80 * math.tanh(term_bonus / 1.6)
    if semantic_score > 0:
        ui = (0.88 * ui) + (0.12 * _clamp_ui_score(semantic_score))

    section = str(
        meta.get("ref_section")
        or ((meta.get("ref_pack") or {}).get("section") if isinstance(meta.get("ref_pack"), dict) else "")
        or meta.get("ref_best_heading_path")
        or meta.get("heading_path")
        or ""
    ).strip()
    loc_quality = str(meta.get("ref_loc_quality") or "").strip().lower()
    if not section:
        ui -= 0.45
    elif loc_quality and loc_quality != "high":
        ui -= 0.15

    try:
        explicit_doc = float(meta.get("explicit_doc_match_score") or 0.0)
    except Exception:
        explicit_doc = 0.0
    if explicit_doc > 0.0:
        ui += min(0.75, 0.12 * explicit_doc)

    return _clamp_ui_score(ui)

def _effective_ui_score(hit: dict) -> tuple[float | None, bool]:
    meta = (hit or {}).get("meta", {}) or {}
    pack_state = str(meta.get("ref_pack_state") or "").strip().lower()
    rank = meta.get("ref_rank") if isinstance(meta.get("ref_rank"), dict) else {}
    if pack_state == "ready":
        calibrated = _calibrated_ui_score(meta, rank)
        if calibrated is not None:
            return calibrated, False
    if pack_state in {"failed", "none", ""}:
        calibrated = _calibrated_ui_score(meta, rank)
        if calibrated is not None and _has_failed_ref_ui_fallback_signal(meta, rank):
            return calibrated, False
        fallback = _failed_ref_fallback_ui_score(meta, rank)
        if fallback is not None and _has_failed_ref_ui_fallback_signal(meta, rank):
            return fallback, False
    return None, pack_state == "pending"


def _has_failed_ref_ui_fallback_signal(meta: dict, rank: dict | None = None) -> bool:
    if not isinstance(meta, dict):
        return False
    rank_d = rank if isinstance(rank, dict) else {}
    if str(meta.get("ref_best_heading_path") or "").strip():
        return True
    raw_locs = meta.get("ref_locs")
    if isinstance(raw_locs, list) and raw_locs:
        return True
    for key in ("ref_show_snippets", "ref_snippets", "ref_overview_snippets"):
        raw_arr = meta.get(key)
        if isinstance(raw_arr, list) and any(str(item or "").strip() for item in raw_arr):
            return True
    try:
        explicit_doc = float(meta.get("explicit_doc_match_score") or 0.0)
    except Exception:
        explicit_doc = 0.0
    if explicit_doc >= 3.0:
        return True
    try:
        score = float((rank_d or {}).get("score") or 0.0)
    except Exception:
        score = 0.0
    return score >= 8.0


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
    allow_llm_translate: bool = True,
) -> str:
    prefer_zh = _prefer_zh_ref_card_locale(prompt, str((citation_meta or {}).get("summary_line") or ""))
    title = str((citation_meta or {}).get("title") or (meta or {}).get("title") or "").strip()
    candidates: list[str] = []

    for key in ("ref_show_snippets",):
        raw_arr = meta.get(key)
        if not isinstance(raw_arr, list):
            continue
        for item in raw_arr[:3]:
            candidates.extend(
                _expand_ref_summary_candidates(
                    str(item or ""),
                    prompt=prompt,
                    title=title,
                    prefer_zh=prefer_zh,
                    allow_llm_translate=allow_llm_translate,
                )
            )
    picked = _pick_ref_card_summary_fallback(prompt=prompt, title=title, candidates=candidates)
    if picked:
        return picked

    citation_summary_source = str((citation_meta or {}).get("summary_source") or "").strip().lower()
    if citation_summary_source == "metadata":
        return _metadata_summary_line_for_ref_card((citation_meta or meta or {}), prompt=prompt)

    citation_candidates = _expand_ref_summary_candidates(
        str((citation_meta or {}).get("summary_line") or ""),
        prompt=prompt,
        title=title,
        prefer_zh=prefer_zh,
        allow_llm_translate=allow_llm_translate,
    )
    citation_summary = _pick_ref_card_summary_fallback(prompt=prompt, title=title, candidates=citation_candidates)
    if citation_summary:
        return citation_summary

    for key in ("ref_overview_snippets",):
        raw_arr = meta.get(key)
        if not isinstance(raw_arr, list):
            continue
        for item in raw_arr[:3]:
            candidates.extend(
                _expand_ref_summary_candidates(
                    str(item or ""),
                    prompt=prompt,
                    title=title,
                    prefer_zh=prefer_zh,
                    allow_llm_translate=allow_llm_translate,
                )
            )
    picked = _pick_ref_card_summary_fallback(prompt=prompt, title=title, candidates=candidates)
    if picked:
        return picked
    return ""


def _ref_summary_identity_terms(*, source_path: str, title: str) -> set[str]:
    out: set[str] = set()
    out.update(_title_identity_keys(source_path))
    out.update(_title_identity_keys(title))
    return {item for item in out if item}


def _ref_summary_focus_score(
    *,
    prompt: str,
    source_path: str,
    title: str,
    text: str,
    anchor_target_kind: str,
    anchor_target_number: int,
) -> float:
    cand = _clean_summary_line(text)
    if not cand:
        return -1000.0
    if _looks_like_title_echo(cand, title):
        return -1000.0
    surface = _normalize_title_identity(cand)
    if not surface:
        return -1000.0

    score = 0.0
    focus_terms = _refs_prompt_focus_terms(prompt)
    identity_terms = _ref_summary_identity_terms(source_path=source_path, title=title)
    exact_focus_hits = _refs_exact_focus_match_count(prompt, cand)

    total_hits = 0
    non_source_hits = 0
    for term in focus_terms:
        if not _focus_term_matches_surface(term, surface):
            continue
        total_hits += 1
        if any(term == ident or term in ident or ident in term for ident in identity_terms):
            continue
        non_source_hits += 1
    score += 6.0 * float(non_source_hits)
    score += 1.5 * float(total_hits)
    score += 2.2 * float(exact_focus_hits)
    keyword_hits = _refs_summary_focus_keyword_hit_count(prompt, surface)
    score += 1.35 * float(keyword_hits)
    title_keyword_hits = _ref_summary_title_keyword_hit_count(title, surface)
    if title_keyword_hits >= 2:
        score += 1.1 * float(title_keyword_hits - 1)

    if _is_definition_focus_prompt(prompt):
        if re.search(r"\b(defin(?:e|es|ed|ition)|introduced?|refers?\s+to|is\s+defined\s+as)\b", cand, flags=re.I):
            score += 2.6
        if total_hits <= 0:
            score -= 2.6
        if keyword_hits >= 2:
            score += 1.6
        if re.search(r"^\s*(?:[A-Z][^:]{0,80}: )?[A-Za-z][^.!?]{0,200}\b(means|is|refers to|describes)\b", cand, flags=re.I):
            score += 1.2
        if re.match(r"^\s*(however|but|additionally|furthermore|moreover|therefore|thus)\b", cand, flags=re.I):
            score -= 2.4
    if _shared_prompt_requests_reference_compare(prompt):
        if re.search(r"\b(compare|compares|compared|comparison|versus|vs\.?)\b", cand, flags=re.I):
            score += 2.6
        if keyword_hits >= 2:
            score += 2.0
        if re.search(r"\b(difference can be summarized|while .* while |whereas)\b", cand, flags=re.I):
            score += 1.2

    kind = str(anchor_target_kind or "").strip().lower()
    num = _positive_int(anchor_target_number)
    if kind and num > 0:
        escaped_num = re.escape(str(num))
        if kind == "equation":
            if re.search(rf"(equation|eq\.?|公式)\s*[\(#\[]?\s*{escaped_num}\b", cand, flags=re.I):
                score += 6.0
        elif kind == "figure":
            if re.search(rf"(figure|fig\.?|图)\s*[\(#\[]?\s*{escaped_num}\b", cand, flags=re.I):
                score += 6.0
        elif kind == "table":
            if re.search(rf"(table|表)\s*[\(#\[]?\s*{escaped_num}\b", cand, flags=re.I):
                score += 6.0
        elif re.search(rf"\b{escaped_num}\b", cand):
            score += 2.5

    length = len(cand)
    if 40 <= length <= 260:
        score += 1.1
    elif length <= 420:
        score += 0.4
    else:
        score -= 0.8

    if re.search(r"\b(supplementary|appendix)\b", cand, flags=re.I):
        score -= 0.6
    if re.search(r"\b(fig|figure|table)\b", cand, flags=re.I) and (not kind):
        score -= 0.4
    if re.search(r"\brate\b", cand, flags=re.I) and keyword_hits <= 1 and _is_definition_focus_prompt(prompt):
        score -= 0.9
    return score


def _normalize_ref_summary_candidate(
    text: str,
    *,
    title: str,
    prefer_zh: bool,
    allow_llm_translate: bool = True,
) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"^\s*#{1,6}\s*", "", raw)
    cand = _summary_excerpt(raw, max_sentences=2, max_len=360)
    if not cand:
        cand = _first_summary_sentence(raw, max_len=220)
    if not cand:
        return ""
    cand = re.sub(r"^\s*#{1,6}\s*", "", cand).strip()
    if _looks_like_title_echo(cand, title):
        return ""
    if _looks_like_front_matter_ref_summary(cand):
        return ""
    if prefer_zh and allow_llm_translate:
        cand = _translate_summary_to_zh(cand)
    cand = _summary_excerpt(cand, max_sentences=2, max_len=360)
    if not cand:
        return ""
    cand = re.sub(r"^\s*#{1,6}\s*", "", cand).strip()
    if _looks_like_title_echo(cand, title):
        return ""
    if _looks_like_front_matter_ref_summary(cand):
        return ""
    return cand


def _split_ref_summary_heading_and_body(raw: str) -> tuple[str, str]:
    text = str(raw or "").strip()
    if not text:
        return "", ""
    lines = [str(line or "").strip() for line in text.splitlines() if str(line or "").strip()]
    if not lines:
        return "", ""
    first = lines[0]
    if re.match(r"^\s*#{1,6}\s+", first):
        heading = re.sub(r"^\s*#{1,6}\s*", "", first).strip()
        body = " ".join(lines[1:]).strip()
        return heading, body
    return "", text


def _heading_numeric_root(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    m = re.match(r"^\s*(\d+)(?:\.\d+)*", raw)
    return str(m.group(1) or "").strip() if m else ""


def _merge_prompt_aligned_heading_path(
    raw_heading: str,
    *,
    fallback_heading_path: str,
    prompt: str,
    source_path: str,
) -> str:
    heading = _sanitize_heading_path_ui(
        str(raw_heading or "").strip(),
        prompt=prompt,
        source_path=source_path,
    )
    if not heading:
        return ""
    if " / " in heading:
        return heading
    fallback = _sanitize_heading_path_ui(
        str(fallback_heading_path or "").strip(),
        prompt=prompt,
        source_path=source_path,
    )
    if not fallback:
        return heading
    parts = [str(part or "").strip() for part in str(fallback).split(" / ") if str(part or "").strip()]
    if not parts:
        return heading
    leaf = str(parts[-1] or "").strip()
    if heading.lower() == leaf.lower():
        return fallback
    if len(parts) < 2:
        return heading
    raw_root = _heading_numeric_root(heading)
    leaf_root = _heading_numeric_root(leaf)
    parent_root = _heading_numeric_root(parts[-2])
    if raw_root and leaf_root and raw_root == leaf_root and ((not parent_root) or parent_root == raw_root):
        merged = _sanitize_heading_path_ui(
            " / ".join(parts[:-1] + [heading]),
            prompt=prompt,
            source_path=source_path,
        )
        if merged:
            return merged
    return heading


def _split_ref_summary_sentences(text: str, *, max_sentences: int = 8) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    raw = re.sub(r"\s+", " ", raw)
    parts = [
        str(part or "").strip()
        for part in re.split(r"(?<=[.!?。！？;；])\s+", raw)
        if str(part or "").strip()
    ]
    return parts[: max(1, int(max_sentences or 8))]


def _is_definition_focus_prompt(prompt: str) -> bool:
    return _shared_prompt_requests_reference_definition(prompt)


@lru_cache(maxsize=512)
def _refs_prompt_focus_keywords(prompt: str) -> tuple[str, ...]:
    focus_terms = _refs_prompt_focus_terms(prompt)
    if not focus_terms:
        return ()
    stopwords = {
        "single",
        "pixel",
        "imaging",
        "paper",
        "papers",
        "library",
        "source",
        "section",
        "please",
        "point",
        "directly",
        "most",
        "other",
        "besides",
        "this",
        "which",
        "what",
        "discuss",
        "discusses",
        "define",
        "defines",
        "defined",
        "comparison",
        "compare",
        "compares",
    }
    out: list[str] = []
    seen: set[str] = set()
    for term in focus_terms:
        for token in re.findall(r"[A-Za-z0-9]{4,}", str(term or "").lower()):
            if token in stopwords or token in seen:
                continue
            seen.add(token)
            out.append(token)
    return tuple(out[:8])


def _refs_prompt_informative_focus_keywords(prompt: str) -> tuple[str, ...]:
    keywords = list(_refs_prompt_focus_keywords(prompt))
    if not keywords:
        return ()
    generic = {
        "deep",
        "learning",
        "model",
        "models",
        "method",
        "methods",
    }
    informative = [token for token in keywords if token not in generic]
    return tuple(informative or keywords)


@lru_cache(maxsize=512)
def _ref_summary_title_keywords(title: str) -> tuple[str, ...]:
    raw = _normalize_title_identity(title)
    if not raw:
        return ()
    stopwords = {
        "single",
        "pixel",
        "imaging",
        "paper",
        "papers",
        "with",
        "from",
        "for",
        "using",
        "based",
        "study",
        "analysis",
        "toward",
        "towards",
        "method",
        "methods",
        "approach",
        "framework",
    }
    out: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-z0-9]{4,}", raw):
        if token in stopwords or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return tuple(out[:10])


def _ref_summary_title_keyword_hit_count(title: str, surface_text: str) -> int:
    surface = _normalize_title_identity(surface_text)
    if not surface:
        return 0
    return sum(1 for token in _ref_summary_title_keywords(title) if token and token in surface)


def _refs_summary_focus_keyword_hit_count(prompt: str, surface_text: str) -> int:
    surface = _normalize_title_identity(surface_text)
    if not surface:
        return 0
    count = 0
    for token in _refs_prompt_focus_keywords(prompt):
        if token and token in surface:
            count += 1
    return count


def _ref_summary_surfaces_match(left: str, right: str) -> bool:
    left_norm = _normalize_title_identity(left)
    right_norm = _normalize_title_identity(right)
    if (not left_norm) or (not right_norm):
        return False
    if left_norm == right_norm:
        return True
    if left_norm in right_norm or right_norm in left_norm:
        return True
    return difflib.SequenceMatcher(None, left_norm, right_norm).ratio() >= 0.72


def _expand_ref_summary_candidates(
    raw: str,
    *,
    prompt: str,
    title: str,
    prefer_zh: bool,
    allow_llm_translate: bool = True,
    allow_focus_prefix: bool = True,
) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    heading, body = _split_ref_summary_heading_and_body(text)
    sentences = _split_ref_summary_sentences(body or text, max_sentences=8)
    candidates: list[str] = []
    seen: set[str] = set()
    definition_prompt = _is_definition_focus_prompt(prompt)

    def _push(candidate_text: str) -> None:
        cand = _normalize_ref_summary_candidate(
            candidate_text,
            title=title,
            prefer_zh=prefer_zh,
            allow_llm_translate=allow_llm_translate,
        )
        if not cand:
            return
        key = cand.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(cand)

    _push(text)
    if body:
        _push(body)
    if heading and body:
        first_body = _first_summary_sentence(body, max_len=260)
        if first_body:
            _push(f"{heading}: {first_body}")
    for sent in sentences:
        _push(sent)
    for idx in range(max(0, len(sentences) - 1)):
        window = f"{sentences[idx]} {sentences[idx + 1]}".strip()
        if len(window) <= 360:
            _push(window)
    focus_keywords = _refs_prompt_focus_keywords(prompt)
    informative_focus_keywords = _refs_prompt_informative_focus_keywords(prompt)
    for sent in sentences[:6]:
        lowered = _normalize_title_identity(sent)
        if not lowered:
            continue
        keyword_hits = sum(1 for token in focus_keywords if token in lowered)
        combined = f"{heading}. {sent}".strip(". ").strip() if heading else sent
        combined_keyword_hits = _refs_summary_focus_keyword_hit_count(prompt, combined)
        if keyword_hits <= 0 and combined_keyword_hits <= 0:
            continue
        if heading:
            _push(f"{heading}: {sent}")
        rendered_terms = _render_focus_terms_for_ref_card(prompt, max_n=2)
        if rendered_terms:
            primary_term = _display_focus_term_for_ref_card(prompt, rendered_terms[0])
            exact_term = _normalize_title_identity(rendered_terms[0])
            sentence_has_term = bool(exact_term and _focus_term_matches_surface(exact_term, sent))
            heading_has_term = bool(exact_term and _focus_term_matches_surface(exact_term, heading))
            if definition_prompt and exact_term and (not sentence_has_term) and (not heading_has_term):
                continue
            if allow_focus_prefix and primary_term and exact_term and exact_term not in _normalize_title_identity(sent):
                prefix_hits = sum(1 for token in informative_focus_keywords if token in lowered)
                if heading:
                    prefix_hits += sum(
                        1
                        for token in informative_focus_keywords
                        if token in _normalize_title_identity(heading)
                    )
                if prefix_hits > 0 and not (
                    _shared_prompt_targets_sci_topic(prompt)
                    and _surface_is_sci_related_predecessor(sent)
                ):
                    _push(f"{primary_term}: {sent}")
                    if heading:
                        _push(f"{primary_term}: {heading}. {sent}")
    return candidates


def _choose_prompt_aligned_ref_summary(
    meta: dict,
    *,
    prompt: str,
    source_path: str,
    citation_meta: dict | None = None,
    anchor_target_kind: str = "",
    anchor_target_number: int = 0,
    allow_llm_translate: bool = True,
) -> str:
    candidate = _choose_prompt_aligned_ref_summary_candidate(
        meta,
        prompt=prompt,
        source_path=source_path,
        citation_meta=citation_meta,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        allow_llm_translate=allow_llm_translate,
    )
    return str((candidate or {}).get("summary") or "").strip()


def _choose_prompt_aligned_ref_summary_candidate(
    meta: dict,
    *,
    prompt: str,
    source_path: str,
    citation_meta: dict | None = None,
    anchor_target_kind: str = "",
    anchor_target_number: int = 0,
    allow_llm_translate: bool = True,
) -> dict:
    focus_terms = _refs_prompt_focus_terms(prompt)
    if not focus_terms and (not str(anchor_target_kind or "").strip()):
        return {}

    prefer_zh = _prefer_zh_ref_card_locale(prompt, str((citation_meta or {}).get("summary_line") or ""))
    title = str((citation_meta or {}).get("title") or (meta or {}).get("title") or "").strip()

    fallback_heading_path = str((meta or {}).get("ref_best_heading_path") or (meta or {}).get("heading_path") or "").strip()
    candidates: list[dict] = []
    seen: dict[str, dict] = {}

    def _push(raw: str, *, heading_path: str = "", source_rank: int = 0) -> None:
        for cand in _expand_ref_summary_candidates(
            raw,
            prompt=prompt,
            title=title,
            prefer_zh=prefer_zh,
            allow_llm_translate=allow_llm_translate,
        ):
            key = cand.lower()
            existing = seen.get(key)
            if isinstance(existing, dict):
                if (not str(existing.get("heading_path") or "").strip()) and str(heading_path or "").strip():
                    existing["heading_path"] = str(heading_path or "").strip()
                existing["source_rank"] = min(int(existing.get("source_rank") or 0), int(source_rank or 0))
                continue
            record = {
                "summary": cand,
                "heading_path": str(heading_path or "").strip(),
                "source_rank": int(source_rank or 0),
            }
            seen[key] = record
            candidates.append(record)

    if isinstance(meta, dict):
        for source_rank, (key, limit) in enumerate((("ref_show_snippets", 4), ("ref_snippets", 4), ("ref_overview_snippets", 3))):
            raw_arr = meta.get(key)
            if not isinstance(raw_arr, list):
                continue
            for item in raw_arr[:limit]:
                raw = str(item or "")
                raw_heading, _raw_body = _split_ref_summary_heading_and_body(raw)
                derived_heading_path = _merge_prompt_aligned_heading_path(
                    raw_heading,
                    fallback_heading_path=fallback_heading_path,
                    prompt=prompt,
                    source_path=source_path,
                ) if raw_heading else ""
                _push(raw, heading_path=derived_heading_path, source_rank=source_rank)
        raw_locs = meta.get("ref_locs")
        if isinstance(raw_locs, list):
            for loc in raw_locs[:3]:
                if not isinstance(loc, dict):
                    continue
                loc_heading_path = _sanitize_heading_path_ui(
                    str(loc.get("heading_path") or loc.get("heading") or "").strip(),
                    prompt=prompt,
                    source_path=source_path,
                )
                for key in ("snippet", "text", "quote", "summary"):
                    raw = str(loc.get(key) or "")
                    if (not loc_heading_path) and raw:
                        raw_heading, _raw_body = _split_ref_summary_heading_and_body(raw)
                        loc_heading_path = _merge_prompt_aligned_heading_path(
                            raw_heading,
                            fallback_heading_path=fallback_heading_path,
                            prompt=prompt,
                            source_path=source_path,
                        ) if raw_heading else ""
                    _push(raw, heading_path=loc_heading_path, source_rank=-1)

    if not candidates:
        return {}

    ranked = sorted(
        candidates,
        key=lambda item: (
            _ref_summary_focus_score(
                prompt=prompt,
                source_path=source_path,
                title=title,
                text=str(item.get("summary") or ""),
                anchor_target_kind=anchor_target_kind,
                anchor_target_number=anchor_target_number,
            ),
            1 if str(item.get("heading_path") or "").strip() else 0,
            -int(item.get("source_rank") or 0),
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else {}
    best_summary = str((best or {}).get("summary") or "").strip()
    if not best_summary:
        return {}
    best_score = _ref_summary_focus_score(
        prompt=prompt,
        source_path=source_path,
        title=title,
        text=best_summary,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
    )
    if best_score < 2.0:
        return {}
    return {
        "summary": best_summary,
        "heading_path": str((best or {}).get("heading_path") or "").strip(),
    }


def _looks_bibliographic_source_block_text(text: str) -> bool:
    raw = " ".join(str(text or "").strip().split())
    if not raw:
        return False
    text_norm = re.sub(r"^\s*(?:\[\d+\]|\d+\.)\s*", "", raw)
    citation_like_head = bool(
        re.match(r"^(?:[A-Z][A-Za-z'`.-]+,\s*(?:[A-Z]\.?\s*){1,4})", text_norm)
    )
    if citation_like_head and re.search(r"\b(?:19|20)\d{2}\b", text_norm) and len(re.findall(r",", text_norm)) >= 4:
        return True
    if citation_like_head and re.search(
        r"\b(et al\.|optica|opt\. express|nat\.|nature|science|photonics|phys\. rev\.|ieee|front\. phys\.)\b",
        text_norm,
        flags=re.I,
    ):
        return True
    if re.search(r"\bdoi\b", text_norm, flags=re.I) and re.search(r"\b(?:19|20)\d{2}\b", text_norm):
        return True
    return False


def _looks_title_like_ref_surface(text: str, title: str) -> bool:
    surface_norm = _normalize_title_identity(str(text or "").strip())
    title_norm = _normalize_title_identity(str(title or "").strip())
    if (not surface_norm) or (not title_norm):
        return False
    if surface_norm == title_norm or surface_norm in title_norm or title_norm in surface_norm:
        return True
    surface_tokens = [
        tok
        for tok in surface_norm.split()
        if tok and len(tok) >= 4 and tok not in _PROMPT_FOCUS_STOPWORDS
    ]
    title_tokens = {
        tok
        for tok in title_norm.split()
        if tok and len(tok) >= 4 and tok not in _PROMPT_FOCUS_STOPWORDS
    }
    if (not surface_tokens) or (not title_tokens):
        return False
    overlap = sum(1 for tok in surface_tokens if tok in title_tokens)
    return bool(
        overlap >= max(4, math.ceil(len(surface_tokens) * 0.75))
        and len(surface_tokens) <= max(18, len(title_tokens) + 4)
        and (not re.search(r"[.!?。！？]", str(text or "")))
    )


def _prompt_prefers_overviewish_ref_summary(prompt: str, *, anchor_target_kind: str = "") -> bool:
    if str(anchor_target_kind or "").strip():
        return False
    text = str(prompt or "").strip().lower()
    if not text:
        return False
    if _shared_prompt_explicitly_requests_multi_paper_list(prompt):
        return True
    return bool(
        re.search(
            r"\b(mention|mentions|mentioned|discuss|discusses|discussed|which papers?|define|defines|defined|definition|what is|introduced?\s+as)\b",
            text,
            flags=re.I,
        )
    )


def _summary_candidate_heading_role_score(
    *,
    prompt: str,
    heading_path: str,
    anchor_target_kind: str,
) -> float:
    heading_norm = _normalize_title_identity(str(heading_path or "").strip())
    if not heading_norm:
        return 0.0
    if not _prompt_prefers_overviewish_ref_summary(prompt, anchor_target_kind=anchor_target_kind):
        return 0.0
    if "abstract" in heading_norm:
        return 2.2
    if "introduction" in heading_norm:
        return 1.8
    if "related work" in heading_norm or "background" in heading_norm or "overview" in heading_norm:
        return 1.0
    if "conclusion" in heading_norm or "discussion" in heading_norm:
        return 0.4
    if re.search(r"\b(method|model|pipeline|architecture|implementation|algorithm)\b", heading_norm):
        return -1.8
    if re.search(r"\b(experiment|results?|evaluation|ablation)\b", heading_norm):
        return -0.7
    return 0.0


def _summary_candidate_heading_prefix_penalty(summary: str, *, heading_path: str) -> float:
    summary_norm = _normalize_title_identity(str(summary or "").strip())
    if not summary_norm:
        return 0.0
    leaf_heading = str(str(heading_path or "").split(" / ")[-1] if heading_path else "").strip()
    leaf_heading = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", leaf_heading).strip()
    leaf_norm = _normalize_title_identity(leaf_heading)
    if (not leaf_norm) or (not summary_norm.startswith(leaf_norm)):
        return 0.0
    if re.search(
        r"\b(model|method|methods|framework|pipeline|introduction|abstract|conclusion|discussion|results?|experiments?|evaluation|overview)\b",
        leaf_norm,
    ):
        return 1.6
    return 0.8


def _summary_candidate_prefixed_title_echo_penalty(summary: str, *, title: str) -> float:
    raw_summary = str(summary or "").strip()
    if not raw_summary or not str(title or "").strip():
        return 0.0
    variants = [raw_summary]
    cur = raw_summary
    for _ in range(2):
        if ":" not in cur:
            break
        cur = str(cur.split(":", 1)[1] or "").strip()
        if cur:
            variants.append(cur)
    if any(_looks_title_like_ref_surface(candidate, title) for candidate in variants):
        return 2.8
    return 0.0


def _looks_prefixed_heading_shell_ref_summary(text: str) -> bool:
    raw = " ".join(str(text or "").strip().split())
    if (not raw) or (":" not in raw):
        return False
    prefix_raw, suffix_raw = raw.split(":", 1)
    prefix_norm = _normalize_title_identity(prefix_raw)
    suffix_norm = _normalize_title_identity(suffix_raw)
    if (not prefix_norm) or (not suffix_norm):
        return False
    prefix_tokens = [
        tok
        for tok in prefix_norm.split()
        if tok and tok not in _PROMPT_FOCUS_STOPWORDS
    ]
    if (not prefix_tokens) or len(prefix_tokens) > 6:
        return False
    return bool(
        re.match(
            r"^(abstract|introduction|background|overview|discussion|conclusion|results?|methods?)\b",
            suffix_norm,
            flags=re.I,
        )
        or re.match(r"^(摘要|引言|背景|概述|讨论|结论|结果|方法)\b", suffix_norm)
    )


def _prompt_aligned_ref_summary_candidate_copy_score(
    candidate: dict,
    *,
    prompt: str,
    source_path: str,
    title: str,
    anchor_target_kind: str,
    anchor_target_number: int,
) -> float:
    summary = str((candidate or {}).get("summary") or "").strip()
    heading_path = str((candidate or {}).get("heading_path") or "").strip()
    if not summary:
        return -1000.0
    score = _ref_summary_focus_score(
        prompt=prompt,
        source_path=source_path,
        title=title,
        text=summary,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
    )
    if _is_ref_card_summary_acceptable(
        prompt=prompt,
        title=title,
        summary_line=summary,
    ):
        score += 2.0
    elif _looks_natural_language_ref_summary(summary):
        score += 0.4
    else:
        score -= 0.9
    if _has_ref_summary_explainer_signal(summary):
        score += 0.7
    if _has_ref_summary_value_signal(summary):
        score += 0.4
    if _looks_natural_language_ref_summary(summary):
        score += 0.5
    score += _summary_candidate_heading_role_score(
        prompt=prompt,
        heading_path=heading_path,
        anchor_target_kind=anchor_target_kind,
    )
    score -= _summary_candidate_heading_prefix_penalty(
        summary,
        heading_path=heading_path,
    )
    score -= _summary_candidate_prefixed_title_echo_penalty(
        summary,
        title=title,
    )
    if _looks_prefixed_heading_shell_ref_summary(summary):
        score -= 3.2
    return score


def _rank_prompt_aligned_ref_summary_candidate(
    candidate: dict,
    *,
    prompt: str,
    source_path: str,
    title: str,
    anchor_target_kind: str,
    anchor_target_number: int,
) -> tuple[float, int, int, int, int, int, int, int, int]:
    summary = str((candidate or {}).get("summary") or "").strip()
    heading_path = str((candidate or {}).get("heading_path") or "").strip()
    raw_focus_surface = str((candidate or {}).get("raw_focus_surface") or "").strip()
    combined_surface = " ".join(part for part in (heading_path, summary) if part)
    summary_score = _prompt_aligned_ref_summary_candidate_copy_score(
        candidate,
        prompt=prompt,
        source_path=source_path,
        title=title,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
    )
    focus_hits = len(_matched_focus_terms_for_ref_card(prompt, surface_text=combined_surface))
    keyword_hits = _refs_summary_focus_keyword_hit_count(prompt, combined_surface)
    heading_depth = heading_path.count(" / ")
    block_boost = 1 if str((candidate or {}).get("source_kind") or "").strip().lower() == "source_block" else 0
    source_rank = -int((candidate or {}).get("source_rank") or 0)
    return (
        float(summary_score),
        _refs_exact_focus_match_count(prompt, summary),
        _refs_exact_focus_match_count(prompt, combined_surface),
        focus_hits,
        keyword_hits,
        _refs_exact_focus_match_count(prompt, raw_focus_surface),
        len(_matched_focus_terms_for_ref_card(prompt, surface_text=raw_focus_surface)),
        -heading_depth,
        block_boost + source_rank,
    )


def _pick_best_prompt_aligned_ref_summary_candidate(
    candidates: list[dict],
    *,
    prompt: str,
    source_path: str,
    title: str,
    anchor_target_kind: str,
    anchor_target_number: int,
) -> dict:
    ranked_rows: list[tuple[tuple[int, int, int, int, float, int, int, int, int], dict]] = []
    for raw in list(candidates or []):
        if not isinstance(raw, dict):
            continue
        summary = str(raw.get("summary") or "").strip()
        if not summary:
            continue
        candidate_score = _rank_prompt_aligned_ref_summary_candidate(
            raw,
            prompt=prompt,
            source_path=source_path,
            title=title,
            anchor_target_kind=anchor_target_kind,
            anchor_target_number=anchor_target_number,
        )
        if float(candidate_score[0]) < 2.0:
            continue
        ranked_rows.append((candidate_score, dict(raw)))
    if not ranked_rows:
        return {}
    ranked_rows.sort(key=lambda item: item[0], reverse=True)
    return ranked_rows[0][1]


def _choose_prompt_aligned_ref_summary_candidate_from_source_blocks(
    *,
    prompt: str,
    source_path: str,
    title: str,
    anchor_target_kind: str = "",
    anchor_target_number: int = 0,
    allow_llm_translate: bool = True,
) -> dict:
    focus_terms = _refs_prompt_focus_terms(prompt)
    if not focus_terms and (not str(anchor_target_kind or "").strip()):
        return {}
    md_path = _resolve_source_md_path(source_path)
    if md_path is None:
        return {}
    try:
        blocks = load_source_blocks(md_path)
    except Exception:
        return {}
    if not blocks:
        return {}

    prefer_zh = _prefer_zh_ref_card_locale(prompt, title)
    candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        block_text = str(block.get("text") or "").strip()
        if (
            (not block_text)
            or _looks_bibliographic_source_block_text(block_text)
            or _looks_title_like_ref_surface(block_text, title)
        ):
            continue
        block_kind = str(block.get("kind") or "").strip().lower()
        if block_kind in {"figure", "table", "equation"} and (not str(anchor_target_kind or "").strip()):
            continue
        heading_path = _normalize_refs_reader_heading_path(
            prompt=prompt,
            source_path=source_path,
            heading_path=str(block.get("heading_path") or "").strip(),
        )
        focus_surface = " ".join(part for part in (heading_path, block_text) if part)
        if (not str(anchor_target_kind or "").strip()):
            exact_hits = _refs_exact_focus_match_count(prompt, focus_surface)
            keyword_hits = _refs_summary_focus_keyword_hit_count(prompt, focus_surface)
            surface_matches = len(_matched_focus_terms_for_ref_card(prompt, surface_text=focus_surface))
            if exact_hits <= 0 and keyword_hits <= 0 and surface_matches <= 0:
                continue
        raw_candidates: list[str] = []
        leaf_heading = str(heading_path.split(" / ")[-1] if heading_path else "").strip()
        if leaf_heading:
            raw_candidates.append(f"## {leaf_heading}\n{block_text}")
        raw_candidates.append(block_text)
        for raw_candidate in raw_candidates:
            for summary in _expand_ref_summary_candidates(
                raw_candidate,
                prompt=prompt,
                title=title,
                prefer_zh=prefer_zh,
                allow_llm_translate=allow_llm_translate,
                allow_focus_prefix=False,
            ):
                key = (str(summary or "").strip().lower(), heading_path.lower())
                if (not key[0]) or key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "summary": str(summary or "").strip(),
                        "heading_path": heading_path,
                        "raw_focus_surface": focus_surface,
                        "source_kind": "source_block",
                        "source_rank": 0,
                        "block_index": idx,
                    }
                )
    return _pick_best_prompt_aligned_ref_summary_candidate(
        candidates,
        prompt=prompt,
        source_path=source_path,
        title=title,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
    )


_GENERIC_REF_WHY_PATTERNS = (
    "给出了与",
    "主题一致",
    "直接参考依据",
    "关键证据来源",
    "定义、方法或结果信息",
)


def _looks_formula_heavy_ref_text(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if re.search(r"(\\[A-Za-z]{2,}|\\tag\{|\$\$|[_^{}]{2,}|=\s*\\|\bint_[a-z]|\bsigma\()", s):
        return True
    mathish = len(re.findall(r"[=+\-*/^$\\{}()[\]_]", s))
    alpha = len(re.findall(r"[A-Za-z\u4e00-\u9fff]", s))
    return mathish >= 12 and mathish > alpha


def _looks_surface_like_ref_summary(text: str) -> bool:
    s = _clean_summary_line(text)
    if not s:
        return False
    if _looks_prefixed_heading_shell_ref_summary(s):
        return True
    if re.match(r"^\s*#{1,6}\s+", s):
        return True
    if re.match(r"^\s*(fig(?:ure)?|table|eq(?:uation)?|appendix)\b", s, flags=re.I):
        return True
    if re.match(r"^\s*[A-Z][A-Za-z& .-]{3,48}\s+Fig\.?\b", s):
        return True
    if re.match(r"^\s*(optics express|science advances|nature communications|cvpr|ieee)\b", s, flags=re.I):
        return True
    if re.search(r"\bOCIS\s+codes?\b", s, flags=re.I):
        return True
    if re.search(r"\b(optical society of america|all rights reserved|copyright)\b", s, flags=re.I):
        return True
    if re.search(r"\$\^\{\d+(?:,\d+)*\}\$", s):
        return True
    if len(re.findall(r"\b[A-Z][A-Z-]{2,}\b", s)) >= 4 and len(re.findall(r"\b(and|with|for|versus|vs\.?)\b", s, flags=re.I)) <= 1:
        return True
    return len(s) > 260


def _looks_generic_ref_why_line(text: str) -> bool:
    s = " ".join(str(text or "").strip().split())
    if not s:
        return True
    lower = s.lower()
    if "..." in s and re.search(r"\b(which|what|where|how|why)\b", lower):
        return True
    if re.search(r"\b(which paper|in my library|point me to|source section)\b", lower):
        return True
    if re.search(r"(瀹氫箟銆佹柟娉曟垨缁撴灉淇℃伅)", s):
        return True
    return any(token in s for token in _GENERIC_REF_WHY_PATTERNS)


def _render_focus_terms_for_ref_card(prompt: str, *, max_n: int = 2) -> list[str]:
    terms = [str(term or "").strip() for term in _refs_prompt_focus_terms(prompt) if str(term or "").strip()]
    out: list[str] = []
    for term in terms:
        if any(
            (term == prev or term in prev or prev in term)
            and (not re.search(r"\b(?:and|vs\.?|versus)\b", prev, flags=re.IGNORECASE))
            for prev in out
        ):
            continue
        out.append(term)
        if len(out) >= max(1, int(max_n or 2)):
            break
    return out


def _looks_focus_prefixed_ref_summary(prompt: str, summary_line: str) -> bool:
    raw = " ".join(str(summary_line or "").strip().split())
    if (not raw) or (":" not in raw):
        return False
    prefix_norm = _normalize_title_identity(str(raw.split(":", 1)[0] or "").strip())
    if not prefix_norm:
        return False
    for term in _render_focus_terms_for_ref_card(prompt, max_n=3):
        term_norm = _normalize_title_identity(term)
        if term_norm and (prefix_norm == term_norm or prefix_norm in term_norm or term_norm in prefix_norm):
            return True
    return False


def _display_focus_term_for_ref_card(prompt: str, term: str) -> str:
    raw_prompt = str(prompt or "").strip()
    raw_term = str(term or "").strip()
    if (not raw_prompt) or (not raw_term):
        return raw_term
    norm = _clean_refs_focus_phrase(raw_term)
    if norm:
        pattern = re.escape(norm).replace(r"\ ", r"[\s\-]+")
        m = re.search(pattern, raw_prompt, flags=re.I)
        if m:
            return " ".join(str(m.group(0) or "").split())
    if _prompt_strongly_prefers_english(prompt):
        return raw_term.title()
    return raw_term


def _matched_focus_terms_for_ref_card(prompt: str, *, surface_text: str) -> list[str]:
    surface = _normalize_title_identity(surface_text)
    if not surface:
        return []
    out: list[str] = []
    for term in _render_focus_terms_for_ref_card(prompt, max_n=3):
        if _focus_term_matches_surface(term, surface):
            out.append(term)
    return out[:2]


def _why_line_explicitly_names_focus_term(prompt: str, why_line: str) -> bool:
    surface = _normalize_title_identity(why_line)
    if not surface:
        return False
    for term in _render_focus_terms_for_ref_card(prompt, max_n=3):
        norm = _normalize_title_identity(term)
        if norm and norm in surface:
            return True
    return False


def _build_prompt_aligned_ref_why_line(
    *,
    prompt: str,
    display_name: str,
    heading_path: str,
    summary_line: str,
    why_line: str,
) -> str:
    loc = " / ".join(part for part in str(heading_path or "").split(" / ") if part).strip()
    surface = " ".join(
        part for part in (
            str(display_name or "").strip(),
            str(heading_path or "").strip(),
            str(summary_line or "").strip(),
            str(why_line or "").strip(),
        ) if part
    )
    matched_terms = _matched_focus_terms_for_ref_card(prompt, surface_text=surface)
    if loc and matched_terms:
        return f"这条命中在“{loc}”直接覆盖了“{'、'.join(matched_terms)}”这个问题焦点，适合作为定位入口。"
    if matched_terms:
        return f"这条命中直接讨论了“{'、'.join(matched_terms)}”，和当前问题的核心概念是正对齐的。"
    if loc:
        return f"这条命中落在“{loc}”，能直接提供和当前问题相关的定义、方法或结果证据。"
    return ""


def _build_prompt_aligned_ref_why_line_v2(
    *,
    prompt: str,
    display_name: str,
    heading_path: str,
    summary_line: str,
    why_line: str,
) -> str:
    prefer_zh = _prefer_zh_ref_card_locale(prompt, display_name, heading_path, summary_line, why_line)
    loc = " / ".join(part for part in str(heading_path or "").split(" / ") if part).strip()
    surface = " ".join(
        part for part in (
            str(display_name or "").strip(),
            str(heading_path or "").strip(),
            str(summary_line or "").strip(),
            str(why_line or "").strip(),
        ) if part
    )
    matched_terms = _matched_focus_terms_for_ref_card(prompt, surface_text=surface)
    if loc and matched_terms:
        if prefer_zh:
            return f"这条命中在“{loc}”直接覆盖了“{' / '.join(matched_terms)}”这个问题焦点，适合作为定位入口。"
        display_terms = [_display_focus_term_for_ref_card(prompt, item) for item in matched_terms]
        return f"This hit lands in '{loc}' and directly covers the focus concept '{' / '.join(display_terms)}', so it is a good entry point."
    if matched_terms:
        if prefer_zh:
            return f"这条命中直接讨论了“{' / '.join(matched_terms)}”，和当前问题的核心概念是正对齐的。"
        display_terms = [_display_focus_term_for_ref_card(prompt, item) for item in matched_terms]
        return f"This hit directly discusses '{' / '.join(display_terms)}', which aligns with the core concept in the user's question."
    if loc:
        if prefer_zh:
            return f"这条命中落在“{loc}”，能直接提供和当前问题相关的定义、方法或结果证据。"
        return f"This hit falls under '{loc}' and can directly provide the definitions, methods, or results relevant to the current question."
    return ""


def _build_prompt_aligned_ref_why_line_v3(
    *,
    prompt: str,
    display_name: str,
    heading_path: str,
    summary_line: str,
    why_line: str,
) -> str:
    prefer_zh = _prefer_zh_ref_card_locale(prompt, display_name, heading_path, summary_line, why_line)
    loc = " / ".join(part for part in str(heading_path or "").split(" / ") if part).strip()
    surface = " ".join(
        part
        for part in (
            str(display_name or "").strip(),
            str(heading_path or "").strip(),
            str(summary_line or "").strip(),
            str(why_line or "").strip(),
        )
        if part
    )
    matched_terms = _matched_focus_terms_for_ref_card(prompt, surface_text=surface)
    if len(matched_terms) >= 2 and _shared_prompt_requests_reference_compare(prompt):
        compare_terms: list[str] = []
        for term in matched_terms:
            parts = re.split(r"\b(?:and|vs\.?|versus)\b", term, flags=re.IGNORECASE)
            for part in parts:
                cleaned = _clean_refs_focus_phrase(part)
                if not cleaned:
                    continue
                norm = _normalize_title_identity(cleaned)
                if norm and norm not in compare_terms:
                    compare_terms.append(norm)
        pair = " / ".join(
            _display_focus_term_for_ref_card(prompt, item)
            for item in (compare_terms or matched_terms)[:2]
        )
        if prefer_zh:
            return f"这条命中在“{loc or heading_path or '该小节'}”直接比较了“{pair}”，和当前的对比问题是正对齐的。"
        return f"This hit directly compares '{pair}' in '{loc or heading_path or 'this section'}', so it is a strong match for the comparison request."
    if matched_terms and _shared_prompt_requests_reference_definition(prompt):
        term = _display_focus_term_for_ref_card(prompt, matched_terms[0])
        if prefer_zh:
            return f"这条命中在“{loc or heading_path or '该小节'}”直接定义或解释了“{term}”，适合作为定位切口。"
        return f"This hit directly defines or explains '{term}' in '{loc or heading_path or 'this section'}', so it is a good entry point."
    if loc and matched_terms:
        if prefer_zh:
            return f"这条命中在“{loc}”直接覆盖了“{' / '.join(matched_terms)}”这个问题焦点，适合作为定位入口。"
        display_terms = [_display_focus_term_for_ref_card(prompt, item) for item in matched_terms]
        return f"This hit lands in '{loc}' and directly covers the focus concept '{' / '.join(display_terms)}', so it is a good entry point."
    if matched_terms:
        if prefer_zh:
            return f"这条命中直接讨论了“{' / '.join(matched_terms)}”，和当前问题的核心概念是正对齐的。"
        display_terms = [_display_focus_term_for_ref_card(prompt, item) for item in matched_terms]
        return f"This hit directly discusses '{' / '.join(display_terms)}', which aligns with the core concept in the user's question."
    if loc:
        if prefer_zh:
            return f"这条命中落在“{loc}”，能直接提供和当前问题相关的定义、方法或结果证据。"
        return f"This hit falls under '{loc}' and can directly provide the definitions, methods, or results relevant to the current question."
    return ""


def _metadata_summary_line_for_ref_card(meta: dict, *, prompt: str) -> str:
    prefer_zh = _prefer_zh_ref_card_locale(prompt, str((meta or {}).get("title") or ""))
    title = _clean_summary_line(str((meta or {}).get("title") or ""))
    venue = _clean_summary_line(str((meta or {}).get("venue") or ""))
    year = str((meta or {}).get("year") or "").strip()
    authors = _clean_summary_line(str((meta or {}).get("authors") or ""))
    author_head = ""
    if authors:
        author_head = re.split(r"[,;&]| and ", authors, maxsplit=1, flags=re.I)[0].strip()
    loc = ""
    if venue and year:
        loc = f"{venue} ({year})"
    elif venue:
        loc = venue
    elif year:
        loc = year
    if prefer_zh:
        if author_head and loc:
            return f"这篇文献当前缺少可用摘要，以下仅根据元数据给出导读：该工作由 {author_head} 发表在 {loc}。"
        if loc:
            return f"这篇文献当前缺少可用摘要，以下仅根据元数据给出导读：该工作发表在 {loc}。"
        if title:
            return "这篇文献当前缺少可用摘要，以下仅根据题名和基础元数据给出导读。"
        return "当前仅检索到有限文献信息，尚未获得可用摘要。"
    if author_head and loc:
        return f"No abstract is available for this paper yet, so this card falls back to metadata only: the work by {author_head} was published in {loc}."
    if loc:
        return f"No abstract is available for this paper yet, so this card falls back to metadata only: the work was published in {loc}."
    if title:
        return "No abstract is available for this paper yet, so this card falls back to the title and basic bibliographic metadata."
    return "Only limited bibliographic metadata is currently available, and no usable abstract was found."


def _build_ref_summary_basis_meta(
    *,
    prompt: str,
    summary_kind: str,
    summary_generation: str,
    summary_line: str = "",
) -> dict[str, str]:
    prefer_zh = _prefer_zh_ref_card_locale(prompt, summary_line)
    kind = str(summary_kind or "").strip().lower()
    generation = str(summary_generation or "").strip().lower()
    if kind == "abstract":
        if generation == "llm_abstract":
            return {
                "summary_generation": "llm_abstract",
                "summary_basis": "基于 abstract 的 LLM 提炼" if prefer_zh else "LLM-distilled from abstract",
            }
        return {
            "summary_generation": generation or "translated_abstract",
            "summary_basis": "基于 abstract 原文整理" if prefer_zh else "Condensed from abstract text",
        }
    if kind == "metadata":
        return {
            "summary_generation": "metadata_only",
            "summary_basis": "仅基于书目信息，非摘要" if prefer_zh else "Metadata only, not an abstract",
        }
    if generation == "llm_grounded":
        return {
            "summary_generation": "llm_grounded",
            "summary_basis": "基于命中章节证据的 LLM 提炼" if prefer_zh else "LLM-distilled from matched section evidence",
        }
    if generation == "deterministic_grounded":
        return {
            "summary_generation": "deterministic_grounded",
            "summary_basis": "基于命中章节证据整理" if prefer_zh else "Condensed from matched section evidence",
        }
    return {
        "summary_generation": generation or "section_grounded",
        "summary_basis": "基于命中章节/定位证据" if prefer_zh else "Based on matched section evidence",
    }


def _build_ref_why_basis_meta(
    *,
    prompt: str,
    why_generation: str,
    why_line: str = "",
) -> dict[str, str]:
    prefer_zh = _prefer_zh_ref_card_locale(prompt, why_line)
    generation = str(why_generation or "").strip().lower()
    if generation == "llm_grounded":
        return {
            "why_generation": "llm_grounded",
            "why_basis": "基于命中章节证据的 LLM 相关性说明" if prefer_zh else "LLM-grounded relevance from matched section evidence",
        }
    if generation == "deterministic_grounded":
        return {
            "why_generation": "deterministic_grounded",
            "why_basis": "基于命中章节和关键词对齐的规则化说明" if prefer_zh else "Rule-based relevance from matched section and focus-term alignment",
        }
    if generation == "navigation":
        return {
            "why_generation": "navigation",
            "why_basis": "基于定位章节与命中证据整理" if prefer_zh else "Based on navigation section and matched evidence",
        }
    return {
        "why_generation": generation or "fallback",
        "why_basis": "基于当前命中证据的保守说明" if prefer_zh else "Conservative relevance note from the available evidence",
    }


def _infer_ref_summary_kind(
    *,
    summary_line: str,
    citation_meta: dict | None,
    used_prompt_aligned_summary: bool,
    used_nav_summary: bool,
) -> str:
    if used_prompt_aligned_summary or used_nav_summary:
        return "guide"
    summary_clean = _clean_summary_line(summary_line)
    citation_summary = _clean_summary_line(str((citation_meta or {}).get("summary_line") or ""))
    citation_source = str((citation_meta or {}).get("summary_source") or "").strip().lower()
    if citation_summary and summary_clean and summary_clean == citation_summary:
        if citation_source == "abstract":
            return "abstract"
        if citation_source == "metadata":
            return "metadata"
    return "guide"


def _build_ref_summary_surface_meta(*, prompt: str, summary_kind: str, summary_line: str = "") -> dict[str, str]:
    prefer_zh = _prefer_zh_ref_card_locale(prompt, summary_line)
    kind = str(summary_kind or "").strip().lower()
    if kind == "abstract":
        return {
            "summary_kind": "abstract",
            "summary_label": "摘要" if prefer_zh else "Abstract",
            "summary_title": "这篇文献讲什么 / 提供什么" if prefer_zh else "What This Paper Covers",
        }
    if kind == "metadata":
        return {
            "summary_kind": "metadata",
            "summary_label": "信息卡" if prefer_zh else "Meta",
            "summary_title": "当前可用的文献信息" if prefer_zh else "Available Bibliographic Info",
        }
    return {
        "summary_kind": "guide",
        "summary_label": "导读" if prefer_zh else "Guide",
        "summary_title": "命中章节讲什么 / 提供什么" if prefer_zh else "What This Matched Section Covers",
    }


def _finalize_abstract_summary_line(*, title: str, abstract_text: str) -> tuple[str, str]:
    abstract_line = _summary_excerpt(abstract_text, max_sentences=5, max_len=900)
    if not abstract_line:
        return "", ""
    llm_summary = _llm_summarize_abstract_zh(title=title, abstract_text=abstract_line)
    if llm_summary:
        return llm_summary, "llm_abstract"
    translated = _translate_summary_to_zh(abstract_line)
    if translated:
        return translated, "translated_abstract"
    return abstract_line, "translated_abstract"


def _has_ref_summary_explainer_signal(text: str) -> bool:
    s = _clean_summary_line(text)
    if not s:
        return False
    return bool(
        re.search(
            r"\b(compare|comparative|analy[sz]e|analysis|evaluat|study|explore|review|survey|introduce|present|propose|design|develop|use)\b",
            s,
            flags=re.I,
        )
        or re.search(r"(比较|对比|分析|评估|研究|探讨|综述|提出|设计|构建|采用|介绍)", s)
    )


def _has_ref_summary_value_signal(text: str) -> bool:
    s = _clean_summary_line(text)
    if not s:
        return False
    return bool(
        re.search(
            r"\b(result|show|demonstrat|improv|outperform|achiev|difference|trade-?off|advantage|limitation|quality|efficiency|robustness|fidelity|performance)\b",
            s,
            flags=re.I,
        )
        or re.search(r"(结果|显示|提升|优于|差异|权衡|优势|局限|质量|效率|鲁棒|保真|性能)", s)
    )


def _looks_natural_language_ref_summary(text: str) -> bool:
    s = _clean_summary_line(text)
    if not s:
        return False
    if _looks_formula_heavy_ref_text(s):
        return False
    if re.search(r"\b(doc|sid|cite)-\d+\b", s, flags=re.I):
        return False
    wordish = len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", s))
    punctuation = len(re.findall(r"[，。；：,.]", s))
    return wordish >= 4 and punctuation >= 1


def _is_ref_card_summary_acceptable(
    *,
    prompt: str,
    title: str,
    summary_line: str,
) -> bool:
    s = _clean_summary_line(summary_line)
    if not s:
        return False
    if _looks_prefixed_heading_shell_ref_summary(s):
        return False
    if _looks_like_title_echo(s, title):
        return False
    if _looks_formula_heavy_ref_text(s):
        return False
    if _looks_surface_like_ref_summary(s):
        return False
    if len(s) < 32:
        return False
    if _is_summary_quality_ok(s):
        return True
    matched_terms = _matched_focus_terms_for_ref_card(prompt, surface_text=s)
    if _prompt_requires_explicit_focus_match(prompt) and _render_focus_terms_for_ref_card(prompt) and (not matched_terms):
        return False
    if matched_terms and (
        (_has_ref_summary_explainer_signal(s) and (_has_ref_summary_value_signal(s) or len(s) >= 48))
        or (len(s) >= 40 and _looks_natural_language_ref_summary(s))
    ):
        return True
    score = _ref_summary_focus_score(
        prompt=prompt,
        source_path="",
        title=title,
        text=s,
        anchor_target_kind="",
        anchor_target_number=0,
    )
    return score >= 1.8


def _looks_like_front_matter_ref_summary(text: str) -> bool:
    cand = _clean_summary_line(text)
    if not cand:
        return False
    if re.search(r"\bOCIS\s+codes?\b", cand, flags=re.I):
        return True
    if re.search(r"\b(optical society of america|all rights reserved|copyright)\b", cand, flags=re.I):
        return True
    if "©" in cand:
        return True
    if len(re.findall(r"\$\^\{\d+(?:,\d+)*\}\$", cand)) >= 1:
        return True
    if len(re.findall(r"\*\*[^*]{2,}\*\*", cand)) >= 2:
        return True
    if len(re.findall(r"\b[A-Z][A-Z][A-Z' -]{3,}\b", cand)) >= 2:
        return True
    return False


def _ref_card_summary_candidate_score(*, prompt: str, title: str, text: str) -> float:
    cand = _clean_summary_line(text)
    if not cand:
        return -1000.0
    if _looks_like_front_matter_ref_summary(cand):
        return -1000.0
    score = _ref_summary_focus_score(
        prompt=prompt,
        source_path="",
        title=title,
        text=cand,
        anchor_target_kind="",
        anchor_target_number=0,
    )
    if _is_summary_quality_ok(cand):
        score += 2.5
    if _has_ref_summary_explainer_signal(cand):
        score += 1.1
    if _has_ref_summary_value_signal(cand):
        score += 0.9
    if _looks_natural_language_ref_summary(cand):
        score += 0.7
    if _looks_surface_like_ref_summary(cand):
        score -= 2.5
    if _looks_prefixed_heading_shell_ref_summary(cand):
        score -= 3.0
    if re.search(r"\bOCIS\s+codes?\b", cand, flags=re.I):
        score -= 3.0
    if re.search(r"\b(optical society of america|all rights reserved|copyright)\b", cand, flags=re.I):
        score -= 2.4
    if len(re.findall(r"\$\^\{\d+(?:,\d+)*\}\$", cand)) >= 1:
        score -= 2.2
    if re.search(r"^\s*(fig(?:ure)?|table|eq(?:uation)?)\s*[\d(#\[]", cand, flags=re.I):
        score -= 1.2
    if re.search(r"^\s*[\(\[]?\d+[\)\].:\- ]", cand):
        score -= 0.9
    if re.search(r"\b(this paper|the paper|this work|the work|method|framework|pipeline)\b", cand, flags=re.I):
        score += 0.8
    if _prompt_requires_explicit_focus_match(prompt):
        matched_terms = _matched_focus_terms_for_ref_card(prompt, surface_text=cand)
        if matched_terms:
            score += 1.4 * float(len(matched_terms))
        else:
            score -= 2.6
    if re.match(r"^[a-z]", cand):
        score -= 0.9
    return score


def _pick_ref_card_summary_fallback(*, prompt: str, title: str, candidates: list[str]) -> str:
    ranked: list[tuple[float, str]] = []
    for raw in candidates or []:
        cand = _summary_excerpt(str(raw or ""), max_sentences=2, max_len=220)
        if not cand:
            continue
        ranked.append((_ref_card_summary_candidate_score(prompt=prompt, title=title, text=cand), cand))
    if not ranked:
        return ""
    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, best = ranked[0]
    if best_score < 1.6:
        return ""
    return best


def _summary_line_needs_polish(
    *,
    prompt: str,
    title: str,
    summary_line: str,
) -> bool:
    s = _clean_summary_line(summary_line)
    if not s:
        return True
    return not _is_ref_card_summary_acceptable(
        prompt=prompt,
        title=title,
        summary_line=s,
    )


def _why_line_needs_polish(
    *,
    prompt: str,
    display_name: str,
    heading_path: str,
    summary_line: str,
    why_line: str,
) -> bool:
    s = " ".join(str(why_line or "").strip().split())
    if not s:
        return True
    if _looks_generic_ref_why_line(s):
        return True
    if _is_definition_focus_prompt(prompt) and (not _why_line_explicitly_names_focus_term(prompt, s)):
        return True
    surface = " ".join(
        part for part in (
            str(display_name or "").strip(),
            str(heading_path or "").strip(),
            str(summary_line or "").strip(),
            s,
        ) if part
    )
    matched_terms = _matched_focus_terms_for_ref_card(prompt, surface_text=surface)
    return bool(_render_focus_terms_for_ref_card(prompt) and (not matched_terms))


def _collect_ref_card_polish_candidates(hit: dict, *, ui_meta: dict, max_items: int = 4) -> list[str]:
    meta = (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}
    out: list[str] = []
    seen: set[str] = set()

    def _push(raw: str) -> None:
        cand = _summary_excerpt(str(raw or ""), max_sentences=2, max_len=280)
        if not cand:
            return
        key = cand.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(cand)

    if isinstance(meta, dict):
        for key, limit in (("ref_show_snippets", 3), ("ref_snippets", 2), ("ref_overview_snippets", 2)):
            raw_arr = meta.get(key)
            if not isinstance(raw_arr, list):
                continue
            for item in raw_arr[:limit]:
                _push(str(item or ""))
        raw_locs = meta.get("ref_locs")
        if isinstance(raw_locs, list):
            for loc in raw_locs[:2]:
                if not isinstance(loc, dict):
                    continue
                for key in ("snippet", "text", "quote", "summary"):
                    _push(str(loc.get(key) or ""))
    for raw in (
        str((ui_meta or {}).get("summary_line") or ""),
        str((ui_meta or {}).get("why_line") or ""),
        str((hit or {}).get("text") or ""),
    ):
        _push(raw)
    return out[: max(1, int(max_items or 4))]


def _refs_card_polish_llm_enabled() -> bool:
    raw_flag = str(os.environ.get("KB_REFS_CARD_POLISH_USE_LLM", "1") or "").strip().lower()
    if raw_flag in {"0", "false", "off", "no"}:
        return False
    try:
        settings = load_settings()
    except Exception:
        return False
    return bool(getattr(settings, "api_key", None))


@lru_cache(maxsize=512)
def _llm_polish_ref_card_copy(
    *,
    prompt: str,
    display_name: str,
    heading_path: str,
    summary_seed: str,
    why_seed: str,
    candidate_payload: str,
) -> tuple[str, str]:
    if (not prompt) or (not candidate_payload):
        return "", ""
    if not _refs_card_polish_llm_enabled():
        return "", ""
    try:
        settings = load_settings()
    except Exception:
        return "", ""
    if not getattr(settings, "api_key", None):
        return "", ""
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
                            "你在润色学术阅读助手里的参考定位卡片文案。"
                            "只输出 JSON，格式为 {\"summary_line\":\"...\",\"why_line\":\"...\"}。"
                            "summary_line: 用 1 句中文概括这篇文献/这一小节在做什么或提供什么，必须基于给定证据，不要照抄公式。"
                            "why_line: 用 1 句中文说明它为什么和用户当前问题直接相关，优先点出命中的概念或章节。"
                            "不要编造论文没有写的内容。不要输出 markdown、序号、DOC/SID/CITE、'当前问题' 这类空泛措辞。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"用户问题：{str(prompt or '').strip()}\n"
                            f"文献标题：{str(display_name or '').strip()}\n"
                            f"章节：{str(heading_path or '').strip()}\n"
                            f"当前摘要候选：{str(summary_seed or '').strip()}\n"
                            f"当前相关性说明候选：{str(why_seed or '').strip()}\n"
                            f"可用证据片段：\n{candidate_payload}\n"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=240,
            )
            or ""
        ).strip()
    except Exception:
        return "", ""

    summary_line = ""
    why_line = ""
    try:
        parsed = json.loads(out)
        if isinstance(parsed, dict):
            summary_line = str(parsed.get("summary_line") or "").strip()
            why_line = str(parsed.get("why_line") or "").strip()
    except Exception:
        m_summary = re.search(r'"summary_line"\s*:\s*"([^"]*)"', out)
        m_why = re.search(r'"why_line"\s*:\s*"([^"]*)"', out)
        summary_line = str(m_summary.group(1) if m_summary else "").strip()
        why_line = str(m_why.group(1) if m_why else "").strip()
    return summary_line, why_line


@lru_cache(maxsize=512)
def _llm_polish_ref_card_copy_v2(
    *,
    prompt: str,
    display_name: str,
    heading_path: str,
    summary_seed: str,
    why_seed: str,
    candidate_payload: str,
) -> tuple[str, str]:
    if (not prompt) or (not candidate_payload):
        return "", ""
    prefer_zh = _prefer_zh_ref_card_locale(prompt, display_name, heading_path, summary_seed, why_seed)
    if not _refs_card_polish_llm_enabled():
        return "", ""
    try:
        settings = load_settings()
    except Exception:
        return "", ""
    if not getattr(settings, "api_key", None):
        return "", ""
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
                            "You are polishing copy for a research reference card in a reading assistant. "
                            "Return JSON only with this schema: "
                            "{\"summary_line\":\"...\",\"why_line\":\"...\"}. "
                            "summary_line must be one concise sentence saying what this paper or section does, compares, or provides. "
                            "Use only the supplied evidence snippets. Do not copy formulas, markdown headings, or venue boilerplate. "
                            "why_line must be one concise sentence explaining why this hit is directly relevant to the user's request. "
                            "Prefer naming the matched concept or section instead of repeating the whole prompt. "
                            f"{'Write both fields in concise Chinese. ' if prefer_zh else 'Write both fields in concise English. '}"
                            "Do not invent facts. Do not output markdown, bullets, DOC/SID/CITE markers, or placeholders."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"User prompt: {str(prompt or '').strip()}\n"
                            f"Paper title: {str(display_name or '').strip()}\n"
                            f"Section heading: {str(heading_path or '').strip()}\n"
                            f"Current summary candidate: {str(summary_seed or '').strip()}\n"
                            f"Current relevance candidate: {str(why_seed or '').strip()}\n"
                            f"Evidence snippets:\n{candidate_payload}\n"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=240,
            )
            or ""
        ).strip()
    except Exception:
        return "", ""

    summary_line = ""
    why_line = ""
    try:
        parsed = json.loads(out)
        if isinstance(parsed, dict):
            summary_line = str(parsed.get("summary_line") or "").strip()
            why_line = str(parsed.get("why_line") or "").strip()
    except Exception:
        m_summary = re.search(r'"summary_line"\s*:\s*"([^"]*)"', out)
        m_why = re.search(r'"why_line"\s*:\s*"([^"]*)"', out)
        summary_line = str(m_summary.group(1) if m_summary else "").strip()
        why_line = str(m_why.group(1) if m_why else "").strip()
    return summary_line, why_line


@lru_cache(maxsize=512)
def _llm_ground_ref_why_line(
    *,
    prompt: str,
    display_name: str,
    heading_path: str,
    summary_line: str,
    why_seed: str,
    candidate_payload: str,
) -> str:
    if (not prompt) or (not candidate_payload):
        return ""
    prefer_zh = _prefer_zh_ref_card_locale(prompt, display_name, heading_path, summary_line, why_seed)
    if not _refs_card_polish_llm_enabled():
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
                            "You are writing the 'why relevant' line for a research reference card. "
                            "Use only the supplied evidence snippets. "
                            "Return JSON only with {\"why_line\":\"...\"}. "
                            "The sentence must explain why this hit is directly relevant to the user's request, "
                            "prefer naming the matched concept, comparison, section, or method. "
                            "Do not restate the whole prompt. Do not invent facts. "
                            f"{'Write concise Chinese. ' if prefer_zh else 'Write concise English. '}"
                            "Do not use markdown, bullets, or placeholders."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"User prompt: {str(prompt or '').strip()}\n"
                            f"Paper title: {str(display_name or '').strip()}\n"
                            f"Section heading: {str(heading_path or '').strip()}\n"
                            f"Current summary: {str(summary_line or '').strip()}\n"
                            f"Current why candidate: {str(why_seed or '').strip()}\n"
                            f"Evidence snippets:\n{candidate_payload}\n"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=140,
            )
            or ""
        ).strip()
    except Exception:
        return ""

    why_line = ""
    try:
        parsed = json.loads(out)
        if isinstance(parsed, dict):
            why_line = str(parsed.get("why_line") or "").strip()
    except Exception:
        m_why = re.search(r'"why_line"\s*:\s*"([^"]*)"', out)
        why_line = str(m_why.group(1) if m_why else "").strip()
    return why_line


def _maybe_polish_single_ref_hit_card(
    *,
    prompt: str,
    hit: dict,
    ui_meta: dict,
    allow_expensive_llm: bool = True,
) -> dict:
    ui = dict(ui_meta or {})
    title = str(ui.get("display_name") or "").strip()
    heading_path = str(ui.get("heading_path") or ui.get("section_label") or "").strip()
    summary_line = _normalize_ref_copy_text(str(ui.get("summary_line") or "").strip())
    why_line = _normalize_ref_copy_text(str(ui.get("why_line") or "").strip())
    summary_kind = str(ui.get("summary_kind") or "").strip().lower()
    why_generation = str(ui.get("why_generation") or "").strip().lower()
    allow_llm_polish = bool(allow_expensive_llm and _refs_card_polish_llm_enabled())

    deterministic_why = _build_prompt_aligned_ref_why_line_v3(
        prompt=prompt,
        display_name=title,
        heading_path=heading_path,
        summary_line=summary_line,
        why_line=why_line,
    )
    if deterministic_why and _why_line_needs_polish(
        prompt=prompt,
        display_name=title,
        heading_path=heading_path,
        summary_line=summary_line,
        why_line=why_line,
    ):
        why_line = deterministic_why
        why_generation = "deterministic_grounded"
        why_basis_meta = _build_ref_why_basis_meta(
            prompt=prompt,
            why_generation=why_generation,
            why_line=why_line,
        )
        ui["why_generation"] = str(why_basis_meta.get("why_generation") or why_generation)
        ui["why_basis"] = str(why_basis_meta.get("why_basis") or "")

    needs_summary = _summary_line_needs_polish(
        prompt=prompt,
        title=title,
        summary_line=summary_line,
    )
    original_needs_summary = bool(needs_summary)
    needs_why = _why_line_needs_polish(
        prompt=prompt,
        display_name=title,
        heading_path=heading_path,
        summary_line=summary_line,
        why_line=why_line,
    )
    candidates = _collect_ref_card_polish_candidates(hit, ui_meta=ui, max_items=6)
    attempt_grounded_why = bool(
        candidates
        and summary_kind != "metadata"
        and allow_llm_polish
        and why_generation != "llm_grounded"
    )
    if not (needs_summary or needs_why or attempt_grounded_why):
        ui["why_line"] = why_line
        return ui
    if not candidates:
        ui["why_line"] = why_line
        return ui
    if needs_summary:
        fallback_summary = _pick_ref_card_summary_fallback(
            prompt=prompt,
            title=title,
            candidates=candidates,
        )
        fallback_summary = _normalize_ref_copy_text(fallback_summary)
        if fallback_summary and _is_ref_card_summary_acceptable(
            prompt=prompt,
            title=title,
            summary_line=fallback_summary,
        ):
            ui["summary_line"] = fallback_summary
            if summary_kind == "guide":
                ui["summary_generation"] = "deterministic_grounded"
                basis_meta = _build_ref_summary_basis_meta(
                    prompt=prompt,
                    summary_kind=summary_kind,
                    summary_generation="deterministic_grounded",
                    summary_line=fallback_summary,
                )
                ui["summary_basis"] = str(basis_meta.get("summary_basis") or "")
            summary_line = fallback_summary
            needs_summary = False
    needs_why = _why_line_needs_polish(
        prompt=prompt,
        display_name=title,
        heading_path=heading_path,
        summary_line=summary_line,
        why_line=why_line,
    )
    candidate_payload = "\n".join(f"- {item}" for item in candidates if item)
    if candidate_payload and summary_kind != "metadata" and allow_llm_polish:
        grounded_why = _llm_ground_ref_why_line(
            prompt=prompt,
            display_name=title,
            heading_path=heading_path,
            summary_line=summary_line,
            why_seed=why_line,
            candidate_payload=candidate_payload,
        )
        grounded_why = _normalize_ref_copy_text(
            _summary_excerpt(grounded_why, max_sentences=2, max_len=160)
        )
        if grounded_why and (not _looks_generic_ref_why_line(grounded_why)):
            why_line = grounded_why
            why_generation = "llm_grounded"
            why_basis_meta = _build_ref_why_basis_meta(
                prompt=prompt,
                why_generation=why_generation,
                why_line=why_line,
            )
            ui["why_generation"] = str(why_basis_meta.get("why_generation") or why_generation)
            ui["why_basis"] = str(why_basis_meta.get("why_basis") or "")
            ui["why_line"] = why_line
            needs_why = False
    if not (needs_summary or needs_why):
        if (
            original_needs_summary
            and allow_llm_polish
            and _summary_line_needs_polish(
                prompt=prompt,
                title=title,
                summary_line=summary_line,
            )
        ):
            needs_summary = True
        else:
            ui["why_line"] = why_line
            return ui
    if not allow_llm_polish:
        ui["why_line"] = why_line
        return ui
    polished_summary, polished_why = _llm_polish_ref_card_copy_v2(
        prompt=prompt,
        display_name=title,
        heading_path=heading_path,
        summary_seed=summary_line,
        why_seed=why_line,
        candidate_payload=candidate_payload,
    )
    polished_summary = _normalize_ref_copy_text(
        _summary_excerpt(polished_summary, max_sentences=2, max_len=220)
    )
    polished_why = _normalize_ref_copy_text(
        _summary_excerpt(polished_why, max_sentences=2, max_len=160)
    )
    if polished_summary and (
        (
            len(_clean_summary_line(polished_summary)) >= 32
            and (not _looks_like_title_echo(polished_summary, title))
            and (not _looks_formula_heavy_ref_text(polished_summary))
        )
        or (not _summary_line_needs_polish(prompt=prompt, title=title, summary_line=polished_summary))
    ):
        ui["summary_line"] = polished_summary
        if summary_kind == "guide":
            ui["summary_generation"] = "llm_grounded"
            basis_meta = _build_ref_summary_basis_meta(
                prompt=prompt,
                summary_kind=summary_kind,
                summary_generation="llm_grounded",
                summary_line=polished_summary,
            )
            ui["summary_basis"] = str(basis_meta.get("summary_basis") or "")
    if polished_why and (not _looks_generic_ref_why_line(polished_why)):
        ui["why_line"] = polished_why
        why_generation = "llm_grounded"
        why_basis_meta = _build_ref_why_basis_meta(
            prompt=prompt,
            why_generation=why_generation,
            why_line=polished_why,
        )
        ui["why_generation"] = str(why_basis_meta.get("why_generation") or why_generation)
        ui["why_basis"] = str(why_basis_meta.get("why_basis") or "")
    else:
        ui["why_line"] = why_line
    return ui


def _maybe_polish_refs_card_copy(*, prompt: str, hits: list[dict], guide_mode: bool) -> list[dict]:
    rows = [dict(hit) for hit in (hits or []) if isinstance(hit, dict)]
    if not rows:
        return rows
    try:
        limit = int(str(os.environ.get("KB_REFS_CARD_POLISH_TOP_N", "2") or "2"))
    except Exception:
        limit = 2
    limit = max(0, min(4, limit))
    if limit <= 0:
        return rows
    polished: list[dict] = []
    for idx, hit in enumerate(rows):
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        if idx >= limit or not isinstance(ui_meta, dict):
            polished.append(hit)
            continue
        hit2 = dict(hit)
        hit2["ui_meta"] = _maybe_polish_single_ref_hit_card(
            prompt=prompt,
            hit=hit,
            ui_meta=ui_meta,
            allow_expensive_llm=True,
        )
        polished.append(hit2)
    return polished


def _compact_reader_open_text(text: str, *, max_len: int = 360) -> str:
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw:
        return ""
    if len(raw) <= max_len:
        return raw
    return raw[:max_len].rstrip() + "..."


_MIXED_QUOTE_SUFFIX_RE = re.compile(
    r"(^|[\s\(\[（【,:：;；，、])(?:[“\"']?)(?P<inner>[A-Za-z][A-Za-z0-9 .:/&+\-]{1,80})[’'](?=(?:中|里|处|部分|章节|小节|一节|该节|本节))"
)


def _normalize_ref_copy_text(text: str) -> str:
    s = " ".join(str(text or "").split())
    if not s:
        return ""

    def _repair_mixed_quote_suffix(match: re.Match[str]) -> str:
        prefix = str(match.group(1) or "")
        inner = str(match.group("inner") or "").strip(" '\"“”‘’")
        if not inner:
            return str(match.group(0) or "")
        return f"{prefix}“{inner}”"

    return _MIXED_QUOTE_SUFFIX_RE.sub(_repair_mixed_quote_suffix, s)


def _normalize_ref_copy_ui_meta(ui_meta: dict | None) -> dict:
    ui = dict(ui_meta or {})
    if not ui:
        return {}
    for key in ("summary_line", "why_line"):
        if key in ui:
            ui[key] = _normalize_ref_copy_text(str(ui.get(key) or ""))
    return ui


def _pick_reader_open_loc_text(loc: dict) -> str:
    if not isinstance(loc, dict):
        return ""
    for key in ("snippet", "text", "quote", "content", "summary", "why"):
        value = _compact_reader_open_text(str(loc.get(key) or ""))
        if value:
            return value
    return ""


def _refs_reader_open_candidate_key(candidate: dict) -> str:
    if not isinstance(candidate, dict):
        return ""
    return "::".join(
        [
            str(candidate.get("headingPath") or "").strip().lower(),
            str(candidate.get("highlightSnippet") or "").strip().lower()[:180],
            str(candidate.get("snippet") or "").strip().lower()[:180],
            str(candidate.get("anchorKind") or "").strip().lower(),
            str(_positive_int(candidate.get("anchorNumber")) or ""),
            str(candidate.get("blockId") or "").strip().lower(),
            str(candidate.get("anchorId") or "").strip().lower(),
        ]
    )


def _normalize_refs_reader_heading_path(*, prompt: str, source_path: str, heading_path: str) -> str:
    heading = _sanitize_heading_path_ui(
        str(heading_path or "").strip(),
        prompt=prompt,
        source_path=source_path,
    )
    if heading and " / " in heading:
        parts = [str(part or "").strip() for part in heading.split(" / ") if str(part or "").strip()]
        if len(parts) >= 2 and _looks_like_doc_title_heading_ui(parts[0], source_path):
            heading = " / ".join(parts[1:]).strip()
        elif len(parts) >= 3 and (not re.match(r"^\d", parts[0])) and re.match(r"^\d", parts[1]):
            heading = " / ".join(parts[1:]).strip()
    return heading


def _build_refs_reader_open_candidate(
    *,
    prompt: str,
    source_path: str,
    heading_path: str,
    snippet: str,
    highlight_snippet: str,
    anchor_kind: str,
    anchor_number: int,
) -> dict | None:
    heading = _normalize_refs_reader_heading_path(
        prompt=prompt,
        source_path=source_path,
        heading_path=heading_path,
    )
    snippet_text = _compact_reader_open_text(snippet)
    highlight_text = _compact_reader_open_text(highlight_snippet or snippet_text)
    candidate = {
        "headingPath": heading or None,
        "snippet": snippet_text or None,
        "highlightSnippet": highlight_text or None,
        "anchorKind": str(anchor_kind or "").strip().lower() or None,
        "anchorNumber": _positive_int(anchor_number) or None,
    }
    if not any(candidate.values()):
        return None
    return {key: value for key, value in candidate.items() if value not in (None, "", [], {})}


def _infer_heading_path_for_summary_from_source_blocks(
    *,
    prompt: str,
    source_path: str,
    summary_line: str,
    anchor_target_kind: str,
    anchor_target_number: int,
) -> str:
    seed = _compact_reader_open_text(summary_line)
    if not seed:
        return ""
    md_path = _resolve_source_md_path(source_path)
    if md_path is None:
        return ""
    try:
        blocks = load_source_blocks(md_path)
    except Exception:
        return ""
    if not blocks:
        return ""
    try:
        matches = match_source_blocks(
            blocks,
            snippet=seed,
            heading_path="",
            prefer_kind=anchor_target_kind,
            target_number=anchor_target_number,
            limit=3,
            score_floor=0.24,
        )
    except Exception:
        matches = []
    for row in matches:
        block = row.get("block") if isinstance(row, dict) else {}
        heading_path = _normalize_refs_reader_heading_path(
            prompt=prompt,
            source_path=source_path,
            heading_path=str((block or {}).get("heading_path") or "").strip(),
        )
        if heading_path:
            return heading_path
    return ""


def _resolve_source_md_path(source_path: str) -> Path | None:
    raw = str(source_path or "").strip()
    if not raw:
        return None
    candidates: list[Path] = []
    direct = Path(raw)
    candidates.append(direct)
    if not direct.is_absolute():
        candidates.append(_REPO_ROOT / raw)
        candidates.append(Path.cwd() / raw)
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        key = str(resolved).strip().lower()
        if (not key) or (key in seen):
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _select_reader_open_exact_snippet(seed_text: str, block_text: str) -> tuple[str, str]:
    seed = _compact_reader_open_text(seed_text)
    block = _compact_reader_open_text(block_text)
    if seed and block:
        seed_key = re.sub(r"\s+", " ", seed).strip().lower()
        block_key = re.sub(r"\s+", " ", block).strip().lower()
        if seed_key and block_key and (seed_key in block_key or block_key in seed_key):
            return seed, seed
    if seed:
        return seed, seed
    return block, block


def _build_refs_exact_candidate_from_block(
    *,
    prompt: str,
    source_path: str,
    block: dict,
    seed_heading_path: str,
    seed_snippet: str,
    anchor_kind: str,
    anchor_number: int,
) -> dict | None:
    if not isinstance(block, dict):
        return None
    block_id = str(block.get("block_id") or "").strip()
    anchor_id = str(block.get("anchor_id") or "").strip()
    if not block_id:
        return None
    heading_path = str(block.get("heading_path") or seed_heading_path or "").strip()
    block_text = str(block.get("text") or block.get("raw_text") or "").strip()
    snippet_text, highlight_text = _select_reader_open_exact_snippet(seed_snippet, block_text)
    candidate = _build_refs_reader_open_candidate(
        prompt=prompt,
        source_path=source_path,
        heading_path=heading_path,
        snippet=snippet_text,
        highlight_snippet=highlight_text,
        anchor_kind=anchor_kind or str(block.get("kind") or ""),
        anchor_number=anchor_number or int(block.get("number") or 0),
    )
    if not isinstance(candidate, dict):
        return None
    candidate["blockId"] = block_id
    if anchor_id:
        candidate["anchorId"] = anchor_id
    return candidate


def _refs_locate_llm_enabled() -> bool:
    raw_flag = str(os.environ.get("KB_REFS_LOCATE_USE_LLM", "1") or "").strip().lower()
    if raw_flag in {"0", "false", "off", "no"}:
        return False
    try:
        settings = load_settings()
    except Exception:
        return False
    return bool(getattr(settings, "api_key", None))


def _should_try_refs_locate_llm(rows: list[dict]) -> bool:
    if len(rows) < 2:
        return False
    try:
        top = float(rows[0].get("score") or 0.0)
        second = float(rows[1].get("score") or 0.0)
    except Exception:
        return False
    if top <= 0.0:
        return False
    margin = top - second
    # Use LLM only when heuristic block matching is genuinely ambiguous.
    return bool(top < 1.08 or margin < 0.14)


@lru_cache(maxsize=512)
def _llm_pick_refs_exact_candidate_index(
    *,
    prompt: str,
    source_path: str,
    anchor_target_kind: str,
    anchor_target_number: int,
    candidates_payload: str,
) -> int:
    if not prompt or not candidates_payload:
        return -1
    if not _refs_locate_llm_enabled():
        return -1
    try:
        settings = load_settings()
    except Exception:
        return -1
    if not getattr(settings, "api_key", None):
        return -1
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
                            "You are selecting the single best reader jump target inside one paper. "
                            "Choose the candidate block that most directly and precisely answers the user prompt. "
                            "Prefer exact mention over broad context, prefer the requested equation/figure number when present, "
                            "and avoid generic surrounding paragraphs when a more explicit block exists. "
                            "Return JSON only, like {\"best\": 2}. Use 1-based indexing. If none is clearly suitable, return {\"best\": 0}."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Prompt: {str(prompt or '').strip()}\n"
                            f"Source: {str(source_path or '').strip()}\n"
                            f"Target anchor kind: {str(anchor_target_kind or '').strip().lower() or 'none'}\n"
                            f"Target anchor number: {int(max(0, int(anchor_target_number or 0)))}\n\n"
                            f"Candidates:\n{candidates_payload}\n"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=120,
            )
            or ""
        ).strip()
    except Exception:
        return -1
    m = re.search(r'"best"\s*:\s*(-?\d+)', out)
    if not m:
        m = re.search(r"\b(-?\d+)\b", out)
    if not m:
        return -1
    try:
        picked = int(m.group(1))
    except Exception:
        return -1
    return picked if picked > 0 else 0


def _resolve_refs_exact_candidates(
    *,
    prompt: str,
    source_path: str,
    anchor_target_kind: str,
    anchor_target_number: int,
    primary_candidate: dict | None,
    secondary_candidates: list[dict],
    allow_llm_disambiguation: bool = True,
) -> list[dict]:
    md_path = _resolve_source_md_path(source_path)
    if md_path is None:
        return []
    try:
        blocks = load_source_blocks(md_path)
    except Exception:
        return []
    if not blocks:
        return []

    seed_candidates = [primary_candidate] if isinstance(primary_candidate, dict) else []
    seed_candidates.extend(item for item in (secondary_candidates or []) if isinstance(item, dict))
    if not seed_candidates:
        return []
    primary_heading_norm = str(
        ((primary_candidate or {}) if isinstance(primary_candidate, dict) else {}).get("headingPath") or ""
    ).strip().lower()

    out_rows: list[dict] = []
    seen_blocks: set[str] = set()
    for seed in seed_candidates[:6]:
        heading_path = str(seed.get("headingPath") or "").strip()
        snippet = str(seed.get("highlightSnippet") or seed.get("snippet") or "").strip()
        score_floor = 0.52 if snippet else 0.68
        if _positive_int(anchor_target_number) > 0:
            score_floor = 0.34 if snippet else 0.58
        try:
            matches = match_source_blocks(
                blocks,
                snippet=snippet,
                heading_path=heading_path,
                prefer_kind=anchor_target_kind,
                target_number=anchor_target_number,
                limit=3,
                score_floor=score_floor,
            )
        except Exception:
            matches = []
        for row in matches:
            block = row.get("block")
            candidate = _build_refs_exact_candidate_from_block(
                prompt=prompt,
                source_path=source_path,
                block=block if isinstance(block, dict) else {},
                seed_heading_path=heading_path,
                seed_snippet=snippet,
                anchor_kind=anchor_target_kind,
                anchor_number=anchor_target_number,
            )
            if not isinstance(candidate, dict):
                continue
            block_id = str(candidate.get("blockId") or "").strip()
            if (not block_id) or (block_id in seen_blocks):
                continue
            seen_blocks.add(block_id)
            out_rows.append(
                {
                    "candidate": candidate,
                    "score": float(row.get("score") or 0.0),
                    "block_text": str(((block or {}) if isinstance(block, dict) else {}).get("text") or "").strip(),
                    "heading_path": heading_path,
                }
            )
            if len(out_rows) >= 5:
                break
        if len(out_rows) >= 5:
            break

    if len(out_rows) <= 1:
        return [dict(item.get("candidate") or {}) for item in out_rows if isinstance(item.get("candidate"), dict)]

    def _exact_candidate_sort_key(item: dict) -> tuple[int, int, float]:
        candidate = dict(item.get("candidate") or {}) if isinstance(item.get("candidate"), dict) else {}
        candidate_heading = str(candidate.get("headingPath") or "").strip().lower()
        seed_heading = str(item.get("heading_path") or "").strip().lower()
        primary_match = int(bool(primary_heading_norm and candidate_heading and candidate_heading == primary_heading_norm))
        seed_match = int(bool(seed_heading and candidate_heading and candidate_heading == seed_heading))
        return (
            primary_match,
            seed_match,
            float(item.get("score") or 0.0),
        )

    out_rows.sort(key=_exact_candidate_sort_key, reverse=True)
    if allow_llm_disambiguation and _should_try_refs_locate_llm(out_rows):
        candidate_lines: list[str] = []
        for idx, row in enumerate(out_rows[:3], start=1):
            candidate = dict(row.get("candidate") or {}) if isinstance(row.get("candidate"), dict) else {}
            candidate_lines.append(
                "\n".join(
                    [
                        f"{idx}. heading: {str(candidate.get('headingPath') or '').strip() or '(none)'}",
                        f"   snippet: {str(candidate.get('highlightSnippet') or candidate.get('snippet') or '').strip()[:260]}",
                        f"   block_text: {str(row.get('block_text') or '').strip()[:260]}",
                        f"   anchor: {str(candidate.get('anchorKind') or '').strip()} {str(candidate.get('anchorNumber') or '').strip()}",
                        f"   heuristic_score: {float(row.get('score') or 0.0):.3f}",
                    ]
                )
            )
        picked = _llm_pick_refs_exact_candidate_index(
            prompt=str(prompt or "").strip(),
            source_path=str(source_path or "").strip(),
            anchor_target_kind=str(anchor_target_kind or "").strip().lower(),
            anchor_target_number=int(_positive_int(anchor_target_number)),
            candidates_payload="\n\n".join(candidate_lines),
        )
        if picked > 0 and picked <= min(3, len(out_rows)):
            chosen = out_rows[picked - 1]
            out_rows = [chosen] + [row for idx, row in enumerate(out_rows) if idx != (picked - 1)]

    return [dict(item.get("candidate") or {}) for item in out_rows if isinstance(item.get("candidate"), dict)]


def _build_refs_reader_open_payload(
    *,
    meta: dict,
    prompt: str,
    source_path: str,
    display_name: str,
    heading_path: str,
    heading: str,
    summary_line: str,
    why_line: str,
    anchor_target_kind: str,
    anchor_target_number: int,
    allow_llm_disambiguation: bool = True,
    allow_exact_locate: bool = True,
) -> dict:
    primary_heading = str(heading_path or heading or "").strip()
    primary_snippet = _compact_reader_open_text(summary_line or why_line)
    primary_candidate = _build_refs_reader_open_candidate(
        prompt=prompt,
        source_path=source_path,
        heading_path=primary_heading,
        snippet=primary_snippet,
        highlight_snippet=primary_snippet,
        anchor_kind=anchor_target_kind,
        anchor_number=anchor_target_number,
    )

    secondary_candidates: list[dict] = []
    seen_secondary: set[str] = set()
    primary_key = _refs_reader_open_candidate_key(primary_candidate or {})

    def _push_secondary(candidate: dict | None) -> None:
        if not isinstance(candidate, dict):
            return
        key = _refs_reader_open_candidate_key(candidate)
        if (not key) or (key == primary_key) or (key in seen_secondary):
            return
        seen_secondary.add(key)
        secondary_candidates.append(candidate)

    raw_locs = meta.get("ref_locs")
    if isinstance(raw_locs, list):
        for loc in raw_locs[:4]:
            if not isinstance(loc, dict):
                continue
            loc_heading = str(loc.get("heading_path") or loc.get("heading") or "").strip()
            loc_snippet = _pick_reader_open_loc_text(loc) or primary_snippet
            _push_secondary(
                _build_refs_reader_open_candidate(
                    prompt=prompt,
                    source_path=source_path,
                    heading_path=loc_heading,
                    snippet=loc_snippet,
                    highlight_snippet=loc_snippet,
                    anchor_kind=anchor_target_kind,
                    anchor_number=anchor_target_number,
                )
            )

    snippet_seed_keys = (
        ("ref_show_snippets", 3),
        ("ref_snippets", 3),
        ("ref_overview_snippets", 2),
    )
    for meta_key, limit in snippet_seed_keys:
        raw_arr = meta.get(meta_key)
        if not isinstance(raw_arr, list):
            continue
        for item in raw_arr[:limit]:
            snippet_text = _compact_reader_open_text(str(item or ""))
            if not snippet_text:
                continue
            _push_secondary(
                _build_refs_reader_open_candidate(
                    prompt=prompt,
                    source_path=source_path,
                    heading_path=primary_heading,
                    snippet=snippet_text,
                    highlight_snippet=snippet_text,
                    anchor_kind=anchor_target_kind,
                    anchor_number=anchor_target_number,
                )
            )

    ref_pack_state = str(meta.get("ref_pack_state") or "").strip().lower()
    if (ref_pack_state == "pending") or (not allow_exact_locate):
        visible_candidates: list[dict] = []
        seen_visible: set[str] = set()

        def _push_visible_pending(candidate: dict | None) -> None:
            if not isinstance(candidate, dict):
                return
            key = _refs_reader_open_candidate_key(candidate)
            if (not key) or (key in seen_visible):
                return
            seen_visible.add(key)
            visible_candidates.append(candidate)

        _push_visible_pending(primary_candidate)
        for candidate in secondary_candidates:
            _push_visible_pending(candidate)
        visible_candidates = visible_candidates[:6]
        effective_primary = visible_candidates[0] if visible_candidates else primary_candidate
        secondary_visible = visible_candidates[1:] if len(visible_candidates) > 1 else []
        reader_open = {
            "sourcePath": source_path,
            "sourceName": display_name,
            "headingPath": str((effective_primary or {}).get("headingPath") or primary_heading or "").strip() or None,
            "snippet": str((effective_primary or {}).get("snippet") or primary_snippet or "").strip() or None,
            "highlightSnippet": str((effective_primary or {}).get("highlightSnippet") or primary_snippet or "").strip() or None,
            "anchorKind": str((effective_primary or {}).get("anchorKind") or anchor_target_kind or "").strip().lower() or None,
            "anchorNumber": _positive_int((effective_primary or {}).get("anchorNumber") or anchor_target_number) or None,
            "strictLocate": False,
            "alternatives": secondary_visible or None,
            "visibleAlternatives": visible_candidates if len(visible_candidates) > 1 else None,
            "evidenceAlternatives": visible_candidates if len(visible_candidates) > 1 else None,
            "initialAltIndex": 0 if visible_candidates else None,
        }
        return {key: value for key, value in reader_open.items() if value not in (None, "", [], {})}

    exact_candidates = _resolve_refs_exact_candidates(
        prompt=prompt,
        source_path=source_path,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        primary_candidate=primary_candidate,
        secondary_candidates=secondary_candidates,
        allow_llm_disambiguation=allow_llm_disambiguation,
    )
    primary_exact = exact_candidates[0] if exact_candidates else None
    related_block_ids = [
        str(candidate.get("blockId") or "").strip()
        for candidate in exact_candidates
        if str(candidate.get("blockId") or "").strip()
    ]
    related_block_ids = list(dict.fromkeys(related_block_ids))[:5]

    effective_primary = primary_exact or primary_candidate
    visible_candidates: list[dict] = []
    seen_visible: set[str] = set()

    def _push_visible(candidate: dict | None) -> None:
        if not isinstance(candidate, dict):
            return
        key = _refs_reader_open_candidate_key(candidate)
        if (not key) or (key in seen_visible):
            return
        seen_visible.add(key)
        visible_candidates.append(candidate)

    _push_visible(effective_primary)
    for candidate in exact_candidates[1:]:
        _push_visible(candidate)
    for candidate in secondary_candidates:
        _push_visible(candidate)

    visible_candidates = visible_candidates[:6]
    secondary_visible = [candidate for candidate in visible_candidates if candidate is not effective_primary]
    secondary_candidates = secondary_visible[:5]
    locate_target = (
        {
            "headingPath": str((primary_exact or {}).get("headingPath") or "").strip() or None,
            "snippet": str((primary_exact or {}).get("snippet") or "").strip() or None,
            "highlightSnippet": str((primary_exact or {}).get("highlightSnippet") or "").strip() or None,
            "blockId": str((primary_exact or {}).get("blockId") or "").strip() or None,
            "anchorId": str((primary_exact or {}).get("anchorId") or "").strip() or None,
            "anchorKind": str((primary_exact or {}).get("anchorKind") or anchor_target_kind or "").strip().lower() or None,
            "anchorNumber": _positive_int((primary_exact or {}).get("anchorNumber") or anchor_target_number) or None,
            "hitLevel": "block",
            "relatedBlockIds": related_block_ids or None,
        }
        if primary_exact
        else None
    )
    if isinstance(locate_target, dict):
        locate_target = {key: value for key, value in locate_target.items() if value not in (None, "", [], {})}
    reader_open = {
        "sourcePath": source_path,
        "sourceName": display_name,
        "headingPath": str((effective_primary or {}).get("headingPath") or primary_heading or "").strip() or None,
        "snippet": str((effective_primary or {}).get("snippet") or primary_snippet or "").strip() or None,
        "highlightSnippet": str((effective_primary or {}).get("highlightSnippet") or primary_snippet or "").strip() or None,
        "blockId": str((effective_primary or {}).get("blockId") or "").strip() or None,
        "anchorId": str((effective_primary or {}).get("anchorId") or "").strip() or None,
        "relatedBlockIds": related_block_ids or None,
        "anchorKind": str((effective_primary or {}).get("anchorKind") or anchor_target_kind or "").strip().lower() or None,
        "anchorNumber": _positive_int((effective_primary or {}).get("anchorNumber") or anchor_target_number) or None,
        "strictLocate": bool(primary_exact),
        "locateTarget": locate_target,
        "alternatives": secondary_candidates or None,
        "visibleAlternatives": visible_candidates if len(visible_candidates) > 1 else None,
        "evidenceAlternatives": visible_candidates if len(visible_candidates) > 1 else None,
        "initialAltIndex": 0 if visible_candidates else None,
    }
    return {key: value for key, value in reader_open.items() if value not in (None, "", [], {})}


def _build_primary_ref_evidence_payload(
    *,
    source_path: str,
    display_name: str,
    reader_open: dict,
    selection_reason: str,
    score: float | None,
) -> dict:
    if not isinstance(reader_open, dict):
        return {}

    def _candidate_to_evidence(candidate: dict | None) -> dict | None:
        if not isinstance(candidate, dict):
            return None
        evidence = {
            "source_path": str(source_path or "").strip() or None,
            "source_name": str(display_name or "").strip() or None,
            "block_id": str(candidate.get("blockId") or "").strip() or None,
            "anchor_id": str(candidate.get("anchorId") or "").strip() or None,
            "heading_path": str(candidate.get("headingPath") or "").strip() or None,
            "snippet": str(candidate.get("snippet") or "").strip() or None,
            "highlight_snippet": str(candidate.get("highlightSnippet") or "").strip() or None,
            "anchor_kind": str(candidate.get("anchorKind") or "").strip().lower() or None,
            "anchor_number": _positive_int(candidate.get("anchorNumber")) or None,
        }
        return {key: value for key, value in evidence.items() if value not in (None, "", [], {})}

    primary_candidate = {
        "headingPath": str(reader_open.get("headingPath") or "").strip(),
        "snippet": str(reader_open.get("snippet") or "").strip(),
        "highlightSnippet": str(reader_open.get("highlightSnippet") or "").strip(),
        "blockId": str(reader_open.get("blockId") or "").strip(),
        "anchorId": str(reader_open.get("anchorId") or "").strip(),
        "anchorKind": str(reader_open.get("anchorKind") or "").strip().lower(),
        "anchorNumber": _positive_int(reader_open.get("anchorNumber")),
    }
    primary_key = _refs_reader_open_candidate_key(primary_candidate)
    primary_evidence = _candidate_to_evidence(primary_candidate)
    if not isinstance(primary_evidence, dict) or not primary_evidence:
        return {}

    alternatives: list[dict] = []
    seen_alt_keys: set[str] = set()
    for raw_candidate in list(reader_open.get("evidenceAlternatives") or reader_open.get("visibleAlternatives") or reader_open.get("alternatives") or []):
        if not isinstance(raw_candidate, dict):
            continue
        key = _refs_reader_open_candidate_key(raw_candidate)
        if (not key) or (key == primary_key) or (key in seen_alt_keys):
            continue
        seen_alt_keys.add(key)
        alt = _candidate_to_evidence(raw_candidate)
        if isinstance(alt, dict) and alt:
            alternatives.append(alt)
        if len(alternatives) >= 5:
            break

    out = dict(primary_evidence)
    if selection_reason:
        out["selection_reason"] = str(selection_reason or "").strip()
    if score is not None:
        try:
            out["score"] = float(score)
        except Exception:
            pass
    out["strict_locate"] = bool(reader_open.get("strictLocate"))
    if alternatives:
        out["alternatives"] = alternatives
    return out


def _normalize_primary_ref_evidence_payload(primary_evidence: dict | None) -> dict:
    if not isinstance(primary_evidence, dict):
        return {}
    out = {
        "source_path": str(primary_evidence.get("source_path") or primary_evidence.get("sourcePath") or "").strip() or None,
        "source_name": str(primary_evidence.get("source_name") or primary_evidence.get("sourceName") or "").strip() or None,
        "block_id": str(primary_evidence.get("block_id") or primary_evidence.get("blockId") or "").strip() or None,
        "anchor_id": str(primary_evidence.get("anchor_id") or primary_evidence.get("anchorId") or "").strip() or None,
        "heading_path": str(primary_evidence.get("heading_path") or primary_evidence.get("headingPath") or "").strip() or None,
        "snippet": str(primary_evidence.get("snippet") or "").strip() or None,
        "highlight_snippet": str(primary_evidence.get("highlight_snippet") or primary_evidence.get("highlightSnippet") or "").strip() or None,
        "anchor_kind": str(primary_evidence.get("anchor_kind") or primary_evidence.get("anchorKind") or "").strip().lower() or None,
        "anchor_number": _positive_int(primary_evidence.get("anchor_number") or primary_evidence.get("anchorNumber")) or None,
        "selection_reason": str(primary_evidence.get("selection_reason") or primary_evidence.get("selectionReason") or "").strip() or None,
    }
    strict_locate_raw = primary_evidence.get("strict_locate")
    if strict_locate_raw is None:
        strict_locate_raw = primary_evidence.get("strictLocate")
    if strict_locate_raw is not None:
        out["strict_locate"] = bool(strict_locate_raw)
    score_raw = primary_evidence.get("score")
    try:
        if score_raw is not None:
            out["score"] = float(score_raw)
    except Exception:
        pass
    alts: list[dict] = []
    for raw_alt in list(primary_evidence.get("alternatives") or []):
        norm_alt = _normalize_primary_ref_evidence_payload(raw_alt)
        if norm_alt:
            alts.append(norm_alt)
        if len(alts) >= 5:
            break
    if alts:
        out["alternatives"] = alts
    return {
        key: value
        for key, value in out.items()
        if value not in (None, "", [], {})
    }


def _extract_hit_primary_ref_evidence(hit: dict | None) -> dict:
    if not isinstance(hit, dict):
        return {}
    ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
    for candidate in (
        ui_meta.get("primary_evidence") if isinstance(ui_meta, dict) else {},
        ((ui_meta.get("reader_open") or {}).get("primaryEvidence") if isinstance(ui_meta.get("reader_open"), dict) else {}),
        ((hit.get("reader_open") or {}).get("primaryEvidence") if isinstance(hit.get("reader_open"), dict) else {}),
    ):
        norm = _normalize_primary_ref_evidence_payload(candidate if isinstance(candidate, dict) else {})
        if norm:
            return norm
    return {}


def _attach_pack_primary_ref_evidence(pack: dict | None) -> dict:
    pack2 = dict(pack or {}) if isinstance(pack, dict) else {}
    existing = _normalize_primary_ref_evidence_payload(pack2.get("primary_evidence") if isinstance(pack2.get("primary_evidence"), dict) else {})
    primary = existing
    if not primary:
        for hit in list(pack2.get("hits") or []):
            primary = _extract_hit_primary_ref_evidence(hit if isinstance(hit, dict) else {})
            if primary:
                break
    if primary:
        pack2["primary_evidence"] = dict(primary)
        heading_path = str(primary.get("heading_path") or "").strip()
        if heading_path:
            pack2["primary_evidence_heading_path"] = heading_path
    return pack2


def _attach_pack_display_contract(pack: dict | None) -> dict:
    pack2 = _attach_pack_primary_ref_evidence(pack)
    hits = [hit for hit in list(pack2.get("hits") or []) if isinstance(hit, dict)]
    guide_filter = pack2.get("guide_filter") if isinstance(pack2.get("guide_filter"), dict) else {}
    pipeline_debug = pack2.get("pipeline_debug") if isinstance(pack2.get("pipeline_debug"), dict) else {}
    payload_mode = str(pack2.get("payload_mode") or "").strip().lower()
    render_status = str(pack2.get("render_status") or "").strip().lower()
    pending = bool(pack2.get("pending")) or bool(pack2.get("enrichment_pending")) or payload_mode == "pending"
    hidden_self_source = bool(guide_filter.get("hidden_self_source"))
    try:
        raw_hit_count = int(pipeline_debug.get("raw_hit_count") or 0)
    except Exception:
        raw_hit_count = 0
    try:
        post_score_gate_hit_count = int(pipeline_debug.get("post_score_gate_hit_count") or 0)
    except Exception:
        post_score_gate_hit_count = 0
    try:
        post_focus_filter_hit_count = int(pipeline_debug.get("post_focus_filter_hit_count") or 0)
    except Exception:
        post_focus_filter_hit_count = 0
    try:
        post_llm_filter_hit_count = int(pipeline_debug.get("post_llm_filter_hit_count") or 0)
    except Exception:
        post_llm_filter_hit_count = 0

    display_state = "empty"
    suppression_reason = ""
    if pending:
        display_state = "pending"
        suppression_reason = "pending_enrichment"
    elif hits:
        display_state = "ready"
    elif hidden_self_source:
        display_state = "hidden_by_guide"
        suppression_reason = "guide_self_source_only"
    elif render_status == "failed":
        display_state = "suppressed"
        suppression_reason = "render_failed"
    elif raw_hit_count > 0:
        display_state = "suppressed"
        if (post_llm_filter_hit_count <= 0) and (post_focus_filter_hit_count > 0):
            suppression_reason = "llm_filter_removed_all"
        elif (post_focus_filter_hit_count <= 0) and (post_score_gate_hit_count > 0):
            suppression_reason = "focus_filter_removed_all"
        elif post_score_gate_hit_count <= 0:
            suppression_reason = "score_gate_removed_all"
        else:
            suppression_reason = "no_renderable_hits"
    else:
        display_state = "empty"
        suppression_reason = "no_candidate_hits"

    pack2["display_state"] = display_state
    if suppression_reason:
        pack2["suppression_reason"] = suppression_reason
    else:
        pack2.pop("suppression_reason", None)
    return pack2


def _doc_list_ref_why_line(*, prompt: str, heading_path: str, prefer_zh: bool) -> str:
    heading = str(heading_path or "").strip()
    focus_action = _shared_prompt_reference_focus_action(prompt)
    if prefer_zh:
        if focus_action == "compare":
            return f"这篇文献作为当前多篇对比查询中的直接命中被保留，定位落在 {heading or '命中章节'}。"
        if focus_action == "define":
            return f"这篇文献作为当前多篇定义/介绍查询中的直接命中被保留，定位落在 {heading or '命中章节'}。"
        return f"这篇文献作为当前多篇库内命中的一项被保留，定位落在 {heading or '命中章节'}。"
    if focus_action == "compare":
        return f"This paper was kept as a direct comparison match for the current multi-paper query, anchored to {heading or 'the matched section'}."
    if focus_action == "define":
        return f"This paper was kept as a direct definition/introduction match for the current multi-paper query, anchored to {heading or 'the matched section'}."
    return f"This paper was kept as one of the direct library matches for the current multi-paper query, anchored to {heading or 'the matched section'}."


def _collect_doc_list_ref_text_candidates(*, raw_item: dict, primary_evidence: dict) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        text = str(value or "").strip()
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(text)

    _push(str(primary_evidence.get("highlight_snippet") or "").strip())
    _push(str(primary_evidence.get("snippet") or "").strip())
    _push(str(raw_item.get("summary_line") or "").strip())
    for alt in list(primary_evidence.get("alternatives") or []):
        if not isinstance(alt, dict):
            continue
        _push(str(alt.get("highlight_snippet") or "").strip())
        _push(str(alt.get("snippet") or "").strip())
    return out


def _primary_ref_evidence_summary_seed(primary_evidence: dict | None) -> str:
    primary = _normalize_primary_ref_evidence_payload(primary_evidence if isinstance(primary_evidence, dict) else {})
    if not primary:
        return ""
    return _compact_reader_open_text(
        str(primary.get("highlight_snippet") or primary.get("snippet") or "").strip()
    )


def _primary_ref_evidence_points_to_same_surface(
    left_primary: dict | None,
    right_primary: dict | None,
) -> bool:
    left = _normalize_primary_ref_evidence_payload(left_primary if isinstance(left_primary, dict) else {})
    right = _normalize_primary_ref_evidence_payload(right_primary if isinstance(right_primary, dict) else {})
    if (not left) or (not right):
        return False

    left_source = str(left.get("source_path") or "").strip()
    right_source = str(right.get("source_path") or "").strip()
    if left_source and right_source and (not _same_source_identity(left_source, right_source)):
        return False

    left_block = str(left.get("block_id") or "").strip()
    right_block = str(right.get("block_id") or "").strip()
    if left_block or right_block:
        return bool(left_block and right_block and left_block == right_block)

    left_anchor = str(left.get("anchor_id") or "").strip()
    right_anchor = str(right.get("anchor_id") or "").strip()
    if left_anchor or right_anchor:
        return bool(left_anchor and right_anchor and left_anchor == right_anchor)

    left_heading = str(left.get("heading_path") or "").strip()
    right_heading = str(right.get("heading_path") or "").strip()
    if left_heading and right_heading and left_heading != right_heading:
        return False

    left_summary = _primary_ref_evidence_summary_seed(left)
    right_summary = _primary_ref_evidence_summary_seed(right)
    if left_summary and right_summary:
        return _ref_summary_surfaces_match(left_summary, right_summary)
    if left_heading and right_heading:
        return True
    return False


def _doc_list_authoritative_primary_is_upgradeable(primary_evidence: dict | None) -> bool:
    primary = _normalize_primary_ref_evidence_payload(primary_evidence if isinstance(primary_evidence, dict) else {})
    if not primary:
        return True
    if bool(primary.get("strict_locate")):
        return False
    if str(primary.get("block_id") or "").strip():
        return False
    if str(primary.get("anchor_id") or "").strip():
        return False
    reason = str(primary.get("selection_reason") or "").strip().lower()
    return reason in {"", "answer_hit_top", "pending_section_seed"}


def _primary_ref_evidence_precision_score(
    *,
    primary_evidence: dict | None,
    prompt: str,
    display_name: str,
) -> tuple[int, int, int, int, int, int, int]:
    primary = _normalize_primary_ref_evidence_payload(primary_evidence if isinstance(primary_evidence, dict) else {})
    if not primary:
        return (0, 0, 0, 0, 0, 0, 0)
    reason = str(primary.get("selection_reason") or "").strip().lower()
    reason_rank = {
        "prompt_aligned_block": 8,
        "prompt_aligned": 7,
        "navigation": 6,
        "fallback": 4,
        "reader_open": 4,
        "strict_locate": 4,
        "shared_refs_pack": 4,
        "answer_hit_top": 0,
        "pending_section_seed": 0,
    }.get(reason, 3 if reason else 0)
    heading_path = _sanitize_heading_path_ui(
        str(primary.get("heading_path") or "").strip(),
        prompt=prompt,
        source_path=str(primary.get("source_path") or "").strip(),
    )
    summary_seed = _primary_ref_evidence_summary_seed(primary)
    summary_seed_usable = bool(
        summary_seed
        and (not _looks_bibliographic_source_block_text(summary_seed))
        and (not _summary_line_needs_polish(
            prompt=prompt,
            title=display_name,
            summary_line=summary_seed,
        ))
    )
    return (
        reason_rank,
        1 if bool(primary.get("strict_locate")) else 0,
        1 if str(primary.get("block_id") or "").strip() else 0,
        1 if str(primary.get("anchor_id") or "").strip() else 0,
        1 if heading_path else 0,
        1 if summary_seed_usable else 0,
        1 if summary_seed else 0,
    )


def _select_doc_list_effective_primary_evidence(
    *,
    prompt: str,
    display_name: str,
    authoritative_primary_evidence: dict | None,
    synthesized_primary_evidence: dict | None,
) -> tuple[dict, str]:
    authoritative = _normalize_primary_ref_evidence_payload(
        authoritative_primary_evidence if isinstance(authoritative_primary_evidence, dict) else {}
    )
    synthesized = _normalize_primary_ref_evidence_payload(
        synthesized_primary_evidence if isinstance(synthesized_primary_evidence, dict) else {}
    )
    if not authoritative:
        return synthesized, "synthesized"
    if not synthesized:
        return authoritative, "authoritative"
    if _primary_ref_evidence_points_to_same_surface(authoritative, synthesized):
        authoritative_score = _primary_ref_evidence_precision_score(
            primary_evidence=authoritative,
            prompt=prompt,
            display_name=display_name,
        )
        synthesized_score = _primary_ref_evidence_precision_score(
            primary_evidence=synthesized,
            prompt=prompt,
            display_name=display_name,
        )
        return (
            (synthesized, "synthesized")
            if synthesized_score > authoritative_score
            else (authoritative, "authoritative")
        )
    if not _doc_list_authoritative_primary_is_upgradeable(authoritative):
        return authoritative, "authoritative"

    authoritative_score = _primary_ref_evidence_precision_score(
        primary_evidence=authoritative,
        prompt=prompt,
        display_name=display_name,
    )
    synthesized_score = _primary_ref_evidence_precision_score(
        primary_evidence=synthesized,
        prompt=prompt,
        display_name=display_name,
    )
    if synthesized_score > authoritative_score:
        return synthesized, "synthesized"
    if authoritative_score > synthesized_score:
        return authoritative, "authoritative"

    auth_reason = str(authoritative.get("selection_reason") or "").strip().lower()
    synth_reason = str(synthesized.get("selection_reason") or "").strip().lower()
    if bool(synthesized.get("strict_locate")) and (not bool(authoritative.get("strict_locate"))):
        return synthesized, "synthesized"
    if synth_reason in {"prompt_aligned_block", "prompt_aligned"} and auth_reason in {"", "answer_hit_top", "pending_section_seed"}:
        return synthesized, "synthesized"
    return authoritative, "authoritative"


def _apply_doc_list_effective_primary_evidence(
    *,
    prompt: str,
    display_name: str,
    fallback_heading_path: str,
    ui_meta: dict | None,
    authoritative_primary_evidence: dict | None,
    authoritative_summary_line: str = "",
) -> tuple[dict, dict]:
    ui_out = dict(ui_meta or {}) if isinstance(ui_meta, dict) else {}
    synthesized_primary = _normalize_primary_ref_evidence_payload(
        ui_out.get("primary_evidence") if isinstance(ui_out.get("primary_evidence"), dict) else {}
    )
    authoritative_primary = _normalize_primary_ref_evidence_payload(
        authoritative_primary_evidence if isinstance(authoritative_primary_evidence, dict) else {}
    )
    effective_primary, selected_source = _select_doc_list_effective_primary_evidence(
        prompt=prompt,
        display_name=display_name,
        authoritative_primary_evidence=authoritative_primary,
        synthesized_primary_evidence=synthesized_primary,
    )
    effective_heading_path = str(
        effective_primary.get("heading_path")
        or ui_out.get("heading_path")
        or fallback_heading_path
        or ""
    ).strip()
    if effective_heading_path and (
        (not str(ui_out.get("heading_path") or "").strip())
        or selected_source == "authoritative"
    ):
            ui_out["heading_path"] = effective_heading_path

    current_summary_line = str(ui_out.get("summary_line") or "").strip()
    effective_summary_seed = _primary_ref_evidence_summary_seed(effective_primary)
    authoritative_summary_seed = _compact_reader_open_text(str(authoritative_summary_line or "").strip())
    if authoritative_summary_seed and _summary_line_needs_polish(
        prompt=prompt,
        title=display_name,
        summary_line=authoritative_summary_seed,
    ):
        authoritative_summary_seed = ""
    if (not authoritative_summary_seed) and authoritative_primary:
        authoritative_summary_seed = _primary_ref_evidence_summary_seed(authoritative_primary)
    authoritative_conflicts_with_synthesized = bool(
        selected_source == "authoritative"
        and authoritative_primary
        and synthesized_primary
        and (not _primary_ref_evidence_points_to_same_surface(authoritative_primary, synthesized_primary))
    )
    if authoritative_conflicts_with_synthesized and authoritative_summary_seed:
        ui_out["summary_line"] = authoritative_summary_seed
    if effective_summary_seed and (
        (not str(ui_out.get("summary_line") or "").strip())
        or (
            _summary_line_needs_polish(
                prompt=prompt,
                title=display_name,
                summary_line=str(ui_out.get("summary_line") or "").strip(),
            )
            and (not _summary_line_needs_polish(
                prompt=prompt,
                title=display_name,
                summary_line=effective_summary_seed,
            ))
        )
    ):
        ui_out["summary_line"] = effective_summary_seed

    if effective_primary:
        ui_out["primary_evidence"] = dict(effective_primary)
        ui_out["primary_evidence_heading_path"] = effective_heading_path
        effective_source = str(
            effective_primary.get("selection_reason")
            or ui_out.get("primary_evidence_source")
            or ("doc_list_authoritative" if selected_source == "authoritative" else "")
        ).strip()
        if effective_source:
            ui_out["primary_evidence_source"] = effective_source
    if authoritative_primary_evidence:
        ui_out["authoritative_primary_evidence"] = dict(
            _normalize_primary_ref_evidence_payload(
                authoritative_primary_evidence if isinstance(authoritative_primary_evidence, dict) else {}
            )
        )
        ui_out["primary_evidence_authority"] = "doc_list_authoritative"
    return ui_out, effective_primary


def _build_doc_list_ref_locs(*, heading_path: str, primary_evidence: dict) -> list[dict]:
    locs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _push(candidate: dict, *, source: str) -> None:
        if not isinstance(candidate, dict):
            return
        loc_heading = str(candidate.get("heading_path") or heading_path or "").strip()
        snippet = _compact_reader_open_text(
            str(candidate.get("highlight_snippet") or candidate.get("snippet") or "").strip()
        )
        if (not loc_heading) and (not snippet):
            return
        key = (loc_heading, snippet)
        if key in seen:
            return
        seen.add(key)
        loc = {
            "heading_path": loc_heading or None,
            "heading": _top_heading(loc_heading) or None,
            "snippet": snippet or None,
            "text": snippet or None,
            "quote": snippet or None,
            "quality": "high" if (loc_heading or snippet) else "medium",
            "source": source,
            "score": 96.0 - (len(locs) * 0.5),
        }
        locs.append({key: value for key, value in loc.items() if value not in (None, "", [], {})})

    _push(primary_evidence, source="doc_list_primary")
    for alt in list(primary_evidence.get("alternatives") or []):
        _push(alt if isinstance(alt, dict) else {}, source="doc_list_alternative")
        if len(locs) >= 4:
            break
    return locs


def _build_doc_list_ref_hit(*, raw_item: dict, idx: int) -> dict:
    source_path = str(raw_item.get("source_path") or "").strip()
    source_name = str(raw_item.get("source_name") or "").strip() or _source_filename(source_path) or f"Reference {idx}"
    primary_evidence = _normalize_primary_ref_evidence_payload(
        raw_item.get("primary_evidence") if isinstance(raw_item.get("primary_evidence"), dict) else {}
    )
    authoritative_summary_line = _compact_reader_open_text(str(raw_item.get("summary_line") or "").strip())
    heading_path = (
        str(raw_item.get("heading_path") or "").strip()
        or str(primary_evidence.get("heading_path") or "").strip()
    )
    section_label, subsection_label = _split_section_subsection(heading_path) if heading_path else ("", "")
    text_candidates = _collect_doc_list_ref_text_candidates(
        raw_item=raw_item,
        primary_evidence=primary_evidence,
    )
    anchor_kind = str(primary_evidence.get("anchor_kind") or "").strip().lower()
    anchor_number = _positive_int(primary_evidence.get("anchor_number"))
    rank_llm = max(72.0, 92.0 - float(max(0, idx - 1)) * 2.0)
    rank_bm25 = max(6.0, 9.4 - float(max(0, idx - 1)) * 0.4)
    meta = {
        "source_path": source_path,
        "source_name": source_name,
        "display_name": source_name,
        "ref_pack_state": "ready",
        "heading_path": heading_path,
        "top_heading": _top_heading(heading_path) or section_label or heading_path,
        "ref_best_heading_path": heading_path,
        "ref_section": section_label or _top_heading(heading_path) or "",
        "ref_subsection": subsection_label or "",
        "ref_loc_quality": "high" if heading_path else "medium",
        "ref_locs": _build_doc_list_ref_locs(
            heading_path=heading_path,
            primary_evidence=primary_evidence,
        ),
        "ref_show_snippets": list(text_candidates[:3]),
        "ref_snippets": list(text_candidates[:3]),
        "ref_overview_snippets": list(text_candidates[:2]),
        "explicit_doc_match_score": 12.0,
        "ref_rank": {
            "llm": rank_llm,
            "bm25": rank_bm25,
            "deep": 2.8,
            "term_bonus": 2.4,
            "semantic_score": 8.8,
            "score": rank_llm,
            "display_score": rank_llm,
        },
    }
    if anchor_kind:
        meta["anchor_target_kind"] = anchor_kind
    if anchor_number > 0:
        meta["anchor_target_number"] = anchor_number
        meta["anchor_match_score"] = 10.0
    if primary_evidence:
        meta["authoritative_primary_evidence"] = dict(primary_evidence)
    return {
        "text": str(text_candidates[0] if text_candidates else (source_name or source_path)).strip(),
        "meta": meta,
    }


def _build_doc_list_reader_open_payload(
    *,
    source_path: str,
    source_name: str,
    heading_path: str,
    summary_line: str,
    primary_evidence: dict,
    reader_open: dict | None,
) -> dict:
    primary = _normalize_primary_ref_evidence_payload(primary_evidence)
    out = dict(reader_open or {}) if isinstance(reader_open, dict) else {}
    if source_path:
        out["sourcePath"] = source_path
    if source_name:
        out["sourceName"] = source_name
    auth_heading = str(primary.get("heading_path") or heading_path or out.get("headingPath") or "").strip()
    auth_snippet = _compact_reader_open_text(
        str(primary.get("snippet") or out.get("snippet") or summary_line or "").strip()
    )
    auth_highlight = _compact_reader_open_text(
        str(primary.get("highlight_snippet") or auth_snippet or out.get("highlightSnippet") or "").strip()
    )
    if auth_heading:
        out["headingPath"] = auth_heading
    if auth_snippet:
        out["snippet"] = auth_snippet
    if auth_highlight:
        out["highlightSnippet"] = auth_highlight
    for src_key, dst_key in (
        ("block_id", "blockId"),
        ("anchor_id", "anchorId"),
        ("anchor_kind", "anchorKind"),
    ):
        value = str(primary.get(src_key) or "").strip()
        if value:
            out[dst_key] = value
    anchor_number = _positive_int(primary.get("anchor_number"))
    if anchor_number > 0:
        out["anchorNumber"] = anchor_number
    if "strict_locate" in primary:
        out["strictLocate"] = bool(primary.get("strict_locate"))
    if primary:
        out["primaryEvidence"] = dict(primary)
    return {
        key: value
        for key, value in out.items()
        if value not in (None, "", [], {})
    }


def _build_doc_list_hit_ui_meta(
    *,
    raw_item: dict,
    idx: int,
    prompt: str,
    allow_expensive_llm: bool,
    allow_exact_locate: bool,
) -> dict:
    source_path = str(raw_item.get("source_path") or "").strip()
    source_name = str(raw_item.get("source_name") or "").strip() or _source_filename(source_path) or f"Reference {idx}"
    primary_evidence = _normalize_primary_ref_evidence_payload(
        raw_item.get("primary_evidence") if isinstance(raw_item.get("primary_evidence"), dict) else {}
    )
    authoritative_summary_line = _compact_reader_open_text(str(raw_item.get("summary_line") or "").strip())
    heading_path = (
        str(raw_item.get("heading_path") or "").strip()
        or str(primary_evidence.get("heading_path") or "").strip()
    )
    hit = _build_doc_list_ref_hit(raw_item=raw_item, idx=idx)
    ui_meta = dict(
        build_hit_ui_meta(
            hit,
            prompt=prompt,
            pdf_root=None,
            lib_store=None,
            allow_expensive_llm=bool(allow_expensive_llm),
            allow_exact_locate=bool(allow_exact_locate),
        )
        or {}
    )
    if not str(ui_meta.get("display_name") or "").strip():
        ui_meta["display_name"] = source_name
    ui_meta, effective_primary_evidence = _apply_doc_list_effective_primary_evidence(
        prompt=prompt,
        display_name=str(ui_meta.get("display_name") or source_name),
        fallback_heading_path=heading_path,
        ui_meta=ui_meta,
        authoritative_primary_evidence=primary_evidence,
        authoritative_summary_line=authoritative_summary_line,
    )
    if not str(ui_meta.get("heading_path") or "").strip() and heading_path:
        ui_meta["heading_path"] = heading_path
    if not str(ui_meta.get("summary_line") or "").strip():
        summary_seed = _compact_reader_open_text(
            str(
                raw_item.get("summary_line")
                or _primary_ref_evidence_summary_seed(effective_primary_evidence)
                or primary_evidence.get("highlight_snippet")
                or primary_evidence.get("snippet")
                or ""
            ).strip()
        )
        if summary_seed:
            ui_meta["summary_line"] = summary_seed
    if _why_line_needs_polish(
        prompt=prompt,
        display_name=str(ui_meta.get("display_name") or source_name),
        heading_path=str(ui_meta.get("heading_path") or heading_path),
        summary_line=str(ui_meta.get("summary_line") or ""),
        why_line=str(ui_meta.get("why_line") or ""),
    ):
        fallback_why = _build_prompt_aligned_ref_why_line_v3(
            prompt=prompt,
            display_name=str(ui_meta.get("display_name") or source_name),
            heading_path=str(ui_meta.get("heading_path") or heading_path),
            summary_line=str(ui_meta.get("summary_line") or ""),
            why_line=str(ui_meta.get("why_line") or ""),
        )
        if not fallback_why:
            fallback_why = _doc_list_ref_why_line(
                prompt=prompt,
                heading_path=str(ui_meta.get("heading_path") or heading_path),
                prefer_zh=bool(_prefer_zh_ref_card_locale(prompt, source_name)),
            )
        if fallback_why:
            why_basis_meta = _build_ref_why_basis_meta(
                prompt=prompt,
                why_generation="deterministic_grounded",
                why_line=fallback_why,
            )
            ui_meta["why_line"] = fallback_why
            ui_meta["why_generation"] = str(why_basis_meta.get("why_generation") or "deterministic_grounded")
            ui_meta["why_basis"] = str(why_basis_meta.get("why_basis") or "")
    score = max(7.8, round(9.55 - (idx - 1) * 0.18, 2))
    ui_meta["score"] = score
    ui_meta["score_pending"] = False
    ui_meta["score_tier"] = _score_tier(score)
    ui_meta["source_path"] = source_path
    reader_open = _build_doc_list_reader_open_payload(
        source_path=source_path,
        source_name=source_name,
        heading_path=str(ui_meta.get("heading_path") or heading_path),
        summary_line=str(ui_meta.get("summary_line") or ""),
        primary_evidence=effective_primary_evidence or primary_evidence,
        reader_open=ui_meta.get("reader_open") if isinstance(ui_meta.get("reader_open"), dict) else {},
    )
    if reader_open:
        ui_meta["reader_open"] = reader_open
    if effective_primary_evidence:
        ui_meta["primary_evidence"] = dict(effective_primary_evidence)
        ui_meta["primary_evidence_heading_path"] = str(
            effective_primary_evidence.get("heading_path")
            or ui_meta.get("heading_path")
            or heading_path
            or ""
        ).strip()
    elif primary_evidence:
        ui_meta["primary_evidence"] = dict(primary_evidence)
        ui_meta["primary_evidence_heading_path"] = str(primary_evidence.get("heading_path") or heading_path or "").strip()
        ui_meta["primary_evidence_source"] = "doc_list_authoritative"
    topic_match_kind = str(raw_item.get("topic_match_kind") or "").strip().lower()
    if topic_match_kind:
        ui_meta["topic_match_kind"] = topic_match_kind
    return ui_meta


def _doc_list_topic_match_why_line(
    *,
    prompt: str,
    heading_path: str,
    match_kind: str,
) -> str:
    kind = str(match_kind or "").strip().lower()
    if not kind:
        return ""
    prefer_zh = bool(_prefer_zh_ref_card_locale(prompt, heading_path))
    loc = " / ".join(part for part in str(heading_path or "").split(" / ") if part).strip()
    zh_fallback_loc = "\u76f8\u5173\u6bb5\u843d"
    en_fallback_loc = "the matched section"
    if kind == "sci_related_predecessor":
        if prefer_zh:
            return "\u8be5\u6587\u8ba8\u8bba\u7684\u662f single-shot compressive spectral imaging\uff0c\u53ef\u4f5c\u4e3a\u4e0e SCI \u76f8\u5173\u7684\u65e9\u671f\u524d\u8eab\u5de5\u4f5c\uff0c\u4f46\u4e0d\u662f\u4e25\u683c\u7684 SCI \u672f\u8bed\u547d\u4e2d\u3002"
        return "This paper is better treated as an early related predecessor: it discusses single-shot compressive spectral imaging, which is SCI-adjacent rather than an exact SCI term match."
    if kind == "explicit_sci_mention":
        if prefer_zh:
            return f"\u8be5\u6587\u5728\u201c{loc or heading_path or zh_fallback_loc}\u201d\u5904\u660e\u786e\u63d0\u5230 Snapshot Compressive Imaging (SCI)\uff0c\u76f4\u63a5\u5bf9\u5e94\u8fd9\u7c7b SCI \u5b9a\u4f4d\u95ee\u9898\u3002"
        return f"The paper explicitly mentions Snapshot Compressive Imaging (SCI) in '{loc or heading_path or en_fallback_loc}', so it is a direct match for this SCI lookup."
    return ""


def _apply_doc_list_topic_match_hints(*, prompt: str, raw_item: dict, ui_meta: dict) -> dict:
    ui = dict(ui_meta or {})
    match_kind = str(raw_item.get("topic_match_kind") or ui.get("topic_match_kind") or "").strip().lower()
    if not match_kind:
        return ui
    ui["topic_match_kind"] = match_kind
    note = _doc_list_topic_match_why_line(
        prompt=prompt,
        heading_path=str(ui.get("heading_path") or raw_item.get("heading_path") or "").strip(),
        match_kind=match_kind,
    )
    current_why = str(ui.get("why_line") or "").strip()
    should_override = bool(
        note
        and (
            match_kind == "sci_related_predecessor"
            or (not current_why)
            or _why_line_needs_polish(
                prompt=prompt,
                display_name=str(ui.get("display_name") or raw_item.get("source_name") or "").strip(),
                heading_path=str(ui.get("heading_path") or raw_item.get("heading_path") or "").strip(),
                summary_line=str(ui.get("summary_line") or raw_item.get("summary_line") or "").strip(),
                why_line=current_why,
            )
            or (not _why_line_explicitly_names_focus_term(prompt, current_why))
        )
    )
    if should_override:
        why_basis_meta = _build_ref_why_basis_meta(
            prompt=prompt,
            why_generation="deterministic_grounded",
            why_line=note,
        )
        ui["why_line"] = note
        ui["why_generation"] = str(why_basis_meta.get("why_generation") or "deterministic_grounded")
        ui["why_basis"] = str(why_basis_meta.get("why_basis") or "")
    if match_kind == "sci_related_predecessor":
        fallback_summary = _compact_reader_open_text(str(raw_item.get("summary_line") or "").strip())
        current_summary = str(ui.get("summary_line") or "").strip()
        display_name = str(ui.get("display_name") or raw_item.get("source_name") or "").strip()
        if fallback_summary and (
            (not current_summary)
            or _summary_line_needs_polish(
                prompt=prompt,
                title=display_name,
                summary_line=current_summary,
            )
            or current_summary.lower().startswith("snapshot compressive imaging:")
            or _looks_like_title_echo(current_summary, display_name)
        ):
            summary_basis_meta = _build_ref_summary_basis_meta(
                prompt=prompt,
                summary_kind=str(ui.get("summary_kind") or "guide"),
                summary_generation="deterministic_grounded",
                summary_line=fallback_summary,
            )
            ui["summary_line"] = fallback_summary
            ui["summary_generation"] = str(summary_basis_meta.get("summary_generation") or "deterministic_grounded")
            ui["summary_basis"] = str(summary_basis_meta.get("summary_basis") or "")
    return ui


def _filter_doc_list_rows_for_guide(
    *,
    doc_rows: list[dict] | None,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
) -> tuple[list[dict], int]:
    rows = [dict(item) for item in list(doc_rows or []) if isinstance(item, dict)]
    guide_path = str(guide_source_path or "").strip()
    guide_name = str(guide_source_name or "").strip()
    guide_active = bool(guide_mode and (guide_path or guide_name))
    if not guide_active:
        return rows, 0
    out: list[dict] = []
    filtered_self = 0
    for raw_item in rows:
        source_path = str(raw_item.get("source_path") or "").strip()
        source_name = str(raw_item.get("source_name") or "").strip() or _source_filename(source_path)
        if _hit_matches_guide_source(
            {
                "source_path": source_path,
                "source_name": source_name,
                "display_name": source_name,
            },
            guide_source_path=guide_path,
            guide_source_name=guide_name,
        ):
            filtered_self += 1
            continue
        out.append(raw_item)
    return out, filtered_self


def build_doc_list_refs_payload(
    *,
    user_msg_id: int | str,
    pack: dict | None,
    doc_list: list[dict] | None,
    allow_expensive_llm: bool = False,
    allow_exact_locate: bool = True,
    apply_copy_polish: bool = True,
    guide_mode: bool = False,
    guide_source_path: str = "",
    guide_source_name: str = "",
) -> dict:
    pack_src = dict(pack or {}) if isinstance(pack, dict) else {}
    prompt = str(pack_src.get("prompt") or "").strip()
    guide_source_path_norm = str(guide_source_path or "").strip()
    guide_source_name_norm = str(guide_source_name or "").strip()
    guide_active = bool(guide_mode and (guide_source_path_norm or guide_source_name_norm))
    prompt_cross_paper_refs = bool(_prompt_likely_cross_paper_refs(prompt))
    doc_rows_all = [dict(item) for item in list(doc_list or []) if isinstance(item, dict)]
    doc_rows, filtered_self_doc_count = _filter_doc_list_rows_for_guide(
        doc_rows=doc_rows_all,
        guide_mode=guide_active,
        guide_source_path=guide_source_path_norm,
        guide_source_name=guide_source_name_norm,
    )
    if doc_rows_all:
        hits: list[dict] = []
        for idx, raw_item in enumerate(doc_rows, start=1):
            source_path = str(raw_item.get("source_path") or "").strip()
            if not source_path:
                continue
            ui_meta = _build_doc_list_hit_ui_meta(
                raw_item=raw_item,
                idx=idx,
                prompt=prompt,
                allow_expensive_llm=bool(allow_expensive_llm),
                allow_exact_locate=bool(allow_exact_locate),
            )
            ui_meta = _normalize_ref_copy_ui_meta(ui_meta)
            ui_meta = _apply_doc_list_topic_match_hints(
                prompt=prompt,
                raw_item=raw_item,
                ui_meta=ui_meta,
            )
            hits.append(
                {
                    "text": str(ui_meta.get("summary_line") or ui_meta.get("why_line") or source_path).strip(),
                    "meta": {
                        "source_path": source_path,
                        "ref_pack_state": "ready",
                        "ref_best_heading_path": str(ui_meta.get("heading_path") or "").strip(),
                    },
                    "ui_meta": ui_meta,
                }
            )
        if apply_copy_polish and hits:
            polished_hits: list[dict] = []
            for hit in hits:
                ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
                if not isinstance(ui_meta, dict):
                    polished_hits.append(hit)
                    continue
                hit2 = dict(hit)
                hit2["ui_meta"] = _normalize_ref_copy_ui_meta(
                    _maybe_polish_single_ref_hit_card(
                        prompt=prompt,
                        hit=hit,
                        ui_meta=ui_meta,
                        allow_expensive_llm=bool(allow_expensive_llm),
                    )
                )
                hit2["ui_meta"] = _apply_doc_list_topic_match_hints(
                    prompt=prompt,
                    raw_item=doc_rows[len(polished_hits)],
                    ui_meta=hit2.get("ui_meta") if isinstance(hit2.get("ui_meta"), dict) else {},
                )
                polished_hits.append(hit2)
            hits = polished_hits
        pack_out = dict(pack_src)
        pack_out["user_msg_id"] = int(user_msg_id) if str(user_msg_id).isdigit() else user_msg_id
        pack_out["hits"] = hits
        pipeline_debug = dict(pack_out.get("pipeline_debug") or {}) if isinstance(pack_out.get("pipeline_debug"), dict) else {}
        pipeline_debug["doc_list_authoritative"] = True
        pipeline_debug["guide_active"] = bool(guide_active)
        pipeline_debug["final_hit_count"] = int(len(hits))
        pipeline_debug["raw_hit_count"] = int(len(hits))
        pipeline_debug["post_score_gate_hit_count"] = int(len(hits))
        pipeline_debug["post_focus_filter_hit_count"] = int(len(hits))
        pipeline_debug["post_llm_filter_hit_count"] = int(len(hits))
        pipeline_debug["filtered_self_hit_count"] = int(filtered_self_doc_count)
        pipeline_debug["prompt_likely_cross_paper_refs"] = bool(prompt_cross_paper_refs)
        pack_out["pipeline_debug"] = pipeline_debug
        if guide_active:
            hidden_self_source = bool(filtered_self_doc_count > 0)
            if (not hidden_self_source) and prompt_cross_paper_refs:
                hidden_self_source = True
            pack_out["guide_filter"] = {
                "active": True,
                "hidden_self_source": hidden_self_source,
                "filtered_hit_count": int(filtered_self_doc_count),
                "guide_source_path": guide_source_path_norm,
                "guide_source_name": guide_source_name_norm or _source_filename(guide_source_path_norm),
            }
        pack_out["payload_mode"] = "full"
        return _attach_pack_display_contract(pack_out)
    prefer_zh = bool(_prefer_zh_ref_card_locale(prompt))
    hits: list[dict] = []
    for idx, raw_item in enumerate(list(doc_list or []), start=1):
        if not isinstance(raw_item, dict):
            continue
        source_path = str(raw_item.get("source_path") or "").strip()
        if not source_path:
            continue
        source_name = str(raw_item.get("source_name") or "").strip() or _source_filename(source_path) or f"Reference {idx}"
        heading_path = str(raw_item.get("heading_path") or "").strip()
        primary_evidence = _normalize_primary_ref_evidence_payload(
            raw_item.get("primary_evidence") if isinstance(raw_item.get("primary_evidence"), dict) else {}
        )
        summary_line = _compact_reader_open_text(
            str(
                raw_item.get("summary_line")
                or primary_evidence.get("highlight_snippet")
                or primary_evidence.get("snippet")
                or ""
            ).strip()
        )
        why_line = _doc_list_ref_why_line(
            prompt=prompt,
            heading_path=heading_path or str(primary_evidence.get("heading_path") or "").strip(),
            prefer_zh=prefer_zh,
        )
        reader_open = {
            "sourcePath": source_path,
            "sourceName": source_name,
            "headingPath": heading_path or str(primary_evidence.get("heading_path") or "").strip() or None,
            "snippet": summary_line or None,
            "highlightSnippet": summary_line or None,
            "strictLocate": bool(primary_evidence.get("strict_locate")),
            "blockId": str(primary_evidence.get("block_id") or "").strip() or None,
            "anchorId": str(primary_evidence.get("anchor_id") or "").strip() or None,
        }
        if primary_evidence:
            reader_open["primaryEvidence"] = dict(primary_evidence)
        score = max(6.6, round(9.6 - (idx - 1) * 0.18, 2))
        ui_meta = {
            "display_name": source_name,
            "heading_path": heading_path,
            "score": score,
            "score_pending": False,
            "score_tier": _score_tier(score),
            "summary_line": summary_line,
            "summary_kind": "guide",
            "summary_label": "导读" if prefer_zh else "Guide",
            "summary_title": "命中章节讲什么 / 提供什么" if prefer_zh else "What This Matched Section Covers",
            "summary_generation": "doc_list_contract",
            "summary_basis": "基于共享多篇文献列表 contract 的展示摘要" if prefer_zh else "Display summary sourced from the shared multi-paper document list contract",
            "why_line": why_line,
            "why_generation": "doc_list_contract",
            "why_basis": "基于共享多篇文献列表 contract 的保留理由" if prefer_zh else "Retention reason sourced from the shared multi-paper document list contract",
            "semantic_badges": [],
            "can_open": True,
            "citation_meta": {},
            "source_path": source_path,
            "reader_open": {k: v for k, v in reader_open.items() if v not in (None, "", [], {})},
        }
        if primary_evidence:
            ui_meta["primary_evidence"] = dict(primary_evidence)
            if not str(ui_meta.get("heading_path") or "").strip():
                ui_meta["heading_path"] = str(primary_evidence.get("heading_path") or "").strip()
        hits.append(
            {
                "text": summary_line or why_line,
                "meta": {
                    "source_path": source_path,
                    "ref_pack_state": "ready",
                    "ref_best_heading_path": str(ui_meta.get("heading_path") or "").strip(),
                },
                "ui_meta": ui_meta,
            }
        )

    pack_out = dict(pack_src)
    pack_out["user_msg_id"] = int(user_msg_id) if str(user_msg_id).isdigit() else user_msg_id
    pack_out["hits"] = hits
    pipeline_debug = dict(pack_out.get("pipeline_debug") or {}) if isinstance(pack_out.get("pipeline_debug"), dict) else {}
    pipeline_debug["doc_list_authoritative"] = True
    pipeline_debug["guide_active"] = bool(guide_active)
    pipeline_debug["final_hit_count"] = int(len(hits))
    if "raw_hit_count" not in pipeline_debug:
        pipeline_debug["raw_hit_count"] = int(len(hits))
    if "post_score_gate_hit_count" not in pipeline_debug:
        pipeline_debug["post_score_gate_hit_count"] = int(len(hits))
    if "post_focus_filter_hit_count" not in pipeline_debug:
        pipeline_debug["post_focus_filter_hit_count"] = int(len(hits))
    if "post_llm_filter_hit_count" not in pipeline_debug:
        pipeline_debug["post_llm_filter_hit_count"] = int(len(hits))
    if "filtered_self_hit_count" not in pipeline_debug:
        pipeline_debug["filtered_self_hit_count"] = 0
    pipeline_debug["prompt_likely_cross_paper_refs"] = bool(prompt_cross_paper_refs)
    pack_out["pipeline_debug"] = pipeline_debug
    if guide_active:
        hidden_self_source = bool(prompt_cross_paper_refs)
        pack_out["guide_filter"] = {
            "active": True,
            "hidden_self_source": hidden_self_source,
            "filtered_hit_count": 0,
            "guide_source_path": guide_source_path_norm,
            "guide_source_name": guide_source_name_norm or _source_filename(guide_source_path_norm),
        }
    pack_out["payload_mode"] = "full"
    return _attach_pack_display_contract(pack_out)


def _resolve_ref_ui_heading_context(
    *,
    prompt: str,
    source_path: str,
    heading_path: str,
    heading_fallback: str = "",
    section_label: str = "",
    subsection_label: str = "",
) -> dict[str, str]:
    heading_path_norm = _sanitize_heading_path_ui(
        str(heading_path or "").strip(),
        prompt=prompt,
        source_path=source_path,
    )
    heading = str(
        heading_fallback
        or _top_heading(heading_path_norm)
        or ""
    ).strip()
    if heading and _is_non_navigational_heading_ui(heading, prompt=prompt, source_path=source_path):
        heading = ""
    if heading and _looks_like_doc_title_heading_ui(heading, source_path):
        heading = ""

    section = str(section_label or "").strip()
    subsection = str(subsection_label or "").strip()
    if section and _is_non_navigational_heading_ui(section, prompt=prompt, source_path=source_path):
        section = ""
    if subsection and _is_non_navigational_heading_ui(subsection, prompt=prompt, source_path=source_path):
        subsection = ""
    if (not section) and heading_path_norm:
        section, subsection = _split_section_subsection(heading_path_norm)
    if section and _looks_like_doc_title_heading_ui(section, source_path):
        section = ""
        subsection = ""

    return {
        "heading_path": heading_path_norm,
        "heading": heading,
        "section_label": section,
        "subsection_label": subsection,
    }


def _should_allow_ref_summary_block_rescue(
    *,
    prompt: str,
    source_path: str,
    ref_pack_state: str,
    allow_exact_locate: bool,
) -> bool:
    if not str(source_path or "").strip():
        return False
    if allow_exact_locate:
        return True
    if str(ref_pack_state or "").strip().lower() != "pending":
        return False
    return bool(_prompt_requires_explicit_focus_match(prompt))


def _select_primary_ref_evidence(
    *,
    meta: dict,
    prompt: str,
    source_path: str,
    display_name: str,
    citation_meta: dict | None,
    heading_context: dict[str, str],
    anchor_target_kind: str,
    anchor_target_number: int,
    allow_exact_locate: bool,
    allow_summary_block_rescue: bool = False,
    allow_llm_translate: bool = True,
) -> dict[str, object]:
    heading_path = str((heading_context or {}).get("heading_path") or "").strip()
    heading = str((heading_context or {}).get("heading") or "").strip()
    section_label = str((heading_context or {}).get("section_label") or "").strip()
    subsection_label = str((heading_context or {}).get("subsection_label") or "").strip()

    nav = _build_ref_navigation(meta, prompt=prompt, heading_fallback=heading)
    used_nav_summary = bool(str(nav.get("summary_line") or nav.get("what") or "").strip())
    summary_line = str(nav.get("summary_line") or nav.get("what") or "").strip()
    if not summary_line:
        summary_line = _fallback_ref_ui_summary_line(
            meta,
            prompt=prompt,
            citation_meta=citation_meta,
            allow_llm_translate=allow_llm_translate,
        )

    used_prompt_aligned_summary = False
    summary_source = "navigation" if used_nav_summary else ("fallback" if summary_line else "")
    selected_heading_path = heading_path

    meta_prompt_aligned_candidate = _choose_prompt_aligned_ref_summary_candidate(
        meta,
        prompt=prompt,
        source_path=source_path,
        citation_meta=citation_meta,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        allow_llm_translate=allow_llm_translate,
    )
    block_prompt_aligned_candidate: dict = {}
    if allow_summary_block_rescue and source_path:
        needs_block_rescue = bool(
            (not meta_prompt_aligned_candidate)
            or (not summary_line)
            or (
                summary_source == "fallback"
                and _looks_focus_prefixed_ref_summary(prompt, summary_line)
            )
            or _summary_line_needs_polish(
                prompt=prompt,
                title=display_name,
                summary_line=summary_line,
            )
        )
        if needs_block_rescue:
            block_prompt_aligned_candidate = _choose_prompt_aligned_ref_summary_candidate_from_source_blocks(
                prompt=prompt,
                source_path=source_path,
                title=str((citation_meta or {}).get("title") or (meta or {}).get("title") or "").strip(),
                anchor_target_kind=anchor_target_kind,
                anchor_target_number=anchor_target_number,
                allow_llm_translate=allow_llm_translate,
            )
    prompt_aligned_candidate = _pick_best_prompt_aligned_ref_summary_candidate(
        [meta_prompt_aligned_candidate, block_prompt_aligned_candidate],
        prompt=prompt,
        source_path=source_path,
        title=str((citation_meta or {}).get("title") or (meta or {}).get("title") or "").strip(),
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
    )
    prompt_aligned_summary = str((prompt_aligned_candidate or {}).get("summary") or "").strip()
    if prompt_aligned_summary:
        candidate_heading_path = _sanitize_heading_path_ui(
            str((prompt_aligned_candidate or {}).get("heading_path") or "").strip(),
            prompt=prompt,
            source_path=source_path,
        )
        if (not candidate_heading_path) and allow_summary_block_rescue:
            candidate_heading_path = _infer_heading_path_for_summary_from_source_blocks(
                prompt=prompt,
                source_path=source_path,
                summary_line=prompt_aligned_summary,
                anchor_target_kind=anchor_target_kind,
                anchor_target_number=anchor_target_number,
            )
        current_unacceptable = bool(
            summary_line
            and _summary_line_needs_polish(
                prompt=prompt,
                title=display_name,
                summary_line=summary_line,
            )
        )
        current_score = _ref_summary_focus_score(
            prompt=prompt,
            source_path=source_path,
            title=str((citation_meta or {}).get("title") or (meta or {}).get("title") or "").strip(),
            text=summary_line,
            anchor_target_kind=anchor_target_kind,
            anchor_target_number=anchor_target_number,
        ) if summary_line else -1000.0
        chosen_score = _ref_summary_focus_score(
            prompt=prompt,
            source_path=source_path,
            title=str((citation_meta or {}).get("title") or (meta or {}).get("title") or "").strip(),
            text=prompt_aligned_summary,
            anchor_target_kind=anchor_target_kind,
            anchor_target_number=anchor_target_number,
        )
        fallback_focus_hits = len(_matched_focus_terms_for_ref_card(prompt, surface_text=summary_line))
        prompt_aligned_focus_hits = len(_matched_focus_terms_for_ref_card(prompt, surface_text=prompt_aligned_summary))
        prefer_prompt_aligned_heading = bool(
            candidate_heading_path
            and candidate_heading_path != heading_path
            and summary_source == "fallback"
            and prompt_aligned_focus_hits >= max(1, fallback_focus_hits)
            and chosen_score >= (current_score - 0.25)
        )
        should_rebind_prompt_aligned_heading = bool(
            candidate_heading_path
            and candidate_heading_path != heading_path
            and _ref_summary_surfaces_match(summary_line, prompt_aligned_summary)
        )
        if (
            (not summary_line)
            or current_unacceptable
            or (chosen_score >= (current_score + 0.75))
            or prefer_prompt_aligned_heading
        ):
            summary_line = prompt_aligned_summary
            used_prompt_aligned_summary = True
            summary_source = (
                "prompt_aligned_block"
                if str((prompt_aligned_candidate or {}).get("source_kind") or "").strip().lower() == "source_block"
                else "prompt_aligned"
            )
            should_rebind_prompt_aligned_heading = bool(
                candidate_heading_path
                and candidate_heading_path != heading_path
            )
        if should_rebind_prompt_aligned_heading:
            selected_heading_path = candidate_heading_path

    selected_section_label = section_label
    selected_subsection_label = subsection_label
    if selected_heading_path and heading_path and selected_heading_path != heading_path:
        selected_section_label = ""
        selected_subsection_label = ""
    resolved_heading_context = _resolve_ref_ui_heading_context(
        prompt=prompt,
        source_path=source_path,
        heading_path=selected_heading_path,
        heading_fallback=str(meta.get("top_heading") or _top_heading(str(meta.get("heading_path") or "")) or "").strip(),
        section_label=selected_section_label,
        subsection_label=selected_subsection_label,
    )

    return {
        "nav": nav,
        "summary_line": summary_line,
        "summary_source": summary_source,
        "used_nav_summary": used_nav_summary,
        "used_prompt_aligned_summary": used_prompt_aligned_summary,
        "heading_path": str(resolved_heading_context.get("heading_path") or "").strip(),
        "heading": str(resolved_heading_context.get("heading") or "").strip(),
        "section_label": str(resolved_heading_context.get("section_label") or "").strip(),
        "subsection_label": str(resolved_heading_context.get("subsection_label") or "").strip(),
    }


def build_hit_ui_meta(
    hit: dict,
    *,
    prompt: str,
    pdf_root: Path | None,
    lib_store: LibraryStore | None,
    preloaded_citation_meta: dict[str, dict] | None = None,
    allow_expensive_llm: bool = True,
    allow_exact_locate: bool = True,
) -> dict:
    meta = (hit or {}).get("meta", {}) or {}
    source_path = str(meta.get("source_path") or "").strip()
    ref_pack_state = str(meta.get("ref_pack_state") or "").strip().lower()
    heading_context = _resolve_ref_ui_heading_context(
        prompt=prompt,
        source_path=source_path,
        heading_path=str(meta.get("ref_best_heading_path") or meta.get("heading_path") or "").strip(),
        heading_fallback=str(
            meta.get("top_heading")
            or _top_heading(str(meta.get("heading_path") or ""))
            or ""
        ).strip(),
        section_label=str(meta.get("ref_section") or "").strip(),
        subsection_label=str(meta.get("ref_subsection") or "").strip(),
    )
    heading_path = str(heading_context.get("heading_path") or "").strip()
    heading = str(heading_context.get("heading") or "").strip()
    section_label = str(heading_context.get("section_label") or "").strip()
    subsection_label = str(heading_context.get("subsection_label") or "").strip()

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

    primary_evidence = _select_primary_ref_evidence(
        meta=meta,
        prompt=prompt,
        source_path=source_path,
        display_name=display_name,
        citation_meta=citation_meta,
        heading_context=heading_context,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        allow_exact_locate=allow_exact_locate,
        allow_summary_block_rescue=_should_allow_ref_summary_block_rescue(
            prompt=prompt,
            source_path=source_path,
            ref_pack_state=ref_pack_state,
            allow_exact_locate=allow_exact_locate,
        ),
        allow_llm_translate=bool(allow_expensive_llm),
    )
    nav = dict(primary_evidence.get("nav") or {}) if isinstance(primary_evidence.get("nav"), dict) else {}
    used_nav_summary = bool(primary_evidence.get("used_nav_summary"))
    used_prompt_aligned_summary = bool(primary_evidence.get("used_prompt_aligned_summary"))
    summary_line = str(primary_evidence.get("summary_line") or "").strip()
    heading_path = str(primary_evidence.get("heading_path") or heading_path).strip()
    heading = str(primary_evidence.get("heading") or heading).strip()
    section_label = str(primary_evidence.get("section_label") or "").strip()
    subsection_label = str(primary_evidence.get("subsection_label") or "").strip()
    summary_source = str(primary_evidence.get("summary_source") or "").strip()
    why_line = str(nav.get("why") or "").strip()
    why_generation = "navigation" if why_line else ""
    if not why_line:
        why_line = _fallback_why_line_ui(
            prompt=prompt,
            heading_label=heading_path or heading,
            section_label=section_label,
            subsection_label=subsection_label,
            find_terms=list(nav.get("find") or []),
        )
        why_generation = "deterministic_grounded" if why_line else "fallback"
    prompt_aligned_why = _build_prompt_aligned_ref_why_line_v3(
        prompt=prompt,
        display_name=display_name,
        heading_path=heading_path or heading,
        summary_line=summary_line,
        why_line=why_line,
    )
    why_focus_matches = _matched_focus_terms_for_ref_card(prompt, surface_text=why_line)
    aligned_why_matches = _matched_focus_terms_for_ref_card(prompt, surface_text=prompt_aligned_why)
    explicit_definition_focus_missing = bool(
        _is_definition_focus_prompt(prompt)
        and why_line
        and (not _why_line_explicitly_names_focus_term(prompt, why_line))
        and _why_line_explicitly_names_focus_term(prompt, prompt_aligned_why)
    )
    if prompt_aligned_why and aligned_why_matches and (
        (not why_line)
        or (not why_focus_matches)
        or why_generation == "navigation"
        or explicit_definition_focus_missing
    ):
        why_line = prompt_aligned_why
        why_generation = "deterministic_grounded"
    reader_open = _build_refs_reader_open_payload(
        meta=meta,
        prompt=prompt,
        source_path=source_path,
        display_name=display_name,
        heading_path=heading_path,
        heading=heading,
        summary_line=summary_line,
        why_line=why_line,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        allow_llm_disambiguation=allow_expensive_llm,
        allow_exact_locate=allow_exact_locate,
    )
    primary_evidence = _build_primary_ref_evidence_payload(
        source_path=source_path,
        display_name=display_name,
        reader_open=reader_open if isinstance(reader_open, dict) else {},
        selection_reason=summary_source,
        score=score,
    )
    if isinstance(reader_open, dict) and primary_evidence:
        reader_open = dict(reader_open)
        reader_open["primaryEvidence"] = dict(primary_evidence)
    summary_kind = _infer_ref_summary_kind(
        summary_line=summary_line,
        citation_meta=citation_meta if isinstance(citation_meta, dict) else {},
        used_prompt_aligned_summary=used_prompt_aligned_summary,
        used_nav_summary=used_nav_summary,
    )
    summary_surface = _build_ref_summary_surface_meta(
        prompt=prompt,
        summary_kind=summary_kind,
        summary_line=summary_line,
    )
    summary_generation = ""
    if summary_kind == "abstract":
        summary_generation = str((citation_meta or {}).get("summary_generation") or "").strip().lower() or "translated_abstract"
    elif summary_kind == "metadata":
        summary_generation = "metadata_only"
    else:
        summary_generation = "section_grounded"
    summary_basis_meta = _build_ref_summary_basis_meta(
        prompt=prompt,
        summary_kind=summary_kind,
        summary_generation=summary_generation,
        summary_line=summary_line,
    )
    why_basis_meta = _build_ref_why_basis_meta(
        prompt=prompt,
        why_generation=why_generation,
        why_line=why_line,
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
        "summary_kind": str(summary_surface.get("summary_kind") or summary_kind),
        "summary_label": str(summary_surface.get("summary_label") or ""),
        "summary_title": str(summary_surface.get("summary_title") or ""),
        "summary_generation": str(summary_basis_meta.get("summary_generation") or summary_generation),
        "summary_basis": str(summary_basis_meta.get("summary_basis") or ""),
        "primary_evidence_source": summary_source,
        "primary_evidence_heading_path": heading_path or heading,
        "primary_evidence": primary_evidence if isinstance(primary_evidence, dict) else {},
        "why_line": why_line,
        "why_generation": str(why_basis_meta.get("why_generation") or why_generation),
        "why_basis": str(why_basis_meta.get("why_basis") or ""),
        "anchor_target_kind": anchor_target_kind,
        "anchor_target_number": anchor_target_number,
        "anchor_match_score": anchor_match_score,
        "explicit_doc_match_score": explicit_doc_match_score,
        "semantic_badges": semantic_badges,
        "can_open": bool(pdf_path),
        "citation_meta": citation_meta if isinstance(citation_meta, dict) else {},
        "source_path": source_path,
        "reader_open": reader_open,
    }


def _refs_hit_rerank_llm_enabled() -> bool:
    raw_flag = str(os.environ.get("KB_REFS_RERANK_USE_LLM", "1") or "").strip().lower()
    if raw_flag in {"0", "false", "off", "no"}:
        return False
    try:
        settings = load_settings()
    except Exception:
        return False
    return bool(getattr(settings, "api_key", None))


def _refs_hit_display_score(hit: dict) -> float:
    ui_meta = (hit or {}).get("ui_meta") if isinstance((hit or {}).get("ui_meta"), dict) else {}
    raw_score = (ui_meta or {}).get("score")
    try:
        return float(raw_score)
    except Exception:
        pass
    fallback_score, _pending = _effective_ui_score(hit if isinstance(hit, dict) else {})
    try:
        return float(fallback_score or 0.0)
    except Exception:
        return 0.0


_PROMPT_FOCUS_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "using", "about", "where", "which", "what",
    "that", "this", "these", "those", "paper", "papers", "library", "source", "sources",
    "section", "please", "point", "directly", "most", "does", "do", "did", "discuss", "discusses",
    "mentioned", "mention", "other", "besides", "find", "show", "explain",
}

_PROMPT_FOCUS_GENERIC_MODIFIERS = {
    "dynamic", "compressive", "physics", "physical", "single", "high", "low",
    "based", "guided", "driven", "general", "specific", "direct", "directly",
}

_PROMPT_FOCUS_PHRASE_PATTERNS = (
    re.compile(
        r"\bwhere\s+(?:in\s+the\s+[^?.!,]{1,80}\s+)?is\s+(.+?)\s+(?:discussed|mentioned|defined|introduced)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:which|what)\s+(?:other\s+)?papers?[^?.!]{0,120}?\b(?:discuss(?:es|ed)?|mention(?:s|ed)?|cover(?:s|ed)?|address(?:es|ed)?|describe(?:s|d)?|use(?:s|d)?|introduce(?:s|d)?|define(?:s|d)?|compare(?:s|d)?)\s+(.+?)(?:[?.!]|$)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bbesides\s+this\s+paper[^?.!]{0,120}?\b(?:discuss(?:es|ed)?|mention(?:s|ed)?|cover(?:s|ed)?|address(?:es|ed)?|describe(?:s|d)?|use(?:s|d)?|introduce(?:s|d)?|define(?:s|d)?|compare(?:s|d)?)\s+(.+?)(?:[?.!]|$)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:which|what)\s+papers?[^?.!]{0,120}?\b(?:directly\s+|most\s+directly\s+)?(?:compare(?:s|d)?|define(?:s|d)?)\s+(.+?)(?:[?.!]|$)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bbesides\s+this\s+paper[^?.!]{0,120}?\b(?:directly\s+|most\s+directly\s+)?(?:compare(?:s|d)?|define(?:s|d)?)\s+(.+?)(?:[?.!]|$)",
        flags=re.IGNORECASE,
    ),
)


def _clean_refs_focus_phrase(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = re.sub(
        r"\b(?:please\s+point\s+me(?:\s+to)?|point\s+me(?:\s+to)?|show\s+me|source\s+section(?:s)?|those\s+sources|source\s+too)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = text.strip(" \t\r\n\"'“”‘’.,;:!?()[]{}")
    return text


def _looks_informative_focus_phrase(raw: str) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    tokens = [tok for tok in _normalize_title_identity(text).split() if tok and tok not in _PROMPT_FOCUS_STOPWORDS]
    if not tokens:
        return False
    if len(tokens) >= 2:
        return True
    token = tokens[0]
    return bool(len(token) >= 4 and (any(ch.isdigit() for ch in token) or "-" in token or token.isupper()))


def _extract_prompt_focus_phrases(prompt: str) -> tuple[str, ...]:
    text = str(prompt or "").strip()
    if not text:
        return ()
    out: list[str] = []
    seen: set[str] = set()

    def _push(raw: str) -> None:
        cleaned = _clean_refs_focus_phrase(raw)
        if not _looks_informative_focus_phrase(cleaned):
            return
        norm = _normalize_title_identity(cleaned)
        if len(norm) < 3 or norm in seen:
            return
        seen.add(norm)
        out.append(norm)

    for pattern in _PROMPT_FOCUS_PHRASE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        raw = str(m.group(1) or "")
        _push(raw)
        if _prompt_requests_compare(text):
            for part in re.split(r"\b(?:and|vs\.?|versus)\b", raw, flags=re.IGNORECASE):
                _push(part)
    return tuple(out[:4])


def _prune_redundant_focus_terms(terms: list[str]) -> tuple[str, ...]:
    items = [str(term or "").strip() for term in terms if str(term or "").strip()]
    out: list[str] = []
    for term in items:
        if any(
            term != other
            and len(other) > len(term)
            and term in other
            and (not re.search(r"\b(?:and|vs\.?|versus)\b", other, flags=re.IGNORECASE))
            for other in items
        ):
            continue
        out.append(term)
    return tuple(out[:8])


def _surface_has_focus_token_sequence(surface_tokens: list[str], term_tokens: list[str]) -> bool:
    if (not surface_tokens) or (not term_tokens) or (len(term_tokens) > len(surface_tokens)):
        return False
    width = len(term_tokens)
    for idx in range(len(surface_tokens) - width + 1):
        if surface_tokens[idx : idx + width] == term_tokens:
            return True
    return False


def _focus_term_adjacent_bigram_hits(surface: str, term_tokens: list[str]) -> int:
    if (not surface) or len(term_tokens) < 2:
        return 0
    hits = 0
    for idx in range(len(term_tokens) - 1):
        phrase = f"{term_tokens[idx]} {term_tokens[idx + 1]}".strip()
        if phrase and re.search(rf"\b{re.escape(phrase)}\b", surface, flags=re.I):
            hits += 1
    return hits


def _focus_term_single_distinctive_token_fallback(term_tokens: list[str], surface_tokens: set[str]) -> bool:
    if len(term_tokens) != 2 or (not surface_tokens):
        return False
    overlap = [tok for tok in term_tokens if tok in surface_tokens]
    if len(overlap) != 1:
        return False
    matched = overlap[0]
    unmatched = term_tokens[0] if matched == term_tokens[1] else term_tokens[1]
    if len(matched) < 10:
        return False
    if matched in _PROMPT_FOCUS_GENERIC_MODIFIERS:
        return False
    return unmatched in _PROMPT_FOCUS_GENERIC_MODIFIERS


def _focus_term_matches_surface(term: str, surface_text: str) -> bool:
    norm_term = _normalize_title_identity(term)
    surface = _normalize_title_identity(surface_text)
    if not norm_term or not surface:
        return False
    if re.search(rf"\b{re.escape(norm_term)}\b", surface, flags=re.I):
        return True
    term_tokens = [
        tok for tok in norm_term.split()
        if tok and tok not in _PROMPT_FOCUS_STOPWORDS and len(tok) >= 4
    ]
    if not term_tokens:
        return False
    surface_tokens = [tok for tok in surface.split() if tok]
    if not surface_tokens:
        return False
    surface_token_set = set(surface_tokens)
    if len(term_tokens) == 1:
        return bool(term_tokens[0] in surface_token_set)
    if len(term_tokens) == 2:
        if _surface_has_focus_token_sequence(surface_tokens, term_tokens):
            return True
        return _focus_term_single_distinctive_token_fallback(term_tokens, surface_token_set)
    if not all(tok in surface_token_set for tok in term_tokens):
        return False
    if _surface_has_focus_token_sequence(surface_tokens, term_tokens):
        return True
    return _focus_term_adjacent_bigram_hits(surface, term_tokens) > 0


def _refs_exact_focus_match_count(prompt: str, surface_text: str) -> int:
    surface = _normalize_title_identity(surface_text)
    if not surface:
        return 0
    count = 0
    for term in _refs_prompt_focus_terms(prompt):
        norm_term = _normalize_title_identity(term)
        if norm_term and re.search(rf"\b{re.escape(norm_term)}\b", surface, flags=re.I):
            count += 1
    return count


@lru_cache(maxsize=512)
def _refs_prompt_focus_terms(prompt: str) -> tuple[str, ...]:
    text = str(prompt or "").strip()
    if not text:
        return ()
    out: list[str] = []
    seen: set[str] = set()

    def _push(raw: str) -> None:
        cleaned = _clean_refs_focus_phrase(raw)
        if not cleaned:
            return
        norm = _normalize_title_identity(cleaned)
        if len(norm) < 3 or norm in seen:
            return
        seen.add(norm)
        out.append(norm)

    prompt_targets_sci = bool(_shared_prompt_targets_sci_topic(text))
    if prompt_targets_sci:
        _push("Snapshot Compressive Imaging")
        _push("SCI")
    topic = _shared_extract_multi_paper_topic(text)
    if topic and (not prompt_targets_sci):
        _push(topic)

    for quoted in re.findall(r"[\"']([^\"']{2,80})[\"']", text):
        _push(quoted)
    for token in re.findall(r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9_-]{1,40}(?![A-Za-z0-9_-])", text):
        raw = str(token or "").strip()
        low = raw.lower()
        if low in _PROMPT_FOCUS_STOPWORDS:
            continue
        has_case_signal = any(ch.isupper() for ch in raw[1:]) or raw.isupper() or any(ch.isdigit() for ch in raw) or ("-" in raw)
        if not has_case_signal:
            continue
        _push(raw)
    for phrase in _extract_prompt_focus_phrases(text):
        _push(phrase)
    return _prune_redundant_focus_terms(out)


def _refs_hit_surface_text(hit: dict) -> str:
    ui_meta = (hit or {}).get("ui_meta") if isinstance((hit or {}).get("ui_meta"), dict) else {}
    meta = (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}
    citation_meta = (ui_meta or {}).get("citation_meta") if isinstance((ui_meta or {}).get("citation_meta"), dict) else {}
    parts = [
        str(hit.get("text") or "").strip(),
        str((ui_meta or {}).get("display_name") or "").strip(),
        str((ui_meta or {}).get("heading_path") or "").strip(),
        str((ui_meta or {}).get("summary_line") or "").strip(),
        str((meta or {}).get("ref_best_heading_path") or "").strip(),
        str((meta or {}).get("ref_section") or "").strip(),
        str((citation_meta or {}).get("title") or "").strip(),
    ]
    joined = " ".join(part for part in parts if part)
    return _normalize_title_identity(joined)


def _refs_raw_hit_surface_text(hit: dict) -> str:
    meta = (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}
    parts: list[str] = [
        str((hit or {}).get("text") or "").strip(),
        str((meta or {}).get("source_path") or "").strip(),
        str((meta or {}).get("ref_best_heading_path") or "").strip(),
        str((meta or {}).get("ref_section") or "").strip(),
        str((meta or {}).get("ref_subsection") or "").strip(),
        str((meta or {}).get("top_heading") or "").strip(),
    ]
    for key in ("ref_show_snippets", "ref_snippets", "ref_overview_snippets", "ref_headings"):
        raw = (meta or {}).get(key)
        if not isinstance(raw, list):
            continue
        parts.extend(str(item or "").strip() for item in raw[:3] if str(item or "").strip())
    return _normalize_title_identity(" ".join(part for part in parts if part))


def _refs_raw_hit_identity_terms(hit: dict) -> set[str]:
    meta = (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}
    identities: set[str] = set()
    for raw in (
        str((meta or {}).get("source_path") or "").strip(),
        str((meta or {}).get("title") or "").strip(),
    ):
        identities.update(_title_identity_keys(raw))
    return {item for item in identities if item}


def _refs_raw_hit_focus_match_count(prompt: str, hit: dict) -> int:
    focus_terms = _refs_prompt_focus_terms(prompt)
    if not focus_terms:
        return 0
    surface = _refs_raw_hit_surface_text(hit)
    if not surface:
        return 0
    count = sum(1 for term in focus_terms if _focus_term_matches_surface(term, surface))
    if count <= 0 and _shared_prompt_targets_sci_topic(prompt) and _surface_is_sci_related_predecessor(surface):
        return 1
    return count


def _refs_raw_hit_non_source_focus_match_count(prompt: str, hit: dict) -> int:
    focus_terms = _refs_prompt_focus_terms(prompt)
    if not focus_terms:
        return 0
    surface = _refs_raw_hit_surface_text(hit)
    if not surface:
        return 0
    identity_terms = _refs_raw_hit_identity_terms(hit)
    count = 0
    for term in focus_terms:
        if not _focus_term_matches_surface(term, surface):
            continue
        if any(term == ident or term in ident or ident in term for ident in identity_terms):
            continue
        count += 1
    if count <= 0 and _shared_prompt_targets_sci_topic(prompt) and _surface_is_sci_related_predecessor(surface):
        return 1
    return count


def _filter_pending_refs_hits_by_prompt_focus(prompt: str, hits: list[dict]) -> list[dict]:
    rows = [hit for hit in (hits or []) if isinstance(hit, dict)]
    if not _prompt_requires_explicit_focus_match(prompt):
        return rows
    rows = [hit for hit in rows if not _refs_hit_focus_terms_only_negated(prompt, hit)]
    focus_terms = _refs_prompt_focus_terms(prompt)
    focus_action = _shared_prompt_reference_focus_action(prompt)
    if not focus_terms:
        return rows
    if focus_action == "compare":
        scored_compare_hits = sorted(
            (
                (_refs_compare_prompt_hit_score(prompt, hit, raw=True), hit)
                for hit in rows
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        compare_hits = [hit for score, hit in scored_compare_hits if score >= _MIN_COMPARE_DIRECT_HIT_SCORE]
        if len(compare_hits) >= 2:
            top_score = float(scored_compare_hits[0][0])
            second_score = float(scored_compare_hits[1][0])
            if top_score >= (second_score + 1.0):
                return [compare_hits[0]]
        return compare_hits
    if _prompt_requests_single_paper_pick(prompt) and focus_action != "compare":
        scored_direct_hits = sorted(
            (
                (_refs_single_paper_pick_hit_score(prompt, hit, raw=True), hit)
                for hit in rows
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [hit for score, hit in scored_direct_hits if score >= _MIN_PENDING_SINGLE_PAPER_DIRECT_HIT_SCORE]
    if len(focus_terms) > 1:
        return [hit for hit in rows if _refs_raw_hit_non_source_focus_match_count(prompt, hit) > 0]
    return [hit for hit in rows if _refs_raw_hit_focus_match_count(prompt, hit) > 0]


def _refs_prompt_source_match_boost(prompt: str, hit: dict) -> float:
    prompt_norm = _normalize_title_identity(prompt)
    if not prompt_norm:
        return 0.0
    ui_meta = (hit or {}).get("ui_meta") if isinstance((hit or {}).get("ui_meta"), dict) else {}
    meta = (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}
    citation_meta = (ui_meta or {}).get("citation_meta") if isinstance((ui_meta or {}).get("citation_meta"), dict) else {}
    identities: set[str] = set()
    for raw in (
        str((meta or {}).get("source_path") or "").strip(),
        str((ui_meta or {}).get("display_name") or "").strip(),
        str((citation_meta or {}).get("title") or "").strip(),
    ):
        identities.update(_title_identity_keys(raw))
    best = 0.0
    for ident in identities:
        if len(ident) < 3:
            continue
        if ident in prompt_norm:
            best = max(best, 2.5 if len(ident) >= 6 else 1.6)
            continue
        ident_tokens = [tok for tok in ident.split() if len(tok) >= 3]
        if ident_tokens and all(tok in prompt_norm for tok in ident_tokens[: min(3, len(ident_tokens))]):
            best = max(best, 2.0)
    return best


def _refs_hit_identity_terms(hit: dict) -> set[str]:
    ui_meta = (hit or {}).get("ui_meta") if isinstance((hit or {}).get("ui_meta"), dict) else {}
    meta = (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}
    citation_meta = (ui_meta or {}).get("citation_meta") if isinstance((ui_meta or {}).get("citation_meta"), dict) else {}
    identities: set[str] = set()
    for raw in (
        str((meta or {}).get("source_path") or "").strip(),
        str((ui_meta or {}).get("display_name") or "").strip(),
        str((citation_meta or {}).get("title") or "").strip(),
    ):
        identities.update(_title_identity_keys(raw))
    return {item for item in identities if item}


def _surface_is_sci_related_predecessor(surface_text: str) -> bool:
    surface = _normalize_title_identity(surface_text)
    if not surface:
        return False
    return bool(
        "single shot compressive spectral imaging" in surface
        or (
            "single shot spectral imaging" in surface
            and "compressive sensing" in surface
        )
    )


def _refs_hit_focus_match_count(prompt: str, hit: dict) -> int:
    focus_terms = _refs_prompt_focus_terms(prompt)
    if not focus_terms:
        return 0
    surface = _refs_hit_surface_text(hit)
    if not surface:
        return 0
    count = sum(1 for term in focus_terms if _focus_term_matches_surface(term, surface))
    if count <= 0 and _shared_prompt_targets_sci_topic(prompt) and _surface_is_sci_related_predecessor(surface):
        return 1
    return count


def _refs_hit_non_source_focus_match_count(prompt: str, hit: dict) -> int:
    focus_terms = _refs_prompt_focus_terms(prompt)
    if not focus_terms:
        return 0
    surface = _refs_hit_surface_text(hit)
    if not surface:
        return 0
    identity_terms = _refs_hit_identity_terms(hit)
    count = 0
    for term in focus_terms:
        if not _focus_term_matches_surface(term, surface):
            continue
        if any(term == ident or term in ident or ident in term for ident in identity_terms):
            continue
        count += 1
    if count <= 0 and _shared_prompt_targets_sci_topic(prompt) and _surface_is_sci_related_predecessor(surface):
        return 1
    return count


def _refs_hit_evidence_surface_text(hit: dict) -> str:
    ui_meta = (hit or {}).get("ui_meta") if isinstance((hit or {}).get("ui_meta"), dict) else {}
    meta = (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}
    parts = [
        str(hit.get("text") or "").strip(),
        str((ui_meta or {}).get("heading_path") or "").strip(),
        str((ui_meta or {}).get("summary_line") or "").strip(),
        str((meta or {}).get("ref_best_heading_path") or "").strip(),
        str((meta or {}).get("ref_section") or "").strip(),
    ]
    joined = " ".join(part for part in parts if part)
    return _normalize_title_identity(joined)


def _focus_term_only_negated_in_surface(term: str, surface: str) -> bool:
    normalized_term = _normalize_title_identity(term)
    normalized_surface = _normalize_title_identity(surface)
    if (not normalized_term) or (not normalized_surface):
        return False
    escaped = re.escape(normalized_term)
    all_count = len(re.findall(rf"\b{escaped}\b", normalized_surface, flags=re.I))
    if all_count <= 0:
        return False
    neg_pattern = (
        rf"\b(?:without|not|no|never|instead\s+of|rather\s+than|free\s+of|excluding|exclude(?:s|d|ing)?|"
        rf"avoid(?:s|ed|ing)?|omit(?:s|ted|ting)?|lack(?:s|ed|ing)?)(?:\s+[a-z0-9-]+){{0,6}}\s+{escaped}\b"
    )
    negated_count = len(re.findall(neg_pattern, normalized_surface, flags=re.I))
    return negated_count >= all_count


def _refs_hit_focus_terms_only_negated(prompt: str, hit: dict) -> bool:
    focus_terms = _refs_prompt_focus_terms(prompt)
    if not focus_terms:
        return False
    evidence_surface = _refs_hit_evidence_surface_text(hit)
    if not evidence_surface:
        return False
    matched_terms = [term for term in focus_terms if _focus_term_matches_surface(term, evidence_surface)]
    if not matched_terms:
        return False
    return all(_focus_term_only_negated_in_surface(term, evidence_surface) for term in matched_terms)


def _prompt_requests_single_paper_pick(prompt: str) -> bool:
    return _shared_prompt_explicitly_requests_single_paper_pick(prompt)


def _prompt_requests_compare(prompt: str) -> bool:
    return _shared_prompt_requests_reference_compare(prompt)


def _prompt_requests_definition(prompt: str) -> bool:
    return _shared_prompt_requests_reference_definition(prompt)


def _refs_hit_directness_surface_text(hit: dict, *, raw: bool) -> str:
    meta = (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}
    ref_pack = (meta or {}).get("ref_pack") if isinstance((meta or {}).get("ref_pack"), dict) else {}
    if raw:
        parts: list[str] = [
            str((hit or {}).get("text") or "").strip(),
            str((meta or {}).get("ref_best_heading_path") or "").strip(),
            str((meta or {}).get("ref_section") or "").strip(),
            str((meta or {}).get("ref_subsection") or "").strip(),
            str((ref_pack or {}).get("what") or "").strip(),
            str((ref_pack or {}).get("why") or "").strip(),
        ]
        for key in ("ref_show_snippets", "ref_snippets", "ref_overview_snippets"):
            raw_items = (meta or {}).get(key)
            if not isinstance(raw_items, list):
                continue
            parts.extend(str(item or "").strip() for item in raw_items[:2] if str(item or "").strip())
        return " ".join(part for part in parts if part)

    ui_meta = (hit or {}).get("ui_meta") if isinstance((hit or {}).get("ui_meta"), dict) else {}
    parts = [
        str((hit or {}).get("text") or "").strip(),
        str((ui_meta or {}).get("heading_path") or "").strip(),
        str((ui_meta or {}).get("summary_line") or "").strip(),
        str((ui_meta or {}).get("why_line") or "").strip(),
        str((meta or {}).get("ref_best_heading_path") or "").strip(),
        str((meta or {}).get("ref_section") or "").strip(),
        str((ref_pack or {}).get("what") or "").strip(),
        str((ref_pack or {}).get("why") or "").strip(),
    ]
    return " ".join(part for part in parts if part)


def _refs_hit_directness_heading_path(hit: dict, *, raw: bool) -> str:
    meta = (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}
    if raw:
        return str(
            (meta or {}).get("ref_best_heading_path")
            or (meta or {}).get("heading_path")
            or (meta or {}).get("ref_section")
            or ""
        ).strip()
    ui_meta = (hit or {}).get("ui_meta") if isinstance((hit or {}).get("ui_meta"), dict) else {}
    return str(
        (ui_meta or {}).get("heading_path")
        or (meta or {}).get("ref_best_heading_path")
        or (meta or {}).get("heading_path")
        or (meta or {}).get("ref_section")
        or ""
    ).strip()


def _refs_single_paper_pick_heading_score(heading_path: str) -> float:
    heading_norm = _normalize_title_identity(str(heading_path or "").strip())
    if not heading_norm:
        return 0.0
    if "abstract" in heading_norm:
        return 2.2
    if "introduction" in heading_norm:
        return 1.8
    if re.search(r"\b(method|methods|model|pipeline|architecture|framework|algorithm)\b", heading_norm):
        return 1.0
    if re.search(r"\b(compare|comparison|analysis|experiment|results?|evaluation)\b", heading_norm):
        return 0.8
    if ("related work" in heading_norm) or ("background" in heading_norm) or ("literature review" in heading_norm):
        return -2.4
    if ("conclusion" in heading_norm) or ("discussion" in heading_norm) or ("future work" in heading_norm):
        return -0.8
    return 0.0


def _refs_single_paper_pick_hit_score(prompt: str, hit: dict, *, raw: bool = False) -> float:
    if not _prompt_requests_single_paper_pick(prompt):
        return -1000.0
    surface = _refs_hit_directness_surface_text(hit, raw=raw)
    if not surface:
        return -1000.0

    if raw:
        focus_hits = _refs_raw_hit_non_source_focus_match_count(prompt, hit)
        identity_surface = " ".join(sorted(_refs_raw_hit_identity_terms(hit)))
    else:
        focus_hits = _refs_hit_non_source_focus_match_count(prompt, hit)
        identity_surface = " ".join(sorted(_refs_hit_identity_terms(hit)))
    title_focus_hits = _refs_focus_match_count_for_text(prompt, identity_surface)
    title_keyword_hits = _refs_summary_focus_keyword_hit_count(prompt, identity_surface)
    surface_keyword_hits = _refs_summary_focus_keyword_hit_count(prompt, surface)
    if focus_hits <= 0 and title_focus_hits <= 0 and title_keyword_hits <= 0:
        return -1000.0

    heading_path = _refs_hit_directness_heading_path(hit, raw=raw)
    heading_score = _refs_single_paper_pick_heading_score(heading_path)
    heading_keyword_hits = _refs_summary_focus_keyword_hit_count(prompt, heading_path)
    surface_low = str(surface or "").strip().lower()
    score = 0.0
    score += 2.4 * float(focus_hits)
    score += 2.1 * float(title_focus_hits)
    score += 1.6 * float(min(2, surface_keyword_hits))
    if heading_keyword_hits > 0:
        score += 0.8
    if title_keyword_hits >= 2:
        score += 1.2
    elif title_keyword_hits == 1 and title_focus_hits <= 0:
        score += 0.6
    score += heading_score
    if title_focus_hits > 0 and focus_hits <= 0 and heading_score >= 0.8:
        score += 1.8

    if _prompt_requests_definition(prompt):
        if re.search(r"\b(defin(?:e|es|ed|ition)|refers?\s+to|is\s+defined\s+as|introduced?\s+as|means)\b", surface_low):
            score += 3.0
        elif surface_keyword_hits > 0 or heading_keyword_hits > 0 or title_keyword_hits > 0:
            score += 1.4
        else:
            score -= 1.8
    else:
        if re.search(
            r"\b(this paper|the paper|this work|the work|we\s+(?:present|propose|introduce|define|describe|analy[sz]e|study|show|demonstrate|develop|use|investigate|explore))\b",
            surface_low,
        ):
            score += 1.4
        if re.search(r"\b(discuss(?:es|ed)?|explain(?:s|ed)?|describe(?:s|d)?|analy[sz]e(?:s|d)?|introduce(?:s|d)?|define(?:s|d)?)\b", surface_low):
            score += 0.9

    if re.search(
        r"\b(mentioned\s+here\s+only|mentioned\s+in\s+passing|generic\s+optimization\s+family|background\s+discussion|"
        r"related\s+work|prior\s+work|previous\s+work|existing\s+methods?|most\s+of\s+the\s+existing\s+methods?|"
        r"many\s+existing\s+methods?|instead\s+of\s+using|widely\s+used|commonly\s+used|citation\s+in\s+related\s+work)\b",
        surface_low,
    ):
        score -= 3.4
    if _looks_negative_ref_reason_text(surface):
        score -= 4.4
    if (
        focus_hits > 0
        and title_focus_hits <= 0
        and title_keyword_hits <= 1
        and heading_score <= 0.0
        and not re.search(
            r"\b(defin(?:e|es|ed|ition)|discuss(?:es|ed)?|explain(?:s|ed)?|describe(?:s|d)?|analy[sz]e(?:s|d)?|introduce(?:s|d)?|compare(?:s|d)?)\b",
            surface_low,
        )
    ):
        score -= 0.6
    return score


def _refs_compare_prompt_hit_score(prompt: str, hit: dict, *, raw: bool = False) -> float:
    if raw:
        surface = _refs_hit_directness_surface_text(hit, raw=True)
        title_surface = " ".join(sorted(_refs_raw_hit_identity_terms(hit)))
        focus_hits = _refs_raw_hit_non_source_focus_match_count(prompt, hit)
    else:
        surface = _refs_hit_surface_text(hit)
        title_surface = " ".join(sorted(_refs_hit_identity_terms(hit)))
        focus_hits = _refs_hit_non_source_focus_match_count(prompt, hit)
    if not surface:
        return -1000.0
    score = 0.0
    score += 2.2 * float(focus_hits)
    if re.search(r"\b(compare|compares|compared|comparison|versus|vs\.?)\b", surface, flags=re.I):
        score += 2.0
    if title_surface and re.search(r"\b(compare|comparison|versus|vs\.?)\b", title_surface, flags=re.I):
        score += 2.8
    if title_surface:
        title_keyword_hits = _refs_summary_focus_keyword_hit_count(prompt, title_surface)
        if title_keyword_hits >= 2:
            score += 2.0
    if re.search(r"\b(directly|systematically|head[\s-]?to[\s-]?head)\b", surface, flags=re.I):
        score += 0.8
    if re.search(r"\b(does\s+not\s+compare|not\s+compare|without\s+comparing|mentions?\s+.*\bbut\s+does\s+not)\b", surface, flags=re.I):
        score -= 4.2
    if re.search(r"\b(background example|mention(?:ed)?\s+in\s+passing|related work)\b", surface, flags=re.I):
        score -= 1.2
    if focus_hits >= 2 and (not re.search(r"\b(compare|compares|compared|comparison|versus|vs\.?)\b", surface, flags=re.I)) and (not re.search(r"\b(compare|comparison|versus|vs\.?)\b", title_surface, flags=re.I)):
        score -= 1.8
    return score


def _prompt_requires_explicit_focus_match(prompt: str) -> bool:
    if not _shared_prompt_requires_reference_focus_match(prompt):
        return False
    return bool(_refs_prompt_focus_terms(prompt))


def _looks_negative_ref_reason_text(text: str) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    patterns = (
        r"\bnot mentioned\b",
        r"\bnot discuss(?:ed)?\b",
        r"\bnot stated\b",
        r"\bdoes not mention\b",
        r"\bdoesn't mention\b",
        r"\bdoes not specify\b",
        r"\bdoesn't specify\b",
        r"\bnot found\b",
        r"\bcannot point\b",
        r"\bno external paper matched\b",
        r"\bno papers? in (?:my|your) library\b",
        r"\bnone of the retrieved documents directly discuss\b",
        r"没有提到",
        r"没有命中",
        r"无法定位",
        r"不能指向",
        r"未提及",
        r"未提到",
    )
    return any(re.search(pat, low, flags=re.I) for pat in patterns)


def _refs_focus_match_count_for_text(prompt: str, text: str) -> int:
    surface = _normalize_title_identity(text)
    if not surface:
        return 0
    return sum(1 for term in _refs_prompt_focus_terms(prompt) if _focus_term_matches_surface(term, surface))


def _should_suppress_negative_ref_hit(prompt: str, hit: dict) -> bool:
    if not _prompt_requires_explicit_focus_match(prompt):
        return False
    ui_meta = (hit or {}).get("ui_meta") if isinstance((hit or {}).get("ui_meta"), dict) else {}
    meta = (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}
    ref_pack = (meta or {}).get("ref_pack") if isinstance((meta or {}).get("ref_pack"), dict) else {}
    why_line = " ".join(
        part for part in (
            str((ui_meta or {}).get("why_line") or "").strip(),
            str((ref_pack or {}).get("why") or "").strip(),
        ) if part
    )
    summary_line = str((ui_meta or {}).get("summary_line") or "").strip()
    if (not _looks_negative_ref_reason_text(why_line)) and (not _looks_negative_ref_reason_text(summary_line)):
        return False
    positive_surface = " ".join(
        part for part in (
            str((hit or {}).get("text") or "").strip(),
            str((ui_meta or {}).get("summary_line") or "").strip(),
            str((ui_meta or {}).get("heading_path") or "").strip(),
            str((meta or {}).get("ref_best_heading_path") or "").strip(),
            str((meta or {}).get("ref_section") or "").strip(),
        ) if part
    )
    return _refs_focus_match_count_for_text(prompt, positive_surface) <= 0


def _filter_refs_hits_by_prompt_focus(prompt: str, hits: list[dict]) -> list[dict]:
    rows = [hit for hit in (hits or []) if isinstance(hit, dict)]
    if not _prompt_requires_explicit_focus_match(prompt):
        return rows
    rows = [hit for hit in rows if not _should_suppress_negative_ref_hit(prompt, hit)]
    rows = [hit for hit in rows if not _refs_hit_focus_terms_only_negated(prompt, hit)]
    focus_terms = _refs_prompt_focus_terms(prompt)
    if _prompt_requests_compare(prompt):
        def _ready_compare_display_score(hit: dict) -> float:
            ready_score = _refs_compare_prompt_hit_score(prompt, hit)
            raw_score = _refs_compare_prompt_hit_score(prompt, hit, raw=True)
            if raw_score <= -999.0:
                return ready_score
            return min(ready_score, raw_score)

        scored_compare_hits = sorted(
            (
                (_ready_compare_display_score(hit), hit)
                for hit in rows
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        compare_hits = [hit for score, hit in scored_compare_hits if score >= _MIN_COMPARE_DIRECT_HIT_SCORE]
        if len(compare_hits) >= 2:
            top_score = float(scored_compare_hits[0][0])
            second_score = float(scored_compare_hits[1][0])
            if top_score >= (second_score + 1.0):
                return [compare_hits[0]]
        return compare_hits
    if _prompt_requests_single_paper_pick(prompt):
        scored_direct_hits = sorted(
            (
                (_refs_single_paper_pick_hit_score(prompt, hit), hit)
                for hit in rows
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [hit for score, hit in scored_direct_hits if score >= _MIN_SINGLE_PAPER_DIRECT_HIT_SCORE]
    matched_non_source = [hit for hit in rows if _refs_hit_non_source_focus_match_count(prompt, hit) > 0]
    if _prompt_explicitly_requests_multi_paper_list(prompt):
        if matched_non_source:
            return matched_non_source
        matched = [hit for hit in rows if _refs_hit_focus_match_count(prompt, hit) > 0]
        return matched if matched else rows
    if len(focus_terms) > 1:
        return matched_non_source if matched_non_source else []
    matched = [hit for hit in rows if _refs_hit_focus_match_count(prompt, hit) > 0]
    return matched if matched else []


def _sort_refs_hits_for_display(*, prompt: str, hits: list[dict]) -> list[dict]:
    decorated: list[tuple[float, float, float, int, dict]] = []
    for idx, hit in enumerate(hits or []):
        if not isinstance(hit, dict):
            continue
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        score = _refs_hit_display_score(hit)
        anchor_score = _non_negative_float((ui_meta or {}).get("anchor_match_score"))
        doc_score = _non_negative_float((ui_meta or {}).get("explicit_doc_match_score"))
        focus_count = float(_refs_hit_focus_match_count(prompt, hit))
        prompt_source_boost = float(_refs_prompt_source_match_boost(prompt, hit))
        decorated.append((focus_count, prompt_source_boost, score, anchor_score, doc_score, -idx, hit))
    decorated.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5]), reverse=True)
    return [item[6] for item in decorated]


def _prompt_explicitly_requests_multi_paper_list(prompt: str) -> bool:
    return _shared_prompt_explicitly_requests_multi_paper_list(prompt)


def _prompt_likely_cross_paper_refs(prompt: str) -> bool:
    low = str(prompt or "").strip().lower()
    if not low:
        return False
    needles = (
        "other paper",
        "other papers",
        "which paper",
        "which papers",
        "besides this paper",
        "in my library",
        "related papers",
        "references in my library",
        "\u54ea\u7bc7",
        "\u54ea\u4e9b\u8bba\u6587",
        "\u8fd8\u6709\u54ea\u4e9b",
        "\u5e93\u91cc",
        "\u522b\u7684\u8bba\u6587",
        "\u5176\u4ed6\u8bba\u6587",
        "\u6709\u54ea\u51e0\u7bc7",
        "\u6709\u54ea\u4e9b",
    )
    return any(token in low for token in needles)


def _should_try_refs_hit_rerank(prompt: str, hits: list[dict]) -> bool:
    rows = [hit for hit in (hits or []) if isinstance(hit, dict)]
    if len(rows) < 2:
        return False
    top = _refs_hit_display_score(rows[0])
    second = _refs_hit_display_score(rows[1])
    third = _refs_hit_display_score(rows[2]) if len(rows) >= 3 else 0.0
    margin = top - second
    top_gap_23 = second - third
    if _prompt_likely_cross_paper_refs(prompt):
        return bool(top < 9.25 or margin < 1.10)
    return bool(top < 8.65 or margin < 0.85 or top_gap_23 < 0.45)


def _refs_hit_relevance_llm_enabled() -> bool:
    raw_flag = str(os.environ.get("KB_REFS_RELEVANCE_USE_LLM", "1") or "").strip().lower()
    if raw_flag in {"0", "false", "off", "no"}:
        return False
    try:
        settings = load_settings()
    except Exception:
        return False
    return bool(getattr(settings, "api_key", None))


def _should_try_refs_hit_relevance_gate(prompt: str, hits: list[dict], *, guide_mode: bool) -> bool:
    rows = [hit for hit in (hits or []) if isinstance(hit, dict)]
    if not rows:
        return False
    if not (_prompt_requires_explicit_focus_match(prompt) or _prompt_likely_cross_paper_refs(prompt)):
        return False
    if _prompt_explicitly_requests_multi_paper_list(prompt) and len(rows) > 1:
        return False
    if guide_mode and _prompt_likely_cross_paper_refs(prompt):
        return True
    if len(rows) == 1:
        return True
    return True


@lru_cache(maxsize=512)
def _llm_filter_refs_hit_indices(
    *,
    prompt: str,
    guide_mode: bool,
    candidates_payload: str,
    candidate_count: int,
) -> tuple[int, ...] | None:
    if (not prompt) or (not candidates_payload) or candidate_count <= 0:
        return None
    if not _refs_hit_relevance_llm_enabled():
        return None
    try:
        settings = load_settings()
    except Exception:
        return None
    if not getattr(settings, "api_key", None):
        return None
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
                            "You are filtering research reference hits for display. "
                            "Keep only hits that directly answer the user's request using the supplied evidence. "
                            "Drop hits that are only broadly related, only match the paper title, or only share a loose topic. "
                            "For definition requests, keep only hits that explicitly define or clearly explain the concept. "
                            "For comparison requests, keep only hits that explicitly compare the requested methods or concepts. "
                            "For cross-paper guide mode, the bound paper has already been filtered out, so judge only the remaining external papers. "
                            "Return JSON only like {\"keep\": [1, 3]}. Use 1-based indices and include each index at most once."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Guide mode: {'true' if guide_mode else 'false'}\n"
                            f"Prompt: {str(prompt or '').strip()}\n\n"
                            f"Candidates:\n{candidates_payload}\n"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=180,
            )
            or ""
        ).strip()
    except Exception:
        return None

    nums: list[int] = []
    parsed_keep_list = False
    try:
        parsed = json.loads(out)
        raw_keep = parsed.get("keep") if isinstance(parsed, dict) else None
        if isinstance(raw_keep, list):
            parsed_keep_list = True
            nums = [int(item) for item in raw_keep]
    except Exception:
        nums = []
    if not nums:
        m = re.search(r'"keep"\s*:\s*\[([^\]]*)\]', out)
        if m:
            parsed_keep_list = True
            nums = [int(item) for item in re.findall(r"-?\d+", str(m.group(1) or ""))]
    if (not nums) and (not parsed_keep_list):
        return None

    seen: set[int] = set()
    out_keep: list[int] = []
    for raw in nums:
        try:
            idx = int(raw)
        except Exception:
            continue
        if idx < 1 or idx > candidate_count or idx in seen:
            continue
        seen.add(idx)
        out_keep.append(idx)
    return tuple(out_keep)


def _maybe_llm_filter_refs_hits(*, prompt: str, hits: list[dict], guide_mode: bool) -> list[dict]:
    rows = [hit for hit in (hits or []) if isinstance(hit, dict)]
    if not rows:
        return rows
    if not _should_try_refs_hit_relevance_gate(prompt, rows, guide_mode=guide_mode):
        return rows

    pool = rows[: min(4, len(rows))]
    candidate_lines: list[str] = []
    for idx, hit in enumerate(pool, start=1):
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        candidate_lines.append(
            "\n".join(
                [
                    f"{idx}. source: {str((ui_meta or {}).get('display_name') or (meta or {}).get('source_path') or '').strip()}",
                    f"   heading: {str((ui_meta or {}).get('heading_path') or '').strip() or '(none)'}",
                    f"   summary: {str((ui_meta or {}).get('summary_line') or '').strip()[:240]}",
                    f"   why: {str((ui_meta or {}).get('why_line') or '').strip()[:200]}",
                    f"   snippet: {str(hit.get('text') or '').strip()[:260]}",
                    f"   ui_score: {_refs_hit_display_score(hit):.3f}",
                    f"   focus_matches: {_refs_hit_focus_match_count(prompt, hit)}",
                    f"   non_source_focus_matches: {_refs_hit_non_source_focus_match_count(prompt, hit)}",
                ]
            )
        )

    keep = _llm_filter_refs_hit_indices(
        prompt=str(prompt or "").strip(),
        guide_mode=bool(guide_mode),
        candidates_payload="\n\n".join(candidate_lines),
        candidate_count=len(pool),
    )
    if keep is None:
        return rows
    if not keep:
        return []

    kept_rows: list[dict] = []
    for idx1 in keep:
        zero = int(idx1) - 1
        if zero < 0 or zero >= len(pool):
            continue
        kept_rows.append(pool[zero])
    return kept_rows


@lru_cache(maxsize=512)
def _llm_rerank_refs_hit_order(
    *,
    prompt: str,
    guide_mode: bool,
    candidates_payload: str,
    candidate_count: int,
) -> tuple[int, ...]:
    if (not prompt) or (not candidates_payload) or candidate_count <= 1:
        return ()
    if not _refs_hit_rerank_llm_enabled():
        return ()
    try:
        settings = load_settings()
    except Exception:
        return ()
    if not getattr(settings, "api_key", None):
        return ()
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
                            "You are ranking library references for display. "
                            "Choose which paper hits the user should see first. "
                            "Prefer papers that directly answer the prompt, explicitly mention the requested concept, "
                            "and provide a precise navigable section or snippet. "
                            "Prefer direct topical relevance over broad similarity. "
                            "If guide mode is true, the current bound paper has already been filtered out, "
                            "so rank only the remaining external papers. "
                            "Return JSON only, like {\"order\": [2, 1, 3]}. Use 1-based indices and include each index at most once."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Guide mode: {'true' if guide_mode else 'false'}\n"
                            f"Prompt: {str(prompt or '').strip()}\n\n"
                            f"Candidates:\n{candidates_payload}\n"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=180,
            )
            or ""
        ).strip()
    except Exception:
        return ()

    nums: list[int] = []
    try:
        parsed = json.loads(out)
        raw_order = parsed.get("order") if isinstance(parsed, dict) else None
        if isinstance(raw_order, list):
            nums = [int(item) for item in raw_order]
    except Exception:
        nums = []
    if not nums:
        m = re.search(r'"order"\s*:\s*\[([^\]]*)\]', out)
        if m:
            nums = [int(item) for item in re.findall(r"-?\d+", str(m.group(1) or ""))]
    if not nums:
        nums = [int(item) for item in re.findall(r"-?\d+", out)]

    seen: set[int] = set()
    out_order: list[int] = []
    for raw in nums:
        try:
            idx = int(raw)
        except Exception:
            continue
        if idx < 1 or idx > candidate_count or idx in seen:
            continue
        seen.add(idx)
        out_order.append(idx)
    return tuple(out_order)


def _maybe_llm_rerank_refs_hits(*, prompt: str, hits: list[dict], guide_mode: bool) -> list[dict]:
    rows = [hit for hit in (hits or []) if isinstance(hit, dict)]
    if len(rows) < 2:
        return rows
    if not _should_try_refs_hit_rerank(prompt, rows):
        return rows

    pool = rows[: min(4, len(rows))]
    candidate_lines: list[str] = []
    for idx, hit in enumerate(pool, start=1):
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        citation_meta = (ui_meta or {}).get("citation_meta") if isinstance((ui_meta or {}).get("citation_meta"), dict) else {}
        cite_bits = [
            str((citation_meta or {}).get("title") or "").strip(),
            str((citation_meta or {}).get("venue") or "").strip(),
            str((citation_meta or {}).get("year") or "").strip(),
        ]
        candidate_lines.append(
            "\n".join(
                [
                    f"{idx}. source: {str((ui_meta or {}).get('display_name') or (meta or {}).get('source_path') or '').strip()}",
                    f"   heading: {str((ui_meta or {}).get('heading_path') or '').strip() or '(none)'}",
                    f"   summary: {str((ui_meta or {}).get('summary_line') or '').strip()[:280]}",
                    f"   why: {str((ui_meta or {}).get('why_line') or '').strip()[:220]}",
                    f"   snippet: {str(hit.get('text') or '').strip()[:240]}",
                    f"   citation: {' | '.join(bit for bit in cite_bits if bit)[:220] or '(none)'}",
                    f"   ui_score: {_refs_hit_display_score(hit):.3f}",
                ]
            )
        )

    order = _llm_rerank_refs_hit_order(
        prompt=str(prompt or "").strip(),
        guide_mode=bool(guide_mode),
        candidates_payload="\n\n".join(candidate_lines),
        candidate_count=len(pool),
    )
    if not order:
        return rows

    ordered_pool: list[dict] = []
    used: set[int] = set()
    for idx1 in order:
        zero = int(idx1) - 1
        if zero < 0 or zero >= len(pool) or zero in used:
            continue
        used.add(zero)
        ordered_pool.append(pool[zero])
    for zero, hit in enumerate(pool):
        if zero in used:
            continue
        ordered_pool.append(hit)
    return ordered_pool + rows[len(pool):]


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


def _resolve_refs_payload_render_variant(
    *,
    render_variant: str,
    allow_citation_prefetch_for_pending: bool,
    allow_expensive_llm_for_ready: bool,
    allow_exact_locate: bool,
) -> tuple[str, bool, bool, bool]:
    variant = str(render_variant or "").strip().lower() or "interactive_full"
    if variant == "fast":
        return variant, False, False, False
    if variant in {"bounded_full", "precomputed_full"}:
        return "bounded_full", False, False, True
    return (
        "interactive_full",
        bool(allow_citation_prefetch_for_pending),
        bool(allow_expensive_llm_for_ready),
        bool(allow_exact_locate),
    )


def enrich_refs_payload(
    refs_by_user: dict[int, dict],
    *,
    pdf_root: Path | None,
    md_root: Path | None,
    lib_store: LibraryStore | None,
    guide_mode: bool = False,
    guide_source_path: str = "",
    guide_source_name: str = "",
    allow_citation_prefetch_for_pending: bool = False,
    allow_expensive_llm_for_ready: bool = True,
    allow_exact_locate: bool = True,
    render_variant: str = "interactive_full",
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    guide_source_path_norm = str(guide_source_path or "").strip()
    guide_source_name_norm = str(guide_source_name or "").strip()
    guide_active = bool(guide_mode and (guide_source_path_norm or guide_source_name_norm))
    (
        _render_variant,
        allow_citation_prefetch_for_pending,
        allow_expensive_llm_for_ready,
        allow_exact_locate,
    ) = _resolve_refs_payload_render_variant(
        render_variant=render_variant,
        allow_citation_prefetch_for_pending=allow_citation_prefetch_for_pending,
        allow_expensive_llm_for_ready=allow_expensive_llm_for_ready,
        allow_exact_locate=allow_exact_locate,
    )
    for user_msg_id, pack in (refs_by_user or {}).items():
        if not isinstance(pack, dict):
            continue
        prompt = str(pack.get("prompt") or "").strip()
        prompt_requires_focus_match = bool(_prompt_requires_explicit_focus_match(prompt))
        prompt_cross_paper_refs = bool(_prompt_likely_cross_paper_refs(prompt))
        prompt_multi_paper_list = bool(_prompt_explicitly_requests_multi_paper_list(prompt))
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
            if guide_active and _hit_matches_guide_source(
                meta,
                guide_source_path=guide_source_path_norm,
                guide_source_name=guide_source_name_norm,
            ):
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
            if prompt_multi_paper_list:
                hits.append(hit2)
                continue
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
        post_score_gate_hit_count = int(len(hits))
        post_focus_filter_hit_count = int(post_score_gate_hit_count)
        post_llm_filter_hit_count = int(post_focus_filter_hit_count)
        if has_pending and hits:
            hits = _filter_pending_refs_hits_by_prompt_focus(prompt, hits)
            post_focus_filter_hit_count = int(len(hits))
            hits = _sort_refs_hits_for_display(prompt=prompt, hits=hits)
            hits = hits[:4]
            post_llm_filter_hit_count = int(len(hits))
        allow_citation_prefetch = bool(
            hits
            and allow_expensive_llm_for_ready
            and ((not has_pending) or allow_citation_prefetch_for_pending)
        )
        preloaded_citation_meta = (
            _prefetch_refs_citation_meta(
                hits,
                pdf_root=pdf_root,
                md_root=md_root,
                lib_store=lib_store,
            )
            if allow_citation_prefetch
            else {}
        )
        allow_hit_llm_refine = bool((not has_pending) and allow_expensive_llm_for_ready)
        for hit2 in hits:
            hit2["ui_meta"] = build_hit_ui_meta(
                hit2,
                prompt=prompt,
                pdf_root=pdf_root,
                lib_store=lib_store,
                preloaded_citation_meta=preloaded_citation_meta,
                allow_expensive_llm=allow_hit_llm_refine,
                allow_exact_locate=bool((not has_pending) and allow_exact_locate),
            )
        if hits and (not has_pending):
            hits = _filter_refs_hits_by_prompt_focus(prompt, hits)
        post_focus_filter_hit_count = int(len(hits))
        if len(hits) > 1:
            hits = _sort_refs_hits_for_display(prompt=prompt, hits=hits)
            if (not has_pending) and allow_expensive_llm_for_ready and (not prompt_multi_paper_list):
                hits = _maybe_llm_rerank_refs_hits(
                    prompt=prompt,
                    hits=hits,
                    guide_mode=guide_active,
                )
        if hits and (not has_pending) and allow_expensive_llm_for_ready and (not prompt_multi_paper_list):
            hits = _maybe_llm_filter_refs_hits(
                prompt=prompt,
                hits=hits,
                guide_mode=guide_active,
            )
        post_llm_filter_hit_count = int(len(hits))
        if hits and (not has_pending) and allow_expensive_llm_for_ready and (not prompt_multi_paper_list):
            hits = _maybe_polish_refs_card_copy(
                prompt=prompt,
                hits=hits,
                guide_mode=guide_active,
            )
        pack2 = dict(pack)
        pack2["hits"] = hits
        pack2["pipeline_debug"] = {
            "guide_active": bool(guide_active),
            "has_pending": bool(has_pending),
            "raw_hit_count": int(len(raw_hits)),
            "post_score_gate_hit_count": int(post_score_gate_hit_count),
            "post_focus_filter_hit_count": int(post_focus_filter_hit_count),
            "post_llm_filter_hit_count": int(post_llm_filter_hit_count),
            "final_hit_count": int(len(hits)),
            "filtered_self_hit_count": int(filtered_self_hits),
            "prompt_requires_explicit_focus_match": bool(prompt_requires_focus_match),
            "prompt_likely_cross_paper_refs": bool(prompt_cross_paper_refs),
            "prompt_explicitly_requests_multi_paper_list": bool(prompt_multi_paper_list),
        }
        if guide_active:
            hidden_self_source = bool(filtered_self_hits > 0)
            if (not hidden_self_source) and _prompt_likely_cross_paper_refs(prompt):
                # Cross-paper guide refs intentionally exclude the bound paper even if
                # it was removed earlier in the pipeline before this UI filter stage.
                hidden_self_source = True
            pack2["guide_filter"] = {
                "active": True,
                "hidden_self_source": hidden_self_source,
                "filtered_hit_count": int(filtered_self_hits),
                "guide_source_path": guide_source_path_norm,
                "guide_source_name": guide_source_name_norm or _source_filename(guide_source_path_norm),
            }
        out[int(user_msg_id)] = _attach_pack_display_contract(pack2)
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
    aid = _extract_arxiv_id_like(s)
    if aid:
        return _arxiv_doi_from_id(aid).lower()
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s, flags=re.I)
    s = s.strip(" \t\r\n.,;:()[]{}<>")
    return s


_ARXIV_ID_RE = re.compile(r"\barxiv\s*[:\s]\s*(\d{4}\.\d{4,5})(?:v\d+)?\b", flags=re.I)
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", flags=re.I)
_ARXIV_DOI_RE = re.compile(r"10\.48550/arxiv[.:](\d{4}\.\d{4,5})(?:v\d+)?", flags=re.I)


def _extract_arxiv_id_like(value: str) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    for pattern in (_ARXIV_ID_RE, _ARXIV_URL_RE, _ARXIV_DOI_RE):
        m = pattern.search(s)
        if m:
            aid = str(m.group(1) or "").strip()
            if aid:
                return aid
    return ""


def _arxiv_doi_from_id(arxiv_id: str) -> str:
    aid = str(arxiv_id or "").strip()
    if not aid:
        return ""
    return f"10.48550/arXiv.{aid}"


def _arxiv_backfill_meta_from_texts(*values: str) -> dict:
    aid = ""
    for raw in values:
        aid = _extract_arxiv_id_like(raw)
        if aid:
            break
    if not aid:
        return {}
    doi = _arxiv_doi_from_id(aid)
    if not doi:
        return {}
    return {
        "doi": doi,
        "doi_url": build_doi_url(doi),
        "arxiv_id": aid,
        "arxiv_url": f"https://arxiv.org/abs/{aid}",
        "match_method": "arxiv_doi_backfill",
    }


def _normalize_title_for_openalex_search(value: str) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    return s[:240].strip()


def _title_similarity_for_openalex(a: str, b: str) -> float:
    na = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(a or "").lower()).strip()
    nb = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(b or "").lower()).strip()
    if not na or not nb:
        return 0.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    ta = set(na.split())
    tb = set(nb.split())
    jac = (len(ta & tb) / len(ta | tb)) if ta and tb else 0.0
    return float(min(1.0, 0.70 * seq + 0.30 * jac))


def _openalex_arxiv_meta_by_title(title: str) -> dict:
    query = _normalize_title_for_openalex_search(title)
    if len(query) < 8:
        return {}
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={"search": query, "per-page": 8},
            timeout=6.0,
            headers={"User-Agent": "Pi-zaya-KB/1.0"},
        )
        if r.status_code != 200:
            return {}
        payload = r.json() or {}
    except Exception:
        return {}
    results = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(results, list) or not results:
        return {}

    best: dict = {}
    best_score = 0.0
    for item in results:
        if not isinstance(item, dict):
            continue
        cand_title = str(item.get("title") or "").strip()
        doi_url = str(item.get("doi") or "").strip()
        if not doi_url:
            continue
        doi_norm = _normalize_doi_like(doi_url)
        if not doi_norm:
            continue
        arxiv_id = _extract_arxiv_id_like(doi_norm) or _extract_arxiv_id_like(str(item.get("ids") or ""))
        if not arxiv_id and ("arxiv" not in doi_norm.lower()):
            continue
        sim = _title_similarity_for_openalex(query, cand_title)
        if sim > best_score:
            best_score = sim
            best = item
    if best_score < 0.84 or not isinstance(best, dict):
        return {}

    doi_norm = _normalize_doi_like(str(best.get("doi") or "").strip())
    if not doi_norm:
        return {}
    out: dict[str, object] = {
        "doi": doi_norm,
        "doi_url": build_doi_url(doi_norm),
        "match_method": "openalex_title_arxiv",
    }
    pub_year = str(best.get("publication_year") or "").strip()
    if pub_year:
        out["year"] = pub_year
    primary_location = best.get("primary_location")
    if isinstance(primary_location, dict):
        source = primary_location.get("source")
        if isinstance(source, dict):
            venue_name = str(source.get("display_name") or "").strip()
            if venue_name:
                out["venue"] = venue_name
    return out


def _should_try_openalex_arxiv_title(meta: dict, *, raw: str) -> bool:
    title = str((meta or {}).get("title") or "").strip()
    if len(title) < 8:
        return False
    venue = str((meta or {}).get("venue") or "").strip().lower()
    s = f"{raw}\n{title}\n{venue}"
    if _extract_arxiv_id_like(s):
        return True
    if "arxiv" in s.lower():
        return True
    return False


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
    if not re.search(
        r"(鎻愬嚭|璁捐|鏋勫缓|閲囩敤|寮曞叆|瀹炵幇|姣旇緝|鍒嗘瀽|璇勪及|develop|propose|introduce|present|compare|analy[sz]e|evaluat)",
        s,
        flags=re.I,
    ):
        return False
    if not re.search(
        r"(缁撴灉|鏄剧ず|鎻愬崌|闄嶄綆|宸紓|浼樺姡|鍔犻€焲浼樹簬|鏈夋晥|鎬ц兘|瀹為獙|result|show|improv|outperform|achiev|difference|trade-?off|advantage|limitation)",
        s,
        flags=re.I,
    ):
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
        elif existing_source == "abstract":
            final_line, generation = _finalize_abstract_summary_line(title=title, abstract_text=existing_line)
            out["summary_line"] = final_line or _translate_summary_to_zh(existing_line)
            out["summary_source"] = "abstract"
            out["summary_generation"] = generation or "translated_abstract"
            return out
        else:
            out["summary_line"] = _translate_summary_to_zh(existing_line)
            out["summary_source"] = existing_source if existing_source in {"fulltext", "abstract", "metadata"} else "fulltext"
            out["summary_generation"] = "fulltext_existing"
            return out

    if allow_crossref_abstract:
        abstract_line = _summary_from_crossref_abstract(out)
        if abstract_line:
            final_line, generation = _finalize_abstract_summary_line(title=title, abstract_text=abstract_line)
            out["summary_line"] = final_line or _translate_summary_to_zh(abstract_line)
            out["summary_source"] = "abstract"
            out["summary_generation"] = generation or "translated_abstract"
            return out
        openalex_line = _summary_from_openalex_abstract(out)
        if openalex_line:
            final_line, generation = _finalize_abstract_summary_line(title=title, abstract_text=openalex_line)
            out["summary_line"] = final_line or _translate_summary_to_zh(openalex_line)
            out["summary_source"] = "abstract"
            out["summary_generation"] = generation or "translated_abstract"
            return out

    fallback = _metadata_summary_line(out)
    if fallback:
        out["summary_line"] = fallback
        out["summary_source"] = "metadata"
        out["summary_generation"] = "metadata_only"
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
        arxiv_backfill = _arxiv_backfill_meta_from_texts(s)
        if arxiv_backfill:
            out.update(arxiv_backfill)

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

    arxiv_backfill0 = _arxiv_backfill_meta_from_texts(
        str(meta.get("doi") or ""),
        str(meta.get("doi_url") or ""),
        str(meta.get("raw") or ""),
        str(meta.get("cite_fmt") or ""),
        str(meta.get("title") or ""),
        str(meta.get("venue") or ""),
    )
    if arxiv_backfill0 and not _normalize_doi_like(str(meta.get("doi") or meta.get("doi_url") or "")):
        meta = _merge_meta_prefer_richer(meta, arxiv_backfill0)

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
    if not _normalize_doi_like(str(meta.get("doi") or meta.get("doi_url") or "")):
        arxiv_backfill1 = _arxiv_backfill_meta_from_texts(
            str(meta.get("raw") or raw0 or ""),
            str(meta.get("cite_fmt") or ""),
            str(meta.get("title") or title or ""),
            str(meta.get("venue") or venue or ""),
        )
        if arxiv_backfill1:
            meta = _merge_meta_prefer_richer(meta, arxiv_backfill1)
    if not _normalize_doi_like(str(meta.get("doi") or meta.get("doi_url") or "")):
        if _should_try_openalex_arxiv_title(meta, raw=raw0 or raw):
            openalex_arxiv = _openalex_arxiv_meta_by_title(str(meta.get("title") or title or ""))
            if openalex_arxiv:
                meta = _merge_meta_prefer_richer(meta, openalex_arxiv)
    enriched = _enrich_bibliometrics(meta)
    if isinstance(enriched, dict):
        meta = enriched
    return _ensure_summary_line(meta, allow_crossref_abstract=True)
