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
import time

from api.deps import load_prefs
from api.reference_card_copy import (
    finalize_ref_card_copy as _finalize_ref_card_copy,
    looks_generic_ref_why_line as _card_copy_looks_generic_ref_why_line,
    looks_templated_ref_why_line as _card_copy_looks_templated_ref_why_line,
)
from api.reference_card_payload import build_ref_card_ui_payload as _build_ref_card_ui_payload
from api.reference_card_quality import (
    LLM_SUMMARY_GENERATIONS,
    LLM_WHY_GENERATIONS,
    attach_refs_pack_polish_contract,
    ref_card_polish_status,
    refs_pack_has_full_llm_copy,
)
from api.reference_intent import (
    refs_prompt_section_intent as _intent_prompt_section_intent,
    refs_prompt_topic_terms as _intent_prompt_topic_terms,
    refs_section_intent_heading_score as _intent_section_intent_heading_score,
    refs_section_intent_terms as _intent_section_intent_terms,
)
from kb.answer_contract import _prefer_zh_locale
from kb.config import load_settings
from kb.citation_meta import (
    fetch_best_crossref_for_reference,
    fetch_best_crossref_meta,
    fetch_crossref_work_by_doi,
    title_similarity,
)
from kb.evidence_text import clean_display_text as _clean_evidence_display_text
from kb.evidence_text import pick_readable_evidence_text as _pick_readable_evidence_text
from kb.file_naming import citation_meta_display_pdf_name
from kb.library_store import LibraryStore
from kb.llm import DeepSeekChat
from kb.reference_query_family import (
    extract_multi_paper_topic as _shared_extract_multi_paper_topic,
    prompt_explicitly_requests_multi_paper_list as _shared_prompt_explicitly_requests_multi_paper_list,
    prompt_explicitly_requests_single_paper_pick as _shared_prompt_explicitly_requests_single_paper_pick,
    prompt_likely_multi_paper_synthesis as _shared_prompt_likely_multi_paper_synthesis,
    prompt_reference_focus_action as _shared_prompt_reference_focus_action,
    prompt_requests_reference_compare as _shared_prompt_requests_reference_compare,
    prompt_requests_reference_definition as _shared_prompt_requests_reference_definition,
    prompt_requires_reference_focus_match as _shared_prompt_requires_reference_focus_match,
    prompt_targets_sci_topic as _shared_prompt_targets_sci_topic,
)
from kb.source_blocks import extract_equation_number, extract_figure_number, load_source_blocks, match_source_blocks
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
    return "auto"


def _refs_card_ui_locale_pref() -> str:
    try:
        prefs = load_prefs()
    except Exception:
        prefs = {}
    raw = str((prefs or {}).get("ui_locale") or "").strip().lower()
    return raw if raw in {"zh", "en"} else ""


def _ref_card_user_locale(prompt: str = "", *fallback_texts: str) -> str:
    pref = _refs_card_locale_pref()
    if pref in {"zh", "en"}:
        return pref

    prompt_text = str(prompt or "").strip()
    if prompt_text:
        if _prefer_zh_locale(prompt_text):
            return "zh"
        if _prompt_strongly_prefers_english(prompt_text):
            return "en"

    ui_pref = _refs_card_ui_locale_pref()
    if ui_pref in {"zh", "en"}:
        return ui_pref

    fallback_parts = [str(text or "").strip() for text in fallback_texts if str(text or "").strip()]
    if fallback_parts:
        return "zh" if _prefer_zh_locale(*fallback_parts) else "en"
    return "en"


def _prefer_zh_ref_card_locale(*texts: str) -> bool:
    prompt = str(texts[0] or "") if texts else ""
    fallback_texts = tuple(str(text or "") for text in texts[1:]) if len(texts) >= 2 else ()
    return _ref_card_user_locale(prompt, *fallback_texts) == "zh"


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
    if str(meta.get("ref_display_reason") or "").strip().lower() == "answer_hit_top":
        return True
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
    except Exception as _exc:
        if _DEV_MODE:
            _print_flush(f"[refs] display_name fallback for {str(source_path or '')[-80:]}: {_exc}")

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
            if re.search(rf"(equation|eq\.?)\s*[\(#\[]?\s*{escaped_num}\b|公式\s*[\(#\[]?\s*{escaped_num}(?!\d)", cand, flags=re.I):
                score += 6.0
        elif kind == "figure":
            if re.search(rf"(figure|fig\.?)\s*[\(#\[]?\s*{escaped_num}\b|图\s*[\(#\[]?\s*{escaped_num}(?!\d)", cand, flags=re.I):
                score += 6.0
        elif kind == "table":
            if re.search(rf"table\s*[\(#\[]?\s*{escaped_num}\b|表\s*[\(#\[]?\s*{escaped_num}(?!\d)", cand, flags=re.I):
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
    protected_space = "__REF_SUMMARY_ABBR_SPACE__"
    raw = re.sub(
        r"\b(Figs?|Eq|Eqs|Secs?|Refs?)\.\s+",
        lambda m: f"{m.group(1)}.{protected_space}",
        raw,
        flags=re.I,
    )
    parts = [
        str(part or "").replace(protected_space, " ").strip()
        for part in re.split(r"(?<=[.!?。！？;；])\s+", raw)
        if str(part or "").strip()
    ]
    return parts[: max(1, int(max_sentences or 8))]


def _trim_definition_clause(text: str) -> str:
    return re.sub(r"[\s,;:.!?]+$", "", str(text or "").strip())


def _definition_followup_clause(text: str) -> str:
    raw = _trim_definition_clause(text)
    if not raw:
        return ""
    raw = re.sub(
        r"^(?:consequently|therefore|thus|hence|as a result|accordingly)\s*,?\s*",
        "",
        raw,
        flags=re.I,
    ).strip()
    if not raw:
        return ""
    if re.match(r"^(?:if|because|however|but|moreover|additionally|furthermore)\b", raw, flags=re.I):
        return ""
    return raw[:1].lower() + raw[1:] if len(raw) > 1 else raw.lower()


def _definition_prompt_summary_rewrites(
    *,
    prompt: str,
    heading: str = "",
    sentence: str,
    next_sentence: str = "",
) -> list[str]:
    if not _is_definition_focus_prompt(prompt):
        return []
    sent = _clean_summary_line(sentence)
    if not sent:
        return []
    if _looks_surface_like_ref_summary(sent) or _looks_formula_heavy_ref_text(sent):
        return []
    focus_terms = _render_focus_terms_for_ref_card(prompt, max_n=1)
    if not focus_terms:
        return []
    focus_term = str(focus_terms[0] or "").strip()
    if not focus_term:
        return []
    display_term = _display_focus_term_for_ref_card(prompt, focus_term)
    surface = " ".join(part for part in (heading, sent, next_sentence) if part)
    if not _focus_term_matches_surface(focus_term, surface):
        keyword_hits = _refs_summary_focus_keyword_hit_count(prompt, surface)
        informative_keywords = _refs_prompt_informative_focus_keywords(prompt)
        min_keyword_hits = min(2, len(informative_keywords)) if informative_keywords else 1
        if keyword_hits < max(1, min_keyword_hits):
            return []

    follow = _definition_followup_clause(next_sentence)
    rewrites: list[str] = []
    prefer_zh = _prefer_zh_ref_card_locale(prompt, heading, sent, next_sentence)

    m_if = re.match(r"^\s*if\s+(.+?),\s*then\s+(.+?)(?:[.!?]|$)", sent, flags=re.I)
    if m_if:
        cond = _trim_definition_clause(m_if.group(1))
        outcome = _trim_definition_clause(m_if.group(2))
        if cond and outcome:
            detail = outcome
            if follow and len(f"{outcome} and {follow}") <= 220:
                detail = f"{outcome} and {follow}"
            rewrites.append(
                f"该文将“{display_term}”解释为：当 {cond} 时，{detail}。"
                if prefer_zh
                else f"The paper defines {display_term} by showing that when {cond}, {detail}."
            )

    m_because = re.match(r"^\s*because\s+(.+?),\s+(.+?)(?:[.!?]|$)", sent, flags=re.I)
    if m_because:
        reason = _trim_definition_clause(m_because.group(1))
        result = _trim_definition_clause(m_because.group(2))
        if reason and result:
            rewrites.append(
                f"该文将“{display_term}”解释为：由于 {reason}，{result}。"
                if prefer_zh
                else f"The paper defines {display_term} by explaining that because {reason}, {result}."
            )

    m_known = re.match(
        r"^\s*(?:this|the)\s+(?:technique|approach|method|strategy|process|scheme)\s+"
        r"is\s+(?:known|defined)\s+as\s+(.+?)(?:[.!?]|$)",
        sent,
        flags=re.I,
    )
    if m_known:
        alias = _trim_definition_clause(m_known.group(1))
        if alias:
            rewrites.append(
                f"该文将“{display_term}”定义为 {alias}。"
                if prefer_zh
                else f"The paper defines {display_term} as {alias}."
            )

    if not rewrites and (not _focus_term_matches_surface(focus_term, sent)):
        clause = _trim_definition_clause(sent)
        if clause and (
            not re.match(
                r"^(?:however|but|moreover|additionally|furthermore|therefore|thus|consequently|hence|accordingly)\b",
                clause,
                flags=re.I,
            )
        ):
            clause = clause[:1].lower() + clause[1:] if len(clause) > 1 else clause.lower()
            rewrites.append(
                f"该文将“{display_term}”解释为：{clause}。"
                if prefer_zh
                else f"The paper defines {display_term} by explaining that {clause}."
            )

    out: list[str] = []
    seen: set[str] = set()
    for item in rewrites:
        cand = _clean_summary_line(item)
        if not cand:
            continue
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


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
    sentences = _split_ref_summary_sentences(body or text, max_sentences=24)
    candidates: list[str] = []
    seen: set[str] = set()
    definition_prompt = _is_definition_focus_prompt(prompt)

    def _push(candidate_text: str) -> None:
        # Skip LLM translation during candidate expansion — translating every
        # variant (40+ per snippet) would make 100+ blocking LLM calls per hit.
        # The caller translates the final selected candidate once via
        # _align_ref_card_copy_to_user_locale instead.
        cand = _normalize_ref_summary_candidate(
            candidate_text,
            title=title,
            prefer_zh=prefer_zh,
            allow_llm_translate=False,
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
    for idx, sent in enumerate(sentences):
        next_sentence = sentences[idx + 1] if (idx + 1) < len(sentences) else ""
        _push(sent)
        for rewrite in _definition_prompt_summary_rewrites(
            prompt=prompt,
            heading=heading,
            sentence=sent,
            next_sentence=next_sentence,
        ):
            _push(rewrite)
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
                if not heading_has_term:
                    continue
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


def _source_block_matches_anchor_target(
    *,
    block_kind: str,
    block_number: int,
    block_text: str,
    heading_path: str,
    anchor_target_kind: str,
    anchor_target_number: int,
) -> bool:
    kind = str(anchor_target_kind or "").strip().lower()
    num = _positive_int(anchor_target_number)
    if not kind or num <= 0:
        return True
    if str(block_kind or "").strip().lower() == kind and _positive_int(block_number) == num:
        return True
    if _refs_heading_anchor_number(kind, heading_path) == num:
        return True
    text = str(block_text or "").strip()
    if kind == "figure":
        return extract_figure_number(text) == num
    if kind == "equation":
        return extract_equation_number(text) == num
    if kind == "table":
        return bool(re.search(rf"\btable\s*[\(#\[]?\s*{re.escape(str(num))}\b|表\s*[\(#\[]?\s*{re.escape(str(num))}(?!\d)", text, flags=re.I))
    return False


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


def _ref_summary_core_clause(text: str) -> str:
    raw = _clean_summary_line(text)
    if not raw:
        return ""
    if ":" not in raw:
        return raw
    prefix_raw, suffix_raw = raw.split(":", 1)
    suffix = str(suffix_raw or "").strip()
    if not suffix:
        return raw
    prefix_norm = _normalize_title_identity(prefix_raw)
    prefix_tokens = [
        tok
        for tok in prefix_norm.split()
        if tok and tok not in _PROMPT_FOCUS_STOPWORDS
    ]
    if prefix_tokens and len(prefix_tokens) <= 6:
        return suffix
    if len(str(prefix_raw or "").strip()) <= 48 and len(str(prefix_raw or "").split()) <= 6:
        return suffix
    return raw


def _looks_fragmentary_ref_summary(text: str) -> bool:
    s = _clean_summary_line(text)
    if not s:
        return False
    core = _ref_summary_core_clause(s)
    if not core:
        return False
    lower = core.lower()
    if re.search(r"[,;:\-/(]$", core):
        return True
    if core.count("(") > core.count(")") or core.count("[") > core.count("]"):
        return True
    if re.match(
        r"^(?:and|or|but|while|whereas|although|though|however|moreover|additionally|furthermore|therefore|thus|which|that|who|whose|whom)\b",
        lower,
        flags=re.I,
    ):
        return True
    if re.match(
        r"^(?:of|for|to|with|by|from|into|onto|between|among|across|through|within|without|under|over|than)\b",
        lower,
        flags=re.I,
    ):
        if (
            "," not in core[:48]
            and (not _has_ref_summary_explainer_signal(core))
            and (not _has_ref_summary_value_signal(core))
        ):
            return True
    if re.match(r"^[a-z]", core):
        if re.search(r"[A-Za-z]{2,}", core) and (not _looks_natural_language_ref_summary(core)):
            return True
    return False


def _looks_why_like_ref_summary(text: str) -> bool:
    s = _clean_summary_line(text)
    if not s:
        return False
    lower = s.lower()
    if re.search(
        r"\b(this hit|good entry point|directly relevant|aligns with the core concept|strong match for the comparison request)\b",
        lower,
        flags=re.I,
    ):
        return True
    if re.match(
        r"^[A-Za-z0-9 .:/&+\-]{6,120}[”\"](?:\u8ba8\u8bba\u4e86|\u6bd4\u8f83\u4e86|\u5b9a\u4e49\u6216\u89e3\u91ca\u4e86)",
        s,
    ):
        return True
    return bool(
        re.search(
            r"(\u9002\u5408\u4f5c\u4e3a\u5b9a\u4f4d\u5165\u53e3|\u76f4\u63a5\u8986\u76d6\u4e86|\u76f4\u63a5\u5b9a\u4e49\u6216\u89e3\u91ca\u4e86|\u548c\u5f53\u524d.{0,24}\u95ee\u9898|\u6b63\u5bf9\u9f50)",
            s,
        )
    )


def _looks_location_only_ref_summary(text: str) -> bool:
    s = _clean_summary_line(text)
    if not s:
        return False
    lower = s.lower()
    if re.search(
        r"\b(the\s+)?relevant discussion appears\b|\bthis hit (?:falls under|lands in)\b",
        lower,
    ):
        return True
    if re.search(r"(相关内容位于|命中落在|定位到|位于“[^”]{1,160}”)", s):
        return True
    return False


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
    if _looks_location_only_ref_summary(summary):
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
    if _looks_focus_prefixed_ref_summary(prompt, summary):
        focus_action = _shared_prompt_reference_focus_action(prompt)
        score -= 5.0 if focus_action in {"compare", "define"} else 2.2
    if _looks_prefixed_heading_shell_ref_summary(summary):
        score -= 3.2
    if _looks_fragmentary_ref_summary(summary):
        score -= 3.4
    if _looks_why_like_ref_summary(summary):
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
        if not _source_block_matches_anchor_target(
            block_kind=block_kind,
            block_number=_positive_int(block.get("number")),
            block_text=block_text,
            heading_path=heading_path,
            anchor_target_kind=anchor_target_kind,
            anchor_target_number=anchor_target_number,
        ):
            continue
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
                        "block_id": str(block.get("block_id") or "").strip(),
                        "anchor_id": str(block.get("anchor_id") or "").strip(),
                        "block_kind": block_kind,
                        "block_number": _positive_int(block.get("number")),
                        "block_text": block_text,
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
    "直接对应",
    "直接讨论",
    "直接相关",
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
    if _card_copy_looks_generic_ref_why_line(s):
        return True
    lower = s.lower()
    if "..." in s and re.search(r"\b(which|what|where|how|why)\b", lower):
        return True
    if re.search(r"\b(which paper|in my library|point me to|source section)\b", lower):
        return True
    if re.search(r"(定义、方法或结果信息)", s):
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
        return f"“{loc}”里有“{'、'.join(matched_terms)}”的原文线索，可用来核对它在文中的定义、方法或结果。"
    if matched_terms:
        return f"这段证据围绕“{'、'.join(matched_terms)}”展开，可用来判断论文如何使用这个概念。"
    if loc:
        return f"可查看“{loc}”里的定义、方法或结果线索。"
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
            return f"“{loc}”里有“{' / '.join(matched_terms)}”的原文线索，可用来核对它在文中的定义、方法或结果。"
        display_terms = [_display_focus_term_for_ref_card(prompt, item) for item in matched_terms]
        return f"'{loc}' contains evidence about '{' / '.join(display_terms)}', useful for checking how the paper defines or uses it."
    if matched_terms:
        if prefer_zh:
            return f"这段证据围绕“{' / '.join(matched_terms)}”展开，可用来判断论文如何使用这个概念。"
        display_terms = [_display_focus_term_for_ref_card(prompt, item) for item in matched_terms]
        return f"The evidence focuses on '{' / '.join(display_terms)}', so it can ground how the paper uses that concept."
    if loc:
        if prefer_zh:
            return f"可查看“{loc}”里的定义、方法或结果线索。"
        return f"Check '{loc}' for the paper's definitions, methods, or result evidence."
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
            return f"“{loc or heading_path or '该小节'}”里比较了“{pair}”，适合用来核对这组差异。"
        return f"'{loc or heading_path or 'this section'}' compares '{pair}', which can ground the contrast."
    if matched_terms and _shared_prompt_requests_reference_definition(prompt):
        term = _display_focus_term_for_ref_card(prompt, matched_terms[0])
        if prefer_zh:
            return f"“{loc or heading_path or '该小节'}”里定义或解释了“{term}”，可用来确认概念含义。"
        return f"'{loc or heading_path or 'this section'}' defines or explains '{term}', useful for checking the concept."
    if loc and matched_terms:
        if prefer_zh:
            return f"“{loc}”里有“{' / '.join(matched_terms)}”的原文线索，可用来核对它在文中的定义、方法或结果。"
        display_terms = [_display_focus_term_for_ref_card(prompt, item) for item in matched_terms]
        return f"'{loc}' contains evidence about '{' / '.join(display_terms)}', useful for checking how the paper defines or uses it."
    if matched_terms:
        if prefer_zh:
            return f"这段证据围绕“{' / '.join(matched_terms)}”展开，可用来判断论文如何使用这个概念。"
        display_terms = [_display_focus_term_for_ref_card(prompt, item) for item in matched_terms]
        return f"The evidence focuses on '{' / '.join(display_terms)}', so it can ground how the paper uses that concept."
    if loc:
        if prefer_zh:
            return f"可查看“{loc}”里的定义、方法或结果线索。"
        return f"Check '{loc}' for the paper's definitions, methods, or result evidence."
    return ""


def _build_prompt_aligned_ref_summary_fallback(
    *,
    prompt: str,
    display_name: str,
    heading_path: str,
    summary_line: str,
    why_line: str,
) -> str:
    prefer_zh = _prefer_zh_ref_card_locale(prompt, display_name, heading_path, summary_line, why_line)
    loc = " / ".join(part for part in str(heading_path or "").split(" / ") if part).strip()
    evidence_summary = _build_evidence_backed_ref_summary_from_seed(
        prompt=prompt,
        title=display_name,
        summary_line=summary_line,
        prefer_zh=prefer_zh,
    )
    if evidence_summary:
        return evidence_summary
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
    focus_action = _shared_prompt_reference_focus_action(prompt)
    if focus_action == "compare" and len(matched_terms) >= 2:
        pair = " / ".join(_display_focus_term_for_ref_card(prompt, item) for item in matched_terms[:2])
        if prefer_zh:
            return f"该文在“{loc or heading_path or '相关小节'}”比较了“{pair}”。"
        return f"The paper compares '{pair}' in '{loc or heading_path or 'this section'}'."
    if focus_action == "define" and matched_terms:
        term = _display_focus_term_for_ref_card(prompt, matched_terms[0])
        if prefer_zh:
            return f"该文在“{loc or heading_path or '相关小节'}”定义或解释了“{term}”。"
        return f"The paper defines or explains '{term}' in '{loc or heading_path or 'this section'}'."
    if matched_terms:
        terms = " / ".join(_display_focus_term_for_ref_card(prompt, item) for item in matched_terms)
        if prefer_zh:
            return f"该文在“{loc or heading_path or '相关小节'}”讨论了“{terms}”。"
        return f"The paper discusses '{terms}' in '{loc or heading_path or 'this section'}'."
    if loc:
        if prefer_zh:
            return f"可查看“{loc}”里的原文线索。"
        return f"Check the source evidence in '{loc}'."
    return ""


def _build_evidence_backed_ref_summary_from_seed(
    *,
    prompt: str,
    title: str,
    summary_line: str,
    prefer_zh: bool,
) -> str:
    seed = _pick_focus_sentence_ref_summary_seed(
        prompt=prompt,
        title=title,
        summary_line=summary_line,
    ) or _summary_excerpt(summary_line, max_sentences=2, max_len=240)
    if not seed:
        return ""
    if _looks_location_only_ref_summary(seed):
        return ""
    if _looks_like_title_echo(seed, title):
        return ""
    if (
        _looks_surface_like_ref_summary(seed)
        or _looks_prefixed_heading_shell_ref_summary(seed)
        or _looks_fragmentary_ref_summary(seed)
        or _looks_why_like_ref_summary(seed)
        or _looks_formula_heavy_ref_text(seed)
    ):
        return ""
    if len(seed) < 32:
        return ""
    if not (_looks_natural_language_ref_summary(seed) or _is_summary_quality_ok(seed)):
        return ""
    if prefer_zh and _ref_copy_clear_locale(seed) == "en":
        compact = _summary_excerpt(seed, max_sentences=2, max_len=260).strip()
        if not compact:
            return ""
        return f"原文片段写到：“{compact}”"
    return seed


def _pick_focus_sentence_ref_summary_seed(*, prompt: str, title: str, summary_line: str) -> str:
    sentences = _split_ref_summary_sentences(summary_line, max_sentences=24)
    if not sentences:
        return ""
    ranked: list[tuple[float, str]] = []
    for sent in sentences:
        cand = _summary_excerpt(sent, max_sentences=1, max_len=240)
        if not cand:
            continue
        if _looks_like_title_echo(cand, title):
            continue
        focus_score = _ref_summary_focus_score(
            prompt=prompt,
            source_path="",
            title=title,
            text=cand,
            anchor_target_kind="",
            anchor_target_number=0,
        )
        if focus_score < 2.0:
            continue
        ranked.append((focus_score, cand))
    if not ranked:
        return ""
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def _ref_copy_clear_locale(text: str) -> str:
    has_cjk = _has_cjk_text(text)
    has_latin = _has_latin_text(text)
    if has_cjk and (not has_latin):
        return "zh"
    if has_latin and (not has_cjk):
        return "en"
    return ""


def _align_ref_card_copy_to_user_locale(
    *,
    prompt: str,
    display_name: str,
    heading_path: str,
    summary_line: str,
    why_line: str,
    summary_kind: str,
    allow_llm_translate: bool,
) -> tuple[str, str]:
    target_locale = _ref_card_user_locale(prompt, display_name, heading_path, summary_line, why_line)
    summary_out = _normalize_ref_copy_text(str(summary_line or "").strip())
    why_out = _normalize_ref_copy_text(str(why_line or "").strip())

    if summary_out and _ref_copy_clear_locale(summary_out) not in {"", target_locale}:
        localized_summary = ""
        if target_locale == "zh" and allow_llm_translate:
            localized_summary = _translate_summary_to_zh(summary_out)
        if (not localized_summary) and str(summary_kind or "").strip().lower() != "metadata":
            localized_summary = _build_prompt_aligned_ref_summary_fallback(
                prompt=prompt,
                display_name=display_name,
                heading_path=heading_path,
                summary_line=summary_out,
                why_line=why_out,
            )
        localized_summary = _normalize_ref_copy_text(str(localized_summary or "").strip())
        if (
            localized_summary
            and _looks_location_only_ref_summary(localized_summary)
            and summary_out
            and (not _looks_location_only_ref_summary(summary_out))
        ):
            localized_summary = ""
        if localized_summary and _ref_copy_clear_locale(localized_summary) in {"", target_locale}:
            summary_out = localized_summary

    if why_out and _ref_copy_clear_locale(why_out) not in {"", target_locale}:
        localized_why = _build_prompt_aligned_ref_why_line_v3(
            prompt=prompt,
            display_name=display_name,
            heading_path=heading_path,
            summary_line=summary_out,
            why_line=why_out,
        )
        localized_why = _normalize_ref_copy_text(str(localized_why or "").strip())
        if localized_why and _ref_copy_clear_locale(localized_why) in {"", target_locale}:
            why_out = localized_why

    return summary_out, why_out


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
    prefer_zh = _ref_card_user_locale(prompt, summary_line) == "zh"
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
    if generation == "llm_pack":
        return {
            "summary_generation": "llm_pack",
            "summary_basis": "基于检索命中证据的 LLM 提炼" if prefer_zh else "LLM-distilled from retrieval evidence",
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
    prefer_zh = _ref_card_user_locale(prompt, why_line) == "zh"
    generation = str(why_generation or "").strip().lower()
    if generation == "llm_pack":
        return {
            "why_generation": "llm_pack",
            "why_basis": "基于检索命中证据的 LLM 相关性说明" if prefer_zh else "LLM-grounded relevance from retrieval evidence",
        }
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
    prefer_zh = _ref_card_user_locale(prompt, summary_line) == "zh"
    kind = str(summary_kind or "").strip().lower()
    if kind == "abstract":
        return {
            "summary_kind": "abstract",
            "summary_label": "摘要" if prefer_zh else "Abstract",
            "summary_title": "这篇文献的核心内容" if prefer_zh else "What This Paper Covers",
        }
    if kind == "metadata":
        return {
            "summary_kind": "metadata",
            "summary_label": "信息卡" if prefer_zh else "Meta",
            "summary_title": "可用文献信息" if prefer_zh else "Available Bibliographic Info",
        }
    return {
        "summary_kind": "guide",
        "summary_label": "导读" if prefer_zh else "Guide",
        "summary_title": "这条证据说明什么" if prefer_zh else "What This Evidence Shows",
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
    if _looks_location_only_ref_summary(s):
        return False
    if _looks_formula_heavy_ref_text(s):
        return False
    if _looks_surface_like_ref_summary(s):
        return False
    if _looks_fragmentary_ref_summary(s):
        return False
    if _looks_why_like_ref_summary(s):
        return False
    if _looks_focus_prefixed_ref_summary(prompt, s):
        focus_action = _shared_prompt_reference_focus_action(prompt)
        core = _ref_summary_core_clause(s)
        if focus_action in {"compare", "define"}:
            return False
        if re.match(r"^(?:because|if|when|while|since|as|figure|fig\.?|table|eq(?:uation)?)\b", core, flags=re.I):
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
    if _looks_location_only_ref_summary(cand):
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
    if _looks_fragmentary_ref_summary(cand):
        score -= 3.4
    if _looks_why_like_ref_summary(cand):
        score -= 3.2
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


def _llm_select_best_evidence_candidate(
    *,
    prompt: str,
    title: str,
    candidates: list[str],
) -> str:
    """Ask the LLM to pick the most informative evidence snippet from a shortlist."""
    if (not prompt) or (len(candidates) < 2):
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
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 6.0),
            max_retries=0,
        )
    except Exception:
        fast_settings = settings
    candidate_lines = "\n".join(
        f"- [{i}] {str(candidate or '').strip()[:300]}"
        for i, candidate in enumerate(candidates[:3])
    )
    try:
        ds = DeepSeekChat(fast_settings)
        out = (
            ds.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are selecting the best evidence snippet for a research reference card. "
                            "Return only the index number (e.g. 0, 1, or 2) of the most informative, "
                            "specific, and query-relevant snippet. Prefer snippets that name concrete "
                            "methods, findings, or contributions over vague location descriptions."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"User query: {str(prompt or '').strip()}\n"
                            f"Paper: {str(title or '').strip()}\n"
                            f"Snippets:\n{candidate_lines}\n"
                            f"Best snippet index:"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=8,
            )
            or ""
        ).strip()
    except Exception:
        return ""
    for ch in out:
        if ch.isdigit():
            idx = int(ch)
            if 0 <= idx < len(candidates):
                return str(candidates[idx] or "").strip()
    return ""


def _pick_ref_card_summary_fallback(*, prompt: str, title: str, candidates: list[str]) -> str:
    ranked: list[tuple[float, str]] = []
    for raw in candidates or []:
        heading, body = _split_ref_summary_heading_and_body(str(raw or ""))
        sentences = _split_ref_summary_sentences(body or str(raw or ""), max_sentences=4)
        variants: list[str] = []

        cand = _summary_excerpt(str(raw or ""), max_sentences=2, max_len=220)
        if cand:
            variants.append(cand)
        for idx, sent in enumerate(sentences[:2]):
            next_sentence = sentences[idx + 1] if (idx + 1) < len(sentences) else ""
            variants.extend(
                _definition_prompt_summary_rewrites(
                    prompt=prompt,
                    heading=heading,
                    sentence=sent,
                    next_sentence=next_sentence,
                )
            )

        seen: set[str] = set()
        for variant in variants:
            key = str(variant or "").strip().lower()
            if (not key) or key in seen:
                continue
            seen.add(key)
            ranked.append(
                (
                    _ref_card_summary_candidate_score(
                        prompt=prompt,
                        title=title,
                        text=variant,
                    ),
                    variant,
                )
            )
    if not ranked:
        return ""
    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, best = ranked[0]
    if best_score < 1.6:
        return ""
    # When multiple candidates score close to the top, ask the LLM to select the most
    # informative one — the deterministic scoring may not capture semantic fitness.
    close_candidates = [
        candidate
        for score, candidate in ranked[:3]
        if score >= max(1.6, best_score - 0.55)
    ]
    if len(close_candidates) >= 2 and _refs_card_polish_llm_enabled():
        llm_pick = _llm_select_best_evidence_candidate(
            prompt=prompt,
            title=title,
            candidates=close_candidates,
        )
        if llm_pick and any(llm_pick in candidate for candidate in close_candidates):
            return llm_pick
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


def _guide_summary_should_prefer_llm_grounding(
    *,
    prompt: str,
    title: str,
    heading_path: str,
    summary_line: str,
    summary_kind: str,
    summary_generation: str,
    allow_llm_polish: bool,
) -> bool:
    if (not allow_llm_polish) or (str(summary_kind or "").strip().lower() not in ("guide", "section_grounded")):
        return False
    generation = str(summary_generation or "").strip().lower()
    if generation == "llm_grounded":
        return False
    s = _clean_summary_line(summary_line)
    if not s:
        return False
    if not _prefer_zh_ref_card_locale(prompt, title, heading_path, s):
        return False
    if _looks_formula_heavy_ref_text(s) or _looks_why_like_ref_summary(s):
        return True
    if _looks_surface_like_ref_summary(s) or _looks_fragmentary_ref_summary(s):
        return True
    # User explicitly demands LLM polish for all languages including CJK.
    return True


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
        cand = _summary_excerpt(str(raw or ""), max_sentences=2, max_len=220)
        if not cand:
            return
        key = cand.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(cand)

    if isinstance(meta, dict):
        for key, limit in (("ref_show_snippets", 2), ("ref_snippets", 2), ("ref_overview_snippets", 1)):
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


def _normalize_ref_copy_similarity_surface(text: str) -> str:
    raw = _clean_summary_line(text)
    if not raw:
        return ""
    raw = raw.lower()
    raw = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def _ref_copy_similarity_ratio(left: str, right: str) -> float:
    left_norm = _normalize_ref_copy_similarity_surface(left)
    right_norm = _normalize_ref_copy_similarity_surface(right)
    if (not left_norm) or (not right_norm):
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if len(left_norm) >= 36 and len(right_norm) >= 36 and ((left_norm in right_norm) or (right_norm in left_norm)):
        return 0.98
    return difflib.SequenceMatcher(None, left_norm, right_norm).ratio()


def _looks_extractive_ref_card_copy(text: str, *, evidence_snippets: list[str]) -> bool:
    cand_norm = _normalize_ref_copy_similarity_surface(text)
    if len(cand_norm) < 36:
        return False
    cand_tokens = set(cand_norm.split())
    for evidence in evidence_snippets or []:
        evidence_norm = _normalize_ref_copy_similarity_surface(evidence)
        if len(evidence_norm) < 24:
            continue
        if _ref_copy_similarity_ratio(cand_norm, evidence_norm) >= 0.84:
            return True
        if not cand_tokens:
            continue
        evidence_tokens = set(evidence_norm.split())
        smaller = min(len(cand_tokens), len(evidence_tokens))
        if smaller < 6:
            continue
        overlap = len(cand_tokens.intersection(evidence_tokens))
        if (overlap / float(smaller)) >= 0.82:
            return True
    return False


def _looks_templated_llm_ref_why_line(text: str) -> bool:
    s = _clean_summary_line(text)
    if not s:
        return False
    if _card_copy_looks_templated_ref_why_line(s):
        return True
    low = s.lower()
    if any(
        token in low
        for token in (
            "this hit is directly relevant",
            "directly relevant because",
            "good entry point",
            "aligns with the core concept",
            "strong match for the comparison request",
            "directly responds to the user's question",
        )
    ):
        return True
    return any(
        token in s
        for token in (
            "这条命中",
            "本条命中",
            "适合作为定位入口",
            "适合作为导读入口",
            "直接回应用户查询",
            "直接覆盖了",
        )
    )


def _accept_llm_ref_summary_line(
    *,
    prompt: str,
    title: str,
    summary_line: str,
    evidence_snippets: list[str],
) -> str:
    out = _normalize_ref_copy_text(_summary_excerpt(summary_line, max_sentences=2, max_len=300))
    if not out:
        return ""
    if _looks_extractive_ref_card_copy(out, evidence_snippets=evidence_snippets):
        return ""
    if _looks_like_title_echo(out, title) or _looks_formula_heavy_ref_text(out):
        return ""
    if _summary_line_needs_polish(prompt=prompt, title=title, summary_line=out):
        return ""
    return out


def _accept_llm_ref_why_line(
    *,
    prompt: str,
    display_name: str,
    heading_path: str,
    summary_line: str,
    why_line: str,
    evidence_snippets: list[str],
) -> str:
    out = _normalize_ref_copy_text(_summary_excerpt(why_line, max_sentences=2, max_len=240))
    if not out:
        return ""
    if _looks_templated_llm_ref_why_line(out):
        return ""
    if _looks_extractive_ref_card_copy(out, evidence_snippets=evidence_snippets):
        return ""
    if _why_line_needs_polish(
        prompt=prompt,
        display_name=display_name,
        heading_path=heading_path,
        summary_line=summary_line,
        why_line=out,
    ):
        return ""
    return "" if _looks_generic_ref_why_line(out) else out


def _reuse_existing_llm_guide_copy(
    *,
    prompt: str,
    title: str,
    heading_path: str,
    summary_kind: str,
    summary_generation: str,
    why_generation: str,
    summary_line: str,
    why_line: str,
    evidence_snippets: list[str],
) -> tuple[str, str]:
    if str(summary_kind or "").strip().lower() not in ("guide", "section_grounded"):
        return "", ""
    if str(summary_generation or "").strip().lower() not in {"llm_grounded", "llm_pack"}:
        return "", ""
    if str(why_generation or "").strip().lower() not in {"llm_grounded", "llm_pack"}:
        return "", ""
    filtered_evidence = [
        snippet
        for snippet in list(evidence_snippets or [])
        if _ref_copy_similarity_ratio(str(snippet or ""), summary_line) < 0.98
        and _ref_copy_similarity_ratio(str(snippet or ""), why_line) < 0.98
    ]
    effective_evidence = list(filtered_evidence)
    accepted_summary = _accept_llm_ref_summary_line(
        prompt=prompt,
        title=title,
        summary_line=summary_line,
        evidence_snippets=effective_evidence,
    )
    if not accepted_summary:
        return "", ""
    accepted_why = _accept_llm_ref_why_line(
        prompt=prompt,
        display_name=title,
        heading_path=heading_path,
        summary_line=accepted_summary,
        why_line=why_line,
        evidence_snippets=effective_evidence,
    )
    if not accepted_why:
        return "", ""
    return accepted_summary, accepted_why


def _refs_card_polish_llm_enabled() -> bool:
    raw_flag = str(os.environ.get("KB_REFS_CARD_POLISH_USE_LLM", "1") or "").strip().lower()
    if raw_flag in {"0", "false", "off", "no"}:
        return False
    # The generation pipeline already validates the API key before reaching
    # this point, so we trust that the key is available.  Skip the redundant
    # load_settings() check here to avoid silent failures in edge cases.
    return True


def _refs_card_polish_timeout_s(default_s: float = 14.0) -> float:
    try:
        raw = float(str(os.environ.get("KB_REFS_CARD_POLISH_TIMEOUT_S", str(default_s)) or str(default_s)))
    except Exception:
        raw = float(default_s)
    return max(2.0, min(45.0, raw))


def _refs_card_polish_max_retries() -> int:
    try:
        raw = int(str(os.environ.get("KB_REFS_CARD_POLISH_MAX_RETRIES", "1") or "1"))
    except Exception:
        raw = 1
    return max(0, min(2, raw))


def _refs_card_polish_top_n() -> int:
    try:
        raw = int(str(os.environ.get("KB_REFS_CARD_POLISH_TOP_N", "6") or "6"))
    except Exception:
        raw = 6
    return max(0, min(8, raw))




def _is_llm_ref_summary_generation(generation: str) -> bool:
    return str(generation or "").strip().lower() in LLM_SUMMARY_GENERATIONS


def _is_llm_ref_why_generation(generation: str) -> bool:
    return str(generation or "").strip().lower() in LLM_WHY_GENERATIONS


def _ref_card_has_llm_copy(ui_meta: dict | None) -> bool:
    return str(ref_card_polish_status(ui_meta).get("polish_status") or "") == "full"


def _refs_hits_have_llm_copy(hits: list[dict] | None) -> bool:
    return refs_pack_has_full_llm_copy({"hits": [hit for hit in list(hits or []) if isinstance(hit, dict)]})


def _suppress_non_llm_ref_card_copy(
    *,
    prompt: str,
    ui_meta: dict,
) -> dict:
    ui = dict(ui_meta or {})
    heading_path = str(ui.get("heading_path") or "").strip()
    display_name = str(ui.get("display_name") or "").strip()
    prefer_zh = bool(
        _prefer_zh_ref_card_locale(
            prompt,
            display_name,
            heading_path,
            str(ui.get("summary_line") or ""),
            str(ui.get("why_line") or ""),
        )
    )
    existing_summary = _normalize_ref_copy_text(str(ui.get("summary_line") or "").strip())
    existing_why = _normalize_ref_copy_text(str(ui.get("why_line") or "").strip())
    summary_needs_replacement = False
    if not _is_llm_ref_summary_generation(str(ui.get("summary_generation") or "")):
        if existing_summary and not _summary_line_needs_polish(
            prompt=prompt,
            title=display_name,
            summary_line=existing_summary,
        ):
            # Keep the existing deterministic summary — it is good enough.
            ui["summary_generation"] = "deterministic_preserved"
        else:
            summary_needs_replacement = True
            # Build a clean section-grounded summary from the heading path.
            leaf = heading_path.rsplit("/", 1)[-1].strip() if heading_path else ""
            if prefer_zh:
                ui["summary_line"] = (
                    f"可查看章节：{leaf}" if leaf
                    else f"该文献保留了可核对的章节线索。"
                )
            else:
                ui["summary_line"] = (
                    f"Check section: {leaf}." if leaf
                    else "This paper has section evidence to inspect."
                )
            ui["summary_generation"] = "deterministic_heading_grounded"
            ui["summary_basis"] = "基于章节定位" if prefer_zh else "Grounded in section heading"
    if not _is_llm_ref_why_generation(str(ui.get("why_generation") or "")):
        if existing_why and not _why_line_needs_polish(
            prompt=prompt,
            display_name=display_name,
            heading_path=heading_path,
            summary_line=str(ui.get("summary_line") or ""),
            why_line=existing_why,
        ):
            ui["why_generation"] = "deterministic_preserved"
        else:
            leaf = heading_path.rsplit("/", 1)[-1].strip() if heading_path else heading_path
            if prefer_zh:
                ui["why_line"] = (
                    f"可用来核对“{leaf}”这一节里的证据。"
                    if leaf
                    else "可用来核对这篇论文里的原文证据。"
                )
            else:
                ui["why_line"] = (
                    f"Use section \"{leaf}\" to inspect the source evidence."
                    if leaf
                    else "Use this paper card to inspect the source evidence."
                )
            ui["why_generation"] = "deterministic_heading_grounded"
            ui["why_basis"] = "基于章节定位" if prefer_zh else "Grounded in section heading"
    return ui


def _suppress_non_llm_ref_card_copy_hits(
    *,
    prompt: str,
    hits: list[dict],
) -> list[dict]:
    out: list[dict] = []
    for hit in list(hits or []):
        if not isinstance(hit, dict):
            continue
        hit2 = dict(hit)
        ui_meta = hit2.get("ui_meta") if isinstance(hit2.get("ui_meta"), dict) else {}
        if isinstance(ui_meta, dict) and not _ref_card_has_llm_copy(ui_meta):
            hit2["ui_meta"] = _suppress_non_llm_ref_card_copy(prompt=prompt, ui_meta=ui_meta)
        out.append(hit2)
    return out


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
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), _refs_card_polish_timeout_s()),
            max_retries=_refs_card_polish_max_retries(),
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
                            "why_line: 用 1 句中文说明用户能用这条证据核对什么，优先点出命中的概念或章节。"
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
                max_tokens=160,
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
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), _refs_card_polish_timeout_s()),
            max_retries=_refs_card_polish_max_retries(),
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
                            "summary_line should be 1-2 compact sentences saying what this paper or section does, compares, or provides. "
                            "When evidence supports it, include the concrete method/task plus the contribution, limitation, or application context. "
                            "Use only the supplied evidence snippets, but synthesize them in fresh wording instead of copying any source sentence. "
                            "Do not reuse long source phrasing, markdown headings, formulas, venue boilerplate, or generic templates. "
                            "why_line should be 1-2 compact sentences explaining what the user can verify with this evidence. "
                            "Name the matched concept, comparison, or section, and add the specific clue that makes the card useful. "
                            "Avoid formulaic phrases such as 'this hit is directly relevant', 'good entry point', or '适合作为定位入口'. "
                            f"{'Write both fields in concise Chinese, roughly 45-90 Chinese characters each. ' if prefer_zh else 'Write both fields in concise English, roughly 25-45 words each. '}"
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
                temperature=0.1,
                max_tokens=220,
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
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), _refs_card_polish_timeout_s(12.0)),
            max_retries=_refs_card_polish_max_retries(),
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
                            "Write 1-2 compact sentences explaining what the user can verify with this evidence. "
                            "Name the matched concept, comparison, section, or method, and include the concrete clue that connects it. "
                            "Do not restate the whole prompt. Do not invent facts. "
                            f"{'Write concise Chinese, roughly 45-90 Chinese characters. ' if prefer_zh else 'Write concise English, roughly 25-45 words. '}"
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


def _prepare_ref_hit_card_llm_grounding(
    *,
    prompt: str,
    hit: dict,
    ui_meta: dict,
    candidates: list[str] | None = None,
) -> dict:
    ui = dict(ui_meta or {})
    title = str(ui.get("display_name") or "").strip()
    heading_path = str(ui.get("heading_path") or ui.get("section_label") or "").strip()
    summary_kind = str(ui.get("summary_kind") or "").strip().lower() or "guide"
    candidate_rows = [str(item).strip() for item in (candidates or []) if str(item).strip()]
    if not candidate_rows:
        candidate_rows = _collect_ref_card_polish_candidates(hit, ui_meta=ui, max_items=4)
    candidate_rows = [item for item in candidate_rows if item]
    if not candidate_rows:
        return {}
    summary_seed = _normalize_ref_copy_text(str(ui.get("summary_line") or "").strip())
    why_seed = _normalize_ref_copy_text(str(ui.get("why_line") or "").strip())
    fallback_summary = _normalize_ref_copy_text(
        _pick_ref_card_summary_fallback(
            prompt=prompt,
            title=title,
            candidates=candidate_rows,
        )
    )
    if fallback_summary and _is_ref_card_summary_acceptable(
        prompt=prompt,
        title=title,
        summary_line=fallback_summary,
    ):
        summary_seed = fallback_summary
    deterministic_why = _normalize_ref_copy_text(
        _build_prompt_aligned_ref_why_line_v3(
            prompt=prompt,
            display_name=title,
            heading_path=heading_path,
            summary_line=summary_seed,
            why_line=why_seed,
        )
    )
    if deterministic_why and (not _looks_generic_ref_why_line(deterministic_why)):
        why_seed = deterministic_why
    candidate_payload = "\n".join(f"- {item}" for item in candidate_rows if item)
    if not candidate_payload:
        return {}
    return {
        "ui_meta": ui,
        "title": title,
        "heading_path": heading_path,
        "summary_kind": summary_kind,
        "summary_seed": summary_seed,
        "why_seed": why_seed,
        "candidates": list(candidate_rows),
        "candidate_payload": candidate_payload,
    }


def _apply_llm_grounded_ref_hit_card_copy(
    *,
    prompt: str,
    prepared: dict,
    polished_summary: str,
    polished_why: str,
) -> dict:
    ui = dict(prepared.get("ui_meta") or {})
    title = str(prepared.get("title") or "").strip()
    heading_path = str(prepared.get("heading_path") or "").strip()
    summary_kind = str(prepared.get("summary_kind") or "guide").strip().lower() or "guide"
    summary_seed = str(prepared.get("summary_seed") or "").strip()
    candidates = [str(item).strip() for item in list(prepared.get("candidates") or []) if str(item).strip()]
    strict_llm_copy = True
    raw_polished_summary = _normalize_ref_copy_text(str(polished_summary or "").strip())
    raw_polished_why = _normalize_ref_copy_text(str(polished_why or "").strip())
    accepted_summary = _accept_llm_ref_summary_line(
        prompt=prompt,
        title=title,
        summary_line=raw_polished_summary,
        evidence_snippets=candidates,
    )
    polished_summary = raw_polished_summary if strict_llm_copy and raw_polished_summary else accepted_summary
    effective_summary = polished_summary or summary_seed
    accepted_why = _accept_llm_ref_why_line(
        prompt=prompt,
        display_name=title,
        heading_path=heading_path,
        summary_line=effective_summary,
        why_line=raw_polished_why,
        evidence_snippets=candidates,
    )
    polished_why = raw_polished_why if strict_llm_copy and raw_polished_why else accepted_why
    if polished_summary:
        ui["summary_line"] = polished_summary
        summary_generation = "llm_grounded"
        ui["summary_generation"] = summary_generation
        basis_meta = _build_ref_summary_basis_meta(
            prompt=prompt,
            summary_kind=summary_kind,
            summary_generation=summary_generation,
            summary_line=polished_summary,
        )
        ui["summary_basis"] = str(basis_meta.get("summary_basis") or "")
    if polished_why:
        ui["why_line"] = polished_why
        why_generation = "llm_grounded"
        why_basis_meta = _build_ref_why_basis_meta(
            prompt=prompt,
            why_generation=why_generation,
            why_line=polished_why,
        )
        ui["why_generation"] = str(why_basis_meta.get("why_generation") or why_generation)
        ui["why_basis"] = str(why_basis_meta.get("why_basis") or "")
    return ui


@lru_cache(maxsize=256)
def _llm_batch_polish_ref_card_copy_v1(
    *,
    prompt: str,
    cards_payload: str,
    card_count: int,
) -> tuple[tuple[int, str, str], ...]:
    if (not prompt) or (not cards_payload) or card_count <= 1:
        return ()
    if not _refs_card_polish_llm_enabled():
        return ()
    try:
        settings = load_settings()
    except Exception:
        return ()
    if not getattr(settings, "api_key", None):
        return ()
    prefer_zh = _prefer_zh_ref_card_locale(prompt, cards_payload)
    try:
        fast_settings = replace(
            settings,
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), _refs_card_polish_timeout_s()),
            max_retries=_refs_card_polish_max_retries(),
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
                            "You are polishing multiple research reference cards in a reading assistant. "
                            "Return JSON only with this schema: "
                            "{\"cards\":[{\"index\":1,\"summary_line\":\"...\",\"why_line\":\"...\"}]}. "
                            "For each card, summary_line should be 1-2 compact sentences saying what the matched paper or section does, provides, compares, or concludes. "
                            "When evidence supports it, include the concrete method/task plus the contribution, limitation, or application context. "
                            "Use only that card's supplied evidence snippets, but synthesize them in fresh wording instead of copying any source sentence. "
                            "Do not reuse long source phrasing, markdown headings, formulas, venue boilerplate, or generic templates. "
                            "For each card, why_line should be 1-2 compact sentences explaining what the user can verify with this evidence. "
                            "Name the matched concept, comparison, or section, and add the specific clue that makes the card useful. "
                            "Avoid formulaic phrases such as 'this hit is directly relevant', 'good entry point', or '适合作为定位入口'. "
                            f"{'Write both fields in concise Chinese, roughly 45-90 Chinese characters each. ' if prefer_zh else 'Write both fields in concise English, roughly 25-45 words each. '}"
                            "Do not invent facts. Do not output markdown, bullets, DOC/SID/CITE markers, or placeholders."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"User prompt: {str(prompt or '').strip()}\n\n"
                            f"Cards:\n{cards_payload}\n"
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=min(760, max(320, 170 * int(card_count) + 60)),
            )
            or ""
        ).strip()
    except Exception:
        return ()
    rows = []
    try:
        parsed = json.loads(out)
        cards = parsed.get("cards") if isinstance(parsed, dict) else None
        if isinstance(cards, list):
            rows = list(cards)
    except Exception:
        rows = []
    if not rows:
        return ()
    out_rows: list[tuple[int, str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("index") or 0)
        except Exception:
            idx = 0
        if idx <= 0:
            continue
        out_rows.append(
            (
                idx,
                str(row.get("summary_line") or "").strip(),
                str(row.get("why_line") or "").strip(),
            )
        )
    return tuple(out_rows)


def _batch_polish_doc_list_ref_hit_cards(
    *,
    prompt: str,
    jobs: list[tuple[int, dict, dict]],
) -> dict[int, dict]:
    prepared_rows: list[dict] = []
    prepared_by_batch_idx: dict[int, dict] = {}
    for job_idx, hit, ui_meta in jobs:
        prepared = _prepare_ref_hit_card_llm_grounding(
            prompt=prompt,
            hit=hit,
            ui_meta=ui_meta,
        )
        if (not prepared) or str(prepared.get("summary_kind") or "").strip().lower() not in ("guide", "section_grounded"):
            continue
        batch_idx = len(prepared_rows) + 1
        prepared2 = dict(prepared)
        prepared2["job_idx"] = int(job_idx)
        prepared2["batch_idx"] = int(batch_idx)
        prepared_rows.append(prepared2)
        prepared_by_batch_idx[batch_idx] = prepared2
    if len(prepared_rows) <= 1:
        return {}
    payload_chunks: list[str] = []
    for row in prepared_rows:
        payload_chunks.append(
            "\n".join(
                [
                    f"Card {int(row.get('batch_idx') or 0)}",
                    f"Paper title: {str(row.get('title') or '').strip()}",
                    f"Section heading: {str(row.get('heading_path') or '').strip()}",
                    f"Current summary candidate: {str(row.get('summary_seed') or '').strip()}",
                    f"Current relevance candidate: {str(row.get('why_seed') or '').strip()}",
                    f"Evidence snippets:\n{str(row.get('candidate_payload') or '').strip()}",
                ]
            )
        )
    outputs = _llm_batch_polish_ref_card_copy_v1(
        prompt=prompt,
        cards_payload="\n\n".join(payload_chunks),
        card_count=len(prepared_rows),
    )
    polished: dict[int, dict] = {}
    for batch_idx, polished_summary, polished_why in outputs:
        prepared = prepared_by_batch_idx.get(int(batch_idx))
        if not isinstance(prepared, dict):
            continue
        job_idx = int(prepared.get("job_idx") or -1)
        if job_idx < 0:
            continue
        polished[job_idx] = _apply_llm_grounded_ref_hit_card_copy(
            prompt=prompt,
            prepared=prepared,
            polished_summary=polished_summary,
            polished_why=polished_why,
        )
    return polished


def _force_llm_ground_ref_hit_card_copy(
    *,
    prompt: str,
    hit: dict,
    ui_meta: dict,
    candidates: list[str],
) -> dict:
    prepared = _prepare_ref_hit_card_llm_grounding(
        prompt=prompt,
        hit=hit,
        ui_meta=ui_meta,
        candidates=candidates,
    )
    if not prepared:
        return dict(ui_meta or {})
    polished_summary, polished_why = _llm_polish_ref_card_copy_v2(
        prompt=prompt,
        display_name=str(prepared.get("title") or "").strip(),
        heading_path=str(prepared.get("heading_path") or "").strip(),
        summary_seed=str(prepared.get("summary_seed") or "").strip(),
        why_seed=str(prepared.get("why_seed") or "").strip(),
        candidate_payload=str(prepared.get("candidate_payload") or "").strip(),
    )
    result = _apply_llm_grounded_ref_hit_card_copy(
        prompt=prompt,
        prepared=prepared,
        polished_summary=polished_summary,
        polished_why=polished_why,
    )
    # If the first LLM pass produced no usable summary, retry with a simpler prompt
    # that focuses on extracting one evidence-grounded sentence.
    needs_retry = not _is_llm_ref_summary_generation(
        str((result or {}).get("summary_generation") or "")
    )
    if needs_retry and prepared.get("candidate_payload"):
        retry_summary, retry_why = _llm_polish_ref_card_copy_v2(
            prompt=prompt,
            display_name=str(prepared.get("title") or "").strip(),
            heading_path=str(prepared.get("heading_path") or "").strip(),
            summary_seed=(
                "Write exactly one concise sentence grounded in the evidence snippets below. "
                "Do not copy raw snippet text; synthesize. "
                "Include the key concept, method, or finding from the matched section."
            ),
            why_seed=(
                "State what this hit contributes to the user's question, in one sentence."
            ),
            candidate_payload=str(prepared.get("candidate_payload") or "").strip(),
        )
        result = _apply_llm_grounded_ref_hit_card_copy(
            prompt=prompt,
            prepared=prepared,
            polished_summary=retry_summary,
            polished_why=retry_why,
        )
    return result


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
    summary_kind = str(ui.get("summary_kind") or "").strip().lower() or "guide"
    ui["summary_kind"] = summary_kind
    summary_generation = str(ui.get("summary_generation") or "").strip().lower()
    why_generation = str(ui.get("why_generation") or "").strip().lower()
    allow_llm_polish = bool(allow_expensive_llm and _refs_card_polish_llm_enabled())
    force_llm_summary = _guide_summary_should_prefer_llm_grounding(
        prompt=prompt,
        title=title,
        heading_path=heading_path,
        summary_line=summary_line,
        summary_kind=summary_kind,
        summary_generation=summary_generation,
        allow_llm_polish=allow_llm_polish,
    )
    candidates = _collect_ref_card_polish_candidates(hit, ui_meta=ui, max_items=4)
    reusable_summary, reusable_why = _reuse_existing_llm_guide_copy(
        prompt=prompt,
        title=title,
        heading_path=heading_path,
        summary_kind=summary_kind,
        summary_generation=summary_generation,
        why_generation=why_generation,
        summary_line=summary_line,
        why_line=why_line,
        evidence_snippets=candidates,
    )
    if reusable_summary and reusable_why:
        ui["summary_line"] = reusable_summary
        ui["why_line"] = reusable_why
        return ui
    force_llm_card = bool(candidates and allow_llm_polish and summary_kind in ("guide", "section_grounded"))
    if force_llm_card:
        return _force_llm_ground_ref_hit_card_copy(
            prompt=prompt,
            hit=hit,
            ui_meta=ui,
            candidates=candidates,
        )

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
            if summary_kind in ("guide", "section_grounded"):
                summary_generation = "deterministic_grounded"
                ui["summary_generation"] = summary_generation
                basis_meta = _build_ref_summary_basis_meta(
                    prompt=prompt,
                    summary_kind=summary_kind,
                    summary_generation=summary_generation,
                    summary_line=fallback_summary,
                )
                ui["summary_basis"] = str(basis_meta.get("summary_basis") or "")
            summary_line = fallback_summary
            needs_summary = False
    force_llm_summary = _guide_summary_should_prefer_llm_grounding(
        prompt=prompt,
        title=title,
        heading_path=heading_path,
        summary_line=summary_line,
        summary_kind=summary_kind,
        summary_generation=summary_generation,
        allow_llm_polish=allow_llm_polish,
    )
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
        grounded_why = _accept_llm_ref_why_line(
            prompt=prompt,
            display_name=title,
            heading_path=heading_path,
            summary_line=summary_line,
            why_line=grounded_why,
            evidence_snippets=candidates,
        )
        if grounded_why:
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
        if force_llm_summary:
            needs_summary = True
        elif (
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
    polished_summary = _accept_llm_ref_summary_line(
        prompt=prompt,
        title=title,
        summary_line=polished_summary,
        evidence_snippets=candidates,
    )
    effective_summary = polished_summary or summary_line
    polished_why = _accept_llm_ref_why_line(
        prompt=prompt,
        display_name=title,
        heading_path=heading_path,
        summary_line=effective_summary,
        why_line=polished_why,
        evidence_snippets=candidates,
    )
    if polished_summary:
        ui["summary_line"] = polished_summary
        if summary_kind in ("guide", "section_grounded"):
            summary_generation = "llm_grounded"
            ui["summary_generation"] = summary_generation
            basis_meta = _build_ref_summary_basis_meta(
                prompt=prompt,
                summary_kind=summary_kind,
                summary_generation=summary_generation,
                summary_line=polished_summary,
            )
            ui["summary_basis"] = str(basis_meta.get("summary_basis") or "")
    if polished_why:
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


def _refs_card_polish_max_workers(job_count: int) -> int:
    try:
        configured = int(str(os.environ.get("KB_REFS_CARD_POLISH_MAX_WORKERS", "2") or "2"))
    except Exception:
        configured = 2
    configured = max(1, min(8, configured))
    return max(1, min(int(job_count or 0), configured))


def _maybe_polish_refs_card_copy(*, prompt: str, hits: list[dict], guide_mode: bool) -> list[dict]:
    rows = [dict(hit) for hit in (hits or []) if isinstance(hit, dict)]
    if not rows:
        return rows
    limit = _refs_card_polish_top_n()
    if limit <= 0:
        return rows
    polished: list[dict] = list(rows)
    jobs: list[tuple[int, dict, dict]] = []
    for idx, hit in enumerate(rows):
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        if idx >= limit or not isinstance(ui_meta, dict):
            continue
        jobs.append((idx, hit, ui_meta))
    if not jobs:
        return polished

    def _polish_one(idx: int, hit: dict, ui_meta: dict) -> tuple[int, dict]:
        return idx, _maybe_polish_single_ref_hit_card(
            prompt=prompt,
            hit=hit,
            ui_meta=ui_meta,
            allow_expensive_llm=True,
        )

    max_workers = _refs_card_polish_max_workers(len(jobs))
    if max_workers <= 1:
        for idx, hit, ui_meta in jobs:
            hit2 = dict(hit)
            hit2["ui_meta"] = _maybe_polish_single_ref_hit_card(
                prompt=prompt,
                hit=hit,
                ui_meta=ui_meta,
                allow_expensive_llm=True,
            )
            polished[idx] = hit2
        return polished
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_polish_one, idx, hit, ui_meta) for idx, hit, ui_meta in jobs]
            for fu in as_completed(futs):
                try:
                    idx, ui_meta = fu.result()
                except Exception:
                    continue
                hit2 = dict(rows[idx])
                hit2["ui_meta"] = ui_meta
                polished[idx] = hit2
    except Exception:
        for idx, hit, ui_meta in jobs:
            hit2 = dict(hit)
            hit2["ui_meta"] = _maybe_polish_single_ref_hit_card(
                prompt=prompt,
                hit=hit,
                ui_meta=ui_meta,
                allow_expensive_llm=True,
            )
            polished[idx] = hit2
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
    heading_path = str(candidate.get("headingPath") or "").strip()
    highlight_snippet = str(candidate.get("highlightSnippet") or "").strip()
    snippet = str(candidate.get("snippet") or "").strip()
    anchor_kind = str(candidate.get("anchorKind") or "").strip().lower()
    anchor_number = _positive_int(candidate.get("anchorNumber"))
    block_id = str(candidate.get("blockId") or "").strip()
    anchor_id = str(candidate.get("anchorId") or "").strip()
    if not any((heading_path, highlight_snippet, snippet, anchor_kind, anchor_number, block_id, anchor_id)):
        return ""
    return "::".join(
        [
            heading_path.lower(),
            highlight_snippet.lower()[:180],
            snippet.lower()[:180],
            anchor_kind,
            str(anchor_number or ""),
            block_id.lower(),
            anchor_id.lower(),
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


def _refs_heading_paths_related(left: str, right: str) -> bool:
    left_norm = str(left or "").strip().lower()
    right_norm = str(right or "").strip().lower()
    if (not left_norm) or (not right_norm):
        return False
    return (
        left_norm == right_norm
        or left_norm.startswith(f"{right_norm} /")
        or right_norm.startswith(f"{left_norm} /")
    )


def _refs_heading_anchor_number(anchor_kind: str, heading_path: str) -> int:
    kind = str(anchor_kind or "").strip().lower()
    heading = str(heading_path or "").strip()
    if (not kind) or (not heading):
        return 0
    if kind == "figure":
        return extract_figure_number(heading)
    if kind == "equation":
        return extract_equation_number(heading)
    if kind == "table":
        m = re.search(r"(?:table|tab\.?|表)\s*[\(#\[]?\s*(\d{1,4})(?!\d)", heading, flags=re.I)
        if not m:
            return 0
        try:
            value = int(str(m.group(1) or "0"))
        except Exception:
            return 0
        return value if value > 0 else 0
    return 0


def _clean_refs_evidence_snippet(
    raw: str,
    *,
    prompt: str,
    source_path: str,
    display_name: str = "",
    heading_path: str = "",
    max_len: int = 360,
) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    title_hint = str(display_name or Path(str(source_path or "")).name or "").strip()
    picked = _pick_readable_evidence_text(
        text,
        source=source_path,
        title=title_hint,
        claim=prompt,
        heading=heading_path,
        max_len=max_len,
    )
    return picked or _clean_evidence_display_text(text, max_len=max_len)


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
    snippet_text = _clean_refs_evidence_snippet(
        snippet,
        prompt=prompt,
        source_path=source_path,
        heading_path=heading,
        max_len=360,
    )
    highlight_text = _clean_refs_evidence_snippet(
        highlight_snippet or snippet_text,
        prompt=prompt,
        source_path=source_path,
        heading_path=heading,
        max_len=360,
    )
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


def _score_refs_exact_surface(
    text: str,
    *,
    prompt: str,
    title: str,
    block_kind: str = "",
    anchor_target_kind: str = "",
) -> float:
    surface = _compact_reader_open_text(text)
    if not surface:
        return -1000.0
    score = 0.0
    block_kind_norm = str(block_kind or "").strip().lower()
    anchor_kind_norm = str(anchor_target_kind or "").strip().lower()
    apply_summary_shape_penalties = block_kind_norm != "paragraph"
    if _looks_bibliographic_source_block_text(surface):
        score -= 5.0
    if title and _looks_title_like_ref_surface(surface, title):
        score -= 5.2
    if _looks_like_front_matter_ref_summary(surface):
        score -= 3.8
    if apply_summary_shape_penalties and _looks_prefixed_heading_shell_ref_summary(surface):
        score -= 3.2
    if apply_summary_shape_penalties and _looks_surface_like_ref_summary(surface):
        score -= 2.8
    if _looks_fragmentary_ref_summary(surface):
        score -= 2.6
    if _looks_why_like_ref_summary(surface):
        score -= 2.6
    if _looks_formula_heavy_ref_text(surface) and anchor_kind_norm != "equation":
        score -= 1.4
    focus_action = _shared_prompt_reference_focus_action(prompt)
    keyword_hits = _refs_summary_focus_keyword_hit_count(prompt, surface) if prompt else 0
    if focus_action == "compare" and re.search(r"\b(compare|comparison|versus|vs\.?|difference|whereas|while)\b", surface, flags=re.I):
        score += 0.9
        if keyword_hits >= 2:
            score += 3.2
    if focus_action == "define":
        if re.search(r"\b(define|defines|defined|definition|refers to|known as|is known as|is called)\b", surface, flags=re.I):
            score += 0.9
        elif re.match(r"^\s*if\b", surface, flags=re.I):
            score += 0.35
    if block_kind_norm == "heading":
        score -= 4.6
    if block_kind_norm in {"figure", "table"} and not anchor_kind_norm:
        score -= 2.8
    if (not anchor_kind_norm) and re.match(r"^\s*(?:fig(?:ure)?\.?|table)\b", surface, flags=re.I):
        score -= 2.4
    if (not anchor_kind_norm) and re.match(r"^\s*\([A-Z]\)\s", surface):
        score -= 2.1
    if _looks_natural_language_ref_summary(surface):
        score += 1.0
    if _has_ref_summary_explainer_signal(surface):
        score += 0.9
    if _has_ref_summary_value_signal(surface):
        score += 0.5
    if len(surface) >= 56:
        score += 0.25
    if block_kind_norm == "paragraph":
        score += 0.35
        if len(surface) >= 120:
            score += 0.35
    elif len(surface) > 420:
        score -= 0.8
    if prompt:
        score += 0.45 * float(_refs_exact_focus_match_count(prompt, surface))
        score += 0.35 * float(len(_matched_focus_terms_for_ref_card(prompt, surface_text=surface)))
        score += 0.15 * float(keyword_hits)
    return score


def _select_reader_open_exact_snippet(
    seed_text: str,
    block_text: str,
    *,
    prompt: str = "",
    title: str = "",
    block_kind: str = "",
    anchor_target_kind: str = "",
) -> tuple[str, str]:
    seed = _compact_reader_open_text(seed_text)
    block = _compact_reader_open_text(block_text)
    if not block:
        return seed, seed
    if not seed:
        return block, block
    seed_score = _score_refs_exact_surface(
        seed,
        prompt=prompt,
        title=title,
        block_kind="",
        anchor_target_kind=anchor_target_kind,
    )
    block_score = _score_refs_exact_surface(
        block,
        prompt=prompt,
        title=title,
        block_kind=block_kind,
        anchor_target_kind=anchor_target_kind,
    )
    if block_score >= (seed_score + 1.0):
        return block, block
    if prompt and _looks_focus_prefixed_ref_summary(prompt, seed) and block_kind.strip().lower() == "paragraph" and block_score > -0.25:
        return block, block
    if prompt and _summary_line_needs_polish(prompt=prompt, title=title, summary_line=seed) and block_score >= (seed_score - 0.15):
        return block, block
    if seed and block:
        seed_key = re.sub(r"\s+", " ", seed).strip().lower()
        block_key = re.sub(r"\s+", " ", block).strip().lower()
        if seed_key and block_key and (seed_key in block_key or block_key in seed_key):
            if (block_score >= (seed_score + 0.35)) and len(block) > (len(seed) + 24):
                return block, block
            return seed, seed
    if (seed_score < -1.5) and (block_score > seed_score):
        return block, block
    return seed, seed


def _build_refs_exact_candidate_from_block(
    *,
    prompt: str,
    source_path: str,
    title: str,
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
    block_kind = str(block.get("kind") or "").strip().lower()
    snippet_text, highlight_text = _select_reader_open_exact_snippet(
        seed_snippet,
        block_text,
        prompt=prompt,
        title=title,
        block_kind=block_kind,
        anchor_target_kind=anchor_kind,
    )
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


def _build_preferred_refs_exact_candidate_from_source_summary(
    *,
    prompt: str,
    source_path: str,
    title: str,
    summary_line: str,
    selected_heading_path: str,
    anchor_target_kind: str,
    anchor_target_number: int,
    prompt_aligned_candidate: dict | None,
) -> dict:
    if not isinstance(prompt_aligned_candidate, dict):
        return {}
    if str(prompt_aligned_candidate.get("source_kind") or "").strip().lower() != "source_block":
        return {}
    block_id = str(prompt_aligned_candidate.get("block_id") or "").strip()
    if not block_id:
        return {}

    candidate_summary = str(prompt_aligned_candidate.get("summary") or "").strip()
    if summary_line and candidate_summary and (not _ref_summary_surfaces_match(summary_line, candidate_summary)):
        return {}

    block_heading_path = _normalize_refs_reader_heading_path(
        prompt=prompt,
        source_path=source_path,
        heading_path=str(prompt_aligned_candidate.get("heading_path") or "").strip(),
    )
    selected_heading = _normalize_refs_reader_heading_path(
        prompt=prompt,
        source_path=source_path,
        heading_path=selected_heading_path,
    )
    if selected_heading and block_heading_path and block_heading_path != selected_heading:
        return {}

    block_kind = str(prompt_aligned_candidate.get("block_kind") or "").strip().lower()
    target_kind = str(anchor_target_kind or "").strip().lower()
    if target_kind and block_kind and block_kind != target_kind:
        return {}

    block_text = str(prompt_aligned_candidate.get("block_text") or "").strip()
    seed_snippet = candidate_summary or summary_line or block_text
    snippet_text, highlight_text = _select_reader_open_exact_snippet(
        seed_snippet,
        block_text,
        prompt=prompt,
        title=title,
        block_kind=block_kind,
        anchor_target_kind=target_kind,
    )
    if (not snippet_text) and block_text:
        snippet_text = _compact_reader_open_text(block_text)
    if not highlight_text:
        highlight_text = snippet_text

    candidate = _build_refs_reader_open_candidate(
        prompt=prompt,
        source_path=source_path,
        heading_path=block_heading_path or selected_heading or selected_heading_path,
        snippet=snippet_text,
        highlight_snippet=highlight_text,
        anchor_kind=target_kind or block_kind,
        anchor_number=anchor_target_number or _positive_int(prompt_aligned_candidate.get("block_number")),
    )
    if not isinstance(candidate, dict):
        return {}
    candidate["blockId"] = block_id
    anchor_id = str(prompt_aligned_candidate.get("anchor_id") or "").strip()
    if anchor_id:
        candidate["anchorId"] = anchor_id
    return candidate


def _refs_locate_llm_enabled() -> bool:
    raw_flag = str(os.environ.get("KB_REFS_LOCATE_USE_LLM", "1") or "").strip().lower()
    if raw_flag in {"0", "false", "off", "no"}:
        return False
    return True


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
    display_name: str = "",
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
                title=display_name,
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
                    "block_kind": str(((block or {}) if isinstance(block, dict) else {}).get("kind") or "").strip().lower(),
                    "heading_path": heading_path,
                }
            )
            if len(out_rows) >= 5:
                break
        if len(out_rows) >= 5:
            break

    if len(out_rows) <= 1:
        return [dict(item.get("candidate") or {}) for item in out_rows if isinstance(item.get("candidate"), dict)]

    anchor_kind_norm = str(anchor_target_kind or "").strip().lower()
    target_anchor_num = _positive_int(anchor_target_number)

    def _exact_candidate_sort_key(item: dict) -> tuple[float, float, int, int, int, int, int, float]:
        candidate = dict(item.get("candidate") or {}) if isinstance(item.get("candidate"), dict) else {}
        candidate_heading = str(candidate.get("headingPath") or "").strip().lower()
        seed_heading = str(item.get("heading_path") or "").strip().lower()
        block_text = str(item.get("block_text") or "").strip()
        surface = block_text or str(candidate.get("highlightSnippet") or candidate.get("snippet") or "").strip()
        primary_match = int(bool(primary_heading_norm and candidate_heading and candidate_heading == primary_heading_norm))
        primary_related = int(bool(primary_heading_norm and candidate_heading and _refs_heading_paths_related(candidate_heading, primary_heading_norm)))
        seed_match = int(bool(seed_heading and candidate_heading and candidate_heading == seed_heading))
        heading_anchor_num = _refs_heading_anchor_number(anchor_kind_norm, candidate_heading)
        target_heading_match = int(bool(target_anchor_num > 0 and heading_anchor_num == target_anchor_num))
        target_heading_conflict = int(bool(target_anchor_num > 0 and heading_anchor_num > 0 and heading_anchor_num != target_anchor_num))
        quality_score = _score_refs_exact_surface(
            surface,
            prompt=prompt,
            title=display_name,
            block_kind=str(item.get("block_kind") or "").strip().lower(),
            anchor_target_kind=anchor_target_kind,
        )
        raw_score = float(item.get("score") or 0.0)
        exact_focus_hits = _refs_exact_focus_match_count(prompt, surface)
        focus_hits = len(_matched_focus_terms_for_ref_card(prompt, surface_text=surface))
        combined_score = (
            float(quality_score)
            + (0.8 * raw_score)
            + (0.25 * float(exact_focus_hits))
            + (0.15 * float(focus_hits))
            + (0.26 * float(primary_match))
            + (0.85 * float(primary_related))
            + (0.12 * float(seed_match))
            + (3.2 * float(target_heading_match))
            - (5.2 * float(target_heading_conflict))
        )
        return (
            float(combined_score),
            float(quality_score),
            target_heading_match,
            -target_heading_conflict,
            primary_related,
            primary_match,
            seed_match,
            raw_score,
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
    preferred_exact_candidate: dict | None = None,
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

    exact_candidates: list[dict] = []
    seen_exact: set[str] = set()

    def _push_exact(candidate: dict | None) -> None:
        if not isinstance(candidate, dict):
            return
        key = _refs_reader_open_candidate_key(candidate)
        if (not key) or (key in seen_exact):
            return
        seen_exact.add(key)
        exact_candidates.append(candidate)

    _push_exact(preferred_exact_candidate)
    for candidate in _resolve_refs_exact_candidates(
        prompt=prompt,
        source_path=source_path,
        display_name=display_name,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        primary_candidate=primary_candidate,
        secondary_candidates=secondary_candidates,
        allow_llm_disambiguation=allow_llm_disambiguation,
    ):
        _push_exact(candidate)
    primary_heading_norm = str((primary_candidate or {}).get("headingPath") or primary_heading or "").strip().lower()
    prompt_is_focus_no_anchor = bool(
        primary_heading_norm
        and (not str(anchor_target_kind or "").strip())
        and _prompt_requires_explicit_focus_match(prompt)
    )
    if (
        len(exact_candidates) >= 1
        and prompt_is_focus_no_anchor
    ):
        top_heading_norm = str((exact_candidates[0].get("headingPath") or "")).strip().lower()
        if top_heading_norm and (not _refs_heading_paths_related(top_heading_norm, primary_heading_norm)):
            found_related = False
            for idx, candidate in enumerate(exact_candidates[1:], start=1):
                candidate_heading_norm = str((candidate.get("headingPath") or "")).strip().lower()
                if _refs_heading_paths_related(candidate_heading_norm, primary_heading_norm):
                    exact_candidates = [candidate] + [item for j, item in enumerate(exact_candidates) if j != idx]
                    found_related = True
                    break
            if not found_related:
                exact_candidates = []
    primary_exact = exact_candidates[0] if exact_candidates else None
    related_block_ids = [
        str(candidate.get("blockId") or "").strip()
        for candidate in exact_candidates
        if str(candidate.get("blockId") or "").strip()
    ]
    related_block_ids = list(dict.fromkeys(related_block_ids))[:5]

    effective_primary = primary_exact or primary_candidate
    if (
        effective_primary is not primary_candidate
        and prompt_is_focus_no_anchor
    ):
        eff_heading = str((effective_primary or {}).get("headingPath") or "").strip().lower()
        if eff_heading and (not _refs_heading_paths_related(eff_heading, primary_heading_norm)):
            effective_primary = dict(effective_primary or {})
            effective_primary["headingPath"] = primary_heading
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
    prompt: str = "",
) -> dict:
    if not isinstance(reader_open, dict):
        return {}

    def _candidate_to_evidence(candidate: dict | None) -> dict | None:
        if not isinstance(candidate, dict):
            return None
        heading_path = str(candidate.get("headingPath") or "").strip()
        snippet = _clean_refs_evidence_snippet(
            str(candidate.get("snippet") or "").strip(),
            prompt=prompt,
            source_path=source_path,
            display_name=display_name,
            heading_path=heading_path,
            max_len=460,
        )
        highlight_snippet = _clean_refs_evidence_snippet(
            str(candidate.get("highlightSnippet") or snippet or "").strip(),
            prompt=prompt,
            source_path=source_path,
            display_name=display_name,
            heading_path=heading_path,
            max_len=460,
        )
        evidence = {
            "source_path": str(source_path or "").strip() or None,
            "source_name": str(display_name or "").strip() or None,
            "block_id": str(candidate.get("blockId") or "").strip() or None,
            "anchor_id": str(candidate.get("anchorId") or "").strip() or None,
            "heading_path": heading_path or None,
            "snippet": snippet or None,
            "highlight_snippet": highlight_snippet or None,
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


_ANSWER_EVIDENCE_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "onto", "this", "that", "these", "those",
    "paper", "work", "study", "method", "methods", "approach", "system", "model", "models",
    "result", "results", "show", "shows", "shown", "use", "uses", "using", "based", "given",
    "question", "answer", "evidence", "reference", "citation", "source", "section", "figure",
    "image", "images", "sample", "samples", "data", "process", "step", "steps", "first",
    "second", "also", "main", "core", "because", "therefore", "however", "then", "than",
}

_ANSWER_EVIDENCE_LOW_SIGNAL_TERMS = {
    "paper", "work", "method", "model", "system", "approach", "image", "imaging", "sample",
    "samples", "result", "results", "data", "process", "section", "reference", "references",
    "citation", "citations", "source", "sources",
}

_ANSWER_EVIDENCE_ALIAS_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("refocus", "refocusing", "defocus", "out of focus", "depth of field", "\u91cd\u65b0\u5bf9\u7126", "\u91cd\u805a\u7126", "\u79bb\u7126", "\u5bf9\u7126"),
        ("digital refocusing", "refocus", "refocusing", "out of focus", "defocus", "depth of field"),
    ),
    (
        ("ray tracing", "ray transfer", "ray optics", "\u5149\u7ebf\u8ffd\u8ff9", "\u5149\u7ebf"),
        ("ray tracing", "ray transfer", "ray transfer matrix", "ray optics"),
    ),
    (
        ("wave optics", "diffraction", "propagation", "angular spectrum", "\u6ce2\u52a8\u5149\u5b66", "\u884d\u5c04", "\u4f20\u64ad"),
        ("wave optics", "diffraction", "wave propagation", "angular spectrum method", "propagation"),
    ),
    (
        ("snr", "signal to noise", "signal-to-noise", "\u4fe1\u566a\u6bd4"),
        ("SNR", "signal-to-noise", "signal to noise"),
    ),
    (
        ("resolution", "spatial resolution", "\u5206\u8fa8\u7387"),
        ("resolution", "spatial resolution"),
    ),
    (
        ("optical sectioning", "sectioning", "\u5149\u5b66\u5207\u7247", "\u5c42\u5207"),
        ("optical sectioning", "sectioning"),
    ),
    (
        ("foveated", "supersampling", "dynamic supersampling"),
        ("foveated", "dynamic supersampling", "supersampling", "adaptive sampling"),
    ),
    (
        ("physics informed", "physics-informed", "spad", "single photon", "single-photon"),
        ("physics-informed", "SPAD", "single-photon", "noise model"),
    ),
    (
        ("hadamard", "fourier"),
        ("Hadamard", "Fourier", "measurement", "sampling"),
    ),
    (
        ("admm", "alternating direction"),
        ("ADMM", "alternating direction method of multipliers"),
    ),
)


def _answer_evidence_push_term(out: list[str], seen: set[str], raw: str) -> None:
    text = str(raw or "").strip()
    if not text:
        return
    norm = _normalize_title_identity(text)
    if len(norm) < 3 or norm in seen:
        return
    if norm in _ANSWER_EVIDENCE_STOPWORDS or norm in _ANSWER_EVIDENCE_LOW_SIGNAL_TERMS:
        return
    seen.add(norm)
    out.append(text)


def _answer_evidence_terms(prompt: str, answer: str) -> list[str]:
    prompt_text = str(prompt or "").strip()
    answer_text = str(answer or "").strip()
    combined = f"{prompt_text}\n{answer_text}".strip()
    if not combined:
        return []
    out: list[str] = []
    seen: set[str] = set()

    for term in _refs_prompt_focus_terms(prompt_text):
        _answer_evidence_push_term(out, seen, term)

    combined_norm = _normalize_title_identity(combined)
    for triggers, aliases in _ANSWER_EVIDENCE_ALIAS_GROUPS:
        triggered = False
        for trigger in triggers:
            trigger_norm = _normalize_title_identity(trigger)
            if trigger_norm and trigger_norm in combined_norm:
                triggered = True
                break
        if triggered:
            for alias in aliases:
                _answer_evidence_push_term(out, seen, alias)

    raw_tokens = re.findall(r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9_-]{1,48}(?![A-Za-z0-9_-])", combined)
    for raw in raw_tokens:
        low = raw.lower()
        has_signal = raw.isupper() or any(ch.isdigit() for ch in raw) or ("-" in raw) or any(ch.isupper() for ch in raw[1:])
        if has_signal and low not in _ANSWER_EVIDENCE_STOPWORDS:
            _answer_evidence_push_term(out, seen, raw)

    tokens = [
        tok
        for tok in re.findall(r"[a-z0-9][a-z0-9-]{1,48}", _normalize_title_identity(answer_text or combined))
        if tok and tok not in _ANSWER_EVIDENCE_STOPWORDS and tok not in _ANSWER_EVIDENCE_LOW_SIGNAL_TERMS
    ]
    for n in (4, 3, 2):
        for idx in range(0, max(0, len(tokens) - n + 1)):
            window = tokens[idx : idx + n]
            if any(tok in _ANSWER_EVIDENCE_STOPWORDS for tok in window):
                continue
            phrase = " ".join(window)
            technical = any(len(tok) >= 7 or "-" in tok or any(ch.isdigit() for ch in tok) for tok in window)
            if not technical:
                continue
            _answer_evidence_push_term(out, seen, phrase)
            if len(out) >= 18:
                return out[:18]

    for seq in re.findall(r"[\u4e00-\u9fff]{2,12}", combined):
        _answer_evidence_push_term(out, seen, seq)
        if len(out) >= 18:
            break
    return out[:18]


def _answer_evidence_term_matches_surface(term: str, surface_text: str) -> bool:
    norm_term = _normalize_title_identity(term)
    surface = _normalize_title_identity(surface_text)
    if (not norm_term) or (not surface):
        return False
    if re.search(r"[\u4e00-\u9fff]", norm_term):
        return norm_term in surface
    return _focus_term_matches_surface(norm_term, surface)


def _answer_evidence_identity_like(term: str, *, source_path: str, display_name: str) -> bool:
    norm = _normalize_title_identity(term)
    if not norm:
        return True
    if norm in _ANSWER_EVIDENCE_LOW_SIGNAL_TERMS:
        return True
    identity_terms = _ref_summary_identity_terms(source_path=source_path, title=display_name)
    return any(norm == ident or norm in ident or ident in norm for ident in identity_terms)


def _primary_evidence_surface(primary: dict | None) -> str:
    if not isinstance(primary, dict):
        return ""
    parts = [
        str(primary.get("heading_path") or primary.get("headingPath") or "").strip(),
        str(primary.get("snippet") or "").strip(),
        str(primary.get("highlight_snippet") or primary.get("highlightSnippet") or "").strip(),
        str(primary.get("source_name") or primary.get("sourceName") or "").strip(),
    ]
    return " ".join(part for part in parts if part)


def _score_primary_ref_evidence_against_answer(
    *,
    primary_evidence: dict | None,
    prompt: str,
    answer: str,
    terms: list[str],
    display_name: str,
    source_path: str,
) -> tuple[float, list[str]]:
    primary = _normalize_primary_ref_evidence_payload(primary_evidence if isinstance(primary_evidence, dict) else {})
    if not primary:
        return -1000.0, []
    surface = _primary_evidence_surface(primary)
    heading_path = str(primary.get("heading_path") or "").strip()
    snippet = str(primary.get("highlight_snippet") or primary.get("snippet") or "").strip()
    if not surface:
        return -1000.0, []
    if _looks_bibliographic_source_block_text(snippet or surface):
        return -1000.0, []
    matched: list[str] = []
    heading_matches = 0
    answer_term_hits = 0
    for term in terms:
        if _answer_evidence_identity_like(term, source_path=source_path, display_name=display_name):
            continue
        if not _answer_evidence_term_matches_surface(term, surface):
            continue
        matched.append(str(term or "").strip())
        answer_term_hits += 1
        if heading_path and _answer_evidence_term_matches_surface(term, heading_path):
            heading_matches += 1
    if not matched and not _refs_summary_focus_keyword_hit_count(prompt, surface):
        return 0.0, []

    heading_norm = _normalize_title_identity(heading_path)
    answer_norm = _normalize_title_identity(answer)
    score = 0.0
    score += 2.2 * float(answer_term_hits)
    score += 1.1 * float(heading_matches)
    score += 0.7 * float(_refs_summary_focus_keyword_hit_count(prompt, surface))
    score += 0.5 * float(_refs_exact_focus_match_count(prompt, surface))
    if bool(primary.get("strict_locate")):
        score += 0.8
    if str(primary.get("block_id") or "").strip():
        score += 0.6
    if str(primary.get("anchor_id") or "").strip():
        score += 0.25
    if heading_norm and re.search(r"\b(method|methods|procedure|algorithm|pipeline|model|experiment|results?|evaluation|analysis|concept)\b", heading_norm):
        score += 1.0
    if "refocus" in answer_norm and "refocus" in heading_norm:
        score += 4.0
    if ("ray tracing" in answer_norm or "wave optics" in answer_norm or "diffraction" in answer_norm) and re.search(
        r"\b(refocus|procedure|method|methods|concept)\b", heading_norm
    ):
        score += 2.2
    if heading_norm in {"abstract", "introduction"} or re.search(r"\b(?:abstract|introduction|related work|references)\b", heading_norm):
        if answer_term_hits <= 1:
            score -= 3.0
        else:
            score -= 0.8
    if _looks_like_front_matter_ref_summary(snippet):
        score -= 3.0
    if _looks_fragmentary_ref_summary(snippet):
        score -= 1.2
    if len(snippet) >= 80:
        score += 0.4
    return score, list(dict.fromkeys(matched))[:8]


def _candidate_primary_from_reader_open(
    reader_open: dict | None,
    *,
    source_path: str,
    display_name: str,
    selection_reason: str,
    score: float | None = None,
) -> dict:
    return _normalize_primary_ref_evidence_payload(
        _build_primary_ref_evidence_payload(
            source_path=source_path,
            display_name=display_name,
            reader_open=reader_open if isinstance(reader_open, dict) else {},
            selection_reason=selection_reason,
            score=score,
            prompt="",
        )
    )


def _iter_pack_primary_ref_evidence_candidates(pack: dict | None) -> list[dict]:
    pack2 = dict(pack or {}) if isinstance(pack, dict) else {}
    out: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()

    def _push(candidate: dict | None) -> None:
        norm = _normalize_primary_ref_evidence_payload(candidate if isinstance(candidate, dict) else {})
        if not norm:
            return
        key = (
            str(norm.get("source_path") or "").strip().lower(),
            str(norm.get("block_id") or "").strip().lower(),
            str(norm.get("anchor_id") or "").strip().lower(),
            _normalize_title_identity(
                " ".join(
                    part
                    for part in (
                        str(norm.get("heading_path") or "").strip(),
                        str(norm.get("highlight_snippet") or norm.get("snippet") or "").strip()[:160],
                    )
                    if part
                )
            ),
        )
        if key in seen:
            return
        seen.add(key)
        out.append(norm)

    existing = pack2.get("primary_evidence") if isinstance(pack2.get("primary_evidence"), dict) else {}
    _push(existing)
    for alt in list((existing or {}).get("alternatives") or []):
        _push(alt if isinstance(alt, dict) else {})

    for hit in list(pack2.get("hits") or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        source_path = str((ui_meta or {}).get("source_path") or (meta or {}).get("source_path") or "").strip()
        display_name = str((ui_meta or {}).get("display_name") or (ui_meta or {}).get("source_name") or "").strip()
        score_raw = (ui_meta or {}).get("score")
        try:
            score = float(score_raw) if score_raw is not None else None
        except Exception:
            score = None
        primary = _extract_hit_primary_ref_evidence(hit)
        _push(primary)
        for alt in list((primary or {}).get("alternatives") or []):
            _push(alt if isinstance(alt, dict) else {})
        reader_open = (ui_meta or {}).get("reader_open") if isinstance((ui_meta or {}).get("reader_open"), dict) else {}
        _push(_candidate_primary_from_reader_open(reader_open, source_path=source_path, display_name=display_name, selection_reason="reader_open", score=score))
        locate_target = (reader_open or {}).get("locateTarget") if isinstance((reader_open or {}).get("locateTarget"), dict) else {}
        if locate_target:
            target_reader = dict(reader_open or {})
            for key in ("headingPath", "snippet", "highlightSnippet", "blockId", "anchorId", "anchorKind", "anchorNumber"):
                if key in locate_target:
                    target_reader[key] = locate_target.get(key)
            target_reader["strictLocate"] = True
            _push(_candidate_primary_from_reader_open(target_reader, source_path=source_path, display_name=display_name, selection_reason="reader_open_locate", score=score))
        for raw_alt in list((reader_open or {}).get("evidenceAlternatives") or (reader_open or {}).get("visibleAlternatives") or (reader_open or {}).get("alternatives") or []):
            if not isinstance(raw_alt, dict):
                continue
            alt_reader = dict(reader_open or {})
            for key in ("headingPath", "snippet", "highlightSnippet", "blockId", "anchorId", "anchorKind", "anchorNumber"):
                alt_reader[key] = raw_alt.get(key)
            alt_reader["strictLocate"] = bool(raw_alt.get("blockId") or raw_alt.get("anchorId"))
            _push(_candidate_primary_from_reader_open(alt_reader, source_path=source_path, display_name=display_name, selection_reason="reader_open_alt", score=score))
    return out


def _pack_hit_source_rows(pack: dict | None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for hit in list((pack or {}).get("hits") or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        source_path = str((ui_meta or {}).get("source_path") or (meta or {}).get("source_path") or "").strip()
        if not source_path:
            continue
        key = source_path.lower()
        if key in seen:
            continue
        seen.add(key)
        display_name = str((ui_meta or {}).get("display_name") or (ui_meta or {}).get("source_name") or _source_filename(source_path)).strip()
        out.append((source_path, display_name))
        if len(out) >= 3:
            break
    return out


def _source_block_to_answer_primary_evidence(
    *,
    block: dict,
    prompt: str,
    source_path: str,
    display_name: str,
    terms: list[str] | None = None,
    selection_reason: str = "answer_aligned_block",
) -> dict:
    if not isinstance(block, dict):
        return {}
    block_id = str(block.get("block_id") or "").strip()
    if not block_id:
        return {}
    heading_path = _normalize_refs_reader_heading_path(
        prompt=prompt,
        source_path=source_path,
        heading_path=str(block.get("heading_path") or "").strip(),
    )
    block_text = str(block.get("text") or block.get("raw_text") or "").strip()
    snippet = _answer_aligned_block_snippet(block_text, terms=list(terms or []))
    if not snippet:
        snippet = _summary_excerpt(block_text, max_sentences=2, max_len=420) or _compact_reader_open_text(block_text)
    snippet = _clean_refs_evidence_snippet(
        snippet,
        prompt=prompt,
        source_path=source_path,
        display_name=display_name,
        heading_path=heading_path,
        max_len=460,
    )
    evidence = {
        "source_path": str(source_path or "").strip() or None,
        "source_name": str(display_name or "").strip() or None,
        "block_id": block_id,
        "anchor_id": str(block.get("anchor_id") or "").strip() or None,
        "heading_path": heading_path or None,
        "snippet": snippet or None,
        "highlight_snippet": snippet or None,
        "anchor_kind": str(block.get("kind") or "").strip().lower() or None,
        "anchor_number": _positive_int(block.get("number")) or None,
        "selection_reason": selection_reason,
        "strict_locate": True,
    }
    return _normalize_primary_ref_evidence_payload(evidence)


def _answer_aligned_block_snippet(block_text: str, *, terms: list[str]) -> str:
    text = str(block_text or "").strip()
    if not text or not terms:
        return ""
    sentences = [part.strip() for part in re.split(r"(?<=[。！？?\.])\s+", _clean_summary_line(text)) if part.strip()]
    if not sentences:
        return ""
    rows: list[tuple[float, int, str]] = []
    for idx, sentence in enumerate(sentences):
        matched = [term for term in terms if _answer_evidence_term_matches_surface(term, sentence)]
        if not matched:
            continue
        score = float(len(matched))
        sentence_norm = _normalize_title_identity(sentence)
        if re.search(r"\b(refocus|ray tracing|ray transfer|wave propagation|diffraction|angular spectrum)\b", sentence_norm):
            score += 1.2
        if 50 <= len(sentence) <= 260:
            score += 0.4
        rows.append((score, idx, sentence))
    if not rows:
        return ""
    rows.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    picked_idx = sorted(idx for _score, idx, _sentence in rows[:3])
    picked = [sentences[idx] for idx in picked_idx]
    snippet = " ".join(picked).strip()
    return _summary_excerpt(snippet, max_sentences=3, max_len=420) or _compact_reader_open_text(snippet, max_len=420)


def _select_answer_aligned_source_block_primary_evidence(
    *,
    pack: dict | None,
    prompt: str,
    answer: str,
    terms: list[str],
) -> tuple[dict, dict]:
    if not terms or not str(answer or "").strip():
        return {}, {}
    rows: list[dict] = []
    for source_path, display_name in _pack_hit_source_rows(pack):
        md_path = _resolve_source_md_path(source_path)
        if md_path is None:
            continue
        try:
            blocks = load_source_blocks(md_path)
        except Exception:
            blocks = []
        for block in list(blocks or [])[:1600]:
            if not isinstance(block, dict):
                continue
            kind = str(block.get("kind") or "").strip().lower()
            if kind in {"heading", "code"}:
                continue
            text = str(block.get("text") or block.get("raw_text") or "").strip()
            heading_path = str(block.get("heading_path") or "").strip()
            if len(text) < 30:
                continue
            candidate = _source_block_to_answer_primary_evidence(
                block=block,
                prompt=prompt,
                source_path=source_path,
                display_name=display_name,
                terms=terms,
            )
            if not candidate:
                continue
            score, matched = _score_primary_ref_evidence_against_answer(
                primary_evidence=candidate,
                prompt=prompt,
                answer=answer,
                terms=terms,
                display_name=display_name,
                source_path=source_path,
            )
            if not matched:
                continue
            heading_norm = _normalize_title_identity(heading_path)
            if kind == "paragraph":
                score += 0.35
            if kind in {"figure", "table"}:
                score += 0.2
            if re.search(r"\b(references|bibliography)\b", heading_norm):
                score -= 8.0
            rows.append({"score": float(score), "matched": matched, "primary": candidate})
    if not rows:
        return {}, {}
    rows.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            len(list(item.get("matched") or [])),
            1 if str(((item.get("primary") or {}) if isinstance(item.get("primary"), dict) else {}).get("heading_path") or "").strip() else 0,
        ),
        reverse=True,
    )
    best = dict(rows[0].get("primary") or {}) if isinstance(rows[0].get("primary"), dict) else {}
    if not best:
        return {}, {}
    alternatives = [
        dict(row.get("primary") or {})
        for row in rows[1:4]
        if isinstance(row.get("primary"), dict) and row.get("primary")
    ]
    if alternatives:
        best["alternatives"] = alternatives
    alignment = {
        "source": "source_blocks",
        "score": round(float(rows[0].get("score") or 0.0), 3),
        "matched_answer_terms": list(dict.fromkeys([str(item) for item in list(rows[0].get("matched") or []) if str(item or "").strip()]))[:8],
        "selected_heading_path": str(best.get("heading_path") or "").strip(),
    }
    return best, alignment


def _select_answer_aligned_primary_ref_evidence(
    *,
    pack: dict | None,
    prompt: str,
    answer: str,
) -> tuple[dict, dict]:
    terms = _answer_evidence_terms(prompt, answer)
    candidates = _iter_pack_primary_ref_evidence_candidates(pack)
    best_existing: dict = {}
    best_existing_score = -1000.0
    best_existing_matches: list[str] = []
    for candidate in candidates:
        source_path = str(candidate.get("source_path") or "").strip()
        display_name = str(candidate.get("source_name") or _source_filename(source_path) or "").strip()
        score, matched = _score_primary_ref_evidence_against_answer(
            primary_evidence=candidate,
            prompt=prompt,
            answer=answer,
            terms=terms,
            display_name=display_name,
            source_path=source_path,
        )
        if score > best_existing_score:
            best_existing = dict(candidate)
            best_existing_score = float(score)
            best_existing_matches = list(matched)

    block_primary, block_alignment = _select_answer_aligned_source_block_primary_evidence(
        pack=pack,
        prompt=prompt,
        answer=answer,
        terms=terms,
    )
    block_score = float((block_alignment or {}).get("score") or -1000.0) if block_primary else -1000.0
    chosen = dict(best_existing)
    chosen_score = best_existing_score
    chosen_matches = list(best_existing_matches)
    chosen_source = "existing"
    if block_primary and (
        (not chosen)
        or block_score >= max(4.8, best_existing_score + 1.0)
        or (best_existing_score < 4.0 and block_score >= 4.8)
    ):
        chosen = dict(block_primary)
        chosen_score = block_score
        chosen_matches = list((block_alignment or {}).get("matched_answer_terms") or [])
        chosen_source = "source_blocks"

    mismatch = bool(terms and (not chosen or chosen_score < 3.2 or len(chosen_matches) <= 0))
    alignment = {
        "answer_term_count": int(len(terms)),
        "answer_terms": list(terms[:12]),
        "matched_answer_terms": list(dict.fromkeys([str(item) for item in chosen_matches if str(item or "").strip()]))[:8],
        "score": round(float(chosen_score if chosen else 0.0), 3),
        "selected_heading_path": str((chosen or {}).get("heading_path") or "").strip(),
        "selected_source": chosen_source if chosen else "",
        "mismatch": bool(mismatch),
    }
    if block_alignment:
        alignment["best_source_block_score"] = block_alignment.get("score")
        alignment["best_source_block_heading_path"] = block_alignment.get("selected_heading_path")
        alignment["best_source_block_terms"] = block_alignment.get("matched_answer_terms")
    return chosen, {key: value for key, value in alignment.items() if value not in (None, "", [], {})}


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
    prompt = str(pack2.get("prompt") or "").strip()
    answer = str(pack2.get("answer") or pack2.get("answer_text") or "").strip()
    aligned_primary: dict = {}
    alignment: dict = {}
    if answer:
        aligned_primary, alignment = _select_answer_aligned_primary_ref_evidence(
            pack=pack2,
            prompt=prompt,
            answer=answer,
        )
    existing = _normalize_primary_ref_evidence_payload(pack2.get("primary_evidence") if isinstance(pack2.get("primary_evidence"), dict) else {})
    primary = aligned_primary or existing
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
    if alignment:
        pack2["primary_evidence_alignment"] = dict(alignment)
        pipeline_debug = dict(pack2.get("pipeline_debug") or {}) if isinstance(pack2.get("pipeline_debug"), dict) else {}
        pipeline_debug["primary_evidence_mismatch"] = bool(alignment.get("mismatch"))
        pipeline_debug["primary_evidence_score"] = alignment.get("score")
        pipeline_debug["primary_evidence_selected_source"] = str(alignment.get("selected_source") or "").strip()
        pack2["pipeline_debug"] = pipeline_debug
    return pack2


_SUGGESTIONS: dict[str, str] = {
    "no_candidate_hits": "No documents matched the query. Try rephrasing with different keywords, or check that relevant documents are ingested in the knowledge base.",
    "score_gate_removed_all": "All BM25 scores were below the relevance threshold. Try a more specific query.",
    "focus_filter_removed_all": "All hits were filtered out because they did not match the prompt's focus terms. Try broadening the question or removing specific constraints.",
    "llm_filter_removed_all": "The LLM relevance filter judged all hits as irrelevant. This may indicate a vocabulary mismatch between the query and documents.",
    "guide_self_source_only": "Guide mode hides the bound source paper. Disable guide mode or ask about other papers.",
    "render_failed": "The reference card rendering pipeline failed unexpectedly. Check server logs for error details.",
    "pending_enrichment": "Results are still being computed. Try again in a few seconds.",
    "no_renderable_hits": "Hits entered the pipeline but none could be rendered as reference cards. Check the pipeline stage counts for details.",
}


def _attach_pack_display_contract(pack: dict | None) -> dict:
    pack2 = _attach_pack_primary_ref_evidence(pack)
    hits = [hit for hit in list(pack2.get("hits") or []) if isinstance(hit, dict)]
    guide_filter = pack2.get("guide_filter") if isinstance(pack2.get("guide_filter"), dict) else {}
    pipeline_debug = pack2.get("pipeline_debug") if isinstance(pack2.get("pipeline_debug"), dict) else {}
    payload_mode = str(pack2.get("payload_mode") or "").strip().lower()
    render_status = str(pack2.get("render_status") or "").strip().lower()
    pending = bool(pack2.get("pending")) or payload_mode == "pending" or (bool(pack2.get("enrichment_pending")) and (not hits))
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
        # Attach upstream diagnostic when retrieval returned zero raw hits.
        if not pipeline_debug.get("retrieval_diag"):
            query_info = pack2.get("used_query") or ""
            query_translated = bool(pack2.get("used_translation"))
            pack_prompt = str(pack2.get("prompt") or "").strip()
            pipeline_debug["retrieval_diag"] = {
                "used_query": str(query_info).strip() or str(pack_prompt)[:200],
                "query_translated": query_translated,
                "likely_empty_reason": (
                    "query_translated_cjk_mismatch"
                    if query_translated
                    else "no_documents_match"
                ),
            }

    pack2["display_state"] = display_state
    if suppression_reason:
        pack2["suppression_reason"] = suppression_reason
        pack2["suggestion"] = _SUGGESTIONS.get(suppression_reason, "No specific suggestion available for this state.")
    else:
        pack2.pop("suppression_reason", None)
        pack2.pop("suggestion", None)
    return attach_refs_pack_polish_contract(pack2)


def _doc_list_ref_why_line(*, prompt: str, heading_path: str, prefer_zh: bool) -> str:
    heading = str(heading_path or "").strip()
    focus_terms = [t for t in _refs_prompt_focus_terms(prompt) if t]
    terms_str = ", ".join(focus_terms[:3]) if focus_terms else ""
    if prefer_zh:
        if heading and terms_str:
            return f"「{heading}」中有「{terms_str}」的原文线索，可用来核对论文怎样使用这些概念。"
        if heading:
            return f"可先查看「{heading}」中的原文线索。"
        if terms_str:
            return f"本文包含「{terms_str}」的可核对线索。"
        return "本文作为库内候选保留，可展开查看具体证据。"
    if heading and terms_str:
        return f"'{heading}' contains evidence about '{terms_str}' for checking the paper's usage."
    if heading:
        return f"Start with '{heading}' to inspect the source evidence."
    if terms_str:
        return f"This paper contains inspectable evidence about '{terms_str}'."
    return f"This paper was kept as a library candidate; expand it to inspect the evidence."


def _collect_doc_list_ref_text_candidates(*, raw_item: dict, primary_evidence: dict) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        text = _clean_refs_evidence_snippet(
            str(value or "").strip(),
            prompt="",
            source_path=str(raw_item.get("source_path") or primary_evidence.get("source_path") or "").strip(),
            display_name=str(raw_item.get("source_name") or primary_evidence.get("source_name") or "").strip(),
            heading_path=str(raw_item.get("heading_path") or primary_evidence.get("heading_path") or "").strip(),
            max_len=460,
        )
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
    return _clean_refs_evidence_snippet(
        str(primary.get("highlight_snippet") or primary.get("snippet") or "").strip(),
        prompt="",
        source_path=str(primary.get("source_path") or "").strip(),
        display_name=str(primary.get("source_name") or "").strip(),
        heading_path=str(primary.get("heading_path") or "").strip(),
        max_len=360,
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
    authoritative_summary_generation: str = "",
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
    current_summary_generation = str(ui_out.get("summary_generation") or "").strip().lower()
    current_summary_is_llm = current_summary_generation in {"llm_grounded", "llm_pack"}
    effective_summary_seed = _primary_ref_evidence_summary_seed(effective_primary)
    authoritative_summary_seed = _compact_reader_open_text(str(authoritative_summary_line or "").strip())
    authoritative_summary_generation_norm = str(authoritative_summary_generation or "").strip().lower()
    authoritative_summary_is_llm = authoritative_summary_generation_norm in {"llm_grounded", "llm_pack"}
    if authoritative_summary_seed and (not authoritative_summary_is_llm) and _summary_line_needs_polish(
        prompt=prompt,
        title=display_name,
        summary_line=authoritative_summary_seed,
    ):
        authoritative_summary_seed = ""
    if (not authoritative_summary_seed) and authoritative_primary:
        authoritative_summary_seed = _primary_ref_evidence_summary_seed(authoritative_primary)
        if authoritative_summary_seed and _summary_line_needs_polish(
            prompt=prompt,
            title=display_name,
            summary_line=authoritative_summary_seed,
        ):
            authoritative_summary_seed = ""
    authoritative_conflicts_with_synthesized = bool(
        selected_source == "authoritative"
        and authoritative_primary
        and synthesized_primary
        and (not _primary_ref_evidence_points_to_same_surface(authoritative_primary, synthesized_primary))
    )
    if authoritative_conflicts_with_synthesized and authoritative_summary_seed:
        ui_out["summary_line"] = authoritative_summary_seed
        if authoritative_summary_is_llm:
            summary_basis_meta = _build_ref_summary_basis_meta(
                prompt=prompt,
                summary_kind="guide",
                summary_generation=authoritative_summary_generation_norm,
                summary_line=authoritative_summary_seed,
            )
            ui_out["summary_generation"] = str(
                summary_basis_meta.get("summary_generation") or authoritative_summary_generation_norm
            )
            ui_out["summary_basis"] = str(summary_basis_meta.get("summary_basis") or "")
    if effective_summary_seed and (
        (not str(ui_out.get("summary_line") or "").strip())
        or (
            (not current_summary_is_llm)
            and
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
        snippet = _clean_refs_evidence_snippet(
            str(candidate.get("highlight_snippet") or candidate.get("snippet") or "").strip(),
            prompt="",
            source_path=str(candidate.get("source_path") or "").strip(),
            heading_path=loc_heading,
            max_len=360,
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
    auth_snippet = _clean_refs_evidence_snippet(
        str(primary.get("snippet") or out.get("snippet") or summary_line or "").strip(),
        prompt="",
        source_path=source_path,
        display_name=source_name,
        heading_path=auth_heading,
        max_len=460,
    )
    auth_highlight = _clean_refs_evidence_snippet(
        str(primary.get("highlight_snippet") or auth_snippet or out.get("highlightSnippet") or "").strip(),
        prompt="",
        source_path=source_path,
        display_name=source_name,
        heading_path=auth_heading,
        max_len=460,
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


def _build_doc_list_hit_ui_seed(
    *,
    raw_item: dict,
    idx: int,
    prompt: str,
) -> tuple[dict, dict, dict]:
    hit = _build_doc_list_ref_hit(raw_item=raw_item, idx=idx)
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    source_path = str((meta or {}).get("source_path") or raw_item.get("source_path") or "").strip()
    source_name = str((meta or {}).get("source_name") or raw_item.get("source_name") or "").strip() or _source_filename(source_path) or f"Reference {idx}"
    primary_evidence = _normalize_primary_ref_evidence_payload(
        raw_item.get("primary_evidence") if isinstance(raw_item.get("primary_evidence"), dict) else {}
    )
    authoritative_summary_line = _compact_reader_open_text(str(raw_item.get("summary_line") or "").strip())
    authoritative_summary_generation = (
        str(raw_item.get("summary_generation") or "").strip().lower()
        if authoritative_summary_line
        else ""
    )
    authoritative_why_line = _normalize_ref_copy_text(str(raw_item.get("why_line") or "").strip())
    authoritative_why_generation = (
        str(raw_item.get("why_generation") or "").strip().lower()
        if authoritative_why_line
        else ""
    )
    heading_context = _resolve_ref_ui_heading_context(
        prompt=prompt,
        source_path=source_path,
        heading_path=str((meta or {}).get("ref_best_heading_path") or raw_item.get("heading_path") or "").strip(),
        heading_fallback=str(
            (meta or {}).get("top_heading")
            or _top_heading(str(raw_item.get("heading_path") or ""))
            or ""
        ).strip(),
        section_label=str((meta or {}).get("ref_section") or "").strip(),
        subsection_label=str((meta or {}).get("ref_subsection") or "").strip(),
    )
    heading_path = str(heading_context.get("heading_path") or raw_item.get("heading_path") or "").strip()
    heading = str(heading_context.get("heading") or "").strip()
    section_label = str(heading_context.get("section_label") or "").strip()
    subsection_label = str(heading_context.get("subsection_label") or "").strip()
    summary_seed = authoritative_summary_line or _compact_reader_open_text(
        str(
            _primary_ref_evidence_summary_seed(primary_evidence)
            or primary_evidence.get("highlight_snippet")
            or primary_evidence.get("snippet")
            or ""
        ).strip()
    )
    summary_generation = authoritative_summary_generation or "section_grounded"
    summary_basis_meta = (
        _build_ref_summary_basis_meta(
            prompt=prompt,
            summary_kind="guide",
            summary_generation=summary_generation,
            summary_line=summary_seed,
        )
        if summary_seed
        else {}
    )
    why_seed = authoritative_why_line or _build_prompt_aligned_ref_why_line_v3(
        prompt=prompt,
        display_name=source_name,
        heading_path=heading_path,
        summary_line=summary_seed,
        why_line="",
    )
    if not why_seed:
        why_seed = _doc_list_ref_why_line(
            prompt=prompt,
            heading_path=heading_path,
            prefer_zh=bool(_prefer_zh_ref_card_locale(prompt, source_name)),
        )
    why_generation = authoritative_why_generation or "deterministic_grounded"
    why_basis_meta = (
        _build_ref_why_basis_meta(
            prompt=prompt,
            why_generation=why_generation,
            why_line=why_seed,
        )
        if why_seed
        else {}
    )
    ui_meta = {
        "display_name": source_name,
        "heading_path": heading_path,
        "heading": heading,
        "section_label": section_label,
        "subsection_label": subsection_label,
        "page_start": None,
        "page_end": None,
        "summary_kind": "guide",
        "summary_label": "导读",
        "summary_title": "这条证据说明什么",
        "source_path": source_path,
        "citation_meta": {},
    }
    if summary_seed:
        ui_meta["summary_line"] = summary_seed
        ui_meta["summary_generation"] = str(summary_basis_meta.get("summary_generation") or summary_generation)
        ui_meta["summary_basis"] = str(summary_basis_meta.get("summary_basis") or "")
    if why_seed:
        ui_meta["why_line"] = why_seed
        ui_meta["why_generation"] = str(why_basis_meta.get("why_generation") or why_generation)
        ui_meta["why_basis"] = str(why_basis_meta.get("why_basis") or "")
    anchor_target_kind = str((meta or {}).get("anchor_target_kind") or "").strip().lower()
    anchor_target_number = _positive_int((meta or {}).get("anchor_target_number"))
    anchor_match_score = _non_negative_float((meta or {}).get("anchor_match_score"))
    explicit_doc_match_score = _non_negative_float((meta or {}).get("explicit_doc_match_score"))
    if anchor_target_kind:
        ui_meta["anchor_target_kind"] = anchor_target_kind
    if anchor_target_number > 0:
        ui_meta["anchor_target_number"] = anchor_target_number
    if anchor_match_score > 0.0:
        ui_meta["anchor_match_score"] = anchor_match_score
    if explicit_doc_match_score > 0.0:
        ui_meta["explicit_doc_match_score"] = explicit_doc_match_score
    return hit, ui_meta, primary_evidence


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
    authoritative_summary_line = _compact_reader_open_text(str(raw_item.get("summary_line") or "").strip())
    primary_evidence = _normalize_primary_ref_evidence_payload(
        raw_item.get("primary_evidence") if isinstance(raw_item.get("primary_evidence"), dict) else {}
    )
    auth_reason = str(primary_evidence.get("selection_reason") or "").strip().lower()
    authoritative_primary_weak = bool(
        primary_evidence
        and (not str(primary_evidence.get("snippet") or primary_evidence.get("highlight_snippet") or "").strip())
        and (not str(primary_evidence.get("block_id") or "").strip())
        and auth_reason in {"", "answer_hit_top", "pending_section_seed"}
    )
    summary_source = ""
    if authoritative_primary_weak:
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
        # Chain A already writes summary_source — capture it.
        summary_source = str(ui_meta.get("summary_source") or "").strip()
    else:
        hit, ui_meta, primary_evidence = _build_doc_list_hit_ui_seed(
            raw_item=raw_item,
            idx=idx,
            prompt=prompt,
        )
        summary_source = "doc_list_seed"
    heading_path = (
        str(ui_meta.get("heading_path") or "").strip()
        or str(raw_item.get("heading_path") or "").strip()
        or str(primary_evidence.get("heading_path") or "").strip()
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
        authoritative_summary_generation=str(raw_item.get("summary_generation") or "").strip(),
    )
    if not str(ui_meta.get("heading_path") or "").strip() and heading_path:
        ui_meta["heading_path"] = heading_path
    current_summary = str(ui_meta.get("summary_line") or "").strip()
    current_summary_generation = str(ui_meta.get("summary_generation") or "").strip().lower()
    current_summary_is_llm = current_summary_generation in {"llm_grounded", "llm_pack"}
    display_name = str(ui_meta.get("display_name") or source_name).strip()
    if (not current_summary_is_llm) and current_summary and (
        _summary_line_needs_polish(
            prompt=prompt,
            title=display_name,
            summary_line=current_summary,
        )
        or _looks_like_title_echo(current_summary, display_name)
        or _looks_why_like_ref_summary(current_summary)
    ):
        fallback_summary = _pick_ref_card_summary_fallback(
            prompt=prompt,
            title=display_name,
            candidates=_collect_doc_list_ref_text_candidates(
                raw_item=raw_item,
                primary_evidence=effective_primary_evidence or primary_evidence,
            ),
        )
        if fallback_summary and (not _looks_like_title_echo(fallback_summary, display_name)):
            summary_basis_meta = _build_ref_summary_basis_meta(
                prompt=prompt,
                summary_kind=str(ui_meta.get("summary_kind") or "guide"),
                summary_generation="deterministic_grounded",
                summary_line=fallback_summary,
            )
            ui_meta["summary_line"] = fallback_summary
            ui_meta["summary_generation"] = str(
                summary_basis_meta.get("summary_generation") or "deterministic_grounded"
            )
            ui_meta["summary_basis"] = str(summary_basis_meta.get("summary_basis") or "")
            summary_source = "doc_list_fallback"
    current_summary = str(ui_meta.get("summary_line") or "").strip()
    current_summary_generation = str(ui_meta.get("summary_generation") or "").strip().lower()
    current_summary_is_llm = current_summary_generation in {"llm_grounded", "llm_pack"}
    if (not current_summary_is_llm) and (
        (not current_summary)
        or _looks_like_title_echo(current_summary, display_name)
        or _looks_why_like_ref_summary(current_summary)
        or _looks_fragmentary_ref_summary(current_summary)
        or _looks_surface_like_ref_summary(current_summary)
        or _looks_formula_heavy_ref_text(current_summary)
    ):
        template_summary = _build_prompt_aligned_ref_summary_fallback(
            prompt=prompt,
            display_name=display_name,
            heading_path=str(ui_meta.get("heading_path") or heading_path),
            summary_line=current_summary,
            why_line=str(ui_meta.get("why_line") or ""),
        )
        if template_summary and (not _summary_line_needs_polish(
            prompt=prompt,
            title=display_name,
            summary_line=template_summary,
        )):
            summary_basis_meta = _build_ref_summary_basis_meta(
                prompt=prompt,
                summary_kind=str(ui_meta.get("summary_kind") or "guide"),
                summary_generation="deterministic_grounded",
                summary_line=template_summary,
            )
            ui_meta["summary_line"] = template_summary
            ui_meta["summary_generation"] = str(
                summary_basis_meta.get("summary_generation") or "deterministic_grounded"
            )
            ui_meta["summary_basis"] = str(summary_basis_meta.get("summary_basis") or "")
            if summary_source != "doc_list_fallback":
                summary_source = "doc_list_prompt_aligned"
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
            summary_source = "doc_list_ultimate_seed"
    # 5th-level ultimate fallback: raw snippet text from the hit
    if not str(ui_meta.get("summary_line") or "").strip():
        raw_snippets: list[str] = []
        for h in list(raw_item.get("hits") or []):
            txt = str(h.get("text") or h.get("snippet") or "").strip()
            if txt and len(txt) > 30:
                raw_snippets.append(txt)
        if not raw_snippets:
            for alt_key in ("summary_line", "highlight_snippet", "snippet", "text"):
                txt = str(raw_item.get(alt_key) or "").strip()
                if txt and len(txt) > 30:
                    raw_snippets.append(txt)
        if raw_snippets:
            fallback_raw = raw_snippets[0][:200].rsplit(" ", 1)[0] if len(raw_snippets[0]) > 200 else raw_snippets[0]
            ui_meta["summary_line"] = fallback_raw
            ui_meta["summary_generation"] = "raw_fallback"
            summary_source = "doc_list_raw_fallback"
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
    aligned_summary_line, aligned_why_line = _align_ref_card_copy_to_user_locale(
        prompt=prompt,
        display_name=str(ui_meta.get("display_name") or source_name),
        heading_path=str(ui_meta.get("heading_path") or heading_path),
        summary_line=str(ui_meta.get("summary_line") or ""),
        why_line=str(ui_meta.get("why_line") or ""),
        summary_kind=str(ui_meta.get("summary_kind") or "guide"),
        allow_llm_translate=bool(allow_expensive_llm),
    )
    if aligned_summary_line:
        ui_meta["summary_line"] = aligned_summary_line
    if aligned_why_line:
        ui_meta["why_line"] = aligned_why_line
    summary_surface = _build_ref_summary_surface_meta(
        prompt=prompt,
        summary_kind=str(ui_meta.get("summary_kind") or "guide"),
        summary_line=str(ui_meta.get("summary_line") or ""),
    )
    ui_meta["summary_kind"] = str(summary_surface.get("summary_kind") or ui_meta.get("summary_kind") or "guide")
    ui_meta["summary_label"] = str(summary_surface.get("summary_label") or "")
    ui_meta["summary_title"] = str(summary_surface.get("summary_title") or "")
    summary_generation = str(ui_meta.get("summary_generation") or "").strip().lower() or "deterministic_grounded"
    why_generation = str(ui_meta.get("why_generation") or "").strip().lower() or "deterministic_grounded"
    if str(ui_meta.get("summary_line") or "").strip():
        summary_basis_meta = _build_ref_summary_basis_meta(
            prompt=prompt,
            summary_kind=str(ui_meta.get("summary_kind") or "guide"),
            summary_generation=summary_generation,
            summary_line=str(ui_meta.get("summary_line") or ""),
        )
        ui_meta["summary_generation"] = str(summary_basis_meta.get("summary_generation") or summary_generation)
        ui_meta["summary_basis"] = str(summary_basis_meta.get("summary_basis") or "")
    if str(ui_meta.get("why_line") or "").strip():
        why_basis_meta = _build_ref_why_basis_meta(
            prompt=prompt,
            why_generation=why_generation,
            why_line=str(ui_meta.get("why_line") or ""),
        )
        ui_meta["why_generation"] = str(why_basis_meta.get("why_generation") or why_generation)
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
    ui_meta["summary_source"] = summary_source
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
    require_llm_copy = True
    current_why_is_llm = _is_llm_ref_why_generation(str(ui.get("why_generation") or ""))
    should_override = bool(
        note
        and (not (require_llm_copy and current_why_is_llm))
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
        current_summary_is_llm = _is_llm_ref_summary_generation(str(ui.get("summary_generation") or ""))
        if fallback_summary and (
            not (require_llm_copy and current_summary_is_llm)
        ) and (
            (not current_summary)
            or _summary_line_needs_polish(
                prompt=prompt,
                title=display_name,
                summary_line=current_summary,
            )
            or bool(re.match(r"^[a-z][a-z0-9 -]{8,60}:\s", current_summary.lower()))
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
    filter_bound_source: bool = False,
) -> tuple[list[dict], int]:
    rows = [dict(item) for item in list(doc_rows or []) if isinstance(item, dict)]
    guide_path = str(guide_source_path or "").strip()
    guide_name = str(guide_source_name or "").strip()
    guide_active = bool(guide_mode and filter_bound_source and (guide_path or guide_name))
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
        filter_bound_source=prompt_cross_paper_refs,
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
            polished_hits: list[dict] = list(hits)
            jobs: list[tuple[int, dict, dict]] = []
            for idx, hit in enumerate(hits):
                ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
                if not isinstance(ui_meta, dict):
                    continue
                jobs.append((idx, hit, ui_meta))

            def _polish_one(idx: int, hit: dict, ui_meta: dict) -> tuple[int, dict]:
                polished_ui = _normalize_ref_copy_ui_meta(
                    _maybe_polish_single_ref_hit_card(
                        prompt=prompt,
                        hit=hit,
                        ui_meta=ui_meta,
                        allow_expensive_llm=bool(allow_expensive_llm),
                    )
                )
                polished_ui = _apply_doc_list_topic_match_hints(
                    prompt=prompt,
                    raw_item=doc_rows[idx],
                    ui_meta=polished_ui,
                )
                return idx, polished_ui

            batch_polished = (
                _batch_polish_doc_list_ref_hit_cards(
                    prompt=prompt,
                    jobs=jobs,
                )
                if bool(allow_expensive_llm)
                else {}
            )
            batch_polished = {
                int(idx): _apply_doc_list_topic_match_hints(
                    prompt=prompt,
                    raw_item=doc_rows[int(idx)],
                    ui_meta=dict(ui_meta or {}),
                )
                for idx, ui_meta in dict(batch_polished or {}).items()
                if str(idx).isdigit() or isinstance(idx, int)
            }
            leftover_jobs = [
                (idx, hit, ui_meta)
                for idx, hit, ui_meta in jobs
                if (
                    idx not in batch_polished
                    or (
                        bool(allow_expensive_llm)
                        and True
                        and not _ref_card_has_llm_copy(batch_polished.get(idx))
                    )
                )
            ]
            for idx, polished_ui in batch_polished.items():
                hit2 = dict(hits[idx])
                hit2["ui_meta"] = polished_ui
                polished_hits[idx] = hit2

            max_workers = _refs_card_polish_max_workers(len(leftover_jobs))
            if max_workers <= 1:
                for idx, hit, ui_meta in leftover_jobs:
                    _, polished_ui = _polish_one(idx, hit, ui_meta)
                    hit2 = dict(hit)
                    hit2["ui_meta"] = polished_ui
                    polished_hits[idx] = hit2
            else:
                try:
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        futs = [ex.submit(_polish_one, idx, hit, ui_meta) for idx, hit, ui_meta in leftover_jobs]
                        for fu in as_completed(futs):
                            try:
                                idx, polished_ui = fu.result()
                            except Exception:
                                continue
                            hit2 = dict(hits[idx])
                            hit2["ui_meta"] = polished_ui
                            polished_hits[idx] = hit2
                except Exception:
                    for idx, hit, ui_meta in leftover_jobs:
                        _, polished_ui = _polish_one(idx, hit, ui_meta)
                        hit2 = dict(hit)
                        hit2["ui_meta"] = polished_ui
                        polished_hits[idx] = hit2
            hits = polished_hits
        if bool(allow_expensive_llm) and True:
            hits = _suppress_non_llm_ref_card_copy_hits(prompt=prompt, hits=hits)
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
        pipeline_debug["copy_polish_allow_expensive_llm"] = bool(allow_expensive_llm)
        pipeline_debug["copy_polish_llm_required"] = True
        pipeline_debug["copy_polish_llm_complete"] = bool(_refs_hits_have_llm_copy(hits))
        raw_qv = pack_src.get("query_variants") if isinstance(pack_src, dict) else []
        if raw_qv:
            pipeline_debug["query_variants"] = list(raw_qv)
        pack_out["pipeline_debug"] = pipeline_debug
        if guide_active:
            hidden_self_source = bool(prompt_cross_paper_refs and (filtered_self_doc_count > 0 or not hits))
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
            "summary_title": "这条证据说明什么" if prefer_zh else "What This Evidence Shows",
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
    if extract_figure_number(prompt) > 0 or extract_equation_number(prompt) > 0:
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
    candidate_title = str(
        (citation_meta or {}).get("title")
        or (meta or {}).get("title")
        or display_name
        or ""
    ).strip()

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
    preferred_exact_candidate: dict = {}

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
            or (bool(str(anchor_target_kind or "").strip()) and anchor_target_number > 0)
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
                title=candidate_title,
                anchor_target_kind=anchor_target_kind,
                anchor_target_number=anchor_target_number,
                allow_llm_translate=allow_llm_translate,
            )
    prompt_aligned_candidate = _pick_best_prompt_aligned_ref_summary_candidate(
        [meta_prompt_aligned_candidate, block_prompt_aligned_candidate],
        prompt=prompt,
        source_path=source_path,
        title=candidate_title,
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
        if candidate_heading_path and anchor_target_kind and anchor_target_number > 0:
            candidate_anchor_num = _refs_heading_anchor_number(anchor_target_kind, candidate_heading_path)
            if candidate_anchor_num > 0 and candidate_anchor_num != anchor_target_number:
                candidate_heading_path = ""
            elif (
                candidate_anchor_num <= 0
                and heading_path
                and (not _refs_heading_paths_related(candidate_heading_path, heading_path))
            ):
                candidate_heading_path = ""
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
            title=candidate_title,
            text=summary_line,
            anchor_target_kind=anchor_target_kind,
            anchor_target_number=anchor_target_number,
        ) if summary_line else -1000.0
        chosen_score = _ref_summary_focus_score(
            prompt=prompt,
            source_path=source_path,
            title=candidate_title,
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
    preferred_exact_candidate = _build_preferred_refs_exact_candidate_from_source_summary(
        prompt=prompt,
        source_path=source_path,
        title=candidate_title,
        summary_line=summary_line,
        selected_heading_path=str(resolved_heading_context.get("heading_path") or "").strip(),
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        prompt_aligned_candidate=prompt_aligned_candidate,
    )

    return {
        "nav": nav,
        "summary_line": summary_line,
        "summary_source": summary_source,
        "used_nav_summary": used_nav_summary,
        "used_prompt_aligned_summary": used_prompt_aligned_summary,
        "preferred_exact_candidate": preferred_exact_candidate,
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
    initial_heading_path = str(meta.get("ref_best_heading_path") or meta.get("heading_path") or "").strip()
    leading_text_heading = _leading_markdown_heading_from_hit_text(str((hit or {}).get("text") or ""))
    if leading_text_heading:
        current_heading_score = _refs_section_intent_heading_score(prompt, initial_heading_path)
        leading_heading_score = _refs_section_intent_heading_score(prompt, leading_text_heading)
        current_norm = _normalize_title_identity(initial_heading_path)
        leading_norm = _normalize_title_identity(leading_text_heading)
        if (
            (not current_norm)
            or current_norm in {"abstract", "references"}
            or (leading_heading_score >= current_heading_score + 0.75 and leading_norm and leading_norm not in current_norm)
        ):
            initial_heading_path = leading_text_heading
    heading_context = _resolve_ref_ui_heading_context(
        prompt=prompt,
        source_path=source_path,
        heading_path=initial_heading_path,
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
    if (not anchor_target_kind) or anchor_target_number <= 0:
        prompt_figure_number = extract_figure_number(prompt)
        if prompt_figure_number > 0:
            anchor_target_kind = "figure"
            anchor_target_number = prompt_figure_number
        else:
            prompt_equation_number = extract_equation_number(prompt)
            if prompt_equation_number > 0:
                anchor_target_kind = "equation"
                anchor_target_number = prompt_equation_number
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
    if bool(meta.get("section_intent_rescue")):
        rescue_heading_path = str(meta.get("ref_best_heading_path") or meta.get("heading_path") or "").strip()
        if rescue_heading_path:
            heading_path = rescue_heading_path
            heading = str(rescue_heading_path.split(" / ")[-1] if " / " in rescue_heading_path else rescue_heading_path).strip()
            section_label = str(meta.get("ref_section") or _top_heading(rescue_heading_path) or "").strip()
            subsection_label = str(meta.get("ref_subsection") or heading).strip()
        rescue_summary = _summary_excerpt(str((hit or {}).get("text") or ""), max_sentences=2, max_len=260)
        if rescue_summary:
            summary_line = rescue_summary
            summary_source = "section_intent_rescue"
    preferred_exact_candidate = (
        dict(primary_evidence.get("preferred_exact_candidate") or {})
        if isinstance(primary_evidence.get("preferred_exact_candidate"), dict)
        else {}
    )
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
    summary_kind = _infer_ref_summary_kind(
        summary_line=summary_line,
        citation_meta=citation_meta if isinstance(citation_meta, dict) else {},
        used_prompt_aligned_summary=used_prompt_aligned_summary,
        used_nav_summary=used_nav_summary,
    )
    summary_line, why_line = _align_ref_card_copy_to_user_locale(
        prompt=prompt,
        display_name=display_name,
        heading_path=heading_path or heading,
        summary_line=summary_line,
        why_line=why_line,
        summary_kind=summary_kind,
        allow_llm_translate=bool(allow_expensive_llm),
    )
    copy_focus_terms = [
        _display_focus_term_for_ref_card(prompt, term)
        for term in _matched_focus_terms_for_ref_card(
            prompt,
            surface_text=" ".join(
                part
                for part in (display_name, heading_path or heading, summary_line, why_line)
                if str(part or "").strip()
            ),
        )
    ]
    summary_line, why_line, copy_changed = _finalize_ref_card_copy(
        summary_line=summary_line,
        why_line=why_line,
        prefer_zh=_prefer_zh_ref_card_locale(prompt, display_name, heading_path or heading, summary_line, why_line),
        focus_terms=copy_focus_terms,
        heading_path=heading_path or heading,
        action=_shared_prompt_reference_focus_action(prompt),
    )
    if copy_changed:
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
        preferred_exact_candidate=preferred_exact_candidate,
        allow_llm_disambiguation=allow_expensive_llm,
        allow_exact_locate=allow_exact_locate,
    )
    if isinstance(reader_open, dict) and anchor_target_kind and anchor_target_number > 0:
        reader_snippet = str(reader_open.get("snippet") or "").strip()
        reader_heading_path = str(reader_open.get("headingPath") or "").strip()
        reader_anchor_matches = bool(
            _refs_heading_anchor_number(anchor_target_kind, reader_heading_path) == anchor_target_number
            or _ref_summary_focus_score(
                prompt=prompt,
                source_path=source_path,
                title=display_name,
                text=reader_snippet,
                anchor_target_kind=anchor_target_kind,
                anchor_target_number=anchor_target_number,
            )
            >= 6.0
        )
        if reader_snippet and reader_anchor_matches:
            current_anchor_score = _ref_summary_focus_score(
                prompt=prompt,
                source_path=source_path,
                title=display_name,
                text=summary_line,
                anchor_target_kind=anchor_target_kind,
                anchor_target_number=anchor_target_number,
            )
            reader_anchor_score = _ref_summary_focus_score(
                prompt=prompt,
                source_path=source_path,
                title=display_name,
                text=reader_snippet,
                anchor_target_kind=anchor_target_kind,
                anchor_target_number=anchor_target_number,
            )
            if reader_anchor_score >= (current_anchor_score + 0.5):
                exact_summary = _build_evidence_backed_ref_summary_from_seed(
                    prompt=prompt,
                    title=display_name,
                    summary_line=reader_snippet,
                    prefer_zh=_prefer_zh_ref_card_locale(prompt, display_name, reader_snippet),
                ) or _summary_excerpt(reader_snippet, max_sentences=2, max_len=240)
                if exact_summary:
                    summary_line = _normalize_ref_copy_text(exact_summary)
                    summary_source = "exact_anchor"
    primary_evidence = _build_primary_ref_evidence_payload(
        source_path=source_path,
        display_name=display_name,
        reader_open=reader_open if isinstance(reader_open, dict) else {},
        selection_reason=summary_source,
        score=score,
        prompt=prompt,
    )
    if isinstance(reader_open, dict) and primary_evidence:
        reader_open = dict(reader_open)
        reader_open["primaryEvidence"] = dict(primary_evidence)
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

    return _build_ref_card_ui_payload(
        display_name=display_name,
        heading_path=heading_path or heading,
        section_label=section_label,
        subsection_label=subsection_label,
        page_start=p0,
        page_end=p1,
        score=score,
        score_pending=bool(score_pending),
        score_tier=_score_tier(score or 0.0) if score is not None else "",
        summary_line=summary_line,
        summary_kind=summary_kind,
        summary_surface=summary_surface,
        summary_generation=summary_generation,
        summary_basis_meta=summary_basis_meta,
        summary_source=summary_source,
        primary_evidence_heading_path=heading_path or heading,
        primary_evidence=primary_evidence if isinstance(primary_evidence, dict) else {},
        why_line=why_line,
        why_generation=why_generation,
        why_basis_meta=why_basis_meta,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        anchor_match_score=anchor_match_score,
        explicit_doc_match_score=explicit_doc_match_score,
        semantic_badges=semantic_badges,
        can_open=bool(pdf_path),
        citation_meta=citation_meta if isinstance(citation_meta, dict) else {},
        source_path=source_path,
        reader_open=reader_open if isinstance(reader_open, dict) else {},
    )


def _refs_hit_rerank_llm_enabled() -> bool:
    raw_flag = str(os.environ.get("KB_REFS_RERANK_USE_LLM", "0") or "").strip().lower()
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


def _refs_hit_raw_retrieval_score(hit: dict) -> float:
    meta = (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}
    rank = (meta or {}).get("ref_rank") if isinstance((meta or {}).get("ref_rank"), dict) else {}
    for value in (
        (rank or {}).get("display_score"),
        (rank or {}).get("score"),
        (hit or {}).get("_bm25_score"),
        (hit or {}).get("score"),
    ):
        try:
            return float(value or 0.0)
        except Exception:
            continue
    return 0.0


def _refs_has_decisive_raw_retrieval_leader(prompt: str, hits: list[dict]) -> bool:
    rows = [hit for hit in list(hits or []) if isinstance(hit, dict)]
    if len(rows) < 2:
        return False
    if (
        _prompt_requires_explicit_focus_match(prompt)
        and not _prompt_requests_compare(prompt)
    ) or _prompt_likely_cross_paper_refs(prompt):
        return False
    top = _refs_hit_raw_retrieval_score(rows[0])
    second = _refs_hit_raw_retrieval_score(rows[1])
    return bool(top >= 10.0 and (top - second) >= 3.0)


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


_ZH_PROMPT_FOCUS_ALIASES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("深度学习", "神经网络", "神经网路"), ("deep learning", "neural network")),
    (("单像素成像", "单像素", "鬼成像"), ("single-pixel imaging", "single pixel imaging", "computational ghost imaging")),
    (("硬件", "实验装置", "实验设置", "装置", "部件"), ("experimental setup", "setup", "hardware", "camera", "lens", "DMD")),
    (("结构化探测", "结构化检测"), ("structured detection", "structured detector")),
    (("激光扫描显微", "扫描显微"), ("laser scanning microscopy", "scanning microscopy")),
    (("图像扫描显微",), ("image scanning microscopy", "ISM")),
    (("共聚焦",), ("confocal", "confocal microscopy")),
    (("权衡", "矛盾", "折中"), ("trade-off", "tradeoff")),
    (("挑战", "局限"), ("challenge", "limitation")),
)


def _refs_prompt_focus_alias_terms(prompt: str) -> tuple[str, ...]:
    text = str(prompt or "").strip()
    if not text:
        return ()
    out: list[str] = []
    seen: set[str] = set()

    def _push(raw: str) -> None:
        norm = _normalize_title_identity(raw)
        if len(norm) < 3 or norm in seen:
            return
        seen.add(norm)
        out.append(norm)

    for triggers, aliases in _ZH_PROMPT_FOCUS_ALIASES:
        if any(trigger and trigger in text for trigger in triggers):
            for alias in aliases:
                _push(alias)
    if re.search(r"(?<![A-Za-z0-9])ISM(?![A-Za-z0-9])", text):
        _push("image scanning microscopy")
        _push("ISM")
    return tuple(out)


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
    for m in re.finditer(
        r"(?:比较|对比)(?:了)?\s*([^？?。.!]{2,140}?)(?:\s*(?:的)?(?:权衡|取舍|差异|区别|不同)|[？?。.!]|$)",
        text,
    ):
        raw = re.sub(r"^(?:哪些|哪几篇|哪几篇文献|哪些文献|文献|论文)\s*", "", str(m.group(1) or "").strip())
        _push(raw)
        for part in re.split(r"\s*(?:和|与|及|以及|、|/|\bvs\.?\b|\bversus\b|\band\b)\s*", raw, flags=re.IGNORECASE):
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
            and (not re.search(r"(?:\b(?:and|vs\.?|versus)\b|和|与|及|以及|、|/)", other, flags=re.IGNORECASE))
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
    for alias_term in _refs_prompt_focus_alias_terms(text):
        _push(alias_term)
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


def _leading_markdown_heading_from_hit_text(text: str) -> str:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw:
        return ""
    for line in raw.splitlines()[:8]:
        s = str(line or "").strip()
        if not s:
            continue
        m = re.match(r"^\s{0,3}#{1,6}\s+(.{2,180})\s*$", s)
        if not m:
            continue
        heading = re.sub(r"\s+", " ", str(m.group(1) or "").strip(" #*\t"))
        heading = re.sub(r"<[^>]+>", " ", heading).strip()
        return heading[:180]
    return ""


def _refs_hit_heading_candidates(hit: dict) -> list[str]:
    ui_meta = (hit or {}).get("ui_meta") if isinstance((hit or {}).get("ui_meta"), dict) else {}
    meta = (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}
    candidates = [
        str((ui_meta or {}).get("heading_path") or "").strip(),
        str((ui_meta or {}).get("primary_evidence_heading_path") or "").strip(),
        str((meta or {}).get("ref_best_heading_path") or "").strip(),
        str((meta or {}).get("heading_path") or "").strip(),
        str((meta or {}).get("ref_section") or "").strip(),
        _leading_markdown_heading_from_hit_text(str((hit or {}).get("text") or "")),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        s = str(item or "").strip()
        if not s:
            continue
        key = _normalize_title_identity(s)
        if (not key) or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _refs_prompt_section_intent(prompt: str) -> str:
    return _intent_prompt_section_intent(prompt)


def _refs_prompt_topic_terms(prompt: str) -> list[str]:
    return _intent_prompt_topic_terms(prompt)


def _refs_section_intent_terms(prompt: str, intent: str) -> tuple[str, ...]:
    return _intent_section_intent_terms(prompt, intent)


def _refs_section_intent_heading_score(prompt: str, heading: str) -> float:
    return _intent_section_intent_heading_score(prompt, heading)


def _refs_hit_section_intent_score(prompt: str, hit: dict) -> float:
    intent = _refs_prompt_section_intent(prompt)
    if not intent:
        return 0.0
    headings = _refs_hit_heading_candidates(hit)
    best = max((_refs_section_intent_heading_score(prompt, h) for h in headings), default=0.0)
    surface = _normalize_title_identity(
        " ".join(
            [
                str((hit or {}).get("text") or ""),
                " ".join(headings),
            ]
        )
    )
    for term in _refs_section_intent_terms(prompt, intent):
        norm = _normalize_title_identity(term)
        if norm and _focus_term_matches_surface(norm, surface):
            best += 0.45
    meta = (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}
    if bool((meta or {}).get("section_intent_rescue")):
        best += 1.0
    return best


def _refs_hit_matches_section_intent(prompt: str, hit: dict) -> bool:
    return _refs_hit_section_intent_score(prompt, hit) >= 4.5


def _source_path_prompt_match_boost(prompt: str, source_path: str) -> float:
    low_prompt = _normalize_title_identity(prompt)
    low_source = _normalize_title_identity(Path(str(source_path or "")).stem)
    if not low_prompt or not low_source:
        return 0.0
    boost = 0.0
    for token in re.findall(r"[a-z0-9]{4,}", low_source):
        if token in low_prompt:
            boost += 0.55
    return min(2.2, boost)


def _pick_section_intent_source_path(prompt: str, hits: list[dict]) -> str:
    rows = [hit for hit in (hits or []) if isinstance(hit, dict)]
    if not rows or not _refs_prompt_section_intent(prompt):
        return ""
    scored: list[tuple[float, str]] = []
    for idx, hit in enumerate(rows[:8]):
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or "").strip()
        if not source_path or is_excluded_source_path(source_path):
            continue
        raw_score = _refs_hit_raw_retrieval_score(hit)
        section_score = _refs_hit_section_intent_score(prompt, hit)
        source_boost = _source_path_prompt_match_boost(prompt, source_path)
        score = (0.18 * raw_score) + section_score + source_boost - (0.02 * idx)
        scored.append((score, source_path))
    if not scored:
        return ""
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _section_intent_block_score(*, prompt: str, block: dict) -> float:
    intent = _refs_prompt_section_intent(prompt)
    if not intent:
        return 0.0
    heading = str((block or {}).get("heading_path") or "").strip()
    text = str((block or {}).get("text") or "").strip()
    if not text:
        return 0.0
    kind = str((block or {}).get("kind") or "").strip().lower()
    if kind in {"equation", "figure", "table", "code"}:
        return 0.0
    score = _refs_section_intent_heading_score(prompt, heading)
    if re.search(r"\s*/\s*(?:figure|table)\s+\d+", heading, flags=re.I):
        score -= 2.5
    if intent == "experiments":
        h_norm = _normalize_title_identity(heading)
        text_norm = _normalize_title_identity(text)
        if "additional study" in h_norm and not re.search(r"\b(additional|ablation|compression|mask)\b", str(prompt or ""), flags=re.I):
            score -= 1.2
        if re.search(r"\b(empirical evidence|quantitative|qualitative|sota|state of the art|results demonstrate)\b", text_norm, flags=re.I):
            score += 1.4
    surface = _normalize_title_identity(f"{heading} {text}")
    for term in _refs_section_intent_terms(prompt, intent):
        norm = _normalize_title_identity(term)
        if norm and _focus_term_matches_surface(norm, surface):
            score += 0.85
    focus_hits = len(_matched_focus_terms_for_ref_card(prompt, surface_text=f"{heading} {text}"))
    score += min(2.4, 0.6 * float(focus_hits))
    if len(text) < 80:
        score -= 0.5
    return score


def _build_section_intent_rescue_hit(prompt: str, hits: list[dict]) -> dict | None:
    intent = _refs_prompt_section_intent(prompt)
    if not intent:
        return None
    source_path = _pick_section_intent_source_path(prompt, hits)
    if not source_path:
        return None
    md_path = _resolve_source_md_path(source_path)
    if md_path is None:
        return None
    try:
        blocks = load_source_blocks(md_path)
    except Exception:
        blocks = []
    if not blocks:
        return None
    ranked: list[tuple[float, dict]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        score = _section_intent_block_score(prompt=prompt, block=block)
        if score >= 4.25:
            ranked.append((score, block))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, block = ranked[0]
    heading_path = str(block.get("heading_path") or "").strip()
    text = str(block.get("text") or block.get("raw_text") or "").strip()
    if not heading_path or not text:
        return None
    template = next(
        (
            hit for hit in (hits or [])
            if isinstance(hit, dict)
            and str(((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "").strip() == source_path
        ),
        None,
    )
    if not isinstance(template, dict):
        return None
    meta0 = template.get("meta") if isinstance(template.get("meta"), dict) else {}
    rank0 = (meta0 or {}).get("ref_rank") if isinstance((meta0 or {}).get("ref_rank"), dict) else {}
    meta = dict(meta0 or {})
    ref_rank = dict(rank0 or {})
    ui_score = max(8.8, min(9.85, 7.25 + (0.22 * float(best_score))))
    ref_rank.update(
        {
            "llm": max(float(ref_rank.get("llm") or 0.0), ui_score * 10.0),
            "llm_score": max(float(ref_rank.get("llm_score") or 0.0), ui_score * 10.0),
            "bm25": max(float(ref_rank.get("bm25") or 0.0), 5.0),
            "term_bonus": max(float(ref_rank.get("term_bonus") or 0.0), 2.4),
            "semantic_score": max(float(ref_rank.get("semantic_score") or 0.0), ui_score),
            "display_score": max(float(ref_rank.get("display_score") or 0.0), ui_score),
            "section_intent": intent,
        }
    )
    meta.update(
        {
            "heading_path": heading_path,
            "top_heading": _top_heading(heading_path),
            "ref_best_heading_path": heading_path,
            "ref_section": _top_heading(heading_path),
            "ref_subsection": str(heading_path.split(" / ")[-1] if " / " in heading_path else heading_path).strip(),
            "ref_loc_quality": "high",
            "ref_pack_state": str(meta.get("ref_pack_state") or "ready").strip() or "ready",
            "ref_rank": ref_rank,
            "section_intent_rescue": True,
            "section_intent": intent,
            "section_intent_block_id": str(block.get("block_id") or "").strip(),
            "section_intent_anchor_id": str(block.get("anchor_id") or "").strip(),
        }
    )
    rescue = dict(template)
    rescue["text"] = text
    rescue["score"] = max(float(template.get("score") or 0.0), ui_score)
    rescue["meta"] = meta
    return rescue


def _maybe_add_section_intent_rescue_hit(prompt: str, hits: list[dict]) -> list[dict]:
    rows = [hit for hit in (hits or []) if isinstance(hit, dict)]
    if not rows or not _refs_prompt_section_intent(prompt):
        return rows
    if any(bool(((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("section_intent_rescue")) for hit in rows):
        return rows
    rescue = _build_section_intent_rescue_hit(prompt, rows)
    if not isinstance(rescue, dict):
        return rows
    rescue_meta = rescue.get("meta") if isinstance(rescue.get("meta"), dict) else {}
    rescue_key = (
        str((rescue_meta or {}).get("source_path") or "").strip(),
        str((rescue_meta or {}).get("section_intent_block_id") or "").strip(),
        str((rescue_meta or {}).get("ref_best_heading_path") or "").strip(),
    )
    for hit in rows:
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        key = (
            str((meta or {}).get("source_path") or "").strip(),
            str((meta or {}).get("section_intent_block_id") or "").strip(),
            str((meta or {}).get("ref_best_heading_path") or (meta or {}).get("heading_path") or "").strip(),
        )
        if key == rescue_key:
            return rows
    return [rescue] + rows


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


def _refs_hit_primary_focus_match_count(prompt: str, hit: dict, *, raw: bool = False) -> int:
    focus_terms = tuple(_refs_prompt_focus_terms(prompt)[:2])
    if not focus_terms:
        return 0
    surface = _refs_raw_hit_surface_text(hit) if raw else _refs_hit_surface_text(hit)
    if not surface:
        return 0
    return sum(1 for term in focus_terms if _focus_term_matches_surface(term, surface))


def _filter_pending_refs_hits_by_prompt_focus(prompt: str, hits: list[dict]) -> list[dict]:
    rows = [hit for hit in (hits or []) if isinstance(hit, dict)]
    bound_hits = _prompt_explicitly_binds_single_source(prompt, rows)
    if bound_hits:
        rows = bound_hits
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
        if len(compare_hits) >= 2 and (not _prompt_explicitly_requests_multi_paper_list(prompt)):
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
    if _prompt_explicitly_requests_multi_paper_list(prompt) and len(focus_terms) > 1:
        primary_matches = [hit for hit in rows if _refs_hit_primary_focus_match_count(prompt, hit, raw=True) > 0]
        if primary_matches:
            return primary_matches
    if len(focus_terms) > 1:
        matched = [hit for hit in rows if _refs_raw_hit_non_source_focus_match_count(prompt, hit) > 0]
        if matched:
            return matched
        return []
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


_SOURCE_ALIAS_STOPWORDS = {
    "acm",
    "aip",
    "cvpr",
    "eccv",
    "iccv",
    "icip",
    "ieee",
    "lpr",
    "mdpi",
    "nips",
    "oe",
    "osa",
    "spie",
}


def _source_explicit_aliases(*values: str) -> set[str]:
    aliases: set[str] = set()
    raw_values = [str(value or "").strip() for value in values if str(value or "").strip()]
    expanded_values: list[str] = []
    for raw in raw_values:
        expanded_values.append(raw)
        name = _source_filename(raw)
        if name:
            expanded_values.append(name)
        _venue, _year, parsed_title = _parse_filename_meta(raw)
        if parsed_title:
            expanded_values.append(parsed_title)
    for raw in expanded_values:
        for token in re.findall(r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9_-]{2,40}(?![A-Za-z0-9_-])", raw):
            cleaned = str(token or "").strip("-_ ")
            if len(cleaned) < 4:
                continue
            norm = _normalize_title_identity(cleaned)
            if len(norm) < 4 or norm in _SOURCE_ALIAS_STOPWORDS:
                continue
            has_alias_shape = bool(
                cleaned.isupper()
                or any(ch.isupper() for ch in cleaned[1:])
                or any(ch.isdigit() for ch in cleaned)
                or ("-" in cleaned)
            )
            if has_alias_shape:
                aliases.add(norm)
    return aliases


def _refs_hit_source_aliases(hit: dict) -> set[str]:
    ui_meta = (hit or {}).get("ui_meta") if isinstance((hit or {}).get("ui_meta"), dict) else {}
    meta = (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}
    citation_meta = (ui_meta or {}).get("citation_meta") if isinstance((ui_meta or {}).get("citation_meta"), dict) else {}
    return _source_explicit_aliases(
        str((meta or {}).get("source_path") or "").strip(),
        str((meta or {}).get("title") or "").strip(),
        str((ui_meta or {}).get("display_name") or "").strip(),
        str((citation_meta or {}).get("title") or "").strip(),
    )


def _prompt_explicitly_binds_single_source(prompt: str, hits: list[dict]) -> list[dict]:
    rows = [hit for hit in list(hits or []) if isinstance(hit, dict)]
    if len(rows) < 2:
        return []
    if (
        _prompt_likely_cross_paper_refs(prompt)
        or _prompt_explicitly_requests_multi_paper_list(prompt)
        or _prompt_requests_compare(prompt)
    ):
        return []
    prompt_norm = _normalize_title_identity(prompt)
    if not prompt_norm:
        return []
    matched: list[dict] = []
    seen_sources: set[str] = set()
    for hit in rows:
        aliases = _refs_hit_source_aliases(hit)
        if not aliases:
            continue
        if not any(alias and alias in prompt_norm for alias in aliases):
            continue
        meta = (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}
        ui_meta = (hit or {}).get("ui_meta") if isinstance((hit or {}).get("ui_meta"), dict) else {}
        source_key = _normalize_title_identity(
            str((meta or {}).get("source_path") or (ui_meta or {}).get("display_name") or "").strip()
        )
        if source_key and source_key in seen_sources:
            continue
        if source_key:
            seen_sources.add(source_key)
        matched.append(hit)
    return matched if len(matched) == 1 else []


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
    if _shared_prompt_requests_reference_compare(prompt):
        return True
    text = str(prompt or "").strip()
    if not text:
        return False
    return bool(re.search(r"(?:比较|对比|权衡|矛盾|不同|区别|差异|相比|相较|取舍)", text))


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
    title_keyword_hits = 0
    has_compare_word = bool(re.search(r"\b(compare|compares|compared|comparison|versus|vs\.?)\b", surface, flags=re.I))
    has_tradeoff_word = bool(re.search(r"\b(trade-?offs?|difference|differences|distinction|sectioning|open pinhole|closed pinhole)\b", surface, flags=re.I))
    if has_compare_word:
        score += 2.0
    elif has_tradeoff_word:
        score += 2.2
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
    if focus_hits >= 2 and (not has_compare_word) and (not has_tradeoff_word) and (not re.search(r"\b(compare|comparison|versus|vs\.?)\b", title_surface, flags=re.I)):
        score -= 1.8
    if len(_refs_prompt_focus_terms(prompt)) > 1 and focus_hits <= 1 and title_keyword_hits < 2:
        score -= 2.4
    return score


def _prompt_requires_explicit_focus_match(prompt: str) -> bool:
    focus_terms = _refs_prompt_focus_terms(prompt)
    if not focus_terms:
        return False
    if _shared_prompt_requires_reference_focus_match(prompt):
        return True
    text = str(prompt or "").strip()
    if not text:
        return False
    return bool(
        _refs_prompt_focus_alias_terms(text)
        and (
            _prompt_requests_compare(text)
            or re.search(r"(?:哪些文献|哪些论文|哪篇|文献|论文|讨论|解释|解决|定义|定位|参考)", text)
        )
    )


def _refs_hit_ui_meta(hit: dict | None) -> dict:
    return (hit or {}).get("ui_meta") if isinstance((hit or {}).get("ui_meta"), dict) else {}


def _refs_hit_meta(hit: dict | None) -> dict:
    return (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}


def _refs_hit_reader_open(hit: dict | None) -> dict:
    ui_meta = _refs_hit_ui_meta(hit)
    reader_open = ui_meta.get("reader_open") if isinstance(ui_meta.get("reader_open"), dict) else {}
    if reader_open:
        return reader_open
    raw = (hit or {}).get("reader_open") if isinstance((hit or {}).get("reader_open"), dict) else {}
    return raw if isinstance(raw, dict) else {}


def _refs_norm_key_text(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(value or "").lower()).strip()


def _refs_hit_source_key(hit: dict | None) -> str:
    ui_meta = _refs_hit_ui_meta(hit)
    meta = _refs_hit_meta(hit)
    source = (
        str(ui_meta.get("source_path") or "").strip()
        or str(meta.get("source_path") or "").strip()
        or str(_refs_hit_reader_open(hit).get("sourcePath") or "").strip()
        or str(ui_meta.get("display_name") or "").strip()
    )
    return _refs_norm_key_text(source.replace("\\", "/"))


def _refs_hit_heading_key(hit: dict | None) -> str:
    ui_meta = _refs_hit_ui_meta(hit)
    meta = _refs_hit_meta(hit)
    reader_open = _refs_hit_reader_open(hit)
    heading = (
        str(ui_meta.get("heading_path") or "").strip()
        or str(ui_meta.get("section_label") or "").strip()
        or str(meta.get("ref_best_heading_path") or "").strip()
        or str(meta.get("heading_path") or "").strip()
        or str(reader_open.get("headingPath") or "").strip()
    )
    return _refs_norm_key_text(heading)


def _refs_hit_locate_key(hit: dict | None) -> str:
    reader_open = _refs_hit_reader_open(hit)
    ui_meta = _refs_hit_ui_meta(hit)
    primary = reader_open.get("primaryEvidence") if isinstance(reader_open.get("primaryEvidence"), dict) else {}
    primary_ui = ui_meta.get("primary_evidence") if isinstance(ui_meta.get("primary_evidence"), dict) else {}
    block_id = str(reader_open.get("blockId") or primary.get("block_id") or primary_ui.get("block_id") or "").strip()
    anchor_id = str(reader_open.get("anchorId") or primary.get("anchor_id") or primary_ui.get("anchor_id") or "").strip()
    anchor_kind = str(reader_open.get("anchorKind") or "").strip().lower()
    anchor_num = str(reader_open.get("anchorNumber") or "").strip()
    if block_id or anchor_id:
        return "loc:" + "|".join([block_id, anchor_id, anchor_kind, anchor_num])
    return ""


def _refs_hit_exact_locate_score(hit: dict | None) -> float:
    ui_meta = _refs_hit_ui_meta(hit)
    reader_open = _refs_hit_reader_open(hit)
    primary = reader_open.get("primaryEvidence") if isinstance(reader_open.get("primaryEvidence"), dict) else {}
    primary_ui = ui_meta.get("primary_evidence") if isinstance(ui_meta.get("primary_evidence"), dict) else {}
    score = 0.0
    if bool(reader_open.get("strictLocate")):
        score += 0.55
    if str(reader_open.get("blockId") or primary.get("block_id") or primary_ui.get("block_id") or "").strip():
        score += 0.30
    if str(reader_open.get("anchorId") or primary.get("anchor_id") or primary_ui.get("anchor_id") or "").strip():
        score += 0.20
    if str(reader_open.get("anchorKind") or "").strip() or _positive_int(reader_open.get("anchorNumber")) > 0:
        score += 0.10
    if bool(primary or primary_ui):
        score += 0.15
    return min(1.30, score)


def _refs_hit_polish_score(hit: dict | None) -> float:
    ui_meta = _refs_hit_ui_meta(hit)
    if not ui_meta:
        return 0.0
    status = str(ref_card_polish_status(ui_meta).get("polish_status") or "").strip().lower()
    if status == "full":
        return 0.60
    if status == "heuristic":
        return 0.20
    if status == "pending":
        return 0.05
    if status == "failed":
        return -0.20
    return 0.0


def _refs_hit_evidence_text(hit: dict | None) -> str:
    ui_meta = _refs_hit_ui_meta(hit)
    meta = _refs_hit_meta(hit)
    reader_open = _refs_hit_reader_open(hit)
    primary = ui_meta.get("primary_evidence") if isinstance(ui_meta.get("primary_evidence"), dict) else {}
    reader_primary = reader_open.get("primaryEvidence") if isinstance(reader_open.get("primaryEvidence"), dict) else {}
    parts: list[str] = []
    for value in (
        primary.get("highlight_snippet"),
        primary.get("snippet"),
        reader_primary.get("highlight_snippet"),
        reader_primary.get("snippet"),
        reader_open.get("highlightSnippet"),
        reader_open.get("snippet"),
        ui_meta.get("summary_line"),
        ui_meta.get("why_line"),
        (hit or {}).get("text"),
    ):
        text = str(value or "").strip()
        if text:
            parts.append(text)
    for key in ("ref_show_snippets", "ref_snippets", "ref_overview_snippets"):
        arr = meta.get(key)
        if isinstance(arr, list):
            for item in arr[:3]:
                text = str(item or "").strip()
                if text:
                    parts.append(text)
    return " ".join(parts)


def _refs_dedupe_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", str(text or "").lower())
    stop = {
        "the", "and", "for", "with", "that", "this", "paper", "section", "method",
        "these", "those", "from", "into", "where", "which", "what", "how",
        "这条", "命中", "证据", "论文", "章节", "方法", "相关", "可以", "用于",
    }
    return {token for token in tokens if token not in stop}


def _refs_evidence_similarity(left: str, right: str) -> float:
    a = _refs_dedupe_tokens(left)
    b = _refs_dedupe_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _refs_evidence_fingerprint(text: str) -> str:
    tokens = sorted(_refs_dedupe_tokens(text))
    if not tokens:
        return ""
    return hashlib.sha1(" ".join(tokens[:32]).encode("utf-8", errors="ignore")).hexdigest()[:16]


def _refs_hits_are_near_duplicates(left: dict, right: dict) -> bool:
    left_source = _refs_hit_source_key(left)
    right_source = _refs_hit_source_key(right)
    if not left_source or left_source != right_source:
        return False
    left_loc = _refs_hit_locate_key(left)
    right_loc = _refs_hit_locate_key(right)
    if left_loc and right_loc and left_loc == right_loc:
        return True
    left_heading = _refs_hit_heading_key(left)
    right_heading = _refs_hit_heading_key(right)
    if left_heading and right_heading and left_heading != right_heading:
        return False
    left_text = _refs_hit_evidence_text(left)
    right_text = _refs_hit_evidence_text(right)
    if left_heading and right_heading and (not left_text or not right_text):
        return True
    left_fp = _refs_evidence_fingerprint(left_text)
    right_fp = _refs_evidence_fingerprint(right_text)
    if left_fp and left_fp == right_fp:
        return True
    return _refs_evidence_similarity(left_text, right_text) >= 0.72


def _refs_hit_duplicate_rank(*, prompt: str, hit: dict, idx: int) -> tuple[float, float, float, float, float, float, int]:
    meta = _refs_hit_meta(hit)
    answer_source_boost = 1.0 if str((meta or {}).get("ref_display_reason") or "").strip().lower() == "answer_hit_top" else 0.0
    return (
        answer_source_boost,
        float(_refs_hit_focus_match_count(prompt, hit)),
        float(_refs_hit_section_intent_score(prompt, hit)),
        _refs_hit_exact_locate_score(hit),
        _refs_hit_polish_score(hit),
        _refs_hit_display_score(hit),
        -int(idx),
    )


def _merge_refs_duplicate_into(keeper: dict, duplicate: dict) -> dict:
    hit = dict(keeper or {})
    ui = dict(_refs_hit_ui_meta(hit))
    ui["merged_duplicate_count"] = _positive_int(ui.get("merged_duplicate_count")) + 1
    headings = [
        str(item or "").strip()
        for item in list(ui.get("merged_duplicate_headings") or [])
        if str(item or "").strip()
    ]
    duplicate_heading = str(_refs_hit_ui_meta(duplicate).get("heading_path") or "").strip()
    if duplicate_heading and duplicate_heading not in headings:
        headings.append(duplicate_heading)
    if headings:
        ui["merged_duplicate_headings"] = headings[:6]
    hit["ui_meta"] = ui
    meta = dict(_refs_hit_meta(hit))
    meta["merged_duplicate_count"] = _positive_int(meta.get("merged_duplicate_count")) + 1
    hit["meta"] = meta
    return hit


def _dedupe_refs_hits_for_display(*, prompt: str, hits: list[dict]) -> tuple[list[dict], int]:
    rows = [dict(hit) for hit in list(hits or []) if isinstance(hit, dict)]
    if len(rows) <= 1:
        return rows, 0
    kept: list[tuple[int, dict]] = []
    removed = 0
    for idx, hit in enumerate(rows):
        match_pos = -1
        for pos, (_keeper_idx, keeper) in enumerate(kept):
            if _refs_hits_are_near_duplicates(keeper, hit):
                match_pos = pos
                break
        if match_pos < 0:
            kept.append((idx, hit))
            continue
        keeper_idx, keeper = kept[match_pos]
        if _refs_hit_duplicate_rank(prompt=prompt, hit=hit, idx=idx) > _refs_hit_duplicate_rank(prompt=prompt, hit=keeper, idx=keeper_idx):
            kept[match_pos] = (idx, _merge_refs_duplicate_into(hit, keeper))
        else:
            kept[match_pos] = (keeper_idx, _merge_refs_duplicate_into(keeper, hit))
        removed += 1
    return [hit for _idx, hit in kept], removed


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


def _refs_hit_matches_expansion_variants(hit: dict) -> bool:
    """Check if a hit's surface text matches significant tokens from its
    query expansion variants.  Used to be lenient with expansion-discovered
    papers that don't match the original prompt's focus terms exactly."""
    expansion_variants = (
        hit.get("_expansion_variants")
        if isinstance(hit.get("_expansion_variants"), list)
        else []
    )
    if not expansion_variants:
        return False
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    surface = " ".join(
        part
        for part in (
            str(hit.get("text") or ""),
            str(meta.get("heading_path") or ""),
            str(meta.get("top_heading") or ""),
        )
        if part
    ).lower()
    if not surface:
        return False
    for variant in expansion_variants:
        tokens = [
            t.lower().strip(".,;:!?()[]\"'")
            for t in variant.split()
            if len(t.strip(".,;:!?()[]\"'")) >= 4
        ]
        for token in tokens:
            if token in surface:
                return True
    return False


def _filter_refs_hits_by_prompt_focus(prompt: str, hits: list[dict]) -> list[dict]:
    rows = [hit for hit in (hits or []) if isinstance(hit, dict)]
    if (
        _prompt_likely_multi_paper_synthesis(prompt)
        and (not _prompt_explicitly_requests_multi_paper_list(prompt))
        and (not _prompt_likely_cross_paper_refs(prompt))
    ):
        return rows
    bound_hits = _prompt_explicitly_binds_single_source(prompt, rows)
    if bound_hits:
        rows = bound_hits
    if not _prompt_requires_explicit_focus_match(prompt):
        return rows
    rows = [hit for hit in rows if not _should_suppress_negative_ref_hit(prompt, hit)]
    rows = [hit for hit in rows if not _refs_hit_focus_terms_only_negated(prompt, hit)]
    force_kept = [hit for hit in rows if _should_force_keep_ref_hit(hit)]

    def _with_force_kept(selected: list[dict]) -> list[dict]:
        out: list[dict] = []
        seen: set[int] = set()
        for hit in list(force_kept or []) + list(selected or []):
            hit_id = id(hit)
            if hit_id in seen:
                continue
            seen.add(hit_id)
            out.append(hit)
        return out

    focus_terms = _refs_prompt_focus_terms(prompt)
    section_matches = [hit for hit in rows if _refs_hit_matches_section_intent(prompt, hit)]
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
        if len(compare_hits) >= 2 and (not _prompt_explicitly_requests_multi_paper_list(prompt)):
            top_score = float(scored_compare_hits[0][0])
            second_score = float(scored_compare_hits[1][0])
            if top_score >= (second_score + 1.0):
                return _with_force_kept([compare_hits[0]])
        return _with_force_kept(compare_hits)
    if _prompt_requests_single_paper_pick(prompt):
        scored_direct_hits = sorted(
            (
                (_refs_single_paper_pick_hit_score(prompt, hit), hit)
                for hit in rows
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return _with_force_kept([hit for score, hit in scored_direct_hits if score >= _MIN_SINGLE_PAPER_DIRECT_HIT_SCORE])
    matched_non_source = [hit for hit in rows if _refs_hit_non_source_focus_match_count(prompt, hit) > 0]
    if _prompt_explicitly_requests_multi_paper_list(prompt):
        if len(focus_terms) > 1:
            primary_matches = [hit for hit in rows if _refs_hit_primary_focus_match_count(prompt, hit) > 0]
            if primary_matches:
                return primary_matches
        if matched_non_source:
            return _with_force_kept(matched_non_source)
        matched = [hit for hit in rows if _refs_hit_focus_match_count(prompt, hit) > 0]
        return _with_force_kept(matched if matched else rows)
    if len(focus_terms) > 1:
        if matched_non_source:
            if section_matches:
                section_ids = {id(hit) for hit in section_matches}
                return _with_force_kept(section_matches + [hit for hit in matched_non_source if id(hit) not in section_ids])
            return _with_force_kept(matched_non_source)
        if section_matches:
            return _with_force_kept(section_matches)
        return _with_force_kept([])
    matched = [hit for hit in rows if _refs_hit_focus_match_count(prompt, hit) > 0]
    if not matched:
        matched = [hit for hit in rows if _refs_hit_matches_expansion_variants(hit)]
    if not matched and section_matches:
        matched = section_matches
    return _with_force_kept(matched if matched else [])


def _sort_refs_hits_for_display(*, prompt: str, hits: list[dict]) -> list[dict]:
    decorated: list[tuple] = []
    prefer_raw_order = not (
        _prompt_requires_explicit_focus_match(prompt)
        or _prompt_likely_cross_paper_refs(prompt)
        or (
            _prompt_likely_multi_paper_synthesis(prompt)
            and (not _prompt_explicitly_requests_multi_paper_list(prompt))
        )
    )
    for idx, hit in enumerate(hits or []):
        if not isinstance(hit, dict):
            continue
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        score = _refs_hit_display_score(hit)
        raw_score = _refs_hit_raw_retrieval_score(hit)
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        answer_source_boost = 1.0 if str((meta or {}).get("ref_display_reason") or "").strip().lower() == "answer_hit_top" else 0.0
        anchor_score = _non_negative_float((ui_meta or {}).get("anchor_match_score"))
        doc_score = _non_negative_float((ui_meta or {}).get("explicit_doc_match_score"))
        focus_count = float(_refs_hit_focus_match_count(prompt, hit))
        prompt_source_boost = float(_refs_prompt_source_match_boost(prompt, hit))
        section_score = float(_refs_hit_section_intent_score(prompt, hit))
        locate_score = _refs_hit_exact_locate_score(hit)
        polish_score = _refs_hit_polish_score(hit)
        if prefer_raw_order:
            decorated.append((answer_source_boost, section_score, raw_score, focus_count, prompt_source_boost, locate_score, polish_score, score, anchor_score, doc_score, -idx, hit))
        else:
            decorated.append((answer_source_boost, focus_count, section_score, prompt_source_boost, locate_score, polish_score, score, anchor_score, doc_score, raw_score, -idx, hit))
    decorated.sort(key=lambda item: item[:-1], reverse=True)
    return [item[-1] for item in decorated]


def _prompt_explicitly_requests_multi_paper_list(prompt: str) -> bool:
    return _shared_prompt_explicitly_requests_multi_paper_list(prompt)


def _prompt_likely_multi_paper_synthesis(prompt: str) -> bool:
    return _shared_prompt_likely_multi_paper_synthesis(prompt)


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
    if _refs_has_decisive_raw_retrieval_leader(prompt, rows):
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
    raw_flag = str(os.environ.get("KB_REFS_RELEVANCE_USE_LLM", "0") or "").strip().lower()
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
    if (
        _prompt_likely_multi_paper_synthesis(prompt)
        and (not _prompt_explicitly_requests_multi_paper_list(prompt))
        and (not _prompt_likely_cross_paper_refs(prompt))
        and len(rows) > 1
    ):
        return False
    if not (_prompt_requires_explicit_focus_match(prompt) or _prompt_likely_cross_paper_refs(prompt)):
        return False
    if _prompt_explicitly_requests_multi_paper_list(prompt) and len(rows) > 1:
        return False
    if guide_mode and _prompt_likely_cross_paper_refs(prompt):
        return True
    if len(rows) == 1:
        return False
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
    section_matches = [hit for hit in rows if _refs_hit_matches_section_intent(prompt, hit)]

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
        if section_matches:
            return section_matches
        return []

    kept_rows: list[dict] = []
    for idx1 in keep:
        zero = int(idx1) - 1
        if zero < 0 or zero >= len(pool):
            continue
        kept_rows.append(pool[zero])
    if section_matches:
        kept_ids = {id(hit) for hit in kept_rows}
        for hit in section_matches:
            if id(hit) not in kept_ids:
                kept_rows.append(hit)
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
        return "bounded_full", False, bool(allow_expensive_llm_for_ready), True
    return (
        "interactive_full",
        bool(allow_citation_prefetch_for_pending),
        bool(allow_expensive_llm_for_ready),
        bool(allow_exact_locate),
    )


def _refs_payload_deadline_near(deadline_at: float | None, min_remaining_s: float = 0.0) -> bool:
    if deadline_at is None:
        return False
    try:
        return (float(deadline_at) - time.perf_counter()) <= max(0.0, float(min_remaining_s))
    except Exception:
        return False


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
    deadline_at: float | None = None,
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
        t_enrich_start = time.time()
        prompt = str(pack.get("prompt") or "").strip()
        prompt_requires_focus_match = bool(_prompt_requires_explicit_focus_match(prompt))
        prompt_cross_paper_refs = bool(_prompt_likely_cross_paper_refs(prompt))
        prompt_multi_paper_list = bool(_prompt_explicitly_requests_multi_paper_list(prompt))
        prompt_ordinary_multi_source_synthesis = bool(
            _prompt_likely_multi_paper_synthesis(prompt)
            and (not prompt_multi_paper_list)
            and (not prompt_cross_paper_refs)
        )
        prompt_multi_source_synthesis = bool(
            prompt_multi_paper_list or prompt_ordinary_multi_source_synthesis
        )
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
            if guide_active and prompt_cross_paper_refs and _hit_matches_guide_source(
                meta,
                guide_source_path=guide_source_path_norm,
                guide_source_name=guide_source_name_norm,
            ):
                filtered_self_hits += 1
                continue
            raw_hits.append(dict(hit))
        raw_hits = _maybe_add_section_intent_rescue_hit(prompt, raw_hits)
        scored_ready = []
        for hit in raw_hits:
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
            if prompt_multi_source_synthesis:
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
        t_score_gate = time.time()
        post_focus_filter_hit_count = int(post_score_gate_hit_count)
        post_llm_filter_hit_count = int(post_focus_filter_hit_count)
        slow_allowed = bool(
            allow_expensive_llm_for_ready
            and (not _refs_payload_deadline_near(deadline_at, 0.35))
        )
        if has_pending and hits:
            hits = _filter_pending_refs_hits_by_prompt_focus(prompt, hits)
            post_focus_filter_hit_count = int(len(hits))
            hits = _sort_refs_hits_for_display(prompt=prompt, hits=hits)
            hits = hits[:4]
            post_llm_filter_hit_count = int(len(hits))
        allow_citation_prefetch = bool(
            hits
            and slow_allowed
            and (_render_variant != "fast")
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
        # LLM translation in build_hit_ui_meta is redundant when polish follows
        # (line 8793) — LLM polish regenerates text in the correct locale.
        allow_hit_llm_refine = False
        exact_locate_allowed = bool(
            (not has_pending)
            and allow_exact_locate
            and (not _refs_payload_deadline_near(deadline_at, 0.20))
        )
        ui_workers = _refs_card_polish_max_workers(len(hits))
        if ui_workers <= 1:
            for hit2 in hits:
                hit2["ui_meta"] = build_hit_ui_meta(
                    hit2,
                    prompt=prompt,
                    pdf_root=pdf_root,
                    lib_store=lib_store,
                    preloaded_citation_meta=preloaded_citation_meta,
                    allow_expensive_llm=allow_hit_llm_refine,
                    allow_exact_locate=exact_locate_allowed,
                )
        else:

            def _build_ui_meta(hit2):
                return hit2, build_hit_ui_meta(
                    hit2,
                    prompt=prompt,
                    pdf_root=pdf_root,
                    lib_store=lib_store,
                    preloaded_citation_meta=preloaded_citation_meta,
                    allow_expensive_llm=allow_hit_llm_refine,
                    allow_exact_locate=exact_locate_allowed,
                )

            with ThreadPoolExecutor(max_workers=ui_workers) as ex:
                futs = [ex.submit(_build_ui_meta, h) for h in hits]
                for fu in as_completed(futs):
                    try:
                        hit2, ui = fu.result()
                        hit2["ui_meta"] = ui
                    except Exception:
                        continue
        if hits and (not has_pending) and (not prompt_ordinary_multi_source_synthesis):
            hits = _filter_refs_hits_by_prompt_focus(prompt, hits)
        post_focus_filter_hit_count = int(len(hits))
        # Clean up internal field used by focus filter expansion check.
        for _h in hits:
            _h.pop("_expansion_variants", None)
        t_focus_filter = time.time()
        if len(hits) > 1:
            hits = _sort_refs_hits_for_display(prompt=prompt, hits=hits)
            slow_allowed = bool(
                allow_expensive_llm_for_ready
                and (not _refs_payload_deadline_near(deadline_at, 0.35))
            )
            if (not has_pending) and slow_allowed and (not prompt_multi_source_synthesis):
                hits = _maybe_llm_rerank_refs_hits(
                    prompt=prompt,
                    hits=hits,
                    guide_mode=guide_active,
                )
        slow_allowed = bool(
            allow_expensive_llm_for_ready
            and (not _refs_payload_deadline_near(deadline_at, 0.35))
        )
        if hits and len(hits) > 1 and (not has_pending) and slow_allowed and (not prompt_multi_source_synthesis):
            hits = _maybe_llm_filter_refs_hits(
                prompt=prompt,
                hits=hits,
                guide_mode=guide_active,
            )
        post_llm_filter_hit_count = int(len(hits))
        deduped_duplicate_hit_count = 0
        if hits and len(hits) > 1 and (not has_pending):
            hits, deduped_duplicate_hit_count = _dedupe_refs_hits_for_display(prompt=prompt, hits=hits)
            if len(hits) > 1:
                hits = _sort_refs_hits_for_display(prompt=prompt, hits=hits)
        display_cap = 0
        if prompt_multi_paper_list:
            display_cap = 6
        elif prompt_multi_source_synthesis:
            display_cap = 4
        if display_cap > 0 and len(hits) > display_cap:
            hits = hits[:display_cap]
        t_final = time.time()
        slow_allowed = bool(
            allow_expensive_llm_for_ready
            and (not _refs_payload_deadline_near(deadline_at, 0.35))
        )
        llm_polish_allowed = bool(hits and (not has_pending) and slow_allowed)
        t_polish_start = time.time()
        if llm_polish_allowed:
            hits = _maybe_polish_refs_card_copy(
                prompt=prompt,
                hits=hits,
                guide_mode=guide_active,
            )
            if len(hits) > 1:
                hits = _sort_refs_hits_for_display(prompt=prompt, hits=hits)
        t_done = time.time()
        pack2 = dict(pack)
        pack2["hits"] = hits
        pack2["pipeline_debug"] = {
            "guide_active": bool(guide_active),
            "has_pending": bool(has_pending),
            "raw_hit_count": int(len(raw_hits)),
            "post_score_gate_hit_count": int(post_score_gate_hit_count),
            "post_focus_filter_hit_count": int(post_focus_filter_hit_count),
            "post_llm_filter_hit_count": int(post_llm_filter_hit_count),
            "deduped_duplicate_hit_count": int(deduped_duplicate_hit_count),
            "final_hit_count": int(len(hits)),
            "display_cap": int(display_cap),
            "filtered_self_hit_count": int(filtered_self_hits),
            "prompt_requires_explicit_focus_match": bool(prompt_requires_focus_match),
            "prompt_likely_cross_paper_refs": bool(prompt_cross_paper_refs),
            "prompt_explicitly_requests_multi_paper_list": bool(prompt_multi_paper_list),
            "prompt_likely_multi_paper_synthesis": bool(prompt_multi_source_synthesis),
            "prompt_ordinary_multi_source_synthesis": bool(prompt_ordinary_multi_source_synthesis),
            "enrich_elapsed_total_s": round(max(0.0, t_done - t_enrich_start), 3),
            "enrich_elapsed_score_gate_s": round(max(0.0, t_score_gate - t_enrich_start), 3),
            "enrich_elapsed_focus_filter_s": round(max(0.0, t_focus_filter - t_enrich_start), 3),
            "enrich_elapsed_llm_filter_s": round(max(0.0, t_final - t_enrich_start), 3),
            "enrich_elapsed_llm_polish_s": round(max(0.0, t_done - t_polish_start), 3),
            "render_variant": str(_render_variant or ""),
            "llm_polish_allowed": bool(llm_polish_allowed),
            "llm_polish_enabled": bool(_refs_card_polish_llm_enabled()),
            "llm_polish_top_n": int(_refs_card_polish_top_n()),
            "llm_polish_timeout_s": float(_refs_card_polish_timeout_s()),
            "llm_polish_max_retries": int(_refs_card_polish_max_retries()),
            "deadline_exhausted": bool(_refs_payload_deadline_near(deadline_at, 0.0)),
            "query_variants": list(pack.get("query_variants") or []),
        }
        if guide_active:
            hidden_self_source = bool(prompt_cross_paper_refs and (filtered_self_hits > 0 or not hits))
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


def _valid_external_abstract_candidate(text: str, *, title: str = "") -> str:
    abstract = _summary_excerpt(text, max_sentences=5, max_len=900)
    if not abstract:
        return ""
    low = abstract.lower()
    if any(
        token in low
        for token in (
            "access through your institution",
            "sign in to access",
            "javascript",
            "cookie",
            "all rights reserved",
            "subscribe to this journal",
            "article navigation",
        )
    ):
        return ""
    if title and _looks_like_title_echo(abstract, title):
        return ""
    if len(re.findall(r"[A-Za-z\u4e00-\u9fff]{3,}", abstract)) < 12:
        return ""
    return _summary_excerpt(abstract, max_sentences=3, max_len=520)


@lru_cache(maxsize=512)
def _semantic_scholar_paper_by_doi(doi: str) -> dict:
    d = _normalize_doi_like(doi)
    if not d or d.startswith("10.48550/arxiv"):
        return {}
    try:
        resp = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(d, safe='')}",
            params={"fields": "title,abstract,year,venue,authors,externalIds,url"},
            headers={"User-Agent": "Pi-zaya-KB/1.0 (Research Assistant)"},
            timeout=4.5,
        )
    except Exception:
        return {}
    if resp.status_code != 200:
        return {}
    try:
        data = resp.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _summary_from_semantic_scholar_abstract(meta: dict) -> str:
    doi_like = str((meta or {}).get("doi") or (meta or {}).get("doi_url") or "").strip()
    doi = _normalize_doi_like(doi_like)
    if not doi:
        return ""
    work = _semantic_scholar_paper_by_doi(doi)
    if not isinstance(work, dict):
        return ""
    external = work.get("externalIds") if isinstance(work.get("externalIds"), dict) else {}
    found_doi = _normalize_doi_like(str((external or {}).get("DOI") or ""))
    if found_doi and found_doi != doi:
        return ""
    title = str((meta or {}).get("title") or "").strip()
    found_title = str(work.get("title") or "").strip()
    if title and found_title and _title_similarity_for_openalex(title, found_title) < 0.86:
        return ""
    return _valid_external_abstract_candidate(str(work.get("abstract") or ""), title=title or found_title)


def _html_meta_content(page: str, names: tuple[str, ...]) -> str:
    html_text = str(page or "")
    if not html_text:
        return ""
    name_set = {name.lower() for name in names}
    for match in re.finditer(r"<meta\b[^>]*>", html_text, flags=re.I):
        tag = match.group(0)
        key_match = re.search(r"\b(?:name|property)\s*=\s*(['\"])(.*?)\1", tag, flags=re.I | re.S)
        if not key_match:
            continue
        key = html.unescape(str(key_match.group(2) or "").strip().lower())
        if key not in name_set:
            continue
        content_match = re.search(r"\bcontent\s*=\s*(['\"])(.*?)\1", tag, flags=re.I | re.S)
        if not content_match:
            continue
        value = html.unescape(str(content_match.group(2) or "")).strip()
        if value:
            return value
    return ""


def _jsonld_description_from_html(page: str) -> str:
    html_text = str(page or "")
    if not html_text:
        return ""
    for match in re.finditer(
        r"<script\b[^>]*type\s*=\s*(['\"])application/ld\+json\1[^>]*>(.*?)</script>",
        html_text,
        flags=re.I | re.S,
    ):
        raw = html.unescape(str(match.group(2) or "")).strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        queue = data if isinstance(data, list) else [data]
        for item in queue:
            if not isinstance(item, dict):
                continue
            for key in ("abstract", "description"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return ""


@lru_cache(maxsize=256)
def _doi_landing_page_abstract(doi: str) -> str:
    d = _normalize_doi_like(doi)
    if not d or d.startswith("10.48550/arxiv"):
        return ""
    try:
        resp = requests.get(
            f"https://doi.org/{quote(d, safe='/')}",
            headers={
                "User-Agent": "Pi-zaya-KB/1.0 (Research Assistant)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            timeout=6.0,
            allow_redirects=True,
        )
    except Exception:
        return ""
    if resp.status_code >= 400:
        return ""
    content_type = str(resp.headers.get("content-type") or "").lower()
    if "html" not in content_type and "xml" not in content_type and "text" not in content_type:
        return ""
    text = str(resp.text or "")
    if not text:
        return ""
    text = text[:500_000]
    return (
        _html_meta_content(
            text,
            (
                "citation_abstract",
                "dc.description",
                "dcterms.description",
                "description",
                "og:description",
                "twitter:description",
            ),
        )
        or _jsonld_description_from_html(text)
    )


def _summary_from_doi_landing_page(meta: dict) -> str:
    doi_like = str((meta or {}).get("doi") or (meta or {}).get("doi_url") or "").strip()
    doi = _normalize_doi_like(doi_like)
    if not doi:
        return ""
    title = str((meta or {}).get("title") or "").strip()
    return _valid_external_abstract_candidate(_doi_landing_page_abstract(doi), title=title)


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
    if (s_norm in t_norm) and (len(s_norm) >= max(24, int(0.68 * len(t_norm)))):
        return True
    s_tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", s)
    t_tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", t)
    if (len(t_tokens) >= 4) and s_tokens:
        common = len(set(s_tokens) & set(t_tokens))
        if common >= max(3, int(0.85 * len(set(t_tokens)))) and len(set(s_tokens)) <= len(set(t_tokens)) + 3:
            return True
        # Second layer: when token overlap is moderate (\u226550%), check sequence similarity
        # to catch paraphrases that reuse title wording without copying it verbatim.
        if common >= max(2, int(0.50 * len(set(t_tokens)))):
            s_seq = " ".join(s_tokens)
            t_seq = " ".join(t_tokens)
            ratio = difflib.SequenceMatcher(None, s_seq, t_seq).ratio()
            if ratio >= 0.72:
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
    if not s:
        return False
    if _looks_fragmentary_ref_summary(s):
        return False
    if _looks_why_like_ref_summary(s):
        return False
    if len(s) < 50:
        return False
    if not re.search(
        r"(提出|设计|构建|采用|引入|实现|比较|分析|评估|develop|propose|introduce|present|compare|analy[sz]e|evaluat)",
        s,
        flags=re.I,
    ):
        return False
    if not re.search(
        r"(结果|显示|提升|降低|差异|优劣|加速|优于|有效|性能|实验|result|show|improv|outperform|achiev|difference|trade-?off|advantage|limitation)",
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
    raw_flag = str(os.environ.get("KB_CITE_SUMMARY_USE_LLM", "0") or "").strip().lower()
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
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 20.0),
            max_retries=1,
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
            out["summary_provider"] = "crossref"
            return out
        openalex_line = _summary_from_openalex_abstract(out)
        if openalex_line:
            final_line, generation = _finalize_abstract_summary_line(title=title, abstract_text=openalex_line)
            out["summary_line"] = final_line or _translate_summary_to_zh(openalex_line)
            out["summary_source"] = "abstract"
            out["summary_generation"] = generation or "translated_abstract"
            out["summary_provider"] = "openalex"
            return out
        semantic_line = _summary_from_semantic_scholar_abstract(out)
        if semantic_line:
            final_line, generation = _finalize_abstract_summary_line(title=title, abstract_text=semantic_line)
            out["summary_line"] = final_line or _translate_summary_to_zh(semantic_line)
            out["summary_source"] = "abstract"
            out["summary_generation"] = generation or "translated_abstract"
            out["summary_provider"] = "semantic_scholar"
            return out
        landing_line = _summary_from_doi_landing_page(out)
        if landing_line:
            final_line, generation = _finalize_abstract_summary_line(title=title, abstract_text=landing_line)
            out["summary_line"] = final_line or _translate_summary_to_zh(landing_line)
            out["summary_source"] = "abstract"
            out["summary_generation"] = generation or "translated_abstract"
            out["summary_provider"] = "doi_landing_page"
            return out

    context_fallback = _contextual_summary_line(out)
    if context_fallback:
        out["summary_line"] = context_fallback
        out["summary_source"] = "citation_context"
        out["summary_generation"] = "citation_context_fallback"
        return out

    fallback = _metadata_summary_line(out)
    if fallback:
        out["summary_line"] = fallback
        out["summary_source"] = "metadata"
        out["summary_generation"] = "metadata_only"
    return out


_EXTERNAL_IDENTITY_KEYS = {
    "title",
    "authors",
    "venue",
    "year",
    "volume",
    "issue",
    "pages",
}

_EXTERNAL_DOI_AND_METRIC_KEYS = {
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


def _safe_float_meta(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _external_meta_seed_title(meta: dict) -> str:
    title = str((meta or {}).get("title") or "").strip()
    if title and not _is_weak_meta_value("title", title):
        return title
    for key in ("cite_fmt", "raw"):
        text = _clean_summary_line(str((meta or {}).get(key) or ""))
        if text and not _is_weak_meta_value("title", text):
            return text[:240]
    return ""


def _external_meta_similarity(base: dict, incoming: dict) -> float:
    explicit = _safe_float_meta((incoming or {}).get("title_similarity"), -1.0)
    if explicit >= 0.0:
        return max(0.0, min(1.0, explicit))
    seed = _external_meta_seed_title(base)
    candidate = str((incoming or {}).get("title") or "").strip()
    if seed and candidate:
        try:
            return max(0.0, min(1.0, float(title_similarity(seed, candidate))))
        except Exception:
            return 0.0
    return 1.0 if (not seed or not candidate) else 0.0


def _store_candidate_external_metadata(out: dict, incoming: dict, *, status: str, reason: str, similarity: float) -> None:
    out["external_metadata_status"] = status
    out["external_metadata_reason"] = reason
    match_method = str((incoming or {}).get("match_method") or "").strip()
    if match_method:
        out["external_match_method"] = match_method
    match_score = (incoming or {}).get("match_score")
    if match_score not in (None, ""):
        out["external_match_score"] = match_score
    if similarity >= 0.0:
        out["external_title_similarity"] = round(max(0.0, min(1.0, similarity)), 4)
    for key in _EXTERNAL_IDENTITY_KEYS | {"doi", "doi_url"}:
        value = (incoming or {}).get(key)
        if value in (None, "", [], {}):
            continue
        out[f"external_{key}"] = value


def _external_meta_merge_mode(base: dict, incoming: dict) -> tuple[str, str, float]:
    base_doi = _normalize_doi_like(str((base or {}).get("doi") or (base or {}).get("doi_url") or ""))
    incoming_doi = _normalize_doi_like(str((incoming or {}).get("doi") or (incoming or {}).get("doi_url") or ""))
    if base_doi and incoming_doi and (base_doi != incoming_doi):
        return "conflict", "外部元数据 DOI 与当前引用已有 DOI 不一致，已保留当前引用信息。", 0.0

    method = str((incoming or {}).get("match_method") or "").strip().lower()
    similarity = _external_meta_similarity(base, incoming)
    seed_title = _external_meta_seed_title(base)
    incoming_title = str((incoming or {}).get("title") or "").strip()
    if seed_title and incoming_title:
        if method in {"bibliographic", "doi", "title", "openalex_title_arxiv"} and similarity < 0.72:
            return (
                "candidate",
                "外部元数据标题与原参考条目相似度较低，已优先保留原参考条目；DOI、被引和期刊指标仅作核对线索。",
                similarity,
            )
        if method == "bibliographic" and similarity < 0.80:
            return (
                "candidate",
                "外部元数据由参考条目模糊匹配得到，标题相似度不够高，已作为候选线索处理。",
                similarity,
            )
    return "trusted", "", similarity


def _contextual_summary_line(meta: dict) -> str:
    context = _summary_excerpt(
        str(
            (meta or {}).get("citation_context")
            or (meta or {}).get("card_evidence")
            or (meta or {}).get("evidence_quote")
            or ""
        ),
        max_sentences=2,
        max_len=280,
    )
    if not context:
        return ""
    claim = _summary_excerpt(
        str((meta or {}).get("answer_claim") or (meta or {}).get("card_claim") or ""),
        max_sentences=1,
        max_len=160,
    )
    location = _clean_summary_line(
        str((meta or {}).get("location_label") or (meta or {}).get("card_locator") or (meta or {}).get("heading_path") or "")
    )
    parts: list[str] = []
    if claim:
        parts.append(f"暂无可用摘要；当前回答主要借它支撑：{claim}")
    else:
        parts.append("暂无可用摘要；可先根据当前论文里的引用语境判断它在回答中的作用。")
    if location:
        parts.append(f"引用位置：{location}。")
    parts.append(f"引用语境：{context}")
    return _summary_excerpt(" ".join(parts), max_sentences=3, max_len=420)


def _merge_meta_prefer_richer(base: dict, incoming: dict) -> dict:
    out = dict(base or {})
    base_doi = _normalize_doi_like(str(out.get("doi") or out.get("doi_url") or ""))
    incoming_doi = _normalize_doi_like(str((incoming or {}).get("doi") or (incoming or {}).get("doi_url") or ""))
    doi_conflict = bool(base_doi and incoming_doi and (base_doi != incoming_doi))
    merge_mode, merge_reason, merge_similarity = _external_meta_merge_mode(out, incoming or {})
    if merge_mode in {"candidate", "conflict"}:
        _store_candidate_external_metadata(
            out,
            incoming or {},
            status=merge_mode,
            reason=merge_reason,
            similarity=merge_similarity,
        )
    elif incoming:
        out.setdefault("external_metadata_status", "trusted")
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
        if merge_mode in {"candidate", "conflict"} and key in _EXTERNAL_IDENTITY_KEYS:
            # A fuzzy external hit may still provide DOI/metrics as a clue, but
            # it must not rewrite the actual cited work identity.
            continue
        if merge_mode == "conflict" and key in _EXTERNAL_DOI_AND_METRIC_KEYS:
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
        if merge_mode == "trusted" and key in _EXTERNAL_IDENTITY_KEYS:
            same_or_new_doi = bool(incoming_doi and ((not base_doi) or incoming_doi == base_doi))
            if same_or_new_doi:
                out[key] = new
                continue
            if key == "title" and _external_meta_similarity(out, incoming or {}) >= 0.94:
                out[key] = new
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
                meta = _merge_meta_prefer_richer(meta, canonical)
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
