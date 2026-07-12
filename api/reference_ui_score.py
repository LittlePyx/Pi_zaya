from __future__ import annotations

import hashlib
import math

_MIN_REF_UI_SCORE = 5.2
_MAX_REF_UI_GAP = 1.8
_MIN_SINGLE_PAPER_DIRECT_HIT_SCORE = 4.25
_MIN_PENDING_SINGLE_PAPER_DIRECT_HIT_SCORE = 3.0
_MIN_COMPARE_DIRECT_HIT_SCORE = 5.0


def _clamp_ui_score(score: float) -> float:
    try:
        v = float(score)
    except Exception:
        v = 0.0
    return max(0.0, min(10.0, v))


def _stable_score_micro_jitter(source_path: str) -> float:
    """Small deterministic jitter to avoid repeated identical decimals."""
    s = str(source_path or "").strip()
    if not s:
        return 0.0
    try:
        h = hashlib.sha1(s.encode("utf-8", "ignore")).digest()
        u = int.from_bytes(h[:2], "big") / 65535.0
    except Exception:
        return 0.0
    return (u - 0.5) * 0.08


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

    evidence_ui = 5.0
    evidence_ui += 1.8 * math.tanh((bm25 - 2.5) / 3.0)
    evidence_ui += 1.2 * math.tanh((deep - 1.5) / 4.0)
    evidence_ui += 0.9 * math.tanh(term_bonus / 1.8)
    if semantic_score > 0:
        evidence_ui = (0.82 * evidence_ui) + (0.18 * _clamp_ui_score(semantic_score))
    evidence_ui = _clamp_ui_score(evidence_ui)

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

    try:
        bm25_spread = max(-1.0, min(1.0, math.tanh((bm25 - 3.0) / 4.0)))
    except Exception:
        bm25_spread = 0.0
    try:
        deep_spread = max(-1.0, min(1.0, math.tanh((deep - 2.0) / 6.0)))
    except Exception:
        deep_spread = 0.0
    ui += (0.14 * bm25_spread) + (0.12 * deep_spread)

    ui += _stable_score_micro_jitter(str(meta.get("source_path") or ""))

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
    if bool(meta.get("paper_guide_fast_exact")):
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
