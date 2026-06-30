from __future__ import annotations

import math

from api.reference_external_ids import _is_weak_meta_value, _normalize_doi_like
from api.reference_summary_text import _clean_summary_line
from kb.citation_meta import title_similarity


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
            continue
        if merge_mode in {"candidate", "conflict"} and key in _EXTERNAL_IDENTITY_KEYS:
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
