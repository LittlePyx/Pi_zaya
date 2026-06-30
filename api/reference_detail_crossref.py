from __future__ import annotations

import re
from typing import Callable


def merge_canonical_for_existing_doi(
    meta: dict,
    *,
    title: str,
    venue: str,
    year: str,
    doi: str,
    fetch_best_crossref_meta: Callable[..., dict | None],
    is_weak_meta_value: Callable[[str, str], bool],
    normalize_doi_like: Callable[[str], str],
    merge_meta_prefer_richer: Callable[[dict, dict], dict],
    build_doi_url: Callable[[str], str],
) -> dict:
    out = dict(meta or {})
    doi_text = str(doi or "").strip()
    if not doi_text:
        return out
    if not str(out.get("doi_url") or "").strip():
        out["doi_url"] = build_doi_url(doi_text)
    try:
        canonical = fetch_best_crossref_meta(
            query_title="" if is_weak_meta_value("title", title) else title,
            doi_hint=doi_text,
            expected_year=year,
            expected_venue=venue,
            min_score=0.90,
            allow_title_only=False,
        )
    except Exception:
        canonical = None
    if isinstance(canonical, dict):
        meta_doi = normalize_doi_like(str(out.get("doi") or out.get("doi_url") or doi_text))
        canonical_doi = normalize_doi_like(str(canonical.get("doi") or canonical.get("doi_url") or ""))
        if meta_doi and canonical_doi and (meta_doi == canonical_doi):
            out = merge_meta_prefer_richer(out, canonical)
        else:
            out = merge_meta_prefer_richer(out, canonical)
        current_doi = str(out.get("doi") or "").strip()
        if current_doi and not str(out.get("doi_url") or "").strip():
            out["doi_url"] = build_doi_url(current_doi)
    return out


def merge_reference_text_crossref(
    meta: dict,
    *,
    raw: str,
    title: str,
    venue: str,
    year: str,
    fetch_best_crossref_for_reference: Callable[..., dict | None],
    fetch_best_crossref_meta: Callable[..., dict | None],
    is_weak_meta_value: Callable[[str, str], bool],
    merge_meta_prefer_richer: Callable[[dict, dict], dict],
    build_doi_url: Callable[[str], str],
) -> dict:
    out = dict(meta or {})
    fetched_ref = None
    raw_text = str(raw or "").strip()
    if raw_text:
        try:
            fetched_ref = fetch_best_crossref_for_reference(reference_text=raw_text, min_score=0.74)
        except Exception:
            fetched_ref = None
    if isinstance(fetched_ref, dict):
        out = merge_meta_prefer_richer(
            out,
            {k: v for k, v in fetched_ref.items() if v not in (None, "", [], {})},
        )
        doi = str(out.get("doi") or "").strip()
        if doi and not str(out.get("doi_url") or "").strip():
            out["doi_url"] = build_doi_url(doi)

    doi = str(out.get("doi") or "").strip()
    if not doi:
        return out
    try:
        canonical = fetch_best_crossref_meta(
            query_title="" if is_weak_meta_value("title", str(out.get("title") or title).strip()) else str(out.get("title") or title).strip(),
            doi_hint=doi,
            expected_year=str(out.get("year") or year).strip(),
            expected_venue=str(out.get("venue") or venue).strip(),
            min_score=0.90,
            allow_title_only=False,
        )
    except Exception:
        canonical = None
    if isinstance(canonical, dict):
        out = merge_meta_prefer_richer(out, canonical)
        current_doi = str(out.get("doi") or "").strip()
        if current_doi and not str(out.get("doi_url") or "").strip():
            out["doi_url"] = build_doi_url(current_doi)
    return out


def merge_title_crossref(
    meta: dict,
    *,
    title: str,
    raw: str,
    venue: str,
    year: str,
    fetch_crossref_meta: Callable[..., dict | None],
    fetch_best_crossref_meta: Callable[..., dict | None],
    is_weak_meta_value: Callable[[str, str], bool],
    merge_meta_prefer_richer: Callable[[dict, dict], dict],
    build_doi_url: Callable[[str], str],
) -> dict:
    out = dict(meta or {})
    search_title = str(title or "").strip()
    if not search_title:
        raw2 = re.sub(r"^\s*(?:\[\s*\d+\s*\]\s*)+", "", str(raw or "")).strip()
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
        and (not is_weak_meta_value("title", search_title))
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
        out = merge_meta_prefer_richer(
            out,
            {k: v for k, v in fetched.items() if v not in (None, "", [], {})},
        )
        doi = str(out.get("doi") or "").strip()
        if doi and not str(out.get("doi_url") or "").strip():
            out["doi_url"] = build_doi_url(doi)
    return out
