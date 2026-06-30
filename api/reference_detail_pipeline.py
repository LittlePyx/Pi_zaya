from __future__ import annotations

from collections.abc import Callable

from api.reference_detail_arxiv import (
    merge_existing_arxiv_backfill,
    merge_missing_doi_arxiv_fallback,
)
from api.reference_detail_crossref import (
    merge_canonical_for_existing_doi,
    merge_reference_text_crossref,
    merge_title_crossref,
)
from api.reference_detail_finalize import enrich_bibliometrics_and_summary
from api.reference_detail_seed import (
    apply_raw_reference_fallback,
    detail_raw_seed,
    seed_detail_raw_fields,
)


def enrich_citation_detail_meta(
    detail: dict,
    *,
    normalize_reference_for_popup: Callable[[dict], dict],
    normalize_doi_like: Callable[[str], str],
    extract_first_doi: Callable[[str], str],
    build_doi_url: Callable[[str], str],
    arxiv_backfill_meta_from_texts: Callable[..., dict],
    fallback_fill_reference_meta_from_raw: Callable[[dict], dict],
    merge_meta_prefer_richer: Callable[[dict, dict], dict],
    fetch_best_crossref_meta: Callable[..., dict | None],
    fetch_best_crossref_for_reference: Callable[..., dict | None],
    fetch_crossref_meta: Callable[..., dict | None],
    is_weak_meta_value: Callable[[str, str], bool],
    should_try_openalex_arxiv_title: Callable[..., bool],
    openalex_arxiv_meta_by_title: Callable[[str], dict],
    enrich_bibliometrics: Callable[[dict], dict],
    ensure_summary_line: Callable[..., dict],
) -> dict:
    meta = normalize_reference_for_popup(detail or {}) or dict(detail or {})
    raw0 = detail_raw_seed(meta)
    meta = seed_detail_raw_fields(
        meta,
        raw=raw0,
        normalize_doi_like=normalize_doi_like,
        extract_first_doi=extract_first_doi,
        build_doi_url=build_doi_url,
    )
    meta = apply_raw_reference_fallback(
        meta,
        raw=raw0,
        arxiv_backfill_meta_from_texts=arxiv_backfill_meta_from_texts,
        fallback_fill_reference_meta_from_raw=fallback_fill_reference_meta_from_raw,
    )

    meta = merge_existing_arxiv_backfill(
        meta,
        arxiv_backfill_meta_from_texts=arxiv_backfill_meta_from_texts,
        normalize_doi_like=normalize_doi_like,
        merge_meta_prefer_richer=merge_meta_prefer_richer,
    )

    title = str(meta.get("title") or "").strip()
    raw = str(
        meta.get("raw")
        or meta.get("card_reference_entry")
        or meta.get("cardReferenceEntry")
        or meta.get("cite_fmt")
        or meta.get("citeFmt")
        or ""
    ).strip()
    venue = str(meta.get("venue") or "").strip()
    year = str(meta.get("year") or "").strip()
    doi = str(meta.get("doi") or "").strip()
    if doi:
        meta = merge_canonical_for_existing_doi(
            meta,
            title=title,
            venue=venue,
            year=year,
            doi=doi,
            fetch_best_crossref_meta=fetch_best_crossref_meta,
            is_weak_meta_value=is_weak_meta_value,
            normalize_doi_like=normalize_doi_like,
            merge_meta_prefer_richer=merge_meta_prefer_richer,
            build_doi_url=build_doi_url,
        )
    if not doi:
        meta = merge_reference_text_crossref(
            meta,
            raw=raw,
            title=title,
            venue=venue,
            year=year,
            fetch_best_crossref_for_reference=fetch_best_crossref_for_reference,
            fetch_best_crossref_meta=fetch_best_crossref_meta,
            is_weak_meta_value=is_weak_meta_value,
            merge_meta_prefer_richer=merge_meta_prefer_richer,
            build_doi_url=build_doi_url,
        )
        doi = str(meta.get("doi") or "").strip()
        if doi:
            return enrich_bibliometrics_and_summary(
                meta,
                enrich_bibliometrics=enrich_bibliometrics,
                ensure_summary_line=ensure_summary_line,
                allow_crossref_abstract=True,
            )

        meta = merge_title_crossref(
            meta,
            title=title,
            raw=raw,
            venue=venue,
            year=year,
            fetch_crossref_meta=fetch_crossref_meta,
            fetch_best_crossref_meta=fetch_best_crossref_meta,
            is_weak_meta_value=is_weak_meta_value,
            merge_meta_prefer_richer=merge_meta_prefer_richer,
            build_doi_url=build_doi_url,
        )
    meta = merge_missing_doi_arxiv_fallback(
        meta,
        raw_seed=raw0,
        raw=raw,
        title=title,
        venue=venue,
        arxiv_backfill_meta_from_texts=arxiv_backfill_meta_from_texts,
        normalize_doi_like=normalize_doi_like,
        merge_meta_prefer_richer=merge_meta_prefer_richer,
        should_try_openalex_arxiv_title=should_try_openalex_arxiv_title,
        openalex_arxiv_meta_by_title=openalex_arxiv_meta_by_title,
    )
    return enrich_bibliometrics_and_summary(
        meta,
        enrich_bibliometrics=enrich_bibliometrics,
        ensure_summary_line=ensure_summary_line,
        allow_crossref_abstract=True,
    )
