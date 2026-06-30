from __future__ import annotations

from pathlib import Path
import re
from typing import Callable

from kb.path_safety import clean_file_source_path_input


def ensure_source_citation_meta(
    *,
    source_path: str,
    pdf_root: Path | None,
    md_root: Path | None,
    lib_store: object | None,
    resolve_pdf_for_source: Callable[[Path | None, str], Path | None],
    has_metrics_payload: Callable[[dict], bool],
    parse_filename_meta: Callable[[str], tuple[str, str, str]],
    source_filename: Callable[[str], str],
    infer_title_from_source_text: Callable[..., str],
    fetch_crossref_meta: Callable[..., dict | None],
    is_weak_meta_value: Callable[[str, str], bool],
    fetch_best_crossref_meta: Callable[..., dict | None],
    merge_meta_prefer_richer: Callable[[dict, dict], dict],
    enrich_bibliometrics: Callable[[dict], dict],
    ensure_summary_line: Callable[..., dict],
) -> dict:
    clean_source_path = clean_file_source_path_input(source_path) or str(source_path or "").strip()
    pdf_path = resolve_pdf_for_source(pdf_root, clean_source_path)
    meta: dict = {}
    if pdf_path is not None and lib_store is not None:
        try:
            stored = lib_store.get_citation_meta(pdf_path)  # type: ignore[attr-defined]
            if isinstance(stored, dict):
                meta = dict(stored)
        except Exception:
            meta = {}

    if has_metrics_payload(meta):
        return ensure_summary_line(meta, allow_crossref_abstract=False)

    venue_hint, year_hint, _ = parse_filename_meta(clean_source_path)
    fallback_title = source_filename(clean_source_path) or str(clean_source_path or "")
    if fallback_title.lower().endswith(".pdf"):
        fallback_title = fallback_title[:-4]
    fallback_title = re.sub(r"\.en\.md$", "", fallback_title, flags=re.I)
    fallback_title = re.sub(r"\.md$", "", fallback_title, flags=re.I)
    search_title = infer_title_from_source_text(
        clean_source_path,
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
        source_path=clean_source_path,
        expected_venue=venue_hint,
        expected_year=year_hint,
        md_root_hint=str(md_root or ""),
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
        meta = merge_meta_prefer_richer(
            meta,
            {k: v for k, v in fetched.items() if v not in (None, "", [], {})},
        )

    enriched = enrich_bibliometrics(meta or {})
    if isinstance(enriched, dict):
        meta = enriched
    if isinstance(meta, dict):
        meta = ensure_summary_line(meta, allow_crossref_abstract=False)

    if pdf_path is not None and lib_store is not None and isinstance(meta, dict) and meta:
        try:
            lib_store.set_citation_meta(pdf_path, meta)  # type: ignore[attr-defined]
        except Exception:
            pass
    return meta if isinstance(meta, dict) else {}
