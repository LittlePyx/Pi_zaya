from __future__ import annotations

"""
Backend-facing reference rendering adapter.

The implementation still lives in ui.refs_renderer while the legacy Streamlit
surface is being retired. Keep production API modules pointed here so the next
extraction can move pure rendering helpers without touching route/business code.
"""

from ui.refs_renderer import (
    _annotate_equation_tags_with_sources,
    _annotate_inpaper_citations_with_hover_meta,
    _build_ref_navigation,
    _enrich_bibliometrics,
    _fallback_fill_reference_meta_from_raw,
    _fallback_why_line_ui,
    _has_metrics_payload,
    _infer_title_from_source_text,
    _is_non_navigational_heading_ui,
    _looks_like_doc_title_heading_ui,
    _normalize_reference_for_popup,
    _open_pdf_at,
    _openalex_work_by_doi,
    _parse_filename_meta,
    _resolve_pdf_for_source,
    _safe_page_range,
    _sanitize_heading_path_ui,
    _score_tier,
    _source_cite_id,
    _split_section_subsection,
    _top_heading,
    fetch_crossref_meta,
)

__all__ = [
    "_annotate_equation_tags_with_sources",
    "_annotate_inpaper_citations_with_hover_meta",
    "_build_ref_navigation",
    "_enrich_bibliometrics",
    "_fallback_fill_reference_meta_from_raw",
    "_fallback_why_line_ui",
    "_has_metrics_payload",
    "_infer_title_from_source_text",
    "_is_non_navigational_heading_ui",
    "_looks_like_doc_title_heading_ui",
    "_normalize_reference_for_popup",
    "_open_pdf_at",
    "_openalex_work_by_doi",
    "_parse_filename_meta",
    "_resolve_pdf_for_source",
    "_safe_page_range",
    "_sanitize_heading_path_ui",
    "_score_tier",
    "_source_cite_id",
    "_split_section_subsection",
    "_top_heading",
    "fetch_crossref_meta",
]
