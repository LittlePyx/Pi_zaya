from __future__ import annotations

import re
from typing import Callable

from api.reference_summary_text import _summary_excerpt


def _ensure_summary_line(
    meta: dict,
    *,
    allow_crossref_abstract: bool,
    looks_low_value_shelf_summary: Callable[[str], bool],
    looks_like_title_echo: Callable[[str, str], bool],
    looks_metadata_only_summary: Callable[[str], bool],
    finalize_abstract_summary_line: Callable[..., tuple[str, str]],
    translate_summary_to_zh: Callable[[str], str],
    attach_summary_quality: Callable[[dict], dict],
    summary_from_crossref_abstract: Callable[[dict], str],
    summary_from_datacite_description: Callable[[dict], str],
    summary_from_europe_pmc_abstract: Callable[[dict], str],
    summary_from_openalex_abstract: Callable[[dict], str],
    summary_from_semantic_scholar_abstract: Callable[[dict], str],
    summary_from_doi_landing_page: Callable[[dict], str],
    contextual_summary_line: Callable[[dict], str],
    metadata_summary_line: Callable[[dict], str],
) -> dict:
    out = dict(meta or {})
    existing_line = _summary_excerpt(str(out.get("summary_line") or ""), max_sentences=3, max_len=360)
    existing_source = str(out.get("summary_source") or "").strip().lower()
    title = str(out.get("title") or "").strip()
    if existing_line and looks_low_value_shelf_summary(existing_line):
        existing_line = ""
        out.pop("summary_line", None)
        out.pop("summary_source", None)
        out.pop("summary_generation", None)
    if existing_line:
        if (existing_source == "metadata") and (
            looks_like_title_echo(existing_line, title)
            or looks_metadata_only_summary(existing_line)
        ):
            existing_line = ""
        elif existing_source == "abstract":
            final_line, generation = finalize_abstract_summary_line(title=title, abstract_text=existing_line)
            out["summary_line"] = final_line or translate_summary_to_zh(existing_line)
            out["summary_source"] = "abstract"
            out["summary_generation"] = generation or "translated_abstract"
            return attach_summary_quality(out)
        else:
            out["summary_line"] = translate_summary_to_zh(existing_line)
            out["summary_source"] = existing_source if existing_source in {"fulltext", "abstract", "metadata"} else "fulltext"
            out["summary_generation"] = "fulltext_existing"
            return attach_summary_quality(out)

    if allow_crossref_abstract:
        abstract_line = summary_from_crossref_abstract(out)
        if abstract_line:
            final_line, generation = finalize_abstract_summary_line(title=title, abstract_text=abstract_line)
            out["summary_line"] = final_line or translate_summary_to_zh(abstract_line)
            out["summary_source"] = "abstract"
            out["summary_generation"] = generation or "translated_abstract"
            out["summary_provider"] = "crossref"
            out["summary_fetch_status"] = "ready"
            return attach_summary_quality(out)
        datacite_line = summary_from_datacite_description(out)
        if datacite_line:
            final_line, generation = finalize_abstract_summary_line(title=title, abstract_text=datacite_line)
            out["summary_line"] = final_line or translate_summary_to_zh(datacite_line)
            out["summary_source"] = "abstract"
            out["summary_generation"] = generation or "translated_abstract"
            out["summary_provider"] = "datacite"
            out["summary_fetch_status"] = "ready"
            return attach_summary_quality(out)
        europe_pmc_line = summary_from_europe_pmc_abstract(out)
        if europe_pmc_line:
            final_line, generation = finalize_abstract_summary_line(title=title, abstract_text=europe_pmc_line)
            out["summary_line"] = final_line or translate_summary_to_zh(europe_pmc_line)
            out["summary_source"] = "abstract"
            out["summary_generation"] = generation or "translated_abstract"
            out["summary_provider"] = "europe_pmc"
            out["summary_fetch_status"] = "ready"
            return attach_summary_quality(out)
        openalex_line = summary_from_openalex_abstract(out)
        if openalex_line:
            final_line, generation = finalize_abstract_summary_line(title=title, abstract_text=openalex_line)
            out["summary_line"] = final_line or translate_summary_to_zh(openalex_line)
            out["summary_source"] = "abstract"
            out["summary_generation"] = generation or "translated_abstract"
            out["summary_provider"] = "openalex"
            out["summary_fetch_status"] = "ready"
            return attach_summary_quality(out)
        semantic_line = summary_from_semantic_scholar_abstract(out)
        if semantic_line:
            final_line, generation = finalize_abstract_summary_line(title=title, abstract_text=semantic_line)
            out["summary_line"] = final_line or translate_summary_to_zh(semantic_line)
            out["summary_source"] = "abstract"
            out["summary_generation"] = generation or "translated_abstract"
            out["summary_provider"] = "semantic_scholar"
            out["summary_fetch_status"] = "ready"
            return attach_summary_quality(out)
        landing_line = summary_from_doi_landing_page(out)
        if landing_line:
            final_line, generation = finalize_abstract_summary_line(title=title, abstract_text=landing_line)
            out["summary_line"] = final_line or translate_summary_to_zh(landing_line)
            out["summary_source"] = "abstract"
            out["summary_generation"] = generation or "translated_abstract"
            out["summary_provider"] = "doi_landing_page"
            out["summary_fetch_status"] = "ready"
            return attach_summary_quality(out)

        provider_status = out.get("summary_fetch_providers")
        provider_values = {
            str(value or "").strip().lower()
            for value in provider_status.values()
        } if isinstance(provider_status, dict) else set()
        if "failed" in provider_values:
            out["summary_fetch_status"] = "retryable"
        elif _normalize_summary_doi(out):
            out["summary_fetch_status"] = "not_provided"
        else:
            out["summary_fetch_status"] = "missing_identity"

    context_fallback = contextual_summary_line(out)
    if context_fallback:
        out["summary_line"] = context_fallback
        out["summary_source"] = "citation_context"
        out["summary_generation"] = "citation_context_fallback"
        return attach_summary_quality(out)

    fallback = metadata_summary_line(out)
    if fallback:
        out["summary_line"] = fallback
        out["summary_source"] = "metadata"
        out["summary_generation"] = "metadata_only"
    return attach_summary_quality(out)


def _normalize_summary_doi(meta: dict) -> str:
    value = str((meta or {}).get("doi") or (meta or {}).get("doi_url") or "").strip().lower()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    return value if value.startswith("10.") and "/" in value else ""
