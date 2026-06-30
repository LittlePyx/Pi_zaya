from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import difflib
import json
import math
import os
from pathlib import Path
import re
import time

from api.reference_card_copy import (
    finalize_ref_card_copy as _finalize_ref_card_copy,
    looks_generic_ref_why_line as _card_copy_looks_generic_ref_why_line,
    looks_templated_ref_why_line as _card_copy_looks_templated_ref_why_line,
)
from api.reference_card_payload import build_ref_card_ui_payload as _build_ref_card_ui_payload
from api.reference_card_locale import (
    _prefer_zh_ref_card_locale,
    _prompt_strongly_prefers_english,
    _ref_card_user_locale,
)
from api.reference_card_quality import (
    LLM_SUMMARY_GENERATIONS,
    LLM_WHY_GENERATIONS,
    attach_refs_pack_polish_contract,
    ref_card_polish_status,
    refs_pack_has_full_llm_copy,
)
from api.reference_focus_terms import (
    _PROMPT_FOCUS_STOPWORDS,
    _clean_refs_focus_phrase,
    _focus_term_matches_surface,
    _refs_exact_focus_match_count,
    _refs_prompt_focus_alias_terms,
    _refs_prompt_focus_terms,
)
from api.reference_external_ids import (
    _arxiv_backfill_meta_from_texts,
    _arxiv_doi_from_id,
    _extract_arxiv_id_like,
    _is_weak_meta_value,
    _normalize_doi_like,
    build_doi_url,
)
from api.reference_external_abstracts import (
    _doi_landing_page_abstract as _external_doi_landing_page_abstract,
    _semantic_scholar_paper_by_doi as _external_semantic_scholar_paper_by_doi,
    _summary_from_crossref_abstract as _external_summary_from_crossref_abstract,
    _summary_from_doi_landing_page as _external_summary_from_doi_landing_page,
    _summary_from_openalex_abstract as _external_summary_from_openalex_abstract,
    _summary_from_semantic_scholar_abstract as _external_summary_from_semantic_scholar_abstract,
    _valid_external_abstract_candidate as _external_valid_external_abstract_candidate,
)
from api.reference_external_meta_merge import _merge_meta_prefer_richer
from api.reference_intent import (
    refs_prompt_section_intent as _intent_prompt_section_intent,
    refs_prompt_topic_terms as _intent_prompt_topic_terms,
    refs_section_intent_heading_score as _intent_section_intent_heading_score,
    refs_section_intent_terms as _intent_section_intent_terms,
)
from api.reference_openalex_arxiv import (
    _normalize_title_for_openalex_search as _openalex_arxiv_normalize_title_for_search,
    _openalex_arxiv_meta_by_title as _external_openalex_arxiv_meta_by_title,
    _should_try_openalex_arxiv_title as _external_should_try_openalex_arxiv_title,
    _title_similarity_for_openalex as _openalex_arxiv_title_similarity,
)
from api.reference_source_identity import (
    _normalize_title_identity,
    _same_source_identity,
    _source_filename,
    _title_identity_keys,
)
from api.reference_source_display import (
    _display_source_name as _source_display_name,
    _hit_matches_guide_source,
)
from api.reference_semantic_badges import _build_semantic_badges
from api.reference_summary_text import (
    _clean_summary_line,
    _first_summary_sentence,
    _has_cjk_text,
    _has_latin_text,
    _looks_like_title_echo,
    _summary_excerpt,
)
from api.reference_summary_quality import (
    _has_summary_action_signal as _summary_quality_has_summary_action_signal,
    _has_summary_result_signal as _summary_quality_has_summary_result_signal,
    _is_summary_quality_ok as _summary_quality_is_summary_quality_ok,
    _looks_low_value_shelf_summary as _summary_quality_looks_low_value_shelf_summary,
    _looks_metadata_only_summary as _summary_quality_looks_metadata_only_summary,
    _summary_quality_contract as _external_summary_quality_contract,
)
from api.reference_summary_fallbacks import (
    _contextual_summary_line as _external_contextual_summary_line,
    _metadata_summary_line as _external_metadata_summary_line,
)
from api.reference_summary_pipeline import _ensure_summary_line as _external_ensure_summary_line
from api.reference_summary_llm import (
    _finalize_abstract_summary_line as _external_finalize_abstract_summary_line,
    _llm_summarize_abstract_zh as _external_llm_summarize_abstract_zh,
    _translate_summary_to_zh as _external_translate_summary_to_zh,
)
from api.reference_source_citation_meta import ensure_source_citation_meta as _external_ensure_source_citation_meta
from api.reference_detail_pipeline import enrich_citation_detail_meta as _detail_pipeline_enrich_citation_detail_meta
from api import reference_card_copy_flow as _card_copy_flow
from api import reference_doc_list as _doc_list
from api import reference_heading_context as _heading_context
from api import reference_hit_context as _hit_context
from api import reference_hit_dedupe as _ref_hit_dedupe
from api import reference_primary_evidence as _primary_evidence
from api import reference_reader_open as _reader_open
from api.reference_ui_score import (
    _MAX_REF_UI_GAP,
    _MIN_COMPARE_DIRECT_HIT_SCORE,
    _MIN_PENDING_SINGLE_PAPER_DIRECT_HIT_SCORE,
    _MIN_REF_UI_SCORE,
    _MIN_SINGLE_PAPER_DIRECT_HIT_SCORE,
    _clamp_ui_score,
    _effective_ui_score,
    _should_force_keep_ref_hit,
)
from api.reference_value_utils import _non_negative_float, _positive_int
from kb.config import load_settings
from kb.citation_meta import (
    extract_first_doi,
    fetch_best_crossref_for_reference,
    fetch_best_crossref_meta,
    fetch_crossref_work_by_doi,
)
from kb.evidence_text import clean_display_text as _clean_evidence_display_text
from kb.evidence_text import finish_evidence_text as _finish_evidence_text
from kb.evidence_text import pick_readable_evidence_text as _pick_readable_evidence_text
from kb.library_store import LibraryStore
from kb.llm import DeepSeekChat
from kb.path_safety import clean_file_source_path_input
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
from api.reference_rendering import (
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


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _display_source_name(source_path: str, pdf_path: Path | None, lib_store: LibraryStore | None) -> str:
    return _source_display_name(source_path, pdf_path, lib_store)


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
    picked = _pick_ref_card_summary_fallback(
        prompt=prompt,
        title=title,
        candidates=candidates,
        allow_llm_select=allow_llm_translate,
    )
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
    citation_summary = _pick_ref_card_summary_fallback(
        prompt=prompt,
        title=title,
        candidates=citation_candidates,
        allow_llm_select=allow_llm_translate,
    )
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
    picked = _pick_ref_card_summary_fallback(
        prompt=prompt,
        title=title,
        candidates=candidates,
        allow_llm_select=allow_llm_translate,
    )
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
    if _looks_synthetic_location_discussion_summary(s):
        return True
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
    if _looks_synthetic_location_discussion_summary(s):
        return True
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
    if _looks_synthetic_location_discussion_summary(summary):
        score -= 5.0
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
    "缁欏嚭浜嗕笌",
    "主题一致",
    "直接参考依据",
    "关键证据来源",
    "定义、方法或结果信息",
    "直接对应",
    "直接讨论",
    "直接相关",
)

_SYNTHETIC_LOCATION_DISCUSSION_RE = re.compile(
    r"^\s*(?:(?:该文|本文|这篇(?:文献|论文|文章)|the\s+paper)\s*)?"
    r"(?:在|于|in\s+)?[“\"']?[^“”\"']{8,220}[”\"']\s*"
    r"(?:讨论了|比较了|定义或解释了|给出了与|directly\s+discusses|discusses|compares|defines|explains)"
    r"[“\"']?[^“”\"']{1,140}[”\"']?\s*[。.]?\s*$",
    flags=re.IGNORECASE,
)


def _looks_synthetic_location_discussion_summary(text: str) -> bool:
    s = _clean_summary_line(text)
    if not s:
        return False
    if _SYNTHETIC_LOCATION_DISCUSSION_RE.match(s):
        return True
    return bool(
        re.match(
            r"^\s*[^“”\"']{4,180}[”\"'](?:讨论了|比较了|定义或解释了)[“\"'][^“”\"']{1,120}[”\"']\s*[。.]?\s*$",
            s,
        )
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
    return _external_finalize_abstract_summary_line(
        title=title,
        abstract_text=abstract_text,
        llm_summarize_abstract_zh=_llm_summarize_abstract_zh,
        translate_summary_to_zh=_translate_summary_to_zh,
    )


def _has_ref_summary_explainer_signal(text: str) -> bool:
    s = _clean_summary_line(text)
    if not s:
        return False
    return bool(
        re.search(
            r"\b(compare|comparative|analy[sz]e|analysis|evaluat|study|explore|review|survey|introduce|present|propose|design|develop|use|suitable|benefit)\b",
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
            r"\b(result|show|demonstrat|improv|outperform|achiev|difference|trade-?off|advantage|limitation|quality|efficiency|robustness|fidelity|performance|binary|dmd|quantization|gamma|grayscale|oblique|periodical|noise|sampling)\b",
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
    if "漏" in cand:
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
    if _looks_synthetic_location_discussion_summary(cand):
        return -1000.0
    if _looks_generic_ref_why_line(cand):
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
    if "..." in cand or "…" in cand:
        score -= 2.0
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


def _pick_ref_card_summary_fallback(
    *,
    prompt: str,
    title: str,
    candidates: list[str],
    allow_llm_select: bool = True,
) -> str:
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
            variants.append(sent)
            if heading:
                variants.append(f"{heading}: {sent}")
            variants.extend(
                _definition_prompt_summary_rewrites(
                    prompt=prompt,
                    heading=heading,
                    sentence=sent,
                    next_sentence=next_sentence,
                )
            )
        for idx in range(max(0, len(sentences) - 1)):
            window = f"{sentences[idx]} {sentences[idx + 1]}".strip()
            if len(window) <= 360:
                variants.append(window)

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
    if allow_llm_select and len(close_candidates) >= 2 and _refs_card_polish_llm_enabled():
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

    def _push_sentence_variants(raw: str) -> None:
        text = str(raw or "").strip()
        if not text:
            return
        heading, body = _split_ref_summary_heading_and_body(text)
        sentences = _split_ref_summary_sentences(body or text, max_sentences=8)
        for sent in sentences[:6]:
            _push(sent)
        for idx in range(max(0, min(len(sentences) - 1, 5))):
            window = f"{sentences[idx]} {sentences[idx + 1]}".strip()
            if len(window) <= 360:
                _push(window)
        if heading and sentences:
            _push(f"{heading}: {sentences[0]}")

    if isinstance(meta, dict):
        for key, limit in (("ref_show_snippets", 2), ("ref_snippets", 2), ("ref_overview_snippets", 1)):
            raw_arr = meta.get(key)
            if not isinstance(raw_arr, list):
                continue
            for item in raw_arr[:limit]:
                raw_item = str(item or "")
                _push(raw_item)
                _push_sentence_variants(raw_item)
        raw_locs = meta.get("ref_locs")
        if isinstance(raw_locs, list):
            for loc in raw_locs[:2]:
                if not isinstance(loc, dict):
                    continue
                for key in ("snippet", "text", "quote", "summary"):
                    raw_loc = str(loc.get(key) or "")
                    _push(raw_loc)
                    _push_sentence_variants(raw_loc)
    for raw in (
        str((ui_meta or {}).get("summary_line") or ""),
        str((ui_meta or {}).get("why_line") or ""),
        str((hit or {}).get("text") or ""),
    ):
        _push(raw)
        _push_sentence_variants(raw)
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


def _apply_deterministic_ref_card_copy_fallback(
    *,
    prompt: str,
    ui_meta: dict,
    candidates: list[str],
) -> dict:
    ui = dict(ui_meta or {})
    title = str(ui.get("display_name") or "").strip()
    heading_path = str(ui.get("heading_path") or ui.get("section_label") or "").strip()
    summary_kind = str(ui.get("summary_kind") or "").strip().lower() or "guide"
    summary_line = _normalize_ref_copy_text(str(ui.get("summary_line") or "").strip())
    why_line = _normalize_ref_copy_text(str(ui.get("why_line") or "").strip())

    if _summary_line_needs_polish(prompt=prompt, title=title, summary_line=summary_line):
        fallback_summary = _normalize_ref_copy_text(
            _pick_ref_card_summary_fallback(
                prompt=prompt,
                title=title,
                candidates=[str(item or "").strip() for item in list(candidates or []) if str(item or "").strip()],
            )
        )
        if fallback_summary and not _summary_line_needs_polish(
            prompt=prompt,
            title=title,
            summary_line=fallback_summary,
        ):
            ui["summary_line"] = fallback_summary
            summary_line = fallback_summary
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

    if _why_line_needs_polish(
        prompt=prompt,
        display_name=title,
        heading_path=heading_path,
        summary_line=summary_line,
        why_line=why_line,
    ):
        deterministic_why = _normalize_ref_copy_text(
            _build_prompt_aligned_ref_why_line_v3(
                prompt=prompt,
                display_name=title,
                heading_path=heading_path,
                summary_line=summary_line,
                why_line=why_line,
            )
        )
        if deterministic_why and not _looks_generic_ref_why_line(deterministic_why):
            ui["why_line"] = deterministic_why
            why_generation = "deterministic_grounded"
            why_basis_meta = _build_ref_why_basis_meta(
                prompt=prompt,
                why_generation=why_generation,
                why_line=deterministic_why,
            )
            ui["why_generation"] = str(why_basis_meta.get("why_generation") or why_generation)
            ui["why_basis"] = str(why_basis_meta.get("why_basis") or "")

    return ui


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
    if not _is_llm_ref_summary_generation(str((result or {}).get("summary_generation") or "")):
        result = _apply_deterministic_ref_card_copy_fallback(
            prompt=prompt,
            ui_meta=result,
            candidates=candidates,
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
    return _reader_open._compact_reader_open_text(text, max_len=max_len)


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
    return _reader_open._pick_reader_open_loc_text(loc)


def _refs_reader_open_candidate_key(candidate: dict) -> str:
    return _reader_open._refs_reader_open_candidate_key(candidate)


def _normalize_refs_reader_heading_path(*, prompt: str, source_path: str, heading_path: str) -> str:
    return _reader_open._normalize_refs_reader_heading_path(
        prompt=prompt,
        source_path=source_path,
        heading_path=heading_path,
        sanitize_heading_path=_sanitize_heading_path_ui,
        looks_like_doc_title_heading=_looks_like_doc_title_heading_ui,
    )


def _refs_heading_paths_related(left: str, right: str) -> bool:
    return _reader_open._refs_heading_paths_related(left, right)


def _refs_heading_anchor_number(anchor_kind: str, heading_path: str) -> int:
    return _reader_open._refs_heading_anchor_number(
        anchor_kind,
        heading_path,
        extract_figure_number=extract_figure_number,
        extract_equation_number=extract_equation_number,
    )


def _clean_refs_evidence_snippet(
    raw: str,
    *,
    prompt: str,
    source_path: str,
    display_name: str = "",
    heading_path: str = "",
    max_len: int = 360,
) -> str:
    return _reader_open._clean_refs_evidence_snippet(
        raw,
        prompt=prompt,
        source_path=source_path,
        display_name=display_name,
        heading_path=heading_path,
        max_len=max_len,
        pick_readable_evidence_text=_pick_readable_evidence_text,
        clean_evidence_display_text=_clean_evidence_display_text,
    )


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
    return _reader_open._build_refs_reader_open_candidate(
        prompt=prompt,
        source_path=source_path,
        heading_path=heading_path,
        snippet=snippet,
        highlight_snippet=highlight_snippet,
        anchor_kind=anchor_kind,
        anchor_number=anchor_number,
        sanitize_heading_path=_sanitize_heading_path_ui,
        looks_like_doc_title_heading=_looks_like_doc_title_heading_ui,
        pick_readable_evidence_text=_pick_readable_evidence_text,
        clean_evidence_display_text=_clean_evidence_display_text,
    )


def _infer_heading_path_for_summary_from_source_blocks(
    *,
    prompt: str,
    source_path: str,
    summary_line: str,
    anchor_target_kind: str,
    anchor_target_number: int,
) -> str:
    return _reader_open._infer_heading_path_for_summary_from_source_blocks(
        prompt=prompt,
        source_path=source_path,
        summary_line=summary_line,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        resolve_source_md_path=_resolve_source_md_path,
        load_source_blocks=load_source_blocks,
        match_source_blocks=match_source_blocks,
        sanitize_heading_path=_sanitize_heading_path_ui,
        looks_like_doc_title_heading=_looks_like_doc_title_heading_ui,
    )


def _resolve_source_md_path(source_path: str) -> Path | None:
    raw = clean_file_source_path_input(source_path)
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
    return _reader_open._score_refs_exact_surface(
        text,
        prompt=prompt,
        title=title,
        block_kind=block_kind,
        anchor_target_kind=anchor_target_kind,
        looks_bibliographic_source_block_text=_looks_bibliographic_source_block_text,
        looks_title_like_ref_surface=_looks_title_like_ref_surface,
        looks_like_front_matter_ref_summary=_looks_like_front_matter_ref_summary,
        looks_prefixed_heading_shell_ref_summary=_looks_prefixed_heading_shell_ref_summary,
        looks_surface_like_ref_summary=_looks_surface_like_ref_summary,
        looks_fragmentary_ref_summary=_looks_fragmentary_ref_summary,
        looks_why_like_ref_summary=_looks_why_like_ref_summary,
        looks_formula_heavy_ref_text=_looks_formula_heavy_ref_text,
        prompt_reference_focus_action=_shared_prompt_reference_focus_action,
        refs_summary_focus_keyword_hit_count=_refs_summary_focus_keyword_hit_count,
        looks_natural_language_ref_summary=_looks_natural_language_ref_summary,
        has_ref_summary_explainer_signal=_has_ref_summary_explainer_signal,
        has_ref_summary_value_signal=_has_ref_summary_value_signal,
        refs_exact_focus_match_count=_refs_exact_focus_match_count,
        matched_focus_terms_for_ref_card=_matched_focus_terms_for_ref_card,
    )


def _select_reader_open_exact_snippet(
    seed_text: str,
    block_text: str,
    *,
    prompt: str = "",
    title: str = "",
    block_kind: str = "",
    anchor_target_kind: str = "",
) -> tuple[str, str]:
    return _reader_open._select_reader_open_exact_snippet(
        seed_text,
        block_text,
        prompt=prompt,
        title=title,
        block_kind=block_kind,
        anchor_target_kind=anchor_target_kind,
        score_refs_exact_surface=_score_refs_exact_surface,
        looks_focus_prefixed_ref_summary=_looks_focus_prefixed_ref_summary,
        summary_line_needs_polish=_summary_line_needs_polish,
    )


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
    return _reader_open._build_refs_exact_candidate_from_block(
        prompt=prompt,
        source_path=source_path,
        title=title,
        block=block,
        seed_heading_path=seed_heading_path,
        seed_snippet=seed_snippet,
        anchor_kind=anchor_kind,
        anchor_number=anchor_number,
        select_reader_open_exact_snippet=_select_reader_open_exact_snippet,
        build_refs_reader_open_candidate=_build_refs_reader_open_candidate,
    )


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
    return _reader_open._build_preferred_refs_exact_candidate_from_source_summary(
        prompt=prompt,
        source_path=source_path,
        title=title,
        summary_line=summary_line,
        selected_heading_path=selected_heading_path,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        prompt_aligned_candidate=prompt_aligned_candidate,
        ref_summary_surfaces_match=_ref_summary_surfaces_match,
        normalize_refs_reader_heading_path=_normalize_refs_reader_heading_path,
        select_reader_open_exact_snippet=_select_reader_open_exact_snippet,
        build_refs_reader_open_candidate=_build_refs_reader_open_candidate,
    )


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
    return _reader_open._resolve_refs_exact_candidates(
        prompt=prompt,
        source_path=source_path,
        display_name=display_name,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        primary_candidate=primary_candidate,
        secondary_candidates=secondary_candidates,
        allow_llm_disambiguation=allow_llm_disambiguation,
        resolve_source_md_path=_resolve_source_md_path,
        load_source_blocks=load_source_blocks,
        match_source_blocks=match_source_blocks,
        build_refs_exact_candidate_from_block=_build_refs_exact_candidate_from_block,
        refs_heading_paths_related=_refs_heading_paths_related,
        refs_heading_anchor_number=_refs_heading_anchor_number,
        score_refs_exact_surface=_score_refs_exact_surface,
        refs_exact_focus_match_count=_refs_exact_focus_match_count,
        matched_focus_terms_for_ref_card=_matched_focus_terms_for_ref_card,
        should_try_refs_locate_llm=_should_try_refs_locate_llm,
        llm_pick_refs_exact_candidate_index=_llm_pick_refs_exact_candidate_index,
    )


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
    return _reader_open._build_refs_reader_open_payload(
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
        build_refs_reader_open_candidate=_build_refs_reader_open_candidate,
        resolve_refs_exact_candidates=_resolve_refs_exact_candidates,
        prompt_requires_explicit_focus_match=_prompt_requires_explicit_focus_match,
        preferred_exact_candidate=preferred_exact_candidate,
        allow_llm_disambiguation=allow_llm_disambiguation,
        allow_exact_locate=allow_exact_locate,
    )


def _build_primary_ref_evidence_payload(
    *,
    source_path: str,
    display_name: str,
    reader_open: dict,
    selection_reason: str,
    score: float | None,
    prompt: str = "",
) -> dict:
    return _reader_open._build_primary_ref_evidence_payload(
        source_path=source_path,
        display_name=display_name,
        reader_open=reader_open,
        selection_reason=selection_reason,
        score=score,
        prompt=prompt,
        clean_refs_evidence_snippet=_clean_refs_evidence_snippet,
    )


def _normalize_primary_ref_evidence_payload(primary_evidence: dict | None) -> dict:
    return _reader_open._normalize_primary_ref_evidence_payload(
        primary_evidence,
        finish_evidence_text=_finish_evidence_text,
    )


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
    return _doc_list._collect_doc_list_ref_text_candidates(
        raw_item=raw_item,
        primary_evidence=primary_evidence,
        clean_refs_evidence_snippet=_clean_refs_evidence_snippet,
    )


def _primary_ref_evidence_summary_seed(primary_evidence: dict | None) -> str:
    return _doc_list._primary_ref_evidence_summary_seed(
        primary_evidence,
        normalize_primary_ref_evidence_payload=_normalize_primary_ref_evidence_payload,
        clean_refs_evidence_snippet=_clean_refs_evidence_snippet,
    )


def _primary_ref_evidence_points_to_same_surface(
    left_primary: dict | None,
    right_primary: dict | None,
) -> bool:
    return _doc_list._primary_ref_evidence_points_to_same_surface(
        left_primary,
        right_primary,
        normalize_primary_ref_evidence_payload=_normalize_primary_ref_evidence_payload,
        primary_ref_evidence_summary_seed=_primary_ref_evidence_summary_seed,
        same_source_identity=_same_source_identity,
        ref_summary_surfaces_match=_ref_summary_surfaces_match,
    )


def _doc_list_authoritative_primary_is_upgradeable(primary_evidence: dict | None) -> bool:
    return _doc_list._doc_list_authoritative_primary_is_upgradeable(
        primary_evidence,
        normalize_primary_ref_evidence_payload=_normalize_primary_ref_evidence_payload,
    )


def _primary_ref_evidence_summary_is_usable(
    primary_evidence: dict | None,
    *,
    prompt: str,
    display_name: str,
) -> bool:
    return _doc_list._primary_ref_evidence_summary_is_usable(
        primary_evidence,
        prompt=prompt,
        display_name=display_name,
        primary_ref_evidence_summary_seed=_primary_ref_evidence_summary_seed,
        looks_bibliographic_source_block_text=_looks_bibliographic_source_block_text,
        summary_line_needs_polish=_summary_line_needs_polish,
    )


def _upgrade_primary_ref_evidence_from_alternatives(
    primary_evidence: dict | None,
    *,
    prompt: str,
    display_name: str,
) -> dict:
    return _doc_list._upgrade_primary_ref_evidence_from_alternatives(
        primary_evidence,
        prompt=prompt,
        display_name=display_name,
        normalize_primary_ref_evidence_payload=_normalize_primary_ref_evidence_payload,
        primary_ref_evidence_summary_is_usable=_primary_ref_evidence_summary_is_usable,
        primary_ref_evidence_precision_score=_primary_ref_evidence_precision_score,
    )


def _primary_ref_evidence_precision_score(
    *,
    primary_evidence: dict | None,
    prompt: str,
    display_name: str,
) -> tuple[int, int, int, int, int, int, int]:
    return _doc_list._primary_ref_evidence_precision_score(
        primary_evidence=primary_evidence,
        prompt=prompt,
        display_name=display_name,
        normalize_primary_ref_evidence_payload=_normalize_primary_ref_evidence_payload,
        sanitize_heading_path=_sanitize_heading_path_ui,
        primary_ref_evidence_summary_seed=_primary_ref_evidence_summary_seed,
        primary_ref_evidence_summary_is_usable=_primary_ref_evidence_summary_is_usable,
    )


def _select_doc_list_effective_primary_evidence(
    *,
    prompt: str,
    display_name: str,
    authoritative_primary_evidence: dict | None,
    synthesized_primary_evidence: dict | None,
) -> tuple[dict, str]:
    return _doc_list._select_doc_list_effective_primary_evidence(
        prompt=prompt,
        display_name=display_name,
        authoritative_primary_evidence=authoritative_primary_evidence,
        synthesized_primary_evidence=synthesized_primary_evidence,
        normalize_primary_ref_evidence_payload=_normalize_primary_ref_evidence_payload,
        upgrade_primary_ref_evidence_from_alternatives=_upgrade_primary_ref_evidence_from_alternatives,
        primary_ref_evidence_points_to_same_surface=_primary_ref_evidence_points_to_same_surface,
        doc_list_authoritative_primary_is_upgradeable=_doc_list_authoritative_primary_is_upgradeable,
        primary_ref_evidence_precision_score=_primary_ref_evidence_precision_score,
    )


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
    return _doc_list._apply_doc_list_effective_primary_evidence(
        prompt=prompt,
        display_name=display_name,
        fallback_heading_path=fallback_heading_path,
        ui_meta=ui_meta,
        authoritative_primary_evidence=authoritative_primary_evidence,
        authoritative_summary_line=authoritative_summary_line,
        authoritative_summary_generation=authoritative_summary_generation,
        normalize_primary_ref_evidence_payload=_normalize_primary_ref_evidence_payload,
        select_doc_list_effective_primary_evidence=_select_doc_list_effective_primary_evidence,
        primary_ref_evidence_summary_seed=_primary_ref_evidence_summary_seed,
        compact_reader_open_text=_compact_reader_open_text,
        summary_line_needs_polish=_summary_line_needs_polish,
        primary_ref_evidence_points_to_same_surface=_primary_ref_evidence_points_to_same_surface,
        build_ref_summary_basis_meta=_build_ref_summary_basis_meta,
    )


def _build_doc_list_ref_locs(*, heading_path: str, primary_evidence: dict) -> list[dict]:
    return _doc_list._build_doc_list_ref_locs(
        heading_path=heading_path,
        primary_evidence=primary_evidence,
        clean_refs_evidence_snippet=_clean_refs_evidence_snippet,
        top_heading=_top_heading,
    )


def _build_doc_list_ref_hit(*, raw_item: dict, idx: int) -> dict:
    return _doc_list._build_doc_list_ref_hit(
        raw_item=raw_item,
        idx=idx,
        source_filename=_source_filename,
        normalize_primary_ref_evidence_payload=_normalize_primary_ref_evidence_payload,
        compact_reader_open_text=_compact_reader_open_text,
        split_section_subsection=_split_section_subsection,
        top_heading=_top_heading,
        clean_refs_evidence_snippet=_clean_refs_evidence_snippet,
    )


def _build_doc_list_reader_open_payload(
    *,
    source_path: str,
    source_name: str,
    heading_path: str,
    summary_line: str,
    primary_evidence: dict,
    reader_open: dict | None,
) -> dict:
    return _reader_open._build_doc_list_reader_open_payload(
        source_path=source_path,
        source_name=source_name,
        heading_path=heading_path,
        summary_line=summary_line,
        primary_evidence=primary_evidence,
        reader_open=reader_open,
        normalize_primary_ref_evidence_payload=_normalize_primary_ref_evidence_payload,
        clean_refs_evidence_snippet=_clean_refs_evidence_snippet,
    )


def _build_doc_list_hit_ui_seed(
    *,
    raw_item: dict,
    idx: int,
    prompt: str,
) -> tuple[dict, dict, dict]:
    return _doc_list._build_doc_list_hit_ui_seed(
        raw_item=raw_item,
        idx=idx,
        prompt=prompt,
        build_doc_list_ref_hit=_build_doc_list_ref_hit,
        source_filename=_source_filename,
        normalize_primary_ref_evidence_payload=_normalize_primary_ref_evidence_payload,
        compact_reader_open_text=_compact_reader_open_text,
        normalize_ref_copy_text=_normalize_ref_copy_text,
        resolve_ref_ui_heading_context=_resolve_ref_ui_heading_context,
        top_heading=_top_heading,
        primary_ref_evidence_summary_seed=_primary_ref_evidence_summary_seed,
        build_ref_summary_basis_meta=_build_ref_summary_basis_meta,
        build_prompt_aligned_ref_why_line=_build_prompt_aligned_ref_why_line_v3,
        doc_list_ref_why_line=_doc_list_ref_why_line,
        prefer_zh_ref_card_locale=_prefer_zh_ref_card_locale,
        build_ref_why_basis_meta=_build_ref_why_basis_meta,
        summary_label="\u5bfc\u8bfb",
        summary_title="\u8fd9\u6761\u8bc1\u636e\u8bf4\u660e\u4ec0\u4e48",
    )


def _apply_doc_list_summary_fallbacks(
    *,
    raw_item: dict,
    prompt: str,
    source_name: str,
    heading_path: str,
    ui_meta: dict | None,
    primary_evidence: dict,
    effective_primary_evidence: dict,
    summary_source: str,
) -> tuple[dict, str]:
    return _doc_list._apply_doc_list_summary_fallbacks(
        raw_item=raw_item,
        prompt=prompt,
        source_name=source_name,
        heading_path=heading_path,
        ui_meta=ui_meta,
        primary_evidence=primary_evidence,
        effective_primary_evidence=effective_primary_evidence,
        summary_source=summary_source,
        summary_line_needs_polish=_summary_line_needs_polish,
        looks_like_title_echo=_looks_like_title_echo,
        looks_why_like_ref_summary=_looks_why_like_ref_summary,
        pick_ref_card_summary_fallback=_pick_ref_card_summary_fallback,
        collect_doc_list_ref_text_candidates=_collect_doc_list_ref_text_candidates,
        build_ref_summary_basis_meta=_build_ref_summary_basis_meta,
        looks_fragmentary_ref_summary=_looks_fragmentary_ref_summary,
        looks_surface_like_ref_summary=_looks_surface_like_ref_summary,
        looks_formula_heavy_ref_text=_looks_formula_heavy_ref_text,
        build_prompt_aligned_ref_summary_fallback=_build_prompt_aligned_ref_summary_fallback,
        compact_reader_open_text=_compact_reader_open_text,
        primary_ref_evidence_summary_seed=_primary_ref_evidence_summary_seed,
    )


def _apply_doc_list_why_fallback(
    *,
    prompt: str,
    source_name: str,
    heading_path: str,
    ui_meta: dict | None,
) -> dict:
    return _doc_list._apply_doc_list_why_fallback(
        prompt=prompt,
        source_name=source_name,
        heading_path=heading_path,
        ui_meta=ui_meta,
        why_line_needs_polish=_why_line_needs_polish,
        build_prompt_aligned_ref_why_line=_build_prompt_aligned_ref_why_line_v3,
        doc_list_ref_why_line=_doc_list_ref_why_line,
        prefer_zh_ref_card_locale=_prefer_zh_ref_card_locale,
        build_ref_why_basis_meta=_build_ref_why_basis_meta,
    )


def _finalize_doc_list_hit_ui_meta(
    *,
    raw_item: dict,
    idx: int,
    prompt: str,
    source_path: str,
    source_name: str,
    heading_path: str,
    ui_meta: dict | None,
    primary_evidence: dict,
    effective_primary_evidence: dict,
    summary_source: str,
    allow_expensive_llm: bool,
) -> dict:
    return _doc_list._finalize_doc_list_hit_ui_meta(
        raw_item=raw_item,
        idx=idx,
        prompt=prompt,
        source_path=source_path,
        source_name=source_name,
        heading_path=heading_path,
        ui_meta=ui_meta,
        primary_evidence=primary_evidence,
        effective_primary_evidence=effective_primary_evidence,
        summary_source=summary_source,
        allow_expensive_llm=allow_expensive_llm,
        align_ref_card_copy_to_user_locale=_align_ref_card_copy_to_user_locale,
        build_ref_summary_surface_meta=_build_ref_summary_surface_meta,
        build_ref_summary_basis_meta=_build_ref_summary_basis_meta,
        build_ref_why_basis_meta=_build_ref_why_basis_meta,
        score_tier=_score_tier,
        build_doc_list_reader_open_payload=_build_doc_list_reader_open_payload,
    )


def _build_doc_list_hit_ui_meta(
    *,
    raw_item: dict,
    idx: int,
    prompt: str,
    allow_expensive_llm: bool,
    allow_exact_locate: bool,
) -> dict:
    return _doc_list._build_doc_list_hit_ui_meta(
        raw_item=raw_item,
        idx=idx,
        prompt=prompt,
        allow_expensive_llm=allow_expensive_llm,
        allow_exact_locate=allow_exact_locate,
        source_filename=_source_filename,
        compact_reader_open_text=_compact_reader_open_text,
        normalize_primary_ref_evidence_payload=_normalize_primary_ref_evidence_payload,
        build_doc_list_ref_hit=_build_doc_list_ref_hit,
        build_hit_ui_meta=build_hit_ui_meta,
        build_doc_list_hit_ui_seed=_build_doc_list_hit_ui_seed,
        apply_doc_list_effective_primary_evidence=_apply_doc_list_effective_primary_evidence,
        apply_doc_list_summary_fallbacks=_apply_doc_list_summary_fallbacks,
        apply_doc_list_why_fallback=_apply_doc_list_why_fallback,
        finalize_doc_list_hit_ui_meta=_finalize_doc_list_hit_ui_meta,
    )


def _doc_list_topic_match_why_line(
    *,
    prompt: str,
    heading_path: str,
    match_kind: str,
) -> str:
    return _doc_list._doc_list_topic_match_why_line(
        prompt=prompt,
        heading_path=heading_path,
        match_kind=match_kind,
        prefer_zh_ref_card_locale=_prefer_zh_ref_card_locale,
    )


def _apply_doc_list_topic_match_hints(*, prompt: str, raw_item: dict, ui_meta: dict) -> dict:
    return _doc_list._apply_doc_list_topic_match_hints(
        prompt=prompt,
        raw_item=raw_item,
        ui_meta=ui_meta,
        doc_list_topic_match_why_line=_doc_list_topic_match_why_line,
        is_llm_ref_why_generation=_is_llm_ref_why_generation,
        why_line_needs_polish=_why_line_needs_polish,
        why_line_explicitly_names_focus_term=_why_line_explicitly_names_focus_term,
        build_ref_why_basis_meta=_build_ref_why_basis_meta,
        compact_reader_open_text=_compact_reader_open_text,
        is_llm_ref_summary_generation=_is_llm_ref_summary_generation,
        summary_line_needs_polish=_summary_line_needs_polish,
        looks_like_title_echo=_looks_like_title_echo,
        build_ref_summary_basis_meta=_build_ref_summary_basis_meta,
    )


def _filter_doc_list_rows_for_guide(
    *,
    doc_rows: list[dict] | None,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
    filter_bound_source: bool = False,
) -> tuple[list[dict], int]:
    return _doc_list._filter_doc_list_rows_for_guide(
        doc_rows=doc_rows,
        guide_mode=guide_mode,
        guide_source_path=guide_source_path,
        guide_source_name=guide_source_name,
        filter_bound_source=filter_bound_source,
        source_filename=_source_filename,
        hit_matches_guide_source=_hit_matches_guide_source,
    )


def _build_doc_list_payload_hits(
    *,
    doc_rows: list[dict] | None,
    prompt: str,
    allow_expensive_llm: bool,
    allow_exact_locate: bool,
) -> list[dict]:
    return _doc_list._build_doc_list_payload_hits(
        doc_rows=doc_rows,
        prompt=prompt,
        allow_expensive_llm=allow_expensive_llm,
        allow_exact_locate=allow_exact_locate,
        build_doc_list_hit_ui_meta=_build_doc_list_hit_ui_meta,
        normalize_ref_copy_ui_meta=_normalize_ref_copy_ui_meta,
        apply_doc_list_topic_match_hints=_apply_doc_list_topic_match_hints,
    )


def _polish_doc_list_payload_hits(
    *,
    prompt: str,
    doc_rows: list[dict] | None,
    hits: list[dict],
    allow_expensive_llm: bool,
) -> list[dict]:
    return _doc_list._polish_doc_list_payload_hits(
        prompt=prompt,
        doc_rows=doc_rows,
        hits=hits,
        allow_expensive_llm=allow_expensive_llm,
        normalize_ref_copy_ui_meta=_normalize_ref_copy_ui_meta,
        maybe_polish_single_ref_hit_card=_maybe_polish_single_ref_hit_card,
        apply_doc_list_topic_match_hints=_apply_doc_list_topic_match_hints,
        batch_polish_doc_list_ref_hit_cards=_batch_polish_doc_list_ref_hit_cards,
        ref_card_has_llm_copy=_ref_card_has_llm_copy,
        refs_card_polish_max_workers=_refs_card_polish_max_workers,
    )


def _finalize_doc_list_payload_pack(
    *,
    user_msg_id: int | str,
    pack_src: dict | None,
    hits: list[dict],
    guide_active: bool,
    guide_source_path_norm: str,
    guide_source_name_norm: str,
    prompt_cross_paper_refs: bool,
    filtered_self_doc_count: int,
    allow_expensive_llm: bool,
) -> dict:
    return _doc_list._finalize_doc_list_payload_pack(
        user_msg_id=user_msg_id,
        pack_src=pack_src,
        hits=hits,
        guide_active=guide_active,
        guide_source_path_norm=guide_source_path_norm,
        guide_source_name_norm=guide_source_name_norm,
        prompt_cross_paper_refs=prompt_cross_paper_refs,
        filtered_self_doc_count=filtered_self_doc_count,
        allow_expensive_llm=allow_expensive_llm,
        refs_hits_have_llm_copy=_refs_hits_have_llm_copy,
        source_filename=_source_filename,
        attach_pack_display_contract=_attach_pack_display_contract,
    )


def _build_legacy_doc_list_payload_hits(
    *,
    doc_list: list[dict] | None,
    prompt: str,
    prefer_zh: bool,
) -> list[dict]:
    return _doc_list._build_legacy_doc_list_payload_hits(
        doc_list=doc_list,
        prompt=prompt,
        prefer_zh=prefer_zh,
        source_filename=_source_filename,
        normalize_primary_ref_evidence_payload=_normalize_primary_ref_evidence_payload,
        compact_reader_open_text=_compact_reader_open_text,
        doc_list_ref_why_line=_doc_list_ref_why_line,
        score_tier=_score_tier,
    )


def _finalize_legacy_doc_list_payload_pack(
    *,
    user_msg_id: int | str,
    pack_src: dict | None,
    hits: list[dict],
    guide_active: bool,
    guide_source_path_norm: str,
    guide_source_name_norm: str,
    prompt_cross_paper_refs: bool,
) -> dict:
    return _doc_list._finalize_legacy_doc_list_payload_pack(
        user_msg_id=user_msg_id,
        pack_src=pack_src,
        hits=hits,
        guide_active=guide_active,
        guide_source_path_norm=guide_source_path_norm,
        guide_source_name_norm=guide_source_name_norm,
        prompt_cross_paper_refs=prompt_cross_paper_refs,
        source_filename=_source_filename,
        attach_pack_display_contract=_attach_pack_display_contract,
    )


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
    return _doc_list._build_doc_list_refs_payload(
        user_msg_id=user_msg_id,
        pack=pack,
        doc_list=doc_list,
        allow_expensive_llm=allow_expensive_llm,
        allow_exact_locate=allow_exact_locate,
        apply_copy_polish=apply_copy_polish,
        guide_mode=guide_mode,
        guide_source_path=guide_source_path,
        guide_source_name=guide_source_name,
        prompt_likely_cross_paper_refs=_prompt_likely_cross_paper_refs,
        filter_doc_list_rows_for_guide=_filter_doc_list_rows_for_guide,
        build_doc_list_payload_hits=_build_doc_list_payload_hits,
        polish_doc_list_payload_hits=_polish_doc_list_payload_hits,
        suppress_non_llm_ref_card_copy_hits=_suppress_non_llm_ref_card_copy_hits,
        finalize_doc_list_payload_pack=_finalize_doc_list_payload_pack,
        prefer_zh_ref_card_locale=_prefer_zh_ref_card_locale,
        build_legacy_doc_list_payload_hits=_build_legacy_doc_list_payload_hits,
        finalize_legacy_doc_list_payload_pack=_finalize_legacy_doc_list_payload_pack,
    )


def _resolve_ref_ui_heading_context(
    *,
    prompt: str,
    source_path: str,
    heading_path: str,
    heading_fallback: str = "",
    section_label: str = "",
    subsection_label: str = "",
) -> dict[str, str]:
    return _heading_context._resolve_ref_ui_heading_context(
        prompt=prompt,
        source_path=source_path,
        heading_path=heading_path,
        heading_fallback=heading_fallback,
        section_label=section_label,
        subsection_label=subsection_label,
        sanitize_heading_path_ui=_sanitize_heading_path_ui,
        top_heading=_top_heading,
        is_non_navigational_heading_ui=_is_non_navigational_heading_ui,
        looks_like_doc_title_heading_ui=_looks_like_doc_title_heading_ui,
        split_section_subsection=_split_section_subsection,
    )


def _should_allow_ref_summary_block_rescue(
    *,
    prompt: str,
    source_path: str,
    ref_pack_state: str,
    allow_exact_locate: bool,
) -> bool:
    return _heading_context._should_allow_ref_summary_block_rescue(
        prompt=prompt,
        source_path=source_path,
        ref_pack_state=ref_pack_state,
        allow_exact_locate=allow_exact_locate,
        extract_figure_number=extract_figure_number,
        extract_equation_number=extract_equation_number,
        prompt_requires_explicit_focus_match=_prompt_requires_explicit_focus_match,
    )


def _resolve_primary_ref_evidence_summary_selection(
    *,
    meta: dict,
    prompt: str,
    source_path: str,
    display_name: str,
    citation_meta: dict | None,
    heading_path: str,
    heading: str,
    anchor_target_kind: str,
    anchor_target_number: int,
    allow_summary_block_rescue: bool,
    allow_llm_translate: bool,
) -> dict[str, object]:
    return _primary_evidence._resolve_primary_ref_evidence_summary_selection(
        meta=meta,
        prompt=prompt,
        source_path=source_path,
        display_name=display_name,
        citation_meta=citation_meta,
        heading_path=heading_path,
        heading=heading,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        allow_summary_block_rescue=allow_summary_block_rescue,
        allow_llm_translate=allow_llm_translate,
        build_ref_navigation=_build_ref_navigation,
        fallback_ref_ui_summary_line=_fallback_ref_ui_summary_line,
        choose_prompt_aligned_ref_summary_candidate=_choose_prompt_aligned_ref_summary_candidate,
        looks_focus_prefixed_ref_summary=_looks_focus_prefixed_ref_summary,
        summary_line_needs_polish=_summary_line_needs_polish,
        sanitize_heading_path_ui=_sanitize_heading_path_ui,
        rank_prompt_aligned_ref_summary_candidate=_rank_prompt_aligned_ref_summary_candidate,
        choose_prompt_aligned_ref_summary_candidate_from_source_blocks=(
            _choose_prompt_aligned_ref_summary_candidate_from_source_blocks
        ),
        pick_best_prompt_aligned_ref_summary_candidate=_pick_best_prompt_aligned_ref_summary_candidate,
        refs_heading_anchor_number=_refs_heading_anchor_number,
        refs_heading_paths_related=_refs_heading_paths_related,
        infer_heading_path_for_summary_from_source_blocks=_infer_heading_path_for_summary_from_source_blocks,
        ref_summary_focus_score=_ref_summary_focus_score,
        matched_focus_terms_for_ref_card=_matched_focus_terms_for_ref_card,
        ref_summary_surfaces_match=_ref_summary_surfaces_match,
    )


def _apply_reader_anchor_summary_override(
    *,
    reader_open: dict | None,
    prompt: str,
    source_path: str,
    display_name: str,
    summary_line: str,
    summary_source: str,
    anchor_target_kind: str,
    anchor_target_number: int,
) -> tuple[str, str]:
    return _primary_evidence._apply_reader_anchor_summary_override(
        reader_open=reader_open,
        prompt=prompt,
        source_path=source_path,
        display_name=display_name,
        summary_line=summary_line,
        summary_source=summary_source,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        refs_heading_anchor_number=_refs_heading_anchor_number,
        ref_summary_focus_score=_ref_summary_focus_score,
        build_evidence_backed_ref_summary_from_seed=_build_evidence_backed_ref_summary_from_seed,
        prefer_zh_ref_card_locale=_prefer_zh_ref_card_locale,
        summary_excerpt=_summary_excerpt,
        normalize_ref_copy_text=_normalize_ref_copy_text,
    )


def _resolve_ref_card_why_line(
    *,
    prompt: str,
    display_name: str,
    heading_path: str,
    heading: str,
    section_label: str,
    subsection_label: str,
    nav: dict | None,
    summary_line: str,
) -> dict[str, str]:
    return _card_copy_flow._resolve_ref_card_why_line(
        prompt=prompt,
        display_name=display_name,
        heading_path=heading_path,
        heading=heading,
        section_label=section_label,
        subsection_label=subsection_label,
        nav=nav,
        summary_line=summary_line,
        fallback_why_line_ui=_fallback_why_line_ui,
        build_prompt_aligned_ref_why_line=_build_prompt_aligned_ref_why_line_v3,
        matched_focus_terms_for_ref_card=_matched_focus_terms_for_ref_card,
        is_definition_focus_prompt=_is_definition_focus_prompt,
        why_line_explicitly_names_focus_term=_why_line_explicitly_names_focus_term,
    )


def _resolve_ref_card_summary_kind_and_copy(
    *,
    prompt: str,
    display_name: str,
    heading_path: str,
    heading: str,
    summary_line: str,
    why_line: str,
    why_generation: str,
    citation_meta: dict | None,
    used_prompt_aligned_summary: bool,
    used_nav_summary: bool,
    allow_llm_translate: bool,
) -> dict[str, object]:
    return _card_copy_flow._resolve_ref_card_summary_kind_and_copy(
        prompt=prompt,
        display_name=display_name,
        heading_path=heading_path,
        heading=heading,
        summary_line=summary_line,
        why_line=why_line,
        why_generation=why_generation,
        citation_meta=citation_meta,
        used_prompt_aligned_summary=used_prompt_aligned_summary,
        used_nav_summary=used_nav_summary,
        allow_llm_translate=allow_llm_translate,
        infer_ref_summary_kind=_infer_ref_summary_kind,
        align_ref_card_copy_to_user_locale=_align_ref_card_copy_to_user_locale,
        matched_focus_terms_for_ref_card=_matched_focus_terms_for_ref_card,
        display_focus_term_for_ref_card=_display_focus_term_for_ref_card,
        ref_card_user_locale=_ref_card_user_locale,
        finalize_ref_card_copy=_finalize_ref_card_copy,
        prompt_reference_focus_action=_shared_prompt_reference_focus_action,
    )


def _build_ref_card_basis_bundle(
    *,
    prompt: str,
    citation_meta: dict | None,
    summary_kind: str,
    summary_line: str,
    why_generation: str,
    why_line: str,
) -> dict[str, object]:
    return _card_copy_flow._build_ref_card_basis_bundle(
        prompt=prompt,
        citation_meta=citation_meta,
        summary_kind=summary_kind,
        summary_line=summary_line,
        why_generation=why_generation,
        why_line=why_line,
        build_ref_summary_surface_meta=_build_ref_summary_surface_meta,
        build_ref_summary_basis_meta=_build_ref_summary_basis_meta,
        build_ref_why_basis_meta=_build_ref_why_basis_meta,
    )


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
    summary_selection = _resolve_primary_ref_evidence_summary_selection(
        meta=meta,
        prompt=prompt,
        source_path=source_path,
        display_name=display_name,
        citation_meta=citation_meta,
        heading_path=heading_path,
        heading=heading,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        allow_summary_block_rescue=allow_summary_block_rescue,
        allow_llm_translate=allow_llm_translate,
    )
    candidate_title = str(summary_selection.get("candidate_title") or "").strip()
    nav = (
        dict(summary_selection.get("nav") or {})
        if isinstance(summary_selection.get("nav"), dict)
        else {}
    )
    used_nav_summary = bool(summary_selection.get("used_nav_summary"))
    used_prompt_aligned_summary = bool(summary_selection.get("used_prompt_aligned_summary"))
    summary_line = str(summary_selection.get("summary_line") or "").strip()
    summary_source = str(summary_selection.get("summary_source") or "").strip()
    selected_heading_path = str(summary_selection.get("selected_heading_path") or heading_path).strip()
    prompt_aligned_candidate = (
        dict(summary_selection.get("prompt_aligned_candidate") or {})
        if isinstance(summary_selection.get("prompt_aligned_candidate"), dict)
        else {}
    )

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


def _build_ref_hit_context(
    *,
    hit: dict,
    prompt: str,
    pdf_root: Path | None,
    lib_store: LibraryStore | None,
    preloaded_citation_meta: dict[str, dict] | None = None,
) -> dict[str, object]:
    return _hit_context._build_ref_hit_context(
        hit=hit,
        prompt=prompt,
        pdf_root=pdf_root,
        lib_store=lib_store,
        preloaded_citation_meta=preloaded_citation_meta,
        leading_markdown_heading_from_hit_text=_leading_markdown_heading_from_hit_text,
        refs_section_intent_heading_score=_refs_section_intent_heading_score,
        normalize_title_identity=_normalize_title_identity,
        resolve_ref_ui_heading_context=_resolve_ref_ui_heading_context,
        top_heading=_top_heading,
        safe_page_range=_safe_page_range,
        effective_ui_score=_effective_ui_score,
        positive_int=_positive_int,
        extract_figure_number=extract_figure_number,
        extract_equation_number=extract_equation_number,
        non_negative_float=_non_negative_float,
        build_semantic_badges=_build_semantic_badges,
        resolve_pdf_for_source=_resolve_pdf_for_source,
        display_source_name=_display_source_name,
    )


def _apply_section_intent_rescue_context(
    *,
    meta: dict,
    hit_text: str,
    heading_path: str,
    heading: str,
    section_label: str,
    subsection_label: str,
    summary_line: str,
    summary_source: str,
) -> dict[str, str]:
    return _hit_context._apply_section_intent_rescue_context(
        meta=meta,
        hit_text=hit_text,
        heading_path=heading_path,
        heading=heading,
        section_label=section_label,
        subsection_label=subsection_label,
        summary_line=summary_line,
        summary_source=summary_source,
        top_heading=_top_heading,
        summary_excerpt=_summary_excerpt,
    )


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
    hit_context = _build_ref_hit_context(
        hit=hit,
        prompt=prompt,
        pdf_root=pdf_root,
        lib_store=lib_store,
        preloaded_citation_meta=preloaded_citation_meta,
    )
    meta = dict(hit_context.get("meta") or {}) if isinstance(hit_context.get("meta"), dict) else {}
    source_path = str(hit_context.get("source_path") or "").strip()
    ref_pack_state = str(hit_context.get("ref_pack_state") or "").strip().lower()
    heading_context = (
        dict(hit_context.get("heading_context") or {})
        if isinstance(hit_context.get("heading_context"), dict)
        else {}
    )
    heading_path = str(hit_context.get("heading_path") or "").strip()
    heading = str(hit_context.get("heading") or "").strip()
    section_label = str(hit_context.get("section_label") or "").strip()
    subsection_label = str(hit_context.get("subsection_label") or "").strip()
    p0 = _positive_int(hit_context.get("page_start"))
    p1 = _positive_int(hit_context.get("page_end"))
    score = hit_context.get("score")
    score_pending = bool(hit_context.get("score_pending"))
    anchor_target_kind = str(hit_context.get("anchor_target_kind") or "").strip().lower()
    anchor_target_number = _positive_int(hit_context.get("anchor_target_number"))
    anchor_match_score = _non_negative_float(hit_context.get("anchor_match_score"))
    explicit_doc_match_score = _non_negative_float(hit_context.get("explicit_doc_match_score"))
    semantic_badges = list(hit_context.get("semantic_badges") or [])
    pdf_path = hit_context.get("pdf_path")
    display_name = str(hit_context.get("display_name") or "").strip()
    citation_meta = (
        dict(hit_context.get("citation_meta") or {})
        if isinstance(hit_context.get("citation_meta"), dict)
        else {}
    )

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
    rescue_context = _apply_section_intent_rescue_context(
        meta=meta,
        hit_text=str((hit or {}).get("text") or ""),
        heading_path=heading_path,
        heading=heading,
        section_label=section_label,
        subsection_label=subsection_label,
        summary_line=summary_line,
        summary_source=summary_source,
    )
    heading_path = str(rescue_context.get("heading_path") or "").strip()
    heading = str(rescue_context.get("heading") or "").strip()
    section_label = str(rescue_context.get("section_label") or "").strip()
    subsection_label = str(rescue_context.get("subsection_label") or "").strip()
    summary_line = str(rescue_context.get("summary_line") or "").strip()
    summary_source = str(rescue_context.get("summary_source") or "").strip()
    preferred_exact_candidate = (
        dict(primary_evidence.get("preferred_exact_candidate") or {})
        if isinstance(primary_evidence.get("preferred_exact_candidate"), dict)
        else {}
    )
    why_copy = _resolve_ref_card_why_line(
        prompt=prompt,
        display_name=display_name,
        heading_path=heading_path,
        heading=heading,
        section_label=section_label,
        subsection_label=subsection_label,
        nav=nav,
        summary_line=summary_line,
    )
    why_line = str(why_copy.get("why_line") or "").strip()
    why_generation = str(why_copy.get("why_generation") or "").strip()
    copy_flow = _resolve_ref_card_summary_kind_and_copy(
        prompt=prompt,
        display_name=display_name,
        heading_path=heading_path,
        heading=heading,
        summary_line=summary_line,
        why_line=why_line,
        why_generation=why_generation,
        citation_meta=citation_meta if isinstance(citation_meta, dict) else {},
        used_prompt_aligned_summary=used_prompt_aligned_summary,
        used_nav_summary=used_nav_summary,
        allow_llm_translate=bool(allow_expensive_llm),
    )
    summary_line = str(copy_flow.get("summary_line") or "").strip()
    why_line = str(copy_flow.get("why_line") or "").strip()
    why_generation = str(copy_flow.get("why_generation") or "").strip()
    summary_kind = str(copy_flow.get("summary_kind") or "").strip()
    render_locale = str(copy_flow.get("render_locale") or "").strip()
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
        summary_line, summary_source = _apply_reader_anchor_summary_override(
            reader_open=reader_open,
            prompt=prompt,
            source_path=source_path,
            display_name=display_name,
            summary_line=summary_line,
            summary_source=summary_source,
            anchor_target_kind=anchor_target_kind,
            anchor_target_number=anchor_target_number,
        )
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
    basis_bundle = _build_ref_card_basis_bundle(
        prompt=prompt,
        citation_meta=citation_meta if isinstance(citation_meta, dict) else {},
        summary_kind=summary_kind,
        summary_line=summary_line,
        why_generation=why_generation,
        why_line=why_line,
    )
    summary_surface = (
        dict(basis_bundle.get("summary_surface") or {})
        if isinstance(basis_bundle.get("summary_surface"), dict)
        else {}
    )
    summary_generation = str(basis_bundle.get("summary_generation") or "").strip()
    summary_basis_meta = (
        dict(basis_bundle.get("summary_basis_meta") or {})
        if isinstance(basis_bundle.get("summary_basis_meta"), dict)
        else {}
    )
    why_basis_meta = (
        dict(basis_bundle.get("why_basis_meta") or {})
        if isinstance(basis_bundle.get("why_basis_meta"), dict)
        else {}
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
        render_locale=render_locale,
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
    return _ref_hit_dedupe._refs_hit_ui_meta(hit)


def _refs_hit_meta(hit: dict | None) -> dict:
    return _ref_hit_dedupe._refs_hit_meta(hit)


def _refs_hit_reader_open(hit: dict | None) -> dict:
    return _ref_hit_dedupe._refs_hit_reader_open(hit)


def _refs_norm_key_text(value: str) -> str:
    return _ref_hit_dedupe._refs_norm_key_text(value)


def _refs_hit_source_key(hit: dict | None) -> str:
    return _ref_hit_dedupe._refs_hit_source_key(hit)


def _refs_hit_heading_key(hit: dict | None) -> str:
    return _ref_hit_dedupe._refs_hit_heading_key(hit)


def _refs_hit_locate_key(hit: dict | None) -> str:
    return _ref_hit_dedupe._refs_hit_locate_key(hit)


def _refs_hit_exact_locate_score(hit: dict | None) -> float:
    return _ref_hit_dedupe._refs_hit_exact_locate_score(hit)


def _refs_hit_polish_score(hit: dict | None) -> float:
    return _ref_hit_dedupe._refs_hit_polish_score(hit)


def _refs_hit_evidence_text(hit: dict | None) -> str:
    return _ref_hit_dedupe._refs_hit_evidence_text(hit)


def _refs_dedupe_tokens(text: str) -> set[str]:
    return _ref_hit_dedupe._refs_dedupe_tokens(text)


def _refs_evidence_similarity(left: str, right: str) -> float:
    return _ref_hit_dedupe._refs_evidence_similarity(left, right)


def _refs_evidence_fingerprint(text: str) -> str:
    return _ref_hit_dedupe._refs_evidence_fingerprint(text)


def _refs_hits_are_near_duplicates(left: dict, right: dict) -> bool:
    return _ref_hit_dedupe._refs_hits_are_near_duplicates(left, right)


def _refs_hit_duplicate_rank(*, prompt: str, hit: dict, idx: int) -> tuple[float, float, float, float, float, float, int]:
    return _ref_hit_dedupe._refs_hit_duplicate_rank(
        prompt=prompt,
        hit=hit,
        idx=idx,
        focus_match_count=_refs_hit_focus_match_count,
        section_intent_score=_refs_hit_section_intent_score,
        display_score=_refs_hit_display_score,
    )


def _merge_refs_duplicate_into(keeper: dict, duplicate: dict) -> dict:
    return _ref_hit_dedupe._merge_refs_duplicate_into(keeper, duplicate)


def _dedupe_refs_hits_for_display(*, prompt: str, hits: list[dict]) -> tuple[list[dict], int]:
    return _ref_hit_dedupe._dedupe_refs_hits_for_display(
        prompt=prompt,
        hits=hits,
        focus_match_count=_refs_hit_focus_match_count,
        section_intent_score=_refs_hit_section_intent_score,
        display_score=_refs_hit_display_score,
    )


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
        r"娌℃湁鎻愬埌",
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


def _normalize_title_for_openalex_search(value: str) -> str:
    return _openalex_arxiv_normalize_title_for_search(value)


def _title_similarity_for_openalex(a: str, b: str) -> float:
    return _openalex_arxiv_title_similarity(a, b)


def _openalex_arxiv_meta_by_title(title: str) -> dict:
    return _external_openalex_arxiv_meta_by_title(title)


def _should_try_openalex_arxiv_title(meta: dict, *, raw: str) -> bool:
    return _external_should_try_openalex_arxiv_title(meta, raw=raw)


def _metadata_summary_line(meta: dict) -> str:
    return _external_metadata_summary_line(meta)


def _summary_from_crossref_abstract(meta: dict) -> str:
    return _external_summary_from_crossref_abstract(meta, fetch_crossref_work_by_doi=fetch_crossref_work_by_doi)


def _summary_from_openalex_abstract(meta: dict) -> str:
    return _external_summary_from_openalex_abstract(meta, openalex_work_by_doi=_openalex_work_by_doi)


def _valid_external_abstract_candidate(text: str, *, title: str = "") -> str:
    return _external_valid_external_abstract_candidate(text, title=title)


@lru_cache(maxsize=512)
def _semantic_scholar_paper_by_doi(doi: str) -> dict:
    return _external_semantic_scholar_paper_by_doi(doi)


def _summary_from_semantic_scholar_abstract(meta: dict) -> str:
    return _external_summary_from_semantic_scholar_abstract(
        meta,
        semantic_scholar_paper_by_doi=_semantic_scholar_paper_by_doi,
        title_similarity=_title_similarity_for_openalex,
    )


@lru_cache(maxsize=256)
def _doi_landing_page_abstract(doi: str) -> str:
    return _external_doi_landing_page_abstract(doi)


def _summary_from_doi_landing_page(meta: dict) -> str:
    return _external_summary_from_doi_landing_page(meta, doi_landing_page_abstract=_doi_landing_page_abstract)


def _has_summary_action_signal(text: str) -> bool:
    return _summary_quality_has_summary_action_signal(text)


def _has_summary_result_signal(text: str) -> bool:
    return _summary_quality_has_summary_result_signal(text)


def _is_summary_quality_ok(text: str) -> bool:
    return _summary_quality_is_summary_quality_ok(
        text,
        looks_fragmentary_ref_summary=_looks_fragmentary_ref_summary,
        looks_why_like_ref_summary=_looks_why_like_ref_summary,
    )


def _looks_low_value_shelf_summary(text: str) -> bool:
    return _summary_quality_looks_low_value_shelf_summary(text)


def _looks_metadata_only_summary(text: str) -> bool:
    return _summary_quality_looks_metadata_only_summary(text)


@lru_cache(maxsize=512)
def _llm_summarize_abstract_zh(title: str, abstract_text: str) -> str:
    return _external_llm_summarize_abstract_zh(
        title,
        abstract_text,
        load_settings_func=load_settings,
        chat_cls=DeepSeekChat,
        is_summary_quality_ok=_is_summary_quality_ok,
    )


@lru_cache(maxsize=512)
def _translate_summary_to_zh(text: str) -> str:
    return _external_translate_summary_to_zh(
        text,
        load_settings_func=load_settings,
        chat_cls=DeepSeekChat,
    )


def _summary_quality_contract(meta: dict) -> dict:
    return _external_summary_quality_contract(
        meta,
        is_summary_quality_ok=_is_summary_quality_ok,
        looks_like_title_echo=_looks_like_title_echo,
    )


def _attach_summary_quality(meta: dict) -> dict:
    out = dict(meta or {})
    out["summary_quality"] = _summary_quality_contract(out)
    return out


def _ensure_summary_line(meta: dict, *, allow_crossref_abstract: bool) -> dict:
    return _external_ensure_summary_line(
        meta,
        allow_crossref_abstract=allow_crossref_abstract,
        looks_low_value_shelf_summary=_looks_low_value_shelf_summary,
        looks_like_title_echo=_looks_like_title_echo,
        looks_metadata_only_summary=_looks_metadata_only_summary,
        finalize_abstract_summary_line=_finalize_abstract_summary_line,
        translate_summary_to_zh=_translate_summary_to_zh,
        attach_summary_quality=_attach_summary_quality,
        summary_from_crossref_abstract=_summary_from_crossref_abstract,
        summary_from_openalex_abstract=_summary_from_openalex_abstract,
        summary_from_semantic_scholar_abstract=_summary_from_semantic_scholar_abstract,
        summary_from_doi_landing_page=_summary_from_doi_landing_page,
        contextual_summary_line=_contextual_summary_line,
        metadata_summary_line=_metadata_summary_line,
    )


def _contextual_summary_line(meta: dict) -> str:
    return _external_contextual_summary_line(meta)


def ensure_source_citation_meta(*, source_path: str, pdf_root: Path | None, md_root: Path | None, lib_store: LibraryStore | None) -> dict:
    return _external_ensure_source_citation_meta(
        source_path=source_path,
        pdf_root=pdf_root,
        md_root=md_root,
        lib_store=lib_store,
        resolve_pdf_for_source=_resolve_pdf_for_source,
        has_metrics_payload=_has_metrics_payload,
        parse_filename_meta=_parse_filename_meta,
        source_filename=_source_filename,
        infer_title_from_source_text=_infer_title_from_source_text,
        fetch_crossref_meta=fetch_crossref_meta,
        is_weak_meta_value=_is_weak_meta_value,
        fetch_best_crossref_meta=fetch_best_crossref_meta,
        merge_meta_prefer_richer=_merge_meta_prefer_richer,
        enrich_bibliometrics=_enrich_bibliometrics,
        ensure_summary_line=_ensure_summary_line,
    )


def enrich_citation_detail_meta(detail: dict) -> dict:
    return _detail_pipeline_enrich_citation_detail_meta(
        detail,
        normalize_reference_for_popup=_normalize_reference_for_popup,
        normalize_doi_like=_normalize_doi_like,
        extract_first_doi=extract_first_doi,
        build_doi_url=build_doi_url,
        arxiv_backfill_meta_from_texts=_arxiv_backfill_meta_from_texts,
        fallback_fill_reference_meta_from_raw=_fallback_fill_reference_meta_from_raw,
        merge_meta_prefer_richer=_merge_meta_prefer_richer,
        fetch_best_crossref_meta=fetch_best_crossref_meta,
        fetch_best_crossref_for_reference=fetch_best_crossref_for_reference,
        fetch_crossref_meta=fetch_crossref_meta,
        is_weak_meta_value=_is_weak_meta_value,
        should_try_openalex_arxiv_title=_should_try_openalex_arxiv_title,
        openalex_arxiv_meta_by_title=_openalex_arxiv_meta_by_title,
        enrich_bibliometrics=_enrich_bibliometrics,
        ensure_summary_line=_ensure_summary_line,
    )
