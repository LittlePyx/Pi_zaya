from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path

from kb import runtime_state as RUNTIME
from kb.bg_queue_state import (
    begin_next_task_or_idle as bg_begin_next_task_or_idle,
    cancel_all as bg_cancel_all,
    enqueue as bg_enqueue,
    finish_task as bg_finish_task,
    remove_queued_tasks_for_pdf as bg_remove_queued_tasks_for_pdf,
    should_cancel as bg_should_cancel,
    snapshot as bg_snapshot,
    update_page_progress as bg_update_page_progress,
)
from kb.answer_contract import (
    _answer_contract_enabled,
    _apply_answer_contract_v1,
    _build_answer_quality_probe,
    _build_default_next_steps,
    _build_paper_guide_grounding_rules,
    _detect_answer_depth,
    _detect_answer_intent,
    _detect_answer_output_mode,
    _enhance_kb_miss_fallback,
    _extract_answer_section_keys,
    _extract_cited_sentences,
    _has_sufficient_answer_sections,
    _normalize_answer_section_name,
    _prefer_zh_locale,
    _reconcile_kb_notice,
    _split_kb_miss_notice,
)
from kb.answer_quality import (
    _gen_answer_quality_summary,
    _gen_record_answer_quality,
)
from kb.generation_citation_validation_runtime import (
    _source_refs_from_index as _citation_validation_source_refs_from_index,
    _validate_structured_citations as _citation_validation_validate_structured_citations,
)
from kb.generation_answer_finalize_runtime import (
    _finalize_generation_answer as _finalize_runtime_finalize_generation_answer,
)
from kb.generation_message_runtime import (
    _build_generation_messages as _generation_build_messages,
    _build_multimodal_user_content as _generation_build_multimodal_user_content,
    _filter_history_for_multimodal_turn as _generation_filter_history_for_multimodal_turn,
)
from kb.generation_state_runtime import (
    _gen_get_task as _state_gen_get_task,
    _gen_mark_cancel as _state_gen_mark_cancel,
    _gen_should_cancel as _state_gen_should_cancel,
    _gen_store_answer as _state_gen_store_answer,
    _gen_store_answer_provenance as _state_gen_store_answer_provenance,
    _gen_store_answer_provenance_async as _state_gen_store_answer_provenance_async,
    _gen_store_answer_provenance_fast as _state_gen_store_answer_provenance_fast,
    _gen_store_partial as _state_gen_store_partial,
    _gen_update_task as _state_gen_update_task,
    _is_live_assistant_text as _state_is_live_assistant_text,
    _live_assistant_task_id as _state_live_assistant_task_id,
    _live_assistant_text as _state_live_assistant_text,
    _should_run_provenance_async_refine as _state_should_run_provenance_async_refine,
)
from kb.paper_guide_answer_selection import (
    _build_answer_hits_for_generation as _selection_build_answer_hits_for_generation,
    _has_anchor_grounded_answer_hits as _selection_has_anchor_grounded_answer_hits,
    _paper_guide_answer_hit_score as _selection_answer_hit_score,
    _paper_guide_focus_heading as _selection_focus_heading,
    _select_paper_guide_answer_hits as _selection_select_answer_hits,
    _stabilize_paper_guide_output_mode as _selection_stabilize_output_mode,
)
from kb.paper_guide_answer_post_runtime import (
    _apply_paper_guide_answer_postprocess as _answer_post_apply_paper_guide_answer_postprocess,
)
from kb.chat_store import ChatStore
from kb.file_ops import _resolve_md_output_paths
from kb.llm import DeepSeekChat
from kb.library_figure_runtime import (
    _build_doc_figure_card as _figure_build_doc_figure_card,
    _collect_doc_figure_assets as _figure_collect_doc_figure_assets,
    _maybe_append_library_figure_markdown as _figure_maybe_append_library_figure_markdown,
    _resolve_doc_image_path as _figure_resolve_doc_image_path,
    _score_figure_card_source_binding as _figure_score_doc_figure_source_binding,
)
from kb.inpaper_citation_grounding import (
    extract_candidate_ref_nums_from_hits,
    extract_candidate_ref_cue_texts,
    extract_citation_context_hints,
    has_explicit_reference_conflict,
    parse_ref_num_set,
    reference_alignment_score,
)
from kb.paper_guide_provenance import (
    _PAPER_GUIDE_PROVENANCE_SCHEMA_VERSION,
    _apply_provenance_required_coverage_contract,
    _apply_provenance_strict_identity_contract,
    _best_evidence_quote_match,
    _build_paper_guide_answer_provenance,
    _collect_paper_guide_block_pool,
    _critical_fact_score,
    _dedupe_str_items,
    _ensure_provenance_block_entry,
    _equation_explanation_score,
    _expand_match_snippet_hints,
    _extract_display_formula_snippet,
    _extract_figure_number,
    _extract_json_object_text,
    _extract_quoted_spans,
    _figure_block_number,
    _formula_anchor_text,
    _formula_char_similarity,
    _formula_claim_alignment_score,
    _formula_token_overlap_score,
    _heading_focus_adjustment,
    _is_display_formula_segment,
    _is_explicit_non_source_segment,
    _is_formula_claim_source_grounded,
    _is_generic_heading_path,
    _is_heading_like_quote_span,
    _is_hit_from_bound_source,
    _is_non_source_scope_boundary,
    _is_rhetorical_shell_sentence,
    _longest_quoted_span,
    _normalize_formula_compare_text,
    _normalize_fs_path_for_match,
    _opens_non_source_scope,
    _ordered_fragment_match_score,
    _pick_blocks_with_llm,
    _quote_binding_score,
    _quote_excerpt_fragments,
    _resolve_paper_guide_md_path,
    _segment_claim_meta,
    _segment_focus_tags,
    _segment_snippet_aliases,
    _segment_type_from_text,
    _select_figure_claim_binding,
    _select_quote_claim_binding,
    _source_basename_identity,
    _source_stem_identity,
    _strict_identity_missing_reasons,
    _strip_provenance_noise_text,
    _summary_block_adjustment,
    _summary_segment_tags,
    _text_token_overlap_score,
)
from kb.paper_guide_shared import (
    _CLAIM_EXPERIMENT_HINT_RE as _PG_CLAIM_EXPERIMENT_HINT_RE,
    _CLAIM_METHOD_HINT_RE as _PG_CLAIM_METHOD_HINT_RE,
    _CJK_WORD_RE as _PG_CJK_WORD_RE,
    _CONTRIBUTION_BLOCK_HINT_RE as _PG_CONTRIBUTION_BLOCK_HINT_RE,
    _CONTRIBUTION_LEADIN_HINT_RE as _PG_CONTRIBUTION_LEADIN_HINT_RE,
    _CRITICAL_FACT_HINT_RE as _PG_CRITICAL_FACT_HINT_RE,
    _DEFINITION_LIKE_BLOCK_HINT_RE as _PG_DEFINITION_LIKE_BLOCK_HINT_RE,
    _DISPLAY_EQ_SEG_RE as _PG_DISPLAY_EQ_SEG_RE,
    _EQUATION_EXPLANATION_HINT_RE as _PG_EQUATION_EXPLANATION_HINT_RE,
    _EQUATION_EXPLANATION_PREFIX_RE as _PG_EQUATION_EXPLANATION_PREFIX_RE,
    _EQ_ENV_SEG_RE as _PG_EQ_ENV_SEG_RE,
    _EXPERIMENT_HEADING_HINTS as _PG_EXPERIMENT_HEADING_HINTS,
    _FIGURE_CLAIM_RE as _PG_FIGURE_CLAIM_RE,
    _FIG_NUMBER_PATTERNS as _PG_FIG_NUMBER_PATTERNS,
    _FORMULA_CMD_RE as _PG_FORMULA_CMD_RE,
    _FORMULA_TOKEN_RE as _PG_FORMULA_TOKEN_RE,
    _GENERIC_HEADING_HINTS as _PG_GENERIC_HEADING_HINTS,
    _LATIN_WORD_RE as _PG_LATIN_WORD_RE,
    _METHOD_HEADING_HINTS as _PG_METHOD_HEADING_HINTS,
    _NON_SOURCE_SEGMENT_HINTS as _PG_NON_SOURCE_SEGMENT_HINTS,
    _QUOTE_ELLIPSIS_RE as _PG_QUOTE_ELLIPSIS_RE,
    _QUOTE_HEADING_LIKE_RE as _PG_QUOTE_HEADING_LIKE_RE,
    _QUOTE_PATTERNS as _PG_QUOTE_PATTERNS,
    _RESULT_BLOCK_HINT_RE as _PG_RESULT_BLOCK_HINT_RE,
    _SEG_SENT_SPLIT_RE as _PG_SEG_SENT_SPLIT_RE,
    _SHELL_ONLY_RE as _PG_SHELL_ONLY_RE,
    _SHELL_PREFIX_RE as _PG_SHELL_PREFIX_RE,
    _SUMMARY_NOVELTY_HINT_RE as _PG_SUMMARY_NOVELTY_HINT_RE,
    _SUMMARY_RESULT_HINT_RE as _PG_SUMMARY_RESULT_HINT_RE,
    _cite_source_id as _shared_cite_source_id,
    _extract_paper_guide_abstract_excerpt,
    _source_name_from_md_path as _shared_source_name_from_md_path,
    _trim_paper_guide_prompt_field,
    _trim_paper_guide_prompt_snippet,
)
from kb.pdf_tools import run_pdf_to_md
from kb.paper_guide_postprocess import (
    _sanitize_paper_guide_answer_for_user,
    _sanitize_structured_cite_tokens,
    _strip_model_ref_section,
)
from kb.paper_guide_citation_surfacing import (
    _collect_paper_guide_candidate_refs_by_source as _surfacing_collect_candidate_refs_by_source,
    _drop_paper_guide_locate_only_line_citations as _surfacing_drop_locate_only_line_citations,
    _inject_paper_guide_card_citations as _surfacing_inject_card_citations,
    _inject_paper_guide_fallback_citations as _surfacing_inject_fallback_citations,
    _inject_paper_guide_focus_citations as _surfacing_inject_focus_citations,
    _promote_paper_guide_numeric_reference_citations as _surfacing_promote_numeric_reference_citations,
)
from kb.paper_guide_context_runtime import (
    _apply_paper_guide_deepread_context as _context_apply_deepread_context,
    _build_paper_guide_context_records as _context_build_context_records,
    _prepare_paper_guide_prompt_context as _context_prepare_prompt_context,
)
from kb.paper_guide_direct_answer_runtime import (
    _build_paper_guide_direct_answer_override as _direct_answer_build_override,
)
from kb.paper_guide_message_builder import (
    _build_generation_prompt_bundle as _message_builder_build_generation_prompt_bundle,
)
from kb.paper_guide_retrieval_runtime import (
    _build_paper_guide_direct_citation_lookup_answer as _retrieval_build_direct_citation_lookup_answer,
    _filter_hits_for_paper_guide as _retrieval_filter_hits_for_paper_guide,
    _extract_paper_guide_local_citation_lookup_refs as _retrieval_extract_local_citation_lookup_refs,
    _paper_guide_citation_lookup_fragments as _retrieval_citation_lookup_fragments,
    _paper_guide_citation_lookup_query_tokens as _retrieval_citation_lookup_query_tokens,
    _paper_guide_citation_lookup_signal_score as _retrieval_citation_lookup_signal_score,
    _paper_guide_deepread_heading as _retrieval_deepread_heading,
    _paper_guide_fallback_deepread_hits as _retrieval_fallback_deepread_hits,
    _paper_guide_has_requested_target_hits as _retrieval_has_requested_target_hits,
    _paper_guide_hit_matches_requested_targets as _retrieval_hit_matches_requested_targets,
    _paper_guide_should_force_rescue as _retrieval_should_force_rescue,
    _select_paper_guide_raw_target_hits as _retrieval_select_raw_target_hits,
    _paper_guide_targeted_box_excerpt_hits as _retrieval_targeted_box_excerpt_hits,
    _paper_guide_targeted_source_block_hits as _retrieval_targeted_source_block_hits,
    _select_paper_guide_deepread_extras as _retrieval_select_deepread_extras,
)
from kb.paper_guide_grounding_runtime import (
    _extract_inline_reference_numbers as _grounding_extract_inline_reference_numbers,
    _build_paper_guide_support_slots as _grounding_build_support_slots,
    _build_paper_guide_support_slots_block as _grounding_build_support_slots_block,
    _extract_inline_reference_specs,
    _extract_paper_guide_locate_anchor as _grounding_extract_locate_anchor,
    _extract_paper_guide_ref_spans as _grounding_extract_ref_spans,
    _inject_paper_guide_support_markers as _grounding_inject_support_markers,
    _is_paper_guide_broad_summary_line as _grounding_is_broad_summary_line,
    _is_paper_guide_support_meta_line as _grounding_is_support_meta_line,
    _normalize_paper_guide_support_surface as _grounding_normalize_support_surface,
    _paper_guide_cue_tokens,
    _paper_guide_support_claim_type as _grounding_support_claim_type,
    _paper_guide_support_cite_policy as _grounding_support_cite_policy,
    _paper_guide_support_focus_tokens as _grounding_support_focus_tokens,
    _paper_guide_support_rule_tokens as _grounding_support_rule_tokens,
    _paper_guide_support_segment_spans as _grounding_support_segment_spans,
    _resolve_paper_guide_support_markers as _grounding_resolve_support_markers,
    _resolve_paper_guide_support_ref_num as _grounding_resolve_support_ref_num,
    _resolve_paper_guide_support_slot_block as _grounding_resolve_support_slot_block,
    _select_paper_guide_support_slot_for_context as _grounding_select_support_slot_for_context,
)
from kb.paper_guide_focus import (
    _PAPER_GUIDE_METHOD_DETAIL_RE,
    _PAPER_GUIDE_METHOD_HEADING_TOKENS,
    _PAPER_GUIDE_METHOD_STRONG_DETAIL_RE,
    _build_paper_guide_special_focus_block as _focus_build_special_focus_block,
    _build_paper_guide_direct_abstract_answer as _focus_build_direct_abstract_answer,
    _repair_paper_guide_focus_answer as _focus_repair_answer,
    _repair_paper_guide_focus_answer_generic as _focus_repair_answer_generic,
    _repair_paper_guide_focus_answer_legacy1 as _focus_repair_answer_legacy1,
    _repair_paper_guide_focus_answer_legacy2 as _focus_repair_answer_legacy2,
    _extract_bound_paper_abstract,
    _extract_bound_paper_figure_caption,
    _extract_bound_paper_method_focus,
    _extract_caption_focus_fragment,
    _extract_caption_panel_letters,
    _extract_caption_prompt_fragment,
    _extract_paper_guide_method_detail_excerpt,
    _extract_paper_guide_method_detail_signals,
    _extract_paper_guide_method_focus_terms,
    _extract_paper_guide_special_focus_excerpt,
    _paper_guide_abstract_requests_translation,
    _paper_guide_answer_has_not_stated_shell,
    _paper_guide_method_detail_is_covered,
    _paper_guide_method_detail_strength,
)
from kb.paper_guide_prompting import (
    _augment_paper_guide_retrieval_prompt,
    _build_paper_guide_citation_grounding_block as _prompting_build_citation_grounding_block,
    _build_paper_guide_evidence_cards_block as _prompting_build_evidence_cards_block,
    _looks_like_reference_list_snippet_local,
    _merge_paper_guide_deepread_context as _prompting_merge_deepread_context,
    _paper_guide_allows_citeless_answer,
    _paper_guide_evidence_card_use_hint as _prompting_evidence_card_use_hint,
    _paper_guide_box_header_number,
    _paper_guide_prompt_family,
    _paper_guide_prompt_requests_exact_method_support,
    _paper_guide_requested_heading_hints,
    _requested_figure_number as _prompting_requested_figure_number,
    _paper_guide_text_matches_requested_box,
    _paper_guide_text_matches_requested_section,
    _paper_guide_text_matches_requested_targets,
)
from kb.reference_index import load_reference_index, resolve_reference_entry
from kb.retrieval_engine import (
    _collect_doc_overview_snippets,
    _deep_read_md_for_context,
    _enrich_grouped_refs_with_llm_pack,
    _extract_md_headings,
    _group_hits_by_doc_for_refs,
    _search_hits_with_fallback,
    _top_heading,
)
from kb.retrieval_heuristics import (
    _is_probably_bad_heading,
    _quick_answer_for_prompt,
    _should_bypass_kb_retrieval,
    _should_prioritize_attached_image,
)
from kb.store import load_all_chunks
from kb.retriever import BM25Retriever
from kb.source_blocks import (
    extract_equation_number,
    has_equation_signal,
    load_source_blocks,
    normalize_inline_markdown,
    normalize_match_text,
    split_answer_segments,
)
from ui.chat_widgets import _normalize_math_markdown
from ui.strings import S

_LIVE_ASSISTANT_PREFIX = "__KB_LIVE_TASK__:"
_CITE_SINGLE_BRACKET_RE = re.compile(
    r"(?<!\[)\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\](?!\])",
    re.IGNORECASE,
)
_CITE_SID_ONLY_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*\]\]",
    re.IGNORECASE,
)
_CITE_NON_NUMERIC_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*[A-Za-z0-9_-]{4,24}\s*:\s*(?!\d{1,4}\s*\]\])[^]\n]+\]\]",
    re.IGNORECASE,
)
_CITE_CANON_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\]\]",
    re.IGNORECASE,
)
_DOC_CONTEXT_LABEL_RE = re.compile(
    r"\bDOC-\d{1,3}(?:-S\d{1,3})?(?:\s*(?:,|/|&|and|or)\s*DOC-\d{1,3}(?:-S\d{1,3})?)*\b",
    re.IGNORECASE,
)
_SUPPORT_MARKER_RE = re.compile(
    r"\[\[\s*SUPPORT\s*:\s*(DOC-(\d{1,3})(?:-S(\d{1,3}))?)\s*\]\]",
    re.IGNORECASE,
)
_SID_INLINE_RE = re.compile(r"\[\s*SID\s*:\s*[A-Za-z0-9_-]{4,24}\s*\]", re.IGNORECASE)
_SID_HEADER_LINE_RE = re.compile(
    r"(?im)^\s*(?:\[\d{1,3}\]|DOC-\d{1,3})\s*\[\s*SID\s*:\s*[A-Za-z0-9_-]{4,24}\s*\][^\n]*\n?",
    re.IGNORECASE,
)
_VISION_IMAGE_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}
# D2 compatibility layer: route runtime values to shared paper-guide
# primitives before removing the legacy local literal definitions.
_FIG_NUMBER_PATTERNS = _PG_FIG_NUMBER_PATTERNS
_DISPLAY_EQ_SEG_RE = _PG_DISPLAY_EQ_SEG_RE
_EQ_ENV_SEG_RE = _PG_EQ_ENV_SEG_RE
_LATIN_WORD_RE = _PG_LATIN_WORD_RE
_CJK_WORD_RE = _PG_CJK_WORD_RE
_FORMULA_TOKEN_RE = _PG_FORMULA_TOKEN_RE
_FORMULA_CMD_RE = _PG_FORMULA_CMD_RE
_SEG_SENT_SPLIT_RE = _PG_SEG_SENT_SPLIT_RE
_CLAIM_EXPERIMENT_HINT_RE = _PG_CLAIM_EXPERIMENT_HINT_RE
_CLAIM_METHOD_HINT_RE = _PG_CLAIM_METHOD_HINT_RE
_GENERIC_HEADING_HINTS = _PG_GENERIC_HEADING_HINTS
_EXPERIMENT_HEADING_HINTS = _PG_EXPERIMENT_HEADING_HINTS
_METHOD_HEADING_HINTS = _PG_METHOD_HEADING_HINTS
_QUOTE_PATTERNS = _PG_QUOTE_PATTERNS
_SHELL_ONLY_RE = _PG_SHELL_ONLY_RE
_SHELL_PREFIX_RE = _PG_SHELL_PREFIX_RE
_CRITICAL_FACT_HINT_RE = _PG_CRITICAL_FACT_HINT_RE
_SUMMARY_NOVELTY_HINT_RE = _PG_SUMMARY_NOVELTY_HINT_RE
_SUMMARY_RESULT_HINT_RE = _PG_SUMMARY_RESULT_HINT_RE
_CONTRIBUTION_BLOCK_HINT_RE = _PG_CONTRIBUTION_BLOCK_HINT_RE
_RESULT_BLOCK_HINT_RE = _PG_RESULT_BLOCK_HINT_RE
_CONTRIBUTION_LEADIN_HINT_RE = _PG_CONTRIBUTION_LEADIN_HINT_RE
_DEFINITION_LIKE_BLOCK_HINT_RE = _PG_DEFINITION_LIKE_BLOCK_HINT_RE
_QUOTE_HEADING_LIKE_RE = _PG_QUOTE_HEADING_LIKE_RE
_FIGURE_CLAIM_RE = _PG_FIGURE_CLAIM_RE


def _perf_log(stage: str, **metrics) -> None:
    parts: list[str] = []
    for key, val in metrics.items():
        if isinstance(val, float):
            parts.append(f"{key}={val:.3f}s")
        else:
            parts.append(f"{key}={val}")
    try:
        print("[kb-perf]", stage, " ".join(parts), flush=True)
    except Exception:
        pass


def _warm_refs_citation_meta_background(source_paths: list[str], *, library_db_path: Path | str | None) -> None:
    uniq_paths: list[str] = []
    seen: set[str] = set()
    for src in source_paths or []:
        s = str(src or "").strip()
        if (not s) or (s in seen):
            continue
        seen.add(s)
        uniq_paths.append(s)
        if len(uniq_paths) >= 8:
            break
    if not uniq_paths:
        return

    def _run() -> None:
        try:
            from api.reference_ui import ensure_source_citation_meta
            from api.routers.library import _md_dir, _pdf_dir
            from kb.library_store import LibraryStore
        except Exception:
            return

        try:
            pdf_root = _pdf_dir()
        except Exception:
            pdf_root = None
        try:
            md_root = _md_dir()
        except Exception:
            md_root = None
        try:
            lib_store = LibraryStore(library_db_path) if library_db_path else None
        except Exception:
            lib_store = None

        def _one(src: str) -> None:
            try:
                ensure_source_citation_meta(
                    source_path=src,
                    pdf_root=pdf_root,
                    md_root=md_root,
                    lib_store=lib_store,
                )
            except Exception:
                return

        max_workers = max(1, min(4, len(uniq_paths)))
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_one, src) for src in uniq_paths]
                for fu in as_completed(futs):
                    try:
                        fu.result()
                    except Exception:
                        continue
        except Exception:
            for src in uniq_paths:
                _one(src)

    try:
        threading.Thread(target=_run, daemon=True, name="kb_refs_meta_warm").start()
    except Exception:
        pass


_PAPER_GUIDE_PREFETCH_LOCK = threading.Lock()
_PAPER_GUIDE_PREFETCH_RECENT: dict[str, float] = {}
_PAPER_GUIDE_PREFETCH_TTL_S = 20.0 * 60.0


def kickoff_paper_guide_prefetch(
    *,
    source_path: str,
    source_name: str = "",
    db_dir: Path | str | None = None,
    md_root: Path | str | None = None,
    library_db_path: Path | str | None = None,
) -> bool:
    raw_source = str(source_path or "").strip()
    if not raw_source:
        return False
    md_path = _resolve_paper_guide_md_path(raw_source, md_root=md_root, db_dir=db_dir)
    key = _normalize_fs_path_for_match(str(md_path) if md_path is not None else raw_source)
    if not key:
        return False

    now = time.time()
    with _PAPER_GUIDE_PREFETCH_LOCK:
        prev = float(_PAPER_GUIDE_PREFETCH_RECENT.get(key) or 0.0)
        if (now - prev) < _PAPER_GUIDE_PREFETCH_TTL_S:
            return False
        _PAPER_GUIDE_PREFETCH_RECENT[key] = now
        if len(_PAPER_GUIDE_PREFETCH_RECENT) > 240:
            old_items = sorted(_PAPER_GUIDE_PREFETCH_RECENT.items(), key=lambda item: float(item[1] or 0.0))
            drop_n = len(_PAPER_GUIDE_PREFETCH_RECENT) - 200
            for k_old, _ in old_items[: max(0, drop_n)]:
                _PAPER_GUIDE_PREFETCH_RECENT.pop(k_old, None)

    seed_name = str(source_name or "").strip() or Path(raw_source).name or Path(raw_source).stem

    def _run() -> None:
        t0 = time.perf_counter()
        deep_jobs = 0
        try:
            if md_path is not None:
                _extract_md_headings(md_path, max_n=96)
                _collect_doc_overview_snippets(md_path, max_n=4, snippet_chars=420)

                deep_queries: list[str] = []
                for q in (
                    f"{seed_name} contribution method experiment limitation",
                    f"{seed_name} abstract introduction method",
                    f"{seed_name} experiment setup results",
                    f"{seed_name} limitation failure future work",
                ):
                    qt = str(q or "").strip()
                    if qt and (qt not in deep_queries):
                        deep_queries.append(qt)
                    if len(deep_queries) >= 4:
                        break
                if deep_queries:
                    max_workers = max(1, min(3, len(deep_queries)))
                    try:
                        with ThreadPoolExecutor(max_workers=max_workers) as ex:
                            futs = [
                                ex.submit(
                                    _deep_read_md_for_context,
                                    md_path,
                                    q,
                                    max_snippets=3,
                                    snippet_chars=1200,
                                )
                                for q in deep_queries
                            ]
                            for fu in as_completed(futs):
                                try:
                                    fu.result()
                                except Exception:
                                    continue
                                deep_jobs += 1
                    except Exception:
                        for q in deep_queries:
                            try:
                                _deep_read_md_for_context(md_path, q, max_snippets=3, snippet_chars=1200)
                                deep_jobs += 1
                            except Exception:
                                continue

            warm_paths = [raw_source]
            if md_path is not None:
                warm_paths.insert(0, str(md_path))
            _warm_refs_citation_meta_background(warm_paths, library_db_path=library_db_path)
            _perf_log(
                "paper_guide.prefetch",
                elapsed=time.perf_counter() - t0,
                source=raw_source,
                md=str(md_path or ""),
                deep_jobs=deep_jobs,
            )
        except Exception as exc:
            _perf_log(
                "paper_guide.prefetch",
                elapsed=time.perf_counter() - t0,
                source=raw_source,
                md=str(md_path or ""),
                error=str(exc)[:120],
            )

    try:
        threading.Thread(target=_run, daemon=True, name="kb_paper_guide_prefetch").start()
    except Exception:
        return False
    return True


_DEICTIC_DOC_RE = re.compile(
    r"(\bthis paper\b|\bthat paper\b|\bthis article\b|\bthat article\b|\bin this paper\b|\bin that paper\b|"
    r"\bthe paper\b|\bthe article\b|"
    r"这篇文章|那篇文章|这篇论文|那篇论文|本文|这篇文献|那篇文献|文中|文里|文章里|论文里)",
    flags=re.I,
)
_EXPLICIT_DOC_RE = re.compile(
    r"(\.pdf\b|[A-Za-z]+-\d{4}[-_ ][A-Za-z0-9][A-Za-z0-9 _\-]{8,}|"
    r"[A-Z][A-Za-z0-9&'._\-]+(?: [A-Za-z0-9&'._\-]+){3,})",
    flags=re.I,
)


def _needs_conversational_source_hint(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    if _EXPLICIT_DOC_RE.search(q):
        return False
    return bool(_DEICTIC_DOC_RE.search(q))


def _pick_recent_source_hint(*, conv_id: str, user_msg_id: int, chat_store: ChatStore) -> str:
    cid = str(conv_id or "").strip()
    if not cid:
        return ""
    try:
        refs_by_user = chat_store.list_message_refs(cid) or {}
    except Exception:
        refs_by_user = {}
    items = sorted(
        (
            (int(mid), rec)
            for mid, rec in refs_by_user.items()
            if isinstance(rec, dict) and int(mid or 0) > 0 and int(mid or 0) < int(user_msg_id or 0)
        ),
        key=lambda x: x[0],
        reverse=True,
    )
    for _mid, rec in items:
        hits = rec.get("hits") or []
        if not isinstance(hits, list):
            continue
        for h in hits:
            if not isinstance(h, dict):
                continue
            meta = h.get("meta", {}) or {}
            src = str(meta.get("source_path") or "").strip()
            if not src:
                continue
            p = Path(src)
            cand0 = re.sub(r"\.en\.md$", ".pdf", p.name, flags=re.I)
            cand1 = re.sub(r"\.en$", "", p.stem, flags=re.I)
            for cand in (cand0, cand1, p.name, p.stem):
                s = str(cand or "").strip()
                if s:
                    return s
    return ""


def _augment_prompt_with_source_hint(prompt: str, source_hint: str) -> str:
    q = str(prompt or "").strip()
    hint = str(source_hint or "").strip()
    if (not q) or (not hint):
        return q
    if hint.lower() in q.lower():
        return q
    return f"{hint} {q}".strip()


def _apply_bound_source_hints(prompt: str, source_hints: list[str], *, limit: int = 2) -> str:
    q = str(prompt or "").strip()
    if not q:
        return q
    out = q
    used = 0
    seen: set[str] = set()
    for raw in source_hints or []:
        hint = str(raw or "").strip()
        if not hint:
            continue
        key = hint.lower()
        if key in seen:
            continue
        seen.add(key)
        out = _augment_prompt_with_source_hint(out, hint)
        used += 1
        if used >= max(1, int(limit)):
            break
    return out

def _paper_guide_deepread_heading(hit: dict) -> str:
    return _retrieval_deepread_heading(hit)


def _select_paper_guide_deepread_extras(
    extras: list[dict],
    *,
    prompt: str,
    prompt_family: str = "",
    limit: int = 2,
) -> list[str]:
    return _retrieval_select_deepread_extras(
        extras,
        prompt=prompt,
        prompt_family=prompt_family,
        limit=limit,
    )


def _merge_paper_guide_deepread_context(base: str, extra: str, *, prompt_family: str = "", prompt: str = "") -> str:
    return _prompting_merge_deepread_context(
        base,
        extra,
        prompt_family=prompt_family,
        prompt=prompt,
    )


def _build_paper_guide_context_records(answer_hits: list[dict], *, paper_guide_mode: bool) -> dict:
    return _context_build_context_records(
        answer_hits,
        paper_guide_mode=paper_guide_mode,
    )


def _apply_paper_guide_deepread_context(
    *,
    ctx_parts: list[str],
    doc_first_idx: dict[str, int],
    paper_guide_card_by_doc_idx: dict[int, dict],
    prompt: str,
    retrieval_prompt: str,
    used_query: str,
    prompt_family: str,
    deep_read: bool,
    answer_hits: list[dict],
    should_cancel=None,
    on_stage=None,
) -> dict:
    return _context_apply_deepread_context(
        ctx_parts=ctx_parts,
        doc_first_idx=doc_first_idx,
        paper_guide_card_by_doc_idx=paper_guide_card_by_doc_idx,
        prompt=prompt,
        retrieval_prompt=retrieval_prompt,
        used_query=used_query,
        prompt_family=prompt_family,
        deep_read=deep_read,
        answer_hits=answer_hits,
        should_cancel=should_cancel,
        on_stage=on_stage,
    )


def _prepare_paper_guide_prompt_context(
    *,
    paper_guide_mode: bool,
    paper_guide_bound_source_ready: bool,
    answer_hits: list[dict],
    paper_guide_evidence_cards: list[dict],
    prompt: str,
    retrieval_prompt: str,
    used_query: str,
    prompt_family: str,
    paper_guide_bound_source_path: str,
    db_dir,
) -> dict:
    return _context_prepare_prompt_context(
        paper_guide_mode=paper_guide_mode,
        paper_guide_bound_source_ready=paper_guide_bound_source_ready,
        answer_hits=answer_hits,
        paper_guide_evidence_cards=paper_guide_evidence_cards,
        prompt=prompt,
        retrieval_prompt=retrieval_prompt,
        used_query=used_query,
        prompt_family=prompt_family,
        paper_guide_bound_source_path=paper_guide_bound_source_path,
        db_dir=db_dir,
    )


def _build_generation_prompt_bundle(
    *,
    prompt: str,
    ctx: str,
    paper_guide_mode: bool,
    paper_guide_bound_source_ready: bool,
    paper_guide_prompt_family: str,
    answer_intent: str,
    answer_depth: str,
    answer_output_mode: str,
    answer_contract_v1: bool,
    has_answer_hits: bool,
    locked_citation_source: dict | None,
    image_first_prompt: bool,
    anchor_grounded_answer: bool,
    paper_guide_special_focus_block: str,
    paper_guide_support_slots_block: str,
    paper_guide_evidence_cards_block: str,
    paper_guide_citation_grounding_block: str,
    image_attachment_count: int = 0,
) -> dict:
    return _message_builder_build_generation_prompt_bundle(
        prompt=prompt,
        ctx=ctx,
        paper_guide_mode=paper_guide_mode,
        paper_guide_bound_source_ready=paper_guide_bound_source_ready,
        paper_guide_prompt_family=paper_guide_prompt_family,
        answer_intent=answer_intent,
        answer_depth=answer_depth,
        answer_output_mode=answer_output_mode,
        answer_contract_v1=answer_contract_v1,
        has_answer_hits=has_answer_hits,
        locked_citation_source=locked_citation_source,
        image_first_prompt=image_first_prompt,
        anchor_grounded_answer=anchor_grounded_answer,
        paper_guide_special_focus_block=paper_guide_special_focus_block,
        paper_guide_support_slots_block=paper_guide_support_slots_block,
        paper_guide_evidence_cards_block=paper_guide_evidence_cards_block,
        paper_guide_citation_grounding_block=paper_guide_citation_grounding_block,
        image_attachment_count=image_attachment_count,
    )


def _build_multimodal_user_content(user: str, image_attachments: list[dict] | None) -> str | list[dict]:
    return _generation_build_multimodal_user_content(
        user,
        image_attachments,
        vision_image_mime_by_suffix=_VISION_IMAGE_MIME_BY_SUFFIX,
    )


def _build_generation_messages(*, system: str, hist: list[dict], user_content: str | list[dict]) -> list[dict]:
    return _generation_build_messages(system=system, hist=hist, user_content=user_content)


def _build_paper_guide_direct_answer_override(
    *,
    paper_guide_mode: bool,
    prompt_family: str,
    prompt_for_user: str,
    paper_guide_focus_source_path: str,
    paper_guide_direct_source_path: str,
    paper_guide_bound_source_path: str,
    answer_hits: list[dict] | None,
    special_focus_block: str,
    db_dir,
    llm=None,
) -> str:
    return _direct_answer_build_override(
        paper_guide_mode=paper_guide_mode,
        prompt_family=prompt_family,
        prompt_for_user=prompt_for_user,
        paper_guide_focus_source_path=paper_guide_focus_source_path,
        paper_guide_direct_source_path=paper_guide_direct_source_path,
        paper_guide_bound_source_path=paper_guide_bound_source_path,
        answer_hits=answer_hits,
        special_focus_block=special_focus_block,
        db_dir=db_dir,
        llm=llm,
        build_direct_abstract_answer=_build_paper_guide_direct_abstract_answer,
        build_direct_citation_lookup_answer=_build_paper_guide_direct_citation_lookup_answer,
    )


def _apply_paper_guide_answer_postprocess(
    answer: str,
    *,
    paper_guide_mode: bool,
    prompt: str,
    prompt_for_user: str,
    prompt_family: str,
    special_focus_block: str,
    focus_source_path: str,
    direct_source_path: str,
    bound_source_path: str,
    db_dir: Path | None,
    answer_hits: list[dict],
    support_slots: list[dict],
    cards: list[dict],
    locked_citation_source: dict | None,
) -> tuple[str, list[dict]]:
    return _answer_post_apply_paper_guide_answer_postprocess(
        answer,
        paper_guide_mode=paper_guide_mode,
        prompt=prompt,
        prompt_for_user=prompt_for_user,
        prompt_family=prompt_family,
        special_focus_block=special_focus_block,
        focus_source_path=focus_source_path,
        direct_source_path=direct_source_path,
        bound_source_path=bound_source_path,
        db_dir=db_dir,
        answer_hits=answer_hits,
        support_slots=support_slots,
        cards=cards,
        locked_citation_source=locked_citation_source,
    )


def _finalize_generation_answer(
    partial: str,
    *,
    prompt: str,
    prompt_for_user: str,
    answer_hits: list[dict],
    db_dir: Path | None,
    locked_citation_source: dict | None,
    answer_intent: str,
    answer_depth: str,
    answer_output_mode: str,
    paper_guide_mode: bool,
    paper_guide_contract_enabled: bool,
    paper_guide_prompt_family: str,
    paper_guide_special_focus_block: str,
    paper_guide_focus_source_path: str,
    paper_guide_direct_source_path: str,
    paper_guide_bound_source_path: str,
    paper_guide_candidate_refs_by_source: dict[str, list[int]] | None,
    paper_guide_support_slots: list[dict] | None,
    paper_guide_evidence_cards: list[dict] | None,
) -> dict:
    return _finalize_runtime_finalize_generation_answer(
        partial,
        prompt=prompt,
        prompt_for_user=prompt_for_user,
        answer_hits=answer_hits,
        db_dir=db_dir,
        locked_citation_source=locked_citation_source,
        answer_intent=answer_intent,
        answer_depth=answer_depth,
        answer_output_mode=answer_output_mode,
        paper_guide_mode=paper_guide_mode,
        paper_guide_contract_enabled=paper_guide_contract_enabled,
        paper_guide_prompt_family=paper_guide_prompt_family,
        paper_guide_special_focus_block=paper_guide_special_focus_block,
        paper_guide_focus_source_path=paper_guide_focus_source_path,
        paper_guide_direct_source_path=paper_guide_direct_source_path,
        paper_guide_bound_source_path=paper_guide_bound_source_path,
        paper_guide_candidate_refs_by_source=paper_guide_candidate_refs_by_source,
        paper_guide_support_slots=paper_guide_support_slots,
        paper_guide_evidence_cards=paper_guide_evidence_cards,
        apply_paper_guide_answer_postprocess=_apply_paper_guide_answer_postprocess,
        maybe_append_library_figure_markdown=_maybe_append_library_figure_markdown,
        validate_structured_citations=_validate_structured_citations,
    )


def _stabilize_paper_guide_output_mode(
    output_mode: str,
    *,
    prompt: str,
    intent: str = "",
    explicit_hint: str = "",
) -> str:
    return _selection_stabilize_output_mode(
        output_mode,
        prompt=prompt,
        intent=intent,
        explicit_hint=explicit_hint,
    )


_INPAPER_QUERY_RE = re.compile(
    r"(\bfig(?:ure)?\b|\beq(?:uation)?\b|\bformula\b|\btheorem\b|\blemma\b|\bdefinition\b|\bproposition\b|\bcorollary\b|"
    r"图|公式|定理|引理|定义|命题|推论|这篇|本文|文中|这篇文章|这篇论文)",
    flags=re.I,
)


def _needs_bound_source_hint(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    if re.search(r"(\.pdf\b|[A-Za-z]+-\d{4}[-_ ][A-Za-z0-9])", q, flags=re.I):
        return False
    if _DEICTIC_DOC_RE.search(q):
        return True
    return bool(_INPAPER_QUERY_RE.search(q))


def _pick_recent_bound_source_hints(*, conv_id: str, chat_store: ChatStore, limit: int = 2) -> list[str]:
    cid = str(conv_id or "").strip()
    if not cid:
        return []
    try:
        rows = chat_store.list_conversation_sources(cid, limit=max(1, int(limit)))
    except Exception:
        rows = []
    out: list[str] = []
    seen: set[str] = set()
    for rec in rows or []:
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("source_name") or "").strip()
        src = str(rec.get("source_path") or "").strip()
        cand = name or Path(src).name or Path(src).stem
        if (not cand) or (cand in seen):
            continue
        seen.add(cand)
        out.append(cand)
        if len(out) >= max(1, int(limit)):
            break
    return out

def _filter_hits_for_paper_guide(
    hits_raw: list[dict],
    *,
    bound_source_path: str,
    bound_source_name: str,
) -> list[dict]:
    return _retrieval_filter_hits_for_paper_guide(
        hits_raw,
        bound_source_path=bound_source_path,
        bound_source_name=bound_source_name,
    )


def _paper_guide_hit_matches_requested_targets(hit: dict, *, prompt: str) -> bool:
    return _retrieval_hit_matches_requested_targets(hit, prompt=prompt)


def _paper_guide_has_requested_target_hits(hits_raw: list[dict], *, prompt: str) -> bool:
    return _retrieval_has_requested_target_hits(hits_raw, prompt=prompt)


def _paper_guide_targeted_box_excerpt_hits(
    *,
    md_path: Path,
    bound_source_path: str,
    prompt: str,
    db_dir: Path | str | None = None,
    limit: int = 4,
) -> list[dict]:
    return _retrieval_targeted_box_excerpt_hits(
        md_path=md_path,
        bound_source_path=bound_source_path,
        prompt=prompt,
        db_dir=db_dir,
        limit=limit,
        resolve_support_slot_block=_resolve_paper_guide_support_slot_block,
    )


def _paper_guide_targeted_source_block_hits(
    *,
    bound_source_path: str,
    prompt: str,
    db_dir: Path | str | None = None,
    limit: int = 4,
) -> list[dict]:
    return _retrieval_targeted_source_block_hits(
        bound_source_path=bound_source_path,
        prompt=prompt,
        db_dir=db_dir,
        limit=limit,
        citation_lookup_query_tokens=_paper_guide_citation_lookup_query_tokens,
        citation_lookup_signal_score=_paper_guide_citation_lookup_signal_score,
        resolve_support_slot_block=_resolve_paper_guide_support_slot_block,
    )


def _paper_guide_fallback_deepread_hits(
    *,
    bound_source_path: str,
    bound_source_name: str,
    query: str,
    prompt: str = "",
    prompt_family: str = "",
    top_k: int,
    db_dir: Path | str | None = None,
) -> list[dict]:
    return _retrieval_fallback_deepread_hits(
        bound_source_path=bound_source_path,
        bound_source_name=bound_source_name,
        query=query,
        prompt=prompt,
        prompt_family=prompt_family,
        top_k=top_k,
        db_dir=db_dir,
        citation_lookup_query_tokens=_paper_guide_citation_lookup_query_tokens,
        citation_lookup_signal_score=_paper_guide_citation_lookup_signal_score,
        resolve_support_slot_block=_resolve_paper_guide_support_slot_block,
    )


# Backward-compat for long-lived Streamlit processes that loaded older runtime_state.
if not hasattr(RUNTIME, "BG_LOCK"):
    RUNTIME.BG_LOCK = threading.Lock()
if not hasattr(RUNTIME, "BG_STATE"):
    RUNTIME.BG_STATE = {
        "queue": [],
        "active_tasks": [],
        "active_count": 0,
        "running": False,
        "done": 0,
        "total": 0,
        "current": "",
        "cur_page_done": 0,
        "cur_page_total": 0,
        "cur_page_msg": "",
        "cancel": False,
        "last": "",
    }
if "active_tasks" not in RUNTIME.BG_STATE:
    RUNTIME.BG_STATE["active_tasks"] = []
if "active_count" not in RUNTIME.BG_STATE:
    RUNTIME.BG_STATE["active_count"] = 0
if not hasattr(RUNTIME, "BG_THREADS"):
    RUNTIME.BG_THREADS = []
if not hasattr(RUNTIME, "GEN_QUALITY_EVENTS"):
    RUNTIME.GEN_QUALITY_EVENTS = []

_BG_STATE = RUNTIME.BG_STATE
_BG_LOCK = RUNTIME.BG_LOCK


def _cite_source_id(source_path: str) -> str:
    return _shared_cite_source_id(source_path)

def _live_assistant_text(task_id: str) -> str:
    return _state_live_assistant_text(task_id, live_assistant_prefix=_LIVE_ASSISTANT_PREFIX)

def _is_live_assistant_text(text: str) -> bool:
    return _state_is_live_assistant_text(text, live_assistant_prefix=_LIVE_ASSISTANT_PREFIX)

def _live_assistant_task_id(text: str) -> str:
    return _state_live_assistant_task_id(text, live_assistant_prefix=_LIVE_ASSISTANT_PREFIX)

def _gen_get_task(session_id: str) -> dict | None:
    return _state_gen_get_task(session_id)

def _gen_update_task(session_id: str, task_id: str, **patch) -> None:
    return _state_gen_update_task(session_id, task_id, time_module=time, **patch)

def _gen_should_cancel(session_id: str, task_id: str) -> bool:
    return _state_gen_should_cancel(session_id, task_id)

def _gen_mark_cancel(session_id: str, task_id: str) -> bool:
    return _state_gen_mark_cancel(session_id, task_id, time_module=time)


def _gen_store_answer(task: dict, answer: str) -> None:
    return _state_gen_store_answer(task, answer, chat_store_cls=ChatStore)

def _gen_store_partial(task: dict, partial: str) -> None:
    return _state_gen_store_partial(task, partial, chat_store_cls=ChatStore)


def _gen_store_answer_provenance(
    task: dict,
    *,
    answer: str,
    answer_hits: list[dict],
    support_resolution: list[dict] | None = None,
) -> None:
    return _state_gen_store_answer_provenance(
        task,
        answer=answer,
        answer_hits=answer_hits,
        support_resolution=support_resolution,
        chat_store_cls=ChatStore,
        build_answer_provenance=_build_paper_guide_answer_provenance,
    )


def _gen_store_answer_provenance_fast(
    task: dict,
    *,
    answer: str,
    answer_hits: list[dict],
    support_resolution: list[dict] | None = None,
) -> None:
    return _state_gen_store_answer_provenance_fast(
        task,
        answer=answer,
        answer_hits=answer_hits,
        support_resolution=support_resolution,
        store_answer_provenance=_gen_store_answer_provenance,
    )


def _should_run_provenance_async_refine(task: dict) -> bool:
    return _state_should_run_provenance_async_refine(task, environ=os.environ)


def _gen_store_answer_provenance_async(
    task: dict,
    *,
    answer: str,
    answer_hits: list[dict],
    support_resolution: list[dict] | None = None,
) -> None:
    return _state_gen_store_answer_provenance_async(
        task,
        answer=answer,
        answer_hits=answer_hits,
        support_resolution=support_resolution,
        store_answer_provenance=_gen_store_answer_provenance,
        perf_log=_perf_log,
        threading_module=threading,
        time_module=time,
    )

def _gen_worker(session_id: str, task_id: str) -> None:
    task = _gen_get_task(session_id) or {}
    if str(task.get("id") or "") != str(task_id or ""):
        return

    worker_t0 = time.perf_counter()
    _gen_update_task(session_id, task_id, status="running", stage="starting", started_at=time.time())

    try:
        conv_id = str(task.get("conv_id") or "")
        prompt = str(task.get("prompt") or "").strip()
        raw_image_atts = task.get("image_attachments") or []
        chat_db = Path(str(task.get("chat_db") or "")).expanduser()
        db_dir = Path(str(task.get("db_dir") or "")).expanduser()
        top_k = int(task.get("top_k") or 6)
        temperature = float(task.get("temperature") or 0.15)
        max_tokens = int(task.get("max_tokens") or 1200)
        deep_read = bool(task.get("deep_read"))
        answer_contract_v1 = _answer_contract_enabled(task)
        answer_depth_auto = bool(task.get("answer_depth_auto", True))
        paper_guide_mode = bool(task.get("paper_guide_mode"))
        answer_mode_hint = str(task.get("answer_mode_hint") or "").strip()
        answer_output_mode_hint = str(task.get("answer_output_mode") or task.get("answer_output_mode_hint") or "").strip()
        answer_intent = _detect_answer_intent(prompt, answer_mode_hint=answer_mode_hint)
        answer_depth = _detect_answer_depth(prompt, intent=answer_intent, auto_depth=answer_depth_auto)
        answer_output_mode = _detect_answer_output_mode(
            prompt,
            answer_output_mode_hint=answer_output_mode_hint,
            answer_mode_hint=answer_mode_hint,
            paper_guide_mode=paper_guide_mode,
            intent=answer_intent,
            anchor_grounded=False,
        )
        if paper_guide_mode:
            answer_output_mode = _stabilize_paper_guide_output_mode(
                answer_output_mode,
                prompt=prompt,
                intent=answer_intent,
                explicit_hint=(answer_output_mode_hint or answer_mode_hint),
            )
        llm_rerank = bool(task.get("llm_rerank", True))
        settings_obj = task.get("settings_obj")
        chat_store = ChatStore(chat_db)
        preferred_sources_raw = task.get("preferred_sources") or []
        paper_guide_bound_source_path = str(task.get("paper_guide_bound_source_path") or "").strip()
        paper_guide_bound_source_name = str(task.get("paper_guide_bound_source_name") or "").strip()
        paper_guide_bound_source_ready = bool(task.get("paper_guide_bound_source_ready"))

        image_attachments: list[dict] = []
        if isinstance(raw_image_atts, list):
            for it in raw_image_atts:
                if not isinstance(it, dict):
                    continue
                p0 = Path(str(it.get("path") or "")).expanduser()
                if (not str(p0)) or (not p0.exists()) or (not p0.is_file()):
                    continue
                mime0 = str(it.get("mime") or "").strip().lower()
                if not mime0.startswith("image/"):
                    mime0 = _VISION_IMAGE_MIME_BY_SUFFIX.get(p0.suffix.lower(), "")
                if not mime0.startswith("image/"):
                    continue
                image_attachments.append(
                    {
                        "path": str(p0),
                        "name": str(it.get("name") or p0.name),
                        "mime": mime0,
                        "sha1": str(it.get("sha1") or "").strip().lower(),
                    }
                )
        if len(image_attachments) > 4:
            image_attachments = image_attachments[:4]

        if (not conv_id) or ((not prompt) and (not image_attachments)):
            raise RuntimeError("invalid task")
        if _gen_should_cancel(session_id, task_id):
            raise RuntimeError("canceled")
        if paper_guide_mode and paper_guide_bound_source_ready and paper_guide_bound_source_path:
            try:
                kickoff_paper_guide_prefetch(
                    source_path=paper_guide_bound_source_path,
                    source_name=paper_guide_bound_source_name,
                    db_dir=db_dir,
                    library_db_path=getattr(settings_obj, "library_db_path", None),
                )
            except Exception:
                pass

        quick_answer = _quick_answer_for_prompt(prompt) if prompt else None
        image_first_prompt = bool(image_attachments) and _should_prioritize_attached_image(prompt)
        bypass_kb = bool(prompt) and (_should_bypass_kb_retrieval(prompt) or image_first_prompt)
        if quick_answer is not None:
            try:
                umid0 = int(task.get("user_msg_id") or 0)
            except Exception:
                umid0 = 0
            if umid0 > 0:
                try:
                    chat_store.upsert_message_refs(
                        user_msg_id=umid0,
                        conv_id=conv_id,
                        prompt=prompt,
                        prompt_sig=str(task.get("prompt_sig") or ""),
                        hits=[],
                        scores=[],
                        used_query="",
                        used_translation=False,
                    )
                except Exception:
                    pass
            _gen_store_answer(task, quick_answer)
            try:
                _gen_store_answer_provenance_fast(task, answer=quick_answer, answer_hits=[])
            except Exception as exc:
                _perf_log("gen.provenance_inline_fast", ok=0, err=str(exc)[:120])
            _perf_log("gen.quick_answer", total=time.perf_counter() - worker_t0, conv_id=conv_id)
            _gen_update_task(
                session_id,
                task_id,
                status="done",
                stage="done",
                answer=quick_answer,
                partial=quick_answer,
                char_count=len(quick_answer),
                answer_intent=answer_intent,
                answer_depth=answer_depth,
                answer_output_mode=answer_output_mode,
                answer_contract_v1=bool(answer_contract_v1),
                finished_at=time.time(),
            )
            return

        try:
            cur_user_msg_id = int(task.get("user_msg_id") or 0)
        except Exception:
            cur_user_msg_id = 0
        retrieval_prompt = str(prompt or "").strip()
        preferred_source_hints: list[str] = []
        if isinstance(preferred_sources_raw, list):
            seen_pref: set[str] = set()
            for it in preferred_sources_raw:
                cand = str(it or "").strip()
                if (not cand) or (cand in seen_pref):
                    continue
                seen_pref.add(cand)
                preferred_source_hints.append(cand)
                if len(preferred_source_hints) >= 3:
                    break
        if paper_guide_mode:
            for cand in (paper_guide_bound_source_path, paper_guide_bound_source_name):
                cand_norm = str(cand or "").strip()
                if (not cand_norm) or (cand_norm in preferred_source_hints):
                    continue
                preferred_source_hints.insert(0, cand_norm)
            if len(preferred_source_hints) > 3:
                preferred_source_hints = preferred_source_hints[:3]
        inferred_source_hint = ""
        if paper_guide_mode and paper_guide_bound_source_ready and preferred_source_hints:
            retrieval_prompt = _apply_bound_source_hints(retrieval_prompt, preferred_source_hints, limit=2)
        if retrieval_prompt and _needs_conversational_source_hint(retrieval_prompt):
            inferred_source_hint = _pick_recent_source_hint(
                conv_id=conv_id,
                user_msg_id=cur_user_msg_id,
                chat_store=chat_store,
            )
            if inferred_source_hint:
                retrieval_prompt = _augment_prompt_with_source_hint(retrieval_prompt, inferred_source_hint)
        if retrieval_prompt and _needs_bound_source_hint(retrieval_prompt):
            if preferred_source_hints:
                for h in preferred_source_hints[:2]:
                    retrieval_prompt = _augment_prompt_with_source_hint(retrieval_prompt, h)
            else:
                bound_hints = _pick_recent_bound_source_hints(conv_id=conv_id, chat_store=chat_store, limit=2)
                for h in bound_hints:
                    retrieval_prompt = _augment_prompt_with_source_hint(retrieval_prompt, h)
        paper_guide_prompt_family = ""
        if paper_guide_mode and paper_guide_bound_source_ready:
            paper_guide_prompt_family = _paper_guide_prompt_family(prompt, intent=answer_intent)
            retrieval_prompt = _augment_paper_guide_retrieval_prompt(
                retrieval_prompt,
                family=paper_guide_prompt_family,
                intent=answer_intent,
                output_mode=answer_output_mode,
            )

        t_load0 = time.perf_counter()
        chunks = load_all_chunks(db_dir)
        retriever = BM25Retriever(chunks)
        _perf_log("gen.load_retriever", elapsed=time.perf_counter() - t_load0, chunks=len(chunks))

        hits_raw: list[dict] = []
        scores_raw: list[float] = []
        used_query = ""
        used_translation = False
        hits: list[dict] = []
        grouped_docs: list[dict] = []
        answer_grouped_docs: list[dict] = []
        refs_async_will_run = False
        refs_async_seed_docs: list[dict] = []
        if prompt and (not bypass_kb):
            _gen_update_task(session_id, task_id, stage="retrieve")
            t_ret0 = time.perf_counter()
            hits_raw, scores_raw, used_query, used_translation = _search_hits_with_fallback(
                retrieval_prompt,
                retriever,
                top_k=top_k,
                settings=settings_obj,
            )
            if paper_guide_mode and paper_guide_bound_source_ready and (paper_guide_bound_source_path or paper_guide_bound_source_name):
                scoped_hits = _filter_hits_for_paper_guide(
                    hits_raw,
                    bound_source_path=paper_guide_bound_source_path,
                    bound_source_name=paper_guide_bound_source_name,
                )
                prompt_targeted = bool(_paper_guide_requested_heading_hints(prompt or retrieval_prompt or ""))
                method_exact_support_targeted = bool(
                    paper_guide_prompt_family in {"method", "reproduce"}
                    and _paper_guide_prompt_requests_exact_method_support(prompt or retrieval_prompt or "")
                )
                prompt_targeted = prompt_targeted or method_exact_support_targeted
                citation_lookup_targeted = paper_guide_prompt_family == "citation_lookup"
                scoped_has_target = _paper_guide_has_requested_target_hits(
                    scoped_hits,
                    prompt=(prompt or retrieval_prompt or ""),
                )
                explicit_target_hits: list[dict] = []
                if (prompt_targeted or citation_lookup_targeted) and paper_guide_bound_source_path:
                    explicit_target_hits = _paper_guide_targeted_source_block_hits(
                        bound_source_path=paper_guide_bound_source_path,
                        prompt=(prompt or retrieval_prompt or ""),
                        db_dir=db_dir,
                        limit=max(2, min(int(top_k or 4), 6)),
                    )
                    if explicit_target_hits:
                        merged_hits: list[dict] = []
                        seen_keys: set[str] = set()
                        for h in list(explicit_target_hits) + list(scoped_hits):
                            if not isinstance(h, dict):
                                continue
                            key = hashlib.sha1(
                                (
                                    str((h.get("meta", {}) or {}).get("block_id") or "")
                                    + "\n"
                                    + str((h.get("meta", {}) or {}).get("heading_path") or "")
                                    + "\n"
                                    + str(h.get("text") or "")
                                ).encode("utf-8", "ignore")
                            ).hexdigest()[:16]
                            if key in seen_keys:
                                continue
                            seen_keys.add(key)
                            merged_hits.append(h)
                        scoped_hits = merged_hits
                        scoped_has_target = True

                # Recall hardening: even when BM25 has hits, it can still miss the relevant section.
                # Add a small, deterministic SourceBlock scan as a supplement (using translated query when available).
                # Keep this limited to avoid adding too much noise/latency on every request.
                try:
                    supplemental_enabled = bool(int(str(os.environ.get("KB_PAPER_GUIDE_SUPPLEMENTAL_SCAN", "1") or "1")))
                except Exception:
                    supplemental_enabled = True
                should_supplement = bool(
                    supplemental_enabled
                    and paper_guide_bound_source_path
                    and paper_guide_prompt_family not in {"abstract"}
                    and (
                        prompt_targeted
                        or citation_lookup_targeted
                        or method_exact_support_targeted
                        or (len(scoped_hits) < max(10, int(top_k or 4) * 3))
                        or paper_guide_prompt_family in {"method", "figure_walkthrough", "reproduce", "compare", "strength_limits"}
                    )
                )
                if should_supplement:
                    scan_prompt = (used_query or "").strip() if bool(used_translation) else ""
                    if not scan_prompt:
                        scan_prompt = str(prompt or retrieval_prompt or "").strip()
                    supplemental_hits = _paper_guide_targeted_source_block_hits(
                        bound_source_path=paper_guide_bound_source_path,
                        prompt=scan_prompt,
                        db_dir=db_dir,
                        limit=max(2, min(int(top_k or 4), 4)),
                    )
                    if supplemental_hits:
                        merged_hits: list[dict] = []
                        seen_keys: set[str] = set()
                        for h in list(supplemental_hits) + list(scoped_hits):
                            if not isinstance(h, dict):
                                continue
                            key = hashlib.sha1(
                                (
                                    str((h.get("meta", {}) or {}).get("block_id") or "")
                                    + "\n"
                                    + str((h.get("meta", {}) or {}).get("heading_path") or "")
                                    + "\n"
                                    + str(h.get("text") or "")
                                ).encode("utf-8", "ignore")
                            ).hexdigest()[:16]
                            if key in seen_keys:
                                continue
                            seen_keys.add(key)
                            merged_hits.append(h)
                        scoped_hits = merged_hits
                should_force_rescue = _retrieval_should_force_rescue(
                    scoped_hits=scoped_hits,
                    prompt=(prompt or retrieval_prompt or ""),
                    prompt_family=paper_guide_prompt_family,
                )
                if should_force_rescue and paper_guide_bound_source_path:
                    fallback_hits = _paper_guide_fallback_deepread_hits(
                        bound_source_path=paper_guide_bound_source_path,
                        bound_source_name=paper_guide_bound_source_name,
                        query=(used_query or retrieval_prompt or prompt or ""),
                        prompt=(prompt or retrieval_prompt or ""),
                        prompt_family=paper_guide_prompt_family,
                        top_k=max(2, min(int(top_k or 4), 4)),
                        db_dir=db_dir,
                    )
                    if fallback_hits:
                        if scoped_hits:
                            merged_hits: list[dict] = []
                            seen_keys: set[str] = set()
                            for h in list(fallback_hits) + list(scoped_hits):
                                if not isinstance(h, dict):
                                    continue
                                key = hashlib.sha1(
                                    (
                                        str((h.get("meta", {}) or {}).get("heading_path") or "")
                                        + "\n"
                                        + str(h.get("text") or "")
                                    ).encode("utf-8", "ignore")
                                ).hexdigest()[:16]
                                if key in seen_keys:
                                    continue
                                seen_keys.add(key)
                                merged_hits.append(h)
                            scoped_hits = merged_hits
                        else:
                            scoped_hits = list(fallback_hits)
                        _perf_log(
                            "gen.paper_guide_scope_fallback",
                            docs=len(fallback_hits),
                            source=paper_guide_bound_source_name or paper_guide_bound_source_path,
                            target_miss=int(prompt_targeted and (not scoped_has_target)),
                            forced=int(should_force_rescue),
                        )
                if len(scoped_hits) != len(hits_raw):
                    _perf_log(
                        "gen.paper_guide_scope",
                        before=len(hits_raw),
                        after=len(scoped_hits),
                        source=paper_guide_bound_source_name or paper_guide_bound_source_path,
                    )
                hits_raw = scoped_hits
                scores_raw = [float(h.get("score", 0.0) or 0.0) for h in hits_raw]
            _perf_log(
                "gen.retrieve",
                elapsed=time.perf_counter() - t_ret0,
                hits_raw=len(hits_raw),
                translated=bool(used_translation),
            )
            hits = _group_hits_by_top_heading(hits_raw, top_k=top_k)

            _gen_update_task(session_id, task_id, stage="refs")
            if not getattr(retriever, "is_empty", False):
                try:
                    t_seed0 = time.perf_counter()
                    grouped_docs = _group_hits_by_doc_for_refs(
                        hits_raw,
                        prompt_text=retrieval_prompt,
                        top_k_docs=top_k,
                        deep_query=(used_query or retrieval_prompt or prompt or ""),
                        deep_read=False,  # fast seed first; deep-read is moved to async refs enrichment
                        llm_rerank=False,
                        settings=settings_obj,
                    )
                    _perf_log("gen.refs_seed", elapsed=time.perf_counter() - t_seed0, docs=len(grouped_docs))
                except Exception:
                    grouped_docs = []
            answer_grouped_docs = list(grouped_docs or [])
            answer_hit_limit = max(1, min(int(top_k), 4))
            guide_strict_mode = bool(paper_guide_mode and paper_guide_bound_source_ready)
            answer_doc_cap = max(
                answer_hit_limit,
                min(
                    int(top_k),
                    5 if (guide_strict_mode and paper_guide_prompt_family in {"overview", "compare", "reproduce", "strength_limits", "figure_walkthrough"}) else (4 if guide_strict_mode else 3),
                ),
            )
            should_sync_deep_seed = bool(hits_raw) and (
                guide_strict_mode
                or _needs_bound_source_hint(prompt or retrieval_prompt or "")
            )
            if should_sync_deep_seed:
                try:
                    t_answer_seed0 = time.perf_counter()
                    rebuilt_for_answer = _group_hits_by_doc_for_refs(
                        hits_raw,
                        prompt_text=retrieval_prompt,
                        top_k_docs=answer_doc_cap,
                        deep_query=(used_query or retrieval_prompt or prompt or ""),
                        deep_read=True,
                        llm_rerank=False,
                        settings=settings_obj,
                    )
                    if rebuilt_for_answer:
                        answer_grouped_docs = rebuilt_for_answer
                    _perf_log(
                        "gen.answer_refs_seed",
                        elapsed=time.perf_counter() - t_answer_seed0,
                        docs=len(answer_grouped_docs),
                    )
                except Exception:
                    pass
            else:
                _perf_log("gen.answer_refs_seed", elapsed=0.0, docs=len(answer_grouped_docs), mode="fast_only")
            # Keep answer path focused on evidence readiness. LLM ref-pack enrichment is
            # deferred to async so it does not block first answer latency.
            _perf_log("gen.answer_refs_enrich", elapsed=0.0, docs=len(answer_grouped_docs), mode="async_only")

            try:
                refs_async_enabled = bool(int(str(os.environ.get("KB_REFS_ASYNC_ENRICH", "1") or "1")))
            except Exception:
                refs_async_enabled = True
            try:
                refs_async_in_paper_guide = bool(int(str(os.environ.get("KB_REFS_ASYNC_ENRICH_IN_PAPER_GUIDE", "0") or "0")))
            except Exception:
                refs_async_in_paper_guide = False
            allow_refs_async = bool(refs_async_enabled and ((not paper_guide_mode) or refs_async_in_paper_guide))

            refs_async_will_run = bool(
                allow_refs_async
                and llm_rerank
                and prompt
                and grouped_docs
                and settings_obj
                and getattr(settings_obj, "api_key", None)
            )
            if refs_async_will_run and grouped_docs:
                try:
                    for d in grouped_docs:
                        if not isinstance(d, dict):
                            continue
                        meta_d = d.get("meta", {}) or {}
                        if not isinstance(meta_d, dict):
                            meta_d = {}
                        meta_d["ref_pack_state"] = "pending"
                        d["meta"] = meta_d
                except Exception:
                    pass
                try:
                    refs_async_seed_docs = copy.deepcopy(grouped_docs)
                except Exception:
                    refs_async_seed_docs = list(grouped_docs)
        else:
            _gen_update_task(
                session_id,
                task_id,
                stage=(
                    "retrieve skipped (image-first prompt)"
                    if image_first_prompt
                    else ("retrieve skipped (general coding request)" if bypass_kb else "retrieve (image-only)")
                ),
            )

        try:
            umid = int(task.get("user_msg_id") or 0)
        except Exception:
            umid = 0
        if umid > 0:
            try:
                chat_store.upsert_message_refs(
                    user_msg_id=umid,
                    conv_id=conv_id,
                    prompt=prompt,
                    prompt_sig=str(task.get("prompt_sig") or ""),
                    hits=list(grouped_docs or []),
                    scores=list(scores_raw or []),
                    used_query=str(used_query or ""),
                    used_translation=bool(used_translation),
                )
            except Exception:
                pass

        def _finalize_task_after_refs_async() -> None:
            snap = _gen_get_task(session_id) or {}
            if str(snap.get("id") or "") != str(task_id or ""):
                return
            if (str(snap.get("status") or "") == "running") and bool(snap.get("answer_ready") or False):
                ans = str(snap.get("answer") or snap.get("partial") or "").strip()
                _gen_update_task(
                    session_id,
                    task_id,
                    status="done",
                    stage="done",
                    answer=ans,
                    partial=ans,
                    char_count=len(ans),
                    finished_at=time.time(),
                )

        if refs_async_will_run and (umid > 0) and refs_async_seed_docs:
            _gen_update_task(session_id, task_id, refs_async_pending=True, refs_async_state="running")

            def _bg_enrich_refs() -> None:
                try:
                    refs_async_top_k_docs = int(str(os.environ.get("KB_REFS_ASYNC_TOP_K", "3") or "3"))
                except Exception:
                    refs_async_top_k_docs = 3
                refs_async_top_k_docs = max(1, min(max(1, int(top_k or 1)), refs_async_top_k_docs))

                try:
                    refs_async_timeout_s = float(str(os.environ.get("KB_REFS_ASYNC_TIMEOUT_S", "12") or "12"))
                except Exception:
                    refs_async_timeout_s = 12.0
                refs_async_timeout_s = max(4.0, min(30.0, refs_async_timeout_s))

                try:
                    refs_async_max_retries = int(str(os.environ.get("KB_REFS_ASYNC_MAX_RETRIES", "0") or "0"))
                except Exception:
                    refs_async_max_retries = 0
                refs_async_max_retries = max(0, min(1, refs_async_max_retries))

                settings_for_refs = settings_obj
                try:
                    settings_for_refs = replace(
                        settings_obj,
                        timeout_s=min(float(getattr(settings_obj, "timeout_s", refs_async_timeout_s) or refs_async_timeout_s), refs_async_timeout_s),
                        max_retries=refs_async_max_retries,
                    )
                except Exception:
                    settings_for_refs = settings_obj

                def _push_partial(partial_docs: list[dict]) -> None:
                    try:
                        cs = ChatStore(chat_db)
                        cs.upsert_message_refs(
                            user_msg_id=umid,
                            conv_id=conv_id,
                            prompt=prompt,
                            prompt_sig=str(task.get("prompt_sig") or ""),
                            hits=list(partial_docs or []),
                            scores=list(scores_raw or []),
                            used_query=str(used_query or ""),
                            used_translation=bool(used_translation),
                        )
                    except Exception:
                        pass

                seed_docs = list(refs_async_seed_docs)[:refs_async_top_k_docs]
                try:
                    if hits_raw:
                        t_rebuild0 = time.perf_counter()
                        rebuilt_docs = _group_hits_by_doc_for_refs(
                            hits_raw,
                            prompt_text=retrieval_prompt,
                            top_k_docs=refs_async_top_k_docs,
                            deep_query=(used_query or retrieval_prompt or prompt or ""),
                            deep_read=True,
                            llm_rerank=False,
                            settings=settings_obj,
                        )
                        if rebuilt_docs:
                            seed_docs = rebuilt_docs
                            for d in seed_docs:
                                if not isinstance(d, dict):
                                    continue
                                meta_d = d.get("meta", {}) or {}
                                if not isinstance(meta_d, dict):
                                    meta_d = {}
                                meta_d["ref_pack_state"] = "pending"
                                d["meta"] = meta_d
                            _push_partial(seed_docs)
                        _perf_log("gen.refs_rebuild", elapsed=time.perf_counter() - t_rebuild0, docs=len(seed_docs))
                except Exception:
                    seed_docs = list(refs_async_seed_docs)

                try:
                    t_pack0 = time.perf_counter()
                    enriched = _enrich_grouped_refs_with_llm_pack(
                        list(seed_docs),
                        question=(prompt or used_query or ""),
                        settings=settings_for_refs,
                        top_k_docs=refs_async_top_k_docs,
                        progress_cb=_push_partial,
                    )
                    _perf_log(
                        "gen.refs_enrich",
                        elapsed=time.perf_counter() - t_pack0,
                        docs=len(enriched),
                        top_k=refs_async_top_k_docs,
                        timeout=refs_async_timeout_s,
                        retries=refs_async_max_retries,
                    )
                except Exception:
                    enriched = []
                snap0 = _gen_get_task(session_id) or {}
                same_task = str(snap0.get("id") or "") == str(task_id or "")
                answer_ready0 = bool(snap0.get("answer_ready") or False)
                # If another task has already replaced this session slot, still allow refs
                # enrichment to be persisted for the original answered message.
                if same_task and _gen_should_cancel(session_id, task_id) and (not answer_ready0):
                    _gen_update_task(session_id, task_id, refs_async_pending=False, refs_async_state="canceled")
                    return
                if enriched:
                    try:
                        cs = ChatStore(chat_db)
                        cs.upsert_message_refs(
                            user_msg_id=umid,
                            conv_id=conv_id,
                            prompt=prompt,
                            prompt_sig=str(task.get("prompt_sig") or ""),
                            hits=list(enriched),
                            scores=list(scores_raw or []),
                            used_query=str(used_query or ""),
                            used_translation=bool(used_translation),
                        )
                    except Exception:
                        pass
                    _gen_update_task(session_id, task_id, refs_async_pending=False, refs_async_state="done", refs_async_docs=int(len(enriched)))
                    try:
                        warm_paths: list[str] = []
                        for d in list(enriched or []):
                            if not isinstance(d, dict):
                                continue
                            meta_d = d.get("meta", {}) or {}
                            src = str(meta_d.get("source_path") or "").strip()
                            if src:
                                warm_paths.append(src)
                        _warm_refs_citation_meta_background(
                            warm_paths,
                            library_db_path=getattr(settings_obj, "library_db_path", None),
                        )
                    except Exception:
                        pass
                else:
                    _gen_update_task(session_id, task_id, refs_async_pending=False, refs_async_state="empty")
                _finalize_task_after_refs_async()

            try:
                threading.Thread(target=_bg_enrich_refs, daemon=True).start()
            except Exception:
                _gen_update_task(session_id, task_id, refs_async_pending=False, refs_async_state="error")
                _finalize_task_after_refs_async()

        _gen_update_task(session_id, task_id, stage="context", used_query=str(used_query or ""), used_translation=bool(used_translation), refs_done=True)

        # Keep prompt compact for fast first-token latency.
        answer_hit_limit = max(
            1,
            min(
                int(top_k),
                5 if (paper_guide_mode and paper_guide_prompt_family in {"overview", "compare", "reproduce", "strength_limits", "figure_walkthrough", "citation_lookup"}) else 4,
            ),
        )
        answer_seed = answer_grouped_docs or grouped_docs or hits
        if paper_guide_mode and paper_guide_bound_source_ready:
            heading_hits_for_answer = list(hits or [])
            grouped_hits_for_answer = list(answer_seed or [])
            raw_target_hits = _select_paper_guide_raw_target_hits(
                hits_raw=list(hits_raw or []),
                prompt=(prompt or retrieval_prompt or ""),
                top_n=answer_hit_limit,
            )
            if raw_target_hits:
                heading_hits_for_answer = raw_target_hits
                if paper_guide_prompt_family == "citation_lookup":
                    grouped_hits_for_answer = list(raw_target_hits)
            answer_hits = _select_paper_guide_answer_hits(
                grouped_docs=grouped_hits_for_answer,
                heading_hits=heading_hits_for_answer,
                prompt=prompt,
                top_n=answer_hit_limit,
            )
        else:
            answer_hits = _build_answer_hits_for_generation(
                grouped_docs=list(answer_seed or []),
                heading_hits=list(hits or []),
                top_n=answer_hit_limit,
            )
        anchor_grounded_answer = _has_anchor_grounded_answer_hits(answer_hits)
        answer_output_mode = _detect_answer_output_mode(
            prompt,
            answer_output_mode_hint=answer_output_mode_hint,
            answer_mode_hint=answer_mode_hint,
            paper_guide_mode=paper_guide_mode,
            intent=answer_intent,
            anchor_grounded=anchor_grounded_answer,
        )
        if paper_guide_mode:
            answer_output_mode = _stabilize_paper_guide_output_mode(
                answer_output_mode,
                prompt=prompt,
                intent=answer_intent,
                explicit_hint=(answer_output_mode_hint or answer_mode_hint),
            )
        locked_citation_source = _pick_locked_citation_source(list(answer_seed or answer_hits))
        answer_hits = _ensure_locked_source_in_answer_hits(
            answer_hits,
            source_rec=locked_citation_source,
            seed_docs=list(answer_seed or []),
            top_n=answer_hit_limit,
        )
        paper_guide_context_records = _build_paper_guide_context_records(
            answer_hits,
            paper_guide_mode=paper_guide_mode,
        )
        ctx_parts = list(paper_guide_context_records.get("ctx_parts") or [])
        doc_first_idx = dict(paper_guide_context_records.get("doc_first_idx") or {})
        paper_guide_evidence_cards = list(paper_guide_context_records.get("paper_guide_evidence_cards") or [])
        paper_guide_card_by_doc_idx = dict(paper_guide_context_records.get("paper_guide_card_by_doc_idx") or {})

        deep_added = 0
        deep_docs = 0
        if deep_read and answer_hits:
            deep_begin = time.monotonic()
            deepread_state = _apply_paper_guide_deepread_context(
                ctx_parts=ctx_parts,
                doc_first_idx=doc_first_idx,
                paper_guide_card_by_doc_idx=paper_guide_card_by_doc_idx,
                prompt=prompt,
                retrieval_prompt=retrieval_prompt,
                used_query=used_query,
                prompt_family=paper_guide_prompt_family,
                deep_read=deep_read,
                answer_hits=answer_hits,
                should_cancel=lambda: _gen_should_cancel(session_id, task_id),
                on_stage=lambda stage: _gen_update_task(session_id, task_id, stage=stage),
            )
            ctx_parts = list(deepread_state.get("ctx_parts") or ctx_parts)
            deep_added = int(deepread_state.get("deep_added") or 0)
            deep_docs = int(deepread_state.get("deep_docs") or 0)
            _perf_log("gen.deep_read", elapsed=time.monotonic() - deep_begin, docs=deep_docs, added=deep_added)

        _gen_update_task(
            session_id,
            task_id,
            deep_read_docs=int(deep_docs),
            deep_read_added=int(deep_added),
            answer_intent=answer_intent,
            answer_depth=answer_depth,
            answer_output_mode=answer_output_mode,
            answer_contract_v1=bool(answer_contract_v1),
            citation_locked_sid=str((locked_citation_source or {}).get("sid") or ""),
            stage="answer",
        )
        ctx = "\n\n---\n\n".join(ctx_parts)
        paper_guide_prompt_context = _prepare_paper_guide_prompt_context(
            paper_guide_mode=paper_guide_mode,
            paper_guide_bound_source_ready=paper_guide_bound_source_ready,
            answer_hits=answer_hits,
            paper_guide_evidence_cards=paper_guide_evidence_cards,
            prompt=prompt,
            retrieval_prompt=retrieval_prompt,
            used_query=used_query,
            prompt_family=paper_guide_prompt_family,
            paper_guide_bound_source_path=paper_guide_bound_source_path,
            db_dir=db_dir,
        )
        paper_guide_evidence_cards_block = str(paper_guide_prompt_context.get("paper_guide_evidence_cards_block") or "")
        paper_guide_support_slots_block = str(paper_guide_prompt_context.get("paper_guide_support_slots_block") or "")
        paper_guide_special_focus_block = str(paper_guide_prompt_context.get("paper_guide_special_focus_block") or "")
        paper_guide_citation_grounding_block = str(paper_guide_prompt_context.get("paper_guide_citation_grounding_block") or "")
        paper_guide_candidate_refs_by_source = dict(paper_guide_prompt_context.get("paper_guide_candidate_refs_by_source") or {})
        paper_guide_support_slots = list(paper_guide_prompt_context.get("paper_guide_support_slots") or [])
        paper_guide_support_resolution: list[dict] = []
        paper_guide_direct_source_path = str(paper_guide_prompt_context.get("paper_guide_direct_source_path") or paper_guide_bound_source_path or "")
        paper_guide_focus_source_path = str(paper_guide_prompt_context.get("paper_guide_focus_source_path") or paper_guide_bound_source_path or "")
        prompt_bundle = _build_generation_prompt_bundle(
            prompt=prompt,
            ctx=ctx,
            paper_guide_mode=paper_guide_mode,
            paper_guide_bound_source_ready=paper_guide_bound_source_ready,
            paper_guide_prompt_family=paper_guide_prompt_family,
            answer_intent=answer_intent,
            answer_depth=answer_depth,
            answer_output_mode=answer_output_mode,
            answer_contract_v1=bool(answer_contract_v1),
            has_answer_hits=bool(answer_hits),
            locked_citation_source=locked_citation_source,
            image_first_prompt=image_first_prompt,
            anchor_grounded_answer=anchor_grounded_answer,
            paper_guide_special_focus_block=paper_guide_special_focus_block,
            paper_guide_support_slots_block=paper_guide_support_slots_block,
            paper_guide_evidence_cards_block=paper_guide_evidence_cards_block,
            paper_guide_citation_grounding_block=paper_guide_citation_grounding_block,
            image_attachment_count=len(image_attachments or []),
        )
        system = str(prompt_bundle.get("system") or "")
        user = str(prompt_bundle.get("user") or "")
        prompt_for_user = str(prompt_bundle.get("prompt_for_user") or prompt or "[Image attachment only request]")
        paper_guide_contract_enabled = bool(prompt_bundle.get("paper_guide_contract_enabled"))
        history = chat_store.get_messages(conv_id)
        try:
            cur_user_msg_id = int(task.get("user_msg_id") or 0)
        except Exception:
            cur_user_msg_id = 0
        try:
            cur_assistant_msg_id = int(task.get("assistant_msg_id") or 0)
        except Exception:
            cur_assistant_msg_id = 0

        hist = _filter_history_for_multimodal_turn(
            history,
            cur_user_msg_id=cur_user_msg_id,
            cur_assistant_msg_id=cur_assistant_msg_id,
            has_current_images=bool(image_attachments),
        )
        hist = hist[-10:]
        user_content = _build_multimodal_user_content(user, image_attachments)
        messages = _build_generation_messages(system=system, hist=hist, user_content=user_content)
        ds = DeepSeekChat(settings_obj)
        direct_answer_override = _build_paper_guide_direct_answer_override(
            paper_guide_mode=paper_guide_mode,
            prompt_family=paper_guide_prompt_family,
            prompt_for_user=prompt_for_user,
            paper_guide_focus_source_path=paper_guide_focus_source_path,
            paper_guide_direct_source_path=paper_guide_direct_source_path,
            paper_guide_bound_source_path=paper_guide_bound_source_path,
            answer_hits=answer_hits,
            special_focus_block=paper_guide_special_focus_block,
            db_dir=db_dir,
            llm=ds,
        )
        partial = ""
        streamed = False
        last_store_ts = 0.0
        last_store_len = 0
        t_answer0 = time.perf_counter()
        if direct_answer_override:
            partial = str(direct_answer_override or "").strip()
            _gen_update_task(session_id, task_id, stage="answer", partial=partial, char_count=len(partial))
            _gen_store_partial(task, partial)
        else:
            try:
                for piece in ds.chat_stream(messages=messages, temperature=temperature, max_tokens=max_tokens):
                    if _gen_should_cancel(session_id, task_id):
                        raise RuntimeError("canceled")
                    partial += piece
                    streamed = True
                    _gen_update_task(session_id, task_id, stage="answer", partial=partial, char_count=len(partial))
                    now = time.monotonic()
                    # Reduce sqlite write frequency while still keeping crash-recovery checkpoints.
                    if (
                        ((now - last_store_ts) >= 0.9 and (len(partial) - last_store_len) >= 48)
                        or (("\n\n" in piece) and (len(partial) - last_store_len) >= 120)
                    ):
                        _gen_store_partial(task, partial)
                        last_store_ts = now
                        last_store_len = len(partial)
            except Exception:
                if streamed:
                    if _gen_should_cancel(session_id, task_id):
                        raise RuntimeError("canceled")
                else:
                    resp = ds.chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
                    partial = str(resp or "")
                    _gen_update_task(session_id, task_id, stage="answer", partial=partial, char_count=len(partial))
            else:
                pass

        if _gen_should_cancel(session_id, task_id):
            answer = (str(partial or "").strip() + "\n\n(Generation canceled)").strip() or "(Generation canceled)"
            _gen_store_answer(task, answer)
            _gen_update_task(session_id, task_id, status="canceled", stage="canceled", answer=answer, partial=answer, char_count=len(answer), finished_at=time.time())
            return

        finalize_state = _finalize_generation_answer(
            partial,
            prompt=prompt,
            prompt_for_user=prompt_for_user,
            answer_hits=answer_hits,
            db_dir=db_dir,
            locked_citation_source=locked_citation_source,
            answer_intent=answer_intent,
            answer_depth=answer_depth,
            answer_output_mode=answer_output_mode,
            paper_guide_mode=paper_guide_mode,
            paper_guide_contract_enabled=paper_guide_contract_enabled,
            paper_guide_prompt_family=paper_guide_prompt_family,
            paper_guide_special_focus_block=paper_guide_special_focus_block,
            paper_guide_focus_source_path=paper_guide_focus_source_path,
            paper_guide_direct_source_path=paper_guide_direct_source_path,
            paper_guide_bound_source_path=paper_guide_bound_source_path,
            paper_guide_candidate_refs_by_source=paper_guide_candidate_refs_by_source,
            paper_guide_support_slots=paper_guide_support_slots,
            paper_guide_evidence_cards=paper_guide_evidence_cards,
        )
        answer = str(finalize_state.get("answer") or "")
        paper_guide_support_resolution = list(finalize_state.get("paper_guide_support_resolution") or [])
        citation_validation = dict(finalize_state.get("citation_validation") or {})
        answer_quality = dict(finalize_state.get("answer_quality") or {})
        _gen_store_answer(task, answer)
        _gen_record_answer_quality(
            session_id=session_id,
            task_id=task_id,
            conv_id=conv_id,
            answer_quality=answer_quality,
        )
        t_prov0 = time.perf_counter()
        try:
            _gen_store_answer_provenance_fast(
                task,
                answer=answer,
                answer_hits=answer_hits,
                support_resolution=paper_guide_support_resolution,
            )
            _perf_log("gen.provenance_inline_fast", elapsed=time.perf_counter() - t_prov0, ok=1)
        except Exception as exc:
            _perf_log("gen.provenance_inline_fast", elapsed=time.perf_counter() - t_prov0, ok=0, err=str(exc)[:120])
        if _should_run_provenance_async_refine(task):
            try:
                _gen_store_answer_provenance_async(
                    task,
                    answer=answer,
                    answer_hits=answer_hits,
                    support_resolution=paper_guide_support_resolution,
                )
                _perf_log("gen.provenance_async_schedule", ok=1)
            except Exception as exc:
                _perf_log("gen.provenance_async_schedule", ok=0, err=str(exc)[:120])
        _gen_update_task(
            session_id,
            task_id,
            status="done",
            stage="done",
            answer=answer,
            partial=answer,
            char_count=len(answer),
            answer_ready=True,
            answer_output_mode=answer_output_mode,
            answer_quality=answer_quality,
            citation_validation=citation_validation,
            finished_at=time.time(),
        )
        _perf_log("gen.answer", elapsed=time.perf_counter() - t_answer0, chars=len(answer))
        _perf_log("gen.total", elapsed=time.perf_counter() - worker_t0, conv_id=conv_id)

    except Exception as e:
        if str(e) == "canceled":
            snap = _gen_get_task(session_id) or {}
            partial = str(snap.get("partial") or "").strip()
            answer = (partial + "\n\n(Generation canceled)").strip() or "(Generation canceled)"
            try:
                _gen_store_answer(task, answer)
            except Exception:
                pass
            _gen_update_task(session_id, task_id, status="canceled", stage="canceled", answer=answer, partial=answer, char_count=len(answer), finished_at=time.time())
            return

        err = S["llm_fail"].format(err=str(e))
        try:
            _gen_store_answer(task, err)
        except Exception:
            pass
        _gen_update_task(session_id, task_id, status="error", stage="error", error=str(e), answer=err, partial=err, char_count=len(err), finished_at=time.time())

def _gen_start_task(task: dict) -> bool:
    sid = str(task.get("session_id") or "").strip()
    tid = str(task.get("id") or "").strip()
    if (not sid) or (not tid):
        return False
    with RUNTIME.GEN_LOCK:
        cur = RUNTIME.GEN_TASKS.get(sid)
        if (
            isinstance(cur, dict)
            and str(cur.get("status") or "") == "running"
            and (not bool(cur.get("answer_ready") or False))
        ):
            return False
        item = dict(task)
        item.setdefault("status", "running")
        item.setdefault("stage", "starting")
        item.setdefault("partial", "")
        item.setdefault("char_count", 0)
        item.setdefault("cancel", False)
        item.setdefault("created_at", time.time())
        item.setdefault("updated_at", time.time())
        RUNTIME.GEN_TASKS[sid] = item
    try:
        threading.Thread(target=_gen_worker, args=(sid, tid), daemon=True).start()
    except Exception:
        with RUNTIME.GEN_LOCK:
            cur = RUNTIME.GEN_TASKS.get(sid)
            if isinstance(cur, dict) and str(cur.get("id") or "") == tid:
                cur2 = dict(cur)
                cur2["status"] = "error"
                cur2["stage"] = "error"
                cur2["answer"] = "绾跨▼鍚姩澶辫触"
                cur2["finished_at"] = time.time()
                RUNTIME.GEN_TASKS[sid] = cur2
        return False
    return True

def _bg_enqueue(task: dict) -> None:
    if "_tid" not in task:
        task = dict(task)
        task["_tid"] = uuid.uuid4().hex
    bg_enqueue(_BG_STATE, _BG_LOCK, task)
    _bg_ensure_started()

def _bg_remove_queued_tasks_for_pdf(pdf_path: Path) -> int:
    """
    Remove queued (not running) conversion tasks for a given PDF.
    Returns removed count.
    """
    return bg_remove_queued_tasks_for_pdf(_BG_STATE, _BG_LOCK, pdf_path)

def _bg_cancel_all() -> None:
    bg_cancel_all(_BG_STATE, _BG_LOCK, "Canceling current background conversion")

def _bg_snapshot() -> dict:
    return bg_snapshot(_BG_STATE, _BG_LOCK)

def _bg_target_worker_count() -> int:
    try:
        raw = str(os.environ.get("KB_BG_CONVERT_MAX_ACTIVE", "") or "").strip()
        if raw:
            return max(1, min(4, int(raw)))
    except Exception:
        pass
    return 2


def _bg_worker_loop() -> None:
    while True:
        task = bg_begin_next_task_or_idle(_BG_STATE, _BG_LOCK)

        if task is None:
            time.sleep(0.35)
            continue

        pdf = Path(task["pdf"])
        out_root = Path(task["out_root"])
        db_dir = Path(task.get("db_dir") or "").expanduser() if task.get("db_dir") else None
        no_llm = bool(task.get("no_llm", False))
        # Equation image fallback should be a last resort.
        # - For full_llm (quality-first), prefer editable/searchable LaTeX over screenshots.
        # - In no-LLM degraded runs, `kb/pdf_tools.run_pdf_to_md` will force-enable it to preserve fidelity.
        eq_image_fallback = bool(task.get("eq_image_fallback", False))
        replace = bool(task.get("replace", False))
        speed_mode = str(task.get("speed_mode", "balanced"))
        if speed_mode == "ultra_fast":
            # Keep VL/LLM path in ultra_fast; converter itself handles speed/quality tradeoff.
            # Forcing no_llm here causes a dramatic quality drop that does not match UI semantics.
            eq_image_fallback = False
        task_id = str(task.get("_tid") or "")

        try:
            md_folder = out_root / pdf.stem
            if replace and md_folder.exists():
                # Safety: only delete inside out_root
                try:
                    md_root = out_root.resolve()
                    target = md_folder.resolve()
                    if str(target).lower().startswith(str(md_root).lower()):
                        import shutil

                        shutil.rmtree(md_folder, ignore_errors=True)
                except Exception:
                    pass

            def _on_progress(page_done: int, page_total: int, msg: str = "") -> None:
                try:
                    bg_update_page_progress(_BG_STATE, _BG_LOCK, page_done, page_total, msg, task_id=task_id)
                except Exception:
                    pass

            def _should_cancel() -> bool:
                return bg_should_cancel(_BG_STATE, _BG_LOCK)

            ok, out_folder = run_pdf_to_md(
                pdf_path=pdf,
                out_root=out_root,
                no_llm=no_llm,
                keep_debug=False,
                eq_image_fallback=eq_image_fallback,
                progress_cb=_on_progress,
                cancel_cb=_should_cancel,
                speed_mode=speed_mode,
                max_active_conversions=_bg_target_worker_count(),
            )
            if ok:
                msg = f"OK: {out_folder}"
            else:
                txt = str(out_folder or "").strip().lower()
                msg = "CANCELLED" if txt == "cancelled" else f"FAIL: {out_folder}"

            # Auto-ingest can add noticeable latency in the conversion UI.
            # Skip it in ultra_fast mode to keep end-to-end time near the 5s target.
            do_auto_ingest = ok and bool(db_dir) and (speed_mode != "ultra_fast")
            if do_auto_ingest and db_dir:
                try:
                    ingest_py = Path(__file__).resolve().parent / "ingest.py"
                    _, md_main, md_exists = _resolve_md_output_paths(out_root, pdf)
                    if ingest_py.exists() and md_exists:
                        subprocess.run(
                            [os.sys.executable, str(ingest_py), "--src", str(md_main), "--db", str(db_dir), "--incremental"],
                            check=False,
                            capture_output=True,
                            text=True,
                        )
                        msg = f"OK+INGEST: {out_folder}"
                except Exception:
                    # Not fatal; conversion still succeeded.
                    pass
        except Exception as e:
            msg = f"FAIL: {e}"

        bg_finish_task(_BG_STATE, _BG_LOCK, msg, task_id=task_id)

def _bg_ensure_started() -> None:
    worker_ver = "2026-03-19.bg.v5"
    desired_workers = _bg_target_worker_count()
    threads = list(getattr(RUNTIME, "BG_THREADS", []) or [])
    running_ver = str(getattr(RUNTIME, "BG_WORKER_VERSION", "") or "")
    live_threads = [t for t in threads if t is not None and t.is_alive()]

    if running_ver != worker_ver:
        try:
            RUNTIME.BG_WORKER_VERSION = worker_ver
        except Exception:
            pass

    if len(live_threads) >= desired_workers:
        RUNTIME.BG_THREADS = live_threads
        RUNTIME.BG_THREAD = live_threads[0] if live_threads else None
        return

    while len(live_threads) < desired_workers:
        t = threading.Thread(target=_bg_worker_loop, daemon=True)
        t.start()
        live_threads.append(t)

    RUNTIME.BG_THREADS = live_threads
    RUNTIME.BG_THREAD = live_threads[0] if live_threads else None
    RUNTIME.BG_WORKER_VERSION = worker_ver

def _build_bg_task(
    *,
    pdf_path: Path,
    out_root: Path,
    db_dir: Path,
    no_llm: bool,
    replace: bool = False,
    speed_mode: str = "balanced",
) -> dict:
    pdf = Path(pdf_path)
    mode = str(speed_mode)
    return {
        "_tid": uuid.uuid4().hex,
        "pdf": str(pdf),
        "out_root": str(out_root),
        "db_dir": str(db_dir),
        # no_llm is controlled only by user-selected mode "no_llm".
        # ultra_fast should remain VL-based (lower quality, but still LLM).
        "no_llm": bool(no_llm),
        # Default OFF across all normal modes; enable only explicitly.
        # In no-LLM runs we still force-enable it inside `run_pdf_to_md` for fidelity.
        "eq_image_fallback": False,
        "replace": bool(replace),
        "speed_mode": mode,
        "name": pdf.name,
    }

def _group_hits_by_top_heading(hits: list[dict], top_k: int) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    grouped: list[dict] = []
    for h in hits:
        meta = h.get("meta", {}) or {}
        src = (meta.get("source_path") or "").strip()
        top = _top_heading(meta.get("heading_path", ""))
        key = (src, top)
        if key in seen:
            continue
        seen.add(key)
        gh = dict(h)
        gh_meta = dict(meta)
        gh_meta["top_heading"] = top
        gh["meta"] = gh_meta
        grouped.append(gh)
        if len(grouped) >= max(1, int(top_k)):
            break
    return grouped


def _hit_source_path(hit: dict) -> str:
    if not isinstance(hit, dict):
        return ""
    meta = hit.get("meta", {}) or {}
    return str(meta.get("source_path") or "").strip()


def _paper_guide_focus_heading(hit: dict) -> str:
    return _selection_focus_heading(hit)


def _paper_guide_answer_hit_score(hit: dict, *, prompt: str) -> float:
    return _selection_answer_hit_score(hit, prompt=prompt)


def _select_paper_guide_answer_hits(
    *,
    grouped_docs: list[dict],
    heading_hits: list[dict],
    prompt: str,
    top_n: int,
) -> list[dict]:
    return _selection_select_answer_hits(
        grouped_docs=grouped_docs,
        heading_hits=heading_hits,
        prompt=prompt,
        top_n=top_n,
    )


def _build_answer_hits_for_generation(
    *,
    grouped_docs: list[dict],
    heading_hits: list[dict],
    top_n: int,
    allow_same_source_multiple: bool = False,
) -> list[dict]:
    return _selection_build_answer_hits_for_generation(
        grouped_docs=grouped_docs,
        heading_hits=heading_hits,
        top_n=top_n,
        allow_same_source_multiple=allow_same_source_multiple,
    )


def _ensure_locked_source_in_answer_hits(
    answer_hits: list[dict],
    *,
    source_rec: dict | None,
    seed_docs: list[dict],
    top_n: int,
) -> list[dict]:
    try:
        limit = max(1, int(top_n))
    except Exception:
        limit = 1
    out = list(answer_hits or [])[:limit]
    if not source_rec:
        return out
    locked_src = str((source_rec or {}).get("source_path") or "").strip()
    if not locked_src:
        return out
    if any(_hit_source_path(h) == locked_src for h in out):
        return out
    locked_hit = None
    for cand in seed_docs or []:
        if _hit_source_path(cand) == locked_src:
            locked_hit = cand
            break
    if not isinstance(locked_hit, dict):
        return out
    out2 = [locked_hit]
    for h in out:
        if _hit_source_path(h) == locked_src:
            continue
        out2.append(h)
        if len(out2) >= limit:
            break
    return out2[:limit]


def _should_prefer_grouped_docs_for_answer(grouped_docs: list[dict]) -> bool:
    for doc in grouped_docs or []:
        if not isinstance(doc, dict):
            continue
        meta = doc.get("meta", {}) or {}
        try:
            doc_score = float(meta.get("explicit_doc_match_score") or 0.0)
        except Exception:
            doc_score = 0.0
        if doc_score >= 6.0:
            return True
        if str(meta.get("anchor_target_kind") or "").strip():
            try:
                anchor_score = float(meta.get("anchor_match_score") or 0.0)
            except Exception:
                anchor_score = 0.0
            if anchor_score > 0.0:
                return True
    return False


def _has_anchor_grounded_answer_hits(answer_hits: list[dict]) -> bool:
    return _selection_has_anchor_grounded_answer_hits(answer_hits)


def _aggregate_answer_sources(answer_hits: list[dict]) -> list[dict]:
    agg_by_src: dict[str, dict] = {}
    for hit in answer_hits or []:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        src = str(meta.get("source_path") or "").strip()
        if not src:
            continue
        rec = agg_by_src.get(src)
        if not isinstance(rec, dict):
            rec = {
                "source_path": src,
                "sid": _cite_source_id(src),
                "source_name": _source_name_from_md_path(src),
                "hits": 0,
                "explicit_doc_score": 0.0,
                "anchor_score": 0.0,
                "source_sha1": "",
            }
            agg_by_src[src] = rec
        rec["hits"] = int(rec.get("hits") or 0) + 1
        sha1 = str(meta.get("source_sha1") or "").strip().lower()
        if sha1 and (not str(rec.get("source_sha1") or "").strip()):
            rec["source_sha1"] = sha1
        try:
            rec["explicit_doc_score"] = max(
                float(rec.get("explicit_doc_score") or 0.0),
                float(meta.get("explicit_doc_match_score") or 0.0),
            )
        except Exception:
            pass
        try:
            rec["anchor_score"] = max(
                float(rec.get("anchor_score") or 0.0),
                float(meta.get("anchor_match_score") or 0.0),
            )
        except Exception:
            pass
    out = list(agg_by_src.values())
    out.sort(
        key=lambda item: (
            float(item.get("anchor_score") or 0.0),
            float(item.get("explicit_doc_score") or 0.0),
            int(item.get("hits") or 0),
            str(item.get("source_name") or ""),
        ),
        reverse=True,
    )
    return out


def _pick_locked_citation_source(answer_hits: list[dict]) -> dict | None:
    ranked = _aggregate_answer_sources(answer_hits)
    if not ranked:
        return None
    if len(ranked) == 1:
        rec = dict(ranked[0])
        rec["lock_reason"] = "single_source"
        return rec

    top = ranked[0]
    second = ranked[1]
    top_anchor = float(top.get("anchor_score") or 0.0)
    sec_anchor = float(second.get("anchor_score") or 0.0)
    if top_anchor > 0.0 and top_anchor >= max(1.0, sec_anchor + 0.25):
        rec = dict(top)
        rec["lock_reason"] = "anchor_dominant"
        return rec

    top_doc = float(top.get("explicit_doc_score") or 0.0)
    sec_doc = float(second.get("explicit_doc_score") or 0.0)
    if top_doc >= 6.0 and top_doc >= max(6.0, sec_doc + 1.5):
        rec = dict(top)
        rec["lock_reason"] = "explicit_doc_dominant"
        return rec

    top_hits = int(top.get("hits") or 0)
    sec_hits = int(second.get("hits") or 0)
    if top_hits >= max(2, sec_hits * 2) and top_doc >= max(4.0, sec_doc):
        rec = dict(top)
        rec["lock_reason"] = "hit_dominant"
        return rec
    return None


def _norm_source_key_local(path_like: str) -> str:
    s = str(path_like or "").strip()
    if not s:
        return ""
    try:
        return str(Path(s).expanduser().resolve(strict=False)).strip().lower()
    except Exception:
        try:
            return str(Path(s).expanduser()).strip().lower()
        except Exception:
            return s.lower()


def _source_refs_from_index(index_data: dict, source_path: str, *, source_sha1: str = "") -> dict[int, dict]:
    return _citation_validation_source_refs_from_index(
        index_data,
        source_path,
        source_sha1=source_sha1,
        norm_source_key_local=_norm_source_key_local,
    )


def _validate_structured_citations(
    answer: str,
    *,
    answer_hits: list[dict],
    db_dir: Path | None,
    locked_source: dict | None = None,
    paper_guide_mode: bool = False,
    paper_guide_candidate_refs_by_source: dict[str, list[int]] | None = None,
    paper_guide_support_slots: list[dict] | None = None,
    paper_guide_support_resolution: list[dict] | None = None,
) -> tuple[str, dict]:
    return _citation_validation_validate_structured_citations(
        answer,
        answer_hits=answer_hits,
        db_dir=db_dir,
        locked_source=locked_source,
        paper_guide_mode=paper_guide_mode,
        paper_guide_candidate_refs_by_source=paper_guide_candidate_refs_by_source,
        paper_guide_support_slots=paper_guide_support_slots,
        paper_guide_support_resolution=paper_guide_support_resolution,
        sanitize_structured_cite_tokens=_sanitize_structured_cite_tokens,
        cite_canon_re=_CITE_CANON_RE,
        cite_source_id=_cite_source_id,
        hit_source_path=_hit_source_path,
        load_reference_index=load_reference_index,
        resolve_reference_entry=resolve_reference_entry,
        source_refs_from_index=_source_refs_from_index,
        extract_candidate_ref_nums_from_hits=extract_candidate_ref_nums_from_hits,
        extract_citation_context_hints=extract_citation_context_hints,
        has_explicit_reference_conflict=has_explicit_reference_conflict,
        select_support_slot_for_context=_select_paper_guide_support_slot_for_context,
        reference_alignment_score=reference_alignment_score,
    )


def _filter_history_for_multimodal_turn(
    history: list[dict],
    *,
    cur_user_msg_id: int,
    cur_assistant_msg_id: int,
    has_current_images: bool,
) -> list[dict]:
    return _generation_filter_history_for_multimodal_turn(
        history,
        cur_user_msg_id=cur_user_msg_id,
        cur_assistant_msg_id=cur_assistant_msg_id,
        has_current_images=has_current_images,
        is_live_assistant_text=_is_live_assistant_text,
    )

def _build_paper_guide_direct_abstract_answer(
    *,
    prompt: str,
    source_path: str,
    db_dir: Path | None,
    llm: DeepSeekChat | None = None,
) -> str:
    return _focus_build_direct_abstract_answer(
        prompt=prompt,
        source_path=source_path,
        db_dir=db_dir,
        llm=llm,
        prefer_zh_locale=_prefer_zh_locale,
        extract_bound_paper_abstract=_extract_bound_paper_abstract,
    )


def _paper_guide_citation_lookup_fragments(text: str) -> list[str]:
    return _retrieval_citation_lookup_fragments(text)


def _extract_paper_guide_local_citation_lookup_refs(text: str, *, prompt: str, max_candidates: int = 6) -> list[int]:
    return _retrieval_extract_local_citation_lookup_refs(text, prompt=prompt, max_candidates=max_candidates)


def _build_paper_guide_direct_citation_lookup_answer(
    *,
    prompt: str,
    source_path: str,
    answer_hits: list[dict] | None,
    special_focus_block: str = "",
    db_dir: Path | None,
) -> str:
    def _reference_entry_lookup(src: str, ref_num: int, *, db_dir: Path | None = None) -> dict:
        md_path = _resolve_paper_guide_md_path(src, db_dir=db_dir)
        idx = load_reference_index(Path(db_dir).expanduser()) if db_dir else {}
        return _source_refs_from_index(idx, str(md_path), source_sha1="").get(int(ref_num)) if md_path is not None else {}

    return _retrieval_build_direct_citation_lookup_answer(
        prompt=prompt,
        source_path=source_path,
        answer_hits=answer_hits,
        special_focus_block=special_focus_block,
        db_dir=db_dir,
        extract_special_focus_excerpt=_extract_paper_guide_special_focus_excerpt,
        reference_entry_lookup=_reference_entry_lookup,
    )

def _extract_inline_reference_numbers(text: str, *, max_candidates: int = 8) -> list[int]:
    return _grounding_extract_inline_reference_numbers(text, max_candidates=max_candidates)


def _select_paper_guide_raw_target_hits(
    *,
    hits_raw: list[dict],
    prompt: str,
    top_n: int,
) -> list[dict]:
    return _retrieval_select_raw_target_hits(
        hits_raw=hits_raw,
        prompt=prompt,
        top_n=top_n,
        answer_hit_score=_paper_guide_answer_hit_score,
    )


def _build_paper_guide_special_focus_block(
    cards: list[dict],
    *,
    prompt: str = "",
    prompt_family: str = "",
    source_path: str = "",
    db_dir: Path | None = None,
    answer_hits: list[dict] | None = None,
) -> str:
    return _focus_build_special_focus_block(
        cards,
        prompt=prompt,
        prompt_family=prompt_family,
        source_path=source_path,
        db_dir=db_dir,
        answer_hits=answer_hits,
        hit_source_path=_hit_source_path,
        requested_figure_number=_requested_figure_number,
        extract_inline_reference_numbers=lambda text: _extract_inline_reference_numbers(text, max_candidates=6),
        paper_guide_cue_tokens=_paper_guide_cue_tokens,
        citation_lookup_query_tokens=_paper_guide_citation_lookup_query_tokens,
        citation_lookup_signal_score=_paper_guide_citation_lookup_signal_score,
        extract_bound_paper_method_focus=_extract_bound_paper_method_focus,
        extract_bound_paper_figure_caption=_extract_bound_paper_figure_caption,
    )


def _repair_paper_guide_focus_answer_legacy1(
    answer: str,
    *,
    prompt: str = "",
    prompt_family: str = "",
    special_focus_block: str = "",
) -> str:
    return _focus_repair_answer_legacy1(
        answer,
        prompt=prompt,
        prompt_family=prompt_family,
        special_focus_block=special_focus_block,
    )


def _repair_paper_guide_focus_answer_legacy2(
    answer: str,
    *,
    prompt: str = "",
    prompt_family: str = "",
    special_focus_block: str = "",
) -> str:
    return _focus_repair_answer_legacy2(
        answer,
        prompt=prompt,
        prompt_family=prompt_family,
        special_focus_block=special_focus_block,
    )


def _repair_paper_guide_focus_answer(
    answer: str,
    *,
    prompt: str = "",
    prompt_family: str = "",
    special_focus_block: str = "",
) -> str:
    return _focus_repair_answer(
        answer,
        prompt=prompt,
        prompt_family=prompt_family,
        special_focus_block=special_focus_block,
    )


def _repair_paper_guide_focus_answer_generic(
    answer: str,
    *,
    prompt: str = "",
    prompt_family: str = "",
    special_focus_block: str = "",
    source_path: str = "",
    db_dir: Path | None = None,
) -> str:
    return _focus_repair_answer_generic(
        answer,
        prompt=prompt,
        prompt_family=prompt_family,
        special_focus_block=special_focus_block,
        source_path=source_path,
        db_dir=db_dir,
        extract_inline_reference_numbers=lambda text: _extract_inline_reference_numbers(text, max_candidates=6),
        extract_bound_paper_method_focus=_extract_bound_paper_method_focus,
    )

def _paper_guide_citation_lookup_query_tokens(prompt: str) -> list[str]:
    return _retrieval_citation_lookup_query_tokens(prompt)


def _paper_guide_citation_lookup_signal_score(
    *,
    prompt: str,
    heading: str,
    text: str,
    inline_refs: list[int] | None = None,
    explicit_ref_list_request: bool = False,
) -> float:
    return _retrieval_citation_lookup_signal_score(
        prompt=prompt,
        heading=heading,
        text=text,
        inline_refs=inline_refs,
        explicit_ref_list_request=explicit_ref_list_request,
    )


def _collect_paper_guide_candidate_refs_by_source(
    cards: list[dict],
    *,
    focus_source_path: str = "",
    special_focus_block: str = "",
    prompt_family: str = "",
    prompt: str = "",
    db_dir: Path | None = None,
) -> dict[str, list[int]]:
    return _surfacing_collect_candidate_refs_by_source(
        cards,
        focus_source_path=focus_source_path,
        special_focus_block=special_focus_block,
        prompt_family=prompt_family,
        prompt=prompt,
        db_dir=db_dir,
        extract_special_focus_excerpt=_extract_paper_guide_special_focus_excerpt,
        extract_bound_method_focus=_extract_bound_paper_method_focus,
        extract_method_focus_terms=_extract_paper_guide_method_focus_terms,
    )


def _inject_paper_guide_fallback_citations(
    answer: str,
    *,
    cards: list[dict],
    prompt_family: str = "",
    max_injections: int = 2,
) -> str:
    return _surfacing_inject_fallback_citations(
        answer,
        cards=cards,
        prompt_family=prompt_family,
        max_injections=max_injections,
    )


def _inject_paper_guide_focus_citations(
    answer: str,
    *,
    special_focus_block: str = "",
    source_path: str = "",
    prompt_family: str = "",
    prompt: str = "",
    db_dir: Path | None = None,
) -> str:
    return _surfacing_inject_focus_citations(
        answer,
        special_focus_block=special_focus_block,
        source_path=source_path,
        prompt_family=prompt_family,
        prompt=prompt,
        db_dir=db_dir,
        cite_source_id=_cite_source_id,
        extract_special_focus_excerpt=_extract_paper_guide_special_focus_excerpt,
        extract_bound_method_focus=_extract_bound_paper_method_focus,
        extract_method_focus_terms=_extract_paper_guide_method_focus_terms,
    )


def _inject_paper_guide_card_citations(
    answer: str,
    *,
    cards: list[dict],
    prompt_family: str = "",
    max_injections: int = 2,
) -> str:
    return _surfacing_inject_card_citations(
        answer,
        cards=cards,
        prompt_family=prompt_family,
        max_injections=max_injections,
    )


def _drop_paper_guide_locate_only_line_citations(
    answer: str,
    *,
    support_resolution: list[dict] | None = None,
) -> str:
    return _surfacing_drop_locate_only_line_citations(
        answer,
        support_resolution=support_resolution,
    )


def _promote_paper_guide_numeric_reference_citations(
    answer: str,
    *,
    locked_source: dict | None = None,
) -> str:
    return _surfacing_promote_numeric_reference_citations(
        answer,
        locked_source=locked_source,
    )


def _paper_guide_evidence_card_use_hint(prompt_family: str) -> str:
    return _prompting_evidence_card_use_hint(prompt_family)




def _is_paper_guide_support_meta_line(text: str) -> bool:
    return _grounding_is_support_meta_line(text)


def _paper_guide_support_segment_spans(answer_markdown: str) -> list[dict]:
    return _grounding_support_segment_spans(answer_markdown)


def _paper_guide_support_focus_tokens(*parts: str, limit: int = 18) -> set[str]:
    return _grounding_support_focus_tokens(*parts, limit=limit)


def _extract_paper_guide_locate_anchor(text: str, *, max_chars: int = 220) -> str:
    return _grounding_extract_locate_anchor(text, max_chars=max_chars)


def _is_paper_guide_broad_summary_line(text: str, *, prompt_family: str = "") -> bool:
    return _grounding_is_broad_summary_line(text, prompt_family=prompt_family)


def _extract_paper_guide_ref_spans(text: str, *, max_spans: int = 4) -> list[dict]:
    return _grounding_extract_ref_spans(text, max_spans=max_spans)


def _paper_guide_support_claim_type(
    *,
    prompt_family: str,
    heading: str = "",
    snippet: str = "",
    candidate_refs: list[int] | None = None,
    ref_spans: list[dict] | None = None,
) -> str:
    return _grounding_support_claim_type(
        prompt_family=prompt_family,
        heading=heading,
        snippet=snippet,
        candidate_refs=candidate_refs,
        ref_spans=ref_spans,
    )


def _paper_guide_support_cite_policy(*, claim_type: str, prompt_family: str) -> str:
    return _grounding_support_cite_policy(claim_type=claim_type, prompt_family=prompt_family)


def _resolve_paper_guide_support_slot_block(
    *,
    source_path: str,
    snippet: str,
    heading: str = "",
    prompt_family: str = "",
    claim_type: str = "",
    db_dir: Path | None = None,
    block_cache: dict[str, tuple[Path, list[dict]]] | None = None,
    atom_cache: dict[str, list[dict]] | None = None,
    target_scope: dict | None = None,
) -> dict:
    return _grounding_resolve_support_slot_block(
        source_path=source_path,
        snippet=snippet,
        heading=heading,
        prompt_family=prompt_family,
        claim_type=claim_type,
        db_dir=db_dir,
        block_cache=block_cache,
        atom_cache=atom_cache,
        target_scope=target_scope,
    )


def _build_paper_guide_support_slots(
    cards: list[dict],
    *,
    prompt: str = "",
    prompt_family: str = "",
    db_dir: Path | None = None,
    max_slots: int = 4,
    target_scope: dict | None = None,
) -> list[dict]:
    return _grounding_build_support_slots(
        cards,
        prompt=prompt,
        prompt_family=prompt_family,
        db_dir=db_dir,
        max_slots=max_slots,
        target_scope=target_scope,
    )


def _build_paper_guide_support_slots_block(
    slots: list[dict],
    *,
    max_slots: int = 4,
) -> str:
    return _grounding_build_support_slots_block(slots, max_slots=max_slots)


def _normalize_paper_guide_support_surface(text: str) -> str:
    return _grounding_normalize_support_surface(text)


def _paper_guide_support_rule_tokens(slot: dict) -> set[str]:
    return _grounding_support_rule_tokens(slot)


def _select_paper_guide_support_slot_for_context(
    slots: list[dict],
    *,
    context_text: str = "",
) -> dict | None:
    return _grounding_select_support_slot_for_context(
        slots,
        context_text=context_text,
    )


def _inject_paper_guide_support_markers(
    answer: str,
    *,
    support_slots: list[dict],
    prompt_family: str = "",
    max_injections: int = 3,
) -> str:
    return _grounding_inject_support_markers(
        answer,
        support_slots=support_slots,
        prompt_family=prompt_family,
        max_injections=max_injections,
    )


def _resolve_paper_guide_support_ref_num(slot: dict, *, context_text: str = "") -> tuple[int | None, str]:
    return _grounding_resolve_support_ref_num(slot, context_text=context_text)


def _resolve_paper_guide_support_markers(
    answer: str,
    *,
    support_slots: list[dict],
    prompt_family: str = "",
    db_dir: Path | None = None,
) -> tuple[str, list[dict]]:
    return _grounding_resolve_support_markers(
        answer,
        support_slots=support_slots,
        prompt_family=prompt_family,
        db_dir=db_dir,
    )


def _build_paper_guide_evidence_cards_block(
    cards: list[dict],
    *,
    prompt: str = "",
    prompt_family: str = "",
    max_cards: int = 4,
) -> str:
    return _prompting_build_evidence_cards_block(
        cards,
        prompt=prompt,
        prompt_family=prompt_family,
        max_cards=max_cards,
    )


def _build_paper_guide_citation_grounding_block(
    answer_hits: list[dict],
    *,
    max_blocks: int = 4,
) -> str:
    return _prompting_build_citation_grounding_block(
        answer_hits,
        max_blocks=max_blocks,
        hit_source_path=_hit_source_path,
        paper_guide_focus_heading=_paper_guide_focus_heading,
        cite_source_id=_cite_source_id,
        extract_candidate_ref_nums=extract_candidate_ref_nums_from_hits,
        extract_candidate_ref_cue_texts=extract_candidate_ref_cue_texts,
    )


def _requested_figure_number(prompt: str, answer_hits: list[dict]) -> int:
    return _prompting_requested_figure_number(prompt, answer_hits)


def _source_name_from_md_path(source_path: str) -> str:
    return _shared_source_name_from_md_path(source_path)


def _resolve_doc_image_path(md_path: Path, raw_ref: str) -> Path | None:
    return _figure_resolve_doc_image_path(md_path, raw_ref)


def _collect_doc_figure_assets(md_path: Path) -> list[dict]:
    return _figure_collect_doc_figure_assets(
        md_path,
        extract_figure_number=_extract_figure_number,
    )


def _build_doc_figure_card(*, source_path: str, figure_num: int) -> dict | None:
    return _figure_build_doc_figure_card(
        source_path=source_path,
        figure_num=figure_num,
        collect_doc_figure_assets=_collect_doc_figure_assets,
        source_name_from_md_path=_source_name_from_md_path,
    )


def _score_figure_card_source_binding(*, prompt: str, meta: dict, figure_num: int, source_path: str) -> float:
    return _figure_score_doc_figure_source_binding(
        prompt=prompt,
        meta=meta,
        figure_num=figure_num,
        source_path=source_path,
        source_name_from_md_path=_source_name_from_md_path,
    )


def _maybe_append_library_figure_markdown(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict],
    bound_source_path: str = "",
) -> str:
    return _figure_maybe_append_library_figure_markdown(
        answer,
        prompt=prompt,
        answer_hits=answer_hits,
        bound_source_path=bound_source_path,
        requested_figure_number=_requested_figure_number,
        build_doc_figure_card=_build_doc_figure_card,
        score_figure_card_source_binding=_score_figure_card_source_binding,
    )
