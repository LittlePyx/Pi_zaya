from __future__ import annotations

import copy
import hashlib
import json
import logging
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
from kb.agent.runner import build_agent_trace_for_completed_answer, build_generation_agent_notes
from kb.agent.tools import generate_grounded_answer as agent_generate_grounded_answer
from kb.generation_agent_finalize_runtime import (
    _gen_agent_source_summary as _agent_finalize_source_summary,
    _gen_answer_contract as _agent_finalize_answer_contract,
    _gen_answer_runtime_check as _agent_finalize_answer_runtime_check,
    _gen_build_agent_completion_payload as _agent_finalize_completion_payload,
    _gen_compact_agent_trace as _agent_finalize_compact_trace,
    _gen_repair_answer_runtime as _agent_finalize_repair_answer_runtime,
    _gen_store_agent_trace_meta as _agent_finalize_store_agent_trace_meta,
    _sync_runtime_repaired_answer_contracts as _agent_finalize_sync_runtime_repaired_answer_contracts,
)
from kb.generation_citation_validation_runtime import (
    _source_refs_from_index as _citation_validation_source_refs_from_index,
    _validate_freeform_numeric_citations as _citation_validation_validate_freeform_numeric_citations,
    _validate_structured_citations as _citation_validation_validate_structured_citations,
)
from kb.generation_answer_finalize_runtime import (
    _build_multi_paper_doc_list_contract as _finalize_runtime_build_multi_paper_doc_list_contract,
    _exclude_bound_source_from_multi_paper_doc_list_contract as _finalize_runtime_exclude_bound_source_from_multi_paper_doc_list_contract,
    _filter_multi_paper_doc_list_contract as _finalize_runtime_filter_multi_paper_doc_list_contract,
    _finalize_generation_answer as _finalize_runtime_finalize_generation_answer,
    _format_multi_paper_list_answer as _finalize_runtime_format_multi_paper_list_answer,
)
from kb.paper_guide_contracts import (
    _build_paper_guide_render_packet_model,
    _paper_guide_model_dump,
)
from kb.generation_message_runtime import (
    _build_generation_messages as _generation_build_messages,
    _build_multimodal_user_content as _generation_build_multimodal_user_content,
    _filter_history_for_multimodal_turn as _generation_filter_history_for_multimodal_turn,
)
from kb.generation_state_runtime import (
    _gen_get_task as _state_gen_get_task,
    _gen_has_active_task_id as _state_gen_has_active_task_id,
    _gen_has_running_for_conversation as _state_gen_has_running_for_conversation,
    _gen_mark_cancel as _state_gen_mark_cancel,
    _gen_store_paper_guide_contract_meta as _state_gen_store_paper_guide_contract_meta,
    _gen_should_cancel as _state_gen_should_cancel,
    _gen_store_answer as _state_gen_store_answer,
    _gen_store_answer_contract_meta as _state_gen_store_answer_contract_meta,
    _gen_store_answer_quality_meta as _state_gen_store_answer_quality_meta,
    _gen_store_answer_runtime_check_meta as _state_gen_store_answer_runtime_check_meta,
    _gen_store_answer_provenance as _state_gen_store_answer_provenance,
    _gen_store_answer_provenance_async as _state_gen_store_answer_provenance_async,
    _gen_store_answer_provenance_fast as _state_gen_store_answer_provenance_fast,
    _gen_store_partial as _state_gen_store_partial,
    _gen_task_blocks_conversation as _state_gen_task_blocks_conversation,
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
from kb.converter.quality_gate import prepare_markdown_for_index
from kb.converter.quality_repair import append_conversion_repair_attempt
from kb.converter.structured_indices import rebuild_structured_indices_for_markdown
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
from kb.path_safety import (
    chat_image_upload_roots,
    resolve_verified_chat_image_upload_path,
)
from kb.reference_query_family import (
    prompt_explicitly_requests_multi_paper_list,
    prompt_likely_multi_paper_synthesis,
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
    _paper_guide_retrieval_confidence_snapshot as _retrieval_confidence_snapshot,
    _paper_guide_citation_lookup_fragments as _retrieval_citation_lookup_fragments,
    _paper_guide_citation_lookup_query_tokens as _retrieval_citation_lookup_query_tokens,
    _paper_guide_citation_lookup_signal_score as _retrieval_citation_lookup_signal_score,
    _paper_guide_deepread_heading as _retrieval_deepread_heading,
    _paper_guide_fallback_deepread_hits as _retrieval_fallback_deepread_hits,
    _paper_guide_has_requested_target_hits as _retrieval_has_requested_target_hits,
    _paper_guide_hit_matches_requested_targets as _retrieval_hit_matches_requested_targets,
    _select_paper_guide_raw_target_hits as _retrieval_select_raw_target_hits,
    _paper_guide_targeted_box_excerpt_hits as _retrieval_targeted_box_excerpt_hits,
    _paper_guide_targeted_source_block_hits as _retrieval_targeted_source_block_hits,
    _select_paper_guide_deepread_extras as _retrieval_select_deepread_extras,
)
from kb.paper_guide.grounder import (
    _extract_inline_reference_numbers as _grounding_extract_inline_reference_numbers,
    _build_paper_guide_segment_locate_target as _grounding_build_segment_locate_target,
    _build_paper_guide_segment_reader_open as _grounding_build_segment_reader_open,
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
from kb.research_trace import (
    add_event as _trace_add_event,
    compact_trace as _trace_compact,
    finish_trace as _trace_finish,
    merge_section as _trace_merge_section,
    new_trace as _trace_new,
    summarize_hits as _trace_summarize_hits,
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
from kb.markdown_rendering import _normalize_math_markdown
from kb.localized_strings import S

logger = logging.getLogger(__name__)

GENERATION_START_FAILED_MESSAGE = "Generation could not be started. Please retry."
GENERATION_START_FAILED_MESSAGE_ZH = "回答任务未能启动，请稍后重试。"
GENERATION_INTERRUPTED_MESSAGE = "Answer was interrupted before completion. Please retry."
GENERATION_INTERRUPTED_MESSAGE_ZH = "回答尚未完成就中断了，请重试。"


def generation_start_failed_message(locale: object = "") -> str:
    raw = str(locale or "").strip().lower()
    return GENERATION_START_FAILED_MESSAGE_ZH if raw.startswith("zh") else GENERATION_START_FAILED_MESSAGE


def generation_interrupted_message(locale: object = "") -> str:
    raw = str(locale or "").strip().lower()
    return GENERATION_INTERRUPTED_MESSAGE_ZH if raw.startswith("zh") else GENERATION_INTERRUPTED_MESSAGE


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


def _llm_provider_label(settings_obj: object | None) -> str:
    settings = settings_obj if settings_obj is not None else None
    base_url = str(getattr(settings, "base_url", "") or "").strip().lower()
    model = str(getattr(settings, "model", "") or "").strip()
    if ("dashscope" in base_url) or ("qwen" in model.lower()):
        return "Qwen"
    if "deepseek" in base_url or "deepseek" in model.lower():
        return "DeepSeek"
    if model:
        return model
    if base_url:
        return base_url
    return "current provider"


def _format_llm_failure_message(*, err: object, settings_obj: object | None) -> str:
    base = S["llm_fail"].format(err=str(err))
    provider = _llm_provider_label(settings_obj)
    detail = []
    if provider:
        detail.append(f"当前 provider：{provider}")
    model = str(getattr(settings_obj, "model", "") or "").strip()
    if model and model != provider:
        detail.append(f"model：{model}")
    if not detail:
        return base
    return f"{base}\n\n" + "；".join(detail) + "。"


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


def _refs_background_llm_polish_enabled() -> bool:
    raw = str(os.environ.get("KB_REFS_BACKGROUND_LLM_POLISH", "") or "").strip().lower()
    if not raw:
        raw = str(os.environ.get("KB_REFS_CARD_POLISH_USE_LLM", "1") or "").strip().lower()
    return raw in {"1", "true", "on", "yes"}


def _is_synthetic_research_basket_hit(hit: dict) -> bool:
    if not isinstance(hit, dict):
        return False
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    source_path = str((meta or {}).get("source_path") or "").strip()
    return bool(
        (meta or {}).get("research_basket_evidence")
        and (
            str((meta or {}).get("basket_source_role") or "").strip() == "synthetic_basket_item"
            or source_path.startswith(_RESEARCH_BASKET_SYNTHETIC_SOURCE_PREFIX)
        )
    )


def _compact_basket_refs_text(value: object, *, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        text = text[: max(1, limit - 1)].rstrip() + "..."
    return text


def _build_synthetic_research_basket_refs_hit(hit: dict) -> dict:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    source_path = str((meta or {}).get("source_path") or "").strip()
    source_name = str((meta or {}).get("source_name") or "").strip()
    title = str((meta or {}).get("title") or "").strip()
    doi = str((meta or {}).get("doi") or "").strip()
    year = str((meta or {}).get("year") or "").strip()
    heading = str((meta or {}).get("heading_path") or "").strip()
    display_name = source_name or (f"Research basket: {title}" if title else "Research basket item")
    text = str(hit.get("text") or "").strip()
    summary = _compact_basket_refs_text(
        text
        or "Selected research-basket metadata is available for this answer.",
        limit=420,
    )
    why_bits = ["The user selected this item in the research basket for the current turn."]
    if doi:
        why_bits.append(f"DOI: {doi}.")
    if year:
        why_bits.append(f"Year: {year}.")
    why = _compact_basket_refs_text(" ".join(why_bits), limit=360)
    meta_out = dict(meta or {})
    meta_out["ref_pack_state"] = "ready"
    meta_out["ref_display_reason"] = str(meta_out.get("ref_display_reason") or "research_basket_evidence")
    meta_out["source_kind"] = "research_basket"
    ui_meta = {
        "display_name": display_name,
        "heading_path": heading,
        "score": 9.2,
        "score_pending": False,
        "score_tier": "high",
        "summary_line": summary,
        "summary_kind": "research_basket",
        "summary_label": "Research basket",
        "summary_title": "Selected Context",
        "summary_generation": "research_basket_context",
        "summary_basis": "User-selected research basket item for this answer",
        "why_line": why,
        "why_generation": "research_basket_context",
        "why_basis": "This card is retained because it was selected by the user, not because it resolves to a local PDF.",
        "semantic_badges": [{"text": "Research basket", "score": 1.0}],
        "can_open": False,
        "citation_meta": {
            "title": title,
            "doi": doi,
            "year": year,
            "source_name": display_name,
            "source_path": "",
        },
        "source_kind": "research_basket",
        "source_path": "",
        "reader_open": {},
    }
    ui_meta = {key: value for key, value in ui_meta.items() if value not in (None, "", [], {})}
    return {
        "text": summary,
        "score": float(hit.get("score") or 0.0) or 9.2,
        "meta": meta_out,
        "ui_meta": ui_meta,
    }


def _append_synthetic_research_basket_refs_hits(payload: dict | None, basket_hits: list[dict]) -> dict | None:
    if not basket_hits:
        return payload
    payload_out = dict(payload or {})
    hits = [dict(item) for item in list(payload_out.get("hits") or []) if isinstance(item, dict)]
    seen = {
        str(((item.get("meta") if isinstance(item.get("meta"), dict) else {}) or {}).get("source_path") or "").strip()
        for item in hits
    }
    added = 0
    for raw_hit in basket_hits:
        if not isinstance(raw_hit, dict):
            continue
        meta = raw_hit.get("meta") if isinstance(raw_hit.get("meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or "").strip()
        if source_path and source_path in seen:
            continue
        hits.append(_build_synthetic_research_basket_refs_hit(raw_hit))
        if source_path:
            seen.add(source_path)
        added += 1
    if added <= 0:
        return payload_out
    payload_out["hits"] = hits
    payload_out["payload_mode"] = str(payload_out.get("payload_mode") or "full").strip() or "full"
    payload_out["display_state"] = "ready"
    payload_out.pop("suppression_reason", None)
    payload_out.pop("suggestion", None)
    debug = dict(payload_out.get("pipeline_debug") or {}) if isinstance(payload_out.get("pipeline_debug"), dict) else {}
    debug["research_basket_synthetic_hit_count"] = int(added)
    debug["final_hit_count"] = int(len(hits))
    payload_out["pipeline_debug"] = debug
    return payload_out


def _build_precomputed_refs_render_payload(
    *,
    user_msg_id: int,
    prompt: str,
    answer: str = "",
    prompt_sig: str,
    hits: list[dict],
    scores: list[float],
    used_query: str,
    used_translation: bool,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
    library_db_path: Path | str | None,
) -> tuple[dict | None, str]:
    mid = int(user_msg_id or 0)
    if mid <= 0:
        return None, ""
    docs = [dict(hit) for hit in list(hits or []) if isinstance(hit, dict)]
    if not docs:
        return None, ""
    docs_with_scores: list[tuple[dict, float]] = []
    for idx, doc in enumerate(docs):
        try:
            score = float(list(scores or [])[idx])
        except Exception:
            try:
                score = float(doc.get("score") or 0.0)
            except Exception:
                score = 0.0
        docs_with_scores.append((doc, score))
    synthetic_basket_docs = [doc for doc, _score in docs_with_scores if _is_synthetic_research_basket_hit(doc)]
    render_docs_with_scores = [
        (doc, score)
        for doc, score in docs_with_scores
        if not _is_synthetic_research_basket_hit(doc)
    ]
    for hit in docs:
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        if str((meta or {}).get("ref_pack_state") or "").strip().lower() == "pending":
            return None, ""
    pack = {
        "user_msg_id": mid,
        "prompt": str(prompt or "").strip(),
        "answer": str(answer or "").strip(),
        "prompt_sig": str(prompt_sig or "").strip(),
        "hits": docs,
        "scores": list(scores or []),
        "used_query": str(used_query or "").strip(),
        "used_translation": bool(used_translation),
    }
    try:
        from api.reference_ui import enrich_refs_payload
        from api.routers.library import _md_dir, _pdf_dir
        from api.routers.references import _refs_pack_render_signature
        from kb.library_store import LibraryStore
    except Exception:
        return None, ""
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
    try:
        if render_docs_with_scores:
            render_pack = dict(pack)
            render_pack["hits"] = [doc for doc, _score in render_docs_with_scores]
            render_pack["scores"] = [score for _doc, score in render_docs_with_scores]
            payload_by_user = enrich_refs_payload(
                {mid: render_pack},
                pdf_root=pdf_root,
                md_root=md_root,
                lib_store=lib_store,
                guide_mode=guide_mode,
                guide_source_path=str(guide_source_path or "").strip(),
                guide_source_name=str(guide_source_name or "").strip(),
                render_variant="bounded_full",
                allow_expensive_llm_for_ready=_refs_background_llm_polish_enabled(),
                allow_exact_locate=True,
            )
            payload = payload_by_user.get(mid) if isinstance(payload_by_user, dict) else None
        else:
            payload = {
                "user_msg_id": mid,
                "prompt": str(prompt or "").strip(),
                "answer": str(answer or "").strip(),
                "prompt_sig": str(prompt_sig or "").strip(),
                "hits": [],
                "scores": [],
                "used_query": str(used_query or "").strip(),
                "used_translation": bool(used_translation),
                "payload_mode": "full",
                "display_state": "ready",
                "pipeline_debug": {
                    "raw_hit_count": int(len(docs)),
                    "post_score_gate_hit_count": 0,
                    "post_focus_filter_hit_count": 0,
                    "post_llm_filter_hit_count": 0,
                },
            }
    except Exception:
        return None, ""
    payload = _append_synthetic_research_basket_refs_hits(payload, synthetic_basket_docs)
    if not isinstance(payload, dict) or (not payload):
        return None, ""
    try:
        sig = _refs_pack_render_signature(
            user_msg_id=mid,
            pack=pack,
            guide_mode=guide_mode,
            guide_source_path=str(guide_source_path or "").strip(),
            guide_source_name=str(guide_source_name or "").strip(),
        )
    except Exception:
        sig = ""
    return payload, str(sig or "").strip()


def _extract_doc_list_contract(paper_guide_contracts: dict | None) -> list[dict]:
    contracts = dict(paper_guide_contracts or {})
    rows = [dict(item) for item in list(contracts.get("doc_list") or []) if isinstance(item, dict)]
    return rows


def _load_stored_doc_list_contract(
    *,
    chat_db: Path | str,
    conv_id: str,
    assistant_msg_id: int,
) -> list[dict]:
    amid = int(assistant_msg_id or 0)
    if amid <= 0:
        return []
    try:
        chat_store = ChatStore(Path(str(chat_db or "")).expanduser())
        messages = chat_store.get_messages(str(conv_id or "").strip())
    except Exception:
        return []
    for rec in list(messages or []):
        try:
            mid = int(rec.get("id") or 0)
        except Exception:
            mid = 0
        if mid != amid:
            continue
        meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
        contracts = meta.get("paper_guide_contracts") if isinstance(meta.get("paper_guide_contracts"), dict) else {}
        return _extract_doc_list_contract(contracts)
    return []


def _await_stored_doc_list_contract(
    *,
    chat_db: Path | str,
    conv_id: str,
    assistant_msg_id: int,
    wait_timeout_s: float = 0.0,
    poll_interval_s: float = 0.1,
) -> list[dict]:
    timeout_s = max(0.0, float(wait_timeout_s or 0.0))
    deadline = time.time() + timeout_s
    while True:
        rows = _load_stored_doc_list_contract(
            chat_db=chat_db,
            conv_id=conv_id,
            assistant_msg_id=assistant_msg_id,
        )
        if rows or time.time() >= deadline:
            return rows
        time.sleep(max(0.02, min(0.25, float(poll_interval_s or 0.1))))


def _build_doc_list_refs_render_payload(
    *,
    user_msg_id: int,
    prompt: str,
    prompt_sig: str,
    hits: list[dict],
    scores: list[float],
    used_query: str,
    used_translation: bool,
    doc_list: list[dict] | None,
    guide_mode: bool = False,
    guide_source_path: str = "",
    guide_source_name: str = "",
) -> tuple[dict | None, str]:
    mid = int(user_msg_id or 0)
    docs = [dict(item) for item in list(doc_list or []) if isinstance(item, dict)]
    if mid <= 0:
        return None, ""
    pack = {
        "user_msg_id": mid,
        "prompt": str(prompt or "").strip(),
        "prompt_sig": str(prompt_sig or "").strip(),
        "hits": [dict(hit) for hit in list(hits or []) if isinstance(hit, dict)],
        "scores": list(scores or []),
        "used_query": str(used_query or "").strip(),
        "used_translation": bool(used_translation),
    }
    try:
        from api.reference_ui import build_doc_list_refs_payload
        from api.routers.references import _refs_pack_render_signature
    except Exception:
        return None, ""
    try:
        payload = build_doc_list_refs_payload(
            user_msg_id=mid,
            pack=pack,
            doc_list=docs,
            allow_expensive_llm=True,
            allow_exact_locate=True,
            guide_mode=bool(guide_mode),
            guide_source_path=str(guide_source_path or "").strip(),
            guide_source_name=str(guide_source_name or "").strip(),
        )
    except Exception:
        return None, ""
    if not isinstance(payload, dict) or (not payload):
        return None, ""
    try:
        sig = _refs_pack_render_signature(
            user_msg_id=mid,
            pack=pack,
            guide_mode=bool(guide_mode),
            guide_source_path=str(guide_source_path or "").strip(),
            guide_source_name=str(guide_source_name or "").strip(),
        )
    except Exception:
        sig = ""
    return payload, str(sig or "").strip()


def _compact_doc_list_surface_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


_INCOMPLETE_STREAM_TRAILING_RE = re.compile(
    r"(?:"
    r"具体来说|简单来说|展开来说|也就是说|换句话说|核心是|可以理解为|"
    r"主要包括|分成|分为|下面|如下|例如|比如|"
    r"specifically|in short|in simple terms|for example|for instance|as follows|including"
    r")\s*[:：]\s*$",
    re.IGNORECASE,
)


def _looks_like_incomplete_stream_partial(
    text: str,
    *,
    paper_guide_mode: bool = False,
    prompt_family: str = "",
    has_hits: bool = False,
) -> bool:
    s = str(text or "").strip()
    if not s:
        return True
    if re.search(r"[,，;；、]\s*$", s):
        return True
    if _INCOMPLETE_STREAM_TRAILING_RE.search(s):
        return True
    if re.search(r"(?m)^\s*(?:[-*+]|\d+[.)])\s*$", s):
        return True
    family = str(prompt_family or "").strip().lower()
    broad_family = family in {
        "overview",
        "compare",
        "method",
        "reproduce",
        "strength_limits",
        "figure_walkthrough",
    }
    if paper_guide_mode and has_hits and broad_family:
        compact = _compact_doc_list_surface_text(s)
        if len(compact) < 90 and re.search(r"[:：]\s*$", compact):
            return True
    return False


def _rendered_primary_precision_score(primary_evidence: dict | None) -> tuple[int, int, int, int, int, int]:
    if not isinstance(primary_evidence, dict) or not primary_evidence:
        return (0, 0, 0, 0, 0, 0)
    reason = str(
        primary_evidence.get("selection_reason")
        or primary_evidence.get("selectionReason")
        or ""
    ).strip().lower()
    reason_rank = {
        "prompt_aligned_block": 7,
        "prompt_aligned": 6,
        "shared_refs_pack": 5,
        "reader_open": 5,
        "strict_locate": 5,
        "pending_section_seed": 2,
        "shared_contract_seed": 1,
        "answer_hit_top": 0,
    }.get(reason, 3 if reason else 0)
    return (
        reason_rank,
        1 if bool(primary_evidence.get("strict_locate") or primary_evidence.get("strictLocate")) else 0,
        1 if str(primary_evidence.get("block_id") or primary_evidence.get("blockId") or "").strip() else 0,
        1 if str(primary_evidence.get("anchor_id") or primary_evidence.get("anchorId") or "").strip() else 0,
        1 if str(primary_evidence.get("heading_path") or primary_evidence.get("headingPath") or "").strip() else 0,
        1
        if _compact_doc_list_surface_text(
            str(
                primary_evidence.get("highlight_snippet")
                or primary_evidence.get("highlightSnippet")
                or primary_evidence.get("snippet")
                or ""
            )
        )
        else 0,
    )


def _normalize_rendered_primary_evidence(
    *,
    candidate: dict | None,
    source_path: str = "",
    source_name: str = "",
    reader_open: dict | None = None,
) -> dict:
    if not isinstance(candidate, dict):
        candidate = {}
    out = {
        key: value
        for key, value in dict(candidate or {}).items()
        if value not in ("", None, [], {})
    }
    if not out and not isinstance(reader_open, dict):
        return {}
    source_path = str(source_path or "").strip()
    source_name = str(source_name or "").strip()
    reader = dict(reader_open or {}) if isinstance(reader_open, dict) else {}

    alias_pairs = (
        ("source_path", ("source_path", "sourcePath")),
        ("source_name", ("source_name", "sourceName")),
        ("block_id", ("block_id", "blockId")),
        ("anchor_id", ("anchor_id", "anchorId")),
        ("heading_path", ("heading_path", "headingPath")),
        ("snippet", ("snippet",)),
        ("highlight_snippet", ("highlight_snippet", "highlightSnippet")),
        ("anchor_kind", ("anchor_kind", "anchorKind")),
        ("selection_reason", ("selection_reason", "selectionReason")),
    )
    for canonical_key, aliases in alias_pairs:
        if str(out.get(canonical_key) or "").strip():
            continue
        for alias in aliases:
            value = str(out.get(alias) or "").strip()
            if value:
                out[canonical_key] = value
                break

    if not str(out.get("source_path") or "").strip() and source_path:
        out["source_path"] = source_path
    if not str(out.get("source_name") or "").strip() and source_name:
        out["source_name"] = source_name

    if reader:
        if not str(out.get("heading_path") or "").strip() and str(reader.get("headingPath") or "").strip():
            out["heading_path"] = str(reader.get("headingPath") or "").strip()
        if not str(out.get("snippet") or "").strip() and _compact_doc_list_surface_text(str(reader.get("snippet") or "").strip()):
            out["snippet"] = _compact_doc_list_surface_text(str(reader.get("snippet") or "").strip())
        if not str(out.get("highlight_snippet") or "").strip() and _compact_doc_list_surface_text(str(reader.get("highlightSnippet") or "").strip()):
            out["highlight_snippet"] = _compact_doc_list_surface_text(str(reader.get("highlightSnippet") or "").strip())
        if not str(out.get("block_id") or "").strip() and str(reader.get("blockId") or "").strip():
            out["block_id"] = str(reader.get("blockId") or "").strip()
        if not str(out.get("anchor_id") or "").strip() and str(reader.get("anchorId") or "").strip():
            out["anchor_id"] = str(reader.get("anchorId") or "").strip()
        if not str(out.get("anchor_kind") or "").strip() and str(reader.get("anchorKind") or "").strip():
            out["anchor_kind"] = str(reader.get("anchorKind") or "").strip().lower()
        anchor_number = reader.get("anchorNumber")
        if ("anchor_number" not in out) and anchor_number not in ("", None):
            out["anchor_number"] = anchor_number
        if "strict_locate" not in out and "strictLocate" in reader:
            out["strict_locate"] = bool(reader.get("strictLocate"))

    anchor_number = out.get("anchor_number")
    if anchor_number not in ("", None):
        try:
            out["anchor_number"] = int(anchor_number)
        except Exception:
            out.pop("anchor_number", None)
    if "strictLocate" in out and "strict_locate" not in out:
        out["strict_locate"] = bool(out.get("strictLocate"))
    return {
        key: value
        for key, value in out.items()
        if value not in ("", None, [], {})
    }


def _extract_rendered_hit_primary_evidence(*, hit: dict, base_row: dict | None = None) -> dict:
    hit_dict = dict(hit or {}) if isinstance(hit, dict) else {}
    row = dict(base_row or {}) if isinstance(base_row, dict) else {}
    meta = hit_dict.get("meta") if isinstance(hit_dict.get("meta"), dict) else {}
    ui_meta = hit_dict.get("ui_meta") if isinstance(hit_dict.get("ui_meta"), dict) else {}
    reader_open = ui_meta.get("reader_open") if isinstance(ui_meta.get("reader_open"), dict) else {}
    source_path = str(
        ui_meta.get("source_path")
        or meta.get("source_path")
        or row.get("source_path")
        or ""
    ).strip()
    source_name = str(
        ui_meta.get("display_name")
        or meta.get("source_name")
        or row.get("source_name")
        or ""
    ).strip()

    candidates: list[dict] = []
    for raw_candidate in (
        ui_meta.get("primary_evidence") if isinstance(ui_meta.get("primary_evidence"), dict) else {},
        (reader_open.get("primaryEvidence") if isinstance(reader_open, dict) else {}),
        (ui_meta.get("authoritative_primary_evidence") if isinstance(ui_meta.get("authoritative_primary_evidence"), dict) else {}),
        (row.get("primary_evidence") if isinstance(row.get("primary_evidence"), dict) else {}),
    ):
        candidate = _normalize_rendered_primary_evidence(
            candidate=raw_candidate if isinstance(raw_candidate, dict) else {},
            source_path=source_path,
            source_name=source_name,
            reader_open=reader_open,
        )
        if candidate:
            candidates.append(candidate)

    best: dict = {}
    best_score = (0, 0, 0, 0, 0, 0)
    for candidate in candidates:
        score = _rendered_primary_precision_score(candidate)
        if (not best) or score > best_score:
            best = dict(candidate)
            best_score = score
    return best


def _pick_doc_list_contract_primary_evidence(doc_list_contract: list[dict] | None) -> dict:
    for raw_item in list(doc_list_contract or []):
        if not isinstance(raw_item, dict):
            continue
        primary = _normalize_rendered_primary_evidence(
            candidate=raw_item.get("primary_evidence") if isinstance(raw_item.get("primary_evidence"), dict) else {},
            source_path=str(raw_item.get("source_path") or "").strip(),
            source_name=str(raw_item.get("source_name") or "").strip(),
        )
        if primary:
            return primary
    return {}


def _sync_multi_paper_primary_evidence_into_contracts(
    *,
    paper_guide_contracts: dict | None,
    doc_list_contract: list[dict] | None,
) -> dict:
    contracts = dict(paper_guide_contracts or {}) if isinstance(paper_guide_contracts, dict) else {}
    doc_list_primary = _pick_doc_list_contract_primary_evidence(doc_list_contract)
    if not doc_list_primary:
        return contracts
    current_primary = (
        dict(contracts.get("primary_evidence") or {})
        if isinstance(contracts.get("primary_evidence"), dict)
        else {}
    )
    if current_primary != doc_list_primary:
        contracts["primary_evidence"] = dict(doc_list_primary)
    render_packet = (
        dict(contracts.get("render_packet") or {})
        if isinstance(contracts.get("render_packet"), dict)
        else {}
    )
    if render_packet:
        render_packet_primary = (
            dict(render_packet.get("primary_evidence") or {})
            if isinstance(render_packet.get("primary_evidence"), dict)
            else {}
        )
        if render_packet_primary != doc_list_primary:
            render_packet["primary_evidence"] = dict(doc_list_primary)
            contracts["render_packet"] = render_packet
    return contracts


def _sync_paper_guide_render_packet_with_provenance(
    *,
    paper_guide_contracts: dict | None,
    provenance: dict | None,
    answer: str,
) -> dict:
    contracts = dict(paper_guide_contracts or {}) if isinstance(paper_guide_contracts, dict) else {}
    if not contracts or not isinstance(provenance, dict):
        return contracts
    segments = [dict(item) for item in list(provenance.get("segments") or []) if isinstance(item, dict)]
    if not segments:
        return contracts
    source_path = str(provenance.get("source_path") or "").strip()
    source_name = str(provenance.get("source_name") or "").strip()
    enriched_segments: list[dict] = []
    for seg_raw in segments:
        seg = dict(seg_raw)
        locate_target = (
            dict(seg.get("locate_target") or {})
            if isinstance(seg.get("locate_target"), dict)
            else _grounding_build_segment_locate_target(seg)
        )
        if locate_target:
            seg["locate_target"] = locate_target
        reader_open = (
            dict(seg.get("reader_open") or {})
            if isinstance(seg.get("reader_open"), dict)
            else _grounding_build_segment_reader_open(
                seg,
                source_path=source_path,
                source_name=source_name,
                locate_target=locate_target,
            )
        )
        if reader_open:
            seg["reader_open"] = reader_open
        enriched_segments.append(seg)
    segments = enriched_segments
    packet = (
        dict(contracts.get("render_packet") or {})
        if isinstance(contracts.get("render_packet"), dict)
        else {}
    )
    primary_evidence = (
        dict(packet.get("primary_evidence") or {})
        if isinstance(packet.get("primary_evidence"), dict)
        else (
            dict(contracts.get("primary_evidence") or {})
            if isinstance(contracts.get("primary_evidence"), dict)
            else (
                dict(provenance.get("primary_evidence") or {})
                if isinstance(provenance.get("primary_evidence"), dict)
                else {}
            )
        )
    )
    rendered_answer = str(
        packet.get("rendered_body")
        or packet.get("answer_markdown")
        or answer
        or ""
    ).strip()
    model = _build_paper_guide_render_packet_model(
        answer_markdown=str(packet.get("answer_markdown") or answer or "").strip(),
        notice=str(packet.get("notice") or "").strip(),
        rendered_body=rendered_answer,
        rendered_content=str(packet.get("rendered_content") or rendered_answer).strip(),
        copy_markdown=str(packet.get("copy_markdown") or answer or rendered_answer).strip(),
        copy_text=str(packet.get("copy_text") or answer or rendered_answer).strip(),
        cite_details=list(packet.get("cite_details") or []),
        citation_validation=(
            packet.get("citation_validation")
            if isinstance(packet.get("citation_validation"), dict)
            else {}
        ),
        # Force the initial task payload to prefer freshly grounded provenance.
        # Otherwise the UI can keep a stale seed locate card until the next message reload.
        locate_target={},
        reader_open={},
        provenance_segments=segments,
        primary_evidence=primary_evidence,
    )
    packet_dump = _paper_guide_model_dump(model)
    if any(packet_dump.values()):
        contracts["render_packet"] = packet_dump
    if primary_evidence and not isinstance(contracts.get("primary_evidence"), dict):
        contracts["primary_evidence"] = dict(primary_evidence)
    return contracts


def _build_doc_list_contract_from_rendered_payload(
    *,
    doc_list_contract: list[dict] | None,
    rendered_payload: dict | None,
) -> list[dict]:
    rows = [dict(item) for item in list(doc_list_contract or []) if isinstance(item, dict)]
    payload = dict(rendered_payload or {}) if isinstance(rendered_payload, dict) else {}
    hits = [dict(item) for item in list(payload.get("hits") or []) if isinstance(item, dict)]
    if not rows or not hits:
        return rows

    rows_by_source: dict[str, dict] = {}
    row_order: list[str] = []
    for row in rows:
        source_path = str(row.get("source_path") or "").strip()
        if (not source_path) or (source_path in rows_by_source):
            continue
        rows_by_source[source_path] = dict(row)
        row_order.append(source_path)

    out: list[dict] = []
    seen: set[str] = set()
    for hit in hits:
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        source_path = str(
            ui_meta.get("source_path")
            or meta.get("source_path")
            or ""
        ).strip()
        if (not source_path) or (source_path in seen):
            continue
        base = dict(rows_by_source.get(source_path) or {})
        row = {
            key: value
            for key, value in base.items()
            if value not in ("", None, [], {})
        }
        source_name = str(
            ui_meta.get("display_name")
            or row.get("source_name")
            or meta.get("source_name")
            or ""
        ).strip()
        if source_path:
            row["source_path"] = source_path
        if source_name:
            row["source_name"] = source_name
        primary_evidence = _extract_rendered_hit_primary_evidence(
            hit=hit,
            base_row=row,
        )
        heading_path = str(
            ui_meta.get("heading_path")
            or primary_evidence.get("heading_path")
            or row.get("heading_path")
            or meta.get("ref_best_heading_path")
            or ""
        ).strip()
        if heading_path:
            row["heading_path"] = heading_path
        summary_line = str(
            ui_meta.get("summary_line")
            or primary_evidence.get("highlight_snippet")
            or primary_evidence.get("snippet")
            or row.get("summary_line")
            or ""
        ).strip()
        if summary_line:
            row["summary_line"] = _compact_doc_list_surface_text(summary_line)
        else:
            row.pop("summary_line", None)
        summary_generation = str(
            ui_meta.get("summary_generation")
            or row.get("summary_generation")
            or ""
        ).strip()
        if summary_generation:
            row["summary_generation"] = summary_generation
        else:
            row.pop("summary_generation", None)
        why_line = str(
            ui_meta.get("why_line")
            or row.get("why_line")
            or ""
        ).strip()
        if why_line:
            row["why_line"] = _compact_doc_list_surface_text(why_line)
        else:
            row.pop("why_line", None)
        why_generation = str(
            ui_meta.get("why_generation")
            or row.get("why_generation")
            or ""
        ).strip()
        if why_generation and why_line:
            row["why_generation"] = why_generation
        else:
            row.pop("why_generation", None)
        if primary_evidence:
            row["primary_evidence"] = {
                key: value
                for key, value in primary_evidence.items()
                if value not in ("", None, [], {})
            }
        topic_match_kind = str(
            ui_meta.get("topic_match_kind")
            or row.get("topic_match_kind")
            or ""
        ).strip()
        if topic_match_kind:
            row["topic_match_kind"] = topic_match_kind
        out.append(row)
        seen.add(source_path)

    for source_path in row_order:
        if source_path in seen:
            continue
        out.append(dict(rows_by_source.get(source_path) or {}))
    return out or rows


def _filter_multi_paper_seed_docs_for_display(
    *,
    prompt: str,
    seed_docs: list[dict] | None,
) -> list[dict]:
    docs = [dict(item) for item in list(seed_docs or []) if isinstance(item, dict)]
    if not docs:
        return []
    try:
        doc_list_seed = _finalize_runtime_build_multi_paper_doc_list_contract(
            prompt=prompt,
            seed_docs=list(docs),
            answer_hits=list(docs),
            evidence_cards=[],
        )
    except Exception:
        return docs
    try:
        filtered_doc_list_seed = _finalize_runtime_filter_multi_paper_doc_list_contract(
            prompt=prompt,
            doc_list=list(doc_list_seed or []),
        )
    except Exception:
        filtered_doc_list_seed = list(doc_list_seed or [])
    effective_doc_list_seed = list(filtered_doc_list_seed or doc_list_seed or [])
    source_order = [
        str(item.get("source_path") or "").strip()
        for item in effective_doc_list_seed
        if isinstance(item, dict) and str(item.get("source_path") or "").strip()
    ]
    if not source_order:
        return docs
    docs_by_source: dict[str, dict] = {}
    for doc in docs:
        meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or "").strip()
        if source_path and source_path not in docs_by_source:
            docs_by_source[source_path] = doc
    filtered: list[dict] = []
    seen: set[str] = set()
    for source_path in source_order:
        doc = docs_by_source.get(source_path)
        if not isinstance(doc, dict) or source_path in seen:
            continue
        filtered.append(doc)
        seen.add(source_path)
    target_count = min(len(source_order), len(docs))
    if len(filtered) < target_count:
        for doc in docs:
            meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
            source_path = str((meta or {}).get("source_path") or "").strip()
            if (not source_path) or source_path in seen:
                continue
            filtered.append(doc)
            seen.add(source_path)
            if len(filtered) >= target_count:
                break
    return filtered or docs


def _rebuild_multi_paper_doc_list_contract_from_available_refs(
    *,
    prompt: str,
    seed_docs: list[dict] | None,
    answer_hits: list[dict] | None,
    evidence_cards: list[dict] | None = None,
    exclude_bound_source: bool = False,
    bound_source_path: str = "",
    bound_source_name: str = "",
) -> list[dict]:
    prompt_text = str(prompt or "").strip()
    if not prompt_text:
        return []
    try:
        rows = _finalize_runtime_build_multi_paper_doc_list_contract(
            prompt=prompt_text,
            seed_docs=list(seed_docs or []),
            answer_hits=list(answer_hits or []),
            evidence_cards=list(evidence_cards or []),
        )
    except Exception:
        rows = []
    rows = [dict(item) for item in list(rows or []) if isinstance(item, dict)]
    if exclude_bound_source and rows:
        try:
            rows = _finalize_runtime_exclude_bound_source_from_multi_paper_doc_list_contract(
                doc_list=rows,
                bound_source_path=str(bound_source_path or ""),
                bound_source_name=str(bound_source_name or ""),
            )
        except Exception:
            pass
    try:
        filtered_rows = _finalize_runtime_filter_multi_paper_doc_list_contract(
            prompt=prompt_text,
            doc_list=list(rows or []),
        )
    except Exception:
        filtered_rows = None
    effective_rows = filtered_rows if isinstance(filtered_rows, list) else rows
    return [dict(item) for item in list(effective_rows or []) if isinstance(item, dict)]


def _select_multi_paper_seed_docs_for_display(
    *,
    prompt_multi_paper_list: bool,
    paper_guide_cross_paper_refs: bool,
    answer_grouped_docs: list[dict] | None,
    grouped_docs: list[dict] | None,
) -> list[dict]:
    if not prompt_multi_paper_list:
        return [
            dict(item)
            for item in list(answer_grouped_docs or grouped_docs or [])
            if isinstance(item, dict)
        ]
    if paper_guide_cross_paper_refs and grouped_docs:
        return [dict(item) for item in list(grouped_docs or []) if isinstance(item, dict)]
    return [
        dict(item)
        for item in list(answer_grouped_docs or grouped_docs or [])
        if isinstance(item, dict)
    ]


def _merge_refs_display_docs_with_answer_hits(
    *,
    refs_seed_docs: list[dict] | None,
    answer_hits: list[dict] | None,
    limit: int,
    answer: str = "",
) -> list[dict]:
    try:
        cap = max(1, int(limit))
    except Exception:
        cap = 4
    out: list[dict] = []
    seen: set[str] = set()
    cited_doc_indexes: set[int] = set()
    for match in re.finditer(r"(?<!\[)\[\s*(\d{1,2})\s*\](?!\])", str(answer or "")):
        try:
            cited_doc_indexes.add(int(match.group(1)))
        except Exception:
            continue

    def _source(hit: dict) -> str:
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        return str((meta or {}).get("source_path") or "").strip()

    def _push(rows: list[dict] | None, *, display_reason: str = "", cited_only: bool = False) -> None:
        for idx, raw in enumerate(list(rows or []), start=1):
            if not isinstance(raw, dict):
                continue
            if cited_only and cited_doc_indexes and idx not in cited_doc_indexes:
                continue
            src = _source(raw)
            key = src or f"idx:{id(raw)}"
            if key in seen:
                continue
            seen.add(key)
            hit = copy.deepcopy(raw)
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            meta2 = dict(meta or {})
            if str(meta2.get("ref_pack_state") or "").strip().lower() == "pending":
                meta2["ref_pack_state"] = "ready"
            if display_reason and not str(meta2.get("ref_display_reason") or "").strip():
                meta2["ref_display_reason"] = str(display_reason or "").strip()
            hit["meta"] = meta2
            out.append(hit)
            if len(out) >= cap:
                return

    # The answer is the strongest signal for what the References panel should
    # explain. Put answer sources first, then fill with the original refs seed.
    _push(answer_hits, display_reason="answer_hit_top", cited_only=True)
    if len(out) < cap:
        _push(refs_seed_docs)
    return out[:cap]


def _select_answer_seed_for_generation(
    *,
    paper_guide_cross_paper_refs: bool,
    answer_grouped_docs: list[dict] | None,
    grouped_docs: list[dict] | None,
    heading_hits: list[dict] | None,
) -> list[dict]:
    if paper_guide_cross_paper_refs and grouped_docs:
        return [dict(item) for item in list(grouped_docs or []) if isinstance(item, dict)]
    return [
        dict(item)
        for item in list(answer_grouped_docs or grouped_docs or heading_hits or [])
        if isinstance(item, dict)
    ]


def _should_sync_deep_seed_for_display(
    *,
    hits_raw: list[dict] | None,
    guide_strict_mode: bool,
    prompt: str,
    retrieval_prompt: str,
) -> bool:
    if not list(hits_raw or []):
        return False
    prompt_text = str(prompt or retrieval_prompt or "").strip()
    return bool(
        guide_strict_mode
        or _needs_bound_source_hint(prompt_text)
        or prompt_explicitly_requests_multi_paper_list(prompt_text)
    )


def _set_refs_hit_pack_state(hits: list[dict] | None, *, state: str) -> list[dict]:
    out: list[dict] = []
    state_norm = str(state or "").strip().lower()
    for raw_hit in list(hits or []):
        if not isinstance(raw_hit, dict):
            continue
        hit = copy.deepcopy(raw_hit)
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        meta2 = dict(meta or {})
        if state_norm:
            meta2["ref_pack_state"] = state_norm
        else:
            meta2.pop("ref_pack_state", None)
        hit["meta"] = meta2
        out.append(hit)
    return out


_PAPER_GUIDE_PREFETCH_LOCK = threading.Lock()
_PAPER_GUIDE_PREFETCH_RECENT: dict[str, float] = {}
_PAPER_GUIDE_PREFETCH_TTL_S = 20.0 * 60.0


def kickoff_paper_guide_prefetch(
    *,
    source_path: str,
    source_name: str = "",
    db_dir: Path | str | None = None,
    md_root: Path | str | None = None,
    pdf_root: Path | str | None = None,
    library_db_path: Path | str | None = None,
) -> bool:
    raw_source = str(source_path or "").strip()
    if not raw_source:
        return False
    md_path = _resolve_paper_guide_md_path(raw_source, md_root=md_root, db_dir=db_dir, pdf_root=pdf_root)
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


def _source_hint_for_query(source_hint: str) -> str:
    raw = str(source_hint or "").strip()
    if not raw:
        return ""
    try:
        path = Path(raw)
        name = path.stem or path.name
    except Exception:
        name = raw
    name = re.sub(r"\.en$", "", str(name or ""), flags=re.I).strip()
    name = re.sub(r"\s+", " ", name)
    return name or raw


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


def _apply_preferred_source_hints(prompt: str, source_hints: list[str], *, limit: int = 6) -> str:
    q = str(prompt or "").strip()
    if not q:
        return q
    out = q
    used = 0
    seen: set[str] = set()
    for raw in list(source_hints or []):
        hint = _source_hint_for_query(str(raw or ""))
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


def _should_apply_implicit_source_hints(*, prompt: str, paper_guide_mode: bool) -> bool:
    prompt_text = str(prompt or "").strip()
    if not prompt_text:
        return True
    if prompt_explicitly_requests_multi_paper_list(prompt_text) and (not paper_guide_mode):
        return False
    return True

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
    paper_guide_reference_opportunities_block: str = "",
    citation_plan_block: str = "",
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
        paper_guide_reference_opportunities_block=paper_guide_reference_opportunities_block,
        citation_plan_block=citation_plan_block,
        image_attachment_count=image_attachment_count,
    )


def _build_multimodal_user_content(
    user: str,
    image_attachments: list[dict] | None,
    *,
    allowed_image_roots: list[Path | str] | None = None,
) -> str | list[dict]:
    return _generation_build_multimodal_user_content(
        user,
        image_attachments,
        vision_image_mime_by_suffix=_VISION_IMAGE_MIME_BY_SUFFIX,
        allowed_image_roots=allowed_image_roots,
    )


def _build_generation_messages(*, system: str, hist: list[dict], user_content: str | list[dict]) -> list[dict]:
    return _generation_build_messages(system=system, hist=hist, user_content=user_content)


def _selected_research_context_items(value) -> list[dict]:
    if not isinstance(value, dict):
        return []
    raw_items = value.get("items")
    if not isinstance(raw_items, list):
        return []
    out: list[dict] = []
    for raw in raw_items[:8]:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip()
        summary = str(raw.get("summary") or "").strip()
        excerpt = str(raw.get("excerpt") or "").strip()
        note = str(raw.get("note") or "").strip()
        if not any((title, summary, excerpt, note)):
            continue
        out.append(dict(raw))
    return out


def _selected_context_prompt_path_label(path_text: object) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""
    normalized = text.replace("\\", "/").rstrip("/")
    label = normalized.rsplit("/", 1)[-1].strip()
    return label or text


def _format_selected_research_context_block(value) -> str:
    items = _selected_research_context_items(value)
    if not items:
        return ""
    lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        title = str(item.get("title") or "").strip() or "Untitled excerpt"
        kind = str(item.get("kind") or "").strip()
        source_name = str(item.get("sourceName") or "").strip()
        location = str(item.get("locationLabel") or "").strip()
        doi = str(item.get("doi") or "").strip()
        block_id = str(item.get("blockId") or item.get("block_id") or "").strip()
        anchor_id = str(item.get("anchorId") or item.get("anchor_id") or "").strip()
        heading_path = str(item.get("headingPath") or item.get("heading_path") or "").strip()
        library_match_path = str(item.get("libraryMatchPath") or item.get("library_match_path") or "").strip()
        library_match_status = str(item.get("libraryMatchStatus") or item.get("library_match_status") or "").strip()
        ref_num = item.get("refNum")
        head_bits = [f"[SHELF-{idx}]"]
        if kind:
            head_bits.append(kind)
        head_bits.append(title[:260])
        lines.append(" ".join(head_bits))
        meta_bits = []
        if source_name:
            meta_bits.append(f"source={source_name[:220]}")
        if location:
            meta_bits.append(f"location={location[:220]}")
        if heading_path and heading_path != location:
            meta_bits.append(f"heading={heading_path[:220]}")
        if block_id:
            meta_bits.append(f"block={block_id[:120]}")
        if anchor_id:
            meta_bits.append(f"anchor={anchor_id[:120]}")
        if ref_num:
            meta_bits.append(f"ref={ref_num}")
        if doi:
            meta_bits.append(f"doi={doi[:180]}")
        if library_match_path:
            library_match_label = _selected_context_prompt_path_label(library_match_path)
            match_label = f"library_match={library_match_label[:220]}"
            if library_match_status:
                match_label = f"{match_label} ({library_match_status[:40]})"
            meta_bits.append(match_label)
        if meta_bits:
            lines.append("  " + "; ".join(meta_bits))
        summary = str(item.get("summary") or "").strip()
        excerpt = str(item.get("excerpt") or "").strip()
        note = str(item.get("note") or "").strip()
        if summary:
            lines.append(f"  summary: {summary[:900]}")
        if excerpt:
            lines.append(f"  excerpt: {excerpt[:900]}")
        if note:
            lines.append(f"  user_note: {note[:520]}")
    block = "\n".join(lines).strip()
    if len(block) > 5200:
        block = block[:5199].rstrip() + "..."
    return block


def _normalize_selected_context_doi(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    return text.strip(" \t\r\n'\"`([{<.,;:)]}>")


def _normalize_selected_context_title(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _selected_context_first_text(item: dict, *keys: str) -> str:
    for key in keys:
        text = str(item.get(key) or "").strip()
        if text:
            return text
    return ""


def _selected_context_kind(item: dict) -> str:
    raw = _selected_context_first_text(item, "kind", "shelfItemKind", "shelf_item_kind")
    return re.sub(r"[\s-]+", "_", raw.strip().lower())


def _selected_context_library_match_usable(item: dict) -> bool:
    path = _selected_context_first_text(item, "libraryMatchPath", "library_match_path")
    if not path:
        return False
    status = _selected_context_first_text(item, "libraryMatchStatus", "library_match_status").lower()
    return status not in {"missing", "none", "not_found", "unmatched", "failed", "error"}


_RESEARCH_BASKET_SYNTHETIC_SOURCE_PREFIX = "__research_basket__/"


def _selected_context_stable_key(item: dict, idx: int) -> str:
    explicit = _selected_context_first_text(item, "key", "id", "itemKey", "item_key")
    if explicit:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", explicit).strip("_")[:80] or f"item_{idx}"
    payload = "\n".join(
        _selected_context_first_text(item, *keys)
        for keys in (
            ("libraryMatchDoi", "library_match_doi", "doi", "doiUrl", "doi_url"),
            ("libraryMatchTitle", "library_match_title", "title", "main", "cardTitle", "card_title"),
            ("sourcePath", "source_path"),
            ("blockId", "block_id"),
            ("anchorId", "anchor_id"),
            ("excerpt", "shelfExcerpt", "shelf_excerpt", "summary", "note"),
        )
    )
    digest = hashlib.sha1(payload.encode("utf-8", "ignore")).hexdigest()[:12] if payload.strip() else f"{idx:02d}"
    return f"item_{idx}_{digest}"


def _selected_context_evidence_text(item: dict, *, title: str, doi: str, year: str, library_match_path: str) -> str:
    lines: list[str] = []
    if title:
        lines.append(f"Title: {title}")
    if doi:
        lines.append(f"DOI: {doi}")
    if year:
        lines.append(f"Year: {year}")
    if library_match_path:
        lines.append(f"Local library match: {library_match_path}")
    for label, keys in (
        ("Summary", ("summary",)),
        ("Selected excerpt", ("excerpt", "shelfExcerpt", "shelf_excerpt", "evidenceQuote", "evidence_quote")),
        ("User note", ("note", "userNote", "user_note")),
    ):
        text = _selected_context_first_text(item, *keys)
        if text:
            lines.append(f"{label}: {text}")
    text = "\n".join(lines).strip()
    if len(text) > 1600:
        text = text[:1599].rstrip() + "..."
    return text


def _selected_research_context_evidence_hits(items: list[dict], *, max_hits: int = 4) -> list[dict]:
    try:
        limit = max(1, int(max_hits))
    except Exception:
        limit = 4
    out: list[dict] = []
    seen: set[str] = set()
    for idx, item in enumerate(list(items or [])[:8], start=1):
        if len(out) >= limit:
            break
        if not isinstance(item, dict):
            continue
        stable_key = _selected_context_stable_key(item, idx)
        kind = _selected_context_kind(item)
        original_source_path = _selected_context_first_text(item, "sourcePath", "source_path")
        original_source_name = _selected_context_first_text(item, "sourceName", "source_name")
        library_match_path = _selected_context_first_text(item, "libraryMatchPath", "library_match_path")
        library_match_status = _selected_context_first_text(item, "libraryMatchStatus", "library_match_status")
        library_match_title = _selected_context_first_text(item, "libraryMatchTitle", "library_match_title")
        library_match_doi = _selected_context_first_text(item, "libraryMatchDoi", "library_match_doi")
        library_match_year = _selected_context_first_text(item, "libraryMatchYear", "library_match_year")
        has_usable_library_match = _selected_context_library_match_usable(item)
        title = (
            library_match_title
            or _selected_context_first_text(item, "title", "main", "cardTitle", "card_title")
        )
        doi = library_match_doi or _selected_context_first_text(item, "doi", "doiUrl", "doi_url")
        year = library_match_year or _selected_context_first_text(item, "year", "publishedYear", "published_year")
        heading_path = _selected_context_first_text(item, "headingPath", "heading_path", "locationLabel", "location_label")
        block_id = _selected_context_first_text(item, "blockId", "block_id")
        anchor_id = _selected_context_first_text(item, "anchorId", "anchor_id", "anchor")
        anchor_kind = _selected_context_first_text(item, "anchorKind", "anchor_kind")
        if has_usable_library_match:
            source_path = library_match_path
            source_role = "matched_library_paper"
        elif original_source_path:
            source_path = original_source_path
            source_role = "selected_source"
        else:
            source_path = f"{_RESEARCH_BASKET_SYNTHETIC_SOURCE_PREFIX}{stable_key}"
            source_role = "synthetic_basket_item"
        source_name = (
            original_source_name
            or title
            or _selected_context_first_text(item, "locationLabel", "location_label")
            or f"Research basket item {idx}"
        )
        if source_role == "synthetic_basket_item" and title:
            source_name = f"Research basket: {title[:120]}"
        text = _selected_context_evidence_text(
            item,
            title=title,
            doi=doi,
            year=year,
            library_match_path=library_match_path if has_usable_library_match else "",
        )
        if not text:
            continue
        dedupe_key = "\n".join(
            (
                str(source_path or "").replace("\\", "/").casefold(),
                str(block_id or "").casefold(),
                str(anchor_id or "").casefold(),
                _normalize_selected_context_doi(doi),
                _normalize_selected_context_title(title),
                hashlib.sha1(text[:320].encode("utf-8", "ignore")).hexdigest()[:12],
            )
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        meta = {
            "source_path": source_path,
            "source_name": source_name,
            "title": title,
            "doi": doi,
            "year": year,
            "heading_path": heading_path,
            "top_heading": _top_heading(heading_path),
            "block_id": block_id,
            "anchor_id": anchor_id,
            "anchor_kind": anchor_kind,
            "anchor_target_kind": anchor_kind,
            "ref_pack_state": "ready",
            "metadata_quality": "ready",
            "source_kind": "research_basket",
            "research_basket_evidence": True,
            "basket_evidence": True,
            "basket_item_index": idx,
            "basket_item_key": stable_key,
            "shelf_item_kind": kind,
            "basket_source_role": source_role,
            "selected_context_source_path": original_source_path,
            "selected_context_source_name": original_source_name,
            "library_match_path": library_match_path,
            "library_match_status": library_match_status,
            "library_match_title": library_match_title,
            "library_match_doi": library_match_doi,
            "library_match_year": library_match_year,
            "citation_context_source": "research_basket",
        }
        meta = {key: value for key, value in meta.items() if value not in (None, "", [], {})}
        out.append(
            {
                "text": text,
                "score": 1000.0 - float(idx),
                "meta": meta,
                "ui_meta": {
                    "source_path": source_path,
                    "source_name": source_name,
                    "display_name": source_name,
                    "heading_path": heading_path,
                    "source_kind": "research_basket",
                },
            }
        )
    return out


def _selected_context_hit_dedupe_key(hit: dict) -> tuple[str, str, str]:
    if not isinstance(hit, dict):
        return ("invalid", "", "")
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    source_path = str((meta or {}).get("source_path") or "").strip().replace("\\", "/").casefold()
    block_id = str((meta or {}).get("block_id") or (meta or {}).get("ref_block_id") or "").strip().casefold()
    anchor_id = str((meta or {}).get("anchor_id") or (meta or {}).get("ref_anchor_id") or "").strip().casefold()
    doi = _normalize_selected_context_doi((meta or {}).get("doi") or (meta or {}).get("library_match_doi"))
    title = _normalize_selected_context_title((meta or {}).get("title") or (meta or {}).get("library_match_title"))
    if source_path and (block_id or anchor_id):
        return ("loc", source_path, f"{block_id}|{anchor_id}")
    if doi:
        return ("doi", doi, "")
    if source_path and title:
        return ("title", source_path, title)
    text = str(hit.get("text") or "").strip()
    return ("text", source_path, hashlib.sha1(text[:320].encode("utf-8", "ignore")).hexdigest()[:12])


def _merge_selected_research_context_evidence_hits(
    answer_hits: list[dict],
    basket_hits: list[dict],
    *,
    limit: int,
) -> list[dict]:
    try:
        cap = max(1, int(limit))
    except Exception:
        cap = max(1, len(answer_hits or []) + len(basket_hits or []))
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for hit in list(basket_hits or []) + list(answer_hits or []):
        if not isinstance(hit, dict):
            continue
        key = _selected_context_hit_dedupe_key(hit)
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
        if len(out) >= cap:
            break
    return out


def _selected_research_context_evidence_contract(basket_hits: list[dict]) -> dict:
    items: list[dict] = []
    for hit in list(basket_hits or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        entry = {
            "source_path": str((meta or {}).get("source_path") or "").strip(),
            "source_name": str((meta or {}).get("source_name") or "").strip(),
            "title": str((meta or {}).get("title") or "").strip(),
            "doi": str((meta or {}).get("doi") or "").strip(),
            "year": str((meta or {}).get("year") or "").strip(),
            "heading_path": str((meta or {}).get("heading_path") or "").strip(),
            "block_id": str((meta or {}).get("block_id") or "").strip(),
            "anchor_id": str((meta or {}).get("anchor_id") or "").strip(),
            "anchor_kind": str((meta or {}).get("anchor_kind") or "").strip(),
            "basket_item_index": int((meta or {}).get("basket_item_index") or 0),
            "basket_item_key": str((meta or {}).get("basket_item_key") or "").strip(),
            "basket_source_role": str((meta or {}).get("basket_source_role") or "").strip(),
            "shelf_item_kind": str((meta or {}).get("shelf_item_kind") or "").strip(),
        }
        items.append({key: value for key, value in entry.items() if value not in (None, "", [], {})})
    if not items:
        return {}
    return {"version": 1, "count": len(items), "items": items}


def _selected_research_context_filter_terms(items: list[dict]) -> dict[str, list]:
    terms: dict[str, list] = {
        "source_paths": [],
        "source_names": [],
        "dois": [],
        "titles": [],
        "item_constraints": [],
    }
    seen: dict[str, set[str]] = {key: set() for key in terms}

    def add(key: str, value: object) -> None:
        raw = str(value or "").strip()
        if not raw:
            return
        if key == "dois":
            raw = _normalize_selected_context_doi(raw)
        elif key == "titles":
            raw = _normalize_selected_context_title(raw)
            if len(raw) < 16 or raw in {"untitled", "untitled excerpt", "reference entry", "selected text"}:
                return
        dedupe_key = raw.lower()
        if (not raw) or dedupe_key in seen[key]:
            return
        seen[key].add(dedupe_key)
        terms[key].append(raw)

    def add_item_constraint(item: dict) -> None:
        source_path = _selected_context_first_text(item, "sourcePath", "source_path")
        source_name = _selected_context_first_text(item, "sourceName", "source_name")
        if not (source_path or source_name):
            return
        constraint = {
            "source_path": source_path,
            "source_name": source_name,
            "block_id": _selected_context_first_text(item, "blockId", "block_id"),
            "anchor_id": _selected_context_first_text(item, "anchorId", "anchor_id", "anchor"),
            "heading_path": _selected_context_first_text(item, "headingPath", "heading_path", "locationLabel", "location_label"),
            "excerpt": _selected_context_first_text(
                item,
                "excerpt",
                "shelfExcerpt",
                "shelf_excerpt",
                "evidenceQuote",
                "evidence_quote",
                "summary",
            ),
        }
        if not any(str(constraint.get(key) or "").strip() for key in ("block_id", "anchor_id", "heading_path", "excerpt")):
            return
        key = "\n".join(
            str(constraint.get(k) or "").strip().lower()
            for k in ("source_path", "source_name", "block_id", "anchor_id", "heading_path", "excerpt")
        )
        if (not key.strip()) or key in seen["item_constraints"]:
            return
        seen["item_constraints"].add(key)
        terms["item_constraints"].append(constraint)

    for item in list(items or [])[:8]:
        if not isinstance(item, dict):
            continue
        raw_doi = (
            item.get("libraryMatchDoi")
            or item.get("library_match_doi")
            or item.get("doi")
            or item.get("doiUrl")
            or item.get("doi_url")
        )
        raw_title = (
            item.get("libraryMatchTitle")
            or item.get("library_match_title")
            or item.get("title")
            or item.get("main")
            or item.get("cardTitle")
            or item.get("card_title")
        )
        doi_term = _normalize_selected_context_doi(raw_doi)
        title_term = _normalize_selected_context_title(raw_title)
        has_bibliographic_identity = bool(
            doi_term
            or (
                len(title_term) >= 16
                and title_term not in {"untitled", "untitled excerpt", "reference entry", "selected text"}
            )
        )
        kind = _selected_context_kind(item)
        if kind == "reference":
            if _selected_context_library_match_usable(item):
                add("source_paths", item.get("libraryMatchPath") or item.get("library_match_path"))
            elif not has_bibliographic_identity:
                add_item_constraint(item)
        else:
            add_item_constraint(item)
        if kind == "reference" or _selected_context_library_match_usable(item):
            add("dois", raw_doi)
            add("titles", raw_title)
    return terms


def _selected_research_context_terms_count(terms: dict[str, list]) -> int:
    return sum(len(list(values or [])) for values in (terms or {}).values())


def _hit_matches_selected_item_constraint(hit: dict, constraint: dict) -> bool:
    if not isinstance(hit, dict):
        return False
    meta = hit.get("meta", {}) or {}
    source_path = str(constraint.get("source_path") or "").strip()
    source_name = str(constraint.get("source_name") or "").strip()
    source_ok = True
    if source_path or source_name:
        source_ok = _is_hit_from_bound_source(
            hit,
            bound_source_path=source_path,
            bound_source_name=source_name,
        )
    if not source_ok:
        return False

    block_id = str(constraint.get("block_id") or "").strip()
    if block_id and block_id == str(meta.get("block_id") or "").strip():
        return True
    anchor_id = str(constraint.get("anchor_id") or "").strip()
    if anchor_id and anchor_id == str(meta.get("anchor_id") or meta.get("anchor") or "").strip():
        return True

    heading = normalize_match_text(str(constraint.get("heading_path") or ""))
    hit_heading = normalize_match_text(str(meta.get("heading_path") or meta.get("top_heading") or ""))
    if heading and hit_heading and len(heading) >= 6:
        if heading == hit_heading or heading in hit_heading or hit_heading in heading:
            return True

    excerpt = normalize_match_text(str(constraint.get("excerpt") or ""))
    hit_text = normalize_match_text(str(hit.get("text") or ""))
    if excerpt and hit_text and len(excerpt) >= 24:
        if excerpt in hit_text or hit_text in excerpt:
            return True
        probe = hit_text[: max(240, min(900, len(excerpt) * 2))]
        if SequenceMatcher(None, excerpt[:900], probe).ratio() >= 0.82:
            return True
        excerpt_tokens = {t for t in re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", excerpt) if len(t) >= 2}
        hit_tokens = {t for t in re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", hit_text) if len(t) >= 2}
        if excerpt_tokens and len(excerpt_tokens) >= 6:
            overlap = len(excerpt_tokens & hit_tokens) / max(1, len(excerpt_tokens))
            if overlap >= 0.58:
                return True
    return False


def _hit_matches_selected_research_context(hit: dict, terms: dict[str, list]) -> bool:
    if not isinstance(hit, dict):
        return False
    meta = hit.get("meta", {}) or {}
    for source_path in terms.get("source_paths") or []:
        if _is_hit_from_bound_source(hit, bound_source_path=str(source_path or ""), bound_source_name=""):
            return True
    for source_name in terms.get("source_names") or []:
        if _is_hit_from_bound_source(hit, bound_source_path="", bound_source_name=str(source_name or "")):
            return True

    haystack = "\n".join(
        str(part or "")
        for part in (
            hit.get("text"),
            meta.get("doi"),
            meta.get("title"),
            meta.get("source_name"),
            meta.get("source_path"),
            meta.get("heading_path"),
        )
    )
    haystack_doi = _normalize_selected_context_doi(haystack)
    for doi in terms.get("dois") or []:
        if doi and doi in haystack_doi:
            return True

    haystack_title = _normalize_selected_context_title(haystack)
    for title in terms.get("titles") or []:
        if title and (title in haystack_title or SequenceMatcher(None, title, haystack_title[: max(len(title) * 3, 240)]).ratio() >= 0.86):
            return True
    for constraint in terms.get("item_constraints") or []:
        if isinstance(constraint, dict) and _hit_matches_selected_item_constraint(hit, constraint):
            return True
    return False


def _filter_hits_for_selected_research_context(
    hits_raw: list[dict],
    selected_items: list[dict],
) -> tuple[list[dict], dict[str, object]]:
    terms = _selected_research_context_filter_terms(selected_items)
    constraint_count = _selected_research_context_terms_count(terms)
    if constraint_count <= 0:
        return [], {
            "active": True,
            "mode": "selected_context_only",
            "before": int(len(hits_raw or [])),
            "after": 0,
            "constraint_count": 0,
        }
    out = [
        hit
        for hit in list(hits_raw or [])
        if isinstance(hit, dict) and _hit_matches_selected_research_context(hit, terms)
    ]
    return out, {
        "active": True,
        "mode": "matched_library_hits" if out else "no_matching_library_hits",
        "before": int(len(hits_raw or [])),
        "after": int(len(out)),
        "constraint_count": int(constraint_count),
        "source_path_count": int(len(terms.get("source_paths") or [])),
        "source_name_count": int(len(terms.get("source_names") or [])),
        "doi_count": int(len(terms.get("dois") or [])),
        "title_count": int(len(terms.get("titles") or [])),
        "item_constraint_count": int(len(terms.get("item_constraints") or [])),
    }


def _normalize_query_scope(value: object) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in {"current", "paper", "current_paper", "source", "reader"}:
        return "current_paper"
    if raw in {"basket", "shelf", "citation_shelf", "selected"}:
        return "basket"
    if raw in {"library", "all", "all_library", "full_library"}:
        return "library"
    return ""


def _effective_query_scope(*, requested: object, paper_guide_mode: bool, has_current_paper: bool, has_basket: bool) -> str:
    scope = _normalize_query_scope(requested)
    if scope == "current_paper" and not has_current_paper:
        scope = ""
    if scope == "basket" and not has_basket:
        scope = ""
    if scope:
        return scope
    if has_current_paper and paper_guide_mode:
        return "current_paper"
    return "library"


def _query_scope_prompt_block(*, scope: str, selected_count: int, current_source_name: str, current_source_path: str) -> str:
    label = str(current_source_name or "").strip() or str(current_source_path or "").strip()
    if scope == "current_paper":
        source_part = f" Current paper: {label}." if label else ""
        return (
            "QUERY SCOPE: Current paper.\n"
            f"- Answer inside the currently opened/bound paper.{source_part}\n"
            "- Use other library snippets only when the user explicitly asks for outside papers or background."
        )
    if scope == "basket":
        return (
            "QUERY SCOPE: Research basket.\n"
            f"- The user selected {max(0, int(selected_count or 0))} basket item(s) for this turn.\n"
            "- Answer only from the selected basket excerpts and any retrieved snippets that match those selected sources or bibliographic identities.\n"
            "- If the selected basket context is insufficient, say what is missing instead of bringing in unrelated library papers."
        )
    return (
        "QUERY SCOPE: Full library.\n"
        "- Search and synthesize across the whole indexed literature library.\n"
        "- When multiple papers are relevant, organize the answer by paper and explain why each paper matches.\n"
        "- Pair important claims with exact retrieved evidence or location markers; say clearly when the library lacks direct support."
    )


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


def _build_paper_guide_supplement_reason_text(reason: str, *, prefer_zh: bool) -> str:
    reason_norm = str(reason or "").strip().lower()
    reason_map_zh = {
        "empty_hits": "当前没有检索到能直接回答该问题的原文片段",
        "target_miss": "检索没有直接命中你问到的具体目标位置",
        "reference_only_hits": "当前命中内容更偏参考文献或外围片段",
        "weak_signal": "当前原文证据信号偏弱",
        "strict_family_without_targeted_support": "当前问题需要更定向的证据，但命中片段不够聚焦",
        "strict_family_weak_overlap": "当前问题与命中片段的词面重叠较弱",
        "strict_family_sparse_hits": "当前问题只命中了少量相关片段",
        "broad_family_weak_overlap": "当前概览类问题与命中片段的重叠较弱",
    }
    reason_map_en = {
        "empty_hits": "no directly answering paper excerpt was retrieved",
        "target_miss": "retrieval did not directly hit the requested target scope",
        "reference_only_hits": "the current hits lean toward references or peripheral snippets",
        "weak_signal": "the current paper evidence signal is weak",
        "strict_family_without_targeted_support": "this question needs more targeted evidence than the current hits provide",
        "strict_family_weak_overlap": "lexical overlap between the question and evidence is weak",
        "strict_family_sparse_hits": "only a small number of related snippets were retrieved",
        "broad_family_weak_overlap": "overlap between this broad question and the evidence is weak",
    }
    if prefer_zh:
        return reason_map_zh.get(reason_norm, "当前问题在原文中的直接支撑较少")
    return reason_map_en.get(reason_norm, "paper support for this question is limited")


def _build_paper_guide_supplement_evidence_digest(
    *,
    answer_hits: list[dict] | None,
    support_resolution: list[dict] | None,
    max_items: int = 2,
) -> str:
    out: list[str] = []
    seen: set[str] = set()

    def _add(heading: str, snippet: str) -> None:
        head = str(heading or "").strip()
        text = normalize_inline_markdown(str(snippet or ""))
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return
        if len(text) > 220:
            text = text[:220].rsplit(" ", 1)[0].rstrip(" ,;:.") + "..."
        label = f"{head}: {text}" if head else text
        key = label.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(label)

    for rec in list(support_resolution or []):
        if not isinstance(rec, dict):
            continue
        heading = str(rec.get("heading_path") or rec.get("primary_heading_path") or "").strip()
        snippet = (
            rec.get("locate_anchor")
            or rec.get("evidence_quote")
            or rec.get("segment_text")
            or rec.get("anchor_text")
            or ""
        )
        _add(heading, str(snippet or ""))
        if len(out) >= max(1, int(max_items or 2)):
            break

    if len(out) < max(1, int(max_items or 2)):
        for hit in list(answer_hits or []):
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta", {}) or {}
            if not isinstance(meta, dict):
                meta = {}
            heading = str(meta.get("heading_path") or meta.get("top_heading") or "").strip()
            snippet = str(hit.get("text") or "").strip()
            _add(heading, snippet)
            if len(out) >= max(1, int(max_items or 2)):
                break

    return "\n".join(f"- {item}" for item in out[: max(1, int(max_items or 2))])


def _build_paper_guide_llm_supplement_lines(
    *,
    settings_obj,
    answer_hits: list[dict] | None,
    prompt_text: str,
    grounded_answer: str,
    prompt_family: str,
    prefer_zh: bool,
    retrieval_confidence_hint: dict[str, object] | None = None,
    support_resolution: list[dict] | None = None,
) -> str:
    try:
        enabled = bool(int(str(os.environ.get("KB_PAPER_GUIDE_SUPPLEMENT_LLM", "1") or "1")))
    except Exception:
        enabled = True
    if not enabled:
        return ""
    if (not settings_obj) or (not getattr(settings_obj, "api_key", None)):
        return ""

    family = str(prompt_family or "").strip().lower()
    if family in {"abstract", "citation_lookup"}:
        return ""

    answer_text = str(grounded_answer or "").strip()
    if not answer_text:
        return ""

    hint = dict(retrieval_confidence_hint or {})
    reason = _build_paper_guide_supplement_reason_text(
        str(hint.get("low_confidence_reason") or hint.get("force_rescue_reason") or "").strip(),
        prefer_zh=bool(prefer_zh),
    )
    evidence_digest = _build_paper_guide_supplement_evidence_digest(
        answer_hits=list(answer_hits or []),
        support_resolution=list(support_resolution or []),
        max_items=2,
    )
    if not evidence_digest:
        evidence_digest = "- (no stable excerpt digest available)"

    try:
        quick_settings = replace(
            settings_obj,
            timeout_s=max(6.0, min(float(getattr(settings_obj, "timeout_s", 12.0) or 12.0), 12.0)),
            max_retries=0,
        )
    except Exception:
        quick_settings = settings_obj

    llm = DeepSeekChat(quick_settings)
    family_hint_zh = {
        "method": "补方法原理、常见适用边界或为什么会这样设计",
        "reproduce": "补复现时常见的实现假设、依赖条件或风险点",
        "equation": "补公式的直觉解释、变量角色或常见使用前提",
        "figure_walkthrough": "补图示通常表示什么、该怎么读、常见误读点",
        "overview": "补领域背景、问题设定或这类方法通常解决什么矛盾",
        "compare": "补常见 trade-off、适用场景和边界",
        "strength_limits": "补优劣势的通用判断框架",
    }
    family_hint_en = {
        "method": "add intuitive mechanism, common applicability boundaries, or why this design is used",
        "reproduce": "add typical implementation assumptions, dependencies, or reproduction risks",
        "equation": "add intuitive reading of the formula, variable roles, or common prerequisites",
        "figure_walkthrough": "add how such figures are usually read, what they typically indicate, or common misreads",
        "overview": "add domain background, problem framing, or what tension this class of methods usually addresses",
        "compare": "add common trade-offs, applicability range, and boundary conditions",
        "strength_limits": "add a general lens for strengths and limitations",
    }
    family_hint = (
        family_hint_zh.get(family, "补一小段有助于理解的背景、直觉或边界")
        if prefer_zh
        else family_hint_en.get(family, "add a small amount of background, intuition, or boundary context")
    )

    if prefer_zh:
        system = (
            "你只负责写一小段“AI补充理解”，用于帮助用户理解论文问题，但这段内容不是论文原文证据。\n"
            "必须遵守：\n"
            "- 只输出 1-2 条 markdown 列表项，每条都以 '- ' 开头。\n"
            "- 保持中文。\n"
            "- 不要写标题、不要写免责声明、不要写前言后语。\n"
            "- 绝不能说“论文指出/论文证明/原文提到”。\n"
            "- 不要输出引用号、[[CITE:...]]、DOC-k、SID、检索诊断。\n"
            "- 重点做通用背景补充、直觉解释、常见边界，而不是重复当前答案。\n"
        )
        user = (
            f"用户问题：\n{str(prompt_text or '').strip()}\n\n"
            f"当前基于论文证据的回答：\n{answer_text[:900]}\n\n"
            f"为什么需要补充：\n- {reason}\n\n"
            f"当前证据摘要（只用于避免补充内容与论文证据冲突，不代表这些话要被你复述成论文结论）：\n"
            f"{evidence_digest}\n\n"
            f"请补 1-2 条 AI 自己的解释性内容，方向优先：{family_hint}。"
        )
    else:
        system = (
            "You only write a short AI supplemental note that helps the user understand the topic, but it is not paper-grounded evidence.\n"
            "Requirements:\n"
            "- Output only 1-2 markdown bullet lines, each starting with '- '.\n"
            "- Stay in English.\n"
            "- Do not write a title, disclaimer, or intro/outro.\n"
            "- Never say 'the paper states', 'the paper proves', or similar evidence wording.\n"
            "- Do not output citation numbers, [[CITE:...]], DOC-k, SID, or retrieval diagnostics.\n"
            "- Add general context, intuition, or boundary conditions instead of repeating the grounded answer.\n"
        )
        user = (
            f"User question:\n{str(prompt_text or '').strip()}\n\n"
            f"Current paper-grounded answer:\n{answer_text[:900]}\n\n"
            f"Why supplementation is needed:\n- {reason}\n\n"
            f"Evidence digest for consistency only (do not present it as if the paper said your supplement):\n"
            f"{evidence_digest}\n\n"
            f"Write 1-2 AI supplemental bullets focused on: {family_hint}."
        )

    try:
        return str(
            llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3,
                max_tokens=220,
            )
            or ""
        ).strip()
    except Exception:
        return ""


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
    research_answer_plan: str = "",
    paper_guide_contracts_seed: dict | None = None,
    paper_guide_retrieval_confidence_hint: dict[str, object] | None = None,
    settings_obj=None,
) -> dict:
    supplement_builder = None
    if settings_obj and getattr(settings_obj, "api_key", None):
        supplement_builder = lambda **kwargs: _build_paper_guide_llm_supplement_lines(
            settings_obj=settings_obj,
            answer_hits=list(answer_hits or []),
            prompt_text=str(kwargs.get("prompt_text") or ""),
            grounded_answer=str(kwargs.get("grounded_answer") or ""),
            prompt_family=str(kwargs.get("prompt_family") or ""),
            prefer_zh=bool(kwargs.get("prefer_zh")),
            retrieval_confidence_hint=dict(kwargs.get("retrieval_confidence_hint") or {}),
            support_resolution=list(kwargs.get("support_resolution") or []),
        )
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
        research_answer_plan=research_answer_plan,
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
        paper_guide_contracts_seed=dict(paper_guide_contracts_seed or {}),
        paper_guide_retrieval_confidence_hint=dict(paper_guide_retrieval_confidence_hint or {}),
        apply_paper_guide_answer_postprocess=_apply_paper_guide_answer_postprocess,
        maybe_append_library_figure_markdown=_maybe_append_library_figure_markdown,
        validate_structured_citations=_validate_structured_citations,
        build_paper_guide_supplement_lines=supplement_builder,
        validate_freeform_numeric_citations=_validate_freeform_numeric_citations,
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


_PAPER_GUIDE_CROSS_PAPER_REFS_RE = re.compile(
    r"(\bother papers?\b|\bwhich other papers?\b|\bbesides this paper\b|\bin my library\b|\bfrom my library\b|"
    r"除了这篇|哪些论文|哪几篇论文|别的论文|其他论文|库里|文库里)",
    flags=re.I,
)


def _paper_guide_requests_cross_paper_refs(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    return bool(_PAPER_GUIDE_CROSS_PAPER_REFS_RE.search(q))


def _should_allow_refs_async_enrich(
    *,
    refs_async_enabled: bool,
    paper_guide_mode: bool,
    refs_async_in_paper_guide: bool,
    paper_guide_cross_paper_refs: bool,
) -> bool:
    if not refs_async_enabled:
        return False
    if not paper_guide_mode:
        return True
    return bool(refs_async_in_paper_guide or paper_guide_cross_paper_refs)


def _select_refs_async_rebuild_hits_raw(
    *,
    hits_raw: list[dict],
    refs_unscoped_hits_raw: list[dict],
    paper_guide_cross_paper_refs: bool,
) -> list[dict]:
    if paper_guide_cross_paper_refs and refs_unscoped_hits_raw:
        return list(refs_unscoped_hits_raw)
    return list(hits_raw)


def _exclude_bound_source_hits_for_cross_paper_refs(
    hits_raw: list[dict],
    *,
    bound_source_path: str,
    bound_source_name: str,
) -> list[dict]:
    out: list[dict] = []
    for hit in list(hits_raw or []):
        if not isinstance(hit, dict):
            continue
        try:
            if _is_hit_from_bound_source(
                hit,
                bound_source_path=bound_source_path,
                bound_source_name=bound_source_name,
            ):
                continue
        except Exception:
            pass
        out.append(hit)
    return out


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


def _paper_guide_retrieval_confidence_snapshot(
    *,
    scoped_hits: list[dict],
    prompt: str,
    prompt_family: str = "",
) -> dict[str, object]:
    return _retrieval_confidence_snapshot(
        scoped_hits=scoped_hits,
        prompt=prompt,
        prompt_family=prompt_family,
    )


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


# Backward-compat for long-lived local processes that loaded older runtime_state.
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


def _gen_has_running_for_conversation(conv_id: str, *, chat_db_path: object = None) -> bool:
    return _state_gen_has_running_for_conversation(conv_id, chat_db_path=chat_db_path)


def _gen_has_active_task_id(task_id: str) -> bool:
    return _state_gen_has_active_task_id(task_id)


def _gen_store_answer(task: dict, answer: str) -> None:
    return _state_gen_store_answer(task, answer, chat_store_cls=ChatStore)

def _gen_store_partial(task: dict, partial: str) -> None:
    return _state_gen_store_partial(task, partial, chat_store_cls=ChatStore)


def _gen_store_answer_quality_meta(task: dict, *, answer_quality: dict | None) -> None:
    return _state_gen_store_answer_quality_meta(
        task,
        answer_quality=answer_quality,
        chat_store_cls=ChatStore,
    )


def _gen_store_answer_runtime_check_meta(task: dict, *, answer_runtime_check: dict | None) -> None:
    return _state_gen_store_answer_runtime_check_meta(
        task,
        answer_runtime_check=answer_runtime_check,
        chat_store_cls=ChatStore,
    )


def _gen_store_answer_contract_meta(task: dict, *, answer_contract: dict | None) -> None:
    return _state_gen_store_answer_contract_meta(
        task,
        answer_contract=answer_contract,
        chat_store_cls=ChatStore,
    )


def _gen_store_research_trace_meta(task: dict, *, research_trace: dict | None) -> None:
    trace = _trace_compact(dict(research_trace or {}))
    if not trace:
        return
    chat_db = Path(str(task.get("chat_db") or "")).expanduser()
    chat_store = ChatStore(chat_db)
    try:
        amid = int(task.get("assistant_msg_id") or 0)
    except Exception:
        amid = 0
    if amid <= 0:
        return
    try:
        chat_store.merge_message_meta(amid, {"research_trace": trace, "trace_id": str(trace.get("trace_id") or "")})
    except Exception:
        pass


def _gen_compact_agent_trace(agent_trace: dict | None) -> dict:
    return _agent_finalize_compact_trace(agent_trace)


def _gen_agent_source_summary(agent_trace: dict | None) -> dict:
    return _agent_finalize_source_summary(agent_trace)


def _gen_answer_runtime_check(
    task: dict,
    *,
    answer: str,
    answer_quality: dict | None = None,
    agent_trace: dict | None = None,
    agent_source_summary: dict | None = None,
    answer_mode: str = "",
    source_blend: str = "",
    runtime_repair: dict | None = None,
) -> dict:
    return _agent_finalize_answer_runtime_check(
        task,
        answer=answer,
        answer_quality=answer_quality,
        agent_trace=agent_trace,
        agent_source_summary=agent_source_summary,
        answer_mode=answer_mode,
        source_blend=source_blend,
        runtime_repair=runtime_repair,
    )


def _gen_answer_contract(
    task: dict,
    *,
    answer_quality: dict | None = None,
    agent_source_summary: dict | None = None,
    answer_runtime_check: dict | None = None,
) -> dict:
    return _agent_finalize_answer_contract(
        task,
        answer_quality=answer_quality,
        agent_source_summary=agent_source_summary,
        answer_runtime_check=answer_runtime_check,
    )


def _gen_repair_answer_runtime(
    task: dict,
    *,
    prompt: str,
    answer: str,
    answer_quality: dict | None = None,
    agent_trace: dict | None = None,
    agent_source_summary: dict | None = None,
    answer_mode: str = "",
    source_blend: str = "",
) -> dict:
    return _agent_finalize_repair_answer_runtime(
        task,
        prompt=prompt,
        answer=answer,
        answer_quality=answer_quality,
        agent_trace=agent_trace,
        agent_source_summary=agent_source_summary,
        answer_mode=answer_mode,
        source_blend=source_blend,
    )


def _sync_runtime_repaired_answer_contracts(paper_guide_contracts: dict | None, *, answer: str) -> dict:
    return _agent_finalize_sync_runtime_repaired_answer_contracts(paper_guide_contracts, answer=answer)


def _gen_store_agent_trace_meta(task: dict, *, agent_trace: dict | None) -> None:
    return _agent_finalize_store_agent_trace_meta(task, agent_trace=agent_trace, chat_store_cls=ChatStore)


def _gen_store_paper_guide_contract_meta(task: dict, *, paper_guide_contracts: dict | None) -> None:
    return _state_gen_store_paper_guide_contract_meta(
        task,
        paper_guide_contracts=paper_guide_contracts,
        chat_store_cls=ChatStore,
    )


def _gen_store_answer_provenance(
    task: dict,
    *,
    answer: str,
    answer_hits: list[dict],
    support_resolution: list[dict] | None = None,
    primary_evidence: dict | None = None,
) -> None:
    return _state_gen_store_answer_provenance(
        task,
        answer=answer,
        answer_hits=answer_hits,
        support_resolution=support_resolution,
        primary_evidence=primary_evidence,
        chat_store_cls=ChatStore,
        build_answer_provenance=_build_paper_guide_answer_provenance,
    )


def _gen_store_answer_provenance_fast(
    task: dict,
    *,
    answer: str,
    answer_hits: list[dict],
    support_resolution: list[dict] | None = None,
    primary_evidence: dict | None = None,
) -> None:
    return _state_gen_store_answer_provenance_fast(
        task,
        answer=answer,
        answer_hits=answer_hits,
        support_resolution=support_resolution,
        primary_evidence=primary_evidence,
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
    primary_evidence: dict | None = None,
) -> None:
    return _state_gen_store_answer_provenance_async(
        task,
        answer=answer,
        answer_hits=answer_hits,
        support_resolution=support_resolution,
        primary_evidence=primary_evidence,
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
    research_trace: dict = {}
    agent_scope_context: dict = {}
    agent_notes_for_trace: dict = {}
    agent_answer_mode = ""
    agent_generation_result_for_trace: dict = {}
    prompt = ""
    settings_obj = None

    try:
        conv_id = str(task.get("conv_id") or "")
        prompt = str(task.get("prompt") or "").strip()
        selected_research_context = task.get("selected_research_context") if isinstance(task.get("selected_research_context"), dict) else {}
        selected_research_context_items = _selected_research_context_items(selected_research_context)
        selected_research_context_block = _format_selected_research_context_block(selected_research_context)
        selected_research_context_evidence_hits: list[dict] = []
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
        requested_query_scope = _normalize_query_scope(task.get("query_scope"))
        effective_query_scope = _effective_query_scope(
            requested=requested_query_scope,
            paper_guide_mode=paper_guide_mode,
            has_current_paper=bool(paper_guide_bound_source_ready and (paper_guide_bound_source_path or paper_guide_bound_source_name)),
            has_basket=bool(selected_research_context_items),
        )
        paper_guide_source_scoped = bool(paper_guide_mode and effective_query_scope == "current_paper")
        query_scope_block = _query_scope_prompt_block(
            scope=effective_query_scope,
            selected_count=len(selected_research_context_items),
            current_source_name=paper_guide_bound_source_name,
            current_source_path=paper_guide_bound_source_path,
        )
        agent_scope_context = {
            "query_scope": str(effective_query_scope or ""),
            "requested_query_scope": str(requested_query_scope or ""),
            "current_source_path": str(paper_guide_bound_source_path or ""),
            "current_source_name": str(paper_guide_bound_source_name or ""),
            "selected_research_context_count": int(len(selected_research_context_items)),
            "scope_source": "existing_rag",
            "task_id": str(task_id or ""),
            "trace_id": str(task.get("trace_id") or ""),
            "conversation_id": str(conv_id or ""),
            "user_message_id": int(task.get("user_msg_id") or 0),
            "assistant_message_id": int(task.get("assistant_msg_id") or 0),
        }
        research_trace = _trace_new(
            session_id=session_id,
            task_id=task_id,
            conv_id=conv_id,
            user_msg_id=int(task.get("user_msg_id") or 0),
            assistant_msg_id=int(task.get("assistant_msg_id") or 0),
            trace_id=str(task.get("trace_id") or ""),
            prompt_sig=str(task.get("prompt_sig") or ""),
            mode="paper_guide" if paper_guide_mode else "normal",
            started_at=time.time(),
        )

        def _trace_commit() -> None:
            _gen_update_task(session_id, task_id, research_trace=_trace_compact(research_trace))

        def _trace_event(stage: str, *, elapsed_s: float | None = None, **payload) -> None:
            nonlocal research_trace
            research_trace = _trace_add_event(research_trace, stage, elapsed_s=elapsed_s, **payload)
            _trace_commit()

        def _trace_section(section: str, payload: dict | None) -> None:
            nonlocal research_trace
            research_trace = _trace_merge_section(research_trace, section, payload)
            _trace_commit()

        _trace_section(
            "request",
            {
                "top_k": int(top_k),
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "deep_read": bool(deep_read),
                "answer_intent": str(answer_intent or ""),
                "answer_depth": str(answer_depth or ""),
                "answer_output_mode": str(answer_output_mode or ""),
                "query_scope": str(effective_query_scope or ""),
                "requested_query_scope": str(requested_query_scope or ""),
                "paper_guide_bound_source": paper_guide_bound_source_name or paper_guide_bound_source_path,
                "image_attachment_count": int(len(raw_image_atts or [])) if isinstance(raw_image_atts, list) else 0,
                "selected_research_context_count": int(len(selected_research_context_items)),
            },
        )

        image_attachments: list[dict] = []
        if isinstance(raw_image_atts, list):
            for it in raw_image_atts:
                if not isinstance(it, dict):
                    continue
                verified = resolve_verified_chat_image_upload_path(it.get("path"), db_dir=db_dir)
                if verified is None:
                    continue
                p0, mime0 = verified
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
        if paper_guide_source_scoped and paper_guide_bound_source_path:
            try:
                try:
                    from api.routers.library import _md_dir, _pdf_dir

                    prefetch_md_root = _md_dir()
                    prefetch_pdf_root = _pdf_dir()
                except Exception:
                    prefetch_md_root = None
                    prefetch_pdf_root = None
                kickoff_paper_guide_prefetch(
                    source_path=paper_guide_bound_source_path,
                    source_name=paper_guide_bound_source_name,
                    db_dir=db_dir,
                    md_root=prefetch_md_root,
                    pdf_root=prefetch_pdf_root,
                    library_db_path=getattr(settings_obj, "library_db_path", None),
                )
            except Exception:
                pass

        quick_answer = _quick_answer_for_prompt(prompt) if prompt and not selected_research_context_block else None
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
            agent_trace = {}
            if bool(task.get("agent_mode")):
                agent_trace = build_agent_trace_for_completed_answer(
                    prompt,
                    quick_answer,
                    evidence_hits=[],
                    status="done",
                    scope_context=agent_scope_context,
                    agent_notes=agent_notes_for_trace,
                    answer_mode=agent_answer_mode,
                )
            agent_source_summary = _gen_agent_source_summary(agent_trace)
            runtime_repair = _gen_repair_answer_runtime(
                task,
                prompt=prompt,
                answer=quick_answer,
                answer_quality={},
                agent_trace=agent_trace,
                agent_source_summary=agent_source_summary,
                answer_mode=agent_answer_mode,
                source_blend=str(agent_scope_context.get("answer_source_blend") or ""),
            )
            if runtime_repair.get("changed"):
                quick_answer = str(runtime_repair.get("answer") or quick_answer).strip()
                agent_trace = build_agent_trace_for_completed_answer(
                    prompt,
                    quick_answer,
                    evidence_hits=[],
                    status="done",
                    scope_context=agent_scope_context,
                    agent_notes=agent_notes_for_trace,
                    answer_mode=agent_answer_mode,
                )
                _trace_section(
                    "answer",
                    {
                        "runtime_repair": True,
                        "runtime_repair_reasons": list(runtime_repair.get("reasons") or [])[:8],
                    },
                )
            _gen_store_answer(task, quick_answer)
            try:
                _gen_store_answer_provenance_fast(task, answer=quick_answer, answer_hits=[])
            except Exception as exc:
                _perf_log("gen.provenance_inline_fast", ok=0, err=str(exc)[:120])
            _perf_log("gen.quick_answer", total=time.perf_counter() - worker_t0, conv_id=conv_id)
            _trace_section("retrieval", {"bypassed": True, "bypass_reason": "quick_answer"})
            _trace_section("answer", {"chars": len(quick_answer), "quick_answer": True})
            research_trace = _trace_finish(research_trace, status="done", total_elapsed_s=time.perf_counter() - worker_t0)
            _gen_store_research_trace_meta(task, research_trace=research_trace)
            if bool(task.get("agent_mode")):
                _gen_store_agent_trace_meta(task, agent_trace=agent_trace)
            agent_completion_payload = _agent_finalize_completion_payload(
                task,
                answer=quick_answer,
                answer_quality={},
                agent_trace=agent_trace,
                answer_mode=agent_answer_mode,
                source_blend=str(agent_scope_context.get("answer_source_blend") or ""),
                runtime_repair=runtime_repair,
            )
            agent_source_summary = dict(agent_completion_payload.get("agent_source_summary") or {})
            answer_runtime_check = dict(agent_completion_payload.get("answer_runtime_check") or {})
            answer_contract = dict(agent_completion_payload.get("answer_contract") or {})
            _gen_store_answer_runtime_check_meta(task, answer_runtime_check=answer_runtime_check)
            _gen_store_answer_contract_meta(task, answer_contract=answer_contract)
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
                research_trace=research_trace,
                agent_trace=agent_trace,
                agent_source_summary=agent_source_summary,
                answer_runtime_check=answer_runtime_check,
                answer_contract=answer_contract,
                finished_at=time.time(),
            )
            return

        try:
            cur_user_msg_id = int(task.get("user_msg_id") or 0)
        except Exception:
            cur_user_msg_id = 0
        retrieval_prompt = str(prompt or "").strip()
        if query_scope_block:
            retrieval_prompt = f"{retrieval_prompt}\n\n{query_scope_block}".strip()
        if selected_research_context_block:
            retrieval_prompt = (
                f"{retrieval_prompt}\n\n"
                "User-selected research basket excerpts for this turn:\n"
                f"{selected_research_context_block}"
            ).strip()
        allow_implicit_source_hints = _should_apply_implicit_source_hints(
            prompt=retrieval_prompt,
            paper_guide_mode=bool(paper_guide_source_scoped),
        )
        preferred_source_hints: list[str] = []
        if isinstance(preferred_sources_raw, list):
            seen_pref: set[str] = set()
            for it in preferred_sources_raw:
                cand = str(it or "").strip()
                if (not cand) or (cand in seen_pref):
                    continue
                seen_pref.add(cand)
                preferred_source_hints.append(cand)
                if len(preferred_source_hints) >= 6:
                    break
        if paper_guide_source_scoped:
            for cand in (paper_guide_bound_source_path, paper_guide_bound_source_name):
                cand_norm = str(cand or "").strip()
                if (not cand_norm) or (cand_norm in preferred_source_hints):
                    continue
                preferred_source_hints.insert(0, cand_norm)
            if len(preferred_source_hints) > 3:
                preferred_source_hints = preferred_source_hints[:3]
        prompt_multi_source_synthesis = bool(
            prompt_likely_multi_paper_synthesis(prompt or retrieval_prompt or "")
            or len(preferred_source_hints) >= 3
            or effective_query_scope == "library"
            or (effective_query_scope == "basket" and len(selected_research_context_items) > 1)
        )
        if (not paper_guide_source_scoped) and preferred_source_hints:
            retrieval_prompt = _apply_preferred_source_hints(
                retrieval_prompt,
                preferred_source_hints,
                limit=6 if prompt_multi_source_synthesis else 3,
            )
        inferred_source_hint = ""
        if paper_guide_source_scoped and preferred_source_hints:
            retrieval_prompt = _apply_bound_source_hints(retrieval_prompt, preferred_source_hints, limit=2)
        if allow_implicit_source_hints and retrieval_prompt and _needs_conversational_source_hint(retrieval_prompt):
            inferred_source_hint = _pick_recent_source_hint(
                conv_id=conv_id,
                user_msg_id=cur_user_msg_id,
                chat_store=chat_store,
            )
            if inferred_source_hint:
                retrieval_prompt = _augment_prompt_with_source_hint(retrieval_prompt, inferred_source_hint)
        if allow_implicit_source_hints and retrieval_prompt and _needs_bound_source_hint(retrieval_prompt):
            if preferred_source_hints:
                for h in preferred_source_hints[:2]:
                    retrieval_prompt = _augment_prompt_with_source_hint(retrieval_prompt, h)
            else:
                bound_hints = _pick_recent_bound_source_hints(conv_id=conv_id, chat_store=chat_store, limit=2)
                for h in bound_hints:
                    retrieval_prompt = _augment_prompt_with_source_hint(retrieval_prompt, h)
        paper_guide_debug: dict[str, object] = {}
        paper_guide_prompt_family = ""
        if paper_guide_source_scoped:
            paper_guide_prompt_family = _paper_guide_prompt_family(prompt, intent=answer_intent)
            retrieval_prompt = _augment_paper_guide_retrieval_prompt(
                retrieval_prompt,
                family=paper_guide_prompt_family,
                intent=answer_intent,
                output_mode=answer_output_mode,
            )
            paper_guide_debug = {
                "initial_prompt_family": str(paper_guide_prompt_family or ""),
                "answer_intent": str(answer_intent or ""),
                "output_mode": str(answer_output_mode or ""),
                "bound_source_ready": bool(paper_guide_bound_source_ready),
                "query_scope": str(effective_query_scope or ""),
            }
            _gen_update_task(
                session_id,
                task_id,
                paper_guide_debug=dict(paper_guide_debug),
            )

        t_load0 = time.perf_counter()
        chunks = load_all_chunks(db_dir)
        retriever = BM25Retriever(chunks)
        _perf_log("gen.load_retriever", elapsed=time.perf_counter() - t_load0, chunks=len(chunks))
        _trace_event("load_retriever", elapsed_s=time.perf_counter() - t_load0, chunks=len(chunks))

        hits_raw: list[dict] = []
        scores_raw: list[float] = []
        used_query = ""
        used_translation = False
        query_variants: list[str] = []
        paper_guide_retrieval_confidence_hint: dict[str, object] = {}
        hits: list[dict] = []
        grouped_docs: list[dict] = []
        answer_grouped_docs: list[dict] = []
        refs_seed_docs_for_display: list[dict] = []
        refs_unscoped_hits_raw: list[dict] = []
        paper_guide_cross_paper_refs = bool(
            paper_guide_source_scoped
            and paper_guide_bound_source_ready
            and _paper_guide_requests_cross_paper_refs(prompt or retrieval_prompt or "")
        )
        refs_async_will_run = False
        refs_async_seed_docs: list[dict] = []
        prompt_multi_paper_list = False
        seed_refs_should_stay_pending = False
        basket_filter_trace: dict[str, object] = {}
        if prompt and (not bypass_kb):
            _gen_update_task(session_id, task_id, stage="retrieve")
            t_ret0 = time.perf_counter()
            hits_raw, scores_raw, used_query, used_translation, query_variants = _search_hits_with_fallback(
                retrieval_prompt,
                retriever,
                top_k=top_k,
                settings=settings_obj,
                allow_expand=True,
            )
            if paper_guide_cross_paper_refs:
                refs_unscoped_hits_raw = _exclude_bound_source_hits_for_cross_paper_refs(
                    list(hits_raw or []),
                    bound_source_path=paper_guide_bound_source_path,
                    bound_source_name=paper_guide_bound_source_name,
                )
            if paper_guide_source_scoped and (paper_guide_bound_source_path or paper_guide_bound_source_name):
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
                        or paper_guide_prompt_family in {"method", "figure_walkthrough", "reproduce", "compare", "strength_limits", "equation", "overview"}
                    )
                )
                if should_supplement:
                    scan_candidates: list[str] = []
                    scan_seen: set[str] = set()
                    for candidate in (
                        prompt or "",
                        retrieval_prompt or "",
                        (used_query or "") if bool(used_translation) else "",
                        used_query or "",
                    ):
                        cand = str(candidate or "").strip()
                        if not cand:
                            continue
                        cand_key = normalize_match_text(cand)
                        if cand_key in scan_seen:
                            continue
                        scan_seen.add(cand_key)
                        scan_candidates.append(cand)
                        if len(scan_candidates) >= 3:
                            break

                    supplemental_hits: list[dict] = []
                    supplemental_seen: set[str] = set()
                    for scan_prompt in scan_candidates:
                        scan_hits = _paper_guide_targeted_source_block_hits(
                            bound_source_path=paper_guide_bound_source_path,
                            prompt=scan_prompt,
                            db_dir=db_dir,
                            limit=max(2, min(int(top_k or 4), 3)),
                        )
                        for h in list(scan_hits or []):
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
                            if key in supplemental_seen:
                                continue
                            supplemental_seen.add(key)
                            supplemental_hits.append(h)
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
                confidence_snapshot = _paper_guide_retrieval_confidence_snapshot(
                    scoped_hits=scoped_hits,
                    prompt=(prompt or retrieval_prompt or ""),
                    prompt_family=paper_guide_prompt_family,
                )
                should_force_rescue = bool(confidence_snapshot.get("force_rescue"))
                force_rescue_reason = str(confidence_snapshot.get("force_rescue_reason") or "").strip()
                should_confidence_rescue = bool(confidence_snapshot.get("low_confidence")) and (not should_force_rescue)
                try:
                    confidence_rescue_enabled = bool(int(str(os.environ.get("KB_PAPER_GUIDE_CONFIDENCE_RESCUE", "1") or "1")))
                except Exception:
                    confidence_rescue_enabled = True
                should_confidence_rescue = bool(should_confidence_rescue and confidence_rescue_enabled)
                _perf_log(
                    "gen.paper_guide_scope_confidence",
                    phase="pre_rescue",
                    family=paper_guide_prompt_family or str(confidence_snapshot.get("family") or ""),
                    hits=int(confidence_snapshot.get("hit_count") or 0),
                    targeted=int(confidence_snapshot.get("targeted_hit_count") or 0),
                    fallback=int(confidence_snapshot.get("fallback_hit_count") or 0),
                    non_reference=int(bool(confidence_snapshot.get("non_reference_signal"))),
                    strong=int(bool(confidence_snapshot.get("strong_signal"))),
                    overlap=int(confidence_snapshot.get("max_overlap") or 0),
                    max_score=float(confidence_snapshot.get("max_score") or 0.0),
                    forced=int(should_force_rescue),
                    low_conf=int(should_confidence_rescue),
                    reason=force_rescue_reason or str(confidence_snapshot.get("low_confidence_reason") or ""),
                )

                should_run_fallback = bool((should_force_rescue or should_confidence_rescue) and paper_guide_bound_source_path)
                if should_run_fallback:
                    fallback_top_k = max(2, min(int(top_k or 4), 4))
                    if should_confidence_rescue and (not should_force_rescue):
                        fallback_top_k = max(2, min(int(top_k or 4), 3))
                    fallback_hits = _paper_guide_fallback_deepread_hits(
                        bound_source_path=paper_guide_bound_source_path,
                        bound_source_name=paper_guide_bound_source_name,
                        query=(used_query or retrieval_prompt or prompt or ""),
                        prompt=(prompt or retrieval_prompt or ""),
                        prompt_family=paper_guide_prompt_family,
                        top_k=fallback_top_k,
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
                        low_confidence=int(should_confidence_rescue),
                        applied=int(bool(fallback_hits)),
                        reason=force_rescue_reason or str(confidence_snapshot.get("low_confidence_reason") or ""),
                    )
                    if fallback_hits:
                        confidence_snapshot = _paper_guide_retrieval_confidence_snapshot(
                            scoped_hits=scoped_hits,
                            prompt=(prompt or retrieval_prompt or ""),
                            prompt_family=paper_guide_prompt_family,
                        )
                        _perf_log(
                            "gen.paper_guide_scope_confidence",
                            phase="post_rescue",
                            family=paper_guide_prompt_family or str(confidence_snapshot.get("family") or ""),
                            hits=int(confidence_snapshot.get("hit_count") or 0),
                            targeted=int(confidence_snapshot.get("targeted_hit_count") or 0),
                            fallback=int(confidence_snapshot.get("fallback_hit_count") or 0),
                            non_reference=int(bool(confidence_snapshot.get("non_reference_signal"))),
                            strong=int(bool(confidence_snapshot.get("strong_signal"))),
                            overlap=int(confidence_snapshot.get("max_overlap") or 0),
                            max_score=float(confidence_snapshot.get("max_score") or 0.0),
                            forced=int(bool(confidence_snapshot.get("force_rescue"))),
                            low_conf=int(bool(confidence_snapshot.get("low_confidence"))),
                            reason=str(confidence_snapshot.get("force_rescue_reason") or confidence_snapshot.get("low_confidence_reason") or ""),
                        )
                if confidence_snapshot:
                    paper_guide_retrieval_confidence_hint = dict(confidence_snapshot)
                if len(scoped_hits) != len(hits_raw):
                    _perf_log(
                        "gen.paper_guide_scope",
                        before=len(hits_raw),
                        after=len(scoped_hits),
                        source=paper_guide_bound_source_name or paper_guide_bound_source_path,
                    )
                hits_raw = scoped_hits
                scores_raw = [float(h.get("score", 0.0) or 0.0) for h in hits_raw]
            if effective_query_scope == "basket":
                basket_scoped_hits, basket_filter_trace = _filter_hits_for_selected_research_context(
                    hits_raw,
                    selected_research_context_items,
                )
                if len(basket_scoped_hits) != len(hits_raw):
                    _perf_log(
                        "gen.basket_scope",
                        before=len(hits_raw),
                        after=len(basket_scoped_hits),
                        mode=str(basket_filter_trace.get("mode") or ""),
                        constraints=int(basket_filter_trace.get("constraint_count") or 0),
                    )
                hits_raw = basket_scoped_hits
                scores_raw = [float(h.get("score", 0.0) or 0.0) for h in hits_raw]
            _perf_log(
                "gen.retrieve",
                elapsed=time.perf_counter() - t_ret0,
                hits_raw=len(hits_raw),
                translated=bool(used_translation),
            )
            _trace_section(
                "retrieval",
                {
                    "bypassed": False,
                    "used_query": str(used_query or ""),
                    "used_translation": bool(used_translation),
                    "query_variants": list(query_variants or []),
                    "raw_hit_count": int(len(hits_raw or [])),
                    "top_hits": _trace_summarize_hits(list(hits_raw or []), limit=6),
                    "paper_guide_cross_paper_refs": bool(paper_guide_cross_paper_refs),
                    "basket_filter": dict(basket_filter_trace or {}),
                },
            )
            _trace_event(
                "retrieve",
                elapsed_s=time.perf_counter() - t_ret0,
                hit_count=int(len(hits_raw or [])),
                query_variant_count=int(len(query_variants or [])),
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
            if refs_unscoped_hits_raw:
                try:
                    cross_paper_grouped_docs = _group_hits_by_doc_for_refs(
                        refs_unscoped_hits_raw,
                        prompt_text=retrieval_prompt,
                        top_k_docs=top_k,
                        deep_query=(used_query or retrieval_prompt or prompt or ""),
                        deep_read=False,
                        llm_rerank=False,
                        settings=settings_obj,
                    )
                    if cross_paper_grouped_docs:
                        grouped_docs = cross_paper_grouped_docs
                        _perf_log(
                            "gen.paper_guide_cross_paper_refs_seed",
                            docs=len(grouped_docs),
                            source=paper_guide_bound_source_name or paper_guide_bound_source_path,
                        )
                except Exception:
                    pass
            answer_hit_limit = max(1, min(int(top_k), 4))
            guide_strict_mode = bool(paper_guide_source_scoped and paper_guide_bound_source_ready)
            answer_doc_cap = max(
                answer_hit_limit,
                min(
                    int(top_k),
                    6
                    if prompt_multi_source_synthesis
                    else (
                        5
                        if (
                            guide_strict_mode
                            and paper_guide_prompt_family in {"overview", "compare", "reproduce", "strength_limits", "figure_walkthrough"}
                        )
                        else (4 if guide_strict_mode else 3)
                    ),
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

            prompt_multi_paper_list = bool(
                effective_query_scope == "library"
                or prompt_explicitly_requests_multi_paper_list(prompt or retrieval_prompt or "")
            )
            try:
                refs_async_enabled = bool(int(str(os.environ.get("KB_REFS_ASYNC_ENRICH", "1") or "1")))
            except Exception:
                refs_async_enabled = True
            try:
                refs_async_in_paper_guide = bool(int(str(os.environ.get("KB_REFS_ASYNC_ENRICH_IN_PAPER_GUIDE", "0") or "0")))
            except Exception:
                refs_async_in_paper_guide = False
            allow_refs_async = _should_allow_refs_async_enrich(
                refs_async_enabled=refs_async_enabled,
                paper_guide_mode=bool(paper_guide_source_scoped),
                refs_async_in_paper_guide=refs_async_in_paper_guide,
                paper_guide_cross_paper_refs=paper_guide_cross_paper_refs,
            )
            refs_seed_docs_for_display = _select_multi_paper_seed_docs_for_display(
                prompt_multi_paper_list=bool(prompt_multi_paper_list),
                paper_guide_cross_paper_refs=bool(paper_guide_cross_paper_refs),
                answer_grouped_docs=list(answer_grouped_docs or []),
                grouped_docs=list(grouped_docs or []),
            )
            if prompt_multi_paper_list and refs_seed_docs_for_display:
                refs_seed_docs_for_display = _filter_multi_paper_seed_docs_for_display(
                    prompt=prompt or retrieval_prompt or "",
                    seed_docs=refs_seed_docs_for_display,
                )

            refs_async_will_run = bool(
                allow_refs_async
                and llm_rerank
                and prompt
                and refs_seed_docs_for_display
                and settings_obj
                and getattr(settings_obj, "api_key", None)
            )

            seed_refs_should_stay_pending = bool(prompt_multi_paper_list)
            if seed_refs_should_stay_pending and refs_seed_docs_for_display:
                try:
                    for d in refs_seed_docs_for_display:
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
                    refs_async_seed_docs = copy.deepcopy(refs_seed_docs_for_display)
                except Exception:
                    refs_async_seed_docs = list(refs_seed_docs_for_display)
            _trace_section(
                "refs",
                {
                    "seed_count": int(len(refs_seed_docs_for_display or [])),
                    "seed_sources": _trace_summarize_hits(list(refs_seed_docs_for_display or []), limit=6),
                    "grouped_doc_count": int(len(grouped_docs or [])),
                    "answer_grouped_doc_count": int(len(answer_grouped_docs or [])),
                    "async_will_run": bool(refs_async_will_run),
                    "async_seed_count": int(len(refs_async_seed_docs or [])),
                    "pending_seed": bool(seed_refs_should_stay_pending),
                },
            )
        else:
            _trace_section(
                "retrieval",
                {
                    "bypassed": True,
                    "bypass_reason": (
                        "image_first_prompt"
                        if image_first_prompt
                        else ("general_request" if bypass_kb else "image_only")
                    ),
                },
            )
            _trace_event("retrieve_skipped", elapsed_s=0.0, reason=str((_trace_compact(research_trace).get("retrieval") or {}).get("bypass_reason") or ""))
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
                    hits=list(refs_seed_docs_for_display or []),
                    scores=list(scores_raw or []),
                    used_query=str(used_query or ""),
                    used_translation=bool(used_translation),
                    render_status="pending" if seed_refs_should_stay_pending else None,
                    render_error="" if seed_refs_should_stay_pending else None,
                    render_error_detail="" if seed_refs_should_stay_pending else None,
                    query_variants=list(query_variants or []),
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
            if prompt_multi_paper_list:
                try:
                    chat_store.set_message_refs_render_state(
                        user_msg_id=umid,
                        render_status="pending",
                        render_error="",
                        render_error_detail="",
                        render_attempts=1,
                    )
                except Exception:
                    pass

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
                    if not prompt_multi_paper_list:
                        return
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
                            render_status="pending",
                            render_error="",
                            render_error_detail="",
                            render_attempts=1,
                        )
                    except Exception:
                        pass

                seed_docs = list(refs_async_seed_docs)[:refs_async_top_k_docs]
                try:
                    rebuild_hits_raw = _select_refs_async_rebuild_hits_raw(
                        hits_raw=hits_raw,
                        refs_unscoped_hits_raw=refs_unscoped_hits_raw,
                        paper_guide_cross_paper_refs=paper_guide_cross_paper_refs,
                    )
                    if rebuild_hits_raw:
                        t_rebuild0 = time.perf_counter()
                        rebuilt_docs = _group_hits_by_doc_for_refs(
                            rebuild_hits_raw,
                            prompt_text=retrieval_prompt,
                            top_k_docs=refs_async_top_k_docs,
                            deep_query=(used_query or retrieval_prompt or prompt or ""),
                            deep_read=True,
                            llm_rerank=False,
                            settings=settings_obj,
                        )
                        if rebuilt_docs:
                            if not prompt_multi_paper_list:
                                seed_docs = rebuilt_docs
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
                        progress_cb=_push_partial if prompt_multi_paper_list else None,
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
                    rendered_payload = None
                    rendered_payload_sig = ""
                    render_status = "full"
                    render_error = ""
                    render_error_detail = ""
                    render_built_at = time.time()
                    render_evidence_sig = ""
                    try:
                        authoritative_doc_list = []
                        if prompt_multi_paper_list:
                            authoritative_doc_list = _await_stored_doc_list_contract(
                                chat_db=chat_db,
                                conv_id=conv_id,
                                assistant_msg_id=int(task.get("assistant_msg_id") or 0),
                                wait_timeout_s=0.0,
                            )
                            if not authoritative_doc_list:
                                authoritative_doc_list = _await_stored_doc_list_contract(
                                    chat_db=chat_db,
                                    conv_id=conv_id,
                                    assistant_msg_id=int(task.get("assistant_msg_id") or 0),
                                    wait_timeout_s=1.6,
                                )
                            if not authoritative_doc_list:
                                authoritative_doc_list = _rebuild_multi_paper_doc_list_contract_from_available_refs(
                                    prompt=prompt or used_query or retrieval_prompt or "",
                                    seed_docs=list(enriched or seed_docs or []),
                                    answer_hits=list(enriched or seed_docs or []),
                                    evidence_cards=[],
                                    exclude_bound_source=bool(paper_guide_cross_paper_refs),
                                    bound_source_path=str(paper_guide_bound_source_path or ""),
                                    bound_source_name=str(paper_guide_bound_source_name or ""),
                                )
                        if prompt_multi_paper_list:
                            rendered_payload, rendered_payload_sig = _build_doc_list_refs_render_payload(
                                user_msg_id=umid,
                                prompt=prompt,
                                prompt_sig=str(task.get("prompt_sig") or ""),
                                hits=list(enriched or []),
                                scores=list(scores_raw or []),
                                used_query=str(used_query or ""),
                                used_translation=bool(used_translation),
                                doc_list=list(authoritative_doc_list or []),
                                guide_mode=bool(paper_guide_source_scoped),
                                guide_source_path=str(paper_guide_bound_source_path or "") if paper_guide_source_scoped else "",
                                guide_source_name=str(paper_guide_bound_source_name or "") if paper_guide_source_scoped else "",
                            )
                        else:
                            rendered_payload, rendered_payload_sig = _build_precomputed_refs_render_payload(
                                user_msg_id=umid,
                                prompt=prompt,
                                prompt_sig=str(task.get("prompt_sig") or ""),
                                hits=list(enriched or []),
                                scores=list(scores_raw or []),
                                used_query=str(used_query or ""),
                                used_translation=bool(used_translation),
                                guide_mode=bool(paper_guide_source_scoped),
                                guide_source_path=str(paper_guide_bound_source_path or "") if paper_guide_source_scoped else "",
                                guide_source_name=str(paper_guide_bound_source_name or "") if paper_guide_source_scoped else "",
                                library_db_path=getattr(settings_obj, "library_db_path", None),
                            )
                    except Exception as exc:
                        rendered_payload = None
                        rendered_payload_sig = ""
                        render_status = "failed"
                        render_error = "render_build_failed"
                        render_error_detail = f"{type(exc).__name__}: {str(exc or '').strip()}"[:500]
                        render_built_at = 0.0
                    if render_status == "full" and ((not isinstance(rendered_payload, dict)) or (not rendered_payload) or (not str(rendered_payload_sig or "").strip())):
                        render_status = "failed"
                        render_error = "render_payload_empty"
                        render_error_detail = "Precomputed refs render returned empty payload or signature."
                        render_built_at = 0.0
                        render_evidence_sig = ""
                    else:
                        render_evidence_sig = str(rendered_payload_sig or "").strip()
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
                            rendered_payload=rendered_payload,
                            rendered_payload_sig=rendered_payload_sig,
                            render_status=render_status,
                            render_error=render_error,
                            render_error_detail=render_error_detail,
                            render_built_at=render_built_at,
                            render_attempts=1,
                            render_evidence_sig=render_evidence_sig,
                        )
                    except Exception:
                        pass
                    refs_async_state = "done" if render_status == "full" else ("awaiting_authoritative_doc_list" if render_status == "pending" else "failed")
                    _gen_update_task(
                        session_id,
                        task_id,
                        refs_async_pending=False,
                        refs_async_state=refs_async_state,
                        refs_async_docs=int(len(enriched)),
                    )
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
                    try:
                        cs = ChatStore(chat_db)
                        cs.set_message_refs_render_state(
                            user_msg_id=umid,
                            render_status="failed",
                            render_error="refs_enrich_empty",
                            render_error_detail="Async refs enrich produced no enriched docs.",
                            render_attempts=1,
                        )
                    except Exception:
                        pass
                    _gen_update_task(session_id, task_id, refs_async_pending=False, refs_async_state="empty")
                _finalize_task_after_refs_async()

            try:
                threading.Thread(target=_bg_enrich_refs, daemon=True).start()
            except Exception:
                try:
                    chat_store.set_message_refs_render_state(
                        user_msg_id=umid,
                        render_status="failed",
                        render_error="refs_async_thread_start_failed",
                        render_error_detail="Failed to start async refs thread.",
                        render_attempts=1,
                    )
                except Exception:
                    pass
                _gen_update_task(session_id, task_id, refs_async_pending=False, refs_async_state="error")
                _finalize_task_after_refs_async()

        _gen_update_task(session_id, task_id, stage="context", used_query=str(used_query or ""), used_translation=bool(used_translation), refs_done=True)

        # Keep prompt compact for fast first-token latency.
        answer_hit_limit = max(
            1,
            min(
                int(top_k),
                6
                if prompt_multi_source_synthesis
                else (
                    5
                    if (
                        paper_guide_source_scoped
                        and paper_guide_prompt_family in {"overview", "compare", "reproduce", "strength_limits", "figure_walkthrough", "citation_lookup"}
                    )
                    else 4
                ),
            ),
        )
        answer_seed = _select_answer_seed_for_generation(
            paper_guide_cross_paper_refs=bool(paper_guide_cross_paper_refs),
            answer_grouped_docs=list(answer_grouped_docs or []),
            grouped_docs=list(grouped_docs or []),
            heading_hits=list(hits or []),
        )
        if paper_guide_source_scoped and paper_guide_bound_source_ready:
            heading_hits_for_answer = list(grouped_docs or []) if paper_guide_cross_paper_refs else list(hits or [])
            if not paper_guide_cross_paper_refs:
                raw_block_hits_for_answer = [
                    dict(hit)
                    for hit in list(hits_raw or [])
                    if isinstance(hit, dict)
                    and bool((hit.get("meta") or {}).get("paper_guide_targeted_block") or (hit.get("meta") or {}).get("paper_guide_fallback"))
                ]
                if raw_block_hits_for_answer:
                    heading_hits_for_answer = [*raw_block_hits_for_answer, *heading_hits_for_answer]
            grouped_hits_for_answer = list(answer_seed or [])
            raw_target_hits = []
            if not paper_guide_cross_paper_refs:
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
        if selected_research_context_items:
            selected_research_context_evidence_hits = _selected_research_context_evidence_hits(
                selected_research_context_items,
                max_hits=min(4, max(1, len(selected_research_context_items))),
            )
            if selected_research_context_evidence_hits:
                merged_answer_hit_limit = max(
                    answer_hit_limit,
                    min(8, answer_hit_limit + len(selected_research_context_evidence_hits)),
                )
                answer_hits = _merge_selected_research_context_evidence_hits(
                    answer_hits,
                    selected_research_context_evidence_hits,
                    limit=merged_answer_hit_limit,
                )
                _trace_section(
                    "basket_context",
                    {
                        "evidence_hit_count": int(len(selected_research_context_evidence_hits)),
                        "merged_answer_hit_count": int(len(answer_hits or [])),
                        "evidence_sources": _trace_summarize_hits(list(selected_research_context_evidence_hits or []), limit=6),
                    },
                )
        anchor_grounded_answer = _has_anchor_grounded_answer_hits(answer_hits)
        answer_output_mode = _detect_answer_output_mode(
            prompt,
            answer_output_mode_hint=answer_output_mode_hint,
            answer_mode_hint=answer_mode_hint,
            paper_guide_mode=bool(paper_guide_source_scoped),
            intent=answer_intent,
            anchor_grounded=anchor_grounded_answer,
        )
        if paper_guide_source_scoped:
            answer_output_mode = _stabilize_paper_guide_output_mode(
                answer_output_mode,
                prompt=prompt,
                intent=answer_intent,
                explicit_hint=(answer_output_mode_hint or answer_mode_hint),
            )
        locked_source_candidates = list(answer_seed or answer_hits)
        if effective_query_scope == "basket" and selected_research_context_evidence_hits:
            locked_source_candidates = [*selected_research_context_evidence_hits, *locked_source_candidates]
        locked_citation_source = _pick_locked_citation_source(locked_source_candidates)
        answer_hits = _ensure_locked_source_in_answer_hits(
            answer_hits,
            source_rec=locked_citation_source,
            seed_docs=list(answer_seed or []),
            top_n=answer_hit_limit,
        )
        _trace_section(
            "answer",
            {
                "answer_hit_count": int(len(answer_hits or [])),
                "answer_sources": _trace_summarize_hits(list(answer_hits or []), limit=6),
                "anchor_grounded": bool(anchor_grounded_answer),
                "locked_source": str((locked_citation_source or {}).get("source_name") or (locked_citation_source or {}).get("source_path") or ""),
                "output_mode": str(answer_output_mode or ""),
            },
        )
        if bool(task.get("agent_mode")):
            try:
                agent_bridge = build_generation_agent_notes(
                    prompt,
                    evidence_hits=list(answer_hits or []),
                    candidate_hits=list(hits_raw or hits or answer_hits or []),
                    scope_context=agent_scope_context,
                )
                agent_bridge_context = dict(agent_bridge.get("context") or {})
                if answer_hits:
                    hybrid_recommended = bool(agent_bridge.get("hybrid_generation_recommended")) and not bool(paper_guide_source_scoped)
                    if hybrid_recommended:
                        agent_notes_for_trace = dict(agent_bridge.get("agent_notes") or {})
                        agent_scope_context.update(agent_bridge_context)
                        agent_answer_mode = str(agent_scope_context.get("answer_mode") or "hybrid_local_external")
                    else:
                        agent_answer_mode = "evidence_grounded"
                        agent_notes_for_trace = {}
                        agent_scope_context.update(
                            {
                                "planner_intent": agent_bridge_context.get("planner_intent") or {},
                                "planner_confidence": str(agent_bridge_context.get("planner_confidence") or ""),
                                "evidence_need": str(agent_bridge_context.get("evidence_need") or ""),
                                "retrieved_hit_count": int(
                                    agent_bridge_context.get("retrieved_hit_count") or len(hits_raw or answer_hits or [])
                                ),
                                "usable_hit_count": int(len(answer_hits or [])),
                                "retrieval_confidence": str(agent_bridge_context.get("retrieval_confidence") or ""),
                                "answer_source_blend": "local_grounded",
                                "answer_mode": agent_answer_mode,
                                "source_policy": "local_only",
                                "hybrid_generation_recommended": False,
                            }
                        )
                else:
                    agent_notes_for_trace = dict(agent_bridge.get("agent_notes") or {})
                    agent_scope_context.update(agent_bridge_context)
                    agent_answer_mode = str(agent_scope_context.get("answer_mode") or "")
                _trace_section(
                    "agent",
                    {
                        "mode": "research_agent",
                        "answer_mode": str(agent_scope_context.get("answer_mode") or agent_answer_mode or ""),
                        "source_policy": str(agent_scope_context.get("source_policy") or ""),
                        "answer_source_blend": str(agent_scope_context.get("answer_source_blend") or ""),
                        "retrieval_confidence": str(agent_scope_context.get("retrieval_confidence") or ""),
                        "usable_hit_count": int(agent_scope_context.get("usable_hit_count") or 0),
                        "hybrid_generation_recommended": bool(agent_scope_context.get("hybrid_generation_recommended")),
                    },
                )
            except Exception as exc:
                agent_scope_context["agent_bridge_error"] = str(exc)[:180]
                _trace_section("agent", {"mode": "research_agent", "bridge_error": str(exc)[:180]})
        paper_guide_context_records = _build_paper_guide_context_records(
            answer_hits,
            paper_guide_mode=bool(paper_guide_source_scoped),
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
            paper_guide_mode=bool(paper_guide_source_scoped),
            paper_guide_bound_source_ready=bool(paper_guide_source_scoped and paper_guide_bound_source_ready),
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
        paper_guide_reference_opportunities_block = str(paper_guide_prompt_context.get("paper_guide_reference_opportunities_block") or "")
        paper_guide_reference_opportunities = list(paper_guide_prompt_context.get("paper_guide_reference_opportunities") or [])
        citation_plan = dict(paper_guide_prompt_context.get("citation_plan") or {})
        citation_plan_block = str(paper_guide_prompt_context.get("citation_plan_block") or "")
        paper_guide_candidate_refs_by_source = dict(paper_guide_prompt_context.get("paper_guide_candidate_refs_by_source") or {})
        paper_guide_support_slots = list(paper_guide_prompt_context.get("paper_guide_support_slots") or [])
        paper_guide_contracts_seed = dict(paper_guide_prompt_context.get("paper_guide_contracts_seed") or {})
        selected_research_context_evidence_contract = _selected_research_context_evidence_contract(
            selected_research_context_evidence_hits
        )
        if selected_research_context_evidence_contract:
            paper_guide_contracts_seed["research_basket_evidence"] = dict(selected_research_context_evidence_contract)
        _trace_section(
            "citation_systems",
            {
                "system_b_opportunity_count": int(len(paper_guide_reference_opportunities or [])),
                "system_a_support_slot_count": int(len(paper_guide_support_slots or [])),
                "evidence_card_count": int(len(paper_guide_evidence_cards or [])),
                "candidate_ref_source_count": int(len(paper_guide_candidate_refs_by_source or {})),
                "citation_plan_intent": str(citation_plan.get("intent") or ""),
                "citation_plan_budget": dict(citation_plan.get("budget") or {}) if isinstance(citation_plan.get("budget"), dict) else {},
            },
        )
        if prompt_multi_paper_list and refs_seed_docs_for_display:
            paper_guide_contracts_seed["doc_list_seed"] = [dict(item) for item in list(refs_seed_docs_for_display or []) if isinstance(item, dict)]
        paper_guide_support_resolution: list[dict] = []
        paper_guide_direct_source_path = (
            str(paper_guide_prompt_context.get("paper_guide_direct_source_path") or paper_guide_bound_source_path or "")
            if paper_guide_source_scoped
            else ""
        )
        paper_guide_focus_source_path = (
            str(paper_guide_prompt_context.get("paper_guide_focus_source_path") or paper_guide_bound_source_path or "")
            if paper_guide_source_scoped
            else ""
        )
        if paper_guide_mode:
            paper_guide_debug.update(
                {
                    "special_focus_present": bool(paper_guide_special_focus_block),
                    "special_focus_prefix": str(paper_guide_special_focus_block or "")[:120],
                    "support_slots_count": int(len(paper_guide_support_slots or [])),
                    "reference_opportunities_count": int(len(paper_guide_reference_opportunities or [])),
                    "focus_source_path": str(paper_guide_focus_source_path or ""),
                    "direct_source_path": str(paper_guide_direct_source_path or ""),
                }
            )
            _gen_update_task(
                session_id,
                task_id,
                paper_guide_debug=dict(paper_guide_debug),
            )
        prompt_bundle = _build_generation_prompt_bundle(
            prompt=prompt,
            ctx=ctx,
            paper_guide_mode=bool(paper_guide_source_scoped),
            paper_guide_bound_source_ready=bool(paper_guide_source_scoped and paper_guide_bound_source_ready),
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
            paper_guide_reference_opportunities_block=paper_guide_reference_opportunities_block,
            citation_plan_block=citation_plan_block,
            image_attachment_count=len(image_attachments or []),
        )
        system = str(prompt_bundle.get("system") or "")
        user = str(prompt_bundle.get("user") or "")
        prompt_for_user = str(prompt_bundle.get("prompt_for_user") or prompt or "[Image attachment only request]")
        paper_guide_contract_enabled = bool(prompt_bundle.get("paper_guide_contract_enabled"))
        research_answer_plan = str(prompt_bundle.get("research_answer_plan") or "").strip()
        if research_answer_plan:
            _trace_section("answer", {"research_answer_plan": research_answer_plan})
            if paper_guide_mode:
                paper_guide_debug["research_answer_plan"] = research_answer_plan
        if query_scope_block:
            user = f"{user.rstrip()}\n\n{query_scope_block}".strip()
        if selected_research_context_block:
            user = (
                f"{user.rstrip()}\n\n"
                "USER-SELECTED RESEARCH BASKET CONTEXT:\n"
                "The user explicitly selected these excerpts for this turn. Use them as supplemental working context; "
                "do not invent bibliographic facts beyond the fields shown, and keep citations/evidence grounded in available sources.\n"
                f"{selected_research_context_block}"
            ).strip()
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
        user_content = _build_multimodal_user_content(
            user,
            image_attachments,
            allowed_image_roots=chat_image_upload_roots(db_dir),
        )
        messages = _build_generation_messages(system=system, hist=hist, user_content=user_content)
        ds = None
        agent_direct_answer_override = ""
        agent_generation_result: dict = {}
        agent_generation_hits = list(answer_hits or []) if agent_answer_mode == "hybrid_local_external" else []
        agent_generation_enabled = bool(
            bool(task.get("agent_mode"))
            and prompt
            and not image_attachments
            and (
                (not answer_hits and agent_answer_mode in {"general_llm", "external_academic_llm"})
                or (
                    bool(agent_generation_hits)
                    and agent_answer_mode == "hybrid_local_external"
                    and bool(agent_scope_context.get("hybrid_generation_recommended"))
                )
            )
        )
        if agent_generation_enabled:
            try:
                agent_generation_result = agent_generate_grounded_answer(
                    prompt,
                    agent_generation_hits,
                    settings=settings_obj,
                    history=hist,
                    agent_notes=agent_notes_for_trace,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                agent_direct_answer_override = str(agent_generation_result.get("answer") or "").strip()
                if agent_direct_answer_override:
                    agent_generation_result_for_trace = {
                        key: value
                        for key, value in dict(agent_generation_result or {}).items()
                        if key not in {"answer", "hits"}
                    }
                    agent_scope_context["agent_llm_used"] = bool(agent_generation_result.get("llm_used"))
                    agent_scope_context["web_search_used"] = bool(agent_generation_result.get("web_search_used"))
                    if agent_generation_result.get("answer_mode"):
                        agent_answer_mode = str(agent_generation_result.get("answer_mode") or agent_answer_mode)
                        agent_scope_context["answer_mode"] = agent_answer_mode
                    if agent_generation_result.get("source_blend"):
                        agent_scope_context["answer_source_blend"] = str(agent_generation_result.get("source_blend") or "")
                    if agent_answer_mode == "hybrid_local_external":
                        agent_scope_context["source_policy"] = "local_plus_external_background"
                    _trace_section(
                        "agent",
                        {
                            "mode": "research_agent",
                            "fallback_generation": True,
                            "hybrid_generation": bool(agent_generation_hits),
                            "answer_mode": str(agent_answer_mode or ""),
                            "llm_used": bool(agent_generation_result.get("llm_used")),
                            "web_search_used": bool(agent_generation_result.get("web_search_used")),
                            "quality_gate": dict(agent_generation_result.get("quality_gate") or {})
                            if isinstance(agent_generation_result.get("quality_gate"), dict)
                            else {},
                            "observation": str(agent_generation_result.get("observation") or "")[:240],
                        },
                    )
            except Exception as exc:
                agent_scope_context["agent_generation_error"] = str(exc)[:180]
                _trace_section("agent", {"mode": "research_agent", "generation_error": str(exc)[:180]})
        direct_answer_override = ""
        if not agent_direct_answer_override:
            ds = DeepSeekChat(settings_obj)
            direct_answer_override = _build_paper_guide_direct_answer_override(
                paper_guide_mode=bool(paper_guide_source_scoped),
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
        if paper_guide_mode:
            paper_guide_debug.update(
                {
                    "prompt_for_user": str(prompt_for_user or ""),
                    "direct_answer_override_used": bool(
                        str(agent_direct_answer_override or direct_answer_override or "").strip()
                    ),
                    "direct_answer_override_prefix": str(
                        agent_direct_answer_override or direct_answer_override or ""
                    ).strip()[:120],
                }
            )
            _gen_update_task(
                session_id,
                task_id,
                paper_guide_debug=dict(paper_guide_debug),
            )
        partial = ""
        streamed = False
        last_store_ts = 0.0
        last_store_len = 0
        t_answer0 = time.perf_counter()
        if agent_direct_answer_override or direct_answer_override:
            partial = str(agent_direct_answer_override or direct_answer_override or "").strip()
            _gen_update_task(session_id, task_id, stage="answer", partial=partial, char_count=len(partial))
            _gen_store_partial(task, partial)
        else:
            try:
                if ds is None:
                    ds = DeepSeekChat(settings_obj)
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
                if _gen_should_cancel(session_id, task_id):
                    raise RuntimeError("canceled")
                should_retry_non_stream = (
                    not streamed
                    or _looks_like_incomplete_stream_partial(
                        partial,
                        paper_guide_mode=bool(paper_guide_source_scoped),
                        prompt_family=paper_guide_prompt_family,
                        has_hits=bool(answer_hits),
                    )
                )
                if should_retry_non_stream:
                    try:
                        resp = ds.chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
                    except Exception:
                        if not streamed:
                            raise
                    else:
                        fallback = str(resp or "").strip()
                        if fallback:
                            partial = fallback
                            _gen_update_task(
                                session_id,
                                task_id,
                                stage="answer",
                                partial=partial,
                                char_count=len(partial),
                                stream_fallback_used=True,
                            )
                        elif not streamed:
                            partial = ""
                            _gen_update_task(session_id, task_id, stage="answer", partial=partial, char_count=0)
            else:
                pass

        _trace_event(
            "llm_answer",
            elapsed_s=time.perf_counter() - t_answer0,
            chars=int(len(partial or "")),
            streamed=bool(streamed),
            direct_override=bool(direct_answer_override),
        )
        if _gen_should_cancel(session_id, task_id):
            answer = (str(partial or "").strip() + "\n\n(Generation canceled)").strip() or "(Generation canceled)"
            _gen_store_answer(task, answer)
            research_trace = _trace_finish(research_trace, status="canceled", total_elapsed_s=time.perf_counter() - worker_t0)
            _gen_store_research_trace_meta(task, research_trace=research_trace)
            _gen_update_task(
                session_id,
                task_id,
                status="canceled",
                stage="canceled",
                answer=answer,
                partial=answer,
                char_count=len(answer),
                research_trace=research_trace,
                finished_at=time.time(),
            )
            return

        t_finalize0 = time.perf_counter()
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
            paper_guide_mode=bool(paper_guide_source_scoped),
            paper_guide_contract_enabled=paper_guide_contract_enabled,
            paper_guide_prompt_family=paper_guide_prompt_family,
            paper_guide_special_focus_block=paper_guide_special_focus_block,
            paper_guide_focus_source_path=paper_guide_focus_source_path,
            paper_guide_direct_source_path=paper_guide_direct_source_path,
            paper_guide_bound_source_path=paper_guide_bound_source_path,
            paper_guide_candidate_refs_by_source=paper_guide_candidate_refs_by_source,
            paper_guide_support_slots=paper_guide_support_slots,
            paper_guide_evidence_cards=paper_guide_evidence_cards,
            research_answer_plan=research_answer_plan,
            paper_guide_contracts_seed=paper_guide_contracts_seed,
            paper_guide_retrieval_confidence_hint=paper_guide_retrieval_confidence_hint,
            settings_obj=settings_obj,
        )
        _trace_event("finalize_answer", elapsed_s=time.perf_counter() - t_finalize0)
        answer = str(finalize_state.get("answer") or "")
        paper_guide_support_resolution = list(finalize_state.get("paper_guide_support_resolution") or [])
        paper_guide_contracts = dict(finalize_state.get("paper_guide_contracts") or {})
        if selected_research_context_items:
            paper_guide_contracts["selected_research_context"] = dict(selected_research_context or {})
        if selected_research_context_evidence_contract:
            paper_guide_contracts["research_basket_evidence"] = dict(selected_research_context_evidence_contract)
        doc_list_rendered_payload = None
        doc_list_rendered_payload_sig = ""
        if prompt_multi_paper_list:
            had_doc_list_contract_key = "doc_list" in paper_guide_contracts
            doc_list_contract = _extract_doc_list_contract(paper_guide_contracts)
            if paper_guide_cross_paper_refs:
                doc_list_contract = _finalize_runtime_exclude_bound_source_from_multi_paper_doc_list_contract(
                    doc_list=doc_list_contract,
                    bound_source_path=str(paper_guide_bound_source_path or ""),
                    bound_source_name=str(paper_guide_bound_source_name or ""),
                )
            filtered_doc_list_contract = _finalize_runtime_filter_multi_paper_doc_list_contract(
                prompt=prompt or prompt_for_user or "",
                doc_list=doc_list_contract,
            )
            doc_list_contract_changed = (filtered_doc_list_contract != doc_list_contract) or (not had_doc_list_contract_key)
            doc_list_contract = list(filtered_doc_list_contract)
            if not doc_list_contract:
                rebuilt_doc_list_contract = _rebuild_multi_paper_doc_list_contract_from_available_refs(
                    prompt=prompt or prompt_for_user or retrieval_prompt or "",
                    seed_docs=list(refs_seed_docs_for_display or []),
                    answer_hits=list(answer_hits or refs_seed_docs_for_display or []),
                    evidence_cards=list(paper_guide_evidence_cards or []),
                    exclude_bound_source=bool(paper_guide_cross_paper_refs),
                    bound_source_path=str(paper_guide_bound_source_path or ""),
                    bound_source_name=str(paper_guide_bound_source_name or ""),
                )
                if rebuilt_doc_list_contract:
                    doc_list_contract = list(rebuilt_doc_list_contract)
                    doc_list_contract_changed = True
            if doc_list_contract_changed:
                paper_guide_contracts["doc_list"] = list(doc_list_contract)
            if umid > 0:
                try:
                    doc_list_rendered_payload, doc_list_rendered_payload_sig = _build_doc_list_refs_render_payload(
                        user_msg_id=umid,
                        prompt=prompt,
                        prompt_sig=str(task.get("prompt_sig") or ""),
                        hits=list(refs_seed_docs_for_display or []),
                        scores=list(scores_raw or []),
                        used_query=str(used_query or ""),
                        used_translation=bool(used_translation),
                        doc_list=list(doc_list_contract or []),
                        guide_mode=bool(paper_guide_source_scoped),
                        guide_source_path=str(paper_guide_bound_source_path or "") if paper_guide_source_scoped else "",
                        guide_source_name=str(paper_guide_bound_source_name or "") if paper_guide_source_scoped else "",
                    )
                except Exception:
                    doc_list_rendered_payload = None
                    doc_list_rendered_payload_sig = ""
                if isinstance(doc_list_rendered_payload, dict) and doc_list_rendered_payload and str(doc_list_rendered_payload_sig or "").strip():
                    rendered_doc_list_contract = _build_doc_list_contract_from_rendered_payload(
                        doc_list_contract=list(doc_list_contract or []),
                        rendered_payload=doc_list_rendered_payload,
                    )
                    if rendered_doc_list_contract != doc_list_contract:
                        doc_list_contract = list(rendered_doc_list_contract)
                        paper_guide_contracts["doc_list"] = list(doc_list_contract)
                        doc_list_contract_changed = True
            if doc_list_contract_changed and doc_list_contract:
                reformatted_answer = _finalize_runtime_format_multi_paper_list_answer(
                    prompt=prompt or prompt_for_user or "",
                    docs=list(doc_list_contract),
                )
                if str(reformatted_answer or "").strip():
                    answer = str(reformatted_answer or "").strip()
                    render_packet = (
                        dict(paper_guide_contracts.get("render_packet") or {})
                        if isinstance(paper_guide_contracts.get("render_packet"), dict)
                        else {}
                    )
                    if render_packet:
                        render_packet["answer_markdown"] = answer
                        render_packet["rendered_body"] = answer
                        render_packet["rendered_content"] = answer
                        render_packet["copy_markdown"] = answer
                        render_packet["copy_text"] = answer
                        paper_guide_contracts["render_packet"] = render_packet
            paper_guide_contracts = _sync_multi_paper_primary_evidence_into_contracts(
                paper_guide_contracts=paper_guide_contracts,
                doc_list_contract=doc_list_contract,
            )
        shared_primary_evidence = (
            dict(paper_guide_contracts.get("primary_evidence") or {})
            if isinstance(paper_guide_contracts.get("primary_evidence"), dict)
            else {}
        )
        citation_validation = dict(finalize_state.get("citation_validation") or {})
        answer_quality = dict(finalize_state.get("answer_quality") or {})
        ref_opps_quality = (
            answer_quality.get("reference_opportunities")
            if isinstance(answer_quality.get("reference_opportunities"), dict)
            else {}
        )
        validation_quality = (
            answer_quality.get("citation_validation")
            if isinstance(answer_quality.get("citation_validation"), dict)
            else citation_validation
        )
        _trace_section(
            "citation_systems",
            {
                "system_b_validated": bool((ref_opps_quality or {}).get("rendered_count")),
                "system_b_validated_count": int((ref_opps_quality or {}).get("rendered_count") or 0),
                "system_b_available_count": int((ref_opps_quality or {}).get("count") or len(paper_guide_reference_opportunities or [])),
                "citation_validation_status": str((validation_quality or {}).get("status") or (citation_validation or {}).get("status") or ""),
                "citation_validation_changed": bool((validation_quality or {}).get("changed") or (citation_validation or {}).get("changed")),
            },
        )
        if paper_guide_mode:
            contract_intent = dict(paper_guide_contracts.get("intent") or {}) if isinstance(paper_guide_contracts.get("intent"), dict) else {}
            paper_guide_debug.update(
                {
                    "final_answer_quality_family": str(answer_quality.get("prompt_family") or ""),
                    "final_contract_intent_family": str(contract_intent.get("family") or ""),
                    "research_answer_plan": research_answer_plan,
                    "final_answer_prefix": str(answer or "")[:160],
                }
            )
        runtime_repair = {"answer": answer, "changed": False, "reasons": []}
        if bool(task.get("agent_mode")):
            pre_repair_agent_trace = build_agent_trace_for_completed_answer(
                prompt,
                answer,
                evidence_hits=answer_hits,
                status="done",
                scope_context=agent_scope_context,
                agent_notes=agent_notes_for_trace,
                answer_mode=agent_answer_mode,
                generation_output=agent_generation_result_for_trace,
            )
            pre_repair_source_summary = _gen_agent_source_summary(pre_repair_agent_trace)
            runtime_repair = _gen_repair_answer_runtime(
                task,
                prompt=prompt,
                answer=answer,
                answer_quality=answer_quality,
                agent_trace=pre_repair_agent_trace,
                agent_source_summary=pre_repair_source_summary,
                answer_mode=agent_answer_mode,
                source_blend=str(agent_scope_context.get("answer_source_blend") or ""),
            )
            if runtime_repair.get("changed"):
                repaired_answer = str(runtime_repair.get("answer") or "").strip()
                if repaired_answer:
                    answer = repaired_answer
                    paper_guide_contracts = _sync_runtime_repaired_answer_contracts(
                        paper_guide_contracts,
                        answer=answer,
                    )
                    if paper_guide_mode:
                        paper_guide_debug["runtime_repair_reasons"] = list(runtime_repair.get("reasons") or [])[:8]
                    _trace_section(
                        "answer",
                        {
                            "runtime_repair": True,
                            "runtime_repair_reasons": list(runtime_repair.get("reasons") or [])[:8],
                        },
                    )
        _gen_store_answer(task, answer)
        _gen_store_answer_quality_meta(task, answer_quality=answer_quality)
        # Store canonical answer_hits source_paths so the renderer resolves [n]
        # against the same ordering the LLM saw, not a separately-filtered display list.
        # IMPORTANT: keep ALL paths (no dedup) so that [k] → canonical_paths[k-1]
        # matches the 1-based DOC-k numbering the LLM sees in its context.
        if cur_assistant_msg_id > 0 and answer_hits:
            try:
                _canon_paths: list[str] = []
                for _h in answer_hits:
                    _sp_h = str((_h.get("meta") or {}).get("source_path") or "").strip()
                    if _sp_h:
                        _canon_paths.append(_sp_h)
                if _canon_paths:
                    chat_store.merge_message_meta(cur_assistant_msg_id, {"canonical_hit_paths": _canon_paths})
            except Exception:
                pass
        if (not prompt_multi_paper_list) and umid > 0 and answer_hits:
            try:
                final_refs_docs = _merge_refs_display_docs_with_answer_hits(
                    refs_seed_docs=list(refs_seed_docs_for_display or []),
                    answer_hits=list(answer_hits or []),
                    limit=max(1, min(int(top_k or 4), 6 if prompt_multi_source_synthesis else 4)),
                    answer=answer,
                )
                _trace_section(
                    "refs",
                    {
                        "final_display_count": int(len(final_refs_docs or [])),
                        "final_display_sources": _trace_summarize_hits(list(final_refs_docs or []), limit=6),
                    },
                )
                rendered_payload = None
                rendered_payload_sig = ""
                if final_refs_docs:
                    t_refs_render0 = time.perf_counter()
                    rendered_payload, rendered_payload_sig = _build_precomputed_refs_render_payload(
                        user_msg_id=umid,
                        prompt=prompt,
                        answer=answer,
                        prompt_sig=str(task.get("prompt_sig") or ""),
                        hits=list(final_refs_docs or []),
                        scores=list(scores_raw or []),
                        used_query=str(used_query or ""),
                        used_translation=bool(used_translation),
                        guide_mode=bool(paper_guide_source_scoped),
                        guide_source_path=str(paper_guide_bound_source_path or "") if paper_guide_source_scoped else "",
                        guide_source_name=str(paper_guide_bound_source_name or "") if paper_guide_source_scoped else "",
                        library_db_path=getattr(settings_obj, "library_db_path", None),
                    )
                    _trace_event(
                        "refs_precompute",
                        elapsed_s=time.perf_counter() - t_refs_render0,
                        rendered=bool(rendered_payload and rendered_payload_sig),
                        hit_count=int(len(final_refs_docs or [])),
                    )
                    primary_alignment = (
                        dict(rendered_payload.get("primary_evidence_alignment") or {})
                        if isinstance(rendered_payload, dict) and isinstance(rendered_payload.get("primary_evidence_alignment"), dict)
                        else {}
                    )
                    primary_evidence_rendered = (
                        dict(rendered_payload.get("primary_evidence") or {})
                        if isinstance(rendered_payload, dict) and isinstance(rendered_payload.get("primary_evidence"), dict)
                        else {}
                    )
                    _trace_section(
                        "refs",
                        {
                            "render_status": "full" if rendered_payload and rendered_payload_sig else "missing",
                            "rendered_payload_sig": str(rendered_payload_sig or "").strip(),
                            "primary_evidence_heading": str(primary_evidence_rendered.get("heading_path") or "").strip(),
                            "primary_evidence_mismatch": bool(primary_alignment.get("mismatch")),
                            "primary_evidence_score": primary_alignment.get("score"),
                            "primary_evidence_terms": list(primary_alignment.get("matched_answer_terms") or [])[:8],
                        },
                    )
                    chat_store.upsert_message_refs(
                        user_msg_id=umid,
                        conv_id=conv_id,
                        prompt=prompt,
                        prompt_sig=str(task.get("prompt_sig") or ""),
                        hits=list(final_refs_docs or []),
                        scores=list(scores_raw or []),
                        used_query=str(used_query or ""),
                        used_translation=bool(used_translation),
                        rendered_payload=rendered_payload,
                        rendered_payload_sig=rendered_payload_sig,
                        render_status="full" if rendered_payload and rendered_payload_sig else None,
                        render_error="" if rendered_payload and rendered_payload_sig else None,
                        render_error_detail="" if rendered_payload and rendered_payload_sig else None,
                        render_built_at=time.time() if rendered_payload and rendered_payload_sig else None,
                        render_attempts=1 if rendered_payload and rendered_payload_sig else None,
                        render_evidence_sig=str(rendered_payload_sig or "").strip(),
                        query_variants=list(query_variants or []),
                    )
            except Exception:
                pass
        if prompt_multi_paper_list and umid > 0:
            doc_list_contract = _extract_doc_list_contract(paper_guide_contracts)
            try:
                rendered_payload = doc_list_rendered_payload
                rendered_payload_sig = doc_list_rendered_payload_sig
                if (not isinstance(rendered_payload, dict)) or (not rendered_payload) or (not str(rendered_payload_sig or "").strip()):
                    rendered_payload, rendered_payload_sig = _build_doc_list_refs_render_payload(
                        user_msg_id=umid,
                        prompt=prompt,
                        prompt_sig=str(task.get("prompt_sig") or ""),
                        hits=list(refs_seed_docs_for_display or []),
                        scores=list(scores_raw or []),
                        used_query=str(used_query or ""),
                        used_translation=bool(used_translation),
                        doc_list=list(doc_list_contract or []),
                        guide_mode=bool(paper_guide_source_scoped),
                        guide_source_path=str(paper_guide_bound_source_path or "") if paper_guide_source_scoped else "",
                        guide_source_name=str(paper_guide_bound_source_name or "") if paper_guide_source_scoped else "",
                    )
                if isinstance(rendered_payload, dict) and rendered_payload and str(rendered_payload_sig or "").strip():
                    ready_seed_hits = _set_refs_hit_pack_state(
                        list(refs_seed_docs_for_display or []),
                        state="ready",
                    )
                    _trace_section(
                        "refs",
                        {
                            "final_display_count": int(len(ready_seed_hits or [])),
                            "final_display_sources": _trace_summarize_hits(list(ready_seed_hits or []), limit=6),
                            "render_status": "full",
                            "rendered_payload_sig": str(rendered_payload_sig or "").strip(),
                            "doc_list_contract_count": int(len(doc_list_contract or [])),
                        },
                    )
                    chat_store.upsert_message_refs(
                        user_msg_id=umid,
                        conv_id=conv_id,
                        prompt=prompt,
                        prompt_sig=str(task.get("prompt_sig") or ""),
                        hits=ready_seed_hits,
                        scores=list(scores_raw or []),
                        used_query=str(used_query or ""),
                        used_translation=bool(used_translation),
                        rendered_payload=rendered_payload,
                        rendered_payload_sig=rendered_payload_sig,
                        render_status="full",
                        render_error="",
                        render_error_detail="",
                        render_built_at=time.time(),
                        render_attempts=1,
                        render_evidence_sig=str(rendered_payload_sig or "").strip(),
                    )
            except Exception:
                pass
        _gen_record_answer_quality(
            session_id=session_id,
            task_id=task_id,
            conv_id=conv_id,
            answer_quality=answer_quality,
        )
        t_prov0 = time.perf_counter()
        try:
            stored_provenance = _gen_store_answer_provenance_fast(
                task,
                answer=answer,
                answer_hits=answer_hits,
                support_resolution=paper_guide_support_resolution,
                primary_evidence=shared_primary_evidence,
            )
            if isinstance(stored_provenance, dict) and stored_provenance:
                synced_contracts = _sync_paper_guide_render_packet_with_provenance(
                    paper_guide_contracts=paper_guide_contracts,
                    provenance=stored_provenance,
                    answer=answer,
                )
                if synced_contracts != paper_guide_contracts:
                    paper_guide_contracts = synced_contracts
                    shared_primary_evidence = (
                        dict(paper_guide_contracts.get("primary_evidence") or {})
                        if isinstance(paper_guide_contracts.get("primary_evidence"), dict)
                        else shared_primary_evidence
                    )
            _perf_log("gen.provenance_inline_fast", elapsed=time.perf_counter() - t_prov0, ok=1)
            _trace_event("provenance_fast", elapsed_s=time.perf_counter() - t_prov0, ok=True)
        except Exception as exc:
            _perf_log("gen.provenance_inline_fast", elapsed=time.perf_counter() - t_prov0, ok=0, err=str(exc)[:120])
            _trace_event("provenance_fast", elapsed_s=time.perf_counter() - t_prov0, ok=False)
        if _should_run_provenance_async_refine(task):
            try:
                _gen_store_answer_provenance_async(
                    task,
                    answer=answer,
                    answer_hits=answer_hits,
                    support_resolution=paper_guide_support_resolution,
                    primary_evidence=shared_primary_evidence,
                )
                _perf_log("gen.provenance_async_schedule", ok=1)
                _trace_section("answer", {"provenance_async_scheduled": True})
            except Exception as exc:
                _perf_log("gen.provenance_async_schedule", ok=0, err=str(exc)[:120])
        _gen_store_paper_guide_contract_meta(task, paper_guide_contracts=paper_guide_contracts)
        _trace_section(
            "answer",
            {
                "chars": int(len(answer or "")),
                "quality_minimum_ok": bool(answer_quality.get("minimum_ok")),
                "quality_prompt_family": str(answer_quality.get("prompt_family") or ""),
                "research_answer_plan": str(answer_quality.get("research_answer_plan") or research_answer_plan or ""),
            },
        )
        research_trace = _trace_finish(research_trace, status="done", total_elapsed_s=time.perf_counter() - worker_t0)
        _gen_store_research_trace_meta(task, research_trace=research_trace)
        agent_trace = {}
        if bool(task.get("agent_mode")):
            agent_trace = build_agent_trace_for_completed_answer(
                prompt,
                answer,
                evidence_hits=answer_hits,
                status="done",
                scope_context=agent_scope_context,
                agent_notes=agent_notes_for_trace,
                answer_mode=agent_answer_mode,
                generation_output=agent_generation_result_for_trace,
            )
            _gen_store_agent_trace_meta(task, agent_trace=agent_trace)
        agent_completion_payload = _agent_finalize_completion_payload(
            task,
            answer=answer,
            answer_quality=answer_quality,
            agent_trace=agent_trace,
            answer_mode=agent_answer_mode,
            source_blend=str(agent_scope_context.get("answer_source_blend") or ""),
            runtime_repair=runtime_repair,
        )
        agent_source_summary = dict(agent_completion_payload.get("agent_source_summary") or {})
        answer_runtime_check = dict(agent_completion_payload.get("answer_runtime_check") or {})
        answer_contract = dict(agent_completion_payload.get("answer_contract") or {})
        _gen_store_answer_runtime_check_meta(task, answer_runtime_check=answer_runtime_check)
        _gen_store_answer_contract_meta(task, answer_contract=answer_contract)
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
            paper_guide_debug=dict(paper_guide_debug),
            citation_validation=citation_validation,
            research_trace=research_trace,
            agent_trace=agent_trace,
            agent_source_summary=agent_source_summary,
            answer_runtime_check=answer_runtime_check,
            answer_contract=answer_contract,
            finished_at=time.time(),
        )
        _perf_log("gen.answer", elapsed=time.perf_counter() - t_answer0, chars=len(answer))
        _perf_log("gen.total", elapsed=time.perf_counter() - worker_t0, conv_id=conv_id)

    except Exception as e:
        snap = _gen_get_task(session_id) or {}
        cancel_requested = str(e) == "canceled" or (
            str(snap.get("id") or "") == str(task_id or "")
            and bool(snap.get("cancel") or False)
        )
        if cancel_requested:
            partial = str(snap.get("partial") or "").strip()
            answer = (partial + "\n\n(Generation canceled)").strip() or "(Generation canceled)"
            try:
                _gen_store_answer(task, answer)
            except Exception:
                pass
            if research_trace:
                research_trace = _trace_finish(research_trace, status="canceled", total_elapsed_s=time.perf_counter() - worker_t0)
                _gen_store_research_trace_meta(task, research_trace=research_trace)
            agent_trace = {}
            if bool(task.get("agent_mode")):
                agent_trace = build_agent_trace_for_completed_answer(
                    prompt,
                    answer,
                    evidence_hits=[],
                    status="canceled",
                    scope_context=agent_scope_context,
                    agent_notes=agent_notes_for_trace,
                    answer_mode=agent_answer_mode,
                )
                _gen_store_agent_trace_meta(task, agent_trace=agent_trace)
            agent_source_summary = _gen_agent_source_summary(agent_trace)
            _gen_update_task(
                session_id,
                task_id,
                status="canceled",
                stage="canceled",
                answer=answer,
                partial=answer,
                char_count=len(answer),
                research_trace=research_trace,
                agent_trace=agent_trace,
                agent_source_summary=agent_source_summary,
                finished_at=time.time(),
            )
            return

        err = _format_llm_failure_message(err=e, settings_obj=settings_obj)
        try:
            _gen_store_answer(task, err)
        except Exception:
            pass
        if research_trace:
            research_trace = _trace_finish(
                research_trace,
                status="error",
                total_elapsed_s=time.perf_counter() - worker_t0,
                error=str(e),
            )
            _gen_store_research_trace_meta(task, research_trace=research_trace)
        agent_trace = {}
        if bool(task.get("agent_mode")):
            agent_trace = build_agent_trace_for_completed_answer(
                prompt,
                err,
                evidence_hits=[],
                status="error",
                scope_context=agent_scope_context,
                agent_notes=agent_notes_for_trace,
                answer_mode=agent_answer_mode,
            )
            _gen_store_agent_trace_meta(task, agent_trace=agent_trace)
        agent_source_summary = _gen_agent_source_summary(agent_trace)
        _gen_update_task(
            session_id,
            task_id,
            status="error",
            stage="error",
            error=str(e),
            answer=err,
            partial=err,
            char_count=len(err),
            research_trace=research_trace,
            agent_trace=agent_trace,
            agent_source_summary=agent_source_summary,
            finished_at=time.time(),
        )

def _gen_start_task(task: dict) -> bool:
    sid = str(task.get("session_id") or "").strip()
    tid = str(task.get("id") or "").strip()
    if (not sid) or (not tid):
        return False
    conv_id = str(task.get("conv_id") or "").strip()
    chat_db_path = task.get("chat_db")
    chat_db_key = ""
    if chat_db_path:
        try:
            chat_db_key = str(Path(str(chat_db_path or "")).expanduser().resolve()).lower()
        except Exception:
            chat_db_key = str(chat_db_path or "").strip().lower()
    RUNTIME.prune_generation_tasks(now=time.time())
    with RUNTIME.GEN_LOCK:
        cur = RUNTIME.GEN_TASKS.get(sid)
        if _state_gen_task_blocks_conversation(cur):
            return False
        if conv_id:
            for cur in RUNTIME.GEN_TASKS.values():
                if not isinstance(cur, dict):
                    continue
                if str(cur.get("conv_id") or "").strip() != conv_id:
                    continue
                if chat_db_key:
                    try:
                        cur_db = str(Path(str(cur.get("chat_db") or "")).expanduser().resolve()).lower()
                    except Exception:
                        cur_db = str(cur.get("chat_db") or "").strip().lower()
                    if cur_db and cur_db != chat_db_key:
                        continue
                if _state_gen_task_blocks_conversation(cur):
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
        logger.exception("generation_thread_start_failed", extra={"session_id": sid, "task_id": tid})
        failure_answer = generation_start_failed_message(task.get("ui_locale") or task.get("locale"))
        try:
            _gen_store_answer(task, failure_answer)
        except Exception:
            pass
        with RUNTIME.GEN_LOCK:
            cur = RUNTIME.GEN_TASKS.get(sid)
            if isinstance(cur, dict) and str(cur.get("id") or "") == tid:
                now = time.time()
                cur2 = dict(cur)
                cur2["status"] = "error"
                cur2["stage"] = "error"
                cur2["error"] = "thread_start_failed"
                cur2["answer"] = failure_answer
                cur2["partial"] = failure_answer
                cur2["char_count"] = len(failure_answer)
                cur2["updated_at"] = now
                cur2["finished_at"] = now
                RUNTIME.GEN_TASKS[sid] = cur2
        return False
    return True

def _bg_enqueue(task: dict) -> bool:
    if "_tid" not in task:
        task = dict(task)
        task["_tid"] = uuid.uuid4().hex
    enqueued = bg_enqueue(_BG_STATE, _BG_LOCK, task)
    if enqueued:
        _bg_ensure_started()
    return bool(enqueued)

def _bg_remove_queued_tasks_for_pdf(pdf_path: Path) -> int:
    """
    Remove queued (not running) conversion tasks for a given PDF.
    Returns removed count.
    """
    return bg_remove_queued_tasks_for_pdf(_BG_STATE, _BG_LOCK, pdf_path)


def _safe_rmtree_child(path_obj: Path, root_obj: Path) -> None:
    try:
        root = Path(root_obj).expanduser().resolve()
        target = Path(path_obj).expanduser().resolve()
        target.relative_to(root)
        if target == root:
            return
        import shutil

        shutil.rmtree(target, ignore_errors=True)
    except Exception:
        return

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


def _bg_ingest_py_path() -> Path:
    return Path(__file__).resolve().parent.parent / "ingest.py"


def _bg_cancel_requested(cancel_cb) -> bool:
    if not callable(cancel_cb):
        return False
    try:
        return bool(cancel_cb())
    except Exception:
        return False


def _bg_conversion_result_message(ok: bool, out_folder: object, cancel_cb, *, prefix: str = "") -> tuple[bool, object, str]:
    if _bg_cancel_requested(cancel_cb):
        return False, "cancelled", "CANCELLED"
    if bool(ok):
        return True, out_folder, f"OK{prefix}: {out_folder}"
    txt = str(out_folder or "").strip().lower()
    fail_prefix = "FAIL" + prefix
    return False, out_folder, "CANCELLED" if txt == "cancelled" else f"{fail_prefix}: {out_folder}"


def _bg_run_ingest_subprocess(args: list[str], cancel_cb) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        while proc.poll() is None:
            if _bg_cancel_requested(cancel_cb):
                try:
                    proc.terminate()
                    proc.wait(timeout=4)
                except Exception:
                    pass
                try:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=2)
                except Exception:
                    pass
                try:
                    stdout, stderr = proc.communicate(timeout=1)
                except Exception:
                    stdout, stderr = "", "cancelled"
                return subprocess.CompletedProcess(args, -15, stdout or "", stderr or "cancelled")
            time.sleep(0.2)
        stdout, stderr = proc.communicate()
        return subprocess.CompletedProcess(args, int(proc.returncode or 0), stdout or "", stderr or "")
    except Exception:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass
        raise


def _bg_record_repair_attempt(md_path: Path | None, **payload) -> None:
    if md_path is None:
        return
    try:
        append_conversion_repair_attempt(md_path, **payload)
    except Exception:
        pass


def _post_convert_quality_gate_enabled() -> bool:
    raw = str(os.environ.get("KB_POST_CONVERT_QUALITY_GATE", "1") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _quality_gate_auto_repair(assessment: dict) -> dict:
    value = (assessment or {}).get("auto_repair") if isinstance(assessment, dict) else {}
    return value if isinstance(value, dict) else {}


_SOURCE_CONVERSION_RETRY_ISSUES = {
    "missing_images",
    "missing_references",
    "mojibake",
    "weak_structure",
    "missing_markdown",
    "source_text_loss",
    "missing_source_pages",
    "page_marker_gaps",
    "reference_index_truncated",
}


def _post_convert_source_retry_enabled() -> bool:
    raw = str(os.environ.get("KB_POST_CONVERT_SOURCE_RETRY", "1") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _quality_assessment_issue_codes(assessment: dict) -> list[str]:
    out: list[str] = []
    for raw in list((assessment or {}).get("blocking_issue_codes") or []) + list((assessment or {}).get("issue_codes") or []):
        code = str(raw or "").strip().lower()
        if code and code not in out:
            out.append(code)
    plan = (assessment or {}).get("repair_plan") if isinstance((assessment or {}).get("repair_plan"), dict) else {}
    for key in ("reconvert_issue_codes", "review_issue_codes", "issue_codes"):
        for raw in list((plan or {}).get(key) or []):
            code = str(raw or "").strip().lower()
            if code and code not in out:
                out.append(code)
    return out


def _post_convert_source_retry_needed(assessment: dict, *, already_retried: bool = False) -> bool:
    if already_retried or not _post_convert_source_retry_enabled():
        return False
    if bool((assessment or {}).get("indexable")):
        return False
    action = str((assessment or {}).get("action") or "").strip().lower()
    if action not in {"reconvert", "autofix"}:
        return False
    codes = set(_quality_assessment_issue_codes(assessment))
    return bool(codes & _SOURCE_CONVERSION_RETRY_ISSUES)


def _post_convert_source_retry_speed_mode(assessment: dict, requested_speed_mode: str) -> str:
    plan = (assessment or {}).get("repair_plan") if isinstance((assessment or {}).get("repair_plan"), dict) else {}
    planned = str((plan or {}).get("speed_mode") or "").strip().lower()
    if planned and planned != "no_llm":
        return planned
    requested = str(requested_speed_mode or "").strip().lower()
    if requested in {"normal", "full_llm"}:
        return "normal"
    return "normal"


def _bg_post_convert_quality_gate(
    md_path: Path,
    *,
    task_id: str = "",
    repair_run_id: str = "",
    speed_mode: str = "",
    source_pdf_path: Path | str | None = None,
) -> dict:
    if not _post_convert_quality_gate_enabled():
        return {
            "enabled": False,
            "indexable": True,
            "status": "skipped",
            "action": "none",
            "auto_repair": {"attempted": False, "changed": False, "unsafe": False, "applied": []},
        }
    try:
        assessment = prepare_markdown_for_index(
            md_path,
            auto_repair=True,
            allow_blocked=False,
            source_pdf_path=source_pdf_path,
        )
    except Exception as exc:
        assessment = {
            "enabled": True,
            "indexable": False,
            "status": "blocked",
            "action": "review",
            "reason": f"Post-convert quality gate failed: {exc}",
            "issue_codes": ["quality_scan_failed"],
            "blocking_issue_codes": ["quality_scan_failed"],
            "auto_repair": {"attempted": False, "changed": False, "unsafe": False, "applied": []},
        }
    else:
        assessment = {**assessment, "enabled": True}

    auto_repair = _quality_gate_auto_repair(assessment)
    if bool(assessment.get("indexable")) and bool(auto_repair.get("changed")):
        try:
            rebuild_structured_indices_for_markdown(
                md_path,
                md_text=md_path.read_text(encoding="utf-8", errors="replace"),
                assets_dir=md_path.parent / "assets",
            )
            assessment["structured_indices_rebuilt"] = True
        except Exception as exc:
            assessment["structured_indices_rebuilt"] = False
            assessment["structured_indices_error"] = str(exc)[:300]

    action = str(assessment.get("action") or "review").strip().lower() or "review"
    status = (
        "blocked"
        if not bool(assessment.get("indexable"))
        else ("autofixed" if bool(auto_repair.get("changed")) else ("ready" if action == "none" else "degraded"))
    )
    plan = assessment.get("repair_plan") if isinstance(assessment.get("repair_plan"), dict) else {}
    detail = (
        "Conversion output was blocked before indexing."
        if status == "blocked"
        else (
            "Conversion output was auto-repaired and accepted before indexing."
            if status == "autofixed"
            else "Conversion output passed post-convert quality gate."
        )
    )
    _bg_record_repair_attempt(
        md_path,
        event="post_convert_quality_gate",
        status=status,
        action=action,
        scope=str(plan.get("scope") or ""),
        speed_mode=speed_mode,
        issue_codes=list(assessment.get("issue_codes") or []),
        task_id=task_id,
        repair_run_id=repair_run_id,
        source="post_convert_quality_gate",
        reason=str(assessment.get("reason") or ""),
        detail=detail,
        extra={
            "indexable": bool(assessment.get("indexable")),
            "quality_status": str(assessment.get("status") or ""),
            "blocking_issue_codes": [
                str(item) for item in list(assessment.get("blocking_issue_codes") or []) if str(item or "").strip()
            ][:30],
            "auto_repair": {
                "attempted": bool(auto_repair.get("attempted")),
                "changed": bool(auto_repair.get("changed")),
                "unsafe": bool(auto_repair.get("unsafe")),
                "applied": [str(item) for item in list(auto_repair.get("applied") or []) if str(item or "").strip()][:20],
            },
            "structured_indices_rebuilt": bool(assessment.get("structured_indices_rebuilt")),
        },
    )
    return assessment


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
        repair_context = task.get("repair_context") if isinstance(task.get("repair_context"), dict) else {}
        repair_run_id = str((repair_context or {}).get("repair_run_id") or "").strip()
        if speed_mode == "ultra_fast":
            # Keep VL/LLM path in ultra_fast; converter itself handles speed/quality tradeoff.
            # Forcing no_llm here causes a dramatic quality drop that does not match UI semantics.
            eq_image_fallback = False
        task_id = str(task.get("_tid") or "")

        try:
            md_folder = out_root / pdf.stem
            if replace and md_folder.exists():
                _safe_rmtree_child(md_folder, out_root)

            last_page_done = 0
            last_page_total = 0

            def _on_progress(page_done: int, page_total: int, msg: str = "") -> None:
                nonlocal last_page_done, last_page_total
                try:
                    last_page_done = max(0, int(page_done or 0))
                except Exception:
                    pass
                try:
                    last_page_total = max(0, int(page_total or 0))
                except Exception:
                    pass
                try:
                    bg_update_page_progress(_BG_STATE, _BG_LOCK, page_done, page_total, msg, task_id=task_id)
                except Exception:
                    pass

            def _should_cancel() -> bool:
                return bg_should_cancel(_BG_STATE, _BG_LOCK)

            def _clear_md_folder_for_retry() -> None:
                _safe_rmtree_child(md_folder, out_root)

            effective_speed_mode = speed_mode
            source_retry_done = False
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
            ok, out_folder, msg = _bg_conversion_result_message(ok, out_folder, _should_cancel)

            _, md_main, md_exists = _resolve_md_output_paths(out_root, pdf)
            if repair_context and md_exists:
                _bg_record_repair_attempt(
                    md_main,
                    event="conversion_finished",
                    status="cancelled" if msg == "CANCELLED" else ("success" if ok else "error"),
                    action=str(repair_context.get("action") or "reconvert"),
                    scope=str(repair_context.get("scope") or ""),
                    speed_mode=speed_mode,
                    issue_codes=list(repair_context.get("issue_codes") or []),
                    task_id=task_id,
                    repair_run_id=repair_run_id,
                    source=str(repair_context.get("source") or "background_conversion"),
                    reason=str(repair_context.get("reason") or ""),
                    detail=msg,
                    extra={"replace": replace, "no_llm": no_llm},
                )

            post_convert_quality: dict = {}
            if ok and md_exists and not _bg_cancel_requested(_should_cancel):
                try:
                    _on_progress(
                        last_page_done,
                        last_page_total,
                        "quality gate: validating converted Markdown",
                    )
                    post_convert_quality = _bg_post_convert_quality_gate(
                        md_main,
                        task_id=task_id,
                        repair_run_id=repair_run_id,
                        speed_mode=speed_mode,
                        source_pdf_path=pdf,
                    )
                    if not bool(post_convert_quality.get("indexable")):
                        msg = f"OK+QUALITY_BLOCKED: {out_folder}"
                    elif bool(_quality_gate_auto_repair(post_convert_quality).get("changed")):
                        msg = f"OK+QUALITY_REPAIRED: {out_folder}"
                except Exception:
                    post_convert_quality = {}

            if (
                ok
                and md_exists
                and not _bg_cancel_requested(_should_cancel)
                and _post_convert_source_retry_needed(post_convert_quality, already_retried=source_retry_done)
            ):
                source_retry_done = True
                retry_speed_mode = _post_convert_source_retry_speed_mode(post_convert_quality, speed_mode)
                retry_issue_codes = _quality_assessment_issue_codes(post_convert_quality)
                retry_plan = post_convert_quality.get("repair_plan") if isinstance(post_convert_quality.get("repair_plan"), dict) else {}
                retry_scope = str((retry_plan or {}).get("scope") or "")
                retry_reason = str(post_convert_quality.get("reason") or "")
                try:
                    _bg_record_repair_attempt(
                        md_main,
                        event="source_quality_retry_queued",
                        status="queued",
                        action="reconvert",
                        scope=retry_scope,
                        speed_mode=retry_speed_mode,
                        issue_codes=retry_issue_codes,
                        task_id=task_id,
                        repair_run_id=repair_run_id,
                        source="post_convert_quality_gate",
                        reason=retry_reason,
                        detail="Quality gate requested an automatic source-level reconversion.",
                        extra={"requested_speed_mode": speed_mode, "retry_speed_mode": retry_speed_mode},
                    )
                except Exception:
                    pass
                _on_progress(
                    last_page_done,
                    last_page_total,
                    f"quality gate: retrying conversion with {retry_speed_mode} profile",
                )
                _clear_md_folder_for_retry()
                retry_ok, retry_out_folder = run_pdf_to_md(
                    pdf_path=pdf,
                    out_root=out_root,
                    no_llm=False,
                    keep_debug=False,
                    eq_image_fallback=False,
                    progress_cb=_on_progress,
                    cancel_cb=_should_cancel,
                    speed_mode=retry_speed_mode,
                    max_active_conversions=_bg_target_worker_count(),
                )
                ok, out_folder, msg = _bg_conversion_result_message(
                    retry_ok,
                    retry_out_folder,
                    _should_cancel,
                    prefix="+SOURCE_RETRY",
                )
                effective_speed_mode = retry_speed_mode
                _, md_main, md_exists = _resolve_md_output_paths(out_root, pdf)
                if md_exists:
                    _bg_record_repair_attempt(
                        md_main,
                        event="source_quality_retry_finished",
                        status="cancelled" if msg == "CANCELLED" else ("success" if ok else "error"),
                        action="reconvert",
                        scope=retry_scope,
                        speed_mode=retry_speed_mode,
                        issue_codes=retry_issue_codes,
                        task_id=task_id,
                        repair_run_id=repair_run_id,
                        source="post_convert_quality_gate",
                        reason=retry_reason,
                        detail=msg,
                        extra={"requested_speed_mode": speed_mode, "retry_speed_mode": retry_speed_mode},
                    )
                post_convert_quality = {}
                if ok and md_exists and not _bg_cancel_requested(_should_cancel):
                    try:
                        _on_progress(
                            last_page_done,
                            last_page_total,
                            "quality gate: validating retried Markdown",
                        )
                        post_convert_quality = _bg_post_convert_quality_gate(
                            md_main,
                            task_id=task_id,
                            repair_run_id=repair_run_id,
                            speed_mode=effective_speed_mode,
                            source_pdf_path=pdf,
                        )
                        if not bool(post_convert_quality.get("indexable")):
                            msg = f"OK+SOURCE_RETRY_QUALITY_BLOCKED: {out_folder}"
                        elif bool(_quality_gate_auto_repair(post_convert_quality).get("changed")):
                            msg = f"OK+SOURCE_RETRY_REPAIRED: {out_folder}"
                        else:
                            msg = f"OK+SOURCE_RETRY_READY: {out_folder}"
                    except Exception:
                        post_convert_quality = {}

            # Auto-ingest can add noticeable latency in the conversion UI.
            # Skip it in ultra_fast mode to keep end-to-end time near the 5s target.
            do_auto_ingest = (
                ok
                and bool(db_dir)
                and (effective_speed_mode != "ultra_fast")
                and not _bg_cancel_requested(_should_cancel)
            )
            if do_auto_ingest and db_dir:
                try:
                    ingest_py = _bg_ingest_py_path()
                    if ingest_py.exists() and md_exists:
                        _on_progress(
                            last_page_done,
                            last_page_total,
                            "ingesting: updating knowledge base index",
                        )
                        ingest_args = [
                            os.sys.executable,
                            str(ingest_py),
                            "--src",
                            str(md_main),
                            "--db",
                            str(db_dir),
                            "--incremental",
                            "--rebuild-structured-indices",
                        ]
                        ingest_proc = _bg_run_ingest_subprocess(
                            ingest_args,
                            _should_cancel,
                        )
                        if _bg_cancel_requested(_should_cancel) or int(getattr(ingest_proc, "returncode", 1) or 0) == -15:
                            msg = "CANCELLED"
                            ingest_status = "cancelled"
                        elif int(getattr(ingest_proc, "returncode", 1) or 0) == 0:
                            if post_convert_quality and not bool(post_convert_quality.get("indexable")):
                                msg = f"OK+QUALITY_BLOCKED: {out_folder}"
                                ingest_status = "blocked"
                            elif bool(_quality_gate_auto_repair(post_convert_quality).get("changed")):
                                msg = f"OK+QUALITY_REPAIRED+INGEST: {out_folder}"
                                ingest_status = "success"
                            else:
                                msg = f"OK+INGEST: {out_folder}"
                                ingest_status = "success"
                        else:
                            msg = f"OK+INGEST_BLOCKED: {out_folder}"
                            ingest_status = "blocked"
                        if repair_context:
                            _bg_record_repair_attempt(
                                md_main,
                                event="ingest_finished",
                                status=ingest_status,
                                action=str(repair_context.get("action") or "reconvert"),
                                scope=str(repair_context.get("scope") or ""),
                                speed_mode=effective_speed_mode,
                                issue_codes=list(repair_context.get("issue_codes") or []),
                                task_id=task_id,
                                repair_run_id=repair_run_id,
                                source="background_ingest",
                                reason=str(repair_context.get("reason") or ""),
                                detail=(
                                    str(getattr(ingest_proc, "stdout", "") or "").strip()[-500:]
                                    or str(getattr(ingest_proc, "stderr", "") or "").strip()[-500:]
                                    or msg
                                ),
                            )
                except Exception:
                    # Not fatal; conversion still succeeded.
                    if repair_context:
                        _bg_record_repair_attempt(
                            md_main if "md_main" in locals() else None,
                            event="ingest_finished",
                            status="error",
                            action=str(repair_context.get("action") or "reconvert"),
                            scope=str(repair_context.get("scope") or ""),
                            speed_mode=effective_speed_mode if "effective_speed_mode" in locals() else speed_mode,
                            issue_codes=list(repair_context.get("issue_codes") or []),
                            task_id=task_id,
                            repair_run_id=repair_run_id,
                            source="background_ingest",
                            reason=str(repair_context.get("reason") or ""),
                            detail="background ingest failed after conversion",
                        )
                    pass
        except Exception as e:
            msg = f"FAIL: {e}"

        if bg_should_cancel(_BG_STATE, _BG_LOCK):
            msg = "CANCELLED"
        bg_finish_task(_BG_STATE, _BG_LOCK, msg, task_id=task_id)

def _bg_ensure_started() -> None:
    worker_ver = "2026-05-29.bg.source-retry.v1"
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
    repair_context: dict | None = None,
) -> dict:
    pdf = Path(pdf_path)
    mode = str(speed_mode)
    task = {
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
    if isinstance(repair_context, dict) and repair_context:
        task["repair_context"] = {
            "action": str(repair_context.get("action") or ""),
            "scope": str(repair_context.get("scope") or ""),
            "reason": str(repair_context.get("reason") or ""),
            "source": str(repair_context.get("source") or ""),
            "repair_run_id": str(repair_context.get("repair_run_id") or ""),
            "issue_codes": [str(item) for item in list(repair_context.get("issue_codes") or []) if str(item or "").strip()][:30],
        }
    return task

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
                "direct_prompt_score": 0.0,
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
            rank = meta.get("ref_rank") if isinstance(meta.get("ref_rank"), dict) else {}
            rec["direct_prompt_score"] = max(
                float(rec.get("direct_prompt_score") or 0.0),
                float(meta.get("direct_prompt_match_score") or (rank or {}).get("direct_prompt") or 0.0),
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
            float(item.get("direct_prompt_score") or 0.0),
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

    top_direct = float(top.get("direct_prompt_score") or 0.0)
    sec_direct = float(second.get("direct_prompt_score") or 0.0)
    if top_direct >= 6.0 and top_direct >= max(6.0, sec_direct + 1.5):
        rec = dict(top)
        rec["lock_reason"] = "direct_prompt_dominant"
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


def _validate_freeform_numeric_citations(
    answer: str,
    *,
    answer_hits: list[dict],
) -> tuple[str, dict]:
    return _citation_validation_validate_freeform_numeric_citations(
        answer,
        answer_hits=answer_hits,
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

_extract_inline_reference_numbers = _grounding_extract_inline_reference_numbers


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




# Package-grounder aliases. Keep these names stable for callers/tests while
# avoiding another layer of one-line passthrough wrappers in task_runtime.
_is_paper_guide_support_meta_line = _grounding_is_support_meta_line
_paper_guide_support_segment_spans = _grounding_support_segment_spans
_paper_guide_support_focus_tokens = _grounding_support_focus_tokens
_extract_paper_guide_locate_anchor = _grounding_extract_locate_anchor
_is_paper_guide_broad_summary_line = _grounding_is_broad_summary_line
_extract_paper_guide_ref_spans = _grounding_extract_ref_spans
_paper_guide_support_claim_type = _grounding_support_claim_type
_paper_guide_support_cite_policy = _grounding_support_cite_policy
_resolve_paper_guide_support_slot_block = _grounding_resolve_support_slot_block
_build_paper_guide_support_slots = _grounding_build_support_slots
_build_paper_guide_support_slots_block = _grounding_build_support_slots_block
_normalize_paper_guide_support_surface = _grounding_normalize_support_surface
_paper_guide_support_rule_tokens = _grounding_support_rule_tokens
_select_paper_guide_support_slot_for_context = _grounding_select_support_slot_for_context
_inject_paper_guide_support_markers = _grounding_inject_support_markers
_resolve_paper_guide_support_ref_num = _grounding_resolve_support_ref_num
_resolve_paper_guide_support_markers = _grounding_resolve_support_markers


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
