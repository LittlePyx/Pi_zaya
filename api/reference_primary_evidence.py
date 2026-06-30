from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def _primary_ref_evidence_candidate_title(
    *,
    meta: Mapping[str, Any] | None,
    citation_meta: Mapping[str, Any] | None,
    display_name: str,
) -> str:
    return str(
        (citation_meta or {}).get("title")
        or (meta or {}).get("title")
        or display_name
        or ""
    ).strip()


def _primary_ref_evidence_base_summary(
    *,
    meta: Mapping[str, Any] | None,
    prompt: str,
    heading: str,
    citation_meta: Mapping[str, Any] | None,
    allow_llm_translate: bool,
    build_ref_navigation: Callable[..., Mapping[str, Any] | None],
    fallback_ref_ui_summary_line: Callable[..., str],
) -> dict[str, Any]:
    nav_raw = build_ref_navigation(meta or {}, prompt=prompt, heading_fallback=heading)
    nav = dict(nav_raw or {}) if isinstance(nav_raw, Mapping) else {}
    used_nav_summary = bool(str(nav.get("summary_line") or nav.get("what") or "").strip())
    summary_line = str(nav.get("summary_line") or nav.get("what") or "").strip()
    if not summary_line:
        summary_line = fallback_ref_ui_summary_line(
            meta or {},
            prompt=prompt,
            citation_meta=citation_meta,
            allow_llm_translate=allow_llm_translate,
        )

    return {
        "nav": nav,
        "summary_line": str(summary_line or "").strip(),
        "summary_source": "navigation" if used_nav_summary else ("fallback" if summary_line else ""),
        "used_nav_summary": used_nav_summary,
    }


def _primary_ref_evidence_current_summary_needs_block_rescue(
    *,
    prompt: str,
    display_name: str,
    summary_line: str,
    summary_source: str,
    looks_focus_prefixed_ref_summary: Callable[..., bool],
    summary_line_needs_polish: Callable[..., bool],
) -> bool:
    return bool(
        (
            summary_source == "fallback"
            and looks_focus_prefixed_ref_summary(prompt, summary_line)
        )
        or summary_line_needs_polish(
            prompt=prompt,
            title=display_name,
            summary_line=summary_line,
        )
    )


def _select_primary_block_prompt_aligned_candidate(
    *,
    prompt: str,
    source_path: str,
    title: str,
    display_name: str,
    summary_line: str,
    summary_source: str,
    heading_path: str,
    meta_prompt_aligned_candidate: Mapping[str, Any] | None,
    anchor_target_kind: str,
    anchor_target_number: int,
    allow_summary_block_rescue: bool,
    allow_llm_translate: bool,
    looks_focus_prefixed_ref_summary: Callable[..., bool],
    summary_line_needs_polish: Callable[..., bool],
    sanitize_heading_path_ui: Callable[..., str],
    rank_prompt_aligned_ref_summary_candidate: Callable[..., tuple],
    choose_prompt_aligned_ref_summary_candidate_from_source_blocks: Callable[..., Mapping[str, Any] | None],
) -> dict[str, Any]:
    if (not allow_summary_block_rescue) or (not str(source_path or "").strip()):
        return {}

    current_summary_needs_block_rescue = _primary_ref_evidence_current_summary_needs_block_rescue(
        prompt=prompt,
        display_name=display_name,
        summary_line=summary_line,
        summary_source=summary_source,
        looks_focus_prefixed_ref_summary=looks_focus_prefixed_ref_summary,
        summary_line_needs_polish=summary_line_needs_polish,
    )
    meta_candidate_heading_path = sanitize_heading_path_ui(
        str((meta_prompt_aligned_candidate or {}).get("heading_path") or "").strip(),
        prompt=prompt,
        source_path=source_path,
    )
    meta_candidate_rebinds_heading = bool(
        meta_candidate_heading_path
        and meta_candidate_heading_path != heading_path
    )
    meta_prompt_score = (
        rank_prompt_aligned_ref_summary_candidate(
            dict(meta_prompt_aligned_candidate or {}),
            prompt=prompt,
            source_path=source_path,
            title=title,
            anchor_target_kind=anchor_target_kind,
            anchor_target_number=anchor_target_number,
        )[0]
        if meta_prompt_aligned_candidate
        else -1000.0
    )
    has_meta_prompt_aligned_candidate = bool(
        meta_prompt_aligned_candidate
        and meta_prompt_score >= 2.0
        and ((not current_summary_needs_block_rescue) or meta_candidate_rebinds_heading)
    )
    needs_block_rescue = bool(
        (bool(str(anchor_target_kind or "").strip()) and anchor_target_number > 0)
        or (
            (not has_meta_prompt_aligned_candidate)
            and (not summary_line)
        )
        or (
            (not has_meta_prompt_aligned_candidate)
            and summary_source == "fallback"
            and looks_focus_prefixed_ref_summary(prompt, summary_line)
        )
        or (
            (not has_meta_prompt_aligned_candidate)
            and current_summary_needs_block_rescue
        )
    )
    if not needs_block_rescue:
        return {}
    out = choose_prompt_aligned_ref_summary_candidate_from_source_blocks(
        prompt=prompt,
        source_path=source_path,
        title=title,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        allow_llm_translate=allow_llm_translate,
    )
    return dict(out or {}) if isinstance(out, Mapping) else {}


def _apply_primary_prompt_aligned_summary_candidate(
    *,
    prompt: str,
    source_path: str,
    title: str,
    display_name: str,
    summary_line: str,
    summary_source: str,
    heading_path: str,
    prompt_aligned_candidate: Mapping[str, Any] | None,
    anchor_target_kind: str,
    anchor_target_number: int,
    allow_summary_block_rescue: bool,
    sanitize_heading_path_ui: Callable[..., str],
    refs_heading_anchor_number: Callable[..., int],
    refs_heading_paths_related: Callable[..., bool],
    infer_heading_path_for_summary_from_source_blocks: Callable[..., str],
    summary_line_needs_polish: Callable[..., bool],
    ref_summary_focus_score: Callable[..., float],
    matched_focus_terms_for_ref_card: Callable[..., list[str]],
    ref_summary_surfaces_match: Callable[..., bool],
) -> dict[str, Any]:
    summary_line_out = str(summary_line or "").strip()
    summary_source_out = str(summary_source or "").strip()
    selected_heading_path = str(heading_path or "").strip()
    used_prompt_aligned_summary = False

    prompt_aligned_summary = str((prompt_aligned_candidate or {}).get("summary") or "").strip()
    if not prompt_aligned_summary:
        return {
            "summary_line": summary_line_out,
            "summary_source": summary_source_out,
            "used_prompt_aligned_summary": used_prompt_aligned_summary,
            "selected_heading_path": selected_heading_path,
        }

    candidate_heading_path = sanitize_heading_path_ui(
        str((prompt_aligned_candidate or {}).get("heading_path") or "").strip(),
        prompt=prompt,
        source_path=source_path,
    )
    if candidate_heading_path and anchor_target_kind and anchor_target_number > 0:
        candidate_anchor_num = refs_heading_anchor_number(anchor_target_kind, candidate_heading_path)
        if candidate_anchor_num > 0 and candidate_anchor_num != anchor_target_number:
            candidate_heading_path = ""
        elif (
            candidate_anchor_num <= 0
            and heading_path
            and (not refs_heading_paths_related(candidate_heading_path, heading_path))
        ):
            candidate_heading_path = ""
    if (not candidate_heading_path) and allow_summary_block_rescue:
        candidate_heading_path = infer_heading_path_for_summary_from_source_blocks(
            prompt=prompt,
            source_path=source_path,
            summary_line=prompt_aligned_summary,
            anchor_target_kind=anchor_target_kind,
            anchor_target_number=anchor_target_number,
        )
    current_unacceptable = bool(
        summary_line_out
        and summary_line_needs_polish(
            prompt=prompt,
            title=display_name,
            summary_line=summary_line_out,
        )
    )
    current_score = (
        ref_summary_focus_score(
            prompt=prompt,
            source_path=source_path,
            title=title,
            text=summary_line_out,
            anchor_target_kind=anchor_target_kind,
            anchor_target_number=anchor_target_number,
        )
        if summary_line_out
        else -1000.0
    )
    chosen_score = ref_summary_focus_score(
        prompt=prompt,
        source_path=source_path,
        title=title,
        text=prompt_aligned_summary,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
    )
    fallback_focus_hits = len(matched_focus_terms_for_ref_card(prompt, surface_text=summary_line_out))
    prompt_aligned_focus_hits = len(matched_focus_terms_for_ref_card(prompt, surface_text=prompt_aligned_summary))
    prefer_prompt_aligned_heading = bool(
        candidate_heading_path
        and candidate_heading_path != heading_path
        and summary_source_out == "fallback"
        and prompt_aligned_focus_hits >= max(1, fallback_focus_hits)
        and chosen_score >= (current_score - 0.25)
    )
    should_rebind_prompt_aligned_heading = bool(
        candidate_heading_path
        and candidate_heading_path != heading_path
        and ref_summary_surfaces_match(summary_line_out, prompt_aligned_summary)
    )
    if (
        (not summary_line_out)
        or current_unacceptable
        or (chosen_score >= (current_score + 0.75))
        or prefer_prompt_aligned_heading
    ):
        summary_line_out = prompt_aligned_summary
        used_prompt_aligned_summary = True
        summary_source_out = (
            "prompt_aligned_block"
            if str((prompt_aligned_candidate or {}).get("source_kind") or "").strip().lower() == "source_block"
            else "prompt_aligned"
        )
        should_rebind_prompt_aligned_heading = bool(
            candidate_heading_path
            and candidate_heading_path != heading_path
        )
    if should_rebind_prompt_aligned_heading:
        selected_heading_path = candidate_heading_path

    return {
        "summary_line": summary_line_out,
        "summary_source": summary_source_out,
        "used_prompt_aligned_summary": used_prompt_aligned_summary,
        "selected_heading_path": selected_heading_path,
    }


def _apply_reader_anchor_summary_override(
    *,
    reader_open: Mapping[str, Any] | None,
    prompt: str,
    source_path: str,
    display_name: str,
    summary_line: str,
    summary_source: str,
    anchor_target_kind: str,
    anchor_target_number: int,
    refs_heading_anchor_number: Callable[..., int],
    ref_summary_focus_score: Callable[..., float],
    build_evidence_backed_ref_summary_from_seed: Callable[..., str],
    prefer_zh_ref_card_locale: Callable[..., bool],
    summary_excerpt: Callable[..., str],
    normalize_ref_copy_text: Callable[[str], str],
) -> tuple[str, str]:
    summary_line_out = str(summary_line or "").strip()
    summary_source_out = str(summary_source or "").strip()
    if (
        (not isinstance(reader_open, Mapping))
        or (not str(anchor_target_kind or "").strip())
        or anchor_target_number <= 0
    ):
        return summary_line_out, summary_source_out

    reader_snippet = str(reader_open.get("snippet") or "").strip()
    reader_heading_path = str(reader_open.get("headingPath") or "").strip()
    reader_anchor_matches = bool(
        refs_heading_anchor_number(anchor_target_kind, reader_heading_path) == anchor_target_number
        or ref_summary_focus_score(
            prompt=prompt,
            source_path=source_path,
            title=display_name,
            text=reader_snippet,
            anchor_target_kind=anchor_target_kind,
            anchor_target_number=anchor_target_number,
        )
        >= 6.0
    )
    if (not reader_snippet) or (not reader_anchor_matches):
        return summary_line_out, summary_source_out

    current_anchor_score = ref_summary_focus_score(
        prompt=prompt,
        source_path=source_path,
        title=display_name,
        text=summary_line_out,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
    )
    reader_anchor_score = ref_summary_focus_score(
        prompt=prompt,
        source_path=source_path,
        title=display_name,
        text=reader_snippet,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
    )
    if reader_anchor_score < (current_anchor_score + 0.5):
        return summary_line_out, summary_source_out

    exact_summary = build_evidence_backed_ref_summary_from_seed(
        prompt=prompt,
        title=display_name,
        summary_line=reader_snippet,
        prefer_zh=prefer_zh_ref_card_locale(prompt, display_name, reader_snippet),
    ) or summary_excerpt(reader_snippet, max_sentences=2, max_len=240)
    if not exact_summary:
        return summary_line_out, summary_source_out
    return normalize_ref_copy_text(exact_summary), "exact_anchor"


def _resolve_primary_ref_evidence_summary_selection(
    *,
    meta: Mapping[str, Any] | None,
    prompt: str,
    source_path: str,
    display_name: str,
    citation_meta: Mapping[str, Any] | None,
    heading_path: str,
    heading: str,
    anchor_target_kind: str,
    anchor_target_number: int,
    allow_summary_block_rescue: bool,
    allow_llm_translate: bool,
    build_ref_navigation: Callable[..., Mapping[str, Any] | None],
    fallback_ref_ui_summary_line: Callable[..., str],
    choose_prompt_aligned_ref_summary_candidate: Callable[..., Mapping[str, Any] | None],
    looks_focus_prefixed_ref_summary: Callable[..., bool],
    summary_line_needs_polish: Callable[..., bool],
    sanitize_heading_path_ui: Callable[..., str],
    rank_prompt_aligned_ref_summary_candidate: Callable[..., tuple],
    choose_prompt_aligned_ref_summary_candidate_from_source_blocks: Callable[..., Mapping[str, Any] | None],
    pick_best_prompt_aligned_ref_summary_candidate: Callable[..., Mapping[str, Any] | None],
    refs_heading_anchor_number: Callable[..., int],
    refs_heading_paths_related: Callable[..., bool],
    infer_heading_path_for_summary_from_source_blocks: Callable[..., str],
    ref_summary_focus_score: Callable[..., float],
    matched_focus_terms_for_ref_card: Callable[..., list[str]],
    ref_summary_surfaces_match: Callable[..., bool],
) -> dict[str, Any]:
    candidate_title = _primary_ref_evidence_candidate_title(
        meta=meta,
        citation_meta=citation_meta,
        display_name=display_name,
    )
    base = _primary_ref_evidence_base_summary(
        meta=meta,
        prompt=prompt,
        heading=heading,
        citation_meta=citation_meta,
        allow_llm_translate=allow_llm_translate,
        build_ref_navigation=build_ref_navigation,
        fallback_ref_ui_summary_line=fallback_ref_ui_summary_line,
    )
    summary_line = str(base.get("summary_line") or "").strip()
    summary_source = str(base.get("summary_source") or "").strip()
    meta_prompt_aligned_candidate_raw = choose_prompt_aligned_ref_summary_candidate(
        meta or {},
        prompt=prompt,
        source_path=source_path,
        citation_meta=citation_meta,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        allow_llm_translate=allow_llm_translate,
    )
    meta_prompt_aligned_candidate = (
        dict(meta_prompt_aligned_candidate_raw or {})
        if isinstance(meta_prompt_aligned_candidate_raw, Mapping)
        else {}
    )
    block_prompt_aligned_candidate = _select_primary_block_prompt_aligned_candidate(
        prompt=prompt,
        source_path=source_path,
        title=candidate_title,
        display_name=display_name,
        summary_line=summary_line,
        summary_source=summary_source,
        heading_path=heading_path,
        meta_prompt_aligned_candidate=meta_prompt_aligned_candidate,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        allow_summary_block_rescue=allow_summary_block_rescue,
        allow_llm_translate=allow_llm_translate,
        looks_focus_prefixed_ref_summary=looks_focus_prefixed_ref_summary,
        summary_line_needs_polish=summary_line_needs_polish,
        sanitize_heading_path_ui=sanitize_heading_path_ui,
        rank_prompt_aligned_ref_summary_candidate=rank_prompt_aligned_ref_summary_candidate,
        choose_prompt_aligned_ref_summary_candidate_from_source_blocks=(
            choose_prompt_aligned_ref_summary_candidate_from_source_blocks
        ),
    )
    prompt_aligned_candidate_raw = pick_best_prompt_aligned_ref_summary_candidate(
        [meta_prompt_aligned_candidate, block_prompt_aligned_candidate],
        prompt=prompt,
        source_path=source_path,
        title=candidate_title,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
    )
    prompt_aligned_candidate = (
        dict(prompt_aligned_candidate_raw or {})
        if isinstance(prompt_aligned_candidate_raw, Mapping)
        else {}
    )
    selected = _apply_primary_prompt_aligned_summary_candidate(
        prompt=prompt,
        source_path=source_path,
        title=candidate_title,
        display_name=display_name,
        summary_line=summary_line,
        summary_source=summary_source,
        heading_path=heading_path,
        prompt_aligned_candidate=prompt_aligned_candidate,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        allow_summary_block_rescue=allow_summary_block_rescue,
        sanitize_heading_path_ui=sanitize_heading_path_ui,
        refs_heading_anchor_number=refs_heading_anchor_number,
        refs_heading_paths_related=refs_heading_paths_related,
        infer_heading_path_for_summary_from_source_blocks=infer_heading_path_for_summary_from_source_blocks,
        summary_line_needs_polish=summary_line_needs_polish,
        ref_summary_focus_score=ref_summary_focus_score,
        matched_focus_terms_for_ref_card=matched_focus_terms_for_ref_card,
        ref_summary_surfaces_match=ref_summary_surfaces_match,
    )

    return {
        "candidate_title": candidate_title,
        "nav": dict(base.get("nav") or {}),
        "used_nav_summary": bool(base.get("used_nav_summary")),
        "summary_line": str(selected.get("summary_line") or "").strip(),
        "summary_source": str(selected.get("summary_source") or "").strip(),
        "used_prompt_aligned_summary": bool(selected.get("used_prompt_aligned_summary")),
        "selected_heading_path": str(selected.get("selected_heading_path") or "").strip(),
        "prompt_aligned_candidate": prompt_aligned_candidate,
    }
