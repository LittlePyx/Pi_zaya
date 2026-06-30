from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def _as_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _resolve_ref_card_why_line(
    *,
    prompt: str,
    display_name: str,
    heading_path: str,
    heading: str,
    section_label: str,
    subsection_label: str,
    nav: Mapping[str, Any] | None,
    summary_line: str,
    fallback_why_line_ui: Callable[..., str],
    build_prompt_aligned_ref_why_line: Callable[..., str],
    matched_focus_terms_for_ref_card: Callable[..., list[str]],
    is_definition_focus_prompt: Callable[[str], bool],
    why_line_explicitly_names_focus_term: Callable[[str, str], bool],
) -> dict[str, str]:
    nav_map = _as_dict(nav)
    heading_label = str(heading_path or heading or "").strip()
    why_line = str(nav_map.get("why") or "").strip()
    why_generation = "navigation" if why_line else ""
    if not why_line:
        why_line = fallback_why_line_ui(
            prompt=prompt,
            heading_label=heading_label,
            section_label=section_label,
            subsection_label=subsection_label,
            find_terms=list(nav_map.get("find") or []),
        )
        why_generation = "deterministic_grounded" if why_line else "fallback"

    prompt_aligned_why = build_prompt_aligned_ref_why_line(
        prompt=prompt,
        display_name=display_name,
        heading_path=heading_label,
        summary_line=summary_line,
        why_line=why_line,
    )
    why_focus_matches = matched_focus_terms_for_ref_card(prompt, surface_text=why_line)
    aligned_why_matches = matched_focus_terms_for_ref_card(prompt, surface_text=prompt_aligned_why)
    explicit_definition_focus_missing = bool(
        is_definition_focus_prompt(prompt)
        and why_line
        and (not why_line_explicitly_names_focus_term(prompt, why_line))
        and why_line_explicitly_names_focus_term(prompt, prompt_aligned_why)
    )
    if prompt_aligned_why and aligned_why_matches and (
        (not why_line)
        or (not why_focus_matches)
        or why_generation == "navigation"
        or explicit_definition_focus_missing
    ):
        why_line = prompt_aligned_why
        why_generation = "deterministic_grounded"

    return {
        "why_line": why_line,
        "why_generation": why_generation,
    }


def _resolve_ref_card_summary_kind_and_copy(
    *,
    prompt: str,
    display_name: str,
    heading_path: str,
    heading: str,
    summary_line: str,
    why_line: str,
    why_generation: str,
    citation_meta: Mapping[str, Any] | None,
    used_prompt_aligned_summary: bool,
    used_nav_summary: bool,
    allow_llm_translate: bool,
    infer_ref_summary_kind: Callable[..., str],
    align_ref_card_copy_to_user_locale: Callable[..., tuple[str, str]],
    matched_focus_terms_for_ref_card: Callable[..., list[str]],
    display_focus_term_for_ref_card: Callable[[str, str], str],
    ref_card_user_locale: Callable[..., str],
    finalize_ref_card_copy: Callable[..., tuple[str, str, bool]],
    prompt_reference_focus_action: Callable[[str], str],
) -> dict[str, Any]:
    heading_label = str(heading_path or heading or "").strip()
    summary_kind = infer_ref_summary_kind(
        summary_line=summary_line,
        citation_meta=_as_dict(citation_meta),
        used_prompt_aligned_summary=used_prompt_aligned_summary,
        used_nav_summary=used_nav_summary,
    )
    summary_out, why_out = align_ref_card_copy_to_user_locale(
        prompt=prompt,
        display_name=display_name,
        heading_path=heading_label,
        summary_line=summary_line,
        why_line=why_line,
        summary_kind=summary_kind,
        allow_llm_translate=allow_llm_translate,
    )
    copy_focus_terms = [
        display_focus_term_for_ref_card(prompt, term)
        for term in matched_focus_terms_for_ref_card(
            prompt,
            surface_text=" ".join(
                part
                for part in (display_name, heading_label, summary_out, why_out)
                if str(part or "").strip()
            ),
        )
    ]
    render_locale = ref_card_user_locale(prompt, display_name, heading_label, summary_out, why_out)
    summary_out, why_out, copy_changed = finalize_ref_card_copy(
        summary_line=summary_out,
        why_line=why_out,
        prefer_zh=render_locale == "zh",
        focus_terms=copy_focus_terms,
        heading_path=heading_label,
        action=prompt_reference_focus_action(prompt),
    )
    why_generation_out = str(why_generation or "").strip()
    if copy_changed:
        why_generation_out = "deterministic_grounded"

    return {
        "summary_line": summary_out,
        "why_line": why_out,
        "why_generation": why_generation_out,
        "summary_kind": summary_kind,
        "render_locale": render_locale,
    }


def _build_ref_card_basis_bundle(
    *,
    prompt: str,
    citation_meta: Mapping[str, Any] | None,
    summary_kind: str,
    summary_line: str,
    why_generation: str,
    why_line: str,
    build_ref_summary_surface_meta: Callable[..., dict[str, Any]],
    build_ref_summary_basis_meta: Callable[..., dict[str, Any]],
    build_ref_why_basis_meta: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    kind = str(summary_kind or "").strip().lower()
    citation = _as_dict(citation_meta)
    if kind == "abstract":
        summary_generation = str(citation.get("summary_generation") or "").strip().lower() or "translated_abstract"
    elif kind == "metadata":
        summary_generation = "metadata_only"
    else:
        summary_generation = "section_grounded"

    summary_surface = build_ref_summary_surface_meta(
        prompt=prompt,
        summary_kind=summary_kind,
        summary_line=summary_line,
    )
    summary_basis_meta = build_ref_summary_basis_meta(
        prompt=prompt,
        summary_kind=summary_kind,
        summary_generation=summary_generation,
        summary_line=summary_line,
    )
    why_basis_meta = build_ref_why_basis_meta(
        prompt=prompt,
        why_generation=why_generation,
        why_line=why_line,
    )
    return {
        "summary_surface": dict(summary_surface or {}),
        "summary_generation": summary_generation,
        "summary_basis_meta": dict(summary_basis_meta or {}),
        "why_basis_meta": dict(why_basis_meta or {}),
    }
