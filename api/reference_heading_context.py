from __future__ import annotations

from collections.abc import Callable


def _resolve_ref_ui_heading_context(
    *,
    prompt: str,
    source_path: str,
    heading_path: str,
    heading_fallback: str = "",
    section_label: str = "",
    subsection_label: str = "",
    sanitize_heading_path_ui: Callable[..., str],
    top_heading: Callable[[str], str],
    is_non_navigational_heading_ui: Callable[..., bool],
    looks_like_doc_title_heading_ui: Callable[[str, str], bool],
    split_section_subsection: Callable[[str], tuple[str, str]],
) -> dict[str, str]:
    heading_path_norm = sanitize_heading_path_ui(
        str(heading_path or "").strip(),
        prompt=prompt,
        source_path=source_path,
    )
    heading = str(
        heading_fallback
        or top_heading(heading_path_norm)
        or ""
    ).strip()
    if heading and is_non_navigational_heading_ui(heading, prompt=prompt, source_path=source_path):
        heading = ""
    if heading and looks_like_doc_title_heading_ui(heading, source_path):
        heading = ""

    section = str(section_label or "").strip()
    subsection = str(subsection_label or "").strip()
    if section and is_non_navigational_heading_ui(section, prompt=prompt, source_path=source_path):
        section = ""
    if subsection and is_non_navigational_heading_ui(subsection, prompt=prompt, source_path=source_path):
        subsection = ""
    if (not section) and heading_path_norm:
        section, subsection = split_section_subsection(heading_path_norm)
    if section and looks_like_doc_title_heading_ui(section, source_path):
        section = ""
        subsection = ""

    return {
        "heading_path": heading_path_norm,
        "heading": heading,
        "section_label": section,
        "subsection_label": subsection,
    }


def _should_allow_ref_summary_block_rescue(
    *,
    prompt: str,
    source_path: str,
    ref_pack_state: str,
    allow_exact_locate: bool,
    extract_figure_number: Callable[[str], int],
    extract_equation_number: Callable[[str], int],
    prompt_requires_explicit_focus_match: Callable[[str], bool],
) -> bool:
    if not str(source_path or "").strip():
        return False
    if allow_exact_locate:
        return True
    if extract_figure_number(prompt) > 0 or extract_equation_number(prompt) > 0:
        return True
    if str(ref_pack_state or "").strip().lower() != "pending":
        return False
    return bool(prompt_requires_explicit_focus_match(prompt))
