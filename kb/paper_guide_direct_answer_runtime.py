from __future__ import annotations

from kb.paper_guide_prompting import (
    _paper_guide_allows_citeless_answer,
    _paper_guide_prompt_family,
    _paper_guide_prompt_requests_doc_map,
    _paper_guide_prompt_requests_exact_method_support,
    _paper_guide_requested_box_numbers,
    _paper_guide_requested_section_targets,
)
from kb.paper_guide_answer_post_runtime import (
    _paper_guide_prompt_requests_exact_citation_support,
    _paper_guide_prompt_requests_exact_equation_support,
    _paper_guide_prompt_requests_exact_figure_caption_support,
    _resolve_exact_citation_lookup_support_from_source,
    _resolve_exact_equation_support_from_source,
    _resolve_exact_figure_panel_caption_support_from_source,
    _resolve_exact_method_support_from_source,
)
from kb.paper_guide_provenance import _extract_figure_number


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
    build_direct_abstract_answer,
    build_direct_citation_lookup_answer,
) -> str:
    if not paper_guide_mode:
        return ""
    prompt_text = str(prompt_for_user or "").strip()
    family = str(prompt_family or "").strip().lower()
    effective_family = family or _paper_guide_prompt_family(prompt_text)

    # Deterministic doc map is generated in postprocess; returning a non-empty override here
    # skips the LLM call for this request.
    if _paper_guide_prompt_requests_doc_map(prompt_text):
        return "Doc map (building verbatim section anchors)..."

    if effective_family == "equation" and _paper_guide_prompt_requests_exact_equation_support(prompt_text):
        rec = _resolve_exact_equation_support_from_source(
            paper_guide_bound_source_path or paper_guide_direct_source_path or paper_guide_focus_source_path,
            prompt=prompt_text,
            db_dir=db_dir,
        )
        equation_markdown = str((rec or {}).get("equation_markdown") or "").strip()
        if equation_markdown:
            return "Equation support (resolving exact equation + variable definitions)..."

    # Deterministic caption-clause resolver in postprocess; skip LLM to avoid mismatched panel anchors.
    if effective_family == "figure_walkthrough" and _paper_guide_prompt_requests_exact_figure_caption_support(prompt_text):
        rec = _resolve_exact_figure_panel_caption_support_from_source(
            paper_guide_bound_source_path or paper_guide_direct_source_path or paper_guide_focus_source_path,
            prompt=prompt_text,
            db_dir=db_dir,
        )
        locate_anchor = str((rec or {}).get("locate_anchor") or "").strip()
        if locate_anchor:
            return "Figure caption (resolving exact panel clause)..."

    if _paper_guide_allows_citeless_answer(effective_family):
        return str(
            build_direct_abstract_answer(
                prompt=prompt_text,
                source_path=paper_guide_direct_source_path,
                db_dir=db_dir,
                llm=llm,
            )
            or ""
        ).strip()

    # Exact-support citation lookup should not depend on the LLM answer formatting.
    # If the user asks "where exactly is that stated", deterministically surface the best in-paper clause
    # with its inline reference number(s).
    if effective_family == "citation_lookup" and _paper_guide_prompt_requests_exact_citation_support(prompt_text):
        rec = _resolve_exact_citation_lookup_support_from_source(
            paper_guide_bound_source_path or paper_guide_direct_source_path or paper_guide_focus_source_path,
            prompt=prompt_text,
            db_dir=db_dir,
        )
        locate_anchor = str((rec or {}).get("locate_anchor") or "").strip()
        heading_path = str((rec or {}).get("heading_path") or "").strip()
        ref_nums = [int(n) for n in list((rec or {}).get("ref_nums") or []) if int(n) > 0]
        if locate_anchor and ref_nums:
            ref_label = ", ".join(f"[{int(n)}]" for n in ref_nums[:4])
            if heading_path:
                return f"The paper cites {ref_label} for this point in {heading_path}:\n> {locate_anchor}"
            return f"The paper cites {ref_label} for this point:\n> {locate_anchor}"

    if effective_family != "citation_lookup":
        if effective_family not in {"method", "reproduce"}:
            return ""
        if not _paper_guide_prompt_requests_exact_method_support(prompt_text):
            return ""
        rec = _resolve_exact_method_support_from_source(
            paper_guide_bound_source_path or paper_guide_direct_source_path or paper_guide_focus_source_path,
            prompt=prompt_text,
            db_dir=db_dir,
        )
        locate_anchor = str((rec or {}).get("locate_anchor") or "").strip()
        heading_path = str((rec or {}).get("heading_path") or "").strip()
        if not locate_anchor:
            return ""
        if heading_path:
            return f"The paper states this explicitly in {heading_path}:\n> {locate_anchor}"
        return f"The paper states this explicitly:\n> {locate_anchor}"

    has_non_ref_target = bool(
        any(sec != "references" for sec in _paper_guide_requested_section_targets(prompt_text))
        or _paper_guide_requested_box_numbers(prompt_text)
        or (_extract_figure_number(prompt_text) > 0)
    )
    if has_non_ref_target:
        return ""

    return str(
        build_direct_citation_lookup_answer(
            prompt=prompt_text,
            source_path=paper_guide_focus_source_path or paper_guide_direct_source_path or paper_guide_bound_source_path,
            answer_hits=answer_hits,
            special_focus_block=special_focus_block,
            db_dir=db_dir,
        )
        or ""
    ).strip()
