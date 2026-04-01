from __future__ import annotations

from pathlib import Path

from kb.answer_contract import (
    _apply_answer_contract_v1,
    _build_answer_quality_probe,
    _enhance_kb_miss_fallback,
    _reconcile_kb_notice,
)
from kb.paper_guide_prompting import _paper_guide_prompt_family
from kb.paper_guide_postprocess import (
    _sanitize_paper_guide_answer_for_user,
    _sanitize_structured_cite_tokens,
    _strip_model_ref_section,
)
from ui.chat_widgets import _normalize_math_markdown


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
    apply_paper_guide_answer_postprocess,
    maybe_append_library_figure_markdown,
    validate_structured_citations,
) -> dict:
    effective_paper_guide_family = (
        str(paper_guide_prompt_family or "").strip().lower()
        or str(_paper_guide_prompt_family(prompt_for_user or prompt) or "").strip().lower()
    )
    sanitize_paper_guide_family = effective_paper_guide_family or "overview"
    answer = _normalize_math_markdown(
        _strip_model_ref_section(_sanitize_structured_cite_tokens(partial or ""))
    ).strip() or "(No text returned)"
    answer = _reconcile_kb_notice(answer, has_hits=bool(answer_hits))
    if paper_guide_contract_enabled:
        answer = _apply_answer_contract_v1(
            answer,
            prompt=prompt,
            has_hits=bool(answer_hits),
            intent=answer_intent,
            depth=answer_depth,
            output_mode=answer_output_mode,
        )
    answer = _enhance_kb_miss_fallback(
        answer,
        has_hits=bool(answer_hits),
        intent=answer_intent,
        depth=answer_depth,
        contract_enabled=bool(paper_guide_contract_enabled),
        output_mode=answer_output_mode,
    )
    answer, paper_guide_support_resolution = apply_paper_guide_answer_postprocess(
        answer,
        paper_guide_mode=paper_guide_mode,
        prompt=prompt,
        prompt_for_user=prompt_for_user,
        prompt_family=paper_guide_prompt_family,
        special_focus_block=paper_guide_special_focus_block,
        focus_source_path=paper_guide_focus_source_path,
        direct_source_path=paper_guide_direct_source_path,
        bound_source_path=paper_guide_bound_source_path,
        db_dir=db_dir,
        answer_hits=answer_hits,
        support_slots=list(paper_guide_support_slots or []),
        cards=list(paper_guide_evidence_cards or []),
        locked_citation_source=locked_citation_source,
    )
    answer = maybe_append_library_figure_markdown(
        answer,
        prompt=prompt,
        answer_hits=answer_hits,
        bound_source_path=paper_guide_bound_source_path,
    )
    answer, citation_validation = validate_structured_citations(
        answer,
        answer_hits=answer_hits,
        db_dir=db_dir,
        locked_source=locked_citation_source,
        paper_guide_mode=bool(paper_guide_mode),
        paper_guide_candidate_refs_by_source=dict(paper_guide_candidate_refs_by_source or {}),
        paper_guide_support_slots=list(paper_guide_support_slots or []),
        paper_guide_support_resolution=list(paper_guide_support_resolution or []),
    )
    # Citation validation may legitimately rewrite or inject structured cite markers for grounding,
    # but the final user-facing paper-guide answer still needs the same family-aware sanitization pass.
    if paper_guide_mode:
        answer = _sanitize_paper_guide_answer_for_user(
            answer,
            has_hits=bool(answer_hits),
            prompt=prompt_for_user or prompt,
            prompt_family=sanitize_paper_guide_family,
        )
    answer_quality = _build_answer_quality_probe(
        answer,
        has_hits=bool(answer_hits),
        contract_enabled=bool(paper_guide_contract_enabled),
        intent=answer_intent,
        depth=answer_depth,
        output_mode=answer_output_mode,
        paper_guide_mode=bool(paper_guide_mode),
        prompt_family=sanitize_paper_guide_family,
    )
    return {
        "answer": answer,
        "paper_guide_support_resolution": list(paper_guide_support_resolution or []),
        "citation_validation": citation_validation,
        "answer_quality": answer_quality,
    }
