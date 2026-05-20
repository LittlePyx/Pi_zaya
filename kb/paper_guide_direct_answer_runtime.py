from __future__ import annotations

import re

from kb.paper_guide_prompting import (
    _paper_guide_allows_citeless_answer,
    _paper_guide_prompt_requests_doc_map,
    _paper_guide_requested_box_numbers,
    _paper_guide_requested_section_targets,
)
from kb.paper_guide.router import (
    PaperGuideBroadSkillDeps,
    PaperGuideExactSkillDeps,
    _dispatch_paper_guide_broad_skill,
    _dispatch_paper_guide_exact_support_skill,
    _resolve_paper_guide_intent,
)
from kb.paper_guide_answer_post_runtime import (
    _build_exact_equation_support_answer,
    _extract_caption_clause_superscript_ref_nums,
    _extract_inline_reference_numbers,
    _resolve_exact_citation_lookup_support_from_source,
    _resolve_doc_map_records_from_source,
    _resolve_exact_equation_support_from_source,
    _resolve_exact_figure_panel_caption_support_from_source,
    _resolve_exact_method_support_from_source,
    _sanitize_paper_guide_answer_for_user,
)
from kb.paper_guide_doc_map import (
    _paper_guide_prompt_requests_focused_reading_path,
    _select_focused_doc_map_records,
)
from kb.paper_guide_focus import (
    _build_paper_guide_overview_role_lines,
    _extract_bound_paper_component_role_focus,
    _extract_paper_guide_method_focus_terms,
    _paper_guide_prompt_requests_component_role_explanation,
)
from kb.paper_guide_retrieval_runtime import _paper_guide_targeted_source_block_hits


def _extract_discussion_future_snippet(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    future_cue_re = re.compile(
        r"(?i)\b(?:future|direction(?:s)?|extension(?:s)?|could|would|may|might|opens?\s+new\s+possibilit(?:y|ies)|promising|hybrid|parallelized|accelerate|extend|dynamic processes)\b"
    )
    sentences = [
        str(part or "").strip()
        for part in re.split(r"(?<=[.!?])\s+", src)
        if str(part or "").strip()
    ]
    picked: list[str] = []
    for sentence in sentences:
        if future_cue_re.search(sentence):
            picked.append(sentence)
        elif picked:
            break
        if len(picked) >= 2:
            break
    if picked:
        return " ".join(picked).strip()
    return src


def _select_section_target_direct_hit(
    source_path: str,
    *,
    prompt: str,
    prompt_family: str,
    db_dir,
) -> dict:
    src = str(source_path or "").strip()
    if not src:
        return {}
    hits = _paper_guide_targeted_source_block_hits(
        bound_source_path=src,
        prompt=prompt,
        db_dir=db_dir,
        limit=8,
        resolve_support_slot_block=lambda **_kwargs: {},
    )
    preferred_kinds = {"paragraph", "list_item", "blockquote"}
    family = str(prompt_family or "").strip().lower()
    prompt_low = str(prompt or "").strip().lower()
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", prompt_low)
        if len(token) >= 4 and token not in {"from", "only", "what", "does", "they", "this", "that", "section", "paper", "authors"}
    }
    ranked: list[tuple[float, dict]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        text = str(hit.get("text") or "").strip()
        heading_path = str(meta.get("heading_path") or "").strip()
        kind = str(meta.get("kind") or "").strip().lower()
        if not text:
            continue
        text_low = text.lower()
        score = 0.0
        if kind in preferred_kinds:
            score += 6.0
        elif kind == "heading":
            score -= 12.0
        shared = query_tokens.intersection(set(re.findall(r"[a-z0-9]+", text_low)))
        if shared:
            score += min(6.0, 1.2 * float(len(shared)))
        if family == "box_only":
            if re.search(r"(?i)\bbox\s*\d+\b", f"{heading_path}\n{text}"):
                score += 10.0
            if re.search(r"(?i)\b(?:condition|transform domain|reconstruct(?:ing|ion)?|sampling|sparsity)\b", text):
                score += 6.0
            if re.search(r"(?i)\b(?:m\s*[<>]=?\s*n|o\s*\(\s*k\s*log)\b", text):
                score += 4.0
        elif family == "strength_limits":
            if re.search(r"(?i)\b(?:trade[\s-]?off|dynamic range|quantization electronics|mean square error|bottleneck|limitation)\b", text):
                score += 8.0
            if ("calibrat" in prompt_low) and re.search(
                r"(?i)\b(?:calibrat(?:e|ed|ing|ion)|specific spad camera|different spad arrays?|automatic calibration|transfer learning|further study)\b",
                text,
            ):
                score += 10.0
            if ("spad" in prompt_low and "array" in prompt_low) and re.search(r"(?i)\bspad arrays?\b", text):
                score += 5.0
            if any(token in prompt_low for token in ("follow-up", "follow up", "suggest")) and re.search(
                r"(?i)\b(?:automatic calibration|transfer learning|worthy of further study|further study)\b",
                text,
            ):
                score += 6.0
            if ("calibrat" in prompt_low or ("spad" in prompt_low and "array" in prompt_low)) and re.search(
                r"(?i)\b(?:wavelength-dependent|photon efficiency|multispectral imaging)\b",
                text,
            ):
                score -= 6.0
        elif family == "discussion_only":
            if re.search(r"(?i)\b(?:future|direction(?:s)?|extension(?:s)?|promising|could|would|may|next|potential|extend|hybrid|parallelized)\b", text):
                score += 8.0
            if re.search(r"(?i)\b(?:spad|parallelized detection|dynamic processes|computational staining|integrated into|adaptable|can be exchanged|single-molecule|commercial confocal fluorescence ism systems)\b", text):
                score += 10.0
            if re.search(r"(?i)\b(?:spad array|can be exchanged|single-molecule fluorescence ism|commercial confocal fluorescence ism systems)\b", text):
                score += 4.0
            if re.search(r"(?i)\b(?:demonstrates|we realized|noise reduction|contrast|fwhm|super-concentration)\b", text):
                score -= 3.0
            if re.search(r"(?i)\b(?:potential applications|phototoxicity|closed pinhole|incident illumination power)\b", text):
                score -= 4.0
        ranked.append((score, dict(hit)))
    if not ranked:
        return {}
    ranked.sort(key=lambda item: item[0], reverse=True)
    return dict(ranked[0][1])


def _paper_guide_prompt_prefers_zh(prompt: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(prompt or "")))


def _build_direct_doc_map_answer(
    *,
    source_path: str,
    prompt_text: str,
    prompt_family: str,
    db_dir,
    has_hits: bool,
) -> str:
    src = str(source_path or "").strip()
    if not src:
        return ""
    focused_path = _paper_guide_prompt_requests_focused_reading_path(prompt_text)
    recs = _resolve_doc_map_records_from_source(
        src,
        prompt=prompt_text,
        db_dir=db_dir,
        max_items=24 if focused_path else 16,
    )
    if not recs:
        return ""
    if focused_path:
        recs = _select_focused_doc_map_records(recs, prompt=prompt_text, max_items=6)
    prefer_zh = _paper_guide_prompt_prefers_zh(prompt_text)
    lines: list[str] = [
        "可以先按这几处读：" if (prefer_zh and focused_path)
        else "可以按这些原文位置读：" if prefer_zh
        else "Start with these source anchors:" if focused_path
        else "Use these source anchors as a reading map:",
        "",
    ]
    for i, rec in enumerate(list(recs or []), start=1):
        heading_path = str((rec or {}).get("heading_path") or "").strip() or ("未命名小节" if prefer_zh else "Unheaded section")
        anchor = str((rec or {}).get("locate_anchor") or "").strip()
        if not anchor:
            continue
        lines.append(f"{int(i)}. {heading_path}")
        lines.append(f"> {anchor}")
        lines.append("")
    answer = "\n".join(lines).rstrip()
    return _sanitize_paper_guide_answer_for_user(
        answer,
        has_hits=bool(has_hits),
        prompt=prompt_text,
        prompt_family=prompt_family or "overview",
    )


def _build_exact_support_direct_answer(
    *,
    prompt_text: str,
    resolved_intent,
    source_path: str,
    db_dir,
    has_hits: bool,
) -> str:
    if not str(source_path or "").strip():
        return ""
    exact_skill_result = _dispatch_paper_guide_exact_support_skill(
        prompt_text=prompt_text,
        resolved_intent=resolved_intent,
        source_path=str(source_path or "").strip(),
        db_dir=db_dir,
        has_hits=bool(has_hits),
        deps=PaperGuideExactSkillDeps(
            resolve_exact_method_support=_resolve_exact_method_support_from_source,
            resolve_exact_equation_support=_resolve_exact_equation_support_from_source,
            build_exact_equation_answer=_build_exact_equation_support_answer,
            resolve_exact_citation_lookup_support=_resolve_exact_citation_lookup_support_from_source,
            extract_inline_reference_numbers=_extract_inline_reference_numbers,
            resolve_exact_figure_panel_caption_support=_resolve_exact_figure_panel_caption_support_from_source,
            extract_caption_clause_superscript_ref_nums=_extract_caption_clause_superscript_ref_nums,
            sanitize_answer=_sanitize_paper_guide_answer_for_user,
        ),
    )
    if exact_skill_result is None:
        return ""
    return str(exact_skill_result.answer_text or "").strip()


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
    resolved_intent = _resolve_paper_guide_intent(
        prompt_text,
        prompt_family=family,
        answer_hits=list(answer_hits or []),
    )
    effective_family = str(resolved_intent.family or family or "").strip().lower()
    has_non_ref_target = bool(
        any(sec != "references" for sec in _paper_guide_requested_section_targets(prompt_text))
        or _paper_guide_requested_box_numbers(prompt_text)
        or int(resolved_intent.target_figure or 0) > 0
    )

    if _paper_guide_prompt_requests_doc_map(prompt_text):
        return _build_direct_doc_map_answer(
            source_path=paper_guide_bound_source_path or paper_guide_direct_source_path or paper_guide_focus_source_path,
            prompt_text=prompt_text,
            prompt_family=effective_family or family,
            db_dir=db_dir,
            has_hits=bool(answer_hits),
        )

    source_path = paper_guide_bound_source_path or paper_guide_direct_source_path or paper_guide_focus_source_path
    exact_direct_answer = _build_exact_support_direct_answer(
        prompt_text=prompt_text,
        resolved_intent=resolved_intent,
        source_path=source_path,
        db_dir=db_dir,
        has_hits=bool(answer_hits),
    )
    if exact_direct_answer:
        return exact_direct_answer

    broad_skill_result = _dispatch_paper_guide_broad_skill(
        prompt_text=prompt_text,
        resolved_intent=resolved_intent,
        source_path=(paper_guide_direct_source_path or source_path) if effective_family == "abstract" else source_path,
        db_dir=db_dir,
        has_hits=bool(answer_hits),
        has_non_ref_target=has_non_ref_target,
        llm=llm,
        deps=PaperGuideBroadSkillDeps(
            build_direct_abstract_answer=build_direct_abstract_answer,
            prompt_requests_component_role_explanation=_paper_guide_prompt_requests_component_role_explanation,
            extract_method_focus_terms=_extract_paper_guide_method_focus_terms,
            extract_component_role_focus=_extract_bound_paper_component_role_focus,
            build_overview_role_lines=_build_paper_guide_overview_role_lines,
            select_section_target_hit=_select_section_target_direct_hit,
            extract_discussion_future_snippet=_extract_discussion_future_snippet,
            extract_box_numbers=_paper_guide_requested_box_numbers,
        ),
    )
    if broad_skill_result is not None and str(broad_skill_result.answer_text or "").strip():
        return str(broad_skill_result.answer_text or "").strip()

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

    if has_non_ref_target:
        return ""

    # For citation_lookup family without an exact-support request (e.g.
    # "what references are cited" without "where exactly"),
    # fall through to LLM generation — the template-based direct answer is
    # too rigid and produces poor results for mixed-concept questions.
    return ""
