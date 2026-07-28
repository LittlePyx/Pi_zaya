from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def _as_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _select_hit_initial_heading_path(
    *,
    meta: Mapping[str, Any] | None,
    hit_text: str,
    prompt: str,
    leading_markdown_heading_from_hit_text: Callable[[str], str],
    refs_section_intent_heading_score: Callable[[str, str], float],
    normalize_title_identity: Callable[[str], str],
) -> str:
    meta_map = _as_dict(meta)
    initial_heading_path = str(meta_map.get("ref_best_heading_path") or meta_map.get("heading_path") or "").strip()
    leading_text_heading = leading_markdown_heading_from_hit_text(str(hit_text or ""))
    if not leading_text_heading:
        return initial_heading_path

    current_heading_score = refs_section_intent_heading_score(prompt, initial_heading_path)
    leading_heading_score = refs_section_intent_heading_score(prompt, leading_text_heading)
    current_norm = normalize_title_identity(initial_heading_path)
    leading_norm = normalize_title_identity(leading_text_heading)
    if (
        (not current_norm)
        or current_norm in {"abstract", "references"}
        or (
            leading_heading_score >= current_heading_score + 0.75
            and leading_norm
            and leading_norm not in current_norm
        )
    ):
        return leading_text_heading
    return initial_heading_path


def _resolve_hit_anchor_target(
    *,
    meta: Mapping[str, Any] | None,
    prompt: str,
    positive_int: Callable[[Any], int],
    extract_figure_number: Callable[[str], int],
    extract_equation_number: Callable[[str], int],
) -> tuple[str, int]:
    meta_map = _as_dict(meta)
    anchor_target_kind = str(meta_map.get("anchor_target_kind") or "").strip().lower()
    anchor_target_number = positive_int(meta_map.get("anchor_target_number"))
    if anchor_target_kind and anchor_target_number > 0:
        return anchor_target_kind, anchor_target_number

    prompt_figure_number = extract_figure_number(prompt)
    if prompt_figure_number > 0:
        return "figure", prompt_figure_number
    prompt_equation_number = extract_equation_number(prompt)
    if prompt_equation_number > 0:
        return "equation", prompt_equation_number
    return anchor_target_kind, anchor_target_number


def _load_hit_citation_context(
    *,
    source_path: str,
    pdf_root: Any,
    lib_store: Any,
    preloaded_citation_meta: Mapping[str, Any] | None,
    resolve_pdf_for_source: Callable[[Any, str], Any],
    display_source_name: Callable[[str, Any, Any], str],
) -> dict[str, Any]:
    pdf_path = resolve_pdf_for_source(pdf_root, source_path)
    display_name = display_source_name(source_path, pdf_path, lib_store)
    citation_meta: dict[str, Any] = {}
    preload_map = _as_dict(preloaded_citation_meta)
    preload_meta = preload_map.get(source_path) if source_path else None
    if isinstance(preload_meta, Mapping) and preload_meta:
        citation_meta = dict(preload_meta)
    if pdf_path is not None and lib_store is not None:
        try:
            if not citation_meta:
                raw_meta = lib_store.get_citation_meta(pdf_path) or {}
                citation_meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
        except Exception:
            if not citation_meta:
                citation_meta = {}
    return {
        "pdf_path": pdf_path,
        "display_name": display_name,
        "citation_meta": citation_meta,
    }


def _build_ref_hit_context(
    *,
    hit: Mapping[str, Any] | None,
    prompt: str,
    pdf_root: Any,
    lib_store: Any,
    preloaded_citation_meta: Mapping[str, Any] | None,
    leading_markdown_heading_from_hit_text: Callable[[str], str],
    refs_section_intent_heading_score: Callable[[str, str], float],
    normalize_title_identity: Callable[[str], str],
    resolve_ref_ui_heading_context: Callable[..., dict[str, str]],
    top_heading: Callable[[str], str],
    safe_page_range: Callable[[Mapping[str, Any]], tuple[int, int]],
    effective_ui_score: Callable[[Mapping[str, Any]], tuple[Any, bool]],
    positive_int: Callable[[Any], int],
    extract_figure_number: Callable[[str], int],
    extract_equation_number: Callable[[str], int],
    non_negative_float: Callable[[Any], float],
    build_semantic_badges: Callable[..., list[Any]],
    resolve_pdf_for_source: Callable[[Any, str], Any],
    display_source_name: Callable[[str, Any, Any], str],
) -> dict[str, Any]:
    hit_map = _as_dict(hit)
    meta = _as_dict(hit_map.get("meta") if isinstance(hit_map.get("meta"), Mapping) else {})
    source_path = str(meta.get("source_path") or "").strip()
    ref_pack_state = str(meta.get("ref_pack_state") or "").strip().lower()
    initial_heading_path = _select_hit_initial_heading_path(
        meta=meta,
        hit_text=str(hit_map.get("text") or ""),
        prompt=prompt,
        leading_markdown_heading_from_hit_text=leading_markdown_heading_from_hit_text,
        refs_section_intent_heading_score=refs_section_intent_heading_score,
        normalize_title_identity=normalize_title_identity,
    )
    heading_context = resolve_ref_ui_heading_context(
        prompt=prompt,
        source_path=source_path,
        heading_path=initial_heading_path,
        heading_fallback=str(
            meta.get("top_heading")
            or top_heading(str(meta.get("heading_path") or ""))
            or ""
        ).strip(),
        section_label=str(meta.get("ref_section") or "").strip(),
        subsection_label=str(meta.get("ref_subsection") or "").strip(),
    )

    p0, p1 = safe_page_range(meta)
    score, score_pending = effective_ui_score(hit_map)
    anchor_target_kind, anchor_target_number = _resolve_hit_anchor_target(
        meta=meta,
        prompt=prompt,
        positive_int=positive_int,
        extract_figure_number=extract_figure_number,
        extract_equation_number=extract_equation_number,
    )
    anchor_match_score = non_negative_float(meta.get("anchor_match_score"))
    explicit_doc_match_score = non_negative_float(meta.get("explicit_doc_match_score"))
    anchor_target_label = str(meta.get("anchor_target_label") or "").strip()
    semantic_badges = build_semantic_badges(
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        anchor_match_score=anchor_match_score,
        explicit_doc_match_score=explicit_doc_match_score,
        anchor_target_label=anchor_target_label,
    )
    citation_context = _load_hit_citation_context(
        source_path=source_path,
        pdf_root=pdf_root,
        lib_store=lib_store,
        preloaded_citation_meta=preloaded_citation_meta,
        resolve_pdf_for_source=resolve_pdf_for_source,
        display_source_name=display_source_name,
    )

    return {
        "meta": meta,
        "source_path": source_path,
        "ref_pack_state": ref_pack_state,
        "heading_context": dict(heading_context or {}),
        "heading_path": str((heading_context or {}).get("heading_path") or "").strip(),
        "heading": str((heading_context or {}).get("heading") or "").strip(),
        "section_label": str((heading_context or {}).get("section_label") or "").strip(),
        "subsection_label": str((heading_context or {}).get("subsection_label") or "").strip(),
        "page_start": p0,
        "page_end": p1,
        "score": score,
        "score_pending": bool(score_pending),
        "anchor_target_kind": anchor_target_kind,
        "anchor_target_number": anchor_target_number,
        "anchor_target_label": anchor_target_label,
        "anchor_match_score": anchor_match_score,
        "explicit_doc_match_score": explicit_doc_match_score,
        "semantic_badges": semantic_badges,
        "pdf_path": citation_context.get("pdf_path"),
        "display_name": str(citation_context.get("display_name") or "").strip(),
        "citation_meta": _as_dict(citation_context.get("citation_meta") if isinstance(citation_context.get("citation_meta"), Mapping) else {}),
    }


def _apply_section_intent_rescue_context(
    *,
    meta: Mapping[str, Any] | None,
    hit_text: str,
    heading_path: str,
    heading: str,
    section_label: str,
    subsection_label: str,
    summary_line: str,
    summary_source: str,
    top_heading: Callable[[str], str],
    summary_excerpt: Callable[..., str],
) -> dict[str, str]:
    meta_map = _as_dict(meta)
    heading_path_out = str(heading_path or "").strip()
    heading_out = str(heading or "").strip()
    section_label_out = str(section_label or "").strip()
    subsection_label_out = str(subsection_label or "").strip()
    summary_line_out = str(summary_line or "").strip()
    summary_source_out = str(summary_source or "").strip()

    if not bool(meta_map.get("section_intent_rescue")):
        return {
            "heading_path": heading_path_out,
            "heading": heading_out,
            "section_label": section_label_out,
            "subsection_label": subsection_label_out,
            "summary_line": summary_line_out,
            "summary_source": summary_source_out,
        }

    rescue_heading_path = str(meta_map.get("ref_best_heading_path") or meta_map.get("heading_path") or "").strip()
    if rescue_heading_path:
        heading_path_out = rescue_heading_path
        heading_out = str(
            rescue_heading_path.split(" / ")[-1]
            if " / " in rescue_heading_path
            else rescue_heading_path
        ).strip()
        section_label_out = str(meta_map.get("ref_section") or top_heading(rescue_heading_path) or "").strip()
        subsection_label_out = str(meta_map.get("ref_subsection") or heading_out).strip()
    rescue_summary = summary_excerpt(str(hit_text or ""), max_sentences=2, max_len=260)
    if rescue_summary:
        summary_line_out = rescue_summary
        summary_source_out = "section_intent_rescue"

    return {
        "heading_path": heading_path_out,
        "heading": heading_out,
        "section_label": section_label_out,
        "subsection_label": subsection_label_out,
        "summary_line": summary_line_out,
        "summary_source": summary_source_out,
    }
