"""Paper Guide grounder.

This package-oriented module is the home for support-marker grounding and
canonical locate-target assembly. It still delegates most heavy matching logic
to the legacy flat module ``kb.paper_guide_grounding_runtime`` so the migration
can stay incremental and low-risk.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from ..paper_guide_structured_index_runtime import load_paper_guide_figure_index


def _legacy_runtime():
    return import_module("kb.paper_guide_grounding_runtime")


def _extract_inline_reference_numbers(text: str, *, max_candidates: int = 8) -> list[int]:
    return _legacy_runtime()._extract_inline_reference_numbers(text, max_candidates=max_candidates)


def _extract_inline_reference_specs(text: str) -> list[str]:
    return _legacy_runtime()._extract_inline_reference_specs(text)


def _build_paper_guide_support_slots(
    cards: list[dict],
    *,
    prompt: str = "",
    prompt_family: str = "",
    db_dir: Path | None = None,
    max_slots: int = 4,
    target_scope: dict | None = None,
) -> list[dict]:
    return _legacy_runtime()._build_paper_guide_support_slots(
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
    return _legacy_runtime()._build_paper_guide_support_slots_block(slots, max_slots=max_slots)


def _extract_paper_guide_locate_anchor(text: str, *, max_chars: int = 220) -> str:
    return _legacy_runtime()._extract_paper_guide_locate_anchor(text, max_chars=max_chars)


def _extract_paper_guide_ref_spans(text: str, *, max_spans: int = 4) -> list[dict]:
    return _legacy_runtime()._extract_paper_guide_ref_spans(text, max_spans=max_spans)


def _inject_paper_guide_support_markers(
    answer: str,
    *,
    support_slots: list[dict],
    prompt_family: str = "",
    max_injections: int = 3,
) -> str:
    return _legacy_runtime()._inject_paper_guide_support_markers(
        answer,
        support_slots=support_slots,
        prompt_family=prompt_family,
        max_injections=max_injections,
    )


def _resolve_reference_index_support_from_source(
    *args,
    **kwargs,
) -> dict:
    return _legacy_runtime()._resolve_reference_index_support_from_source(*args, **kwargs)


def _resolve_paper_guide_support_markers(
    answer: str,
    *,
    support_slots: list[dict],
    prompt_family: str = "",
    db_dir: Path | None = None,
) -> tuple[str, list[dict]]:
    return _legacy_runtime()._resolve_paper_guide_support_markers(
        answer,
        support_slots=support_slots,
        prompt_family=prompt_family,
        db_dir=db_dir,
    )


def _resolve_paper_guide_support_slot_block(
    source_path: str,
    *,
    snippet: str,
    heading: str = "",
    prompt_family: str = "",
    claim_type: str = "",
    db_dir: Path | None = None,
    block_cache: dict[str, tuple[Path, list[dict]]] | None = None,
    atom_cache: dict[str, list[dict]] | None = None,
    target_scope: dict | None = None,
) -> dict:
    return _legacy_runtime()._resolve_paper_guide_support_slot_block(
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


def _resolve_paper_guide_support_ref_num(slot: dict, *, context_text: str = "") -> tuple[int | None, str]:
    return _legacy_runtime()._resolve_paper_guide_support_ref_num(slot, context_text=context_text)


def _paper_guide_cue_tokens(text: str) -> list[str]:
    return _legacy_runtime()._paper_guide_cue_tokens(text)


def _is_paper_guide_broad_summary_line(text: str, *, prompt_family: str = "") -> bool:
    return _legacy_runtime()._is_paper_guide_broad_summary_line(text, prompt_family=prompt_family)


def _is_paper_guide_support_meta_line(text: str) -> bool:
    return _legacy_runtime()._is_paper_guide_support_meta_line(text)


def _normalize_paper_guide_support_surface(text: str) -> str:
    return _legacy_runtime()._normalize_paper_guide_support_surface(text)


def _paper_guide_support_claim_type(
    *,
    prompt_family: str,
    heading: str = "",
    snippet: str = "",
    candidate_refs: list[int] | None = None,
    ref_spans: list[dict] | None = None,
) -> str:
    return _legacy_runtime()._paper_guide_support_claim_type(
        prompt_family=prompt_family,
        heading=heading,
        snippet=snippet,
        candidate_refs=candidate_refs,
        ref_spans=ref_spans,
    )


def _paper_guide_support_cite_policy(*, claim_type: str, prompt_family: str) -> str:
    return _legacy_runtime()._paper_guide_support_cite_policy(
        claim_type=claim_type,
        prompt_family=prompt_family,
    )


def _paper_guide_support_focus_tokens(*parts: str, limit: int = 18) -> set[str]:
    return _legacy_runtime()._paper_guide_support_focus_tokens(*parts, limit=limit)


def _paper_guide_support_rule_tokens(slot: dict) -> set[str]:
    return _legacy_runtime()._paper_guide_support_rule_tokens(slot)


def _paper_guide_support_segment_spans(answer_markdown: str) -> list[dict]:
    return _legacy_runtime()._paper_guide_support_segment_spans(answer_markdown)


def _select_paper_guide_support_slot_for_context(
    slots: list[dict],
    *,
    context_text: str = "",
) -> dict | None:
    return _legacy_runtime()._select_paper_guide_support_slot_for_context(
        slots,
        context_text=context_text,
    )


def _extract_caption_fragment_for_letters(text: str, letters: set[str]) -> str:
    return _legacy_runtime()._extract_caption_fragment_for_letters(text, letters)


def _extract_caption_fragment_for_letters_fallback(text: str, letters: set[str]) -> str:
    raw = str(text or "").strip()
    panels = {
        str(ch or "").strip().lower()
        for ch in set(letters or set())
        if str(ch or "").strip()
    }
    if (not raw) or (not panels):
        return ""

    markers: list[tuple[int, str]] = []
    import re

    for m in re.finditer(r"\(\s*([A-Za-z])\s*\)", raw):
        markers.append((int(m.start()), str(m.group(1) or "").strip().lower()))
    for m in re.finditer(r"\*\*\s*([A-Za-z])\s*\*\*", raw):
        markers.append((int(m.start()), str(m.group(1) or "").strip().lower()))
    for m in re.finditer(r"(?m)(?:^|[.;:])\s*([A-Za-z])\s+(?=[A-Z])", raw):
        markers.append((int(m.start(1)), str(m.group(1) or "").strip().lower()))
    if not markers:
        return ""
    markers.sort(key=lambda item: (int(item[0]), str(item[1])))

    for panel in panels:
        start = -1
        end = -1
        for pos, letter in markers:
            if letter == panel:
                start = int(pos)
                break
        if start < 0:
            continue
        for pos, _letter in markers:
            if int(pos) > start:
                end = int(pos)
                break
        frag = raw[start:end].strip() if end > 0 else raw[start:].strip()
        frag = frag.rstrip(" ;,.")
        if frag:
            return frag
    return ""


def _ground_paper_guide_answer_support(
    answer: str,
    *,
    support_slots: list[dict],
    prompt_family: str = "",
    db_dir: Path | None = None,
    max_injections: int = 3,
) -> tuple[str, list[dict]]:
    text = _inject_paper_guide_support_markers(
        answer,
        support_slots=support_slots,
        prompt_family=prompt_family,
        max_injections=max_injections,
    )
    return _resolve_paper_guide_support_markers(
        text,
        support_slots=support_slots,
        prompt_family=prompt_family,
        db_dir=db_dir,
    )


def _extract_paper_guide_segment_anchor_number(seg: dict | None) -> int:
    if not isinstance(seg, dict):
        return 0
    for raw_num in (
        seg.get("equation_number"),
        seg.get("support_slot_figure_number"),
    ):
        try:
            value = int(raw_num or 0)
        except Exception:
            value = 0
        if value > 0:
            return value
    return 0


def _resolve_paper_guide_panel_clause_snippet(
    seg: dict | None,
    *,
    block_lookup: dict[str, dict] | None = None,
    md_path: str | Path | None = None,
) -> str:
    if not isinstance(seg, dict):
        return ""
    claim_type = str(seg.get("claim_type") or "").strip().lower()
    if claim_type != "figure_panel":
        return ""
    panel_letters = {
        str(ch or "").strip().lower()
        for ch in list(seg.get("support_slot_panel_letters") or [])
        if str(ch or "").strip()
    }
    if not panel_letters:
        return ""

    lookup = {
        str(block_id or "").strip(): dict(block)
        for block_id, block in dict(block_lookup or {}).items()
        if str(block_id or "").strip() and isinstance(block, dict)
    }

    def _block_text_for(block_id: str) -> str:
        block = lookup.get(str(block_id or "").strip())
        if not isinstance(block, dict):
            return ""
        return str(block.get("raw_text") or block.get("text") or "").strip()

    candidate_ids: list[str] = []
    for raw in (
        seg.get("primary_block_id"),
        *(list(seg.get("evidence_block_ids") or [])),
        *(list(seg.get("related_block_ids") or [])),
    ):
        block_id = str(raw or "").strip()
        if block_id and block_id not in candidate_ids:
            candidate_ids.append(block_id)
    for block_id in candidate_ids:
        frag = _extract_caption_fragment_for_letters(_block_text_for(block_id), panel_letters)
        if not frag:
            frag = _extract_caption_fragment_for_letters_fallback(_block_text_for(block_id), panel_letters)
        if frag:
            return str(frag).strip()

    fig_no = 0
    try:
        fig_no = int(seg.get("support_slot_figure_number") or 0)
    except Exception:
        fig_no = 0
    md_path_str = str(md_path or "").strip()
    if fig_no <= 0 or not md_path_str:
        return ""
    try:
        rows = load_paper_guide_figure_index(md_path_str)
    except Exception:
        rows = []
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        try:
            row_no = int(row.get("paper_figure_number") or row.get("figure_number") or 0)
        except Exception:
            row_no = 0
        if row_no != fig_no:
            continue
        caption = str(row.get("caption") or "").strip()
        frag = _extract_caption_fragment_for_letters(caption, panel_letters)
        if not frag:
            frag = _extract_caption_fragment_for_letters_fallback(caption, panel_letters)
        if frag:
            return str(frag).strip()
    return ""


def _build_paper_guide_segment_locate_target(
    seg: dict | None,
    *,
    panel_clause_snippet: str = "",
) -> dict:
    if not isinstance(seg, dict):
        return {}
    snippet_aliases = [
        str(item or "").strip()
        for item in list(seg.get("snippet_aliases") or [])
        if str(item or "").strip()
    ]
    related_block_ids = [
        str(item or "").strip()
        for item in list(seg.get("related_block_ids") or [])
        if str(item or "").strip()
    ]
    anchor_number = _extract_paper_guide_segment_anchor_number(seg)
    primary_snippet = (
        str(panel_clause_snippet or "").strip()
        or str(seg.get("evidence_quote") or "").strip()
        or str(seg.get("anchor_text") or "").strip()
        or str(seg.get("text") or "").strip()
    )
    locate_target = {
        "segmentId": str(seg.get("segment_id") or "").strip() or None,
        "sourceSegmentId": str(seg.get("segment_id") or "").strip() or None,
        "headingPath": str(seg.get("primary_heading_path") or "").strip() or None,
        "snippet": primary_snippet or None,
        "highlightSnippet": primary_snippet or None,
        "evidenceQuote": str(seg.get("evidence_quote") or "").strip() or None,
        "anchorText": (
            str(panel_clause_snippet or "").strip() or str(seg.get("anchor_text") or "").strip() or None
        ),
        "hitLevel": str(seg.get("hit_level") or "").strip() or None,
        "blockId": str(seg.get("primary_block_id") or "").strip() or None,
        "anchorId": str(seg.get("primary_anchor_id") or "").strip() or None,
        "anchorKind": str(seg.get("anchor_kind") or "").strip() or None,
        "anchorNumber": int(anchor_number or 0) or None,
        "claimType": str(seg.get("claim_type") or "").strip() or None,
        "locatePolicy": str(seg.get("locate_policy") or "").strip() or None,
        "locateSurfacePolicy": str(seg.get("locate_surface_policy") or "").strip() or None,
        "snippetAliases": snippet_aliases or None,
        "relatedBlockIds": related_block_ids or None,
    }
    if any(value for value in locate_target.values()):
        return locate_target
    return {}


def _build_paper_guide_segment_reader_open(
    seg: dict | None,
    *,
    source_path: str,
    source_name: str,
    locate_target: dict | None,
    alternative_candidates: list[dict] | None = None,
    claim_group: dict | None = None,
) -> dict:
    if not isinstance(seg, dict):
        return {}
    locate = dict(locate_target or {}) if isinstance(locate_target, dict) else {}
    related_block_ids = [
        str(item or "").strip()
        for item in list(seg.get("related_block_ids") or [])
        if str(item or "").strip()
    ]
    reader_open = {
        "sourcePath": str(source_path or "").strip() or None,
        "sourceName": str(source_name or "").strip() or None,
        "headingPath": locate.get("headingPath"),
        "snippet": locate.get("snippet"),
        "highlightSnippet": locate.get("highlightSnippet"),
        "blockId": locate.get("blockId"),
        "anchorId": locate.get("anchorId"),
        "relatedBlockIds": related_block_ids or None,
        "anchorKind": locate.get("anchorKind"),
        "anchorNumber": locate.get("anchorNumber"),
        "strictLocate": True,
        "locateTarget": locate if any(value for value in locate.values()) else None,
        "claimGroup": dict(claim_group or {}) if isinstance(claim_group, dict) and any(claim_group.values()) else None,
        "alternatives": list(alternative_candidates or []) or None,
        "visibleAlternatives": list(alternative_candidates or []) or None,
        "evidenceAlternatives": list(alternative_candidates or []) or None,
        "initialAltIndex": 0,
    }
    if any(
        value
        for key, value in reader_open.items()
        if key not in {"sourcePath", "sourceName", "strictLocate", "initialAltIndex"}
    ):
        return reader_open
    return {}


def __getattr__(name: str):  # pragma: no cover
    return getattr(_legacy_runtime(), name)


def __dir__():  # pragma: no cover
    return sorted(set(globals().keys()) | set(dir(_legacy_runtime())))


__all__ = [
    "_build_paper_guide_support_slots",
    "_build_paper_guide_support_slots_block",
    "_build_paper_guide_segment_locate_target",
    "_build_paper_guide_segment_reader_open",
    "_extract_caption_fragment_for_letters",
    "_extract_inline_reference_specs",
    "_extract_inline_reference_numbers",
    "_extract_paper_guide_locate_anchor",
    "_extract_paper_guide_ref_spans",
    "_extract_paper_guide_segment_anchor_number",
    "_ground_paper_guide_answer_support",
    "_inject_paper_guide_support_markers",
    "_is_paper_guide_broad_summary_line",
    "_is_paper_guide_support_meta_line",
    "_normalize_paper_guide_support_surface",
    "_paper_guide_cue_tokens",
    "_paper_guide_support_cite_policy",
    "_paper_guide_support_claim_type",
    "_paper_guide_support_focus_tokens",
    "_paper_guide_support_rule_tokens",
    "_paper_guide_support_segment_spans",
    "_resolve_paper_guide_panel_clause_snippet",
    "_resolve_paper_guide_support_markers",
    "_resolve_paper_guide_support_ref_num",
    "_resolve_paper_guide_support_slot_block",
    "_resolve_reference_index_support_from_source",
    "_select_paper_guide_support_slot_for_context",
]
