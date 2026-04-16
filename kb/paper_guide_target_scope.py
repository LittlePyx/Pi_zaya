from __future__ import annotations

import re

from kb.paper_guide_focus import _extract_caption_panel_letters
from kb.paper_guide_prompting import (
    _paper_guide_prompt_family,
    _paper_guide_requested_box_numbers,
    _paper_guide_requested_heading_hints,
    _paper_guide_requested_section_targets,
    _paper_guide_text_matches_requested_box,
    _paper_guide_text_matches_requested_section,
)
from kb.paper_guide_provenance import _extract_figure_number


def _extract_prompt_panel_letters(prompt: str) -> list[str]:
    q = str(prompt or "").strip()
    if not q:
        return []
    letters: set[str] = set()
    windows = [
        str(match.group(1) or "")
        for match in re.finditer(
            r"\bpanel(?:s)?\b([^\n.!?]{0,96})",
            q,
            flags=re.IGNORECASE,
        )
    ]
    windows.extend(
        str(match.group(1) or "")
        for match in re.finditer(
            r"\bfig(?:ure)?\.?\s*\d+\b([^\n.!?]{0,96})",
            q,
            flags=re.IGNORECASE,
        )
        if re.search(r"\bpanel(?:s)?\b|\([a-g]\)", str(match.group(1) or ""), flags=re.IGNORECASE)
    )
    for window in windows:
        for paren_match in re.finditer(r"\(([a-g])\)", window, flags=re.IGNORECASE):
            letters.add(str(paren_match.group(1) or "").strip().lower())
        letters.update(
            str(ch or "").strip().lower()
            for ch in re.findall(r"\b([a-g])\b", window, flags=re.IGNORECASE)
            if str(ch or "").strip()
        )
    return sorted(letters)


def _normalize_paper_guide_target_scope(scope: dict | None) -> dict:
    raw = dict(scope or {}) if isinstance(scope, dict) else {}
    family = str(raw.get("prompt_family") or "").strip().lower()

    sections: list[str] = []
    seen_sections: set[str] = set()
    for item in list(raw.get("requested_sections") or []):
        key = str(item or "").strip().lower()
        if (not key) or (key in seen_sections):
            continue
        seen_sections.add(key)
        sections.append(key)

    boxes: list[int] = []
    seen_boxes: set[int] = set()
    for item in list(raw.get("requested_boxes") or []):
        try:
            value = int(item)
        except Exception:
            continue
        if value <= 0 or value in seen_boxes:
            continue
        seen_boxes.add(value)
        boxes.append(value)

    try:
        target_figure_num = int(raw.get("target_figure_num") or raw.get("target_figure_number") or 0)
    except Exception:
        target_figure_num = 0

    panels: list[str] = []
    seen_panels: set[str] = set()
    for item in list(raw.get("target_panel_letters") or []):
        key = str(item or "").strip().lower()
        if (not key) or (key in seen_panels):
            continue
        seen_panels.add(key)
        panels.append(key)

    heading_hints: list[str] = []
    seen_hints: set[str] = set()
    for item in list(raw.get("heading_hints") or []):
        key = str(item or "").strip()
        low = key.lower()
        if (not key) or (low in seen_hints):
            continue
        seen_hints.add(low)
        heading_hints.append(key)

    require_scope_match = bool(
        sections
        or boxes
        or target_figure_num > 0
        or panels
        or raw.get("require_scope_match")
    )

    prefer_caption_atoms = bool(
        raw.get("prefer_caption_atoms")
        or family == "figure_walkthrough"
        or panels
    )
    prefer_ref_atoms = bool(
        raw.get("prefer_ref_atoms")
        or family == "citation_lookup"
    )
    prefer_sentence_atoms = bool(
        raw.get("prefer_sentence_atoms")
        or family in {"method", "citation_lookup", "discussion_only", "box_only"}
        or sections
        or boxes
    )
    prefer_exact_anchor = bool(
        raw.get("prefer_exact_anchor")
        or family in {"method", "citation_lookup", "figure_walkthrough", "discussion_only", "box_only"}
        or require_scope_match
    )
    allow_non_target_fallback = bool(
        raw.get("allow_non_target_fallback")
        if "allow_non_target_fallback" in raw
        else (not require_scope_match)
    )

    return {
        "prompt_family": family,
        "requested_sections": sections,
        "requested_boxes": boxes,
        "target_figure_num": target_figure_num if target_figure_num > 0 else 0,
        "target_panel_letters": panels,
        "heading_hints": heading_hints,
        "require_scope_match": require_scope_match,
        "prefer_caption_atoms": prefer_caption_atoms,
        "prefer_ref_atoms": prefer_ref_atoms,
        "prefer_sentence_atoms": prefer_sentence_atoms,
        "prefer_exact_anchor": prefer_exact_anchor,
        "allow_non_target_fallback": allow_non_target_fallback,
    }


def _build_paper_guide_target_scope(
    prompt: str,
    *,
    prompt_family: str = "",
) -> dict:
    q = str(prompt or "").strip()
    family = str(prompt_family or "").strip().lower() or _paper_guide_prompt_family(q)
    requested_sections = _paper_guide_requested_section_targets(q)
    requested_boxes = _paper_guide_requested_box_numbers(q)
    target_figure_num = _extract_figure_number(q)
    target_panel_letters = _extract_prompt_panel_letters(q)
    heading_hints = _paper_guide_requested_heading_hints(q)

    scope = {
        "prompt_family": family,
        "requested_sections": requested_sections,
        "requested_boxes": requested_boxes,
        "target_figure_num": int(target_figure_num or 0),
        "target_panel_letters": target_panel_letters,
        "heading_hints": heading_hints,
        "require_scope_match": bool(
            requested_sections
            or requested_boxes
            or int(target_figure_num or 0) > 0
            or target_panel_letters
        ),
        "prefer_caption_atoms": bool(family == "figure_walkthrough" or target_panel_letters),
        "prefer_ref_atoms": bool(family == "citation_lookup"),
        "prefer_sentence_atoms": bool(
            family in {"method", "citation_lookup", "discussion_only", "box_only"}
            or requested_sections
            or requested_boxes
        ),
        "prefer_exact_anchor": bool(
            family in {"method", "citation_lookup", "figure_walkthrough", "discussion_only", "box_only"}
            or requested_sections
            or requested_boxes
            or int(target_figure_num or 0) > 0
            or target_panel_letters
        ),
        "allow_non_target_fallback": not bool(
            requested_sections
            or requested_boxes
            or int(target_figure_num or 0) > 0
            or target_panel_letters
        ),
    }
    return _normalize_paper_guide_target_scope(scope)


def _paper_guide_target_scope_has_targets(scope: dict | None) -> bool:
    scope_norm = _normalize_paper_guide_target_scope(scope)
    return bool(
        list(scope_norm.get("requested_sections") or [])
        or list(scope_norm.get("requested_boxes") or [])
        or int(scope_norm.get("target_figure_num") or 0) > 0
        or list(scope_norm.get("target_panel_letters") or [])
    )


def _paper_guide_target_scope_matches_text(
    scope: dict | None,
    *,
    text: str = "",
    heading: str = "",
    box_number: int = 0,
    figure_number: int = 0,
    panel_letters: list[str] | None = None,
) -> bool:
    scope_norm = _normalize_paper_guide_target_scope(scope)
    if not _paper_guide_target_scope_has_targets(scope_norm):
        return True

    src_text = str(text or "").strip()
    src_heading = str(heading or "").strip()
    joined = "\n".join(part for part in [src_heading, src_text] if part)

    requested_boxes = [int(n) for n in list(scope_norm.get("requested_boxes") or []) if int(n) > 0]
    if requested_boxes:
        try:
            explicit_box_number = int(box_number or 0)
        except Exception:
            explicit_box_number = 0
        if explicit_box_number > 0:
            if explicit_box_number not in requested_boxes:
                return False
        elif not any(
            _paper_guide_text_matches_requested_box(src_heading, n)
            or _paper_guide_text_matches_requested_box(src_text[:640], n)
            or _paper_guide_text_matches_requested_box(joined[:1200], n)
            for n in requested_boxes
        ):
            return False

    requested_sections = [str(sec or "").strip().lower() for sec in list(scope_norm.get("requested_sections") or []) if str(sec or "").strip()]
    if requested_sections:
        if not any(
            _paper_guide_text_matches_requested_section(src_heading, sec)
            or _paper_guide_text_matches_requested_section(src_text[:640], sec)
            or _paper_guide_text_matches_requested_section(joined[:1200], sec)
            for sec in requested_sections
        ):
            return False

    try:
        target_figure_num = int(scope_norm.get("target_figure_num") or 0)
    except Exception:
        target_figure_num = 0
    if target_figure_num > 0:
        try:
            explicit_figure_number = int(figure_number or 0)
        except Exception:
            explicit_figure_number = 0
        figure_match = (
            explicit_figure_number == target_figure_num
            if explicit_figure_number > 0
            else _extract_figure_number(joined[:1400]) == target_figure_num
        )
        if not figure_match:
            return False

    target_panels = {
        str(ch or "").strip().lower()
        for ch in list(scope_norm.get("target_panel_letters") or [])
        if str(ch or "").strip()
    }
    if target_panels:
        explicit_panels = {
            str(ch or "").strip().lower()
            for ch in list(panel_letters or [])
            if str(ch or "").strip()
        }
        found_panels = explicit_panels or {
            str(ch or "").strip().lower()
            for ch in _extract_caption_panel_letters(joined[:1400])
            if str(ch or "").strip()
        }
        if not found_panels.intersection(target_panels):
            return False

    return True
