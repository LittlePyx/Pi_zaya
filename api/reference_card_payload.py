from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from api.reference_card_quality import attach_ref_card_polish_contract


def _as_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_text(value: Any) -> str:
    return str(value or "")


def build_ref_card_ui_payload(
    *,
    display_name: str,
    heading_path: str,
    section_label: str,
    subsection_label: str,
    page_start: int,
    page_end: int,
    score: float | None,
    score_pending: bool,
    score_tier: str,
    summary_line: str,
    summary_kind: str,
    summary_surface: Mapping[str, Any] | None,
    summary_generation: str,
    summary_basis_meta: Mapping[str, Any] | None,
    summary_source: str,
    primary_evidence_heading_path: str,
    primary_evidence: Mapping[str, Any] | None,
    why_line: str,
    why_generation: str,
    why_basis_meta: Mapping[str, Any] | None,
    anchor_target_kind: str,
    anchor_target_number: int,
    anchor_match_score: float,
    explicit_doc_match_score: float,
    semantic_badges: Sequence[Any],
    can_open: bool,
    citation_meta: Mapping[str, Any] | None,
    source_path: str,
    reader_open: Mapping[str, Any] | None,
    render_locale: str = "",
) -> dict[str, Any]:
    """Assemble the stable UI contract for one reference card.

    Keep this module free of retrieval, scoring, and LLM behavior. It is the
    boundary between reference selection logic and frontend-facing card shape.
    """

    surface = _as_dict(summary_surface)
    summary_basis = _as_dict(summary_basis_meta)
    why_basis = _as_dict(why_basis_meta)

    payload = {
        "display_name": display_name,
        "heading_path": heading_path,
        "section_label": section_label,
        "subsection_label": subsection_label,
        "page_start": page_start,
        "page_end": page_end,
        "score": score,
        "score_pending": bool(score_pending),
        "score_tier": score_tier,
        "summary_line": summary_line,
        "summary_kind": _as_text(surface.get("summary_kind") or summary_kind),
        "summary_label": _as_text(surface.get("summary_label")),
        "summary_title": _as_text(surface.get("summary_title")),
        "summary_generation": _as_text(summary_basis.get("summary_generation") or summary_generation),
        "summary_basis": _as_text(summary_basis.get("summary_basis")),
        "primary_evidence_source": summary_source,
        "summary_source": summary_source,
        "primary_evidence_heading_path": primary_evidence_heading_path,
        "primary_evidence": _as_dict(primary_evidence),
        "why_line": why_line,
        "why_generation": _as_text(why_basis.get("why_generation") or why_generation),
        "why_basis": _as_text(why_basis.get("why_basis")),
        "anchor_target_kind": anchor_target_kind,
        "anchor_target_number": anchor_target_number,
        "anchor_match_score": anchor_match_score,
        "explicit_doc_match_score": explicit_doc_match_score,
        "semantic_badges": list(semantic_badges),
        "can_open": bool(can_open),
        "citation_meta": _as_dict(citation_meta),
        "source_path": source_path,
        "reader_open": _as_dict(reader_open),
        "render_locale": _as_text(render_locale),
    }
    return attach_ref_card_polish_contract(payload)
