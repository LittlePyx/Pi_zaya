from __future__ import annotations

from typing import Any, TypedDict


class PaperGuideRefSpan(TypedDict, total=False):
    text: str
    nums: list[int]
    scope: str


class PaperGuideTargetScope(TypedDict, total=False):
    prompt_family: str
    requested_sections: list[str]
    requested_boxes: list[int]
    target_figure_num: int
    target_panel_letters: list[str]
    heading_hints: list[str]
    require_scope_match: bool
    prefer_caption_atoms: bool
    prefer_ref_atoms: bool
    prefer_sentence_atoms: bool
    prefer_exact_anchor: bool
    allow_non_target_fallback: bool


class PaperGuideEvidenceAtom(TypedDict, total=False):
    atom_id: str
    atom_kind: str
    block_id: str
    anchor_id: str
    heading_path: str
    text: str
    locate_anchor: str
    figure_number: int
    panel_letters: list[str]
    box_number: int
    inline_refs: list[int]
    block_kind: str
    order_index: int
    atom_score: float
    scope_match: bool


class PaperGuideSupportSlot(TypedDict, total=False):
    doc_idx: int
    support_id: str
    support_example: str
    cite_example: str
    sid: str
    source_path: str
    heading: str
    heading_path: str
    cue: str
    snippet: str
    locate_anchor: str
    claim_type: str
    cite_policy: str
    candidate_refs: list[int]
    ref_spans: list[PaperGuideRefSpan]
    evidence_atom_id: str
    evidence_atom_kind: str
    evidence_atom_text: str
    figure_number: int
    box_number: int
    panel_letters: list[str]
    target_scope: PaperGuideTargetScope
    block_id: str
    anchor_id: str
    deepread_texts: list[str]


class PaperGuideSupportResolution(TypedDict, total=False):
    doc_idx: int
    support_id: str
    sid: str
    source_path: str
    block_id: str
    anchor_id: str
    heading_path: str
    locate_anchor: str
    claim_type: str
    cite_policy: str
    candidate_refs: list[int]
    ref_spans: list[PaperGuideRefSpan]
    evidence_atom_id: str
    evidence_atom_kind: str
    evidence_atom_text: str
    figure_number: int
    box_number: int
    panel_letters: list[str]
    target_scope: PaperGuideTargetScope
    resolved_ref_num: int
    citation_resolution_mode: str
    segment_text: str
    line_index: int
    segment_index: int
    segment_kind: str
    segment_snippet_key: str


def _paper_guide_contract_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_paper_guide_candidate_refs(items: Any, *, limit: int = 8) -> list[int]:
    try:
        max_items = max(1, int(limit))
    except Exception:
        max_items = 8
    out: list[int] = []
    seen: set[int] = set()
    for item in list(items or []):
        try:
            value = int(item)
        except Exception:
            continue
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= max_items:
            break
    return out


def _normalize_paper_guide_panel_letters(items: Any, *, limit: int = 8) -> list[str]:
    try:
        max_items = max(1, int(limit))
    except Exception:
        max_items = 8
    out: list[str] = []
    seen: set[str] = set()
    for item in list(items or []):
        value = str(item or "").strip().lower()
        if (not value) or (value in seen):
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= max_items:
            break
    return out


def _normalize_paper_guide_ref_spans(items: Any, *, limit: int = 4) -> list[PaperGuideRefSpan]:
    try:
        max_items = max(1, int(limit))
    except Exception:
        max_items = 4
    out: list[PaperGuideRefSpan] = []
    seen: set[tuple[str, tuple[int, ...], str]] = set()
    for raw in list(items or []):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        nums = _normalize_paper_guide_candidate_refs(raw.get("nums"), limit=8)
        scope = str(raw.get("scope") or "").strip()
        key = (text.lower(), tuple(nums), scope.lower())
        if (not text and not nums) or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "text": text,
                "nums": nums,
                "scope": scope,
            }
        )
        if len(out) >= max_items:
            break
    return out


def _build_paper_guide_support_resolution(
    *,
    doc_idx: Any = 0,
    support_id: str = "",
    sid: str = "",
    source_path: str = "",
    block_id: str = "",
    anchor_id: str = "",
    heading_path: str = "",
    locate_anchor: str = "",
    claim_type: str = "",
    cite_policy: str = "",
    candidate_refs: Any = None,
    ref_spans: Any = None,
    evidence_atom_id: str = "",
    evidence_atom_kind: str = "",
    evidence_atom_text: str = "",
    figure_number: Any = 0,
    box_number: Any = 0,
    panel_letters: Any = None,
    target_scope: dict | None = None,
    resolved_ref_num: Any = 0,
    citation_resolution_mode: str = "",
    segment_text: str = "",
    line_index: Any = -1,
    segment_index: Any = 0,
    segment_kind: str = "",
    segment_snippet_key: str = "",
) -> PaperGuideSupportResolution:
    return {
        "doc_idx": _paper_guide_contract_int(doc_idx, default=0),
        "support_id": str(support_id or "").strip(),
        "sid": str(sid or "").strip(),
        "source_path": str(source_path or "").strip(),
        "block_id": str(block_id or "").strip(),
        "anchor_id": str(anchor_id or "").strip(),
        "heading_path": str(heading_path or "").strip(),
        "locate_anchor": str(locate_anchor or "").strip(),
        "claim_type": str(claim_type or "").strip(),
        "cite_policy": str(cite_policy or "").strip(),
        "candidate_refs": _normalize_paper_guide_candidate_refs(candidate_refs, limit=6),
        "ref_spans": _normalize_paper_guide_ref_spans(ref_spans, limit=4),
        "evidence_atom_id": str(evidence_atom_id or "").strip(),
        "evidence_atom_kind": str(evidence_atom_kind or "").strip(),
        "evidence_atom_text": str(evidence_atom_text or "").strip(),
        "figure_number": _paper_guide_contract_int(figure_number, default=0),
        "box_number": _paper_guide_contract_int(box_number, default=0),
        "panel_letters": _normalize_paper_guide_panel_letters(panel_letters, limit=6),
        "target_scope": dict(target_scope or {}) if isinstance(target_scope, dict) else {},
        "resolved_ref_num": _paper_guide_contract_int(resolved_ref_num, default=0),
        "citation_resolution_mode": str(citation_resolution_mode or "").strip(),
        "segment_text": str(segment_text or "").strip(),
        "line_index": _paper_guide_contract_int(line_index, default=-1),
        "segment_index": _paper_guide_contract_int(segment_index, default=0),
        "segment_kind": str(segment_kind or "").strip(),
        "segment_snippet_key": str(segment_snippet_key or "").strip(),
    }


def _normalize_paper_guide_support_resolution(raw: dict | None) -> PaperGuideSupportResolution:
    src = dict(raw or {}) if isinstance(raw, dict) else {}
    out = dict(src)
    out.update(
        _build_paper_guide_support_resolution(
            doc_idx=src.get("doc_idx"),
            support_id=src.get("support_id"),
            sid=src.get("sid"),
            source_path=src.get("source_path"),
            block_id=src.get("block_id"),
            anchor_id=src.get("anchor_id"),
            heading_path=src.get("heading_path"),
            locate_anchor=src.get("locate_anchor"),
            claim_type=src.get("claim_type"),
            cite_policy=src.get("cite_policy"),
            candidate_refs=src.get("candidate_refs"),
            ref_spans=src.get("ref_spans"),
            evidence_atom_id=src.get("evidence_atom_id"),
            evidence_atom_kind=src.get("evidence_atom_kind"),
            evidence_atom_text=src.get("evidence_atom_text"),
            figure_number=src.get("figure_number"),
            box_number=src.get("box_number"),
            panel_letters=src.get("panel_letters"),
            target_scope=src.get("target_scope"),
            resolved_ref_num=src.get("resolved_ref_num"),
            citation_resolution_mode=src.get("citation_resolution_mode"),
            segment_text=src.get("segment_text"),
            line_index=src.get("line_index"),
            segment_index=src.get("segment_index"),
            segment_kind=src.get("segment_kind"),
            segment_snippet_key=src.get("segment_snippet_key"),
        )
    )
    return out


def _normalize_paper_guide_support_slot(raw: dict | None) -> PaperGuideSupportSlot:
    src = dict(raw or {}) if isinstance(raw, dict) else {}
    out = dict(src)
    out["doc_idx"] = _paper_guide_contract_int(src.get("doc_idx"), default=0)
    out["support_id"] = str(src.get("support_id") or "").strip()
    out["support_example"] = str(src.get("support_example") or "").strip()
    out["cite_example"] = str(src.get("cite_example") or "").strip()
    out["sid"] = str(src.get("sid") or "").strip()
    out["source_path"] = str(src.get("source_path") or "").strip()
    out["heading"] = str(src.get("heading") or "").strip()
    out["heading_path"] = str(src.get("heading_path") or "").strip()
    out["cue"] = str(src.get("cue") or "").strip()
    out["snippet"] = str(src.get("snippet") or "").strip()
    out["locate_anchor"] = str(src.get("locate_anchor") or "").strip()
    out["claim_type"] = str(src.get("claim_type") or "").strip()
    out["cite_policy"] = str(src.get("cite_policy") or "").strip()
    out["candidate_refs"] = _normalize_paper_guide_candidate_refs(src.get("candidate_refs"), limit=6)
    out["ref_spans"] = _normalize_paper_guide_ref_spans(src.get("ref_spans"), limit=4)
    out["evidence_atom_id"] = str(src.get("evidence_atom_id") or "").strip()
    out["evidence_atom_kind"] = str(src.get("evidence_atom_kind") or "").strip()
    out["evidence_atom_text"] = str(src.get("evidence_atom_text") or "").strip()
    out["figure_number"] = _paper_guide_contract_int(src.get("figure_number"), default=0)
    out["box_number"] = _paper_guide_contract_int(src.get("box_number"), default=0)
    out["panel_letters"] = _normalize_paper_guide_panel_letters(src.get("panel_letters"), limit=6)
    out["target_scope"] = dict(src.get("target_scope") or {}) if isinstance(src.get("target_scope"), dict) else {}
    out["block_id"] = str(src.get("block_id") or "").strip()
    out["anchor_id"] = str(src.get("anchor_id") or "").strip()
    out["deepread_texts"] = [
        str(item or "").strip()
        for item in list(src.get("deepread_texts") or [])
        if str(item or "").strip()
    ]
    return out
