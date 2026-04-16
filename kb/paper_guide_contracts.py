from __future__ import annotations

import re
from typing import Any

from typing_extensions import TypedDict

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except Exception:  # pragma: no cover - compatibility shim
    ConfigDict = None


class _PaperGuideBaseModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="allow")
    else:  # pragma: no cover - pydantic v1 fallback
        class Config:
            extra = "allow"


def _paper_guide_model_dump(model: BaseModel | None) -> dict[str, Any]:
    if not isinstance(model, BaseModel):
        return {}
    try:
        return dict(model.model_dump(mode="python"))
    except Exception:
        try:
            return dict(model.dict())
        except Exception:
            return {}


def _paper_guide_text_tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", str(value or "").lower())
        if token
    }


def _paper_guide_token_overlap(lhs: Any, rhs: Any) -> float:
    lhs_tokens = _paper_guide_text_tokens(lhs)
    rhs_tokens = _paper_guide_text_tokens(rhs)
    if not lhs_tokens or not rhs_tokens:
        return 0.0
    return float(len(lhs_tokens.intersection(rhs_tokens))) / float(min(len(lhs_tokens), len(rhs_tokens)))


def _paper_guide_is_negative_locate_surface_text(value: Any) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    return bool(
        re.search(
            r"(?i)\b(?:not stated|not mentioned|does not mention|doesn't mention|"
            r"does not specify|doesn't specify|cannot be determined|not found|"
            r"no external paper matched|no other papers matched|does not include)\b",
            raw,
        )
    )


def _paper_guide_should_suppress_render_locate(
    seg: dict[str, Any] | None,
    *,
    locate_target: dict[str, Any] | None = None,
    reader_open: dict[str, Any] | None = None,
) -> bool:
    segment = dict(seg or {}) if isinstance(seg, dict) else {}
    target = dict(locate_target or {}) if isinstance(locate_target, dict) else {}
    reader = dict(reader_open or {}) if isinstance(reader_open, dict) else {}
    anchor_kind = str(
        target.get("anchorKind")
        or segment.get("anchor_kind")
        or reader.get("anchorKind")
        or ""
    ).strip().lower()
    if anchor_kind in {"equation", "figure", "quote", "blockquote", "inline_formula"}:
        return False

    claim_type = str(segment.get("claim_type") or "").strip().lower()
    texts = [
        str(target.get("snippet") or "").strip(),
        str(target.get("highlightSnippet") or "").strip(),
        str(target.get("evidenceQuote") or "").strip(),
        str(target.get("anchorText") or "").strip(),
        str(segment.get("text") or "").strip(),
        str(segment.get("evidence_quote") or "").strip(),
        str(segment.get("anchor_text") or "").strip(),
    ]
    has_negative_surface = any(_paper_guide_is_negative_locate_surface_text(text) for text in texts if text)
    if not has_negative_surface:
        return False
    return claim_type in {"", "evidence_note_claim", "shell_sentence", "critical_fact_claim"}


def _paper_guide_segment_hit_level(seg: dict[str, Any]) -> str:
    raw = str(seg.get("hit_level") or "").strip().lower()
    if raw in {"exact", "block", "heading", "none"}:
        return raw
    if str(seg.get("primary_block_id") or "").strip():
        if str(seg.get("primary_anchor_id") or "").strip() or str(seg.get("anchor_kind") or "").strip():
            return "exact"
        return "block"
    if str(seg.get("primary_heading_path") or seg.get("heading_path") or "").strip():
        return "heading"
    return "none"


def _paper_guide_primary_render_segment_score(seg: dict[str, Any]) -> float:
    score = 0.0
    locate_target = seg.get("locate_target") if isinstance(seg.get("locate_target"), dict) else {}
    reader_open = seg.get("reader_open") if isinstance(seg.get("reader_open"), dict) else {}
    if locate_target:
        score += 2.0
    if reader_open:
        score += 2.0

    hit_level = _paper_guide_segment_hit_level(seg)
    if hit_level == "exact":
        score += 4.0
    elif hit_level == "block":
        score += 2.0
    elif hit_level == "heading":
        score -= 0.5
    else:
        score -= 1.0

    if str(seg.get("locate_policy") or "").strip().lower() == "required":
        score += 1.25
    if str(seg.get("locate_surface_policy") or locate_target.get("locateSurfacePolicy") or "").strip().lower() == "primary":
        score += 0.75
    if bool(seg.get("must_locate")) or bool(reader_open.get("strictLocate")):
        score += 0.75

    mapping_source = str(seg.get("mapping_source") or "").strip().lower()
    if mapping_source == "support_slot":
        score += 3.0
    elif mapping_source == "fast":
        score += 0.5

    evidence_confidence = float(seg.get("evidence_confidence") or 0.0)
    mapping_quality = float(seg.get("mapping_quality") or 0.0)
    score += min(2.0, max(0.0, evidence_confidence) * 0.8)
    score += min(1.5, max(0.0, mapping_quality) * 0.8)

    support_claim_type = str(seg.get("support_slot_claim_type") or "").strip().lower()
    claim_type = support_claim_type or str(seg.get("claim_type") or "").strip().lower()
    if support_claim_type:
        score += 2.5
    if str(seg.get("support_locate_anchor") or "").strip():
        score += 3.0
    if int(seg.get("resolved_ref_num") or 0) > 0:
        score += 1.5
    if int(seg.get("support_slot_figure_number") or 0) > 0:
        score += 1.25
    if list(seg.get("support_slot_panel_letters") or []):
        score += 1.25

    if claim_type in {
        "prior_work",
        "method_detail",
        "figure_panel",
        "figure_claim",
        "formula_claim",
        "equation_explanation_claim",
        "inline_formula_claim",
        "doc_map",
    }:
        score += 1.25

    anchor_text = (
        str(seg.get("support_locate_anchor") or "").strip()
        or str(locate_target.get("anchorText") or "").strip()
        or str(seg.get("anchor_text") or "").strip()
        or str(seg.get("evidence_quote") or "").strip()
        or str(locate_target.get("evidenceQuote") or "").strip()
        or str(locate_target.get("snippet") or "").strip()
    )
    segment_text = (
        str(seg.get("text") or "").strip()
        or str(seg.get("raw_markdown") or "").strip()
        or str(seg.get("display_markdown") or "").strip()
    )
    overlap = _paper_guide_token_overlap(segment_text, anchor_text)
    score += min(2.5, overlap * 5.0)

    if claim_type in {"critical_fact_claim", "shell_sentence"} and (not support_claim_type):
        if overlap < 0.18 and (not str(seg.get("support_locate_anchor") or "").strip()):
            score -= 4.5

    return score


def _select_paper_guide_primary_render_segment(
    visible_segments: list[dict[str, Any]],
    all_segments: list[dict[str, Any]],
    *,
    answer_context: str = "",
) -> dict[str, Any]:
    def _answer_alignment_adjustment(item: dict[str, Any]) -> float:
        anchor = (
            str(item.get("support_locate_anchor") or "").strip()
            or str(((item.get("locate_target") or {}).get("anchorText")) if isinstance(item.get("locate_target"), dict) else "").strip()
            or str(item.get("anchor_text") or "").strip()
            or str(item.get("evidence_quote") or "").strip()
            or str(((item.get("locate_target") or {}).get("snippet")) if isinstance(item.get("locate_target"), dict) else "").strip()
        )
        overlap = _paper_guide_token_overlap(answer_context, anchor)
        adjust = min(3.5, overlap * 7.0)
        support_claim = str(item.get("support_slot_claim_type") or "").strip().lower()
        claim_type = str(item.get("claim_type") or "").strip().lower()
        if support_claim == "own_result":
            if overlap < 0.20:
                adjust -= 6.0
            elif overlap < 0.35:
                adjust -= 2.0
        if claim_type == "blockquote_claim" and len(anchor) > 180:
            adjust -= 4.0
        return adjust

    scored_visible = [
        item
        for item in visible_segments
        if isinstance(item.get("locate_target"), dict) or isinstance(item.get("reader_open"), dict)
    ]
    preferred_visible = [
        item
        for item in scored_visible
        if not _paper_guide_should_suppress_render_locate(
            item,
            locate_target=item.get("locate_target") if isinstance(item.get("locate_target"), dict) else {},
            reader_open=item.get("reader_open") if isinstance(item.get("reader_open"), dict) else {},
        )
    ]
    candidate_visible = preferred_visible or scored_visible
    if candidate_visible:
        return max(
            candidate_visible,
            key=lambda item: (
                _paper_guide_primary_render_segment_score(item)
                + _answer_alignment_adjustment(item),
                float(item.get("mapping_quality") or 0.0),
                float(item.get("evidence_confidence") or 0.0),
            ),
        )
    if visible_segments:
        return visible_segments[0]
    if all_segments:
        return all_segments[0]
    return {}


class PaperGuideIntentModel(_PaperGuideBaseModel):
    prompt: str = ""
    family: str = ""
    exact_support: bool = False
    beginner_mode: bool = False
    target_figure: int = 0
    target_panels: list[str] = Field(default_factory=list)
    target_equation: int = 0
    target_scope: dict[str, Any] = Field(default_factory=dict)


class PaperGuideSupportRecordModel(_PaperGuideBaseModel):
    doc_idx: int = 0
    support_id: str = ""
    sid: str = ""
    source_path: str = ""
    block_id: str = ""
    anchor_id: str = ""
    heading_path: str = ""
    locate_anchor: str = ""
    claim_type: str = ""
    cite_policy: str = ""
    candidate_refs: list[int] = Field(default_factory=list)
    ref_spans: list[PaperGuideRefSpan] = Field(default_factory=list)
    evidence_atom_id: str = ""
    evidence_atom_kind: str = ""
    evidence_atom_text: str = ""
    figure_number: int = 0
    box_number: int = 0
    panel_letters: list[str] = Field(default_factory=list)
    target_scope: dict[str, Any] = Field(default_factory=dict)
    resolved_ref_num: int = 0
    citation_resolution_mode: str = ""
    segment_text: str = ""
    line_index: int = -1
    segment_index: int = 0
    segment_kind: str = ""
    segment_snippet_key: str = ""
    support_example: str = ""
    cite_example: str = ""
    heading: str = ""
    cue: str = ""
    snippet: str = ""
    deepread_texts: list[str] = Field(default_factory=list)


class PaperGuideSupportPackModel(_PaperGuideBaseModel):
    family: str = ""
    answer_markdown: str = ""
    support_records: list[PaperGuideSupportRecordModel] = Field(default_factory=list)
    needs_supplement: bool = False


class PaperGuideEvidenceCardModel(_PaperGuideBaseModel):
    doc_idx: int = 0
    sid: str = ""
    source_path: str = ""
    heading: str = ""
    cue: str = ""
    snippet: str = ""
    candidate_refs: list[int] = Field(default_factory=list)
    deepread_texts: list[str] = Field(default_factory=list)


class PaperGuideRetrievalBundleModel(_PaperGuideBaseModel):
    prompt_family: str = ""
    target_scope: dict[str, Any] = Field(default_factory=dict)
    evidence_cards: list[PaperGuideEvidenceCardModel] = Field(default_factory=list)
    candidate_refs_by_source: dict[str, list[int]] = Field(default_factory=dict)
    direct_source_path: str = ""
    focus_source_path: str = ""
    bound_source_path: str = ""


class PaperGuideGroundingTraceSegmentModel(_PaperGuideBaseModel):
    segment_id: str = ""
    text: str = ""
    source_path: str = ""
    primary_block_id: str = ""
    primary_anchor_id: str = ""
    heading_path: str = ""
    anchor_kind: str = ""
    anchor_number: int = 0
    claim_type: str = ""
    cite_policy: str = "locate_only"
    locate_policy: str = "optional"
    evidence_quote: str = ""
    support_block_ids: list[str] = Field(default_factory=list)
    evidence_block_ids: list[str] = Field(default_factory=list)


class PaperGuideCitationDetailModel(_PaperGuideBaseModel):
    num: int = 0
    anchor: str = ""
    source_name: str = ""
    source_path: str = ""
    raw: str = ""
    title: str = ""
    authors: str = ""
    venue: str = ""
    year: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    doi_url: str = ""
    cite_fmt: str = ""


class PaperGuideRenderPacketModel(_PaperGuideBaseModel):
    answer_markdown: str = ""
    notice: str = ""
    rendered_body: str = ""
    rendered_content: str = ""
    copy_markdown: str = ""
    copy_text: str = ""
    cite_details: list[PaperGuideCitationDetailModel] = Field(default_factory=list)
    citation_validation: dict[str, Any] = Field(default_factory=dict)
    locate_target: dict[str, Any] = Field(default_factory=dict)
    reader_open: dict[str, Any] = Field(default_factory=dict)
    segment_ids: list[str] = Field(default_factory=list)
    visible_segment_ids: list[str] = Field(default_factory=list)
    provenance_segment_count: int = 0
    visible_segment_count: int = 0


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


def _normalize_paper_guide_text_list(items: Any, *, limit: int = 4) -> list[str]:
    try:
        max_items = max(1, int(limit))
    except Exception:
        max_items = 4
    out: list[str] = []
    seen: set[str] = set()
    for item in list(items or []):
        value = str(item or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= max_items:
            break
    return out


def _normalize_paper_guide_id_list(items: Any, *, limit: int = 8) -> list[str]:
    try:
        max_items = max(1, int(limit))
    except Exception:
        max_items = 8
    out: list[str] = []
    seen: set[str] = set()
    for item in list(items or []):
        value = str(item or "").strip()
        if (not value) or (value in seen):
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= max_items:
            break
    return out


def _normalize_paper_guide_shallow_dict(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key or "").strip()
        if not name or value is None:
            continue
        if isinstance(value, str):
            val = value.strip()
            if not val:
                continue
            out[name] = val
            continue
        if isinstance(value, list):
            items: list[Any] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str):
                    item_text = item.strip()
                    if not item_text:
                        continue
                    items.append(item_text)
                    continue
                if isinstance(item, dict):
                    child = _normalize_paper_guide_shallow_dict(item)
                    if child:
                        items.append(child)
                    continue
                items.append(item)
            if items:
                out[name] = items
            continue
        if isinstance(value, dict):
            child = _normalize_paper_guide_shallow_dict(value)
            if child:
                out[name] = child
            continue
        out[name] = value
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
    out["deepread_texts"] = _normalize_paper_guide_text_list(src.get("deepread_texts"), limit=4)
    return out


def _build_paper_guide_intent_model(
    *,
    prompt: str = "",
    family: str = "",
    exact_support: bool = False,
    beginner_mode: bool = False,
    target_figure: Any = 0,
    target_panels: Any = None,
    target_equation: Any = 0,
    target_scope: dict | None = None,
    **extra: Any,
) -> PaperGuideIntentModel:
    return PaperGuideIntentModel(
        prompt=str(prompt or "").strip(),
        family=str(family or "").strip().lower(),
        exact_support=bool(exact_support),
        beginner_mode=bool(beginner_mode),
        target_figure=_paper_guide_contract_int(target_figure, default=0),
        target_panels=_normalize_paper_guide_panel_letters(target_panels, limit=8),
        target_equation=_paper_guide_contract_int(target_equation, default=0),
        target_scope=dict(target_scope or {}) if isinstance(target_scope, dict) else {},
        **dict(extra or {}),
    )


def _paper_guide_support_record_model_from_resolution(
    raw: dict | PaperGuideSupportResolution | None,
) -> PaperGuideSupportRecordModel:
    src = _normalize_paper_guide_support_resolution(raw if isinstance(raw, dict) else {})
    return PaperGuideSupportRecordModel(**src)


def _paper_guide_support_record_model_from_slot(
    raw: dict | PaperGuideSupportSlot | None,
) -> PaperGuideSupportRecordModel:
    src = _normalize_paper_guide_support_slot(raw if isinstance(raw, dict) else {})
    return PaperGuideSupportRecordModel(**src)


def _build_paper_guide_support_pack_model(
    *,
    family: str = "",
    answer_markdown: str = "",
    support_records: list[dict] | list[PaperGuideSupportRecordModel] | None = None,
    needs_supplement: bool = False,
    **extra: Any,
) -> PaperGuideSupportPackModel:
    normalized_records: list[PaperGuideSupportRecordModel] = []
    for raw in list(support_records or []):
        if isinstance(raw, PaperGuideSupportRecordModel):
            normalized_records.append(raw)
            continue
        if not isinstance(raw, dict):
            continue
        if any(
            key in raw
            for key in (
                "resolved_ref_num",
                "citation_resolution_mode",
                "segment_text",
                "segment_index",
            )
        ):
            normalized_records.append(_paper_guide_support_record_model_from_resolution(raw))
        else:
            normalized_records.append(_paper_guide_support_record_model_from_slot(raw))
    return PaperGuideSupportPackModel(
        family=str(family or "").strip().lower(),
        answer_markdown=str(answer_markdown or "").strip(),
        support_records=normalized_records,
        needs_supplement=bool(needs_supplement),
        **dict(extra or {}),
    )


def _normalize_paper_guide_candidate_refs_by_source(
    raw: Any,
    *,
    source_limit: int = 12,
    per_source_limit: int = 8,
) -> dict[str, list[int]]:
    try:
        max_sources = max(1, int(source_limit))
    except Exception:
        max_sources = 12
    out: dict[str, list[int]] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        source_path = str(key or "").strip()
        if (not source_path) or (source_path in out):
            continue
        out[source_path] = _normalize_paper_guide_candidate_refs(value, limit=per_source_limit)
        if len(out) >= max_sources:
            break
    return out


def _paper_guide_evidence_card_model_from_raw(
    raw: dict | None,
) -> PaperGuideEvidenceCardModel:
    src = dict(raw or {}) if isinstance(raw, dict) else {}
    return PaperGuideEvidenceCardModel(
        doc_idx=_paper_guide_contract_int(src.get("doc_idx"), default=0),
        sid=str(src.get("sid") or "").strip(),
        source_path=str(src.get("source_path") or "").strip(),
        heading=str(src.get("heading") or "").strip(),
        cue=str(src.get("cue") or "").strip(),
        snippet=str(src.get("snippet") or "").strip(),
        candidate_refs=_normalize_paper_guide_candidate_refs(src.get("candidate_refs"), limit=6),
        deepread_texts=_normalize_paper_guide_text_list(src.get("deepread_texts"), limit=3),
        **{
            key: value
            for key, value in src.items()
            if key
            not in {
                "doc_idx",
                "sid",
                "source_path",
                "heading",
                "cue",
                "snippet",
                "candidate_refs",
                "deepread_texts",
            }
        },
    )


def _build_paper_guide_retrieval_bundle_model(
    *,
    prompt_family: str = "",
    target_scope: dict | None = None,
    evidence_cards: list[dict] | list[PaperGuideEvidenceCardModel] | None = None,
    candidate_refs_by_source: dict[str, list[int]] | None = None,
    direct_source_path: str = "",
    focus_source_path: str = "",
    bound_source_path: str = "",
    **extra: Any,
) -> PaperGuideRetrievalBundleModel:
    normalized_cards: list[PaperGuideEvidenceCardModel] = []
    for raw in list(evidence_cards or []):
        if isinstance(raw, PaperGuideEvidenceCardModel):
            normalized_cards.append(raw)
            continue
        if not isinstance(raw, dict):
            continue
        normalized_cards.append(_paper_guide_evidence_card_model_from_raw(raw))
    return PaperGuideRetrievalBundleModel(
        prompt_family=str(prompt_family or "").strip().lower(),
        target_scope=dict(target_scope or {}) if isinstance(target_scope, dict) else {},
        evidence_cards=normalized_cards,
        candidate_refs_by_source=_normalize_paper_guide_candidate_refs_by_source(candidate_refs_by_source or {}),
        direct_source_path=str(direct_source_path or "").strip(),
        focus_source_path=str(focus_source_path or "").strip(),
        bound_source_path=str(bound_source_path or "").strip(),
        **dict(extra or {}),
    )


def _paper_guide_grounding_trace_segment_model_from_raw(
    raw: dict | None,
) -> PaperGuideGroundingTraceSegmentModel:
    src = dict(raw or {}) if isinstance(raw, dict) else {}
    support_block_ids = [
        str(item or "").strip()
        for item in list(src.get("support_block_ids") or [])
        if str(item or "").strip()
    ]
    evidence_block_ids = [
        str(item or "").strip()
        for item in list(src.get("evidence_block_ids") or [])
        if str(item or "").strip()
    ]
    return PaperGuideGroundingTraceSegmentModel(
        segment_id=str(src.get("segment_id") or src.get("support_id") or src.get("sid") or "").strip(),
        text=str(src.get("text") or src.get("segment_text") or src.get("raw_markdown") or "").strip(),
        source_path=str(src.get("source_path") or "").strip(),
        primary_block_id=str(src.get("primary_block_id") or src.get("block_id") or "").strip(),
        primary_anchor_id=str(src.get("primary_anchor_id") or src.get("anchor_id") or "").strip(),
        heading_path=str(src.get("primary_heading_path") or src.get("heading_path") or "").strip(),
        anchor_kind=str(src.get("anchor_kind") or "").strip(),
        anchor_number=_paper_guide_contract_int(
            src.get("anchor_number")
            or src.get("equation_number")
            or src.get("support_slot_figure_number"),
            default=0,
        ),
        claim_type=str(src.get("claim_type") or "").strip(),
        cite_policy=str(src.get("cite_policy") or "").strip() or "locate_only",
        locate_policy=str(src.get("locate_policy") or "").strip() or "optional",
        evidence_quote=str(src.get("evidence_quote") or "").strip(),
        support_block_ids=support_block_ids,
        evidence_block_ids=evidence_block_ids,
        **{
            key: value
            for key, value in src.items()
            if key
            not in {
                "segment_id",
                "text",
                "raw_markdown",
                "source_path",
                "primary_block_id",
                "block_id",
                "primary_anchor_id",
                "anchor_id",
                "primary_heading_path",
                "heading_path",
                "anchor_kind",
                "anchor_number",
                "equation_number",
                "support_slot_figure_number",
                "claim_type",
                "cite_policy",
                "locate_policy",
                "evidence_quote",
                "support_block_ids",
                "evidence_block_ids",
            }
        },
    )


def _paper_guide_citation_detail_model_from_raw(
    raw: dict | None,
) -> PaperGuideCitationDetailModel:
    src = dict(raw or {}) if isinstance(raw, dict) else {}
    return PaperGuideCitationDetailModel(
        num=_paper_guide_contract_int(src.get("num"), default=0),
        anchor=str(src.get("anchor") or "").strip(),
        source_name=str(src.get("source_name") or "").strip(),
        source_path=str(src.get("source_path") or "").strip(),
        raw=str(src.get("raw") or "").strip(),
        title=str(src.get("title") or "").strip(),
        authors=str(src.get("authors") or "").strip(),
        venue=str(src.get("venue") or "").strip(),
        year=str(src.get("year") or "").strip(),
        volume=str(src.get("volume") or "").strip(),
        issue=str(src.get("issue") or "").strip(),
        pages=str(src.get("pages") or "").strip(),
        doi=str(src.get("doi") or "").strip(),
        doi_url=str(src.get("doi_url") or "").strip(),
        cite_fmt=str(src.get("cite_fmt") or "").strip(),
        **{
            key: value
            for key, value in src.items()
            if key
            not in {
                "num",
                "anchor",
                "source_name",
                "source_path",
                "raw",
                "title",
                "authors",
                "venue",
                "year",
                "volume",
                "issue",
                "pages",
                "doi",
                "doi_url",
                "cite_fmt",
            }
        },
    )


def _build_paper_guide_render_packet_model(
    *,
    answer_markdown: str = "",
    notice: str = "",
    rendered_body: str = "",
    rendered_content: str = "",
    copy_markdown: str = "",
    copy_text: str = "",
    cite_details: list[dict] | list[PaperGuideCitationDetailModel] | None = None,
    citation_validation: dict[str, Any] | None = None,
    locate_target: dict[str, Any] | None = None,
    reader_open: dict[str, Any] | None = None,
    provenance_segments: list[dict] | None = None,
    **extra: Any,
) -> PaperGuideRenderPacketModel:
    normalized_cites: list[PaperGuideCitationDetailModel] = []
    for raw in list(cite_details or []):
        if isinstance(raw, PaperGuideCitationDetailModel):
            normalized_cites.append(raw)
            continue
        if not isinstance(raw, dict):
            continue
        normalized_cites.append(_paper_guide_citation_detail_model_from_raw(raw))

    segments = [dict(item) for item in list(provenance_segments or []) if isinstance(item, dict)]
    segment_ids = _normalize_paper_guide_id_list(
        [item.get("segment_id") for item in segments],
        limit=12,
    )
    visible_segments = [
        item
        for item in segments
        if str(item.get("locate_policy") or item.get("locatePolicy") or "").strip().lower() != "hidden"
    ]
    visible_segment_ids = _normalize_paper_guide_id_list(
        [item.get("segment_id") for item in visible_segments],
        limit=12,
    )

    answer_context = (
        str(rendered_body or "").strip()
        or str(answer_markdown or "").strip()
        or str(copy_text or "").strip()
        or str(rendered_content or "").strip()
    )
    primary_segment = _select_paper_guide_primary_render_segment(
        visible_segments,
        segments,
        answer_context=answer_context,
    )
    primary_locate_target = (
        locate_target
        if isinstance(locate_target, dict) and locate_target
        else (
            primary_segment.get("locate_target")
            if isinstance(primary_segment, dict) and isinstance(primary_segment.get("locate_target"), dict)
            else {}
        )
    )
    primary_reader_open = (
        reader_open
        if isinstance(reader_open, dict) and reader_open
        else (
            primary_segment.get("reader_open")
            if isinstance(primary_segment, dict) and isinstance(primary_segment.get("reader_open"), dict)
            else {}
        )
    )
    if _paper_guide_should_suppress_render_locate(
        primary_segment if isinstance(primary_segment, dict) else {},
        locate_target=primary_locate_target if isinstance(primary_locate_target, dict) else {},
        reader_open=primary_reader_open if isinstance(primary_reader_open, dict) else {},
    ):
        primary_locate_target = {}
        primary_reader_open = {}

    return PaperGuideRenderPacketModel(
        answer_markdown=str(answer_markdown or "").strip(),
        notice=str(notice or "").strip(),
        rendered_body=str(rendered_body or "").strip(),
        rendered_content=str(rendered_content or "").strip(),
        copy_markdown=str(copy_markdown or "").strip(),
        copy_text=str(copy_text or "").strip(),
        cite_details=normalized_cites,
        citation_validation=_normalize_paper_guide_shallow_dict(citation_validation),
        locate_target=_normalize_paper_guide_shallow_dict(primary_locate_target),
        reader_open=_normalize_paper_guide_shallow_dict(primary_reader_open),
        segment_ids=segment_ids,
        visible_segment_ids=visible_segment_ids,
        provenance_segment_count=len(segments),
        visible_segment_count=len(visible_segments),
        **dict(extra or {}),
    )
