from __future__ import annotations

import re

from kb.inpaper_citation_grounding import parse_ref_num_set
from kb.paper_guide_focus import (
    _extract_caption_fragment_for_letters,
    _extract_caption_panel_letters,
)
from kb.paper_guide_prompting import _paper_guide_box_header_number
from kb.paper_guide_provenance import _extract_figure_number
from kb.source_blocks import normalize_inline_markdown, normalize_match_text

_INLINE_REF_PATTERNS = (
    r"\[(\d{1,4}(?:\s*(?:[\-鈥撯€斺垝,])\s*\d{1,4})*)\]",
    r"\$\^\{(\d{1,4}(?:\s*(?:[\-鈥撯€斺垝,])\s*\d{1,4})*)\}\$",
    r"\^\{(\d{1,4}(?:\s*(?:[\-鈥撯€斺垝,])\s*\d{1,4})*)\}",
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z0-9\[]|Fig(?:ure)?\.?\s*\d+|[a-z]\)))")
_CLAUSE_PUNCT_RE = re.compile(r"[.;!?]")
_CLAUSE_CONJ_RE = re.compile(r"(?i)\b(?:and|but|however|whereas|while|then)\b")


def _extract_inline_reference_specs_local(text: str) -> list[str]:
    src = str(text or "").strip()
    if not src:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for pattern in _INLINE_REF_PATTERNS:
        for spec in re.findall(pattern, src):
            key = str(spec or "").strip()
            if (not key) or (key in seen):
                continue
            seen.add(key)
            out.append(key)
    return out


def _extract_inline_reference_numbers_local(text: str, *, max_candidates: int = 8) -> list[int]:
    src = str(text or "").strip()
    if not src:
        return []
    try:
        limit = max(1, int(max_candidates))
    except Exception:
        limit = 8
    out: list[int] = []
    seen: set[int] = set()
    for spec in _extract_inline_reference_specs_local(src):
        for item in parse_ref_num_set(str(spec or "").strip(), max_items=max(8, limit * 2)):
            try:
                value = int(item)
            except Exception:
                continue
            if value <= 0 or value in seen:
                continue
            seen.add(value)
            out.append(value)
            if len(out) >= limit:
                return out
    return out


def _normalize_atom_text(text: str) -> str:
    src = normalize_inline_markdown(str(text or "").replace("\r\n", "\n").replace("\r", "\n"))
    src = re.sub(r"\s+", " ", src).strip()
    return src


def _split_evidence_sentences(text: str) -> list[str]:
    src = _normalize_atom_text(text)
    if not src:
        return []
    protected = re.sub(r"(?i)\bet al\.", "et al<prd>", src)
    protected = re.sub(r"(?i)\bfig\.", "fig<prd>", protected)
    protected = re.sub(r"(?i)\beq\.", "eq<prd>", protected)
    parts = [part.strip().replace("<prd>", ".") for part in _SENTENCE_SPLIT_RE.split(protected) if str(part or "").strip()]
    if len(parts) <= 1 and ";" in src:
        parts = [part.strip() for part in re.split(r";\s+", src) if str(part or "").strip()]
    if not parts:
        return []
    merged: list[str] = []
    for part in parts:
        if merged and len(part) < 24:
            merged[-1] = (merged[-1].rstrip() + " " + part.lstrip()).strip()
            continue
        merged.append(part)
    return [item for item in merged if len(item) >= 12]


def _extract_ref_local_clause(text: str, *, match_start: int, match_end: int) -> str:
    src = re.sub(r"\s+", " ", str(text or "").strip())
    if not src:
        return ""
    left = 0
    for punct in _CLAUSE_PUNCT_RE.finditer(src[:match_start]):
        prefix = src[max(0, int(punct.start()) - 12) : int(punct.end())]
        if re.search(r"(?i)(?:et al|fig|eq)\.$", prefix):
            continue
        left = max(left, int(punct.end()))
    conj_window_left = max(left, match_start - 220)
    for conj in _CLAUSE_CONJ_RE.finditer(src[conj_window_left:match_start]):
        left = max(left, conj_window_left + int(conj.end()))

    right = len(src)
    punct_after = _CLAUSE_PUNCT_RE.search(src[match_end:])
    if punct_after:
        right = min(right, match_end + int(punct_after.start()))
    conj_after = _CLAUSE_CONJ_RE.search(src[match_end : min(len(src), match_end + 220)])
    if conj_after:
        right = min(right, match_end + int(conj_after.start()))

    clause = src[left:right].strip(" ,;:")
    return _normalize_atom_text(clause)


def _append_atom(
    atoms: list[dict],
    *,
    atom_id: str,
    atom_kind: str,
    block: dict,
    text: str,
    figure_number: int,
    panel_letters: list[str],
    box_number: int,
    inline_refs: list[int],
    dedupe: set[tuple[str, str, str]],
) -> None:
    clean = _normalize_atom_text(text)
    if len(clean) < 10:
        return
    key = (str(block.get("block_id") or "").strip(), str(atom_kind or "").strip(), normalize_match_text(clean[:320]))
    if key in dedupe:
        return
    dedupe.add(key)
    atoms.append(
        {
            "atom_id": atom_id,
            "atom_kind": str(atom_kind or "").strip(),
            "block_id": str(block.get("block_id") or "").strip(),
            "anchor_id": str(block.get("anchor_id") or "").strip(),
            "heading_path": str(block.get("heading_path") or "").strip(),
            "text": clean[:1200],
            "locate_anchor": clean[:900],
            "figure_number": int(figure_number or 0),
            "panel_letters": [str(ch or "").strip().lower() for ch in list(panel_letters or []) if str(ch or "").strip()],
            "box_number": int(box_number or 0),
            "inline_refs": [int(n) for n in list(inline_refs or []) if int(n) > 0][:8],
            "block_kind": str(block.get("kind") or "").strip().lower(),
            "order_index": int(block.get("order_index") or 0),
        }
    )


def _build_paper_guide_evidence_atoms(
    blocks: list[dict],
    *,
    max_atoms_per_block: int = 10,
) -> list[dict]:
    try:
        per_block_limit = max(3, int(max_atoms_per_block))
    except Exception:
        per_block_limit = 10
    atoms: list[dict] = []
    dedupe: set[tuple[str, str, str]] = set()
    active_box_number = 0
    active_box_heading_norm = ""
    active_box_order_index = 0
    for block in list(blocks or []):
        if not isinstance(block, dict):
            continue
        block_id = str(block.get("block_id") or "").strip()
        if not block_id:
            continue
        raw_text = str(block.get("raw_text") or block.get("text") or "").strip()
        if not raw_text:
            continue
        heading_path = str(block.get("heading_path") or "").strip()
        figure_number = _extract_figure_number(f"{heading_path}\n{raw_text[:2400]}")
        panel_letters_all = sorted(
            str(ch or "").strip().lower()
            for ch in _extract_caption_panel_letters(raw_text[:2400])
            if str(ch or "").strip()
        )
        try:
            order_index = int(block.get("order_index") or 0)
        except Exception:
            order_index = 0
        explicit_box_number = _paper_guide_box_header_number(heading_path) or _paper_guide_box_header_number(raw_text[:240])
        if explicit_box_number > 0:
            active_box_number = int(explicit_box_number)
            active_box_heading_norm = normalize_match_text(heading_path)
            active_box_order_index = int(order_index or 0)
            box_number = int(explicit_box_number)
        else:
            box_number = 0
            if (
                active_box_number > 0
                and normalize_match_text(heading_path) == active_box_heading_norm
                and int(order_index or 0) > int(active_box_order_index or 0)
                and (int(order_index or 0) - int(active_box_order_index or 0)) <= 20
            ):
                box_number = int(active_box_number)
        local_count = 0

        if panel_letters_all:
            for letter in panel_letters_all:
                if local_count >= per_block_limit:
                    break
                fragment = _extract_caption_fragment_for_letters(raw_text, {letter})
                if not fragment:
                    continue
                inline_refs = _extract_inline_reference_numbers_local(fragment, max_candidates=6)
                _append_atom(
                    atoms,
                    atom_id=f"{block_id}:cap:{letter}",
                    atom_kind="caption_clause",
                    block=block,
                    text=fragment,
                    figure_number=figure_number,
                    panel_letters=[letter],
                    box_number=box_number,
                    inline_refs=inline_refs,
                    dedupe=dedupe,
                )
                local_count += 1

        sentences = _split_evidence_sentences(raw_text[:3200])
        for sent_idx, sentence in enumerate(sentences, start=1):
            if local_count >= per_block_limit:
                break
            inline_refs = _extract_inline_reference_numbers_local(sentence, max_candidates=6)
            _append_atom(
                atoms,
                atom_id=f"{block_id}:sent:{sent_idx}",
                atom_kind="sentence",
                block=block,
                text=sentence,
                figure_number=figure_number,
                panel_letters=sorted(
                    str(ch or "").strip().lower()
                    for ch in _extract_caption_panel_letters(sentence[:800])
                    if str(ch or "").strip()
                ),
                box_number=box_number,
                inline_refs=inline_refs,
                dedupe=dedupe,
            )
            local_count += 1
            if not inline_refs or local_count >= per_block_limit:
                continue
            sentence_src = re.sub(r"\s+", " ", str(sentence or "").strip())
            ref_atom_ord = 0
            for pattern in _INLINE_REF_PATTERNS:
                for match in re.finditer(pattern, sentence_src):
                    clause = _extract_ref_local_clause(
                        sentence_src,
                        match_start=int(match.start()),
                        match_end=int(match.end()),
                    )
                    if not clause:
                        continue
                    ref_nums = _extract_inline_reference_numbers_local(clause, max_candidates=4)
                    if not ref_nums:
                        continue
                    ref_atom_ord += 1
                    _append_atom(
                        atoms,
                        atom_id=f"{block_id}:ref:{sent_idx}:{ref_atom_ord}",
                        atom_kind="ref_span",
                        block=block,
                        text=clause,
                        figure_number=figure_number,
                        panel_letters=sorted(
                            str(ch or "").strip().lower()
                            for ch in _extract_caption_panel_letters(clause[:400])
                            if str(ch or "").strip()
                        ),
                        box_number=box_number,
                        inline_refs=ref_nums,
                        dedupe=dedupe,
                    )
                    local_count += 1
                    if local_count >= per_block_limit:
                        break
                if local_count >= per_block_limit:
                    break
    return atoms
