from __future__ import annotations

import re
from pathlib import Path

from kb.inpaper_citation_grounding import (
    extract_citation_context_hints,
    parse_ref_num_set,
)
from kb.paper_guide_contracts import (
    _build_paper_guide_support_resolution,
    _normalize_paper_guide_support_slot,
)
from kb.paper_guide_evidence_atoms import _build_paper_guide_evidence_atoms
from kb.paper_guide_focus import (
    _extract_caption_fragment_for_letters,
    _extract_caption_panel_letters,
    _extract_caption_prompt_fragment,
)
from kb.paper_guide_prompting import _paper_guide_prompt_family
from kb.paper_guide_provenance import (
    _best_evidence_quote,
    _block_support_metrics,
    _extract_figure_number,
    _is_explicit_non_source_segment,
    _is_generic_heading_path,
    _is_rhetorical_shell_sentence,
    _resolve_paper_guide_md_path,
    _strip_provenance_noise_text,
    _summary_segment_tags,
    _text_token_overlap_score,
)
from kb.paper_guide_shared import (
    _extract_paper_guide_abstract_excerpt,
    _trim_paper_guide_prompt_field,
    _trim_paper_guide_prompt_snippet,
)
from kb.paper_guide_target_scope import (
    _build_paper_guide_target_scope,
    _normalize_paper_guide_target_scope,
    _paper_guide_target_scope_has_targets,
    _paper_guide_target_scope_matches_text,
)
from kb.source_blocks import (
    load_source_blocks,
    match_source_blocks,
    normalize_inline_markdown,
    normalize_match_text,
)

_PAPER_GUIDE_CITE_STOPWORDS = {
    "this",
    "that",
    "these",
    "those",
    "with",
    "from",
    "using",
    "used",
    "into",
    "than",
    "then",
    "their",
    "there",
    "which",
    "where",
    "while",
    "when",
    "under",
    "through",
    "between",
    "based",
    "shows",
    "showing",
    "paper",
    "figure",
    "panel",
    "result",
    "results",
    "evidence",
    "conclusion",
}
_PAPER_GUIDE_CITE_SHORT_TOKENS = {
    "apr",
    "cnr",
    "nec",
    "psf",
    "ism",
    "rvt",
}
_PAPER_GUIDE_CITE_STRONG_TOKENS = {
    "apr",
    "rvt",
    "phase",
    "correlation",
    "registration",
    "pinhole",
    "closed",
    "open",
    "figure",
    "panel",
    "cnr",
    "nec",
    "ipsf",
    "iscat",
    "iism",
}
_CITE_CANON_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\]\]",
    re.IGNORECASE,
)
_SUPPORT_MARKER_RE = re.compile(
    r"\[\[\s*SUPPORT\s*:\s*(DOC-(\d{1,3})(?:-S(\d{1,3}))?)\s*\]\]",
    re.IGNORECASE,
)

_PAPER_GUIDE_SUPPORT_META_LINE_RE = re.compile(
    r"(?i)^(?:"
    r"based on the retrieved (?:context|evidence|snippets)\b|"
    r"the retrieved context\b|"
    r"the retrieved evidence\b|"
    r"the provided (?:context|evidence|snippets)\b|"
    r"the discussion section is not present\b|"
    r"the paper does not (?:state|specify)\b|"
    r"the retrieved context does not (?:state|include|specify)\b|"
    r"all provided excerpts\b|"
    r"there is no discussion of\b|"
    r"therefore, per the instruction\b|"
    r"source\s*:|caption anchor\s*:|figure anchor\s*:|reference locate\s*:|参考定位\s*:|"
    r"specifically\s*:|thus, based on the provided evidence\s*:|"
    r"no supporting (?:equation|figure|section)\b|"
    r"if the discussion section were available\b"
    r")"
)
_PAPER_GUIDE_SUPPORT_METHOD_DETAIL_RE = re.compile(
    r"(?i)\b(?:workflow|pipeline|radial variance transform|rvt|phase correlation|image registration|"
    r"shift vectors?|reappl(?:y|ied)|applied back|central pixel image|off-axis raw images)\b"
)
_PAPER_GUIDE_SUPPORT_COMPARE_DETAIL_RE = re.compile(
    r"(?i)\b(?:cnr|nec|fwhm|resolution|open[- ]?pinhole|closed[- ]?pinhole|trade[- ]?off|"
    r"noise(?:-equivalent)? contrast|contrast-to-noise)\b"
)
_PAPER_GUIDE_SUPPORT_PHRASE_HINTS = (
    "phase correlation",
    "image registration",
    "radial variance transform",
    "local degree of symmetry",
    "intensity only map",
    "interferogram",
    "shift vectors",
    "original iism dataset",
    "applied back",
    "central pixel image",
    "transform domain",
    "k log(n/k)",
    "optimization problem",
)


def _extract_inline_reference_specs(text: str) -> list[str]:
    src = str(text or "").strip()
    if not src:
        return []
    specs: list[str] = []
    seen: set[str] = set()
    for pattern in (
        r"(?<![A-Za-z0-9])\[(\d{1,4}(?:\s*(?:[\-–—−,])\s*\d{1,4})*)\](?![A-Za-z])",
        r"\$\^\{(\d{1,4}(?:\s*(?:[\-–—−,])\s*\d{1,4})*)\}\$",
        r"\^\{(\d{1,4}(?:\s*(?:[\-–—−,])\s*\d{1,4})*)\}",
    ):
        for spec in re.findall(pattern, src):
            key = str(spec or "").strip()
            if (not key) or (key in seen):
                continue
            seen.add(key)
            specs.append(key)
    return specs


def _extract_inline_reference_numbers(text: str, *, max_candidates: int = 8) -> list[int]:
    src = str(text or "").strip()
    if not src:
        return []
    try:
        limit = max(1, int(max_candidates))
    except Exception:
        limit = 8
    out: list[int] = []
    seen: set[int] = set()
    for spec in _extract_inline_reference_specs(src):
        for item in parse_ref_num_set(str(spec or "").strip(), max_items=max(8, limit * 2)):
            try:
                n = int(item)
            except Exception:
                continue
            if n <= 0 or n in seen:
                continue
            seen.add(n)
            out.append(n)
            if len(out) >= limit:
                return out
    return out


def _paper_guide_cue_tokens(text: str) -> list[str]:
    s = str(text or "").strip().lower()
    if not s:
        return []
    s = re.sub(r"\[\[cite:[^\]]+\]\]", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\[\d{1,4}(?:\s*(?:[\-–—−,])\s*\d{1,4})*\]", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    out: list[str] = []
    seen: set[str] = set()
    for tok in re.findall(r"[a-z0-9]{3,}", s):
        if (len(tok) < 4) and (tok not in _PAPER_GUIDE_CITE_SHORT_TOKENS) and (not any(ch.isdigit() for ch in tok)):
            continue
        if tok in _PAPER_GUIDE_CITE_STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= 12:
            break
    return out


def _is_paper_guide_support_meta_line(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    if _is_rhetorical_shell_sentence(raw):
        return True
    if _is_explicit_non_source_segment(raw):
        return True
    if _PAPER_GUIDE_SUPPORT_META_LINE_RE.search(raw):
        return True
    return False


def _paper_guide_support_segment_spans(answer_markdown: str) -> list[dict]:
    lines = str(answer_markdown or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    spans: list[dict] = []
    buf: list[str] = []
    segment_kind = "paragraph"
    start_idx = -1
    in_fence = False

    def _push(kind: str, start_line: int, end_line: int, raw_text: str) -> None:
        raw = str(raw_text or "").strip()
        if not raw:
            return
        text = normalize_inline_markdown(raw)
        if len(text) < 10:
            return
        spans.append(
            {
                "segment_index": len(spans),
                "kind": kind,
                "line_start": int(start_line),
                "line_end": int(end_line),
                "text": text[:1600],
                "snippet_key": normalize_match_text(text[:360]),
            }
        )

    def _flush(end_line: int) -> None:
        nonlocal buf, segment_kind, start_idx
        if not buf:
            return
        _push(segment_kind, start_idx, end_line, "\n".join(buf))
        buf = []
        segment_kind = "paragraph"
        start_idx = -1

    for idx, raw in enumerate(lines):
        line = str(raw or "")
        if re.match(r"^\s*(```+|~~~+)\s*", line):
            if in_fence:
                in_fence = False
                _flush(max(start_idx, idx - 1))
            else:
                _flush(max(start_idx, idx - 1))
                in_fence = True
            continue
        if in_fence:
            continue
        if not line.strip():
            _flush(max(start_idx, idx - 1))
            continue
        if re.match(r"^\s{0,3}(#{1,6})\s+(.*)$", line):
            _flush(max(start_idx, idx - 1))
            continue
        list_match = re.match(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$", line)
        if list_match:
            _flush(max(start_idx, idx - 1))
            _push("list_item", idx, idx, str(list_match.group(1) or ""))
            continue
        if re.match(r"^\s*\|.*\|\s*$", line):
            _flush(max(start_idx, idx - 1))
            _push("table", idx, idx, line)
            continue
        quote_match = re.match(r"^\s*>\s?(.*)$", line)
        if quote_match:
            _flush(max(start_idx, idx - 1))
            _push("blockquote", idx, idx, str(quote_match.group(1) or ""))
            continue
        if start_idx < 0:
            start_idx = idx
        segment_kind = "paragraph"
        buf.append(line)
    _flush(max(start_idx, len(lines) - 1))
    return spans


def _paper_guide_support_focus_tokens(*parts: str, limit: int = 18) -> set[str]:
    out: list[str] = []
    seen: set[str] = set()
    try:
        max_items = max(4, int(limit))
    except Exception:
        max_items = 18
    for part in parts:
        for tok in _paper_guide_cue_tokens(str(part or "")):
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= max_items:
                return set(out)
    return set(out)


def _merge_paper_guide_ints(*seqs: list[int], limit: int = 8) -> list[int]:
    try:
        max_items = max(1, int(limit))
    except Exception:
        max_items = 8
    out: list[int] = []
    seen: set[int] = set()
    for seq in seqs:
        for item in list(seq or []):
            try:
                value = int(item)
            except Exception:
                continue
            if value <= 0 or value in seen:
                continue
            seen.add(value)
            out.append(value)
            if len(out) >= max_items:
                return out
    return out


def _merge_paper_guide_ref_spans(*seqs: list[dict], limit: int = 4) -> list[dict]:
    try:
        max_items = max(1, int(limit))
    except Exception:
        max_items = 4
    out: list[dict] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for seq in seqs:
        for item in list(seq or []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            nums = tuple(
                int(n)
                for n in list(item.get("nums") or [])
                if str(n).strip().isdigit() and int(n) > 0
            )
            key = (normalize_match_text(text[:220]), nums)
            if (not text and not nums) or key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "text": text,
                    "nums": list(nums),
                    "scope": str(item.get("scope") or "").strip(),
                }
            )
            if len(out) >= max_items:
                return out
    return out


def _score_paper_guide_evidence_atom(
    atom: dict,
    *,
    probe: str,
    heading: str = "",
    prompt_family: str = "",
    claim_type: str = "",
    target_scope: dict | None = None,
    query_tokens: set[str] | None = None,
    query_phrases: list[str] | None = None,
    method_focus: bool = False,
    compare_focus: bool = False,
) -> tuple[float, bool]:
    if not isinstance(atom, dict):
        return float("-inf"), False
    atom_text = str(atom.get("text") or atom.get("locate_anchor") or "").strip()
    if not atom_text:
        return float("-inf"), False
    atom_heading = str(atom.get("heading_path") or "").strip()
    atom_kind = str(atom.get("atom_kind") or "").strip().lower()
    atom_tokens = _paper_guide_support_focus_tokens(atom_heading, atom_text)
    shared_tokens = set(query_tokens or set()).intersection(atom_tokens)
    strong_shared = shared_tokens.intersection(_PAPER_GUIDE_CITE_STRONG_TOKENS)
    family = str(prompt_family or "").strip().lower()
    claim = str(claim_type or "").strip().lower()
    score = 0.0
    score += 1.15 * float(_text_token_overlap_score(probe, atom_text))
    score += 0.25 * float(_text_token_overlap_score(f"{heading}\n{probe}", f"{atom_heading}\n{atom_text}"))
    if shared_tokens:
        score += min(0.9, 0.18 * float(len(shared_tokens)))
    if strong_shared:
        score += min(0.72, 0.22 * float(len(strong_shared)))
    if query_phrases:
        score += min(
            1.0,
            0.34
            * float(
                sum(
                    1
                    for phrase in list(query_phrases or [])
                    if str(phrase or "").strip()
                    and str(phrase or "").strip() in normalize_match_text(atom_text[:1600])
                )
            ),
        )
    if atom_kind == "caption_clause":
        score += 0.22
    if atom_kind == "sentence":
        score += 0.1
    if atom_kind == "ref_span":
        score += 0.18

    if claim == "figure_panel":
        if atom_kind == "caption_clause":
            score += 2.35
        elif atom_kind == "sentence":
            score -= 0.35
    elif family == "citation_lookup" or claim in {"prior_work", "borrowed_tool"}:
        if atom_kind == "ref_span":
            score += 2.2
        elif atom_kind == "sentence" and list(atom.get("inline_refs") or []):
            score += 0.8
    elif claim in {"method_detail", "borrowed_tool"}:
        if atom_kind == "sentence":
            score += 0.85
        if method_focus and _PAPER_GUIDE_SUPPORT_METHOD_DETAIL_RE.search(atom_text):
            score += 0.65
    elif compare_focus and _PAPER_GUIDE_SUPPORT_COMPARE_DETAIL_RE.search(atom_text):
        score += 0.45
    elif family == "box_only":
        if atom_kind == "sentence":
            score += 0.9
        elif atom_kind == "ref_span":
            score -= 0.35
        if re.search(r"(?i)\b(?:transform domain|optimization problem|k\s*\\?log\s*\(n/k\)|m\s*[>=鈮]\s*o\()", atom_text):
            score += 0.95

    inline_refs = [int(n) for n in list(atom.get("inline_refs") or []) if int(n) > 0]
    if inline_refs:
        if family == "citation_lookup":
            score += min(1.4, 0.44 * float(len(inline_refs)))
        else:
            score += min(0.45, 0.12 * float(len(inline_refs)))

    target_scope_norm = _normalize_paper_guide_target_scope(target_scope)
    has_scope_targets = _paper_guide_target_scope_has_targets(target_scope_norm)
    scope_match = _paper_guide_target_scope_matches_text(
        target_scope_norm,
        text=atom_text,
        heading=atom_heading,
        box_number=int(atom.get("box_number") or 0),
        figure_number=int(atom.get("figure_number") or 0),
        panel_letters=list(atom.get("panel_letters") or []),
    )
    if has_scope_targets:
        if scope_match:
            score += 1.65
        else:
            score -= 1.05

    try:
        target_figure_num = int(target_scope_norm.get("target_figure_num") or 0)
    except Exception:
        target_figure_num = 0
    atom_figure_num = int(atom.get("figure_number") or 0)
    if target_figure_num > 0:
        if atom_figure_num == target_figure_num:
            score += 1.15
        else:
            score -= 0.6

    target_panels = {
        str(ch or "").strip().lower()
        for ch in list(target_scope_norm.get("target_panel_letters") or [])
        if str(ch or "").strip()
    }
    atom_panels = {
        str(ch or "").strip().lower()
        for ch in list(atom.get("panel_letters") or [])
        if str(ch or "").strip()
    }
    if target_panels:
        overlap = target_panels.intersection(atom_panels)
        if overlap:
            score += 1.8 + (0.24 * float(len(overlap)))
            if overlap == target_panels:
                score += 0.8
        elif atom_panels:
            score -= 0.85
        elif claim == "figure_panel":
            score -= 0.55

    target_boxes = {
        int(n)
        for n in list(target_scope_norm.get("requested_boxes") or [])
        if str(n).strip().isdigit() and int(n) > 0
    }
    try:
        atom_box_number = int(atom.get("box_number") or 0)
    except Exception:
        atom_box_number = 0
    if target_boxes:
        if atom_box_number > 0 and atom_box_number in target_boxes:
            score += 1.55
            if family == "box_only":
                score += 0.7
        elif atom_box_number > 0:
            score -= 0.95

    if len(atom_text) <= 220:
        score += 0.26
    elif len(atom_text) >= 540:
        score -= 0.18
    return score, scope_match


def _select_paper_guide_support_atoms(
    atoms: list[dict],
    *,
    probe: str,
    heading: str = "",
    prompt_family: str = "",
    claim_type: str = "",
    target_scope: dict | None = None,
) -> list[dict]:
    candidates = [dict(atom) for atom in list(atoms or []) if isinstance(atom, dict)]
    if not candidates:
        return []
    target_scope_norm = _normalize_paper_guide_target_scope(target_scope)
    query_tokens = _paper_guide_support_focus_tokens(heading, probe)
    query_text_norm = normalize_match_text(f"{heading}\n{probe}")
    query_phrases = [phrase for phrase in _PAPER_GUIDE_SUPPORT_PHRASE_HINTS if phrase in query_text_norm]
    method_focus = bool(
        str(claim_type or "").strip().lower() in {"method_detail", "borrowed_tool"}
        or _PAPER_GUIDE_SUPPORT_METHOD_DETAIL_RE.search(f"{heading}\n{probe}")
    )
    compare_focus = bool(
        str(claim_type or "").strip().lower() == "compare_result"
        or _PAPER_GUIDE_SUPPORT_COMPARE_DETAIL_RE.search(f"{heading}\n{probe}")
    )
    scored: list[dict] = []
    for atom in candidates:
        score, scope_match = _score_paper_guide_evidence_atom(
            atom,
            probe=probe,
            heading=heading,
            prompt_family=prompt_family,
            claim_type=claim_type,
            target_scope=target_scope_norm,
            query_tokens=query_tokens,
            query_phrases=query_phrases,
            method_focus=method_focus,
            compare_focus=compare_focus,
        )
        atom["atom_score"] = float(score)
        atom["scope_match"] = bool(scope_match)
        scored.append(atom)

    scored.sort(
        key=lambda atom: (
            float(atom.get("atom_score") or 0.0),
            1 if bool(atom.get("scope_match")) else 0,
            1 if str(atom.get("atom_kind") or "").strip().lower() == "caption_clause" else 0,
            1 if str(atom.get("atom_kind") or "").strip().lower() == "ref_span" else 0,
            len(str(atom.get("locate_anchor") or atom.get("text") or "").strip()),
        ),
        reverse=True,
    )
    target_panels = {
        str(ch or "").strip().lower()
        for ch in list(target_scope_norm.get("target_panel_letters") or [])
        if str(ch or "").strip()
    }
    if str(claim_type or "").strip().lower() == "figure_panel" and target_panels:
        selected: list[dict] = []
        used_panels: set[str] = set()
        preferred_rows = [
            atom
            for atom in scored
            if str(atom.get("atom_kind") or "").strip().lower() == "caption_clause"
        ]
        for pool in (preferred_rows, scored):
            for atom in pool:
                if atom in selected:
                    continue
                atom_panels = {
                    str(ch or "").strip().lower()
                    for ch in list(atom.get("panel_letters") or [])
                    if str(ch or "").strip()
                }
                overlap = target_panels.intersection(atom_panels)
                if not overlap:
                    continue
                selected.append(atom)
                used_panels.update(overlap)
                if used_panels >= target_panels:
                    break
            if used_panels >= target_panels:
                break
        if selected:
            return selected
    return scored[:4]


def _extract_paper_guide_locate_anchor(text: str, *, max_chars: int = 220) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    pieces = [
        _trim_paper_guide_prompt_snippet(str(part or "").strip(), max_chars=max_chars)
        for part in re.split(r"(?<=[.;:])\s+|\n+", src)
        if str(part or "").strip()
    ]
    for piece in pieces:
        if len(piece) >= 24:
            return piece
    return _trim_paper_guide_prompt_snippet(src, max_chars=max_chars)


def _paper_guide_support_heading_with_figure(heading_path: str, *, figure_number: int) -> str:
    heading = str(heading_path or "").strip()
    try:
        fig_num = int(figure_number or 0)
    except Exception:
        fig_num = 0
    if fig_num <= 0:
        return heading
    if re.search(rf"(?i)\bfig(?:ure)?\.?\s*{int(fig_num)}\b", heading):
        return heading
    if heading:
        return f"{heading} / Figure {int(fig_num)}"
    return f"Figure {int(fig_num)}"


def _paper_guide_support_heading_with_box(heading_path: str, *, box_number: int) -> str:
    heading = str(heading_path or "").strip()
    try:
        box_num = int(box_number or 0)
    except Exception:
        box_num = 0
    if box_num <= 0:
        return heading
    if re.search(rf"(?i)\bbox\s*{int(box_num)}\b", heading):
        return heading
    if heading:
        return f"{heading} / Box {int(box_num)}"
    return f"Box {int(box_num)}"


def _is_paper_guide_broad_summary_line(text: str, *, prompt_family: str = "") -> bool:
    raw = _strip_provenance_noise_text(text)
    if not raw:
        return False
    family = str(prompt_family or "").strip().lower()
    low = normalize_inline_markdown(raw).lower()
    summary_like = bool(
        re.search(
            r"\b(?:in summary|overall|panel-by-panel walkthrough|walk me through|walkthrough|all claims are directly supported|"
            r"grounded strictly in the retrieved context|supported by the figure caption|supported by the provided caption)\b",
            low,
            flags=re.IGNORECASE,
        )
    )
    if family == "figure_walkthrough":
        figure_num = _extract_figure_number(raw)
        panel_group_count = len(
            re.findall(
                r"\(([a-g](?:\s*[–—-]\s*[a-g]|(?:\s*,\s*[a-g])+){0,1})\)",
                raw,
                flags=re.IGNORECASE,
            )
        )
        if re.search(
            r"^\s*(?:in summary,\s*)?figure\s*\d+\s+(?:demonstrates|shows|establishes|illustrates|summarizes)\b",
            low,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(r"^\s*in summary,\s*figure\s*\d+\b", low, flags=re.IGNORECASE):
            return True
        if figure_num > 0 and summary_like:
            return True
        if figure_num > 0 and re.search(r"\bwalks the reader\b", low, flags=re.IGNORECASE):
            return True
        if figure_num > 0 and panel_group_count >= 2 and re.search(
            r"\b(?:demonstrates|shows|establishes|illustrates|validates|benchmarks|overcomes)\b",
            low,
            flags=re.IGNORECASE,
        ):
            return True
    if family == "compare" and (
        summary_like
        or re.search(r"^\s*(?:thus|overall|in summary|the trade-off is)\b", low, flags=re.IGNORECASE)
    ):
        return True
    if family == "overview" and (
        summary_like
        or bool(_summary_segment_tags(raw))
    ):
        return True
    return False


def _extract_paper_guide_ref_spans(text: str, *, max_spans: int = 4) -> list[dict]:
    src = str(text or "").strip()
    if not src:
        return []
    try:
        limit = max(1, int(max_spans))
    except Exception:
        limit = 4
    spans: list[dict] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    parts = re.split(r"(?<=[.;:])\s+|\n+", src)
    for part in parts:
        frag = _trim_paper_guide_prompt_snippet(str(part or "").strip(), max_chars=240)
        if not frag:
            continue
        for spec in _extract_inline_reference_specs(frag):
            nums = [int(n) for n in parse_ref_num_set(spec, max_items=8) if int(n) > 0]
            if not nums:
                continue
            key = (normalize_match_text(frag), tuple(nums))
            if key in seen:
                continue
            seen.add(key)
            spans.append(
                {
                    "text": frag,
                    "nums": nums,
                    "scope": "same_sentence",
                }
            )
            if len(spans) >= limit:
                return spans
    return spans


def _paper_guide_support_claim_type(
    *,
    prompt_family: str,
    heading: str = "",
    snippet: str = "",
    candidate_refs: list[int] | None = None,
    ref_spans: list[dict] | None = None,
) -> str:
    family = str(prompt_family or "").strip().lower()
    heading_low = str(heading or "").strip().lower()
    snippet_low = str(snippet or "").strip().lower()
    detail_text = f"{heading_low}\n{snippet_low}"
    has_refs = bool(candidate_refs or ref_spans)
    if family == "abstract":
        return "abstract_quote"
    if family == "figure_walkthrough":
        if _extract_caption_panel_letters(f"{heading}\n{snippet}"):
            return "figure_panel"
        if _PAPER_GUIDE_SUPPORT_METHOD_DETAIL_RE.search(detail_text):
            if has_refs and re.search(r"\b(?:library|tool|software|package|as detailed in|according to)\b", detail_text):
                return "borrowed_tool"
            return "method_detail"
        if _PAPER_GUIDE_SUPPORT_COMPARE_DETAIL_RE.search(detail_text):
            return "compare_result"
        return "figure_panel"
    if family == "method":
        if has_refs and re.search(r"\b(?:library|tool|software|package|as detailed in|according to)\b", snippet_low):
            return "borrowed_tool"
        return "method_detail"
    if family == "citation_lookup":
        if has_refs:
            return "prior_work"
        return "own_result"
    if family == "compare":
        return "compare_result"
    if has_refs and ("reference" in heading_low or "background" in heading_low or "prior" in snippet_low):
        return "prior_work"
    if family in {"overview", "strength_limits"} and has_refs:
        return "prior_work"
    return "own_result"


def _paper_guide_support_cite_policy(*, claim_type: str, prompt_family: str) -> str:
    claim = str(claim_type or "").strip().lower()
    family = str(prompt_family or "").strip().lower()
    if claim in {"prior_work", "borrowed_tool", "method_detail"}:
        return "prefer_ref"
    if claim in {"abstract_quote", "figure_panel", "own_result"}:
        return "locate_only"
    if family in {"abstract", "figure_walkthrough"}:
        return "locate_only"
    return "allow_none"


def _resolve_paper_guide_support_slot_block(
    *,
    source_path: str,
    snippet: str,
    heading: str = "",
    prompt_family: str = "",
    claim_type: str = "",
    db_dir: Path | None = None,
    block_cache: dict[str, tuple[Path, list[dict]]] | None = None,
    atom_cache: dict[str, list[dict]] | None = None,
    target_scope: dict | None = None,
) -> dict:
    src = str(source_path or "").strip()
    probe = str(snippet or "").strip()
    if (not src) or (not probe):
        return {}
    md_path: Path | None = None
    blocks: list[dict] = []
    cache = block_cache if isinstance(block_cache, dict) else None
    cached = cache.get(src) if cache is not None else None
    if isinstance(cached, tuple) and len(cached) == 2:
        md_path = cached[0]
        blocks = list(cached[1] or [])
    else:
        md_path = _resolve_paper_guide_md_path(src, db_dir=db_dir)
        if md_path is None:
            return {}
        try:
            blocks = load_source_blocks(md_path)
        except Exception:
            blocks = []
        if cache is not None:
            cache[src] = (md_path, list(blocks or []))
    if not blocks:
        return {}

    target_scope_norm = _normalize_paper_guide_target_scope(target_scope)
    heading_norm = normalize_match_text(heading)
    family = str(prompt_family or "").strip().lower()
    claim = str(claim_type or "").strip().lower()
    probe_norm = normalize_match_text(probe)
    query_text_norm = normalize_match_text(f"{heading}\n{probe}")
    method_focus = claim in {"method_detail", "borrowed_tool"} or bool(_PAPER_GUIDE_SUPPORT_METHOD_DETAIL_RE.search(f"{heading}\n{probe}"))
    compare_focus = claim == "compare_result" or bool(_PAPER_GUIDE_SUPPORT_COMPARE_DETAIL_RE.search(f"{heading}\n{probe}"))

    # In figure-walkthrough prompts we often want two different behaviors:
    # - figure-panel slots should stay scoped to the requested figure/panels
    # - method-detail slots should be allowed to land in the Methods section
    # The prompt-level target scope typically includes a figure number, which would
    # otherwise suppress the method-detail evidence selection.
    if family == "figure_walkthrough" and claim in {"method_detail", "borrowed_tool"}:
        relaxed = dict(target_scope_norm)
        relaxed["target_figure_num"] = 0
        relaxed["target_panel_letters"] = []
        relaxed["require_scope_match"] = bool(
            list(relaxed.get("requested_sections") or [])
            or list(relaxed.get("requested_boxes") or [])
        )
        relaxed["prefer_caption_atoms"] = False
        relaxed["prefer_sentence_atoms"] = True
        relaxed["allow_non_target_fallback"] = True
        target_scope_norm = _normalize_paper_guide_target_scope(relaxed)
    target_fig = int(target_scope_norm.get("target_figure_num") or 0) or _extract_figure_number(f"{heading}\n{probe}")
    target_panel_letters = {
        str(ch or "").strip().lower()
        for ch in list(target_scope_norm.get("target_panel_letters") or [])
        if str(ch or "").strip()
    } or _extract_caption_panel_letters(f"{heading}\n{probe}")
    if family == "figure_walkthrough" and claim in {"method_detail", "borrowed_tool"}:
        target_fig = 0
        target_panel_letters = set()
    query_tokens = _paper_guide_support_focus_tokens(heading, probe)
    query_phrases = [phrase for phrase in _PAPER_GUIDE_SUPPORT_PHRASE_HINTS if phrase in query_text_norm]
    block_lookup = {
        str(block.get("block_id") or "").strip(): dict(block)
        for block in list(blocks or [])
        if isinstance(block, dict) and str(block.get("block_id") or "").strip()
    }
    atoms_cache_local = atom_cache if isinstance(atom_cache, dict) else None
    atoms = list(atoms_cache_local.get(src) or []) if atoms_cache_local is not None else []
    if not atoms:
        try:
            atoms = _build_paper_guide_evidence_atoms(blocks)
        except Exception:
            atoms = []
        if atoms_cache_local is not None:
            atoms_cache_local[src] = list(atoms or [])
    selected_atoms = _select_paper_guide_support_atoms(
        atoms,
        probe=probe,
        heading=heading,
        prompt_family=family,
        claim_type=claim,
        target_scope=target_scope_norm,
    )
    best_atom = dict(selected_atoms[0]) if selected_atoms else {}
    best_atom_block = block_lookup.get(str(best_atom.get("block_id") or "").strip()) if best_atom else None
    if isinstance(best_atom_block, dict):
        best_atom_score = float(best_atom.get("atom_score") or 0.0)
        atom_kind = str(best_atom.get("atom_kind") or "").strip().lower()
        should_use_best_atom = bool(best_atom_score >= 1.45)
        if claim == "figure_panel" and atom_kind == "caption_clause":
            should_use_best_atom = True
        if family == "citation_lookup" and atom_kind == "ref_span":
            should_use_best_atom = True
        if method_focus and atom_kind == "sentence" and best_atom_score >= 1.1:
            should_use_best_atom = True
        if _paper_guide_target_scope_has_targets(target_scope_norm) and bool(best_atom.get("scope_match")) and best_atom_score >= 1.0:
            should_use_best_atom = True
        if should_use_best_atom:
            inline_refs = [int(n) for n in list(best_atom.get("inline_refs") or []) if int(n) > 0]
            ref_spans = []
            if inline_refs:
                ref_spans.append(
                    {
                        "text": str(best_atom.get("locate_anchor") or best_atom.get("text") or "").strip(),
                        "nums": inline_refs,
                        "scope": "same_clause" if atom_kind == "ref_span" else "same_sentence",
                    }
                )
            locate_anchor = str(best_atom.get("locate_anchor") or best_atom.get("text") or "").strip()
            panel_letters = [
                str(ch or "").strip().lower()
                for ch in list(best_atom.get("panel_letters") or [])
                if str(ch or "").strip()
            ]
            if (
                claim == "figure_panel"
                and atom_kind == "caption_clause"
                and target_fig > 0
                and len(target_panel_letters) >= 2
            ):
                # Multi-panel probe: prefer a caption fragment that includes all requested panels,
                # instead of returning the single best caption clause.
                block_text = str(best_atom_block.get("raw_text") or best_atom_block.get("text") or "").strip()
                merged = _extract_caption_fragment_for_letters(block_text, target_panel_letters) if block_text else ""
                if merged:
                    locate_anchor = merged
                    panel_letters = sorted(target_panel_letters)
            return {
                "md_path": str(md_path or ""),
                "block_id": str(best_atom_block.get("block_id") or "").strip(),
                "anchor_id": str(best_atom_block.get("anchor_id") or "").strip(),
                "heading_path": _paper_guide_support_heading_with_box(
                    _paper_guide_support_heading_with_figure(
                        str(best_atom_block.get("heading_path") or "").strip(),
                        figure_number=int(best_atom.get("figure_number") or target_fig or 0),
                    ),
                    box_number=int(best_atom.get("box_number") or 0),
                ),
                "locate_anchor": locate_anchor,
                "block_text": str(best_atom_block.get("raw_text") or best_atom_block.get("text") or "").strip(),
                "snippet": str(best_atom.get("text") or "").strip(),
                "evidence_atom_id": str(best_atom.get("atom_id") or "").strip(),
                "evidence_atom_kind": atom_kind,
                "evidence_atom_text": str(best_atom.get("text") or "").strip(),
                "figure_number": int(best_atom.get("figure_number") or target_fig or 0),
                "box_number": int(best_atom.get("box_number") or 0),
                "panel_letters": panel_letters,
                "candidate_refs": inline_refs,
                "ref_spans": ref_spans,
                "matching_atoms": [dict(atom) for atom in list(selected_atoms or []) if isinstance(atom, dict)],
                "target_scope": dict(target_scope_norm),
            }
    ranked_rows = [
        dict(row)
        for row in list(
            match_source_blocks(
                blocks,
                snippet=probe,
                prefer_kind="",
                target_number=0,
                limit=12,
            )
            or []
        )
        if isinstance(row, dict)
    ]
    seen_ranked_ids: set[str] = set()
    for row in ranked_rows:
        block = dict(row.get("block") or {}) if isinstance(row, dict) else {}
        block_id = str(block.get("block_id") or "").strip()
        if block_id:
            seen_ranked_ids.add(block_id)
    if query_phrases or target_fig > 0:
        extra_rows: list[tuple[int, int, dict]] = []
        for raw_block in blocks:
            block = dict(raw_block or {})
            block_id = str(block.get("block_id") or "").strip()
            if (not block_id) or (block_id in seen_ranked_ids):
                continue
            block_heading_norm = normalize_match_text(str(block.get("heading_path") or ""))
            block_text = str(block.get("raw_text") or block.get("text") or "").strip()
            block_text_norm = normalize_match_text(block_text[:2600])
            phrase_hits = sum(
                1
                for phrase in query_phrases
                if (phrase in block_text_norm) or (phrase in block_heading_norm)
            )
            fig_match = 0
            if target_fig > 0:
                try:
                    fig_match = 1 if _extract_figure_number(f"{block.get('heading_path') or ''}\n{block_text[:1400]}") == target_fig else 0
                except Exception:
                    fig_match = 0
            if phrase_hits <= 0 and fig_match <= 0:
                continue
            extra_rows.append((int(phrase_hits), int(fig_match), block))
        extra_rows.sort(
            key=lambda item: (
                int(item[0]),
                int(item[1]),
                len(str(item[2].get("raw_text") or item[2].get("text") or "").strip()),
            ),
            reverse=True,
        )
        for phrase_hits, fig_match, block in extra_rows[:12]:
            ranked_rows.append(
                {
                    "block": block,
                    "score": (0.34 * int(phrase_hits)) + (0.24 if int(fig_match) > 0 else 0.0),
                }
            )
    if claim == "figure_panel" and target_fig > 0 and target_panel_letters:
        caption_best: dict | None = None
        caption_best_score = float("-inf")
        for raw_block in blocks:
            block = dict(raw_block or {})
            kind = str(block.get("kind") or "").strip().lower()
            if kind not in {"paragraph", "list_item", "blockquote"}:
                continue
            block_text = str(block.get("raw_text") or block.get("text") or "").strip()
            if _extract_figure_number(f"{block.get('heading_path') or ''}\n{block_text[:1600]}") != target_fig:
                continue
            block_letters = _extract_caption_panel_letters(block_text)
            overlap = target_panel_letters.intersection(block_letters)
            if not overlap:
                continue
            score = 0.0
            if re.match(rf"^\s*(?:\*{{0,2}})?(?:fig(?:ure)?\.?\s*{target_fig}\b)", block_text, flags=re.IGNORECASE):
                score += 1.8
            score += 1.4 * float(len(overlap))
            if overlap == target_panel_letters:
                score += 1.2
            clause_fragment = _extract_caption_fragment_for_letters(block_text, target_panel_letters)
            if clause_fragment:
                score += 0.8
                score += 0.45 * float(_text_token_overlap_score(probe, clause_fragment))
            score += 0.22 * float(_text_token_overlap_score(probe, block_text))
            if score > caption_best_score:
                caption_best_score = score
                caption_best = block
        if isinstance(caption_best, dict) and caption_best_score >= 1.8:
            caption_text = str(caption_best.get("raw_text") or caption_best.get("text") or "").strip()
            locate_anchor = (
                _extract_caption_fragment_for_letters(caption_text, target_panel_letters)
                or _best_evidence_quote(probe, caption_best)
                or _extract_paper_guide_locate_anchor(caption_text)
            )
            return {
                "md_path": str(md_path or ""),
                "block_id": str(caption_best.get("block_id") or "").strip(),
                "anchor_id": str(caption_best.get("anchor_id") or "").strip(),
                "heading_path": _paper_guide_support_heading_with_figure(
                    str(caption_best.get("heading_path") or "").strip(),
                    figure_number=int(target_fig or _extract_figure_number(caption_text[:1600]) or 0),
                ),
                "locate_anchor": locate_anchor,
                "block_text": caption_text,
                "figure_number": int(target_fig or _extract_figure_number(caption_text[:1600]) or 0),
                "box_number": 0,
                "panel_letters": sorted(target_panel_letters),
                "target_scope": dict(target_scope_norm),
            }
    best_block: dict | None = None
    best_score = float("-inf")
    for row in ranked_rows or []:
        block = dict(row.get("block") or {}) if isinstance(row, dict) else {}
        if not block:
            continue
        score = float(row.get("score") or 0.0)
        metrics = _block_support_metrics(probe, block)
        support_score = float(metrics.get("support_score") or 0.0)
        quote_score = float(metrics.get("quote_score") or 0.0)
        heading_adjust = float(metrics.get("heading_adjust") or 0.0)
        block_heading_norm = normalize_match_text(str(block.get("heading_path") or ""))
        block_text = str(block.get("raw_text") or block.get("text") or "").strip()
        block_text_norm = normalize_match_text(block_text[:1600])
        block_fig = _extract_figure_number(f"{block.get('heading_path') or ''}\n{block_text[:800]}")
        block_tokens = _paper_guide_support_focus_tokens(str(block.get("heading_path") or ""), block_text[:1200])
        shared_tokens = query_tokens.intersection(block_tokens)
        strong_shared = shared_tokens.intersection(_PAPER_GUIDE_CITE_STRONG_TOKENS)
        score += (0.92 * support_score) + (0.18 * quote_score) + heading_adjust
        if heading_norm and block_heading_norm:
            if heading_norm in block_heading_norm or block_heading_norm.endswith(heading_norm):
                score += 0.22
        if query_tokens:
            if shared_tokens:
                score += min(0.72, 0.12 * len(shared_tokens))
                score += min(0.54, 0.18 * len(strong_shared))
            elif len(query_tokens) >= 2:
                score -= 0.32
        if query_phrases:
            score += min(
                1.08,
                0.34
                * sum(
                    1
                    for phrase in query_phrases
                    if (phrase in block_text_norm) or (phrase in block_heading_norm)
                ),
            )
        if target_fig > 0:
            if block_fig == target_fig:
                score += 0.85
            elif family == "figure_walkthrough" or claim in {"figure_panel", "compare_result", "method_detail"}:
                score -= 0.45
        if _paper_guide_target_scope_has_targets(target_scope_norm):
            scope_match = _paper_guide_target_scope_matches_text(
                target_scope_norm,
                text=block_text[:2200],
                heading=str(block.get("heading_path") or ""),
                figure_number=int(block_fig or 0),
            )
            if scope_match:
                score += 1.2
            else:
                score -= 0.95
        if method_focus:
            if any(token in block_heading_norm for token in ("materials and methods", "method", "adaptive pixel reassignment", "radial variance transform", "data analysis")):
                score += 0.35
            if any(
                token in block_text_norm
                for token in (
                    "phase correlation",
                    "image registration",
                    "shift vectors",
                    "shift vector",
                    "radial variance transform",
                    "rvt",
                    "applied back",
                    "original iism dataset",
                )
            ):
                score += 0.45
            if ("original iism dataset" in block_text_norm) and ("shift vectors" in block_text_norm):
                score += 1.1
            if ("applied back" in block_text_norm) and ("original iism dataset" in block_text_norm):
                score += 0.9
            if any(token in block_heading_norm for token in ("microscope setup", "hardware control")) and not strong_shared:
                score -= 0.7
        elif compare_focus:
            if any(token in block_heading_norm for token in ("results", "resolution", "discussion", "figure")):
                score += 0.28
        elif family == "figure_walkthrough":
            if any(token in block_heading_norm for token in ("results", "figure", "caption", "legend")):
                score += 0.28
            if any(token in block_heading_norm for token in ("microscope setup", "hardware control")) and target_fig > 0 and block_fig != target_fig:
                score -= 0.5
        if _is_generic_heading_path(str(block.get("heading_path") or "")) and not shared_tokens:
            score -= 0.6
        if (not block_heading_norm) and probe_norm and (probe_norm not in block_text_norm) and not shared_tokens:
            score -= 0.5
        if probe and block_text:
            probe_len = len(probe)
            block_len = len(block_text)
            if probe_len >= 80 and block_len < max(48, int(probe_len * 0.35)):
                score -= 0.65
            if (not block_heading_norm) and probe_len >= 80 and block_len < max(64, int(probe_len * 0.75)):
                score -= 0.9
            if block_len <= 140 and support_score < 0.18 and quote_score < 0.12:
                score -= 0.45
        if score > best_score:
            best_score = score
            best_block = block
    if not isinstance(best_block, dict):
        return {}
    block_text = str(best_block.get("raw_text") or best_block.get("text") or "").strip()
    locate_anchor = _best_evidence_quote(probe, best_block) or _extract_paper_guide_locate_anchor(block_text)
    block_box_number = 0
    explicit_boxes = [
        int(n)
        for n in list(target_scope_norm.get("requested_boxes") or [])
        if str(n).strip().isdigit() and int(n) > 0
    ]
    if explicit_boxes:
        block_scope_match = _paper_guide_target_scope_matches_text(
            target_scope_norm,
            text=block_text[:2200],
            heading=str(best_block.get("heading_path") or ""),
        )
        if block_scope_match:
            block_box_number = int(explicit_boxes[0])
    return {
        "md_path": str(md_path or ""),
        "block_id": str(best_block.get("block_id") or "").strip(),
        "anchor_id": str(best_block.get("anchor_id") or "").strip(),
        "heading_path": _paper_guide_support_heading_with_box(
            _paper_guide_support_heading_with_figure(
                str(best_block.get("heading_path") or "").strip(),
                figure_number=int(target_fig or _extract_figure_number(f"{best_block.get('heading_path') or ''}\n{block_text[:1200]}") or 0),
            ),
            box_number=int(block_box_number or 0),
        ),
        "locate_anchor": locate_anchor,
        "block_text": block_text,
        "figure_number": int(target_fig or _extract_figure_number(f"{best_block.get('heading_path') or ''}\n{block_text[:1200]}") or 0),
        "box_number": int(block_box_number or 0),
        "target_scope": dict(target_scope_norm),
    }


def _build_paper_guide_support_slots(
    cards: list[dict],
    *,
    prompt: str = "",
    prompt_family: str = "",
    db_dir: Path | None = None,
    max_slots: int = 4,
    target_scope: dict | None = None,
) -> list[dict]:
    try:
        limit = max(1, int(max_slots))
    except Exception:
        limit = 4
    family = str(prompt_family or "").strip().lower() or _paper_guide_prompt_family(prompt)
    target_scope_norm = _normalize_paper_guide_target_scope(
        target_scope or _build_paper_guide_target_scope(prompt, prompt_family=family)
    )
    slots: list[dict] = []
    seen: set[tuple[int, str, str]] = set()
    block_cache: dict[str, tuple[Path, list[dict]]] = {}
    atom_cache: dict[str, list[dict]] = {}
    for raw_card in cards or []:
        if len(slots) >= limit:
            break
        if not isinstance(raw_card, dict):
            continue
        try:
            doc_idx = max(1, int(raw_card.get("doc_idx") or 0))
        except Exception:
            doc_idx = len(slots) + 1
        sid = str(raw_card.get("sid") or "").strip()
        source_path = str(raw_card.get("source_path") or "").strip()
        heading = str(raw_card.get("heading") or "").strip()
        cue = str(raw_card.get("cue") or "").strip()
        primary = str(raw_card.get("snippet") or "").strip()
        primary_panel_letters = _extract_caption_panel_letters(primary) if (family == "figure_walkthrough" and primary) else set()
        deepread_texts = [str(item or "").strip() for item in list(raw_card.get("deepread_texts") or []) if str(item or "").strip()]

        snippet = ""
        if family == "abstract":
            for cand in deepread_texts + [primary]:
                snippet = _extract_paper_guide_abstract_excerpt(cand, max_chars=560)
                if snippet:
                    break
        elif family == "figure_walkthrough":
            for cand in deepread_texts + [primary]:
                if re.search(r"(?:^|\b)(?:fig(?:ure)?\.?\s*\d+|panel\b|caption\b)", cand, flags=re.IGNORECASE):
                    snippet = _extract_caption_prompt_fragment(cand, prompt=prompt)
                    if not snippet:
                        snippet = _trim_paper_guide_prompt_snippet(cand, max_chars=520)
                    if snippet:
                        break
        if not snippet:
            for cand in [primary, *deepread_texts]:
                snippet = _trim_paper_guide_prompt_snippet(cand, max_chars=420)
                if snippet:
                    break
        if not snippet or not source_path:
            continue

        ref_spans = _extract_paper_guide_ref_spans("\n".join([snippet, cue] + deepread_texts[:1]), max_spans=4)
        candidate_refs: list[int] = []
        seen_refs: set[int] = set()
        for span in ref_spans:
            for item in list(span.get("nums") or []):
                try:
                    n = int(item)
                except Exception:
                    continue
                if n <= 0 or n in seen_refs:
                    continue
                seen_refs.add(n)
                candidate_refs.append(n)
        for item in list(raw_card.get("candidate_refs") or []):
            try:
                n = int(item)
            except Exception:
                continue
            if n <= 0 or n in seen_refs:
                continue
            seen_refs.add(n)
            candidate_refs.append(n)

        claim_type = _paper_guide_support_claim_type(
            prompt_family=family,
            heading=heading,
            snippet=snippet,
            candidate_refs=candidate_refs,
            ref_spans=ref_spans,
        )
        support_meta = _resolve_paper_guide_support_slot_block(
            source_path=source_path,
            snippet=snippet,
            heading=heading,
            prompt_family=family,
            claim_type=claim_type,
            db_dir=db_dir,
            block_cache=block_cache,
            atom_cache=atom_cache,
            target_scope=target_scope_norm,
        )
        cite_policy = _paper_guide_support_cite_policy(claim_type=claim_type, prompt_family=family)
        desired_panels = {
            str(ch or "").strip().lower()
            for ch in list(target_scope_norm.get("target_panel_letters") or [])
            if str(ch or "").strip()
        }
        slot_variants: list[dict] = []
        if (
            family == "figure_walkthrough"
            and claim_type == "figure_panel"
            and desired_panels
            and primary_panel_letters
            and primary_panel_letters.issubset(desired_panels)
        ):
            # Only expand panel clauses into multiple slots when the incoming card snippet
            # is already panel-focused (does not contain extra panel letters beyond the target).
            block_text = str((support_meta or {}).get("block_text") or "").strip()
            letters_to_expand = [ch for ch in sorted(primary_panel_letters) if ch in desired_panels]
            for ch in letters_to_expand:
                locate = _extract_caption_fragment_for_letters(block_text, {ch}) if block_text else ""
                if not locate:
                    locate = _extract_caption_fragment_for_letters(snippet, {ch}) if snippet else ""
                variant = dict(support_meta or {})
                variant["panel_letters"] = [str(ch)]
                # Avoid de-duplication collapsing multiple per-panel variants that share the same evidence atom id.
                base_atom_id = str(variant.get("evidence_atom_id") or "").strip()
                variant["evidence_atom_id"] = f"{base_atom_id}:{ch}" if base_atom_id else f"panel:{ch}"
                if locate:
                    variant["locate_anchor"] = locate
                slot_variants.append(variant)
        if not slot_variants:
            slot_variants = [dict(support_meta or {})]

        for variant in slot_variants:
            if len(slots) >= limit:
                break
            variant_ref_spans = _merge_paper_guide_ref_spans(
                list(variant.get("ref_spans") or []),
                ref_spans,
                limit=4,
            )
            variant_ref_nums = [
                int(n)
                for span in variant_ref_spans
                if isinstance(span, dict)
                for n in list(span.get("nums") or [])
                if str(n).strip().isdigit() and int(n) > 0
            ]
            variant_candidate_refs = _merge_paper_guide_ints(
                list(variant.get("candidate_refs") or []),
                variant_ref_nums,
                candidate_refs,
                limit=6,
            )
            variant_snippet = str(variant.get("snippet") or "").strip() or snippet
            locate_anchor = str(variant.get("locate_anchor") or "").strip()
            panel_letters = [str(ch or "").strip().lower() for ch in list(variant.get("panel_letters") or []) if str(ch or "").strip()]
            if family == "figure_walkthrough" and claim_type == "figure_panel" and desired_panels:
                # Keep the prompt-extracted panel fragment as the user-facing snippet, and
                # derive the locate anchor from the resolved caption text when possible.
                variant_snippet = snippet or variant_snippet
                block_text = str(variant.get("block_text") or "").strip()
                if len(slot_variants) > 1 and panel_letters:
                    letter_set = {str(ch or "").strip().lower() for ch in panel_letters if str(ch or "").strip()}
                    clause = _extract_caption_fragment_for_letters(block_text, letter_set) if block_text else ""
                    if clause:
                        locate_anchor = clause
                    panel_letters = sorted(letter_set)
                else:
                    clause = _extract_caption_fragment_for_letters(block_text, desired_panels) if block_text else ""
                    if clause:
                        locate_anchor = clause
                    panel_letters = sorted(desired_panels)
            if not locate_anchor:
                locate_anchor = _extract_paper_guide_locate_anchor(variant_snippet)
            key_text = (
                str(variant.get("evidence_atom_id") or "").strip()
                or str(variant.get("evidence_atom_text") or "").strip()
                or locate_anchor
                or variant_snippet
            )
            key = (int(doc_idx), normalize_match_text(source_path), normalize_match_text(key_text[:240]))
            if key in seen:
                continue
            seen.add(key)
            slot = {
                "doc_idx": int(doc_idx),
                "sid": sid,
                "source_path": source_path,
                "heading": heading,
                "block_id": str(variant.get("block_id") or "").strip(),
                "anchor_id": str(variant.get("anchor_id") or "").strip(),
                "heading_path": str(variant.get("heading_path") or heading).strip(),
                "snippet": variant_snippet,
                "cue": str(variant.get("cue") or cue or "").strip(),
                "deepread_texts": deepread_texts[:2],
                "locate_anchor": locate_anchor,
                "claim_type": claim_type,
                "cite_policy": cite_policy,
                "ref_spans": variant_ref_spans,
                "candidate_refs": variant_candidate_refs[:6],
                "figure_number": int(variant.get("figure_number") or target_scope_norm.get("target_figure_num") or 0),
                "box_number": int(
                    variant.get("box_number")
                    or next(
                        (
                            int(n)
                            for n in list(target_scope_norm.get("requested_boxes") or [])
                            if str(n).strip().isdigit() and int(n) > 0
                        ),
                        0,
                    )
                    or 0
                ),
                "panel_letters": panel_letters,
                "evidence_atom_id": str(variant.get("evidence_atom_id") or "").strip(),
                "evidence_atom_kind": str(variant.get("evidence_atom_kind") or "").strip(),
                "evidence_atom_text": str(variant.get("evidence_atom_text") or "").strip(),
                "target_scope": dict(variant.get("target_scope") or target_scope_norm),
            }
            slots.append(slot)
    doc_counts: dict[int, int] = {}
    for slot in slots:
        try:
            doc_key = int(slot.get("doc_idx") or 0)
        except Exception:
            doc_key = 0
        if doc_key > 0:
            doc_counts[doc_key] = int(doc_counts.get(doc_key, 0)) + 1
    doc_ordinals: dict[int, int] = {}
    for slot in slots:
        try:
            doc_key = int(slot.get("doc_idx") or 0)
        except Exception:
            doc_key = 0
        ordinal = 1
        if doc_key > 0:
            ordinal = int(doc_ordinals.get(doc_key, 0)) + 1
            doc_ordinals[doc_key] = ordinal
        support_id = f"DOC-{int(doc_key)}"
        if int(doc_counts.get(doc_key, 0)) > 1:
            support_id = f"{support_id}-S{int(ordinal)}"
        slot["support_id"] = support_id
        slot["support_example"] = f"[[SUPPORT:{support_id}]]"
        sid = str(slot.get("sid") or "").strip()
        candidate_refs = [int(n) for n in list(slot.get("candidate_refs") or []) if int(n) > 0]
        ref_span_nums: list[int] = []
        for span in list(slot.get("ref_spans") or []):
            if not isinstance(span, dict):
                continue
            for item in list(span.get("nums") or []):
                try:
                    n = int(item)
                except Exception:
                    continue
                if n <= 0 or n in ref_span_nums:
                    continue
                ref_span_nums.append(n)
        preferred_cite_refs = ref_span_nums or candidate_refs
        if sid and preferred_cite_refs:
            slot["cite_example"] = f"[[CITE:{sid}:{int(preferred_cite_refs[0])}]]"
    return slots


def _build_paper_guide_support_slots_block(
    slots: list[dict],
    *,
    max_slots: int = 4,
) -> str:
    try:
        limit = max(1, int(max_slots))
    except Exception:
        limit = 4
    lines: list[str] = []
    for slot in list(slots or [])[:limit]:
        if not isinstance(slot, dict):
            continue
        try:
            doc_idx = max(1, int(slot.get("doc_idx") or 0))
        except Exception:
            doc_idx = len(lines) + 1
        parts = [f"DOC-{doc_idx}"]
        sid = _trim_paper_guide_prompt_field(str(slot.get("sid") or "").strip(), max_chars=24)
        if sid:
            parts.append(f"sid={sid}")
        heading = _trim_paper_guide_prompt_field(str(slot.get("heading_path") or slot.get('heading') or "").strip(), max_chars=96)
        if heading:
            parts.append(f"heading={heading}")
        claim_type = str(slot.get("claim_type") or "").strip()
        if claim_type:
            parts.append(f"claim_type={claim_type}")
        cite_policy = str(slot.get("cite_policy") or "").strip()
        if cite_policy:
            parts.append(f"cite_policy={cite_policy}")
        atom_kind = str(slot.get("evidence_atom_kind") or "").strip()
        if atom_kind:
            parts.append(f"atom={atom_kind}")
        panels = [str(ch or "").strip().lower() for ch in list(slot.get("panel_letters") or []) if str(ch or "").strip()]
        if panels:
            parts.append("panels=" + ",".join(panels[:3]))
        support_example = str(slot.get("support_example") or "").strip()
        if support_example:
            parts.append(f"support_example={support_example}")
        refs = [int(n) for n in list(slot.get("candidate_refs") or []) if int(n) > 0]
        if refs:
            parts.append("refs=" + ", ".join(str(n) for n in refs[:6]))
        cite_example = str(slot.get("cite_example") or "").strip()
        if cite_example:
            parts.append(f"cite_example={cite_example}")
        lines.append("- " + " | ".join(parts))
        snippet = _trim_paper_guide_prompt_snippet(str(slot.get("snippet") or "").strip(), max_chars=420)
        if snippet:
            lines.append("  snippet: " + snippet.replace("\n", "\n  "))
        locate_anchor = _trim_paper_guide_prompt_field(str(slot.get("locate_anchor") or "").strip(), max_chars=220)
        if locate_anchor:
            lines.append("  locate_anchor: " + locate_anchor)
        ref_spans = list(slot.get("ref_spans") or [])
        if ref_spans:
            span = ref_spans[0] or {}
            nums = ", ".join(str(int(n)) for n in list(span.get("nums") or [])[:4] if int(n) > 0)
            span_text = _trim_paper_guide_prompt_field(str(span.get("text") or "").strip(), max_chars=180)
            if nums or span_text:
                lines.append("  ref_span: " + " | ".join(part for part in [f"nums={nums}" if nums else "", span_text] if part))
    if not lines:
        return ""
    return (
        "Paper-guide support slots:\n"
        "- End each paper-grounded claim or bullet with the exact support_example marker instead of guessing a paper reference number.\n"
        "- Runtime will resolve [[SUPPORT:...]] into the final structured citation or locate-only grounding.\n"
        "- If cite_policy=locate_only, still use the support marker; do not invent a paper reference number.\n"
        + "\n".join(lines)
    )


def _normalize_paper_guide_support_surface(text: str) -> str:
    s = str(text or "")
    if not s:
        return ""
    s = _CITE_CANON_RE.sub("", s)
    s = _SUPPORT_MARKER_RE.sub("", s)
    s = normalize_inline_markdown(s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _paper_guide_support_rule_tokens(slot: dict) -> set[str]:
    if not isinstance(slot, dict):
        return set()
    cue_parts: list[str] = []
    for part in (
        slot.get("cue"),
        slot.get("heading_path"),
        slot.get("heading"),
        slot.get("snippet"),
        slot.get("locate_anchor"),
    ):
        txt = str(part or "").strip()
        if txt:
            cue_parts.append(txt)
    for extra in list(slot.get("deepread_texts") or [])[:1]:
        txt = str(extra or "").strip()
        if txt:
            cue_parts.append(txt)
    return set(_paper_guide_cue_tokens(" ".join(cue_parts)))


def _select_paper_guide_support_slot_for_context(
    slots: list[dict],
    *,
    context_text: str = "",
) -> dict | None:
    candidates = [dict(slot) for slot in list(slots or []) if isinstance(slot, dict)]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    surface = _normalize_paper_guide_support_surface(context_text)
    line_tokens = set(_paper_guide_cue_tokens(surface))
    line_panel_letters = _extract_caption_panel_letters(context_text)
    if not line_tokens:
        return candidates[0]
    best_slot: dict | None = None
    best_score = float("-inf")
    for slot in candidates:
        slot_tokens = _paper_guide_support_rule_tokens(slot)
        shared = line_tokens.intersection(slot_tokens)
        score = float(len(shared))
        if shared.intersection(_PAPER_GUIDE_CITE_STRONG_TOKENS):
            score += 0.45
        claim_type = str(slot.get("claim_type") or "").strip().lower()
        if claim_type in {"method_detail", "borrowed_tool"} and re.search(
            r"\b(?:apr|rvt|phase correlation|registration|shift|workflow|pipeline)\b",
            surface,
            flags=re.IGNORECASE,
        ):
            score += 0.3
        elif claim_type == "figure_panel" and re.search(r"\b(?:figure|fig|panel)\b", surface, flags=re.IGNORECASE):
            score += 0.25
        elif claim_type == "compare_result" and re.search(
            r"\b(?:cnr|nec|fwhm|resolution|pinhole|trade[- ]?off)\b",
            surface,
            flags=re.IGNORECASE,
        ):
            score += 0.25
        if line_panel_letters:
            slot_panel_letters = _extract_caption_panel_letters(
                " ".join(
                    [
                        str(slot.get("snippet") or "").strip(),
                        str(slot.get("locate_anchor") or "").strip(),
                    ]
                )
            )
            if claim_type == "figure_panel":
                if slot_panel_letters and line_panel_letters.intersection(slot_panel_letters):
                    score += 1.4
                elif slot_panel_letters:
                    score -= 0.5
            else:
                score -= 0.65
        if score > best_score:
            best_score = score
            best_slot = slot
    return best_slot or candidates[0]


def _inject_paper_guide_support_markers(
    answer: str,
    *,
    support_slots: list[dict],
    prompt_family: str = "",
    max_injections: int = 3,
) -> str:
    text = str(answer or "").strip()
    family = str(prompt_family or "").strip().lower()
    if (not text) or (not support_slots) or family == "abstract":
        return text
    try:
        limit = max(1, int(max_injections))
    except Exception:
        limit = 3
    if family == "figure_walkthrough":
        limit = max(limit, min(5, max(1, len(support_slots or []))))

    rules: list[dict[str, object]] = []
    for slot in support_slots or []:
        if not isinstance(slot, dict):
            continue
        marker = str(slot.get("support_example") or "").strip()
        if not marker:
            continue
        cue_parts: list[str] = []
        for part in (
            slot.get("cue"),
            slot.get("heading_path"),
            slot.get("heading"),
            slot.get("snippet"),
            slot.get("locate_anchor"),
        ):
            txt = str(part or "").strip()
            if txt:
                cue_parts.append(txt)
        for extra in list(slot.get("deepread_texts") or [])[:1]:
            txt = str(extra or "").strip()
            if txt:
                cue_parts.append(txt)
        tokens = set(_paper_guide_cue_tokens(" ".join(cue_parts)))
        if not tokens:
            continue
        rules.append(
            {
                "marker": marker,
                "tokens": tokens,
                "claim_type": str(slot.get("claim_type") or "").strip().lower(),
                "cite_policy": str(slot.get("cite_policy") or "").strip().lower(),
                "panel_letters": {
                    str(ch or "").strip().lower()
                    for ch in list(slot.get("panel_letters") or [])
                    if str(ch or "").strip()
                }
                or _extract_caption_panel_letters(" ".join(cue_parts)),
            }
        )
    if not rules:
        return text

    def _family_compatible(claim_type: str, line_text: str) -> bool:
        claim = str(claim_type or "").strip().lower()
        line = str(line_text or "")
        if not family:
            return True
        if family == "overview":
            if claim in {"own_result", "prior_work"}:
                return True
            if claim in {"method_detail", "borrowed_tool"}:
                return bool(re.search(r"\b(?:apr|implementation|algorithm|phase correlation|registration)\b", line, flags=re.IGNORECASE))
            return False
        if family == "method":
            return claim not in {"figure_panel"}
        if family == "citation_lookup":
            if claim in {"prior_work", "borrowed_tool", "method_detail", "own_result"}:
                return bool(
                    re.search(
                        r"\b(?:reference|references|citation|cited|introduced|attributed|hadamard|fourier|wavelet|richardson|lucy)\b",
                        line,
                        flags=re.IGNORECASE,
                    )
                )
            return False
        if family == "compare":
            return claim in {"compare_result", "own_result", "figure_panel", "method_detail"}
        if family == "figure_walkthrough":
            return claim in {"figure_panel", "own_result", "compare_result"}
        return True

    lines = text.splitlines()
    used_markers: set[str] = set()
    injected = 0
    skip_line_re = re.compile(
        r"(?i)(retrieved library snippets support this conclusion|check the cited section/figure|compare this result against one baseline paper)"
    )
    header_re = re.compile(
        r"^(Conclusion|Evidence|Limits|Next Steps|Abstract text|Chinese translation)\s*:?\s*$",
        flags=re.IGNORECASE,
    )
    for idx, line in enumerate(lines):
        if injected >= limit:
            break
        raw = str(line or "")
        stripped = raw.strip()
        if (not stripped) or ("[[SUPPORT:" in stripped) or ("[[CITE:" in stripped):
            continue
        if _is_paper_guide_support_meta_line(stripped):
            continue
        if header_re.match(stripped):
            continue
        if _is_paper_guide_broad_summary_line(stripped, prompt_family=family):
            continue
        if family == "figure_walkthrough" and re.match(
            r"^\s*(?:[-*+]\s+)?(?:blue|green|red|cyan|magenta|yellow|left|right|top|bottom)\s*:",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue
        if stripped.startswith("|") or skip_line_re.search(stripped):
            continue
        line_tokens = set(_paper_guide_cue_tokens(stripped))
        if not line_tokens:
            continue
        line_panel_letters = _extract_caption_panel_letters(stripped)
        best_rule: dict[str, object] | None = None
        best_score = 0.0
        best_shared: set[str] = set()
        for rule in rules:
            marker = str(rule.get("marker") or "").strip()
            if (not marker) or (marker in used_markers):
                continue
            claim_type = str(rule.get("claim_type") or "").strip().lower()
            if not _family_compatible(claim_type, stripped):
                continue
            shared = line_tokens.intersection(set(rule.get("tokens") or set()))
            if not shared:
                continue
            score = float(len(shared))
            cite_policy = str(rule.get("cite_policy") or "").strip().lower()
            if shared.intersection(_PAPER_GUIDE_CITE_STRONG_TOKENS):
                score += 0.45
            if cite_policy == "prefer_ref":
                score += 0.1
            if family == "method":
                if claim_type == "method_detail":
                    score += 0.35
                if re.search(r"\b(?:implementation detail|phase correlation|registration|apr)\b", stripped, flags=re.IGNORECASE):
                    score += 0.45
            elif family == "compare":
                if claim_type in {"compare_result", "own_result"}:
                    score += 0.3
                if re.search(r"\b(?:open|closed|pinhole|cnr|nec|fwhm|resolution|noise)\b", stripped, flags=re.IGNORECASE):
                    score += 0.45
            elif family == "figure_walkthrough":
                if claim_type == "figure_panel":
                    score += 0.35
                if re.search(r"\b(?:figure|fig|panel)\b", stripped, flags=re.IGNORECASE):
                    score += 0.45
                if line_panel_letters:
                    rule_panels = {str(ch or "").strip().lower() for ch in set(rule.get("panel_letters") or set()) if str(ch or "").strip()}
                    if claim_type == "figure_panel":
                        score += 0.9
                        if rule_panels:
                            overlap = line_panel_letters.intersection(rule_panels)
                            if overlap:
                                score += 1.35 + (0.2 * float(len(overlap)))
                            else:
                                score -= 0.65
                    else:
                        score -= 0.75
            elif family == "overview":
                if claim_type in {"own_result", "prior_work"}:
                    score += 0.2
            if score > best_score:
                best_score = score
                best_rule = rule
                best_shared = set(shared)
        min_score = 2.0
        if family in {"method", "compare", "figure_walkthrough", "overview"}:
            min_score = 1.35
        if best_shared.intersection(_PAPER_GUIDE_CITE_STRONG_TOKENS):
            min_score = min(min_score, 1.0)
        if not best_rule or best_score < min_score:
            continue
        marker = str(best_rule.get("marker") or "").strip()
        if not marker:
            continue
        lines[idx] = raw.rstrip() + " " + marker
        used_markers.add(marker)
        injected += 1
    return "\n".join(lines).strip()


def _resolve_paper_guide_support_ref_num(slot: dict, *, context_text: str = "") -> tuple[int | None, str]:
    if not isinstance(slot, dict):
        return None, "missing_slot"
    cite_policy = str(slot.get("cite_policy") or "").strip().lower()
    if cite_policy == "locate_only":
        return None, "locate_only"
    if _is_paper_guide_support_meta_line(str(context_text or "")):
        return None, "missing_local_evidence"
    explicit_context_refs = _extract_inline_reference_numbers(str(context_text or ""), max_candidates=4)
    ref_spans = [dict(item) for item in list(slot.get("ref_spans") or []) if isinstance(item, dict)]
    if explicit_context_refs:
        local_ref_nums: list[int] = []
        for span in ref_spans:
            for item in list(span.get("nums") or []):
                try:
                    n = int(item)
                except Exception:
                    continue
                if n > 0 and n not in local_ref_nums:
                    local_ref_nums.append(n)
        for item in list(slot.get("candidate_refs") or []):
            try:
                n = int(item)
            except Exception:
                continue
            if n > 0 and n not in local_ref_nums:
                local_ref_nums.append(n)
        for n in explicit_context_refs:
            if int(n) > 0 and int(n) in local_ref_nums:
                return int(n), "context_explicit_ref"
    if ref_spans:
        context_tokens = set(_paper_guide_cue_tokens(str(context_text or "")))
        best_num: int | None = None
        best_score = float("-inf")
        for span in ref_spans:
            nums = [int(n) for n in list(span.get("nums") or []) if int(n) > 0]
            if len(nums) != 1:
                continue
            span_text = str(span.get("text") or "").strip()
            span_tokens = set(_paper_guide_cue_tokens(span_text))
            score = 0.0
            span_scope = str(span.get("scope") or "").strip().lower()
            if span_scope == "same_clause":
                score += 1.45
            elif span_scope == "same_sentence":
                score += 1.25
            if context_tokens and span_tokens:
                score += 0.55 * float(len(context_tokens.intersection(span_tokens)))
            elif span_tokens:
                score += 0.2 * float(len(span_tokens))
            if score > best_score:
                best_score = score
                best_num = int(nums[0])
        if int(best_num or 0) > 0:
            return int(best_num), "slot_ref_span"
    candidate_refs = [int(n) for n in list(slot.get("candidate_refs") or []) if int(n) > 0]
    uniq = list(dict.fromkeys(candidate_refs))
    if len(uniq) == 1:
        return int(uniq[0]), "slot_candidate"
    hints = extract_citation_context_hints(str(context_text or ""), token_start=0, token_end=len(str(context_text or "")))
    if uniq and (str(hints.get("doi") or "").strip() or (str(hints.get("author") or "").strip() and str(hints.get("year") or "").strip())):
        return int(uniq[0]), "context_hint_candidate"
    return None, "no_ref"


def _resolve_paper_guide_support_markers(
    answer: str,
    *,
    support_slots: list[dict],
    prompt_family: str = "",
    db_dir: Path | None = None,
) -> tuple[str, list[dict]]:
    text = str(answer or "")
    if not text:
        return text, []
    slot_by_support_id: dict[str, dict] = {}
    slots_by_doc_idx: dict[int, list[dict]] = {}
    normalized_slots = [
        _normalize_paper_guide_support_slot(slot)
        for slot in list(support_slots or [])
        if isinstance(slot, dict)
    ]
    for slot in normalized_slots:
        if not isinstance(slot, dict):
            continue
        support_id = str(slot.get("support_id") or "").strip()
        if support_id:
            slot_by_support_id[support_id] = slot
        try:
            doc_idx = int(slot.get("doc_idx") or 0)
        except Exception:
            doc_idx = 0
        if doc_idx > 0:
            slots_by_doc_idx.setdefault(doc_idx, []).append(slot)
    if not slots_by_doc_idx:
        return _SUPPORT_MARKER_RE.sub("", text), []

    resolutions: list[dict] = []
    lines = text.splitlines()
    for idx, raw_line in enumerate(lines):
        if "[[SUPPORT:" not in str(raw_line or ""):
            continue
        line = str(raw_line or "")
        surface = _normalize_paper_guide_support_surface(line)
        if _is_paper_guide_support_meta_line(surface):
            lines[idx] = _SUPPORT_MARKER_RE.sub("", line).rstrip()
            continue

        def _repl(m: re.Match[str]) -> str:
            support_id = str(m.group(1) or "").strip()
            try:
                doc_idx = int(m.group(2) or 0)
            except Exception:
                doc_idx = 0
            slot = slot_by_support_id.get(support_id) if support_id else None
            if not isinstance(slot, dict):
                slot = _select_paper_guide_support_slot_for_context(
                    slots_by_doc_idx.get(doc_idx, []),
                    context_text=surface,
                )
            if not isinstance(slot, dict):
                resolutions.append(
                    _build_paper_guide_support_resolution(
                        doc_idx=int(doc_idx),
                        support_id=support_id,
                        segment_text=surface,
                        line_index=int(idx),
                        citation_resolution_mode="missing_slot",
                    )
                )
                return ""
            rebound = _resolve_paper_guide_support_slot_block(
                source_path=str(slot.get("source_path") or "").strip(),
                snippet=surface,
                heading=str(slot.get("heading_path") or slot.get("heading") or "").strip(),
                prompt_family=prompt_family,
                claim_type=str(slot.get("claim_type") or "").strip(),
                db_dir=db_dir,
                target_scope=dict(slot.get("target_scope") or {}),
            )
            slot_ref_spans = _merge_paper_guide_ref_spans(
                list(rebound.get("ref_spans") or []),
                list(slot.get("ref_spans") or []),
                limit=4,
            )
            slot_candidate_refs = _merge_paper_guide_ints(
                list(rebound.get("candidate_refs") or []),
                [
                    int(n)
                    for span in slot_ref_spans
                    if isinstance(span, dict)
                    for n in list(span.get("nums") or [])
                    if str(n).strip().isdigit() and int(n) > 0
                ],
                list(slot.get("candidate_refs") or []),
                limit=6,
            )
            slot_for_ref = dict(slot)
            slot_for_ref["ref_spans"] = slot_ref_spans
            slot_for_ref["candidate_refs"] = slot_candidate_refs
            block_id = str(rebound.get("block_id") or slot.get("block_id") or "").strip()
            anchor_id = str(rebound.get("anchor_id") or slot.get("anchor_id") or "").strip()
            heading_path = str(rebound.get("heading_path") or slot.get("heading_path") or slot.get("heading") or "").strip()
            locate_anchor = str(rebound.get("locate_anchor") or slot.get("locate_anchor") or "").strip()
            ref_num, mode = _resolve_paper_guide_support_ref_num(slot_for_ref, context_text=line)
            resolutions.append(
                _build_paper_guide_support_resolution(
                    doc_idx=int(doc_idx),
                    support_id=support_id or str(slot.get("support_id") or "").strip(),
                    sid=str(slot.get("sid") or "").strip(),
                    source_path=str(slot.get("source_path") or "").strip(),
                    block_id=block_id,
                    anchor_id=anchor_id,
                    heading_path=heading_path,
                    locate_anchor=locate_anchor,
                    claim_type=str(slot.get("claim_type") or "").strip(),
                    cite_policy=str(slot.get("cite_policy") or "").strip(),
                    candidate_refs=slot_candidate_refs,
                    ref_spans=slot_ref_spans,
                    evidence_atom_id=str(rebound.get("evidence_atom_id") or slot.get("evidence_atom_id") or "").strip(),
                    evidence_atom_kind=str(rebound.get("evidence_atom_kind") or slot.get("evidence_atom_kind") or "").strip(),
                    evidence_atom_text=str(rebound.get("evidence_atom_text") or slot.get("evidence_atom_text") or "").strip(),
                    figure_number=int(rebound.get("figure_number") or slot.get("figure_number") or 0),
                    box_number=int(rebound.get("box_number") or slot.get("box_number") or 0),
                    panel_letters=list(rebound.get("panel_letters") or slot.get("panel_letters") or []),
                    target_scope=dict(rebound.get("target_scope") or slot.get("target_scope") or {}),
                    resolved_ref_num=int(ref_num or 0),
                    citation_resolution_mode=mode,
                    segment_text=surface,
                    line_index=int(idx),
                )
            )
            if int(ref_num or 0) > 0:
                sid = str(slot.get("sid") or "").strip()
                if sid:
                    return f"[[CITE:{sid}:{int(ref_num)}]]"
            return ""

        replaced = _SUPPORT_MARKER_RE.sub(_repl, line)
        cleaned = re.sub(r"[ \t]{2,}", " ", replaced).rstrip()
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        lines[idx] = cleaned

    out = "\n".join(lines)
    out = _SUPPORT_MARKER_RE.sub("", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    segment_spans = _paper_guide_support_segment_spans(out)
    for rec in resolutions:
        try:
            line_index_raw = rec.get("line_index")
            line_index = int(line_index_raw) if str(line_index_raw).strip() else -1
        except Exception:
            line_index = -1
        if line_index < 0:
            continue
        for span in segment_spans:
            try:
                start_line_raw = span.get("line_start")
                end_line_raw = span.get("line_end")
                start_line = int(start_line_raw) if str(start_line_raw).strip() else -1
                end_line = int(end_line_raw) if str(end_line_raw).strip() else -1
            except Exception:
                continue
            if start_line <= line_index <= end_line:
                rec["segment_index"] = int(span.get("segment_index") or 0)
                rec["segment_kind"] = str(span.get("kind") or "").strip()
                rec["segment_snippet_key"] = str(span.get("snippet_key") or "").strip()
                rec["segment_text"] = str(span.get("text") or rec.get("segment_text") or "").strip()
                break
    return out.strip(), resolutions
