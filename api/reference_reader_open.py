from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re

from api.reference_value_utils import _positive_int


PRIMARY_REF_EVIDENCE_MAX_LEN = 520


def _compact_reader_open_text(text: str, *, max_len: int = 360) -> str:
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw:
        return ""
    if len(raw) <= max_len:
        return raw
    return raw[:max_len].rstrip() + "..."


def _pick_reader_open_loc_text(loc: dict) -> str:
    if not isinstance(loc, dict):
        return ""
    for key in ("snippet", "text", "quote", "content", "summary", "why"):
        value = _compact_reader_open_text(str(loc.get(key) or ""))
        if value:
            return value
    return ""


def _refs_reader_open_candidate_key(candidate: dict) -> str:
    if not isinstance(candidate, dict):
        return ""
    heading_path = str(candidate.get("headingPath") or "").strip()
    highlight_snippet = str(candidate.get("highlightSnippet") or "").strip()
    snippet = str(candidate.get("snippet") or "").strip()
    anchor_kind = str(candidate.get("anchorKind") or "").strip().lower()
    anchor_number = _positive_int(candidate.get("anchorNumber"))
    block_id = str(candidate.get("blockId") or "").strip()
    anchor_id = str(candidate.get("anchorId") or "").strip()
    if not any((heading_path, highlight_snippet, snippet, anchor_kind, anchor_number, block_id, anchor_id)):
        return ""
    return "::".join(
        [
            heading_path.lower(),
            highlight_snippet.lower()[:180],
            snippet.lower()[:180],
            anchor_kind,
            str(anchor_number or ""),
            block_id.lower(),
            anchor_id.lower(),
        ]
    )


def _normalize_refs_reader_heading_path(
    *,
    prompt: str,
    source_path: str,
    heading_path: str,
    sanitize_heading_path: Callable[..., str],
    looks_like_doc_title_heading: Callable[[str, str], bool],
) -> str:
    heading = sanitize_heading_path(
        str(heading_path or "").strip(),
        prompt=prompt,
        source_path=source_path,
    )
    if heading and " / " in heading:
        parts = [str(part or "").strip() for part in heading.split(" / ") if str(part or "").strip()]
        if len(parts) >= 2 and looks_like_doc_title_heading(parts[0], source_path):
            heading = " / ".join(parts[1:]).strip()
        elif len(parts) >= 3 and (not re.match(r"^\d", parts[0])) and re.match(r"^\d", parts[1]):
            heading = " / ".join(parts[1:]).strip()
    return heading


def _refs_heading_paths_related(left: str, right: str) -> bool:
    left_norm = str(left or "").strip().lower()
    right_norm = str(right or "").strip().lower()
    if (not left_norm) or (not right_norm):
        return False
    return (
        left_norm == right_norm
        or left_norm.startswith(f"{right_norm} /")
        or right_norm.startswith(f"{left_norm} /")
    )


def _refs_heading_anchor_number(
    anchor_kind: str,
    heading_path: str,
    *,
    extract_figure_number: Callable[[str], int],
    extract_equation_number: Callable[[str], int],
) -> int:
    kind = str(anchor_kind or "").strip().lower()
    heading = str(heading_path or "").strip()
    if (not kind) or (not heading):
        return 0
    if kind == "figure":
        return extract_figure_number(heading)
    if kind == "equation":
        return extract_equation_number(heading)
    if kind == "table":
        m = re.search(r"(?:table|tab\.?|表)\s*[\(#\[]?\s*(\d{1,4})(?!\d)", heading, flags=re.I)
        if not m:
            return 0
        try:
            value = int(str(m.group(1) or "0"))
        except Exception:
            return 0
        return value if value > 0 else 0
    return 0


def _clean_refs_evidence_snippet(
    raw: str,
    *,
    prompt: str,
    source_path: str,
    display_name: str = "",
    heading_path: str = "",
    max_len: int = 360,
    pick_readable_evidence_text: Callable[..., str],
    clean_evidence_display_text: Callable[..., str],
) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    title_hint = str(display_name or Path(str(source_path or "")).name or "").strip()
    picked = pick_readable_evidence_text(
        text,
        source=source_path,
        title=title_hint,
        claim=prompt,
        heading=heading_path,
        max_len=max_len,
    )
    return picked or clean_evidence_display_text(text, max_len=max_len)


def _build_refs_reader_open_candidate(
    *,
    prompt: str,
    source_path: str,
    heading_path: str,
    snippet: str,
    highlight_snippet: str,
    anchor_kind: str,
    anchor_number: int,
    sanitize_heading_path: Callable[..., str],
    looks_like_doc_title_heading: Callable[[str, str], bool],
    pick_readable_evidence_text: Callable[..., str],
    clean_evidence_display_text: Callable[..., str],
) -> dict | None:
    heading = _normalize_refs_reader_heading_path(
        prompt=prompt,
        source_path=source_path,
        heading_path=heading_path,
        sanitize_heading_path=sanitize_heading_path,
        looks_like_doc_title_heading=looks_like_doc_title_heading,
    )
    snippet_text = _clean_refs_evidence_snippet(
        snippet,
        prompt=prompt,
        source_path=source_path,
        heading_path=heading,
        max_len=360,
        pick_readable_evidence_text=pick_readable_evidence_text,
        clean_evidence_display_text=clean_evidence_display_text,
    )
    highlight_text = _clean_refs_evidence_snippet(
        highlight_snippet or snippet_text,
        prompt=prompt,
        source_path=source_path,
        heading_path=heading,
        max_len=360,
        pick_readable_evidence_text=pick_readable_evidence_text,
        clean_evidence_display_text=clean_evidence_display_text,
    )
    candidate = {
        "headingPath": heading or None,
        "snippet": snippet_text or None,
        "highlightSnippet": highlight_text or None,
        "anchorKind": str(anchor_kind or "").strip().lower() or None,
        "anchorNumber": _positive_int(anchor_number) or None,
    }
    if not any(candidate.values()):
        return None
    return {key: value for key, value in candidate.items() if value not in (None, "", [], {})}


def _infer_heading_path_for_summary_from_source_blocks(
    *,
    prompt: str,
    source_path: str,
    summary_line: str,
    anchor_target_kind: str,
    anchor_target_number: int,
    resolve_source_md_path: Callable[[str], object],
    load_source_blocks: Callable[[object], list],
    match_source_blocks: Callable[..., list],
    sanitize_heading_path: Callable[..., str],
    looks_like_doc_title_heading: Callable[[str, str], bool],
) -> str:
    seed = _compact_reader_open_text(summary_line)
    if not seed:
        return ""
    md_path = resolve_source_md_path(source_path)
    if md_path is None:
        return ""
    try:
        blocks = load_source_blocks(md_path)
    except Exception:
        return ""
    if not blocks:
        return ""
    try:
        matches = match_source_blocks(
            blocks,
            snippet=seed,
            heading_path="",
            prefer_kind=anchor_target_kind,
            target_number=anchor_target_number,
            limit=3,
            score_floor=0.24,
        )
    except Exception:
        matches = []
    for row in matches:
        block = row.get("block") if isinstance(row, dict) else {}
        heading_path = _normalize_refs_reader_heading_path(
            prompt=prompt,
            source_path=source_path,
            heading_path=str((block or {}).get("heading_path") or "").strip(),
            sanitize_heading_path=sanitize_heading_path,
            looks_like_doc_title_heading=looks_like_doc_title_heading,
        )
        if heading_path:
            return heading_path
    return ""


def _score_refs_exact_surface(
    text: str,
    *,
    prompt: str,
    title: str,
    block_kind: str = "",
    anchor_target_kind: str = "",
    looks_bibliographic_source_block_text: Callable[[str], bool],
    looks_title_like_ref_surface: Callable[[str, str], bool],
    looks_like_front_matter_ref_summary: Callable[[str], bool],
    looks_prefixed_heading_shell_ref_summary: Callable[[str], bool],
    looks_surface_like_ref_summary: Callable[[str], bool],
    looks_fragmentary_ref_summary: Callable[[str], bool],
    looks_why_like_ref_summary: Callable[[str], bool],
    looks_formula_heavy_ref_text: Callable[[str], bool],
    prompt_reference_focus_action: Callable[[str], str],
    refs_summary_focus_keyword_hit_count: Callable[[str, str], int],
    looks_natural_language_ref_summary: Callable[[str], bool],
    has_ref_summary_explainer_signal: Callable[[str], bool],
    has_ref_summary_value_signal: Callable[[str], bool],
    refs_exact_focus_match_count: Callable[[str, str], int],
    matched_focus_terms_for_ref_card: Callable[..., list],
) -> float:
    surface = _compact_reader_open_text(text)
    if not surface:
        return -1000.0
    score = 0.0
    block_kind_norm = str(block_kind or "").strip().lower()
    anchor_kind_norm = str(anchor_target_kind or "").strip().lower()
    apply_summary_shape_penalties = block_kind_norm != "paragraph"
    if looks_bibliographic_source_block_text(surface):
        score -= 5.0
    if title and looks_title_like_ref_surface(surface, title):
        score -= 5.2
    if looks_like_front_matter_ref_summary(surface):
        score -= 3.8
    if apply_summary_shape_penalties and looks_prefixed_heading_shell_ref_summary(surface):
        score -= 3.2
    if apply_summary_shape_penalties and looks_surface_like_ref_summary(surface):
        score -= 2.8
    if looks_fragmentary_ref_summary(surface):
        score -= 2.6
    if looks_why_like_ref_summary(surface):
        score -= 2.6
    if looks_formula_heavy_ref_text(surface) and anchor_kind_norm != "equation":
        score -= 1.4
    focus_action = prompt_reference_focus_action(prompt)
    keyword_hits = refs_summary_focus_keyword_hit_count(prompt, surface) if prompt else 0
    if focus_action == "compare" and re.search(
        r"\b(compare|comparison|versus|vs\.?|difference|whereas|while)\b",
        surface,
        flags=re.I,
    ):
        score += 0.9
        if keyword_hits >= 2:
            score += 3.2
    if focus_action == "define":
        if re.search(r"\b(define|defines|defined|definition|refers to|known as|is known as|is called)\b", surface, flags=re.I):
            score += 0.9
        elif re.match(r"^\s*if\b", surface, flags=re.I):
            score += 0.35
    if block_kind_norm == "heading":
        score -= 4.6
    if block_kind_norm in {"figure", "table"} and not anchor_kind_norm:
        score -= 2.8
    if (not anchor_kind_norm) and re.match(r"^\s*(?:fig(?:ure)?\.?|table)\b", surface, flags=re.I):
        score -= 2.4
    if (not anchor_kind_norm) and re.match(r"^\s*\([A-Z]\)\s", surface):
        score -= 2.1
    if looks_natural_language_ref_summary(surface):
        score += 1.0
    if has_ref_summary_explainer_signal(surface):
        score += 0.9
    if has_ref_summary_value_signal(surface):
        score += 0.5
    if len(surface) >= 56:
        score += 0.25
    if block_kind_norm == "paragraph":
        score += 0.35
        if len(surface) >= 120:
            score += 0.35
    elif len(surface) > 420:
        score -= 0.8
    if prompt:
        score += 0.45 * float(refs_exact_focus_match_count(prompt, surface))
        score += 0.35 * float(len(matched_focus_terms_for_ref_card(prompt, surface_text=surface)))
        score += 0.15 * float(keyword_hits)
    return score


def _select_reader_open_exact_snippet(
    seed_text: str,
    block_text: str,
    *,
    prompt: str = "",
    title: str = "",
    block_kind: str = "",
    anchor_target_kind: str = "",
    score_refs_exact_surface: Callable[..., float],
    looks_focus_prefixed_ref_summary: Callable[[str, str], bool],
    summary_line_needs_polish: Callable[..., bool],
) -> tuple[str, str]:
    seed = _compact_reader_open_text(seed_text)
    block = _compact_reader_open_text(block_text)
    if not block:
        return seed, seed
    if not seed:
        return block, block
    seed_score = score_refs_exact_surface(
        seed,
        prompt=prompt,
        title=title,
        block_kind="",
        anchor_target_kind=anchor_target_kind,
    )
    block_score = score_refs_exact_surface(
        block,
        prompt=prompt,
        title=title,
        block_kind=block_kind,
        anchor_target_kind=anchor_target_kind,
    )
    if block_score >= (seed_score + 1.0):
        return block, block
    if (
        prompt
        and looks_focus_prefixed_ref_summary(prompt, seed)
        and block_kind.strip().lower() == "paragraph"
        and block_score > -0.25
    ):
        return block, block
    if prompt and summary_line_needs_polish(prompt=prompt, title=title, summary_line=seed) and block_score >= (seed_score - 0.15):
        return block, block
    if seed and block:
        seed_key = re.sub(r"\s+", " ", seed).strip().lower()
        block_key = re.sub(r"\s+", " ", block).strip().lower()
        if seed_key and block_key and (seed_key in block_key or block_key in seed_key):
            if (block_score >= (seed_score + 0.35)) and len(block) > (len(seed) + 24):
                return block, block
            return seed, seed
    if (seed_score < -1.5) and (block_score > seed_score):
        return block, block
    return seed, seed


def _build_refs_exact_candidate_from_block(
    *,
    prompt: str,
    source_path: str,
    title: str,
    block: dict,
    seed_heading_path: str,
    seed_snippet: str,
    anchor_kind: str,
    anchor_number: int,
    select_reader_open_exact_snippet: Callable[..., tuple[str, str]],
    build_refs_reader_open_candidate: Callable[..., dict | None],
) -> dict | None:
    if not isinstance(block, dict):
        return None
    block_id = str(block.get("block_id") or "").strip()
    anchor_id = str(block.get("anchor_id") or "").strip()
    if not block_id:
        return None
    heading_path = str(block.get("heading_path") or seed_heading_path or "").strip()
    block_text = str(block.get("text") or block.get("raw_text") or "").strip()
    block_kind = str(block.get("kind") or "").strip().lower()
    snippet_text, highlight_text = select_reader_open_exact_snippet(
        seed_snippet,
        block_text,
        prompt=prompt,
        title=title,
        block_kind=block_kind,
        anchor_target_kind=anchor_kind,
    )
    candidate = build_refs_reader_open_candidate(
        prompt=prompt,
        source_path=source_path,
        heading_path=heading_path,
        snippet=snippet_text,
        highlight_snippet=highlight_text,
        anchor_kind=anchor_kind or str(block.get("kind") or ""),
        anchor_number=anchor_number or int(block.get("number") or 0),
    )
    if not isinstance(candidate, dict):
        return None
    candidate["blockId"] = block_id
    if anchor_id:
        candidate["anchorId"] = anchor_id
    return candidate


def _build_preferred_refs_exact_candidate_from_source_summary(
    *,
    prompt: str,
    source_path: str,
    title: str,
    summary_line: str,
    selected_heading_path: str,
    anchor_target_kind: str,
    anchor_target_number: int,
    prompt_aligned_candidate: dict | None,
    ref_summary_surfaces_match: Callable[[str, str], bool],
    normalize_refs_reader_heading_path: Callable[..., str],
    select_reader_open_exact_snippet: Callable[..., tuple[str, str]],
    build_refs_reader_open_candidate: Callable[..., dict | None],
) -> dict:
    if not isinstance(prompt_aligned_candidate, dict):
        return {}
    if str(prompt_aligned_candidate.get("source_kind") or "").strip().lower() != "source_block":
        return {}
    block_id = str(prompt_aligned_candidate.get("block_id") or "").strip()
    if not block_id:
        return {}

    candidate_summary = str(prompt_aligned_candidate.get("summary") or "").strip()
    if summary_line and candidate_summary and (not ref_summary_surfaces_match(summary_line, candidate_summary)):
        return {}

    block_heading_path = normalize_refs_reader_heading_path(
        prompt=prompt,
        source_path=source_path,
        heading_path=str(prompt_aligned_candidate.get("heading_path") or "").strip(),
    )
    selected_heading = normalize_refs_reader_heading_path(
        prompt=prompt,
        source_path=source_path,
        heading_path=selected_heading_path,
    )
    if selected_heading and block_heading_path and block_heading_path != selected_heading:
        return {}

    block_kind = str(prompt_aligned_candidate.get("block_kind") or "").strip().lower()
    target_kind = str(anchor_target_kind or "").strip().lower()
    if target_kind and block_kind and block_kind != target_kind:
        return {}

    block_text = str(prompt_aligned_candidate.get("block_text") or "").strip()
    seed_snippet = candidate_summary or summary_line or block_text
    snippet_text, highlight_text = select_reader_open_exact_snippet(
        seed_snippet,
        block_text,
        prompt=prompt,
        title=title,
        block_kind=block_kind,
        anchor_target_kind=target_kind,
    )
    if (not snippet_text) and block_text:
        snippet_text = _compact_reader_open_text(block_text)
    if not highlight_text:
        highlight_text = snippet_text

    candidate = build_refs_reader_open_candidate(
        prompt=prompt,
        source_path=source_path,
        heading_path=block_heading_path or selected_heading or selected_heading_path,
        snippet=snippet_text,
        highlight_snippet=highlight_text,
        anchor_kind=target_kind or block_kind,
        anchor_number=anchor_target_number or _positive_int(prompt_aligned_candidate.get("block_number")),
    )
    if not isinstance(candidate, dict):
        return {}
    candidate["blockId"] = block_id
    anchor_id = str(prompt_aligned_candidate.get("anchor_id") or "").strip()
    if anchor_id:
        candidate["anchorId"] = anchor_id
    return candidate


def _resolve_refs_exact_candidates(
    *,
    prompt: str,
    source_path: str,
    display_name: str = "",
    anchor_target_kind: str,
    anchor_target_number: int,
    primary_candidate: dict | None,
    secondary_candidates: list[dict],
    allow_llm_disambiguation: bool = True,
    resolve_source_md_path: Callable[[str], object],
    load_source_blocks: Callable[[object], list],
    match_source_blocks: Callable[..., list],
    build_refs_exact_candidate_from_block: Callable[..., dict | None],
    refs_heading_paths_related: Callable[[str, str], bool],
    refs_heading_anchor_number: Callable[[str, str], int],
    score_refs_exact_surface: Callable[..., float],
    refs_exact_focus_match_count: Callable[[str, str], int],
    matched_focus_terms_for_ref_card: Callable[..., list],
    should_try_refs_locate_llm: Callable[[list[dict]], bool],
    llm_pick_refs_exact_candidate_index: Callable[..., int],
) -> list[dict]:
    md_path = resolve_source_md_path(source_path)
    if md_path is None:
        return []
    try:
        blocks = load_source_blocks(md_path)
    except Exception:
        return []
    if not blocks:
        return []

    seed_candidates = [primary_candidate] if isinstance(primary_candidate, dict) else []
    seed_candidates.extend(item for item in (secondary_candidates or []) if isinstance(item, dict))
    if not seed_candidates:
        return []
    primary_heading_norm = str(
        ((primary_candidate or {}) if isinstance(primary_candidate, dict) else {}).get("headingPath") or ""
    ).strip().lower()

    out_rows: list[dict] = []
    seen_blocks: set[str] = set()
    for seed in seed_candidates[:6]:
        heading_path = str(seed.get("headingPath") or "").strip()
        snippet = str(seed.get("highlightSnippet") or seed.get("snippet") or "").strip()
        score_floor = 0.52 if snippet else 0.68
        if _positive_int(anchor_target_number) > 0:
            score_floor = 0.34 if snippet else 0.58
        try:
            matches = match_source_blocks(
                blocks,
                snippet=snippet,
                heading_path=heading_path,
                prefer_kind=anchor_target_kind,
                target_number=anchor_target_number,
                limit=3,
                score_floor=score_floor,
            )
        except Exception:
            matches = []
        for row in matches:
            block = row.get("block")
            candidate = build_refs_exact_candidate_from_block(
                prompt=prompt,
                source_path=source_path,
                title=display_name,
                block=block if isinstance(block, dict) else {},
                seed_heading_path=heading_path,
                seed_snippet=snippet,
                anchor_kind=anchor_target_kind,
                anchor_number=anchor_target_number,
            )
            if not isinstance(candidate, dict):
                continue
            block_id = str(candidate.get("blockId") or "").strip()
            if (not block_id) or (block_id in seen_blocks):
                continue
            seen_blocks.add(block_id)
            out_rows.append(
                {
                    "candidate": candidate,
                    "score": float(row.get("score") or 0.0),
                    "block_text": str(((block or {}) if isinstance(block, dict) else {}).get("text") or "").strip(),
                    "block_kind": str(((block or {}) if isinstance(block, dict) else {}).get("kind") or "").strip().lower(),
                    "heading_path": heading_path,
                }
            )
            if len(out_rows) >= 5:
                break
        if len(out_rows) >= 5:
            break

    if len(out_rows) <= 1:
        return [dict(item.get("candidate") or {}) for item in out_rows if isinstance(item.get("candidate"), dict)]

    anchor_kind_norm = str(anchor_target_kind or "").strip().lower()
    target_anchor_num = _positive_int(anchor_target_number)

    def _exact_candidate_sort_key(item: dict) -> tuple[float, float, int, int, int, int, int, float]:
        candidate = dict(item.get("candidate") or {}) if isinstance(item.get("candidate"), dict) else {}
        candidate_heading = str(candidate.get("headingPath") or "").strip().lower()
        seed_heading = str(item.get("heading_path") or "").strip().lower()
        block_text = str(item.get("block_text") or "").strip()
        surface = block_text or str(candidate.get("highlightSnippet") or candidate.get("snippet") or "").strip()
        primary_match = int(bool(primary_heading_norm and candidate_heading and candidate_heading == primary_heading_norm))
        primary_related = int(bool(primary_heading_norm and candidate_heading and refs_heading_paths_related(candidate_heading, primary_heading_norm)))
        seed_match = int(bool(seed_heading and candidate_heading and candidate_heading == seed_heading))
        heading_anchor_num = refs_heading_anchor_number(anchor_kind_norm, candidate_heading)
        target_heading_match = int(bool(target_anchor_num > 0 and heading_anchor_num == target_anchor_num))
        target_heading_conflict = int(bool(target_anchor_num > 0 and heading_anchor_num > 0 and heading_anchor_num != target_anchor_num))
        quality_score = score_refs_exact_surface(
            surface,
            prompt=prompt,
            title=display_name,
            block_kind=str(item.get("block_kind") or "").strip().lower(),
            anchor_target_kind=anchor_target_kind,
        )
        raw_score = float(item.get("score") or 0.0)
        exact_focus_hits = refs_exact_focus_match_count(prompt, surface)
        focus_hits = len(matched_focus_terms_for_ref_card(prompt, surface_text=surface))
        combined_score = (
            float(quality_score)
            + (0.8 * raw_score)
            + (0.25 * float(exact_focus_hits))
            + (0.15 * float(focus_hits))
            + (0.26 * float(primary_match))
            + (0.85 * float(primary_related))
            + (0.12 * float(seed_match))
            + (3.2 * float(target_heading_match))
            - (5.2 * float(target_heading_conflict))
        )
        return (
            float(combined_score),
            float(quality_score),
            target_heading_match,
            -target_heading_conflict,
            primary_related,
            primary_match,
            seed_match,
            raw_score,
        )

    out_rows.sort(key=_exact_candidate_sort_key, reverse=True)
    if allow_llm_disambiguation and should_try_refs_locate_llm(out_rows):
        candidate_lines: list[str] = []
        for idx, row in enumerate(out_rows[:3], start=1):
            candidate = dict(row.get("candidate") or {}) if isinstance(row.get("candidate"), dict) else {}
            candidate_lines.append(
                "\n".join(
                    [
                        f"{idx}. heading: {str(candidate.get('headingPath') or '').strip() or '(none)'}",
                        f"   snippet: {str(candidate.get('highlightSnippet') or candidate.get('snippet') or '').strip()[:260]}",
                        f"   block_text: {str(row.get('block_text') or '').strip()[:260]}",
                        f"   anchor: {str(candidate.get('anchorKind') or '').strip()} {str(candidate.get('anchorNumber') or '').strip()}",
                        f"   heuristic_score: {float(row.get('score') or 0.0):.3f}",
                    ]
                )
            )
        picked = llm_pick_refs_exact_candidate_index(
            prompt=str(prompt or "").strip(),
            source_path=str(source_path or "").strip(),
            anchor_target_kind=str(anchor_target_kind or "").strip().lower(),
            anchor_target_number=int(_positive_int(anchor_target_number)),
            candidates_payload="\n\n".join(candidate_lines),
        )
        if picked > 0 and picked <= min(3, len(out_rows)):
            chosen = out_rows[picked - 1]
            out_rows = [chosen] + [row for idx, row in enumerate(out_rows) if idx != (picked - 1)]

    return [dict(item.get("candidate") or {}) for item in out_rows if isinstance(item.get("candidate"), dict)]


def _build_refs_reader_open_payload(
    *,
    meta: dict,
    prompt: str,
    source_path: str,
    display_name: str,
    heading_path: str,
    heading: str,
    summary_line: str,
    why_line: str,
    anchor_target_kind: str,
    anchor_target_number: int,
    build_refs_reader_open_candidate: Callable[..., dict | None],
    resolve_refs_exact_candidates: Callable[..., list[dict]],
    prompt_requires_explicit_focus_match: Callable[[str], bool],
    preferred_exact_candidate: dict | None = None,
    allow_llm_disambiguation: bool = True,
    allow_exact_locate: bool = True,
) -> dict:
    primary_heading = str(heading_path or heading or "").strip()
    primary_snippet = _compact_reader_open_text(summary_line or why_line)
    primary_candidate = build_refs_reader_open_candidate(
        prompt=prompt,
        source_path=source_path,
        heading_path=primary_heading,
        snippet=primary_snippet,
        highlight_snippet=primary_snippet,
        anchor_kind=anchor_target_kind,
        anchor_number=anchor_target_number,
    )

    secondary_candidates: list[dict] = []
    seen_secondary: set[str] = set()
    primary_key = _refs_reader_open_candidate_key(primary_candidate or {})

    def _push_secondary(candidate: dict | None) -> None:
        if not isinstance(candidate, dict):
            return
        key = _refs_reader_open_candidate_key(candidate)
        if (not key) or (key == primary_key) or (key in seen_secondary):
            return
        seen_secondary.add(key)
        secondary_candidates.append(candidate)

    raw_locs = (meta or {}).get("ref_locs") if isinstance(meta, dict) else None
    if isinstance(raw_locs, list):
        for loc in raw_locs[:4]:
            if not isinstance(loc, dict):
                continue
            loc_heading = str(loc.get("heading_path") or loc.get("heading") or "").strip()
            loc_snippet = _pick_reader_open_loc_text(loc) or primary_snippet
            _push_secondary(
                build_refs_reader_open_candidate(
                    prompt=prompt,
                    source_path=source_path,
                    heading_path=loc_heading,
                    snippet=loc_snippet,
                    highlight_snippet=loc_snippet,
                    anchor_kind=anchor_target_kind,
                    anchor_number=anchor_target_number,
                )
            )

    snippet_seed_keys = (
        ("ref_show_snippets", 3),
        ("ref_snippets", 3),
        ("ref_overview_snippets", 2),
    )
    for meta_key, limit in snippet_seed_keys:
        raw_arr = (meta or {}).get(meta_key) if isinstance(meta, dict) else None
        if not isinstance(raw_arr, list):
            continue
        for item in raw_arr[:limit]:
            snippet_text = _compact_reader_open_text(str(item or ""))
            if not snippet_text:
                continue
            _push_secondary(
                build_refs_reader_open_candidate(
                    prompt=prompt,
                    source_path=source_path,
                    heading_path=primary_heading,
                    snippet=snippet_text,
                    highlight_snippet=snippet_text,
                    anchor_kind=anchor_target_kind,
                    anchor_number=anchor_target_number,
                )
            )

    ref_pack_state = str((meta or {}).get("ref_pack_state") or "").strip().lower() if isinstance(meta, dict) else ""
    if (ref_pack_state == "pending") or (not allow_exact_locate):
        visible_candidates: list[dict] = []
        seen_visible: set[str] = set()

        def _push_visible_pending(candidate: dict | None) -> None:
            if not isinstance(candidate, dict):
                return
            key = _refs_reader_open_candidate_key(candidate)
            if (not key) or (key in seen_visible):
                return
            seen_visible.add(key)
            visible_candidates.append(candidate)

        _push_visible_pending(primary_candidate)
        for candidate in secondary_candidates:
            _push_visible_pending(candidate)
        visible_candidates = visible_candidates[:6]
        effective_primary = visible_candidates[0] if visible_candidates else primary_candidate
        secondary_visible = visible_candidates[1:] if len(visible_candidates) > 1 else []
        reader_open = {
            "sourcePath": source_path,
            "sourceName": display_name,
            "headingPath": str((effective_primary or {}).get("headingPath") or primary_heading or "").strip() or None,
            "snippet": str((effective_primary or {}).get("snippet") or primary_snippet or "").strip() or None,
            "highlightSnippet": str((effective_primary or {}).get("highlightSnippet") or primary_snippet or "").strip() or None,
            "anchorKind": str((effective_primary or {}).get("anchorKind") or anchor_target_kind or "").strip().lower() or None,
            "anchorNumber": _positive_int((effective_primary or {}).get("anchorNumber") or anchor_target_number) or None,
            "strictLocate": False,
            "alternatives": secondary_visible or None,
            "visibleAlternatives": visible_candidates if len(visible_candidates) > 1 else None,
            "evidenceAlternatives": visible_candidates if len(visible_candidates) > 1 else None,
            "initialAltIndex": 0 if visible_candidates else None,
        }
        return {key: value for key, value in reader_open.items() if value not in (None, "", [], {})}

    exact_candidates: list[dict] = []
    seen_exact: set[str] = set()

    def _push_exact(candidate: dict | None) -> None:
        if not isinstance(candidate, dict):
            return
        key = _refs_reader_open_candidate_key(candidate)
        if (not key) or (key in seen_exact):
            return
        seen_exact.add(key)
        exact_candidates.append(candidate)

    _push_exact(preferred_exact_candidate)
    for candidate in resolve_refs_exact_candidates(
        prompt=prompt,
        source_path=source_path,
        display_name=display_name,
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        primary_candidate=primary_candidate,
        secondary_candidates=secondary_candidates,
        allow_llm_disambiguation=allow_llm_disambiguation,
    ):
        _push_exact(candidate)
    primary_heading_norm = str((primary_candidate or {}).get("headingPath") or primary_heading or "").strip().lower()
    prompt_is_focus_no_anchor = bool(
        primary_heading_norm
        and (not str(anchor_target_kind or "").strip())
        and prompt_requires_explicit_focus_match(prompt)
    )
    if len(exact_candidates) >= 1 and prompt_is_focus_no_anchor:
        top_heading_norm = str((exact_candidates[0].get("headingPath") or "")).strip().lower()
        if top_heading_norm and (not _refs_heading_paths_related(top_heading_norm, primary_heading_norm)):
            found_related = False
            for idx, candidate in enumerate(exact_candidates[1:], start=1):
                candidate_heading_norm = str((candidate.get("headingPath") or "")).strip().lower()
                if _refs_heading_paths_related(candidate_heading_norm, primary_heading_norm):
                    exact_candidates = [candidate] + [item for j, item in enumerate(exact_candidates) if j != idx]
                    found_related = True
                    break
            if not found_related:
                exact_candidates = []
    primary_exact = exact_candidates[0] if exact_candidates else None
    related_block_ids = [
        str(candidate.get("blockId") or "").strip()
        for candidate in exact_candidates
        if str(candidate.get("blockId") or "").strip()
    ]
    related_block_ids = list(dict.fromkeys(related_block_ids))[:5]

    effective_primary = primary_exact or primary_candidate
    if effective_primary is not primary_candidate and prompt_is_focus_no_anchor:
        eff_heading = str((effective_primary or {}).get("headingPath") or "").strip().lower()
        if eff_heading and (not _refs_heading_paths_related(eff_heading, primary_heading_norm)):
            effective_primary = dict(effective_primary or {})
            effective_primary["headingPath"] = primary_heading
    visible_candidates: list[dict] = []
    seen_visible: set[str] = set()

    def _push_visible(candidate: dict | None) -> None:
        if not isinstance(candidate, dict):
            return
        key = _refs_reader_open_candidate_key(candidate)
        if (not key) or (key in seen_visible):
            return
        seen_visible.add(key)
        visible_candidates.append(candidate)

    _push_visible(effective_primary)
    for candidate in exact_candidates[1:]:
        _push_visible(candidate)
    for candidate in secondary_candidates:
        _push_visible(candidate)

    visible_candidates = visible_candidates[:6]
    secondary_visible = [candidate for candidate in visible_candidates if candidate is not effective_primary]
    secondary_candidates = secondary_visible[:5]
    locate_target = (
        {
            "headingPath": str((primary_exact or {}).get("headingPath") or "").strip() or None,
            "snippet": str((primary_exact or {}).get("snippet") or "").strip() or None,
            "highlightSnippet": str((primary_exact or {}).get("highlightSnippet") or "").strip() or None,
            "blockId": str((primary_exact or {}).get("blockId") or "").strip() or None,
            "anchorId": str((primary_exact or {}).get("anchorId") or "").strip() or None,
            "anchorKind": str((primary_exact or {}).get("anchorKind") or anchor_target_kind or "").strip().lower() or None,
            "anchorNumber": _positive_int((primary_exact or {}).get("anchorNumber") or anchor_target_number) or None,
            "hitLevel": "block",
            "relatedBlockIds": related_block_ids or None,
        }
        if primary_exact
        else None
    )
    if isinstance(locate_target, dict):
        locate_target = {key: value for key, value in locate_target.items() if value not in (None, "", [], {})}
    reader_open = {
        "sourcePath": source_path,
        "sourceName": display_name,
        "headingPath": str((effective_primary or {}).get("headingPath") or primary_heading or "").strip() or None,
        "snippet": str((effective_primary or {}).get("snippet") or primary_snippet or "").strip() or None,
        "highlightSnippet": str((effective_primary or {}).get("highlightSnippet") or primary_snippet or "").strip() or None,
        "blockId": str((effective_primary or {}).get("blockId") or "").strip() or None,
        "anchorId": str((effective_primary or {}).get("anchorId") or "").strip() or None,
        "relatedBlockIds": related_block_ids or None,
        "anchorKind": str((effective_primary or {}).get("anchorKind") or anchor_target_kind or "").strip().lower() or None,
        "anchorNumber": _positive_int((effective_primary or {}).get("anchorNumber") or anchor_target_number) or None,
        "strictLocate": bool(primary_exact),
        "locateTarget": locate_target,
        "alternatives": secondary_candidates or None,
        "visibleAlternatives": visible_candidates if len(visible_candidates) > 1 else None,
        "evidenceAlternatives": visible_candidates if len(visible_candidates) > 1 else None,
        "initialAltIndex": 0 if visible_candidates else None,
    }
    return {key: value for key, value in reader_open.items() if value not in (None, "", [], {})}


def _build_primary_ref_evidence_payload(
    *,
    source_path: str,
    display_name: str,
    reader_open: dict,
    selection_reason: str,
    score: float | None,
    prompt: str = "",
    clean_refs_evidence_snippet: Callable[..., str],
) -> dict:
    if not isinstance(reader_open, dict):
        return {}

    def _candidate_to_evidence(candidate: dict | None) -> dict | None:
        if not isinstance(candidate, dict):
            return None
        heading_path = str(candidate.get("headingPath") or "").strip()
        snippet = clean_refs_evidence_snippet(
            str(candidate.get("snippet") or "").strip(),
            prompt=prompt,
            source_path=source_path,
            display_name=display_name,
            heading_path=heading_path,
            max_len=460,
        )
        highlight_snippet = clean_refs_evidence_snippet(
            str(candidate.get("highlightSnippet") or snippet or "").strip(),
            prompt=prompt,
            source_path=source_path,
            display_name=display_name,
            heading_path=heading_path,
            max_len=460,
        )
        evidence = {
            "source_path": str(source_path or "").strip() or None,
            "source_name": str(display_name or "").strip() or None,
            "block_id": str(candidate.get("blockId") or "").strip() or None,
            "anchor_id": str(candidate.get("anchorId") or "").strip() or None,
            "heading_path": heading_path or None,
            "snippet": snippet or None,
            "highlight_snippet": highlight_snippet or None,
            "anchor_kind": str(candidate.get("anchorKind") or "").strip().lower() or None,
            "anchor_number": _positive_int(candidate.get("anchorNumber")) or None,
        }
        return {key: value for key, value in evidence.items() if value not in (None, "", [], {})}

    primary_candidate = {
        "headingPath": str(reader_open.get("headingPath") or "").strip(),
        "snippet": str(reader_open.get("snippet") or "").strip(),
        "highlightSnippet": str(reader_open.get("highlightSnippet") or "").strip(),
        "blockId": str(reader_open.get("blockId") or "").strip(),
        "anchorId": str(reader_open.get("anchorId") or "").strip(),
        "anchorKind": str(reader_open.get("anchorKind") or "").strip().lower(),
        "anchorNumber": _positive_int(reader_open.get("anchorNumber")),
    }
    primary_key = _refs_reader_open_candidate_key(primary_candidate)
    primary_evidence = _candidate_to_evidence(primary_candidate)
    if not isinstance(primary_evidence, dict) or not primary_evidence:
        return {}

    alternatives: list[dict] = []
    seen_alt_keys: set[str] = set()
    for raw_candidate in list(reader_open.get("evidenceAlternatives") or reader_open.get("visibleAlternatives") or reader_open.get("alternatives") or []):
        if not isinstance(raw_candidate, dict):
            continue
        key = _refs_reader_open_candidate_key(raw_candidate)
        if (not key) or (key == primary_key) or (key in seen_alt_keys):
            continue
        seen_alt_keys.add(key)
        alt = _candidate_to_evidence(raw_candidate)
        if isinstance(alt, dict) and alt:
            alternatives.append(alt)
        if len(alternatives) >= 5:
            break

    out = dict(primary_evidence)
    if selection_reason:
        out["selection_reason"] = str(selection_reason or "").strip()
    if score is not None:
        try:
            out["score"] = float(score)
        except Exception:
            pass
    out["strict_locate"] = bool(reader_open.get("strictLocate"))
    if alternatives:
        out["alternatives"] = alternatives
    return out


def _normalize_primary_ref_evidence_payload(
    primary_evidence: dict | None,
    *,
    finish_evidence_text: Callable[..., str],
) -> dict:
    if not isinstance(primary_evidence, dict):
        return {}
    snippet = finish_evidence_text(
        str(primary_evidence.get("snippet") or "").strip(),
        max_len=PRIMARY_REF_EVIDENCE_MAX_LEN,
    )
    highlight_snippet = finish_evidence_text(
        str(primary_evidence.get("highlight_snippet") or primary_evidence.get("highlightSnippet") or "").strip(),
        max_len=PRIMARY_REF_EVIDENCE_MAX_LEN,
    )
    out = {
        "source_path": str(primary_evidence.get("source_path") or primary_evidence.get("sourcePath") or "").strip() or None,
        "source_name": str(primary_evidence.get("source_name") or primary_evidence.get("sourceName") or "").strip() or None,
        "block_id": str(primary_evidence.get("block_id") or primary_evidence.get("blockId") or "").strip() or None,
        "anchor_id": str(primary_evidence.get("anchor_id") or primary_evidence.get("anchorId") or "").strip() or None,
        "heading_path": str(primary_evidence.get("heading_path") or primary_evidence.get("headingPath") or "").strip() or None,
        "snippet": snippet or None,
        "highlight_snippet": highlight_snippet or snippet or None,
        "anchor_kind": str(primary_evidence.get("anchor_kind") or primary_evidence.get("anchorKind") or "").strip().lower() or None,
        "anchor_number": _positive_int(primary_evidence.get("anchor_number") or primary_evidence.get("anchorNumber")) or None,
        "page_start": _positive_int(primary_evidence.get("page_start") or primary_evidence.get("pageStart")) or None,
        "page_end": _positive_int(primary_evidence.get("page_end") or primary_evidence.get("pageEnd")) or None,
        "selection_reason": str(primary_evidence.get("selection_reason") or primary_evidence.get("selectionReason") or "").strip() or None,
    }
    strict_locate_raw = primary_evidence.get("strict_locate")
    if strict_locate_raw is None:
        strict_locate_raw = primary_evidence.get("strictLocate")
    if strict_locate_raw is not None:
        out["strict_locate"] = bool(strict_locate_raw)
    score_raw = primary_evidence.get("score")
    try:
        if score_raw is not None:
            out["score"] = float(score_raw)
    except Exception:
        pass
    alts: list[dict] = []
    for raw_alt in list(primary_evidence.get("alternatives") or []):
        norm_alt = _normalize_primary_ref_evidence_payload(
            raw_alt,
            finish_evidence_text=finish_evidence_text,
        )
        if norm_alt:
            alts.append(norm_alt)
        if len(alts) >= 5:
            break
    if alts:
        out["alternatives"] = alts
    return {
        key: value
        for key, value in out.items()
        if value not in (None, "", [], {})
    }


def _build_doc_list_reader_open_payload(
    *,
    source_path: str,
    source_name: str,
    heading_path: str,
    summary_line: str,
    primary_evidence: dict,
    reader_open: dict | None,
    normalize_primary_ref_evidence_payload: Callable[[dict | None], dict],
    clean_refs_evidence_snippet: Callable[..., str],
) -> dict:
    primary = normalize_primary_ref_evidence_payload(primary_evidence)
    out = dict(reader_open or {}) if isinstance(reader_open, dict) else {}
    if source_path:
        out["sourcePath"] = source_path
    if source_name:
        out["sourceName"] = source_name
    auth_heading = str(primary.get("heading_path") or heading_path or out.get("headingPath") or "").strip()
    auth_snippet = clean_refs_evidence_snippet(
        str(primary.get("snippet") or out.get("snippet") or summary_line or "").strip(),
        prompt="",
        source_path=source_path,
        display_name=source_name,
        heading_path=auth_heading,
        max_len=460,
    )
    auth_highlight = clean_refs_evidence_snippet(
        str(primary.get("highlight_snippet") or auth_snippet or out.get("highlightSnippet") or "").strip(),
        prompt="",
        source_path=source_path,
        display_name=source_name,
        heading_path=auth_heading,
        max_len=460,
    )
    if auth_heading:
        out["headingPath"] = auth_heading
    if auth_snippet:
        out["snippet"] = auth_snippet
    if auth_highlight:
        out["highlightSnippet"] = auth_highlight
    for src_key, dst_key in (
        ("block_id", "blockId"),
        ("anchor_id", "anchorId"),
        ("anchor_kind", "anchorKind"),
    ):
        value = str(primary.get(src_key) or "").strip()
        if value:
            out[dst_key] = value
    anchor_number = _positive_int(primary.get("anchor_number"))
    if anchor_number > 0:
        out["anchorNumber"] = anchor_number
    if "strict_locate" in primary:
        out["strictLocate"] = bool(primary.get("strict_locate"))
    if primary:
        out["primaryEvidence"] = dict(primary)
    return {
        key: value
        for key, value in out.items()
        if value not in (None, "", [], {})
    }
