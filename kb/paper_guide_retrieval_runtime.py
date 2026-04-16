from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from pathlib import Path

from kb.paper_guide_evidence_atoms import _build_paper_guide_evidence_atoms
from kb.paper_guide_focus import _extract_caption_panel_letters
from kb.paper_guide.grounder import (
    _extract_inline_reference_numbers,
    _paper_guide_cue_tokens,
)
from kb.paper_guide_prompting import (
    _PAPER_GUIDE_CITATION_LOOKUP_ATTRIBUTION_RE,
    _PAPER_GUIDE_CITATION_LOOKUP_PROMPT_RE,
    _PAPER_GUIDE_CITATION_LOOKUP_QUERY_STOPWORDS,
    _PAPER_GUIDE_REF_SECTION_RE,
    _augment_paper_guide_retrieval_prompt,
    _looks_like_reference_list_snippet_local,
    _paper_guide_box_header_number,
    _paper_guide_prompt_family,
    _paper_guide_prompt_requests_exact_method_support,
    _paper_guide_requested_box_numbers,
    _paper_guide_requested_heading_hints,
    _paper_guide_requested_section_targets,
    _paper_guide_text_matches_requested_box,
    _paper_guide_text_matches_requested_section,
    _paper_guide_text_matches_requested_targets,
)
from kb.paper_guide_shared import (
    _trim_paper_guide_prompt_field,
    _trim_paper_guide_prompt_snippet,
)
from kb.paper_guide_provenance import (
    _extract_figure_number,
    _is_hit_from_bound_source,
    _resolve_paper_guide_md_path,
)
from kb.retrieval_engine import _deep_read_md_for_context
from kb.source_blocks import load_source_blocks, normalize_inline_markdown, normalize_match_text


def _paper_guide_seed_query_tokens_for_targeted_scan(
    *,
    prompt: str,
    family: str,
    bound_source_path: str,
) -> set[str]:
    q = str(prompt or "").strip()
    tokens = set(_paper_guide_cue_tokens(q))
    augmented = _augment_paper_guide_retrieval_prompt(
        q,
        family=family,
    )
    tokens.update(_paper_guide_cue_tokens(augmented))
    if tokens:
        return tokens
    raw_src = str(bound_source_path or "").strip()
    src_seed = re.sub(r"\.pdf$", "", raw_src, flags=re.IGNORECASE)
    seeded = f"{src_seed} method methods result results figure equation reference discussion"
    return set(_paper_guide_cue_tokens(seeded))


def _paper_guide_best_block_for_fallback_hit(
    *,
    hit_text: str,
    block_rows: list[dict],
) -> dict | None:
    snippet = str(hit_text or "").strip()
    if not snippet:
        return None
    snippet_norm = normalize_match_text(snippet)
    if not snippet_norm:
        return None
    snippet_tokens = set(_paper_guide_cue_tokens(snippet[:1200]))
    snippet_head = snippet_norm[:220]
    best_row: dict | None = None
    best_score = 0.0
    for row in block_rows:
        block_norm = str(row.get("norm_text") or "")
        if not block_norm:
            continue
        score = 0.0
        if snippet_norm == block_norm:
            score += 140.0
        elif snippet_norm in block_norm:
            score += 120.0
        elif snippet_head and snippet_head in block_norm:
            score += 78.0
        elif (len(block_norm) >= 60) and (block_norm in snippet_norm):
            score += 48.0
        block_tokens = set(row.get("tokens") or [])
        shared = snippet_tokens.intersection(block_tokens) if snippet_tokens and block_tokens else set()
        if shared:
            score += min(26.0, 2.4 * float(len(shared)))
        try:
            ratio = SequenceMatcher(None, snippet_norm[:320], block_norm[:420]).ratio()
        except Exception:
            ratio = 0.0
        score += 16.0 * float(ratio)
        if score > best_score:
            best_score = score
            best_row = row
    if (best_row is None) or (best_score < 7.5):
        return None
    return best_row


def _paper_guide_deepread_heading(hit: dict) -> str:
    if not isinstance(hit, dict):
        return ""
    meta = hit.get("meta", {}) or {}
    heading = str(meta.get("heading_path") or "").strip()
    if heading:
        return heading
    text = str(hit.get("text") or "").strip()
    if not text:
        return ""
    for ln in text.splitlines():
        s = str(ln or "").strip()
        if not s:
            continue
        if s.startswith("#"):
            return s.lstrip("#").strip()
        break
    return ""


def _select_paper_guide_deepread_extras(
    extras: list[dict],
    *,
    prompt: str,
    prompt_family: str = "",
    limit: int = 2,
) -> list[str]:
    try:
        keep_limit = max(1, int(limit))
    except Exception:
        keep_limit = 2

    family = str(prompt_family or "").strip().lower()
    q = str(prompt or "").strip()
    wants_references = bool(
        re.search(r"(\breference\b|\bcitation\b|\bcite\b|参考文献|引用|引文|bibliography)", q, flags=re.IGNORECASE)
    )
    target_sections = _paper_guide_requested_section_targets(q)
    target_boxes = _paper_guide_requested_box_numbers(q)
    target_figure_num = _extract_figure_number(q)
    target_panels = _extract_caption_panel_letters(q)
    is_citation_lookup = bool(_paper_guide_prompt_family(q) == "citation_lookup")

    ranked: list[tuple[float, str]] = []
    for ex in extras or []:
        if not isinstance(ex, dict):
            continue
        text = str(ex.get("text") or "").strip()
        if not text:
            continue
        heading = _paper_guide_deepread_heading(ex)
        heading_norm = normalize_match_text(heading)
        text_norm = normalize_match_text(text[:240])
        if (not wants_references) and (
            (_PAPER_GUIDE_REF_SECTION_RE.search(heading or "") is not None)
            or _looks_like_reference_list_snippet_local(text)
        ):
            continue

        try:
            score = float(ex.get("score") or 0.0)
        except Exception:
            score = 0.0

        if target_boxes:
            box_match = any(
                _paper_guide_text_matches_requested_box(heading, n)
                or _paper_guide_text_matches_requested_box(text[:360], n)
                for n in target_boxes
            )
            score += 48.0 if box_match else -16.0

        if target_sections:
            heading_match = any(_paper_guide_text_matches_requested_section(heading, sec) for sec in target_sections)
            text_match = any(_paper_guide_text_matches_requested_section(text[:320], sec) for sec in target_sections)
            if heading_match:
                score += 26.0
            elif text_match:
                score += 11.0
            elif heading_norm:
                score -= 8.0

        if target_figure_num > 0:
            heading_fig = _extract_figure_number(heading)
            text_fig = _extract_figure_number(text[:320])
            if heading_fig == target_figure_num or text_fig == target_figure_num:
                score += 18.0
            elif family == "figure_walkthrough":
                score -= 6.0
            if target_panels:
                panel_overlap = len(target_panels.intersection(_extract_caption_panel_letters(text[:400])))
                score += 6.0 * float(panel_overlap)

        if is_citation_lookup:
            inline_refs = _extract_inline_reference_numbers(text, max_candidates=4)
            if inline_refs:
                score += 6.0
            query_tokens = set(_paper_guide_cue_tokens(q))
            shared_tokens = set(_paper_guide_cue_tokens(text[:320])).intersection(query_tokens)
            if shared_tokens:
                score += min(8.0, 2.0 * float(len(shared_tokens)))
            if _paper_guide_text_matches_requested_section(heading, "references"):
                score += 4.0

        if family == "abstract":
            if "abstract" in heading_norm or text.lstrip().lower().startswith("# abstract"):
                score += 100.0
            elif "introduction" in heading_norm:
                score += 8.0
            elif heading_norm:
                score -= 24.0
            if any(token in heading_norm for token in ("results", "discussion", "materials and methods", "methods", "appendix")):
                score -= 20.0
            if "here we introduce" in text_norm or "next generation technique" in text_norm:
                score += 2.0
        elif family == "method":
            heading_low = heading.lower()
            text_low = text.lower()
            if re.search(r"\bapr\b|adaptive pixel[- ]?reassignment", q, flags=re.IGNORECASE):
                if "adaptive pixel-reassignment" in heading_low or "adaptive pixel reassignment" in heading_low:
                    score += 12.0
                if any(token in text_low for token in ("phase correlation", "image registration", "radial variance transform", "rvt")):
                    score += 10.0
                if "microscope setup" in heading_low:
                    score -= 2.8
            if any(token in heading_norm for token in ("adaptive pixel reassignment", "algorithm", "analysis", "rvt")):
                score += 3.6
            if any(token in text_norm for token in ("phase correlation", "image registration", "radial variance transform", "rvt")):
                score += 4.0

        ranked.append((score, text))

    ranked.sort(key=lambda item: item[0], reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for _score, text in ranked:
        key = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:12]
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= keep_limit:
            break
    return out


def _filter_hits_for_paper_guide(
    hits_raw: list[dict],
    *,
    bound_source_path: str,
    bound_source_name: str,
) -> list[dict]:
    out: list[dict] = []
    for hit in hits_raw or []:
        if not isinstance(hit, dict):
            continue
        if _is_hit_from_bound_source(
            hit,
            bound_source_path=bound_source_path,
            bound_source_name=bound_source_name,
        ):
            out.append(hit)
    return out


def _paper_guide_hit_matches_requested_targets(hit: dict, *, prompt: str) -> bool:
    if not isinstance(hit, dict):
        return False
    meta = hit.get("meta", {}) or {}
    for cand in (
        meta.get("heading_path"),
        meta.get("top_heading"),
        hit.get("text"),
    ):
        if _paper_guide_text_matches_requested_targets(str(cand or ""), prompt=prompt):
            return True
    return False


def _paper_guide_has_requested_target_hits(hits_raw: list[dict], *, prompt: str) -> bool:
    if not _paper_guide_requested_heading_hints(prompt):
        return True
    return any(_paper_guide_hit_matches_requested_targets(hit, prompt=prompt) for hit in list(hits_raw or []))


def _paper_guide_targeted_box_excerpt_hits(
    *,
    md_path: Path,
    bound_source_path: str,
    prompt: str,
    db_dir: Path | str | None = None,
    limit: int = 4,
    resolve_support_slot_block=None,
) -> list[dict]:
    target_boxes = _paper_guide_requested_box_numbers(prompt)
    if not target_boxes:
        return []
    if not callable(resolve_support_slot_block):
        return []
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    query_tokens = set(_paper_guide_cue_tokens(prompt))
    ranked: list[tuple[float, dict]] = []
    for box_num in target_boxes:
        start_m = re.search(
            rf"<!--\s*box:start\s+id={int(box_num)}\s*-->",
            text,
            flags=re.IGNORECASE,
        )
        end_m = re.search(
            rf"<!--\s*box:end\s+id={int(box_num)}\s*-->",
            text,
            flags=re.IGNORECASE,
        )
        if not start_m or not end_m or end_m.start() <= start_m.end():
            continue
        body = str(text[start_m.end() : end_m.start()] or "").strip()
        if not body:
            continue
        chunks = [
            str(chunk or "").strip()
            for chunk in re.split(r"\n{2,}", body)
            if str(chunk or "").strip() and "<!--" not in str(chunk or "")
        ]
        for chunk in chunks:
            score = 36.0
            shared = set(_paper_guide_cue_tokens(chunk[:900])).intersection(query_tokens)
            if shared:
                score += min(18.0, 2.5 * float(len(shared)))
            if _paper_guide_box_header_number(chunk) == int(box_num):
                score -= 12.0
            if re.search(r"\btransform domain\b", chunk, flags=re.IGNORECASE):
                score += 8.0
            if re.search(r"M\s*\\ge\s*O\(K\s*\\log\(N/K\)\)", chunk):
                score += 12.0
            if re.search(r"\bHadamard\b", prompt, flags=re.IGNORECASE) and re.search(r"Hadamard", chunk, flags=re.IGNORECASE):
                score += 6.0
            support_meta = resolve_support_slot_block(
                source_path=bound_source_path,
                snippet=chunk,
                heading=f"Box {int(box_num)}",
                prompt_family="overview",
                claim_type="own_result",
                db_dir=db_dir,
            )
            meta = {
                "source_path": str(md_path),
                "heading_path": str(support_meta.get("heading_path") or f"Box {int(box_num)}").strip(),
                "block_id": str(support_meta.get("block_id") or "").strip(),
                "anchor_id": str(support_meta.get("anchor_id") or "").strip(),
                "kind": "paper_guide_box_excerpt",
                "paper_guide_targeted_block": True,
                "paper_guide_target_scope": "box_excerpt",
            }
            ranked.append(
                (
                    score,
                    {
                        "text": chunk[:1200],
                        "score": score,
                        "meta": meta,
                    },
                )
            )
    ranked.sort(key=lambda item: item[0], reverse=True)
    out: list[dict] = []
    seen: set[str] = set()
    for _score, hit in ranked:
        key = hashlib.sha1(
            (
                str((hit.get("meta") or {}).get("block_id") or "")
                + "\n"
                + str(hit.get("text") or "")
            ).encode("utf-8", "ignore")
        ).hexdigest()[:16]
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
        if len(out) >= max(1, int(limit or 4)):
            break
    return out


def _paper_guide_targeted_source_block_hits(
    *,
    bound_source_path: str,
    prompt: str,
    db_dir: Path | str | None = None,
    limit: int = 4,
    citation_lookup_query_tokens=None,
    citation_lookup_signal_score=None,
    resolve_support_slot_block=None,
) -> list[dict]:
    md_path = _resolve_paper_guide_md_path(
        bound_source_path,
        db_dir=db_dir,
    )
    if md_path is None:
        return []
    q = str(prompt or "").strip()
    if not q:
        return []
    # Index-first targeted hits: for figure panel requests, use figure_index.json caption clauses
    # rather than scanning generic blocks. This improves stability and panel-specific locate.
    try:
        family0 = str(_paper_guide_prompt_family(q) or "").strip().lower()
    except Exception:
        family0 = ""
    try:
        target_fig0 = int(_requested_figure_number(q, []) or 0)
    except Exception:
        target_fig0 = 0
    target_panels0 = _extract_caption_panel_letters(q)
    if family0 == "figure_walkthrough" and target_fig0 > 0 and target_panels0:
        try:
            from kb.paper_guide_structured_index_runtime import load_paper_guide_figure_index
            from kb.paper_guide.grounder import _extract_caption_fragment_for_letters

            for row in load_paper_guide_figure_index(md_path):
                try:
                    fig_no = int(row.get("paper_figure_number") or 0)
                except Exception:
                    fig_no = 0
                if fig_no != int(target_fig0):
                    continue
                caption = str(row.get("caption") or "").strip()
                frag = _extract_caption_fragment_for_letters(caption, set(target_panels0))
                if not frag:
                    continue
                caption_block_id = str(row.get("caption_block_id") or "").strip()
                caption_anchor_id = str(row.get("caption_anchor_id") or "").strip()
                # Use caption block/anchor if available; otherwise fall back to figure placeholder.
                if not caption_block_id:
                    caption_block_id = str(row.get("figure_block_id") or "").strip()
                if not caption_anchor_id:
                    caption_anchor_id = str(row.get("anchor_id") or "").strip()
                meta = {
                    "source_path": str(md_path),
                    "source_sha1": compute_file_sha1(md_path),
                    "top_heading": "",
                    "heading_path": str(row.get("heading_path") or "").strip(),
                    "block_id": caption_block_id,
                    "anchor_id": caption_anchor_id,
                    "kind": "paper_guide_figure_index_clause",
                    "paper_guide_targeted_block": True,
                    "paper_guide_target_scope": "figure_panel",
                    "figure_number": int(target_fig0),
                    "panel_letters": sorted(target_panels0),
                }
                return [
                    {
                        "text": str(frag).strip()[:1200],
                        "score": 999.0,
                        "meta": meta,
                    }
                ]
        except Exception:
            pass
    box_excerpt_hits = _paper_guide_targeted_box_excerpt_hits(
        md_path=md_path,
        bound_source_path=bound_source_path,
        prompt=q,
        db_dir=db_dir,
        limit=max(2, min(int(limit or 4), 6)),
        resolve_support_slot_block=resolve_support_slot_block,
    )
    ranked: list[tuple[float, dict]] = [
        (float(hit.get("score") or 0.0), hit) for hit in box_excerpt_hits if isinstance(hit, dict)
    ]
    explicit_hints = _paper_guide_requested_heading_hints(q)
    family = str(_paper_guide_prompt_family(q) or "").strip().lower()
    is_citation_lookup = bool(family == "citation_lookup")
    explicit_ref_list_request = bool(
        re.search(r"(?i)\b(?:reference\s+list|works?\s+cited|bibliography)\b", q)
    )
    require_target_match = bool(explicit_hints) and (not is_citation_lookup)
    if is_citation_lookup and callable(citation_lookup_query_tokens):
        query_tokens = set(citation_lookup_query_tokens(q))
    else:
        query_tokens = _paper_guide_seed_query_tokens_for_targeted_scan(
            prompt=q,
            family=family,
            bound_source_path=bound_source_path,
        )
    blocks = list(load_source_blocks(md_path))
    target_boxes = set(_paper_guide_requested_box_numbers(q))
    box_context_indices: set[int] = set()
    if target_boxes:
        active_box = 0
        window_remaining = 0
        for idx, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            text = str(block.get("raw_text") or block.get("text") or "").strip()
            header_box = _paper_guide_box_header_number(text)
            if header_box > 0:
                if header_box in target_boxes:
                    active_box = int(header_box)
                    window_remaining = 18
                    box_context_indices.add(idx)
                    continue
                active_box = 0
                window_remaining = 0
            if active_box > 0 and window_remaining > 0:
                box_context_indices.add(idx)
                window_remaining -= 1
    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        text = str(block.get("raw_text") or block.get("text") or "").strip()
        heading = str(block.get("heading_path") or "").strip()
        if not text:
            continue
        combined = "\n".join(part for part in (heading, text[:800]) if part)
        box_context_match = idx in box_context_indices
        target_match = box_context_match or _paper_guide_text_matches_requested_targets(combined, prompt=q)
        if require_target_match and (not target_match):
            continue
        score = 0.0
        if target_match:
            score += 32.0
        if box_context_match:
            score += 14.0
        heading_low = heading.lower()
        tokens = set(_paper_guide_cue_tokens(combined))
        shared = tokens.intersection(query_tokens)
        if shared:
            score += min(14.0, 2.0 * float(len(shared)))
        q_low = q.lower()
        text_low = text.lower()
        if family == "method":
            if any(token in heading_low for token in ("method", "methods", "materials and methods", "methodology", "implementation", "algorithm", "analysis")):
                score += 7.0
            if any(token in text_low for token in ("phase correlation", "image registration", "algorithm", "workflow", "rvt")):
                score += 3.5
        elif family == "reproduce":
            if any(token in heading_low for token in ("materials and methods", "methods", "setup", "data acquisition", "protocol", "implementation")):
                score += 6.0
            if any(token in text_low for token in ("camera", "laser", "dwell time", "exposure", "acquisition", "hardware")):
                score += 3.2
        elif family == "overview":
            if any(token in heading_low for token in ("abstract", "introduction", "results", "discussion", "conclusion")):
                score += 4.5
        elif family == "equation":
            if any(token in heading_low for token in ("equation", "formula", "method", "background")):
                score += 5.0
            if any(token in text for token in ("\\tag{", "$$", "\\[", "\\]", "where ", "denotes", "represents")):
                score += 3.8
        elif family == "figure_walkthrough":
            if any(token in heading_low for token in ("figure", "caption", "panel", "results")):
                score += 5.0
            if any(token in text_low for token in ("figure", "fig.", "panel", "caption")):
                score += 2.5
        if "hadamard" in q_low and "hadamard" in text_low:
            score += 12.0
        if "richardson" in q_low and "richardson" in text_low:
            score += 8.0
        if "lucy" in q_low and "lucy" in text_low:
            score += 8.0
        if is_citation_lookup:
            inline_refs = _extract_inline_reference_numbers(text, max_candidates=4)
            if inline_refs:
                score += 8.0
            if callable(citation_lookup_signal_score):
                score += citation_lookup_signal_score(
                    prompt=q,
                    heading=heading,
                    text=text,
                    inline_refs=inline_refs,
                    explicit_ref_list_request=explicit_ref_list_request,
                )
        if score <= 0.0:
            continue
        meta = {
            "source_path": str(md_path),
            "heading_path": heading,
            "block_id": str(block.get("block_id") or "").strip(),
            "anchor_id": str(block.get("anchor_id") or "").strip(),
            "kind": str(block.get("kind") or "").strip(),
            "paper_guide_targeted_block": True,
        }
        if box_context_match:
            meta["paper_guide_target_scope"] = "box_context"
        ranked.append(
            (
                score,
                {
                    "text": text[:1200],
                    "score": score,
                    "meta": meta,
                },
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    out: list[dict] = []
    seen: set[str] = set()
    for _score, hit in ranked:
        key = hashlib.sha1(
            (
                str((hit.get("meta") or {}).get("heading_path") or "")
                + "\n"
                + str(hit.get("text") or "")
            ).encode("utf-8", "ignore")
        ).hexdigest()[:16]
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
        if len(out) >= max(1, int(limit or 4)):
            break
    return out


def _paper_guide_should_force_rescue(
    *,
    scoped_hits: list[dict],
    prompt: str,
    prompt_family: str = "",
) -> bool:
    snapshot = _paper_guide_retrieval_confidence_snapshot(
        scoped_hits=scoped_hits,
        prompt=prompt,
        prompt_family=prompt_family,
    )
    return bool(snapshot.get("force_rescue"))


def _paper_guide_retrieval_confidence_snapshot(
    *,
    scoped_hits: list[dict],
    prompt: str,
    prompt_family: str = "",
) -> dict[str, object]:
    hits = [hit for hit in list(scoped_hits or []) if isinstance(hit, dict)]
    q = str(prompt or "").strip()
    family = str(prompt_family or _paper_guide_prompt_family(q)).strip().lower()
    method_exact_support = bool(
        family == "method" and _paper_guide_prompt_requests_exact_method_support(q)
    )
    explicit_targeting = bool(
        _paper_guide_requested_heading_hints(q)
        or _paper_guide_requested_box_numbers(q)
        or (_extract_figure_number(q) > 0)
        or method_exact_support
    )
    has_requested_target_hits = True
    if explicit_targeting:
        has_requested_target_hits = _paper_guide_has_requested_target_hits(hits, prompt=q)

    explicit_ref_list_request = bool(
        re.search(r"(?i)\b(?:reference\s+list|works?\s+cited|bibliography)\b", q)
    )
    query_tokens = _paper_guide_seed_query_tokens_for_targeted_scan(
        prompt=q,
        family=family,
        bound_source_path="",
    )
    strong_signal = False
    non_reference_signal = False
    targeted_hit_count = 0
    fallback_hit_count = 0
    reference_like_hit_count = 0
    max_overlap = 0
    max_score = 0.0
    for hit in hits:
        meta = hit.get("meta", {}) or {}
        try:
            max_score = max(max_score, float(hit.get("score") or 0.0))
        except Exception:
            max_score = max_score
        is_targeted = bool(meta.get("paper_guide_targeted_block"))
        is_fallback = bool(meta.get("paper_guide_fallback"))
        if is_targeted:
            targeted_hit_count += 1
        if is_fallback:
            fallback_hit_count += 1
        if is_targeted or is_fallback:
            strong_signal = True
            non_reference_signal = True
            continue
        heading = str(meta.get("heading_path") or meta.get("top_heading") or "").strip()
        text = str(hit.get("text") or "").strip()
        if not text:
            continue
        looks_reference = bool(_PAPER_GUIDE_REF_SECTION_RE.search(heading) or _looks_like_reference_list_snippet_local(text))
        if looks_reference:
            reference_like_hit_count += 1
            if family == "citation_lookup" or explicit_ref_list_request:
                non_reference_signal = True
            else:
                continue
        else:
            non_reference_signal = True
        overlap = 0
        if query_tokens:
            overlap = len(set(_paper_guide_cue_tokens("\n".join(part for part in (heading, text[:1200]) if part))).intersection(query_tokens))
        if overlap > max_overlap:
            max_overlap = overlap
        if overlap >= 2:
            strong_signal = True
        elif overlap >= 1 and len(text) >= 120:
            strong_signal = True

    force_rescue = False
    force_rescue_reason = ""
    if not hits:
        force_rescue = True
        force_rescue_reason = "empty_hits"
    elif explicit_targeting and (not has_requested_target_hits):
        force_rescue = True
        force_rescue_reason = "target_miss"
    elif (family != "citation_lookup") and (not explicit_ref_list_request) and (not non_reference_signal):
        force_rescue = True
        force_rescue_reason = "reference_only_hits"
    elif not strong_signal:
        force_rescue = True
        force_rescue_reason = "weak_signal"
    elif family in {"citation_lookup", "method", "figure_walkthrough", "reproduce"}:
        if (targeted_hit_count + fallback_hit_count) <= 0:
            force_rescue = True
            force_rescue_reason = "strict_family_without_targeted_support"

    low_confidence = False
    low_confidence_reason = ""
    if force_rescue:
        low_confidence = True
        low_confidence_reason = force_rescue_reason or "force_rescue"
    else:
        strict_family = family in {"citation_lookup", "method", "reproduce", "equation", "figure_walkthrough"}
        broad_family = family in {"overview", "compare", "strength_limits", "abstract"}
        has_strong_targeted_hit = (targeted_hit_count + fallback_hit_count) > 0
        if (not has_strong_targeted_hit) and strict_family:
            if max_overlap <= 1 and max_score < 11.0:
                low_confidence = True
                low_confidence_reason = "strict_family_weak_overlap"
            elif len(hits) < 3 and max_overlap < 2:
                low_confidence = True
                low_confidence_reason = "strict_family_sparse_hits"
        elif (not has_strong_targeted_hit) and broad_family:
            if (max_overlap <= 1) and (max_score < 9.5) and (len(hits) < 8):
                low_confidence = True
                low_confidence_reason = "broad_family_weak_overlap"

    return {
        "hit_count": len(hits),
        "family": family,
        "explicit_targeting": explicit_targeting,
        "explicit_ref_list_request": explicit_ref_list_request,
        "has_requested_target_hits": has_requested_target_hits,
        "targeted_hit_count": targeted_hit_count,
        "fallback_hit_count": fallback_hit_count,
        "reference_like_hit_count": reference_like_hit_count,
        "non_reference_signal": non_reference_signal,
        "strong_signal": strong_signal,
        "max_overlap": max_overlap,
        "max_score": max_score,
        "force_rescue": force_rescue,
        "force_rescue_reason": force_rescue_reason,
        "low_confidence": low_confidence,
        "low_confidence_reason": low_confidence_reason,
    }


def _paper_guide_fallback_deepread_hits(
    *,
    bound_source_path: str,
    bound_source_name: str,
    query: str,
    prompt: str = "",
    prompt_family: str = "",
    top_k: int,
    db_dir: Path | str | None = None,
    citation_lookup_query_tokens=None,
    citation_lookup_signal_score=None,
    resolve_support_slot_block=None,
) -> list[dict]:
    md_path = _resolve_paper_guide_md_path(
        bound_source_path,
        db_dir=db_dir,
    )
    if md_path is None:
        return []

    q = str(query or "").strip()
    if not q:
        q = f"{bound_source_name or md_path.stem} contribution method experiment limitation"
    prompt_raw = str(prompt or "").strip()
    prompt_effective = prompt_raw or q
    family_eff = str(prompt_family or _paper_guide_prompt_family(prompt_effective or q)).strip().lower()
    # If retrieval provided a translated/English query, prefer it for targeted scans when
    # the original prompt is CJK-heavy. This improves recall for English papers.
    try:
        if q and prompt_raw and (re.search(r"[A-Za-z]", prompt_raw) is None) and (re.search(r"[A-Za-z]", q) is not None):
            prompt_effective = q
    except Exception:
        prompt_effective = prompt_effective
    prompt_candidates: list[str] = []
    for candidate in (prompt_raw, q):
        cand = str(candidate or "").strip()
        if not cand:
            continue
        if any(normalize_match_text(cand) == normalize_match_text(existing) for existing in prompt_candidates):
            continue
        prompt_candidates.append(cand)
    has_lexical_candidate = any(bool(_paper_guide_cue_tokens(cand)) for cand in prompt_candidates)
    if not has_lexical_candidate:
        augmented_candidate = _augment_paper_guide_retrieval_prompt(
            prompt_effective or q,
            family=family_eff,
        )
        cand_aug = str(augmented_candidate or "").strip()
        if cand_aug and not any(normalize_match_text(cand_aug) == normalize_match_text(existing) for existing in prompt_candidates):
            prompt_candidates.append(cand_aug)
            prompt_effective = cand_aug
    if not prompt_candidates:
        seed = f"{bound_source_name or md_path.stem} method result figure equation reference discussion"
        prompt_candidates.append(seed)
        prompt_effective = seed

    targeted_ranked: list[tuple[float, dict]] = []
    targeted_seen: set[str] = set()
    for candidate in prompt_candidates:
        targeted_hits = _paper_guide_targeted_source_block_hits(
            bound_source_path=bound_source_path,
            prompt=candidate,
            db_dir=db_dir,
            limit=max(2, min(int(top_k or 4), 4)),
            citation_lookup_query_tokens=citation_lookup_query_tokens,
            citation_lookup_signal_score=citation_lookup_signal_score,
            resolve_support_slot_block=resolve_support_slot_block,
        )
        for hit in targeted_hits:
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta", {}) or {}
            key = hashlib.sha1(
                (
                    str(meta.get("block_id") or "")
                    + "\n"
                    + str(meta.get("heading_path") or "")
                    + "\n"
                    + str(hit.get("text") or "")
                ).encode("utf-8", "ignore")
            ).hexdigest()[:16]
            if key in targeted_seen:
                continue
            targeted_seen.add(key)
            try:
                score = float(hit.get("score") or 0.0)
            except Exception:
                score = 0.0
            targeted_ranked.append((score, dict(hit)))
    if targeted_ranked:
        targeted_ranked.sort(key=lambda item: item[0], reverse=True)
        return [hit for _score, hit in targeted_ranked[: max(1, int(top_k or 4))]]
    deep_hits = _deep_read_md_for_context(
        md_path,
        q,
        max_snippets=max(2, min(int(top_k or 4), 4)),
        snippet_chars=1200,
    )
    selected_texts = _select_paper_guide_deepread_extras(
        deep_hits,
        prompt=prompt_effective,
        prompt_family=prompt_family,
        limit=max(1, min(int(top_k or 4), 4)),
    )
    if not selected_texts:
        return []
    selected_keys = {
        hashlib.sha1(str(text or "").encode("utf-8", "ignore")).hexdigest()[:12]
        for text in selected_texts
        if str(text or "").strip()
    }
    out: list[dict] = []
    block_rows: list[dict] = []
    try:
        for block in list(load_source_blocks(md_path)):
            if not isinstance(block, dict):
                continue
            block_text = str(block.get("raw_text") or block.get("text") or "").strip()
            if not block_text:
                continue
            block_rows.append(
                {
                    "block": block,
                    "text": block_text,
                    "norm_text": normalize_match_text(block_text[:2200]),
                    "tokens": _paper_guide_cue_tokens(block_text[:1400]),
                }
            )
    except Exception:
        block_rows = []
    for idx, h in enumerate(deep_hits, start=1):
        if not isinstance(h, dict):
            continue
        text = str(h.get("text") or "").strip()
        key = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:12] if text else ""
        if selected_keys and key not in selected_keys:
            continue
        rec = dict(h)
        meta = rec.get("meta", {}) or {}
        if not isinstance(meta, dict):
            meta = {}
        meta["source_path"] = str(md_path)
        meta["paper_guide_fallback"] = True
        rec["meta"] = meta
        try:
            score0 = float(rec.get("score") or 0.0)
        except Exception:
            score0 = 0.0
        if score0 <= 0.0:
            rec["score"] = max(0.01, 1.0 - (idx - 1) * 0.1)
        best_row = _paper_guide_best_block_for_fallback_hit(
            hit_text=text,
            block_rows=block_rows,
        )
        if best_row:
            block_best = best_row.get("block") or {}
            block_id = str(block_best.get("block_id") or "").strip()
            anchor_id = str(block_best.get("anchor_id") or "").strip()
            heading_best = str(block_best.get("heading_path") or "").strip()
            if heading_best and (not str(meta.get("heading_path") or "").strip()):
                meta["heading_path"] = heading_best
            if block_id:
                meta["block_id"] = block_id
            if anchor_id:
                meta["anchor_id"] = anchor_id
            if block_id or anchor_id:
                meta["paper_guide_rebound_block"] = True
        out.append(rec)
    return out


def _paper_guide_citation_lookup_fragments(text: str) -> list[str]:
    src = str(text or "").strip()
    if not src:
        return []
    # Be conservative here: normalize_inline_markdown is helpful for matching, but it can
    # mangle inline-math citation superscripts like `$^{[4]}$` into `$^{[4]}` (dropping the
    # closing `$`). For citation lookup we want fragments that remain verbatim-matchable.
    norm = re.sub(r"\s+", " ", src.replace("\r\n", "\n").replace("\r", "\n")).strip()
    chunks: list[str] = []
    seen: set[str] = set()
    for raw_line in re.split(r"\n+", norm):
        line = str(raw_line or "").strip(" >-*")
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])|(?<=\])\s+(?=[A-Z])", line)
        for raw_part in parts:
            part = str(raw_part or "").strip()
            if len(part) < 20:
                continue
            key = normalize_match_text(part)
            if key in seen:
                continue
            seen.add(key)
            chunks.append(part)
    return chunks or [norm]


def _paper_guide_citation_lookup_query_tokens(prompt: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tok in _paper_guide_cue_tokens(prompt):
        if tok in _PAPER_GUIDE_CITATION_LOOKUP_QUERY_STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    # Lightweight semantic expansion for citation lookup prompts:
    # converted papers often use "compressive sensing" while user asks "compressive sampling" (or vice versa).
    alias_map: dict[str, tuple[str, ...]] = {
        "sampling": ("sensing",),
        "sensing": ("sampling",),
    }
    for tok in list(out):
        for alt in alias_map.get(tok, ()):
            if (not alt) or (alt in _PAPER_GUIDE_CITATION_LOOKUP_QUERY_STOPWORDS):
                continue
            if alt in seen:
                continue
            seen.add(alt)
            out.append(alt)
    return out


def _focus_citation_fragment_for_refs(text: str, *, target_refs: list[int], prompt: str = "") -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    refs_target = {int(n) for n in list(target_refs or []) if int(n) > 0}
    if not refs_target:
        return src
    q_tokens = set(_paper_guide_citation_lookup_query_tokens(prompt))
    sentence_like: list[str] = []
    for base in re.split(r"(?<=[.!?])\s+|\n+", src):
        s = str(base or "").strip(" >-*")
        if not s:
            continue
        sentence_like.append(s)
        # Long lines in converted markdown often hold multiple citation clauses;
        # split once more to isolate the clause around the target reference.
        if len(s) >= 120:
            for sub in re.split(r"(?<=[,;:])\s+", s):
                ss = str(sub or "").strip(" >-*")
                if ss and (ss != s):
                    sentence_like.append(ss)

    best = ""
    best_score = float("-inf")
    seen: set[str] = set()
    for cand in sentence_like:
        key = normalize_match_text(cand)
        if (not key) or (key in seen):
            continue
        seen.add(key)
        refs = _extract_inline_reference_numbers(cand, max_candidates=8)
        if not refs:
            continue
        refs_set = {int(n) for n in refs if int(n) > 0}
        overlap = refs_set.intersection(refs_target)
        if not overlap:
            continue
        extra = refs_set.difference(refs_target)
        score = 10.0
        score += 4.0 * float(len(overlap))
        score -= 5.0 * float(len(extra))
        if len(cand) < 28:
            score -= 6.0
        score -= min(4.0, 0.006 * float(len(cand)))
        if q_tokens:
            tok_overlap = len(set(_paper_guide_citation_lookup_query_tokens(cand)).intersection(q_tokens))
            score += min(4.0, 1.2 * float(tok_overlap))
        if score > best_score:
            best_score = score
            best = cand
    return best or src


def _paper_guide_prompt_prefers_single_reference(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    if re.search(r"(?i)\b(?:which|what)\s+references?\b", q):
        if re.search(r"(?i)\b(?:which|what)\s+references\b", q):
            return False
        return True
    if re.search(r"(?i)\b(?:which|what)\s+(?:in-?paper\s+)?citation\b", q):
        return True
    if re.search(r"(?i)\bcite(?:d|s)?\b[^\n]{0,60}\bfor\b", q):
        return True
    return False


def _select_primary_refs_for_prompt(*, fragment: str, prompt: str, refs: list[int], max_keep: int = 4) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for raw in list(refs or []):
        try:
            n = int(raw)
        except Exception:
            continue
        if n <= 0 or n in seen:
            continue
        seen.add(n)
        ordered.append(n)
    if not ordered:
        return []
    keep = max(1, int(max_keep or 4))
    if len(ordered) <= 1:
        return ordered[:keep]
    if not _paper_guide_prompt_prefers_single_reference(prompt):
        return ordered[:keep]
    q_tokens = set(_paper_guide_citation_lookup_query_tokens(prompt))
    best_ref = ordered[0]
    best_score = float("-inf")
    for ref_num in ordered:
        focused = _focus_citation_fragment_for_refs(
            fragment,
            target_refs=[int(ref_num)],
            prompt=prompt,
        )
        focused_refs = _extract_inline_reference_numbers(focused, max_candidates=6)
        score = _paper_guide_citation_lookup_signal_score(
            prompt=prompt,
            heading="",
            text=focused,
            inline_refs=focused_refs,
            explicit_ref_list_request=False,
        )
        if q_tokens:
            overlap = len(set(_paper_guide_citation_lookup_query_tokens(focused)).intersection(q_tokens))
            score += 2.4 * float(overlap)
        if int(ref_num) in focused_refs:
            score += 6.0
        score -= min(2.0, 0.004 * float(len(focused)))
        if score > best_score:
            best_score = score
            best_ref = int(ref_num)
    return [best_ref]


def _paper_guide_citation_lookup_signal_score(
    *,
    prompt: str,
    heading: str,
    text: str,
    inline_refs: list[int] | None = None,
    explicit_ref_list_request: bool = False,
) -> float:
    heading_low = str(heading or "").strip().lower()
    # heading_path often prefixes the paper title, which can contain very generic query tokens
    # (e.g., "single-pixel imaging") and would otherwise swamp the scoring.
    heading_leaf_low = heading_low
    if " / " in heading_leaf_low:
        heading_leaf_low = heading_leaf_low.split(" / ", 1)[1].strip()
    text_low = str(text or "").strip().lower()
    score = 0.0
    topic_tokens = _paper_guide_citation_lookup_query_tokens(prompt)
    shared_topics_text = [tok for tok in topic_tokens if tok in text_low]
    shared_topics_heading = [tok for tok in topic_tokens if (tok not in shared_topics_text) and (tok in heading_leaf_low)]
    if shared_topics_text:
        score += min(12.0, 3.2 * float(len(shared_topics_text)))
    if shared_topics_heading:
        # Heading-only lexical overlap is weaker evidence than direct text overlap.
        score += min(2.4, 0.8 * float(len(shared_topics_heading)))

    # Heuristic: for citation-lookup questions phrased like "cite for X", strongly prefer fragments
    # that actually contain X (so we don't pick a nearby sentence with many refs but missing the target term).
    focus_tokens: list[str] = []
    m = re.search(r"(?i)\bcite(?:d|s)?\b[^\n]{0,80}\bfor\b\s+([^?.!,;\n]{1,120})", prompt)
    if m:
        raw_focus = str(m.group(1) or "").strip().lower()
        generic_focus_tokens = {"single", "pixel", "imaging", "authors", "method", "methods", "approach"}
        cand = [
            tok
            for tok in _paper_guide_citation_lookup_query_tokens(raw_focus)
            if tok and (tok not in generic_focus_tokens)
        ]
        for tok in cand:
            if tok not in focus_tokens:
                focus_tokens.append(tok)
        if not focus_tokens:
            for tok in _paper_guide_citation_lookup_query_tokens(raw_focus):
                if tok and tok not in focus_tokens:
                    focus_tokens.append(tok)
        focus_tokens = focus_tokens[:4]
    if focus_tokens:
        focus_text_overlap = [tok for tok in focus_tokens if tok in text_low]
        focus_heading_overlap = [tok for tok in focus_tokens if (tok not in focus_text_overlap) and (tok in heading_leaf_low)]
        if focus_text_overlap:
            score += min(14.0, 6.0 * float(len(focus_text_overlap)))
        elif focus_heading_overlap:
            score += min(4.0, 2.0 * float(len(focus_heading_overlap)))
        elif inline_refs:
            score -= 10.0
    has_inline_refs = bool(inline_refs)
    # Guardrail: "citation-like" sentences with many refs are common (e.g., listing optimization priors).
    # If the fragment does not match the question's topical tokens at all, down-rank it even if it contains refs.
    shared_topics = list(shared_topics_text) + list(shared_topics_heading)
    if has_inline_refs and (not shared_topics_text):
        score -= 6.0
    if has_inline_refs and len(shared_topics_text) >= 2:
        score += 10.0
    elif has_inline_refs and shared_topics_text:
        score += 4.0
    if _PAPER_GUIDE_CITATION_LOOKUP_ATTRIBUTION_RE.search(text):
        score += 8.0 if has_inline_refs else 3.0
    is_reference_section = _paper_guide_text_matches_requested_section(heading, "references")
    looks_like_reference_entry = bool(re.match(r"(?i)^\s*(?:references\b|\[\d{1,4}\])", str(text or "").strip()))
    if is_reference_section or looks_like_reference_entry:
        score += 2.0 if explicit_ref_list_request else -5.0
    return score


def _extract_paper_guide_local_citation_lookup_refs(text: str, *, prompt: str, max_candidates: int = 6) -> list[int]:
    frag = normalize_inline_markdown(str(text or "").strip())
    if not frag:
        return []
    spec_pat = r"(\d{1,4}(?:\s*(?:[\-–—−,])\s*\d{1,4})*)"
    for tok in _paper_guide_citation_lookup_query_tokens(prompt):
        if not tok:
            continue
        patterns = (
            rf"(?i)\b{re.escape(tok)}\b[^\[\]\n]{{0,32}}\[({spec_pat})\](?![A-Za-z])",
            rf"(?i)\b{re.escape(tok)}\b[^\n]{{0,32}}\$\^\{{({spec_pat})\}}\$",
            rf"(?i)\b{re.escape(tok)}\b[^\n]{{0,32}}\^\{{({spec_pat})\}}",
        )
        for pattern in patterns:
            m = re.search(pattern, frag)
            if not m:
                continue
            out: list[int] = []
            for part in re.split(r"\s*[,，]\s*|\s*[–—−-]\s*", str(m.group(1) or "").strip()):
                try:
                    n = int(part)
                except Exception:
                    continue
                if n > 0:
                    out.append(n)
            if out:
                return out[: max(1, int(max_candidates or 6))]
    return []


def _select_paper_guide_local_citation_lookup_refs(
    text: str,
    *,
    prompt: str,
    max_candidates: int = 6,
) -> list[int]:
    base_refs = _extract_paper_guide_local_citation_lookup_refs(
        text,
        prompt=prompt,
        max_candidates=max_candidates,
    )
    raw = str(text or "").strip()
    frag = normalize_inline_markdown(raw)
    if not frag:
        return base_refs
    try:
        limit = max(1, int(max_candidates))
    except Exception:
        limit = 6
    scored_rows: list[tuple[float, list[int]]] = []
    if base_refs:
        scored_rows.append((2.5 + (0.3 * float(len(base_refs))), list(base_refs[:limit])))
    pseudo_block = {
        "block_id": "blk_local_citation_lookup",
        "anchor_id": "a_local_citation_lookup",
        "heading_path": "",
        "raw_text": raw,
        "text": frag,
        "kind": "paragraph",
        "order_index": 1,
    }
    try:
        atoms = _build_paper_guide_evidence_atoms([pseudo_block], max_atoms_per_block=10)
    except Exception:
        atoms = []
    query_tokens = set(_paper_guide_citation_lookup_query_tokens(prompt))
    for atom in atoms:
        if str(atom.get("atom_kind") or "").strip().lower() != "ref_span":
            continue
        refs = [int(n) for n in list(atom.get("inline_refs") or []) if int(n) > 0]
        if not refs:
            continue
        atom_text = str(atom.get("text") or "").strip()
        atom_tokens = set(_paper_guide_cue_tokens(atom_text))
        shared = query_tokens.intersection(atom_tokens)
        score = min(4.0, 0.85 * float(len(shared)))
        if len(refs) == 1:
            score += 1.8
        elif len(refs) > 1:
            score -= min(1.4, 0.45 * float(len(refs) - 1))
        if re.search(r"(?i)\b(?:reported|report(?:ed)? soon after|seminal paper|introduced|attributed|by)\b", atom_text):
            score += 1.6
        if re.search(r"(?i)\bet al\.", atom_text):
            score += 0.9
        if re.search(r"(?i)\b(?:duarte|richardson|lucy|hadamard|fourier|wavelet)\b", atom_text):
            score += 1.5
        scored_rows.append((score, refs[:limit]))
    if not scored_rows:
        return []
    scored_rows.sort(key=lambda item: (float(item[0]), 1 if len(item[1]) == 1 else 0), reverse=True)
    return list(scored_rows[0][1])[:limit]


def _select_paper_guide_raw_target_hits(
    *,
    hits_raw: list[dict],
    prompt: str,
    top_n: int,
    answer_hit_score=None,
) -> list[dict]:
    family = _paper_guide_prompt_family(prompt)
    explicit_ref_list_request = bool(
        re.search(r"(?i)\b(?:reference\s+list|works?\s+cited|bibliography)\b", str(prompt or ""))
    )
    requested_hints = _paper_guide_requested_heading_hints(prompt)
    method_exact_support = bool(
        family == "method" and _paper_guide_prompt_requests_exact_method_support(str(prompt or ""))
    )
    if (not requested_hints) and (family != "citation_lookup") and (not method_exact_support):
        return []
    try:
        limit = max(1, int(top_n))
    except Exception:
        limit = 1
    ranked: list[tuple[float, dict]] = []
    seen: set[tuple[str, str, str]] = set()
    score_fn = answer_hit_score or (lambda hit, *, prompt: float(hit.get("score") or 0.0))
    for hit in hits_raw or []:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        text = str(hit.get("text") or "").strip()
        heading = str(meta.get("heading_path") or "").strip()
        signal_score = 0.0
        if family == "citation_lookup":
            inline_refs = _extract_inline_reference_numbers(text, max_candidates=6)
            signal_score = _paper_guide_citation_lookup_signal_score(
                prompt=prompt,
                heading=heading,
                text=text,
                inline_refs=inline_refs,
                explicit_ref_list_request=explicit_ref_list_request,
            )
            if (signal_score <= 0.0) and (not bool(meta.get("paper_guide_targeted_block"))):
                continue
        elif not (
            _paper_guide_hit_matches_requested_targets(hit, prompt=prompt)
            or bool(meta.get("paper_guide_targeted_block"))
        ):
            continue
        src = str(meta.get("source_path") or "").strip()
        block_id = str(meta.get("block_id") or "").strip()
        key = (
            src,
            block_id or normalize_match_text(heading),
            hashlib.sha1(text[:320].encode("utf-8", "ignore")).hexdigest()[:12],
        )
        if key in seen:
            continue
        seen.add(key)
        base_score = float(score_fn(hit, prompt=prompt))
        rank_score = base_score
        if family == "citation_lookup":
            # For citation lookup, prioritize semantic/attribution signal over raw retrieval rank.
            # This avoids selecting nearby but off-topic citation sentences.
            rank_score = (12.0 * float(signal_score)) + (0.25 * base_score)
            if bool(meta.get("paper_guide_targeted_block")):
                rank_score += 0.8
        ranked.append((rank_score, hit))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [dict(hit) for _score, hit in ranked[:limit]]


def _build_paper_guide_direct_citation_lookup_answer(
    *,
    prompt: str,
    source_path: str,
    answer_hits: list[dict] | None,
    special_focus_block: str = "",
    db_dir: Path | None,
    extract_special_focus_excerpt=None,
    reference_entry_lookup=None,
) -> str:
    q = str(prompt or "").strip()
    if _paper_guide_prompt_family(q) != "citation_lookup":
        return ""
    src = str(source_path or "").strip()
    explicit_ref_list_request = bool(
        re.search(r"(?i)\b(?:reference\s+list|works?\s+cited|bibliography)\b", q)
    )
    candidates: list[dict[str, str]] = []
    for hit in list(answer_hits or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        text = str(hit.get("text") or "").strip()
        if not text:
            continue
        candidates.append(
            {
                "text": text,
                "heading": str(meta.get("heading_path") or meta.get("top_heading") or "").strip(),
            }
        )
    extract_focus = extract_special_focus_excerpt or (lambda block: "")
    focus_excerpt = str(extract_focus(special_focus_block) or "").strip()
    if focus_excerpt:
        candidates.append({"text": focus_excerpt, "heading": ""})
    targeted_candidates: list[dict[str, str]] = []
    if src:
        # Fallback pool: when grouped answer hits are off-target, add a small targeted scan.
        try:
            targeted_hits = _paper_guide_targeted_source_block_hits(
                bound_source_path=src,
                prompt=q,
                db_dir=db_dir,
                limit=6,
                citation_lookup_query_tokens=_paper_guide_citation_lookup_query_tokens,
                citation_lookup_signal_score=_paper_guide_citation_lookup_signal_score,
                resolve_support_slot_block=lambda **_kwargs: {},
            )
        except Exception:
            targeted_hits = []
        for hit in list(targeted_hits or []):
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta", {}) or {}
            text = str(hit.get("text") or "").strip()
            if not text:
                continue
            targeted_candidates.append(
                {
                    "text": text,
                    "heading": str(meta.get("heading_path") or "").strip(),
                }
            )

    focus_query_tokens = {
        tok
        for tok in _paper_guide_citation_lookup_query_tokens(q)
        if len(str(tok or "").strip()) >= 6
    }
    if not focus_query_tokens:
        focus_query_tokens = set(_paper_guide_citation_lookup_query_tokens(q))

    def _best_from(candidate_rows: list[dict[str, str]]) -> dict:
        best_fragment_local = ""
        best_heading_local = ""
        best_score_local = float("-inf")
        best_refs_local: list[int] = []
        best_focus_overlap = 0
        for cand in list(candidate_rows or []):
            heading = str(cand.get("heading") or "").strip()
            for frag in _paper_guide_citation_lookup_fragments(str(cand.get("text") or "")):
                refs = _extract_inline_reference_numbers(frag, max_candidates=6)
                if not refs:
                    continue
                heading_norm = str(heading or "").strip()
                is_reference_like = (
                    _paper_guide_text_matches_requested_section(heading_norm, "references")
                    or bool(re.match(r"(?i)^\s*\[\d{1,4}\]", frag))
                )
                score = _paper_guide_citation_lookup_signal_score(
                    prompt=q,
                    heading=heading_norm,
                    text=frag,
                    inline_refs=refs,
                    explicit_ref_list_request=explicit_ref_list_request,
                )
                score += min(6.0, 2.0 * float(len(refs)))
                if explicit_ref_list_request:
                    if is_reference_like:
                        score += 6.0
                else:
                    score += 6.0 if not is_reference_like else -8.0
                focus_overlap = 0
                if focus_query_tokens:
                    low = str(frag or "").strip().lower()
                    focus_overlap = sum(1 for tok in focus_query_tokens if tok and tok in low)
                    score += min(8.0, 2.0 * float(focus_overlap))
                if score > best_score_local:
                    best_score_local = score
                    best_fragment_local = frag
                    best_heading_local = heading
                    best_refs_local = refs
                    best_focus_overlap = focus_overlap
        return {
            "fragment": best_fragment_local,
            "heading": best_heading_local,
            "score": float(best_score_local),
            "refs": list(best_refs_local),
            "focus_overlap": int(best_focus_overlap),
        }

    best_row = _best_from(candidates)
    rescue_row = _best_from(targeted_candidates)
    if str(rescue_row.get("fragment") or "").strip():
        if not str(best_row.get("fragment") or "").strip():
            best_row = rescue_row
        else:
            base_score = float(best_row.get("score") or 0.0)
            rescue_score = float(rescue_row.get("score") or 0.0)
            base_focus = int(best_row.get("focus_overlap") or 0)
            rescue_focus = int(rescue_row.get("focus_overlap") or 0)
            # Prefer rescue snippets when they clearly carry more query-specific focus terms.
            if (rescue_focus > base_focus and rescue_score >= (base_score - 1.5)) or (
                base_score < 8.0 and rescue_score > base_score
            ):
                best_row = rescue_row
    best_fragment = str(best_row.get("fragment") or "").strip()
    best_heading = str(best_row.get("heading") or "").strip()
    best_score = float(best_row.get("score") or float("-inf"))
    best_refs = [int(n) for n in list(best_row.get("refs") or []) if int(n) > 0]
    if (not best_fragment) or (not best_refs) or best_score < 6.0:
        return ""
    local_refs = _select_paper_guide_local_citation_lookup_refs(best_fragment, prompt=q, max_candidates=4)
    if local_refs:
        best_refs = local_refs
    best_refs = _select_primary_refs_for_prompt(
        fragment=best_fragment,
        prompt=q,
        refs=best_refs,
        max_keep=4,
    )
    best_fragment = _focus_citation_fragment_for_refs(
        best_fragment,
        target_refs=best_refs,
        prompt=q,
    )

    ref_lines: list[str] = []
    for n in best_refs[:4]:
        ref = reference_entry_lookup(src, int(n), db_dir=db_dir) if callable(reference_entry_lookup) else {}
        ref_raw = str((ref or {}).get("raw") or "").strip()
        ref_title = str((ref or {}).get("title") or "").strip()
        if ref_title:
            ref_lines.append(f"- [{int(n)}] {ref_title}.")
        elif ref_raw:
            ref_lines.append(f"- {_trim_paper_guide_prompt_snippet(ref_raw, max_chars=220)}")
        else:
            ref_lines.append(f"- [{int(n)}]")

    ref_label = ", ".join(f"[{int(n)}]" for n in best_refs[:4])
    heading_label = _trim_paper_guide_prompt_field(best_heading, max_chars=96)
    lines = [f"The paper cites {ref_label} for this point."]
    if heading_label:
        lines.append(f"This is stated in {heading_label}:")
    else:
        lines.append("This is stated explicitly in the paper:")
    lines.append(f"> {best_fragment}")
    if ref_lines:
        lines.append("")
        lines.append("Reference entries:")
        lines.extend(ref_lines)
    return "\n".join(lines).strip()
