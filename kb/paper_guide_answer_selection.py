from __future__ import annotations

import hashlib
import re

from kb.paper_guide.grounder import _extract_inline_reference_numbers
from kb.paper_guide_prompting import (
    _paper_guide_box_header_number,
    _paper_guide_prompt_family,
    _paper_guide_requested_box_numbers,
    _paper_guide_requested_heading_hints,
    _paper_guide_requested_section_targets,
)
from kb.paper_guide_provenance import (
    _extract_figure_number,
    _is_generic_heading_path,
    _text_token_overlap_score,
)
from kb.paper_guide_retrieval_runtime import (
    _paper_guide_citation_lookup_signal_score,
    _paper_guide_hit_matches_requested_targets,
)
from kb.paper_guide_shared import (
    _CLAIM_EXPERIMENT_HINT_RE,
    _CLAIM_METHOD_HINT_RE,
    _EXPERIMENT_HEADING_HINTS,
    _GENERIC_HEADING_HINTS,
    _METHOD_HEADING_HINTS,
)
from kb.source_blocks import extract_equation_number, normalize_match_text


def _stabilize_paper_guide_output_mode(
    output_mode: str,
    *,
    prompt: str,
    intent: str = "",
    explicit_hint: str = "",
) -> str:
    mode = str(output_mode or "").strip().lower() or "reading_guide"
    if explicit_hint:
        return mode
    family = _paper_guide_prompt_family(prompt, intent=intent)
    if (family in {"abstract", "figure_walkthrough", "overview", "compare", "reproduce", "method", "equation"}) and mode == "critical_review":
        return "reading_guide"
    return mode


def _split_heading_path_parts(heading_path: str) -> list[str]:
    return [part.strip() for part in str(heading_path or "").split(" / ") if part.strip()]


def _is_generic_heading_part(heading: str) -> bool:
    norm = normalize_match_text(heading)
    if not norm:
        return False
    return any(token in norm for token in _GENERIC_HEADING_HINTS)


def _paper_guide_focus_heading(hit: dict) -> str:
    if not isinstance(hit, dict):
        return ""
    meta = hit.get("meta", {}) or {}
    heading_path = (
        str(meta.get("ref_best_heading_path") or "").strip()
        or str(meta.get("heading_path") or "").strip()
        or str(meta.get("top_heading") or "").strip()
    )
    parts = _split_heading_path_parts(heading_path)
    if not parts:
        return ""
    specific_parts = [part for part in parts if not _is_generic_heading_part(part)]
    if len(specific_parts) >= 2:
        return " / ".join(specific_parts[-2:])
    if specific_parts:
        return specific_parts[-1]
    if len(parts) >= 2:
        return " / ".join(parts[-2:])
    return parts[-1]


def _looks_like_title_only_hit(hit: dict) -> bool:
    if not isinstance(hit, dict):
        return False
    text = str(hit.get("text") or "").strip()
    if not text:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    if not lines[0].startswith("# "):
        return False
    if len(lines) <= 2:
        return True
    if len(lines) == 3 and ("@" in lines[-1] or "$^{" in lines[-1]):
        return True
    return False


def _looks_like_heading_only_hit(hit: dict) -> bool:
    if not isinstance(hit, dict):
        return False
    meta = hit.get("meta", {}) or {}
    kind = str(meta.get("kind") or "").strip().lower()
    if kind == "heading":
        return True
    text = str(hit.get("text") or "").strip()
    if not text:
        return False
    focus_heading = _paper_guide_focus_heading(hit)
    heading_leaf = str((focus_heading or str(meta.get("heading_path") or "").strip()).split(" / ")[-1] or "").strip()
    if not heading_leaf:
        return False
    return normalize_match_text(text) == normalize_match_text(heading_leaf)


def _paper_guide_answer_hit_score(hit: dict, *, prompt: str) -> float:
    if not isinstance(hit, dict):
        return float("-inf")
    meta = hit.get("meta", {}) or {}
    try:
        score = float(hit.get("score") or 0.0)
    except Exception:
        score = 0.0

    heading_path = str(meta.get("heading_path") or meta.get("ref_best_heading_path") or meta.get("top_heading") or "").strip()
    focus_heading = _paper_guide_focus_heading(hit)
    heading_norm = normalize_match_text(focus_heading or heading_path)
    prompt_norm = normalize_match_text(prompt)
    text = str(hit.get("text") or "")
    text_norm = normalize_match_text(text[:1600])
    target_equation_number = extract_equation_number(prompt)
    deep_learning_topic = bool(re.search(r"(深度学习|神经网络|卷积网络|cnn|deep\s+learning|neural\s+network)", prompt, flags=re.IGNORECASE))
    family = _paper_guide_prompt_family(prompt)
    if _paper_guide_hit_matches_requested_targets(hit, prompt=prompt):
        score += 22.0
    if bool(meta.get("paper_guide_targeted_block")):
        score += 10.0

    if heading_norm:
        score += 1.8 * _text_token_overlap_score(prompt_norm, heading_norm)
    if text_norm:
        score += 0.35 * _text_token_overlap_score(prompt_norm, text_norm[:480])

    if _looks_like_title_only_hit(hit):
        score -= 18.0
    if _looks_like_heading_only_hit(hit):
        score -= 14.0
    if _paper_guide_box_header_number(text) > 0 and len(text.strip()) <= 96:
        score -= 12.0
    if _is_generic_heading_path(heading_path):
        score -= 3.5
    if "abstract" in heading_norm:
        score -= 1.2
    if "reference" in heading_norm:
        # Some papers only mention topics like deep learning in the reference list.
        # If the user explicitly asks about it, allow reference snippets to compete.
        if deep_learning_topic and any(tok in text_norm for tok in ("deep learning", "neural", "cnn")):
            score -= 1.0
        else:
            score -= 7.0

    if family == "abstract":
        if "abstract" in heading_norm:
            score += 6.5
        if "introduction" in heading_norm:
            score += 1.2
        if any(token in heading_norm for token in ("results", "discussion", "materials and methods", "method", "reference")):
            score -= 4.0
        if any(
            token in text_norm
            for token in (
                "here we introduce",
                "next generation technique",
                "label free imaging inside live cells",
            )
        ):
            score += 1.5
    elif family == "overview":
        if any(token in heading_norm for token in ("abstract", "introduction", "discussion", "result")):
            score += 3.8
        if any(token in heading_norm for token in ("materials and methods", "microscope setup", "hardware control", "data acquisition")):
            score -= 1.8
        if any(
            token in text_norm
            for token in (
                "here, we introduce",
                "in this work, we propose",
                "our results establish",
                "core contribution",
                "opens new avenues",
            )
        ):
            score += 2.0
    elif family == "compare":
        if any(token in heading_norm for token in ("results", "resolution", "discussion")):
            score += 3.2
        if any(token in text_norm for token in ("cnr", "nec", "fwhm", "open-pinhole", "closed-pinhole", "iism-apr", "same incident illumination power")):
            score += 3.0
    elif family == "figure_walkthrough":
        if any(token in heading_norm for token in ("figure", "caption", "legend", "panel")):
            score += 5.0
        if any(token in text_norm for token in ("figure 1", "fig 1", "fig. 1", "panel", "caption", "line profiles", "open-pinhole", "closed-pinhole")):
            score += 3.2
        if any(token in heading_norm for token in ("abstract", "introduction", "materials and methods")):
            score -= 1.6
    elif family == "reproduce":
        if any(token in heading_norm for token in ("materials and methods", "microscope setup", "hardware control", "data acquisition", "data analysis", "software packages")):
            score += 5.0
        if any(token in text_norm for token in ("cobolt", "hamamatsu", "picoquant", "symphotime", "pyLabLib", "dwell time", "camera exposure", "scan control")):
            score += 3.0
        if any(token in heading_norm for token in ("abstract", "discussion")):
            score -= 1.5
    elif family == "equation":
        if any(token in heading_norm for token in ("method", "background", "equation", "formula", "derivation")):
            score += 4.0
        if any(token in text for token in ("\\tag{", "$$", "where ", "denotes", "represents", "is defined as")):
            score += 4.5
        if str(meta.get("anchor_target_kind") or "").strip().lower() == "equation":
            score += 8.0
        try:
            hit_equation_number = int(meta.get("anchor_target_number") or 0)
        except Exception:
            hit_equation_number = 0
        if target_equation_number > 0 and hit_equation_number == target_equation_number:
            score += 10.0
        if target_equation_number > 0 and any(
            token in text for token in (f"\\tag{{{int(target_equation_number)}}}", f"Equation ({int(target_equation_number)})")
        ):
            score += 8.0
    elif family == "citation_lookup":
        target_sections = _paper_guide_requested_section_targets(prompt)
        non_ref_sections = [sec for sec in target_sections if sec != "references"]
        explicit_ref_list_request = bool(
            re.search(r"(?i)\b(?:reference\s+list|works?\s+cited|bibliography)\b", prompt)
        )
        if any(token in heading_norm for token in ("references", "acquisition and image reconstruction strategies", "results", "discussion")):
            score += 4.0
        if non_ref_sections:
            if any(normalize_match_text(sec) in normalize_match_text(heading_path) for sec in non_ref_sections):
                score += 5.0
            elif "references" in normalize_match_text(heading_path):
                score -= 2.5
        elif (not explicit_ref_list_request) and ("references" in normalize_match_text(heading_path)):
            score -= 3.0
        inline_refs = _extract_inline_reference_numbers(text, max_candidates=4)
        if inline_refs:
            score += 4.5
        if any(
            token in text_norm
            for token in (
                "hadamard",
                "fourier",
                "richardson",
                "lucy",
                "attributed",
                "introduced",
                "reference list",
                "works cited",
            )
        ):
            score += 2.4
        score += _paper_guide_citation_lookup_signal_score(
            prompt=prompt,
            heading=heading_path,
            text=text,
            inline_refs=inline_refs,
            explicit_ref_list_request=explicit_ref_list_request,
        )
    elif family == "strength_limits":
        if any(token in heading_norm for token in ("discussion", "results", "resolution")):
            score += 3.4
        if any(token in text_norm for token in ("not stated", "good agreement", "remaining difference", "could be improved", "quantified")):
            score += 2.0
        if str(meta.get("kind") or "").strip().lower() in {"paragraph", "list_item", "blockquote"}:
            score += 1.8

    if _CLAIM_METHOD_HINT_RE.search(prompt):
        if any(token in heading_norm for token in _METHOD_HEADING_HINTS):
            score += 4.5
        if any(token in heading_norm for token in ("principle", "setup", "analysis", "algorithm", "adaptive")):
            score += 2.2
        if any(
            token in text_norm
            for token in (
                "we developed",
                "we introduced",
                "we modified",
                "to adapt",
                "workflow",
                "algorithm",
                "camera acquisition",
                "illumination is provided",
            )
        ):
            score += 1.8
        if re.search(r"\bapr\b|adaptive pixel[- ]?reassignment", prompt, flags=re.IGNORECASE):
            if any(token in heading_norm for token in ("adaptive pixel reassignment", "rvt", "radial variance transform")):
                score += 4.2
            if any(token in text.lower() for token in ("phase correlation", "image registration", "radial variance transform", "rvt")):
                score += 5.0
            if "introduction" in heading_norm:
                score -= 1.0

    if _CLAIM_EXPERIMENT_HINT_RE.search(prompt):
        if any(token in heading_norm for token in _EXPERIMENT_HEADING_HINTS):
            score += 3.0

    return score


def _select_paper_guide_answer_hits(
    *,
    grouped_docs: list[dict],
    heading_hits: list[dict],
    prompt: str,
    top_n: int,
) -> list[dict]:
    try:
        limit = max(1, int(top_n))
    except Exception:
        limit = 1

    wants_references = bool(
        re.search(
            r"(\breference\b|\bcitation\b|\bcite\b|\[\d{1,3}\]|参考文献|引用|引文)",
            str(prompt or ""),
            flags=re.IGNORECASE,
        )
    )
    family = _paper_guide_prompt_family(prompt)
    explicit_ref_list_request = bool(
        re.search(r"(?i)\b(?:reference\s+list|works?\s+cited|bibliography)\b", str(prompt or ""))
    )
    requested_sections = _paper_guide_requested_section_targets(prompt)
    non_ref_target_requested = bool(
        any(sec != "references" for sec in requested_sections)
        or _paper_guide_requested_box_numbers(prompt)
        or (_extract_figure_number(prompt) > 0)
    )
    ranked: list[tuple[float, dict]] = []
    seen_raw: set[tuple[str, str, str]] = set()
    for hit in list(heading_hits or []) + list(grouped_docs or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        src = str(meta.get("source_path") or "").strip()
        heading = str(meta.get("heading_path") or meta.get("ref_best_heading_path") or meta.get("top_heading") or "").strip()
        text = str(hit.get("text") or "").strip()
        raw_key = (
            src,
            normalize_match_text(heading),
            hashlib.sha1(text[:320].encode("utf-8", "ignore")).hexdigest()[:12],
        )
        if raw_key in seen_raw:
            continue
        seen_raw.add(raw_key)
        score = _paper_guide_answer_hit_score(hit, prompt=prompt)
        rec = dict(hit)
        meta_out = dict(meta)
        focus_heading = _paper_guide_focus_heading(hit)
        if focus_heading:
            meta_out["top_heading"] = focus_heading
        rec["meta"] = meta_out
        ranked.append((score, rec))

    ranked.sort(key=lambda item: item[0], reverse=True)
    out: list[dict] = []
    seen_out: set[tuple[str, str]] = set()
    target_filtered = bool(_paper_guide_requested_heading_hints(prompt))
    if family == "citation_lookup" and (not explicit_ref_list_request) and (not non_ref_target_requested):
        target_filtered = False

    def _matches_effective_target(hit: dict) -> bool:
        meta_hit = hit.get("meta", {}) or {}
        return _paper_guide_hit_matches_requested_targets(hit, prompt=prompt) or bool(meta_hit.get("paper_guide_targeted_block"))

    has_target_ranked = target_filtered and any(
        _matches_effective_target(hit)
        for _score, hit in ranked
    )
    for _score, hit in ranked:
        meta = hit.get("meta", {}) or {}
        src = str(meta.get("source_path") or "").strip()
        focus = str(meta.get("top_heading") or meta.get("heading_path") or "").strip()
        block_id = str(meta.get("block_id") or "").strip()
        focus_norm = normalize_match_text(focus)
        if has_target_ranked and (not _matches_effective_target(hit)):
            continue
        if (not wants_references) and ("reference" in focus_norm):
            continue
        if _looks_like_title_only_hit(hit) and out:
            continue
        out_key = (src, block_id or focus_norm) if target_filtered else (src, focus_norm)
        if out_key in seen_out:
            continue
        seen_out.add(out_key)
        out.append(hit)
        if len(out) >= limit:
            break
    return out[:limit]


def _hit_source_path(hit: dict) -> str:
    if not isinstance(hit, dict):
        return ""
    meta = hit.get("meta", {}) or {}
    return str(meta.get("source_path") or "").strip()


def _build_answer_hits_for_generation(
    *,
    grouped_docs: list[dict],
    heading_hits: list[dict],
    top_n: int,
    allow_same_source_multiple: bool = False,
) -> list[dict]:
    try:
        limit = max(1, int(top_n))
    except Exception:
        limit = 1
    out: list[dict] = []
    seen_src: set[str] = set()

    def _push(pool: list[dict]) -> None:
        nonlocal out
        for hit in pool or []:
            if not isinstance(hit, dict):
                continue
            src = _hit_source_path(hit)
            if (not allow_same_source_multiple) and src and (src in seen_src):
                continue
            out.append(hit)
            if (not allow_same_source_multiple) and src:
                seen_src.add(src)
            if len(out) >= limit:
                return

    _push(grouped_docs)
    if len(out) < limit:
        _push(heading_hits)
    if out:
        return out[:limit]
    return list((grouped_docs or heading_hits or [])[:limit])


def _has_anchor_grounded_answer_hits(answer_hits: list[dict]) -> bool:
    for hit in answer_hits or []:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        if not str(meta.get("anchor_target_kind") or "").strip():
            continue
        try:
            anchor_score = float(meta.get("anchor_match_score") or 0.0)
        except Exception:
            anchor_score = 0.0
        if anchor_score > 0.0:
            return True
    return False
