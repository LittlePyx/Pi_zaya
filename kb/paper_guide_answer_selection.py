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
    _paper_guide_semantic_query_terms,
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
        # A targeted SourceBlock is already constrained to the bound paper and
        # ranked against every block in that paper.  Grouped reference cards
        # carry larger document-level BM25 scores that are not comparable and
        # used to push the precise paragraph behind a broad abstract.  Give the
        # verified block enough priority to lead generation and citation plans.
        score += 45.0

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


def _paper_guide_method_aspect_priority_hits(
    ranked: list[tuple[float, dict]],
    *,
    prompt: str,
) -> list[dict]:
    """Preserve one source block for each explicitly requested method aspect."""

    q = str(prompt or "")
    asks_updates = bool(
        re.search(
            r"(?i)(?:\bPDHG\b|\bprimal[- ]dual\b|\bdual update\b|\bprimal update\b|"
            r"\bproximal\b|\u5bf9\u5076\u66f4\u65b0|\u539f\u59cb\u66f4\u65b0|\u8fd1\u7aef\u7b97\u5b50)",
            q,
        )
    )
    asks_initialization = bool(
        re.search(
            r"(?i)(?:\binitiali[sz](?:e|ed|ation)\b|\bstarting point\b|\binitial guess\b|"
            r"\bzero[- ]initiali[sz]ation\b|\bpseudo[- ]inverse\b|\bFBP\b|\u521d\u59cb\u5316|\u521d\u59cb\u731c\u6d4b)",
            q,
        )
    )
    if not (asks_updates and asks_initialization):
        return []

    def _update_bonus(hit: dict) -> float:
        text = normalize_match_text(
            f"{str(((hit.get('meta') or {}).get('heading_path') or ''))} {str(hit.get('text') or '')}"
        )
        if ("primal proximal" in text) and ("dual proximal" in text):
            return 24.0
        if ("algorithm 2" in text) and ("learned proximal" in text):
            return 18.0
        if ("proximal operators" in text) and any(token in text for token in ("replaced", "learned", "network")):
            return 8.0
        return float("-inf")

    def _initialization_bonus(hit: dict) -> float:
        text = normalize_match_text(
            f"{str(((hit.get('meta') or {}).get('heading_path') or ''))} {str(hit.get('text') or '')}"
        )
        if ("zero initialization" in text) and ("pseudo inverse" in text):
            return 24.0
        if ("initial guess" in text) and any(token in text for token in ("final results", "complexity", "earlier reconstruction")):
            return 20.0
        if any(token in text for token in ("zero initialization", "starting point", "initial guess")):
            return 8.0
        return float("-inf")

    selected: list[dict] = []
    for bonus_fn in (_update_bonus, _initialization_bonus):
        best_hit: dict | None = None
        best_score = float("-inf")
        for base_score, hit in ranked:
            bonus = bonus_fn(hit)
            if bonus == float("-inf"):
                continue
            candidate_score = float(base_score) + bonus
            if candidate_score > best_score:
                best_score = candidate_score
                best_hit = hit
        if isinstance(best_hit, dict) and all(best_hit is not item for item in selected):
            selected.append(best_hit)
    return selected


def _paper_guide_named_acronym_priority_hits(
    ranked: list[tuple[float, dict]],
    *,
    prompt: str,
) -> list[dict]:
    """Reserve one source block for each named mechanism in a compound question."""

    q = str(prompt or "").strip()
    if re.search(
        r"(?i)\b(?:observation|measurement|forward|imaging)\s+"
        r"(?:model|matrix|equation)s?\b|\u89c2\u6d4b(?:\u6a21\u578b|\u77e9\u9635|\u65b9\u7a0b)",
        q,
    ):
        # Observation-model comparisons often require one equation paragraph
        # that deliberately defines both named systems together.
        return []
    named: list[str] = []
    for token in re.findall(
        r"(?<![A-Za-z0-9])([A-Z][A-Z0-9+_-]{1,15})(?![A-Za-z0-9])",
        q,
    ):
        normalized = str(token or "").upper()
        if normalized in {"PDF", "DOI", "RGB", "PSNR", "SSIM", "SNR", "CNR"}:
            continue
        # Dataset/version identifiers such as SA-1B are quantitative facets,
        # not a second named mechanism to balance against SAM. Treating them
        # as mechanism acronyms can reserve a low-value appendix occurrence
        # and consume one of the bounded answer passages before the exact
        # dataset-statistics block is considered.
        if any(char.isdigit() for char in normalized):
            continue
        if normalized not in named:
            named.append(normalized)
    if len(named) < 2:
        return []

    semantic_terms = {
        str(token or "").casefold()
        for token in _paper_guide_semantic_query_terms(q)
        if str(token or "").strip()
    }
    selected: list[dict] = []
    for acronym in named:
        matching: list[tuple[float, dict, str]] = []
        for base_score, hit in ranked:
            meta = hit.get("meta", {}) or {}
            surface = " ".join(
                (
                    str(meta.get("heading_path") or meta.get("top_heading") or ""),
                    str(hit.get("text") or ""),
                )
            )
            if not re.search(
                rf"(?<![A-Z0-9]){re.escape(acronym)}(?![A-Z0-9])",
                surface.upper(),
            ):
                continue
            matching.append((float(base_score), hit, surface))
        if not matching:
            continue
        exclusive = [
            item
            for item in matching
            if sum(
                bool(
                    re.search(
                        rf"(?<![A-Z0-9]){re.escape(other)}(?![A-Z0-9])",
                        item[2].upper(),
                    )
                )
                for other in named
            )
            == 1
        ]
        exclusive_ids = {id(item[1]) for item in exclusive}

        def priority_key(item: tuple[float, dict, str]) -> tuple[int, int, float]:
            acronym_sentences = [
                sentence
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", item[2])
                if re.search(
                    rf"(?<![A-Z0-9]){re.escape(acronym)}(?![A-Z0-9])",
                    sentence.upper(),
                )
            ]
            focused_surface = " ".join(acronym_sentences) or item[2]
            source_terms = {
                token.casefold()
                for token in re.findall(
                    r"[A-Za-z][A-Za-z0-9_-]{2,}", focused_surface
                )
            }
            return (
                len(source_terms & semantic_terms),
                int(id(item[1]) in exclusive_ids),
                item[0],
            )

        _score, chosen, _surface = max(matching, key=priority_key)
        if all(chosen is not item for item in selected):
            selected.append(chosen)
    return selected


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
    # Keep the immutable section-heading page alongside paragraphs that begin
    # on the following page.  PDF conversion can place a heading at the foot
    # of p. N and its first equation/paragraph at p. N+1.  Treating only the
    # content block's page as the section location makes an otherwise exact
    # citation miss the section the reader was told to open.
    section_starts: dict[tuple[str, str], tuple[int, str]] = {}
    for _score, hit in ranked:
        meta = hit.get("meta", {}) or {}
        if not _looks_like_heading_only_hit(hit):
            continue
        source_key = str(meta.get("source_path") or "").replace("\\", "/").casefold()
        heading_key = normalize_match_text(str(meta.get("heading_path") or ""))
        try:
            page_start = int(meta.get("page_start") or 0)
        except (TypeError, ValueError):
            page_start = 0
        heading_text = str(hit.get("text") or "").strip()
        if not source_key or not heading_key or page_start <= 0 or not heading_text:
            continue
        previous = section_starts.get((source_key, heading_key))
        if previous is None or page_start < previous[0]:
            section_starts[(source_key, heading_key)] = (page_start, heading_text)
    for _score, hit in ranked:
        meta = dict(hit.get("meta") or {})
        source_key = str(meta.get("source_path") or "").replace("\\", "/").casefold()
        heading_key = normalize_match_text(str(meta.get("heading_path") or ""))
        section_start = section_starts.get((source_key, heading_key))
        if section_start is None:
            continue
        try:
            page_start = int(meta.get("page_start") or 0)
        except (TypeError, ValueError):
            page_start = 0
        if page_start > section_start[0]:
            meta["section_page_start"] = int(section_start[0])
            meta["section_heading_text"] = str(section_start[1])
            hit["meta"] = meta
    out: list[dict] = []
    seen_out: set[tuple[str, str]] = set()
    target_filtered = bool(_paper_guide_requested_heading_hints(prompt))
    if family == "citation_lookup" and (not explicit_ref_list_request) and (not non_ref_target_requested):
        target_filtered = False

    def _matches_effective_target(hit: dict) -> bool:
        meta_hit = hit.get("meta", {}) or {}
        if target_filtered:
            # Supplemental SourceBlock scans mark every returned row as targeted,
            # including high-scoring generic sections. When the question names a
            # section, only an actual section match may satisfy the hard gate.
            if _paper_guide_hit_matches_requested_targets(hit, prompt=prompt):
                return True
            if _paper_guide_requested_box_numbers(prompt) or (_extract_figure_number(prompt) > 0):
                return bool(meta_hit.get("paper_guide_targeted_block"))
            return False
        return bool(meta_hit.get("paper_guide_targeted_block"))

    has_target_ranked = target_filtered and any(
        _matches_effective_target(hit)
        for _score, hit in ranked
    )
    priority_hits = _paper_guide_named_acronym_priority_hits(ranked, prompt=prompt)
    if family == "method":
        for hit in _paper_guide_method_aspect_priority_hits(ranked, prompt=prompt):
            if all(hit is not item for item in priority_hits):
                priority_hits.append(hit)
    ordered_ranked = [
        (float("inf"), hit)
        for hit in priority_hits
        if (not has_target_ranked) or _matches_effective_target(hit)
    ]
    ordered_ranked.extend(
        (score, hit)
        for score, hit in ranked
        if all(hit is not priority for priority in priority_hits)
    )
    for _score, hit in ordered_ranked:
        meta = hit.get("meta", {}) or {}
        src = str(meta.get("source_path") or "").strip()
        focus = str(meta.get("top_heading") or meta.get("heading_path") or "").strip()
        block_id = str(meta.get("block_id") or "").strip()
        is_targeted_block = bool(meta.get("paper_guide_targeted_block"))
        focus_norm = normalize_match_text(focus)
        if has_target_ranked and (not _matches_effective_target(hit)):
            continue
        if (not wants_references) and ("reference" in focus_norm):
            continue
        if _looks_like_title_only_hit(hit) and out:
            continue
        out_key = (src, block_id or focus_norm) if (target_filtered or is_targeted_block) else (src, focus_norm)
        if out_key in seen_out:
            continue
        seen_out.add(out_key)
        out.append(hit)
        if len(out) >= limit:
            break

    # A multi-fact question may need two short immutable blocks from the same
    # PDF page: e.g. one "Images" paragraph and one "Masks" paragraph, or an
    # equation followed by the sentence that says it is unweighted.  Raw score
    # alone tends to spend the last bounded passage on a broad ablation.  Swap
    # in at most two same-page companions only when they add a term explicitly
    # requested by the question.  This preserves the context/latency cap while
    # improving factual completeness instead of broadening retrieval.
    semantic_surface = " ".join(_paper_guide_semantic_query_terms(prompt))

    def _facet_terms(value: object) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(
                r"[A-Za-z][A-Za-z0-9_-]{1,}|\d+(?:\.\d+)?%?|[\u4e00-\u9fff]{2,8}",
                f"{value or ''}",
            )
            if len(token.rstrip("%")) >= 2
        }

    requested_terms = _facet_terms(f"{prompt} {semantic_surface}")

    def _candidate_surface(hit: dict) -> str:
        meta = hit.get("meta", {}) or {}
        return " ".join(
            (
                str(meta.get("heading_path") or meta.get("top_heading") or ""),
                str(hit.get("text") or ""),
            )
        )

    def _source_page(hit: dict) -> tuple[str, int]:
        meta = hit.get("meta", {}) or {}
        source = str(meta.get("source_path") or "").replace("\\", "/").casefold()
        try:
            page = int(meta.get("page_start") or 0)
        except (TypeError, ValueError):
            page = 0
        return source, page

    facet_doc_frequency: dict[str, int] = {term: 0 for term in requested_terms}
    for _score, ranked_hit in ranked:
        ranked_terms = _facet_terms(_candidate_surface(ranked_hit))
        for term in requested_terms & ranked_terms:
            facet_doc_frequency[term] = facet_doc_frequency.get(term, 0) + 1

    chosen_ids = {id(hit) for hit in out}
    protected_companion_ids: set[int] = set()
    protected_base_ids: set[int] = set()
    companion_rows: list[tuple[int, float, int, float, dict, dict]] = []
    for base in list(out):
        base_source, base_page = _source_page(base)
        if not base_source or base_page <= 0:
            continue
        base_requested = requested_terms & _facet_terms(_candidate_surface(base))
        for candidate_score, candidate in ranked:
            if id(candidate) in chosen_ids or _looks_like_heading_only_hit(candidate):
                continue
            candidate_source, candidate_page = _source_page(candidate)
            if candidate_source != base_source or candidate_page != base_page:
                continue
            candidate_requested = requested_terms & _facet_terms(_candidate_surface(candidate))
            gained = candidate_requested - base_requested
            if not gained or len(candidate_requested) < 2:
                continue
            numeric_gain = sum(
                bool(re.fullmatch(r"\d+(?:\.\d+)?%?|\d+[a-z]+", token))
                for token in gained
            )
            rarity_gain = sum(
                1.0 / max(1, facet_doc_frequency.get(term, 1))
                for term in gained
            )
            companion_rows.append(
                (
                    numeric_gain,
                    rarity_gain,
                    len(gained),
                    float(candidate_score),
                    candidate,
                    base,
                )
            )
    companion_rows.sort(
        key=lambda item: (item[0], item[1], item[2], item[3]),
        reverse=True,
    )
    replacements = 0
    for _numeric_gain, _rarity, _gain, _score, candidate, base in companion_rows:
        if replacements >= 2 or id(candidate) in chosen_ids:
            continue
        candidate_terms = requested_terms & _facet_terms(_candidate_surface(candidate))
        victim_rows: list[tuple[int, float, int]] = []
        for index, current in enumerate(out):
            if (
                current is base
                or id(current) in protected_companion_ids
                or id(current) in protected_base_ids
            ):
                continue
            current_source, _current_page = _source_page(current)
            candidate_source, _candidate_page = _source_page(candidate)
            if current_source != candidate_source:
                continue
            current_terms = requested_terms & _facet_terms(_candidate_surface(current))
            other_terms: set[str] = set()
            for other_index, other in enumerate(out):
                if other_index == index:
                    continue
                other_terms.update(
                    requested_terms & _facet_terms(_candidate_surface(other))
                )
            unique_current_terms = current_terms - other_terms
            current_heading = normalize_match_text(
                str(((current.get("meta") or {}).get("heading_path") or ""))
            )
            low_value = int(
                bool(re.search(r"\bablation|appendix|references?\b", current_heading))
            )
            try:
                current_score = float(current.get("score") or 0.0)
            except (TypeError, ValueError):
                current_score = 0.0
            victim_rows.append(
                (
                    len(unique_current_terms) - 3 * low_value,
                    current_score,
                    index,
                )
            )
        if not victim_rows:
            continue
        victim_overlap, _victim_score, victim_index = min(victim_rows)
        base_terms = requested_terms & _facet_terms(_candidate_surface(base))
        candidate_gain = candidate_terms - base_terms
        if len(candidate_gain) < max(1, victim_overlap):
            continue
        chosen_ids.discard(id(out[victim_index]))
        out[victim_index] = candidate
        chosen_ids.add(id(candidate))
        protected_companion_ids.add(id(candidate))
        protected_base_ids.add(id(base))
        replacements += 1
    return out[:limit]


def _hit_source_path(hit: dict) -> str:
    if not isinstance(hit, dict):
        return ""
    meta = hit.get("meta", {}) or {}
    return str(meta.get("source_path") or "").strip()


def _multi_source_answer_hit_is_low_value(hit: dict, *, prompt: str = "") -> bool:
    if not isinstance(hit, dict):
        return True
    if _looks_like_title_only_hit(hit) or _looks_like_heading_only_hit(hit):
        return True
    meta = hit.get("meta", {}) or {}
    heading = normalize_match_text(
        str(meta.get("heading_path") or meta.get("ref_best_heading_path") or meta.get("top_heading") or "")
    )
    if re.search(
        r"(?:^|\s)(?:references|bibliography|acknowledgements?|acknowledgments?|"
        r"data(?: and code)? availability|code availability)(?:\s|$)",
        heading,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(r"\bauthor biographies?\b|\bbiographical notes?\b", heading, flags=re.IGNORECASE):
        asks_about_authors = bool(
            re.search(
                r"\bauthors?\b|\bbiograph|\beducation\b|\bdegree\b|\baffiliation\b|"
                r"作者|传记|简历|教育经历|学历|学位|任职|职位|研究方向",
                str(prompt or ""),
                flags=re.IGNORECASE,
            )
        )
        if not asks_about_authors:
            return True
    text_prefix = normalize_match_text(str(hit.get("text") or "")[:900])
    if re.search(r"\backnowledgements?\b|\backnowledgments?\b", text_prefix, flags=re.IGNORECASE):
        return True
    if re.search(r"\bdata(?: and code)? availability\b|\bcode availability\b", text_prefix, flags=re.IGNORECASE):
        return True
    if re.search(r"\bdata from\b.{0,320}\bzenodo\b", text_prefix, flags=re.IGNORECASE):
        return True
    return False


def _rescue_multi_source_answer_hits(
    *,
    grouped_docs: list[dict],
    raw_hits: list[dict],
    prompt: str,
) -> list[dict]:
    """Choose one useful in-memory evidence hit per already selected source."""

    def source_key(hit: dict) -> str:
        return _hit_source_path(hit).replace("\\", "/").casefold()

    raw_by_source: dict[str, list[dict]] = {}
    for hit in raw_hits or []:
        if not isinstance(hit, dict):
            continue
        key = source_key(hit)
        if not key:
            continue
        raw_by_source.setdefault(key, []).append(hit)

    rescued: list[dict] = []
    for grouped in grouped_docs or []:
        if not isinstance(grouped, dict):
            continue
        key = source_key(grouped)
        grouped_meta = grouped.get("meta", {}) or {}
        if str(grouped_meta.get("structured_kind") or "").strip().lower() in {
            "table_metric",
            "table_row",
        }:
            # The grouped reference has already preserved the table-aware
            # retriever's winning metric series.  Re-scoring raw hits here with
            # the generic paper-guide heuristic can favor a shorter ablation
            # row from the same paper and make the generated answer disagree
            # with both retrieval rank and the reference card.
            rescued.append(dict(grouped))
            continue
        candidates = [grouped, *raw_by_source.get(key, [])]
        ranked: list[tuple[float, int, dict]] = []
        seen: set[tuple[str, str]] = set()
        for idx, candidate in enumerate(candidates):
            if not isinstance(candidate, dict) or _multi_source_answer_hit_is_low_value(candidate, prompt=prompt):
                continue
            meta = candidate.get("meta", {}) or {}
            fingerprint = (
                str(meta.get("block_id") or meta.get("chunk_id") or "").strip(),
                hashlib.sha1(str(candidate.get("text") or "")[:480].encode("utf-8", "ignore")).hexdigest()[:12],
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            ranked.append(
                (
                    _paper_guide_answer_hit_score(candidate, prompt=prompt),
                    1 if idx == 0 else 0,
                    candidate,
                )
            )
        if not ranked:
            continue
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = dict(ranked[0][2])
        if ranked[0][1] == 0:
            meta_out = dict(selected.get("meta", {}) or {})
            meta_out["multi_source_representative_rescue"] = True
            selected["meta"] = meta_out
        rescued.append(selected)
    return rescued


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


def _merge_same_source_answer_hits(
    hits: list[dict],
    *,
    max_passages: int = 5,
    passage_char_limit: int = 1100,
) -> list[dict]:
    """Represent complementary passages from one paper as one citeable doc.

    Generation context labels are document identifiers. Giving several chunks
    from the same paper separate ``DOC-n`` labels makes ordinary numeric model
    citations ambiguous with the paper's bibliography. Preserve the passages,
    headings, and pages in one source bundle so all answer claims cite DOC-1;
    the citation plan can still bind each claim to its exact passage.
    """

    rows = [dict(hit) for hit in list(hits or []) if isinstance(hit, dict)]
    if len(rows) <= 1:
        return rows
    source_keys = {
        _hit_source_path(hit).replace("\\", "/").casefold()
        for hit in rows
        if _hit_source_path(hit)
    }
    if len(source_keys) != 1:
        return rows
    try:
        limit = max(1, int(max_passages or 5))
    except Exception:
        limit = 5
    try:
        text_limit = max(320, int(passage_char_limit or 1100))
    except Exception:
        text_limit = 1100

    passages: list[str] = []
    passage_meta: list[dict] = []
    seen: set[str] = set()
    for hit in rows:
        if len(passages) >= limit:
            break
        meta = dict(hit.get("meta") or {})
        body = str(hit.get("text") or "").strip()
        if not body:
            continue
        fingerprint = hashlib.sha1(
            normalize_match_text(body[:520]).encode("utf-8", "ignore")
        ).hexdigest()[:16]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        if len(body) > text_limit:
            body = body[:text_limit].rsplit(" ", 1)[0].rstrip() + "..."
        heading = str(meta.get("heading_path") or meta.get("top_heading") or "").strip()
        try:
            page_start = int(meta.get("page_start") or 0)
            page_end = int(meta.get("page_end") or page_start or 0)
        except Exception:
            page_start = 0
            page_end = 0
        label_parts = [part for part in (heading, f"p. {page_start}" if page_start > 0 else "") if part]
        label = " | ".join(label_parts) or f"Passage {len(passages) + 1}"
        passages.append(f"[Source passage {len(passages) + 1}: {label}]\n{body}")
        passage_meta.append(
            {
                key: value
                for key, value in {
                    "heading_path": heading,
                    "page_start": page_start,
                    "page_end": page_end,
                    "block_id": str(meta.get("block_id") or "").strip(),
                    "anchor_id": str(meta.get("anchor_id") or "").strip(),
                    "anchor_kind": str(meta.get("anchor_kind") or "").strip(),
                    "section_page_start": int(meta.get("section_page_start") or 0),
                    "section_heading_text": str(meta.get("section_heading_text") or "").strip(),
                    "score": float(hit.get("score") or 0.0),
                    "text": body,
                }.items()
                if value not in ("", 0)
            }
        )
    if len(passages) <= 1:
        return rows[:1]

    merged = dict(rows[0])
    merged["text"] = "\n\n".join(passages)
    merged["score"] = max(float(hit.get("score") or 0.0) for hit in rows)
    merged_meta = dict(merged.get("meta") or {})
    merged_meta["same_source_evidence_bundle"] = True
    merged_meta["source_passage_count"] = len(passages)
    merged_meta["source_passages"] = passage_meta
    merged["meta"] = merged_meta
    return [merged]


def _bundle_answer_hits_by_source(hits: list[dict]) -> list[dict]:
    """Bundle repeated passages per paper while preserving paper order."""

    ordered_keys: list[str] = []
    grouped: dict[str, list[dict]] = {}
    passthrough: list[tuple[int, dict]] = []
    for index, hit in enumerate(list(hits or [])):
        if not isinstance(hit, dict):
            continue
        source_key = _hit_source_path(hit).replace("\\", "/").casefold()
        if not source_key:
            passthrough.append((index, dict(hit)))
            continue
        if source_key not in grouped:
            ordered_keys.append(source_key)
            grouped[source_key] = []
        grouped[source_key].append(dict(hit))

    out: list[dict] = []
    for source_key in ordered_keys:
        out.extend(_merge_same_source_answer_hits(grouped[source_key]))
    out.extend(hit for _index, hit in sorted(passthrough, key=lambda item: item[0]))
    return out


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
