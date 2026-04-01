from __future__ import annotations

import re
from pathlib import Path

from kb.paper_guide_citation_surfacing import (
    _drop_paper_guide_locate_only_line_citations,
    _inject_paper_guide_card_citations,
    _inject_paper_guide_focus_citations,
    _promote_paper_guide_numeric_reference_citations,
)
from kb.paper_guide_focus import (
    _extract_bound_paper_figure_caption,
    _extract_bound_paper_method_focus,
    _extract_paper_guide_method_detail_excerpt,
    _extract_paper_guide_method_focus_terms,
    _repair_paper_guide_focus_answer_generic,
)
from kb.paper_guide_grounding_runtime import (
    _extract_inline_reference_numbers,
    _inject_paper_guide_support_markers,
    _resolve_paper_guide_support_markers,
    _paper_guide_cue_tokens,
)
from kb.paper_guide_postprocess import _sanitize_paper_guide_answer_for_user
from kb.paper_guide_provenance import _resolve_paper_guide_md_path
from kb.paper_guide_provenance import _extract_figure_number
from kb.paper_guide_retrieval_runtime import (
    _paper_guide_citation_lookup_fragments,
    _paper_guide_citation_lookup_query_tokens,
    _paper_guide_citation_lookup_signal_score,
    _select_paper_guide_local_citation_lookup_refs,
    _paper_guide_targeted_source_block_hits,
)
from kb.paper_guide_prompting import (
    _paper_guide_prompt_family,
    _paper_guide_prompt_requests_doc_map,
    _paper_guide_prompt_requests_exact_method_support,
    _requested_figure_number,
)
from kb.paper_guide_target_scope import _extract_prompt_panel_letters
from kb.inpaper_citation_grounding import parse_ref_num_set
from kb.source_blocks import extract_equation_number, load_source_blocks, normalize_inline_markdown


def _surface_plain_text_box_formula(answer: str, *, support_resolution: list[dict]) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    if "k log(n/k)" in text.lower() or "m >= o(k log(n/k))" in text.lower():
        return text
    formula_source = text
    for rec in list(support_resolution or []):
        if not isinstance(rec, dict):
            continue
        locate_anchor = str(rec.get("locate_anchor") or "").strip()
        if locate_anchor:
            formula_source = locate_anchor
            break
    plain = str(formula_source or "")
    plain = plain.replace("\\ge", ">=").replace("\\log", "log")
    plain = re.sub(r"\$+", "", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    match = re.search(r"(?i)(m\s*>=\s*o\(\s*k\s*log\s*\(\s*n\s*/\s*k\s*\)\s*\))", plain)
    if not match:
        return text
    return f"{text}\n\nPlain-text condition: {match.group(1)}."


def _resolve_doc_map_records_from_source(
    source_path: str,
    *,
    prompt: str,
    db_dir: Path | None,
    max_items: int = 16,
) -> list[dict]:
    src = str(source_path or "").strip()
    if not src:
        return []
    md_path = _resolve_paper_guide_md_path(src, db_dir=db_dir)
    if not isinstance(md_path, Path) or (not md_path.exists()):
        return []
    try:
        blocks = list(load_source_blocks(md_path) or [])
    except Exception:
        blocks = []
    if not blocks:
        return []

    try:
        limit = max(4, int(max_items))
    except Exception:
        limit = 16

    heading_bad_re = re.compile(
        r"(?i)\b(references?|bibliography|works?\s+cited|appendi(?:x|ces)|supplementary|acknowledg(e)?ments?)\b"
    )
    wants_refs = bool(re.search(r"(?i)(reference|references|bibliography|cite|citation|参考文献|引用)", str(prompt or "")))

    def pick_anchor(raw_text: str) -> str:
        raw = str(raw_text or "").strip()
        if not raw:
            return ""
        clean = normalize_inline_markdown(raw)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) < 40:
            return ""
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])|(?<=[.!?])\s+(?=[\"'(])", clean)
        parts = [p.strip() for p in parts if p.strip()]
        for p in parts[:8]:
            if len(p) < 45:
                continue
            if len(p) > 260:
                continue
            return p
        return clean[:220].rstrip(".") + ("." if clean and clean[-1].isalnum() else "")

    out: list[dict] = []
    seen_heading: set[str] = set()
    for block in blocks:
        if not isinstance(block, dict):
            continue
        heading_path = str(block.get("heading_path") or "").strip()
        if not heading_path:
            continue
        heading_key = heading_path.lower()
        if heading_key in seen_heading:
            continue
        if (not wants_refs) and heading_bad_re.search(heading_path):
            continue
        kind = str(block.get("kind") or "").strip().lower()
        if kind not in {"paragraph", "list_item", "blockquote"}:
            continue
        text = str(block.get("raw_text") or block.get("text") or "").strip()
        if not text:
            continue
        anchor = pick_anchor(text)
        if not anchor:
            continue
        seen_heading.add(heading_key)
        out.append(
            {
                "source_path": src,
                "block_id": str(block.get("block_id") or "").strip(),
                "anchor_id": str(block.get("anchor_id") or "").strip(),
                "heading_path": heading_path,
                "locate_anchor": anchor,
                "claim_type": "doc_map",
                "cite_policy": "locate_only",
                "segment_text": anchor,
                "segment_index": -1,
            }
        )
        if len(out) >= limit:
            break
    return out


def _leading_inline_heading_label(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    patterns = (
        r"^\s*\*\*\s*([^*\n]{3,80}?)\s*\.\s*\*\*",
        r"^\s*__\s*([^_\n]{3,80}?)\s*\.\s*__",
        r"^\s*([A-Z][A-Za-z0-9][A-Za-z0-9 /\-]{2,60}?)\.\s+(?:The|We|In|Using|After|As|This)\b",
    )
    for pattern in patterns:
        m = re.match(pattern, raw)
        if not m:
            continue
        label = str(m.group(1) or "").strip(" .:;-")
        if len(label) < 3:
            continue
        return label
    return ""


def _resolve_exact_method_support_from_source(
    source_path: str,
    *,
    prompt: str,
    db_dir: Path | None,
) -> dict:
    src = str(source_path or "").strip()
    if not src:
        return {}
    md_path = _resolve_paper_guide_md_path(src, db_dir=db_dir)
    if not isinstance(md_path, Path) or (not md_path.exists()):
        return {}
    try:
        blocks = load_source_blocks(md_path)
    except Exception:
        blocks = []
    if not blocks:
        return {}
    focus_terms = _extract_paper_guide_method_focus_terms(prompt)
    low_prompt = str(prompt or "").strip().lower()
    exact_cues = [
        cue
        for cue in ("original iism dataset", "applied back", "reapplied", "re-applied", "shift vectors")
        if cue in low_prompt
    ]
    detail_patterns: dict[str, tuple[str, ...]] = {
        "framework": ("framework", "pytorch", "tensorflow", "keras"),
        "optimizer": ("optimizer", "adam", "adamw", "sgd"),
        "learning_rate": ("learning rate", "lr"),
        "batch_size": ("batch size",),
        "iterations": ("iterations", "iteration", "epochs", "epoch"),
        "rays": ("rays",),
        "beat_frequency": ("beat frequency",),
        "sampling_rate": ("sampling rate", "data acquisition card", "sampling", "dac"),
    }
    requested_detail_keys = {
        name
        for name, patterns in detail_patterns.items()
        if any(str(pattern or "").strip() and str(pattern).lower() in low_prompt for pattern in patterns)
    }

    # Generic training/implementation-detail prompts (PyTorch/optimizer/batch size/iterations/etc.).
    wants_phrases: list[str] = []
    for phrase in (
        "implementation details",
        "network training",
        "training",
        "optimizer",
        "adam",
        "pytorch",
        "learning rate",
        "batch size",
        "iterations",
        "epochs",
        "epoch",
        "rays",
        "learning rate",
        "beat frequency",
        "sampling rate",
        "data acquisition card",
    ):
        if phrase in low_prompt:
            wants_phrases.append(phrase)
    cue_tokens = [tok for tok in _paper_guide_cue_tokens(prompt) if tok and len(tok) >= 3]

    def _candidate_sentences(text: str) -> list[str]:
        raw = " ".join(str(text or "").replace("\r\n", "\n").replace("\r", "\n").split())
        if not raw:
            return []
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=\])\s+(?=[A-Z])", raw)
        out: list[str] = []
        for part in parts:
            s = str(part or "").strip()
            if len(s) < 28:
                continue
            if len(s) > 620:
                s = s[:620].rsplit(" ", 1)[0].strip()
            out.append(s)
        return out or [raw[:620]]

    best_record: dict = {}
    best_score = float("-inf")
    for block in list(blocks or []):
        if not isinstance(block, dict):
            continue
        kind = str(block.get("kind") or "").strip().lower()
        if kind not in {"paragraph", "list_item", "blockquote"}:
            continue
        text = str(block.get("raw_text") or block.get("text") or "").strip()
        if not text:
            continue
        text_low = text.lower()
        # Skip obvious author/affiliation lines unless explicitly requested.
        if ("@" in text) and not any(k in low_prompt for k in ("email", "author", "affiliation")):
            if not re.search(r"\bwe\b|\bimplement\b|\btrain\b|\boptimizer\b|\bbatch\b|\biterations?\b", text_low):
                continue

        heading_path = str(block.get("heading_path") or "").strip()
        inline_heading = _leading_inline_heading_label(text)
        inline_heading_low = inline_heading.lower()
        if inline_heading and inline_heading_low not in heading_path.lower():
            if inline_heading_low in low_prompt:
                heading_path = f"{heading_path} / {inline_heading}" if heading_path else inline_heading
        heading_low = heading_path.lower()

        # iISM/APR exact-support prompts: keep the old focused-excerpt behavior.
        if exact_cues:
            excerpt = _extract_paper_guide_method_detail_excerpt(text, focus_terms=focus_terms)
            if not excerpt:
                continue
            low = excerpt.lower()
            score = 0.0
            matched_terms = [term for term in focus_terms if term and term.lower() in low]
            score += min(6.0, 1.15 * float(len(matched_terms)))
            if "shift vectors" in low:
                score += 2.0
            if "original iism dataset" in low:
                score += 8.0
            if "applied back" in low:
                score += 4.6
            if re.search(r"\bre-?appl(?:y|ied)\b", low):
                score += 2.2
            if re.search(r"\bappl(?:ied|y)\b", low) and "original iism" in low:
                score += 3.2
            if "results" in heading_low and ("original iism dataset" in low or "applied back" in low):
                score += 2.4
            if ("materials and methods" in heading_low or heading_low.startswith("methods")) and "original iism dataset" not in low:
                score -= 1.2
            if all(cue not in low for cue in exact_cues):
                score -= 4.0
            if score > best_score:
                best_score = score
                best_record = {
                    "source_path": src,
                    "block_id": str(block.get("block_id") or "").strip(),
                    "anchor_id": str(block.get("anchor_id") or "").strip(),
                    "heading_path": heading_path,
                    "locate_anchor": excerpt,
                    "claim_type": "method_detail",
                    "cite_policy": "locate_only",
                    "segment_text": excerpt,
                }
            continue

        # Generic method-detail selection: score headings + cue overlap + prompt-specific phrases,
        # then pick the best matching sentence within that block.
        heading_boost = 0.0
        if "implementation details" in heading_low:
            heading_boost += 14.0
        if "experimental setup" in heading_low or "experiments" in heading_low:
            heading_boost += 6.0
        if heading_low.startswith("methods") or " / 3. method" in heading_low or " / method" in heading_low:
            heading_boost += 3.0
        if "abstract" in heading_low or "references" in heading_low:
            heading_boost -= 6.0

        shared = [tok for tok in cue_tokens if tok in text_low or tok in heading_low]
        token_score = min(16.0, 1.7 * float(len(shared)))

        phrase_score = 0.0
        for phrase in wants_phrases:
            if phrase and phrase in text_low:
                phrase_score += 6.0
        if ("batch size" in low_prompt) and ("batch size" in text_low):
            phrase_score += 10.0
        if ("iterations" in low_prompt) and ("iterations" in text_low):
            phrase_score += 10.0
        if ("rays" in low_prompt) and ("rays" in text_low):
            phrase_score += 6.0
        if ("pytorch" in low_prompt) and ("pytorch" in text_low):
            phrase_score += 8.0
        if ("adam" in low_prompt) and ("adam" in text_low):
            phrase_score += 7.0
        if ("learning rate" in low_prompt) and ("learning rate" in text_low):
            phrase_score += 10.0
        if ("beat frequency" in low_prompt) and ("beat frequency" in text_low):
            phrase_score += 10.0
        if ("sampling rate" in low_prompt) and ("sampling rate" in text_low):
            phrase_score += 10.0
        if ("data acquisition card" in low_prompt) and ("data acquisition card" in text_low):
            phrase_score += 8.0

        sentence_records: list[tuple[float, int, str, set[str]]] = []
        for sent_idx, sent in enumerate(_candidate_sentences(text)):
            sent_low = sent.lower()
            coverage = {
                name
                for name in requested_detail_keys
                if any(str(pattern).lower() in sent_low for pattern in detail_patterns.get(name, ()))
            }
            s = 0.0
            if ("batch size" in low_prompt) and ("batch size" in sent_low):
                s += 10.0
            if ("iterations" in low_prompt) and ("iterations" in sent_low):
                s += 10.0
            if ("rays" in low_prompt) and ("rays" in sent_low):
                s += 6.0
            if ("learning rate" in low_prompt) and ("learning rate" in sent_low):
                s += 10.0
            if ("beat frequency" in low_prompt) and ("beat frequency" in sent_low):
                s += 10.0
            if ("sampling rate" in low_prompt) and ("sampling rate" in sent_low):
                s += 10.0
            if ("data acquisition card" in low_prompt) and ("data acquisition card" in sent_low):
                s += 8.0
            s += min(10.0, 1.3 * float(len([tok for tok in cue_tokens if tok in sent_low])))
            s += max(0.0, 4.0 - (0.004 * float(len(sent))))
            if coverage:
                s += 6.0 * float(len(coverage))
            sentence_records.append((s, sent_idx, sent, coverage))

        best_sent = ""
        best_sent_score = float("-inf")
        covered_details: set[str] = set()
        if requested_detail_keys and sentence_records:
            selected_records: list[tuple[float, int, str, set[str]]] = []
            remaining = list(sentence_records)
            while remaining and len(selected_records) < 3:
                best_choice = None
                best_choice_score = float("-inf")
                for rec in remaining:
                    base_score, sent_idx, _sent, coverage = rec
                    novelty = len(set(coverage).difference(covered_details))
                    score_adj = float(base_score) + (8.0 * float(novelty))
                    if score_adj > best_choice_score:
                        best_choice_score = score_adj
                        best_choice = rec
                if best_choice is None:
                    break
                remaining.remove(best_choice)
                selected_records.append(best_choice)
                covered_details.update(set(best_choice[3]))
                if covered_details.issuperset(requested_detail_keys):
                    break
            selected_records.sort(key=lambda item: item[1])
            best_sent = " ".join(str(item[2] or "").strip() for item in selected_records if str(item[2] or "").strip()).strip()
            best_sent_score = float(sum(float(item[0]) for item in selected_records)) + (6.0 * float(len(covered_details)))
        else:
            for sent_score, _sent_idx, sent, coverage in sentence_records:
                if sent_score > best_sent_score:
                    best_sent_score = sent_score
                    best_sent = sent
                    covered_details = set(coverage)
        if not best_sent:
            continue

        coverage_bonus = 0.0
        if requested_detail_keys:
            missing_count = max(0, len(requested_detail_keys) - len(covered_details))
            coverage_bonus += 6.0 * float(len(covered_details))
            coverage_bonus -= 3.0 * float(missing_count)

        score = heading_boost + token_score + phrase_score + best_sent_score + coverage_bonus
        if score > best_score:
            best_score = score
            best_record = {
                "source_path": src,
                "block_id": str(block.get("block_id") or "").strip(),
                "anchor_id": str(block.get("anchor_id") or "").strip(),
                "heading_path": heading_path,
                "locate_anchor": best_sent,
                "claim_type": "method_detail",
                "cite_policy": "locate_only",
                "segment_text": best_sent,
            }
    return best_record


def _extract_exact_method_support_from_source(
    source_path: str,
    *,
    prompt: str,
    db_dir: Path | None,
) -> str:
    rec = _resolve_exact_method_support_from_source(
        source_path,
        prompt=prompt,
        db_dir=db_dir,
    )
    return str(rec.get("locate_anchor") or "").strip()


def _paper_guide_prompt_requests_exact_equation_support(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    has_equation_marker = bool(
        re.search(r"(?i)\b(?:equation|eq\.?|formula)\b", q) or ("公式" in q) or ("方程" in q)
    )
    if not has_equation_marker:
        return False
    if extract_equation_number(q) > 0:
        return True
    return bool(
        re.search(
            r"(?i)(?:\bvariable(?:s)?\b|\bdefine(?:s|d)?\b|\bdefinition\b|\bdenote(?:s|d)?\b|\brepresent(?:s|ed)?\b|"
            r"\bwhat\s+does\b|\bpoint\s+me\s+to\b|\bexact\s+supporting\s+part\b|\bexact\s+supporting\s+sentence\b|"
            r"\u53d8\u91cf|\u7b26\u53f7|\u5b9a\u4e49|\u539f\u6587|\u652f\u6301|\u54ea\u91cc)",
            q,
        )
    )


def _looks_like_equation_explanation_block(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    plain = normalize_inline_markdown(raw)
    low = plain.lower()
    if low.startswith("where "):
        return True
    signal_score = 0
    if "where " in low:
        signal_score += 1
    if any(token in low for token in ("denotes", "represents", "is defined as", "near and far bounds")):
        signal_score += 1
    if len(re.findall(r"\$[^$]{1,120}\$", raw)) >= 2:
        signal_score += 1
    if re.search(r"(?i)\b(?:variable|variables|symbol|symbols|parameter|parameters)\b", plain):
        signal_score += 1
    return signal_score >= 2


def _resolve_exact_equation_support_from_source(
    source_path: str,
    *,
    prompt: str,
    db_dir: Path | None,
) -> dict:
    src = str(source_path or "").strip()
    q = str(prompt or "").strip()
    if (not src) or (not q):
        return {}
    equation_number = int(extract_equation_number(q) or 0)
    if equation_number <= 0:
        return {}

    md_path = _resolve_paper_guide_md_path(src, db_dir=db_dir)
    if not isinstance(md_path, Path) or (not md_path.exists()):
        return {}
    try:
        blocks = list(load_source_blocks(md_path) or [])
    except Exception:
        blocks = []
    if not blocks:
        return {}

    cue_tokens = [tok for tok in _paper_guide_cue_tokens(q) if tok not in {"equation", "formula", "variables"}]
    best_index = -1
    best_block: dict = {}
    best_score = float("-inf")
    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        kind = str(block.get("kind") or "").strip().lower()
        raw_text = str(block.get("raw_text") or block.get("text") or "").strip()
        if not raw_text:
            continue
        try:
            block_equation_number = int(block.get("number") or 0)
        except Exception:
            block_equation_number = 0
        if block_equation_number <= 0:
            block_equation_number = int(extract_equation_number(raw_text) or 0)
        if block_equation_number != equation_number:
            continue
        heading_low = str(block.get("heading_path") or "").strip().lower()
        text_low = raw_text.lower()
        score = 0.0
        if kind == "equation":
            score += 18.0
        if f"\\tag{{{int(equation_number)}}}" in raw_text:
            score += 8.0
        if any(token in heading_low for token in ("method", "background", "equation", "formula", "derivation")):
            score += 4.0
        if "$$" in raw_text:
            score += 2.0
        if cue_tokens:
            score += min(8.0, 2.0 * float(len([tok for tok in cue_tokens if tok in text_low or tok in heading_low])))
        if score > best_score:
            best_score = score
            best_index = idx
            best_block = block
    if best_index < 0 or not best_block:
        return {}

    target_heading = str(best_block.get("heading_path") or "").strip()

    def _same_heading(block: dict) -> bool:
        heading = str(block.get("heading_path") or "").strip()
        if target_heading:
            return heading == target_heading
        return not heading

    leadin_text = ""
    leadin_block: dict = {}
    leadin_score = float("-inf")
    for offset in range(1, 4):
        idx = best_index - offset
        if idx < 0:
            break
        block = blocks[idx]
        if not isinstance(block, dict):
            continue
        kind = str(block.get("kind") or "").strip().lower()
        if kind not in {"paragraph", "list_item", "blockquote"}:
            continue
        if not _same_heading(block):
            continue
        raw_text = str(block.get("raw_text") or block.get("text") or "").strip()
        if not raw_text:
            continue
        plain = normalize_inline_markdown(raw_text)
        low = plain.lower()
        score = 0.0
        if any(token in low for token in ("can be written as", "is defined as", "is given by", "can be expressed as")):
            score += 8.0
        if any(token in low for token in ("the color", "ray", "rendering", "volume")):
            score += 3.0
        if _looks_like_equation_explanation_block(raw_text):
            score -= 4.0
        score -= 0.1 * float(offset)
        if score > leadin_score:
            leadin_score = score
            leadin_text = raw_text
            leadin_block = block

    explanation_text = ""
    explanation_block: dict = {}
    explanation_score = float("-inf")
    for offset in range(1, 5):
        idx = best_index + offset
        if idx >= len(blocks):
            break
        block = blocks[idx]
        if not isinstance(block, dict):
            continue
        kind = str(block.get("kind") or "").strip().lower()
        if kind == "equation":
            break
        if kind not in {"paragraph", "list_item", "blockquote"}:
            continue
        if not _same_heading(block):
            break
        raw_text = str(block.get("raw_text") or block.get("text") or "").strip()
        if not raw_text:
            continue
        plain = normalize_inline_markdown(raw_text)
        low = plain.lower()
        score = 0.0
        if _looks_like_equation_explanation_block(raw_text):
            score += 10.0
        if low.startswith("where "):
            score += 8.0
        if any(token in low for token in ("denotes", "represents", "is defined as", "near and far bounds")):
            score += 4.0
        if cue_tokens:
            score += min(6.0, 1.5 * float(len([tok for tok in cue_tokens if tok in low])))
        score -= 0.1 * float(offset)
        if score > explanation_score:
            explanation_score = score
            explanation_text = raw_text
            explanation_block = block

    equation_markdown = str(best_block.get("raw_text") or best_block.get("text") or "").strip()
    if not equation_markdown:
        return {}

    return {
        "source_path": src,
        "heading_path": target_heading,
        "equation_number": int(equation_number),
        "equation_markdown": equation_markdown,
        "equation_block_id": str(best_block.get("block_id") or "").strip(),
        "equation_anchor_id": str(best_block.get("anchor_id") or "").strip(),
        "equation_anchor": normalize_inline_markdown(equation_markdown)[:900],
        "leadin_text": str(leadin_text or "").strip(),
        "leadin_block_id": str((leadin_block or {}).get("block_id") or "").strip(),
        "leadin_anchor_id": str((leadin_block or {}).get("anchor_id") or "").strip(),
        "explanation_text": str(explanation_text or "").strip(),
        "explanation_block_id": str((explanation_block or {}).get("block_id") or "").strip(),
        "explanation_anchor_id": str((explanation_block or {}).get("anchor_id") or "").strip(),
    }


def _build_exact_equation_support_answer(record: dict) -> tuple[str, list[dict]]:
    rec = dict(record or {})
    equation_number = int(rec.get("equation_number") or 0)
    heading_path = str(rec.get("heading_path") or "").strip()
    equation_markdown = str(rec.get("equation_markdown") or "").strip()
    explanation_text = str(rec.get("explanation_text") or "").strip()
    leadin_text = str(rec.get("leadin_text") or "").strip()

    lines: list[str] = []
    intro = f"Equation ({int(equation_number)}) is stated"
    if heading_path:
        intro += f" in {heading_path}:"
    else:
        intro += " in the paper:"
    lines.append(intro)
    if leadin_text:
        lines.append(f"The lead-in sentence says: {leadin_text}")
    if equation_markdown:
        lines.append("")
        lines.append(equation_markdown)

    support_resolution: list[dict] = []
    if equation_markdown:
        support_resolution.append(
            {
                "source_path": str(rec.get("source_path") or "").strip(),
                "block_id": str(rec.get("equation_block_id") or "").strip(),
                "anchor_id": str(rec.get("equation_anchor_id") or "").strip(),
                "heading_path": heading_path,
                "locate_anchor": equation_markdown,
                "claim_type": "formula_claim",
                "cite_policy": "locate_only",
                "segment_text": equation_markdown,
                "segment_index": -1,
                "equation_number": int(equation_number or 0),
            }
        )

    if explanation_text:
        explanation_line = f"The variable definitions appear immediately after the equation: {explanation_text}"
        lines.append("")
        lines.append(explanation_line)
        support_resolution.append(
            {
                "source_path": str(rec.get("source_path") or "").strip(),
                "block_id": str(rec.get("explanation_block_id") or "").strip(),
                "anchor_id": str(rec.get("explanation_anchor_id") or "").strip(),
                "heading_path": heading_path,
                "locate_anchor": explanation_line,
                "claim_type": "equation_explanation_claim",
                "cite_policy": "locate_only",
                "segment_text": explanation_line,
                "segment_index": -1,
                "equation_number": int(equation_number or 0),
            }
        )

    return "\n".join(lines).strip(), support_resolution


_PAPER_GUIDE_CITATION_EXACT_SUPPORT_RE = re.compile(
    r"(?i)("
    r"where\s+is\s+that\s+stated\s+exactly|where\s+exactly|exact\s+supporting\s+part|"
    r"exact\s+supporting\s+sentence(?:s)?|exact\s+supporting\s+sentence\(s\)|"
    r"exact\s+sentence(?:s)?|supporting\s+sentence(?:s)?|point\s+me\s+to|"
    r"\u5f15\u7528\u7f16\u53f7|\u53c2\u8003\u6587\u732e\u7f16\u53f7|\u6587\u5185\u5f15\u7528|\u6587\u5185\u53c2\u8003|"
    r"\u539f\u6587.*?(?:\u54ea\u91cc|\u54ea\u513f).*?(?:\u5199|\u8bf4|\u63d0\u5230)|"
    r"(?:\u54ea\u91cc|\u54ea\u513f).{0,6}\u660e\u786e.{0,10}(?:\u5199|\u8bf4|\u63d0\u5230)|"
    r"\u7ed9\u51fa.*?(?:\u539f\u6587|\u539f\u53e5).*?(?:\u652f\u6301|\u8bc1\u636e)|"
    r"\u53ef\u5b9a\u4f4d.*?(?:\u539f\u6587|\u652f\u6301|\u8bc1\u636e)"
    r")"
)


def _paper_guide_prompt_requests_exact_citation_support(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    return bool(_PAPER_GUIDE_CITATION_EXACT_SUPPORT_RE.search(q))


_PAPER_GUIDE_FIGURE_CAPTION_EXACT_SUPPORT_RE = re.compile(
    r"(?i)("
    r"exact\s+supporting\s+caption\s+clause|caption\s+clause|exact\s+caption|"
    r"\u56fe\u6ce8.*?(?:\u539f\u6587|\u539f\u53e5)|"
    r"(?:\u539f\u6587|\u539f\u53e5).*?\u56fe\u6ce8|"
    r"\u56fe\u6ce8.*?\u53e5"
    r")"
)


def _paper_guide_prompt_requests_exact_figure_caption_support(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    return bool(_PAPER_GUIDE_FIGURE_CAPTION_EXACT_SUPPORT_RE.search(q))


def _extract_caption_panel_clause(text: str, *, panel_letter: str) -> str:
    raw = str(text or "").strip()
    panel = str(panel_letter or "").strip().lower()
    if (not raw) or (not panel) or (panel < "a") or (panel > "z"):
        return ""
    # Extract the panel clause from a caption block. Support common styles:
    # - "(f) ...; (g) ..." (parenthesized)
    # - "**f** ..." (bold markdown)
    # - "f The ..." / "f ..." (plain, often after punctuation or at start)
    markers: list[tuple[int, str]] = []
    for m in re.finditer(r"(?i)\(\s*([a-z])\s*\)", raw):
        markers.append((int(m.start()), str(m.group(1) or "").strip().lower()))
    for m in re.finditer(r"(?i)\*\*\s*([a-z])\s*\*\*", raw):
        markers.append((int(m.start()), str(m.group(1) or "").strip().lower()))
    # Plain markers like "a The ..." at the start or after punctuation.
    for m in re.finditer(r"(?im)(?:^|[.;:])\s*([a-z])\s+(?=[A-Z])", raw):
        markers.append((int(m.start(1)), str(m.group(1) or "").strip().lower()))

    if not markers:
        return ""
    markers.sort(key=lambda it: (int(it[0]), str(it[1])))

    # Pick the first occurrence of the requested panel letter.
    start = -1
    for pos, letter in markers:
        if letter == panel:
            start = int(pos)
            break
    if start < 0:
        return ""
    end = -1
    for pos, _letter in markers:
        if int(pos) > start:
            end = int(pos)
            break

    clause = raw[start:end].strip() if end > 0 else raw[start:].strip()
    clause = re.sub(r"\s+", " ", clause).strip()
    return clause[:900]


def _extract_caption_clause_superscript_ref_nums(text: str, *, max_nums: int = 6) -> list[int]:
    """
    Extract numeric citation-like superscripts from a caption clause.

    Conservative heuristic: ignore a superscript if the preceding non-space character is a digit
    (to avoid treating 10^{15} as a reference).
    """
    src = str(text or "")
    if not src:
        return []
    try:
        limit = max(1, int(max_nums))
    except Exception:
        limit = 6
    out: list[int] = []
    seen: set[int] = set()
    for m in re.finditer(r"\^\{\s*([0-9][0-9,\s\-\u2013\u2014\u2212]*[0-9])\s*\}", src):
        caret_pos = int(m.start(0))
        prev = ""
        j = caret_pos - 1
        while j >= 0:
            ch = src[j]
            if ch.isspace():
                j -= 1
                continue
            prev = ch
            break
        if prev.isdigit():
            continue
        spec = str(m.group(1) or "").strip()
        if not spec:
            continue
        for n in parse_ref_num_set(spec, max_items=12):
            if n <= 0 or n in seen:
                continue
            seen.add(n)
            out.append(int(n))
            if len(out) >= limit:
                return out
    return out


def _resolve_exact_figure_panel_caption_support_from_source(
    source_path: str,
    *,
    prompt: str,
    db_dir: Path | None,
) -> dict:
    src = str(source_path or "").strip()
    q = str(prompt or "").strip()
    if (not src) or (not q):
        return {}
    panel_letters = _extract_prompt_panel_letters(q)
    if not panel_letters:
        return {}
    panel = str(panel_letters[0] or "").strip().lower()
    fig_num = int(_extract_figure_number(q) or 0)
    if fig_num <= 0:
        # If the prompt doesn't specify a figure number, don't guess.
        return {}

    md_path = _resolve_paper_guide_md_path(src, db_dir=db_dir)
    if not isinstance(md_path, Path) or (not md_path.exists()):
        return {}

    best: dict = {}
    best_score = float("-inf")
    fig_marker_re = re.compile(rf"(?i)\b(?:figure|fig)\.?\s*{int(fig_num)}\b")
    caption_start_re = re.compile(r"(?i)^\s*(?:\*\*)?\s*(?:figure|fig)\.?\s*(\d{1,4})\b")
    scope_left = 0
    scope_heading = ""
    blocks = list(load_source_blocks(md_path) or [])
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = str(block.get("kind") or "").strip().lower()
        if kind not in {"paragraph", "list_item", "blockquote"}:
            continue
        text = str(block.get("raw_text") or block.get("text") or "").strip()
        if not text:
            continue
        heading = str(block.get("heading_path") or "").strip()

        # If we're entering another figure caption, close the current scope.
        m_other = caption_start_re.match(text)
        if m_other:
            try:
                other_num = int(m_other.group(1))
            except Exception:
                other_num = 0
            if other_num > 0 and other_num != int(fig_num):
                scope_left = 0
                scope_heading = ""

        # Caption blocks are often split, e.g.:
        # "Figure 6. ..." (one paragraph) then "a ... b ..." (next paragraph).
        # Track a short window after we see the "Figure N" marker in any block.
        if fig_marker_re.search(text):
            scope_left = max(scope_left, 4)
            scope_heading = heading

        in_scope = bool(fig_marker_re.search(text)) or bool(
            scope_left > 0 and (not scope_heading or heading == scope_heading)
        )
        if scope_left > 0:
            scope_left -= 1
        if not in_scope:
            continue

        clause = _extract_caption_panel_clause(text, panel_letter=panel)
        if not clause:
            continue
        # Score by matching prompt cue tokens in the clause (avoid paper-title bleed).
        q_tokens = [tok for tok in _paper_guide_cue_tokens(q) if tok not in {"figure", "panel", "caption"}]
        clause_low = clause.lower()
        shared = [tok for tok in q_tokens if tok and tok in clause_low]
        score = 10.0 + min(10.0, 2.0 * float(len(shared)))
        # Prefer shorter, more clause-like excerpts over whole-captions.
        score += max(0.0, 6.0 - (0.004 * float(len(clause))))
        if score > best_score:
            best_score = score
            best = {
                "source_path": src,
                "block_id": str(block.get("block_id") or "").strip(),
                "anchor_id": str(block.get("anchor_id") or "").strip(),
                "heading_path": heading,
                "locate_anchor": clause,
                "claim_type": "figure_panel",
                "cite_policy": "locate_only",
                "segment_text": clause,
                "segment_index": -1,
                "figure_number": int(fig_num),
                "panel_letters": [panel],
            }
    return best if best_score >= 8.0 else {}


def _resolve_exact_citation_lookup_support_from_source(
    source_path: str,
    *,
    prompt: str,
    db_dir: Path | None,
) -> dict:
    src = str(source_path or "").strip()
    q = str(prompt or "").strip()
    if (not src) or (not q):
        return {}
    md_path = _resolve_paper_guide_md_path(src, db_dir=db_dir)

    hits = _paper_guide_targeted_source_block_hits(
        bound_source_path=src,
        prompt=q,
        db_dir=db_dir,
        limit=10,
        citation_lookup_query_tokens=_paper_guide_citation_lookup_query_tokens,
        citation_lookup_signal_score=_paper_guide_citation_lookup_signal_score,
        resolve_support_slot_block=None,
    )
    explicit_ref_list_request = bool(
        re.search(r"(?i)\b(?:reference\s+list|works?\s+cited|bibliography)\b", q)
    )
    query_tokens = set(_paper_guide_citation_lookup_query_tokens(q))
    query_focus_tokens: set[str] = set()
    for tok in re.findall(r"\b[A-Za-z]+\d{2,}\b", q):
        query_focus_tokens.add(str(tok or "").strip().lower())
    for tok in re.findall(r"\b[A-Z]{3,}\b", q):
        query_focus_tokens.add(str(tok or "").strip().lower())
    if re.search(r"(?i)\bpascal\b", q):
        query_focus_tokens.add("pascal")
    if re.search(r"(?i)\bvoc\b", q):
        query_focus_tokens.add("voc")
    focus_term = ""
    focus_match = re.search(r"(?i)\bcite(?:d|s)?\b[^\n]{0,80}\bfor\b\s+(?:the\s+)?([A-Za-z][A-Za-z0-9-]{1,36})", q)
    if focus_match:
        cand_focus = str(focus_match.group(1) or "").strip().lower()
        if cand_focus and cand_focus not in {"the", "a", "an", "this", "that", "it"}:
            focus_term = cand_focus
    focus_entity_tokens: set[str] = set(tok for tok in query_focus_tokens if any(ch.isdigit() for ch in tok))
    if "pascal" in query_focus_tokens:
        focus_entity_tokens.add("pascal")
    if "voc" in query_focus_tokens:
        focus_entity_tokens.add("voc")
    best_fragment = ""
    best_heading = ""
    best_refs: list[int] = []
    best_hit_meta: dict = {}
    best_score = float("-inf")
    seen_fragments: set[str] = set()
    scored_fragments: list[dict] = []

    def _consider_fragment(*, heading: str, frag: str, meta: dict) -> None:
        nonlocal best_fragment, best_heading, best_refs, best_hit_meta, best_score
        refs = _extract_inline_reference_numbers(frag, max_candidates=6)
        if not refs:
            return
        frag_low = str(frag or "").strip().lower()
        focus_shared = [tok for tok in query_focus_tokens if tok in frag_low]
        entity_shared = [tok for tok in focus_entity_tokens if tok in frag_low]
        score = _paper_guide_citation_lookup_signal_score(
            prompt=q,
            heading=heading,
            text=frag,
            inline_refs=refs,
            explicit_ref_list_request=explicit_ref_list_request,
        )
        score += min(6.0, 2.0 * float(len(refs)))
        if query_tokens:
            shared = set(_paper_guide_cue_tokens(frag)).intersection(query_tokens)
            score += min(12.0, 2.4 * float(len(shared)))
        if focus_shared:
            score += min(16.0, 4.0 * float(len(focus_shared)))
        elif query_focus_tokens:
            score -= 4.0
        if entity_shared:
            score += min(24.0, 8.0 * float(len(entity_shared)))
        elif focus_entity_tokens:
            score -= 12.0
        scored_fragments.append(
            {
                "heading": str(heading or "").strip(),
                "frag": str(frag or "").strip(),
                "refs": list(refs),
                "meta": dict(meta or {}),
                "score": float(score),
                "entity_shared": set(entity_shared),
            }
        )
        if score > best_score:
            best_score = score
            best_fragment = frag
            best_heading = heading
            best_refs = refs
            best_hit_meta = dict(meta or {})

    for hit in hits:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        heading = str(meta.get("heading_path") or meta.get("top_heading") or "").strip()
        text = str(hit.get("text") or "").strip()
        if not text:
            continue
        for frag in _paper_guide_citation_lookup_fragments(text):
            frag_key = normalize_inline_markdown(str(frag or "")).strip().lower()
            if not frag_key or frag_key in seen_fragments:
                continue
            seen_fragments.add(frag_key)
            _consider_fragment(heading=heading, frag=frag, meta=meta)

    # Always include a direct source-block scan for exact citation support.
    # Retrieval-only candidates can miss split lines around captions/figures.
    if isinstance(md_path, Path) and md_path.exists():
        try:
            blocks = list(load_source_blocks(md_path) or [])
        except Exception:
            blocks = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            kind = str(block.get("kind") or "").strip().lower()
            if kind not in {"paragraph", "list_item", "blockquote"}:
                continue
            heading = str(block.get("heading_path") or "").strip()
            text = str(block.get("raw_text") or block.get("text") or "").strip()
            if not text:
                continue
            meta = {
                "block_id": str(block.get("block_id") or "").strip(),
                "anchor_id": str(block.get("anchor_id") or "").strip(),
                "heading_path": heading,
            }
            fragments = _paper_guide_citation_lookup_fragments(text)
            combined_fragments = list(fragments)
            max_window = min(3, len(fragments))
            for window in range(2, max_window + 1):
                for idx in range(len(fragments) - window + 1):
                    combo = " ".join(str(fragments[idx + off] or "").strip() for off in range(window)).strip()
                    if combo:
                        combined_fragments.append(combo)
            for frag in combined_fragments:
                frag_key = normalize_inline_markdown(str(frag or "")).strip().lower()
                if not frag_key or frag_key in seen_fragments:
                    continue
                seen_fragments.add(frag_key)
                _consider_fragment(heading=heading, frag=frag, meta=meta)

    # If the query contains focus entities (e.g. VOC2007 / VOC2012), merge top fragments
    # that each cover different focus entities so split-line attributions can be recovered.
    if focus_entity_tokens and scored_fragments:
        target_focus = 2 if len(focus_entity_tokens) >= 2 else 1
        ranked = sorted(
            [row for row in scored_fragments if set(row.get("entity_shared") or set())],
            key=lambda row: (
                len(set(row.get("entity_shared") or set())),
                float(row.get("score") or 0.0),
            ),
            reverse=True,
        )
        merged_focus: set[str] = set()
        merged_refs: list[int] = []
        merged_frags: list[str] = []
        merged_heading = ""
        merged_meta: dict = {}
        for row in ranked:
            row_focus = set(row.get("entity_shared") or set())
            if merged_refs and not (row_focus - merged_focus):
                continue
            merged_focus.update(row_focus)
            frag_text = str(row.get("frag") or "").strip()
            if frag_text:
                merged_frags.append(frag_text)
            local_refs = _select_paper_guide_local_citation_lookup_refs(frag_text, prompt=q, max_candidates=4)
            use_refs = [int(n) for n in (local_refs or list(row.get("refs") or [])) if int(n) > 0]
            for n in use_refs:
                if n not in merged_refs:
                    merged_refs.append(n)
            if (not merged_heading) and str(row.get("heading") or "").strip():
                merged_heading = str(row.get("heading") or "").strip()
            if (not merged_meta) and isinstance(row.get("meta"), dict):
                merged_meta = dict(row.get("meta") or {})
            if len(merged_focus) >= target_focus and len(merged_refs) >= target_focus:
                break
        if len(merged_focus) >= target_focus and len(merged_refs) >= target_focus:
            merged_anchor = " ".join(
                part
                for part in dict.fromkeys(
                    str(part or "").strip()
                    for part in merged_frags
                )
                if part
            ).strip()
            if merged_anchor:
                best_fragment = merged_anchor
                best_heading = merged_heading or best_heading
                best_refs = list(merged_refs[:4])
                if merged_meta:
                    best_hit_meta = dict(merged_meta)
                best_score = max(best_score, 24.0)

    if (not best_fragment) or (not best_refs) or best_score < 6.0:
        return {}

    local_refs = _select_paper_guide_local_citation_lookup_refs(best_fragment, prompt=q, max_candidates=4)
    if local_refs and ((len(best_refs) <= 1) or (not focus_entity_tokens)):
        best_refs = local_refs

    # Some papers phrase attributions as a long sentence starting with a subordinate clause
    # ("When ..., ... (METHOD) ^{[n]} ..."). Those "When ..." leads can be treated as rhetorical
    # and occasionally get dropped during provenance segment selection. Prefer a later clause
    # that still contains the method term + inline ref so locate can bind reliably.
    frag0 = str(best_fragment or "").strip()
    low0 = frag0.lower()
    if low0.startswith(("when ", "while ", "if ")):
        ref_pos = frag0.find("$^{[")
        if ref_pos < 0:
            ref_pos = frag0.find("^{[")
        search_end = ref_pos if ref_pos > 0 else min(len(frag0), 320)
        comma = frag0.rfind(",", 0, search_end)
        if comma > 0:
            cand = frag0[comma + 1 :].strip()
            if len(cand) >= 40 and _extract_inline_reference_numbers(cand, max_candidates=6):
                # Keep this trim only when the candidate still carries prompt-focus cue tokens.
                # Otherwise it can drop the method term itself (for example ADMM) and hurt locate.
                cand_low = cand.lower()
                if focus_term and (focus_term not in cand_low):
                    cand = ""
                if not cand:
                    pass
                cand_tokens = set(_paper_guide_cue_tokens(cand))
                if cand and ((not query_tokens) or cand_tokens.intersection(query_tokens)):
                    best_fragment = cand

    return {
        "source_path": src,
        "block_id": str(best_hit_meta.get("block_id") or "").strip(),
        "anchor_id": str(best_hit_meta.get("anchor_id") or "").strip(),
        "heading_path": str(best_heading or best_hit_meta.get("heading_path") or "").strip(),
        "locate_anchor": str(best_fragment or "").strip(),
        "claim_type": "prior_work",
        "cite_policy": "prefer_ref",
        "segment_text": str(best_fragment or "").strip(),
        "segment_index": -1,
        "ref_nums": list(best_refs[:4]),
    }


def _force_exact_method_support_surface(
    answer: str,
    *,
    prompt: str,
    source_path: str,
    db_dir: Path | None,
    support_resolution: list[dict],
    support_slots: list[dict],
) -> str:
    text = str(answer or "").strip()
    if not _paper_guide_prompt_requests_exact_method_support(prompt):
        return text
    best = None
    for rec in list(support_resolution or []):
        if not isinstance(rec, dict):
            continue
        if str(rec.get("locate_anchor") or "").strip():
            best = rec
            break
    if best is None:
        for slot in list(support_slots or []):
            if not isinstance(slot, dict):
                continue
            if str(slot.get("locate_anchor") or "").strip():
                best = slot
                break
    locate_anchor = ""
    heading_path = ""
    if isinstance(best, dict):
        locate_anchor = str(best.get("locate_anchor") or "").strip()
        heading_path = str(best.get("heading_path") or best.get("heading") or "").strip()
    if not locate_anchor:
        focus_terms = _extract_paper_guide_method_focus_terms(prompt)
        excerpt = _extract_bound_paper_method_focus(
            source_path,
            db_dir=db_dir,
            focus_terms=focus_terms,
        )
        excerpt = _extract_paper_guide_method_detail_excerpt(
            excerpt,
            focus_terms=focus_terms,
        )
        locate_anchor = str(excerpt or "").strip()
        heading_path = heading_path or "retrieved method evidence"
    # For "exact supporting part" method prompts, the best UX is to surface the exact sentence from source blocks
    # even if the initial retrieved slot was off-target (common when BM25 hits are dominated by the title/intro).
    exact_record = _resolve_exact_method_support_from_source(
        source_path,
        prompt=prompt,
        db_dir=db_dir,
    )
    if exact_record:
        exact_anchor = str(exact_record.get("locate_anchor") or "").strip()
        exact_heading = str(exact_record.get("heading_path") or "").strip()
        if exact_anchor:
            locate_anchor = exact_anchor
            heading_path = exact_heading or heading_path or "retrieved method evidence"
    if not locate_anchor:
        return text
    if locate_anchor.lower() in text.lower():
        return text
    if heading_path:
        return f"The paper states this explicitly in {heading_path}:\n> {locate_anchor}"
    return f"The paper states this explicitly:\n> {locate_anchor}"


def _apply_paper_guide_answer_postprocess(
    answer: str,
    *,
    paper_guide_mode: bool,
    prompt: str,
    prompt_for_user: str,
    prompt_family: str,
    special_focus_block: str,
    focus_source_path: str,
    direct_source_path: str,
    bound_source_path: str,
    db_dir: Path | None,
    answer_hits: list[dict],
    support_slots: list[dict],
    cards: list[dict],
    locked_citation_source: dict | None,
) -> tuple[str, list[dict]]:
    text = str(answer or "").strip()
    if not paper_guide_mode:
        return text, []

    support_resolution: list[dict] = []
    source_path = str(focus_source_path or direct_source_path or bound_source_path or "").strip()
    family = str(prompt_family or "").strip().lower()
    prompt_text = str(prompt_for_user or prompt or "").strip()

    if _paper_guide_prompt_requests_doc_map(prompt_text):
        doc_source_path = str(bound_source_path or source_path or "").strip()
        recs = _resolve_doc_map_records_from_source(
            doc_source_path,
            prompt=prompt_text,
            db_dir=db_dir,
            max_items=16,
        )
        if recs:
            lines: list[str] = []
            lines.append("Doc map (verbatim anchors by section):")
            lines.append("")
            for i, rec in enumerate(recs, start=1):
                heading_path = str(rec.get("heading_path") or "").strip() or "Unheaded section"
                anchor = str(rec.get("locate_anchor") or "").strip()
                if not anchor:
                    continue
                lines.append(f"{int(i)}. {heading_path}")
                lines.append(f"> {anchor}")
                lines.append("")
            out = "\n".join(lines).rstrip()
            out = _sanitize_paper_guide_answer_for_user(
                out,
                has_hits=bool(answer_hits),
                prompt=prompt_text,
                prompt_family=family or "overview",
            )
            return out, list(recs)

    effective_family = family or _paper_guide_prompt_family(prompt_text)
    if effective_family == "equation" and _paper_guide_prompt_requests_exact_equation_support(prompt_text):
        equation_source_path = str(bound_source_path or source_path or "").strip()
        rec = _resolve_exact_equation_support_from_source(
            equation_source_path,
            prompt=prompt_text,
            db_dir=db_dir,
        )
        equation_markdown = str((rec or {}).get("equation_markdown") or "").strip()
        if equation_markdown:
            out, support_resolution = _build_exact_equation_support_answer(rec)
            out = _sanitize_paper_guide_answer_for_user(
                out,
                has_hits=bool(answer_hits),
                prompt=prompt_text,
                prompt_family=effective_family,
            )
            return out, list(support_resolution or [])

    if effective_family == "citation_lookup" and _paper_guide_prompt_requests_exact_citation_support(prompt_text):
        citation_source_path = str(bound_source_path or source_path or "").strip()
        rec = _resolve_exact_citation_lookup_support_from_source(
            citation_source_path,
            prompt=prompt_text,
            db_dir=db_dir,
        )
        locate_anchor = str((rec or {}).get("locate_anchor") or "").strip()
        heading_path = str((rec or {}).get("heading_path") or "").strip()
        ref_nums = [int(n) for n in list((rec or {}).get("ref_nums") or []) if int(n) > 0]
        if locate_anchor and (not ref_nums):
            ref_nums = _extract_inline_reference_numbers(locate_anchor, max_candidates=4)
        if locate_anchor and ref_nums:
            ref_label = ", ".join(f"[{int(n)}]" for n in ref_nums[:4])
            lines = [f"The paper cites {ref_label} for this point."]
            if heading_path:
                lines.append(f"This is stated in {heading_path}:")
            else:
                lines.append("This is stated explicitly in the paper:")
            lines.append(f"> {locate_anchor}")
            out = "\n".join(lines).strip()
            rec_out = dict(rec or {})
            if ref_nums and not list(rec_out.get("candidate_refs") or []):
                rec_out["candidate_refs"] = list(ref_nums[:4])
            if ref_nums and int(ref_nums[0]) > 0 and int(rec_out.get("resolved_ref_num") or 0) <= 0:
                rec_out["resolved_ref_num"] = int(ref_nums[0])
            out = _sanitize_paper_guide_answer_for_user(
                out,
                has_hits=bool(answer_hits),
                prompt=prompt_text,
                prompt_family=effective_family,
            )
            return out, [rec_out]

    # Exact-support figure prompts should bind the jump target to the specific caption clause (panel),
    # not a best-effort fuzzy match to a nearby paragraph or the figure asset block.
    if effective_family == "figure_walkthrough" and _paper_guide_prompt_requests_exact_figure_caption_support(prompt_text):
        fig_source_path = str(bound_source_path or source_path or "").strip()
        rec = _resolve_exact_figure_panel_caption_support_from_source(
            fig_source_path,
            prompt=prompt_text,
            db_dir=db_dir,
        )
        locate_anchor = str((rec or {}).get("locate_anchor") or "").strip()
        heading_path = str((rec or {}).get("heading_path") or "").strip()
        panel_letters = [
            str(ch or "").strip().lower()
            for ch in list((rec or {}).get("panel_letters") or [])
            if str(ch or "").strip()
        ]
        try:
            fig_num = int((rec or {}).get("figure_number") or 0)
        except Exception:
            fig_num = 0
        if locate_anchor:
            prefix = "Figure caption"
            if fig_num > 0 and panel_letters:
                prefix = f"Figure {int(fig_num)} caption for panel ({panel_letters[0]})"
            elif fig_num > 0:
                prefix = f"Figure {int(fig_num)} caption"
            lines = [f"{prefix} states:"]
            if heading_path:
                lines.append(f"Section: {heading_path}")
            lines.append(f"> {locate_anchor}")
            # Surface obvious reference superscripts in the clause (for example SPC$^{15}$) as structured cites
            # so the UI can show the reference entry instead of silently dropping it.
            ref_nums = _extract_caption_clause_superscript_ref_nums(locate_anchor, max_nums=4)
            if ref_nums:
                # Use in-paper numeric citation format so it can be rendered into hover meta
                # without leaking raw structured cite markers into the user-facing markdown.
                label = ", ".join(f"[{int(n)}]" for n in ref_nums if int(n) > 0)
                if label:
                    lines.append(f"References in this clause: {label}")
            out = "\n".join(lines).strip()
            rec_out = dict(rec or {})
            if ref_nums and not list(rec_out.get("candidate_refs") or []):
                rec_out["candidate_refs"] = list(ref_nums)
                if len(ref_nums) == 1:
                    rec_out["resolved_ref_num"] = int(ref_nums[0])
            out = _sanitize_paper_guide_answer_for_user(
                out,
                has_hits=bool(answer_hits),
                prompt=prompt_text,
                prompt_family=effective_family,
            )
            return out, [rec_out]

    # Exact-support method prompts should not depend on model output formatting or SUPPORT marker injection.
    # Instead, deterministically surface the best matching sentence from source blocks and return a single
    # locate-only support record so the locate gate can bind the jump target.
    if effective_family in {"method", "reproduce"} and _paper_guide_prompt_requests_exact_method_support(prompt_text):
        method_source_path = str(bound_source_path or source_path or "").strip()
        rec = _resolve_exact_method_support_from_source(
            method_source_path,
            prompt=prompt_text,
            db_dir=db_dir,
        )
        locate_anchor = str((rec or {}).get("locate_anchor") or "").strip()
        heading_path = str((rec or {}).get("heading_path") or "").strip()
        if locate_anchor:
            if heading_path:
                out = f"The paper states this explicitly in {heading_path}:\n> {locate_anchor}"
            else:
                out = f"The paper states this explicitly:\n> {locate_anchor}"
            rec_out = dict(rec or {})
            rec_out.setdefault("segment_text", locate_anchor)
            rec_out["segment_index"] = -1
            out = _sanitize_paper_guide_answer_for_user(
                out,
                has_hits=bool(answer_hits),
                prompt=prompt_text,
                prompt_family=effective_family,
            )
            return out, [rec_out]

    if special_focus_block:
        text = _repair_paper_guide_focus_answer_generic(
            text,
            prompt=prompt_text,
            prompt_family=effective_family,
            special_focus_block=special_focus_block,
            source_path=source_path,
            db_dir=db_dir,
        )
    elif effective_family == "figure_walkthrough":
        figure_num = _requested_figure_number(prompt_text, answer_hits)
        if figure_num > 0:
            figure_caption = _extract_bound_paper_figure_caption(
                source_path,
                figure_num=figure_num,
                db_dir=db_dir,
            )
            if figure_caption:
                text = _repair_paper_guide_focus_answer_generic(
                    text,
                    prompt=prompt_text,
                    prompt_family=effective_family,
                    special_focus_block=f"Paper-guide figure focus:\n- Caption excerpt:\n{figure_caption}",
                    source_path=source_path,
                    db_dir=db_dir,
                )

    text = _inject_paper_guide_support_markers(
        text,
        support_slots=support_slots,
        prompt_family=effective_family,
    )
    text, support_resolution = _resolve_paper_guide_support_markers(
        text,
        support_slots=support_slots,
        prompt_family=effective_family,
        db_dir=db_dir,
    )
    if effective_family == "citation_lookup":
        normalized_support: list[dict] = []
        for rec in list(support_resolution or []):
            if not isinstance(rec, dict):
                continue
            rec_out = dict(rec)
            try:
                resolved_ref_num = int(rec_out.get("resolved_ref_num") or 0)
            except Exception:
                resolved_ref_num = 0
            if resolved_ref_num <= 0:
                candidate_refs: list[int] = []
                for item in list(rec_out.get("candidate_refs") or []):
                    try:
                        n = int(item)
                    except Exception:
                        continue
                    if n > 0 and n not in candidate_refs:
                        candidate_refs.append(n)
                for item in list(rec_out.get("support_ref_candidates") or []):
                    try:
                        n = int(item)
                    except Exception:
                        continue
                    if n > 0 and n not in candidate_refs:
                        candidate_refs.append(n)
                if not candidate_refs:
                    locate_anchor = str(rec_out.get("locate_anchor") or rec_out.get("segment_text") or "").strip()
                    if locate_anchor:
                        candidate_refs = _select_paper_guide_local_citation_lookup_refs(
                            locate_anchor,
                            prompt=prompt_text,
                            max_candidates=4,
                        )
                    if not candidate_refs and locate_anchor:
                        candidate_refs = _extract_inline_reference_numbers(locate_anchor, max_candidates=4)
                if candidate_refs:
                    rec_out["candidate_refs"] = [int(n) for n in candidate_refs if int(n) > 0][:6]
                    rec_out["resolved_ref_num"] = int(rec_out["candidate_refs"][0])
            normalized_support.append(rec_out)
        support_resolution = normalized_support
    text = _inject_paper_guide_focus_citations(
        text,
        special_focus_block=special_focus_block,
        source_path=source_path,
        prompt_family=effective_family,
        prompt=prompt_text,
        db_dir=db_dir,
    )
    text = _inject_paper_guide_card_citations(
        text,
        cards=cards,
        prompt_family=effective_family,
    )
    text = _drop_paper_guide_locate_only_line_citations(
        text,
        support_resolution=support_resolution,
    )
    if locked_citation_source and effective_family == "method":
        text = _promote_paper_guide_numeric_reference_citations(
            text,
            locked_source=locked_citation_source,
        )
    if effective_family in {"method", "reproduce"}:
        # For exact-method-support questions, always resolve against the bound paper path if available.
        # (focus_source_path can be a PDF name/path in some flows and may not map back to the md reliably.)
        method_source_path = str(bound_source_path or source_path or "").strip()
        exact_source_support = (
            _resolve_exact_method_support_from_source(
                method_source_path,
                prompt=prompt_text,
                db_dir=db_dir,
            )
            if _paper_guide_prompt_requests_exact_method_support(prompt_text)
            else {}
        )
        text = _force_exact_method_support_surface(
            text,
            prompt=prompt_text,
            source_path=method_source_path,
            db_dir=db_dir,
            support_resolution=support_resolution,
            support_slots=support_slots,
        )
        if exact_source_support and not any(
            str(rec.get("locate_anchor") or "").strip()
            for rec in list(support_resolution or [])
            if isinstance(rec, dict)
        ):
            rec_out = dict(exact_source_support)
            # This record is synthesized post-answer and should be matched to the quote segment by surface,
             # not by a default segment index (0), which would bind it to the wrong segment.
            rec_out.setdefault("segment_index", -1)
            support_resolution.append(rec_out)
    elif effective_family == "box_only":
        text = _surface_plain_text_box_formula(
            text,
            support_resolution=support_resolution,
        )
    text = _sanitize_paper_guide_answer_for_user(
        text,
        has_hits=bool(answer_hits),
        prompt=prompt_text,
        prompt_family=effective_family,
    )
    return text, support_resolution
