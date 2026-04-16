from __future__ import annotations

import re
from pathlib import Path

from kb.inpaper_citation_grounding import parse_ref_num_set
from kb.paper_guide_focus import (
    _extract_bound_paper_method_focus,
    _extract_paper_guide_method_focus_terms,
    _extract_paper_guide_special_focus_excerpt,
)
from kb.paper_guide.grounder import (
    _PAPER_GUIDE_CITE_STRONG_TOKENS,
    _extract_inline_reference_numbers,
    _paper_guide_cue_tokens,
)
from kb.paper_guide_prompting import _PAPER_GUIDE_CITATION_LOOKUP_PROMPT_RE

_CITE_SINGLE_BRACKET_RE = re.compile(
    r"(?<!\[)\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\](?!\])",
    re.IGNORECASE,
)
_CITE_CANON_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\]\]",
    re.IGNORECASE,
)
_PAPER_GUIDE_NUMERIC_REF_RE = re.compile(
    r"(?i)(\b(?:reference|references|ref(?:s)?\.?)\s*)(\[(\d{1,4}(?:\s*(?:[\-–—−,])\s*\d{1,4})*)\])"
)


def _collect_paper_guide_candidate_refs_by_source(
    cards: list[dict],
    *,
    focus_source_path: str = "",
    special_focus_block: str = "",
    prompt_family: str = "",
    prompt: str = "",
    db_dir: Path | None = None,
    extract_special_focus_excerpt=None,
    extract_bound_method_focus=None,
    extract_method_focus_terms=None,
) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}

    def _add(src: str, nums: list[int]) -> None:
        source = str(src or "").strip()
        if (not source) or (not nums):
            return
        bucket = out.setdefault(source, [])
        seen = set(int(n) for n in bucket if int(n) > 0)
        for item in nums:
            try:
                n = int(item)
            except Exception:
                continue
            if n <= 0 or n in seen:
                continue
            seen.add(n)
            bucket.append(n)

    for card in cards or []:
        if not isinstance(card, dict):
            continue
        source_path = str(card.get("source_path") or focus_source_path or "").strip()
        refs: list[int] = []
        for item in list(card.get("candidate_refs") or []):
            try:
                n = int(item)
            except Exception:
                continue
            if n > 0:
                refs.append(n)
        if not refs:
            refs = _extract_inline_reference_numbers(
                " ".join(
                    str(part or "").strip()
                    for part in (
                        card.get("cue"),
                        card.get("heading"),
                        card.get("snippet"),
                    )
                    if str(part or "").strip()
                ),
                max_candidates=6,
            )
        _add(source_path, refs)

    extract_focus = extract_special_focus_excerpt or _extract_paper_guide_special_focus_excerpt
    focus_excerpt = str(extract_focus(str(special_focus_block or "")) or "").strip()
    family = str(prompt_family or "").strip().lower()
    if (not _extract_inline_reference_numbers(focus_excerpt, max_candidates=2)) and family == "method":
        bound_focus_getter = extract_bound_method_focus or _extract_bound_paper_method_focus
        focus_terms_getter = extract_method_focus_terms or _extract_paper_guide_method_focus_terms
        bound_focus_excerpt = bound_focus_getter(
            str(focus_source_path or "").strip(),
            db_dir=db_dir,
            focus_terms=focus_terms_getter(prompt),
        )
        if bound_focus_excerpt:
            focus_excerpt = bound_focus_excerpt
    if focus_excerpt:
        _add(
            str(focus_source_path or "").strip(),
            _extract_inline_reference_numbers(focus_excerpt, max_candidates=6),
        )
    return out


def _inject_paper_guide_fallback_citations(
    answer: str,
    *,
    cards: list[dict],
    prompt_family: str = "",
    max_injections: int = 2,
) -> str:
    text = str(answer or "").strip()
    family = str(prompt_family or "").strip().lower()
    if (not text) or ("[[CITE:" in text) or family != "method":
        return text
    try:
        limit = max(1, int(max_injections))
    except Exception:
        limit = 2

    rules: list[dict[str, object]] = []
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        sid = str(card.get("sid") or "").strip()
        refs = []
        for item in list(card.get("candidate_refs") or []):
            try:
                refs.append(int(item))
            except Exception:
                continue
        if (not sid) or (not refs):
            continue
        cue_source = str(card.get("cue") or "").strip()
        if not cue_source:
            snippet = str(card.get("snippet") or "").strip()
            if re.search(r"\[\d{1,4}(?:\s*(?:[\-–—−,])\s*\d{1,4})*\]", snippet):
                cue_source = snippet
        tokens = _paper_guide_cue_tokens(cue_source)
        if len(tokens) < 2:
            continue
        rules.append(
            {
                "marker": f"[[CITE:{sid}:{refs[0]}]]",
                "tokens": tokens,
            }
        )

    if not rules:
        return text

    lines = text.splitlines()
    used_markers: set[str] = set()
    injected = 0
    for idx, line in enumerate(lines):
        if injected >= limit:
            break
        raw = str(line or "")
        stripped = raw.strip()
        if (not stripped) or ("[[CITE:" in stripped):
            continue
        if re.match(r"^(Conclusion|Evidence|Limits|Next Steps|Abstract text|Chinese translation)\s*:?\s*$", stripped, flags=re.IGNORECASE):
            continue
        if stripped.startswith("|"):
            continue
        line_tokens = set(_paper_guide_cue_tokens(stripped))
        if len(line_tokens) < 2:
            continue
        best_rule: dict[str, object] | None = None
        best_score = 0
        for rule in rules:
            marker = str(rule.get("marker") or "").strip()
            if (not marker) or (marker in used_markers):
                continue
            shared = line_tokens.intersection(set(rule.get("tokens") or []))
            score = len(shared)
            if score > best_score:
                best_score = score
                best_rule = rule
        if not best_rule or best_score < 2:
            continue
        marker = str(best_rule.get("marker") or "").strip()
        if not marker:
            continue
        lines[idx] = raw.rstrip() + " " + marker
        used_markers.add(marker)
        injected += 1

    return "\n".join(lines).strip()


def _inject_paper_guide_focus_citations(
    answer: str,
    *,
    special_focus_block: str = "",
    source_path: str = "",
    prompt_family: str = "",
    prompt: str = "",
    db_dir: Path | None = None,
    cite_source_id=None,
    extract_special_focus_excerpt=None,
    extract_bound_method_focus=None,
    extract_method_focus_terms=None,
) -> str:
    text = str(answer or "").strip()
    family = str(prompt_family or "").strip().lower()
    if family != "method":
        return text
    if _PAPER_GUIDE_CITATION_LOOKUP_PROMPT_RE.search(str(prompt or "").strip()):
        return text
    extract_focus = extract_special_focus_excerpt or _extract_paper_guide_special_focus_excerpt
    focus_excerpt = str(extract_focus(str(special_focus_block or "")) or "").strip()
    src = str(source_path or "").strip()
    if (not text) or (not focus_excerpt) or (not src):
        return text
    refs = _extract_inline_reference_numbers(focus_excerpt, max_candidates=4)
    if (not refs) and family == "method":
        bound_focus_getter = extract_bound_method_focus or _extract_bound_paper_method_focus
        focus_terms_getter = extract_method_focus_terms or _extract_paper_guide_method_focus_terms
        bound_focus_excerpt = bound_focus_getter(
            src,
            db_dir=db_dir,
            focus_terms=focus_terms_getter(prompt),
        )
        if bound_focus_excerpt:
            focus_excerpt = bound_focus_excerpt
            refs = _extract_inline_reference_numbers(focus_excerpt, max_candidates=4)
    if not refs:
        return text
    sid = str(cite_source_id(src) if callable(cite_source_id) else "").strip()
    if not sid:
        return text
    marker = f"[[CITE:{sid}:{int(refs[0])}]]"
    if marker in text:
        return text
    rule_tokens = set(_paper_guide_cue_tokens(focus_excerpt))
    if len(rule_tokens) < 2:
        return text

    lines = text.splitlines()
    best_idx = -1
    best_score = float("-inf")
    for idx, line in enumerate(lines):
        raw = str(line or "")
        stripped = raw.strip()
        if (not stripped) or ("[[CITE:" in stripped):
            continue
        if re.match(r"^(Conclusion|Evidence|Limits|Next Steps|Abstract text|Chinese translation)\s*:?\s*$", stripped, flags=re.IGNORECASE):
            continue
        if stripped.startswith("|"):
            continue
        line_tokens = set(_paper_guide_cue_tokens(stripped))
        if not line_tokens:
            continue
        score = float(len(line_tokens.intersection(rule_tokens)))
        if family == "method" and re.search(r"implementation detail", stripped, flags=re.IGNORECASE):
            score += 3.0
        elif family == "figure_walkthrough" and re.search(r"caption anchor", stripped, flags=re.IGNORECASE):
            score += 2.0
        if score > best_score:
            best_score = score
            best_idx = idx
    min_score = 2.0
    if family in {"method", "compare", "figure_walkthrough"}:
        min_score = 1.0
    if best_idx < 0 or best_score < min_score:
        return text
    raw_line = str(lines[best_idx] or "")
    replaced = False

    def _replace_first_inline_ref(match: re.Match[str]) -> str:
        nonlocal replaced
        if replaced:
            return str(match.group(0) or "")
        spec = str(match.group(1) or "").strip()
        nums = parse_ref_num_set(spec, max_items=4)
        if refs[0] not in nums:
            return str(match.group(0) or "")
        replaced = True
        return marker

    new_line = re.sub(
        r"\[(\d{1,4}(?:\s*(?:[\-–—−,])\s*\d{1,4})*)\]",
        _replace_first_inline_ref,
        raw_line,
        count=1,
    )
    if not replaced:
        new_line = raw_line.rstrip() + " " + marker
    lines[best_idx] = new_line
    return "\n".join(lines).strip()


def _inject_paper_guide_card_citations(
    answer: str,
    *,
    cards: list[dict],
    prompt_family: str = "",
    max_injections: int = 2,
) -> str:
    text = str(answer or "").strip()
    family = str(prompt_family or "").strip().lower()
    if (not text) or ("[[CITE:" in text) or family != "method":
        return text
    if re.search(r"\[\d{1,4}(?:\s*(?:[\-–—−,])\s*\d{1,4})*\]", text):
        return text
    try:
        limit = max(1, int(max_injections))
    except Exception:
        limit = 2

    rules: list[dict[str, object]] = []
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        sid = str(card.get("sid") or "").strip()
        refs: list[int] = []
        for item in list(card.get("candidate_refs") or []):
            try:
                n = int(item)
            except Exception:
                continue
            if n > 0:
                refs.append(n)
        if (not sid) or (not refs):
            continue
        cue_parts: list[str] = []
        for part in (
            str(card.get("cue") or "").strip(),
            str(card.get("heading") or "").strip(),
            str(card.get("snippet") or "").strip(),
        ):
            if part:
                cue_parts.append(part)
        for extra in list(card.get("deepread_texts") or [])[:1]:
            extra_text = str(extra or "").strip()
            if extra_text:
                cue_parts.append(extra_text)
        tokens = _paper_guide_cue_tokens(" ".join(cue_parts))
        if len(tokens) < 2:
            continue
        rules.append(
            {
                "marker": f"[[CITE:{sid}:{refs[0]}]]",
                "tokens": set(tokens),
            }
        )

    if not rules:
        return _inject_paper_guide_fallback_citations(
            text,
            cards=cards,
            prompt_family=prompt_family,
            max_injections=max_injections,
        )

    lines = text.splitlines()
    used_markers: set[str] = set()
    injected = 0
    for idx, line in enumerate(lines):
        if injected >= limit:
            break
        raw = str(line or "")
        stripped = raw.strip()
        if (not stripped) or ("[[CITE:" in stripped):
            continue
        if re.match(r"^(Conclusion|Evidence|Limits|Next Steps|Abstract text|Chinese translation)\s*:?\s*$", stripped, flags=re.IGNORECASE):
            continue
        if stripped.startswith("|"):
            continue
        line_tokens = set(_paper_guide_cue_tokens(stripped))
        if len(line_tokens) < 2:
            continue
        best_rule: dict[str, object] | None = None
        best_score = 0
        best_shared: set[str] = set()
        for rule in rules:
            marker = str(rule.get("marker") or "").strip()
            if (not marker) or (marker in used_markers):
                continue
            shared = line_tokens.intersection(set(rule.get("tokens") or set()))
            score = len(shared)
            if score > best_score:
                best_score = score
                best_rule = rule
                best_shared = set(shared)
        min_score = 2
        if family in {"method", "compare", "figure_walkthrough"} and best_score == 1:
            if best_shared.intersection(_PAPER_GUIDE_CITE_STRONG_TOKENS):
                min_score = 1
        if not best_rule or best_score < min_score:
            continue
        marker = str(best_rule.get("marker") or "").strip()
        if not marker:
            continue
        lines[idx] = raw.rstrip() + " " + marker
        used_markers.add(marker)
        injected += 1

    out = "\n".join(lines).strip()
    if ("[[CITE:" not in out) and out:
        return _inject_paper_guide_fallback_citations(
            out,
            cards=cards,
            prompt_family=prompt_family,
            max_injections=max_injections,
        )
    return out


def _drop_paper_guide_locate_only_line_citations(
    answer: str,
    *,
    support_resolution: list[dict] | None = None,
) -> str:
    text = str(answer or "")
    if ("[[CITE:" not in text) and ("[CITE:" not in text):
        return text
    lines = text.splitlines()
    changed = False
    for rec in list(support_resolution or []):
        if not isinstance(rec, dict):
            continue
        if str(rec.get("cite_policy") or "").strip().lower() != "locate_only":
            continue
        try:
            resolved_ref_num = int(rec.get("resolved_ref_num") or 0)
        except Exception:
            resolved_ref_num = 0
        if resolved_ref_num > 0:
            continue
        try:
            line_index_raw = rec.get("line_index")
            line_index = int(line_index_raw) if str(line_index_raw).strip() else -1
        except Exception:
            line_index = -1
        if line_index < 0 or line_index >= len(lines):
            continue
        raw_line = str(lines[line_index] or "")
        new_line = _CITE_CANON_RE.sub("", raw_line)
        new_line = _CITE_SINGLE_BRACKET_RE.sub("", new_line)
        new_line = re.sub(r"[ \t]{2,}", " ", new_line).rstrip()
        new_line = re.sub(r"\s+([,.;:!?])", r"\1", new_line)
        if new_line != raw_line:
            lines[line_index] = new_line
            changed = True
    return "\n".join(lines).strip() if changed else text


def _promote_paper_guide_numeric_reference_citations(
    answer: str,
    *,
    locked_source: dict | None = None,
) -> str:
    text = str(answer or "")
    locked_sid = str((locked_source or {}).get("sid") or "").strip().lower()
    if (not text) or (not locked_sid):
        return text

    def _repl(m: re.Match[str]) -> str:
        prefix = str(m.group(1) or "")
        spec = str(m.group(3) or "").strip()
        nums = parse_ref_num_set(spec, max_items=8)
        if not nums:
            return str(m.group(0) or "")
        markers = "".join(f"[[CITE:{locked_sid}:{int(n)}]]" for n in nums if int(n) > 0)
        if not markers:
            return str(m.group(0) or "")
        return prefix + markers

    return _PAPER_GUIDE_NUMERIC_REF_RE.sub(_repl, text)
