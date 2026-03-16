from __future__ import annotations

import json
import os
import re
import uuid
from difflib import SequenceMatcher
from pathlib import Path

from kb.paper_guide_shared import (
    DeepSeekChat,
    _CLAIM_EXPERIMENT_HINT_RE,
    _CLAIM_METHOD_HINT_RE,
    _CJK_WORD_RE,
    _CONTRIBUTION_BLOCK_HINT_RE,
    _CONTRIBUTION_LEADIN_HINT_RE,
    _CRITICAL_FACT_HINT_RE,
    _DEFINITION_LIKE_BLOCK_HINT_RE,
    _DISPLAY_EQ_SEG_RE,
    _EQUATION_EXPLANATION_HINT_RE,
    _EQUATION_EXPLANATION_PREFIX_RE,
    _EQ_ENV_SEG_RE,
    _EXPERIMENT_HEADING_HINTS,
    _FIGURE_CLAIM_RE,
    _FIG_NUMBER_PATTERNS,
    _FORMULA_CMD_RE,
    _FORMULA_TOKEN_RE,
    _GENERIC_HEADING_HINTS,
    _LATIN_WORD_RE,
    _METHOD_HEADING_HINTS,
    _NON_SOURCE_SEGMENT_HINTS,
    _QUOTE_ELLIPSIS_RE,
    _QUOTE_HEADING_LIKE_RE,
    _QUOTE_PATTERNS,
    _RESULT_BLOCK_HINT_RE,
    _SEG_SENT_SPLIT_RE,
    _SHELL_ONLY_RE,
    _SHELL_PREFIX_RE,
    _SUMMARY_NOVELTY_HINT_RE,
    _SUMMARY_RESULT_HINT_RE,
    _resolve_md_output_paths,
    extract_equation_number,
    has_equation_signal,
    load_source_blocks,
    match_source_blocks,
    normalize_inline_markdown,
    normalize_match_text,
    split_answer_segments,
)

_PAPER_GUIDE_PROVENANCE_SCHEMA_VERSION = 4


def _normalize_fs_path_for_match(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        p = Path(raw).expanduser().resolve(strict=False)
    except Exception:
        p = Path(raw).expanduser()
    return str(p).replace("\\", "/").strip().lower()


def _source_basename_identity(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    name = Path(raw).name or raw
    low = str(name).strip().lower()
    low = re.sub(r"\.en\.md$", "", low, flags=re.IGNORECASE)
    low = re.sub(r"\.md$", "", low, flags=re.IGNORECASE)
    low = re.sub(r"\.pdf$", "", low, flags=re.IGNORECASE)
    low = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", low)
    return re.sub(r"\s+", " ", low).strip()


def _source_stem_identity(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        p = Path(raw)
        name = str(p.name or raw).strip()
    except Exception:
        name = raw
    return _source_basename_identity(name)


def _resolve_paper_guide_md_path(
    source_path: str,
    *,
    md_root: Path | str | None = None,
    db_dir: Path | str | None = None,
) -> Path | None:
    raw = str(source_path or "").strip()
    if not raw:
        return None
    src = Path(raw).expanduser()
    try:
        if src.is_file() and src.suffix.lower().endswith(".md"):
            return src.resolve(strict=False)
    except Exception:
        pass

    if src.suffix.lower() != ".pdf":
        return None

    roots: list[Path] = []
    if md_root:
        try:
            roots.append(Path(md_root).expanduser())
        except Exception:
            pass
    if db_dir:
        try:
            roots.append(Path(db_dir).expanduser())
        except Exception:
            pass
        try:
            roots.append(Path(db_dir).expanduser().parent / "md_output")
        except Exception:
            pass
    try:
        roots.append(src.parent)
    except Exception:
        pass

    uniq_roots: list[Path] = []
    seen_roots: set[str] = set()
    for root in roots:
        key = _normalize_fs_path_for_match(str(root))
        if (not key) or (key in seen_roots):
            continue
        seen_roots.add(key)
        uniq_roots.append(root)

    for root in uniq_roots:
        try:
            _md_dir, md_main, md_exists = _resolve_md_output_paths(root, src)
        except Exception:
            continue
        if not md_exists:
            continue
        try:
            if md_main.is_file():
                return md_main.resolve(strict=False)
        except Exception:
            continue

    return None


def _is_hit_from_bound_source(
    hit: dict,
    *,
    bound_source_path: str,
    bound_source_name: str,
) -> bool:
    meta = (hit or {}).get("meta", {}) or {}
    hit_source = str(meta.get("source_path") or "").strip()
    if not hit_source:
        return False
    hit_path_norm = _normalize_fs_path_for_match(hit_source)
    if not hit_path_norm:
        return False

    target_path_norm = _normalize_fs_path_for_match(bound_source_path)
    if target_path_norm:
        if hit_path_norm == target_path_norm:
            return True
        hit_stem = _source_stem_identity(hit_source)
        target_stem = _source_stem_identity(bound_source_path)
        if target_stem and hit_stem and (hit_stem == target_stem) and (len(target_stem) >= 8):
            return True
        return False

    target_name = _source_basename_identity(bound_source_name)
    if not target_name:
        return False
    hit_stem = _source_stem_identity(hit_source)
    return bool(hit_stem and (hit_stem == target_name))


def _is_display_formula_segment(text: str, *, segment_kind: str = "") -> bool:
    src = str(text or "").strip()
    if not src:
        return False
    if _DISPLAY_EQ_SEG_RE.search(src):
        return True
    if _EQ_ENV_SEG_RE.search(src):
        return True
    if "\\[" in src and "\\]" in src:
        return True

    n0 = extract_equation_number(src)
    if n0 > 0 and has_equation_signal(src):
        return True

    if not has_equation_signal(src):
        return False

    clean = normalize_inline_markdown(src)
    if not clean:
        return False
    latin_words = len(_LATIN_WORD_RE.findall(clean))
    cjk_words = len(_CJK_WORD_RE.findall(clean))
    formula_tokens = len(_FORMULA_TOKEN_RE.findall(src))
    non_empty_lines = [line for line in src.splitlines() if str(line).strip()]
    kind = str(segment_kind or "").strip().lower()

    if formula_tokens >= 8 and (latin_words + cjk_words) <= 12 and len(non_empty_lines) <= 3:
        return True
    if kind == "paragraph" and formula_tokens >= 10 and (latin_words + cjk_words) <= 14:
        return True
    return False


def _formula_token_overlap_score(a: str, b: str) -> float:
    ta = set(_FORMULA_CMD_RE.findall(str(a or "")))
    tb = set(_FORMULA_CMD_RE.findall(str(b or "")))
    if not ta or not tb:
        return 0.0
    overlap = sum(1 for token in ta if token in tb)
    return float(overlap) / (max(1.0, (len(ta) * len(tb)) ** 0.5))


def _text_token_overlap_score(a: str, b: str) -> float:
    x = normalize_match_text(a)
    y = normalize_match_text(b)
    if (not x) or (not y):
        return 0.0
    ta = set(re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fff]{1,2}", x))
    tb = set(re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fff]{1,2}", y))
    if not ta or not tb:
        return 0.0
    overlap = sum(1 for token in ta if token in tb)
    return float(overlap) / (max(1.0, (len(ta) * len(tb)) ** 0.5))


def _normalize_formula_compare_text(text: str) -> str:
    src = str(text or "")
    if not src:
        return ""
    s = src.lower()
    s = re.sub(r"\$+", "", s)
    s = re.sub(r"\\tag\{\s*\d{1,4}\s*\}", "", s)
    s = re.sub(r"\\(?:left|right|mathbf|mathrm|mathit|boldsymbol|operatorname)\b", "", s)
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\\[,;:!]", "", s)
    s = re.sub(r"\s+", "", s)
    return s[:2000]


def _formula_char_similarity(a: str, b: str) -> float:
    x = _normalize_formula_compare_text(a)
    y = _normalize_formula_compare_text(b)
    if (not x) or (not y):
        return 0.0
    try:
        return float(SequenceMatcher(None, x, y).ratio())
    except Exception:
        return 0.0


def _segment_snippet_aliases(text: str) -> list[str]:
    src = normalize_inline_markdown(str(text or ""))
    if not src:
        return []
    aliases: list[str] = []
    seen: set[str] = set()

    def _push(raw: str) -> None:
        key = normalize_match_text(str(raw or ""))
        if (not key) or (key in seen):
            return
        seen.add(key)
        aliases.append(key[:360])

    _push(src)
    if len(src) > 200:
        _push(src[:200])

    sent_list = [s.strip() for s in _SEG_SENT_SPLIT_RE.split(src) if str(s).strip()]
    if sent_list:
        first = sent_list[0]
        if len(first) >= 14:
            _push(first)
        if len(sent_list) >= 2:
            pair = f"{sent_list[0]} {sent_list[1]}".strip()
            if len(pair) >= 18:
                _push(pair)

    return aliases[:6]


def _strip_provenance_noise_text(text: str) -> str:
    src = normalize_inline_markdown(str(text or ""))
    if not src:
        return ""
    src = re.sub(r"\[\d{1,3}(?:\s*[-,\u2013\u2014]\s*\d{1,3})*\]", " ", src)
    src = re.sub(r"\(\s*\d{1,3}(?:\s*[-,\u2013\u2014]\s*\d{1,3})*\s*\)", " ", src)
    src = re.sub(r"(?:see|参见)\s*\[\d{1,3}(?:\s*[-,\u2013\u2014]\s*\d{1,3})*\]", " ", src, flags=re.IGNORECASE)
    src = re.sub(r"\s+", " ", src)
    return src.strip()


def _extract_quoted_spans(text: str, *, min_len: int = 10) -> list[str]:
    src = _strip_provenance_noise_text(text)
    if not src:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for pattern in _QUOTE_PATTERNS:
        for m in pattern.finditer(src):
            item = str(m.group(1) or "").strip()
            if len(item) < max(1, int(min_len)):
                continue
            key = normalize_match_text(item)
            if (not key) or (key in seen):
                continue
            seen.add(key)
            out.append(item[:360])
            if len(out) >= 6:
                return out
    return out


def _longest_quoted_span(text: str, *, min_len: int = 10) -> str:
    spans = _extract_quoted_spans(text, min_len=min_len)
    if not spans:
        return ""
    spans.sort(key=lambda item: len(str(item or "")), reverse=True)
    return str(spans[0] or "").strip()


def _is_heading_like_quote_span(text: str) -> bool:
    raw = _strip_provenance_noise_text(text)
    if not raw:
        return True
    compact = re.sub(r"\s+", " ", raw).strip(" :;,.!?()[]{}\"'“”‘’")
    if not compact:
        return True
    if _QUOTE_HEADING_LIKE_RE.fullmatch(compact):
        return True
    if re.search(r"[。！？!?;；:：]", compact):
        return False
    latin_words = _LATIN_WORD_RE.findall(compact)
    if 0 < len(latin_words) <= 8 and len(compact) <= 80:
        verb_like = re.search(
            r"\b(?:is|are|was|were|be|being|been|can|cannot|could|should|would|will|use|used|"
            r"using|estimate|estimated|show|shown|train|training|feed|feeding|make|making|"
            r"exploit|exploits|provide|providing|compare|comparison)\b",
            compact,
            re.IGNORECASE,
        )
        if not verb_like:
            return True
    if len(compact) <= 28 and not re.search(r"\d", compact):
        return True
    return False


def _is_rhetorical_shell_sentence(text: str) -> bool:
    raw = _strip_provenance_noise_text(text)
    if not raw:
        return True
    if _SHELL_ONLY_RE.match(raw):
        return True
    if _SHELL_PREFIX_RE.match(raw):
        return True
    if raw.endswith(":") or raw.endswith("："):
        informative_tail = re.sub(r"[:：]\s*$", "", raw).strip()
        if len(informative_tail) <= 32:
            return True
        if re.search(r"(?:说明|表明|意味着|提示|可见|证实)\s*$", informative_tail):
            return True
    return False


def _is_explicit_non_source_segment(text: str) -> bool:
    raw = _strip_provenance_noise_text(text)
    if not raw:
        return False
    low = raw.lower()
    if "补充说明" in raw and ("检索片段" in raw or "retrieved context" in low):
        return True
    for hint in _NON_SOURCE_SEGMENT_HINTS:
        if hint and ((hint in raw) or (hint in low)):
            return True
    return False


def _opens_non_source_scope(text: str) -> bool:
    raw = str(text or "").strip()
    clean = _strip_provenance_noise_text(raw)
    if (not clean) or (not _is_explicit_non_source_segment(raw)):
        return False
    head = clean[:48].lower()
    if clean.endswith(":") or clean.endswith("："):
        return True
    return ("supplement" in head) or ("补充说明" in clean[:24]) or ("说明" in clean[:16])


def _is_non_source_scope_boundary(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    clean = _strip_provenance_noise_text(raw)
    if not clean:
        return False
    if _is_explicit_non_source_segment(raw):
        return False
    if raw.startswith("---"):
        return True
    if re.match(r"^\s{0,3}#{1,6}\s+", raw):
        return True
    return bool(
        re.match(
            r"^\s*(?:结论|依据|边界|限制|下一步|note|notes|boundary|limitations?|conclusion|next steps?)\s*[:：]",
            clean,
            re.IGNORECASE,
        )
    )


def _critical_fact_score(text: str) -> float:
    raw = _strip_provenance_noise_text(text)
    if not raw:
        return 0.0
    score = 0.0
    if len(raw) >= 18:
        score += 0.12
    if len(raw) >= 32:
        score += 0.12
    if len(raw) >= 56:
        score += 0.08
    if re.search(r"\d", raw):
        score += 0.08
    if re.search(r"[A-Z][A-Za-z0-9+-]{1,}", raw):
        score += 0.08
    if has_equation_signal(raw):
        score += 0.14
    if extract_equation_number(raw) > 0:
        score += 0.12
    if _FIGURE_CLAIM_RE.search(raw):
        score += 0.12
    if _CLAIM_EXPERIMENT_HINT_RE.search(raw):
        score += 0.10
    if _CLAIM_METHOD_HINT_RE.search(raw):
        score += 0.08
    if _CRITICAL_FACT_HINT_RE.search(raw):
        score += 0.14
    return float(score)


def _segment_type_from_text(text: str, *, segment_kind: str = "") -> str:
    src = normalize_inline_markdown(str(text or ""))
    if not src:
        return "other"
    head = src[:48]
    kind = str(segment_kind or "").strip().lower()
    if _is_display_formula_segment(src, segment_kind=kind):
        return "equation_explanation"
    if head.startswith("结论") or head.startswith("核心结论"):
        return "claim"
    if head.startswith("依据") or head.startswith("证据") or head.startswith("原文"):
        return "evidence"
    if head.startswith("下一步") or head.startswith("建议") or head.startswith("行动"):
        return "next_step"
    if kind == "list_item":
        return "bullet"
    return "prose"


def _segment_focus_tags(text: str) -> set[str]:
    src = normalize_inline_markdown(str(text or ""))
    if not src:
        return set()
    tags: set[str] = set()
    if _CLAIM_EXPERIMENT_HINT_RE.search(src):
        tags.add("experiment")
    if _CLAIM_METHOD_HINT_RE.search(src):
        tags.add("method")
    if has_equation_signal(src):
        tags.add("formula")
    return tags


def _summary_segment_tags(text: str) -> set[str]:
    src = normalize_inline_markdown(str(text or ""))
    if not src:
        return set()
    tags: set[str] = set()
    if _SUMMARY_NOVELTY_HINT_RE.search(src):
        tags.add("novelty")
    if _SUMMARY_RESULT_HINT_RE.search(src):
        tags.add("result")
    return tags


def _summary_block_adjustment(segment_text: str, block: dict | None) -> float:
    if not isinstance(block, dict):
        return 0.0
    tags = _summary_segment_tags(segment_text)
    if not tags:
        return 0.0
    block_text = normalize_inline_markdown(str(block.get("raw_text") or block.get("text") or ""))
    if not block_text:
        return 0.0
    heading = normalize_match_text(str(block.get("heading_path") or ""))
    kind = str(block.get("kind") or "").strip().lower()
    score = 0.0
    text_has_contribution = bool(_CONTRIBUTION_BLOCK_HINT_RE.search(block_text))
    text_has_result = bool(_RESULT_BLOCK_HINT_RE.search(block_text))
    definition_like = bool(_DEFINITION_LIKE_BLOCK_HINT_RE.search(block_text))
    leadin_only = bool(_CONTRIBUTION_LEADIN_HINT_RE.search(block_text))

    if kind == "heading":
        score -= 0.22
    if kind == "equation":
        score -= 0.36

    if "novelty" in tags:
        if text_has_contribution:
            score += 0.72
            if kind == "list_item":
                score += 0.18
            if "abstract" in heading:
                score += 0.18
            elif "introduction" in heading:
                score += 0.10
        if definition_like:
            score -= 0.88
        if leadin_only:
            score -= 0.14

    if "result" in tags:
        if text_has_result:
            score += 0.58
            if kind == "list_item":
                score += 0.12
            if any(token in heading for token in _EXPERIMENT_HEADING_HINTS):
                score += 0.16
            elif "abstract" in heading:
                score += 0.08
        if definition_like:
            score -= 0.46
        if leadin_only:
            score -= 0.10

    return float(score)


def _heading_focus_adjustment(segment_text: str, heading_path: str) -> float:
    heading = normalize_match_text(heading_path)
    if not heading:
        return 0.0
    tags = _segment_focus_tags(segment_text)
    generic = any(token in heading for token in _GENERIC_HEADING_HINTS)
    experiment = any(token in heading for token in _EXPERIMENT_HEADING_HINTS)
    method = any(token in heading for token in _METHOD_HEADING_HINTS)
    score = 0.0
    if generic:
        score -= 0.18
        if "abstract" in heading:
            score -= 0.08
        if "related work" in heading or "reference" in heading:
            score -= 0.12
    if "experiment" in tags:
        if experiment:
            score += 0.26
        elif generic:
            score -= 0.14
    if "method" in tags:
        if method:
            score += 0.18
        elif generic and ("experiment" not in tags):
            score -= 0.06
    return score


def _is_generic_heading_path(heading_path: str) -> bool:
    heading = normalize_match_text(heading_path)
    if not heading:
        return False
    return any(token in heading for token in _GENERIC_HEADING_HINTS)


def _best_evidence_quote_match(segment_text: str, block: dict | None) -> tuple[str, float]:
    if not isinstance(block, dict):
        return "", 0.0
    block_text_raw = str(block.get("raw_text") or block.get("text") or "").strip()
    if not block_text_raw:
        return "", 0.0
    block_text = normalize_inline_markdown(block_text_raw)
    if not block_text:
        return "", 0.0
    if has_equation_signal(segment_text) or str(block.get("kind") or "").strip().lower() == "equation":
        snippet = block_text[:320]
        score = _text_token_overlap_score(segment_text, snippet)
        if snippet:
            try:
                score += 0.22 * float(
                    SequenceMatcher(
                        None,
                        normalize_match_text(segment_text)[:420],
                        normalize_match_text(snippet)[:420],
                    ).ratio()
                )
            except Exception:
                pass
        return snippet, score

    segment_key = normalize_match_text(segment_text)
    best = ""
    best_score = -1.0
    sentences = [s.strip() for s in _SEG_SENT_SPLIT_RE.split(block_text) if str(s).strip()]
    if not sentences:
        sentences = [block_text]
    for sent in sentences[:16]:
        sent_norm = normalize_match_text(sent)
        if not sent_norm:
            continue
        score = 0.0
        if sent_norm == segment_key:
            score += 1.2
        elif segment_key and (segment_key in sent_norm or sent_norm in segment_key):
            score += 0.82
        score += _text_token_overlap_score(segment_text, sent)
        try:
            score += 0.28 * float(SequenceMatcher(None, segment_key[:420], sent_norm[:420]).ratio())
        except Exception:
            pass
        if has_equation_signal(sent):
            score += 0.22 * _formula_token_overlap_score(segment_text, sent)
            score += 0.12 * _formula_char_similarity(segment_text, sent)
        if score > best_score:
            best = sent
            best_score = score
    if best and best_score >= 0.16:
        return best[:320], max(0.0, best_score)
    fallback = block_text[:220]
    fallback_score = _text_token_overlap_score(segment_text, fallback)
    try:
        fallback_score += 0.18 * float(
            SequenceMatcher(
                None,
                normalize_match_text(segment_text)[:420],
                normalize_match_text(fallback)[:420],
            ).ratio()
        )
    except Exception:
        pass
    return fallback, max(0.0, fallback_score)


def _best_evidence_quote(segment_text: str, block: dict | None) -> str:
    return _best_evidence_quote_match(segment_text, block)[0]


def _expand_match_snippet_hints(text: str, *, max_items: int = 6) -> list[str]:
    raw = str(text or "")
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def _push(item: str) -> None:
        clean = normalize_inline_markdown(item)
        if len(clean) < 14:
            return
        key = normalize_match_text(clean)
        if (not key) or (key in seen):
            return
        seen.add(key)
        out.append(clean[:360])

    for line in raw.splitlines():
        stripped = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", str(line or "")).strip()
        if not stripped:
            continue
        clean = normalize_inline_markdown(stripped)
        if len(clean) >= 24:
            _push(stripped)
            if len(out) >= max_items:
                return out

    clean_raw = normalize_inline_markdown(raw)
    if clean_raw:
        for sent in [s.strip() for s in _SEG_SENT_SPLIT_RE.split(clean_raw) if str(s).strip()]:
            if len(sent) < 22:
                continue
            _push(sent)
            if len(out) >= max_items:
                return out
        _push(clean_raw)
    return out[:max_items]


def _block_support_metrics(segment_text: str, block: dict | None) -> dict[str, float | str | bool]:
    if not isinstance(block, dict):
        return {
            "quote": "",
            "quote_score": 0.0,
            "support_score": 0.0,
            "heading_adjust": 0.0,
            "generic_heading": False,
        }
    block_text = normalize_inline_markdown(str(block.get("raw_text") or block.get("text") or ""))
    if not block_text:
        return {
            "quote": "",
            "quote_score": 0.0,
            "support_score": 0.0,
            "heading_adjust": 0.0,
            "generic_heading": False,
        }
    heading_path = str(block.get("heading_path") or "").strip()
    quote, quote_score = _best_evidence_quote_match(segment_text, block)
    block_overlap = _text_token_overlap_score(segment_text, block_text)
    block_ratio = 0.0
    try:
        block_ratio = float(
            SequenceMatcher(
                None,
                normalize_match_text(segment_text)[:420],
                normalize_match_text(block_text)[:420],
            ).ratio()
        )
    except Exception:
        block_ratio = 0.0
    support_score = max(quote_score, block_overlap + (0.22 * block_ratio))
    heading_adjust = _heading_focus_adjustment(segment_text, heading_path)
    return {
        "quote": quote,
        "quote_score": float(max(0.0, quote_score)),
        "support_score": float(max(0.0, support_score)),
        "heading_adjust": float(heading_adjust),
        "generic_heading": bool(_is_generic_heading_path(heading_path)),
    }


def _extract_json_object_text(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{[\s\S]*\}", text)
    return str(m.group(0) or "").strip() if m else ""


def _extract_figure_number(text: str) -> int:
    raw = str(text or "").strip()
    if not raw:
        return 0
    for pat in _FIG_NUMBER_PATTERNS:
        m = pat.search(raw)
        if not m:
            continue
        try:
            n = int(m.group(1))
        except Exception:
            n = 0
        if n > 0:
            return n
    return 0


def _extract_display_formula_snippet(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    m = _DISPLAY_EQ_SEG_RE.search(raw)
    if m:
        return str(m.group(0) or "").strip()[:1200]
    if _EQ_ENV_SEG_RE.search(raw):
        return raw[:1200]
    if "\\[" in raw and "\\]" in raw:
        return raw[:1200]
    return ""


def _strip_inline_formula_delimiters(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if raw.startswith("$$") and raw.endswith("$$") and len(raw) >= 4:
        raw = raw[2:-2]
    elif raw.startswith("$") and raw.endswith("$") and len(raw) >= 2:
        raw = raw[1:-1]
    elif raw.startswith("\\(") and raw.endswith("\\)") and len(raw) >= 4:
        raw = raw[2:-2]
    raw = re.sub(r"\\tag\{\s*\d{1,4}\s*\}", "", raw)
    return raw.strip()


def _extract_inline_formula_spans(text: str, *, max_items: int = 10) -> list[str]:
    raw = str(text or "")
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def _push(item: str) -> None:
        cleaned = _strip_inline_formula_delimiters(item)
        if not cleaned or len(cleaned) < 3 or not has_equation_signal(cleaned):
            return
        key = _normalize_formula_compare_text(cleaned)
        if (not key) or (key in seen):
            return
        seen.add(key)
        out.append(cleaned[:900])

    for pattern in (
        re.compile(r"(?<!\\)(?<!\$)\$(?!\$)(.+?)(?<!\\)\$(?!\$)", re.DOTALL),
        re.compile(r"\\\((.+?)\\\)", re.DOTALL),
    ):
        for match in pattern.finditer(raw):
            _push(str(match.group(1) or ""))
            if len(out) >= max_items:
                return out
    return out


def _inline_formula_value_score(text: str) -> float:
    raw = _strip_inline_formula_delimiters(text)
    if not raw or not has_equation_signal(raw):
        return 0.0
    score = 0.0
    if "=" in raw:
        score += 0.42
    token_count = len(_FORMULA_TOKEN_RE.findall(raw))
    cmd_count = len(_FORMULA_CMD_RE.findall(raw))
    if token_count >= 3:
        score += 0.22
    if token_count >= 6:
        score += 0.16
    if cmd_count >= 1:
        score += 0.18
    if re.search(r"\\(?:int|sum|prod|frac|exp|log|softmax|sigma|mu|theta|phi)\b", raw):
        score += 0.42
    if len(raw) >= 18:
        score += 0.18
    return float(score)


def _pick_primary_inline_formula_span(text: str) -> str:
    spans = _extract_inline_formula_spans(text)
    if not spans:
        return ""
    ranked = sorted(
        spans,
        key=lambda item: (
            -_inline_formula_value_score(item),
            -len(str(item or "")),
        ),
    )
    return str(ranked[0] or "").strip()


def _split_formula_equation_sides(text: str) -> tuple[str, str]:
    raw = _strip_inline_formula_delimiters(text)
    if not raw:
        return "", ""
    parts = re.split(r"(?<![<>!=])=(?![=>])", raw, maxsplit=1)
    if len(parts) != 2:
        return raw, ""
    return str(parts[0] or "").strip(), str(parts[1] or "").strip()


def _best_inline_formula_match(segment: dict | None, block: dict | None) -> tuple[str, float]:
    if not isinstance(segment, dict) or not isinstance(block, dict):
        return "", 0.0
    block_text_raw = str(block.get("raw_text") or block.get("text") or "").strip()
    if not block_text_raw:
        return "", 0.0
    spans = _extract_inline_formula_spans(block_text_raw)
    if not spans:
        return "", 0.0

    segment_formula_raw = str(segment.get("raw_markdown") or segment.get("text") or "")
    formula_src = (
        _extract_display_formula_snippet(segment_formula_raw)
        or _pick_primary_inline_formula_span(segment_formula_raw)
        or str(segment.get("anchor_text") or segment.get("evidence_quote") or segment.get("text") or "").strip()
    )
    formula_src = _strip_inline_formula_delimiters(formula_src)
    if not formula_src or not has_equation_signal(formula_src):
        return "", 0.0

    lhs, rhs = _split_formula_equation_sides(formula_src)
    block_norm = normalize_match_text(normalize_inline_markdown(block_text_raw))
    low_block = block_text_raw.lower()
    best_span = ""
    best_score = 0.0
    for span in spans:
        value_score = _inline_formula_value_score(span)
        if value_score < 0.4:
            continue
        score = 0.0
        score += 0.72 * _formula_token_overlap_score(formula_src, span)
        score += 0.46 * _formula_char_similarity(formula_src, span)
        score += 0.18 * _text_token_overlap_score(formula_src, span)
        if rhs:
            rhs_score = 0.0
            rhs_score += 0.84 * _formula_token_overlap_score(rhs, span)
            rhs_score += 0.58 * _formula_char_similarity(rhs, span)
            rhs_score += 0.16 * _text_token_overlap_score(rhs, span)
            if lhs and normalize_match_text(lhs) and normalize_match_text(lhs) in block_norm:
                rhs_score += 0.24
            score = max(score, rhs_score)
        if "defined as" in low_block or "where " in low_block:
            score += 0.08
        score += min(0.28, value_score * 0.18)
        if score > best_score:
            best_span = span
            best_score = score
    return best_span, float(best_score)


def _select_inline_formula_claim_binding(
    segment: dict | None,
    block_lookup: dict[str, dict] | None = None,
) -> tuple[dict | None, str, float]:
    if not isinstance(segment, dict):
        return None, "", 0.0
    lookup = block_lookup or {}
    blocks = [block for block in lookup.values() if isinstance(block, dict)]
    if not blocks:
        return None, "", 0.0

    current_primary_id = str(segment.get("primary_block_id") or "").strip()
    current_primary = lookup.get(current_primary_id) if current_primary_id else None
    current_heading = normalize_match_text(str((current_primary or {}).get("heading_path") or segment.get("primary_heading_path") or ""))
    try:
        current_order = int((current_primary or {}).get("order_index") or 0)
    except Exception:
        current_order = 0
    segment_formula_raw = str(segment.get("raw_markdown") or segment.get("text") or "")
    formula_src = (
        _extract_display_formula_snippet(segment_formula_raw)
        or _pick_primary_inline_formula_span(segment_formula_raw)
        or str(segment.get("anchor_text") or segment.get("evidence_quote") or segment.get("text") or "").strip()
    )
    formula_src = _strip_inline_formula_delimiters(formula_src)
    lhs, rhs = _split_formula_equation_sides(formula_src)
    formula_src_norm = _normalize_formula_compare_text(formula_src)
    rhs_norm = _normalize_formula_compare_text(rhs)

    best_block: dict | None = None
    best_span = ""
    best_score = 0.0
    for block in blocks:
        block_text_raw = str(block.get("raw_text") or block.get("text") or "").strip()
        if not block_text_raw or not has_equation_signal(block_text_raw):
            continue
        if _is_display_formula_segment(block_text_raw, segment_kind=str(block.get("kind") or "")):
            continue
        block_id = str(block.get("block_id") or "").strip()
        if not block_id:
            continue
        span, score = _best_inline_formula_match(segment, block)
        if not span:
            continue
        span_norm = _normalize_formula_compare_text(span)
        exactish = bool(
            span_norm
            and (
                (formula_src_norm and (formula_src_norm == span_norm or formula_src_norm in span_norm or span_norm in formula_src_norm))
                or (rhs_norm and (rhs_norm == span_norm or rhs_norm in span_norm or span_norm in rhs_norm))
            )
        )
        if not exactish:
            continue
        block_heading = normalize_match_text(str(block.get("heading_path") or ""))
        try:
            block_order = int(block.get("order_index") or 0)
        except Exception:
            block_order = 0
        if current_heading and block_heading and block_heading == current_heading:
            score += 0.14
        if current_order > 0 and block_order > 0:
            distance = abs(block_order - current_order)
            score += max(0.0, 0.24 - min(distance, 5) * 0.05)
            if block_order > current_order and distance <= 2:
                score += 0.06
        if score > best_score:
            best_block = block
            best_span = span
            best_score = score
    if best_block is None or best_score < 0.56:
        return None, "", 0.0
    return best_block, best_span, float(best_score)


def _is_formula_claim_source_grounded(segment: dict | None, primary_block: dict | None) -> bool:
    if not isinstance(segment, dict):
        return False
    formula_src = _extract_display_formula_snippet(str(segment.get("raw_markdown") or segment.get("text") or ""))
    if not formula_src:
        return True
    support_formula = (
        _extract_display_formula_snippet(str((primary_block or {}).get("raw_text") or (primary_block or {}).get("text") or ""))
        or _extract_display_formula_snippet(str(segment.get("evidence_quote") or ""))
        or str(segment.get("evidence_quote") or "").strip()
    )
    if not support_formula or not has_equation_signal(support_formula):
        return False
    if normalize_match_text(formula_src) == normalize_match_text(support_formula):
        return True
    formula_tok = _formula_token_overlap_score(formula_src, support_formula)
    formula_char = _formula_char_similarity(formula_src, support_formula)
    formula_text = _text_token_overlap_score(formula_src, support_formula)
    if formula_tok >= 0.84 and formula_char >= 0.78:
        return True
    if formula_char >= 0.88 and formula_text >= 0.76:
        return True
    return False


def _formula_claim_alignment_score(segment: dict | None, primary_block: dict | None) -> float:
    if not isinstance(segment, dict) or not isinstance(primary_block, dict):
        return 0.0
    formula_src = _extract_display_formula_snippet(str(segment.get("raw_markdown") or segment.get("text") or ""))
    support_formula = _extract_display_formula_snippet(
        str(primary_block.get("raw_text") or primary_block.get("text") or "")
    )
    if (not formula_src) or (not support_formula):
        return 0.0
    score = 0.0
    if normalize_match_text(formula_src) == normalize_match_text(support_formula):
        score += 0.9
    score += 0.94 * _formula_token_overlap_score(formula_src, support_formula)
    score += 0.78 * _formula_char_similarity(formula_src, support_formula)
    score += 0.18 * _text_token_overlap_score(formula_src, support_formula)
    try:
        seg_eq = int(segment.get("equation_number") or 0)
    except Exception:
        seg_eq = 0
    try:
        block_eq = int(primary_block.get("number") or 0)
    except Exception:
        block_eq = 0
    if seg_eq > 0 and block_eq > 0 and seg_eq == block_eq:
        score += 0.72
    return float(score)


def _formula_anchor_text(raw_markdown: str, segment_text: str, primary_block: dict | None) -> str:
    raw = str(raw_markdown or "").strip()
    seg = str(segment_text or "").strip()
    if raw:
        m = _DISPLAY_EQ_SEG_RE.search(raw)
        if m:
            return str(m.group(0) or "").strip()[:900]
        if _EQ_ENV_SEG_RE.search(raw):
            return raw[:900]
    if isinstance(primary_block, dict):
        block_text = str(primary_block.get("raw_text") or primary_block.get("text") or "").strip()
        if block_text:
            return block_text[:900]
    return seg[:900]


def _dedupe_str_items(items: list[object] | tuple[object, ...] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in list(items or []):
        text = str(item or "").strip()
        if (not text) or (text in seen):
            continue
        seen.add(text)
        out.append(text)
    return out


_CLAIM_GROUP_SECTION_BOUNDARY_RE = re.compile(
    r"^(?:conclusion|core conclusion|evidence|original text|next steps?|suggestions?|risks?|limits?|"
    r"\u7ed3\u8bba|\u6838\u5fc3\u7ed3\u8bba|\u4f9d\u636e|\u8bc1\u636e|\u539f\u6587|\u4e0b\u4e00\u6b65|"
    r"\u5efa\u8bae|\u98ce\u9669|\u9650\u5236)\s*[:\uFF1A]?",
    flags=re.IGNORECASE,
)
_CLAIM_GROUP_LEAD_TAIL_RE = re.compile(
    r"(?:below|include|includes|including|consists? of|steps?|pipeline|shows?|indicates?|therefore|"
    r"\u5982\u4e0b|\u5305\u62ec|\u5206\u4e3a|\u6b65\u9aa4|\u6d41\u7a0b|\u8868\u660e|\u8bf4\u660e|\u53ef\u89c1|\u56e0\u6b64)$",
    flags=re.IGNORECASE,
)
_CLAIM_GROUP_CORE_HINT_RE = re.compile(
    r"(\b(?:gt|ground[ -]?truth|pose|camera|train|training|input|output|pipeline|rendering|volume)\b|"
    r"\u4f7f\u7528|\u91c7\u7528|\u8f93\u5165|\u8f93\u51fa|\u6062\u590d|\u91cd\u5efa|\u4f30\u8ba1|"
    r"\u56fa\u5b9a|\u8bad\u7ec3|\u751f\u6210|\u8868\u5f81|\u6f14\u7ece|\u7ea6\u675f|\u5bf9\u6bd4|"
    r"\u5229\u7528|\u5148\u7528|\u518d\u5c06|\u6765\u81ea|\u5bf9\u5e94)",
    flags=re.IGNORECASE,
)


def _score_claim_group_content_core(
    text: str,
    *,
    segment_kind: str = "",
    segment_type: str = "",
    evidence_mode: str = "",
) -> float:
    raw = _strip_provenance_noise_text(text).replace("\n", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return 0.0
    if _is_rhetorical_shell_sentence(raw):
        return 0.04

    score = 0.18
    length = len(raw)
    if length >= 18:
        score += 0.12
    if length >= 28:
        score += 0.12
    if length >= 46:
        score += 0.08
    if re.search(r"[\"'`\u2018\u2019\u201c\u201d]", raw):
        score += 0.12
    if re.search(r"[()\uFF08\uFF09=]", raw):
        score += 0.08
    if re.search(r"\d", raw):
        score += 0.08
    if re.search(r"[A-Z][A-Za-z0-9+\-]{1,}", raw):
        score += 0.08
    if _CLAIM_GROUP_CORE_HINT_RE.search(raw):
        score += 0.12

    kind = str(segment_kind or "").strip().lower()
    seg_type = str(segment_type or "").strip().lower()
    mode = str(evidence_mode or "").strip().lower()
    if kind == "list_item":
        score += 0.08
    if seg_type in {"bullet", "evidence", "equation_explanation"}:
        score += 0.08
    if mode == "direct":
        score += 0.06
    if re.search(r"[:\uFF1A]$", raw):
        score -= 0.18
    return max(0.0, min(1.2, float(score)))


def _is_likely_claim_group_lead(text: str, *, segment_kind: str = "", segment_type: str = "") -> bool:
    raw = _strip_provenance_noise_text(text).replace("\n", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return False
    if _is_rhetorical_shell_sentence(raw):
        return True
    if re.search(r"[:\uFF1A]$", raw):
        return True
    kind = str(segment_kind or "").strip().lower()
    seg_type = str(segment_type or "").strip().lower()
    if kind == "list_item" or seg_type == "bullet":
        return bool(_CLAIM_GROUP_LEAD_TAIL_RE.search(raw))
    return False


def _is_likely_claim_group_boundary(segment: dict | None) -> bool:
    if not isinstance(segment, dict):
        return True
    seg_type = str(segment.get("segment_type") or "").strip().lower()
    if seg_type in {"claim", "next_step"}:
        return True
    text = _strip_provenance_noise_text(str(segment.get("text") or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return True
    return bool(_CLAIM_GROUP_SECTION_BOUNDARY_RE.search(text))


def _pick_claim_group_target_segment(
    segments: list[dict],
    start_index: int,
) -> tuple[dict, int] | None:
    if start_index < 0 or start_index >= len(segments):
        return None
    current = segments[start_index]
    if not isinstance(current, dict):
        return None
    current_text = str(current.get("text") or "").strip()
    current_score = _score_claim_group_content_core(
        current_text,
        segment_kind=str(current.get("kind") or ""),
        segment_type=str(current.get("segment_type") or ""),
        evidence_mode=str(current.get("evidence_mode") or ""),
    )
    promote = _is_likely_claim_group_lead(
        current_text,
        segment_kind=str(current.get("kind") or ""),
        segment_type=str(current.get("segment_type") or ""),
    ) or current_score < 0.42
    if (not promote) and current_score >= 0.5:
        return current, 0

    best_segment: dict | None = None
    best_distance = 0
    best_score = -1.0
    upper = min(len(segments), start_index + 5)
    for idx in range(start_index + 1, upper):
        candidate = segments[idx]
        if not isinstance(candidate, dict):
            continue
        if idx > (start_index + 1) and _is_likely_claim_group_boundary(candidate):
            break
        if str(candidate.get("segment_type") or "").strip().lower() == "next_step":
            break
        if str(candidate.get("evidence_mode") or "").strip().lower() != "direct":
            continue
        candidate_text = str(candidate.get("text") or "").strip()
        if (not candidate_text) or _is_rhetorical_shell_sentence(candidate_text):
            continue
        candidate_score = _score_claim_group_content_core(
            candidate_text,
            segment_kind=str(candidate.get("kind") or ""),
            segment_type=str(candidate.get("segment_type") or ""),
            evidence_mode=str(candidate.get("evidence_mode") or ""),
        )
        if candidate_score < 0.46:
            continue
        distance = idx - start_index
        total = candidate_score - max(0, distance - 1) * 0.12
        if str(candidate.get("kind") or "").strip().lower() == "list_item":
            total += 0.06
        if str(candidate.get("evidence_mode") or "").strip().lower() == "direct":
            total += 0.04
        if str(candidate.get("segment_type") or "").strip().lower() == "bullet":
            total += 0.04
        if (best_segment is None) or (total > best_score):
            best_segment = candidate
            best_distance = distance
            best_score = total

    if best_segment is not None:
        return best_segment, best_distance
    if current_score >= 0.5 and not _is_likely_claim_group_lead(
        current_text,
        segment_kind=str(current.get("kind") or ""),
        segment_type=str(current.get("segment_type") or ""),
    ):
        return current, 0
    return None


def _assign_claim_group_targets(segments: list[dict]) -> list[dict]:
    out: list[dict] = []
    for idx, seg0 in enumerate(list(segments or [])):
        if not isinstance(seg0, dict):
            continue
        seg = dict(seg0)
        segment_id = str(seg.get("segment_id") or "").strip() or f"seg_{idx + 1}"
        claim_type = str(seg.get("claim_type") or "").strip().lower()
        locate_policy = str(seg.get("locate_policy") or "").strip().lower()
        evidence_mode = str(seg.get("evidence_mode") or "").strip().lower()
        block_ids = _dedupe_str_items(
            [seg.get("primary_block_id")]
            + list(seg.get("support_block_ids") or [])
            + list(seg.get("evidence_block_ids") or [])
        )
        keep_self_target = bool(seg.get("must_locate")) or claim_type in {
            "quote_claim",
            "blockquote_claim",
            "formula_claim",
            "inline_formula_claim",
            "equation_explanation_claim",
            "figure_claim",
        }
        target: tuple[dict, int] | None = None
        if keep_self_target:
            target = (seg, 0)
        elif evidence_mode == "direct" and locate_policy != "hidden" and block_ids:
            target = _pick_claim_group_target_segment(segments, idx)
        if target is None:
            target = (seg, 0)
        target_seg, distance = target
        target_segment_id = str(target_seg.get("segment_id") or "").strip() or segment_id
        seg["claim_group_target_segment_id"] = target_segment_id
        seg["claim_group_target_distance"] = int(max(0, distance))
        lead_text = _strip_provenance_noise_text(str(seg.get("text") or ""))
        lead_text = re.sub(r"\s+", " ", lead_text).strip()
        if distance > 0 and lead_text:
            seg["claim_group_lead_text"] = lead_text[:600]
            if not str(seg.get("claim_group_kind") or "").strip():
                seg["claim_group_kind"] = "content_core_bundle"
            if not str(seg.get("claim_group_id") or "").strip():
                seg["claim_group_id"] = f"content_core_bundle:{target_segment_id}"
        else:
            seg["claim_group_lead_text"] = ""
        out.append(seg)
    return out


def _segment_claim_meta(
    *,
    segment_text: str,
    raw_markdown: str,
    segment_kind: str,
    evidence_mode: str,
    primary_block: dict | None,
    evidence_quote: str,
    mapping_quality: float,
) -> dict[str, object]:
    del mapping_quality
    seg_text = _strip_provenance_noise_text(segment_text)
    raw_md = str(raw_markdown or "").strip()
    kind = str(segment_kind or "").strip().lower()
    mode = str(evidence_mode or "").strip().lower()
    quote_spans = _extract_quoted_spans(raw_md or seg_text, min_len=10)
    quote_anchor = _longest_quoted_span(raw_md or seg_text, min_len=10)
    heading_like_quote = bool(quote_anchor) and _is_heading_like_quote_span(quote_anchor)
    anchor_text = ""
    anchor_kind = ""
    claim_type = "critical_fact_claim"
    explicit_non_source = _is_explicit_non_source_segment(raw_md or seg_text)
    eq_number = extract_equation_number(raw_md or seg_text) if has_equation_signal(raw_md or seg_text) else 0
    figure_number = _extract_figure_number(raw_md or seg_text) if _FIGURE_CLAIM_RE.search(raw_md or seg_text) else 0
    primary_kind = str((primary_block or {}).get("kind") or "").strip().lower()
    primary_inline_formula = _pick_primary_inline_formula_span(raw_md or seg_text)
    inline_formula_dominant = False
    if primary_inline_formula and not _DISPLAY_EQ_SEG_RE.search(raw_md) and not _EQ_ENV_SEG_RE.search(raw_md):
        outer_text = re.sub(r"(?<!\\)(?<!\$)\$(?!\$).+?(?<!\\)\$(?!\$)", " ", raw_md or seg_text, flags=re.DOTALL)
        outer_text = re.sub(r"\\\(.+?\\\)", " ", outer_text, flags=re.DOTALL)
        outer_clean = normalize_inline_markdown(outer_text).strip()
        formula_clean = normalize_inline_markdown(primary_inline_formula).strip()
        inline_formula_dominant = bool(
            formula_clean
            and (
                kind == "blockquote"
                or len(formula_clean) >= max(16, int(len(_strip_provenance_noise_text(raw_md or seg_text)) * 0.38))
                or len(outer_clean) <= 40
            )
            and _inline_formula_value_score(primary_inline_formula) >= 0.72
        )

    if explicit_non_source:
        claim_type = "shell_sentence"
        anchor_kind = ""
        anchor_text = ""
    elif inline_formula_dominant:
        claim_type = "inline_formula_claim"
        anchor_kind = "inline_formula"
        anchor_text = primary_inline_formula[:900]
    elif _is_display_formula_segment(raw_md or seg_text, segment_kind=kind):
        claim_type = "formula_claim"
        anchor_kind = "equation"
        anchor_text = _formula_anchor_text(raw_md, seg_text, primary_block)
    elif primary_kind == "figure" or figure_number > 0:
        claim_type = "figure_claim"
        anchor_kind = "figure"
        figure_anchor = str(evidence_quote or seg_text or raw_md).strip()
        if figure_number > 0 and not re.search(
            rf"(?:\bfig(?:ure)?\.?\s*{int(figure_number)}\b|图\s*{int(figure_number)}\b|第\s*{int(figure_number)}\s*张图)",
            figure_anchor,
            re.IGNORECASE,
        ):
            figure_anchor = f"Figure {int(figure_number)}. {figure_anchor}".strip()
        anchor_text = figure_anchor[:480]
    elif kind == "blockquote":
        claim_type = "blockquote_claim"
        anchor_kind = "blockquote"
        anchor_text = str(evidence_quote or seg_text).strip()[:600]
    elif quote_spans and (not heading_like_quote):
        claim_type = "quote_claim"
        anchor_kind = "quote"
        anchor_text = quote_anchor[:600]
    elif heading_like_quote:
        claim_type = "shell_sentence"
        anchor_kind = ""
        anchor_text = ""
    elif _is_rhetorical_shell_sentence(seg_text):
        claim_type = "shell_sentence"
        anchor_kind = ""
        anchor_text = ""
    else:
        claim_type = "critical_fact_claim"
        anchor_kind = "sentence"
        anchor_text = str(evidence_quote or seg_text).strip()[:320]

    has_identity = bool(str((primary_block or {}).get("block_id") or "").strip()) and bool(
        str(anchor_kind or "").strip()
    ) and bool(str(anchor_text or evidence_quote or "").strip())
    must_locate = bool(
        mode == "direct"
        and has_identity
        and (
            claim_type in {"quote_claim", "blockquote_claim", "formula_claim", "inline_formula_claim", "figure_claim"}
            or (
                claim_type == "critical_fact_claim"
                and _critical_fact_score(anchor_text or seg_text) >= 0.34
                and (len(str(anchor_text or "").strip()) >= 14)
            )
        )
    )
    if claim_type == "shell_sentence":
        must_locate = False
    if claim_type == "formula_claim" and eq_number <= 0:
        try:
            eq_number = int((primary_block or {}).get("number") or 0)
        except Exception:
            eq_number = 0
    return {
        "claim_type": claim_type,
        "must_locate": bool(must_locate),
        "anchor_kind": anchor_kind,
        "anchor_text": str(anchor_text or "").strip(),
        "equation_number": int(eq_number or 0),
        "quote_spans": quote_spans,
    }


def _equation_explanation_score(
    *,
    segment_text: str,
    raw_markdown: str,
    formula_text: str,
    equation_number: int = 0,
    source_distance: int = 0,
) -> float:
    raw = _strip_provenance_noise_text(raw_markdown or segment_text)
    if not raw:
        return 0.0
    if _is_rhetorical_shell_sentence(raw):
        return 0.0
    score = 0.0
    if _EQUATION_EXPLANATION_PREFIX_RE.search(raw):
        score += 0.6
    if _EQUATION_EXPLANATION_HINT_RE.search(raw):
        score += 0.42
    if has_equation_signal(raw) and not _is_display_formula_segment(raw):
        score += 0.22
    if int(equation_number or 0) > 0:
        raw_eq = extract_equation_number(raw)
        if raw_eq > 0 and raw_eq == int(equation_number):
            score += 0.5
    score += 0.58 * _formula_token_overlap_score(raw, formula_text)
    score += 0.24 * _formula_char_similarity(raw, formula_text)
    score += 0.16 * _text_token_overlap_score(raw, formula_text)
    if int(source_distance or 0) == 1:
        score += 0.14
    elif int(source_distance or 0) == 2:
        score += 0.06
    return float(score)


def _figure_block_number(block: dict | None) -> int:
    if not isinstance(block, dict):
        return 0
    try:
        n = int(block.get("number") or 0)
    except Exception:
        n = 0
    if n > 0:
        return int(n)
    return _extract_figure_number(str(block.get("raw_text") or block.get("text") or ""))


def _select_figure_claim_binding(
    segment: dict | None,
    block_lookup: dict[str, dict] | None = None,
) -> tuple[dict | None, dict | None]:
    if not isinstance(segment, dict):
        return None, None
    lookup = block_lookup or {}
    blocks = [block for block in lookup.values() if isinstance(block, dict)]
    if not blocks:
        return None, None

    raw = " ".join(
        [
            str(segment.get("anchor_text") or ""),
            str(segment.get("evidence_quote") or ""),
            str(segment.get("raw_markdown") or segment.get("text") or ""),
        ]
    ).strip()
    figure_number = _extract_figure_number(raw)
    if figure_number <= 0:
        current_primary = lookup.get(str(segment.get("primary_block_id") or "").strip()) or {}
        figure_number = _figure_block_number(current_primary)
    if figure_number <= 0:
        return None, None

    def _order_of(block: dict | None) -> int:
        if not isinstance(block, dict):
            return 0
        try:
            order = int(block.get("order_index") or 0)
        except Exception:
            order = 0
        if order > 0:
            return order
        try:
            line_start = int(block.get("line_start") or 0)
        except Exception:
            line_start = 0
        return max(0, line_start)

    def _caption_score(block: dict | None, figure_order: int) -> float:
        if not isinstance(block, dict):
            return -1.0
        text = str(block.get("raw_text") or block.get("text") or "").strip()
        if _extract_figure_number(text) != figure_number:
            return -1.0
        score = 0.0
        if re.match(rf"^\s*(?:fig(?:ure)?\.?\s*{figure_number}\b|图\s*{figure_number}\b)", text, re.IGNORECASE):
            score += 1.3
        if "single snapshot compressed image" in text.lower():
            score += 0.22
        order = _order_of(block)
        if figure_order > 0 and order > 0:
            dist = abs(order - figure_order)
            if dist == 1:
                score += 0.42
            elif dist == 0:
                score += 0.28
            else:
                score += max(0.0, 0.18 - (0.04 * float(dist)))
        return score

    figure_blocks = [
        block
        for block in blocks
        if str(block.get("kind") or "").strip().lower() == "figure"
        and _figure_block_number(block) == figure_number
    ]
    current_primary_id = str(segment.get("primary_block_id") or "").strip()
    current_primary = lookup.get(current_primary_id) or {}

    figure_block: dict | None = None
    if figure_blocks:
        figure_blocks.sort(
            key=lambda block: (
                0 if str(block.get("block_id") or "").strip() == current_primary_id else 1,
                _order_of(block) if _order_of(block) > 0 else 10**9,
                str(block.get("block_id") or ""),
            )
        )
        figure_block = figure_blocks[0]

    figure_order = _order_of(figure_block or current_primary)
    caption_candidates = [
        block
        for block in blocks
        if str(block.get("kind") or "").strip().lower() in {"paragraph", "list_item", "blockquote"}
        and _extract_figure_number(str(block.get("raw_text") or block.get("text") or "")) == figure_number
    ]
    caption_block: dict | None = None
    if caption_candidates:
        caption_candidates.sort(
            key=lambda block: (
                -_caption_score(block, figure_order),
                _order_of(block) if _order_of(block) > 0 else 10**9,
                str(block.get("block_id") or ""),
            )
        )
        caption_block = caption_candidates[0]
        if _caption_score(caption_block, figure_order) < 0.18:
            caption_block = None

    return figure_block, caption_block


def _quote_excerpt_fragments(text: str) -> list[str]:
    raw = _strip_provenance_noise_text(text)
    if not raw:
        return []
    normalized = _QUOTE_ELLIPSIS_RE.sub(" … ", raw)
    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"\s+…\s+", normalized):
        clean = normalize_inline_markdown(part).strip(" \"'`*[](){}")
        if len(clean) < 14:
            continue
        key = normalize_match_text(clean)
        if (not key) or (key in seen):
            continue
        seen.add(key)
        out.append(clean[:360])
        if len(out) >= 4:
            break
    return out


def _ordered_fragment_match_score(needle: str, haystack: str) -> tuple[float, int]:
    fragments = _quote_excerpt_fragments(needle)
    if len(fragments) < 2:
        return 0.0, 0
    haystack_norm = normalize_match_text(haystack)
    if not haystack_norm:
        return 0.0, 0
    matched = 0
    cursor = 0
    for fragment in fragments:
        fragment_norm = normalize_match_text(fragment)
        if not fragment_norm:
            continue
        idx = haystack_norm.find(fragment_norm, cursor)
        if idx < 0:
            break
        matched += 1
        cursor = idx + len(fragment_norm)
    if matched <= 0:
        return 0.0, 0
    score = 0.26 * float(matched)
    if matched >= 2:
        score += 0.84
    if matched >= len(fragments):
        score += 0.24
    return float(score), int(matched)


def _quote_binding_score(needle: str, block: dict | None) -> tuple[float, str, int]:
    if not isinstance(block, dict):
        return -1.0, "", 0
    block_text_raw = str(block.get("raw_text") or block.get("text") or "").strip()
    block_text = normalize_inline_markdown(block_text_raw)
    probe = normalize_inline_markdown(needle)
    if not block_text or not probe:
        return -1.0, "", 0
    block_norm = normalize_match_text(block_text)
    probe_norm = normalize_match_text(probe)

    score = 0.0
    if probe_norm and probe_norm in block_norm:
        score += 1.54

    ordered_score, matched_fragments = _ordered_fragment_match_score(probe, block_text)
    score += ordered_score

    support_quote, quote_score = _best_evidence_quote_match(probe, block)
    score += 0.92 * float(quote_score)
    score += 0.52 * _text_token_overlap_score(probe, block_text)
    try:
        score += 0.34 * float(SequenceMatcher(None, probe_norm[:520], block_norm[:520]).ratio())
    except Exception:
        pass

    heading_path = str(block.get("heading_path") or "").strip()
    score += 0.08 * _text_token_overlap_score(probe, heading_path)
    if str(block.get("kind") or "").strip().lower() in {"paragraph", "blockquote"}:
        score += 0.05

    return float(score), str(support_quote or "").strip(), int(matched_fragments)


def _select_quote_claim_binding(
    segment: dict | None,
    block_lookup: dict[str, dict] | None = None,
) -> tuple[dict | None, str]:
    if not isinstance(segment, dict):
        return None, ""
    lookup = block_lookup or {}
    blocks = [
        block
        for block in lookup.values()
        if isinstance(block, dict)
        and str(block.get("kind") or "").strip().lower() in {"paragraph", "list_item", "blockquote"}
    ]
    if not blocks:
        return None, ""

    raw_markdown = str(segment.get("raw_markdown") or segment.get("text") or "").strip()
    quoted_spans = _extract_quoted_spans(raw_markdown, min_len=12)
    seeds = _dedupe_str_items(
        quoted_spans
        + [
            str(segment.get("anchor_text") or "").strip(),
            str(segment.get("evidence_quote") or "").strip(),
        ]
    )
    if not seeds and str(segment.get("kind") or "").strip().lower() == "blockquote":
        seeds = [normalize_inline_markdown(raw_markdown or str(segment.get("text") or ""))[:600]]
    if not seeds:
        return None, ""

    current_primary_id = str(segment.get("primary_block_id") or "").strip()
    current_primary = lookup.get(current_primary_id) if current_primary_id else None
    current_best = max((_quote_binding_score(seed, current_primary)[0] for seed in seeds), default=-1.0)

    best_block: dict | None = None
    best_quote = ""
    best_score = -1.0
    best_fragment_hits = 0
    for block in blocks:
        block_best_score = -1.0
        block_best_quote = ""
        block_best_hits = 0
        for seed in seeds:
            score, support_quote, matched_fragments = _quote_binding_score(seed, block)
            if score > block_best_score:
                block_best_score = score
                block_best_quote = support_quote
                block_best_hits = matched_fragments
        if block_best_score > best_score:
            best_block = block
            best_quote = block_best_quote
            best_score = block_best_score
            best_fragment_hits = block_best_hits

    if not isinstance(best_block, dict):
        return None, ""
    if best_score < 0.96:
        return None, ""
    if current_primary_id and str(best_block.get("block_id") or "").strip() == current_primary_id:
        return best_block, best_quote
    if current_best > 0 and best_score < (current_best + 0.12):
        return None, ""

    if best_fragment_hits >= 2:
        block_text = str(best_block.get("raw_text") or best_block.get("text") or "").strip()
        if block_text:
            best_quote = normalize_inline_markdown(block_text)[:900]
    elif not best_quote:
        best_quote = _best_evidence_quote(seeds[0], best_block)

    return best_block, str(best_quote or "").strip()


def _strict_identity_missing_reasons(segment: dict | None) -> list[str]:
    if not isinstance(segment, dict):
        return ["invalid_segment"]
    reasons: list[str] = []
    primary_block_id = str(segment.get("primary_block_id") or "").strip()
    evidence_block_ids = [
        str(item or "").strip()
        for item in list(segment.get("evidence_block_ids") or [])
        if str(item or "").strip()
    ]
    anchor_kind = str(segment.get("anchor_kind") or "").strip().lower()
    anchor_text = str(segment.get("anchor_text") or "").strip()
    evidence_quote = str(segment.get("evidence_quote") or "").strip()
    if not primary_block_id:
        reasons.append("missing_primary_block_id")
    if not evidence_block_ids:
        reasons.append("missing_evidence_block_ids")
    if not anchor_kind:
        reasons.append("missing_anchor_kind")
    if not (anchor_text or evidence_quote):
        reasons.append("missing_anchor_text_or_evidence_quote")
    return reasons


def _pick_blocks_with_llm(
    *,
    ds: DeepSeekChat | None,
    segment_text: str,
    segment_kind: str,
    is_formula_segment: bool,
    candidate_rows: list[dict],
    max_pick: int = 2,
) -> list[str]:
    if ds is None:
        return []
    rows = [row for row in list(candidate_rows or []) if isinstance(row, dict)]
    if len(rows) < 2:
        return []

    cand_rows = rows[: min(6, len(rows))]
    allowed_ids: set[str] = set()
    cand_lines: list[str] = []
    for idx, row in enumerate(cand_rows, start=1):
        block = row.get("block")
        if not isinstance(block, dict):
            continue
        block_id = str(block.get("block_id") or "").strip()
        if not block_id:
            continue
        allowed_ids.add(block_id)
        score = float(row.get("score") or 0.0)
        kind = str(block.get("kind") or "").strip()
        heading = str(block.get("heading_path") or "").strip()
        number = int(block.get("number") or 0)
        block_text_raw = str(block.get("raw_text") or block.get("text") or "")
        text = normalize_inline_markdown(block_text_raw)[:420]
        cand_lines.append(
            f"{idx}. block_id={block_id} kind={kind} number={number} score={score:.3f}\n"
            f"   heading={heading}\n"
            f"   text={text}"
        )
    if len(cand_lines) < 2:
        return []

    sys_msg = (
        "You are a strict evidence mapper. Choose source blocks that directly support the answer segment. "
        "If no candidate clearly supports, return empty ids."
    )
    formula_rule = (
        "- For formula segments, prioritize same equation number; if absent, allow mathematically equivalent form with different symbols.\n"
        if is_formula_segment else
        "- For non-formula segments, choose directly supporting original statements, not broad topic neighbors.\n"
    )
    user_msg = (
        "Task: choose at most "
        f"{max(1, int(max_pick))} block_id values.\n"
        "Rules:\n"
        "- Prefer exact semantic grounding over loose topic similarity.\n"
        + formula_rule
        + "- Do NOT guess.\n"
        + "Return JSON only: {\"ids\": [\"block_id\", ...]}.\n\n"
        + f"segment_kind={segment_kind}\n"
        + f"is_formula_segment={int(bool(is_formula_segment))}\n"
        + f"segment_text={segment_text[:520]}\n\n"
        + "candidates:\n"
        + "\n".join(cand_lines)
    )
    try:
        out = ds.chat(
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=220,
        )
    except Exception:
        return []

    obj_text = _extract_json_object_text(out)
    if not obj_text:
        return []
    try:
        obj = json.loads(obj_text)
    except Exception:
        return []
    ids_raw = obj.get("ids") if isinstance(obj, dict) else []
    if not isinstance(ids_raw, list):
        return []
    picked: list[str] = []
    seen: set[str] = set()
    for it in ids_raw:
        bid = str(it or "").strip()
        if (not bid) or (bid in seen) or (bid not in allowed_ids):
            continue
        seen.add(bid)
        picked.append(bid)
        if len(picked) >= max(1, int(max_pick)):
            break
    return picked


def _ensure_provenance_block_entry(block_map: dict[str, dict], block: dict) -> None:
    block_id = str(block.get("block_id") or "").strip()
    if (not block_id) or (block_id in block_map):
        return
    block_map[block_id] = {
        "block_id": block_id,
        "anchor_id": str(block.get("anchor_id") or "").strip(),
        "kind": str(block.get("kind") or "").strip(),
        "heading_path": str(block.get("heading_path") or "").strip(),
        "text": str(block.get("text") or ""),
        "line_start": int(block.get("line_start") or 0),
        "line_end": int(block.get("line_end") or 0),
        "number": int(block.get("number") or 0),
    }


def _collect_paper_guide_block_pool(
    *,
    blocks: list[dict],
    answer_hits: list[dict],
    bound_source_path: str,
    bound_source_name: str,
) -> list[dict]:
    best_by_block: dict[str, dict] = {}

    def _block_formula_seed_score(block: dict) -> float:
        kind = str(block.get("kind") or "").strip().lower()
        text = str(block.get("text") or "")
        score = 0.0
        if kind == "equation":
            score += 0.34
        if "$$" in text or "\\begin{equation" in text.lower():
            score += 0.38
        if has_equation_signal(text):
            score += 0.22
        try:
            if int(block.get("number") or 0) > 0:
                score += 0.18
        except Exception:
            pass
        return score

    for hit in answer_hits or []:
        if not isinstance(hit, dict):
            continue
        if not _is_hit_from_bound_source(
            hit,
            bound_source_path=bound_source_path,
            bound_source_name=bound_source_name,
        ):
            continue
        meta = hit.get("meta", {}) or {}
        if not isinstance(meta, dict):
            meta = {}
        heading_hints: list[str] = []
        for value in (
            meta.get("ref_best_heading_path"),
            meta.get("heading_path"),
            meta.get("top_heading"),
        ):
            text = normalize_inline_markdown(str(value or ""))
            if text and text not in heading_hints:
                heading_hints.append(text)
        raw_locs = meta.get("ref_locs")
        if isinstance(raw_locs, list):
            for loc in raw_locs[:4]:
                if not isinstance(loc, dict):
                    continue
                hp = normalize_inline_markdown(str(loc.get("heading_path") or loc.get("heading") or ""))
                if hp and hp not in heading_hints:
                    heading_hints.append(hp)
        snippet_hints: list[str] = []
        raw_snips = meta.get("ref_show_snippets")
        if isinstance(raw_snips, list):
            for item in raw_snips[:4]:
                for text in _expand_match_snippet_hints(str(item or ""), max_items=4):
                    if text and text not in snippet_hints:
                        snippet_hints.append(text)
        hit_text = normalize_inline_markdown(str(hit.get("text") or ""))
        for text in _expand_match_snippet_hints(str(hit_text or ""), max_items=6):
            if text and text not in snippet_hints:
                snippet_hints.append(text)
        prefer_kind = str(meta.get("anchor_target_kind") or "").strip().lower()
        try:
            target_number = int(meta.get("anchor_target_number") or 0)
        except Exception:
            target_number = 0

        match_jobs: list[tuple[str, str]] = []
        if snippet_hints:
            for snippet in snippet_hints[:4]:
                for heading in heading_hints[:2] or [""]:
                    match_jobs.append((snippet, heading))
        elif heading_hints:
            for heading in heading_hints[:2]:
                match_jobs.append(("", heading))
        elif target_number > 0:
            match_jobs.append(("", ""))

        for snippet, heading in match_jobs[:8]:
            rows = match_source_blocks(
                blocks,
                snippet=snippet,
                heading_path=heading,
                prefer_kind=prefer_kind,
                target_number=target_number,
                limit=3,
            )
            for row in rows:
                block = row.get("block")
                if not isinstance(block, dict):
                    continue
                block_id = str(block.get("block_id") or "").strip()
                if not block_id:
                    continue
                score = float(row.get("score") or 0.0)
                prev = best_by_block.get(block_id)
                if prev is None or score > float(prev.get("score") or 0.0):
                    best_by_block[block_id] = {
                        "score": score,
                        "block": block,
                    }
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        block_id = str(block.get("block_id") or "").strip()
        if not block_id:
            continue
        seed_score = _block_formula_seed_score(block)
        if seed_score <= 0.0:
            continue
        prev = best_by_block.get(block_id)
        if prev is None or seed_score > float(prev.get("score") or 0.0):
            best_by_block[block_id] = {
                "score": seed_score,
                "block": block,
            }

    ranked = sorted(best_by_block.values(), key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return ranked[:24]


def _apply_provenance_strict_identity_contract(
    segments: list[dict] | None,
) -> tuple[list[dict], dict[str, object]]:
    hardened: list[dict] = []
    must_locate_candidate_count = 0
    strict_identity_count = 0
    identity_missing_reasons: dict[str, int] = {}
    identity_missing_segments: list[dict[str, object]] = []

    for seg0 in list(segments or []):
        if not isinstance(seg0, dict):
            continue
        seg = dict(seg0)
        missing_reasons: list[str] = []
        if bool(seg.get("must_locate")):
            must_locate_candidate_count += 1
            missing_reasons = _strict_identity_missing_reasons(seg)
            if missing_reasons:
                seg["must_locate"] = False
                for reason in missing_reasons:
                    identity_missing_reasons[reason] = int(identity_missing_reasons.get(reason) or 0) + 1
                identity_missing_segments.append(
                    {
                        "segment_id": str(seg.get("segment_id") or "").strip(),
                        "claim_type": str(seg.get("claim_type") or "").strip(),
                        "reasons": missing_reasons,
                    }
                )
            else:
                strict_identity_count += 1
        seg["strict_identity_missing_reasons"] = missing_reasons
        hardened.append(seg)

    must_locate_count = sum(1 for seg in hardened if bool(seg.get("must_locate")))
    strict_identity_ready = bool(must_locate_candidate_count == strict_identity_count)
    if must_locate_candidate_count <= 0:
        strict_identity_ready = True
    return hardened, {
        "provenance_schema_version": int(_PAPER_GUIDE_PROVENANCE_SCHEMA_VERSION),
        "must_locate_candidate_count": int(must_locate_candidate_count),
        "must_locate_count": int(must_locate_count),
        "strict_identity_count": int(strict_identity_count),
        "strict_identity_ready": bool(strict_identity_ready),
        "identity_missing_reasons": identity_missing_reasons,
        "identity_missing_segments": identity_missing_segments,
    }


def _apply_provenance_required_coverage_contract(
    segments: list[dict] | None,
    *,
    block_lookup: dict[str, dict] | None = None,
) -> list[dict]:
    out: list[dict] = []
    lookup = block_lookup or {}
    block_order = {
        str(block_id): int((block or {}).get("order_index") or 0)
        for block_id, block in lookup.items()
        if str(block_id or "").strip()
    }
    formula_bundles: list[dict[str, object]] = []
    non_source_scope_active = False

    for seg0 in list(segments or []):
        if not isinstance(seg0, dict):
            continue
        seg = dict(seg0)
        claim_type = str(seg.get("claim_type") or "").strip().lower()
        evidence_mode = str(seg.get("evidence_mode") or "").strip().lower()
        segment_id = str(seg.get("segment_id") or "").strip()
        primary_block_id = str(seg.get("primary_block_id") or "").strip()
        raw_markdown = str(seg.get("raw_markdown") or seg.get("text") or "").strip()
        explicit_non_source = _is_explicit_non_source_segment(raw_markdown)
        seg["related_block_ids"] = _dedupe_str_items(seg.get("related_block_ids") or [])
        if claim_type in {"quote_claim", "blockquote_claim"} and evidence_mode == "direct":
            quote_block, quote_support = _select_quote_claim_binding(seg, lookup)
            if isinstance(quote_block, dict):
                quote_anchor = _longest_quoted_span(raw_markdown or str(seg.get("text") or ""), min_len=12)
                primary_block_id = str(quote_block.get("block_id") or "").strip()
                primary_anchor_id = str(quote_block.get("anchor_id") or "").strip()
                primary_heading_path = str(quote_block.get("heading_path") or "").strip()
                old_primary_block_id = str(seg.get("primary_block_id") or "").strip()
                evidence_block_ids = _dedupe_str_items([primary_block_id] + list(seg.get("evidence_block_ids") or []))
                support_block_ids = _dedupe_str_items(
                    ([old_primary_block_id] if old_primary_block_id and old_primary_block_id != primary_block_id else [])
                    + list(seg.get("support_block_ids") or [])
                )
                seg["primary_block_id"] = primary_block_id
                seg["primary_anchor_id"] = primary_anchor_id
                seg["primary_heading_path"] = primary_heading_path
                seg["evidence_block_ids"] = evidence_block_ids
                seg["support_block_ids"] = support_block_ids
                if quote_support:
                    seg["evidence_quote"] = quote_support[:900]
                seg["anchor_text"] = str(quote_anchor or quote_support or seg.get("text") or "").strip()[:600]
        primary_block = lookup.get(primary_block_id) if primary_block_id else None
        if claim_type == "inline_formula_claim" and evidence_mode == "direct":
            inline_block, inline_anchor_text, inline_score = _select_inline_formula_claim_binding(seg, lookup)
            if isinstance(inline_block, dict):
                inline_block_id = str(inline_block.get("block_id") or "").strip()
                inline_anchor_id = str(inline_block.get("anchor_id") or "").strip()
                inline_heading_path = str(inline_block.get("heading_path") or "").strip()
                old_primary_block_id = str(seg.get("primary_block_id") or "").strip()
                evidence_block_ids = _dedupe_str_items(
                    [inline_block_id]
                    + ([old_primary_block_id] if old_primary_block_id and old_primary_block_id != inline_block_id else [])
                    + list(seg.get("evidence_block_ids") or [])
                )
                support_block_ids = _dedupe_str_items(
                    ([old_primary_block_id] if old_primary_block_id and old_primary_block_id != inline_block_id else [])
                    + list(seg.get("support_block_ids") or [])
                )
                seg["must_locate"] = True
                seg["anchor_kind"] = "inline_formula"
                seg["anchor_text"] = str(inline_anchor_text or seg.get("anchor_text") or "").strip()[:900]
                seg["evidence_quote"] = _best_evidence_quote(str(seg.get("text") or ""), inline_block)[:900]
                seg["primary_block_id"] = inline_block_id
                seg["primary_anchor_id"] = inline_anchor_id
                seg["primary_heading_path"] = inline_heading_path
                seg["evidence_block_ids"] = evidence_block_ids
                seg["support_block_ids"] = support_block_ids
                seg["related_block_ids"] = _dedupe_str_items(
                    ([old_primary_block_id] if old_primary_block_id and old_primary_block_id != inline_block_id else [])
                    + list(seg.get("related_block_ids") or [])
                )
                seg["formula_origin"] = "explanation"
                seg["mapping_quality"] = max(float(seg.get("mapping_quality") or 0.0), float(inline_score))
                primary_block_id = inline_block_id
                primary_block = inline_block
            else:
                seg["must_locate"] = False
                seg["locate_policy"] = "hidden"
                seg["locate_surface_policy"] = "hidden"
                seg["formula_origin"] = "derived"
                seg["claim_group_id"] = ""
                seg["claim_group_kind"] = ""
                out.append(seg)
                if explicit_non_source and _opens_non_source_scope(raw_markdown):
                    non_source_scope_active = True
                continue
        if claim_type == "formula_claim" and evidence_mode == "direct":
            inline_formula_surface = _pick_primary_inline_formula_span(raw_markdown or str(seg.get("text") or ""))
            display_formula_surface = _extract_display_formula_snippet(raw_markdown or str(seg.get("text") or ""))
            if inline_formula_surface and (not display_formula_surface):
                inline_block, inline_anchor_text, inline_score = _select_inline_formula_claim_binding(seg, lookup)
                if isinstance(inline_block, dict):
                    inline_block_id = str(inline_block.get("block_id") or "").strip()
                    inline_anchor_id = str(inline_block.get("anchor_id") or "").strip()
                    inline_heading_path = str(inline_block.get("heading_path") or "").strip()
                    old_primary_block_id = str(seg.get("primary_block_id") or "").strip()
                    evidence_block_ids = _dedupe_str_items(
                        [inline_block_id]
                        + ([old_primary_block_id] if old_primary_block_id and old_primary_block_id != inline_block_id else [])
                        + list(seg.get("evidence_block_ids") or [])
                    )
                    support_block_ids = _dedupe_str_items(
                        ([old_primary_block_id] if old_primary_block_id and old_primary_block_id != inline_block_id else [])
                        + list(seg.get("support_block_ids") or [])
                    )
                    seg["claim_type"] = "inline_formula_claim"
                    seg["must_locate"] = True
                    seg["anchor_kind"] = "inline_formula"
                    seg["anchor_text"] = str(inline_anchor_text or inline_formula_surface).strip()[:900]
                    seg["evidence_quote"] = _best_evidence_quote(str(seg.get("text") or ""), inline_block)[:900]
                    seg["primary_block_id"] = inline_block_id
                    seg["primary_anchor_id"] = inline_anchor_id
                    seg["primary_heading_path"] = inline_heading_path
                    seg["evidence_block_ids"] = evidence_block_ids
                    seg["support_block_ids"] = support_block_ids
                    seg["related_block_ids"] = _dedupe_str_items(
                        ([old_primary_block_id] if old_primary_block_id and old_primary_block_id != inline_block_id else [])
                        + list(seg.get("related_block_ids") or [])
                    )
                    seg["formula_origin"] = "explanation"
                    seg["mapping_quality"] = max(float(seg.get("mapping_quality") or 0.0), float(inline_score))
                    primary_block_id = inline_block_id
                    primary_block = inline_block
                    claim_type = "inline_formula_claim"
                else:
                    seg["must_locate"] = False
                    seg["locate_policy"] = "hidden"
                    seg["locate_surface_policy"] = "hidden"
                    seg["formula_origin"] = "derived"
                    seg["claim_group_id"] = ""
                    seg["claim_group_kind"] = ""
                    out.append(seg)
                    if explicit_non_source and _opens_non_source_scope(raw_markdown):
                        non_source_scope_active = True
                    continue
            if not _is_formula_claim_source_grounded(seg, primary_block):
                inline_block, inline_anchor_text, inline_score = _select_inline_formula_claim_binding(seg, lookup)
                if isinstance(inline_block, dict):
                    inline_block_id = str(inline_block.get("block_id") or "").strip()
                    inline_anchor_id = str(inline_block.get("anchor_id") or "").strip()
                    inline_heading_path = str(inline_block.get("heading_path") or "").strip()
                    old_primary_block_id = str(seg.get("primary_block_id") or "").strip()
                    evidence_block_ids = _dedupe_str_items(
                        [inline_block_id]
                        + ([old_primary_block_id] if old_primary_block_id and old_primary_block_id != inline_block_id else [])
                        + list(seg.get("evidence_block_ids") or [])
                    )
                    support_block_ids = _dedupe_str_items(
                        ([old_primary_block_id] if old_primary_block_id and old_primary_block_id != inline_block_id else [])
                        + list(seg.get("support_block_ids") or [])
                    )
                    seg["claim_type"] = "inline_formula_claim"
                    seg["must_locate"] = True
                    seg["anchor_kind"] = "inline_formula"
                    seg["anchor_text"] = str(inline_anchor_text or "").strip()[:900]
                    seg["evidence_quote"] = _best_evidence_quote(str(seg.get("text") or ""), inline_block)[:900]
                    seg["primary_block_id"] = inline_block_id
                    seg["primary_anchor_id"] = inline_anchor_id
                    seg["primary_heading_path"] = inline_heading_path
                    seg["evidence_block_ids"] = evidence_block_ids
                    seg["support_block_ids"] = support_block_ids
                    seg["related_block_ids"] = _dedupe_str_items(
                        ([old_primary_block_id] if old_primary_block_id and old_primary_block_id != inline_block_id else [])
                        + list(seg.get("related_block_ids") or [])
                    )
                    seg["formula_origin"] = "explanation"
                    seg["mapping_quality"] = max(float(seg.get("mapping_quality") or 0.0), float(inline_score))
                    primary_block_id = inline_block_id
                    primary_block = inline_block
                    claim_type = "inline_formula_claim"
                else:
                    seg["must_locate"] = False
                    seg["locate_policy"] = "hidden"
                    seg["locate_surface_policy"] = "hidden"
                    seg["formula_origin"] = "derived"
                    seg["claim_group_id"] = ""
                    seg["claim_group_kind"] = ""
                    out.append(seg)
                    if explicit_non_source and _opens_non_source_scope(raw_markdown):
                        non_source_scope_active = True
                    continue
        if claim_type == "figure_claim" and evidence_mode == "direct":
            figure_block, caption_block = _select_figure_claim_binding(seg, lookup)
            if figure_block or caption_block:
                primary_block = figure_block or caption_block or {}
                primary_block_id = str(primary_block.get("block_id") or "").strip()
                primary_anchor_id = str(primary_block.get("anchor_id") or "").strip()
                primary_heading_path = str(primary_block.get("heading_path") or "").strip()
                old_primary_block_id = str(seg.get("primary_block_id") or "").strip()
                evidence_block_ids = _dedupe_str_items(
                    [primary_block_id]
                    + ([str(caption_block.get("block_id") or "").strip()] if isinstance(caption_block, dict) else [])
                    + list(seg.get("evidence_block_ids") or [])
                )
                support_block_ids = _dedupe_str_items(
                    ([old_primary_block_id] if old_primary_block_id and old_primary_block_id != primary_block_id else [])
                    + list(seg.get("support_block_ids") or [])
                )
                seg["primary_block_id"] = primary_block_id
                seg["primary_anchor_id"] = primary_anchor_id
                seg["primary_heading_path"] = primary_heading_path
                seg["evidence_block_ids"] = evidence_block_ids
                seg["support_block_ids"] = support_block_ids
                if isinstance(caption_block, dict):
                    caption_text = str(caption_block.get("raw_text") or caption_block.get("text") or "").strip()
                    if caption_text:
                        seg["anchor_text"] = caption_text[:480]
                        seg["evidence_quote"] = caption_text[:900]
        if non_source_scope_active and _is_non_source_scope_boundary(raw_markdown):
            non_source_scope_active = False
        non_source_scoped = bool(explicit_non_source or non_source_scope_active)
        locate_policy = "hidden"
        locate_surface_policy = "hidden"
        claim_group_id = ""
        claim_group_kind = ""
        formula_origin = str(seg.get("formula_origin") or "").strip().lower()

        if non_source_scoped:
            locate_policy = "hidden"
            locate_surface_policy = "hidden"
            seg["must_locate"] = False
        elif claim_type in {"quote_claim", "blockquote_claim"} and evidence_mode == "direct":
            locate_policy = "required"
            locate_surface_policy = "primary"
            claim_group_kind = "quote_bundle"
            claim_group_id = f"quote_bundle:{primary_block_id or segment_id or uuid.uuid4().hex[:8]}"
        elif claim_type == "formula_claim" and evidence_mode == "direct":
            locate_policy = "required"
            locate_surface_policy = "primary"
            formula_origin = "source"
        elif claim_type == "inline_formula_claim" and evidence_mode == "direct":
            locate_policy = "required"
            locate_surface_policy = "secondary"
            formula_origin = "explanation"
        elif claim_type == "figure_claim" and evidence_mode == "direct":
            locate_policy = "required"
            locate_surface_policy = "primary"
        elif claim_type == "shell_sentence":
            locate_policy = "hidden"
        elif bool(seg.get("must_locate")):
            locate_policy = "required"
            locate_surface_policy = "primary"
        elif evidence_mode == "direct":
            locate_policy = "optional"
            locate_surface_policy = "hidden"

        if claim_type == "formula_claim" and evidence_mode == "direct" and not non_source_scoped:
            bundle_key = primary_block_id or segment_id or uuid.uuid4().hex[:8]
            claim_group_id = f"formula_bundle:{bundle_key}"
            claim_group_kind = "formula_bundle"
            formula_bundles.append(
                {
                    "segment_id": segment_id,
                    "segment_index": int(seg.get("segment_index") or 0),
                    "primary_block_id": primary_block_id,
                    "primary_anchor_id": str(seg.get("primary_anchor_id") or "").strip(),
                    "primary_heading_path": str(seg.get("primary_heading_path") or "").strip(),
                    "equation_number": int(seg.get("equation_number") or 0),
                    "anchor_text": str(seg.get("anchor_text") or "").strip(),
                    "evidence_quote": str(seg.get("evidence_quote") or "").strip(),
                    "claim_group_id": claim_group_id,
                    "claim_group_kind": claim_group_kind,
                    "source_block_order": int(block_order.get(primary_block_id) or 0),
                    "alignment_score": _formula_claim_alignment_score(seg, primary_block),
                }
            )
        elif claim_type == "inline_formula_claim" and evidence_mode == "direct" and not non_source_scoped:
            related_formula_block_id = next(
                (
                    str(item or "").strip()
                    for item in list(seg.get("related_block_ids") or [])
                    if str(item or "").strip()
                ),
                "",
            )
            bundle_key = related_formula_block_id or primary_block_id or segment_id or uuid.uuid4().hex[:8]
            claim_group_id = f"formula_bundle:{bundle_key}"
            claim_group_kind = "formula_bundle"

        seg["locate_policy"] = locate_policy
        seg["locate_surface_policy"] = locate_surface_policy
        seg["claim_group_id"] = claim_group_id
        seg["claim_group_kind"] = claim_group_kind
        seg["formula_origin"] = formula_origin
        out.append(seg)
        if explicit_non_source and _opens_non_source_scope(raw_markdown):
            non_source_scope_active = True

    if not out or not formula_bundles:
        inline_formula_groups: dict[str, list[dict]] = {}
        for seg in out:
            if not isinstance(seg, dict):
                continue
            if str(seg.get("claim_type") or "").strip().lower() != "inline_formula_claim":
                continue
            if str(seg.get("claim_group_kind") or "").strip().lower() != "formula_bundle":
                continue
            inline_formula_groups.setdefault(str(seg.get("claim_group_id") or "").strip(), []).append(seg)
        for items in inline_formula_groups.values():
            dedupe_by_key: dict[str, list[dict]] = {}
            for seg in items:
                key = " :: ".join(
                    [
                        str(seg.get("primary_block_id") or "").strip(),
                        _normalize_formula_compare_text(str(seg.get("anchor_text") or ""))[:420],
                    ]
                ).strip()
                if not key:
                    continue
                dedupe_by_key.setdefault(key, []).append(seg)
            for dup_items in dedupe_by_key.values():
                if len(dup_items) <= 1:
                    continue
                dup_items.sort(
                    key=lambda seg: (
                        -len(str(seg.get("anchor_text") or seg.get("evidence_quote") or "")),
                        int(seg.get("segment_index") or 0),
                    )
                )
                keep = dup_items[0]
                keep["locate_surface_policy"] = "secondary"
                for seg in dup_items[1:]:
                    seg["must_locate"] = False
                    seg["locate_policy"] = "hidden"
                    seg["locate_surface_policy"] = "hidden"
        quote_groups: dict[str, list[dict]] = {}
        for seg in out:
            if not isinstance(seg, dict):
                continue
            if str(seg.get("claim_group_kind") or "").strip().lower() != "quote_bundle":
                continue
            quote_groups.setdefault(str(seg.get("claim_group_id") or "").strip(), []).append(seg)
        for items in quote_groups.values():
            dedupe_by_key: dict[str, list[dict]] = {}
            for seg in items:
                key = " :: ".join(
                    [
                        str(seg.get("primary_block_id") or "").strip(),
                        normalize_match_text(
                            str(seg.get("anchor_text") or seg.get("evidence_quote") or seg.get("text") or "")
                        )[:420],
                    ]
                ).strip()
                if not key:
                    continue
                dedupe_by_key.setdefault(key, []).append(seg)
            for dup_items in dedupe_by_key.values():
                if len(dup_items) <= 1:
                    continue
                dup_items.sort(
                    key=lambda seg: (
                        -len(str(seg.get("anchor_text") or seg.get("evidence_quote") or "")),
                        int(seg.get("segment_index") or 0),
                    )
                )
                keep = dup_items[0]
                keep["locate_surface_policy"] = "primary"
                for seg in dup_items[1:]:
                    seg["must_locate"] = False
                    seg["locate_policy"] = "hidden"
                    seg["locate_surface_policy"] = "hidden"
        return _assign_claim_group_targets(out)

    bundle_items_by_id: dict[str, list[dict[str, object]]] = {}
    for bundle in formula_bundles:
        bundle_id = str(bundle.get("claim_group_id") or "").strip()
        if not bundle_id:
            continue
        bundle_items_by_id.setdefault(bundle_id, []).append(bundle)

    bundle_formula_segment_ids: set[str] = set()
    representative_bundles: list[dict[str, object]] = []
    for bundle_id, items in bundle_items_by_id.items():
        ranked_items = sorted(
            items,
            key=lambda item: (
                -float(item.get("alignment_score") or 0.0),
                -(1 if int(item.get("equation_number") or 0) > 0 else 0),
                -len(str(item.get("anchor_text") or item.get("evidence_quote") or "")),
                int(item.get("segment_index") or 0),
            ),
        )
        if not ranked_items:
            continue
        primary_item = ranked_items[0]
        representative_bundles.append(primary_item)
        primary_segment_id = str(primary_item.get("segment_id") or "").strip()
        for seg in out:
            if str(seg.get("claim_group_id") or "").strip() != bundle_id:
                continue
            if str(seg.get("claim_type") or "").strip().lower() != "formula_claim":
                continue
            segment_id = str(seg.get("segment_id") or "").strip()
            if not segment_id:
                continue
            bundle_formula_segment_ids.add(segment_id)
            if segment_id == primary_segment_id:
                seg["must_locate"] = True
                seg["locate_policy"] = "required"
                seg["locate_surface_policy"] = "primary"
                seg["formula_origin"] = "source"
                continue
            seg["must_locate"] = False
            seg["locate_policy"] = "hidden"
            seg["locate_surface_policy"] = "hidden"
            seg["formula_origin"] = "derived"

    used_segment_ids: set[str] = set(bundle_formula_segment_ids)
    for bundle in representative_bundles:
        formula_segment_id = str(bundle.get("segment_id") or "").strip()
        if formula_segment_id:
            used_segment_ids.add(formula_segment_id)
        formula_block_id = str(bundle.get("primary_block_id") or "").strip()
        formula_anchor_id = str(bundle.get("primary_anchor_id") or "").strip()
        formula_heading_path = str(bundle.get("primary_heading_path") or "").strip()
        formula_number = int(bundle.get("equation_number") or 0)
        formula_text = str(bundle.get("anchor_text") or bundle.get("evidence_quote") or "").strip()
        formula_order = int(bundle.get("source_block_order") or 0)
        claim_group_id = str(bundle.get("claim_group_id") or "").strip()
        claim_group_kind = str(bundle.get("claim_group_kind") or "").strip()
        formula_seg_index = int(bundle.get("segment_index") or 0)
        if not formula_block_id:
            continue

        for seg in out:
            segment_id = str(seg.get("segment_id") or "").strip()
            if (not segment_id) or (segment_id in used_segment_ids):
                continue
            evidence_mode = str(seg.get("evidence_mode") or "").strip().lower()
            if evidence_mode != "direct":
                continue
            claim_type = str(seg.get("claim_type") or "").strip().lower()
            if claim_type in {"formula_claim", "inline_formula_claim", "quote_claim", "blockquote_claim", "figure_claim"}:
                continue
            raw_markdown = str(seg.get("raw_markdown") or seg.get("text") or "").strip()
            segment_text = str(seg.get("text") or "").strip()
            if not segment_text:
                continue
            source_primary_block_id = str(seg.get("primary_block_id") or "").strip()
            source_order = int(block_order.get(source_primary_block_id) or 0)
            source_distance = 0
            if formula_order > 0 and source_order > 0:
                source_distance = abs(source_order - formula_order)
            else:
                source_distance = abs(int(seg.get("segment_index") or 0) - formula_seg_index)
            if source_distance <= 0 or source_distance > 2:
                continue
            score = _equation_explanation_score(
                segment_text=segment_text,
                raw_markdown=raw_markdown,
                formula_text=formula_text,
                equation_number=formula_number,
                source_distance=source_distance,
            )
            if score < 0.58:
                continue
            explanation_block = lookup.get(source_primary_block_id) if source_primary_block_id else None
            explanation_block_id = str((explanation_block or {}).get("block_id") or source_primary_block_id).strip()
            explanation_anchor_id = str((explanation_block or {}).get("anchor_id") or seg.get("primary_anchor_id") or "").strip()
            explanation_heading_path = str((explanation_block or {}).get("heading_path") or seg.get("primary_heading_path") or "").strip()
            explanation_anchor_kind = str(seg.get("anchor_kind") or "").strip().lower() or "sentence"
            if explanation_anchor_kind in {"equation", "figure", "quote"}:
                explanation_anchor_kind = "sentence"
            related_block_ids = _dedupe_str_items([formula_block_id] + list(seg.get("related_block_ids") or []))
            if (not explanation_block_id) or explanation_block_id == formula_block_id:
                seg["claim_type"] = "equation_explanation_claim"
                seg["must_locate"] = False
                seg["locate_policy"] = "hidden"
                seg["locate_surface_policy"] = "hidden"
                seg["formula_origin"] = "explanation"
                seg["claim_group_id"] = claim_group_id
                seg["claim_group_kind"] = claim_group_kind
                seg["related_block_ids"] = related_block_ids
                used_segment_ids.add(segment_id)
                continue
            evidence_block_ids = _dedupe_str_items(
                [explanation_block_id, formula_block_id]
                + list(seg.get("evidence_block_ids") or [])
            )
            support_block_ids = _dedupe_str_items(
                ([formula_block_id] if formula_block_id and formula_block_id != explanation_block_id else [])
                + ([source_primary_block_id] if source_primary_block_id and source_primary_block_id not in {formula_block_id, explanation_block_id} else [])
                + list(seg.get("support_block_ids") or [])
            )
            seg["claim_type"] = "equation_explanation_claim"
            seg["must_locate"] = True
            seg["anchor_kind"] = explanation_anchor_kind
            seg["equation_number"] = int(formula_number or 0)
            seg["primary_block_id"] = explanation_block_id
            seg["primary_anchor_id"] = explanation_anchor_id
            seg["primary_heading_path"] = explanation_heading_path or formula_heading_path
            seg["evidence_block_ids"] = evidence_block_ids
            seg["support_block_ids"] = support_block_ids
            seg["locate_policy"] = "required"
            seg["locate_surface_policy"] = "secondary"
            seg["claim_group_id"] = claim_group_id
            seg["claim_group_kind"] = claim_group_kind
            seg["formula_origin"] = "explanation"
            seg["related_block_ids"] = related_block_ids
            anchor_text = str(seg.get("anchor_text") or "").strip()
            if not anchor_text:
                seg["anchor_text"] = str(segment_text or raw_markdown)[:900]
            used_segment_ids.add(segment_id)

    quote_groups: dict[str, list[dict]] = {}
    for seg in out:
        if not isinstance(seg, dict):
            continue
        if str(seg.get("claim_group_kind") or "").strip().lower() != "quote_bundle":
            continue
        quote_groups.setdefault(str(seg.get("claim_group_id") or "").strip(), []).append(seg)
    for items in quote_groups.values():
        dedupe_by_key: dict[str, list[dict]] = {}
        for seg in items:
            key = " :: ".join(
                [
                    str(seg.get("primary_block_id") or "").strip(),
                    normalize_match_text(
                        str(seg.get("anchor_text") or seg.get("evidence_quote") or seg.get("text") or "")
                    )[:420],
                ]
            ).strip()
            if not key:
                continue
            dedupe_by_key.setdefault(key, []).append(seg)
        for dup_items in dedupe_by_key.values():
            if len(dup_items) <= 1:
                continue
            dup_items.sort(
                key=lambda seg: (
                    -len(str(seg.get("anchor_text") or seg.get("evidence_quote") or "")),
                    int(seg.get("segment_index") or 0),
                )
            )
            keep = dup_items[0]
            keep["locate_surface_policy"] = "primary"
            for seg in dup_items[1:]:
                seg["must_locate"] = False
                seg["locate_policy"] = "hidden"
                seg["locate_surface_policy"] = "hidden"

    inline_formula_groups: dict[str, list[dict]] = {}
    for seg in out:
        if not isinstance(seg, dict):
            continue
        if str(seg.get("claim_type") or "").strip().lower() != "inline_formula_claim":
            continue
        if str(seg.get("claim_group_kind") or "").strip().lower() != "formula_bundle":
            continue
        inline_formula_groups.setdefault(str(seg.get("claim_group_id") or "").strip(), []).append(seg)
    for items in inline_formula_groups.values():
        dedupe_by_key: dict[str, list[dict]] = {}
        for seg in items:
            key = " :: ".join(
                [
                    str(seg.get("primary_block_id") or "").strip(),
                    _normalize_formula_compare_text(str(seg.get("anchor_text") or ""))[:420],
                ]
            ).strip()
            if not key:
                continue
            dedupe_by_key.setdefault(key, []).append(seg)
        for dup_items in dedupe_by_key.values():
            if len(dup_items) <= 1:
                continue
            dup_items.sort(
                key=lambda seg: (
                    -len(str(seg.get("anchor_text") or seg.get("evidence_quote") or "")),
                    int(seg.get("segment_index") or 0),
                )
            )
            keep = dup_items[0]
            keep["locate_surface_policy"] = "secondary"
            for seg in dup_items[1:]:
                seg["must_locate"] = False
                seg["locate_policy"] = "hidden"
                seg["locate_surface_policy"] = "hidden"

    return _assign_claim_group_targets(out)


def _build_paper_guide_answer_provenance(
    *,
    answer: str,
    answer_hits: list[dict],
    bound_source_path: str,
    bound_source_name: str,
    db_dir: Path | str | None,
    settings_obj: object | None = None,
    llm_rerank: bool = False,
) -> dict | None:
    source_path = str(bound_source_path or "").strip()
    if not source_path:
        return None
    md_path = _resolve_paper_guide_md_path(source_path, db_dir=db_dir)
    if md_path is None:
        return None
    try:
        blocks = load_source_blocks(md_path)
    except Exception:
        return None
    if not blocks:
        return None
    block_lookup = {
        str(block.get("block_id") or "").strip(): dict(block)
        for block in blocks
        if isinstance(block, dict) and str(block.get("block_id") or "").strip()
    }

    evidence_pool = _collect_paper_guide_block_pool(
        blocks=blocks,
        answer_hits=answer_hits,
        bound_source_path=bound_source_path,
        bound_source_name=bound_source_name,
    )
    mapping_mode = "fast"
    empty_contract_meta = {
        "provenance_schema_version": int(_PAPER_GUIDE_PROVENANCE_SCHEMA_VERSION),
        "must_locate_candidate_count": 0,
        "must_locate_count": 0,
        "strict_identity_count": 0,
        "strict_identity_ready": True,
        "identity_missing_reasons": {},
        "identity_missing_segments": [],
    }

    if not evidence_pool:
        return {
            "version": int(_PAPER_GUIDE_PROVENANCE_SCHEMA_VERSION),
            "source_path": source_path,
            "source_name": str(bound_source_name or Path(source_path).name or "").strip(),
            "md_path": str(md_path),
            "doc_id": str(blocks[0].get("doc_id") or ""),
            "segments": [],
            "block_map": {},
            "status": "no_evidence_pool",
            "mapping_mode": mapping_mode,
            "llm_rerank_enabled": bool(llm_rerank),
            "llm_rerank_calls": 0,
            **empty_contract_meta,
        }

    candidate_blocks = [dict(item.get("block") or {}) for item in evidence_pool if isinstance(item.get("block"), dict)]
    block_map: dict[str, dict] = {}
    segments_out: list[dict] = []
    segments = split_answer_segments(answer)
    llm_picker: DeepSeekChat | None = None
    llm_calls_used = 0
    try:
        llm_max_calls = int(os.environ.get("KB_PROVENANCE_LLM_MAX_CALLS", "3") or 3)
    except Exception:
        llm_max_calls = 3
    llm_max_calls = max(0, min(6, llm_max_calls))
    if bool(llm_rerank) and (llm_max_calls > 0):
        try:
            llm_picker = DeepSeekChat(settings_obj) if settings_obj is not None else None
        except Exception:
            llm_picker = None

    def _is_formula_block(block: dict) -> bool:
        kind = str(block.get("kind") or "").strip().lower()
        text = str(block.get("text") or "")
        if kind == "equation":
            return True
        if "$$" in text or "\\begin{equation" in text.lower():
            return True
        if has_equation_signal(text) and ("=" in text or "\\tag{" in text.lower()):
            return True
        return False

    global_formula_blocks = [dict(block) for block in (blocks or []) if isinstance(block, dict) and _is_formula_block(block)]
    if not global_formula_blocks:
        global_formula_blocks = [dict(block) for block in candidate_blocks if isinstance(block, dict) and _is_formula_block(block)]

    for idx, segment in enumerate(segments, start=1):
        seg_text = normalize_inline_markdown(str(segment.get("text") or ""))
        if len(seg_text) < 10:
            continue
        seg_kind = str(segment.get("kind") or "paragraph")
        raw_markdown = str(segment.get("raw_markdown") or segment.get("raw_text") or seg_text).strip()
        word_count = len(_LATIN_WORD_RE.findall(seg_text)) + len(_CJK_WORD_RE.findall(seg_text))
        is_formula = _is_display_formula_segment(seg_text, segment_kind=seg_kind)
        eq_number = extract_equation_number(seg_text) if has_equation_signal(seg_text) else 0
        quoted_spans = _extract_quoted_spans(raw_markdown or seg_text, min_len=12)
        quote_anchor = _longest_quoted_span(raw_markdown or seg_text, min_len=12)
        probe_text = str(quote_anchor or seg_text).strip()
        summary_tags = _summary_segment_tags(seg_text) if not is_formula else set()
        prefer_kind = "equation" if is_formula else ""
        base_blocks = candidate_blocks
        if (not is_formula) and (str(seg_kind).strip().lower() == "blockquote" or bool(quote_anchor)):
            base_blocks = blocks
        if is_formula and global_formula_blocks:
            base_blocks = global_formula_blocks
        if is_formula:
            rank_limit = 10
        elif summary_tags:
            rank_limit = max(12, min(24, len(base_blocks or [])))
        else:
            rank_limit = 8 if quote_anchor else 5
        ranked = match_source_blocks(
            base_blocks,
            snippet=probe_text,
            prefer_kind=prefer_kind,
            target_number=eq_number,
            limit=rank_limit,
            score_floor=(0.12 if summary_tags else None),
        )
        if (not ranked) and (not is_formula) and quote_anchor and base_blocks is not candidate_blocks:
            ranked = match_source_blocks(
                candidate_blocks,
                snippet=probe_text,
                prefer_kind="",
                target_number=0,
                limit=6,
            )
        if is_formula and (not ranked):
            ranked = match_source_blocks(
                candidate_blocks,
                snippet=seg_text,
                prefer_kind="equation",
                target_number=eq_number,
                limit=10,
            )
        if is_formula:
            ranked_formula = []
            ranked_other = []
            for row in ranked:
                block0 = row.get("block")
                if isinstance(block0, dict) and _is_formula_block(block0):
                    score = float(row.get("score") or 0.0)
                    formula_text = str(block0.get("text") or "")
                    formula_tok = _formula_token_overlap_score(seg_text, formula_text)
                    formula_char = _formula_char_similarity(seg_text, formula_text)
                    score += (0.92 * formula_tok) + (0.65 * formula_char)
                    if eq_number > 0:
                        try:
                            if int(block0.get("number") or 0) == int(eq_number):
                                score += 1.15
                        except Exception:
                            pass
                    ranked_formula.append(
                        {
                            "score": score,
                            "block": block0,
                            "formula_tok": formula_tok,
                            "formula_char": formula_char,
                        }
                    )
                else:
                    ranked_other.append(row)
            ranked_formula.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
            if eq_number <= 0 and ranked_formula:
                top_tok = float(ranked_formula[0].get("formula_tok") or 0.0)
                top_char = float(ranked_formula[0].get("formula_char") or 0.0)
                if top_tok < 0.24 and top_char < 0.42:
                    ranked_formula = []
            ranked = ranked_formula + ([] if ranked_formula else ranked_other)
        else:
            ranked_text = []
            ranked_formula = []
            for row in ranked:
                block0 = row.get("block")
                if isinstance(block0, dict) and _is_formula_block(block0):
                    ranked_formula.append(row)
                else:
                    ranked_text.append(row)
            ranked = ranked_text + ranked_formula

        segment_mapping_source = "fast"
        chosen_ids: list[str] = []
        best_score = float(ranked[0].get("score") or 0.0) if ranked else 0.0
        second_score = float(ranked[1].get("score") or 0.0) if len(ranked) > 1 else 0.0
        score_gap = best_score - second_score

        formula_needs_llm = bool(
            is_formula and (
                (eq_number > 0 and score_gap < 0.18)
                or (eq_number <= 0 and ((best_score < 1.55) or (score_gap < 0.26)))
            )
        )
        if (
            llm_picker is not None
            and llm_calls_used < llm_max_calls
            and len(ranked) >= 2
            and (formula_needs_llm or ((best_score < 0.98) and (score_gap < 0.14)))
        ):
            llm_ids = _pick_blocks_with_llm(
                ds=llm_picker,
                segment_text=probe_text,
                segment_kind=seg_kind,
                is_formula_segment=is_formula,
                candidate_rows=ranked[:6],
                max_pick=(2 if is_formula else 1),
            )
            llm_calls_used += 1
            if llm_ids:
                by_id = {
                    str((row.get("block") or {}).get("block_id") or "").strip(): row
                    for row in ranked
                    if isinstance(row.get("block"), dict)
                }
                boosted: list[dict] = []
                consumed: set[str] = set()
                for bid in llm_ids:
                    row = by_id.get(bid)
                    if not isinstance(row, dict):
                        continue
                    block0 = row.get("block")
                    if not isinstance(block0, dict):
                        continue
                    consumed.add(bid)
                    boosted.append({
                        "score": float(row.get("score") or 0.0) + 1.25,
                        "block": block0,
                    })
                for row in ranked:
                    block0 = row.get("block")
                    if not isinstance(block0, dict):
                        continue
                    bid = str(block0.get("block_id") or "").strip()
                    if bid in consumed:
                        continue
                    boosted.append(row)
                ranked = boosted
                best_score = float(ranked[0].get("score") or 0.0) if ranked else best_score
                segment_mapping_source = "llm_refined"

        primary_support_metrics: dict[str, float | str | bool] = {
            "quote": "",
            "quote_score": 0.0,
            "support_score": 0.0,
            "heading_adjust": 0.0,
            "generic_heading": False,
            "summary_adjust": 0.0,
        }
        if (not is_formula) and ranked:
            rescored: list[dict] = []
            for row in ranked:
                block0 = row.get("block")
                if not isinstance(block0, dict):
                    continue
                metrics = _block_support_metrics(probe_text, block0)
                summary_adjust = _summary_block_adjustment(seg_text, block0)
                capped_base = min(float(row.get("score") or 0.0), 1.32)
                final_score = capped_base
                final_score += 0.96 * float(metrics.get("support_score") or 0.0)
                final_score += float(metrics.get("heading_adjust") or 0.0)
                final_score += summary_adjust
                rescored.append(
                    {
                        "score": final_score,
                        "block": block0,
                        "support_score": float(metrics.get("support_score") or 0.0),
                        "quote_score": float(metrics.get("quote_score") or 0.0),
                        "support_quote": str(metrics.get("quote") or ""),
                        "heading_adjust": float(metrics.get("heading_adjust") or 0.0),
                        "generic_heading": bool(metrics.get("generic_heading")),
                        "summary_adjust": float(summary_adjust),
                    }
                )
            rescored.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
            ranked = rescored
            best_score = float(ranked[0].get("score") or 0.0) if ranked else 0.0
            second_score = float(ranked[1].get("score") or 0.0) if len(ranked) > 1 else 0.0
            score_gap = best_score - second_score

        min_score = 0.44 if is_formula else 0.63
        dynamic_floor = max(min_score, best_score - (0.14 if is_formula else 0.10))
        keep_limit = 2
        for row in ranked:
            score = float(row.get("score") or 0.0)
            block = row.get("block")
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("block_id") or "").strip()
            if not block_id:
                continue
            block_is_formula = _is_formula_block(block)
            if not is_formula and block_is_formula and score < (dynamic_floor + 0.12):
                continue
            if not is_formula:
                support_score = float(row.get("support_score") or 0.0)
                quote_score = float(row.get("quote_score") or 0.0)
                heading_adjust = float(row.get("heading_adjust") or 0.0)
                generic_heading = bool(row.get("generic_heading"))
                summary_adjust = float(row.get("summary_adjust") or 0.0)
                if summary_tags:
                    if support_score < 0.18 and summary_adjust < 0.60:
                        continue
                    if generic_heading and support_score < 0.42 and summary_adjust < 0.60:
                        continue
                else:
                    if support_score < 0.24:
                        continue
                    if generic_heading and support_score < 0.42:
                        continue
                if quote_score < 0.18 and heading_adjust < -0.10:
                    continue
                if summary_tags and summary_adjust <= -0.48 and support_score < 0.58:
                    continue
            if score < dynamic_floor:
                continue
            if block_id in chosen_ids:
                continue
            chosen_ids.append(block_id)
            _ensure_provenance_block_entry(block_map, block)
            if len(chosen_ids) == 1:
                primary_support_metrics = {
                    "quote": str(row.get("support_quote") or ""),
                    "quote_score": float(row.get("quote_score") or 0.0),
                    "support_score": float(row.get("support_score") or 0.0),
                    "heading_adjust": float(row.get("heading_adjust") or 0.0),
                    "generic_heading": bool(row.get("generic_heading")),
                    "summary_adjust": float(row.get("summary_adjust") or 0.0),
                }
            if len(chosen_ids) >= keep_limit:
                break

        if is_formula and not chosen_ids:
            eq_ranked = match_source_blocks(
                global_formula_blocks or blocks,
                snippet=seg_text,
                prefer_kind="equation",
                target_number=eq_number,
                limit=8,
            )
            eq_ranked2: list[dict] = []
            for row in eq_ranked:
                block = row.get("block")
                if not isinstance(block, dict):
                    continue
                if not _is_formula_block(block):
                    continue
                formula_text = str(block.get("text") or "")
                formula_tok = _formula_token_overlap_score(seg_text, formula_text)
                formula_char = _formula_char_similarity(seg_text, formula_text)
                score = float(row.get("score") or 0.0)
                score += (0.88 * formula_tok) + (0.62 * formula_char)
                if eq_number > 0:
                    try:
                        if int(block.get("number") or 0) == int(eq_number):
                            score += 1.1
                    except Exception:
                        pass
                eq_ranked2.append(
                    {
                        "score": score,
                        "block": block,
                        "formula_tok": formula_tok,
                        "formula_char": formula_char,
                    }
                )
            eq_ranked2.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
            if eq_number <= 0 and eq_ranked2:
                top_tok = float(eq_ranked2[0].get("formula_tok") or 0.0)
                top_char = float(eq_ranked2[0].get("formula_char") or 0.0)
                if top_tok < 0.24 and top_char < 0.42:
                    eq_ranked2 = []
            eq_best = float(eq_ranked2[0].get("score") or 0.0) if eq_ranked2 else 0.0
            eq_floor = max(0.24, eq_best - 0.26)
            for row in eq_ranked2:
                score = float(row.get("score") or 0.0)
                block = row.get("block")
                if not isinstance(block, dict):
                    continue
                block_id = str(block.get("block_id") or "").strip()
                if (not block_id) or (score < eq_floor) or (block_id in chosen_ids):
                    continue
                chosen_ids.append(block_id)
                _ensure_provenance_block_entry(block_map, block)
                if len(chosen_ids) >= 2:
                    break
            if chosen_ids:
                best_score = max(best_score, eq_best)
        if is_formula and (not chosen_ids) and word_count >= 10:
            prose_ranked = match_source_blocks(
                blocks,
                snippet=seg_text,
                prefer_kind="",
                target_number=0,
                limit=6,
            )
            prose_floor = 0.5
            for row in prose_ranked:
                score = float(row.get("score") or 0.0)
                block = row.get("block")
                if not isinstance(block, dict):
                    continue
                if _is_formula_block(block):
                    continue
                block_id = str(block.get("block_id") or "").strip()
                if (not block_id) or (score < prose_floor):
                    continue
                chosen_ids.append(block_id)
                _ensure_provenance_block_entry(block_map, block)
                if len(chosen_ids) >= 2:
                    break
            if chosen_ids:
                best_score = max(best_score, float(prose_ranked[0].get("score") or 0.0))

        evidence_mode = "direct" if chosen_ids else "synthesis"
        snippet_aliases = _segment_snippet_aliases(seg_text)
        for quote_span in quoted_spans:
            key = normalize_match_text(quote_span)
            if (not key) or (key in snippet_aliases):
                continue
            snippet_aliases.append(key[:360])
            if len(snippet_aliases) >= 8:
                break
        primary_block_id = str(chosen_ids[0] or "").strip() if chosen_ids else ""
        support_block_ids = [str(item or "").strip() for item in chosen_ids[1:] if str(item or "").strip()]
        primary_block = block_lookup.get(primary_block_id) if primary_block_id else None
        if primary_block and (not is_formula):
            support_score = float(primary_support_metrics.get("support_score") or 0.0)
            generic_heading = bool(primary_support_metrics.get("generic_heading"))
            summary_adjust = float(primary_support_metrics.get("summary_adjust") or 0.0)
            reject_primary = False
            if summary_tags:
                if support_score < 0.18 and summary_adjust < 0.60:
                    reject_primary = True
                if generic_heading and support_score < 0.42 and summary_adjust < 0.60:
                    reject_primary = True
            elif support_score < 0.24 or (generic_heading and support_score < 0.42):
                reject_primary = True
            if reject_primary:
                chosen_ids = []
                primary_block_id = ""
                support_block_ids = []
                primary_block = None
                evidence_mode = "synthesis"
                primary_support_metrics = {
                    "quote": "",
                    "quote_score": 0.0,
                    "support_score": 0.0,
                    "heading_adjust": 0.0,
                    "generic_heading": False,
                    "summary_adjust": 0.0,
                }
        primary_anchor_id = str((primary_block or {}).get("anchor_id") or "").strip()
        primary_heading_path = str((primary_block or {}).get("heading_path") or "").strip()
        evidence_quote = str(primary_support_metrics.get("quote") or "")
        if (not evidence_quote) and primary_block:
            evidence_quote = _best_evidence_quote(probe_text, primary_block)
        evidence_confidence = round(best_score, 4) if chosen_ids else 0.0
        mapping_quality = round(float(primary_support_metrics.get("support_score") or 0.0), 4) if chosen_ids else 0.0
        claim_meta = _segment_claim_meta(
            segment_text=seg_text,
            raw_markdown=raw_markdown,
            segment_kind=seg_kind,
            evidence_mode=evidence_mode,
            primary_block=primary_block,
            evidence_quote=evidence_quote,
            mapping_quality=mapping_quality,
        )
        segments_out.append(
            {
                "segment_id": f"seg_{idx:03d}",
                "segment_index": idx,
                "kind": seg_kind,
                "segment_type": _segment_type_from_text(seg_text, segment_kind=seg_kind),
                "text": seg_text,
                "raw_markdown": raw_markdown[:4000],
                "snippet_key": str(segment.get("snippet_key") or normalize_match_text(seg_text[:360])),
                "snippet_aliases": snippet_aliases,
                "evidence_mode": evidence_mode,
                "evidence_block_ids": chosen_ids,
                "primary_block_id": primary_block_id,
                "primary_anchor_id": primary_anchor_id,
                "primary_heading_path": primary_heading_path,
                "support_block_ids": support_block_ids,
                "evidence_quote": evidence_quote,
                "evidence_confidence": evidence_confidence,
                "mapping_quality": mapping_quality,
                "mapping_source": segment_mapping_source,
                "claim_type": str(claim_meta.get("claim_type") or "").strip(),
                "must_locate": bool(claim_meta.get("must_locate")),
                "anchor_kind": str(claim_meta.get("anchor_kind") or "").strip(),
                "anchor_text": str(claim_meta.get("anchor_text") or "").strip()[:900],
                "equation_number": int(claim_meta.get("equation_number") or 0),
            }
        )

    if llm_calls_used > 0:
        mapping_mode = "llm_refined"
    segments_with_policy = _apply_provenance_required_coverage_contract(
        segments_out,
        block_lookup=block_lookup,
    )
    segments_hardened, contract_meta = _apply_provenance_strict_identity_contract(segments_with_policy)

    return {
        "version": int(_PAPER_GUIDE_PROVENANCE_SCHEMA_VERSION),
        "source_path": source_path,
        "source_name": str(bound_source_name or Path(source_path).name or "").strip(),
        "md_path": str(md_path),
        "doc_id": str(blocks[0].get("doc_id") or ""),
        "segments": segments_hardened,
        "block_map": block_map,
        "status": "ready",
        "candidate_block_count": len(candidate_blocks),
        "mapping_mode": mapping_mode,
        "llm_rerank_enabled": bool(llm_rerank),
        "llm_rerank_calls": int(llm_calls_used),
        **contract_meta,
    }
