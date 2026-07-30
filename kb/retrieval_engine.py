from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .chunking import chunk_markdown
from .llm import DeepSeekChat
from .paper_guide_prompting import _paper_guide_prompt_requests_citation_lookup
from .paper_guide_structured_index_runtime import (
    figure_key_for_scope,
    filter_figure_index_rows,
    normalize_figure_scope,
)
from .retrieval_heuristics import (
    _aspects_from_snippets,
    _clean_snippet_for_display,
    _doc_term_bonus,
    _has_cjk,
    _has_latin,
    _is_noise_snippet_text,
    _is_probably_bad_heading,
    _norm_text_for_match,
    _normalize_heading,
    _pick_best_heading_for_doc,
    _preferred_section_keys,
    _query_term_profile,
    _score_tokens,
)
from .retriever import BM25Retriever
from .reference_query_family import prompt_explicitly_requests_multi_paper_list as _prompt_explicitly_requests_multi_paper_list
from .source_blocks import load_anchor_index_cached as _load_anchor_index_cached
from .source_filters import is_excluded_source_path
from .store import compute_file_sha1
from .tokenize import tokenize
from .runtime_cache import cache_get as _runtime_cache_get, cache_set as _runtime_cache_set

# These callbacks are injected by the API/runtime layer to reuse the shared cache.
_CACHE_GET: Callable[[str, str], Any] = _runtime_cache_get
_CACHE_SET: Callable[..., None] = _runtime_cache_set


def configure_cache(cache_get: Callable[[str, str], Any], cache_set: Callable[..., None]) -> None:
    global _CACHE_GET, _CACHE_SET
    _CACHE_GET = cache_get
    _CACHE_SET = cache_set


def _cache_get(bucket: str, key: str):
    return _CACHE_GET(bucket, key)


def _cache_set(bucket: str, key: str, val, *, max_items: int = 600) -> None:
    _CACHE_SET(bucket, key, val, max_items=max_items)


def _is_temp_source_path(source_path: str) -> bool:
    s = (source_path or "").strip()
    if not s:
        return True
    # If the caller explicitly points KB_DB_DIR to a directory that happens to
    # contain "tmp_*" in its path, we still must treat its documents as eligible
    # for retrieval. The temp filter is meant to exclude converter artifacts
    # under an otherwise "real" DB, not to break an intentional DB root switch.
    try:
        raw_db_dir = str(os.environ.get("KB_DB_DIR", "") or "").strip()
        if raw_db_dir:
            db_dir = Path(raw_db_dir).expanduser().resolve()
            src = Path(s).expanduser().resolve()
            if src.is_relative_to(db_dir):
                return False
    except Exception:
        pass
    if is_excluded_source_path(s):
        return True
    p = Path(s)
    low_parts = [str(x).strip().lower() for x in p.parts]
    low_name = p.name.lower()
    low_stem = p.stem.lower()
    if any(x in {"temp", "__pycache__"} for x in low_parts):
        return True
    if any(x.startswith("__upload__") or x.startswith("_tmp_") or x.startswith("tmp_") for x in low_parts):
        return True
    if low_name.startswith("__upload__") or low_stem.startswith("__upload__"):
        return True
    if low_name.startswith("_tmp_") or low_stem.startswith("_tmp_"):
        return True
    if low_name.startswith("tmp_") or low_stem.startswith("tmp_"):
        return True
    return False

def _top_heading(heading_path: str) -> str:
    hp = (heading_path or "").strip()
    if not hp:
        return ""
    return hp.split(" / ", 1)[0].strip()


def _normalize_heading_path_for_display(heading_path: str) -> str:
    hp = (heading_path or "").strip()
    if not hp:
        return ""
    parts: list[str] = []
    for raw in hp.split(" / "):
        t = _normalize_heading(str(raw).strip())
        if not t:
            continue
        parts.append(t)
    return " / ".join(parts)


def _split_heading_path_levels(heading_path: str) -> tuple[str, str]:
    hp = _normalize_heading_path_for_display(heading_path)
    if not hp:
        return "", ""
    parts = [p.strip() for p in hp.split(" / ") if p.strip()]
    if not parts:
        return "", ""
    return parts[0], " / ".join(parts[1:]).strip()


_REF_HEADING_RE = re.compile(
    r"\b(references?|bibliography|works?\s+cited|citation|acknowledg(e)?ments?|appendi(?:x|ces)|supplementary)\b",
    flags=re.I,
)
_VENUE_HEADING_EXACT = {
    "nature photonics",
    "science advances",
    "nature communications",
    "physical review letters",
    "optics express",
    "optics letters",
    "applied optics",
}
_VENUE_HEAD_TOKENS = {
    "nature",
    "science",
    "ieee",
    "acm",
    "cvpr",
    "iccv",
    "eccv",
    "neurips",
    "icml",
    "ijcai",
    "aaai",
    "conference",
    "proceedings",
    "journal",
    "transactions",
    "letters",
    "communication",
    "communications",
    "photonics",
    "optics",
    "review",
    "advances",
    "arxiv",
}
_VENUE_JOIN_TOKENS = {"of", "on", "for", "and", "the", "in", "&"}
_COMMON_SECTION_TOKENS = {
    "abstract",
    "introduction",
    "background",
    "related",
    "work",
    "method",
    "methods",
    "approach",
    "model",
    "setup",
    "experiment",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "implementation",
    "evaluation",
    "analysis",
}
_TITLE_STOP_TOKENS = {
    "paper",
    "using",
    "with",
    "for",
    "from",
    "toward",
    "towards",
    "based",
    "via",
    "adaptive",
    "dynamic",
    "single",
    "pixel",
    "imaging",
}


def _wants_reference_navigation(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    # Keep bibliography navigation aligned with the centralized paper-guide
    # intent classifier.  In particular, output constraints such as
    # “给出对应引用” / “give supporting citations” ask for answer grounding;
    # they are not requests to retrieve the paper's References section.
    if _paper_guide_prompt_requests_citation_lookup(q):
        return True
    return bool(
        re.search(
            r"(?i)(?:"
            r"(?:打开|查看|跳转到?|定位到?|展示|列出).{0,12}(?:参考文献|文献列表|引用列表)|"
            r"(?:参考文献|文献列表|引用列表).{0,12}(?:章节|部分|列表|第\s*\d+\s*条)|"
            r"(?:open|show|view|navigate\s+to|go\s+to|list).{0,16}(?:references?|bibliography)|"
            r"(?:references?|bibliography).{0,16}(?:section|list|entry\s*\[?\d+\]?)"
            r")",
            q,
        )
    )


def _is_reference_heading_like(h: str) -> bool:
    s = _normalize_heading(h)
    if not s:
        return False
    return bool(_REF_HEADING_RE.search(s))


def _is_venue_heading_like(h: str) -> bool:
    s = _normalize_heading(h)
    if not s:
        return False
    low = s.lower()
    if low in _VENUE_HEADING_EXACT:
        return True

    toks = re.findall(r"[a-z][a-z0-9.+-]*", low)
    if not toks:
        return False
    if any(t in _COMMON_SECTION_TOKENS for t in toks):
        return False
    venue_hit = any(t in _VENUE_HEAD_TOKENS for t in toks)
    if (len(toks) <= 6) and venue_hit and all((t in _VENUE_HEAD_TOKENS or t in _VENUE_JOIN_TOKENS) for t in toks):
        return True

    letters = re.sub(r"[^A-Za-z]", "", s)
    if letters and (letters == letters.upper()) and (len(toks) <= 5) and venue_hit:
        return True
    return False


def _looks_like_doc_title_heading(h: str, source_path: str) -> bool:
    s = _normalize_heading(h)
    src = str(source_path or "").strip()
    if not s or not src:
        return False
    h_norm = _norm_text_for_match(s)
    if len(h_norm) < 24:
        return False

    stem = Path(src).stem
    stem = re.sub(r"(19|20)\d{2}", " ", stem)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem_norm = _norm_text_for_match(stem)
    if not stem_norm:
        return False
    if h_norm in stem_norm:
        return True

    h_toks = [t for t in tokenize(h_norm) if len(t) >= 3 and t not in _TITLE_STOP_TOKENS]
    s_toks = [t for t in tokenize(stem_norm) if len(t) >= 3 and t not in _TITLE_STOP_TOKENS]
    if len(h_toks) < 3 or len(s_toks) < 3:
        return False
    hs = set(h_toks)
    ss = set(s_toks)
    inter = hs & ss
    if len(inter) < 3:
        return False
    return (len(inter) / max(1, len(hs))) >= 0.66


def _is_non_navigational_heading(h: str, *, question: str, source_path: str = "") -> bool:
    s = _normalize_heading(h)
    if not s:
        return True
    if _is_probably_bad_heading(s):
        return True
    if _is_venue_heading_like(s):
        return True
    if (not _wants_reference_navigation(question)) and _is_reference_heading_like(s):
        return True
    return False


def _is_low_quality_navigation_heading(h: str, *, question: str, source_path: str = "") -> bool:
    s = _normalize_heading(h)
    if not s:
        return True
    if _is_non_navigational_heading(s, question=question, source_path=source_path):
        return True
    if _looks_like_doc_title_heading(s, source_path):
        return True
    return False


def _looks_like_structured_section_heading(h: str) -> bool:
    s = _normalize_heading(h)
    if not s:
        return False
    low = s.lower()
    if re.match(r"^\d+(\.\d+){0,3}\b", low):
        return True
    if re.match(r"^(section|sec\.?|chapter|part|appendix)\b", low):
        return True
    return bool(re.match(r"^[ivxlcdm]+\.\s+", low))


def _sanitize_heading_path_for_navigation(heading_path: str, *, question: str, source_path: str = "") -> str:
    hp = _normalize_heading_path_for_display(heading_path)
    if not hp:
        return ""
    parts = [p.strip() for p in hp.split(" / ") if p.strip()]
    if not parts:
        return ""

    keep: list[str] = []
    for p in parts:
        if _is_non_navigational_heading(p, question=question, source_path=source_path):
            continue
        if keep and keep[-1].lower() == p.lower():
            continue
        keep.append(p)
    if len(keep) >= 2:
        first = keep[0]
        second = keep[1]
        if (
            len(first) >= 36
            and _looks_like_structured_section_heading(second)
            and (not _looks_like_structured_section_heading(first))
        ):
            keep = keep[1:]
    if keep and _looks_like_doc_title_heading(keep[0], source_path):
        keep = keep[1:] if len(keep) >= 2 else []
    if not keep:
        return ""
    return " / ".join(keep[:3])


_DOC_HINT_STOP_TOKENS = {
    "paper",
    "pdf",
    "article",
    "figure",
    "fig",
    "table",
    "equation",
    "eq",
    "formula",
    "theorem",
    "lemma",
    "definition",
    "proposition",
    "corollary",
    "图",
    "表",
    "公式",
    "定理",
    "引理",
    "定义",
    "命题",
    "推论",
    "文章",
    "论文",
    "这篇",
    "这个",
    "什么",
    "讲了",
    "讲的",
    "内容",
}
_SMALL_CN_NUMS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_ANCHOR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "section",
        re.compile(
            r"\b(?:section|sec\.?|chapter)\s*([0-9]+(?:\.[0-9]+){0,3})\b",
            flags=re.I,
        ),
    ),
    (
        "section",
        re.compile(r"第\s*([0-9]+(?:\.[0-9]+){0,3})\s*(?:节|章)"),
    ),
    ("figure", re.compile(r"\bfig(?:ure)?\.?\s*([0-9ivxlcdm]+)\b", flags=re.I)),
    ("figure", re.compile(r"\bfig(?:ure)?\.?\s*S\s*([0-9ivxlcdm]+)\b", flags=re.I)),
    ("table", re.compile(r"\btable\.?\s*([0-9ivxlcdm]+)\b", flags=re.I)),
    ("equation", re.compile(r"\b(?:eq(?:uation)?|formula)\.?\s*[\(\[（]?\s*([0-9ivxlcdm]+)\s*[\)\]）]?", flags=re.I)),
    ("theorem", re.compile(r"\btheorem\.?\s*([0-9ivxlcdm]+)\b", flags=re.I)),
    ("lemma", re.compile(r"\blemma\.?\s*([0-9ivxlcdm]+)\b", flags=re.I)),
    ("definition", re.compile(r"\bdefinition\.?\s*([0-9ivxlcdm]+)\b", flags=re.I)),
    ("proposition", re.compile(r"\bproposition\.?\s*([0-9ivxlcdm]+)\b", flags=re.I)),
    ("corollary", re.compile(r"\bcorollary\.?\s*([0-9ivxlcdm]+)\b", flags=re.I)),
    ("figure", re.compile(r"图\s*([零一二两三四五六七八九十百\d]+)(?!\d)")),
    ("table", re.compile(r"表\s*([零一二两三四五六七八九十百\d]+)(?!\d)")),
    ("figure", re.compile(r"第\s*([零一二两三四五六七八九十百\d]+)\s*(?:张)?图")),
    ("table", re.compile(r"第\s*([零一二两三四五六七八九十百\d]+)\s*表")),
    ("equation", re.compile(r"(?:公式|式)\s*[\(\[（]?\s*([零一二两三四五六七八九十百\d]+)\s*[\)\]）]?")),
    ("theorem", re.compile(r"定理\s*[\(\[（]?\s*([零一二两三四五六七八九十百\d]+)\s*[\)\]）]?")),
    ("lemma", re.compile(r"引理\s*[\(\[（]?\s*([零一二两三四五六七八九十百\d]+)\s*[\)\]）]?")),
    ("definition", re.compile(r"定义\s*[\(\[（]?\s*([零一二两三四五六七八九十百\d]+)\s*[\)\]）]?")),
    ("proposition", re.compile(r"命题\s*[\(\[（]?\s*([零一二两三四五六七八九十百\d]+)\s*[\)\]）]?")),
    ("corollary", re.compile(r"推论\s*[\(\[（]?\s*([零一二两三四五六七八九十百\d]+)\s*[\)\]）]?")),
]
_ANCHOR_KIND_LABELS = {
    "section": ("section", "sec", "chapter", "节", "章"),
    "figure": ("figure", "fig", "图", "张图"),
    "table": ("table", "表"),
    "equation": ("equation", "eq", "formula", "公式", "式"),
    "theorem": ("theorem", "定理"),
    "lemma": ("lemma", "引理"),
    "definition": ("definition", "定义"),
    "proposition": ("proposition", "命题"),
    "corollary": ("corollary", "推论"),
}


def _parse_small_roman(text: str) -> int | None:
    s = str(text or "").strip().lower()
    if not s or not re.fullmatch(r"[ivxlcdm]+", s):
        return None
    vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    prev = 0
    for ch in reversed(s):
        cur = vals.get(ch, 0)
        if cur < prev:
            total -= cur
        else:
            total += cur
            prev = cur
    return total if total > 0 else None


def _parse_small_cn_number(text: str) -> int | None:
    s = str(text or "").strip()
    if not s:
        return None
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return None
    if len(s) == 1:
        return _SMALL_CN_NUMS.get(s)
    if s == "十":
        return 10
    if s.startswith("十"):
        tail = _SMALL_CN_NUMS.get(s[1:], 0)
        return 10 + int(tail)
    if s.endswith("十"):
        head = _SMALL_CN_NUMS.get(s[:-1])
        return (int(head) * 10) if head is not None else None
    if "十" in s:
        left, right = s.split("十", 1)
        left_n = _SMALL_CN_NUMS.get(left, 1 if left == "" else None)
        right_n = _SMALL_CN_NUMS.get(right, 0 if right == "" else None)
        if left_n is None or right_n is None:
            return None
        return (int(left_n) * 10) + int(right_n)
    return None


def _parse_anchor_number(text: str) -> int | None:
    s = str(text or "").strip()
    if not s:
        return None
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return None
    roman = _parse_small_roman(s)
    if roman is not None:
        return roman
    return _parse_small_cn_number(s)


def _figure_scope_for_anchor_match(text: str, match: re.Match[str]) -> str:
    src = str(text or "")
    start = max(0, int(match.start()) - 40)
    window = src[start : int(match.end())]
    matched = str(match.group(0) or "")
    if re.search(r"(?:extended\s+data|扩展数据|扩展)\s*$", src[start : int(match.start())], re.IGNORECASE):
        return "extended_data"
    if re.search(r"(?:supplementary|supplemental|补充)\s*$", src[start : int(match.start())], re.IGNORECASE):
        return "supplementary"
    if re.search(r"\bfig(?:ure)?\.?\s*S\s*\d+\b", matched, re.IGNORECASE):
        return "supplementary"
    if re.search(r"\bextended\s+data\s+fig(?:ure)?\.?\s*\d+\b|(?:扩展数据|扩展)\s*图\s*\d+", window, re.IGNORECASE):
        return "extended_data"
    if re.search(
        r"\b(?:supplementary|supplemental)\s+fig(?:ure)?\.?\s*S?\s*\d+\b|补充\s*图\s*\d+",
        window,
        re.IGNORECASE,
    ):
        return "supplementary"
    return "main"


def _extract_explicit_anchor_hint(question: str) -> dict[str, object]:
    q = str(question or "").strip()
    if not q:
        return {}
    ranked_hints: list[tuple[int, int, dict]] = []
    for pattern_index, (kind, pat) in enumerate(_ANCHOR_PATTERNS):
        for m in pat.finditer(q):
            raw_number = str(m.group(1) or "").strip()
            number_text = raw_number
            if kind == "section" and re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,3}", raw_number):
                number_parts = [str(int(part)) for part in raw_number.split(".")]
                number_text = ".".join(number_parts)
                num = int(number_parts[0])
            else:
                num = _parse_anchor_number(raw_number)
            if num is None or num <= 0:
                continue
            phrases: list[str] = []
            labels = _ANCHOR_KIND_LABELS.get(kind) or ()
            if kind == "section":
                phrases.extend(
                    [
                        f"section {number_text}",
                        f"sec. {number_text}",
                        f"chapter {number_text}",
                        number_text,
                        f"第{number_text}节",
                        f"第 {number_text} 节",
                    ]
                )
            for lab in (() if kind == "section" else labels):
                if lab in {"fig", "eq"}:
                    phrases.append(f"{lab}. {num}")
                    phrases.append(f"{lab} {num}")
                elif lab == "张图":
                    phrases.append(f"第{num}张图")
                elif lab in {"图", "表", "公式", "式", "定理", "引理", "定义", "命题", "推论"}:
                    phrases.append(f"{lab}{num}")
                    phrases.append(f"{lab} {num}")
                    phrases.append(f"第{num}{lab}")
                else:
                    phrases.append(f"{lab} {num}")
            figure_scope = _figure_scope_for_anchor_match(q, m) if kind == "figure" else ""
            if kind == "figure" and figure_scope == "extended_data":
                phrases = [f"Extended Data Figure {num}", f"Extended Data Fig. {num}", f"扩展数据图{num}"]
            elif kind == "figure" and figure_scope == "supplementary":
                phrases = [f"Supplementary Figure {num}", f"Supplementary Fig. {num}", f"Fig. S{num}", f"补充图{num}"]
            hint = {
                "kind": kind,
                "number": int(num),
                "number_text": number_text,
                "label": f"{kind} {number_text}",
                "phrases": list(dict.fromkeys([p for p in phrases if str(p or "").strip()])),
            }
            if kind == "figure":
                hint["figure_scope"] = figure_scope
                hint["figure_key"] = figure_key_for_scope(figure_scope, int(num))
            ranked_hints.append((int(m.start()), pattern_index, hint))
    ranked_hints.sort(key=lambda item: (int(item[0]), int(item[1])))
    hints: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for _start, _pattern_index, hint in ranked_hints:
        key = (
            str(hint.get("kind") or ""),
            str(hint.get("number_text") or hint.get("number") or ""),
            str(hint.get("figure_scope") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        hints.append(hint)
    if not hints:
        return {}
    # Return primary (first) hint with all hints accessible via "all_hints".
    primary = dict(hints[0])
    primary["all_hints"] = hints
    return primary


def _source_prompt_match_score(prompt_text: str, source_path: str) -> float:
    prompt_raw = str(prompt_text or "").strip()
    src = str(source_path or "").strip()
    if (not prompt_raw) or (not src):
        return 0.0
    prompt_low = prompt_raw.lower()
    prompt_norm = _norm_text_for_match(prompt_raw)
    p = Path(src)
    candidates = [p.name, p.stem, re.sub(r"^[A-Za-z]+-\d{4}[-_ ]*", "", p.stem)]
    score = 0.0
    for cand in candidates:
        c = str(cand or "").strip()
        if not c:
            continue
        c_low = c.lower()
        c_norm = _norm_text_for_match(c)
        if c_low and (c_low in prompt_low):
            score = max(score, 9.0 if c_low.endswith(".pdf") else 8.0)
        if c_norm and (len(c_norm) >= 12) and (c_norm in prompt_norm):
            score = max(score, 8.0)

    # A natural Chinese paper pointer can identify an English-titled source even
    # when the user does not repeat any title words verbatim. Keep this alias
    # deliberately narrow: generic perovskite/device questions must not be
    # promoted to laser papers (or vice versa).
    source_surface = _direct_phrase_surface(" ".join(str(x or "") for x in candidates))
    wants_perovskite_laser = (
        ("钙钛矿" in prompt_raw and ("激光器" in prompt_raw or "激光" in prompt_raw))
        or bool(re.search(r"\bperovskite\s+(?:laser|lasing)\b", prompt_low))
    )
    is_perovskite_laser_source = (
        "perovskite" in source_surface
        and bool(re.search(r"\b(?:laser|lasing)\b", source_surface))
    )
    if wants_perovskite_laser and is_perovskite_laser_source:
        score = max(score, 7.5)

    # Some method acronyms are user-facing shorthand rather than literal title
    # tokens. Resolve only high-specificity aliases here so the downstream
    # explicit-focus filter does not discard a paper that an alias expansion
    # correctly retrieved.
    source_identity_aliases = (
        ("pidl", ("physics informed deep learning", "single photon imaging")),
        ("piln", ("part based image loop network",)),
        ("hatnet", ("dual scale transformer", "large scale single pixel imaging")),
        ("ista-net", ("ista net", "interpretable optimization inspired deep network")),
        (
            "cassi",
            (
                "single shot compressive spectral imaging",
                "dual disperser",
            ),
        ),
    )
    for alias, required_source_phrases in source_identity_aliases:
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", prompt_low) and all(
            phrase in source_surface for phrase in required_source_phrases
        ):
            score = max(score, 7.5)

    prompt_tokens = [t for t in tokenize(prompt_norm) if len(t) >= 3 and t not in _DOC_HINT_STOP_TOKENS]
    src_tokens = [
        t
        for t in tokenize(_norm_text_for_match(" ".join(str(x or "") for x in candidates)))
        if len(t) >= 3 and t not in _DOC_HINT_STOP_TOKENS
    ]
    for token in re.findall(r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9_-]{2,40}(?![A-Za-z0-9_-])", " ".join(candidates)):
        raw_token = str(token or "").strip().strip("-_")
        low_token = raw_token.lower()
        if low_token in _DOC_HINT_STOP_TOKENS or len(low_token) < 4:
            continue
        has_identity_signal = (
            raw_token.isupper()
            or any(ch.isupper() for ch in raw_token[1:])
            or any(ch.isdigit() for ch in raw_token)
            or ("-" in raw_token)
        )
        if has_identity_signal and low_token in prompt_norm:
            score = max(score, 6.5)
    if prompt_tokens and src_tokens:
        inter = set(prompt_tokens) & set(src_tokens)
        if len(inter) >= 3:
            score += 2.0 + min(3.0, 0.6 * len(inter))
            ratio = len(inter) / max(1, len(set(src_tokens)))
            if ratio >= 0.55:
                score += 2.0
    return float(score)


_DIRECT_PROMPT_STOP_TOKENS = _DOC_HINT_STOP_TOKENS | {
    "a",
    "an",
    "and",
    "answer",
    "are",
    "around",
    "be",
    "by",
    "can",
    "could",
    "define",
    "defined",
    "defines",
    "definition",
    "direct",
    "directly",
    "discuss",
    "discussed",
    "discusses",
    "does",
    "for",
    "from",
    "give",
    "how",
    "in",
    "is",
    "it",
    "library",
    "me",
    "mention",
    "mentioned",
    "mentions",
    "most",
    "my",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "please",
    "point",
    "section",
    "show",
    "source",
    "sources",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "what",
    "where",
    "which",
    "why",
    "with",
}


def _direct_phrase_surface(text: str) -> str:
    s = _norm_text_for_match(text)
    s = re.sub(r"[-_/]+", " ", s)
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s)
    return " ".join(s.split())


def _extract_direct_prompt_phrases(prompt_text: str) -> tuple[str, ...]:
    """Extract user-written technical phrases that should anchor source choice."""

    raw = str(prompt_text or "").strip()
    if not raw:
        return ()
    out: list[str] = []
    seen: set[str] = set()

    def _push_phrase(value: str) -> None:
        phrase = _direct_phrase_surface(value)
        if not phrase or phrase in seen:
            return
        toks = [t for t in tokenize(phrase) if t]
        if not toks:
            return
        informative = [
            t
            for t in toks
            if t not in _DIRECT_PROMPT_STOP_TOKENS
            and (len(t) >= 4 or any(ch.isdigit() for ch in t))
        ]
        if len(toks) >= 2:
            if not informative:
                return
        elif len(toks) == 1:
            tok = toks[0]
            if tok in _DIRECT_PROMPT_STOP_TOKENS:
                return
            raw_hit = re.search(rf"(?<![A-Za-z0-9_-]){re.escape(tok)}(?![A-Za-z0-9_-])", raw, flags=re.I)
            raw_token = raw_hit.group(0) if raw_hit else tok
            has_identity_signal = (
                raw_token.isupper()
                or any(ch.isupper() for ch in raw_token[1:])
                or any(ch.isdigit() for ch in raw_token)
                or ("-" in raw_token)
            )
            if len(tok) < 5 and not has_identity_signal:
                return
        seen.add(phrase)
        out.append(phrase)

    for quoted in re.findall(r"[\"“”‘’]([^\"“”‘’]{2,120})[\"“”‘’]", raw):
        _push_phrase(quoted)

    latin_runs = re.findall(
        r"[A-Za-z][A-Za-z0-9+_.-]*(?:\s+[A-Za-z][A-Za-z0-9+_.-]*){0,11}",
        raw,
    )
    for run in latin_runs:
        raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9+_.-]*", run)
        tokens = []
        for token in raw_tokens:
            normed = _direct_phrase_surface(token)
            if not normed:
                continue
            parts = [p for p in normed.split() if p and p not in _DIRECT_PROMPT_STOP_TOKENS]
            tokens.extend(parts)
        if len(tokens) == 1:
            _push_phrase(tokens[0])
            continue
        if len(tokens) >= 2:
            max_n = min(5, len(tokens))
            for n in range(max_n, 1, -1):
                for idx in range(0, len(tokens) - n + 1):
                    phrase_tokens = tokens[idx : idx + n]
                    if not any(len(t) >= 6 or any(ch.isdigit() for ch in t) for t in phrase_tokens):
                        continue
                    _push_phrase(" ".join(phrase_tokens))

    # Prefer specific phrases first. Keep the list small so generic English
    # questions do not produce a cloud of weak title matches.
    out.sort(key=lambda item: (len(item.split()), len(item)), reverse=True)
    return tuple(out[:12])


def _direct_prompt_match_score(
    *,
    prompt_text: str,
    source_path: str,
    snippets: list[str],
    headings: list[str],
) -> tuple[float, tuple[str, ...]]:
    phrases = _extract_direct_prompt_phrases(prompt_text)
    if not phrases:
        return 0.0, ()
    p = Path(str(source_path or "").strip())
    title_surface = _direct_phrase_surface(
        " ".join(
            str(x or "")
            for x in (p.name, p.stem, re.sub(r"^[A-Za-z]+-\d{4}[-_ ]*", "", p.stem))
            if str(x or "").strip()
        )
    )
    heading_surface = _direct_phrase_surface(" ".join(str(x or "") for x in list(headings or [])[:8]))
    snippet_surface = _direct_phrase_surface(" ".join(str(x or "") for x in list(snippets or [])[:4]))
    score = 0.0
    matched: list[str] = []
    for phrase in phrases:
        toks = phrase.split()
        if not toks:
            continue
        title_hit = phrase in title_surface
        heading_hit = phrase in heading_surface
        snippet_hit = phrase in snippet_surface
        if not (title_hit or heading_hit or snippet_hit):
            continue
        is_single = len(toks) == 1
        phrase_score = 0.0
        if title_hit:
            phrase_score += 3.4 if is_single else 5.6 + min(2.0, 0.35 * len(toks))
        if heading_hit:
            phrase_score += 1.8 if is_single else 3.4
        if snippet_hit:
            phrase_score += 1.2 if is_single else 2.3
        if title_hit and (heading_hit or snippet_hit):
            phrase_score += 2.2
        elif (heading_hit and snippet_hit) and (not is_single):
            phrase_score += 1.1
        score += phrase_score
        matched.append(phrase)
    return float(min(score, 18.0)), tuple(matched[:5])


def _clean_doc_focus_phrase(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = re.sub(
        r"\b(?:please\s+point\s+me(?:\s+to)?|point\s+me(?:\s+to)?|show\s+me|source\s+section(?:s)?|those\s+sources)\b.*$",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.I)
    return text.strip(" \t\r\n\"'“”‘’?,;:!?()[]{}")


def _looks_informative_doc_focus_phrase(raw: str) -> bool:
    norm = _norm_text_for_match(raw)
    if not norm:
        return False
    toks = [t for t in tokenize(norm) if len(t) >= 3 and t not in _DOC_HINT_STOP_TOKENS]
    if len(toks) >= 2:
        return True
    if len(toks) == 1:
        tok = toks[0]
        return len(tok) >= 6 or any(ch.isdigit() for ch in tok) or ("-" in tok)
    return False


def _extract_doc_focus_phrases(prompt_text: str) -> tuple[str, ...]:
    text = str(prompt_text or "").strip()
    if not text:
        return ()
    patterns = (
        re.compile(
            r"\bwhere\s+(?:in\s+the\s+[^?.!,]{1,80}\s+)?is\s+(.+?)\s+(?:discussed|mentioned|defined|introduced)\b",
            flags=re.I,
        ),
        re.compile(
            r"\b(?:which|what)\s+(?:other\s+)?papers?[^?.!]{0,120}?\b(?:discuss(?:es|ed)?|mention(?:s|ed)?|cover(?:s|ed)?|address(?:es|ed)?|describe(?:s|d)?|use(?:s|d)?|introduce(?:s|d)?|define(?:s|d)?|compare(?:s|d)?)\s+(.+?)(?:[?.!]|$)",
            flags=re.I,
        ),
        re.compile(
            r"\bbesides\s+this\s+paper[^?.!]{0,120}?\b(?:discuss(?:es|ed)?|mention(?:s|ed)?|cover(?:s|ed)?|address(?:es|ed)?|describe(?:s|d)?|use(?:s|d)?|introduce(?:s|d)?|define(?:s|d)?|compare(?:s|d)?)\s+(.+?)(?:[?.!]|$)",
            flags=re.I,
        ),
        re.compile(
            r"\b(?:which|what)\s+papers?[^?.!]{0,120}?\b(?:directly\s+|most\s+directly\s+)?(?:compare(?:s|d)?|define(?:s|d)?)\s+(.+?)(?:[?.!]|$)",
            flags=re.I,
        ),
        re.compile(
            r"\bbesides\s+this\s+paper[^?.!]{0,120}?\b(?:directly\s+|most\s+directly\s+)?(?:compare(?:s|d)?|define(?:s|d)?)\s+(.+?)(?:[?.!]|$)",
            flags=re.I,
        ),
    )
    out: list[str] = []
    seen: set[str] = set()

    def _push(raw: str) -> None:
        cleaned = _clean_doc_focus_phrase(raw)
        if not _looks_informative_doc_focus_phrase(cleaned):
            return
        norm = _norm_text_for_match(cleaned)
        if len(norm) < 3 or norm in seen:
            return
        seen.add(norm)
        out.append(norm)

    for quoted in re.findall(r"[\"“”‘’]([^\"“”‘’]{2,120})[\"“”‘’]", text):
        _push(str(quoted or ""))
    for pattern in patterns:
        m = pattern.search(text)
        if not m:
            continue
        raw = str(m.group(1) or "")
        _push(raw)
        if re.search(r"\b(compare|compares|compared|comparison|versus|vs\.?)\b", text, flags=re.I):
            for part in re.split(r"\b(?:and|vs\.?|versus)\b", raw, flags=re.I):
                _push(part)
    return tuple(out[:6])


def _focus_phrase_matches_doc_surface(phrase: str, surface_text: str) -> bool:
    norm_phrase = _norm_text_for_match(phrase)
    norm_surface = _norm_text_for_match(surface_text)
    if not norm_phrase or not norm_surface:
        return False
    if norm_phrase in norm_surface:
        return True
    phrase_toks = [t for t in tokenize(norm_phrase) if len(t) >= 4 and t not in _DOC_HINT_STOP_TOKENS]
    if len(phrase_toks) < 2:
        return False
    surface_tokens = set(tokenize(norm_surface))
    overlap = [tok for tok in phrase_toks if tok in surface_tokens]
    if len(overlap) >= min(2, len(phrase_toks)):
        return True
    if len(phrase_toks) >= 3 and len(overlap) >= max(2, len(phrase_toks) - 1):
        return True
    return False


def _doc_focus_match_score(
    *,
    prompt_text: str,
    source_path: str,
    snippets: list[str],
    headings: list[str],
) -> float:
    phrases = _extract_doc_focus_phrases(prompt_text)
    if not phrases:
        return 0.0
    p = Path(str(source_path or "").strip())
    title_surface = " ".join(
        str(x or "")
        for x in (p.name, p.stem, re.sub(r"^[A-Za-z]+-\d{4}[-_ ]*", "", p.stem))
        if str(x or "").strip()
    )
    body_surface = " ".join(str(x or "") for x in list(headings or [])[:8] + list(snippets or [])[:4] if str(x or "").strip())
    prompt_low = str(prompt_text or "").lower()
    is_compare = bool(re.search(r"\b(compare|compares|compared|comparison|versus|vs\.?)\b", prompt_low))
    is_define = bool(re.search(r"\b(defin(?:e|es|ed|ition)|what\s+is|introduced?\s+as)\b", prompt_low))
    explain_surface = body_surface
    score = 0.0
    matched = 0
    for phrase in phrases:
        title_hit = _focus_phrase_matches_doc_surface(phrase, title_surface)
        body_hit = _focus_phrase_matches_doc_surface(phrase, body_surface)
        if title_hit:
            score += 5.2
        if body_hit:
            score += 4.1
        if title_hit or body_hit:
            matched += 1
        if is_define and body_hit and re.search(r"\b(defin(?:e|es|ed|ition)|refers?\s+to|introduced?\s+as|is\s+defined\s+as)\b", explain_surface, flags=re.I):
            score += 1.8
    if is_compare and matched >= 2 and re.search(r"\b(compare|compares|compared|comparison|versus|vs\.?)\b", f"{title_surface}\n{body_surface}", flags=re.I):
        score += 4.6
    if matched >= 1 and len(phrases) == 1:
        score += 1.2
    return float(min(score, 18.0))


def _build_doc_anchor_focus_query(prompt_text: str, source_path: str, anchor_hint: dict[str, object]) -> str:
    q = str(prompt_text or "").strip()
    src = Path(str(source_path or "").strip())
    for cand in (src.name, src.stem, re.sub(r"^[A-Za-z]+-\d{4}[-_ ]*", "", src.stem)):
        s = str(cand or "").strip()
        if not s:
            continue
        q = re.sub(re.escape(s), " ", q, flags=re.I)
    parts: list[str] = []
    phrases = anchor_hint.get("phrases") if isinstance(anchor_hint, dict) else None
    if isinstance(phrases, list):
        parts.extend(str(x).strip() for x in phrases if str(x or "").strip())
    remain_tokens = [
        t
        for t in tokenize(_norm_text_for_match(q))
        if len(t) >= 2 and t not in _DOC_HINT_STOP_TOKENS
    ]
    parts.extend(remain_tokens[:6])
    return " ".join(dict.fromkeys(parts)).strip()


def _anchor_text_bonus(text: str, anchor_hint: dict[str, object]) -> float:
    if not isinstance(anchor_hint, dict) or not anchor_hint:
        return 0.0
    kind = str(anchor_hint.get("kind") or "").strip().lower()
    try:
        num = int(anchor_hint.get("number") or 0)
    except Exception:
        num = 0
    number_text = str(anchor_hint.get("number_text") or num or "").strip()
    if (not kind) or (num <= 0):
        return 0.0
    low = str(text or "").lower()
    if not low:
        return 0.0
    score = 0.0
    patterns_by_kind = {
        "section": (
            rf"(?:^|[\n/]|\b)(?:section|sec\.?|chapter)?\s*"
            rf"{re.escape(number_text)}(?=\s|[.:：、\-]|$)|"
            rf"第\s*{re.escape(number_text)}\s*(?:节|章)"
        ),
        "figure": rf"(?:fig(?:ure)?\.?\s*{num}\b|图\s*{num}(?!\d)|图{num}(?!\d)|第\s*{num}\s*张图)",
        "table": rf"(?:table\.?\s*{num}\b|表\s*{num}(?!\d)|表{num}(?!\d)|第\s*{num}\s*表)",
        "equation": rf"(?:eq(?:uation)?\.?\s*{num}\b|formula\s*{num}\b|公式\s*{num}(?!\d)|公式{num}(?!\d)|式\s*{num}(?!\d)|式{num}(?!\d)|[\(（]\s*{num}\s*[\)）]|\\tag\{{\s*{num}\s*\}})",
        "theorem": rf"(?:theorem\.?\s*{num}\b|定理\s*{num}(?!\d)|定理{num}(?!\d))",
        "lemma": rf"(?:lemma\.?\s*{num}\b|引理\s*{num}(?!\d)|引理{num}(?!\d))",
        "definition": rf"(?:definition\.?\s*{num}\b|定义\s*{num}(?!\d)|定义{num}(?!\d))",
        "proposition": rf"(?:proposition\.?\s*{num}\b|命题\s*{num}(?!\d)|命题{num}(?!\d))",
        "corollary": rf"(?:corollary\.?\s*{num}\b|推论\s*{num}(?!\d)|推论{num}(?!\d))",
    }
    pat = patterns_by_kind.get(kind)
    first_anchor_pos: int | None = None
    if pat:
        m_anchor = re.search(pat, low, flags=re.I)
        if m_anchor:
            first_anchor_pos = int(m_anchor.start())
            score += 25.0
            # Stronger boost when the anchor text is at the very start of the snippet
            # (this IS the figure caption / equation definition itself).
            if first_anchor_pos <= 120:
                score += 15.0
    labels = _ANCHOR_KIND_LABELS.get(kind) or ()
    if any(lab in low for lab in labels):
        score += 3.0
    if number_text in low:
        score += 1.5
    if kind in {"figure", "table"}:
        label_pat = r"(?:fig(?:ure)?\.?|图)" if kind == "figure" else r"(?:table\.?|表)"
        direct_caption_pat = rf"(?:^|[\n/]\s*|!\[[^\]]*)\**\s*{label_pat}\s*{num}(?!\d)"
        m_direct = re.search(direct_caption_pat, low, flags=re.I)
        if m_direct:
            score += 20.0
            direct_pos = int(m_direct.start())
            if direct_pos <= 80:
                score += 15.0
            elif first_anchor_pos is not None and direct_pos > (int(first_anchor_pos) + 120):
                score -= 6.0
        other_caption_pat = (
            r"(?:^|[\n/]\s*|!\[[^\]]*)\**\s*(?:fig(?:ure)?\.?|图)\s*(\d+)(?!\d)"
            if kind == "figure"
            else r"(?:^|[\n/]\s*|!\[[^\]]*)\**\s*(?:table\.?|表)\s*(\d+)(?!\d)"
        )
        m_other = re.search(other_caption_pat, low, flags=re.I)
        if m_other:
            try:
                other_num = int(m_other.group(1))
            except Exception:
                other_num = num
            if other_num != num and first_anchor_pos is not None and int(m_other.start()) < first_anchor_pos:
                score -= 8.0
    return score


def _anchor_regexes(anchor_hint: dict[str, object]) -> list[re.Pattern[str]]:
    if not isinstance(anchor_hint, dict) or not anchor_hint:
        return []
    kind = str(anchor_hint.get("kind") or "").strip().lower()
    try:
        num = int(anchor_hint.get("number") or 0)
    except Exception:
        num = 0
    number_text = str(anchor_hint.get("number_text") or num or "").strip()
    if (not kind) or (num <= 0):
        return []
    raw_patterns = {
        "section": [
            rf"(?:^|\n)\s*\#{{1,6}}\s*{re.escape(number_text)}(?=\s|[.:：、\-]|$)",
            rf"\b(?:section|sec\.?|chapter)\s*{re.escape(number_text)}\b",
            rf"第\s*{re.escape(number_text)}\s*(?:节|章)",
        ],
        "figure": [
            rf"fig(?:ure)?\.?\s*{num}\b",
            rf"第\s*{num}\s*张?图",
            rf"图\s*{num}(?!\d)",
            rf"图{num}(?!\d)",
        ],
        "table": [
            rf"table\.?\s*{num}\b",
            rf"第\s*{num}\s*表",
            rf"表\s*{num}(?!\d)",
            rf"表{num}(?!\d)",
        ],
        "equation": [
            rf"eq(?:uation)?\.?\s*{num}\b",
            rf"formula\s*{num}\b",
            rf"(?:公式|式)\s*{num}(?!\d)",
            rf"(?:公式|式){num}(?!\d)",
            rf"[\(（]\s*{num}\s*[\)）]",
            rf"\\tag\{{\s*{num}\s*\}}",
        ],
        "theorem": [rf"theorem\.?\s*{num}\b", rf"定理\s*{num}(?!\d)", rf"定理{num}(?!\d)"],
        "lemma": [rf"lemma\.?\s*{num}\b", rf"引理\s*{num}(?!\d)", rf"引理{num}(?!\d)"],
        "definition": [rf"definition\.?\s*{num}\b", rf"定义\s*{num}(?!\d)", rf"定义{num}(?!\d)"],
        "proposition": [rf"proposition\.?\s*{num}\b", rf"命题\s*{num}(?!\d)", rf"命题{num}(?!\d)"],
        "corollary": [rf"corollary\.?\s*{num}\b", rf"推论\s*{num}(?!\d)", rf"推论{num}(?!\d)"],
    }
    out: list[re.Pattern[str]] = []
    for pat in raw_patterns.get(kind, []):
        try:
            out.append(re.compile(pat, flags=re.I))
        except Exception:
            continue
    return out


def _find_anchor_snippets_in_md(
    md_path: Path,
    anchor_hint: dict[str, object],
    *,
    max_snippets: int = 3,
    snippet_chars: int = 1600,
) -> list[dict]:
    md_path = Path(md_path)
    if not md_path.exists():
        return []
    pats = _anchor_regexes(anchor_hint)
    if not pats:
        return []

    # Pre-indexed anchor lookup: O(1) via source_blocks cache.
    anchor_index = _load_anchor_index_cached(md_path)
    hint_kind = str(anchor_hint.get("kind") or "").strip().lower()
    hint_number = 0
    try:
        hint_number = int(anchor_hint.get("number") or 0)
    except Exception:
        hint_number = 0
    hint_figure_scope = normalize_figure_scope(anchor_hint.get("figure_scope")) if hint_kind == "figure" else ""
    hint_figure_key = figure_key_for_scope(hint_figure_scope, hint_number)
    if hint_kind and hint_number > 0:
        kind_plural = hint_kind + "s" if hint_kind != "equation" else "equations"
        entries = anchor_index.get(kind_plural, []) if kind_plural in anchor_index else anchor_index.get(hint_kind + "s", [])
        selected_entries = (
            filter_figure_index_rows(
                entries,
                figure_number=hint_number,
                figure_scope=hint_figure_scope,
            )
            if hint_kind == "figure"
            else [entry for entry in entries if int(entry.get("number") or 0) == hint_number]
        )
        for entry in selected_entries:
            if int(entry.get("number") or 0) == hint_number:
                caption = str(entry.get("caption_text") or entry.get("text_fragment") or "").strip()
                heading = str(entry.get("heading_path") or "").strip()
                block_id = str(entry.get("block_id") or "").strip()
                if caption:
                    meta = {"source_path": str(md_path), "anchor_read": True, "heading_path": heading, "block_id": block_id}
                    if hint_kind == "figure":
                        meta["figure_scope"] = str(entry.get("figure_scope") or hint_figure_scope).strip()
                        meta["figure_key"] = str(entry.get("figure_key") or hint_figure_key).strip()
                        meta["anchor_target_scope"] = hint_figure_scope
                        meta["anchor_target_key"] = hint_figure_key
                    out = [{"score": 95.0, "id": block_id, "text": caption, "meta": meta}]
                    # If we have more entries for the same figure/table (e.g., caption paragraphs),
                    # include them as additional snippets.
                    for extra in selected_entries:
                        if int(extra.get("number") or 0) != hint_number:
                            continue
                        if extra.get("block_id") == block_id:
                            continue
                        extra_text = str(extra.get("caption_text") or extra.get("text_fragment") or "").strip()
                        if extra_text and extra_text != caption:
                            extra_meta = {"source_path": str(md_path), "anchor_read": True,
                                          "heading_path": str(extra.get("heading_path") or heading),
                                          "block_id": str(extra.get("block_id") or "")}
                            if hint_kind == "figure":
                                extra_meta["figure_scope"] = str(extra.get("figure_scope") or hint_figure_scope).strip()
                                extra_meta["figure_key"] = str(extra.get("figure_key") or hint_figure_key).strip()
                                extra_meta["anchor_target_scope"] = hint_figure_scope
                                extra_meta["anchor_target_key"] = hint_figure_key
                            out.append({
                                "score": 85.0,
                                "id": str(extra.get("block_id") or ""),
                                "text": extra_text,
                                "meta": extra_meta,
                            })
                    if len(out) >= max(1, int(max_snippets)):
                        out = out[:max(1, int(max_snippets))]
                    return out

    text = _read_text_cached(md_path)
    if not text.strip():
        return []

    chunks = chunk_markdown(text, source_path=str(md_path), chunk_size=900, overlap=0)
    scored: list[tuple[float, dict]] = []
    for c in chunks:
        body = str(c.get("text") or "").strip()
        if len(body) < 40:
            continue
        hits = 0
        first_match_pos: int | None = None
        for pat in pats:
            try:
                matches = list(pat.finditer(body))
                if hint_kind == "figure" and hint_figure_scope:
                    matches = [
                        match
                        for match in matches
                        if _figure_scope_for_anchor_match(body, match) == hint_figure_scope
                    ]
                hits += len(matches)
                if matches and ((first_match_pos is None) or (matches[0].start() < first_match_pos)):
                    first_match_pos = int(matches[0].start())
            except Exception:
                continue
        if hits <= 0:
            continue
        score = 40.0 + (12.0 * float(hits))
        body_low = body.lower()
        if body_low.lstrip().startswith(("fig.", "figure", "图", "table", "表", "equation", "theorem", "lemma", "definition", "proposition", "corollary", "定理", "引理", "定义", "命题", "推论")):
            score += 8.0
        meta = dict((c.get("meta") or {}))
        meta.setdefault("source_path", str(md_path))
        meta["anchor_read"] = True
        if hint_kind == "figure":
            meta["figure_scope"] = hint_figure_scope
            meta["figure_key"] = hint_figure_key
            meta["anchor_target_scope"] = hint_figure_scope
            meta["anchor_target_key"] = hint_figure_key
        body_out = body
        if first_match_pos is not None and len(body_out) > snippet_chars:
            start = max(0, int(first_match_pos) - min(240, max(80, snippet_chars // 5)))
            end = min(len(body_out), start + max(240, int(snippet_chars)))
            body_out = body_out[start:end].strip()
            if start > 0:
                body_out = "..." + body_out
            if end < len(body):
                body_out = body_out.rstrip() + "..."
        elif len(body_out) > snippet_chars:
            body_out = body_out[:snippet_chars].rstrip() + "..."
        scored.append((float(score), {"score": float(score), "id": str(c.get("id") or ""), "text": body_out, "meta": meta}))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict] = []
    seen: set[str] = set()
    for _score, item in scored:
        body = str(item.get("text") or "").strip()
        if (not body) or (body in seen):
            continue
        seen.add(body)
        out.append(item)
        if len(out) >= max(1, int(max_snippets)):
            break
    return out


_INTENT_METHOD_RE = re.compile(
    r"(怎么|如何|怎样|方法|实现|步骤|流程|原理|机制|算法|模型|编码|采样|掩膜|推导|公式|"
    r"\bhow\b|\bmethod\b|\bmethods\b|\bapproach\b|\bimplement(?:ation)?\b|\balgorithm\b|\bmodel\b|\bequation\b|\bderive\b)",
    flags=re.I,
)
_INTENT_RESULT_RE = re.compile(
    r"(结果|性能|指标|对比|比较|实验|消融|准确率|误差|提升|"
    r"\bresult\b|\bresults\b|\bperformance\b|\bmetric\b|\bevaluation\b|\bcompare\b|\bablation\b|\bbenchmark\b)",
    flags=re.I,
)
_INTENT_LIMIT_RE = re.compile(
    r"(局限|限制|不足|失效|失败|未来工作|讨论|结论|"
    r"\blimitation\b|\blimitations\b|\bfailure\b|\bfuture work\b|\bdiscussion\b|\bconclusion\b)",
    flags=re.I,
)
_INTENT_BG_RE = re.compile(
    r"(是什么|定义|含义|概念|背景|综述|介绍|"
    r"\bwhat is\b|\bdefinition\b|\bbackground\b|\boverview\b|\bintro(?:duction)?\b)",
    flags=re.I,
)
_HEADING_METHOD_RE = re.compile(
    r"\b(method|methods|approach|algorithm|implementation|model|theory|derivation|equation|setup|pipeline)\b|"
    r"(方法|算法|实现|模型|原理|推导|公式|系统|装置|编码|采样|掩膜|实验设置)",
    flags=re.I,
)
_HEADING_RESULT_RE = re.compile(
    r"\b(result|results|evaluation|experiment|experiments|benchmark|ablation|analysis)\b|"
    r"(结果|实验|评估|性能|指标|对比|消融|分析)",
    flags=re.I,
)
_HEADING_BG_RE = re.compile(
    r"\b(abstract|introduction|background|overview|related\s+work|preliminar(?:y|ies))\b|"
    r"(摘要|引言|背景|概述|相关工作|预备知识|定义)",
    flags=re.I,
)
_HEADING_DISCUSSION_RE = re.compile(
    r"\b(discussion|conclusion|limitations?|future\s+work)\b|(讨论|结论|局限|未来工作)",
    flags=re.I,
)


def _question_intent_flags(question: str) -> dict[str, bool]:
    q = str(question or "").strip()
    return {
        "references": _wants_reference_navigation(q),
        "method": bool(_INTENT_METHOD_RE.search(q)),
        "result": bool(_INTENT_RESULT_RE.search(q)),
        "limitation": bool(_INTENT_LIMIT_RE.search(q)),
        "background": bool(_INTENT_BG_RE.search(q)),
    }


def _is_discussion_or_conclusion_heading(h: str) -> bool:
    s = _normalize_heading(h)
    if not s:
        return False
    return bool(_HEADING_DISCUSSION_RE.search(s))


def _heading_intent_bonus_for_question(heading: str, question: str) -> float:
    h = _normalize_heading(heading)
    if not h:
        return 0.0
    low = h.lower()
    intent = _question_intent_flags(question)
    is_ref = _is_reference_heading_like(low)
    is_disc = _is_discussion_or_conclusion_heading(low)
    is_method = bool(_HEADING_METHOD_RE.search(low))
    is_result = bool(_HEADING_RESULT_RE.search(low))
    is_bg = bool(_HEADING_BG_RE.search(low))

    score = 0.0
    if intent["references"]:
        if is_ref:
            score += 3.2
        elif is_disc:
            score -= 0.6
        return score

    if is_ref:
        score -= 3.4

    if intent["limitation"]:
        if is_disc:
            score += 2.8
        if is_method:
            score += 0.6
        if is_result:
            score += 0.8
        return score

    # For "how/method" queries, avoid discussion/conclusion starts.
    if intent["method"]:
        if is_method:
            score += 3.2
        if is_bg:
            score += 1.1
        if is_result:
            score += 0.6
        if is_disc:
            score -= 4.0
        return score

    if intent["result"]:
        if is_result:
            score += 2.8
        if is_method:
            score += 1.0
        if is_bg:
            score += 0.2
        if is_disc:
            score -= 1.5
        return score

    if intent["background"]:
        if is_bg:
            score += 2.6
        if is_method:
            score += 0.9
        if is_result:
            score += 0.2
        if is_disc:
            score -= 2.4
        return score

    # Generic default: mild penalty for discussion/conclusion as start location.
    if is_disc:
        score -= 1.2
    return score


def _should_avoid_discussion_for_question(question: str) -> bool:
    flags = _question_intent_flags(question)
    if flags["references"] or flags["limitation"]:
        return False
    # Default behavior: discussion/conclusion are not good entry points unless user explicitly asks for them.
    return True


def _best_loc_heading_for_question(meta: dict, *, question: str, source_path: str = "") -> tuple[str, str]:
    if not isinstance(meta, dict):
        return "", ""
    locs = meta.get("ref_locs")
    if not isinstance(locs, list) or (not locs):
        return "", ""
    best_score = -1e9
    best_path = ""
    for loc in locs:
        if not isinstance(loc, dict):
            continue
        hp = _sanitize_heading_path_for_navigation(
            str(loc.get("heading_path") or loc.get("heading") or "").strip(),
            question=question,
            source_path=source_path,
        )
        if not hp:
            continue
        top_h, _sub_h = _split_heading_path_levels(hp)
        if not top_h or _is_non_navigational_heading(top_h, question=question, source_path=source_path):
            continue
        if _should_avoid_discussion_for_question(question) and _is_discussion_or_conclusion_heading(top_h):
            continue
        try:
            base = float(loc.get("score_adj", loc.get("score", 0.0)) or 0.0)
        except Exception:
            base = 0.0
        score = base + _heading_intent_bonus_for_question(hp, question)
        if score > best_score:
            best_score = score
            best_path = hp
    if not best_path:
        return "", ""
    sec, sub = _split_heading_path_levels(best_path)
    return sec, sub


def _page_range_from_meta(meta: dict) -> tuple[int | None, int | None]:
    def _to_pos_int(x) -> int | None:
        try:
            v = int(x)
        except Exception:
            return None
        return v if v > 0 else None

    p0 = _to_pos_int(meta.get("page_start"))
    p1 = _to_pos_int(meta.get("page_end"))
    if p0 is None:
        p0 = _to_pos_int(meta.get("page"))
    if p0 is None:
        p0 = _to_pos_int(meta.get("page_num"))
    if p0 is None:
        p0 = _to_pos_int(meta.get("page_idx"))
    if p1 is None:
        p1 = p0
    if (p0 is not None) and (p1 is not None) and p1 < p0:
        p0, p1 = p1, p0
    return p0, p1

def _translate_query_for_search(settings, prompt_text: str) -> str | None:
    """
    Translate a CJK-heavy query to a compact English search query (keywords),
    so BM25 can match English papers.
    """
    q_raw = (prompt_text or "").strip()
    if not q_raw:
        return None
    # Strip bound-source hints / file paths before language heuristics, otherwise a
    # Chinese question with an appended PDF path looks "latin-heavy" and won't translate.
    q = q_raw
    try:
        q = re.sub(r"(?i)\b[a-z]:\\\\[^\s]{6,}\b", " ", q)  # Windows paths
        q = re.sub(r"(?i)\bhttps?://[^\s]{6,}\b", " ", q)  # URLs
        q = re.sub(r"(?i)\b[^\s]+\\.(pdf|md|txt|docx?)\b", " ", q)  # filenames
        q = " ".join(q.split())
    except Exception:
        q = q_raw
    # If a bound-source hint was prepended (often a long Latin-only title), drop it for translation.
    try:
        m_cjk = re.search(r"[\u4e00-\u9fff]", q)
        if m_cjk and m_cjk.start() >= 12:
            prefix = q[: m_cjk.start()]
            if (re.search(r"[\u4e00-\u9fff]", prefix) is None) and (re.search(r"[A-Za-z]", prefix) is not None):
                q = q[m_cjk.start() :].strip()
    except Exception:
        q = q

    if not q:
        return None
    if not _has_cjk(q):
        return None
    # Allow translation for mixed queries like:
    #   "这篇文章核心问题是什么 NatPhoton-2019 ..."
    # where the bound-source hint introduces Latin tokens but the query is still CJK-heavy.
    if _has_latin(q):
        try:
            cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", q))
            latin_chars = len(re.findall(r"[A-Za-z]", q))
        except Exception:
            cjk_chars, latin_chars = 0, 0
        # If the query is mostly Latin, translation is unnecessary and can hurt recall.
        if cjk_chars < 6 or latin_chars > (cjk_chars * 3):
            return None

    key = hashlib.sha1((str(getattr(settings, "api_key", None)) + "|" + q).encode("utf-8", "ignore")).hexdigest()[:16]
    cached = _cache_get("trans", key)
    if isinstance(cached, str) and cached.strip():
        return cached.strip()

    # Fast dictionary-based fallback for common KB terms.
    # This keeps retrieval responsive even when translation LLM is slow/unavailable.
    terms: list[str] = []
    mapping = [
        ("钙钛矿激光器", "perovskite laser"),
        ("光泵浦", "optically pumped"),
        ("器件问题", "device challenge"),
        ("\u5355\u50cf\u7d20", "single-pixel"),
        ("\u5355\u5149\u5b50", "single-photon"),
        ("\u5355\u66dd\u5149", "single-shot"),
        ("\u5355\u6b21\u66dd\u5149", "single-shot"),
        ("\u538b\u7f29\u6210\u50cf", "compressive imaging"),
        ("\u538b\u7f29\u611f\u77e5", "compressed sensing"),
        ("\u538b\u7f29\u7387", "compression ratio"),
        ("\u91c7\u6837\u6570", "number of measurements"),
        ("\u91c7\u6837\u7387", "sampling rate"),
        ("\u6d4b\u91cf\u6570", "number of measurements"),
        ("\u6d4b\u91cf", "measurements"),
        ("\u6a21\u5f0f\u6570", "number of patterns"),
        ("\u91cd\u5efa\u8d28\u91cf", "reconstruction quality"),
        ("\u6838\u5fc3\u95ee\u9898", "core problem"),
        ("\u5173\u952e\u95ee\u9898", "key problem"),
        ("\u95ee\u9898\u610f\u8bc6", "problem formulation"),
        ("\u4e3a\u4ec0\u4e48", "motivation"),
        ("\u4f20\u7edf\u65b9\u6848", "conventional approach"),
        ("\u4e0d\u591f\u597d", "limitations"),
        ("\u4f18\u7f3a\u70b9", "advantages disadvantages"),
        ("\u4f18\u70b9", "advantages"),
        ("\u7f3a\u70b9", "disadvantages"),
        ("\u9002\u7528\u573a\u666f", "applications"),
        ("\u9002\u7528", "applicable"),
        ("\u4e3b\u6d41", "mainstream"),
        ("\u91cd\u5efa\u65b9\u6cd5", "reconstruction methods"),
        ("\u91cd\u5efa\u7b97\u6cd5", "reconstruction algorithm"),
        ("\u7b97\u6cd5", "algorithm"),
        ("\u6df1\u5ea6\u5b66\u4e60", "deep learning"),
        ("\u795e\u7ecf\u7f51\u7edc", "neural network"),
        ("\u7ed3\u6784\u5316\u63a2\u6d4b", "structured detection"),
        ("\u6fc0\u5149\u626b\u63cf\u663e\u5fae", "laser scanning microscopy"),
        ("\u5171\u805a\u7126", "confocal"),
        ("\u6743\u8861", "trade-off"),
        ("\u4e3b\u8981\u8d21\u732e", "main contribution"),
        ("\u6838\u5fc3\u8d21\u732e", "core contribution"),
        ("\u8d21\u732e", "contribution"),
        ("\u539f\u7406", "principle"),
        ("\u673a\u5236", "mechanism"),
        ("\u65b9\u6cd5", "method"),
        ("\u7b97\u6cd5", "algorithm"),
        ("\u5b9e\u9a8c", "experiment"),
        ("\u7ed3\u679c", "results"),
        ("\u8ba8\u8bba", "discussion"),
        ("\u7ed3\u8bba", "conclusion"),
        ("\u5c40\u9650", "limitation"),
        ("\u7f3a\u70b9", "limitation"),
        ("\u672a\u6765\u5de5\u4f5c", "future work"),
        ("\u590d\u73b0", "reproducibility"),
        ("\u590d\u73b0\u6027", "reproducibility"),
        ("\u5f15\u7528", "citation"),
        ("\u53c2\u8003\u6587\u732e", "references"),
        ("\u76ee\u5f55", "table of contents"),
        ("\u5927\u7eb2", "outline"),
        ("\u56fe", "figure"),
        # Do not match the single character ``表`` here: substring matching
        # would turn words such as ``代表性`` into an unrelated ``table``
        # query. Cover actual table intent with explicit phrases instead.
        ("\u8868\u683c", "table"),
        ("\u8868\u4e2d", "table"),
        ("\u8868\u5185", "table"),
        ("\u516c\u5f0f", "equation"),
        ("\u5b9a\u4e49", "definition"),
        ("\u5c0f\u6ce2", "wavelet"),
        ("\u6210\u50cf", "imaging"),
        ("\u91cd\u5efa", "reconstruction"),
        ("\u91c7\u6837", "sampling"),
        ("\u63a9\u819c", "mask pattern"),
        ("\u7f16\u7801", "coding"),
        ("\u5149\u8c31", "spectral"),
        ("\u8d85\u6750\u6599", "metamaterial"),
        ("\u8d85\u8868\u9762", "metasurface"),
        ("\u590d\u7528", "multiplexing"),
        ("\u9891\u5206", "frequency-division"),
        ("基准测试", "benchmark"),
        ("基准", "benchmark"),
        ("最高", "highest"),
        ("最大", "maximum"),
        ("最低", "lowest"),
        ("最小", "minimum"),
        ("模型", "model"),
        ("并列", "tie"),
        ("代表性应用", "representative applications"),
        ("哪些应用", "applications"),
        ("什么场景", "use cases"),
        ("综述", "review"),
    ]
    ascii_terms = [
        token
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9+._-]{1,}\b", q)
        if token.lower() not in {"the", "a", "an", "of", "for", "and", "or", "paper", "pdf", "md", "en"}
    ]
    terms.extend(ascii_terms[:8])
    for zh, en_term in mapping:
        if zh in q:
            terms.append(en_term)
    if terms:
        uniq: list[str] = []
        seen = set()
        for t in terms:
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)
        heuristic = " ".join(uniq[:8]).strip()
        if heuristic:
            _cache_set("trans", key, heuristic, max_items=500)
            return heuristic

    # No API key: we can't call the translation LLM, but the heuristic fallback above may still help.
    if not getattr(settings, "api_key", None):
        return None

    try:
        settings_fast = replace(
            settings,
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 8.0),
            max_retries=1,
        )
    except Exception:
        settings_fast = settings
    ds = DeepSeekChat(settings_fast)
    system = (
        "You translate Chinese research questions into an English search query for academic retrieval.\n"
        "Rules:\n"
        "- Output ONLY the English search query (no quotes, no explanations).\n"
        "- Prefer keywords and key phrases; include useful synonyms.\n"
        "- Keep it compact (8-18 tokens).\n"
        "- DO NOT confuse terms:\n"
        "  - 单曝光/单次曝光 -> single-shot, single exposure, snapshot\n"
        "  - 单像素 -> single-pixel\n"
        "  - 压缩成像 -> compressive imaging\n"
        "  - 光谱成像 -> spectral imaging\n"
        "- If the user didn't mention 单像素 avoid adding single-pixel.\n"
    )
    user = f"Question: {q}\n\nEnglish search keywords:"
    try:
        out = (ds.chat(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.0, max_tokens=80) or "").strip()
    except Exception:
        out = ""
    out = " ".join(out.split())
    _cache_set("trans", key, out, max_items=500)
    return out or None

def _llm_semantic_rerank_score(settings, *, question: str, doc_headings: list[str], snippets: list[str]) -> tuple[float, str]:
    """
    LLM-based semantic relevance scoring, grounded on snippets (not filenames).

    Returns: (score_0_100, short_reason)
    """
    if not settings or (not getattr(settings, "api_key", None)):
        return 0.0, ""

    q = (question or "").strip()
    if not q:
        return 0.0, ""

    hs = [h for h in (doc_headings or []) if isinstance(h, str)]
    hs = [h for h in hs if h and (not _is_probably_bad_heading(h))][:25]

    sn = []
    for s in (snippets or [])[:4]:
        s2 = " ".join((s or "").strip().split())
        if len(s2) > 420:
            s2 = s2[:420].rstrip() + "..."
        if s2:
            sn.append(s2)
    if not sn:
        return 0.0, ""

    cache_key = hashlib.sha1(("\n".join(sn) + "|" + q).encode("utf-8", "ignore")).hexdigest()[:16]
    v0 = _cache_get("rerank", cache_key)
    if isinstance(v0, dict):
        try:
            return float(v0.get("score", 0.0) or 0.0), str(v0.get("why", "") or "")
        except Exception:
            return 0.0, ""

    ds = DeepSeekChat(settings)
    en = _has_latin(q) and (not _has_cjk(q))
    if en:
        sys = (
            "You are a strict academic retriever reranker.\n"
            "Output JSON ONLY: {\"score\":number,\"why\":string}.\n"
            "Rules:\n"
            "- score is 0..100 and MUST reflect how directly the provided snippets answer the question.\n"
            "- Penalize false friends (e.g., single-shot vs single-pixel) when mismatched.\n"
            "- Use only the snippets/headings; do NOT use filenames.\n"
            "- why: <= 18 words.\n"
        )
    else:
        sys = (
            "你是严格的学术检索重排器。\n"
            "只能输出 JSON：{\"score\":number,\"why\":string}。\n"
            "规则：\n"
            "- score 为 0..100，必须反映“这些片段是否直接回答用户问题”。\n"
            "- 遇到术语假朋友要扣分（如 single-shot vs single-pixel）。\n"
            "- 只能根据 snippets/headings 判断，不能根据文件名判断。\n"
            "- why <= 18 个字，写清楚为什么。\n"
        )

    user = (
        f"Question: {q}\n\n"
        "Available headings:\n- " + "\n- ".join(hs) + "\n\n"
        "Snippets:\n- " + "\n- ".join(sn) + "\n"
    )
    try:
        out = (ds.chat(messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}], temperature=0.0, max_tokens=160) or "").strip()
    except Exception:
        out = ""

    if out.startswith("```"):
        out = out.strip().strip("`")
        out = out.replace("json", "", 1).strip()

    try:
        data = json.loads(out)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return 0.0, ""

    try:
        score = float(data.get("score", 0.0) or 0.0)
    except Exception:
        score = 0.0
    score = max(0.0, min(100.0, score))
    why = str(data.get("why") or "").strip()
    _cache_set("rerank", cache_key, {"score": score, "why": why}, max_items=600)
    return score, why


def _expand_query_via_llm(settings, prompt_text: str) -> list[str]:
    """
    Generate up to 3 search query variants via LLM for BM25 recall expansion.
    Always prepends the original query as the first variant.
    Returns [prompt_text] on failure or when expansion is not beneficial.
    """
    q = (prompt_text or "").strip()
    if not q or len(q) < 4:
        return [q] if q else []

    cache_key = hashlib.sha1((str(getattr(settings, "api_key", None)) + "|expand|" + q).encode("utf-8", "ignore")).hexdigest()[:16]
    cached = _cache_get("query_expand", cache_key)
    if isinstance(cached, list) and len(cached) >= 1:
        # Always ensure original query is first
        return [q] + [v for v in cached if v and v != q]

    if not getattr(settings, "api_key", None):
        return [q]

    try:
        settings_fast = replace(
            settings,
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 8.0),
            max_retries=0,
        )
    except Exception:
        settings_fast = settings
    ds = DeepSeekChat(settings_fast)

    has_cjk = bool(re.search(r"[一-鿿]", q))
    system = (
        "You generate search query variants for academic paper retrieval.\n"
        "Rules:\n"
        "- Output up to 3 alternative search queries, one per line, no numbering.\n"
        "- Each query should be a compact keyword phrase (8-20 tokens).\n"
        "- Focus on the topic's core concepts and related terminology.\n"
        "- Explore BROADENING expansions: capture synonyms and parallel concepts\n"
        "  that would find papers on the same topic using different terminology.\n"
        + (
            "- For CJK queries: generate ALL variants in English (academic papers are in English).\n"
            "  Do NOT keep CJK characters — translate core concepts to English keywords.\n"
            if has_cjk
            else "- For English input, include synonym variants that may not share keywords.\n"
        )
        + "- Do NOT add quotes, numbering, or commentary.\n"
        "- If the query is already well-formed and has no useful variants, output only: NONE\n"
    )
    user = f"Query: {q}\n\nAlternative search queries:"
    try:
        out = (
            ds.chat(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3,
                max_tokens=160,
            )
            or ""
        ).strip()
    except Exception:
        out = ""

    variants: list[str] = []
    if out and out.strip().upper() != "NONE":
        for line in out.split("\n"):
            line = " ".join(line.strip().strip('"\'「」').split())
            if line and len(line) >= 4 and line.lower() != q.lower():
                variants.append(line)
        # Limit to 3 variants
        variants = variants[:3]

    _cache_set("query_expand", cache_key, variants, max_items=300)
    return [q] + variants


def _deterministic_query_variants(prompt_text: str) -> list[str]:
    """Cheap domain-aware query variants for common research questions.

    These variants cover mechanism/term aliases that are easy to miss with
    lexical BM25, especially mixed Chinese-English questions where we avoid a
    translation call because the prompt already contains a long English title.
    """

    q = str(prompt_text or "").strip()
    if not q:
        return []
    low = q.lower()
    variants: list[str] = []

    def has_any(*needles: str) -> bool:
        return any(str(item or "").lower() in low for item in needles if str(item or "").strip())

    def add(value: str) -> None:
        v = " ".join(str(value or "").split()).strip()
        if not v:
            return
        key = v.lower()
        if key == low:
            return
        if key in {x.lower() for x in variants}:
            return
        variants.append(v)

    if has_any("hatnet"):
        add(
            "HATNet Dual-Scale Transformer Large-Scale Single-Pixel Imaging "
            "tensor ISTA deep unfolding tensor gradient descent hybrid-attention denoising"
        )
        add(
            "HATNet stages TGD U-shaped denoiser S-SA C-SA Kronecker SPI "
            "learnable step size proximal mapping"
        )

    if has_any("learned primal-dual", "learned primal dual") and has_any(
        "pdhg",
        "primal-dual",
        "primal dual",
        "可学习网络",
        "替换",
        "初始化",
    ):
        add(
            "Learned PDHG proximal operators replaced parametrized convolutional neural networks "
            "dual update primal update fixed iterations"
        )
        if has_any("初始化", "initialization", "initialise", "initialize", "starting point", "zero", "fbp"):
            add(
                "Learned Primal-Dual choice of starting point zero initialization "
                "filtered back-projection FBP pseudo-inverse final results"
            )

    if has_any("ista-net", "ista net") and has_any(
        "ista",
        "iteration",
        "iterative",
        "algorithm",
        "unroll",
        "unfold",
        "phase",
        "module",
        "迭代",
        "展开",
        "模块",
        "可学习",
    ):
        add(
            "ISTA-Net maps ISTA update steps fixed phases each phase one iteration "
            "r(k) module x(k) module learnable step size shrinkage threshold nonlinear transforms"
        )

    if has_any(
        "refocus",
        "refocusing",
        "out of focus",
        "\u79bb\u7126",
        "\u91cd\u805a\u7126",
        "\u91cd\u65b0\u5bf9\u7126",
        "\u91cd\u5bf9\u7126",
    ):
        add(
            "digital refocusing out-of-focus sample ray tracing wave propagation "
            "diffraction angular information"
        )
    if has_any("perovskite laser", "perovskite lasing", "钙钛矿激光器", "钙钛矿激光"):
        # Keep the base expansion technology-neutral. A generic perovskite-laser
        # question may refer to optical pumping, nanowire lasers, or another
        # device family; it must not silently become the dual-cavity Nature paper.
        add("perovskite laser lasing optical gain threshold cavity device")
        if has_any(
            "dual-cavity",
            "dual cavity",
            "electrically driven",
            "electrical injection",
            "electrically injected",
            "peled",
            "双腔",
            "电驱动",
            "电注入",
            "电泵浦",
        ):
            add(
                "electrically driven lasing dual-cavity perovskite device microcavity "
                "PeLED electrical injection laser"
            )
    if has_any("trade-off", "tradeoff", "\u6743\u8861", "\u539a\u6837\u672c", "thick sample", "thick samples", "s2ism"):
        add(
            "structured detection microscopy thick samples resolution SNR signal-to-noise "
            "optical sectioning out-of-focus background"
        )
    if has_any(
        "benefit",
        "risk",
        "advantage",
        "advantages",
        "disadvantage",
        "challenge",
        "challenges",
        "limitation",
        "limitations",
        "\u597d\u5904",
        "\u574f\u5904",
        "\u98ce\u9669",
        "\u4f18\u52bf",
        "\u6311\u6218",
        "\u5c40\u9650",
        "\u7f3a\u70b9",
    ) and has_any(
        "deep learning",
        "\u6df1\u5ea6\u5b66\u4e60",
    ):
        add(
            "deep learning single-pixel imaging advantages challenges data generalization "
            "interpretability speed reconstruction quality"
        )
    if has_any("pidl"):
        add(
            "physics-informed deep learning computational single-photon imaging "
            "physical prior data generator neural network loss inference"
        )
    if has_any("single-photon", "single photon", "\u5355\u5149\u5b50") and has_any(
        "physics-informed deep learning",
        "physics informed deep learning",
        "pidl",
    ):
        add(
            "High-resolution single-photon imaging physics-informed deep learning "
            "physical multi-source noise model SPAD arrays crosstalk dark count rate"
        )
    if (
        has_any("single-photon", "single photon", "\u5355\u5149\u5b50")
        and has_any(
            "physics-informed deep learning",
            "physics informed deep learning",
            "pidl",
        )
        and has_any(
            "detector",
            "photodetector",
            "review",
            "survey",
            "\u63a2\u6d4b\u5668",
            "\u7efc\u8ff0",
            "\u600e\u4e48\u642d\u914d\u8bfb",
        )
    ):
        add(
            "Emerging single-photon detection technique high-performance "
            "photodetector review SPAD"
        )
        add(
            "High-resolution single-photon imaging physics-informed deep learning "
            "real SPAD noise"
        )
    if has_any("piln", "part-based image-loop", "image-loop network"):
        add(
            "part-based image-loop network single-pixel imaging ILNet physical model "
            "untrained neural network inference"
        )
    if has_any("piln", "part-based image-loop", "image-loop network") and has_any(
        "review",
        "survey",
        "taxonomy",
        "position",
        "relationship",
        "mainline",
        "\u7efc\u8ff0",
        "\u4e3b\u7ebf",
        "\u4f4d\u7f6e",
        "\u5173\u7cfb",
    ):
        add(
            "data-driven model-driven hybrid-driven single-pixel imaging deep learning "
            "taxonomy untrained network measurements physical model"
        )
    if has_any("origin", "source", "comes from", "prior", "previous", "\u6765\u6e90", "\u51fa\u5904", "\u4e4b\u524d", "\u5df2\u6709"):
        add("prior work existing method background reference citation source")
    wants_cassi = has_any("cassi")
    wants_sci_lineage = bool(
        has_any(
            "snapshot compressive imaging",
            "snapshot compressive",
            "compressive snapshot",
            "压缩快照",
            "快照压缩",
        )
        and has_any("3d", "three-dimensional", "三维", "光谱", "spectral")
    )
    if wants_cassi or wants_sci_lineage:
        add(
            "CASSI single-shot compressive spectral imaging dual-disperser "
            "two dispersive elements binary-valued aperture spectral datacube"
        )
    if wants_cassi and has_any("dcd") and has_any("dltr"):
        add(
            "DLTR dimension-discriminative low-rank tensor hyperspectral image "
            "overlapped cubic patches mode unfolding weighted rank regularization"
        )
    if wants_sci_lineage or has_any("scinerf", "scigs"):
        add(
            "SCINeRF neural radiance fields snapshot compressive image "
            "physical imaging process SCI training NeRF 3D scene"
        )
        add(
            "SCIGS 3D Gaussian Splatting snapshot compressive image "
            "single compressed image dynamic 3D scenes"
        )
    microscopy_method_count = sum(
        (
            has_any("structured detection", "s2ism", "\u7ed3\u6784\u5316\u63a2\u6d4b"),
            has_any("interferometric", "iism", "\u5e72\u6d89"),
            has_any("light-field", "light field", "qclfm", "\u5149\u573a"),
        )
    )
    if microscopy_method_count >= 2:
        add(
            "Structured detection for simultaneous super-resolution and optical "
            "sectioning s2ISM laser scanning microscopy"
        )
        add(
            "Interferometric image scanning microscopy label-free live cells "
            "enhanced lateral resolution"
        )
        add(
            "Quantum correlation light-field microscope position angular information "
            "volumetric reconstruction extreme depth of field"
        )
    if has_any("3d single-pixel video", "3d single pixel video") and has_any(
        "detector",
        "探测器",
        "frame",
        "fps",
        "速度",
        "实时",
    ):
        add(
            "3D single-pixel video four spatially-separated single-pixel detectors "
            "continuous real-time 3D video 8 frames per second 64 x 64 pixels"
        )
    if has_any(
        "single-pixel compressive holography",
        "single pixel compressive holography",
        "单像素压缩全息",
        "单像素全息",
    ) and has_any("throughput", "phase", "相移", "吞吐"):
        add(
            "high-throughput single-pixel compressive holography beat frequency "
            "heterodyne holography phase stepping naturally in time"
        )
    if has_any(
        "sequential compressed sensing",
        "sequentially designed compressed sensing",
        "顺序压缩感知",
        "序贯压缩感知",
    ):
        add(
            "sequential adaptive compressed sensing signal support recovery "
            "distilled sensing exact support lower SNR"
        )
    if has_any("spad") and has_any(
        "geiger",
        "breakdown",
        "quench",
        "淬灭",
        "击穿",
        "雪崩",
    ):
        add(
            "SPAD operates in Geiger mode reverse bias breakdown voltage "
            "quenching circuit avalanche single photon detection"
        )
    if has_any(
        "beginner",
        "getting started",
        "reading roadmap",
        "刚开始",
        "入门",
        "主线",
        "先读",
    ) and has_any("single-pixel imaging", "single pixel imaging", "单像素成像"):
        add("Principles and prospects for single-pixel imaging review camera architecture applications")
        add("Hadamard single-pixel imaging versus Fourier single-pixel imaging sampling basis comparison")
        add("Advances and Challenges of Single-Pixel Imaging Based on Deep Learning review taxonomy")
    if (
        has_any("single-pixel", "single pixel", "\u5355\u50cf\u7d20")
        and has_any(
            "applications",
            "use case",
            "use cases",
            "representative",
            "review",
            "survey",
            "\u4ec0\u4e48\u573a\u666f",
            "\u503c\u5f97\u7528",
            "\u4ee3\u8868\u6027\u5e94\u7528",
            "\u7efc\u8ff0",
        )
    ):
        add(
            "Principles and prospects for single-pixel imaging applications "
            "wavelengths outside FPA technology high frame rates three dimensions "
            "hazardous gas leaks autonomous vehicles camera architecture review"
        )
    return variants[:4]


def _merge_expanded_results(
    results: list[tuple[list[dict], list[float], str]],
    top_k: int,
    *,
    weights: list[float] | None = None,
) -> tuple[list[dict], list[float]]:
    """
    Reciprocal Rank Fusion (RRF) over multiple query result sets.
    Each entry: (hits, scores, query_text).
    Deduplicates by chunk_id. Returns merged (hits, scores).

    When *weights* is provided (one per result set), each set's RRF
    contribution is multiplied by its weight.  This lets the original
    query (weight > 1.0) dominate over expansion variants.
    """
    rrf_scores: dict[str, float] = {}
    hit_map: dict[str, dict] = {}
    chunk_order: list[str] = []

    for result_idx, (hits, _scores, _query) in enumerate(results):
        weight = weights[result_idx] if weights else 1.0
        seen_in_result: set[str] = set()
        for rank, h in enumerate(hits or []):
            meta = h.get("meta") if isinstance(h.get("meta"), dict) else {}
            chunk_id = str(
                meta.get("chunk_id")
                or h.get("chunk_id")
                or h.get("id")
                or h.get("text", "")[:120]
            )
            if chunk_id in seen_in_result:
                continue
            seen_in_result.add(chunk_id)
            if chunk_id not in hit_map:
                hit_map[chunk_id] = h
                chunk_order.append(chunk_id)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + weight / (20.0 + rank)

    # Sort by RRF score descending
    ordered = sorted(chunk_order, key=lambda cid: rrf_scores.get(cid, 0.0), reverse=True)
    merged_hits = [hit_map[cid] for cid in ordered[:top_k]]
    merged_scores = [rrf_scores.get(cid, 0.0) for cid in ordered[:top_k]]
    return merged_hits, merged_scores


def _search_hits_with_fallback(
    prompt_text: str,
    retriever: BM25Retriever,
    top_k: int,
    settings,
    *,
    allow_translate: bool = True,
    allow_expand: bool = False,
    whole_library: bool = False,
) -> tuple[list[dict], list[float], str, bool, list[str]]:
    """
    Returns: (hits_raw, scores, used_query, used_translation, query_variants)
    """
    q1 = re.split(
        r"(?im)^QUERY SCOPE:\s*(?:Current paper|Research basket|Full library)\.\s*$",
        str(prompt_text or ""),
        maxsplit=1,
    )[0].strip()
    # A library query is allowed to inspect a wider internal candidate window.
    # The answer path still selects only the best evidence-bearing documents, so
    # this improves cross-document recall without exposing low-quality matches.
    candidate_limit = max(10, top_k * (16 if whole_library else 6))
    hits1 = retriever.search(q1, top_k=candidate_limit) if q1 else []
    hits1 = [h for h in (hits1 or []) if not _is_temp_source_path(str((h.get("meta") or {}).get("source_path") or ""))]
    scores1 = [float(h.get("score", 0.0) or 0.0) for h in hits1]
    best1 = float(max(scores1) if scores1 else 0.0)
    query_variants: list[str] = [q1] if q1 else []

    # If the query is CJK-only, BM25 over English corpora can return all-zeros (but still returns arbitrary docs).
    # Try translating to English to get meaningful retrieval.
    used_trans = False
    q2 = _translate_query_for_search(settings, q1) if bool(allow_translate) else None
    if q2:
        hits2 = retriever.search(q2, top_k=candidate_limit)
        hits2 = [h for h in (hits2 or []) if not _is_temp_source_path(str((h.get("meta") or {}).get("source_path") or ""))]
        scores2 = [float(h.get("score", 0.0) or 0.0) for h in hits2]
        best2 = float(max(scores2) if scores2 else 0.0)
        query_variants.append(q2)
        # RRF-merge original and translated results instead of replacing
        # one with the other, so hits found by either query are preserved.
        # Original query (q1) is weighted 2x so it dominates ranking.
        if hits1 and hits2:
            merged_q1q2, merged_s1s2 = _merge_expanded_results(
                [(hits1, scores1, q1), (hits2, scores2, q2)],
                top_k=candidate_limit,
                # For a CJK question over an English academic corpus, the
                # translated query is the primary lexical signal.  Giving the
                # untranslated query priority lets incidental CJK matches bury
                # the actually relevant English papers.
                weights=([0.35, 2.0] if _has_cjk(q1) else [2.0, 1.0]),
            )
            merged_best = float(max(merged_s1s2) if merged_s1s2 else 0.0)
            # RRF and BM25 scores live on different scales, so comparing their
            # maxima can reject useful translated hits whenever the original
            # query has even a weak lexical match. Both result sets are already
            # represented in the fusion; accept the fused ranking directly.
            hits1, scores1, best1 = merged_q1q2, merged_s1s2, merged_best
            used_trans = True
            for _h in hits1:
                _h["_bm25_score"] = _h.get("score", 0.0)
        elif not hits1 and hits2:
            hits1, scores1, best1 = hits2, scores2, best2
            for _h in hits1:
                _h["_bm25_score"] = _h.get("score", 0.0)
            used_trans = True

    # LLM-based query expansion to improve recall for synonym-variant queries.
    deterministic_variants = [
        v
        for v in _deterministic_query_variants(q1)
        if v and v.lower() != q1.lower() and (not q2 or v.lower() != q2.lower())
    ]
    if deterministic_variants:
        all_results: list[tuple[list[dict], list[float], str]] = [(hits1, scores1, q1)]
        for variant in deterministic_variants:
            v_hits = retriever.search(variant, top_k=candidate_limit)
            v_hits = [h for h in (v_hits or []) if not _is_temp_source_path(str((h.get("meta") or {}).get("source_path") or ""))]
            v_scores = [float(h.get("score", 0.0) or 0.0) for h in v_hits]
            all_results.append((v_hits, v_scores, variant))
            if variant not in query_variants:
                query_variants.append(variant)
        merged_hits, merged_scores = _merge_expanded_results(
            all_results,
            top_k=candidate_limit,
            weights=[1.5] + [2.5 for _ in deterministic_variants],
        )
        for _h in merged_hits:
            _h["_expansion_variants"] = list(deterministic_variants)
        for _h in merged_hits:
            _h["_bm25_score"] = _h.get("score", 0.0)
        hits1, scores1, best1 = merged_hits, merged_scores, float(max(merged_scores) if merged_scores else best1)

    # LLM-based query expansion to improve recall for synonym-variant queries.
    if (
        bool(allow_expand)
        and not deterministic_variants
        and q1
        and getattr(settings, "api_key", None)
        and getattr(settings, "query_expansion_enabled", False)
    ):
        expanded = _expand_query_via_llm(settings, q1)
        # expanded[0] is q1; skip it since we already searched q1
        new_variants = [v for v in expanded[1:] if v and v.lower() != q1.lower() and (not q2 or v.lower() != q2.lower())]
        if new_variants:
            all_results: list[tuple[list[dict], list[float], str]] = [(hits1, scores1, q1)]
            if q2 and not used_trans:
                all_results.append((hits2, scores2, q2))
            for variant in new_variants:
                v_hits = retriever.search(variant, top_k=candidate_limit)
                v_hits = [h for h in (v_hits or []) if not _is_temp_source_path(str((h.get("meta") or {}).get("source_path") or ""))]
                v_scores = [float(h.get("score", 0.0) or 0.0) for h in v_hits]
                all_results.append((v_hits, v_scores, variant))
                query_variants.append(variant)

            # RRF merge all result sets (always use merged results — RRF handles
            # low-quality expansions by weighting at rank depth, and the original
            # query results are included in the pool so nothing is lost).
            # Original query (q1) is weighted 2x, all other variants 1x.
            merged_hits, merged_scores = _merge_expanded_results(
                all_results,
                top_k=candidate_limit,
                weights=[2.0 if i == 0 else 1.0 for i in range(len(all_results))],
            )
            # Preserve original BM25 score so downstream sorters can use it.
            for _h in merged_hits:
                _h["_bm25_score"] = _h.get("score", 0.0)
            # Tag hits with the variant list so the focus filter can be more
            # lenient for expansion-discovered papers.
            _expansion_variants = [v for v in query_variants if v.lower() != q1.lower()]
            if _expansion_variants:
                for _h in merged_hits:
                    _h["_expansion_variants"] = list(_expansion_variants)
            return merged_hits, merged_scores, q1, used_trans, query_variants

    # Last-resort fallback: if all queries returned empty, try a broad search
    # with the original query to avoid starving downstream rendering.
    if not hits1 and q1:
        hits_fb = retriever.search(q1, top_k=max(20, top_k * 20))
        hits_fb = [h for h in (hits_fb or []) if not _is_temp_source_path(str((h.get("meta") or {}).get("source_path") or ""))]
        if hits_fb:
            scores_fb = [float(h.get("score", 0.0) or 0.0) for h in hits_fb]
            return hits_fb, scores_fb, q1, used_trans, query_variants

    return hits1, scores1, q1, used_trans, query_variants


_READING_ROADMAP_QUERY_RE = re.compile(
    r"(?:刚开始|入门|主线|路线|先读|阅读顺序|哪几篇|"
    r"\b(?:beginner|getting started|roadmap|reading order|read first|which papers?)\b)",
    flags=re.I,
)
_FOUNDATIONAL_SOURCE_RE = re.compile(
    r"(?:review|survey|overview|tutorial|principles?|prospects?|foundations?|"
    r"advances?[- _]and[- _]challenges?)",
    flags=re.I,
)
_COMPARATIVE_SOURCE_RE = re.compile(
    r"(?:\bversus\b|\bvs\.?\b|comparison|comparative|benchmark)",
    flags=re.I,
)


def _reading_roadmap_source_role_bonus(prompt_text: str, source_path: str) -> float:
    """Favor complementary literature roles for a beginner reading route."""

    if not _READING_ROADMAP_QUERY_RE.search(str(prompt_text or "")):
        return 0.0
    source_surface = Path(str(source_path or "")).stem.replace("_", " ").replace("-", " ")
    if _FOUNDATIONAL_SOURCE_RE.search(source_surface):
        return 18.0
    if _COMPARATIVE_SOURCE_RE.search(source_surface):
        return 14.0
    return 0.0

def _group_hits_by_doc_for_refs(
    hits_raw: list[dict],
    prompt_text: str,
    top_k_docs: int,
    *,
    deep_query: str = "",
    deep_read: bool = False,
    llm_rerank: bool = False,
    settings=None,
) -> list[dict]:
    """
    Merge hits from the same markdown doc into a single ref entry.
    """
    by_doc: dict[str, list[dict]] = {}
    for h in hits_raw or []:
        meta = h.get("meta", {}) or {}
        src = (meta.get("source_path") or "").strip()
        if (not src) or _is_temp_source_path(src):
            continue
        by_doc.setdefault(src, []).append(h)

    # Pre-sort docs by best lexical hit; later stages can override with deep-read/LLM semantics.
    doc_order: list[tuple[float, str]] = []
    doc_hint_scores: dict[str, float] = {}
    doc_focus_scores: dict[str, float] = {}
    doc_direct_scores: dict[str, float] = {}
    doc_direct_terms: dict[str, tuple[str, ...]] = {}
    doc_roadmap_role_scores: dict[str, float] = {}
    anchor_hint = _extract_explicit_anchor_hint(prompt_text or deep_query or "")
    profile = _query_term_profile(prompt_text, deep_query or "")
    reading_roadmap_query = bool(
        _READING_ROADMAP_QUERY_RE.search(str(prompt_text or deep_query or ""))
    )
    _bm25_scores: list[float] = []
    for src, hs in by_doc.items():
        try:
            best_score = float(max(float(h.get("score", 0.0) or 0.0) for h in hs))
        except Exception:
            best_score = 0.0
        _bm25_scores.append(best_score)
        doc_hint_score = _source_prompt_match_score(prompt_text or deep_query or "", src)
        doc_focus_score = _doc_focus_match_score(
            prompt_text=(prompt_text or deep_query or ""),
            source_path=src,
            snippets=[str((h.get("text") or "")).strip() for h in hs[:6] if str((h.get("text") or "")).strip()],
            headings=[
                str(((h.get("meta", {}) or {}).get("heading_path") or (h.get("meta", {}) or {}).get("top_heading") or "")).strip()
                for h in hs[:6]
                if isinstance(h.get("meta"), dict)
            ],
        )
        direct_score, direct_terms = _direct_prompt_match_score(
            prompt_text=(prompt_text or deep_query or ""),
            source_path=src,
            snippets=[str((h.get("text") or "")).strip() for h in hs[:6] if str((h.get("text") or "")).strip()],
            headings=[
                str(((h.get("meta", {}) or {}).get("heading_path") or (h.get("meta", {}) or {}).get("top_heading") or "")).strip()
                for h in hs[:6]
                if isinstance(h.get("meta"), dict)
            ],
        )
        doc_hint_scores[src] = float(doc_hint_score)
        doc_focus_scores[src] = float(doc_focus_score)
        doc_direct_scores[src] = float(direct_score)
        doc_direct_terms[src] = tuple(direct_terms or ())
        pre_term_bonus = _doc_term_bonus(
            profile,
            Path(src).name,
            [str((h.get("text") or "")).strip() for h in hs[:6] if str((h.get("text") or "")).strip()],
        )
        roadmap_role_bonus = _reading_roadmap_source_role_bonus(
            prompt_text or deep_query or "",
            src,
        )
        doc_roadmap_role_scores[src] = float(roadmap_role_bonus)
        # Candidate preselection must honor topic qualifiers too. Otherwise a
        # wider full-library pool can fill the bounded document scorer with
        # generic high-BM25 papers before relevant qualified papers are scored.
        doc_order.append(
            (
                best_score
                + (1.6 * doc_hint_score)
                + (1.05 * doc_focus_score)
                + (1.8 * direct_score)
                + (4.0 * pre_term_bonus)
                + roadmap_role_bonus,
                src,
            )
        )
    doc_order.sort(key=lambda x: x[0], reverse=True)

    docs: list[dict] = []
    nav_question = (prompt_text or deep_query or "").strip()
    # Normalize BM25 scores across all candidate docs (0-1 range).
    _bm25_global_max = max(_bm25_scores) if _bm25_scores else 1.0
    if _bm25_global_max <= 0:
        _bm25_global_max = 1.0
    # Bound work: only consider a limited number of candidate docs.
    max_docs_consider = max(int(top_k_docs) * 2, 12)
    # Quality-first refs: if deep_read is enabled, expand more candidate docs than before.
    deep_expand_docs = min(max_docs_consider, max(int(top_k_docs) * 2, 6)) if deep_read else 0
    for _best, src in doc_order[:max_docs_consider]:
        hs = by_doc.get(src) or []
        hs2 = sorted(hs, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        expansion_variants = list(
            dict.fromkeys(
                str(variant or "").strip()
                for hit in hs2[:6]
                for variant in list(hit.get("_expansion_variants") or [])
                if str(variant or "").strip()
            )
        )
        expansion_focus_query = " ".join(expansion_variants[:3]).strip()
        primary_table_hit = next(
            (
                h
                for h in hs2[:1]
                if str(((h.get("meta") or {}).get("structured_kind") or "")).strip().lower()
                in {"table_metric", "table_row"}
                and str(h.get("text") or "").strip()
            ),
            None,
        )
        primary_table_text = str((primary_table_hit or {}).get("text") or "").strip()
        primary_table_meta = (
            dict((primary_table_hit or {}).get("meta") or {})
            if isinstance((primary_table_hit or {}).get("meta"), dict)
            else {}
        )
        best_score = float(hs2[0].get("score", 0.0) or 0.0) if hs2 else 0.0
        doc_hint_score = float(doc_hint_scores.get(src, 0.0) or 0.0)
        doc_focus_score = float(doc_focus_scores.get(src, 0.0) or 0.0)
        direct_score = float(doc_direct_scores.get(src, 0.0) or 0.0)
        direct_terms = tuple(doc_direct_terms.get(src) or ())
        roadmap_role_bonus = float(doc_roadmap_role_scores.get(src, 0.0) or 0.0)
        force_anchor_focus = bool(anchor_hint) and (doc_hint_score >= 6.0)
        anchor_focus_query = (
            _build_doc_anchor_focus_query(prompt_text or deep_query or "", src, anchor_hint)
            if force_anchor_focus
            else ""
        )
        # Candidate headings: (score, top_heading)
        cand: list[tuple[float, str]] = []
        snippets: list[str] = []
        snippet_anchor_bonus: dict[str, float] = {}
        locs_full: list[dict] = []
        for h in hs2[:6]:
            meta = h.get("meta", {}) or {}
            sc_h = float(h.get("score", 0.0) or 0.0)
            top = (meta.get("top_heading") or _top_heading(meta.get("heading_path", "")) or "").strip()
            anchor_bonus = 0.0
            if force_anchor_focus:
                anchor_bonus = _anchor_text_bonus(
                    "\n".join(
                        x
                        for x in [
                            str(meta.get("heading_path") or "").strip(),
                            str(h.get("text") or "").strip(),
                        ]
                        if x
                    ),
                    anchor_hint,
                )
            if top and (not _is_non_navigational_heading(top, question=nav_question, source_path=src)):
                hp_raw = _normalize_heading_path_for_display(str(meta.get("heading_path") or ""))
                hp = _sanitize_heading_path_for_navigation(hp_raw or top, question=nav_question, source_path=src)
                sc_adj = sc_h + _heading_intent_bonus_for_question(hp or top, nav_question) + (2.0 * anchor_bonus)
                cand.append((sc_adj, top))
                if hp:
                    p0, p1 = _page_range_from_meta(meta)
                    top_h, _sub_h = _split_heading_path_levels(hp)
                    locs_full.append(
                        {
                            "heading_path": hp,
                            "heading": top_h or (_normalize_heading(top) or top),
                            "score": sc_h,
                            "score_adj": sc_adj,
                            "page_start": p0,
                            "page_end": p1,
                            "source": "hit",
                        }
                    )
            t = (h.get("text") or "").strip()
            if t:
                if _should_skip_reference_like_snippet(
                    t,
                    heading_path=str(meta.get("heading_path") or top or ""),
                    question=nav_question,
                    source_path=src,
                ) and not (force_anchor_focus and anchor_bonus > 0.0):
                    t = ""
            if t:
                if force_anchor_focus and anchor_bonus > 0.0:
                    snippet_anchor_bonus[t] = max(float(snippet_anchor_bonus.get(t, 0.0) or 0.0), float(anchor_bonus))
                if t not in snippets:
                    snippets.append(t)

        # Optional deep-read for better section targeting + aspects + ranking.
        deep_best = 0.0
        use_expansion_deep_read = bool(expansion_focus_query and doc_hint_score >= 6.0)
        expansion_augmented_query = " ".join(
            part
            for part in [
                str(prompt_text or deep_query or "").strip(),
                expansion_focus_query if use_expansion_deep_read else "",
            ]
            if part
        ).strip()
        do_deep_read = (
            (deep_read and deep_query and (len(docs) < deep_expand_docs))
            or force_anchor_focus
            or use_expansion_deep_read
        )
        if do_deep_read:
            read_query = (
                anchor_focus_query
                or expansion_augmented_query
                or deep_query
                or prompt_text
                or ""
            ).strip()
            anchor_extra: list[dict] = []
            if force_anchor_focus:
                try:
                    anchor_extra = _find_anchor_snippets_in_md(
                        Path(src),
                        anchor_hint,
                        max_snippets=3,
                        snippet_chars=900,
                    )
                except Exception:
                    anchor_extra = []
            try:
                deep_extra = _deep_read_md_for_context(
                    Path(src),
                    read_query,
                    max_snippets=(5 if force_anchor_focus else 3),
                    snippet_chars=1600,
                )
            except Exception:
                deep_extra = []
            extra = list(anchor_extra or [])
            seen_extra_text = {str(x.get("text") or "").strip() for x in extra if str(x.get("text") or "").strip()}
            for ex in deep_extra or []:
                tx0 = str(ex.get("text") or "").strip()
                if tx0 and (tx0 in seen_extra_text):
                    continue
                if tx0:
                    seen_extra_text.add(tx0)
                extra.append(ex)
            for ex in extra or []:
                meta_ex = ex.get("meta", {}) or {}
                sc_ex = float(ex.get("score", 0.0) or 0.0)
                anchor_bonus_ex = (
                    _anchor_text_bonus(
                        "\n".join(
                            x
                            for x in [
                                str(meta_ex.get("heading_path") or "").strip(),
                                str(ex.get("text") or "").strip(),
                            ]
                            if x
                        ),
                        anchor_hint,
                    )
                    if force_anchor_focus
                    else 0.0
                )
                if force_anchor_focus and bool(meta_ex.get("anchor_read")) and anchor_bonus_ex > 0.0:
                    anchor_bonus_ex += 10.0
                hp2_raw = str(meta_ex.get("heading_path", "") or "").strip()
                hp2 = _sanitize_heading_path_for_navigation(
                    _normalize_heading_path_for_display(hp2_raw),
                    question=nav_question,
                    source_path=src,
                )
                top2 = _top_heading(hp2 or hp2_raw)
                if top2 and (not _is_non_navigational_heading(top2, question=nav_question, source_path=src)):
                    sc2_raw = sc_ex + 0.2 + (0.35 * anchor_bonus_ex)
                    sc2_adj = sc2_raw + _heading_intent_bonus_for_question(hp2 or top2, nav_question)
                    cand.append((sc2_adj, top2))
                    if hp2:
                        p0, p1 = _page_range_from_meta(meta_ex)
                        locs_full.append(
                            {
                                "heading_path": hp2,
                                "heading": _normalize_heading(top2) or top2,
                                "score": sc2_raw,
                                "score_adj": sc2_adj,
                                "page_start": p0,
                                "page_end": p1,
                                "source": "deep",
                            }
                        )
                tx = (ex.get("text") or "").strip()
                if tx:
                    if _should_skip_reference_like_snippet(
                        tx,
                        heading_path=hp2 or hp2_raw,
                        question=nav_question,
                        source_path=src,
                    ) and not (force_anchor_focus and anchor_bonus_ex > 0.0):
                        tx = ""
                if tx:
                    if force_anchor_focus and anchor_bonus_ex > 0.0:
                        snippet_anchor_bonus[tx] = max(float(snippet_anchor_bonus.get(tx, 0.0) or 0.0), float(anchor_bonus_ex))
                    if tx not in snippets:
                        snippets.append(tx)
                try:
                    deep_best = max(deep_best, float(ex.get("score", 0.0) or 0.0) + (2.0 * anchor_bonus_ex))
                except Exception:
                    pass

        # Final heading MUST be a real heading from this doc.
        # Prefer headings already attached to hits (grounded), but also read real md headings for navigation.
        best_heading = _pick_best_heading_for_doc(cand, prompt_text)
        headings_for_pack: list[str] = []
        try:
            prefer = _preferred_section_keys(prompt_text)
            picked = _pick_heading_from_md(
                Path(src),
                anchor_focus_query or deep_query or prompt_text,
                prefer=prefer,
                source_path=src,
            )
            if picked:
                best_heading = picked
        except Exception:
            pass
        try:
            hs_raw = _extract_md_headings(Path(src), max_n=40)
            hs_keep: list[str] = []
            seen_hs: set[str] = set()
            for hh in hs_raw:
                hp_h = _sanitize_heading_path_for_navigation(hh, question=nav_question, source_path=src)
                top_hh, _sub_hh = _split_heading_path_levels(hp_h)
                hh2 = _normalize_heading(top_hh or hp_h)
                if not hh2:
                    continue
                if _is_non_navigational_heading(hh2, question=nav_question, source_path=src):
                    continue
                key_h = hh2.lower()
                if key_h in seen_hs:
                    continue
                seen_hs.add(key_h)
                hs_keep.append(hh2)
            headings_for_pack = hs_keep
        except Exception:
            headings_for_pack = []
        if not headings_for_pack:
            # Minimal heading set: only what we have seen from hits/deep-read.
            seen_h2: list[str] = []
            for _sc2, hh in sorted(cand, key=lambda x: x[0], reverse=True):
                hh2 = _normalize_heading(hh)
                if (
                    (not hh2)
                    or _is_non_navigational_heading(hh2, question=nav_question, source_path=src)
                    or hh2 in seen_h2
                ):
                    continue
                seen_h2.append(hh2)
                if len(seen_h2) >= 22:
                    break
            headings_for_pack = seen_h2
        aspects = _aspects_from_snippets(snippets[:3], prompt_text)
        try:
            overview_snips = _collect_doc_overview_snippets(Path(src), max_n=3, snippet_chars=360)
        except Exception:
            overview_snips = []
        if not overview_snips:
            overview_snips = [
                _clean_snippet_for_display(s, max_chars=360)
                for s in snippets[:2]
                if str(s or "").strip()
            ]

        # Build display snippets: pick the most relevant, non-noise snippets.
        q_for_pick = (
            anchor_focus_query
            or expansion_augmented_query
            or deep_query
            or prompt_text
            or ""
        ).strip()
        q_tokens = [t for t in tokenize(q_for_pick) if len(t) >= 3]
        scored_snips: list[tuple[float, str]] = []
        for s in snippets:
            s2 = (s or "").strip()
            if not s2:
                continue
            anchor_snip_bonus = float(snippet_anchor_bonus.get(s2, 0.0) or 0.0)
            if _is_noise_snippet_text(s2) and (anchor_snip_bonus <= 0.0):
                continue
            try:
                sc = _score_tokens(s2, q_tokens) if q_tokens else 0.0
            except Exception:
                sc = 0.0
            if force_anchor_focus:
                sc += anchor_snip_bonus
            # Prefer snippets that literally contain key phrases for single-shot/single-pixel disambiguation.
            low = _norm_text_for_match(s2)
            if profile.get("wants_single_shot") and any(k in low for k in ["single-shot", "single shot", "single exposure", "snapshot"]):
                sc += 3.0
            if profile.get("wants_single_shot") and any(k in low for k in ["single-pixel", "single pixel"]):
                sc -= 3.0
            scored_snips.append((float(sc), s2))
        scored_snips.sort(key=lambda x: x[0], reverse=True)
        show_snips = [_clean_snippet_for_display(s, max_chars=900) for _, s in scored_snips[:2]]
        show_snips = [s for s in show_snips if str(s or "").strip()]
        if primary_table_text:
            # A document group can contain many tables that share the same
            # dataset and metric tokens.  The raw retriever has already chosen
            # the best structured table hit using table type, metric label and
            # method-vs-ablation intent.  Do not discard that decision here by
            # re-ranking only on token density (which favors shorter ablation
            # rows over the complete benchmark series).
            primary_table_show = _clean_snippet_for_display(primary_table_text, max_chars=900)
            if primary_table_show:
                show_snips = [primary_table_show]
        if force_anchor_focus:
            anchored_raw = [
                s
                for s in sorted(
                    snippets,
                    key=lambda x: float(snippet_anchor_bonus.get(x, 0.0) or 0.0),
                    reverse=True,
                )
                if float(snippet_anchor_bonus.get(s, 0.0) or 0.0) > 0.0
            ]
            anchored = [_clean_snippet_for_display(s, max_chars=900) for s in anchored_raw]
            anchored = [s for s in anchored if str(s or "").strip()]
            if anchored:
                rest = [s for s in show_snips if s not in anchored]
                show_snips = (anchored + rest)[:2]

        # Best location candidates (prefer deep-read heading_path with subsection detail)
        locs_full.sort(
            key=lambda x: (
                float(x.get("score_adj", x.get("score", 0.0)) or 0.0),
                float(x.get("score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        locs2_good: list[dict] = []
        locs2_fallback: list[dict] = []
        seen_h = set()
        for loc in locs_full:
            hh_path = _sanitize_heading_path_for_navigation(
                str(loc.get("heading_path") or ""),
                question=nav_question,
                source_path=src,
            )
            if not hh_path:
                hh_path = _sanitize_heading_path_for_navigation(
                    str(loc.get("heading") or ""),
                    question=nav_question,
                    source_path=src,
                )
            if not hh_path:
                continue
            top_h, sub_h = _split_heading_path_levels(hh_path)
            hh_key = hh_path.lower()
            if not top_h or _is_non_navigational_heading(top_h, question=nav_question, source_path=src) or hh_key in seen_h:
                continue
            if _should_avoid_discussion_for_question(nav_question) and _is_discussion_or_conclusion_heading(top_h):
                continue
            if sub_h and _is_non_navigational_heading(sub_h, question=nav_question, source_path=src):
                hh_path = top_h
                sub_h = ""
                hh_key = hh_path.lower()
                if hh_key in seen_h:
                    continue
            seen_h.add(hh_key)
            is_low_quality = _is_low_quality_navigation_heading(top_h, question=nav_question, source_path=src)
            ent = {
                "heading": top_h,
                "heading_path": hh_path,
                "score": float(loc.get("score_adj", loc.get("score", 0.0)) or 0.0),
                "source": str(loc.get("source") or ""),
                "quality": ("low" if is_low_quality else "high"),
            }
            if sub_h:
                ent["subsection"] = sub_h
            try:
                p0 = int(loc.get("page_start")) if loc.get("page_start") is not None else None
            except Exception:
                p0 = None
            try:
                p1 = int(loc.get("page_end")) if loc.get("page_end") is not None else None
            except Exception:
                p1 = None
            if p0 is not None and p0 > 0:
                ent["page_start"] = p0
            if p1 is not None and p1 > 0:
                ent["page_end"] = p1
            if is_low_quality:
                locs2_fallback.append(ent)
            else:
                locs2_good.append(ent)
            if (len(locs2_good) + len(locs2_fallback)) >= 6:
                break
        locs2 = (locs2_good + locs2_fallback)[:3]

        # Heuristic base score: BM25 + small deep-read signal + term mismatch penalties.
        doc_name = Path(src).name
        term_bonus = _doc_term_bonus(profile, doc_name, snippets[:3])
        deep_scaled = 1.6 * (deep_best ** 0.6) if deep_best > 0 else 0.0
        anchor_best = max((float(v or 0.0) for v in snippet_anchor_bonus.values()), default=0.0)
        # Normalize BM25 to 0-1 range; cap anchor bonus so it can't dominate.
        norm_bm25 = best_score / _bm25_global_max if _bm25_global_max > 0 else 0.0
        anchor_capped = min(anchor_best, 20.0)
        combined = (
            (5.0 * norm_bm25)  # BM25 contribution scaled to ~0-5
            + (0.25 * deep_scaled)
            + term_bonus
            + (1.5 * doc_hint_score)
            + (1.15 * doc_focus_score)
            + (1.35 * direct_score)
            + (0.35 * anchor_capped)
            + roadmap_role_bonus
        )

        meta_out = {"source_path": src}
        src_sha1 = _file_sha1_cached(Path(src))
        if src_sha1:
            meta_out["source_sha1"] = src_sha1
        if best_heading and (not _is_low_quality_navigation_heading(best_heading, question=nav_question, source_path=src)):
            meta_out["top_heading"] = best_heading
        meta_out["ref_aspects"] = aspects
        anchor_primary_text = ""
        if force_anchor_focus:
            snippets_sorted = sorted(snippets, key=lambda s: float(snippet_anchor_bonus.get(s, 0.0) or 0.0), reverse=True)
            meta_out["ref_snippets"] = snippets_sorted[:3]
            anchor_primary_text = next(
                (str(s or "").strip() for s in snippets_sorted if float(snippet_anchor_bonus.get(s, 0.0) or 0.0) > 0.0 and str(s or "").strip()),
                "",
            )
        else:
            meta_out["ref_snippets"] = (
                show_snips[:3] if use_expansion_deep_read and show_snips else snippets[:3]
            )
        if primary_table_text:
            # Keep the card, locator and async reference summary grounded on
            # the same table series used by the answer.  Competing tables from
            # the same paper are intentionally excluded from this one-card
            # evidence surface.
            meta_out["ref_snippets"] = [primary_table_text]
        meta_out["ref_show_snippets"] = show_snips[:3]
        meta_out["ref_overview_snippets"] = [x for x in overview_snips if str(x or "").strip()][:3]
        meta_out["ref_locs"] = locs2
        meta_out["ref_headings"] = headings_for_pack
        if doc_hint_score > 0.0:
            meta_out["explicit_doc_match_score"] = float(doc_hint_score)
        if direct_score > 0.0:
            meta_out["direct_prompt_match_score"] = float(direct_score)
            meta_out["direct_prompt_match_terms"] = list(direct_terms)
        if roadmap_role_bonus > 0.0:
            meta_out["reading_roadmap_role_score"] = float(roadmap_role_bonus)
        if primary_table_text:
            for key in (
                "structured_kind",
                "table_index",
                "table_number",
                "table_metric",
                "table_metric_label",
                "table_metric_direction",
                "table_subject_label",
                "table_subject_kind",
                "block_id",
                "table_block_id",
                "anchor_id",
                "line_start",
                "line_end",
            ):
                value = primary_table_meta.get(key)
                if value not in (None, "", 0):
                    meta_out[key] = value
            meta_out["structured_evidence_locked"] = True
            primary_heading = str(primary_table_meta.get("heading_path") or "").strip()
            if primary_heading:
                meta_out["heading_path"] = primary_heading
            primary_page_start, primary_page_end = _page_range_from_meta(primary_table_meta)
            if primary_page_start is not None and primary_page_start > 0:
                meta_out["page_start"] = int(primary_page_start)
                meta_out["page_end"] = int(primary_page_end or primary_page_start)
        if force_anchor_focus and anchor_hint:
            meta_out["anchor_target_kind"] = str(anchor_hint.get("kind") or "")
            meta_out["anchor_target_number"] = int(anchor_hint.get("number") or 0)
            if str(anchor_hint.get("number_text") or "").strip():
                meta_out["anchor_target_label"] = str(anchor_hint.get("number_text") or "").strip()
            if str(anchor_hint.get("kind") or "").strip().lower() == "figure":
                target_scope = normalize_figure_scope(anchor_hint.get("figure_scope"))
                target_key = str(anchor_hint.get("figure_key") or figure_key_for_scope(target_scope, int(anchor_hint.get("number") or 0))).strip()
                if target_scope:
                    meta_out["anchor_target_scope"] = target_scope
                    meta_out["figure_scope"] = target_scope
                if target_key:
                    meta_out["anchor_target_key"] = target_key
                    meta_out["figure_key"] = target_key
            meta_out["anchor_match_score"] = float(anchor_best)
        if locs2:
            loc0 = locs2[0]
            hp0 = str(loc0.get("heading_path") or "").strip()
            sec0, sub0 = _split_heading_path_levels(hp0 or str(loc0.get("heading") or ""))
            q0 = str(loc0.get("quality") or "").strip().lower()
            high_quality_loc = (q0 != "low") and (not _is_low_quality_navigation_heading(sec0, question=nav_question, source_path=src))
            meta_out["ref_loc_quality"] = ("high" if high_quality_loc else "low")
            if high_quality_loc:
                if hp0:
                    meta_out["ref_best_heading_path"] = hp0
                if sec0:
                    meta_out["ref_section"] = sec0
                if sub0 and (not _is_non_navigational_heading(sub0, question=nav_question, source_path=src)):
                    meta_out["ref_subsection"] = sub0
            if loc0.get("page_start") is not None:
                meta_out["page_start"] = int(loc0.get("page_start"))
            if loc0.get("page_end") is not None:
                meta_out["page_end"] = int(loc0.get("page_end"))
        else:
            meta_out["ref_loc_quality"] = "none"
        meta_out["ref_rank"] = {
            "bm25": best_score,
            "deep": deep_best,
            "term_bonus": term_bonus,
            "focus_bonus": doc_focus_score,
            "direct_prompt": direct_score,
            "reading_roadmap_role": roadmap_role_bonus,
            "llm": 0.0,
            "why": "",
            "score": combined,
            "display_score": combined,
            "semantic_score": 0.0,
        }

        docs.append(
            {
                "score": float(combined),
                "id": f"doc:{hashlib.sha1(src.encode('utf-8','ignore')).hexdigest()[:12]}",
                "text": (
                    anchor_primary_text
                    if anchor_primary_text
                    else (
                        show_snips[0]
                        if show_snips
                        else (snippets[0] if snippets else "")
                    )
                ),
                "meta": meta_out,
            }
        )

    # Optional LLM pack: one-shot semantic rerank + strong directional one-liner pieces (grounded on snippets/headings).
    if llm_rerank and settings and docs:
        pack = _llm_refs_pack(settings, question=(prompt_text or deep_query or ""), docs=docs)
        if isinstance(pack, dict) and pack:
            docs = _apply_llm_pack_to_grouped_docs(docs, pack=pack, question=nav_question)

    docs.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

    # Compound topic requests should not be padded with papers that match only
    # one half of the topic (for example, generic deep learning or conventional
    # SPI). Preserve the broad pool when no qualified document exists so a
    # sparse library can still return a clearly bounded fallback.
    if (
        profile.get("wants_deep_learning")
        and profile.get("wants_single_pixel")
        and not reading_roadmap_query
    ):
        qualified_docs = [
            doc
            for doc in docs
            if float((((doc.get("meta") or {}).get("ref_rank") or {}).get("term_bonus") or 0.0)) >= 3.0
        ]
        if qualified_docs:
            docs = qualified_docs

    # Precision-first filtering: when semantic rerank is available, drop weakly related docs
    # instead of filling the list with lexical look-alikes.
    if llm_rerank and docs:
        docs = _semantic_filter_docs_by_llm(docs)

    return docs[: max(1, int(top_k_docs))]

def _group_hits_by_doc_for_refs_fast(hits_raw: list[dict], top_k_docs: int) -> list[dict]:
    """
    Fast fallback for background QA worker:
    - no full-md deep read
    - no LLM rerank
    - no heading extraction from file
    """
    by_doc: dict[str, dict] = {}
    for h in hits_raw or []:
        meta = h.get("meta", {}) or {}
        src = (meta.get("source_path") or "").strip()
        if (not src) or _is_temp_source_path(src):
            continue
        cur = by_doc.get(src)
        sc = float(h.get("score", 0.0) or 0.0)
        top = (meta.get("top_heading") or _top_heading(meta.get("heading_path", "")) or "").strip()
        if top and (_is_reference_heading_like(top) or _is_venue_heading_like(top)):
            top = ""
        txt = (h.get("text") or "").strip()
        src_sha1_fast = _file_sha1_cached(Path(src))
        if (cur is None) or (sc > float(cur.get("score", 0.0) or 0.0)):
            meta_fast = {
                "source_path": src,
                "top_heading": ("" if _is_probably_bad_heading(top) else top),
                "ref_snippets": [txt] if txt else [],
                "ref_show_snippets": [_clean_snippet_for_display(txt, max_chars=900)] if txt else [],
                "ref_overview_snippets": [_clean_snippet_for_display(txt, max_chars=360)] if txt else [],
                "ref_locs": ([{"heading": top, "score": sc}] if top and (not _is_probably_bad_heading(top)) else []),
                "ref_headings": ([top] if top and (not _is_probably_bad_heading(top)) else []),
                "ref_aspects": [],
                "ref_rank": {"bm25": sc, "deep": 0.0, "term_bonus": 0.0, "llm": 0.0, "why": "", "score": sc},
            }
            if src_sha1_fast:
                meta_fast["source_sha1"] = src_sha1_fast
            by_doc[src] = {
                "score": sc,
                "id": f"doc:{hashlib.sha1(src.encode('utf-8','ignore')).hexdigest()[:12]}",
                "text": txt,
                "meta": meta_fast,
            }
        elif txt:
            m = by_doc[src].get("meta") or {}
            arr = list(m.get("ref_snippets") or [])
            if txt not in arr and len(arr) < 2:
                arr.append(txt)
                m["ref_snippets"] = arr
                m["ref_show_snippets"] = [_clean_snippet_for_display(x, max_chars=900) for x in arr]
                m["ref_overview_snippets"] = [_clean_snippet_for_display(x, max_chars=360) for x in arr[:2]]
                by_doc[src]["meta"] = m
    docs = list(by_doc.values())
    docs.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    return docs[: max(1, int(top_k_docs))]

def _read_text_cached(path: Path) -> str:
    p = Path(path)
    try:
        mtime = float(p.stat().st_mtime)
    except Exception:
        mtime = 0.0
    key = f"{str(p)}|{mtime}"
    v0 = _cache_get("file_text", key)
    if isinstance(v0, str):
        return v0
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        text = ""
    _cache_set("file_text", key, text, max_items=220)
    return text


def _file_sha1_cached(path: Path) -> str:
    p = Path(path)
    try:
        st = p.stat()
        key = f"{str(p)}|{int(st.st_mtime)}|{int(st.st_size)}"
    except Exception:
        key = str(p)
    v0 = _cache_get("file_sha1", key)
    if isinstance(v0, str):
        return v0
    try:
        if p.exists() and p.is_file():
            out = str(compute_file_sha1(p) or "").strip().lower()
        else:
            out = ""
    except Exception:
        out = ""
    _cache_set("file_sha1", key, out, max_items=300)
    return out

def _extract_md_headings(md_path: Path, *, max_n: int = 80) -> list[str]:
    """
    Extract real headings from the markdown file (ground truth for navigation).
    Returns plain heading titles (without leading #), preserving numbering if present.
    """
    md_path = Path(md_path)
    if not md_path.exists():
        return []
    text = _read_text_cached(md_path)
    if not text:
        return []
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("#"):
            continue
        level = len(s) - len(s.lstrip("#"))
        if level <= 0 or level > 4:
            continue
        title = s[level:].strip()
        title = _normalize_heading(title)
        if not title or _is_probably_bad_heading(title):
            continue
        if title not in out:
            out.append(title)
        if len(out) >= max_n:
            break
    return out

def _pick_heading_from_md(md_path: Path, query: str, *, prefer: list[str], source_path: str = "") -> str:
    """
    Pick a heading that most likely contains the answer, from real md headings only.
    """
    hs = _extract_md_headings(md_path)
    if not hs:
        return ""
    q = (query or "").strip()
    if not q:
        return hs[0]
    q_toks = [t for t in tokenize(q) if len(t) >= 3]

    def score(h: str) -> float:
        low = h.lower()
        if _is_non_navigational_heading(h, question=q, source_path=source_path):
            return -1e6
        if _should_avoid_discussion_for_question(q) and _is_discussion_or_conclusion_heading(h):
            return -1000.0
        # Keyword overlap
        base = 0.0
        if q_toks:
            ht = tokenize(h)
            ct = Counter(ht)
            base += float(sum(ct.get(t, 0) for t in q_toks))
        base += _heading_intent_bonus_for_question(h, q)
        # Preference boost (method/results/intro etc)
        bonus = 0.0
        for i, k in enumerate(prefer):
            if k in low:
                bonus += 3.0 - i * 0.25
                break
        if _looks_like_doc_title_heading(h, source_path):
            bonus -= 2.2
        # Slightly prefer medium-length headings
        bonus += max(0.0, (50 - abs(len(h) - 38)) / 200.0)
        return base + bonus

    best = max(hs, key=score)
    # Avoid pointing to REFERENCES unless asked.
    wants_refs = bool(re.search(r"(参考文献|引用|cite|citation|reference)", q, flags=re.I))
    if _is_non_navigational_heading(best, question=q, source_path=source_path) and not wants_refs:
        for h in hs:
            if _is_non_navigational_heading(h, question=q, source_path=source_path):
                continue
            if not _is_probably_bad_heading(h):
                return h
        return ""
    return best

def _deep_read_md_for_context(md_path: Path, query: str, *, max_snippets: int = 3, snippet_chars: int = 1400) -> list[dict]:
    """
    Read the full .md and extract the most relevant snippets (by token overlap),
    then return in the same dict shape as retriever hits.
    """
    md_path = Path(md_path)
    if not md_path.exists():
        return []
    text = _read_text_cached(md_path)
    if not text.strip():
        return []

    q_tokens = [t for t in tokenize(query or "") if len(t) >= 3]
    if not q_tokens:
        return []

    # Cache per (file mtime, query) to avoid repeated full-doc scans across reruns / background tasks.
    try:
        mtime = float(md_path.stat().st_mtime)
    except Exception:
        mtime = 0.0
    cache_key = hashlib.sha1((str(md_path) + "|" + str(mtime) + "|" + (query or "")).encode("utf-8", "ignore")).hexdigest()[:16]
    v0 = _cache_get("deep_read", cache_key)
    if isinstance(v0, list):
        try:
            return list(v0)
        except Exception:
            return []

    chunks = chunk_markdown(text, source_path=str(md_path), chunk_size=900, overlap=0)
    scored: list[tuple[float, dict]] = []
    for c in chunks:
        body = (c.get("text") or "").strip()
        if len(body) < 80:
            continue
        s = _score_tokens(body, q_tokens)
        if s <= 0.0:
            continue
        scored.append((s, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict] = []
    for rank, (s, c) in enumerate(scored[: max(1, int(max_snippets))], start=1):
        meta = dict((c.get("meta") or {}))
        meta.setdefault("source_path", str(md_path))
        meta["deep_read"] = True
        body = (c.get("text") or "").strip()
        if len(body) > snippet_chars:
            body = body[:snippet_chars].rstrip() + "..."
        out.append({"score": float(s), "id": f"deep:{hashlib.sha1((str(md_path)+'|'+str(rank)).encode('utf-8','ignore')).hexdigest()[:12]}", "text": body, "meta": meta})
    _cache_set("deep_read", cache_key, out, max_items=320)
    return out


def _looks_like_reference_list_snippet(text: str) -> bool:
    s = " ".join(str(text or "").strip().split())
    if not s:
        return False
    if _REF_HEADING_RE.search(s[:160]):
        return True
    if len(re.findall(r"\[\d{1,3}\]", s)) >= 2:
        return True
    if re.match(r"^\[\d{1,3}\]\s+[A-Z][A-Za-z][^.!?]{8,}", s):
        low = s.lower()
        if (
            re.search(r"\b(?:19|20)\d{2}\b", s)
            or "proceedings" in low
            or "conference" in low
            or "arxiv" in low
            or "ieee" in low
            or "springer" in low
        ):
            return True
    return False


def _should_skip_reference_like_snippet(text: str, *, heading_path: str, question: str, source_path: str = "") -> bool:
    if _wants_reference_navigation(question):
        return False
    hp = _normalize_heading_path_for_display(str(heading_path or "").strip())
    top_h = _top_heading(hp or heading_path)
    if top_h and _is_reference_heading_like(top_h):
        return True
    return _looks_like_reference_list_snippet(text)


_OVERVIEW_HEADING_GOOD_RE = re.compile(
    r"(abstract|introduction|background|overview|summary|contribution|method|approach|results?|discussion|conclusion|"
    r"摘要|引言|背景|概述|方法|实验|结果|讨论|结论)",
    flags=re.I,
)
_OVERVIEW_HEADING_BAD_RE = re.compile(
    r"(references?|bibliography|works?\s+cited|appendi(?:x|ces)|supplementary|acknowledg(e)?ments?|"
    r"参考文献|附录|补充材料|致谢)",
    flags=re.I,
)
_OVERVIEW_TEXT_SIGNAL_RE = re.compile(
    r"(\bwe\s+(propose|present|introduce|develop)\b|\bour\s+(method|approach|framework|system)\b|"
    r"\bexperiments?\s+(show|demonstrate|indicate)\b|\bresults?\s+(show|demonstrate|indicate)\b|"
    r"本文(提出|介绍|研究)|我们(提出|设计|实现)|实验结果(表明|显示)|结果表明)",
    flags=re.I,
)


def _collect_doc_overview_snippets(md_path: Path, *, max_n: int = 3, snippet_chars: int = 360) -> list[str]:
    """
    Build doc-level overview snippets (paper summary evidence), independent of query.
    """
    md_path = Path(md_path)
    if not md_path.exists():
        return []

    try:
        mtime = float(md_path.stat().st_mtime)
    except Exception:
        mtime = 0.0
    key_raw = f"doc_overview_v2|{str(md_path)}|{mtime}|{int(max_n)}|{int(snippet_chars)}"
    cache_key = hashlib.sha1(key_raw.encode("utf-8", "ignore")).hexdigest()[:16]
    v0 = _cache_get("doc_overview", cache_key)
    if isinstance(v0, list):
        try:
            return [str(x).strip() for x in v0 if str(x).strip()][: max(1, int(max_n))]
        except Exception:
            return []

    text = _read_text_cached(md_path)
    if not text.strip():
        return []

    chunks = chunk_markdown(text, source_path=str(md_path), chunk_size=900, overlap=0)
    scored: list[tuple[float, int, str, str]] = []
    for idx, c in enumerate(chunks[:90]):
        body_raw = " ".join(str(c.get("text") or "").strip().split())
        body_raw = re.sub(r"^\s*#{1,6}\s*", "", body_raw).strip()
        body_raw = re.sub(r"^(?:\d+(?:\.\d+)*)\s+[A-Z][A-Z0-9\s]{2,40}\s+", "", body_raw).strip()
        if len(body_raw) < 90:
            continue
        meta = c.get("meta", {}) or {}
        hp = _normalize_heading_path_for_display(str(meta.get("heading_path") or ""))
        h_top, _h_sub = _split_heading_path_levels(hp)
        heading = (h_top or hp).strip()
        if heading and _OVERVIEW_HEADING_BAD_RE.search(heading):
            continue
        score = 0.0
        if heading and _OVERVIEW_HEADING_GOOD_RE.search(heading):
            score += 2.2
        if (not heading) and idx <= 2:
            score += 0.9
        if idx < 12:
            score += max(0.0, 1.1 - (0.09 * idx))
        if _OVERVIEW_TEXT_SIGNAL_RE.search(body_raw):
            score += 0.7
        score += min(0.6, len(body_raw) / 1500.0)
        scored.append((score, idx, heading, body_raw))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    out: list[str] = []
    seen_h: set[str] = set()
    seen_txt: set[str] = set()
    for _sc, _idx, heading, body in scored:
        k_h = heading.lower()
        txt = _clean_snippet_for_display(body, max_chars=max(160, int(snippet_chars)))
        if not txt:
            continue
        k_t = txt.lower()
        if k_t in seen_txt:
            continue
        if k_h and (k_h in seen_h) and (len(out) >= 1):
            continue
        if _OVERVIEW_HEADING_BAD_RE.search(txt):
            continue
        seen_h.add(k_h)
        seen_txt.add(k_t)
        out.append(txt)
        if len(out) >= int(max_n):
            break

    if not out:
        lines = [ln.strip() for ln in text.splitlines()[:120] if ln.strip()]
        for ln in lines:
            if ln.startswith("#"):
                continue
            if _OVERVIEW_HEADING_BAD_RE.search(ln):
                continue
            txt = _clean_snippet_for_display(ln, max_chars=max(140, int(snippet_chars)))
            if txt:
                out.append(txt)
            if len(out) >= int(max_n):
                break

    out2 = out[: max(1, int(max_n))]
    _cache_set("doc_overview", cache_key, out2, max_items=320)
    return out2


def _sanitize_llm_start_text(start_text: str, *, question: str, source_path: str = "") -> str:
    s = " ".join(str(start_text or "").strip().split())
    if not s:
        return ""
    allow_refs = _wants_reference_navigation(question)
    if (not allow_refs) and _REF_HEADING_RE.search(s):
        return ""
    if _should_avoid_discussion_for_question(question) and _is_discussion_or_conclusion_heading(s):
        return ""

    # Normalize backticked heading paths if present.
    m = re.search(r"`([^`]{2,180})`", s)
    if m:
        hp = _sanitize_heading_path_for_navigation(m.group(1), question=question, source_path=source_path)
        if hp:
            s = s[: m.start()] + f"`{hp}`" + s[m.end() :]
        else:
            s = (s[: m.start()] + s[m.end() :]).strip(" ;,|")
    s_compact = re.sub(r"[\s`|,;:，；。：·\-_/\\(){}\[\]]+", "", s)
    if len(s_compact) < 6:
        return ""
    if re.search(r"(先从\s*开始|start\s+with\s*$)", s, flags=re.I):
        return ""
    # Short strings that look like venue labels are not useful as a reading start.
    if len(s) <= 80 and _is_venue_heading_like(s):
        return ""
    if _should_avoid_discussion_for_question(question) and _is_discussion_or_conclusion_heading(s):
        return ""
    return s


_GUIDE_GENERIC_PATTERNS = (
    "this paper can provide information related to",
    "directly relevant information points",
    "这篇文献可提供与该问题相关的信息",
    "命中与问题直接相关的信息点",
    "与当前问题相关内容",
    "可用于回答问题的证据",
    "information related to the question",
    "evidence for the current question",
)
_GUIDE_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "at",
    "by",
    "as",
    "is",
    "are",
    "be",
    "this",
    "that",
    "these",
    "those",
    "method",
    "methods",
    "approach",
    "approaches",
    "model",
    "models",
    "algorithm",
    "algorithms",
    "experiment",
    "experiments",
    "evaluation",
    "analysis",
    "problem",
    "problems",
    "challenge",
    "challenges",
    "constraint",
    "constraints",
    "bottleneck",
    "bottlenecks",
    "metric",
    "metrics",
    "performance",
    "paper",
    "study",
    "work",
    "introduction",
    "background",
    "discussion",
    "conclusion",
    "results",
    "section",
    "sections",
    "data",
    "dataset",
    "datasets",
    "figure",
    "table",
    "supplementary",
    "appendix",
    "文献",
    "论文",
    "研究",
    "问题",
    "挑战",
    "瓶颈",
    "约束",
    "相关",
    "信息",
    "内容",
    "章节",
    "小节",
    "方法",
    "结果",
    "实验",
    "模型",
    "算法",
    "数据",
    "related",
    "relevant",
    "with",
    "using",
    "use",
    "used",
    "based",
    "via",
    "from",
    "into",
    "over",
    "under",
    "through",
    "between",
    "across",
    "improve",
    "improves",
    "improved",
    "enable",
    "enables",
    "provide",
    "provides",
    "proposed",
    "propose",
}


def _looks_generic_guidance_text(text: str) -> bool:
    s = " ".join(str(text or "").strip().split())
    if not s:
        return True
    low = s.lower()
    if any(k in low for k in _GUIDE_GENERIC_PATTERNS):
        return True
    toks = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", low) if t not in _GUIDE_STOPWORDS]
    if (len(set(toks)) <= 2) and (len(s) <= 96):
        return True
    return False


def _extract_anchor_terms_from_meta(meta: dict, *, question: str = "", max_n: int = 5) -> list[str]:
    if not isinstance(meta, dict):
        return []
    texts: list[str] = []
    target_kind = str(meta.get("anchor_target_kind") or "").strip().lower()
    try:
        target_num = int(meta.get("anchor_target_number") or 0)
    except Exception:
        target_num = 0
    if target_kind and target_num > 0:
        if target_kind == "figure":
            texts.extend([f"Figure {target_num}", f"Fig. {target_num}", f"图{target_num}", f"第{target_num}张图"])
        elif target_kind == "equation":
            texts.extend([f"Equation ({target_num})", f"Eq. {target_num}", f"公式{target_num}", f"式({target_num})", f"\\tag{{{target_num}}}"])
        elif target_kind == "table":
            texts.extend([f"Table {target_num}", f"表{target_num}"])
        else:
            texts.extend([f"{target_kind} {target_num}", f"{target_kind} ({target_num})"])
    for s in (meta.get("ref_show_snippets") or [])[:3]:
        s2 = " ".join(str(s or "").strip().split())
        if s2:
            texts.append(s2)
    for s in (meta.get("ref_snippets") or [])[:2]:
        s2 = " ".join(str(s or "").strip().split())
        if s2:
            texts.append(s2)
    for loc in (meta.get("ref_locs") or [])[:3]:
        if not isinstance(loc, dict):
            continue
        hp = str(loc.get("heading_path") or loc.get("heading") or "").strip()
        if hp:
            texts.append(hp)
    if not texts:
        return []

    q_toks = set(tokenize(str(question or "").lower()))
    score: dict[str, float] = {}

    def _add(term: str, w: float) -> None:
        t = str(term or "").strip()
        if not t:
            return
        k = t.lower()
        if len(k) <= 2:
            return
        if k in _GUIDE_STOPWORDS:
            return
        if (k in q_toks) and (len(k) <= 5):
            return
        score[t] = float(score.get(t, 0.0) + w)

    all_text = "\n".join(texts)
    for ab in re.findall(r"\b[A-Z]{2,8}\b", all_text):
        _add(ab, 2.4)
    for hy in re.findall(r"\b[A-Za-z]{3,}(?:-[A-Za-z0-9]{2,})+\b", all_text):
        _add(hy, 2.0)
    for w in re.findall(r"\b[A-Za-z][A-Za-z0-9]{3,}\b", all_text):
        wl = w.lower()
        if wl in _GUIDE_STOPWORDS:
            continue
        _add(w, 1.0)
    for zh in re.findall(r"[\u4e00-\u9fff]{2,8}", all_text):
        if zh in {"这篇文献", "当前问题", "相关信息"}:
            continue
        _add(zh, 1.4)

    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    out: list[str] = []
    seen_low: set[str] = set()
    for t, _s in ranked:
        low = t.lower()
        if low in seen_low:
            continue
        if any((low in ex) or (ex in low) for ex in seen_low if len(ex) >= 4):
            continue
        seen_low.add(low)
        out.append(t)
        if len(out) >= int(max_n):
            break
    return out


def _split_sentences_for_guidance(text: str, *, max_n: int = 20) -> list[str]:
    s = " ".join(str(text or "").replace("\n", " ").split())
    if not s:
        return []
    parts = re.split(r"(?<=[。！？；.!?;])\s+|[。！？；]", s)
    out: list[str] = []
    for p in parts:
        p2 = " ".join(str(p or "").strip().split())
        if len(p2) < 8:
            continue
        out.append(p2)
        if len(out) >= int(max_n):
            break
    return out


def _trim_clause(text: str, *, max_chars: int = 64) -> str:
    s = " ".join(str(text or "").strip().split())
    if not s:
        return ""
    s = re.sub(r"^[,;:，；：\-]+", "", s).strip()
    s = re.sub(r"[。！？.!?;；]+$", "", s).strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _looks_like_keyword_list_text(text: str) -> bool:
    s = " ".join(str(text or "").strip().split())
    if not s:
        return True
    if len(s) <= 64 and (s.count(",") + s.count("，") + s.count(";") + s.count("；")) >= 2:
        return True
    low = s.lower()
    verb_markers = (
        "提出",
        "采用",
        "利用",
        "通过",
        "实现",
        "验证",
        "提升",
        "对比",
        "解决",
        "propose",
        "introduce",
        "use",
        "using",
        "achieve",
        "improve",
        "show",
        "demonstrate",
        "outperform",
    )
    if not any(v in low for v in verb_markers):
        toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", low)
        if (len(toks) >= 3) and ((low.count(",") + low.count(";")) >= 2):
            return True
    return False


def _contains_question_echo(text: str, question: str) -> bool:
    t = " ".join(str(text or "").strip().split()).lower()
    q = " ".join(str(question or "").strip().split()).lower()
    if not t or not q:
        return False
    q_compact = re.sub(r"[\s`'\"“”‘’，。！？,.?!:;；：()（）\-_/\\]+", "", q)
    t_compact = re.sub(r"[\s`'\"“”‘’，。！？,.?!:;；：()（）\-_/\\]+", "", t)
    if len(q_compact) < 10:
        return False
    # Match medium-length chunks from the question; indicates likely paraphrase/echo.
    for n in (24, 18, 14):
        if len(q_compact) < n:
            continue
        max_start = min(len(q_compact) - n, 28)
        for s in range(max_start + 1):
            chunk = q_compact[s : s + n]
            if chunk and (chunk in t_compact):
                return True
    return False


def _looks_latin_heavy(text: str) -> bool:
    s = str(text or "")
    if not s.strip():
        return False
    n_cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
    n_lat = len(re.findall(r"[A-Za-z]", s))
    return (n_lat >= 18) and (n_lat >= (2 * n_cjk + 8))


def _anchor_specificity_score(term: str) -> float:
    t = " ".join(str(term or "").strip().split())
    if not t:
        return -1e9
    low = t.lower()
    score = 0.0
    # Prefer phrase-like / technical-looking anchors.
    if ("-" in t) or (" " in t):
        score += 2.4
    if re.search(r"\d", t):
        score += 1.8
    if re.search(r"[A-Z]{2,}", t):
        score += 1.5
    if len(t) >= 12:
        score += 1.1
    # Penalize over-generic academic words.
    if low in _GUIDE_STOPWORDS:
        score -= 2.5
    if re.fullmatch(r"[a-z]+", low) and len(low) <= 6:
        score -= 0.8
    return score


def _pick_specific_terms(cands: list[str], *, max_n: int = 3) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    ranked = sorted(
        [str(x or "").strip() for x in (cands or []) if str(x or "").strip()],
        key=_anchor_specificity_score,
        reverse=True,
    )
    for t in ranked:
        k = t.lower()
        if k in seen:
            continue
        if any((k in s) or (s in k) for s in seen if len(s) >= 5):
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= int(max_n):
            break
    return out


def _calibrate_refs_pack_score(
    *,
    raw_score: float,
    meta: dict | None,
    section: str,
) -> float:
    """Calibrate LLM score with retrieval evidence to avoid score collapse."""
    m = meta if isinstance(meta, dict) else {}
    rank = m.get("ref_rank") if isinstance(m.get("ref_rank"), dict) else {}

    def _to_float(v, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(float(lo), min(float(hi), float(v)))

    raw = _clamp(_to_float(raw_score, 0.0), 0.0, 100.0)
    bm25 = max(0.0, _to_float(rank.get("bm25"), 0.0))
    deep = max(0.0, _to_float(rank.get("deep"), 0.0))
    term_bonus = _to_float(rank.get("term_bonus"), 0.0)
    sem = max(0.0, _to_float(rank.get("semantic_score"), 0.0))
    explicit_doc = max(0.0, _to_float(m.get("explicit_doc_match_score"), 0.0))
    anchor_kind = str(m.get("anchor_target_kind") or "").strip().lower()
    anchor_match = max(0.0, _to_float(m.get("anchor_match_score"), 0.0))
    has_anchor = bool(anchor_kind) and anchor_match > 0.0

    bm25_n = _clamp((bm25 - 0.5) / 6.0, 0.0, 1.0)
    deep_n = _clamp(deep / 4.0, 0.0, 1.0)
    term_n = _clamp((term_bonus + 1.0) / 5.0, 0.0, 1.0)
    sem_n = _clamp(sem / 10.0, 0.0, 1.0)
    explicit_n = _clamp(explicit_doc / 10.0, 0.0, 1.0)
    evidence = (
        (0.34 * bm25_n)
        + (0.18 * deep_n)
        + (0.16 * term_n)
        + (0.22 * sem_n)
        + (0.10 * explicit_n)
    )
    calibrated = ((0.70 * (raw / 100.0)) + (0.30 * evidence)) * 100.0

    if str(section or "").strip():
        calibrated += 2.0
    else:
        calibrated -= 4.0

    if term_bonus < 0.0:
        calibrated += max(-12.0, 3.0 * term_bonus)

    if (not has_anchor) and term_bonus <= 0.0 and bm25 < 1.0:
        calibrated = min(calibrated, 60.0)
    elif (not has_anchor) and term_bonus <= 0.0 and bm25 < 2.0:
        calibrated = min(calibrated, 68.0)

    if has_anchor:
        calibrated = max(calibrated, min(98.0, 82.0 + min(14.0, 0.8 * anchor_match)))

    return _clamp(calibrated, 0.0, 100.0)


def _postprocess_refs_pack(result: dict[int, dict], docs: list[dict], *, question: str) -> dict[int, dict]:
    if not isinstance(result, dict):
        return {}
    q = (question or "").strip()
    cjk = _has_cjk(q)

    source_by_i: dict[int, str] = {}
    meta_by_i: dict[int, dict] = {}
    for i, d in enumerate(docs or [], start=1):
        meta = d.get("meta", {}) or {}
        meta_by_i[i] = meta
        source_by_i[i] = str(meta.get("source_path") or "").strip()

    def _anchor_target_label(meta: dict) -> str:
        kind = str((meta or {}).get("anchor_target_kind") or "").strip().lower()
        try:
            num = int((meta or {}).get("anchor_target_number") or 0)
        except Exception:
            num = 0
        if (not kind) or num <= 0:
            return ""
        if cjk:
            if kind == "figure":
                return f"图{num}"
            if kind == "equation":
                return f"公式{num}"
            if kind == "table":
                return f"表{num}"
            if kind == "theorem":
                return f"定理{num}"
            if kind == "lemma":
                return f"引理{num}"
            return f"{kind}{num}"
        if kind == "figure":
            return f"Figure {num}"
        if kind == "equation":
            return f"Equation ({num})"
        if kind == "table":
            return f"Table {num}"
        return f"{kind} {num}"

    def _anchor_conflict_text(text: str) -> bool:
        low = " ".join(str(text or "").strip().split()).lower()
        if not low:
            return False
        pats = (
            "未直接给出",
            "无法确认",
            "未包含",
            "仅提及",
            "并未给出",
            "not directly",
            "cannot confirm",
            "not enough context",
            "only partial",
            "not explicitly given",
            "does not directly give",
        )
        return any(p in low for p in pats)

    def _anchor_grounded_why(meta: dict, sec: str) -> str:
        label = _anchor_target_label(meta)
        if not label:
            return ""
        loc = str(sec or (meta.get("ref_section") or meta.get("top_heading") or "")).strip()
        if cjk:
            if loc:
                return f"问题直接询问{label}的具体内容，而该文已在“{loc}”附近命中对应编号的原文片段，可直接作为回答依据。"
            return f"问题直接询问{label}的具体内容，而该文已命中对应编号的原文片段，可直接作为回答依据。"
        if loc:
            return f"The question asks for {label}, and this paper directly matches that numbered item near '{loc}', so it can be used as direct evidence."
        return f"The question asks for {label}, and this paper directly matches that numbered item in the retrieved snippets."

    def _clean_model_text(s: str, *, max_sentences: int, min_cjk_chars: int = 14) -> str:
        t = " ".join(str(s or "").strip().split())
        if not t:
            return ""
        t = t.replace("...", " ").replace("…", " ")
        t = re.sub(r"\s{2,}", " ", t).strip()
        if _contains_question_echo(t, q):
            return ""
        if cjk:
            compact = re.sub(r"\s+", "", t)
            if (not _has_cjk(t)) or _looks_latin_heavy(t) or (len(compact) < max(6, int(min_cjk_chars))):
                return ""
            if _looks_like_keyword_list_text(t):
                return ""
            if re.search(r"(该文针对|核心做法是|并报告|该文主要解决|方法上)", t):
                return ""
        else:
            if _looks_generic_guidance_text(t) or _looks_like_keyword_list_text(t):
                return ""
            if re.search(r"\b(the paper tackles|the core method is|the reported result)\b", t, flags=re.I):
                return ""
        # Keep only the first 1..N sentences to avoid verbose blocks.
        parts = re.split(r"(?<=[。！？.!?])\s+|[；;]", t)
        keep: list[str] = []
        for p in parts:
            p2 = " ".join(str(p or "").strip().split())
            if not p2:
                continue
            keep.append(p2)
            if len(keep) >= max(1, int(max_sentences)):
                break
        if not keep:
            return ""
        out = " ".join(keep).strip()
        if cjk and (not re.search(r"[。！？]$", out)):
            out += "。"
        if (not cjk) and (not re.search(r"[.!?]$", out)):
            out += "."
        return out

    for i, it in list(result.items()):
        if not isinstance(it, dict):
            continue
        idx = int(i)
        src_i = str(source_by_i.get(idx, "") or "")
        meta_i = meta_by_i.get(idx, {}) or {}

        sec = str(it.get("section") or "").strip()
        sec = _sanitize_heading_path_for_navigation(sec, question=q, source_path=src_i)
        sec_top, _sec_sub = _split_heading_path_levels(sec)
        sec = sec_top or sec
        if sec and (
            _is_non_navigational_heading(sec, question=q, source_path=src_i)
            or _is_low_quality_navigation_heading(sec, question=q, source_path=src_i)
        ):
            sec = ""
        if sec and _should_avoid_discussion_for_question(q) and _is_discussion_or_conclusion_heading(sec):
            sec = ""
        if not sec:
            sec_alt, _sub_alt = _best_loc_heading_for_question(meta_i, question=q, source_path=src_i)
            if sec_alt:
                sec = sec_alt

        what = _clean_model_text(str(it.get("what") or ""), max_sentences=3, min_cjk_chars=16)
        why = _clean_model_text(str(it.get("why") or ""), max_sentences=2, min_cjk_chars=8)
        start = _sanitize_llm_start_text(str(it.get("start") or "").strip(), question=q, source_path=src_i)
        gain = _clean_model_text(str(it.get("gain") or ""), max_sentences=2, min_cjk_chars=10)
        raw_find = it.get("find") if isinstance(it.get("find"), list) else []
        find = [str(x or "").strip() for x in (raw_find or []) if str(x or "").strip()]
        find = [x for x in find if not _looks_generic_guidance_text(x)][:4]
        score_cal = _calibrate_refs_pack_score(
            raw_score=float(it.get("score", 0.0) or 0.0),
            meta=meta_i,
            section=sec,
        )

        has_anchor_hit = bool(str(meta_i.get("anchor_target_kind") or "").strip()) and float(meta_i.get("anchor_match_score", 0.0) or 0.0) > 0.0
        if has_anchor_hit and (_anchor_conflict_text(why) or (not why)):
            why = _anchor_grounded_why(meta_i, sec)
        if has_anchor_hit and _anchor_conflict_text(gain):
            gain = ""

        it["score"] = float(score_cal)
        it["what"] = what
        it["why"] = why
        it["gain"] = gain
        it["start"] = start
        it["section"] = sec
        it["find"] = find
        result[i] = it

    return result


def _parse_json_object_lenient(text: str) -> dict | None:
    s = str(text or "").strip()
    if not s:
        return None
    if s.startswith("```"):
        s = s.strip().strip("`")
        s = re.sub(r"^\s*json\s*", "", s, flags=re.I).strip()
    try:
        data = json.loads(s)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    # Fallback: extract the largest JSON-looking object from mixed text.
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    blob = str(m.group(0) or "").strip()
    if not blob:
        return None
    try:
        data = json.loads(blob)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _llm_refs_pack_docwise_items(settings, *, question: str, items: list[dict], on_item=None) -> list[dict]:
    """
    Fallback path when one-shot pack generation fails.
    Generate each doc's ref pack independently so one bad sample doesn't block all docs.
    """
    if (not settings) or (not getattr(settings, "api_key", None)):
        return []
    q = (question or "").strip()
    if (not q) or (not isinstance(items, list)) or (not items):
        return []

    try:
        settings_fast = replace(
            settings,
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 18.0),
            max_retries=0,
        )
    except Exception:
        settings_fast = settings

    try:
        base_docwise_timeout_s = float(getattr(settings_fast, "timeout_s", 18.0) or 18.0)
    except Exception:
        base_docwise_timeout_s = 18.0
    base_docwise_timeout_s = max(6.0, min(18.0, base_docwise_timeout_s))
    retry_docwise_timeout_s = max(base_docwise_timeout_s, min(24.0, max(base_docwise_timeout_s + 4.0, base_docwise_timeout_s * 1.25)))

    sys = (
        "You are an academic paper summarizer for retrieval references.\n"
        "Return JSON ONLY with keys: score, what, why, section.\n"
        "Rules:\n"
        "- score: 0..100 (how relevant this paper is to the question).\n"
        "- Use broad score distribution; do not repeat fixed constants across many docs.\n"
        "- Strong direct evidence: 80-95; partial relevance: 45-75; weak/noisy relation: <=40.\n"
        "- what: 1-2 fluent sentences summarizing the paper itself (goal/method/evidence), independent of the question.\n"
        "- why: 1-2 fluent sentences explaining relevance to the question, with concrete snippet/location evidence.\n"
        "- section: choose from provided headings if possible; else empty string.\n"
        "- Match user language (Chinese question -> Chinese output).\n"
        "- Output JSON only. No markdown fences.\n"
    )

    def _is_usable_docwise_result(data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        what = str(data.get("what") or "").strip()
        why = str(data.get("why") or "").strip()
        return bool(what and why)

    def _run_docwise_once(it: dict, *, timeout_s: float, max_tokens: int) -> dict | None:
        if not isinstance(it, dict):
            return None
        try:
            idx = int(it.get("i") or 0)
        except Exception:
            idx = 0
        if idx <= 0:
            return None
        payload = {
            "question": q,
            "doc": {
                "i": idx,
                "headings": list(it.get("headings") or [])[:10],
                "locs": list(it.get("locs") or [])[:2],
                "overview_snippets": list(it.get("overview_snippets") or [])[:1],
                "snippets": list(it.get("snippets") or [])[:2],
                "anchors": list(it.get("anchors") or [])[:3],
            },
        }
        user = json.dumps(payload, ensure_ascii=False)
        try:
            try:
                local_settings = replace(
                    settings_fast,
                    timeout_s=float(timeout_s),
                    max_retries=0,
                )
            except Exception:
                local_settings = settings_fast
            ds = DeepSeekChat(local_settings)
            out = (ds.chat(
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
            ) or "").strip()
        except Exception:
            return None
        data = _parse_json_object_lenient(out)
        if not isinstance(data, dict):
            return None
        return {
            "i": idx,
            "score": data.get("score", 0.0),
            "why": str(data.get("why") or "").strip(),
            "what": str(data.get("what") or "").strip(),
            "start": str(data.get("start") or "").strip(),
            "gain": str(data.get("gain") or "").strip(),
            "find": data.get("find") if isinstance(data.get("find"), list) else [],
            "section": str(data.get("section") or "").strip(),
        }

    def _one_doc(it: dict) -> dict | None:
        rec = _run_docwise_once(it, timeout_s=base_docwise_timeout_s, max_tokens=280)
        if _is_usable_docwise_result(rec or {}):
            return rec
        rec_retry = _run_docwise_once(it, timeout_s=retry_docwise_timeout_s, max_tokens=420)
        if _is_usable_docwise_result(rec_retry or {}):
            return rec_retry
        return rec if _is_usable_docwise_result(rec or {}) else None

    arr: list[dict] = []
    seed = [x for x in items[:8] if isinstance(x, dict)]
    max_workers = max(1, min(6, len(seed)))
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_one_doc, it) for it in seed]
            for fu in as_completed(futs):
                try:
                    rec = fu.result()
                except Exception:
                    rec = None
                if isinstance(rec, dict):
                    arr.append(rec)
                    if callable(on_item):
                        try:
                            on_item(dict(rec))
                        except Exception:
                            pass
    except Exception:
        for it in seed:
            rec = _one_doc(it)
            if isinstance(rec, dict):
                arr.append(rec)
                if callable(on_item):
                    try:
                        on_item(dict(rec))
                    except Exception:
                        pass
    return arr


def _build_llm_refs_pack_items(question: str, docs: list[dict]) -> tuple[list[dict], dict[int, str]]:
    q = (question or "").strip()
    items: list[dict] = []
    source_by_i: dict[int, str] = {}
    for i, d in enumerate(docs, start=1):
        meta = d.get("meta", {}) or {}
        src_i = str(meta.get("source_path") or "").strip()
        source_by_i[i] = src_i
        headings = [h for h in (meta.get("ref_headings") or []) if isinstance(h, str)]
        hs_clean: list[str] = []
        hs_seen: set[str] = set()
        for hh in headings:
            hp_h = _sanitize_heading_path_for_navigation(hh, question=q, source_path=src_i)
            top_h, _sub_h = _split_heading_path_levels(hp_h)
            hh2 = _normalize_heading(top_h or hp_h)
            if not hh2:
                continue
            if _is_non_navigational_heading(hh2, question=q, source_path=src_i):
                continue
            k = hh2.lower()
            if k in hs_seen:
                continue
            hs_seen.add(k)
            hs_clean.append(hh2)
            if len(hs_clean) >= 8:
                break
        headings = hs_clean
        locs_payload: list[dict] = []
        raw_locs = meta.get("ref_locs")
        if isinstance(raw_locs, list):
            for loc in raw_locs[:2]:
                if not isinstance(loc, dict):
                    continue
                hp = str(loc.get("heading_path") or loc.get("heading") or "").strip()
                if not hp:
                    continue
                hp = _sanitize_heading_path_for_navigation(hp, question=q, source_path=src_i)
                if not hp:
                    continue
                rec = {"heading_path": hp}
                try:
                    p0 = int(loc.get("page_start")) if loc.get("page_start") is not None else None
                except Exception:
                    p0 = None
                try:
                    p1 = int(loc.get("page_end")) if loc.get("page_end") is not None else None
                except Exception:
                    p1 = None
                if p0 is not None and p0 > 0:
                    rec["page_start"] = p0
                if p1 is not None and p1 > 0:
                    rec["page_end"] = p1
                locs_payload.append(rec)
        snippets = []
        for s in (meta.get("ref_show_snippets") or [])[:2]:
            s2 = " ".join(str(s).strip().split())
            if len(s2) > 360:
                s2 = s2[:360].rstrip() + "..."
            if s2:
                snippets.append(s2)
        overview_snippets = []
        for s in (meta.get("ref_overview_snippets") or [])[:2]:
            s2 = " ".join(str(s).strip().split())
            if len(s2) > 360:
                s2 = s2[:360].rstrip() + "..."
            if s2:
                overview_snippets.append(s2)
        if not snippets:
            s = " ".join((d.get("text") or "").strip().split())
            if len(s) > 360:
                s = s[:360].rstrip() + "..."
            if s:
                snippets.append(s)
        anchors_i = _extract_anchor_terms_from_meta(meta, question=q, max_n=5)
        target_anchor = {}
        kind_i = str(meta.get("anchor_target_kind") or "").strip().lower()
        try:
            num_i = int(meta.get("anchor_target_number") or 0)
        except Exception:
            num_i = 0
        if kind_i and num_i > 0:
            target_anchor = {"kind": kind_i, "number": num_i}
        items.append(
            {
                "i": i,
                "headings": headings,
                "locs": locs_payload,
                "overview_snippets": overview_snippets,
                "snippets": snippets,
                "anchors": anchors_i,
                "target_anchor": target_anchor,
            }
        )
    return items, source_by_i


def _llm_refs_pack_batch(
    settings,
    *,
    question: str,
    docs: list[dict],
    items: list[dict],
    source_by_i: dict[int, str],
) -> dict[int, dict]:
    if not settings or (not getattr(settings, "api_key", None)):
        return {}
    q = (question or "").strip()
    if (not q) or (not docs) or (not items):
        return {}

    try:
        settings_fast = replace(
            settings,
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 14.0),
            max_retries=0,
        )
    except Exception:
        settings_fast = settings
    ds = DeepSeekChat(settings_fast)
    sys = (
        "You are a strict academic retriever reranker and reading guide generator.\n"
        "Output JSON ONLY: "
        "{\"items\":[{\"i\":int,\"score\":number,\"why\":string,\"what\":string,\"start\":string,\"gain\":string,\"find\":[string],\"section\":string}]}.\n"
        "Rules:\n"
        "- score: 0..100, based on how directly snippets answer the question.\n"
        "- Use broad score distribution; avoid repeated fixed constants (e.g., 97.6 for many docs).\n"
        "- Strong direct evidence: 80-95; partial relevance: 45-75; weak/noisy relation: <=40.\n"
        "- Penalize false-friend term mismatch (e.g., single-shot vs single-pixel).\n"
        "- Use ONLY snippets/headings; DO NOT use filenames.\n"
        "- If target_anchor is provided and snippets/anchors directly contain that numbered figure/equation/theorem, do NOT claim the item is missing, not directly given, or unverifiable.\n"
        "- section MUST be chosen from provided headings; otherwise empty string.\n"
        "- Prefer using candidate locs.heading_path when writing the start field.\n"
        "- NEVER use journal/venue names as section (e.g., Nature Photonics, Science Advances).\n"
        "- NEVER point start to References/Bibliography unless the question explicitly asks for citations/references.\n"
        "- If no reliable section exists, set section to empty and give a paragraph-level start strategy from snippets.\n"
        "- For HOW/METHOD questions, DO NOT set section/start to Discussion or Conclusion unless the user explicitly asks limitations/discussion.\n"
        "- Match the user's language (Chinese question -> Chinese output).\n"
        "- If the question is Chinese, write fluent Chinese sentences; avoid broken English fragments unless they are exact method names.\n"
        "- For Chinese output, `what` and `why` MUST be fluent natural Chinese; avoid broken English except exact proper nouns.\n"
        "- Avoid rigid templates and avoid repeating the same wording across fields.\n"
        "- In `why`, explicitly point to a concrete location using section (and page range when available in locs).\n"
        "- `what` MUST be a paper-level overview independent of the current question.\n"
        "- Build `what` from overview_snippets first; use snippets only as fallback if overview_snippets are weak.\n"
        "- `what`: write 1-3 complete sentences (typically 2) summarizing the paper's goal, core method, and key evidence.\n"
        "- Do NOT use fixed writing templates like '该文针对...核心做法是...并报告...'.\n"
        "- Do NOT output ellipsis ('...' or '…') in `what`.\n"
        "- If one part is weakly supported, state uncertainty briefly instead of fabricating.\n"
        "- `why` MUST focus on why this paper is relevant to the current question, and point to concrete evidence in snippets/locs.\n"
        "- When target_anchor is present, `why` should explicitly state that the matching numbered item is found in snippets/locs and where it appears.\n"
        "- Keep `what` and `why` semantically distinct; do not paraphrase one into the other.\n"
        "- start: where to start reading (section/subsection + what to look for first).\n"
        "- gain: what the user can extract from this paper that helps answer the question.\n"
        "- find: 2-4 concrete items to look for (methods, settings, formulas, results, ablations, etc.).\n"
        "- `find` items must be clean noun phrases; do NOT output table rows, pipe-delimited text, or metric-only number dumps.\n"
        "- MUST include at least 1 paper-specific anchor term directly present in snippets (e.g., method/component/dataset/metric names).\n"
        "- Prefer terms from the provided anchors list when available.\n"
        "- Across different items, avoid reusing the same wording; keep each item distinct.\n"
        "- Do NOT output generic template phrases or broad taxonomies unless snippets explicitly support them.\n"
        "- Prefer concrete nouns and terms from snippets over abstract wording.\n"
        "- Keep each field concise but informative. Avoid boilerplate and avoid repeating the same words across fields.\n"
        "- If evidence is weak or only partial, reduce score and state the limitation in why/gain.\n"
    )
    payload = {"question": q, "allow_reference_section": bool(_wants_reference_navigation(q)), "docs": items}
    user = json.dumps(payload, ensure_ascii=False)
    out = ""
    try:
        max_tokens = min(720, max(360, 220 * len(items) + 80))
        out = (
            ds.chat(
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            or ""
        ).strip()
    except Exception:
        out = ""
    data = _parse_json_object_lenient(out)
    arr = data.get("items") if isinstance(data, dict) else None
    if not isinstance(arr, list):
        return {}

    result: dict[int, dict] = {}
    for it in arr:
        if not isinstance(it, dict):
            continue
        try:
            i = int(it.get("i"))
        except Exception:
            continue
        doc_meta_i = {}
        if 1 <= i <= len(docs):
            try:
                doc_meta_i = (docs[i - 1].get("meta", {}) or {})
            except Exception:
                doc_meta_i = {}
        try:
            sc = float(it.get("score", 0.0) or 0.0)
        except Exception:
            sc = 0.0
        sc = max(0.0, min(100.0, sc))
        src_i = str(source_by_i.get(i) or "").strip()
        sec_raw = str(it.get("section") or "").strip()
        sec_path = _sanitize_heading_path_for_navigation(sec_raw, question=q, source_path=src_i)
        sec_top, _sec_sub = _split_heading_path_levels(sec_path)
        sec_final = sec_top.strip()
        if sec_final and _is_low_quality_navigation_heading(sec_final, question=q, source_path=src_i):
            sec_final = ""
        if sec_final and _should_avoid_discussion_for_question(q) and _is_discussion_or_conclusion_heading(sec_final):
            sec_final = ""
        if not sec_final:
            sec_alt, _sub_alt = _best_loc_heading_for_question(doc_meta_i, question=q, source_path=src_i)
            if sec_alt:
                sec_final = sec_alt

        start_raw = str(it.get("start") or "").strip()
        start_final = _sanitize_llm_start_text(start_raw, question=q, source_path=src_i)
        result[i] = {
            "score": sc,
            "why": str(it.get("why") or "").strip(),
            "what": str(it.get("what") or "").strip(),
            "start": start_final,
            "gain": str(it.get("gain") or "").strip(),
            "find": [str(x).strip() for x in (it.get("find") or []) if str(x).strip()][:4] if isinstance(it.get("find"), list) else [],
            "section": sec_final,
        }

    result = _postprocess_refs_pack(result, docs, question=q)
    return result


def _llm_refs_pack(settings, *, question: str, docs: list[dict]) -> dict[int, dict]:
    """
    LLM-only refs pack with bounded latency:
    - split docs into small batches
    - run batches in parallel
    - fallback docwise only for missing items

    Returns: {idx -> {"score":float, "why":str, "what":str, "start":str, "gain":str, "find":[str], "section":str}}
    """
    if not settings or (not getattr(settings, "api_key", None)):
        return {}
    q = (question or "").strip()
    if not q or not docs:
        return {}

    items, source_by_i = _build_llm_refs_pack_items(q, docs)

    try:
        sig_parts = ["refs_pack_v11", q]
        for d in docs:
            src = str((d.get("meta", {}) or {}).get("source_path") or "")
            try:
                mtime = float(Path(src).stat().st_mtime) if src else 0.0
            except Exception:
                mtime = 0.0
            sig_parts.append(src + "|" + str(mtime))
        sig = "|".join(sig_parts)
        cache_key = hashlib.sha1(sig.encode("utf-8", "ignore")).hexdigest()[:16]
    except Exception:
        cache_key = hashlib.sha1(q.encode("utf-8", "ignore")).hexdigest()[:16]

    v0 = _cache_get("refs_pack", cache_key)
    if isinstance(v0, dict):
        return v0

    if _prompt_explicitly_requests_multi_paper_list(q):
        try:
            settings_fast = replace(
                settings,
                timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 10.0),
                max_retries=0,
            )
        except Exception:
            settings_fast = settings
        try:
            arr_retry = _llm_refs_pack_docwise_items(
                settings_fast,
                question=q,
                items=items,
            )
        except Exception:
            arr_retry = []
        result: dict[int, dict] = {}
        for rec in arr_retry:
            if not isinstance(rec, dict):
                continue
            try:
                idx = int(rec.get("i") or 0)
            except Exception:
                idx = 0
            if idx <= 0:
                continue
            result[idx] = dict(rec)
        result = _postprocess_refs_pack(result, docs, question=q)
        _cache_set("refs_pack", cache_key, result, max_items=260)
        return result

    item_batches: list[list[dict]] = []
    batch_size = 2 if len(items) > 4 else 3
    batch_size = max(1, batch_size)
    for pos in range(0, len(items), batch_size):
        batch = [it for it in items[pos : pos + batch_size] if isinstance(it, dict)]
        if batch:
            item_batches.append(batch)

    pack_batch: dict[int, dict] = {}
    if len(item_batches) <= 1:
        try:
            pack_batch = _llm_refs_pack_batch(
                settings,
                question=q,
                docs=docs,
                items=items,
                source_by_i=source_by_i,
            )
        except Exception:
            pack_batch = {}
    else:
        max_workers = max(1, min(3, len(item_batches)))
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [
                    ex.submit(
                        _llm_refs_pack_batch,
                        settings,
                        question=q,
                        docs=docs,
                        items=batch,
                        source_by_i=source_by_i,
                    )
                    for batch in item_batches
                ]
                for fu in as_completed(futs):
                    try:
                        rec = fu.result()
                    except Exception:
                        rec = {}
                    if isinstance(rec, dict) and rec:
                        pack_batch.update(rec)
        except Exception:
            pack_batch = {}

    ready_ids = {int(i) for i, rec in (pack_batch or {}).items() if isinstance(rec, dict)}
    missing_items = [it for it in items if int(it.get("i") or 0) not in ready_ids]
    arr: list[dict] = [dict(rec) for rec in (pack_batch or {}).values() if isinstance(rec, dict)]
    if missing_items:
        try:
            settings_fast = replace(
                settings,
                timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 18.0),
                max_retries=0,
            )
        except Exception:
            settings_fast = settings
        try:
            arr_retry = _llm_refs_pack_docwise_items(settings_fast, question=q, items=missing_items)
        except Exception:
            arr_retry = []
        for rec in arr_retry:
            if isinstance(rec, dict):
                arr.append(rec)

    result: dict[int, dict] = {}
    for rec in arr:
        if not isinstance(rec, dict):
            continue
        try:
            idx = int(rec.get("i") or 0)
        except Exception:
            idx = 0
        if idx <= 0:
            continue
        result[idx] = dict(rec)
    result = _postprocess_refs_pack(result, docs, question=q)
    _cache_set("refs_pack", cache_key, result, max_items=260)
    return result


def _apply_llm_pack_to_grouped_docs(
    docs: list[dict],
    *,
    pack: dict[int, dict],
    question: str,
    clear_missing: bool = True,
) -> list[dict]:
    if not isinstance(docs, list) or (not docs):
        return docs
    if not isinstance(pack, dict) or (not pack):
        return docs
    q = (question or "").strip()
    for i, d in enumerate(docs, start=1):
        if not isinstance(d, dict):
            continue
        meta = d.get("meta", {}) or {}
        src_meta = str(meta.get("source_path") or "").strip()
        pr = pack.get(i) or {}
        if not isinstance(pr, dict):
            if clear_missing and str(meta.get("ref_pack_state") or "").strip().lower() == "pending":
                meta["ref_pack_state"] = "none"
                d["meta"] = meta
            continue
        try:
            llm_score = float(pr.get("score", 0.0) or 0.0)
        except Exception:
            llm_score = 0.0
        llm_score = max(0.0, min(100.0, llm_score))
        llm_why = str(pr.get("why") or "").strip()
        llm_start = _sanitize_llm_start_text(str(pr.get("start") or "").strip(), question=q, source_path=src_meta)
        sec_raw = str(pr.get("section") or "").strip()
        sec_path = _sanitize_heading_path_for_navigation(sec_raw, question=q, source_path=src_meta)
        sec, _sec_sub = _split_heading_path_levels(sec_path)
        if sec and _is_low_quality_navigation_heading(sec, question=q, source_path=src_meta):
            sec = ""
        if sec and _should_avoid_discussion_for_question(q) and _is_discussion_or_conclusion_heading(sec):
            sec = ""
        if not sec:
            sec_alt, _sub_alt = _best_loc_heading_for_question(meta, question=q, source_path=src_meta)
            if sec_alt:
                sec = sec_alt

        meta["ref_pack"] = {
            "score": llm_score,
            "why": llm_why,
            "what": str(pr.get("what") or "").strip(),
            "start": llm_start,
            "gain": str(pr.get("gain") or "").strip(),
            "find": [str(x).strip() for x in (pr.get("find") or []) if str(x).strip()][:4] if isinstance(pr.get("find"), list) else [],
            "section": sec,
        }
        meta["ref_pack_state"] = "ready"

        if sec and (not _is_non_navigational_heading(sec, question=q, source_path=src_meta)):
            meta["top_heading"] = sec
            meta["ref_section"] = sec
            meta["ref_loc_quality"] = "high"
            meta.pop("ref_subsection", None)
            locs_meta = meta.get("ref_locs") or []
            if isinstance(locs_meta, list):
                for loc in locs_meta:
                    if not isinstance(loc, dict):
                        continue
                    hp = str(loc.get("heading_path") or "").strip()
                    top_h, sub_h = _split_heading_path_levels(hp or str(loc.get("heading") or ""))
                    if top_h and (top_h.lower() == sec.lower()):
                        if hp:
                            meta["ref_best_heading_path"] = hp
                        if sub_h and (not _is_non_navigational_heading(sub_h, question=q, source_path=src_meta)):
                            meta["ref_subsection"] = sub_h
                        break
        elif str(meta.get("ref_loc_quality") or "").strip().lower() == "low":
            meta.pop("ref_best_heading_path", None)
            meta.pop("ref_section", None)
            meta.pop("ref_subsection", None)

        r = meta.get("ref_rank") or {}
        try:
            bm25 = float(r.get("bm25", 0.0) or 0.0)
        except Exception:
            bm25 = 0.0
        try:
            deep_best = float(r.get("deep", 0.0) or 0.0)
        except Exception:
            deep_best = 0.0
        try:
            term_bonus = float(r.get("term_bonus", 0.0) or 0.0)
        except Exception:
            term_bonus = 0.0
        deep_scaled = 1.6 * (deep_best ** 0.6) if deep_best > 0 else 0.0
        display_score = llm_score / 10.0
        combined2 = display_score + (0.25 * deep_scaled) + (0.10 * bm25) + (0.50 * term_bonus)
        meta["ref_rank"] = {
            "bm25": bm25,
            "deep": deep_best,
            "term_bonus": term_bonus,
            "llm": llm_score,
            "why": llm_why,
            "score": display_score,
            "display_score": display_score,
            "semantic_score": combined2,
        }
        d["score"] = display_score
        d["meta"] = meta
    return docs


def _semantic_filter_docs_by_llm(docs: list[dict]) -> list[dict]:
    if not isinstance(docs, list) or (not docs):
        return []
    llm_scores: list[float] = []
    for d in docs:
        meta = d.get("meta", {}) or {}
        rank = meta.get("ref_rank") or {}
        try:
            llm_sc = float(rank.get("llm", 0.0) or 0.0)
        except Exception:
            llm_sc = 0.0
        if llm_sc > 0:
            llm_scores.append(llm_sc)
    if not llm_scores:
        return docs

    best_llm = max(llm_scores)
    # Adaptive: keep docs within 60% of best score, with a floor of 20.
    # This avoids the hard 28.0 floor that kept everything when best_llm was weak.
    sem_keep_min = max(20.0, best_llm * 0.40)
    filtered: list[dict] = []
    for d in docs:
        meta = d.get("meta", {}) or {}
        rank = meta.get("ref_rank") or {}
        try:
            llm_sc = float(rank.get("llm", 0.0) or 0.0)
        except Exception:
            llm_sc = 0.0
        if llm_sc >= sem_keep_min:
            filtered.append(d)
    return filtered or docs


def _enrich_grouped_refs_with_llm_pack(
    docs: list[dict],
    *,
    question: str,
    settings=None,
    top_k_docs: int | None = None,
    progress_cb=None,
) -> list[dict]:
    """
    Enrich already-grouped refs with LLM pack (semantic rerank + reading guide),
    intended for async/background refinement without blocking answer streaming.
    """
    if not isinstance(docs, list) or (not docs):
        return []
    if (not settings) or (not getattr(settings, "api_key", None)):
        docs_no_llm = copy.deepcopy(docs)
        for d in docs_no_llm:
            if not isinstance(d, dict):
                continue
            m = d.get("meta", {}) or {}
            if str(m.get("ref_pack_state") or "").strip().lower() == "pending":
                m["ref_pack_state"] = "failed"
                d["meta"] = m
        return docs_no_llm
    q = (question or "").strip()
    if not q:
        docs_no_q = copy.deepcopy(docs)
        for d in docs_no_q:
            if not isinstance(d, dict):
                continue
            m = d.get("meta", {}) or {}
            if str(m.get("ref_pack_state") or "").strip().lower() == "pending":
                m["ref_pack_state"] = "failed"
                d["meta"] = m
        return docs_no_q

    docs2 = copy.deepcopy(docs)
    pack_batch: dict[int, dict] = {}
    partial_pack: dict[int, dict] = {}
    items, _source_by_i = _build_llm_refs_pack_items(q, docs2)
    used_multi_paper_docwise_fast_path = bool(_prompt_explicitly_requests_multi_paper_list(q))

    def _on_item(rec: dict) -> None:
        if not isinstance(rec, dict):
            return
        try:
            idx = int(rec.get("i") or 0)
        except Exception:
            idx = 0
        if idx <= 0:
            return
        partial_pack[idx] = dict(rec)
        _apply_llm_pack_to_grouped_docs(docs2, pack={idx: dict(rec)}, question=q, clear_missing=False)
        if callable(progress_cb):
            try:
                progress_cb(copy.deepcopy(docs2))
            except Exception:
                pass

    try:
        pack_batch = _llm_refs_pack(settings, question=q, docs=docs2)
    except Exception:
        pack_batch = {}

    if isinstance(pack_batch, dict) and pack_batch:
        _apply_llm_pack_to_grouped_docs(docs2, pack=pack_batch, question=q, clear_missing=False)
        if callable(progress_cb):
            try:
                progress_cb(copy.deepcopy(docs2))
            except Exception:
                pass

    ready_ids = {int(i) for i, rec in (pack_batch or {}).items() if isinstance(rec, dict)}
    missing_items = [it for it in items if int(it.get("i") or 0) not in ready_ids]

    arr: list[dict] = [dict(rec) for rec in (pack_batch or {}).values() if isinstance(rec, dict)]
    if missing_items and (not used_multi_paper_docwise_fast_path):
        try:
            arr_retry = _llm_refs_pack_docwise_items(settings, question=q, items=missing_items, on_item=_on_item)
        except Exception:
            arr_retry = []
        for rec in arr_retry:
            if isinstance(rec, dict):
                arr.append(rec)

    ready_ids = {int(x.get("i") or 0) for x in arr if isinstance(x, dict)}
    if not ready_ids:
        for d in docs2:
            if not isinstance(d, dict):
                continue
            m = d.get("meta", {}) or {}
            if str(m.get("ref_pack_state") or "").strip().lower() == "pending":
                m["ref_pack_state"] = "failed"
                d["meta"] = m
    else:
        for i, d in enumerate(docs2, start=1):
            if not isinstance(d, dict):
                continue
            if i in ready_ids:
                continue
            m = d.get("meta", {}) or {}
            if str(m.get("ref_pack_state") or "").strip().lower() == "pending":
                m["ref_pack_state"] = "failed"
                d["meta"] = m
    docs2.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    if top_k_docs is not None:
        try:
            k = max(1, int(top_k_docs))
        except Exception:
            k = len(docs2)
        docs2 = docs2[:k]
    return docs2
