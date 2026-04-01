from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .chunking import chunk_markdown
from .llm import DeepSeekChat
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
from .source_filters import is_excluded_source_path
from .store import compute_file_sha1
from .tokenize import tokenize

# These callbacks are injected by app.py so this module can reuse the shared runtime cache.
_CACHE_GET: Callable[[str, str], Any] = lambda _bucket, _key: None
_CACHE_SET: Callable[..., None] = lambda _bucket, _key, _val, **_kw: None


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
    return bool(re.search(r"(参考文献|引用|cite|citation|reference|bibliography)", q, flags=re.I))


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
    ("figure", re.compile(r"\bfig(?:ure)?\.?\s*([0-9ivxlcdm]+)\b", flags=re.I)),
    ("table", re.compile(r"\btable\.?\s*([0-9ivxlcdm]+)\b", flags=re.I)),
    ("equation", re.compile(r"\b(?:eq(?:uation)?|formula)\.?\s*[\(\[（]?\s*([0-9ivxlcdm]+)\s*[\)\]）]?", flags=re.I)),
    ("theorem", re.compile(r"\btheorem\.?\s*([0-9ivxlcdm]+)\b", flags=re.I)),
    ("lemma", re.compile(r"\blemma\.?\s*([0-9ivxlcdm]+)\b", flags=re.I)),
    ("definition", re.compile(r"\bdefinition\.?\s*([0-9ivxlcdm]+)\b", flags=re.I)),
    ("proposition", re.compile(r"\bproposition\.?\s*([0-9ivxlcdm]+)\b", flags=re.I)),
    ("corollary", re.compile(r"\bcorollary\.?\s*([0-9ivxlcdm]+)\b", flags=re.I)),
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


def _extract_explicit_anchor_hint(question: str) -> dict[str, object]:
    q = str(question or "").strip()
    if not q:
        return {}
    for kind, pat in _ANCHOR_PATTERNS:
        m = pat.search(q)
        if not m:
            continue
        num = _parse_anchor_number(m.group(1))
        if num is None or num <= 0:
            continue
        phrases: list[str] = []
        labels = _ANCHOR_KIND_LABELS.get(kind) or ()
        for lab in labels:
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
        return {
            "kind": kind,
            "number": int(num),
            "label": f"{kind} {num}",
            "phrases": list(dict.fromkeys([p for p in phrases if str(p or "").strip()])),
        }
    return {}


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

    prompt_tokens = [t for t in tokenize(prompt_norm) if len(t) >= 3 and t not in _DOC_HINT_STOP_TOKENS]
    src_tokens = [
        t
        for t in tokenize(_norm_text_for_match(" ".join(str(x or "") for x in candidates)))
        if len(t) >= 3 and t not in _DOC_HINT_STOP_TOKENS
    ]
    if prompt_tokens and src_tokens:
        inter = set(prompt_tokens) & set(src_tokens)
        if len(inter) >= 3:
            score += 2.0 + min(3.0, 0.6 * len(inter))
            ratio = len(inter) / max(1, len(set(src_tokens)))
            if ratio >= 0.55:
                score += 2.0
    return float(score)


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
    if (not kind) or (num <= 0):
        return 0.0
    low = str(text or "").lower()
    if not low:
        return 0.0
    score = 0.0
    patterns_by_kind = {
        "figure": rf"(?:fig(?:ure)?\.?\s*{num}\b|图\s*{num}\b|图{num}\b|第\s*{num}\s*张图)",
        "table": rf"(?:table\.?\s*{num}\b|表\s*{num}\b|表{num}\b|第\s*{num}\s*表)",
        "equation": rf"(?:eq(?:uation)?\.?\s*{num}\b|formula\s*{num}\b|公式\s*{num}\b|公式{num}\b|式\s*{num}\b|式{num}\b|[\(（]\s*{num}\s*[\)）]|\\tag\{{\s*{num}\s*\}})",
        "theorem": rf"(?:theorem\.?\s*{num}\b|定理\s*{num}\b|定理{num}\b)",
        "lemma": rf"(?:lemma\.?\s*{num}\b|引理\s*{num}\b|引理{num}\b)",
        "definition": rf"(?:definition\.?\s*{num}\b|定义\s*{num}\b|定义{num}\b)",
        "proposition": rf"(?:proposition\.?\s*{num}\b|命题\s*{num}\b|命题{num}\b)",
        "corollary": rf"(?:corollary\.?\s*{num}\b|推论\s*{num}\b|推论{num}\b)",
    }
    pat = patterns_by_kind.get(kind)
    if pat and re.search(pat, low, flags=re.I):
        score += 12.0
    labels = _ANCHOR_KIND_LABELS.get(kind) or ()
    if any(lab in low for lab in labels):
        score += 2.0
    if str(num) in low:
        score += 1.2
    return score


def _anchor_regexes(anchor_hint: dict[str, object]) -> list[re.Pattern[str]]:
    if not isinstance(anchor_hint, dict) or not anchor_hint:
        return []
    kind = str(anchor_hint.get("kind") or "").strip().lower()
    try:
        num = int(anchor_hint.get("number") or 0)
    except Exception:
        num = 0
    if (not kind) or (num <= 0):
        return []
    raw_patterns = {
        "figure": [
            rf"fig(?:ure)?\.?\s*{num}\b",
            rf"第\s*{num}\s*张?图",
            rf"图\s*{num}\b",
            rf"图{num}\b",
        ],
        "table": [
            rf"table\.?\s*{num}\b",
            rf"第\s*{num}\s*表",
            rf"表\s*{num}\b",
            rf"表{num}\b",
        ],
        "equation": [
            rf"eq(?:uation)?\.?\s*{num}\b",
            rf"formula\s*{num}\b",
            rf"(?:公式|式)\s*{num}\b",
            rf"(?:公式|式){num}\b",
            rf"[\(（]\s*{num}\s*[\)）]",
            rf"\\tag\{{\s*{num}\s*\}}",
        ],
        "theorem": [rf"theorem\.?\s*{num}\b", rf"定理\s*{num}\b", rf"定理{num}\b"],
        "lemma": [rf"lemma\.?\s*{num}\b", rf"引理\s*{num}\b", rf"引理{num}\b"],
        "definition": [rf"definition\.?\s*{num}\b", rf"定义\s*{num}\b", rf"定义{num}\b"],
        "proposition": [rf"proposition\.?\s*{num}\b", rf"命题\s*{num}\b", rf"命题{num}\b"],
        "corollary": [rf"corollary\.?\s*{num}\b", rf"推论\s*{num}\b", rf"推论{num}\b"],
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
        ("\u8868", "table"),
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
    ]
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
            max_retries=0,
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
            "浣犳槸涓ユ牸鐨勫鏈绱㈤噸鎺掑櫒銆俓n"
            "鍙兘杈撳嚭 JSON锛歿\"score\":number,\"why\":string}銆俓n"
            "瑙勫垯锛歕n"
            "- score 涓?0..100锛屽繀椤诲弽鏄犫€滆繖浜涚墖娈垫槸鍚︾洿鎺ュ洖绛旂敤鎴烽棶棰樷€濄€俓n"
            "- 閬囧埌鏈鍋囨湅鍙嬭鎵ｅ垎锛堝 single-shot vs single-pixel锛夈€俓n"
            "- 鍙兘鏍规嵁 snippets/headings 鍒ゆ柇锛屼笉鑳芥牴鎹枃浠跺悕鍒ゆ柇銆俓n"
            "- why锛?= 18 涓瓧锛屽啓娓呮涓轰粈涔堛€俓n"
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

def _search_hits_with_fallback(
    prompt_text: str,
    retriever: BM25Retriever,
    top_k: int,
    settings,
    *,
    allow_translate: bool = True,
) -> tuple[list[dict], list[float], str, bool]:
    """
    Returns: (hits_raw, scores, used_query, used_translation)
    """
    q1 = (prompt_text or "").strip()
    hits1 = retriever.search(q1, top_k=max(10, top_k * 6)) if q1 else []
    hits1 = [h for h in (hits1 or []) if not _is_temp_source_path(str((h.get("meta") or {}).get("source_path") or ""))]
    scores1 = [float(h.get("score", 0.0) or 0.0) for h in hits1]
    best1 = float(max(scores1) if scores1 else 0.0)

    # If the query is CJK-only, BM25 over English corpora can return all-zeros (but still returns arbitrary docs).
    # Try translating to English to get meaningful retrieval.
    used_trans = False
    q2 = _translate_query_for_search(settings, q1) if bool(allow_translate) else None
    if q2:
        hits2 = retriever.search(q2, top_k=max(10, top_k * 6))
        hits2 = [h for h in (hits2 or []) if not _is_temp_source_path(str((h.get("meta") or {}).get("source_path") or ""))]
        scores2 = [float(h.get("score", 0.0) or 0.0) for h in hits2]
        best2 = float(max(scores2) if scores2 else 0.0)
        # Prefer translated retrieval when it yields a more meaningful signal.
        if (not hits1 and hits2) or (best1 <= 0.0 and best2 > 0.0) or (best2 > (best1 * 1.25 + 0.10)):
            return hits2, scores2, q2, True

    return hits1, scores1, q1, used_trans

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
    anchor_hint = _extract_explicit_anchor_hint(prompt_text or deep_query or "")
    for src, hs in by_doc.items():
        try:
            best_score = float(max(float(h.get("score", 0.0) or 0.0) for h in hs))
        except Exception:
            best_score = 0.0
        doc_hint_score = _source_prompt_match_score(prompt_text or deep_query or "", src)
        doc_hint_scores[src] = float(doc_hint_score)
        doc_order.append((best_score + (1.6 * doc_hint_score), src))
    doc_order.sort(key=lambda x: x[0], reverse=True)

    docs: list[dict] = []
    profile = _query_term_profile(prompt_text, deep_query or "")
    nav_question = (prompt_text or deep_query or "").strip()
    # Bound work: only consider a limited number of candidate docs.
    max_docs_consider = max(int(top_k_docs) * 2, 12)
    # Quality-first refs: if deep_read is enabled, expand more candidate docs than before.
    deep_expand_docs = min(max_docs_consider, max(int(top_k_docs) * 2, 6)) if deep_read else 0
    for _best, src in doc_order[:max_docs_consider]:
        hs = by_doc.get(src) or []
        hs2 = sorted(hs, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        best_score = float(hs2[0].get("score", 0.0) or 0.0) if hs2 else 0.0
        doc_hint_score = float(doc_hint_scores.get(src, 0.0) or 0.0)
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
                sc_adj = sc_h + _heading_intent_bonus_for_question(hp or top, nav_question) + (0.35 * anchor_bonus)
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
                ):
                    t = ""
            if t:
                if force_anchor_focus and anchor_bonus > 0.0:
                    snippet_anchor_bonus[t] = max(float(snippet_anchor_bonus.get(t, 0.0) or 0.0), float(anchor_bonus))
                if t not in snippets:
                    snippets.append(t)

        # Optional deep-read for better section targeting + aspects + ranking.
        deep_best = 0.0
        do_deep_read = (deep_read and deep_query and (len(docs) < deep_expand_docs)) or force_anchor_focus
        if do_deep_read:
            read_query = (anchor_focus_query or deep_query or prompt_text or "").strip()
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
                    ):
                        tx = ""
                if tx:
                    if force_anchor_focus and anchor_bonus_ex > 0.0:
                        snippet_anchor_bonus[tx] = max(float(snippet_anchor_bonus.get(tx, 0.0) or 0.0), float(anchor_bonus_ex))
                    if tx not in snippets:
                        snippets.append(tx)
                try:
                    deep_best = max(deep_best, float(ex.get("score", 0.0) or 0.0) + (0.20 * anchor_bonus_ex))
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
        q_for_pick = (anchor_focus_query or deep_query or prompt_text or "").strip()
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
        combined = (
            (0.75 * best_score)
            + (0.25 * deep_scaled)
            + term_bonus
            + (1.5 * doc_hint_score)
            + (0.35 * anchor_best)
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
            meta_out["ref_snippets"] = snippets[:3]
        meta_out["ref_show_snippets"] = show_snips[:3]
        meta_out["ref_overview_snippets"] = [x for x in overview_snips if str(x or "").strip()][:3]
        meta_out["ref_locs"] = locs2
        meta_out["ref_headings"] = headings_for_pack
        if doc_hint_score > 0.0:
            meta_out["explicit_doc_match_score"] = float(doc_hint_score)
        if force_anchor_focus and anchor_hint:
            meta_out["anchor_target_kind"] = str(anchor_hint.get("kind") or "")
            meta_out["anchor_target_number"] = int(anchor_hint.get("number") or 0)
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
    wants_refs = bool(re.search(r"(鍙傝€冩枃鐚畖寮曠敤|cite|citation|reference)", q, flags=re.I))
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
        rec = _run_docwise_once(it, timeout_s=18.0, max_tokens=280)
        if _is_usable_docwise_result(rec or {}):
            return rec
        rec_retry = _run_docwise_once(it, timeout_s=24.0, max_tokens=420)
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
    sem_keep_min = max(28.0, best_llm - 35.0)
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
    if missing_items:
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
