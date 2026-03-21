from __future__ import annotations

import re

from kb.citation_meta import extract_first_doi, extract_year_hint

_RANGE_DASH_CLASS = r"\-\u2013\u2014\u2212"
_INPAPER_NUMERIC_RE = re.compile(rf"\[(\d{{1,4}}(?:\s*(?:[{_RANGE_DASH_CLASS},])\s*\d{{1,4}})*)\]")
_AUTHOR_ETAL_RE = re.compile(r"\b([A-Z][A-Za-z'`-]{1,40})\s+et\s+al\.?\b", flags=re.I)
_AUTHOR_YEAR_PAREN_RE = re.compile(r"\b([A-Z][A-Za-z'`-]{1,40})\s*\(\s*((?:19|20)\d{2})\s*\)")
_AUTHOR_YEAR_INLINE_RE = re.compile(r"\b([A-Z][A-Za-z'`-]{1,40})\s*,?\s+((?:19|20)\d{2})\b")


def parse_ref_num_set(spec: str, *, max_items: int = 48) -> list[int]:
    text = str(spec or "").strip()
    if not text:
        return []
    out: list[int] = []
    seen: set[int] = set()
    for part in re.split(r"\s*,\s*", text):
        s = str(part or "").strip()
        if not s:
            continue
        m = re.fullmatch(rf"(\d{{1,4}})\s*(?:[{_RANGE_DASH_CLASS}])\s*(\d{{1,4}})", s)
        if m:
            try:
                a = int(m.group(1))
                b = int(m.group(2))
            except Exception:
                continue
            if a <= 0 or b <= 0:
                continue
            lo = min(a, b)
            hi = max(a, b)
            if (hi - lo) > 12:
                continue
            for n in range(lo, hi + 1):
                if n in seen:
                    continue
                seen.add(n)
                out.append(n)
                if len(out) >= max(1, int(max_items)):
                    return out
            continue
        try:
            n = int(s)
        except Exception:
            continue
        if n <= 0 or n in seen:
            continue
        seen.add(n)
        out.append(n)
        if len(out) >= max(1, int(max_items)):
            return out
    return out


def extract_candidate_ref_nums_from_hits(
    answer_hits: list[dict],
    *,
    source_path: str = "",
    max_candidates: int = 48,
) -> list[int]:
    want_src = str(source_path or "").strip()
    out: list[int] = []
    seen: set[int] = set()

    def _push_from_text(text: str) -> None:
        nonlocal out
        for m in _INPAPER_NUMERIC_RE.finditer(str(text or "")):
            for n in parse_ref_num_set(m.group(1), max_items=max_candidates):
                if n in seen:
                    continue
                seen.add(n)
                out.append(n)
                if len(out) >= max(1, int(max_candidates)):
                    return

    for hit in answer_hits or []:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        src = str(meta.get("source_path") or "").strip()
        if want_src and src and src != want_src:
            continue
        _push_from_text(str(hit.get("text") or ""))
        if len(out) >= max(1, int(max_candidates)):
            return out
        snippets = meta.get("ref_show_snippets")
        if isinstance(snippets, list):
            for item in snippets:
                _push_from_text(str(item or ""))
                if len(out) >= max(1, int(max_candidates)):
                    return out
    return out


def _trim_candidate_cue_text(text: str, *, max_chars: int) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s:
        return ""
    s = s.replace("|", "/")
    try:
        limit = max(24, int(max_chars))
    except Exception:
        limit = 180
    if len(s) <= limit:
        return s
    m = _INPAPER_NUMERIC_RE.search(s)
    if not m:
        return s[: max(0, limit - 3)].rstrip() + "..."
    start = max(0, int(m.start()) - max(0, limit // 3))
    end = min(len(s), start + limit)
    chunk = s[start:end].strip()
    if start > 0:
        chunk = "..." + chunk.lstrip()
    if end < len(s):
        chunk = chunk.rstrip() + "..."
    return chunk


def extract_candidate_ref_cue_texts(
    hit: dict,
    *,
    max_cues: int = 2,
    max_chars: int = 180,
) -> list[str]:
    if not isinstance(hit, dict):
        return []
    try:
        limit = max(1, int(max_cues))
    except Exception:
        limit = 2

    meta = hit.get("meta", {}) or {}
    texts: list[str] = []
    primary = str(hit.get("text") or "").strip()
    if primary:
        texts.append(primary)
    snippets = meta.get("ref_show_snippets")
    if isinstance(snippets, list):
        for item in snippets:
            s = str(item or "").strip()
            if s:
                texts.append(s)

    out: list[str] = []
    seen: set[str] = set()
    for raw in texts:
        if not _INPAPER_NUMERIC_RE.search(raw):
            continue
        cue = _trim_candidate_cue_text(raw, max_chars=max_chars)
        if not cue:
            continue
        key = cue.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cue)
        if len(out) >= limit:
            break
    return out


def extract_citation_context_hints(answer_text: str, *, token_start: int, token_end: int) -> dict[str, object]:
    text = str(answer_text or "")
    st = max(0, int(token_start) - 260)
    ed = min(len(text), int(token_end) + 120)
    window = str(text[st:ed] or "")
    left = str(text[max(0, int(token_start) - 220) : int(token_start)] or "")

    doi_hint = str(extract_first_doi(window) or "").strip().lower()
    year_hint = str(extract_year_hint(window) or extract_year_hint(left) or "").strip()

    author_hint = ""
    author_confident = False
    for pattern in (_AUTHOR_ETAL_RE, _AUTHOR_YEAR_PAREN_RE, _AUTHOR_YEAR_INLINE_RE):
        matches = list(pattern.finditer(window))
        if not matches:
            continue
        m = matches[-1]
        author_hint = str(m.group(1) or "").strip().lower()
        if author_hint in {"the", "this", "that", "these", "those", "figure", "table", "section", "equation", "model", "method"}:
            author_hint = ""
            continue
        author_confident = pattern is not _AUTHOR_YEAR_INLINE_RE or bool(year_hint)
        break

    return {
        "doi": doi_hint,
        "year": year_hint,
        "author": author_hint,
        "author_confident": bool(author_confident and author_hint),
        "window": window,
    }


def reference_alignment_score(ref: dict, hints: dict[str, object]) -> float:
    if not isinstance(ref, dict):
        return float("-inf")
    doi_hint = str((hints or {}).get("doi") or "").strip().lower()
    year_hint = str((hints or {}).get("year") or "").strip()
    author_hint = str((hints or {}).get("author") or "").strip().lower()
    author_confident = bool((hints or {}).get("author_confident"))

    ref_doi = str(ref.get("doi") or "").strip().lower()
    if (not ref_doi) and str(ref.get("raw") or "").strip():
        ref_doi = str(extract_first_doi(str(ref.get("raw") or "")) or "").strip().lower()
    ref_year = str(ref.get("year") or extract_year_hint(" ".join([str(ref.get("raw") or ""), str(ref.get("cite_fmt") or ""), str(ref.get("title") or "")])) or "").strip()
    ref_hay = " ".join(
        [
            str(ref.get("authors") or "").strip(),
            str(ref.get("title") or "").strip(),
            str(ref.get("venue") or "").strip(),
            str(ref.get("raw") or "").strip(),
        ]
    ).lower()
    ref_hay_norm = re.sub(r"[^a-z0-9]+", " ", ref_hay).strip()

    score = 0.0
    if doi_hint and ref_doi:
        if doi_hint == ref_doi:
            score += 8.0
        else:
            score -= 10.0
    if year_hint and ref_year:
        if year_hint == ref_year:
            score += 2.0
        else:
            score -= 3.0 if (author_confident and author_hint) else 1.0
    if author_hint and ref_hay_norm:
        if author_hint in ref_hay_norm:
            score += 2.5 if author_confident else 1.0
        elif author_confident and year_hint:
            score -= 3.0
    return score


def has_explicit_reference_conflict(ref: dict, hints: dict[str, object]) -> bool:
    if not isinstance(ref, dict):
        return False
    doi_hint = str((hints or {}).get("doi") or "").strip().lower()
    year_hint = str((hints or {}).get("year") or "").strip()
    author_hint = str((hints or {}).get("author") or "").strip().lower()
    author_confident = bool((hints or {}).get("author_confident"))

    ref_doi = str(ref.get("doi") or "").strip().lower()
    if (not ref_doi) and str(ref.get("raw") or "").strip():
        ref_doi = str(extract_first_doi(str(ref.get("raw") or "")) or "").strip().lower()
    if doi_hint and ref_doi and doi_hint != ref_doi:
        return True

    ref_year = str(ref.get("year") or extract_year_hint(" ".join([str(ref.get("raw") or ""), str(ref.get("cite_fmt") or ""), str(ref.get("title") or "")])) or "").strip()
    if author_confident and author_hint and year_hint:
        if ref_year and ref_year != year_hint:
            return True
        ref_hay = " ".join([str(ref.get("authors") or ""), str(ref.get("raw") or "")]).lower()
        ref_hay_norm = re.sub(r"[^a-z0-9]+", " ", ref_hay).strip()
        if ref_hay_norm and author_hint not in ref_hay_norm:
            return True
    return False
