from __future__ import annotations

import re

from kb.citation_meta import extract_first_doi, extract_year_hint

_RANGE_DASH_CLASS = r"\-\u2013\u2014\u2212"
_INPAPER_NUMERIC_RE = re.compile(rf"\[(\d{{1,4}}(?:\s*(?:[{_RANGE_DASH_CLASS},])\s*\d{{1,4}})*)\]")
_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPERSCRIPT_TRANSLATION = str.maketrans(
    {
        **{char: str(index) for index, char in enumerate(_SUPERSCRIPT_DIGITS)},
        "⁻": "-",
        "⁺": "+",
    }
)
_SUPERSCRIPT_NUMERIC_RE = re.compile(
    rf"(?<![{_SUPERSCRIPT_DIGITS}⁻⁺])"
    rf"([{_SUPERSCRIPT_DIGITS}]{{1,4}}(?:\s*(?:[⁻{_RANGE_DASH_CLASS},;，、；])\s*[{_SUPERSCRIPT_DIGITS}]{{1,4}})*)"
    rf"(?![{_SUPERSCRIPT_DIGITS}⁻⁺ⁱⁿ])"
)
_SUPERSCRIPT_UNIT_TAIL_RE = re.compile(
    r"(?:^|[^A-Za-zµμΩ])(?:mm|cm|nm|pm|µm|μm|km|m|s|ms|µs|μs|ns|Hz|kHz|MHz|GHz|W|mW|A|V|K|Pa|mol)$",
    flags=re.IGNORECASE,
)
_LATIN_SURNAME_RE = r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'`-]{1,40}"
_AUTHOR_ETAL_RE = re.compile(r"\b([A-Z][A-Za-z'`-]{1,40})\s+et\s+al\.?\b", flags=re.I)
_AUTHOR_YEAR_PAREN_RE = re.compile(r"\b([A-Z][A-Za-z'`-]{1,40})\s*\(\s*((?:19|20)\d{2})\s*\)")
_AUTHOR_YEAR_CONNECTOR_RE = re.compile(
    rf"\b(?:{_LATIN_SURNAME_RE}\s*(?:(?i:and)|&|和|与|、)\s*)?"
    rf"({_LATIN_SURNAME_RE})\s*(?:,?\s*(?:(?i:in)|于|在)\s*)?((?:19|20)\d{{2}})\b"
)
_AUTHOR_YEAR_INLINE_RE = re.compile(r"\b([A-Z][A-Za-z'`-]{1,40})\s*,?\s+((?:19|20)\d{2})\b")


def parse_ref_num_set(spec: str, *, max_items: int = 48) -> list[int]:
    text = str(spec or "").translate(_SUPERSCRIPT_TRANSLATION).strip()
    text = re.sub(r"[，、;；]", ",", text)
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


def _inside_markdown_math(text: str, offset: int) -> bool:
    prefix = str(text or "")[: max(0, int(offset))]
    return len(re.findall(r"(?<!\\)\$", prefix)) % 2 == 1


def _looks_like_superscript_citation(text: str, start: int, end: int, spec: str) -> bool:
    source = str(text or "")
    before = source[start - 1] if start > 0 else ""
    after = source[end] if end < len(source) else ""
    if before in {"⁻", "⁺", "^", "_"} or before.isdigit():
        return False
    if after and (after.isalnum() or after in _SUPERSCRIPT_DIGITS or after in "ⁱⁿ"):
        return False
    if _inside_markdown_math(source, start):
        return False
    prefix = source[:start].rstrip()
    if _SUPERSCRIPT_UNIT_TAIL_RE.search(prefix):
        return False
    token_match = re.search(r"([A-Za-z])$", prefix)
    if token_match and len(parse_ref_num_set(spec, max_items=16)) == 1:
        # A lone superscript after a one-letter variable is overwhelmingly likely
        # to be an exponent (R², x³), not a bibliography marker.
        previous_token = re.search(r"([A-Za-z]+)$", prefix)
        if previous_token and len(previous_token.group(1)) == 1:
            return False
    return bool(parse_ref_num_set(spec, max_items=64))


def _looks_like_bracket_citation(text: str, start: int, spec: str) -> bool:
    marker = str(spec or "").strip()
    if re.fullmatch(r"\d{1,3},\d{3}", marker):
        return False
    numbers = [part for part in re.split(r"\s*[,;]\s*", marker) if part]
    if len(numbers) >= 3:
        prefix = str(text or "")[max(0, int(start) - 100) : int(start)]
        if re.search(
            r"\b(?:dimensions?|shape|size|kernel|weights?|vector|matrix|tensor|resolution)"
            r"\b[^.!?]{0,70}$",
            prefix,
            flags=re.IGNORECASE,
        ):
            return False
    return True


def iter_inpaper_numeric_citations(text: str) -> list[tuple[str, int, int, str]]:
    """Return numeric citation specs as ``(spec, start, end, style)`` rows.

    Besides bracket markers, Nature-style Unicode superscripts such as
    ``³⁰⁻³³`` and ``⁴³`` are supported. Unit exponents and inline math are
    deliberately excluded.
    """

    source = str(text or "")
    out: list[tuple[str, int, int, str]] = []
    for match in _INPAPER_NUMERIC_RE.finditer(source):
        if _inside_markdown_math(source, int(match.start())):
            continue
        if not _looks_like_bracket_citation(source, int(match.start()), str(match.group(1) or "")):
            continue
        out.append((str(match.group(1) or ""), int(match.start()), int(match.end()), "bracket"))
    for match in _SUPERSCRIPT_NUMERIC_RE.finditer(source):
        spec = str(match.group(1) or "")
        start = int(match.start())
        end = int(match.end())
        if not _looks_like_superscript_citation(source, start, end, spec):
            continue
        out.append((spec, start, end, "unicode_superscript"))
    out.sort(key=lambda item: (int(item[1]), int(item[2])))
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
        for spec, _start, _end, _style in iter_inpaper_numeric_citations(str(text or "")):
            for n in parse_ref_num_set(spec, max_items=max_candidates):
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
    markers = iter_inpaper_numeric_citations(s)
    if not markers:
        return s[: max(0, limit - 3)].rstrip() + "..."
    start = max(0, int(markers[0][1]) - max(0, limit // 3))
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
        if not iter_inpaper_numeric_citations(raw):
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
    for pattern in (_AUTHOR_ETAL_RE, _AUTHOR_YEAR_PAREN_RE, _AUTHOR_YEAR_CONNECTOR_RE, _AUTHOR_YEAR_INLINE_RE):
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
