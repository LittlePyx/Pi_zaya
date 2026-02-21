from __future__ import annotations

import difflib
import html
import re
from functools import lru_cache
from typing import Any

import requests


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", flags=re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_YEAR_RE = re.compile(r"^(19\d{2}|20\d{2})$")
_REF_LEAD_RE = re.compile(r"^\s*(?:\[\d{1,4}\]|\d{1,4}\.)\s*")
_REF_BAD_PATTERNS = (
    "copyright",
    "all rights reserved",
    "www.",
    "http://",
    "https://",
    "\\text",
    "acknowledg",
    "grant no",
)


def normalize_title_for_match(title: str) -> str:
    s = (title or "").strip()
    if not s:
        return ""
    s = re.sub(r"\.(pdf|md)$", "", s, flags=re.IGNORECASE)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\u4e00-\u9fff ]+", " ", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    return s


def title_similarity(a: str, b: str) -> float:
    na = normalize_title_for_match(a)
    nb = normalize_title_for_match(b)
    if not na or not nb:
        return 0.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    ta = set(na.split())
    tb = set(nb.split())
    jac = (len(ta & tb) / len(ta | tb)) if (ta and tb) else 0.0
    return min(1.0, (0.72 * seq) + (0.28 * jac))


def _clean_doi(doi: str) -> str:
    d = (doi or "").strip()
    if not d:
        return ""
    d = d.strip(" \t\r\n.,;:()[]{}<>")
    return d


def extract_first_doi(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    m = _DOI_RE.search(s)
    if not m:
        return ""
    return _clean_doi(m.group(0))


def extract_year_hint(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    years = re.findall(r"(19\d{2}|20\d{2})", s)
    if not years:
        return ""
    return str(years[-1])


def extract_first_author_family_hint(text: str) -> str:
    """
    Best-effort extract first author's family name from a reference line.
    Example:
    - "M. J. Sun, M. P. Edgar, ..." -> "sun"
    - "Sun M J, Edgar M P, ..." -> "sun"
    """
    s = _REF_LEAD_RE.sub("", (text or "").strip())
    if not s:
        return ""
    head = s.split(",", 1)[0].strip()
    if not head:
        return ""
    toks = [t for t in re.split(r"\s+", head) if t]
    if not toks:
        return ""
    # Prefer the longest alphabetic token in the first author span.
    best = ""
    for t in toks:
        t2 = re.sub(r"[^A-Za-z\-']", "", t).strip("-'")
        if len(t2) > len(best):
            best = t2
    return best.lower() if len(best) >= 2 else ""


def _clean_reference_for_query(text: str) -> str:
    s = _REF_LEAD_RE.sub("", (text or "").strip())
    s = re.sub(r"\s{2,}", " ", s)
    # Remove very noisy tails.
    s = re.sub(r"(?:\bwww\.[^\s]+|\bhttps?://[^\s]+)", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if len(s) > 360:
        s = s[:360].rstrip(" ,.;:")
    return s


def is_promising_reference_text(text: str) -> bool:
    s = _clean_reference_for_query(text)
    if not s:
        return False
    if len(s) < 24:
        return False
    # Extremely long entries are often merged garbage.
    if len(s) > 420:
        return False
    low = s.lower()
    if any(k in low for k in _REF_BAD_PATTERNS):
        return False
    if not extract_year_hint(s):
        return False
    alpha_words = re.findall(r"[A-Za-z]{2,}", s)
    if len(alpha_words) < 4:
        return False
    return True


def _extract_year(item: dict[str, Any]) -> str:
    for k in ("published-print", "published-online", "issued", "created"):
        dt = item.get(k) or {}
        parts = dt.get("date-parts", [[]])
        if parts and parts[0]:
            y = str(parts[0][0])
            if _YEAR_RE.fullmatch(y):
                return y
    return ""


def _format_authors(item: dict[str, Any]) -> str:
    authors_list = item.get("author", []) or []
    names: list[str] = []
    for a in authors_list:
        if not isinstance(a, dict):
            continue
        family = str(a.get("family") or "").strip()
        given = str(a.get("given") or "").strip()
        if not family:
            continue
        given_clean = re.sub(r"[.,]", "", given).strip()
        initial = given_clean[:1] if given_clean else ""
        names.append(f"{family} {initial}".strip())
    if not names:
        return ""
    if len(names) > 3:
        return ", ".join(names[:3]) + ", et al"
    return ", ".join(names)


def _meta_from_item(item: dict[str, Any], *, fallback_title: str = "") -> dict[str, str]:
    title_list = item.get("title", []) or []
    title = html.unescape(str(title_list[0] if title_list else fallback_title)).strip()
    venue_list = item.get("container-title", []) or []
    venue = html.unescape(str(venue_list[0] if venue_list else "")).strip()
    return {
        "title": title,
        "authors": _format_authors(item) or "[Unknown Authors]",
        "venue": venue,
        "year": _extract_year(item),
        "volume": str(item.get("volume") or "").strip(),
        "issue": str(item.get("issue") or "").strip(),
        "pages": str(item.get("page") or "").strip(),
        "doi": str(item.get("DOI") or "").strip(),
    }


def _candidate_biblio_text(item: dict[str, Any]) -> str:
    meta = _meta_from_item(item)
    parts = [
        str(meta.get("authors") or ""),
        str(meta.get("title") or ""),
        str(meta.get("venue") or ""),
        str(meta.get("year") or ""),
        str(meta.get("volume") or ""),
        str(meta.get("issue") or ""),
        str(meta.get("pages") or ""),
        str(meta.get("doi") or ""),
    ]
    return " ".join(x for x in parts if x).strip()


def _text_similarity(a: str, b: str) -> float:
    na = normalize_title_for_match(a)
    nb = normalize_title_for_match(b)
    if not na or not nb:
        return 0.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    ta = set(na.split())
    tb = set(nb.split())
    jac = (len(ta & tb) / len(ta | tb)) if (ta and tb) else 0.0
    return min(1.0, (0.64 * seq) + (0.36 * jac))


def _author_family_set(item: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for a in (item.get("author", []) or []):
        if not isinstance(a, dict):
            continue
        fam = str(a.get("family") or "").strip().lower()
        fam = re.sub(r"[^a-z\-']", "", fam).strip("-'")
        if fam:
            out.add(fam)
    return out


@lru_cache(maxsize=512)
def _crossref_get_work_by_doi(doi: str) -> dict[str, Any] | None:
    d = _clean_doi(doi)
    if not d:
        return None
    url = f"https://api.crossref.org/works/{d}"
    headers = {"User-Agent": "Pi-zaya-KB/1.0 (Research Assistant)"}
    try:
        resp = requests.get(url, headers=headers, timeout=3.5)
        if resp.status_code != 200:
            return None
        data = resp.json()
        item = data.get("message", {})
        if not isinstance(item, dict):
            return None
        return item
    except Exception:
        return None


def fetch_crossref_work_by_doi(doi: str) -> dict[str, Any] | None:
    d = _clean_doi(doi)
    if not d:
        return None
    item = _crossref_get_work_by_doi(d)
    return item if isinstance(item, dict) else None


def fetch_crossref_references_by_doi(doi: str) -> list[dict[str, Any]]:
    item = fetch_crossref_work_by_doi(doi)
    if not isinstance(item, dict):
        return []
    refs = item.get("reference")
    if not isinstance(refs, list):
        return []
    out: list[dict[str, Any]] = []
    for r in refs:
        if isinstance(r, dict):
            out.append(r)
    return out


@lru_cache(maxsize=1024)
def _crossref_search_title_raw(title: str, rows: int) -> list[dict[str, Any]]:
    q = normalize_title_for_match(title)
    if not q or len(q) < 5:
        return []
    params = {
        "query.title": q,
        "rows": int(max(1, min(8, rows))),
        "select": "author,published-print,published-online,issued,created,container-title,volume,issue,page,DOI,title",
    }
    headers = {"User-Agent": "Pi-zaya-KB/1.0 (Research Assistant)"}
    url = "https://api.crossref.org/works"
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=3.0)
        if resp.status_code != 200:
            return []
        data = resp.json()
        items = data.get("message", {}).get("items", [])
        if not isinstance(items, list):
            return []
        out: list[dict[str, Any]] = []
        for it in items:
            if isinstance(it, dict):
                out.append(it)
        return out
    except Exception:
        return []


@lru_cache(maxsize=1024)
def _crossref_search_bibliographic_raw(reference_text: str, rows: int) -> list[dict[str, Any]]:
    q = _clean_reference_for_query(reference_text)
    q = _WS_RE.sub(" ", q).strip()
    if not q or len(q) < 8:
        return []
    params = {
        "query.bibliographic": q,
        "rows": int(max(1, min(7, rows))),
    }
    headers = {"User-Agent": "Pi-zaya-KB/1.0 (Research Assistant)"}
    url = "https://api.crossref.org/works"
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=2.8)
        if resp.status_code != 200:
            return []
        data = resp.json()
        items = data.get("message", {}).get("items", [])
        if not isinstance(items, list):
            return []
        out: list[dict[str, Any]] = []
        for it in items:
            if isinstance(it, dict):
                out.append(it)
        return out
    except Exception:
        return []


def _venue_similarity(expected: str, got: str) -> float:
    ne = normalize_title_for_match(expected)
    ng = normalize_title_for_match(got)
    if not ne or not ng:
        return 0.0
    if ne == ng:
        return 1.0
    if ne in ng or ng in ne:
        return 0.92
    return title_similarity(ne, ng)


def fetch_best_crossref_meta(
    *,
    query_title: str,
    expected_year: str = "",
    expected_venue: str = "",
    doi_hint: str = "",
    min_score: float = 0.90,
    allow_title_only: bool = False,
) -> dict[str, Any] | None:
    q = (query_title or "").strip()
    y = (expected_year or "").strip()
    v = (expected_venue or "").strip()
    d = _clean_doi(doi_hint)

    if d:
        item = _crossref_get_work_by_doi(d)
        if item:
            meta = _meta_from_item(item, fallback_title=q)
            sim = title_similarity(q, meta.get("title", "")) if q else 1.0
            # If DOI comes from noisy text, require title agreement to trust it.
            if (not q) or (sim >= 0.80):
                out = dict(meta)
                out["match_method"] = "doi"
                out["title_similarity"] = round(sim, 4)
                out["match_score"] = round(max(0.97, sim), 4)
                return out

    if not q:
        return None

    items = _crossref_search_title_raw(q, 5)
    if not items:
        return None

    best_meta: dict[str, Any] | None = None
    best_score = -1.0
    best_title_sim = 0.0
    best_year_match = False
    best_venue_sim = 0.0

    for it in items:
        meta = _meta_from_item(it, fallback_title=q)
        cand_title = meta.get("title", "")
        t_sim = title_similarity(q, cand_title)
        y_match = bool(y and (meta.get("year", "") == y))
        v_sim = _venue_similarity(v, meta.get("venue", "")) if v else 0.0
        score = t_sim

        if y:
            score += 0.06 if y_match else -0.10
        if v:
            score += 0.05 * (2.0 * v_sim - 1.0)

        score = max(0.0, min(1.0, score))
        if score > best_score:
            best_score = score
            best_title_sim = t_sim
            best_year_match = y_match
            best_venue_sim = v_sim
            best_meta = meta

    if not best_meta:
        return None

    # Hard safety gates:
    # - If expected year is known, candidate year must match.
    # - If year is unknown but expected venue is known, venue must match well.
    # - If neither year nor venue is known, allow title-only only when explicitly enabled.
    if y and (not best_year_match):
        return None
    if (not y) and v and (best_venue_sim < 0.90):
        return None
    if (not y) and (not v) and (not allow_title_only):
        return None

    # Strict quality gate: prefer no result over wrong result.
    if best_score < float(min_score):
        return None
    if best_title_sim < 0.88:
        return None
    if allow_title_only and (not y) and (not v):
        # Title-only mode is intentionally stricter.
        if best_title_sim < 0.94:
            return None
        if best_score < max(float(min_score), 0.96):
            return None

    out = dict(best_meta)
    out["match_method"] = "title"
    out["title_similarity"] = round(best_title_sim, 4)
    out["match_score"] = round(best_score, 4)
    return out


def fetch_best_crossref_for_reference(
    *,
    reference_text: str,
    min_score: float = 0.62,
) -> dict[str, Any] | None:
    """
    Resolve noisy bibliography lines using Crossref's bibliographic query.
    This is designed for references without DOI / incomplete titles.
    """
    raw = _clean_reference_for_query(reference_text)
    if not raw or len(raw) < 8:
        return None

    year_hint = extract_year_hint(raw)
    author_hint = extract_first_author_family_hint(raw)
    doi_hint = extract_first_doi(raw)
    if doi_hint:
        by_doi = fetch_best_crossref_meta(
            query_title="",
            doi_hint=doi_hint,
            allow_title_only=False,
            min_score=0.90,
        )
        if isinstance(by_doi, dict):
            out0 = dict(by_doi)
            out0["match_method"] = "doi"
            return out0

    if (not doi_hint) and (not is_promising_reference_text(raw)):
        return None

    items = _crossref_search_bibliographic_raw(raw, 15)
    if not items:
        return None

    best_meta: dict[str, Any] | None = None
    best_score = -1.0
    best_text_sim = 0.0
    best_year_match = False
    best_author_match = False

    for it in items:
        meta = _meta_from_item(it, fallback_title="")
        cand_txt = _candidate_biblio_text(it)
        t_sim = _text_similarity(raw, cand_txt)
        y = str(meta.get("year") or "").strip()
        y_match = bool(year_hint and y and (year_hint == y))
        y_near = False
        if year_hint and y and (not y_match):
            try:
                y_near = abs(int(year_hint) - int(y)) <= 1
            except Exception:
                y_near = False

        author_match = False
        if author_hint:
            fams = _author_family_set(it)
            author_match = author_hint in fams

        score = t_sim
        if year_hint:
            if y_match:
                score += 0.18
            elif y_near:
                score += 0.06
            else:
                score -= 0.12
        if author_hint:
            score += 0.14 if author_match else -0.08
        # Light boost for records with DOI.
        if str(meta.get("doi") or "").strip():
            score += 0.03

        score = max(0.0, min(1.0, score))
        if score > best_score:
            best_score = score
            best_meta = meta
            best_text_sim = t_sim
            best_year_match = y_match
            best_author_match = author_match

    if not best_meta:
        return None
    if best_score < float(min_score):
        return None
    if year_hint and (not best_year_match) and best_score < 0.72:
        return None
    if author_hint and (not best_author_match) and best_score < 0.74:
        return None

    out = dict(best_meta)
    out["match_method"] = "bibliographic"
    out["title_similarity"] = round(best_text_sim, 4)
    out["match_score"] = round(best_score, 4)
    return out
