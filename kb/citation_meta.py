from __future__ import annotations

import difflib
import html
import os
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
_VENUE_QUERY_ALIASES: tuple[tuple[str, str], ...] = (
    (r"\bNat\.\s*Commun\.?\b", "Nature Communications"),
    (r"\bNat\.\s*Photonics\b", "Nature Photonics"),
    (r"\bNat\.\s*Methods\b", "Nature Methods"),
    (r"\bNat\.\s*Phys\.?\b", "Nature Physics"),
    (r"\bNat\.\s*Biomed\.\s*Eng\.?\b", "Nature Biomedical Engineering"),
    (r"\bSci\.\s*Adv\.?\b", "Science Advances"),
    (r"\bLight\s+Sci\.\s*Appl\.?\b", "Light Science Applications"),
    (r"\bPhys\.\s*Rev\.\s*A\b", "Physical Review A"),
    (r"\bPhys\.\s*Rev\.\s*Lett\.?\b", "Physical Review Letters"),
    (r"\bOpt\.\s*Express\b", "Optics Express"),
    (r"\bOpt\.\s*Lett\.?\b", "Optics Letters"),
    (r"\bOpt\.\s*Laser\s+Eng\.?\b", "Optics and Lasers in Engineering"),
    (r"\bOpt\.\s*Laser\s+Technol\.?\b", "Optics and Laser Technology"),
    (r"\bAppl\.\s*Phys\.\s*Lett\.?\b", "Applied Physics Letters"),
    (r"\bPhotonics\s+Res\.?\b", "Photonics Research"),
    (r"\bLaser\s+Photonics\s+Rev\.?\b", "Laser and Photonics Reviews"),
    (r"\bIEEE\s+Trans\.\s*Inform\.\s*Theory\b", "IEEE Transactions on Information Theory"),
    (r"\bIEEE\s+Trans\.\s*Inf\.\s*Theory\b", "IEEE Transactions on Information Theory"),
    (r"\bIEEE\s+Signal\s+Proc\.\s*Mag\.?\b", "IEEE Signal Processing Magazine"),
    (r"\bIEEE\s+Signal\s+Process\.\s*Mag\.?\b", "IEEE Signal Processing Magazine"),
    (r"\bIEEE\s+Trans\.\s*Pattern\s+Anal\.\s*Mach\.\s*Intell\.?\b", "IEEE Transactions on Pattern Analysis and Machine Intelligence"),
    (r"\bIEEE\s+Trans\.\s*Comput\.\s*Imag\.?\b", "IEEE Transactions on Computational Imaging"),
    (r"\bJ\.\s*Lightwave\s+Technol\.?\b", "Journal of Lightwave Technology"),
    (r"\bJ\.\s*Opt\.\s*Soc\.\s*Am\.\s*A\b", "Journal of the Optical Society of America A"),
    (r"\bAppl\.\s*Optics\b", "Applied Optics"),
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


def _normalize_doi_like(doi: str) -> str:
    d = (doi or "").strip()
    if not d:
        return ""
    d = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", d, flags=re.IGNORECASE)
    return _clean_doi(d)


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


def _strip_reference_query_markup(text: str) -> str:
    s = str(text or "")
    if not s:
        return ""
    s = re.sub(r"<!--\s*kb_page:\s*\d+\s*-->", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    s = s.replace("\u9225?", "-").replace("\u922d?", "-")
    return s


def _expand_reference_venue_aliases(text: str) -> str:
    s = str(text or "")
    if not s:
        return ""
    for pattern, replacement in _VENUE_QUERY_ALIASES:
        s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)
    return s


def _reference_hint_text(text: str) -> str:
    s = _REF_LEAD_RE.sub("", (text or "").strip())
    s = _strip_reference_query_markup(s)
    s = _expand_reference_venue_aliases(s)
    return _WS_RE.sub(" ", s).strip()


def _clean_reference_for_query(text: str) -> str:
    s = _REF_LEAD_RE.sub("", (text or "").strip())
    s = _strip_reference_query_markup(s)
    s = _expand_reference_venue_aliases(s)
    s = re.sub(r"[,;:(){}\[\]]", " ", s)
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
    if not venue:
        publisher = html.unescape(str(item.get("publisher") or "")).strip()
        if publisher:
            venue = publisher
    if not venue:
        institution = item.get("institution") or []
        if isinstance(institution, list) and institution:
            first = institution[0]
            if isinstance(first, dict):
                venue = html.unescape(str(first.get("name") or "")).strip()
            else:
                venue = html.unescape(str(first or "")).strip()
    return {
        "title": title,
        "authors": _format_authors(item) or "[Unknown Authors]",
        "venue": venue,
        "year": _extract_year(item),
        "volume": str(item.get("volume") or "").strip(),
        "issue": str(item.get("issue") or "").strip(),
        "pages": str(item.get("page") or item.get("article-number") or "").strip(),
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


def _token_set(text: str) -> set[str]:
    return set(normalize_title_for_match(text).split())


def _page_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for token in re.findall(r"[A-Za-z]?\d{1,8}[A-Za-z]?", str(text or "")):
        t = token.strip().lower()
        if t:
            out.add(t)
    return out


def _structured_biblio_score(
    raw: str,
    item: dict[str, Any],
    meta: dict[str, Any],
    *,
    year_hint: str,
    author_hint: str,
) -> tuple[float, bool, bool]:
    raw_clean = _clean_reference_for_query(raw)
    raw_tokens = _token_set(raw_clean)
    y = str(meta.get("year") or "").strip()
    y_match = bool(year_hint and y and (year_hint == y))
    author_match = False
    if author_hint:
        author_match = author_hint in _author_family_set(item)
    venue = str(meta.get("venue") or "").strip()
    venue_sim = _venue_similarity(raw_clean, venue) if venue else 0.0
    venue_match = bool(venue_sim >= 0.88)

    volume = normalize_title_for_match(str(meta.get("volume") or ""))
    volume_match = bool(volume and volume in raw_tokens)
    page_match = False
    for token in _page_tokens(str(meta.get("pages") or "")):
        if token in raw_tokens:
            page_match = True
            break

    score = 0.0
    if y_match:
        score += 0.24
    if author_match:
        score += 0.22
    if venue_match:
        score += 0.20
    elif venue_sim >= 0.72:
        score += 0.10
    if volume_match:
        score += 0.12
    if page_match:
        score += 0.18
    if str(meta.get("doi") or "").strip():
        score += 0.04
    return min(1.0, score), y_match, author_match


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


def _openalex_author_display_name(author_ship: dict[str, Any]) -> str:
    author = author_ship.get("author") if isinstance(author_ship, dict) else None
    if not isinstance(author, dict):
        return ""
    return html.unescape(str(author.get("display_name") or "")).strip()


def _openalex_author_family_set(item: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    authorships = item.get("authorships") if isinstance(item, dict) else None
    if not isinstance(authorships, list):
        return out
    for authorship in authorships:
        name = _openalex_author_display_name(authorship if isinstance(authorship, dict) else {})
        if not name:
            continue
        token = name.replace(",", " ").split()
        fam = token[-1] if token else ""
        fam = re.sub(r"[^a-zA-Z\-']", "", fam).strip("-'").lower()
        if fam:
            out.add(fam)
    return out


def _format_openalex_authors(item: dict[str, Any]) -> str:
    authorships = item.get("authorships") if isinstance(item, dict) else None
    if not isinstance(authorships, list):
        return ""
    names: list[str] = []
    for authorship in authorships:
        name = _openalex_author_display_name(authorship if isinstance(authorship, dict) else {})
        if name:
            names.append(name)
    if not names:
        return ""
    if len(names) > 3:
        return ", ".join(names[:3]) + ", et al"
    return ", ".join(names)


def _openalex_source_name(item: dict[str, Any]) -> str:
    for location_key in ("primary_location", "best_oa_location"):
        location = item.get(location_key) if isinstance(item, dict) else None
        if not isinstance(location, dict):
            continue
        source = location.get("source")
        if isinstance(source, dict):
            name = html.unescape(str(source.get("display_name") or "")).strip()
            if name:
                return name
    locations = item.get("locations") if isinstance(item, dict) else None
    if isinstance(locations, list):
        for location in locations:
            if not isinstance(location, dict):
                continue
            source = location.get("source")
            if isinstance(source, dict):
                name = html.unescape(str(source.get("display_name") or "")).strip()
                if name:
                    return name
    return ""


def _openalex_pages(item: dict[str, Any]) -> str:
    biblio = item.get("biblio") if isinstance(item, dict) else None
    if not isinstance(biblio, dict):
        return ""
    first = str(biblio.get("first_page") or "").strip()
    last = str(biblio.get("last_page") or "").strip()
    if first and last and first != last:
        return f"{first}-{last}"
    return first or last


def _meta_from_openalex_item(item: dict[str, Any], *, fallback_title: str = "") -> dict[str, str]:
    title = html.unescape(str(item.get("title") or fallback_title or "")).strip()
    biblio = item.get("biblio") if isinstance(item, dict) else None
    if not isinstance(biblio, dict):
        biblio = {}
    return {
        "title": title,
        "authors": _format_openalex_authors(item) or "[Unknown Authors]",
        "venue": _openalex_source_name(item),
        "year": str(item.get("publication_year") or "").strip(),
        "volume": str(biblio.get("volume") or "").strip(),
        "issue": str(biblio.get("issue") or "").strip(),
        "pages": _openalex_pages(item),
        "doi": _normalize_doi_like(str(item.get("doi") or "")),
    }


@lru_cache(maxsize=1024)
def _openalex_search_title_raw(title: str, rows: int) -> list[dict[str, Any]]:
    q = str(title or "").strip()
    if not q or len(q) < 8:
        return []
    mailto = str(os.environ.get("KB_OPENALEX_MAILTO") or "").strip()
    params: dict[str, Any] = {
        "search": q[:240],
        "per-page": int(max(1, min(12, rows))),
        "select": "id,doi,title,publication_year,authorships,primary_location,best_oa_location,locations,biblio",
    }
    if mailto:
        params["mailto"] = mailto
    headers = {"User-Agent": "Pi-zaya-KB/1.0 (Research Assistant)"}
    try:
        resp = requests.get("https://api.openalex.org/works", params=params, headers=headers, timeout=4.5)
        if resp.status_code != 200:
            return []
        payload = resp.json() or {}
        results = payload.get("results") if isinstance(payload, dict) else []
        if not isinstance(results, list):
            return []
        return [item for item in results if isinstance(item, dict)]
    except Exception:
        return []


def fetch_best_openalex_meta(
    *,
    query_title: str,
    reference_text: str = "",
    expected_year: str = "",
    expected_venue: str = "",
    min_score: float = 0.88,
) -> dict[str, Any] | None:
    q = str(query_title or "").strip()
    raw = _clean_reference_for_query(reference_text)
    if not q:
        q = raw
    q = _WS_RE.sub(" ", q).strip()
    if len(q) < 8:
        return None

    hint_text = _reference_hint_text(reference_text)
    year_hint = str(expected_year or "").strip()
    if not _YEAR_RE.fullmatch(year_hint):
        year_hint = extract_year_hint(hint_text or raw)
    author_hint = extract_first_author_family_hint(hint_text or raw)
    venue_hint = str(expected_venue or "").strip()

    items = _openalex_search_title_raw(q, 8)
    if not items:
        return None

    best_meta: dict[str, str] | None = None
    best_score = -1.0
    best_title_sim = 0.0
    best_year_match = False
    best_author_match = False

    for item in items:
        meta = _meta_from_openalex_item(item, fallback_title=q)
        cand_title = str(meta.get("title") or "").strip()
        t_sim = title_similarity(q, cand_title)
        year = str(meta.get("year") or "").strip()
        y_match = bool(year_hint and year and year_hint == year)
        y_near = False
        if year_hint and year and (not y_match):
            try:
                y_near = abs(int(year_hint) - int(year)) <= 1
            except Exception:
                y_near = False

        author_match = False
        if author_hint:
            author_match = author_hint in _openalex_author_family_set(item)

        venue_sim = _venue_similarity(venue_hint, str(meta.get("venue") or "")) if venue_hint else 0.0
        score = t_sim
        if year_hint:
            if y_match:
                score += 0.08
            elif y_near:
                score += 0.03
            else:
                score -= 0.14
        if author_hint:
            score += 0.08 if author_match else -0.05
        if venue_hint:
            score += 0.04 * (2.0 * venue_sim - 1.0)
        if str(meta.get("doi") or "").strip():
            score += 0.02
        score = max(0.0, min(1.0, score))

        if (
            score > best_score
            or (
                abs(score - best_score) <= 1e-9
                and (
                    (y_match and not best_year_match)
                    or (author_match and not best_author_match)
                    or (y_match and author_match and not (best_year_match and best_author_match))
                )
            )
        ):
            best_score = score
            best_meta = meta
            best_title_sim = t_sim
            best_year_match = y_match
            best_author_match = author_match

    if not best_meta:
        return None
    if best_title_sim < 0.88:
        return None
    if year_hint and (not best_year_match) and best_score < 0.92:
        return None
    if author_hint and (not best_author_match) and best_score < 0.92:
        return None
    if best_score < float(min_score):
        return None

    out: dict[str, Any] = dict(best_meta)
    out["match_method"] = "openalex_title"
    out["title_similarity"] = round(best_title_sim, 4)
    out["match_score"] = round(best_score, 4)
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
        "select": "author,published-print,published-online,issued,created,container-title,publisher,institution,volume,issue,page,DOI,title",
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
        "select": "author,published-print,published-online,issued,created,container-title,publisher,volume,issue,page,article-number,DOI,title",
    }
    headers = {"User-Agent": "Pi-zaya-KB/1.0 (Research Assistant)"}
    url = "https://api.crossref.org/works"
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=6.5)
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


def _venue_tokens(text: str) -> list[str]:
    s = str(text or "").strip()
    if not s:
        return []
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = s.replace("&", " and ")
    norm = normalize_title_for_match(s)
    return [w for w in norm.split() if w]


def _venue_shape_key(text: str) -> str:
    toks = _venue_tokens(text)
    if not toks:
        return ""
    return "".join(t[: min(4, len(t))] for t in toks)


def _venue_initials(text: str) -> str:
    toks = _venue_tokens(text)
    if not toks:
        return ""
    return "".join(t[:1] for t in toks)


def _venue_similarity(expected: str, got: str) -> float:
    ne = normalize_title_for_match(expected)
    ng = normalize_title_for_match(got)
    if not ne or not ng:
        return 0.0
    if ne == ng:
        return 1.0
    if ne in ng or ng in ne:
        return 0.92
    se = _venue_shape_key(expected)
    sg = _venue_shape_key(got)
    if se and sg and (se == sg or se in sg or sg in se):
        return 0.96
    ie = _venue_initials(expected)
    ig = _venue_initials(got)
    if ie and ig and (ie == ig):
        return 0.94
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

        rank_score = score
        if (
            rank_score > best_score
            or (
                abs(rank_score - best_score) <= 1e-9
                and (v_sim > best_venue_sim or (v_sim == best_venue_sim and y_match and (not best_year_match)))
            )
        ):
            best_score = rank_score
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
    out["match_score"] = round(max(0.0, min(1.0, best_score)), 4)
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

    hint_text = _reference_hint_text(reference_text)
    year_hint = extract_year_hint(hint_text or raw)
    author_hint = extract_first_author_family_hint(hint_text or raw)
    doi_hint = extract_first_doi(hint_text or reference_text or raw)
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
    best_structured_score = 0.0
    best_year_match = False
    best_author_match = False

    for it in items:
        meta = _meta_from_item(it, fallback_title="")
        cand_txt = _candidate_biblio_text(it)
        t_sim = _text_similarity(raw, cand_txt)
        structured_score, structured_year_match, structured_author_match = _structured_biblio_score(
            raw,
            it,
            meta,
            year_hint=year_hint,
            author_hint=author_hint,
        )
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

        score = max(0.0, min(1.0, max(score, structured_score)))
        if (
            score > best_score
            or (
                abs(score - best_score) <= 1e-9
                and structured_score > best_structured_score
            )
        ):
            best_score = score
            best_meta = meta
            best_text_sim = t_sim
            best_structured_score = structured_score
            best_year_match = bool(y_match or structured_year_match)
            best_author_match = bool(author_match or structured_author_match)

    if not best_meta:
        return None
    if best_score < float(min_score):
        return None
    if best_text_sim < 0.42 and best_structured_score < 0.78:
        return None
    if year_hint and (not best_year_match) and best_score < 0.72:
        return None
    if author_hint and (not best_author_match) and best_score < 0.74:
        return None

    out = dict(best_meta)
    out["match_method"] = "bibliographic_compact" if best_structured_score >= 0.78 and best_text_sim < 0.62 else "bibliographic"
    out["title_similarity"] = round(best_text_sim, 4)
    out["match_score"] = round(best_score, 4)
    if best_structured_score:
        out["structured_match_score"] = round(best_structured_score, 4)
    return out
