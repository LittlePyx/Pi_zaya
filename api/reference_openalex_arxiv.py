from __future__ import annotations

import difflib
import re

import requests

from api.reference_external_ids import _extract_arxiv_id_like, _normalize_doi_like, build_doi_url


def _normalize_title_for_openalex_search(value: str) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    return s[:240].strip()


def _title_similarity_for_openalex(a: str, b: str) -> float:
    na = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(a or "").lower()).strip()
    nb = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(b or "").lower()).strip()
    if not na or not nb:
        return 0.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    ta = set(na.split())
    tb = set(nb.split())
    jac = (len(ta & tb) / len(ta | tb)) if ta and tb else 0.0
    return float(min(1.0, 0.70 * seq + 0.30 * jac))


def _openalex_arxiv_meta_by_title(title: str) -> dict:
    query = _normalize_title_for_openalex_search(title)
    if len(query) < 8:
        return {}
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={"search": query, "per-page": 8},
            timeout=6.0,
            headers={"User-Agent": "Pi-zaya-KB/1.0"},
        )
        if r.status_code != 200:
            return {}
        payload = r.json() or {}
    except Exception:
        return {}
    results = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(results, list) or not results:
        return {}

    best: dict = {}
    best_score = 0.0
    for item in results:
        if not isinstance(item, dict):
            continue
        cand_title = str(item.get("title") or "").strip()
        doi_url = str(item.get("doi") or "").strip()
        if not doi_url:
            continue
        doi_norm = _normalize_doi_like(doi_url)
        if not doi_norm:
            continue
        arxiv_id = _extract_arxiv_id_like(doi_norm) or _extract_arxiv_id_like(str(item.get("ids") or ""))
        if not arxiv_id and ("arxiv" not in doi_norm.lower()):
            continue
        sim = _title_similarity_for_openalex(query, cand_title)
        if sim > best_score:
            best_score = sim
            best = item
    if best_score < 0.84 or not isinstance(best, dict):
        return {}

    doi_norm = _normalize_doi_like(str(best.get("doi") or "").strip())
    if not doi_norm:
        return {}
    out: dict[str, object] = {
        "doi": doi_norm,
        "doi_url": build_doi_url(doi_norm),
        "match_method": "openalex_title_arxiv",
    }
    pub_year = str(best.get("publication_year") or "").strip()
    if pub_year:
        out["year"] = pub_year
    primary_location = best.get("primary_location")
    if isinstance(primary_location, dict):
        source = primary_location.get("source")
        if isinstance(source, dict):
            venue_name = str(source.get("display_name") or "").strip()
            if venue_name:
                out["venue"] = venue_name
    return out


def _should_try_openalex_arxiv_title(meta: dict, *, raw: str) -> bool:
    title = str((meta or {}).get("title") or "").strip()
    if len(title) < 8:
        return False
    venue = str((meta or {}).get("venue") or "").strip().lower()
    s = f"{raw}\n{title}\n{venue}"
    if _extract_arxiv_id_like(s):
        return True
    if "arxiv" in s.lower():
        return True
    return False
