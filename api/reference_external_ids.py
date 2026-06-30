from __future__ import annotations

import re
from urllib.parse import quote

_ARXIV_ID_RE = re.compile(r"\barxiv\s*[:\s]\s*(\d{4}\.\d{4,5})(?:v\d+)?\b", flags=re.I)
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", flags=re.I)
_ARXIV_DOI_RE = re.compile(r"10\.48550/arxiv[.:](\d{4}\.\d{4,5})(?:v\d+)?", flags=re.I)


def build_doi_url(doi_or_url: str) -> str:
    raw = str(doi_or_url or "").strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    return "https://doi.org/" + quote(raw, safe="/:;._-()")


def _is_weak_meta_value(key: str, value: str) -> bool:
    s = str(value or "").strip()
    if not s:
        return True
    if key == "title":
        if len(s) <= 4:
            return True
        if len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", s)) <= 1:
            return True
        if re.fullmatch(r"[A-Za-z][A-Za-z.\s&-]{1,40}\(\d{4}\)\.?", s):
            return True
        if re.fullmatch(r"[A-Za-z][A-Za-z.\s&-]{1,40}\d{4}\.?", s):
            return True
    if key == "authors":
        if len(s) <= 3:
            return True
        if len(re.findall(r"[A-Za-z\u4e00-\u9fff]+", s)) <= 1:
            return True
    if key == "venue":
        if len(s) <= 1:
            return True
    return False


def _normalize_doi_like(value: str) -> str:
    s = str(value or "").strip().lower()
    if not s:
        return ""
    aid = _extract_arxiv_id_like(s)
    if aid:
        return _arxiv_doi_from_id(aid).lower()
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s, flags=re.I)
    s = s.strip(" \t\r\n.,;:()[]{}<>")
    return s


def _extract_arxiv_id_like(value: str) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    for pattern in (_ARXIV_ID_RE, _ARXIV_URL_RE, _ARXIV_DOI_RE):
        m = pattern.search(s)
        if m:
            aid = str(m.group(1) or "").strip()
            if aid:
                return aid
    return ""


def _arxiv_doi_from_id(arxiv_id: str) -> str:
    aid = str(arxiv_id or "").strip()
    if not aid:
        return ""
    return f"10.48550/arXiv.{aid}"


def _arxiv_backfill_meta_from_texts(*values: str) -> dict:
    aid = ""
    for raw in values:
        aid = _extract_arxiv_id_like(raw)
        if aid:
            break
    if not aid:
        return {}
    doi = _arxiv_doi_from_id(aid)
    if not doi:
        return {}
    return {
        "doi": doi,
        "doi_url": build_doi_url(doi),
        "arxiv_id": aid,
        "arxiv_url": f"https://arxiv.org/abs/{aid}",
        "match_method": "arxiv_doi_backfill",
    }


__all__ = [
    "_arxiv_backfill_meta_from_texts",
    "_arxiv_doi_from_id",
    "_extract_arxiv_id_like",
    "_is_weak_meta_value",
    "_normalize_doi_like",
    "build_doi_url",
]
