from __future__ import annotations

from functools import lru_cache
import re
from typing import Callable
from urllib.parse import quote

import requests

from api.reference_external_ids import _normalize_doi_like
from api.reference_summary_text import (
    _html_meta_content,
    _jsonld_description_from_html,
    _looks_like_title_echo,
    _openalex_abstract_text,
    _summary_excerpt,
)


def _summary_from_crossref_abstract(meta: dict, *, fetch_crossref_work_by_doi: Callable[[str], dict | None]) -> str:
    doi_like = str((meta or {}).get("doi") or (meta or {}).get("doi_url") or "").strip()
    doi = _normalize_doi_like(doi_like)
    if not doi:
        return ""
    try:
        work = fetch_crossref_work_by_doi(doi)
    except Exception:
        work = None
    if not isinstance(work, dict):
        return ""
    abstract = str(work.get("abstract") or "").strip()
    if not abstract:
        return ""
    return _summary_excerpt(abstract, max_sentences=3, max_len=520)


def _summary_from_openalex_abstract(meta: dict, *, openalex_work_by_doi: Callable[[str], dict | None]) -> str:
    doi_like = str((meta or {}).get("doi") or (meta or {}).get("doi_url") or "").strip()
    doi = _normalize_doi_like(doi_like)
    if not doi:
        return ""
    try:
        work = openalex_work_by_doi(doi)
    except Exception:
        work = None
    abstract = _openalex_abstract_text(work if isinstance(work, dict) else {})
    if not abstract:
        return ""
    return _summary_excerpt(abstract, max_sentences=3, max_len=520)


def _valid_external_abstract_candidate(text: str, *, title: str = "") -> str:
    abstract = _summary_excerpt(text, max_sentences=5, max_len=900)
    if not abstract:
        return ""
    low = abstract.lower()
    if any(
        token in low
        for token in (
            "access through your institution",
            "sign in to access",
            "javascript",
            "cookie",
            "all rights reserved",
            "subscribe to this journal",
            "article navigation",
        )
    ):
        return ""
    if title and _looks_like_title_echo(abstract, title):
        return ""
    if len(re.findall(r"[A-Za-z\u4e00-\u9fff]{3,}", abstract)) < 12:
        return ""
    return _summary_excerpt(abstract, max_sentences=3, max_len=520)


@lru_cache(maxsize=512)
def _semantic_scholar_paper_by_doi(doi: str) -> dict:
    d = _normalize_doi_like(doi)
    if not d or d.startswith("10.48550/arxiv"):
        return {}
    try:
        resp = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(d, safe='')}",
            params={"fields": "title,abstract,year,venue,authors,externalIds,url"},
            headers={"User-Agent": "Pi-zaya-KB/1.0 (Research Assistant)"},
            timeout=4.5,
        )
    except Exception:
        return {}
    if resp.status_code != 200:
        return {}
    try:
        data = resp.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _summary_from_semantic_scholar_abstract(
    meta: dict,
    *,
    semantic_scholar_paper_by_doi: Callable[[str], dict],
    title_similarity: Callable[[str, str], float],
) -> str:
    doi_like = str((meta or {}).get("doi") or (meta or {}).get("doi_url") or "").strip()
    doi = _normalize_doi_like(doi_like)
    if not doi:
        return ""
    work = semantic_scholar_paper_by_doi(doi)
    if not isinstance(work, dict):
        return ""
    external = work.get("externalIds") if isinstance(work.get("externalIds"), dict) else {}
    found_doi = _normalize_doi_like(str((external or {}).get("DOI") or ""))
    if found_doi and found_doi != doi:
        return ""
    title = str((meta or {}).get("title") or "").strip()
    found_title = str(work.get("title") or "").strip()
    if title and found_title and title_similarity(title, found_title) < 0.86:
        return ""
    return _valid_external_abstract_candidate(str(work.get("abstract") or ""), title=title or found_title)


@lru_cache(maxsize=256)
def _doi_landing_page_abstract(doi: str) -> str:
    d = _normalize_doi_like(doi)
    if not d or d.startswith("10.48550/arxiv"):
        return ""
    try:
        resp = requests.get(
            f"https://doi.org/{quote(d, safe='/')}",
            headers={
                "User-Agent": "Pi-zaya-KB/1.0 (Research Assistant)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            timeout=6.0,
            allow_redirects=True,
        )
    except Exception:
        return ""
    if resp.status_code >= 400:
        return ""
    content_type = str(resp.headers.get("content-type") or "").lower()
    if "html" not in content_type and "xml" not in content_type and "text" not in content_type:
        return ""
    text = str(resp.text or "")
    if not text:
        return ""
    text = text[:500_000]
    return (
        _html_meta_content(
            text,
            (
                "citation_abstract",
                "dc.description",
                "dcterms.description",
                "description",
                "og:description",
                "twitter:description",
            ),
        )
        or _jsonld_description_from_html(text)
    )


def _summary_from_doi_landing_page(meta: dict, *, doi_landing_page_abstract: Callable[[str], str]) -> str:
    doi_like = str((meta or {}).get("doi") or (meta or {}).get("doi_url") or "").strip()
    doi = _normalize_doi_like(doi_like)
    if not doi:
        return ""
    title = str((meta or {}).get("title") or "").strip()
    return _valid_external_abstract_candidate(doi_landing_page_abstract(doi), title=title)
