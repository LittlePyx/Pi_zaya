from __future__ import annotations

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


def _summary_from_crossref_abstract(
    meta: dict,
    *,
    fetch_crossref_work_by_doi: Callable[[str], dict | None],
    fetch_crossref_work_by_doi_status: Callable[[str], tuple[dict | None, str]] | None = None,
) -> str:
    doi_like = str((meta or {}).get("doi") or (meta or {}).get("doi_url") or "").strip()
    doi = _normalize_doi_like(doi_like)
    if not doi:
        return ""
    fetch_status = ""
    try:
        work = fetch_crossref_work_by_doi(doi)
    except Exception:
        work = None
    if isinstance(work, dict):
        fetch_status = "ready"
    elif fetch_crossref_work_by_doi_status is not None:
        # The legacy metadata cache may contain a process-lifetime None from a
        # transient timeout/429.  The status-aware request deliberately bypasses
        # that negative cache so the shelf can recover on retry.
        try:
            work, fetch_status = fetch_crossref_work_by_doi_status(doi)
        except Exception:
            work, fetch_status = None, "failed"
    if fetch_status:
        provider_status = dict(meta.get("summary_fetch_providers") or {})
        provider_status["crossref"] = fetch_status
        meta["summary_fetch_providers"] = provider_status
    if not isinstance(work, dict):
        return ""
    abstract = str(work.get("abstract") or "").strip()
    if not abstract:
        if fetch_status == "ready":
            provider_status = dict(meta.get("summary_fetch_providers") or {})
            provider_status["crossref"] = "not_provided"
            meta["summary_fetch_providers"] = provider_status
        return ""
    summary = _summary_excerpt(abstract, max_sentences=3, max_len=520)
    if not summary and fetch_status == "ready":
        provider_status = dict(meta.get("summary_fetch_providers") or {})
        provider_status["crossref"] = "not_provided"
        meta["summary_fetch_providers"] = provider_status
    return summary


def _summary_from_openalex_abstract(meta: dict, *, openalex_work_by_doi: Callable[[str], dict | None]) -> str:
    doi_like = str((meta or {}).get("doi") or (meta or {}).get("doi_url") or "").strip()
    doi = _normalize_doi_like(doi_like)
    if not doi:
        return ""
    try:
        work = openalex_work_by_doi(doi)
    except Exception:
        work = None
    provider_status = dict(meta.get("summary_fetch_providers") or {})
    if isinstance(work, dict):
        provider_status["openalex"] = "ready" if _openalex_abstract_text(work) else "not_provided"
    else:
        # The legacy OpenAlex adapter collapses both exact-work 404 and
        # transport failures to None.  It cannot safely drive a retry state;
        # Crossref and Semantic Scholar contribute explicit transient status.
        provider_status["openalex"] = "not_provided"
    meta["summary_fetch_providers"] = provider_status
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
        return {"_kb_fetch_status": "failed"}
    if resp.status_code == 404:
        return {"_kb_fetch_status": "not_found"}
    if resp.status_code != 200:
        return {"_kb_fetch_status": "failed", "_kb_http_status": int(resp.status_code or 0)}
    try:
        data = resp.json()
    except Exception:
        return {"_kb_fetch_status": "failed"}
    if not isinstance(data, dict):
        return {"_kb_fetch_status": "failed"}
    return {**data, "_kb_fetch_status": "ready"}


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
    provider_status = dict(meta.get("summary_fetch_providers") or {})
    fetch_status = str(work.get("_kb_fetch_status") or "").strip().lower()
    if fetch_status:
        provider_status["semantic_scholar"] = fetch_status
    else:
        provider_status["semantic_scholar"] = "ready" if str(work.get("abstract") or "").strip() else "not_provided"
    meta["summary_fetch_providers"] = provider_status
    if provider_status["semantic_scholar"] in {"failed", "not_found"}:
        return ""
    external = work.get("externalIds") if isinstance(work.get("externalIds"), dict) else {}
    found_doi = _normalize_doi_like(str((external or {}).get("DOI") or ""))
    if found_doi and found_doi != doi:
        provider_status["semantic_scholar"] = "identity_mismatch"
        meta["summary_fetch_providers"] = provider_status
        return ""
    title = str((meta or {}).get("title") or "").strip()
    found_title = str(work.get("title") or "").strip()
    if title and found_title and title_similarity(title, found_title) < 0.86:
        provider_status["semantic_scholar"] = "identity_mismatch"
        meta["summary_fetch_providers"] = provider_status
        return ""
    abstract = _valid_external_abstract_candidate(str(work.get("abstract") or ""), title=title or found_title)
    if not abstract and provider_status["semantic_scholar"] == "ready":
        provider_status["semantic_scholar"] = "not_provided"
        meta["summary_fetch_providers"] = provider_status
    return abstract


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
    abstract = _valid_external_abstract_candidate(doi_landing_page_abstract(doi), title=title)
    provider_status = dict(meta.get("summary_fetch_providers") or {})
    provider_status["doi_landing_page"] = "ready" if abstract else "not_provided"
    meta["summary_fetch_providers"] = provider_status
    return abstract
