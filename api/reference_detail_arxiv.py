from __future__ import annotations

from typing import Callable


def merge_existing_arxiv_backfill(
    meta: dict,
    *,
    arxiv_backfill_meta_from_texts: Callable[..., dict],
    normalize_doi_like: Callable[[str], str],
    merge_meta_prefer_richer: Callable[[dict, dict], dict],
) -> dict:
    out = dict(meta or {})
    arxiv_backfill = arxiv_backfill_meta_from_texts(
        str(out.get("doi") or ""),
        str(out.get("doi_url") or ""),
        str(out.get("raw") or ""),
        str(out.get("cite_fmt") or ""),
        str(out.get("title") or ""),
        str(out.get("venue") or ""),
    )
    if arxiv_backfill and not normalize_doi_like(str(out.get("doi") or out.get("doi_url") or "")):
        out = merge_meta_prefer_richer(out, arxiv_backfill)
    return out


def merge_missing_doi_arxiv_fallback(
    meta: dict,
    *,
    raw_seed: str,
    raw: str,
    title: str,
    venue: str,
    arxiv_backfill_meta_from_texts: Callable[..., dict],
    normalize_doi_like: Callable[[str], str],
    merge_meta_prefer_richer: Callable[[dict, dict], dict],
    should_try_openalex_arxiv_title: Callable[..., bool],
    openalex_arxiv_meta_by_title: Callable[[str], dict],
) -> dict:
    out = dict(meta or {})
    if not normalize_doi_like(str(out.get("doi") or out.get("doi_url") or "")):
        arxiv_backfill = arxiv_backfill_meta_from_texts(
            str(out.get("raw") or raw_seed or ""),
            str(out.get("cite_fmt") or ""),
            str(out.get("title") or title or ""),
            str(out.get("venue") or venue or ""),
        )
        if arxiv_backfill:
            out = merge_meta_prefer_richer(out, arxiv_backfill)
    if not normalize_doi_like(str(out.get("doi") or out.get("doi_url") or "")):
        if should_try_openalex_arxiv_title(out, raw=raw_seed or raw):
            openalex_arxiv = openalex_arxiv_meta_by_title(str(out.get("title") or title or ""))
            if openalex_arxiv:
                out = merge_meta_prefer_richer(out, openalex_arxiv)
    return out
