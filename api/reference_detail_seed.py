from __future__ import annotations

import re
from typing import Callable


def detail_raw_seed(meta: dict) -> str:
    return str(
        (meta or {}).get("raw")
        or (meta or {}).get("card_reference_entry")
        or (meta or {}).get("cardReferenceEntry")
        or (meta or {}).get("cite_fmt")
        or (meta or {}).get("citeFmt")
        or ""
    ).strip()


def seed_detail_raw_fields(
    meta: dict,
    *,
    raw: str,
    normalize_doi_like: Callable[[str], str],
    extract_first_doi: Callable[[str], str],
    build_doi_url: Callable[[str], str],
) -> dict:
    out = dict(meta or {})
    raw_text = str(raw or "").strip()
    if not raw_text:
        return out
    if not str(out.get("raw") or "").strip():
        out["raw"] = raw_text
    if not str(out.get("cite_fmt") or out.get("citeFmt") or "").strip():
        out["cite_fmt"] = raw_text
    if not normalize_doi_like(str(out.get("doi") or out.get("doi_url") or "")):
        raw_doi = extract_first_doi(raw_text)
        if raw_doi:
            out["doi"] = raw_doi
            out["doi_url"] = build_doi_url(raw_doi)
    return out


def fallback_parse_raw_reference(
    raw: str,
    *,
    meta: dict,
    arxiv_backfill_meta_from_texts: Callable[..., dict],
    fallback_fill_reference_meta_from_raw: Callable[[dict], dict],
) -> dict:
    s = str(raw or "").strip()
    s = re.sub(r"^\s*(?:\[\s*\d+\s*\]\s*)+", "", s)
    s = s.replace("*", "")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return {}

    out: dict[str, object] = {}
    arxiv_backfill = arxiv_backfill_meta_from_texts(s)
    if arxiv_backfill:
        out.update(arxiv_backfill)

    year_m = re.search(r"\((19|20)\d{2}\)", s)
    if year_m:
        out["year"] = year_m.group(0).strip("()")
    else:
        year2 = re.search(r"\b(19|20)\d{2}\b", s)
        if year2:
            out["year"] = year2.group(0)

    try:
        shared = fallback_fill_reference_meta_from_raw(
            {
                "raw": s,
                "venue": str((meta or {}).get("venue") or "").strip(),
                "title": str((meta or {}).get("title") or "").strip(),
                "authors": str((meta or {}).get("authors") or "").strip(),
                "year": str((meta or {}).get("year") or "").strip(),
                "pages": str((meta or {}).get("pages") or "").strip(),
                "volume": str((meta or {}).get("volume") or "").strip(),
            }
        )
    except Exception:
        shared = {}
    if isinstance(shared, dict):
        for key in ("authors", "title", "venue", "year", "volume", "issue", "pages"):
            value = str(shared.get(key) or "").strip()
            if value:
                out.setdefault(key, value)

    etal_match = re.match(r"^(?P<authors>.+?\bet al\.)\s+(?P<title>.+?)\.\s+(?P<venue>.+)$", s, flags=re.I)
    if etal_match:
        out.setdefault("authors", etal_match.group("authors").strip(" ."))
        out.setdefault("title", etal_match.group("title").strip(" ."))
        out.setdefault("venue", etal_match.group("venue").strip(" ."))
        return out

    if not any(str(out.get(key) or "").strip() for key in ("authors", "title", "venue")):
        parts = [p.strip(" .") for p in re.split(r"\.\s+", s) if p.strip(" .")]
        if len(parts) >= 3:
            out.setdefault("authors", parts[0])
            out.setdefault("title", parts[1])
            out.setdefault("venue", parts[2])
        elif len(parts) == 2:
            out.setdefault("authors", parts[0])
            out.setdefault("title", parts[1])
    return out


def apply_raw_reference_fallback(
    meta: dict,
    *,
    raw: str,
    arxiv_backfill_meta_from_texts: Callable[..., dict],
    fallback_fill_reference_meta_from_raw: Callable[[dict], dict],
) -> dict:
    out = dict(meta or {})
    raw_text = str(raw or "").strip()
    if not raw_text:
        return out
    parsed = fallback_parse_raw_reference(
        raw_text,
        meta=out,
        arxiv_backfill_meta_from_texts=arxiv_backfill_meta_from_texts,
        fallback_fill_reference_meta_from_raw=fallback_fill_reference_meta_from_raw,
    )
    for key, value in parsed.items():
        if value and not str(out.get(key) or "").strip():
            out[key] = value
    return out
