from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Mapping

from kb.citation_meta import normalize_title_for_match
from kb.reference_index import CROSSREF_CACHE_FILE_NAME, INDEX_FILE_NAME


PUBLIC_CITATION_META_FIELDS = (
    "title",
    "authors",
    "venue",
    "year",
    "volume",
    "issue",
    "pages",
    "doi",
    "doi_url",
    "venue_kind",
    "citation_count",
    "journal_if",
    "journal_quartile",
    "conference_tier",
    "conference_ccf",
    "conference_name",
    "conference_acronym",
)


def public_citation_meta(value: Mapping[str, Any] | None) -> dict[str, Any]:
    data = value if isinstance(value, Mapping) else {}
    out: dict[str, Any] = {}
    for field in PUBLIC_CITATION_META_FIELDS:
        item = data.get(field)
        if item in (None, "", [], {}):
            continue
        out[field] = item
    doi = _clean_doi(out.get("doi") or out.get("doi_url"))
    if doi:
        out["doi"] = doi
        out["doi_url"] = f"https://doi.org/{doi}"
    return out


def _portable_basename(value: Any) -> str:
    normalized = str(value or "").strip().replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] if normalized else ""


def source_identity_key(value: Any) -> str:
    name = _portable_basename(value)
    lowered = name.lower()
    for suffix in (".en.md", ".zh.md", ".md", ".pdf"):
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def load_local_source_citation_meta(
    source_path: str,
    *,
    source_name: str = "",
    db_dir: str | Path | None,
) -> dict[str, Any]:
    """Return cached source-paper metadata without performing network I/O."""
    if not str(source_path or "").strip() or not db_dir:
        return {}
    root = Path(db_dir).expanduser()
    index = _read_json(root / INDEX_FILE_NAME)
    doc = _source_doc(index, source_path=source_path, source_name=source_name)
    if not doc:
        return {}

    cache = _read_json(root / CROSSREF_CACHE_FILE_NAME)
    direct = _source_doc_public_meta(doc)
    doi = _clean_doi(doc.get("source_doi") or direct.get("doi"))
    if not doi:
        doi = _cached_source_work_doi(cache, doc)

    cached: dict[str, Any] = {}
    doi_bucket = cache.get("doi") if isinstance(cache.get("doi"), Mapping) else {}
    if doi and isinstance(doi_bucket.get(doi.lower()), Mapping):
        cached = public_citation_meta(doi_bucket.get(doi.lower()))
    if not cached:
        title = _source_doc_title(doc)
        title_key = normalize_title_for_match(title)[:260]
        title_bucket = cache.get("title") if isinstance(cache.get("title"), Mapping) else {}
        if title_key and isinstance(title_bucket.get(title_key), Mapping):
            cached = public_citation_meta(title_bucket.get(title_key))

    merged = dict(cached)
    merged.update(direct)
    if doi:
        merged.setdefault("doi", doi)
        merged.setdefault("doi_url", f"https://doi.org/{doi}")
    return public_citation_meta(merged)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {}
    return _read_json_cached(str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))


@lru_cache(maxsize=12)
def _read_json_cached(path: str, _mtime_ns: int, _size: int) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _norm_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").lower().rstrip("/")


def _path_suffix_score(left: str, right: str) -> int:
    lhs = [part for part in _norm_path(left).split("/") if part]
    rhs = [part for part in _norm_path(right).split("/") if part]
    score = 0
    for offset in range(1, min(len(lhs), len(rhs), 8) + 1):
        if lhs[-offset:] != rhs[-offset:]:
            break
        score = offset
    return score


def _source_doc(
    index: Mapping[str, Any],
    *,
    source_path: str,
    source_name: str,
) -> dict[str, Any]:
    docs = index.get("docs")
    if not isinstance(docs, Mapping):
        return {}
    wanted_path = _norm_path(source_path)
    for key, raw_doc in docs.items():
        if not isinstance(raw_doc, Mapping):
            continue
        if wanted_path in {_norm_path(key), _norm_path(raw_doc.get("path"))}:
            return dict(raw_doc)

    identities = {
        key
        for key in (source_identity_key(source_path), source_identity_key(source_name))
        if key
    }
    candidates: list[tuple[int, dict[str, Any]]] = []
    for raw_doc in docs.values():
        if not isinstance(raw_doc, Mapping):
            continue
        doc_identities = {
            key
            for key in (
                source_identity_key(raw_doc.get("path")),
                source_identity_key(raw_doc.get("name")),
                source_identity_key(raw_doc.get("stem")),
            )
            if key
        }
        if not identities.intersection(doc_identities):
            continue
        candidates.append(
            (_path_suffix_score(source_path, str(raw_doc.get("path") or "")), dict(raw_doc))
        )
    if len(candidates) == 1:
        return candidates[0][1]
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        if candidates[0][0] > candidates[1][0]:
            return candidates[0][1]
    return {}


def _source_doc_public_meta(doc: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "title": ("source_title", "title"),
        "authors": ("source_authors", "authors"),
        "venue": ("source_venue", "venue"),
        "year": ("source_year", "year"),
        "volume": ("source_volume", "volume"),
        "issue": ("source_issue", "issue"),
        "pages": ("source_pages", "pages"),
        "doi": ("source_doi", "doi"),
        "doi_url": ("source_doi_url", "doi_url"),
    }
    out: dict[str, Any] = {}
    for field, keys in aliases.items():
        for key in keys:
            value = doc.get(key)
            if value not in (None, "", [], {}):
                out[field] = value
                break
    return public_citation_meta(out)


def _source_doc_title(doc: Mapping[str, Any]) -> str:
    direct = str(doc.get("source_title") or doc.get("title") or "").strip()
    if direct:
        return direct
    _venue, _year, title = _source_name_hints(
        str(doc.get("name") or doc.get("path") or "")
    )
    return title


def _source_name_hints(value: str) -> tuple[str, str, str]:
    name = _portable_basename(value)
    lowered = name.lower()
    for suffix in (".en.md", ".zh.md", ".md", ".pdf"):
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            break
    match = re.match(r"^(?P<venue>.+?)-(?P<year>19\d{2}|20\d{2})-(?P<title>.+)$", name)
    if not match:
        return "", "", name.strip()
    return (
        str(match.group("venue") or "").strip(),
        str(match.group("year") or "").strip(),
        str(match.group("title") or "").strip(),
    )


def _cached_source_work_doi(cache: Mapping[str, Any], doc: Mapping[str, Any]) -> str:
    source_work = cache.get("source_work")
    if not isinstance(source_work, Mapping):
        return ""
    venue, year, title = _source_name_hints(
        str(doc.get("name") or doc.get("path") or "")
    )
    title = _source_doc_title(doc) or title
    if not title:
        return ""
    key = (
        f"{normalize_title_for_match(title)[:220]}|"
        f"{year}|{normalize_title_for_match(venue)[:120]}"
    )
    return _clean_doi(source_work.get(key))


def _clean_doi(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.I)
    match = re.search(r"10\.\d{4,9}/[^\s\"<>]+", text, flags=re.I)
    return match.group(0).rstrip(".,;:)]}").strip() if match else ""
