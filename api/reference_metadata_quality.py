from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from api.reference_ui import enrich_citation_detail_meta
from kb.citation_meta import extract_first_doi, normalize_title_for_match, title_similarity
from kb.reference_index import CROSSREF_CACHE_FILE_NAME, INDEX_FILE_NAME


_DOI_URL_RE = re.compile(r"^https?://(?:dx\.)?doi\.org/", flags=re.I)
_REPAIR_SOURCE = "shelf_metadata_repair"
_PERSIST_FIELDS = (
    "title",
    "authors",
    "venue",
    "year",
    "volume",
    "issue",
    "pages",
    "doi",
    "doi_url",
    "citation_count",
    "journal_if",
    "journal_quartile",
    "conference_tier",
    "conference_ccf",
    "summary_line",
)


def _text(value: Any) -> str:
    return str(value or "").replace("\u00a0", " ").strip()


def _norm_doi(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    text = _DOI_URL_RE.sub("", text).strip()
    doi = extract_first_doi(text)
    if doi:
        return doi.strip(" \t\r\n.,;:()[]{}<>")
    if text.lower().startswith("10."):
        return text.strip(" \t\r\n.,;:()[]{}<>")
    return ""


def _doi_url(doi: Any) -> str:
    clean = _norm_doi(doi)
    return f"https://doi.org/{clean}" if clean else ""


def _raw_reference_text(detail: Mapping[str, Any] | None) -> str:
    data = detail or {}
    return _text(
        data.get("raw")
        or data.get("cite_fmt")
        or data.get("citeFmt")
        or data.get("card_reference_entry")
        or data.get("cardReferenceEntry")
    )


def _canonicalize_detail(detail: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(detail or {})
    for canonical, aliases in {
        "doi_url": ("doiUrl",),
        "cite_fmt": ("citeFmt",),
        "card_title": ("cardTitle",),
        "card_reference_entry": ("cardReferenceEntry",),
        "source_path": ("sourcePath",),
        "source_name": ("sourceName",),
        "ref_num": ("refNum", "reference_num", "referenceNum"),
    }.items():
        if _text(out.get(canonical)):
            continue
        for alias in aliases:
            if _text(out.get(alias)):
                out[canonical] = out.get(alias)
                break
    reference_entry = _text(out.get("card_reference_entry") or out.get("cardReferenceEntry"))
    if reference_entry:
        if not _text(out.get("raw")):
            out["raw"] = reference_entry
        if not _text(out.get("cite_fmt")):
            out["cite_fmt"] = reference_entry
    if not _text(out.get("doi")):
        raw_doi = _norm_doi(out.get("doi_url") or out.get("doiUrl") or _raw_reference_text(out))
        if raw_doi:
            out["doi"] = raw_doi
            if not _text(out.get("doi_url")):
                out["doi_url"] = _doi_url(raw_doi)
    return out


def _issue(code: str, label: str, *, field: str, severity: str = "warning", detail: str = "") -> dict[str, Any]:
    return {
        "code": code,
        "label": label,
        "field": field,
        "severity": severity,
        "detail": detail[:240],
    }


def _looks_weak_title(title: str) -> bool:
    t = _text(title)
    if len(t) < 8:
        return True
    low = t.lower()
    if low in {"reference", "citation", "source", "paper", "untitled"}:
        return True
    if re.fullmatch(r"(?:ref(?:erence)?\s*)?#?\d{1,4}", low):
        return True
    tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", t)
    return len(tokens) <= 1


def _looks_weak_authors(authors: str) -> bool:
    text = _text(authors)
    if not text:
        return True
    if text.lower() in {"unknown", "[unknown authors]", "anonymous"}:
        return True
    tokens = re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", text)
    return len(tokens) <= 1


def _looks_weak_venue(venue: str) -> bool:
    text = _text(venue)
    if not text:
        return True
    if text.lower() in {"unknown", "journal", "conference", "proceedings"}:
        return True
    tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text)
    return len(tokens) <= 1


def _has_repair_seed(detail: Mapping[str, Any]) -> bool:
    return any(
        _text(detail.get(key))
        for key in (
            "doi",
            "doi_url",
            "doiUrl",
            "raw",
            "cite_fmt",
            "citeFmt",
            "card_reference_entry",
            "cardReferenceEntry",
            "title",
            "card_title",
            "cardTitle",
            "source_name",
            "sourceName",
            "source_path",
            "sourcePath",
        )
    )


def citation_metadata_quality(detail: Mapping[str, Any] | None) -> dict[str, Any]:
    data: Mapping[str, Any] = _canonicalize_detail(detail)
    title = _text(data.get("title") or data.get("card_title") or data.get("cardTitle"))
    authors = _text(data.get("authors"))
    venue = _text(data.get("venue"))
    year = _text(data.get("year"))
    doi = _norm_doi(data.get("doi") or data.get("doi_url") or data.get("doiUrl"))
    raw = _raw_reference_text(data)
    source = _text(data.get("source_path") or data.get("sourcePath") or data.get("source_name") or data.get("sourceName"))
    external_status = _text(data.get("external_metadata_status") or data.get("externalMetadataStatus")).lower()
    external_doi = _norm_doi(data.get("external_doi") or data.get("externalDoi") or data.get("external_doi_url") or data.get("externalDoiUrl"))
    issues: list[dict[str, Any]] = []

    if not source:
        issues.append(_issue("missing_source", "Missing source identity", field="source", severity="error"))
    if _looks_weak_title(title):
        issues.append(_issue("weak_or_missing_title", "Missing reliable title", field="title", severity="error", detail=title))
    if _looks_weak_authors(authors):
        issues.append(_issue("missing_authors", "Missing authors", field="authors", severity="error"))
    if _looks_weak_venue(venue):
        issues.append(_issue("missing_venue", "Missing journal or conference", field="venue", severity="error"))
    if not re.fullmatch(r"(?:18|19|20)\d{2}", year):
        issues.append(_issue("missing_year", "Missing publication year", field="year", severity="warning"))
    if not doi:
        raw_doi = _norm_doi(raw)
        if raw_doi:
            issues.append(_issue("doi_not_promoted", "DOI present in reference text but not promoted", field="doi", severity="error", detail=raw_doi))
        else:
            issues.append(_issue("missing_doi", "Missing DOI", field="doi", severity="warning"))

    identity_complete = bool(
        source
        and not _looks_weak_title(title)
        and not _looks_weak_authors(authors)
        and not _looks_weak_venue(venue)
        and re.fullmatch(r"(?:18|19|20)\d{2}", year)
        and doi
    )
    external_conflicts_with_visible_doi = bool(doi and external_doi and doi.lower() != external_doi.lower())
    external_needs_review = bool(
        external_status == "conflict"
        or (external_status == "candidate" and ((not identity_complete) or external_conflicts_with_visible_doi))
    )
    if external_status in {"candidate", "conflict"} and external_needs_review:
        issues.append(
            _issue(
                f"external_metadata_{external_status}",
                "External metadata is not trusted yet",
                field="external_metadata_status",
                severity="warning" if external_status == "candidate" else "error",
                detail=_text(data.get("external_metadata_reason") or data.get("externalMetadataReason")),
            )
        )

    error_count = sum(1 for item in issues if str(item.get("severity") or "") == "error")
    warning_count = sum(1 for item in issues if str(item.get("severity") or "") == "warning")
    if error_count:
        status = "error"
    elif warning_count:
        status = "warning"
    else:
        status = "ready"
    score = max(0, 100 - (error_count * 24) - (warning_count * 8))
    missing_fields = [str(item.get("field") or "") for item in issues if str(item.get("field") or "")]
    return {
        "contract_version": 1,
        "ok": status == "ready",
        "status": status,
        "score": score,
        "missing_fields": sorted(set(missing_fields)),
        "issues": issues,
        "repairable": bool(_has_repair_seed(data)),
        "retryable": bool(status != "ready" and _has_repair_seed(data)),
        "doi": doi,
    }


def _changed_fields(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    fields = [
        "title",
        "authors",
        "venue",
        "year",
        "volume",
        "issue",
        "pages",
        "doi",
        "doi_url",
        "citation_count",
        "journal_if",
        "journal_quartile",
        "conference_tier",
        "conference_ccf",
        "summary_line",
    ]
    out: list[str] = []
    for field in fields:
        if _text(before.get(field)) != _text(after.get(field)):
            out.append(field)
    return out


def _json_load_dict(path: Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _json_save_dict(path: Path, data: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _norm_path_text(value: Any) -> str:
    text = _text(value).replace("\\", "/").rstrip("/")
    return text.lower()


def _path_name(value: Any) -> str:
    text = _text(value).replace("\\", "/").rstrip("/")
    if not text:
        return ""
    return text.rsplit("/", 1)[-1].lower()


def _path_suffix_score(want: Any, candidate: Any) -> int:
    left = [part for part in _norm_path_text(want).split("/") if part]
    right = [part for part in _norm_path_text(candidate).split("/") if part]
    if not left or not right:
        return 0
    limit = min(len(left), len(right), 8)
    score = 0
    for idx in range(1, limit + 1):
        if left[-idx:] == right[-idx:]:
            score = idx
        else:
            break
    return score


def _int_value(value: Any) -> int:
    try:
        n = int(str(value or "").strip())
    except Exception:
        return 0
    return n if n > 0 else 0


def _ref_num(meta: Mapping[str, Any]) -> int:
    for key in ("ref_num", "refNum", "num", "number", "reference_num", "referenceNum"):
        n = _int_value(meta.get(key))
        if n > 0:
            return n
    anchor = _text(meta.get("anchor") or meta.get("key"))
    match = re.search(r"(?:ref|reference|bib|b)[^\d]{0,8}(\d{1,4})\b", anchor, flags=re.I)
    if match:
        return _int_value(match.group(1))
    return 0


def _year_ok(value: Any) -> bool:
    return bool(re.fullmatch(r"(?:18|19|20)\d{2}", _text(value)))


def _value_equal(field: str, left: Any, right: Any) -> bool:
    if field == "doi":
        return _norm_doi(left).lower() == _norm_doi(right).lower()
    return _text(left) == _text(right)


def _should_replace_field(field: str, old: Any, new: Any) -> bool:
    new_text = _text(new)
    if field not in {"citation_count"} and not new_text:
        return False
    old_text = _text(old)
    if field == "doi":
        old_doi = _norm_doi(old)
        new_doi = _norm_doi(new)
        return bool(new_doi and (not old_doi or old_doi.lower() == new_doi.lower()))
    if field == "doi_url":
        old_doi = _norm_doi(old)
        new_doi = _norm_doi(new)
        return bool(new_doi and (not old_doi or old_doi.lower() == new_doi.lower()))
    if not old_text:
        return True
    if field == "title":
        if _looks_weak_title(old_text):
            return True
        return len(new_text) > len(old_text) and title_similarity(old_text, new_text) >= 0.72
    if field == "authors":
        if _looks_weak_authors(old_text):
            return True
        return len(new_text) > len(old_text) and old_text.lower() in new_text.lower()
    if field == "venue":
        if _looks_weak_venue(old_text):
            return True
        return len(new_text) > len(old_text) and old_text.lower() in new_text.lower()
    if field == "year":
        return (not _year_ok(old_text)) and _year_ok(new_text)
    if field in {"volume", "issue", "pages", "journal_quartile", "conference_tier", "conference_ccf"}:
        return False
    if field in {"citation_count", "journal_if", "summary_line"}:
        return False
    return False


def _metadata_payload(meta: Mapping[str, Any]) -> dict[str, Any]:
    data = _canonicalize_detail(meta)
    doi = _norm_doi(data.get("doi") or data.get("doi_url"))
    out: dict[str, Any] = {}
    for field in _PERSIST_FIELDS:
        value = data.get(field)
        if field == "title":
            value = data.get("title") or data.get("card_title") or data.get("cardTitle")
        elif field == "doi":
            value = doi
        elif field == "doi_url":
            value = _text(data.get("doi_url") or data.get("doiUrl")) or _doi_url(doi)
        if field == "citation_count":
            try:
                n = int(value)
            except Exception:
                continue
            out[field] = n
            continue
        if _text(value):
            out[field] = _text(value)
    if doi and not out.get("doi_url"):
        out["doi_url"] = _doi_url(doi)
    return out


def _reference_entry_payload(ref: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    merged = _canonicalize_detail({**dict(meta or {}), **dict(ref or {})})
    raw = _raw_reference_text(ref) or _raw_reference_text(meta)
    if raw:
        merged.setdefault("raw", raw)
        merged.setdefault("cite_fmt", raw)
    doi = _norm_doi(merged.get("doi") or merged.get("doi_url") or raw)
    if doi:
        merged["doi"] = doi
        merged["doi_url"] = _doi_url(doi)
    payload = _metadata_payload(merged)
    for key in ("raw", "cite_fmt", "source_path", "source_name", "ref_num", "num"):
        value = merged.get(key)
        if _text(value) and not _text(payload.get(key)):
            payload[key] = _text(value)
    if doi:
        payload.setdefault("doi", doi)
        payload.setdefault("doi_url", _doi_url(doi))
    return payload


def _merge_metadata(existing: Mapping[str, Any] | None, incoming: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    out = dict(existing or {})
    changed = False
    for field, value in dict(incoming or {}).items():
        if not _should_replace_field(field, out.get(field), value):
            continue
        next_value = _norm_doi(value) if field == "doi" else value
        if field == "doi_url":
            next_value = _doi_url(value)
        if not _value_equal(field, out.get(field), next_value):
            out[field] = next_value
            changed = True
    for field, value in {
        "metadata_repaired_at": time.time(),
        "metadata_repair_source": _REPAIR_SOURCE,
    }.items():
        if out.get(field) != value:
            out[field] = value
            changed = True
    return out, changed


def _cache_record(meta: Mapping[str, Any]) -> dict[str, Any]:
    payload = _metadata_payload(meta)
    if not payload:
        return {}
    payload["metadata_repaired_at"] = time.time()
    payload["metadata_repair_source"] = _REPAIR_SOURCE
    return payload


def _persist_crossref_cache(meta: Mapping[str, Any], db_dir: Path) -> bool:
    payload = _cache_record(meta)
    doi = _norm_doi(payload.get("doi") or meta.get("doi") or meta.get("doi_url"))
    if not doi:
        return False
    path = Path(db_dir) / CROSSREF_CACHE_FILE_NAME
    cache = _json_load_dict(path)
    changed = False
    for bucket in ("doi", "bib", "source_refs", "source_work", "title"):
        if not isinstance(cache.get(bucket), dict):
            cache[bucket] = {}
            changed = True
    doi_key = doi.lower()
    existing = cache["doi"].get(doi_key)
    next_record = {**dict(existing or {}), **payload, "doi": doi, "doi_url": _doi_url(doi)}
    if existing != next_record:
        cache["doi"][doi_key] = next_record
        changed = True

    title_key = normalize_title_for_match(_text(payload.get("title") or meta.get("title") or meta.get("card_title")))[:260]
    if title_key:
        existing_title = cache["title"].get(title_key)
        title_record = {**dict(existing_title or {}), **next_record}
        if existing_title != title_record:
            cache["title"][title_key] = title_record
            changed = True

    raw_key = normalize_title_for_match(_text(meta.get("raw") or meta.get("cite_fmt") or meta.get("citeFmt")))[:260]
    if raw_key:
        existing_bib = cache["bib"].get(raw_key)
        bib_record = {**dict(existing_bib or {}), **next_record}
        if existing_bib != bib_record:
            cache["bib"][raw_key] = bib_record
            changed = True

    if not changed:
        return False
    cache["version"] = 1
    cache["updated_at"] = time.time()
    _json_save_dict(path, cache)
    return True


def _source_match_score(meta: Mapping[str, Any], key: str, doc: Mapping[str, Any]) -> int:
    source_path = _text(meta.get("source_path") or meta.get("sourcePath"))
    source_name = _text(meta.get("source_name") or meta.get("sourceName"))
    doc_path = _text(doc.get("path") or key)
    doc_name = _text(doc.get("name") or _path_name(doc_path))
    score = 0
    if source_path:
        if _norm_path_text(source_path) == _norm_path_text(doc_path) or _norm_path_text(source_path) == _norm_path_text(key):
            score += 120
        score += min(24, _path_suffix_score(source_path, doc_path) * 4)
        if _path_name(source_path) and _path_name(source_path) == _path_name(doc_path):
            score += 18
    if source_name:
        source_name_norm = normalize_title_for_match(source_name)
        doc_name_norm = normalize_title_for_match(doc_name)
        doc_stem_norm = normalize_title_for_match(Path(doc_name).stem)
        if source_name_norm and source_name_norm in {doc_name_norm, doc_stem_norm}:
            score += 45
        elif source_name_norm and (source_name_norm in doc_name_norm or doc_stem_norm in source_name_norm):
            score += 16
    return score


def _reference_match_score(meta: Mapping[str, Any], ref: Mapping[str, Any], ref_key: str) -> float:
    score = 0.0
    wanted_num = _ref_num(meta)
    ref_num = _int_value(ref.get("num") or ref_key)
    if wanted_num > 0 and ref_num == wanted_num:
        score += 95.0
    meta_doi = _norm_doi(meta.get("doi") or meta.get("doi_url") or _raw_reference_text(meta))
    ref_doi = _norm_doi(ref.get("doi") or ref.get("doi_url") or ref.get("raw"))
    if meta_doi and ref_doi:
        if meta_doi.lower() == ref_doi.lower():
            score += 110.0
        else:
            score -= 80.0
    elif meta_doi:
        ref_raw_doi = _norm_doi(ref.get("raw"))
        if ref_raw_doi and ref_raw_doi.lower() == meta_doi.lower():
            score += 75.0

    meta_title = _text(meta.get("title") or meta.get("card_title") or meta.get("raw"))
    ref_title = _text(ref.get("title") or ref.get("raw"))
    title_score = title_similarity(meta_title, ref_title)
    if title_score >= 0.82:
        score += 58.0 * title_score
    elif title_score >= 0.68:
        score += 30.0 * title_score

    meta_raw = normalize_title_for_match(_raw_reference_text(meta) or meta_title)
    ref_raw = normalize_title_for_match(_text(ref.get("raw") or ref_title))
    if meta_raw and ref_raw:
        if meta_raw in ref_raw or ref_raw in meta_raw:
            score += 34.0
        elif title_similarity(meta_raw, ref_raw) >= 0.78:
            score += 20.0
    if _year_ok(meta.get("year")) and _text(meta.get("year")) == _text(ref.get("year")):
        score += 5.0
    return score


def _find_reference_doc(data: Mapping[str, Any], meta: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    docs = data.get("docs")
    if not isinstance(docs, dict) or not docs:
        return None
    best_key = ""
    best_doc: dict[str, Any] | None = None
    best_score = 0
    for key, raw_doc in docs.items():
        if not isinstance(raw_doc, dict):
            continue
        score = _source_match_score(meta, str(key), raw_doc)
        if score > best_score:
            best_key = str(key)
            best_doc = raw_doc
            best_score = score
    if best_doc is None or best_score < 16:
        return None
    return best_key, best_doc


def _find_reference_entry(doc: Mapping[str, Any], meta: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    refs = doc.get("refs")
    if not isinstance(refs, dict) or not refs:
        return None
    wanted_num = _ref_num(meta)
    if wanted_num > 0:
        direct = refs.get(str(wanted_num))
        if isinstance(direct, dict):
            return str(wanted_num), direct
    best_key = ""
    best_ref: dict[str, Any] | None = None
    best_score = 0.0
    for key, raw_ref in refs.items():
        if not isinstance(raw_ref, dict):
            continue
        score = _reference_match_score(meta, raw_ref, str(key))
        if score > best_score:
            best_key = str(key)
            best_ref = raw_ref
            best_score = score
    if best_ref is None or best_score < 44.0:
        return None
    return best_key, best_ref


def _persist_reference_index(meta: Mapping[str, Any], db_dir: Path) -> bool:
    path = Path(db_dir) / INDEX_FILE_NAME
    if not path.exists():
        return False
    data = _json_load_dict(path)
    found_doc = _find_reference_doc(data, meta)
    if not found_doc:
        return False
    _doc_key, doc = found_doc
    found_ref = _find_reference_entry(doc, meta)
    if not found_ref:
        return False
    ref_key, ref = found_ref
    incoming = _metadata_payload(meta)
    if not incoming:
        return False
    if incoming.get("doi") and _norm_doi(ref.get("doi")) and _norm_doi(ref.get("doi")).lower() != _norm_doi(incoming.get("doi")).lower():
        return False
    merged, changed = _merge_metadata(ref, incoming)
    if not changed:
        return False
    if incoming.get("doi") or incoming.get("title"):
        merged["crossref_ok"] = True
        merged["match_method"] = _REPAIR_SOURCE
    refs = doc.get("refs")
    if not isinstance(refs, dict):
        return False
    refs[ref_key] = merged
    doc["refs"] = refs
    doc["metadata_repaired_at"] = time.time()
    data["updated_at"] = time.time()
    _json_save_dict(path, data)
    return True


def _metadata_from_reference_index(meta: Mapping[str, Any], db_dir: str | Path | None) -> dict[str, Any]:
    if not db_dir:
        return {}
    path = Path(db_dir) / INDEX_FILE_NAME
    if not path.exists():
        return {}
    data = _json_load_dict(path)
    found_doc = _find_reference_doc(data, meta)
    if not found_doc:
        return {}
    _doc_key, doc = found_doc
    found_ref = _find_reference_entry(doc, meta)
    if not found_ref:
        return {}
    ref_key, ref = found_ref
    payload = _reference_entry_payload(ref, meta)
    if ref_key and not _text(payload.get("ref_num")):
        payload["ref_num"] = ref_key
    if _text(doc.get("path")) and not _text(payload.get("source_path")):
        payload["source_path"] = _text(doc.get("path"))
    if _text(doc.get("name")) and not _text(payload.get("source_name")):
        payload["source_name"] = _text(doc.get("name"))
    if payload:
        payload["metadata_repair_source"] = "reference_index"
    return payload


def _metadata_from_crossref_cache(meta: Mapping[str, Any], db_dir: str | Path | None) -> dict[str, Any]:
    if not db_dir:
        return {}
    path = Path(db_dir) / CROSSREF_CACHE_FILE_NAME
    if not path.exists():
        return {}
    cache = _json_load_dict(path)
    if not cache:
        return {}

    data = _canonicalize_detail(meta)
    doi = _norm_doi(data.get("doi") or data.get("doi_url") or _raw_reference_text(data)).lower()
    raw_key = normalize_title_for_match(_raw_reference_text(data))[:260]
    title_key = normalize_title_for_match(_text(data.get("title") or data.get("card_title") or data.get("cardTitle")))[:260]
    candidates: list[tuple[str, Mapping[str, Any]]] = []

    if doi:
        bucket = cache.get("doi")
        if isinstance(bucket, Mapping):
            value = bucket.get(doi)
            if isinstance(value, Mapping):
                candidates.append(("crossref_cache:doi", value))

    if raw_key:
        bucket = cache.get("bib")
        if isinstance(bucket, Mapping):
            value = bucket.get(raw_key)
            if isinstance(value, Mapping):
                candidates.append(("crossref_cache:bib", value))

    if title_key:
        bucket = cache.get("title")
        if isinstance(bucket, Mapping):
            value = bucket.get(title_key)
            if isinstance(value, Mapping):
                candidates.append(("crossref_cache:title", value))

    for source, candidate in candidates:
        payload = _metadata_payload(candidate)
        if not payload:
            continue
        cand_doi = _norm_doi(payload.get("doi") or payload.get("doi_url")).lower()
        if doi and cand_doi and cand_doi != doi:
            continue
        payload["metadata_repair_source"] = source
        return payload
    return {}


def persist_repaired_citation_metadata(meta: Mapping[str, Any], db_dir: str | Path | None = None) -> list[str]:
    if not db_dir:
        return []
    data = _canonicalize_detail(meta)
    root = Path(db_dir)
    targets: list[str] = []
    try:
        if _persist_crossref_cache(data, root):
            targets.append("crossref_cache")
    except Exception:
        pass
    try:
        if _persist_reference_index(data, root):
            targets.append("reference_index")
    except Exception:
        pass
    return targets


def _classify_error(exc: Exception) -> tuple[str, str]:
    text = f"{type(exc).__name__}: {str(exc or '').strip()}"
    folded = text.lower()
    if any(token in folded for token in ("connection", "connect", "network", "econnrefused", "dns", "name resolution")):
        return "connection", text[:240]
    if "timeout" in folded or "timed out" in folded:
        return "timeout", text[:240]
    return "error", text[:240]


def repair_citation_metadata_item(detail: Mapping[str, Any] | None, *, db_dir: str | Path | None = None) -> dict[str, Any]:
    original = _canonicalize_detail(detail)
    key = _text(original.get("key") or original.get("anchor") or original.get("doi") or original.get("title"))
    before = citation_metadata_quality(original)
    try:
        local_meta = _metadata_from_reference_index(original, db_dir)
        cache_meta = _metadata_from_crossref_cache({**original, **local_meta}, db_dir)
        repair_sources: list[str] = []
        if local_meta:
            repair_sources.append("reference_index")
        if cache_meta:
            repair_sources.append(str(cache_meta.get("metadata_repair_source") or "crossref_cache"))
        seed = _canonicalize_detail({**original, **local_meta, **cache_meta}) if (local_meta or cache_meta) else original
        repaired = enrich_citation_detail_meta(seed)
        if not isinstance(repaired, dict):
            repaired = {}
        merged = _canonicalize_detail({**seed, **repaired})
        raw_doi = _norm_doi(_raw_reference_text(seed))
        if raw_doi and not _norm_doi(merged.get("doi") or merged.get("doi_url")):
            merged["doi"] = raw_doi
            merged["doi_url"] = _doi_url(raw_doi)
        after = citation_metadata_quality(merged)
        promoted_raw_doi = bool(raw_doi and _norm_doi(merged.get("doi") or merged.get("doi_url")) == raw_doi)
        changed = _changed_fields(original, merged)
        if promoted_raw_doi and "doi" not in changed:
            changed.append("doi")
        persisted_targets = persist_repaired_citation_metadata(merged, db_dir) if changed or after.get("ok") else []
        if after["ok"]:
            repair_status = "repaired" if changed or not before["ok"] else "ready"
        elif changed:
            repair_status = "partial"
        elif after.get("retryable"):
            repair_status = "retryable"
        else:
            repair_status = "unchanged"
        merged["metadata_quality"] = after
        merged["metadata_repair_status"] = repair_status
        if repair_sources:
            merged["metadata_repair_sources"] = repair_sources
        if changed:
            merged["metadata_changed_fields"] = changed
        return {
            "key": key,
            "ok": bool(after.get("ok")),
            "changed": bool(changed),
            "changed_fields": changed,
            "repair_status": repair_status,
            "retryable": bool(after.get("retryable")),
            "fixed_issue_codes": [
                str(issue.get("code") or "")
                for issue in list(before.get("issues") or [])
                if isinstance(issue, Mapping)
                and str(issue.get("code") or "")
                and str(issue.get("code") or "") not in {
                    str(after_issue.get("code") or "")
                    for after_issue in list(after.get("issues") or [])
                    if isinstance(after_issue, Mapping)
                }
            ],
            "remaining_issue_codes": [
                str(issue.get("code") or "")
                for issue in list(after.get("issues") or [])
                if isinstance(issue, Mapping) and str(issue.get("code") or "")
            ],
            "repair_sources": repair_sources,
            "before": before,
            "after": after,
            "meta": merged,
            "persisted": bool(persisted_targets),
            "persisted_targets": persisted_targets,
        }
    except Exception as exc:
        error_kind, error_detail = _classify_error(exc)
        return {
            "key": key,
            "ok": False,
            "changed": False,
            "changed_fields": [],
            "repair_status": "retryable" if error_kind in {"connection", "timeout"} else "error",
            "retryable": error_kind in {"connection", "timeout"},
            "error_kind": error_kind,
            "error_detail": error_detail,
            "before": before,
            "after": before,
            "meta": original,
            "persisted": False,
            "persisted_targets": [],
        }


def repair_citation_metadata_batch(
    items: list[Mapping[str, Any]],
    *,
    limit: int = 40,
    db_dir: str | Path | None = None,
) -> dict[str, Any]:
    limited = [item for item in list(items or []) if isinstance(item, Mapping)][: max(0, int(limit))]
    results = [repair_citation_metadata_item(item, db_dir=db_dir) for item in limited]
    ready = sum(1 for item in results if bool((item.get("after") or {}).get("ok")))
    ready_before = sum(1 for item in results if bool((item.get("before") or {}).get("ok")))
    partial = sum(1 for item in results if str(item.get("repair_status") or "") == "partial")
    retryable = sum(1 for item in results if bool(item.get("retryable")))
    failed = sum(1 for item in results if str(item.get("repair_status") or "") == "error")
    changed = sum(1 for item in results if bool(item.get("changed")))
    persisted = sum(1 for item in results if bool(item.get("persisted")))
    fixed_counter: dict[str, int] = {}
    remaining_counter: dict[str, int] = {}
    changed_field_counter: dict[str, int] = {}
    source_counter: dict[str, int] = {}
    before_scores: list[int] = []
    after_scores: list[int] = []
    for item in results:
        before = item.get("before") if isinstance(item.get("before"), Mapping) else {}
        after = item.get("after") if isinstance(item.get("after"), Mapping) else {}
        before_scores.append(_int_value(before.get("score")))
        after_scores.append(_int_value(after.get("score")))
        for code in item.get("fixed_issue_codes") or []:
            text = _text(code)
            if text:
                fixed_counter[text] = fixed_counter.get(text, 0) + 1
        for code in item.get("remaining_issue_codes") or []:
            text = _text(code)
            if text:
                remaining_counter[text] = remaining_counter.get(text, 0) + 1
        for field in item.get("changed_fields") or []:
            text = _text(field)
            if text:
                changed_field_counter[text] = changed_field_counter.get(text, 0) + 1
        for source in item.get("repair_sources") or []:
            text = _text(source)
            if text:
                source_counter[text] = source_counter.get(text, 0) + 1

    def _counter_items(counter: Mapping[str, int], limit: int = 8) -> list[dict[str, Any]]:
        rows = sorted(counter.items(), key=lambda pair: (-int(pair[1]), pair[0]))
        return [{"name": str(name), "count": int(count)} for name, count in rows[:limit]]

    before_avg = int(round(sum(before_scores) / len(before_scores))) if before_scores else 0
    after_avg = int(round(sum(after_scores) / len(after_scores))) if after_scores else 0
    return {
        "ok": failed == 0,
        "requested": len(limited),
        "ready": int(ready),
        "partial": int(partial),
        "retryable": int(retryable),
        "failed": int(failed),
        "changed": int(changed),
        "persisted": int(persisted),
        "impact": {
            "requested": len(limited),
            "ready_before": int(ready_before),
            "ready_after": int(ready),
            "ready_delta": int(ready - ready_before),
            "changed": int(changed),
            "persisted": int(persisted),
            "before_avg_score": before_avg,
            "after_avg_score": after_avg,
            "score_delta": int(after_avg - before_avg),
            "fixed_issue_codes": _counter_items(fixed_counter),
            "remaining_issue_codes": _counter_items(remaining_counter),
            "changed_fields": _counter_items(changed_field_counter),
            "repair_sources": _counter_items(source_counter),
        },
        "items": results,
    }


def metadata_identity_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return title_similarity(_text(left.get("title") or left.get("raw")), _text(right.get("title") or right.get("raw")))
