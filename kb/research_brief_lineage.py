from __future__ import annotations

import hashlib
import json
import re
from typing import Any


_MATRIX_FIELDS = (
    "method",
    "dataset_or_experiment",
    "metric",
    "key_result",
    "limitation",
)
_BLOCKED_EXPORT_STATUSES = {
    "matrix_missing",
    "source_revision_missing",
    "integrity_mismatch",
    "revision_mismatch",
}


def _text(value: object, *, limit: int = 1_200) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _normal(value: object) -> str:
    return _text(value, limit=8_000).casefold()


def _source_identity(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").casefold()


def _stable(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def matrix_contract_fingerprint(record: dict[str, Any] | None) -> str:
    """Hash only matrix state that can influence a matrix-backed brief."""

    source = record if isinstance(record, dict) else {}
    payload = {
        "rows": list(source.get("rows") or []),
        "evidence": list(source.get("evidence") or []),
        "source_items": list(source.get("source_items") or []),
        "comparison_audits": list(source.get("comparison_audits") or []),
        "quality_status": str(source.get("quality_status") or ""),
    }
    return hashlib.sha256(_stable(payload).encode("utf-8")).hexdigest()


def _row_index(record: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(list((record or {}).get("rows") or [])):
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("id") or "").strip() or f"row-{index + 1}"
        out[row_id] = row
    return out


def _row_source(row: dict[str, Any] | None) -> str:
    source = row if isinstance(row, dict) else {}
    return _source_identity(source.get("source_path") or source.get("source_name"))


def _row_label(row: dict[str, Any] | None, row_id: str) -> str:
    source = row if isinstance(row, dict) else {}
    return _text(source.get("paper") or source.get("source_name") or source.get("source_path") or row_id, limit=300)


def _cell_contract(row: dict[str, Any], field: str) -> dict[str, Any]:
    cells = row.get("cells") if isinstance(row.get("cells"), dict) else {}
    cell = cells.get(field) if isinstance(cells.get(field), dict) else {}
    return {
        "value": _text(cell.get("value"), limit=2_000),
        "support_status": _text(cell.get("support_status"), limit=80),
        "evidence_ids": [str(item or "") for item in list(cell.get("evidence_ids") or [])],
        "manual_override": bool(cell.get("manual_override")),
    }


def _comparison_index(record: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(list((record or {}).get("comparison_audits") or [])):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip() or f"comparison-{index + 1}"
        out[item_id] = item
    return out


def _source_item_key(item: dict[str, Any], index: int) -> str:
    return (
        _source_identity(item.get("sourcePath") or item.get("source_path"))
        or _normal(item.get("doi"))
        or _normal(item.get("key") or item.get("id") or item.get("title"))
        or f"source-{index + 1}"
    )


def _source_item_index(record: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(list((record or {}).get("source_items") or [])):
        if isinstance(item, dict):
            out[_source_item_key(item, index)] = item
    return out


def _matrix_hit_signatures(record: dict[str, Any] | None) -> set[tuple[str, str]]:
    if not isinstance(record, dict):
        return set()
    from kb.evidence_matrix import evidence_matrix_hits

    signatures: set[tuple[str, str]] = set()
    for hit in evidence_matrix_hits(record, limit=100):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        source = _source_identity(meta.get("source_path") or meta.get("source_name"))
        quote = _normal(hit.get("text"))
        if source and quote:
            signatures.add((source, quote))
    return signatures


def matrix_change_impact(
    historical: dict[str, Any],
    current: dict[str, Any],
    *,
    brief_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    old_rows = _row_index(historical)
    new_rows = _row_index(current)
    changed_rows: list[dict[str, Any]] = []
    changed_fields = 0
    changed_fields_by_source: dict[str, set[str]] = {}
    for row_id in sorted(set(old_rows) | set(new_rows)):
        old_row = old_rows.get(row_id)
        new_row = new_rows.get(row_id)
        fields: list[str] = []
        change = "changed"
        if old_row is None:
            change = "added"
            fields = [*_MATRIX_FIELDS, "source"]
        elif new_row is None:
            change = "removed"
            fields = [*_MATRIX_FIELDS, "source"]
        else:
            if _row_source(old_row) != _row_source(new_row) or _text(old_row.get("source_status")) != _text(new_row.get("source_status")):
                fields.append("source")
            for field in _MATRIX_FIELDS:
                if _stable(_cell_contract(old_row, field)) != _stable(_cell_contract(new_row, field)):
                    fields.append(field)
            if _text(old_row.get("notes"), limit=4_000) != _text(new_row.get("notes"), limit=4_000):
                fields.append("notes")
        if not fields:
            continue
        changed_fields += len(fields)
        for row in (old_row, new_row):
            identity = _row_source(row)
            if identity:
                changed_fields_by_source.setdefault(identity, set()).update(fields)
        changed_rows.append(
            {
                "row_id": row_id,
                "source_name": _row_label(new_row or old_row, row_id),
                "change": change,
                "fields": fields,
            }
        )

    old_comparisons = _comparison_index(historical)
    new_comparisons = _comparison_index(current)
    changed_comparisons: list[dict[str, str]] = []
    for comparison_id in sorted(set(old_comparisons) | set(new_comparisons)):
        old_item = old_comparisons.get(comparison_id)
        new_item = new_comparisons.get(comparison_id)
        if _stable(old_item or {}) == _stable(new_item or {}):
            continue
        change = "added" if old_item is None else ("removed" if new_item is None else "changed")
        item = new_item or old_item or {}
        changed_comparisons.append(
            {
                "comparison_id": comparison_id,
                "change": change,
                "left_source_name": _text(item.get("left_source_name"), limit=240),
                "right_source_name": _text(item.get("right_source_name"), limit=240),
            }
        )

    old_sources = _source_item_index(historical)
    new_sources = _source_item_index(current)
    changed_sources: list[dict[str, str]] = []
    for source_id in sorted(set(old_sources) | set(new_sources)):
        old_item = old_sources.get(source_id)
        new_item = new_sources.get(source_id)
        if _stable(old_item or {}) == _stable(new_item or {}):
            continue
        change = "added" if old_item is None else ("removed" if new_item is None else "changed")
        item = new_item or old_item or {}
        changed_sources.append(
            {
                "source_id": source_id,
                "source_name": _text(item.get("title") or item.get("sourceName") or item.get("sourcePath"), limit=300),
                "change": change,
            }
        )

    old_hits = _matrix_hit_signatures(historical)
    new_hits = _matrix_hit_signatures(current)
    invalidated_hits = old_hits - new_hits
    changed_source_keys = {str(item.get("source_id") or "") for item in changed_sources}
    affected_citations: list[int] = []
    for item in list(brief_evidence or []):
        if not isinstance(item, dict):
            continue
        citation_number = int(item.get("citation_number") or 0)
        if citation_number <= 0:
            continue
        source = _source_identity(item.get("source_path") or item.get("source_name"))
        quote = _normal(item.get("evidence_quote"))
        source_quote = _normal(item.get("source_evidence_quote"))
        comparison_id = _text(item.get("comparison_audit_id"), limit=120)
        matrix_field = _text(item.get("matrix_field"), limit=80)
        invalidated = (source, quote) in invalidated_hits or bool(source_quote and (source, source_quote) in invalidated_hits)
        if source and source in changed_source_keys:
            invalidated = True
        if comparison_id and any(row["comparison_id"] == comparison_id for row in changed_comparisons):
            invalidated = True
        if matrix_field and source:
            invalidated = invalidated or matrix_field in changed_fields_by_source.get(source, set())
        if invalidated and citation_number not in affected_citations:
            affected_citations.append(citation_number)

    return {
        "changed_row_count": len(changed_rows),
        "changed_field_count": changed_fields,
        "changed_comparison_count": len(changed_comparisons),
        "changed_source_count": len(changed_sources),
        "affected_citation_count": len(affected_citations),
        "affected_citation_numbers": sorted(affected_citations),
        "rows": changed_rows[:40],
        "comparisons": changed_comparisons[:40],
        "sources": changed_sources[:40],
    }


def research_brief_lineage(
    record: dict[str, Any],
    *,
    current_matrix: dict[str, Any] | None,
    historical_matrix: dict[str, Any] | None = None,
    include_impact: bool = False,
    summary_only: bool = False,
) -> dict[str, Any]:
    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
    matrix_id = str(quality.get("source_matrix_id") or "").strip()
    saved_revision = max(0, int(quality.get("source_matrix_revision") or 0))
    historical_verified = bool(
        str(record.get("quality_status") or "") == "verified"
        and str(quality.get("source_matrix_quality_status") or "verified") == "verified"
    )
    lineage: dict[str, Any] = {
        "contract_version": 1,
        "status": "untracked",
        "source_matrix_id": matrix_id,
        "source_matrix_title": str(quality.get("source_matrix_title") or ""),
        "source_matrix_revision": saved_revision,
        "current_matrix_revision": 0,
        "source_matrix_quality_status": str(quality.get("source_matrix_quality_status") or ""),
        "current_matrix_quality_status": "",
        "historical_verified": historical_verified,
        "latest_verified": False,
        "refresh_available": False,
        "export_allowed": True,
        "export_mode": "untracked",
        "reasons": [],
        "impact": {},
    }
    if not matrix_id or saved_revision <= 0:
        lineage["reasons"] = ["no_source_matrix_lineage"]
        return lineage
    if not isinstance(current_matrix, dict):
        lineage.update(
            {
                "status": "matrix_missing",
                "export_allowed": False,
                "export_mode": "blocked",
                "reasons": ["source_matrix_missing"],
            }
        )
        return lineage

    current_revision = max(1, int(current_matrix.get("revision") or 1))
    current_quality = str(current_matrix.get("quality_status") or "draft")
    lineage.update(
        {
            "source_matrix_title": str(current_matrix.get("title") or lineage["source_matrix_title"]),
            "current_matrix_revision": current_revision,
            "current_matrix_quality_status": current_quality,
            "refresh_available": current_quality == "verified",
        }
    )
    if saved_revision > current_revision:
        lineage.update(
            {
                "status": "revision_mismatch",
                "export_allowed": False,
                "export_mode": "blocked",
                "reasons": ["source_matrix_revision_ahead_of_current"],
            }
        )
        return lineage

    saved_fingerprint = str(quality.get("source_matrix_fingerprint") or "").strip()
    if saved_revision == current_revision:
        if saved_fingerprint and saved_fingerprint != matrix_contract_fingerprint(current_matrix):
            status = "integrity_mismatch"
            reasons = ["source_matrix_changed_without_revision"]
        elif current_quality != "verified":
            status = "matrix_unverified"
            reasons = ["current_source_matrix_not_verified"]
        else:
            status = "current"
            reasons = []
        lineage.update(
            {
                "status": status,
                "latest_verified": historical_verified and status == "current",
                "export_allowed": status not in _BLOCKED_EXPORT_STATUSES,
                "export_mode": "current" if status == "current" else ("blocked" if status in _BLOCKED_EXPORT_STATUSES else "historical"),
                "reasons": reasons,
            }
        )
        return lineage

    if not isinstance(historical_matrix, dict) and summary_only:
        equivalent = bool(
            saved_fingerprint
            and saved_fingerprint == matrix_contract_fingerprint(current_matrix)
            and current_quality == "verified"
        )
        status = (
            "current_equivalent"
            if equivalent
            else ("matrix_updated" if current_quality == "verified" else "matrix_updated_unverified")
        )
        lineage.update(
            {
                "status": status,
                "latest_verified": historical_verified and equivalent,
                "export_allowed": True,
                "export_mode": "current_equivalent" if equivalent else "historical",
                "reasons": [
                    (
                        "source_matrix_revision_advanced_without_evidence_change"
                        if equivalent
                        else "source_matrix_updated"
                    ),
                    *(["current_source_matrix_not_verified"] if current_quality != "verified" else []),
                ],
            }
        )
        return lineage
    if not isinstance(historical_matrix, dict):
        lineage.update(
            {
                "status": "source_revision_missing",
                "export_allowed": False,
                "export_mode": "blocked",
                "reasons": ["source_matrix_revision_missing"],
            }
        )
        return lineage
    if saved_fingerprint and saved_fingerprint != matrix_contract_fingerprint(historical_matrix):
        lineage.update(
            {
                "status": "integrity_mismatch",
                "export_allowed": False,
                "export_mode": "blocked",
                "reasons": ["source_matrix_revision_integrity_mismatch"],
            }
        )
        return lineage

    equivalent = matrix_contract_fingerprint(historical_matrix) == matrix_contract_fingerprint(current_matrix)
    if equivalent and current_quality == "verified":
        status = "current_equivalent"
        reasons = ["source_matrix_revision_advanced_without_evidence_change"]
    elif current_quality == "verified":
        status = "matrix_updated"
        reasons = ["source_matrix_updated"]
    else:
        status = "matrix_updated_unverified"
        reasons = ["source_matrix_updated", "current_source_matrix_not_verified"]
    lineage.update(
        {
            "status": status,
            "latest_verified": historical_verified and status == "current_equivalent",
            "export_allowed": True,
            "export_mode": "current_equivalent" if status == "current_equivalent" else "historical",
            "reasons": reasons,
        }
    )
    if include_impact and not equivalent:
        lineage["impact"] = matrix_change_impact(
            historical_matrix,
            current_matrix,
            brief_evidence=[item for item in list(record.get("evidence") or []) if isinstance(item, dict)],
        )
    return lineage


def research_brief_lineage_note(record: dict[str, Any]) -> str:
    lineage = record.get("lineage") if isinstance(record.get("lineage"), dict) else {}
    if not lineage or str(lineage.get("status") or "") == "untracked":
        return ""
    saved = int(lineage.get("source_matrix_revision") or 0)
    current = int(lineage.get("current_matrix_revision") or 0)
    status = str(lineage.get("status") or "untracked")
    return (
        f"Matrix lineage: {lineage.get('source_matrix_title') or lineage.get('source_matrix_id') or 'unknown'}; "
        f"brief source revision: {saved or 'unknown'}; current matrix revision: {current or 'missing'}; "
        f"freshness: {status}; export mode: {lineage.get('export_mode') or 'unknown'}."
    )
