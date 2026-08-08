from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from kb.evidence_matrix import MATRIX_CELL_FIELDS
from kb.research_brief import research_brief_context


WATCH_CONTRACT_VERSION = 1
ACTIONABLE_KINDS = {
    "source_added",
    "source_removed",
    "source_unavailable",
    "source_content_changed",
}


def _text(value: object, *, limit: int = 1_200) -> str:
    return " ".join(str(value or "").split())[: max(0, int(limit))]


def source_identity(value: object) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    try:
        return str(Path(raw).expanduser().resolve(strict=False)).replace("\\", "/").casefold()
    except Exception:
        return raw.casefold()


def _json_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_signature(value: object) -> dict[str, Any]:
    raw = str(value or "").strip()
    if not raw:
        return {"exists": False, "size": 0, "mtime_ns": 0, "sha256": ""}
    try:
        path = Path(raw).expanduser().resolve(strict=False)
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "exists": path.is_file(),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": digest.hexdigest(),
        }
    except (OSError, ValueError):
        return {"exists": False, "size": 0, "mtime_ns": 0, "sha256": ""}


def source_watch_snapshot(
    selected_items: list[dict[str, Any]],
    *,
    shelf_revision: int = 0,
) -> dict[str, Any]:
    context_items = [
        item
        for item in list(research_brief_context(selected_items).get("items") or [])
        if isinstance(item, dict)
    ]
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in context_items:
        source_path = str(item.get("sourcePath") or "").strip()
        identity = source_identity(source_path)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        metadata = {
            "title": _text(item.get("title"), limit=500),
            "source_name": _text(item.get("sourceName"), limit=500),
            "doi": _text(item.get("doi"), limit=400).casefold(),
            "authors": _text(item.get("authors"), limit=800),
            "year": _text(item.get("year"), limit=40),
        }
        file_signature = _file_signature(source_path)
        sources.append(
            {
                "identity": identity,
                "source_item_key": _text(item.get("key"), limit=500),
                "source_path": source_path,
                "source_name": metadata["source_name"] or metadata["title"] or Path(source_path).name,
                "title": metadata["title"],
                "metadata": metadata,
                "metadata_fingerprint": _json_hash(metadata),
                "content": file_signature,
                "content_fingerprint": str(file_signature.get("sha256") or ""),
            }
        )
    sources.sort(key=lambda item: str(item.get("identity") or ""))
    return {
        "contract_version": WATCH_CONTRACT_VERSION,
        "captured_at": time.time(),
        "shelf_revision": max(0, int(shelf_revision or 0)),
        "source_count": len(sources),
        "sources": sources,
        "fingerprint": _json_hash(
            [
                {
                    "identity": item["identity"],
                    "metadata_fingerprint": item["metadata_fingerprint"],
                    "content_fingerprint": item["content_fingerprint"],
                    "exists": bool((item.get("content") or {}).get("exists")),
                }
                for item in sources
            ]
        ),
    }


def evidence_watch_scope_items(
    shelf_items: list[dict[str, Any]],
    *,
    tracked_items: list[dict[str, Any]] | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Keep tracked matrix sources stable, then fill remaining source slots."""
    eligible: list[tuple[str, dict[str, Any]]] = []
    by_identity: dict[str, dict[str, Any]] = {}
    for raw in list(shelf_items or []):
        if not isinstance(raw, dict):
            continue
        context = list(research_brief_context([raw]).get("items") or [])
        source = context[0] if context and isinstance(context[0], dict) else {}
        identity = source_identity(source.get("sourcePath"))
        if not identity or not str(source.get("sourceName") or "").strip() or identity in by_identity:
            continue
        item = dict(raw)
        by_identity[identity] = item
        eligible.append((identity, item))

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_limit = max(1, min(8, int(limit or 8)))
    for raw in list(tracked_items or []):
        if not isinstance(raw, dict):
            continue
        context = list(research_brief_context([raw]).get("items") or [])
        source = context[0] if context and isinstance(context[0], dict) else {}
        identity = source_identity(source.get("sourcePath"))
        current = by_identity.get(identity)
        if current is None or identity in seen:
            continue
        seen.add(identity)
        selected.append(current)
        if len(selected) >= source_limit:
            return selected
    for identity, item in eligible:
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(item)
        if len(selected) >= source_limit:
            break
    return selected


def matrix_source_watch_snapshot(matrix: dict[str, Any]) -> dict[str, Any] | None:
    quality = matrix.get("quality") if isinstance(matrix.get("quality"), dict) else {}
    snapshot = quality.get("source_watch_snapshot")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("sources"), list):
        return None
    return dict(snapshot)


def _impact_for_source(
    matrix: dict[str, Any],
    *,
    identity: str,
    briefs: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [item for item in list(matrix.get("rows") or []) if isinstance(item, dict)]
    affected_rows = [row for row in rows if source_identity(row.get("source_path")) == identity]
    row_ids = {str(row.get("id") or "") for row in affected_rows if str(row.get("id") or "")}
    fields: set[str] = set()
    for row in affected_rows:
        cells = row.get("cells") if isinstance(row.get("cells"), dict) else {}
        for field in MATRIX_CELL_FIELDS:
            cell = cells.get(field) if isinstance(cells.get(field), dict) else {}
            if str(cell.get("value") or "").strip() or list(cell.get("evidence_ids") or []):
                fields.add(field)
    comparison_ids = [
        str(item.get("id") or "")
        for item in list(matrix.get("comparison_audits") or [])
        if isinstance(item, dict)
        and (
            str(item.get("left_row_id") or "") in row_ids
            or str(item.get("right_row_id") or "") in row_ids
        )
        and str(item.get("id") or "")
    ]
    brief_impacts: list[dict[str, Any]] = []
    matrix_id = str(matrix.get("id") or "")
    for brief in briefs:
        quality = brief.get("quality") if isinstance(brief.get("quality"), dict) else {}
        if str(quality.get("source_matrix_id") or "") != matrix_id:
            continue
        citations = sorted(
            {
                int(item.get("citation_number") or 0)
                for item in list(brief.get("evidence") or [])
                if isinstance(item, dict)
                and source_identity(item.get("source_path")) == identity
                and int(item.get("citation_number") or 0) > 0
            }
        )
        if citations:
            brief_impacts.append(
                {
                    "brief_id": str(brief.get("id") or ""),
                    "title": str(brief.get("title") or ""),
                    "revision": int(brief.get("revision") or 1),
                    "citation_numbers": citations,
                }
            )
    return {
        "affected_row_ids": sorted(row_ids),
        "affected_fields": sorted(fields),
        "affected_comparison_ids": sorted(comparison_ids),
        "affected_briefs": brief_impacts,
        "affected_brief_count": len(brief_impacts),
        "affected_citation_count": sum(len(item["citation_numbers"]) for item in brief_impacts),
    }


def build_evidence_watch_events(
    matrix: dict[str, Any],
    *,
    baseline: dict[str, Any],
    current: dict[str, Any],
    briefs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    old_sources = {
        str(item.get("identity") or ""): item
        for item in list(baseline.get("sources") or [])
        if isinstance(item, dict) and str(item.get("identity") or "")
    }
    new_sources = {
        str(item.get("identity") or ""): item
        for item in list(current.get("sources") or [])
        if isinstance(item, dict) and str(item.get("identity") or "")
    }
    all_identities = sorted(set(old_sources) | set(new_sources))
    matrix_id = str(matrix.get("id") or "")
    matrix_revision = int(matrix.get("revision") or 1)
    events: list[dict[str, Any]] = []
    for identity in all_identities:
        before = old_sources.get(identity)
        after = new_sources.get(identity)
        kind = ""
        severity = "info"
        if before is None and after is not None:
            kind = "source_added"
            severity = "warning"
        elif before is not None and after is None:
            kind = "source_removed"
            severity = "error"
        elif before is not None and after is not None:
            before_exists = bool((before.get("content") or {}).get("exists"))
            after_exists = bool((after.get("content") or {}).get("exists"))
            if before_exists and not after_exists:
                kind = "source_unavailable"
                severity = "error"
            elif str(before.get("content_fingerprint") or "") != str(after.get("content_fingerprint") or ""):
                kind = "source_content_changed"
                severity = "error"
            elif str(before.get("metadata_fingerprint") or "") != str(after.get("metadata_fingerprint") or ""):
                kind = "source_metadata_changed"
                severity = "info"
        if not kind:
            continue
        source = after or before or {}
        impact = _impact_for_source(matrix, identity=identity, briefs=list(briefs or []))
        if kind == "source_added":
            impact["candidate_fields"] = list(MATRIX_CELL_FIELDS)
        event_key_payload = {
            "matrix_id": matrix_id,
            "kind": kind,
            "identity": identity,
            "before_content": str((before or {}).get("content_fingerprint") or ""),
            "after_content": str((after or {}).get("content_fingerprint") or ""),
            "before_metadata": str((before or {}).get("metadata_fingerprint") or ""),
            "after_metadata": str((after or {}).get("metadata_fingerprint") or ""),
        }
        events.append(
            {
                "event_key": _json_hash(event_key_payload),
                "contract_version": WATCH_CONTRACT_VERSION,
                "project_id": str(matrix.get("project_id") or ""),
                "matrix_id": matrix_id,
                "matrix_title": str(matrix.get("title") or ""),
                "matrix_revision": matrix_revision,
                "kind": kind,
                "severity": severity,
                "actionable": kind in ACTIONABLE_KINDS,
                "source_identity": identity,
                "source_item_key": str(source.get("source_item_key") or ""),
                "source_path": str(source.get("source_path") or ""),
                "source_name": str(source.get("source_name") or source.get("title") or ""),
                "before": before or {},
                "after": after or {},
                "impact": impact,
            }
        )
    return events


def evidence_watch_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    active = [item for item in events if isinstance(item, dict)]
    return {
        "total": len(active),
        "actionable": sum(1 for item in active if bool(item.get("actionable"))),
        "metadata_only": sum(1 for item in active if str(item.get("kind") or "") == "source_metadata_changed"),
        "high_severity": sum(1 for item in active if str(item.get("severity") or "") == "error"),
        "affected_matrix_count": len({str(item.get("matrix_id") or "") for item in active}),
        "affected_brief_count": len(
            {
                str(brief.get("brief_id") or "")
                for item in active
                for brief in list((item.get("impact") or {}).get("affected_briefs") or [])
                if isinstance(brief, dict) and str(brief.get("brief_id") or "")
            }
        ),
    }
