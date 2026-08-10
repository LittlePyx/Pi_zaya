from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from api.deps import get_chat_store, get_settings
from api.routers.evidence_matrices import _scan_project_evidence_changes
from kb.research_brief_lineage import research_brief_lineage
from kb.research_gap import (
    ACTIVE_RESEARCH_GAP_STATUSES,
    build_project_research_gaps,
    find_research_gap_candidates,
    research_gap_summary,
)


router = APIRouter(prefix="/api", tags=["research-gaps"])


class ResearchGapIgnoreBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reason: str = Field("", max_length=500)


def _project_or_404(project_id: str) -> dict:
    project = get_chat_store().get_project(str(project_id or "").strip())
    if project is None:
        raise HTTPException(404, "project not found")
    return project


def _gap_or_404(project_id: str, gap_id: str) -> dict[str, Any]:
    gap = get_chat_store().get_research_gap(str(gap_id or "").strip())
    if gap is None or str(gap.get("project_id") or "") != str(project_id or ""):
        raise HTTPException(404, "research gap not found")
    return gap


def _project_matrices(project_id: str) -> list[dict[str, Any]]:
    store = get_chat_store()
    records: list[dict[str, Any]] = []
    for summary in store.list_evidence_matrices(project_id, limit=300):
        record = store.get_evidence_matrix(str(summary.get("id") or ""))
        if isinstance(record, dict):
            records.append(record)
    return records


def _project_briefs_with_lineage(
    project_id: str,
    *,
    matrix_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    store = get_chat_store()
    records: list[dict[str, Any]] = []
    for summary in store.list_research_briefs(project_id, limit=300):
        brief = store.get_research_brief(str(summary.get("id") or ""))
        if not isinstance(brief, dict):
            continue
        quality = brief.get("quality") if isinstance(brief.get("quality"), dict) else {}
        matrix_id = str(quality.get("source_matrix_id") or "")
        current = matrix_by_id.get(matrix_id)
        saved_revision = int(quality.get("source_matrix_revision") or 0)
        historical = None
        if isinstance(current, dict) and saved_revision > 0 and int(current.get("revision") or 0) > saved_revision:
            historical = store.get_evidence_matrix_revision(matrix_id, saved_revision)
        enriched = dict(brief)
        enriched["lineage"] = research_brief_lineage(
            brief,
            current_matrix=current,
            historical_matrix=historical,
            include_impact=True,
            summary_only=False,
        )
        records.append(enriched)
    return records


def _scan_project_research_gaps(project_id: str) -> dict[str, Any]:
    _project_or_404(project_id)
    matrices = _project_matrices(project_id)
    matrix_by_id = {str(item.get("id") or ""): item for item in matrices if str(item.get("id") or "")}
    briefs = _project_briefs_with_lineage(project_id, matrix_by_id=matrix_by_id)
    changes = _scan_project_evidence_changes(project_id)
    gaps = build_project_research_gaps(
        project_id=project_id,
        matrices=matrices,
        briefs=briefs,
        evidence_changes=[item for item in list(changes.get("items") or []) if isinstance(item, dict)],
    )
    active = get_chat_store().sync_research_gap_items(project_id=project_id, gaps=gaps)
    return {
        "items": active,
        "summary": research_gap_summary(active),
        "scanned_at": time.time(),
        "matrix_count": len(matrices),
        "brief_count": len(briefs),
        "source_change_count": len(list(changes.get("items") or [])),
    }


@router.get("/projects/{project_id}/research-gaps")
def list_research_gaps(
    project_id: str,
    status: str = Query("active", pattern="^(active|open|in_progress|ignored|resolved)$"),
    limit: int = Query(300, ge=1, le=1000),
):
    _project_or_404(project_id)
    items = get_chat_store().list_project_research_gaps(project_id, status=status, limit=limit)
    return {"items": items, "summary": research_gap_summary(items)}


@router.post("/projects/{project_id}/research-gaps/scan")
def scan_research_gaps(project_id: str):
    return _scan_project_research_gaps(project_id)


@router.post("/projects/{project_id}/research-gaps/{gap_id}/ignore")
def ignore_research_gap(project_id: str, gap_id: str, body: ResearchGapIgnoreBody):
    _project_or_404(project_id)
    gap = _gap_or_404(project_id, gap_id)
    if str(gap.get("status") or "") not in ACTIVE_RESEARCH_GAP_STATUSES:
        raise HTTPException(409, "research gap is no longer active")
    if not bool(gap.get("dismissible")):
        raise HTTPException(400, "this gap must be resolved in its source workflow")
    updated = get_chat_store().set_research_gap_status(
        gap_id,
        project_id=project_id,
        status="ignored",
        action={"ignored_at": time.time(), "ignore_reason": str(body.reason or "")},
    )
    if updated is None:
        raise HTTPException(404, "research gap not found")
    return updated


def _excluded_matrix_sources(gap: dict[str, Any]) -> list[str]:
    matrix_id = str(gap.get("matrix_id") or "")
    matrix = get_chat_store().get_evidence_matrix(matrix_id) if matrix_id else None
    if not isinstance(matrix, dict):
        return []
    sources = [
        str(item.get("source_path") or "")
        for item in list(matrix.get("rows") or [])
        if isinstance(item, dict) and str(item.get("source_path") or "")
    ]
    sources.extend(
        str(item.get("sourcePath") or item.get("source_path") or "")
        for item in list(matrix.get("source_items") or [])
        if isinstance(item, dict) and str(item.get("sourcePath") or item.get("source_path") or "")
    )
    return sources


def _gap_candidates(gap: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    return find_research_gap_candidates(
        gap,
        db_dir=get_settings().db_dir,
        excluded_source_paths=_excluded_matrix_sources(gap),
        limit=limit,
    )


@router.get("/projects/{project_id}/research-gaps/{gap_id}/candidates")
def list_research_gap_candidates(
    project_id: str,
    gap_id: str,
    limit: int = Query(5, ge=1, le=12),
):
    _project_or_404(project_id)
    gap = _gap_or_404(project_id, gap_id)
    if str(gap.get("status") or "") not in ACTIVE_RESEARCH_GAP_STATUSES:
        raise HTTPException(409, "research gap is no longer active")
    if not bool(gap.get("candidate_searchable")):
        raise HTTPException(400, "this research gap does not support local candidate search")
    items = _gap_candidates(gap, limit=limit)
    return {"items": items, "query": str(gap.get("candidate_query") or ""), "gap_id": gap_id}


@router.post("/projects/{project_id}/research-gaps/{gap_id}/candidates/{candidate_id}/confirm")
def confirm_research_gap_candidate(project_id: str, gap_id: str, candidate_id: str):
    _project_or_404(project_id)
    gap = _gap_or_404(project_id, gap_id)
    if str(gap.get("status") or "") not in ACTIVE_RESEARCH_GAP_STATUSES:
        raise HTTPException(409, "research gap is no longer active")
    candidate = next(
        (item for item in _gap_candidates(gap, limit=12) if str(item.get("id") or "") == str(candidate_id or "")),
        None,
    )
    if not isinstance(candidate, dict):
        raise HTTPException(409, "candidate is no longer available; search again")
    shelf_item = {
        "key": f"research-gap:{candidate['id']}",
        "anchor": str(candidate.get("anchor_id") or candidate.get("block_id") or candidate.get("chunk_id") or candidate["id"]),
        "title": str(candidate.get("title") or candidate.get("source_name") or ""),
        "main": str(candidate.get("title") or candidate.get("source_name") or ""),
        "sourceName": str(candidate.get("source_name") or ""),
        "sourcePath": str(candidate.get("source_path") or ""),
        "shelfItemKind": "citation",
        "shelfOrigin": "research_gap",
        "shelfExcerpt": str(candidate.get("evidence_quote") or ""),
        "evidenceQuote": str(candidate.get("evidence_quote") or ""),
        "headingPath": str(candidate.get("heading_path") or ""),
        "locationLabel": str(candidate.get("location_label") or ""),
        "pageStart": candidate.get("page_start"),
        "pageEnd": candidate.get("page_end"),
        "blockId": str(candidate.get("block_id") or ""),
        "anchorId": str(candidate.get("anchor_id") or ""),
        "note": f"Candidate accepted for research gap: {gap.get('title') or gap_id}",
    }
    shelf = get_chat_store().append_citation_shelf_item(
        item=shelf_item,
        project_id=project_id,
        scope="project",
        open=True,
    )
    if shelf is None:
        raise HTTPException(409, "project literature basket is unavailable")
    updated = get_chat_store().set_research_gap_status(
        gap_id,
        project_id=project_id,
        status="in_progress",
        action={
            "candidate_confirmed_at": time.time(),
            "candidate_id": str(candidate.get("id") or ""),
            "candidate_source_path": str(candidate.get("source_path") or ""),
            "candidate_source_name": str(candidate.get("source_name") or ""),
        },
    )
    return {"gap": updated or gap, "candidate": candidate, "shelf": shelf}
