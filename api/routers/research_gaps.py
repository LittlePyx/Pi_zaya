from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from api.deps import get_chat_store, get_settings
from api.routers.evidence_matrices import _indexed_source_is_fresh, _scan_project_evidence_changes
from kb.evidence_matrix import (
    apply_evidence_matrix_cell_repair,
    apply_evidence_matrix_source_expansion,
    audit_evidence_comparison,
    evidence_matrix_cell_repair_candidates,
    evidence_comparison_quality,
    evidence_matrix_source_expansion_preview,
    find_evidence_comparison_candidates,
)
from kb.evidence_watch import source_identity, source_watch_snapshot
from kb.research_brief_lineage import research_brief_lineage
from kb.research_gap import (
    ACTIVE_RESEARCH_GAP_STATUSES,
    build_project_research_gaps,
    find_research_gap_candidates,
    research_gap_summary,
)
from kb.project_status import build_project_research_status
from kb.store import load_all_chunks


router = APIRouter(prefix="/api", tags=["research-gaps"])


class ResearchGapIgnoreBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reason: str = Field("", max_length=500)


class ResearchGapRepairApplyBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    expected_revision: int = Field(..., ge=1)


class ResearchGapExpansionApplyBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    expected_revision: int = Field(..., ge=1)


class ComparisonCandidateAuditBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    expected_revision: int = Field(..., ge=1)
    confirmed_mappings: list[
        Literal["task", "dataset", "evaluation_protocol"]
    ] = Field(default_factory=list, max_length=3)


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


def _affected_briefs_for_matrix(
    project_id: str,
    matrix: dict[str, Any],
    *,
    brief_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    matrix_id = str(matrix.get("id") or "")
    affected: list[dict[str, Any]] = []
    for brief in _project_briefs_with_lineage(project_id, matrix_by_id={matrix_id: matrix}):
        brief_id = str(brief.get("id") or "")
        quality = brief.get("quality") if isinstance(brief.get("quality"), dict) else {}
        if str(quality.get("source_matrix_id") or "") != matrix_id:
            continue
        if brief_ids and brief_id not in brief_ids:
            continue
        lineage = brief.get("lineage") if isinstance(brief.get("lineage"), dict) else {}
        affected.append(
            {
                "id": brief_id,
                "title": str(brief.get("title") or ""),
                "revision": int(brief.get("revision") or 1),
                "lineage_status": str(lineage.get("status") or "untracked"),
                "update_ready": str(lineage.get("status") or "") == "matrix_updated",
                "impact": dict(lineage.get("impact") or {}),
            }
        )
    return affected


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


def _comparison_matrix(project_id: str, matrix_id: str) -> dict[str, Any]:
    matrix = get_chat_store().get_evidence_matrix(str(matrix_id or "").strip())
    if not isinstance(matrix, dict) or str(matrix.get("project_id") or "") != str(project_id or ""):
        raise HTTPException(404, "evidence matrix not found")
    if str(matrix.get("quality_status") or "") != "verified":
        raise HTTPException(400, "comparison candidates require a verified evidence matrix")
    source_paths = [
        str(item.get("source_path") or "")
        for item in list(matrix.get("rows") or [])
        if isinstance(item, dict)
        and str(item.get("source_status") or "active") == "active"
        and str(item.get("source_path") or "")
    ]
    if any(not _indexed_source_is_fresh(source_path) for source_path in source_paths):
        raise HTTPException(
            409,
            "a matrix source is not freshly indexed; reindex changed papers before finding comparisons",
        )
    return matrix


def _comparison_candidate_result(
    matrix: dict[str, Any],
    *,
    limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    chunks = [
        item
        for item in load_all_chunks(get_settings().db_dir)
        if isinstance(item, dict)
    ]
    result = find_evidence_comparison_candidates(
        matrix,
        db_dir=get_settings().db_dir,
        corpus_chunks=chunks,
        limit=limit,
    )
    return result, chunks


def _project_comparison_candidate_scan(
    matrices: list[dict[str, Any]],
) -> dict[str, Any]:
    started = time.perf_counter()
    eligible = [
        matrix
        for matrix in matrices
        if str(matrix.get("quality_status") or "") == "verified"
        and len([item for item in list(matrix.get("rows") or []) if isinstance(item, dict)]) >= 2
    ]
    fresh: list[dict[str, Any]] = []
    skipped_stale = 0
    first_stale_matrix_id = ""
    for matrix in eligible:
        source_paths = [
            str(item.get("source_path") or "")
            for item in list(matrix.get("rows") or [])
            if isinstance(item, dict)
            and str(item.get("source_status") or "active") == "active"
            and str(item.get("source_path") or "")
        ]
        if any(not _indexed_source_is_fresh(source_path) for source_path in source_paths):
            skipped_stale += 1
            if not first_stale_matrix_id:
                first_stale_matrix_id = str(matrix.get("id") or "")
            continue
        fresh.append(matrix)
    chunks = [item for item in load_all_chunks(get_settings().db_dir) if isinstance(item, dict)] if fresh else []
    candidate_count = 0
    first_candidate_matrix_id = ""
    examined_row_pairs = 0
    structured_observation_count = 0
    per_matrix: list[dict[str, Any]] = []
    for matrix in fresh:
        result = find_evidence_comparison_candidates(
            matrix,
            db_dir=get_settings().db_dir,
            corpus_chunks=chunks,
            limit=8,
        )
        count = int(result.get("candidate_count") or 0)
        candidate_count += count
        examined_row_pairs += int(result.get("examined_row_pairs") or 0)
        structured_observation_count += int(result.get("structured_observation_count") or 0)
        matrix_id = str(matrix.get("id") or "")
        if count and not first_candidate_matrix_id:
            first_candidate_matrix_id = matrix_id
        per_matrix.append(
            {
                "matrix_id": matrix_id,
                "matrix_revision": int(matrix.get("revision") or 0),
                "candidate_count": count,
                "phase_timings_ms": dict(result.get("phase_timings_ms") or {}),
            }
        )
    return {
        "contract_version": 1,
        "candidate_count": candidate_count,
        "first_candidate_matrix_id": first_candidate_matrix_id,
        "eligible_matrix_count": len(eligible),
        "scanned_matrix_count": len(fresh),
        "skipped_stale_matrix_count": skipped_stale,
        "first_stale_matrix_id": first_stale_matrix_id,
        "scan_complete": len(fresh) == len(eligible),
        "examined_row_pairs": examined_row_pairs,
        "structured_observation_count": structured_observation_count,
        "matrix_results": per_matrix,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def _project_research_status(project_id: str, *, refresh: bool) -> dict[str, Any]:
    total_started = time.perf_counter()
    project = _project_or_404(project_id)
    artifact_started = time.perf_counter()
    matrices = _project_matrices(project_id)
    matrix_by_id = {str(item.get("id") or ""): item for item in matrices if str(item.get("id") or "")}
    briefs = _project_briefs_with_lineage(project_id, matrix_by_id=matrix_by_id)
    shelf = get_chat_store().get_citation_shelf(project_id=project_id, scope="project") or {}
    artifact_ms = (time.perf_counter() - artifact_started) * 1000.0

    gap_started = time.perf_counter()
    if refresh:
        changes = _scan_project_evidence_changes(project_id)
        generated = build_project_research_gaps(
            project_id=project_id,
            matrices=matrices,
            briefs=briefs,
            evidence_changes=[item for item in list(changes.get("items") or []) if isinstance(item, dict)],
        )
        gaps = get_chat_store().sync_research_gap_items(project_id=project_id, gaps=generated)
    else:
        gaps = get_chat_store().list_project_research_gaps(project_id, status="active", limit=300)
    gap_ms = (time.perf_counter() - gap_started) * 1000.0

    comparison_scan = _project_comparison_candidate_scan(matrices) if refresh else {
        "eligible_matrix_count": len(
            [
                item
                for item in matrices
                if str(item.get("quality_status") or "") == "verified"
                and len([row for row in list(item.get("rows") or []) if isinstance(row, dict)]) >= 2
            ]
        ),
        "scanned_matrix_count": 0,
        "skipped_stale_matrix_count": 0,
        "first_stale_matrix_id": "",
        "candidate_count": 0,
        "scan_complete": False,
        "elapsed_ms": 0.0,
    }
    assemble_started = time.perf_counter()
    payload = build_project_research_status(
        project=project,
        citation_shelf=shelf,
        matrices=matrices,
        briefs=briefs,
        gaps=gaps,
        comparison_scan=comparison_scan,
    )
    assemble_ms = (time.perf_counter() - assemble_started) * 1000.0
    payload.update(
        {
            "refreshed": refresh,
            "generated_at": time.time(),
            "comparison_scan": comparison_scan,
            "phase_timings_ms": {
                "load_artifacts": round(artifact_ms, 3),
                "scan_and_sync_gaps": round(gap_ms, 3),
                "scan_comparison_candidates": round(float(comparison_scan.get("elapsed_ms") or 0.0), 3),
                "assemble": round(assemble_ms, 3),
                "total": round((time.perf_counter() - total_started) * 1000.0, 3),
            },
        }
    )
    return payload


@router.get("/projects/{project_id}/research-status")
def get_project_research_status(project_id: str):
    return _project_research_status(project_id, refresh=False)


@router.post("/projects/{project_id}/research-status/refresh")
def refresh_project_research_status(project_id: str):
    return _project_research_status(project_id, refresh=True)


@router.get("/projects/{project_id}/evidence-matrices/{matrix_id}/comparison-candidates")
def list_evidence_comparison_candidates(
    project_id: str,
    matrix_id: str,
    limit: int = Query(8, ge=1, le=50),
):
    _project_or_404(project_id)
    matrix = _comparison_matrix(project_id, matrix_id)
    result, _chunks = _comparison_candidate_result(matrix, limit=limit)
    return result


@router.post(
    "/projects/{project_id}/evidence-matrices/{matrix_id}/comparison-candidates/{candidate_id}/audit"
)
def audit_evidence_comparison_candidate(
    project_id: str,
    matrix_id: str,
    candidate_id: str,
    body: ComparisonCandidateAuditBody,
):
    _project_or_404(project_id)
    matrix = _comparison_matrix(project_id, matrix_id)
    current_revision = int(matrix.get("revision") or 1)
    if current_revision != int(body.expected_revision):
        raise HTTPException(
            409,
            f"evidence matrix revision conflict; current revision is {current_revision}",
        )
    candidate_result, chunks = _comparison_candidate_result(matrix, limit=100)
    candidate = next(
        (
            item
            for item in list(candidate_result.get("items") or [])
            if isinstance(item, dict) and str(item.get("id") or "") == str(candidate_id or "")
        ),
        None,
    )
    if not isinstance(candidate, dict):
        raise HTTPException(409, "comparison candidate is no longer available; scan again")
    required = {
        str(item or "")
        for item in list(candidate.get("required_confirmations") or [])
        if str(item or "")
    }
    confirmed = {str(item or "") for item in body.confirmed_mappings if str(item or "")}
    if confirmed - required:
        raise HTTPException(400, "only candidate-marked semantic mappings can be confirmed")
    missing = sorted(required - confirmed)
    if missing:
        raise HTTPException(
            400,
            "confirm every reviewed semantic mapping before auditing: " + ", ".join(missing),
        )
    dimensions = []
    for item in list(candidate.get("dimensions") or []):
        if not isinstance(item, dict):
            continue
        dimension = str(item.get("dimension") or "")
        dimensions.append(
            {
                "dimension": dimension,
                "left_value": str(item.get("left_value") or ""),
                "right_value": str(item.get("right_value") or ""),
                "mapping_confirmed": dimension in confirmed,
            }
        )
    audit = audit_evidence_comparison(
        rows=[item for item in list(matrix.get("rows") or []) if isinstance(item, dict)],
        spec={
            "mode": "ranking",
            "left_row_id": str(candidate.get("left_row_id") or ""),
            "right_row_id": str(candidate.get("right_row_id") or ""),
            "dimensions": dimensions,
            "left_target": str(candidate.get("left_target") or ""),
            "right_target": str(candidate.get("right_target") or ""),
            "target_mapping_confirmed": False,
            "left_result": str(candidate.get("left_result") or ""),
            "right_result": str(candidate.get("right_result") or ""),
        },
        db_dir=get_settings().db_dir,
        corpus_chunks=chunks,
    )
    audits = [
        item
        for item in list(matrix.get("comparison_audits") or [])
        if isinstance(item, dict) and str(item.get("id") or "") != str(audit.get("id") or "")
    ]
    audits.append(audit)
    quality = dict(matrix.get("quality") or {})
    quality["contract_version"] = max(2, int(quality.get("contract_version") or 1))
    quality.update(evidence_comparison_quality(audits))
    quality["last_comparison_candidate_audit"] = {
        "contract_version": 1,
        "candidate_id": str(candidate.get("id") or ""),
        "status": str(audit.get("status") or ""),
        "confirmed_mappings": sorted(confirmed),
        "left_row_id": str(candidate.get("left_row_id") or ""),
        "right_row_id": str(candidate.get("right_row_id") or ""),
        "audited_at": time.time(),
    }
    store = get_chat_store()
    record, conflict = store.update_evidence_matrix(
        matrix_id,
        expected_revision=body.expected_revision,
        comparison_audits=audits,
        quality=quality,
    )
    if conflict:
        current = int((record or {}).get("revision") or 0)
        raise HTTPException(409, f"evidence matrix revision conflict; current revision is {current}")
    if not isinstance(record, dict):
        raise HTTPException(404, "evidence matrix not found")
    scan = _scan_project_research_gaps(project_id)
    return {
        "candidate": candidate,
        "audit": audit,
        "matrix": record,
        "affected_briefs": _affected_briefs_for_matrix(project_id, record),
        "research_gaps": scan,
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


def _candidate_or_conflict(gap: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    candidate = next(
        (
            item
            for item in _gap_candidates(gap, limit=12)
            if str(item.get("id") or "") == str(candidate_id or "")
        ),
        None,
    )
    if not isinstance(candidate, dict):
        raise HTTPException(409, "candidate is no longer available; search again")
    return candidate


def _expansion_matrix(gap: dict[str, Any]) -> dict[str, Any]:
    if not bool(gap.get("candidate_searchable")):
        raise HTTPException(400, "this research gap does not support matrix source expansion")
    matrix_id = str(gap.get("matrix_id") or "")
    matrix = get_chat_store().get_evidence_matrix(matrix_id) if matrix_id else None
    if not isinstance(matrix, dict) or str(matrix.get("project_id") or "") != str(gap.get("project_id") or ""):
        raise HTTPException(409, "the research gap's evidence matrix is unavailable")
    if int(matrix.get("revision") or 0) != int(gap.get("matrix_revision") or 0):
        raise HTTPException(409, "the evidence matrix changed; scan research gaps again")
    return matrix


def _confirmed_candidate_shelf_item(
    project_id: str,
    gap: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    action = gap.get("action") if isinstance(gap.get("action"), dict) else {}
    if (
        str(action.get("candidate_id") or "") != str(candidate.get("id") or "")
        or source_identity(action.get("candidate_source_path")) != source_identity(candidate.get("source_path"))
    ):
        raise HTTPException(409, "confirm this candidate in the project literature basket before expanding the matrix")
    shelf = get_chat_store().get_citation_shelf(project_id=project_id, scope="project") or {}
    expected_key = f"research-gap:{candidate.get('id') or ''}"
    item = next(
        (
            entry
            for entry in list(shelf.get("items") or [])
            if isinstance(entry, dict)
            and str(entry.get("key") or entry.get("id") or "") == expected_key
            and source_identity(entry.get("sourcePath") or entry.get("source_path"))
            == source_identity(candidate.get("source_path"))
        ),
        None,
    )
    if not isinstance(item, dict):
        raise HTTPException(409, "the confirmed candidate is no longer in the project literature basket")
    return shelf, item


def _expansion_preview(
    project_id: str,
    gap: dict[str, Any],
    candidate_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    matrix = _expansion_matrix(gap)
    candidate = _candidate_or_conflict(gap, candidate_id)
    source_path = str(candidate.get("source_path") or "")
    if not source_path or not _indexed_source_is_fresh(source_path):
        raise HTTPException(409, "the candidate paper is not freshly indexed; reindex it before expanding the matrix")
    shelf, source_item = _confirmed_candidate_shelf_item(project_id, gap, candidate)
    try:
        preview = evidence_matrix_source_expansion_preview(
            matrix,
            gap,
            candidate,
            source_item,
            db_dir=get_settings().db_dir,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return matrix, candidate, shelf, preview


def _repair_matrix(gap: dict[str, Any]) -> dict[str, Any]:
    if str(gap.get("kind") or "") not in {"missing_cell", "unsupported_cell"}:
        raise HTTPException(400, "this research gap does not support same-source cell repair")
    matrix_id = str(gap.get("matrix_id") or "")
    matrix = get_chat_store().get_evidence_matrix(matrix_id) if matrix_id else None
    if not isinstance(matrix, dict) or str(matrix.get("project_id") or "") != str(gap.get("project_id") or ""):
        raise HTTPException(409, "the research gap's evidence matrix is unavailable")
    if int(matrix.get("revision") or 0) != int(gap.get("matrix_revision") or 0):
        raise HTTPException(409, "the evidence matrix changed; scan research gaps again")
    source_path = str(gap.get("source_path") or "")
    if not source_path or not _indexed_source_is_fresh(source_path):
        raise HTTPException(409, "the source paper is not freshly indexed; reindex it before repairing evidence")
    return matrix


def _gap_repairs(
    gap: dict[str, Any],
    matrix: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    return evidence_matrix_cell_repair_candidates(
        matrix,
        gap,
        db_dir=get_settings().db_dir,
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


@router.get("/projects/{project_id}/research-gaps/{gap_id}/candidates/{candidate_id}/expansion")
def preview_research_gap_source_expansion(
    project_id: str,
    gap_id: str,
    candidate_id: str,
):
    _project_or_404(project_id)
    gap = _gap_or_404(project_id, gap_id)
    if str(gap.get("status") or "") not in ACTIVE_RESEARCH_GAP_STATUSES:
        raise HTTPException(409, "research gap is no longer active")
    matrix, candidate, _shelf, preview = _expansion_preview(project_id, gap, candidate_id)
    return {
        "candidate": candidate,
        "preview": preview,
        "matrix_id": str(matrix.get("id") or ""),
        "matrix_revision": int(matrix.get("revision") or 1),
    }


@router.post("/projects/{project_id}/research-gaps/{gap_id}/candidates/{candidate_id}/expansion/apply")
def apply_research_gap_source_expansion(
    project_id: str,
    gap_id: str,
    candidate_id: str,
    body: ResearchGapExpansionApplyBody,
):
    _project_or_404(project_id)
    gap = _gap_or_404(project_id, gap_id)
    if str(gap.get("status") or "") not in ACTIVE_RESEARCH_GAP_STATUSES:
        raise HTTPException(409, "research gap is no longer active")
    matrix, candidate, shelf, preview = _expansion_preview(project_id, gap, candidate_id)
    if int(matrix.get("revision") or 1) != int(body.expected_revision):
        raise HTTPException(
            409,
            f"evidence matrix revision conflict; current revision is {int(matrix.get('revision') or 1)}",
        )
    try:
        payload = apply_evidence_matrix_source_expansion(
            matrix,
            gap,
            preview,
            db_dir=get_settings().db_dir,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    watch_snapshot = source_watch_snapshot(
        payload["source_items"],
        shelf_revision=int(shelf.get("revision") or 0),
    )
    payload["quality"]["source_watch_snapshot"] = watch_snapshot
    store = get_chat_store()
    record, conflict = store.update_evidence_matrix(
        str(matrix.get("id") or ""),
        expected_revision=body.expected_revision,
        rows=payload["rows"],
        evidence=payload["evidence"],
        source_items=payload["source_items"],
        comparison_flags=payload["comparison_flags"],
        comparison_audits=payload["comparison_audits"],
        quality_status=str(payload["quality_status"]),
        quality=payload["quality"],
    )
    if conflict:
        current_revision = int((record or {}).get("revision") or 0)
        raise HTTPException(409, f"evidence matrix revision conflict; current revision is {current_revision}")
    if not isinstance(record, dict):
        raise HTTPException(404, "evidence matrix not found")
    store.set_evidence_watch_baseline(
        str(record.get("id") or ""),
        project_id=project_id,
        matrix_revision=int(record.get("revision") or 1),
        snapshot=watch_snapshot,
    )
    store.set_research_gap_status(
        gap_id,
        project_id=project_id,
        status="in_progress",
        action={
            "source_expanded_at": time.time(),
            "expanded_candidate_id": str(candidate.get("id") or ""),
            "expanded_source_path": str(candidate.get("source_path") or ""),
            "expanded_row_id": str(payload.get("new_row_id") or ""),
            "matrix_revision": int(record.get("revision") or 1),
        },
    )
    scan = _scan_project_research_gaps(project_id)
    original_gap_preserved = any(
        str(item.get("gap_key") or "") == str(gap.get("gap_key") or "")
        for item in list(scan.get("items") or [])
        if isinstance(item, dict)
    )
    return {
        "gap": store.get_research_gap(gap_id) or gap,
        "candidate": candidate,
        "preview": preview,
        "matrix": record,
        "new_row_id": str(payload.get("new_row_id") or ""),
        "preserved_row_count": int(payload.get("preserved_row_count") or 0),
        "reaudited_comparison_count": int(payload.get("reaudited_comparison_count") or 0),
        "comparison_flag_count": len(list(payload.get("comparison_flags") or [])),
        "original_gap_preserved": original_gap_preserved,
        "affected_briefs": _affected_briefs_for_matrix(project_id, record),
        "research_gaps": scan,
    }


@router.get("/projects/{project_id}/research-gaps/{gap_id}/repairs")
def list_research_gap_repairs(
    project_id: str,
    gap_id: str,
    limit: int = Query(3, ge=1, le=8),
):
    _project_or_404(project_id)
    gap = _gap_or_404(project_id, gap_id)
    if str(gap.get("status") or "") not in ACTIVE_RESEARCH_GAP_STATUSES:
        raise HTTPException(409, "research gap is no longer active")
    matrix = _repair_matrix(gap)
    return {
        "items": _gap_repairs(gap, matrix, limit=limit),
        "gap_id": gap_id,
        "matrix_id": str(matrix.get("id") or ""),
        "matrix_revision": int(matrix.get("revision") or 1),
        "source_path": str(gap.get("source_path") or ""),
    }


@router.post("/projects/{project_id}/research-gaps/{gap_id}/repairs/{repair_id}/apply")
def apply_research_gap_repair(
    project_id: str,
    gap_id: str,
    repair_id: str,
    body: ResearchGapRepairApplyBody,
):
    _project_or_404(project_id)
    gap = _gap_or_404(project_id, gap_id)
    if str(gap.get("status") or "") not in ACTIVE_RESEARCH_GAP_STATUSES:
        raise HTTPException(409, "research gap is no longer active")
    matrix = _repair_matrix(gap)
    if int(matrix.get("revision") or 1) != int(body.expected_revision):
        raise HTTPException(
            409,
            f"evidence matrix revision conflict; current revision is {int(matrix.get('revision') or 1)}",
        )
    repair = next(
        (
            item
            for item in _gap_repairs(gap, matrix, limit=8)
            if str(item.get("id") or "") == str(repair_id or "")
        ),
        None,
    )
    if not isinstance(repair, dict):
        raise HTTPException(409, "repair candidate is no longer available; search again")
    try:
        payload = apply_evidence_matrix_cell_repair(
            matrix,
            gap,
            repair,
            db_dir=get_settings().db_dir,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    store = get_chat_store()
    record, conflict = store.update_evidence_matrix(
        str(matrix.get("id") or ""),
        expected_revision=body.expected_revision,
        rows=payload["rows"],
        evidence=payload["evidence"],
        comparison_flags=payload["comparison_flags"],
        comparison_audits=payload["comparison_audits"],
        quality_status=str(payload["quality_status"]),
        quality=payload["quality"],
    )
    if conflict:
        current_revision = int((record or {}).get("revision") or 0)
        raise HTTPException(409, f"evidence matrix revision conflict; current revision is {current_revision}")
    if not isinstance(record, dict):
        raise HTTPException(404, "evidence matrix not found")
    snapshot = payload["quality"].get("source_watch_snapshot")
    if isinstance(snapshot, dict):
        store.set_evidence_watch_baseline(
            str(record.get("id") or ""),
            project_id=project_id,
            matrix_revision=int(record.get("revision") or 1),
            snapshot=snapshot,
        )
    store.set_research_gap_status(
        gap_id,
        project_id=project_id,
        status="in_progress",
        action={
            "repair_applied_at": time.time(),
            "repair_candidate_id": str(repair.get("id") or ""),
            "repair_evidence_id": str(repair.get("evidence_id") or ""),
            "repair_source_path": str(repair.get("source_path") or ""),
            "matrix_revision": int(record.get("revision") or 1),
        },
    )
    scan = _scan_project_research_gaps(project_id)
    affected_brief_ids = {
        str(item.get("brief_id") or "")
        for item in list((gap.get("impact") or {}).get("affected_briefs") or [])
        if isinstance(item, dict) and str(item.get("brief_id") or "")
    }
    affected_briefs = _affected_briefs_for_matrix(
        project_id,
        record,
        brief_ids=affected_brief_ids,
    )
    return {
        "gap": store.get_research_gap(gap_id) or gap,
        "repair": repair,
        "matrix": record,
        "reaudited_comparison_count": int(payload.get("reaudited_comparison_count") or 0),
        "affected_briefs": affected_briefs,
        "research_gaps": scan,
    }


@router.post("/projects/{project_id}/research-gaps/{gap_id}/candidates/{candidate_id}/confirm")
def confirm_research_gap_candidate(project_id: str, gap_id: str, candidate_id: str):
    _project_or_404(project_id)
    gap = _gap_or_404(project_id, gap_id)
    if str(gap.get("status") or "") not in ACTIVE_RESEARCH_GAP_STATUSES:
        raise HTTPException(409, "research gap is no longer active")
    candidate = _candidate_or_conflict(gap, candidate_id)
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
