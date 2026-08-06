from __future__ import annotations

import re
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from api.deps import get_chat_store, get_settings
from kb.evidence_matrix import (
    COMPARISON_DIMENSIONS,
    MATRIX_CELL_FIELDS,
    audit_evidence_comparison,
    build_project_evidence_matrix,
    evidence_comparison_quality,
    evidence_matrix_csv,
    evidence_matrix_markdown,
    evidence_matrix_quality,
    evidence_matrix_xlsx,
    reaudit_evidence_comparisons,
)
from kb.maintenance import create_auto_snapshot
from kb.research_brief import research_brief_context, select_research_brief_sources


router = APIRouter(prefix="/api", tags=["evidence-matrices"])

_MAX_TITLE_CHARS = 240
_MAX_OBJECTIVE_CHARS = 4_000
_MAX_ITEM_KEYS = 8
_MAX_NOTE_CHARS = 4_000
_MAX_CELL_CHARS = 520


class EvidenceMatrixCreateBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field("Untitled evidence matrix", max_length=_MAX_TITLE_CHARS)
    objective: str = Field("", max_length=_MAX_OBJECTIVE_CHARS)
    source_conv_id: str | None = Field(None, max_length=120)


class EvidenceMatrixGenerateBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field("Evidence matrix", max_length=_MAX_TITLE_CHARS)
    objective: str = Field("", max_length=_MAX_OBJECTIVE_CHARS)
    item_keys: list[str] = Field(default_factory=list, max_length=_MAX_ITEM_KEYS)
    source_conv_id: str | None = Field(None, max_length=120)
    matrix_id: str | None = Field(None, max_length=120)
    expected_revision: int | None = Field(None, ge=1)


class EvidenceMatrixCellUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field: Literal["method", "dataset_or_experiment", "metric", "key_result", "limitation"]
    value: str = Field("", max_length=_MAX_CELL_CHARS)


class EvidenceMatrixRowUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    row_id: str = Field(..., min_length=1, max_length=120)
    notes: str | None = Field(None, max_length=_MAX_NOTE_CHARS)
    cells: list[EvidenceMatrixCellUpdate] = Field(default_factory=list, max_length=5)


class EvidenceMatrixUpdateBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    expected_revision: int = Field(..., ge=1)
    title: str | None = Field(None, max_length=_MAX_TITLE_CHARS)
    objective: str | None = Field(None, max_length=_MAX_OBJECTIVE_CHARS)
    row_updates: list[EvidenceMatrixRowUpdate] = Field(default_factory=list, max_length=_MAX_ITEM_KEYS)


class EvidenceMatrixRestoreBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    revision: int = Field(..., ge=1)
    expected_revision: int = Field(..., ge=1)


class EvidenceComparisonDimensionBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dimension: Literal["task", "dataset", "evaluation_protocol", "metric"]
    left_value: str = Field(..., min_length=2, max_length=240)
    right_value: str = Field(..., min_length=2, max_length=240)
    mapping_confirmed: bool = False


class EvidenceComparisonAuditBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    expected_revision: int = Field(..., ge=1)
    mode: Literal["ranking", "replication"] = "ranking"
    left_row_id: str = Field(..., min_length=1, max_length=120)
    right_row_id: str = Field(..., min_length=1, max_length=120)
    dimensions: list[EvidenceComparisonDimensionBody] = Field(..., min_length=4, max_length=4)
    left_target: str = Field(..., min_length=2, max_length=240)
    right_target: str = Field(..., min_length=2, max_length=240)
    target_mapping_confirmed: bool = False
    left_result: str = Field(..., min_length=1, max_length=80)
    right_result: str = Field(..., min_length=1, max_length=80)


def _project_or_404(project_id: str) -> dict:
    project = get_chat_store().get_project(str(project_id or "").strip())
    if project is None:
        raise HTTPException(404, "project not found")
    return project


def _matrix_or_404(matrix_id: str) -> dict:
    record = get_chat_store().get_evidence_matrix(str(matrix_id or "").strip())
    if record is None:
        raise HTTPException(404, "evidence matrix not found")
    return record


def _conflict_response(record: dict | None) -> None:
    revision = int((record or {}).get("revision") or 0)
    raise HTTPException(409, f"evidence matrix revision conflict; current revision is {revision}")


def _download_name(record: dict, suffix: str) -> str:
    title = str(record.get("title") or "evidence-matrix").strip().lower()
    slug = re.sub(r"[^a-z0-9_-]+", "-", title).strip("-")[:64] or "evidence-matrix"
    return f"{slug}-r{int(record.get('revision') or 1)}.{suffix}"


def _quality_with_comparisons(quality: dict[str, Any], audits: list[dict[str, Any]]) -> dict[str, Any]:
    result = dict(quality or {})
    result["contract_version"] = max(2, int(result.get("contract_version") or 1))
    result.update(evidence_comparison_quality(audits))
    return result


def _selected_shelf_items(project_id: str, item_keys: list[str]) -> list[dict[str, Any]]:
    shelf = get_chat_store().get_citation_shelf(project_id=project_id, scope="project")
    shelf_items = [item for item in list((shelf or {}).get("items") or []) if isinstance(item, dict)]
    requested = {str(key or "").strip() for key in item_keys if str(key or "").strip()}
    if requested:
        shelf_by_key = {
            str(item.get("key") or item.get("id") or "").strip(): item
            for item in shelf_items
            if str(item.get("key") or item.get("id") or "").strip()
        }
        unavailable = sorted(
            key
            for key in requested
            if key not in shelf_by_key or not select_research_brief_sources([shelf_by_key[key]])
        )
        if unavailable:
            raise HTTPException(
                400,
                "selected literature-basket items lack local full-text evidence: "
                + ", ".join(unavailable[:_MAX_ITEM_KEYS]),
            )
    selected = select_research_brief_sources(shelf_items, item_keys=item_keys)
    if not selected:
        raise HTTPException(400, "no selected literature-basket item has local full-text evidence")
    return selected


def _updated_rows(
    current_rows: list[dict[str, Any]],
    updates: list[EvidenceMatrixRowUpdate],
) -> tuple[list[dict[str, Any]], bool, bool]:
    by_id = {
        str(row.get("id") or ""): dict(row)
        for row in current_rows
        if isinstance(row, dict) and str(row.get("id") or "")
    }
    cells_changed = False
    notes_changed = False
    for update in updates:
        row = by_id.get(str(update.row_id or ""))
        if row is None:
            raise HTTPException(400, f"unknown evidence matrix row: {update.row_id}")
        if update.notes is not None and str(row.get("notes") or "") != update.notes:
            row["notes"] = update.notes
            notes_changed = True
        cells = dict(row.get("cells") or {}) if isinstance(row.get("cells"), dict) else {}
        for cell_update in update.cells:
            field = str(cell_update.field)
            if field not in MATRIX_CELL_FIELDS:
                continue
            current_cell = dict(cells.get(field) or {}) if isinstance(cells.get(field), dict) else {}
            next_value = str(cell_update.value or "").strip()
            if str(current_cell.get("value") or "").strip() == next_value:
                continue
            current_cell.update(
                {
                    "field": field,
                    "value": next_value,
                    "support_status": "needs_review" if next_value else "missing",
                    "manual_override": True,
                }
            )
            cells[field] = current_cell
            cells_changed = True
        row["cells"] = cells
        by_id[update.row_id] = row
    ordered = [by_id.get(str(row.get("id") or ""), dict(row)) for row in current_rows if isinstance(row, dict)]
    return ordered, cells_changed, notes_changed


@router.get("/projects/{project_id}/evidence-matrices")
def list_evidence_matrices(project_id: str, limit: int = Query(80, ge=1, le=300)):
    _project_or_404(project_id)
    return get_chat_store().list_evidence_matrices(project_id, limit=limit)


@router.post("/projects/{project_id}/evidence-matrices")
def create_evidence_matrix(project_id: str, body: EvidenceMatrixCreateBody):
    _project_or_404(project_id)
    record = get_chat_store().create_evidence_matrix(
        project_id=project_id,
        source_conv_id=body.source_conv_id,
        title=body.title,
        objective=body.objective,
        quality_status="draft",
        quality={"contract_version": 1, "reasons": ["manual_draft"]},
    )
    if record is None:
        raise HTTPException(404, "project not found")
    return record


@router.post("/projects/{project_id}/evidence-matrices/generate")
def generate_evidence_matrix(project_id: str, body: EvidenceMatrixGenerateBody):
    _project_or_404(project_id)
    current: dict[str, Any] | None = None
    if body.matrix_id:
        current = _matrix_or_404(body.matrix_id)
        if str(current.get("project_id") or "") != str(project_id or ""):
            raise HTTPException(404, "evidence matrix not found in project")
        if body.expected_revision is None:
            raise HTTPException(400, "expected_revision is required when refreshing an evidence matrix")
        if int(current.get("revision") or 1) != int(body.expected_revision):
            _conflict_response(current)
    selected_items = _selected_shelf_items(project_id, body.item_keys)
    rows, evidence, comparison_flags = build_project_evidence_matrix(
        selected_items,
        objective=body.objective,
        db_dir=get_settings().db_dir,
        existing_rows=list((current or {}).get("rows") or []),
    )
    comparison_audits = reaudit_evidence_comparisons(
        rows=rows,
        audits=[item for item in list((current or {}).get("comparison_audits") or []) if isinstance(item, dict)],
        db_dir=get_settings().db_dir,
    )
    quality_status, quality = evidence_matrix_quality(
        rows=rows,
        evidence=evidence,
        selected_items=selected_items,
        comparison_flags=comparison_flags,
        comparison_audits=comparison_audits,
    )
    source_items = [
        item
        for item in list(research_brief_context(selected_items, conversation_id=body.source_conv_id or "").get("items") or [])
        if isinstance(item, dict)
    ]
    title = str(body.title or "").strip() or str(body.objective or "").strip()[:_MAX_TITLE_CHARS]
    store = get_chat_store()
    if body.matrix_id:
        record, conflict = store.update_evidence_matrix(
            body.matrix_id,
            expected_revision=body.expected_revision,
            title=title,
            objective=body.objective,
            rows=rows,
            evidence=evidence,
            source_items=source_items,
            comparison_flags=comparison_flags,
            comparison_audits=comparison_audits,
            quality_status=quality_status,
            quality=quality,
        )
        if conflict:
            _conflict_response(record)
        if record is None:
            raise HTTPException(404, "evidence matrix not found")
        return record
    record = store.create_evidence_matrix(
        project_id=project_id,
        source_conv_id=body.source_conv_id,
        title=title,
        objective=body.objective,
        rows=rows,
        evidence=evidence,
        source_items=source_items,
        comparison_flags=comparison_flags,
        comparison_audits=comparison_audits,
        quality_status=quality_status,
        quality=quality,
    )
    if record is None:
        raise HTTPException(404, "project not found")
    return record


@router.get("/evidence-matrices/{matrix_id}")
def get_evidence_matrix(matrix_id: str):
    return _matrix_or_404(matrix_id)


@router.patch("/evidence-matrices/{matrix_id}")
def update_evidence_matrix(matrix_id: str, body: EvidenceMatrixUpdateBody):
    current = _matrix_or_404(matrix_id)
    rows, cells_changed, _notes_changed = _updated_rows(
        [item for item in list(current.get("rows") or []) if isinstance(item, dict)],
        body.row_updates,
    )
    objective_changed = body.objective is not None and body.objective != str(current.get("objective") or "")
    quality_status: str | None = None
    quality: dict[str, Any] | None = None
    if cells_changed:
        quality_status, quality = evidence_matrix_quality(
            rows=rows,
            evidence=[item for item in list(current.get("evidence") or []) if isinstance(item, dict)],
            selected_items=[item for item in list(current.get("source_items") or []) if isinstance(item, dict)],
            comparison_flags=[item for item in list(current.get("comparison_flags") or []) if isinstance(item, dict)],
            comparison_audits=[item for item in list(current.get("comparison_audits") or []) if isinstance(item, dict)],
        )
        quality["edited_after_verification"] = True
        quality["reasons"] = sorted({*list(quality.get("reasons") or []), "edited_after_verification"})
        quality_status = "needs_review"
    elif objective_changed:
        quality = dict(current.get("quality") or {})
        quality.update(
            {
                "edited_after_verification": True,
                "reasons": sorted({*list(quality.get("reasons") or []), "objective_changed"}),
            }
        )
        quality_status = "draft"
    record, conflict = get_chat_store().update_evidence_matrix(
        matrix_id,
        expected_revision=body.expected_revision,
        title=body.title,
        objective=body.objective,
        rows=rows if body.row_updates else None,
        quality_status=quality_status,
        quality=quality,
    )
    if conflict:
        _conflict_response(record)
    if record is None:
        raise HTTPException(404, "evidence matrix not found")
    return record


@router.post("/evidence-matrices/{matrix_id}/comparison-audits")
def audit_matrix_comparison(matrix_id: str, body: EvidenceComparisonAuditBody):
    current = _matrix_or_404(matrix_id)
    if int(current.get("revision") or 1) != int(body.expected_revision):
        _conflict_response(current)
    if str(current.get("quality_status") or "") != "verified":
        raise HTTPException(400, "comparison audit requires a verified evidence matrix")
    dimensions = [item.model_dump() for item in body.dimensions]
    if {str(item.get("dimension") or "") for item in dimensions} != set(COMPARISON_DIMENSIONS):
        raise HTTPException(400, "comparison audit requires task, dataset, evaluation_protocol, and metric")
    audit = audit_evidence_comparison(
        rows=[item for item in list(current.get("rows") or []) if isinstance(item, dict)],
        spec={
            "mode": body.mode,
            "left_row_id": body.left_row_id,
            "right_row_id": body.right_row_id,
            "dimensions": dimensions,
            "left_target": body.left_target,
            "right_target": body.right_target,
            "target_mapping_confirmed": body.target_mapping_confirmed,
            "left_result": body.left_result,
            "right_result": body.right_result,
        },
        db_dir=get_settings().db_dir,
    )
    audits = [
        item
        for item in list(current.get("comparison_audits") or [])
        if isinstance(item, dict) and str(item.get("id") or "") != str(audit.get("id") or "")
    ]
    audits.append(audit)
    quality = _quality_with_comparisons(dict(current.get("quality") or {}), audits)
    record, conflict = get_chat_store().update_evidence_matrix(
        matrix_id,
        expected_revision=body.expected_revision,
        comparison_audits=audits,
        quality=quality,
    )
    if conflict:
        _conflict_response(record)
    if record is None:
        raise HTTPException(404, "evidence matrix not found")
    return record


@router.delete("/evidence-matrices/{matrix_id}/comparison-audits/{comparison_id}")
def delete_matrix_comparison(
    matrix_id: str,
    comparison_id: str,
    expected_revision: int = Query(..., ge=1),
):
    current = _matrix_or_404(matrix_id)
    if int(current.get("revision") or 1) != int(expected_revision):
        _conflict_response(current)
    audits = [
        item
        for item in list(current.get("comparison_audits") or [])
        if isinstance(item, dict) and str(item.get("id") or "") != str(comparison_id or "")
    ]
    if len(audits) == len(list(current.get("comparison_audits") or [])):
        raise HTTPException(404, "comparison audit not found")
    quality = _quality_with_comparisons(dict(current.get("quality") or {}), audits)
    record, conflict = get_chat_store().update_evidence_matrix(
        matrix_id,
        expected_revision=expected_revision,
        comparison_audits=audits,
        quality=quality,
    )
    if conflict:
        _conflict_response(record)
    if record is None:
        raise HTTPException(404, "evidence matrix not found")
    return record


@router.get("/evidence-matrices/{matrix_id}/revisions")
def list_evidence_matrix_revisions(matrix_id: str, limit: int = Query(40, ge=1, le=200)):
    _matrix_or_404(matrix_id)
    return get_chat_store().list_evidence_matrix_revisions(matrix_id, limit=limit)


@router.get("/evidence-matrices/{matrix_id}/revisions/{revision}")
def get_evidence_matrix_revision(matrix_id: str, revision: int):
    _matrix_or_404(matrix_id)
    record = get_chat_store().get_evidence_matrix_revision(matrix_id, revision)
    if record is None:
        raise HTTPException(404, "evidence matrix revision not found")
    return record


@router.post("/evidence-matrices/{matrix_id}/restore")
def restore_evidence_matrix(matrix_id: str, body: EvidenceMatrixRestoreBody):
    _matrix_or_404(matrix_id)
    record, conflict = get_chat_store().restore_evidence_matrix_revision(
        matrix_id,
        body.revision,
        expected_revision=body.expected_revision,
    )
    if conflict:
        _conflict_response(record)
    if record is None:
        raise HTTPException(404, "evidence matrix revision not found")
    return record


@router.delete("/evidence-matrices/{matrix_id}")
def delete_evidence_matrix(matrix_id: str):
    record = _matrix_or_404(matrix_id)
    snapshot = create_auto_snapshot(
        get_settings(),
        action="evidence_matrix_delete",
        label=matrix_id,
        metadata={
            "matrix_id": matrix_id,
            "project_id": str(record.get("project_id") or ""),
            "revision": int(record.get("revision") or 1),
        },
    )
    if bool(snapshot.get("block_operation")):
        detail = str(snapshot.get("error") or snapshot.get("reason") or "automatic backup failed")
        raise HTTPException(503, f"automatic backup failed before evidence_matrix_delete: {detail}")
    if not get_chat_store().delete_evidence_matrix(matrix_id):
        raise HTTPException(404, "evidence matrix not found")
    return {"ok": True, "auto_backup": snapshot}


@router.get("/evidence-matrices/{matrix_id}/export")
def export_evidence_matrix(
    matrix_id: str,
    format: Literal["markdown", "csv", "xlsx"] = Query("markdown"),
):
    record = _matrix_or_404(matrix_id)
    if format == "xlsx":
        content = evidence_matrix_xlsx(record)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        suffix = "xlsx"
    elif format == "csv":
        content = evidence_matrix_csv(record)
        media_type = "text/csv; charset=utf-8"
        suffix = "csv"
    else:
        content = evidence_matrix_markdown(record).encode("utf-8")
        media_type = "text/markdown; charset=utf-8"
        suffix = "md"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{_download_name(record, suffix)}"'},
    )
