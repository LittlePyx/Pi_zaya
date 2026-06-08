from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.deps import get_settings
from api.routers.settings import production_readiness_payload
from kb.maintenance import (
    acknowledge_latest_restore_review,
    cleanup_backup_archives,
    create_backup_archive,
    create_diagnostics_archive,
    list_backup_archives,
    maintenance_status,
    public_restore_audit_events,
    resolve_backup_archive,
    restore_backup_archive,
    restore_dry_run_backup_archive,
    verify_backup_archive,
)

router = APIRouter(prefix="/api/maintenance", tags=["maintenance"])


class CreateBackupBody(BaseModel):
    label: str | None = None


class CleanupBackupsBody(BaseModel):
    keep: int | None = None
    dry_run: bool = False


class RestoreBackupBody(BaseModel):
    confirm: str = ""
    components: dict[str, bool] = {}
    create_pre_restore_backup: bool = True
    force: bool = False


class RestoreReviewAcknowledgeBody(BaseModel):
    checks: dict[str, bool] = Field(default_factory=dict)


@router.get("/diagnostics/export")
def export_diagnostics():
    settings = get_settings()
    archive = create_diagnostics_archive(settings, readiness_payload=production_readiness_payload(settings))
    return FileResponse(
        str(archive),
        media_type="application/zip",
        filename=archive.name,
    )


@router.get("/status")
def get_maintenance_status():
    return maintenance_status(get_settings())


@router.get("/backups")
def list_backups():
    return {"items": list_backup_archives()}


@router.post("/backups")
def create_backup(body: CreateBackupBody | None = None):
    settings = get_settings()
    return create_backup_archive(settings, label=(body.label if body else "") or "")


@router.get("/restore-audit")
def list_restore_audit(limit: int = 20):
    safe_limit = max(1, min(int(limit or 20), 50))
    return {"items": public_restore_audit_events(limit=safe_limit)}


@router.post("/restore-review/acknowledge")
def acknowledge_restore_review(body: RestoreReviewAcknowledgeBody | None = None):
    result = acknowledge_latest_restore_review(checks=(body.checks if body else None))
    if not bool(result.get("ok")):
        raise HTTPException(status_code=409, detail=result)
    return result


@router.post("/backups/cleanup")
def cleanup_backups(body: CleanupBackupsBody | None = None):
    keep = body.keep if body else None
    if keep is not None and int(keep) < 1:
        raise HTTPException(status_code=400, detail="keep must be >= 1")
    return cleanup_backup_archives(keep=keep, dry_run=bool(body.dry_run) if body else False)


@router.get("/backups/{name}/verify")
def verify_backup(name: str):
    try:
        archive = resolve_backup_archive(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="backup not found") from exc
    return verify_backup_archive(archive)


@router.get("/backups/{name}/restore-dry-run")
def restore_backup_dry_run(name: str):
    try:
        archive = resolve_backup_archive(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="backup not found") from exc
    return restore_dry_run_backup_archive(get_settings(), archive)


@router.post("/backups/{name}/restore")
def restore_backup(name: str, body: RestoreBackupBody):
    try:
        archive = resolve_backup_archive(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="backup not found") from exc
    expected = f"RESTORE {archive.name}"
    if str(body.confirm or "") != expected:
        raise HTTPException(status_code=400, detail={"message": "confirmation text mismatch", "expected": expected})
    result = restore_backup_archive(
        get_settings(),
        archive,
        confirm=body.confirm,
        components=body.components,
        create_pre_restore_backup=bool(body.create_pre_restore_backup),
        force=bool(body.force),
    )
    if not bool(result.get("ok")):
        raise HTTPException(status_code=409, detail=result)
    return result


@router.get("/backups/{name}")
def download_backup(name: str):
    try:
        archive = resolve_backup_archive(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="backup not found") from exc
    return FileResponse(
        str(archive),
        media_type="application/zip",
        filename=archive.name,
    )
