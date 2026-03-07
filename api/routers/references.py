from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.deps import get_chat_store, get_settings, load_prefs
from api.reference_ui import enrich_citation_detail_meta, enrich_refs_payload, ensure_source_citation_meta, open_reference_source
from kb.library_store import LibraryStore
from api.sse import sse_generator, sse_response
from kb.reference_sync import (
    start_reference_sync,
    snapshot as refsync_snapshot,
)

router = APIRouter(prefix="/api/references", tags=["references"])


def _md_dir() -> Path:
    from api.routers.library import _md_dir
    return _md_dir()


def _pdf_dir() -> Path:
    from api.routers.library import _pdf_dir
    return _pdf_dir()


def _lib_store() -> LibraryStore:
    return LibraryStore(get_settings().library_db_path)


@router.post("/sync")
def start_sync(workers: int | None = None, crossref_budget_s: float | None = None):
    s = get_settings()
    try:
        workers_default = int(os.environ.get("KB_REFSYNC_WORKERS", "6") or 6)
    except Exception:
        workers_default = 6
    if workers is None:
        workers = workers_default
    workers_final = int(max(1, min(16, int(workers))))

    try:
        budget_default = float(os.environ.get("KB_CROSSREF_BUDGET_S", "45") or 45.0)
    except Exception:
        budget_default = 45.0
    if crossref_budget_s is None:
        crossref_budget_s = budget_default
    budget_final = float(max(5.0, min(180.0, float(crossref_budget_s))))

    result = start_reference_sync(
        src_root=_md_dir(),
        db_dir=s.db_dir,
        pdf_root=_pdf_dir(),
        library_db_path=s.library_db_path,
        crossref_time_budget_s=budget_final,
        doi_prefetch_workers=workers_final,
    )
    return result


@router.get("/sync/status")
async def sync_status():
    def poll():
        snap = refsync_snapshot()
        return {
            **snap,
            "done": snap.get("status") in ("done", "error", "idle"),
        }
    return sse_response(sse_generator(poll, interval=0.5))


@router.get("/conversation/{conv_id}")
def get_conversation_refs(conv_id: str):
    refs = get_chat_store().list_message_refs(conv_id)
    return enrich_refs_payload(refs, pdf_root=_pdf_dir(), md_root=_md_dir(), lib_store=_lib_store())


class OpenReferenceBody(BaseModel):
    source_path: str
    page: int | None = None


class CitationMetaBody(BaseModel):
    source_path: str


class BibliometricsBody(BaseModel):
    meta: dict


_ASSET_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


@router.post("/open")
def open_reference(body: OpenReferenceBody):
    ok, message = open_reference_source(
        source_path=body.source_path,
        pdf_root=_pdf_dir(),
        page=body.page,
    )
    if not ok:
        raise HTTPException(404, message)
    return {"ok": True, "message": message}


@router.post("/citation-meta")
def get_reference_citation_meta(body: CitationMetaBody):
    return ensure_source_citation_meta(
        source_path=body.source_path,
        pdf_root=_pdf_dir(),
        md_root=_md_dir(),
        lib_store=_lib_store(),
    )


@router.post("/bibliometrics")
def get_bibliometrics(body: BibliometricsBody):
    return enrich_citation_detail_meta(body.meta or {})


@router.get("/asset")
def get_reference_asset(path: str):
    raw = str(path or "").strip()
    if not raw:
        raise HTTPException(404, "asset not found")
    try:
        resolved = Path(raw).expanduser().resolve()
    except Exception:
        raise HTTPException(404, "asset not found")
    if (not resolved.exists()) or (not resolved.is_file()):
        raise HTTPException(404, "asset not found")
    if resolved.suffix.lower() not in _ASSET_IMAGE_EXTS:
        raise HTTPException(404, "asset not found")
    root = _md_dir().resolve()
    try:
        resolved.relative_to(root)
    except Exception:
        raise HTTPException(404, "asset not found")
    media_type = str(mimetypes.guess_type(str(resolved))[0] or "application/octet-stream")
    return FileResponse(str(resolved), media_type=media_type, filename=resolved.name)
