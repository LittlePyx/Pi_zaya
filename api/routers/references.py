from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter

from api.deps import get_settings, load_prefs
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


@router.post("/sync")
def start_sync():
    s = get_settings()
    result = start_reference_sync(
        src_root=_md_dir(),
        db_dir=s.db_dir,
        pdf_root=_pdf_dir(),
        library_db_path=s.library_db_path,
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
