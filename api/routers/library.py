from __future__ import annotations

import os
import subprocess
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

from api.deps import get_settings, load_prefs
from api.sse import sse_generator, sse_response
from kb.task_runtime import (
    _bg_enqueue,
    _bg_snapshot,
    _bg_cancel_all,
    _build_bg_task,
    _bg_ensure_started,
)
from kb.file_ops import (
    _list_pdf_paths_fast,
    _next_pdf_dest_path,
    _sha1_bytes,
    _cleanup_tmp_uploads,
    _cleanup_tmp_md_artifacts,
)

router = APIRouter(prefix="/api/library", tags=["library"])


def _pdf_dir() -> Path:
    prefs = load_prefs()
    s = get_settings()
    return Path(prefs.get("pdf_dir") or os.environ.get("KB_PDF_DIR") or str(Path(s.db_dir).parent / "pdfs")).expanduser().resolve()


def _md_dir() -> Path:
    prefs = load_prefs()
    s = get_settings()
    return Path(prefs.get("md_dir") or os.environ.get("KB_MD_DIR") or str(Path(s.db_dir).parent / "md_output")).expanduser().resolve()


@router.get("/pdfs")
def list_pdfs():
    d = _pdf_dir()
    if not d.exists():
        return []
    return [{"name": p.name, "path": str(p)} for p in _list_pdf_paths_fast(d)]


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), base_name: str = Form("")):
    pdf_d = _pdf_dir()
    pdf_d.mkdir(parents=True, exist_ok=True)
    data = await file.read()
    sha1 = _sha1_bytes(data)

    # duplicate check
    for existing in _list_pdf_paths_fast(pdf_d):
        try:
            if _sha1_bytes(existing.read_bytes()) == sha1:
                return {"duplicate": True, "existing": existing.name, "sha1": sha1}
        except Exception:
            continue

    name = (base_name.strip() or Path(file.filename or "upload").stem).strip() or "upload"
    dest = _next_pdf_dest_path(pdf_d, name)
    dest.write_bytes(data)
    return {"duplicate": False, "path": str(dest), "name": dest.name, "sha1": sha1}


class ConvertBody(BaseModel):
    pdf_name: str
    speed_mode: str = "balanced"
    no_llm: bool = False
    replace: bool = False


@router.post("/convert")
def start_convert(body: ConvertBody):
    s = get_settings()
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    md_d.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_d / body.pdf_name
    task = _build_bg_task(
        pdf_path=pdf_path,
        out_root=md_d,
        db_dir=s.db_dir,
        no_llm=body.no_llm,
        replace=body.replace,
        speed_mode=body.speed_mode,
    )
    _bg_enqueue(task)
    return {"ok": True, "task_id": task.get("_tid", "")}


@router.get("/convert/status")
async def convert_status():
    def poll():
        snap = _bg_snapshot()
        return {
            "running": snap.get("running", False),
            "done": not snap.get("running", False) and snap.get("total", 0) > 0,
            "total": snap.get("total", 0),
            "completed": snap.get("done", 0),
            "current": snap.get("current", ""),
            "cur_page_done": snap.get("cur_page_done", 0),
            "cur_page_total": snap.get("cur_page_total", 0),
            "cur_page_msg": snap.get("cur_page_msg", ""),
            "last": snap.get("last", ""),
        }
    return sse_response(sse_generator(poll, interval=0.5))


@router.post("/convert/cancel")
def cancel_convert():
    _bg_cancel_all()
    return {"ok": True}


@router.post("/reindex")
def reindex():
    s = get_settings()
    md_d = _md_dir()
    ingest_py = Path(__file__).resolve().parent.parent.parent / "ingest.py"
    if not ingest_py.exists():
        return {"ok": False, "error": "ingest.py not found"}
    result = subprocess.run(
        [os.sys.executable, str(ingest_py), "--src", str(md_d), "--db", str(s.db_dir), "--incremental", "--prune"],
        capture_output=True, text=True, timeout=300,
    )
    return {"ok": result.returncode == 0, "stdout": result.stdout[-500:], "stderr": result.stderr[-500:]}
