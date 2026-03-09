from __future__ import annotations

import hashlib
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi import File, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.chat_render import enrich_messages_with_reference_render
from api.deps import get_chat_store, get_settings
from api.routers.library import (
    _md_dir,
    auto_rename_saved_pdf_in_library,
    quick_ingest_pdf,
    refine_pdf_with_full_llm_replace,
    save_pdf_to_library,
)
from kb.file_ops import _path_exists
from kb.pdf_tools import ensure_dir
from kb.task_runtime import kickoff_paper_guide_prefetch

router = APIRouter(prefix="/api", tags=["chat"])


class CreateConvBody(BaseModel):
    title: str = "新对话"
    project_id: str | None = None
    mode: str = "normal"
    bound_source_path: str = ""
    bound_source_name: str = ""
    bound_source_ready: bool = False


class CreateProjectBody(BaseModel):
    name: str = "未命名项目"


class AppendMsgBody(BaseModel):
    role: str = "user"
    content: str


class UpdateMsgBody(BaseModel):
    content: str


class UpdateTitleBody(BaseModel):
    title: str


class UpdateProjectBody(BaseModel):
    project_id: str | None = None


class UpdateConversationGuideBody(BaseModel):
    mode: str | None = None
    bound_source_path: str | None = None
    bound_source_name: str | None = None
    bound_source_ready: bool | None = None


class RenameProjectBody(BaseModel):
    name: str


class UploadJobBody(BaseModel):
    job_id: str


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
IMAGE_MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}
_CHAT_UPLOAD_JOB_LOCK = threading.Lock()
_CHAT_UPLOAD_JOBS: dict[str, dict] = {}
_CHAT_QUALITY_REFINE_LOCK = threading.Lock()
_CHAT_QUALITY_REFINE_RUNNING: set[str] = set()


def _chat_image_dir() -> Path:
    from api.deps import get_settings

    settings = get_settings()
    out = Path(settings.db_dir) / "_chat_uploads" / "images"
    ensure_dir(out)
    return out


def _safe_upload_stem(name: str) -> str:
    s0 = str(name or "").strip()
    out_chars: list[str] = []
    for ch in s0:
        try:
            if ch.isalnum():
                out_chars.append(ch)
            elif ch in ("-", "_", "."):
                out_chars.append(ch)
            elif ch.isspace():
                out_chars.append("-")
            else:
                out_chars.append("_")
        except Exception:
            out_chars.append("_")
    out = "".join(out_chars).strip(" ._-")
    return (out or "upload")[:72]


def _chat_image_url(path: str) -> str:
    return f"/api/chat/uploads/image?path={quote(str(path or '').strip(), safe='')}"


def _normalize_chat_image_attachment(item: dict) -> dict:
    rec = dict(item or {})
    path = str(rec.get("path") or "").strip()
    if path:
        rec["url"] = _chat_image_url(path)
    return rec


def _normalize_message_attachments(message: dict) -> dict:
    rec = dict(message or {})
    attachments = []
    for item in list(rec.get("attachments") or []):
        if not isinstance(item, dict):
            continue
        attachments.append(_normalize_chat_image_attachment(item))
    rec["attachments"] = attachments
    return rec


def _bind_pdf_source_to_conversation(*, conv_id: str, source_path: str, source_name: str = "") -> None:
    cid = str(conv_id or "").strip()
    src = str(source_path or "").strip()
    if (not cid) or (not src):
        return
    try:
        store = get_chat_store()
        if not store.get_conversation(cid):
            return
        store.bind_conversation_source(cid, src, source_name)
    except Exception:
        return


def _kickoff_paper_guide_prefetch_if_needed(
    *,
    mode: str,
    source_path: str,
    source_name: str,
    source_ready: bool,
) -> None:
    mode_norm = str(mode or "").strip().lower()
    src = str(source_path or "").strip()
    if mode_norm != "paper_guide":
        return
    if (not src) or (not bool(source_ready)):
        return
    try:
        settings = get_settings()
    except Exception:
        settings = None
    try:
        md_root = _md_dir()
    except Exception:
        md_root = None
    try:
        kickoff_paper_guide_prefetch(
            source_path=src,
            source_name=str(source_name or "").strip(),
            db_dir=(getattr(settings, "db_dir", None) if settings is not None else None),
            md_root=md_root,
            library_db_path=(getattr(settings, "library_db_path", None) if settings is not None else None),
        )
    except Exception:
        return


def _coerce_bool_flag(value: object) -> bool:
    try:
        return bool(int(value or 0))
    except Exception:
        return bool(value)


def _chat_pdf_ingest_status_payload(job_id: str, record: dict) -> dict:
    rec = dict(record or {})
    ingest_status = str(rec.get("ingest_status") or "")
    return {
        "ingest_job_id": job_id,
        "kind": "pdf",
        "name": str(rec.get("name") or ""),
        "sha1": str(rec.get("sha1") or ""),
        "path": str(rec.get("path") or ""),
        "status": "saved" if ingest_status in {"processing", "renaming", "converting", "ingesting", "ready"} else ("duplicate" if ingest_status == "duplicate" else "error"),
        "ready": bool(rec.get("ready")),
        "ingest_status": ingest_status,
        "md_path": str(rec.get("md_path") or ""),
        "error": str(rec.get("error") or ""),
        "quality_status": str(rec.get("quality_status") or ""),
        "quality_stage": str(rec.get("quality_stage") or ""),
        "quality_error": str(rec.get("quality_error") or ""),
    }


def _speed_mode_needs_quality_refine(speed_mode: str) -> bool:
    mode = str(speed_mode or "").strip().lower()
    return mode in {"ultra_fast", "no_llm", "fast"}


def _start_chat_pdf_quality_refine(job_id: str) -> None:
    rec = _get_chat_pdf_ingest_job(job_id) or {}
    if not isinstance(rec, dict):
        return
    if (not bool(rec.get("ready"))) or str(rec.get("ingest_status") or "") != "ready":
        return
    if str(rec.get("quality_status") or "") not in {"pending", "error"}:
        return
    pdf_path = Path(str(rec.get("path") or "")).expanduser()
    if (not str(pdf_path)) or (not _path_exists(pdf_path)):
        _set_chat_pdf_ingest_job(job_id, {"quality_status": "error", "quality_error": "pdf file not found"})
        return
    try:
        dedupe_key = str(pdf_path.resolve())
    except Exception:
        dedupe_key = str(pdf_path)
    with _CHAT_QUALITY_REFINE_LOCK:
        if dedupe_key in _CHAT_QUALITY_REFINE_RUNNING:
            return
        _CHAT_QUALITY_REFINE_RUNNING.add(dedupe_key)
    _set_chat_pdf_ingest_job(
        job_id,
        {
            "cancel_requested": False,
            "quality_status": "running",
            "quality_stage": "refining",
            "quality_error": "",
        },
    )

    def _run() -> None:
        try:
            result = refine_pdf_with_full_llm_replace(
                pdf_path=pdf_path,
                progress_cb=lambda stage: _set_chat_pdf_ingest_job(job_id, {"quality_status": "running", "quality_stage": str(stage or "refining")}),
                cancel_cb=lambda: _chat_pdf_ingest_cancel_requested(job_id),
            )
            if bool(result.get("ready")):
                rec_done = _get_chat_pdf_ingest_job(job_id) or {}
                md_path = str(result.get("md_path") or rec_done.get("md_path") or "")
                _set_chat_pdf_ingest_job(
                    job_id,
                    {
                        "quality_status": "ready",
                        "quality_stage": "ready",
                        "quality_error": "",
                        "md_path": md_path,
                    },
                )
                _bind_pdf_source_to_conversation(
                    conv_id=str(rec_done.get("conv_id") or ""),
                    source_path=md_path or str(rec_done.get("path") or ""),
                    source_name=str(rec_done.get("name") or ""),
                )
            elif bool(result.get("cancelled")):
                _set_chat_pdf_ingest_job(job_id, {"quality_status": "cancelled", "quality_stage": "cancelled", "quality_error": "cancelled"})
            else:
                _set_chat_pdf_ingest_job(
                    job_id,
                    {
                        "quality_status": "error",
                        "quality_stage": "error",
                        "quality_error": str(result.get("error") or "quality refine failed"),
                    },
                )
        except Exception as exc:
            _set_chat_pdf_ingest_job(
                job_id,
                {
                    "quality_status": "error",
                    "quality_stage": "error",
                    "quality_error": str(exc),
                },
            )
        finally:
            with _CHAT_QUALITY_REFINE_LOCK:
                _CHAT_QUALITY_REFINE_RUNNING.discard(dedupe_key)

    threading.Thread(target=_run, daemon=True, name=f"chat_pdf_refine_{job_id[:8]}").start()


def _set_chat_pdf_ingest_job(job_id: str, payload: dict) -> None:
    with _CHAT_UPLOAD_JOB_LOCK:
        current = dict(_CHAT_UPLOAD_JOBS.get(job_id) or {})
        current.update(payload or {})
        current["updated_at"] = time.time()
        _CHAT_UPLOAD_JOBS[job_id] = current


def _get_chat_pdf_ingest_job(job_id: str) -> dict | None:
    with _CHAT_UPLOAD_JOB_LOCK:
        rec = _CHAT_UPLOAD_JOBS.get(job_id)
        if not isinstance(rec, dict):
            return None
        return dict(rec)


def _chat_pdf_ingest_cancel_requested(job_id: str) -> bool:
    with _CHAT_UPLOAD_JOB_LOCK:
        rec = _CHAT_UPLOAD_JOBS.get(job_id)
        return bool(isinstance(rec, dict) and rec.get("cancel_requested"))


def _terminate_job_proc(proc: object | None) -> None:
    if proc is None:
        return
    try:
        poll = getattr(proc, "poll", None)
        if callable(poll) and poll() is not None:
            return
    except Exception:
        return
    try:
        terminate = getattr(proc, "terminate", None)
        wait = getattr(proc, "wait", None)
        if callable(terminate):
            terminate()
        if callable(wait):
            wait(timeout=4)
    except Exception:
        pass
    try:
        poll = getattr(proc, "poll", None)
        kill = getattr(proc, "kill", None)
        wait = getattr(proc, "wait", None)
        if callable(poll) and poll() is None and callable(kill):
            kill()
        if callable(wait):
            wait(timeout=2)
    except Exception:
        pass


def _start_chat_pdf_ingest_job(
    *,
    pdf_path: Path,
    speed_mode: str,
    display_name: str,
    sha1: str = "",
    conv_id: str = "",
) -> str:
    job_id = uuid.uuid4().hex
    quality_pending = _speed_mode_needs_quality_refine(speed_mode)
    _set_chat_pdf_ingest_job(
        job_id,
        {
            "job_id": job_id,
            "name": display_name,
            "sha1": sha1,
            "path": str(pdf_path),
            "conv_id": str(conv_id or "").strip(),
            "ready": False,
            "ingest_status": "renaming",
            "speed_mode": str(speed_mode or "balanced"),
            "cancel_requested": False,
            "error": "",
            "md_path": "",
            "quality_status": "pending" if quality_pending else "none",
            "quality_stage": "pending" if quality_pending else "",
            "quality_error": "",
            "created_at": time.time(),
        },
    )

    def _run() -> None:
        def _progress(stage: str) -> None:
            if _chat_pdf_ingest_cancel_requested(job_id):
                return
            _set_chat_pdf_ingest_job(job_id, {"ingest_status": str(stage or "processing")})

        def _set_ingest_proc(proc: object | None) -> None:
            _set_chat_pdf_ingest_job(job_id, {"ingest_proc": proc})

        current_pdf_path = pdf_path
        current_display_name = str(display_name or pdf_path.name)
        if not _chat_pdf_ingest_cancel_requested(job_id):
            try:
                renamed = auto_rename_saved_pdf_in_library(pdf_path=pdf_path, base_name=current_display_name)
                renamed_path = Path(str(renamed.get("path") or "")).expanduser()
                if _path_exists(renamed_path):
                    current_pdf_path = renamed_path
                current_display_name = str(renamed.get("name") or current_display_name)
                _set_chat_pdf_ingest_job(
                    job_id,
                    {
                        "path": str(current_pdf_path),
                        "name": current_display_name,
                        "sha1": str(renamed.get("sha1") or sha1 or ""),
                    },
                )
            except Exception:
                current_pdf_path = pdf_path
                current_display_name = str(display_name or pdf_path.name)

        try:
            result = quick_ingest_pdf(
                pdf_path=current_pdf_path,
                speed_mode=speed_mode,
                progress_cb=_progress,
                cancel_cb=lambda: _chat_pdf_ingest_cancel_requested(job_id),
                ingest_proc_cb=_set_ingest_proc,
            )
        except Exception as exc:
            result = {"ready": False, "error": str(exc)}
        if _chat_pdf_ingest_cancel_requested(job_id) or bool(result.get("cancelled")):
            _set_chat_pdf_ingest_job(
                job_id,
                {
                    "ready": False,
                    "ingest_status": "cancelled",
                    "error": "cancelled",
                    "ingest_proc": None,
                },
            )
            return
        _set_chat_pdf_ingest_job(
            job_id,
            {
                "ready": bool(result.get("ready")),
                "ingest_status": "ready" if bool(result.get("ready")) else "error",
                "error": str(result.get("error") or ""),
                "md_path": str(result.get("md_path") or ""),
                "ingest_proc": None,
            },
        )
        if bool(result.get("ready")):
            rec_done = _get_chat_pdf_ingest_job(job_id) or {}
            _bind_pdf_source_to_conversation(
                conv_id=str(rec_done.get("conv_id") or ""),
                source_path=str(rec_done.get("md_path") or rec_done.get("path") or ""),
                source_name=str(rec_done.get("name") or ""),
            )
            if _speed_mode_needs_quality_refine(speed_mode) and bool(result.get("out_folder")):
                _start_chat_pdf_quality_refine(job_id)

    threading.Thread(target=_run, daemon=True, name=f"chat_pdf_ingest_{job_id[:8]}").start()
    return job_id


def _cancel_chat_pdf_ingest_job(job_id: str) -> dict | None:
    proc: object | None = None
    with _CHAT_UPLOAD_JOB_LOCK:
        rec = _CHAT_UPLOAD_JOBS.get(job_id)
        if not isinstance(rec, dict):
            return None
        current = dict(rec)
        ingest_status = str(current.get("ingest_status") or "")
        quality_status = str(current.get("quality_status") or "")
        ingest_running = ingest_status in {"processing", "renaming", "converting", "ingesting"}
        quality_running = quality_status in {"pending", "running"}
        if (not ingest_running) and (not quality_running):
            return current

        current["cancel_requested"] = True
        if ingest_running:
            current["ingest_status"] = "cancelled"
            current["ready"] = False
            current["error"] = "cancelled"
            proc = current.get("ingest_proc")
        if quality_running:
            current["quality_status"] = "cancelled"
            current["quality_stage"] = "cancelled"
            current["quality_error"] = "cancelled"
        _CHAT_UPLOAD_JOBS[job_id] = current
    _terminate_job_proc(proc)
    return _get_chat_pdf_ingest_job(job_id)


def _retry_chat_pdf_ingest_job(job_id: str) -> dict | None:
    rec = _get_chat_pdf_ingest_job(job_id)
    if not isinstance(rec, dict):
        return None
    status = str(rec.get("ingest_status") or "")
    if status in {"processing", "renaming", "converting", "ingesting"}:
        raise RuntimeError("job still running")
    pdf_path = Path(str(rec.get("path") or "")).expanduser()
    if not _path_exists(pdf_path):
        raise FileNotFoundError("pdf file not found")
    new_job_id = _start_chat_pdf_ingest_job(
        pdf_path=pdf_path,
        speed_mode=str(rec.get("speed_mode") or "balanced"),
        display_name=str(rec.get("name") or pdf_path.name),
        sha1=str(rec.get("sha1") or ""),
        conv_id=str(rec.get("conv_id") or ""),
    )
    return _get_chat_pdf_ingest_job(new_job_id)


def _retry_chat_pdf_quality_refine_job(job_id: str) -> dict | None:
    rec = _get_chat_pdf_ingest_job(job_id)
    if not isinstance(rec, dict):
        return None
    if (not bool(rec.get("ready"))) or str(rec.get("ingest_status") or "") != "ready":
        raise RuntimeError("ingest not ready")
    quality_status = str(rec.get("quality_status") or "")
    if quality_status in {"pending", "running"}:
        raise RuntimeError("quality refine still running")
    if quality_status in {"none", ""}:
        raise RuntimeError("quality refine not enabled for this job")
    pdf_path = Path(str(rec.get("path") or "")).expanduser()
    if not _path_exists(pdf_path):
        raise FileNotFoundError("pdf file not found")
    _set_chat_pdf_ingest_job(
        job_id,
        {
            "cancel_requested": False,
            "quality_status": "pending",
            "quality_stage": "pending",
            "quality_error": "",
        },
    )
    _start_chat_pdf_quality_refine(job_id)
    return _get_chat_pdf_ingest_job(job_id)


def _detect_image_ext(raw_name: str, raw_mime: str, data: bytes) -> str:
    suffix = Path(raw_name).suffix.lower()
    if suffix in IMAGE_EXTS:
        return suffix
    ext = IMAGE_MIME_TO_EXT.get(str(raw_mime or "").strip().lower(), "")
    if ext:
        return ext
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    if (len(data) >= 12) and (data[:4] == b"RIFF") and (data[8:12] == b"WEBP"):
        return ".webp"
    if data.startswith(b"BM"):
        return ".bmp"
    return ".png"


def _save_chat_image(*, raw_name: str, raw_mime: str, data: bytes, sha1: str) -> dict:
    img_dir = _chat_image_dir()
    ext = _detect_image_ext(raw_name, raw_mime, data)
    stem_seed = Path(raw_name).stem or f"pasted-{int(time.time())}"
    safe_stem = _safe_upload_stem(stem_seed)
    dest_img = img_dir / f"{safe_stem}-{sha1[:10]}{ext}"
    duplicate = dest_img.exists()
    if not duplicate:
        dest_img.write_bytes(data)
    return {
        "kind": "image",
        "status": "duplicate" if duplicate else "saved",
        "name": raw_name or dest_img.name,
        "sha1": sha1,
        "mime": raw_mime or "image/*",
        "path": str(dest_img),
        "attachment": {
            "sha1": sha1,
            "path": str(dest_img),
            "name": raw_name or dest_img.name,
            "mime": raw_mime or "image/*",
            "url": _chat_image_url(str(dest_img)),
        },
    }


@router.get("/chat/uploads/image")
def get_chat_upload_image(path: str):
    root = _chat_image_dir().resolve()
    try:
        resolved = Path(str(path or "")).expanduser().resolve()
    except Exception:
        raise HTTPException(404, "image not found")
    if resolved != root and root not in resolved.parents:
        raise HTTPException(404, "image not found")
    if (not resolved.exists()) or (not resolved.is_file()):
        raise HTTPException(404, "image not found")
    media_type = IMAGE_MIME_TO_EXT.get("", "")
    suffix = resolved.suffix.lower()
    if suffix == ".png":
        media_type = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    elif suffix == ".webp":
        media_type = "image/webp"
    elif suffix == ".gif":
        media_type = "image/gif"
    elif suffix == ".bmp":
        media_type = "image/bmp"
    else:
        media_type = "application/octet-stream"
    return FileResponse(str(resolved), media_type=media_type, filename=resolved.name)


@router.get("/chat/uploads/status")
def get_chat_upload_status(job_ids: str = ""):
    wanted = [str(x or "").strip() for x in str(job_ids or "").split(",") if str(x or "").strip()]
    if not wanted:
        return {"items": []}
    items: list[dict] = []
    with _CHAT_UPLOAD_JOB_LOCK:
        for job_id in wanted:
            rec = _CHAT_UPLOAD_JOBS.get(job_id)
            if not isinstance(rec, dict):
                continue
            items.append(_chat_pdf_ingest_status_payload(job_id, rec))
    return {"items": items}


@router.post("/chat/uploads/cancel")
def cancel_chat_upload_job(body: UploadJobBody):
    rec = _cancel_chat_pdf_ingest_job(str(body.job_id or "").strip())
    if rec is None:
        raise HTTPException(404, "upload job not found")
    return {"item": _chat_pdf_ingest_status_payload(str(body.job_id or "").strip(), rec)}


@router.post("/chat/uploads/retry")
def retry_chat_upload_job(body: UploadJobBody):
    job_id = str(body.job_id or "").strip()
    if not job_id:
        raise HTTPException(400, "job_id required")
    try:
        rec = _retry_chat_pdf_ingest_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    if rec is None:
        raise HTTPException(404, "upload job not found")
    new_job_id = str(rec.get("job_id") or "")
    return {"item": _chat_pdf_ingest_status_payload(new_job_id, rec)}


@router.post("/chat/uploads/quality/retry")
def retry_chat_upload_quality_job(body: UploadJobBody):
    job_id = str(body.job_id or "").strip()
    if not job_id:
        raise HTTPException(400, "job_id required")
    try:
        rec = _retry_chat_pdf_quality_refine_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    if rec is None:
        raise HTTPException(404, "upload job not found")
    return {"item": _chat_pdf_ingest_status_payload(job_id, rec)}


@router.post("/chat/uploads")
async def upload_chat_files(
    files: list[UploadFile] = File(...),
    quick_ingest: bool = Form(True),
    speed_mode: str = Form("balanced"),
    conv_id: str = Form(""),
):
    results: list[dict] = []
    seen_sha1: set[str] = set()
    conv_id_norm = str(conv_id or "").strip()

    for up in list(files or []):
        raw_name = str(getattr(up, "filename", "") or "").strip()
        raw_mime = str(getattr(up, "content_type", "") or "").strip().lower()
        try:
            data = await up.read()
        except Exception:
            data = b""
        if not data:
            results.append({
                "kind": "unknown",
                "status": "error",
                "name": raw_name or "upload",
                "error": "empty upload",
            })
            continue

        sha1 = hashlib.sha1(data).hexdigest()
        if sha1 in seen_sha1:
            results.append({
                "kind": "unknown",
                "status": "duplicate",
                "name": raw_name or "upload",
                "sha1": sha1,
            })
            continue
        seen_sha1.add(sha1)

        suffix = Path(raw_name).suffix.lower()
        is_pdf = bool((suffix == ".pdf") or (raw_mime == "application/pdf") or data.startswith(b"%PDF"))
        is_image = bool(raw_mime.startswith("image/") or (suffix in IMAGE_EXTS))

        if is_image and (not is_pdf):
            try:
                results.append(_save_chat_image(raw_name=raw_name, raw_mime=raw_mime, data=data, sha1=sha1))
            except Exception as exc:
                results.append({
                    "kind": "image",
                    "status": "error",
                    "name": raw_name or "image",
                    "sha1": sha1,
                    "error": str(exc),
                })
            continue

        if is_pdf:
            try:
                saved = save_pdf_to_library(file_name=raw_name or "upload.pdf", data=data, fast_mode=True)
                result = {
                    "kind": "pdf",
                    "status": "duplicate" if saved.get("duplicate") else "saved",
                    "name": str(saved.get("name") or raw_name or "upload.pdf"),
                    "sha1": sha1,
                    "path": str(saved.get("path") or ""),
                    "duplicate": bool(saved.get("duplicate")),
                    "existing": str(saved.get("existing") or ""),
                    "ready": False,
                    "ingest_status": "idle",
                    "quality_status": "none",
                    "quality_stage": "",
                    "quality_error": "",
                }
                if (not saved.get("duplicate")) and quick_ingest:
                    pdf_path = Path(str(saved.get("path") or "")).expanduser()
                    if _path_exists(pdf_path):
                        job_id = _start_chat_pdf_ingest_job(
                            pdf_path=pdf_path,
                            speed_mode=speed_mode,
                            display_name=str(result.get("name") or pdf_path.name),
                            sha1=sha1,
                            conv_id=conv_id_norm,
                        )
                        result["ingest_job_id"] = job_id
                        result["ingest_status"] = "renaming"
                        if _speed_mode_needs_quality_refine(speed_mode):
                            result["quality_status"] = "pending"
                            result["quality_stage"] = "pending"
                    else:
                        result["status"] = "error"
                        result["ingest_status"] = "error"
                        result["error"] = "pdf saved but ingest job not started"
                else:
                    result["ready"] = bool(saved.get("duplicate"))
                    result["ingest_status"] = "ready" if bool(saved.get("duplicate")) else "idle"
                    result["quality_status"] = "none"
                    if bool(saved.get("duplicate")):
                        _bind_pdf_source_to_conversation(
                            conv_id=conv_id_norm,
                            source_path=str(saved.get("path") or ""),
                            source_name=str(saved.get("name") or raw_name or "upload.pdf"),
                        )
                results.append(result)
            except Exception as exc:
                results.append({
                    "kind": "pdf",
                    "status": "error",
                    "name": raw_name or "upload.pdf",
                    "sha1": sha1,
                    "error": str(exc),
                })
            continue

        results.append({
            "kind": "unknown",
            "status": "unsupported",
            "name": raw_name or "upload",
            "sha1": sha1,
            "mime": raw_mime,
            "error": "unsupported file type",
        })

    return {"items": results}


@router.get("/projects")
def list_projects():
    return get_chat_store().list_projects()


@router.post("/projects")
def create_project(body: CreateProjectBody):
    project_id = get_chat_store().create_project(body.name)
    return {"id": project_id}


@router.patch("/projects/{project_id}")
def rename_project(project_id: str, body: RenameProjectBody):
    ok = get_chat_store().rename_project(project_id, body.name)
    if not ok:
        raise HTTPException(404, "project not found")
    return {"ok": True}


@router.delete("/projects/{project_id}")
def delete_project(project_id: str):
    get_chat_store().delete_project(project_id)
    return {"ok": True}


@router.get("/conversations")
def list_conversations(limit: int = 80, project_id: str | None = None, include_archived: bool = False):
    pid = project_id
    if isinstance(pid, str):
        pid = pid.strip() or None
    lim = max(1, min(300, int(limit or 80)))
    return get_chat_store().list_conversations(
        project_id=pid,
        limit=lim,
        include_archived=bool(include_archived),
    )


@router.post("/conversations")
def create_conversation(body: CreateConvBody):
    project_id = body.project_id
    if isinstance(project_id, str):
        project_id = project_id.strip() or None
    mode = str(body.mode or "").strip() or "normal"
    bound_source_path = str(body.bound_source_path or "").strip()
    bound_source_name = str(body.bound_source_name or "").strip()
    source_ready = bool(body.bound_source_ready)
    conv_id = get_chat_store().create_conversation(
        body.title,
        project_id=project_id,
        mode=mode,
        bound_source_path=bound_source_path,
        bound_source_name=bound_source_name,
        bound_source_ready=source_ready,
    )
    _kickoff_paper_guide_prefetch_if_needed(
        mode=mode,
        source_path=bound_source_path,
        source_name=bound_source_name,
        source_ready=source_ready,
    )
    return {"id": conv_id}


@router.get("/conversations/{conv_id}")
def get_conversation(conv_id: str):
    conv = get_chat_store().get_conversation(conv_id)
    if conv is None:
        raise HTTPException(404, "conversation not found")
    _kickoff_paper_guide_prefetch_if_needed(
        mode=str(conv.get("mode") or ""),
        source_path=str(conv.get("bound_source_path") or ""),
        source_name=str(conv.get("bound_source_name") or ""),
        source_ready=_coerce_bool_flag(conv.get("bound_source_ready")),
    )
    return conv


@router.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    get_chat_store().delete_conversation(conv_id)
    return {"ok": True}


@router.get("/conversations/{conv_id}/messages")
def get_messages(conv_id: str, limit: int | None = None):
    store = get_chat_store()
    messages = [_normalize_message_attachments(msg) for msg in store.get_messages(conv_id, limit=limit)]
    refs_by_user = store.list_message_refs(conv_id) or {}
    return enrich_messages_with_reference_render(messages, refs_by_user, conv_id=conv_id)


@router.post("/conversations/{conv_id}/messages")
def append_message(conv_id: str, body: AppendMsgBody):
    msg_id = get_chat_store().append_message(conv_id, body.role, body.content)
    return {"id": msg_id}


@router.patch("/messages/{msg_id}")
def update_message(msg_id: int, body: UpdateMsgBody):
    ok = get_chat_store().update_message_content(msg_id, body.content)
    if not ok:
        raise HTTPException(404, "message not found")
    return {"ok": True}


@router.delete("/messages/{msg_id}")
def delete_message(msg_id: int):
    ok = get_chat_store().delete_message(msg_id)
    if not ok:
        raise HTTPException(404, "message not found")
    return {"ok": True}


@router.get("/conversations/{conv_id}/refs")
def list_refs(conv_id: str):
    return get_chat_store().list_message_refs(conv_id)


@router.patch("/conversations/{conv_id}/title")
def update_title(conv_id: str, body: UpdateTitleBody):
    get_chat_store().set_title_if_default(conv_id, body.title)
    return {"ok": True}


@router.patch("/conversations/{conv_id}/project")
def update_conversation_project(conv_id: str, body: UpdateProjectBody):
    project_id = body.project_id
    if isinstance(project_id, str):
        project_id = project_id.strip() or None
    ok = get_chat_store().set_conversation_project(conv_id, project_id)
    if not ok:
        raise HTTPException(404, "conversation not found")
    return {"ok": True}


@router.patch("/conversations/{conv_id}/guide")
def update_conversation_guide(conv_id: str, body: UpdateConversationGuideBody):
    store = get_chat_store()
    ok = store.set_conversation_guide(
        conv_id,
        mode=body.mode,
        bound_source_path=body.bound_source_path,
        bound_source_name=body.bound_source_name,
        bound_source_ready=body.bound_source_ready,
    )
    if not ok:
        raise HTTPException(404, "conversation not found")
    conv = store.get_conversation(conv_id) or {}
    _kickoff_paper_guide_prefetch_if_needed(
        mode=str(conv.get("mode") or ""),
        source_path=str(conv.get("bound_source_path") or ""),
        source_name=str(conv.get("bound_source_name") or ""),
        source_ready=_coerce_bool_flag(conv.get("bound_source_ready")),
    )
    return {"ok": True}
