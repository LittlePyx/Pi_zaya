from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import File, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.chat_render import enrich_messages_with_reference_render
from api.contracts.research_agent import ResearchAgentRequest, ResearchAgentResponse
from api.deps import get_chat_store, get_settings, load_prefs
from api.internal_access import require_management_api
from api.upload_limits import (
    is_probably_pdf,
    max_chat_upload_files,
    max_image_upload_bytes,
    max_pdf_upload_bytes,
    read_upload_limited,
)
from api.routers.library import (
    _md_dir,
    _pdf_dir,
    auto_rename_saved_pdf_in_library,
    quick_ingest_pdf,
    refine_pdf_with_full_llm_replace,
    save_pdf_to_library,
)
from kb.file_ops import _path_exists
from kb.agent.runner import run_research_agent
from kb.agent.schema import validate_agent_trace
from kb.maintenance import create_auto_snapshot
from kb.paper_guide_provenance import _resolve_paper_guide_md_path
from kb.path_safety import (
    clean_file_source_path_input,
    image_ext_for_mime,
    path_is_within_roots,
    resolve_existing_file_under_roots,
    resolve_verified_image_file_under_roots,
    resolve_verified_pdf_file_under_roots,
    resolved_path,
    sniff_image_ext,
    unique_resolved_roots,
    verified_image_bytes_mime,
)
from kb.pdf_tools import ensure_dir
from kb.reader_session_store import ReaderSessionStore
from kb.task_runtime import (
    _gen_has_active_task_id,
    _is_live_assistant_text,
    _live_assistant_task_id,
    generation_interrupted_message,
    kickoff_paper_guide_prefetch,
)

# Backward-compatible import name for callers that used the router-local model.
ResearchAgentBody = ResearchAgentRequest

router = APIRouter(prefix="/api", tags=["chat"])

_CHAT_TITLE_MAX_CHARS = 240
_CHAT_PROJECT_NAME_MAX_CHARS = 120
_CHAT_MESSAGE_MAX_CHARS = 80_000
_CHAT_SOURCE_PATH_MAX_CHARS = 1_200
_CHAT_SOURCE_NAME_MAX_CHARS = 500
_CHAT_READER_TITLE_MAX_CHARS = 240
_CHAT_READER_CONVERSATION_ID_MAX_CHARS = 120
_CHAT_READER_PAYLOAD_MAX_JSON_CHARS = 120_000
_CHAT_STATE_MAX_JSON_CHARS = 160_000
_CHAT_CITATION_SHELF_MAX_ITEMS = 120
_CHAT_CITATION_SHELF_MAX_JSON_CHARS = 260_000
_CHAT_CITATION_SHELF_ITEM_MAX_JSON_CHARS = 40_000


def _bounded_json_size(value: Any, *, name: str, max_json_chars: int) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except Exception as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc
    if len(encoded) > int(max_json_chars):
        raise ValueError(f"{name} is too large; max {int(max_json_chars)} JSON chars")
    return value


def _bounded_dict(value: Any, *, name: str, max_json_chars: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return _bounded_json_size(value, name=name, max_json_chars=max_json_chars)


def _bounded_dict_list(value: Any, *, name: str, max_items: int, max_json_chars: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    if len(value) > int(max_items):
        raise ValueError(f"{name} has too many items; max {int(max_items)}")
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{name} items must be objects")
    return _bounded_json_size(value, name=name, max_json_chars=max_json_chars)


def _dangerous_auto_snapshot(action: str, *, label: str = "", metadata: dict | None = None) -> dict:
    snapshot = create_auto_snapshot(
        get_settings(),
        action=action,
        label=label,
        metadata=metadata or {},
    )
    if bool(snapshot.get("block_operation")):
        detail = str(snapshot.get("error") or snapshot.get("reason") or "automatic backup failed")
        raise HTTPException(503, f"automatic backup failed before {action}: {detail}")
    return snapshot


class CreateConvBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field("新会话", max_length=_CHAT_TITLE_MAX_CHARS)
    project_id: str | None = Field(None, max_length=120)
    mode: str = Field("normal", max_length=40)
    bound_source_path: str = Field("", max_length=_CHAT_SOURCE_PATH_MAX_CHARS)
    bound_source_name: str = Field("", max_length=_CHAT_SOURCE_NAME_MAX_CHARS)
    bound_source_ready: bool = False


class CreateProjectBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field("未命名项目", max_length=_CHAT_PROJECT_NAME_MAX_CHARS)


class AppendMsgBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = Field("user", max_length=32)
    content: str = Field(..., max_length=_CHAT_MESSAGE_MAX_CHARS)


class UpdateMsgBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str = Field(..., max_length=_CHAT_MESSAGE_MAX_CHARS)


class UpdateTitleBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(..., max_length=_CHAT_TITLE_MAX_CHARS)


class UpdateProjectBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    project_id: str | None = Field(None, max_length=120)


class UpdateConversationGuideBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str | None = Field(None, max_length=40)
    bound_source_path: str | None = Field(None, max_length=_CHAT_SOURCE_PATH_MAX_CHARS)
    bound_source_name: str | None = Field(None, max_length=_CHAT_SOURCE_NAME_MAX_CHARS)
    bound_source_ready: bool | None = None


class RenameProjectBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., max_length=_CHAT_PROJECT_NAME_MAX_CHARS)


class UploadJobBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: str = Field(..., max_length=120)


class ReaderSessionCreateBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    payload: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    title: str = Field("", max_length=_CHAT_READER_TITLE_MAX_CHARS)
    conversation_id: str = Field("", max_length=_CHAT_READER_CONVERSATION_ID_MAX_CHARS)
    message_id: int | None = None

    @field_validator("payload")
    @classmethod
    def _check_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_dict(value, name="reader session payload", max_json_chars=_CHAT_READER_PAYLOAD_MAX_JSON_CHARS)

    @field_validator("state")
    @classmethod
    def _check_state(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_dict(value, name="reader session state", max_json_chars=_CHAT_STATE_MAX_JSON_CHARS)


class ReaderSessionStatePatchBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: dict[str, Any] = Field(default_factory=dict)

    @field_validator("state")
    @classmethod
    def _check_state(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_dict(value, name="reader session state", max_json_chars=_CHAT_STATE_MAX_JSON_CHARS)


class ConversationReaderStatePatchBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: dict[str, Any] = Field(default_factory=dict)

    @field_validator("state")
    @classmethod
    def _check_state(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_dict(value, name="conversation reader state", max_json_chars=_CHAT_STATE_MAX_JSON_CHARS)


class ConversationResearchStatePatchBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: dict[str, Any] = Field(default_factory=dict)

    @field_validator("state")
    @classmethod
    def _check_state(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_dict(value, name="conversation research state", max_json_chars=_CHAT_STATE_MAX_JSON_CHARS)


class CitationShelfBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[dict[str, Any]] = Field(default_factory=list, max_length=_CHAT_CITATION_SHELF_MAX_ITEMS)
    open: bool = False
    scope: str | None = Field(None, max_length=40)
    project_id: str | None = Field(None, max_length=120)
    allow_empty_overwrite: bool = False

    @field_validator("items")
    @classmethod
    def _check_items(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _bounded_dict_list(
            value,
            name="citation shelf items",
            max_items=_CHAT_CITATION_SHELF_MAX_ITEMS,
            max_json_chars=_CHAT_CITATION_SHELF_MAX_JSON_CHARS,
        )


class CitationShelfAppendBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    item: dict[str, Any] = Field(default_factory=dict)
    open: bool = True
    scope: str | None = Field(None, max_length=40)
    project_id: str | None = Field(None, max_length=120)

    @field_validator("item")
    @classmethod
    def _check_item(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_dict(value, name="citation shelf item", max_json_chars=_CHAT_CITATION_SHELF_ITEM_MAX_JSON_CHARS)


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
_CHAT_UPLOAD_JOB_TTL_S = 6 * 60 * 60
_CHAT_UPLOAD_JOB_MAX_ITEMS = 300
_CHAT_QUALITY_REFINE_LOCK = threading.Lock()
_CHAT_QUALITY_REFINE_RUNNING: set[str] = set()


def _chat_image_dir() -> Path:
    from api.deps import get_settings

    settings = get_settings()
    out = Path(settings.db_dir) / "_chat_uploads" / "images"
    ensure_dir(out)
    return out


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _reader_markdown_roots() -> list[Path]:
    try:
        settings = get_settings()
    except Exception:
        settings = None
    db_dir = getattr(settings, "db_dir", None) if settings is not None else None
    return unique_resolved_roots([
        _md_dir(),
        _project_root() / "tmp",
        db_dir,
        (Path(db_dir).expanduser() / "db") if db_dir else None,
        (Path(db_dir).expanduser().parent / "md_output") if db_dir else None,
    ])


def _reader_session_store() -> ReaderSessionStore:
    settings = get_settings()
    return ReaderSessionStore(Path(settings.db_dir) / "_reader_sessions.json")


def _normalize_reader_session_payload(payload: dict[str, Any], *, require_allowed_source: bool = False) -> dict[str, Any]:
    rec = dict(payload or {})
    source_path = str(rec.get("sourcePath") or rec.get("source_path") or "").strip()
    if source_path:
        if require_allowed_source:
            source_path = _resolve_allowed_reader_source_path(source_path)
        rec["sourcePath"] = source_path
        rec.pop("source_path", None)
    source_name = str(rec.get("sourceName") or rec.get("source_name") or "").strip()
    if source_name:
        rec["sourceName"] = source_name
        rec.pop("source_name", None)
    return rec


def _resolve_reader_state_source_path(source_path: str) -> str:
    raw = str(source_path or "").strip()
    if not raw:
        raise HTTPException(400, "source_path required")
    return _resolve_allowed_reader_source_path(raw)


def _clean_reader_source_path_input(source_path: str) -> str:
    return clean_file_source_path_input(source_path)


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


def _safe_upload_display_name(raw_name: str, *, fallback: str = "upload") -> str:
    clean = unquote(str(raw_name or ""), errors="replace")
    clean = re.sub(r"[\x00-\x1f\x7f]+", " ", clean).strip().replace("\\", "/")
    clean = clean.rsplit("/", 1)[-1].strip()
    clean = re.sub(r"\s+", " ", clean).strip()
    if clean in {"", ".", ".."}:
        clean = str(fallback or "upload").strip() or "upload"
    return clean[:160]


def _chat_image_url(path: str) -> str:
    return f"/api/chat/uploads/image?path={quote(_chat_image_public_path(path), safe='')}"


def _chat_image_public_path(path: str) -> str:
    raw = clean_file_source_path_input(path)
    if not raw:
        return ""
    return Path(raw.replace("\\", "/")).name


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


def _trace_int(value: object) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _trace_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _agent_trace_audit_summary(trace: dict, validation: dict | None = None) -> dict:
    existing = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
    verification = trace.get("verification") if isinstance(trace.get("verification"), dict) else {}
    context = trace.get("context") if isinstance(trace.get("context"), dict) else {}
    plan = trace.get("plan") if isinstance(trace.get("plan"), list) else []
    steps = trace.get("steps") if isinstance(trace.get("steps"), list) else []
    errors = trace.get("errors") if isinstance(trace.get("errors"), list) else []
    return {
        "available": True,
        "mode": str(trace.get("mode") or ""),
        "question_type": str(existing.get("question_type") or trace.get("question_type") or "unknown"),
        "status": str(existing.get("status") or trace.get("status") or ""),
        "query_scope": str(existing.get("query_scope") or context.get("query_scope") or ""),
        "requested_query_scope": str(existing.get("requested_query_scope") or context.get("requested_query_scope") or ""),
        "total_claims": _trace_int(existing.get("total_claims") if "total_claims" in existing else verification.get("total_claims")),
        "supported_claims": _trace_int(existing.get("supported_claims") if "supported_claims" in existing else verification.get("supported_claims")),
        "unsupported_claims": _trace_int(existing.get("unsupported_claims") if "unsupported_claims" in existing else verification.get("unsupported_claims")),
        "support_ratio": _trace_float(existing.get("support_ratio")),
        "plan_step_count": _trace_int(existing.get("plan_step_count") if "plan_step_count" in existing else len(plan)),
        "tool_call_count": _trace_int(existing.get("tool_call_count") if "tool_call_count" in existing else len(steps)),
        "has_errors": bool(existing.get("has_errors")) if "has_errors" in existing else bool(errors),
        "schema_ok": bool((validation or {}).get("ok", False)),
    }


def _recover_stale_live_assistant_messages(messages: list[dict], store) -> list[dict]:
    if not messages:
        return messages
    try:
        locale = load_prefs().get("ui_locale")
    except Exception:
        locale = ""
    replacement = generation_interrupted_message(locale)
    for msg in messages:
        if str(msg.get("role") or "") != "assistant":
            continue
        content = str(msg.get("content") or "")
        if not _is_live_assistant_text(content):
            continue
        task_id = _live_assistant_task_id(content)
        if task_id and _gen_has_active_task_id(task_id):
            continue
        try:
            mid = int(msg.get("id") or 0)
        except Exception:
            mid = 0
        if mid <= 0:
            continue
        meta = dict(msg.get("meta") or {})
        meta.update({
            "generation_status": "interrupted",
            "generation_task_id": task_id,
        })
        try:
            store.update_message_content(mid, replacement)
            store.merge_message_meta(mid, meta)
        except Exception:
            pass
        msg["content"] = replacement
        msg["meta"] = meta
    return messages


def _resolve_allowed_paper_guide_source_path(source_path: str) -> str:
    raw = _clean_reader_source_path_input(source_path)
    if not raw:
        return ""
    try:
        pdf_root = _pdf_dir()
        md_root = _md_dir()
    except Exception as exc:
        raise HTTPException(500, "library directories unavailable") from exc

    suffix = Path(raw).suffix.lower()
    if suffix.endswith(".md"):
        md_path = _resolve_paper_guide_md_path(raw, md_root=md_root, db_dir=getattr(get_settings(), "db_dir", None), pdf_root=pdf_root)
        if md_path is None:
            raise HTTPException(400, "source markdown must be within the configured markdown directory")
        return str(md_path)

    if suffix == ".pdf":
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = Path(pdf_root) / candidate
        resolved = resolved_path(candidate)
        if resolved is None or resolved.suffix.lower() != ".pdf":
            raise HTTPException(400, "source path must point to a PDF")
        if not path_is_within_roots(resolved, [pdf_root]):
            raise HTTPException(400, "source PDF must be within the configured PDF directory")
        if not resolved.is_file():
            raise HTTPException(404, "source PDF not found")
        return str(resolved)

    raise HTTPException(400, "source path must point to a PDF or converted markdown")


def _resolve_allowed_reader_source_path(source_path: str) -> str:
    raw = _clean_reader_source_path_input(source_path)
    if not raw:
        raise HTTPException(400, "reader sourcePath required")
    suffix = Path(raw).suffix.lower()
    if suffix.endswith(".md"):
        try:
            pdf_root = _pdf_dir()
            md_root = _md_dir()
        except Exception as exc:
            raise HTTPException(500, "library directories unavailable") from exc
        try:
            settings = get_settings()
        except Exception:
            settings = None
        md_path = _resolve_paper_guide_md_path(raw, md_root=md_root, db_dir=getattr(settings, "db_dir", None), pdf_root=pdf_root)
        if md_path is None:
            md_path = resolve_existing_file_under_roots(raw, _reader_markdown_roots())
        if md_path is None:
            raise HTTPException(400, "source markdown must be within an allowed reader directory")
        return str(md_path)
    if suffix == ".pdf":
        return _resolve_allowed_paper_guide_source_path(raw)
    raise HTTPException(400, "source path must point to a PDF or converted markdown")


def _resolve_chat_upload_pdf_path(path_raw: str) -> Path:
    raw = str(path_raw or "").strip()
    if not raw:
        raise FileNotFoundError("pdf file not found")
    try:
        pdf_root = _pdf_dir()
    except Exception as exc:
        raise RuntimeError("library PDF directory is unavailable") from exc
    resolved = resolve_verified_pdf_file_under_roots(raw, [pdf_root])
    if resolved is not None:
        return resolved
    candidate = resolved_path(raw)
    if candidate is None:
        raise FileNotFoundError("pdf file not found")
    if not path_is_within_roots(candidate, [pdf_root]):
        raise ValueError("pdf file must be within the configured PDF directory")
    if candidate.suffix.lower() != ".pdf":
        raise ValueError("pdf file must be a PDF")
    try:
        exists = candidate.is_file()
    except Exception:
        exists = False
    if exists:
        raise ValueError("pdf file is not a valid PDF")
    raise FileNotFoundError("pdf file not found")


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
        pdf_root = _pdf_dir()
    except Exception:
        pdf_root = None
    try:
        kickoff_paper_guide_prefetch(
            source_path=src,
            source_name=str(source_name or "").strip(),
            db_dir=(getattr(settings, "db_dir", None) if settings is not None else None),
            md_root=md_root,
            pdf_root=pdf_root,
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
        _prune_chat_upload_jobs_locked()
        current = dict(_CHAT_UPLOAD_JOBS.get(job_id) or {})
        current.update(payload or {})
        current["updated_at"] = time.time()
        _CHAT_UPLOAD_JOBS[job_id] = current


def _chat_upload_job_running(record: dict) -> bool:
    ingest_status = str(record.get("ingest_status") or "")
    quality_status = str(record.get("quality_status") or "")
    return ingest_status in {"processing", "renaming", "converting", "ingesting"} or quality_status in {"pending", "running"}


def _prune_chat_upload_jobs_locked(*, now: float | None = None) -> int:
    current_time = time.time() if now is None else float(now)
    cutoff = current_time - _CHAT_UPLOAD_JOB_TTL_S
    removed = 0
    for job_id, rec in list(_CHAT_UPLOAD_JOBS.items()):
        if not isinstance(rec, dict) or _chat_upload_job_running(rec):
            continue
        try:
            ts = float(rec.get("updated_at") or rec.get("created_at") or current_time)
        except Exception:
            ts = current_time
        if ts < cutoff:
            _CHAT_UPLOAD_JOBS.pop(job_id, None)
            removed += 1
    if len(_CHAT_UPLOAD_JOBS) > _CHAT_UPLOAD_JOB_MAX_ITEMS:
        candidates: list[tuple[float, str]] = []
        for job_id, rec in _CHAT_UPLOAD_JOBS.items():
            if not isinstance(rec, dict) or _chat_upload_job_running(rec):
                continue
            try:
                ts = float(rec.get("updated_at") or rec.get("created_at") or current_time)
            except Exception:
                ts = current_time
            candidates.append((ts, job_id))
        overflow = len(_CHAT_UPLOAD_JOBS) - _CHAT_UPLOAD_JOB_MAX_ITEMS
        for _ts, job_id in sorted(candidates)[: max(0, overflow)]:
            _CHAT_UPLOAD_JOBS.pop(job_id, None)
            removed += 1
    return removed


def _get_chat_pdf_ingest_job(job_id: str) -> dict | None:
    with _CHAT_UPLOAD_JOB_LOCK:
        _prune_chat_upload_jobs_locked()
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
            payload = {
                "ready": False,
                "ingest_status": "cancelled",
                "error": "cancelled",
                "ingest_proc": None,
            }
            if quality_pending:
                payload.update({
                    "quality_status": "cancelled",
                    "quality_stage": "cancelled",
                    "quality_error": "cancelled",
                })
            _set_chat_pdf_ingest_job(job_id, payload)
            return
        ready = bool(result.get("ready"))
        result_error = str(result.get("error") or "")
        quality_update: dict[str, str] = {}
        if quality_pending and not ready:
            quality_update = {
                "quality_status": "error",
                "quality_stage": "error",
                "quality_error": result_error or "ingest failed before quality refine",
            }
        elif quality_pending and ready and not bool(result.get("out_folder")):
            quality_update = {
                "quality_status": "error",
                "quality_stage": "error",
                "quality_error": "quality refine was not started after ingest",
            }
        payload = {
            "ready": ready,
            "ingest_status": "ready" if ready else "error",
            "error": result_error,
            "md_path": str(result.get("md_path") or ""),
            "ingest_proc": None,
        }
        payload.update(quality_update)
        _set_chat_pdf_ingest_job(
            job_id,
            payload,
        )
        if ready:
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
    pdf_path = _resolve_chat_upload_pdf_path(str(rec.get("path") or ""))
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
    _resolve_chat_upload_pdf_path(str(rec.get("path") or ""))
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


def _sniff_image_ext(data: bytes) -> str:
    return sniff_image_ext(data)


def _claimed_image_ext(raw_name: str, raw_mime: str) -> str:
    suffix = Path(_safe_upload_display_name(raw_name)).suffix.lower()
    if suffix in IMAGE_EXTS:
        return suffix
    return IMAGE_MIME_TO_EXT.get(str(raw_mime or "").strip().lower(), "")


def _save_chat_image(*, raw_name: str, data: bytes, sha1: str) -> dict:
    img_dir = _chat_image_dir()
    mime = verified_image_bytes_mime(data)
    if not mime:
        raise ValueError("invalid image file")
    ext = _sniff_image_ext(data) or image_ext_for_mime(mime)
    if not ext:
        raise ValueError("invalid image file")
    display_name = _safe_upload_display_name(raw_name, fallback=f"pasted-{int(time.time())}{ext}")
    stem_seed = Path(display_name).stem or f"pasted-{int(time.time())}"
    safe_stem = _safe_upload_stem(stem_seed)
    dest_img = img_dir / f"{safe_stem}-{sha1[:10]}{ext}"
    duplicate = False
    if dest_img.exists():
        existing_mime = resolve_verified_image_file_under_roots(dest_img, [img_dir])
        duplicate = bool(existing_mime and existing_mime[1] == mime)
    if not duplicate:
        tmp = dest_img.with_name(f".{dest_img.name}.{uuid.uuid4().hex[:10]}.tmp")
        try:
            tmp.write_bytes(data)
            tmp.replace(dest_img)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
    return {
        "kind": "image",
        "status": "duplicate" if duplicate else "saved",
        "name": display_name or dest_img.name,
        "sha1": sha1,
        "mime": mime,
        "path": dest_img.name,
        "attachment": {
            "sha1": sha1,
            "path": dest_img.name,
            "name": display_name or dest_img.name,
            "mime": mime,
            "url": _chat_image_url(dest_img.name),
        },
    }


@router.post("/chat/research-agent", response_model=ResearchAgentResponse)
def run_chat_research_agent(body: ResearchAgentRequest) -> ResearchAgentResponse:
    query = str(body.query or body.prompt or "").strip()
    if not query:
        raise HTTPException(400, "query required")
    settings = get_settings()
    source_lock_path = str(body.source_lock_path or "").strip()
    if source_lock_path:
        source_lock_path = _resolve_allowed_paper_guide_source_path(source_lock_path)
    source_lock_name = str(body.source_lock_name or "").strip()
    payload = run_research_agent(
        query,
        db_dir=settings.db_dir,
        settings=settings,
        top_k=body.top_k,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        query_scope=body.query_scope,
        selected_research_context=body.prompt_context or {},
        current_source_path=source_lock_path,
        current_source_name=source_lock_name,
    )
    return ResearchAgentResponse.model_validate(payload)


@router.get("/chat/uploads/image")
def get_chat_upload_image(path: str):
    verified = resolve_verified_image_file_under_roots(path, [_chat_image_dir()])
    if verified is None:
        raise HTTPException(404, "image not found")
    resolved, media_type = verified
    return FileResponse(str(resolved), media_type=media_type, filename=resolved.name)


@router.get("/chat/uploads/status")
def get_chat_upload_status(job_ids: str = ""):
    wanted = [str(x or "").strip() for x in str(job_ids or "").split(",") if str(x or "").strip()]
    if not wanted:
        return {"items": []}
    items: list[dict] = []
    with _CHAT_UPLOAD_JOB_LOCK:
        _prune_chat_upload_jobs_locked()
        for job_id in wanted:
            rec = _CHAT_UPLOAD_JOBS.get(job_id)
            if not isinstance(rec, dict):
                continue
            items.append(_chat_pdf_ingest_status_payload(job_id, rec))
    return {"items": items}


@router.post("/chat/uploads/cancel", dependencies=[Depends(require_management_api)])
def cancel_chat_upload_job(body: UploadJobBody):
    rec = _cancel_chat_pdf_ingest_job(str(body.job_id or "").strip())
    if rec is None:
        raise HTTPException(404, "upload job not found")
    return {"item": _chat_pdf_ingest_status_payload(str(body.job_id or "").strip(), rec)}


@router.post("/chat/uploads/retry", dependencies=[Depends(require_management_api)])
def retry_chat_upload_job(body: UploadJobBody):
    job_id = str(body.job_id or "").strip()
    if not job_id:
        raise HTTPException(400, "job_id required")
    try:
        rec = _retry_chat_pdf_ingest_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    if rec is None:
        raise HTTPException(404, "upload job not found")
    new_job_id = str(rec.get("job_id") or "")
    return {"item": _chat_pdf_ingest_status_payload(new_job_id, rec)}


@router.post("/chat/uploads/quality/retry", dependencies=[Depends(require_management_api)])
def retry_chat_upload_quality_job(body: UploadJobBody):
    job_id = str(body.job_id or "").strip()
    if not job_id:
        raise HTTPException(400, "job_id required")
    try:
        rec = _retry_chat_pdf_quality_refine_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    if rec is None:
        raise HTTPException(404, "upload job not found")
    return {"item": _chat_pdf_ingest_status_payload(job_id, rec)}


@router.post("/chat/uploads", dependencies=[Depends(require_management_api)])
async def upload_chat_files(
    files: list[UploadFile] = File(...),
    quick_ingest: bool = Form(True),
    speed_mode: str = Form("balanced"),
    conv_id: str = Form(""),
):
    results: list[dict] = []
    seen_sha1: set[str] = set()
    conv_id_norm = str(conv_id or "").strip()
    settings = get_settings()
    uploads = list(files or [])
    if len(uploads) > max_chat_upload_files(settings):
        raise HTTPException(413, "too many uploaded files")
    pdf_limit = max_pdf_upload_bytes(settings)
    image_limit = max_image_upload_bytes(settings)

    for up in uploads:
        raw_name = str(getattr(up, "filename", "") or "").strip()
        raw_mime = str(getattr(up, "content_type", "") or "").strip().lower()
        suffix = Path(raw_name).suffix.lower()
        claimed_pdf = bool((suffix == ".pdf") or (raw_mime == "application/pdf"))
        claimed_image = bool(_claimed_image_ext(raw_name, raw_mime))
        read_limit = pdf_limit if claimed_pdf else (image_limit if claimed_image else max(pdf_limit, image_limit))
        try:
            data = await read_upload_limited(
                up,
                max_bytes=read_limit,
                label=raw_name or "chat upload",
            )
        except HTTPException as exc:
            if exc.status_code == 413:
                raise
            results.append({
                "kind": "unknown",
                "status": "error",
                "name": raw_name or "upload",
                "error": str(exc.detail or "failed to read upload"),
            })
            continue
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

        is_pdf = bool(claimed_pdf or is_probably_pdf(data))
        detected_image_ext = _sniff_image_ext(data)
        is_image = bool(claimed_image or detected_image_ext)

        if is_image and (not is_pdf):
            if (not claimed_image) and len(data) > image_limit:
                raise HTTPException(413, f"{raw_name or 'image'} exceeds the {image_limit} byte upload limit")
            if not detected_image_ext:
                results.append({
                    "kind": "image",
                    "status": "error",
                    "name": raw_name or "image",
                    "sha1": sha1,
                    "error": "invalid image file",
                })
                continue
            try:
                results.append(_save_chat_image(raw_name=raw_name, data=data, sha1=sha1))
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
            if not is_probably_pdf(data):
                results.append({
                    "kind": "pdf",
                    "status": "error",
                    "name": raw_name or "upload.pdf",
                    "sha1": sha1,
                    "error": "invalid PDF file",
                })
                continue
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


@router.post("/reader/sessions")
def create_reader_session(body: ReaderSessionCreateBody):
    payload = _normalize_reader_session_payload(body.payload, require_allowed_source=True)
    source_path = str(payload.get("sourcePath") or "").strip()
    if not source_path:
        raise HTTPException(400, "reader sourcePath required")
    title = str(body.title or payload.get("sourceName") or Path(source_path).name).strip()
    return _reader_session_store().create(
        payload,
        state=body.state,
        title=title,
        conversation_id=body.conversation_id,
        message_id=body.message_id,
    )


@router.get("/reader/sessions/{session_id}")
def get_reader_session(session_id: str):
    record = _reader_session_store().get(session_id)
    if record is None:
        raise HTTPException(404, "reader session not found")
    return record


@router.patch("/reader/sessions/{session_id}/state")
def update_reader_session_state(session_id: str, body: ReaderSessionStatePatchBody):
    record = _reader_session_store().update_state(session_id, body.state)
    if record is None:
        raise HTTPException(404, "reader session not found")
    return record


@router.get("/conversations/{conv_id}/reader-state")
def get_conversation_reader_state(conv_id: str, source_path: str = Query("")):
    raw_src = str(source_path or "").strip()
    src = _resolve_reader_state_source_path(source_path)
    store = get_chat_store()
    if store.get_conversation(conv_id) is None:
        raise HTTPException(404, "conversation not found")
    record = store.get_conversation_reader_state(conv_id, src)
    if record and (record.get("state") or raw_src == src):
        return record
    if raw_src and raw_src != src:
        legacy = store.get_conversation_reader_state(conv_id, raw_src)
        legacy_state = legacy.get("state") if isinstance(legacy, dict) else {}
        if isinstance(legacy_state, dict) and legacy_state:
            migrated = store.patch_conversation_reader_state(conv_id, src, legacy_state)
            return migrated or {**legacy, "source_path": src}
    return record


@router.patch("/conversations/{conv_id}/reader-state")
def patch_conversation_reader_state(
    conv_id: str,
    body: ConversationReaderStatePatchBody,
    source_path: str = Query(""),
):
    src = _resolve_reader_state_source_path(source_path)
    record = get_chat_store().patch_conversation_reader_state(conv_id, src, body.state)
    if record is None:
        raise HTTPException(404, "conversation not found")
    return record


@router.get("/conversations/{conv_id}/research-state")
def get_conversation_research_state(conv_id: str):
    record = get_chat_store().get_conversation_research_state(conv_id)
    if record is None:
        raise HTTPException(404, "conversation not found")
    return record


@router.patch("/conversations/{conv_id}/research-state")
def patch_conversation_research_state(conv_id: str, body: ConversationResearchStatePatchBody):
    record = get_chat_store().patch_conversation_research_state(conv_id, body.state)
    if record is None:
        raise HTTPException(404, "conversation not found")
    return record


@router.get("/projects")
def list_projects():
    return get_chat_store().list_projects()


@router.get("/sidebar")
def get_sidebar(limit: int = 80, include_archived: bool = False):
    lim = max(1, min(300, int(limit or 80)))
    return get_chat_store().sidebar_snapshot(limit=lim, include_archived=bool(include_archived))


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
    store = get_chat_store()
    if store.get_project(project_id) is None:
        raise HTTPException(404, "project not found")
    auto_backup = _dangerous_auto_snapshot(
        "chat_project_delete",
        label=project_id,
        metadata={"project_id": project_id},
    )
    ok = store.delete_project(project_id)
    if not ok:
        raise HTTPException(404, "project not found")
    return {"ok": True, "auto_backup": auto_backup}


@router.get("/chat/citation-shelf")
def get_citation_shelf(
    conv_id: str | None = Query(None),
    project_id: str | None = Query(None),
    scope: str = Query("project"),
):
    record = get_chat_store().get_citation_shelf(
        conv_id=conv_id,
        project_id=project_id,
        scope=scope,
    )
    if record is None:
        raise HTTPException(404, "conversation not found")
    return record


@router.patch("/chat/citation-shelf")
def save_citation_shelf(
    body: CitationShelfBody,
    conv_id: str | None = Query(None),
    project_id: str | None = Query(None),
    scope: str = Query("project"),
):
    resolved_scope = str(body.scope or scope or "project").strip() or "project"
    resolved_project_id = body.project_id if body.project_id is not None else project_id
    record = get_chat_store().save_citation_shelf(
        conv_id=conv_id,
        project_id=resolved_project_id,
        scope=resolved_scope,
        items=body.items,
        open=body.open,
        allow_empty_overwrite=body.allow_empty_overwrite,
    )
    if record is None:
        raise HTTPException(404, "conversation not found")
    return record


@router.post("/chat/citation-shelf/items")
def append_citation_shelf_item(
    body: CitationShelfAppendBody,
    conv_id: str | None = Query(None),
    project_id: str | None = Query(None),
    scope: str = Query("project"),
):
    resolved_scope = str(body.scope or scope or "project").strip() or "project"
    resolved_project_id = body.project_id if body.project_id is not None else project_id
    record = get_chat_store().append_citation_shelf_item(
        conv_id=conv_id,
        project_id=resolved_project_id,
        scope=resolved_scope,
        item=body.item,
        open=body.open,
    )
    if record is None:
        raise HTTPException(404, "conversation not found")
    return record


@router.delete("/chat/citation-shelf")
def delete_citation_shelf(
    conv_id: str | None = Query(None),
    project_id: str | None = Query(None),
    scope: str = Query("project"),
):
    store = get_chat_store()
    if conv_id and store.get_conversation(conv_id) is None:
        raise HTTPException(404, "conversation not found")
    auto_backup = _dangerous_auto_snapshot(
        "chat_citation_shelf_delete",
        label=str(scope or "project"),
        metadata={"conv_id": conv_id or "", "project_id": project_id or "", "scope": scope},
    )
    record = store.delete_citation_shelf(
        conv_id=conv_id,
        project_id=project_id,
        scope=scope,
    )
    if record is None:
        raise HTTPException(404, "conversation not found")
    record["auto_backup"] = auto_backup
    return record


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
    if bound_source_path:
        bound_source_path = _resolve_allowed_paper_guide_source_path(bound_source_path)
    store = get_chat_store()
    if project_id and store.get_project(project_id) is None:
        raise HTTPException(404, "project not found")
    conv_id = store.create_conversation(
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


def _merge_cached_reference_render_payload(conv_id: str, refs_by_user: dict) -> dict:
    merged: dict = dict(refs_by_user or {}) if isinstance(refs_by_user, dict) else {}
    try:
        from api.routers.references import _get_compatible_cached_conversation_refs_payload

        cached = _get_compatible_cached_conversation_refs_payload(
            conv_id=conv_id,
            refs=merged,
        )
    except Exception:
        cached = None
    if not isinstance(cached, dict):
        return merged
    for raw_key, pack in cached.items():
        if not isinstance(pack, dict):
            continue
        try:
            key = int(raw_key)
        except Exception:
            continue
        if key <= 0:
            continue
        current = merged.get(key) or merged.get(str(key))
        if isinstance(current, dict):
            next_pack = dict(current)
            next_pack["rendered_payload"] = pack
            merged[key] = next_pack
        else:
            merged[key] = pack
    return merged


@router.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    store = get_chat_store()
    if store.get_conversation(conv_id) is None:
        raise HTTPException(404, "conversation not found")
    auto_backup = _dangerous_auto_snapshot(
        "chat_conversation_delete",
        label=conv_id,
        metadata={"conv_id": conv_id},
    )
    ok = store.delete_conversation(conv_id)
    if not ok:
        raise HTTPException(404, "conversation not found")
    return {"ok": True, "auto_backup": auto_backup}


@router.get("/conversations/{conv_id}/messages")
def get_messages(conv_id: str, limit: int | None = None, render_packet_only: int | None = None):
    store = get_chat_store()
    messages = [_normalize_message_attachments(msg) for msg in store.get_messages(conv_id, limit=limit)]
    messages = _recover_stale_live_assistant_messages(messages, store)
    refs_by_user = _merge_cached_reference_render_payload(conv_id, store.list_message_refs(conv_id) or {})
    conv = store.get_conversation(conv_id) or {}
    mode = str(conv.get("mode") or "").strip().lower()
    # Default: enable in paper-guide conversations so frontend can exercise contract-first mode.
    # If query param is provided, it is treated as an explicit override (0 disables).
    render_packet_flag = (
        bool(int(render_packet_only or 0))
        if render_packet_only is not None
        else (mode == "paper_guide")
    )
    return enrich_messages_with_reference_render(
        messages,
        refs_by_user,
        conv_id=conv_id,
        chat_store=store,
        render_packet_only=render_packet_flag,
    )


@router.get("/conversations/{conv_id}/messages_page")
def get_messages_page(conv_id: str, limit: int = 24, before_id: int | None = None, render_packet_only: int | None = None):
    store = get_chat_store()
    messages, has_more_before, oldest_loaded_id, newest_loaded_id = store.get_messages_page(
        conv_id,
        limit=limit,
        before_id=before_id,
    )
    refs_by_user = _merge_cached_reference_render_payload(conv_id, store.list_message_refs(conv_id) or {})
    conv = store.get_conversation(conv_id) or {}
    mode = str(conv.get("mode") or "").strip().lower()
    render_packet_flag = (
        bool(int(render_packet_only or 0))
        if render_packet_only is not None
        else (mode == "paper_guide")
    )
    rendered = enrich_messages_with_reference_render(
        _recover_stale_live_assistant_messages([_normalize_message_attachments(msg) for msg in messages], store),
        refs_by_user,
        conv_id=conv_id,
        chat_store=store,
        render_packet_only=render_packet_flag,
    )
    return {
        "messages": rendered,
        "has_more_before": bool(has_more_before),
        "oldest_loaded_id": oldest_loaded_id,
        "newest_loaded_id": newest_loaded_id,
    }


@router.get("/messages/{msg_id}/agent-trace")
def get_message_agent_trace(msg_id: int, conv_id: str = Query("")):
    store = get_chat_store()
    msg = store.get_message(msg_id)
    if msg is None:
        raise HTTPException(404, "message not found")
    conv_id_expected = str(conv_id or "").strip()
    conv_id_actual = str(msg.get("conv_id") or "").strip()
    if conv_id_expected and conv_id_expected != conv_id_actual:
        raise HTTPException(404, "message not found")

    meta = msg.get("meta") if isinstance(msg.get("meta"), dict) else {}
    trace = meta.get("agent_trace") if isinstance(meta.get("agent_trace"), dict) else {}
    if not trace:
        return {
            "message_id": int(msg_id),
            "conv_id": conv_id_actual,
            "available": False,
            "agent_trace": {},
            "summary": {"available": False},
        }
    validation = validate_agent_trace(trace)
    return {
        "message_id": int(msg_id),
        "conv_id": conv_id_actual,
        "available": True,
        "agent_trace": trace,
        "summary": _agent_trace_audit_summary(trace, validation),
        "schema_errors": list(validation.get("errors") or [])[:8],
    }


@router.post("/conversations/{conv_id}/messages")
def append_message(conv_id: str, body: AppendMsgBody):
    store = get_chat_store()
    if store.get_conversation(conv_id) is None:
        raise HTTPException(404, "conversation not found")
    msg_id = store.append_message(conv_id, body.role, body.content)
    return {"id": msg_id}


@router.patch("/messages/{msg_id}")
def update_message(msg_id: int, body: UpdateMsgBody):
    ok = get_chat_store().update_message_content(msg_id, body.content, touch_conversation=True)
    if not ok:
        raise HTTPException(404, "message not found")
    return {"ok": True}


@router.delete("/messages/{msg_id}")
def delete_message(msg_id: int):
    store = get_chat_store()
    if not store.message_exists(msg_id):
        raise HTTPException(404, "message not found")
    auto_backup = _dangerous_auto_snapshot(
        "chat_message_delete",
        label=str(msg_id),
        metadata={"message_id": int(msg_id)},
    )
    ok = store.delete_message(msg_id)
    if not ok:
        raise HTTPException(404, "message not found")
    return {"ok": True, "auto_backup": auto_backup}


@router.get("/conversations/{conv_id}/refs")
def list_refs(conv_id: str):
    return get_chat_store().list_message_refs(conv_id)


@router.patch("/conversations/{conv_id}/title")
def update_title(conv_id: str, body: UpdateTitleBody):
    title = str(body.title or "").replace("\n", " ").strip()
    if not title:
        raise HTTPException(400, "title is required")
    ok = get_chat_store().set_title(conv_id, body.title)
    if not ok:
        raise HTTPException(404, "conversation not found")
    return {"ok": True}


@router.patch("/conversations/{conv_id}/project")
def update_conversation_project(conv_id: str, body: UpdateProjectBody):
    project_id = body.project_id
    if isinstance(project_id, str):
        project_id = project_id.strip() or None
    store = get_chat_store()
    if store.get_conversation(conv_id) is None:
        raise HTTPException(404, "conversation not found")
    if project_id and store.get_project(project_id) is None:
        raise HTTPException(404, "project not found")
    ok = store.set_conversation_project(conv_id, project_id)
    if not ok:
        raise HTTPException(404, "conversation not found")
    return {"ok": True}


@router.patch("/conversations/{conv_id}/guide")
def update_conversation_guide(conv_id: str, body: UpdateConversationGuideBody):
    store = get_chat_store()
    bound_source_path = body.bound_source_path
    if isinstance(bound_source_path, str) and bound_source_path.strip():
        bound_source_path = _resolve_allowed_paper_guide_source_path(bound_source_path)
    ok = store.set_conversation_guide(
        conv_id,
        mode=body.mode,
        bound_source_path=bound_source_path,
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
