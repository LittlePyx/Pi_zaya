from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException, UploadFile

from kb.path_safety import is_probably_pdf_bytes


DEFAULT_MAX_PDF_UPLOAD_BYTES = 80 * 1024 * 1024
DEFAULT_MAX_IMAGE_UPLOAD_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_CHAT_UPLOAD_FILES = 12
_READ_CHUNK_BYTES = 1024 * 1024


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return max(1, int(raw))
    except Exception:
        return int(default)


def _env_megabytes(name: str, default_bytes: int) -> int:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return int(default_bytes)
    try:
        return max(1, int(float(raw) * 1024 * 1024))
    except Exception:
        return int(default_bytes)


def max_pdf_upload_bytes(settings: object | None = None) -> int:
    value = getattr(settings, "max_pdf_upload_bytes", None) if settings is not None else None
    if value:
        try:
            return max(1, int(value))
        except Exception:
            pass
    if os.environ.get("KB_MAX_PDF_UPLOAD_BYTES"):
        return _env_int("KB_MAX_PDF_UPLOAD_BYTES", DEFAULT_MAX_PDF_UPLOAD_BYTES)
    return _env_megabytes("KB_MAX_PDF_UPLOAD_MB", DEFAULT_MAX_PDF_UPLOAD_BYTES)


def max_image_upload_bytes(settings: object | None = None) -> int:
    value = getattr(settings, "max_image_upload_bytes", None) if settings is not None else None
    if value:
        try:
            return max(1, int(value))
        except Exception:
            pass
    if os.environ.get("KB_MAX_IMAGE_UPLOAD_BYTES"):
        return _env_int("KB_MAX_IMAGE_UPLOAD_BYTES", DEFAULT_MAX_IMAGE_UPLOAD_BYTES)
    return _env_megabytes("KB_MAX_IMAGE_UPLOAD_MB", DEFAULT_MAX_IMAGE_UPLOAD_BYTES)


def max_chat_upload_files(settings: object | None = None) -> int:
    value = getattr(settings, "max_chat_upload_files", None) if settings is not None else None
    if value:
        try:
            return max(1, int(value))
        except Exception:
            pass
    return _env_int("KB_MAX_CHAT_UPLOAD_FILES", DEFAULT_MAX_CHAT_UPLOAD_FILES)


def is_probably_pdf(data: bytes) -> bool:
    return is_probably_pdf_bytes(data)


def ensure_pdf_upload(data: bytes, *, file_name: str = "", content_type: str = "") -> None:
    suffix = Path(str(file_name or "upload.pdf")).suffix.lower()
    mime = str(content_type or "").strip().lower()
    claimed_pdf = suffix == ".pdf" or mime == "application/pdf"
    if not claimed_pdf:
        raise HTTPException(400, "upload must be a PDF")
    if not is_probably_pdf(data):
        raise HTTPException(400, "invalid PDF file")


async def read_upload_limited(
    upload: UploadFile,
    *,
    max_bytes: int,
    label: str = "upload",
    require_non_empty: bool = True,
) -> bytes:
    limit = max(1, int(max_bytes or 1))
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            chunk = await upload.read(_READ_CHUNK_BYTES)
        except Exception as exc:
            raise HTTPException(400, f"failed to read {label}") from exc
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, f"{label} exceeds the {limit} byte upload limit")
        chunks.append(bytes(chunk))
    data = b"".join(chunks)
    if require_non_empty and not data:
        raise HTTPException(400, "empty file")
    return data
