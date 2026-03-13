from __future__ import annotations

import mimetypes
import os
import re
from pathlib import Path
from urllib.parse import quote
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.deps import get_chat_store, get_settings, load_prefs
from api.reference_ui import enrich_citation_detail_meta, enrich_refs_payload, ensure_source_citation_meta, open_reference_source
from kb.file_ops import _resolve_md_output_paths
from kb.library_store import LibraryStore
from kb.source_blocks import load_source_blocks, source_blocks_to_reader_anchors
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
    conversation = get_chat_store().get_conversation(conv_id) or {}
    guide_mode = str(conversation.get("mode") or "").strip().lower() == "paper_guide"
    guide_source_path = str(conversation.get("bound_source_path") or "").strip()
    guide_source_name = str(conversation.get("bound_source_name") or "").strip()
    refs = get_chat_store().list_message_refs(conv_id)
    return enrich_refs_payload(
        refs,
        pdf_root=_pdf_dir(),
        md_root=_md_dir(),
        lib_store=_lib_store(),
        guide_mode=guide_mode,
        guide_source_path=guide_source_path,
        guide_source_name=guide_source_name,
    )


class OpenReferenceBody(BaseModel):
    source_path: str
    page: int | None = None


class CitationMetaBody(BaseModel):
    source_path: str


class BibliometricsBody(BaseModel):
    meta: dict


class ReaderDocBody(BaseModel):
    source_path: str


_ASSET_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_MD_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_MD_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_MD_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
_MD_FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*")
_EQ_NUMBER_RE = re.compile(r"(?:\b(?:eq|equation|公式)\s*[#(（]?\s*|[\(（])(\d{1,4})(?:\s*[)）])", re.IGNORECASE)
_INLINE_EQ_RE = re.compile(r"\$[^$]{1,280}\$")
_TEX_CMD_RE = re.compile(r"\\[a-zA-Z]{2,}")


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


def _resolve_reader_md_path(source_path: str) -> Path | None:
    raw = str(source_path or "").strip()
    if not raw:
        return None
    src = Path(raw).expanduser()
    if src.suffix.lower().endswith(".md"):
        try:
            if src.exists() and src.is_file():
                return src.resolve(strict=False)
        except Exception:
            return None
        return None

    pdf_root = _pdf_dir()
    md_root = _md_dir()

    pdf_candidate = src
    try:
        if (not pdf_candidate.is_absolute()) and (Path(pdf_candidate).name == str(pdf_candidate)):
            pdf_candidate = pdf_root / pdf_candidate
    except Exception:
        pass

    try:
        if not (pdf_candidate.exists() and pdf_candidate.is_file()):
            return None
    except Exception:
        return None

    try:
        _md_folder, md_main, md_exists = _resolve_md_output_paths(md_root, pdf_candidate)
    except Exception:
        return None
    if (not md_exists) or (not md_main.exists()) or (not md_main.is_file()):
        return None
    try:
        return md_main.resolve(strict=False)
    except Exception:
        return md_main


def _rewrite_md_asset_links(md_text: str, *, md_path: Path, md_root: Path) -> str:
    text = str(md_text or "")
    if not text:
        return text

    def _replace(m: re.Match[str]) -> str:
        alt = str(m.group(1) or "")
        raw = str(m.group(2) or "").strip()
        if not raw:
            return m.group(0)
        url = raw.strip().strip("<>").split()[0].strip()
        low = url.lower()
        if low.startswith(("http://", "https://", "data:", "#", "/api/")):
            return m.group(0)
        try:
            cand = Path(url).expanduser()
            if not cand.is_absolute():
                cand = (md_path.parent / cand).resolve(strict=False)
            else:
                cand = cand.resolve(strict=False)
            if (not cand.exists()) or (not cand.is_file()):
                return m.group(0)
            try:
                cand.relative_to(md_root.resolve())
            except Exception:
                return m.group(0)
            asset_url = f"/api/references/asset?path={quote(str(cand), safe='')}"
            return f"![{alt}]({asset_url})"
        except Exception:
            return m.group(0)

    return _MD_IMAGE_RE.sub(_replace, text)


def _strip_md_inline_for_anchor(input_text: str) -> str:
    text = str(input_text or "")
    if not text:
        return ""
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _has_equation_signal(text: str) -> bool:
    src = str(text or "")
    if not src:
        return False
    if "$$" in src:
        return True
    low = src.lower()
    if "\\begin{equation" in low or "\\[" in src:
        return True
    if _INLINE_EQ_RE.search(src):
        return True
    if _TEX_CMD_RE.search(src) and re.search(r"[=^_]", src):
        return True
    return False


def _extract_equation_number(text: str) -> int:
    src = str(text or "")
    if not src:
        return 0
    m = _EQ_NUMBER_RE.search(src)
    if not m:
        return 0
    try:
        v = int(str(m.group(1) or "0"))
    except Exception:
        return 0
    return v if v > 0 else 0


def _anchor_id(kind: str, index: int) -> str:
    prefix_map = {
        "heading": "hd",
        "paragraph": "p",
        "equation": "eq",
        "list_item": "li",
        "blockquote": "bq",
        "code": "cd",
        "table": "tb",
    }
    prefix = prefix_map.get(str(kind or "").strip().lower(), "a")
    return f"{prefix}_{int(max(1, index)):05d}"


def _build_reader_anchors(md_text: str, *, md_path: Path) -> tuple[list[dict], list[dict]]:
    blocks = load_source_blocks(md_path, md_text=md_text)
    anchors = source_blocks_to_reader_anchors(blocks)
    return anchors, blocks


@router.post("/reader/doc")
def get_reader_doc(body: ReaderDocBody):
    source_path = str(body.source_path or "").strip()
    if not source_path:
        raise HTTPException(400, "source_path required")
    md_path = _resolve_reader_md_path(source_path)
    if md_path is None:
        raise HTTPException(404, "markdown not found for source")
    try:
        md_text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        raise HTTPException(500, "failed to read markdown")

    md_root = _md_dir()
    md_render = _rewrite_md_asset_links(md_text, md_path=md_path, md_root=md_root)
    anchors, blocks = _build_reader_anchors(md_text, md_path=md_path)
    source_name = md_path.name
    low = source_name.lower()
    if low.endswith(".en.md"):
        source_name = source_name[:-6] + ".pdf"
    elif low.endswith(".md"):
        source_name = source_name[:-3] + ".pdf"

    return {
        "ok": True,
        "source_path": source_path,
        "source_name": source_name,
        "md_path": str(md_path),
        "markdown": md_render,
        "anchors": anchors,
        "blocks": blocks,
    }


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
