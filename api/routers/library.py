from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from api.deps import get_settings, load_prefs
from api.sse import sse_generator, sse_response
from kb.file_naming import (
    build_display_pdf_filename,
    build_storage_base_name,
    merge_citation_meta_file_labels,
    merge_citation_meta_name_fields,
)
from kb.task_runtime import (
    _bg_enqueue,
    _bg_snapshot,
    _bg_cancel_all,
    _build_bg_task,
    _bg_ensure_started,
    _bg_remove_queued_tasks_for_pdf,
)
from kb.file_ops import (
    _list_pdf_paths_fast,
    _next_pdf_dest_path,
    _persist_upload_pdf,
    _resolve_md_output_paths,
    _sha1_bytes,
    _cleanup_tmp_uploads,
    _cleanup_tmp_md_artifacts,
    _write_tmp_upload,
    _path_exists,
    _path_is_dir,
    _path_is_file,
    _to_os_path,
)
from kb.library_store import LibraryStore
from kb.pdf_tools import PdfMetaSuggestion, extract_pdf_meta_suggestion, run_pdf_to_md, open_in_explorer
from kb.reference_sync import start_reference_sync

router = APIRouter(prefix="/api/library", tags=["library"])
_RENAME_SUGGEST_CACHE: dict[str, dict] = {}


def _pdf_dir() -> Path:
    prefs = load_prefs()
    s = get_settings()
    return Path(prefs.get("pdf_dir") or os.environ.get("KB_PDF_DIR") or str(Path(s.db_dir).parent / "pdfs")).expanduser().resolve()


def _md_dir() -> Path:
    prefs = load_prefs()
    s = get_settings()
    return Path(prefs.get("md_dir") or os.environ.get("KB_MD_DIR") or str(Path(s.db_dir).parent / "md_output")).expanduser().resolve()


def _library_store() -> LibraryStore:
    return LibraryStore(get_settings().library_db_path)


def _ingest_py_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "ingest.py"


def _safe_delete_file(path_obj: Path) -> tuple[bool, str]:
    p = Path(path_obj)
    try:
        if not _path_exists(p):
            return True, "not found"
    except Exception:
        pass
    err = ""
    try:
        os.remove(_to_os_path(p))
    except Exception as exc:
        err = str(exc)
        try:
            p.unlink()
            err = ""
        except Exception as exc2:
            err = str(exc2) or err
    try:
        if _path_exists(p):
            return False, err or "still exists after delete"
    except Exception:
        pass
    return True, ""


def _safe_delete_tree(path_obj: Path) -> tuple[bool, str]:
    p = Path(path_obj)
    try:
        if not _path_exists(p):
            return True, "not found"
        if not _path_is_dir(p):
            return False, "target is not a directory"
    except Exception:
        pass
    err = ""
    try:
        shutil.rmtree(_to_os_path(p), ignore_errors=False)
    except Exception as exc:
        err = str(exc)
        try:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass
    try:
        if _path_exists(p):
            return False, err or "directory still exists after delete"
    except Exception:
        pass
    return True, ""


def _normalized_path_key(raw: str | Path) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    try:
        return str(Path(s).expanduser().resolve())
    except Exception:
        return s


def _build_task_maps_from_snapshot(snap: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    by_path: dict[str, dict] = {}
    by_name: dict[str, dict] = {}
    queue = list((snap or {}).get("queue") or [])
    for idx, task in enumerate(queue, start=1):
        if not isinstance(task, dict):
            continue
        task_pdf = str(task.get("pdf") or "").strip()
        if not task_pdf:
            continue
        task_name = str(task.get("name") or Path(task_pdf).name).strip()
        info = {
            "queued": True,
            "running": False,
            "replace": bool(task.get("replace", False)),
            "queue_pos": int(idx),
            "task_id": str(task.get("_tid") or ""),
        }
        key = _normalized_path_key(task_pdf)
        if key:
            prev = by_path.get(key)
            if not isinstance(prev, dict):
                by_path[key] = dict(info)
            else:
                prev["queued"] = True
                prev["replace"] = bool(prev.get("replace")) or bool(info.get("replace"))
                prev["queue_pos"] = min(int(prev.get("queue_pos") or idx), int(info.get("queue_pos") or idx))
                by_path[key] = prev
        if task_name:
            prev_n = by_name.get(task_name)
            if not isinstance(prev_n, dict):
                by_name[task_name] = dict(info)
            else:
                prev_n["queued"] = True
                prev_n["replace"] = bool(prev_n.get("replace")) or bool(info.get("replace"))
                prev_n["queue_pos"] = min(int(prev_n.get("queue_pos") or idx), int(info.get("queue_pos") or idx))
                by_name[task_name] = prev_n

    current_name = str((snap or {}).get("current") or "").strip()
    if bool((snap or {}).get("running")) and current_name:
        current_replace = bool((snap or {}).get("cur_task_replace", False))
        cur = by_name.get(current_name) if isinstance(by_name.get(current_name), dict) else {
            "queued": False,
            "running": False,
            "replace": False,
            "queue_pos": 0,
            "task_id": str((snap or {}).get("cur_task_id") or ""),
        }
        cur["running"] = True
        cur["replace"] = bool(cur.get("replace")) or current_replace
        by_name[current_name] = cur
    return by_path, by_name


def _library_file_item(pdf: Path, *, md_root: Path, task_by_path: dict[str, dict], task_by_name: dict[str, dict]) -> dict:
    md_folder, md_main, md_exists = _resolve_md_output_paths(md_root, pdf)
    key = _normalized_path_key(pdf)
    info = task_by_path.get(key) if key else None
    if not isinstance(info, dict):
        info = task_by_name.get(pdf.name) if isinstance(task_by_name.get(pdf.name), dict) else {}
    queued = bool((info or {}).get("queued"))
    running = bool((info or {}).get("running"))
    replace_task = bool((info or {}).get("replace"))
    queue_pos = int((info or {}).get("queue_pos") or 0)
    task_state = "running" if running else ("queued" if queued else "idle")
    queued_or_running = bool(queued or running)
    reconverting = bool(replace_task and queued_or_running)
    category = "converted" if (md_exists and (not reconverting) and (not queued_or_running)) else "pending"
    if task_state == "running":
        status = "running_reconvert" if replace_task else "running"
    elif task_state == "queued":
        status = "queued_reconvert" if replace_task else "queued"
    else:
        status = "converted" if category == "converted" else "pending"
    return {
        "name": pdf.name,
        "path": str(pdf),
        "md_exists": bool(md_exists),
        "md_path": str(md_main) if md_exists else "",
        "md_folder": str(md_folder),
        "category": category,
        "task_state": task_state,
        "status": status,
        "replace_task": bool(replace_task),
        "queue_pos": int(queue_pos),
    }


def _collect_library_files(*, pdf_dir: Path, md_dir: Path, scope: str = "200") -> dict:
    pdfs_all = list(_list_pdf_paths_fast(pdf_dir))
    pdfs_all.sort(key=lambda p: p.name.lower())

    scope_raw = str(scope or "200").strip().lower()
    limit = 200
    if scope_raw in {"all", "*", "0", "full"}:
        limit = 0
    else:
        try:
            limit = max(1, min(5000, int(scope_raw)))
        except Exception:
            limit = 200

    view = pdfs_all if limit <= 0 else pdfs_all[:limit]
    snap = _bg_snapshot()
    task_by_path, task_by_name = _build_task_maps_from_snapshot(snap)
    items = [_library_file_item(pdf, md_root=md_dir, task_by_path=task_by_path, task_by_name=task_by_name) for pdf in view]

    pending = sum(1 for item in items if str(item.get("category") or "") == "pending")
    converted = sum(1 for item in items if str(item.get("category") or "") == "converted")
    queued = sum(1 for item in items if str(item.get("task_state") or "") == "queued")
    running = sum(1 for item in items if str(item.get("task_state") or "") == "running")
    reconverting = sum(1 for item in items if bool(item.get("replace_task")) and str(item.get("task_state") or "") in {"queued", "running"})

    return {
        "items": items,
        "counts": {
            "total_view": len(items),
            "total_all": len(pdfs_all),
            "pending": int(pending),
            "converted": int(converted),
            "queued": int(queued),
            "running": int(running),
            "reconverting": int(reconverting),
        },
        "truncated": bool(limit > 0 and len(pdfs_all) > len(view)),
        "scope": "all" if limit <= 0 else str(limit),
        "queue": {
            "running": bool(snap.get("running", False)),
            "current": str(snap.get("current", "")),
            "done": int(snap.get("done", 0) or 0),
            "total": int(snap.get("total", 0) or 0),
        },
    }


def _parse_rename_scan_limit(scope: str) -> int:
    raw = str(scope or "30").strip().lower()
    if raw in {"all", "*", "0", "full"}:
        return 0
    m = re.search(r"\d+", raw)
    if m:
        try:
            return max(1, min(2000, int(m.group(0))))
        except Exception:
            return 30
    return 30


def _recent_pdf_paths(pdf_dir: Path, limit: int) -> list[Path]:
    if limit <= 0:
        return list(_list_pdf_paths_fast(pdf_dir))
    pairs: list[tuple[float, Path]] = []
    for p in _list_pdf_paths_fast(pdf_dir):
        try:
            mtime = float(p.stat().st_mtime)
        except Exception:
            mtime = 0.0
        pairs.append((mtime, p))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in pairs[:limit]]


def _suggest_dest_for_base(*, pdf_dir: Path, current_pdf: Path, base_name: str, max_suffix: int = 200) -> Path:
    base = str(base_name or "").strip() or current_pdf.stem
    cand = pdf_dir / f"{base}.pdf"
    try:
        if cand.resolve() == current_pdf.resolve():
            return current_pdf
    except Exception:
        if str(cand) == str(current_pdf):
            return current_pdf
    if not cand.exists():
        return cand
    k = 2
    while k <= int(max_suffix):
        next_cand = pdf_dir / f"{base}-{k}.pdf"
        try:
            if next_cand.resolve() == current_pdf.resolve():
                return current_pdf
        except Exception:
            if str(next_cand) == str(current_pdf):
                return current_pdf
        if not next_cand.exists():
            return next_cand
        k += 1
    return pdf_dir / f"{base}-{max_suffix + 1}.pdf"


def _sync_md_after_pdf_rename_basic(*, md_root: Path, src_pdf: Path, dest_pdf: Path) -> dict:
    try:
        old_dir = (Path(md_root) / src_pdf.stem).expanduser()
        new_dir = (Path(md_root) / dest_pdf.stem).expanduser()
        target_dir = new_dir

        if old_dir.exists() and old_dir.is_dir() and (str(old_dir) != str(new_dir)):
            if (not new_dir.exists()):
                try:
                    old_dir.rename(new_dir)
                except Exception as exc:
                    return {"ok": False, "msg": f"md folder rename failed: {exc}"}
            else:
                target_dir = new_dir
        elif old_dir.exists() and old_dir.is_dir():
            target_dir = old_dir
        elif new_dir.exists() and new_dir.is_dir():
            target_dir = new_dir
        else:
            return {"ok": True, "msg": "no md folder"}

        old_main = target_dir / f"{src_pdf.stem}.en.md"
        new_main = target_dir / f"{dest_pdf.stem}.en.md"
        if old_main.exists() and old_main.is_file() and (str(old_main) != str(new_main)) and (not new_main.exists()):
            try:
                old_main.rename(new_main)
            except Exception as exc:
                return {"ok": False, "msg": f"md main rename failed: {exc}"}
        return {"ok": True, "msg": "md synced"}
    except Exception as exc:
        return {"ok": False, "msg": str(exc)}


def _build_rename_suggestion_item(*, pdf_path: Path, pdf_dir: Path, md_dir: Path, use_llm: bool) -> dict:
    try:
        st = pdf_path.stat()
        cache_key = f"{pdf_path.resolve()}|{int(st.st_mtime)}|{int(st.st_size)}|llm:{int(bool(use_llm))}"
    except Exception:
        cache_key = f"{pdf_path}|llm:{int(bool(use_llm))}"
    cached = _RENAME_SUGGEST_CACHE.get(cache_key)
    if isinstance(cached, dict):
        return dict(cached)

    settings = get_settings()
    try:
        suggestion = extract_pdf_meta_suggestion(pdf_path, settings=settings if use_llm else None)
    except Exception:
        suggestion = PdfMetaSuggestion()
    venue = str(getattr(suggestion, "venue", "") or "").strip()
    year = str(getattr(suggestion, "year", "") or "").strip()
    title = str(getattr(suggestion, "title", "") or "").strip() or pdf_path.stem
    base_name = build_storage_base_name(
        venue=venue,
        year=year,
        title=title,
        pdf_dir=pdf_dir,
        md_out_root=md_dir,
    )
    dest = _suggest_dest_for_base(pdf_dir=pdf_dir, current_pdf=pdf_path, base_name=base_name)
    display_full_name = build_display_pdf_filename(
        venue=venue,
        year=year,
        title=title,
        fallback_name=pdf_path.name,
    )
    md_folder, md_main, md_exists = _resolve_md_output_paths(md_dir, pdf_path)
    out = {
        "name": pdf_path.name,
        "path": str(pdf_path),
        "suggested_name": dest.name,
        "suggested_stem": dest.stem,
        "display_full_name": display_full_name,
        "diff": str(dest.name) != str(pdf_path.name),
        "meta": {
            "venue": venue,
            "year": year,
            "title": title,
        },
        "md_exists": bool(md_exists),
        "md_path": str(md_main) if md_exists else "",
        "md_folder": str(md_folder),
    }
    try:
        if len(_RENAME_SUGGEST_CACHE) > 4000:
            _RENAME_SUGGEST_CACHE.clear()
        _RENAME_SUGGEST_CACHE[cache_key] = dict(out)
    except Exception:
        pass
    return out


def _existing_pdf_record(pdf_dir: Path, sha1: str, lib_store: LibraryStore | None = None) -> dict | None:
    store = lib_store or _library_store()
    record = None
    try:
        record = store.get_by_sha1(sha1)
    except Exception:
        record = None

    if isinstance(record, dict):
        existing_path = Path(str(record.get("path") or "")).expanduser()
        if existing_path.exists() and existing_path.is_file():
            return {
                "name": existing_path.name,
                "path": str(existing_path),
                "sha1": sha1,
            }

    for existing in _list_pdf_paths_fast(pdf_dir):
        try:
            if _sha1_bytes(existing.read_bytes()) == sha1:
                return {
                    "name": existing.name,
                    "path": str(existing),
                    "sha1": sha1,
                }
        except Exception:
            continue
    return None


def save_pdf_to_library(*, file_name: str, data: bytes, base_name: str = "", fast_mode: bool = False, allow_duplicate: bool = False) -> dict:
    settings = get_settings()
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    pdf_d.mkdir(parents=True, exist_ok=True)
    md_d.mkdir(parents=True, exist_ok=True)

    sha1 = _sha1_bytes(data)
    lib_store = _library_store()
    if not bool(allow_duplicate):
        existing = _existing_pdf_record(pdf_d, sha1, lib_store=lib_store)
        if existing:
            return {
                "duplicate": True,
                "existing": str(existing.get("name") or ""),
                "path": str(existing.get("path") or ""),
                "name": str(existing.get("name") or ""),
                "sha1": sha1,
            }

    raw_name_pdf = str(file_name or "upload.pdf").strip() or "upload.pdf"
    tmp_path = _write_tmp_upload(pdf_d, raw_name_pdf, data)
    dest_pdf: Path | None = None
    try:
        if fast_mode:
            sug = PdfMetaSuggestion()
        else:
            try:
                sug = extract_pdf_meta_suggestion(tmp_path, settings=settings)
            except Exception:
                sug = PdfMetaSuggestion()

        fallback_title = Path((base_name or "").strip() or raw_name_pdf).stem or "Untitled"
        venue = str(getattr(sug, "venue", "") or "").strip()
        year = str(getattr(sug, "year", "") or "").strip()
        title = str(getattr(sug, "title", "") or "").strip() or fallback_title

        base = build_storage_base_name(
            venue=venue,
            year=year,
            title=title,
            pdf_dir=pdf_d,
            md_out_root=md_d,
        )
        dest_pdf = _next_pdf_dest_path(pdf_d, base)
        display_full_name = build_display_pdf_filename(
            venue=venue,
            year=year,
            title=title,
            fallback_name=raw_name_pdf,
        )
        citation_meta = merge_citation_meta_file_labels(
            getattr(sug, "crossref_meta", None) if isinstance(getattr(sug, "crossref_meta", None), dict) else None,
            display_full_name=display_full_name,
            storage_filename=dest_pdf.name,
        )
        citation_meta = merge_citation_meta_name_fields(
            citation_meta,
            venue=venue,
            year=year,
            title=title,
        )

        _persist_upload_pdf(tmp_path, dest_pdf, data)
        lib_store.upsert(sha1, dest_pdf, citation_meta=citation_meta)
        return {
            "duplicate": False,
            "path": str(dest_pdf),
            "name": dest_pdf.name,
            "sha1": sha1,
            "citation_meta": citation_meta,
        }
    finally:
        try:
            if tmp_path.exists() and (dest_pdf is None or tmp_path.resolve() != dest_pdf.resolve()):
                tmp_path.unlink()
        except Exception:
            pass


def auto_rename_saved_pdf_in_library(*, pdf_path: Path, base_name: str = "", use_llm: bool = True, also_md: bool = True) -> dict:
    settings = get_settings()
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    lib_store = _library_store()
    src_pdf = Path(pdf_path).expanduser().resolve()
    if (not src_pdf.exists()) or (not src_pdf.is_file()):
        return {"ok": False, "error": "pdf not found", "path": str(src_pdf), "name": src_pdf.name}

    try:
        sha1 = _sha1_bytes(src_pdf.read_bytes())
    except Exception:
        sha1 = ""

    try:
        sug = extract_pdf_meta_suggestion(src_pdf, settings=settings if use_llm else None)
    except Exception:
        sug = PdfMetaSuggestion()

    fallback_title = Path((base_name or "").strip() or src_pdf.stem).stem or "Untitled"
    venue = str(getattr(sug, "venue", "") or "").strip()
    year = str(getattr(sug, "year", "") or "").strip()
    title = str(getattr(sug, "title", "") or "").strip() or fallback_title

    base = build_storage_base_name(
        venue=venue,
        year=year,
        title=title,
        pdf_dir=pdf_d,
        md_out_root=md_d,
    )
    cand_pdf = _next_pdf_dest_path(pdf_d, base)
    dest_pdf = cand_pdf if cand_pdf.resolve() != src_pdf else src_pdf
    if (not dest_pdf.exists()) and (dest_pdf.resolve() != src_pdf):
        try:
            src_pdf.rename(dest_pdf)
        except Exception:
            dest_pdf = src_pdf

    display_full_name = build_display_pdf_filename(
        venue=venue,
        year=year,
        title=title,
        fallback_name=src_pdf.name,
    )
    citation_meta = merge_citation_meta_file_labels(
        getattr(sug, "crossref_meta", None) if isinstance(getattr(sug, "crossref_meta", None), dict) else None,
        display_full_name=display_full_name,
        storage_filename=dest_pdf.name,
    )
    citation_meta = merge_citation_meta_name_fields(
        citation_meta,
        venue=venue,
        year=year,
        title=title,
    )
    if sha1:
        lib_store.upsert(sha1, dest_pdf, citation_meta=citation_meta)
    else:
        try:
            lib_store.update_path(src_pdf, dest_pdf)
            lib_store.set_citation_meta(dest_pdf, citation_meta)
        except Exception:
            pass

    md_sync = {"ok": True, "msg": "skipped"}
    if bool(also_md) and (str(dest_pdf) != str(src_pdf)):
        md_sync = _sync_md_after_pdf_rename_basic(md_root=md_d, src_pdf=src_pdf, dest_pdf=dest_pdf)

    return {
        "ok": True,
        "path": str(dest_pdf),
        "name": dest_pdf.name,
        "sha1": sha1,
        "citation_meta": citation_meta,
        "renamed": str(dest_pdf) != str(src_pdf),
        "md_sync": md_sync,
    }


def quick_ingest_pdf(
    *,
    pdf_path: Path,
    speed_mode: str = "ultra_fast",
    progress_cb: Callable[[str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    ingest_proc_cb: Callable[[subprocess.Popen | None], None] | None = None,
) -> dict:
    settings = get_settings()
    md_d = _md_dir()
    md_d.mkdir(parents=True, exist_ok=True)

    def _report(stage: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(stage)
        except Exception:
            pass

    no_llm = str(speed_mode or "").strip().lower() == "no_llm"
    _report("converting")
    ok, out_folder = run_pdf_to_md(
        pdf_path=Path(pdf_path),
        out_root=md_d,
        no_llm=no_llm,
        keep_debug=False,
        eq_image_fallback=False,
        cancel_cb=cancel_cb,
        speed_mode=speed_mode,
    )
    if not ok:
        return {
            "ready": False,
            "error": str(out_folder or "convert failed"),
            "cancelled": str(out_folder or "").strip().lower() == "cancelled",
        }

    _, md_main, md_exists = _resolve_md_output_paths(md_d, Path(pdf_path))
    if not md_exists:
        return {
            "ready": False,
            "error": "markdown output missing",
        }

    _report("ingesting")
    ingest_res = _ingest_markdown_incremental(
        md_main=md_main,
        db_dir=Path(settings.db_dir).expanduser(),
        cancel_cb=cancel_cb,
        ingest_proc_cb=ingest_proc_cb,
    )
    if not bool(ingest_res.get("ready")):
        return ingest_res

    return {
        "ready": True,
        "md_path": str(md_main),
        "out_folder": str(out_folder),
    }


def _ingest_markdown_incremental(
    *,
    md_main: Path,
    db_dir: Path,
    cancel_cb: Callable[[], bool] | None = None,
    ingest_proc_cb: Callable[[subprocess.Popen | None], None] | None = None,
) -> dict:
    ingest_py = _ingest_py_path()
    if not ingest_py.exists():
        return {
            "ready": False,
            "error": "ingest.py not found",
        }

    def _terminate_proc(proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is not None:
                return
        except Exception:
            return
        try:
            proc.terminate()
            proc.wait(timeout=4)
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)
        except Exception:
            pass

    proc = subprocess.Popen(
        [os.sys.executable, str(ingest_py), "--src", str(md_main), "--db", str(db_dir), "--incremental"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if ingest_proc_cb is not None:
        try:
            ingest_proc_cb(proc)
        except Exception:
            pass
    try:
        while proc.poll() is None:
            if cancel_cb is not None:
                try:
                    if bool(cancel_cb()):
                        _terminate_proc(proc)
                        return {
                            "ready": False,
                            "error": "cancelled",
                            "cancelled": True,
                        }
                except Exception:
                    pass
            time.sleep(0.2)
        stderr_text = ""
        try:
            if proc.stderr is not None:
                stderr_text = str(proc.stderr.read() or "")
        except Exception:
            stderr_text = ""
    finally:
        if ingest_proc_cb is not None:
            try:
                ingest_proc_cb(None)
            except Exception:
                pass

    if proc.returncode != 0:
        return {
            "ready": False,
            "error": (stderr_text or "ingest failed").strip()[-500:],
        }
    return {
        "ready": True,
    }


def refine_pdf_with_full_llm_replace(
    *,
    pdf_path: Path,
    progress_cb: Callable[[str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    ingest_proc_cb: Callable[[subprocess.Popen | None], None] | None = None,
) -> dict:
    settings = get_settings()
    md_d = _md_dir()
    md_d.mkdir(parents=True, exist_ok=True)
    pdf = Path(pdf_path).expanduser().resolve()

    def _report(stage: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(stage)
        except Exception:
            pass

    shadow_root = md_d / "temp" / "_quality_refine_jobs" / f"{pdf.stem}-{uuid.uuid4().hex[:10]}"
    shadow_root.mkdir(parents=True, exist_ok=True)

    def _cleanup_shadow() -> None:
        try:
            if shadow_root.exists():
                shutil.rmtree(shadow_root, ignore_errors=True)
        except Exception:
            pass

    def _cancel_requested() -> bool:
        if cancel_cb is None:
            return False
        try:
            return bool(cancel_cb())
        except Exception:
            return False

    _report("refining")
    ok, out_folder = run_pdf_to_md(
        pdf_path=pdf,
        out_root=shadow_root,
        no_llm=False,
        keep_debug=False,
        eq_image_fallback=False,
        cancel_cb=cancel_cb,
        speed_mode="full_llm",
    )
    if not ok:
        _cleanup_shadow()
        return {
            "ready": False,
            "error": str(out_folder or "refine convert failed"),
            "cancelled": str(out_folder or "").strip().lower() == "cancelled",
        }

    if _cancel_requested():
        _cleanup_shadow()
        return {
            "ready": False,
            "error": "cancelled",
            "cancelled": True,
        }

    shadow_folder, shadow_md_main, shadow_exists = _resolve_md_output_paths(shadow_root, pdf)
    if (not shadow_exists) or (not shadow_md_main.exists()):
        _cleanup_shadow()
        return {
            "ready": False,
            "error": "refine markdown output missing",
        }

    target_folder = md_d / pdf.stem
    backup_root = md_d / "temp" / "_quality_refine_backup"
    backup_folder = backup_root / f"{pdf.stem}-{uuid.uuid4().hex[:8]}"
    had_backup = False

    def _rollback_target() -> None:
        if not had_backup:
            return
        try:
            if target_folder.exists():
                shutil.rmtree(target_folder, ignore_errors=True)
        except Exception:
            pass
        try:
            if backup_folder.exists():
                target_folder.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(backup_folder), str(target_folder))
        except Exception:
            pass

    try:
        if target_folder.exists():
            backup_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target_folder), str(backup_folder))
            had_backup = True
        target_folder.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(shadow_folder), str(target_folder))
    except Exception as exc:
        _rollback_target()
        _cleanup_shadow()
        return {
            "ready": False,
            "error": f"refine swap failed: {exc}",
        }
    finally:
        _cleanup_shadow()

    _, target_md_main, target_exists = _resolve_md_output_paths(md_d, pdf)
    if (not target_exists) or (not target_md_main.exists()):
        _rollback_target()
        return {
            "ready": False,
            "error": "refine target markdown missing",
        }

    _report("refine_ingesting")
    ingest_res = _ingest_markdown_incremental(
        md_main=target_md_main,
        db_dir=Path(settings.db_dir).expanduser(),
        cancel_cb=cancel_cb,
        ingest_proc_cb=ingest_proc_cb,
    )
    if not bool(ingest_res.get("ready")):
        _rollback_target()
        return ingest_res

    if had_backup:
        try:
            if backup_folder.exists():
                shutil.rmtree(backup_folder, ignore_errors=True)
        except Exception:
            pass

    return {
        "ready": True,
        "md_path": str(target_md_main),
        "out_folder": str(out_folder),
        "refined": True,
    }


@router.get("/pdfs")
def list_pdfs():
    d = _pdf_dir()
    if not d.exists():
        return []
    return [{"name": p.name, "path": str(p)} for p in _list_pdf_paths_fast(d)]


@router.get("/files")
def list_library_files(scope: str = "200"):
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    pdf_d.mkdir(parents=True, exist_ok=True)
    md_d.mkdir(parents=True, exist_ok=True)
    return _collect_library_files(pdf_dir=pdf_d, md_dir=md_d, scope=scope)


class ConvertPendingBody(BaseModel):
    speed_mode: str = "balanced"
    no_llm: bool = False
    limit: int = 0


def _inspect_pdf_upload(*, file_name: str, data: bytes, use_llm: bool = True) -> dict:
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    pdf_d.mkdir(parents=True, exist_ok=True)
    md_d.mkdir(parents=True, exist_ok=True)

    raw_name_pdf = str(file_name or "upload.pdf").strip() or "upload.pdf"
    sha1 = _sha1_bytes(data)
    existing = _existing_pdf_record(pdf_d, sha1)

    tmp_path = _write_tmp_upload(pdf_d, raw_name_pdf, data)
    try:
        if bool(use_llm):
            try:
                sug = extract_pdf_meta_suggestion(tmp_path, settings=get_settings())
            except Exception:
                sug = PdfMetaSuggestion()
        else:
            sug = PdfMetaSuggestion()

        fallback_title = Path(raw_name_pdf).stem or "Untitled"
        venue = str(getattr(sug, "venue", "") or "").strip()
        year = str(getattr(sug, "year", "") or "").strip()
        title = str(getattr(sug, "title", "") or "").strip() or fallback_title
        base = build_storage_base_name(
            venue=venue,
            year=year,
            title=title,
            pdf_dir=pdf_d,
            md_out_root=md_d,
        )
        dest = _next_pdf_dest_path(pdf_d, base)
        display_full_name = build_display_pdf_filename(
            venue=venue,
            year=year,
            title=title,
            fallback_name=raw_name_pdf,
        )
        return {
            "name": raw_name_pdf,
            "sha1": sha1,
            "duplicate": bool(existing),
            "existing": str((existing or {}).get("name") or ""),
            "existing_path": str((existing or {}).get("path") or ""),
            "suggested_name": dest.name,
            "suggested_stem": dest.stem,
            "display_full_name": display_full_name,
            "meta": {
                "venue": venue,
                "year": year,
                "title": title,
            },
        }
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


@router.get("/rename/suggestions")
def list_rename_suggestions(scope: str = "30", use_llm: bool = True):
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    pdf_d.mkdir(parents=True, exist_ok=True)
    md_d.mkdir(parents=True, exist_ok=True)
    limit = _parse_rename_scan_limit(scope)
    pdfs = _recent_pdf_paths(pdf_d, limit)
    if limit <= 0:
        pdfs.sort(key=lambda p: p.name.lower())
    items: list[dict] = []
    workers = 2 if bool(use_llm) else 6
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut_map = {
            ex.submit(_build_rename_suggestion_item, pdf_path=p, pdf_dir=pdf_d, md_dir=md_d, use_llm=bool(use_llm)): idx
            for idx, p in enumerate(pdfs)
        }
        temp: list[tuple[int, dict]] = []
        for fut in as_completed(fut_map):
            idx = int(fut_map[fut])
            try:
                rec = fut.result()
            except Exception:
                p = pdfs[idx]
                rec = {
                    "name": p.name,
                    "path": str(p),
                    "suggested_name": p.name,
                    "suggested_stem": p.stem,
                    "display_full_name": p.name,
                    "diff": False,
                    "meta": {"venue": "", "year": "", "title": ""},
                    "md_exists": False,
                    "md_path": "",
                    "md_folder": "",
                    "error": "suggestion failed",
                }
            temp.append((idx, rec))
        temp.sort(key=lambda x: x[0])
        items = [it for _, it in temp]
    changed = sum(1 for item in items if bool(item.get("diff")))
    return {
        "items": items,
        "scope": "all" if limit <= 0 else str(limit),
        "use_llm": bool(use_llm),
        "total_scanned": len(items),
        "changed": int(changed),
    }


class RenameApplyBody(BaseModel):
    pdf_names: list[str] = []
    base_overrides: dict[str, str] = {}
    use_llm: bool = True
    also_md: bool = True


@router.post("/rename/apply")
def apply_rename_suggestions(body: RenameApplyBody):
    pdf_d = _pdf_dir()
    names = [str(name or "").strip() for name in list(body.pdf_names or [])]
    names = [name for name in names if name and (Path(name).name == name)]
    if not names:
        raise HTTPException(400, "pdf_names required")

    items: list[dict] = []
    renamed = 0
    failed = 0
    skipped = 0
    for name in names:
        src_pdf = (pdf_d / name).expanduser()
        if not _path_is_file(src_pdf):
            items.append({"name": name, "ok": False, "error": "pdf not found"})
            failed += 1
            continue
        override = str((body.base_overrides or {}).get(name) or "").strip()
        result = auto_rename_saved_pdf_in_library(
            pdf_path=src_pdf,
            base_name=override,
            use_llm=bool(body.use_llm),
            also_md=bool(body.also_md),
        )
        ok = bool(result.get("ok"))
        was_renamed = bool(result.get("renamed"))
        if ok and was_renamed:
            renamed += 1
        elif ok:
            skipped += 1
        else:
            failed += 1
        items.append({
            "name": name,
            **result,
        })

    return {
        "ok": failed == 0,
        "renamed": int(renamed),
        "skipped": int(skipped),
        "failed": int(failed),
        "needs_reindex": bool(renamed > 0),
        "items": items,
    }


@router.post("/convert/pending")
def convert_pending(body: ConvertPendingBody):
    s = get_settings()
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    md_d.mkdir(parents=True, exist_ok=True)

    listing = _collect_library_files(pdf_dir=pdf_d, md_dir=md_d, scope="all")
    items = list(listing.get("items") or [])
    limit = max(0, int(body.limit or 0))
    no_llm = bool(body.no_llm) or (str(body.speed_mode or "").strip().lower() == "no_llm")

    enqueued = 0
    skipped_busy = 0
    pending_total = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("category") or "") != "pending":
            continue
        pending_total += 1
        if str(item.get("task_state") or "") != "idle":
            skipped_busy += 1
            continue
        pdf_path = Path(str(item.get("path") or "")).expanduser()
        if not _path_is_file(pdf_path):
            continue
        task = _build_bg_task(
            pdf_path=pdf_path,
            out_root=md_d,
            db_dir=Path(s.db_dir).expanduser(),
            no_llm=no_llm,
            replace=False,
            speed_mode=str(body.speed_mode or "balanced"),
        )
        _bg_enqueue(task)
        enqueued += 1
        if limit > 0 and enqueued >= limit:
            break

    return {
        "ok": True,
        "enqueued": int(enqueued),
        "skipped_busy": int(skipped_busy),
        "pending_total": int(pending_total),
    }


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), base_name: str = Form("")):
    data = await file.read()
    return save_pdf_to_library(
        file_name=str(file.filename or "upload.pdf"),
        data=data,
        base_name=base_name,
    )


@router.post("/upload/inspect")
async def inspect_upload_pdf(file: UploadFile = File(...), use_llm: bool = Form(True)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    return _inspect_pdf_upload(
        file_name=str(file.filename or "upload.pdf"),
        data=data,
        use_llm=bool(use_llm),
    )


@router.post("/upload/commit")
async def commit_upload_pdf(
    file: UploadFile = File(...),
    base_name: str = Form(""),
    convert_now: bool = Form(False),
    speed_mode: str = Form("balanced"),
    allow_duplicate: bool = Form(False),
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    saved = save_pdf_to_library(
        file_name=str(file.filename or "upload.pdf"),
        data=data,
        base_name=base_name,
        allow_duplicate=bool(allow_duplicate),
    )
    enqueued = False
    task_id = ""
    if bool(convert_now) and (not bool(saved.get("duplicate"))):
        s = get_settings()
        md_d = _md_dir()
        md_d.mkdir(parents=True, exist_ok=True)
        pdf_path = Path(str(saved.get("path") or "")).expanduser()
        if _path_is_file(pdf_path):
            no_llm = str(speed_mode or "").strip().lower() == "no_llm"
            task = _build_bg_task(
                pdf_path=pdf_path,
                out_root=md_d,
                db_dir=s.db_dir,
                no_llm=no_llm,
                replace=False,
                speed_mode=str(speed_mode or "balanced"),
            )
            _bg_enqueue(task)
            enqueued = True
            task_id = str(task.get("_tid") or "")
    return {
        **saved,
        "enqueued": bool(enqueued),
        "task_id": task_id,
    }


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
    no_llm = bool(body.no_llm) or (str(body.speed_mode or "").strip().lower() == "no_llm")
    task = _build_bg_task(
        pdf_path=pdf_path,
        out_root=md_d,
        db_dir=s.db_dir,
        no_llm=no_llm,
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


class OpenLibraryFileBody(BaseModel):
    pdf_name: str
    target: str = "pdf"  # pdf | md | pdf_dir | md_dir


@router.post("/file/open")
def open_library_file(body: OpenLibraryFileBody):
    target = str(body.target or "pdf").strip().lower()
    pdf_d = _pdf_dir()
    md_d = _md_dir()

    if target == "pdf_dir":
        open_in_explorer(pdf_d)
        return {"ok": True, "target": "pdf_dir", "path": str(pdf_d)}
    if target == "md_dir":
        open_in_explorer(md_d)
        return {"ok": True, "target": "md_dir", "path": str(md_d)}

    pdf_name = str(body.pdf_name or "").strip()
    if (not pdf_name) or (Path(pdf_name).name != pdf_name):
        raise HTTPException(400, "invalid pdf_name")
    pdf_path = (pdf_d / pdf_name).expanduser()

    if target == "pdf":
        if not _path_is_file(pdf_path):
            raise HTTPException(404, "pdf not found")
        open_in_explorer(pdf_path)
        return {"ok": True, "target": "pdf", "path": str(pdf_path)}
    if target == "md":
        _, md_main, md_exists = _resolve_md_output_paths(md_d, pdf_path)
        if (not md_exists) or (not _path_is_file(md_main)):
            raise HTTPException(404, "markdown not found")
        open_in_explorer(md_main)
        return {"ok": True, "target": "md", "path": str(md_main)}
    raise HTTPException(400, "invalid target")


class DeleteLibraryFileBody(BaseModel):
    pdf_name: str
    also_md: bool = True
    remove_queued: bool = True


@router.post("/file/delete")
def delete_library_file(body: DeleteLibraryFileBody):
    pdf_name = str(body.pdf_name or "").strip()
    if (not pdf_name) or (Path(pdf_name).name != pdf_name):
        raise HTTPException(400, "invalid pdf_name")
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    pdf_path = (pdf_d / pdf_name).expanduser()
    if not _path_is_file(pdf_path):
        raise HTTPException(404, "pdf not found")

    snap = _bg_snapshot()
    running_now = bool(snap.get("running"))
    current_name = str(snap.get("current") or "").strip()
    if running_now and current_name and (current_name == pdf_name):
        raise HTTPException(409, "file is currently converting")

    removed_queued = 0
    if bool(body.remove_queued):
        try:
            removed_queued = int(_bg_remove_queued_tasks_for_pdf(pdf_path) or 0)
        except Exception:
            removed_queued = 0

    pdf_ok, pdf_err = _safe_delete_file(pdf_path)
    md_deleted = False
    md_warn = ""
    if bool(body.also_md):
        try:
            md_root = md_d.resolve()
            target = (md_d / pdf_path.stem).resolve()
            if target != md_root and md_root in target.parents and _path_exists(target):
                ok_md, msg_md = _safe_delete_tree(target)
                md_deleted = bool(ok_md)
                if not ok_md:
                    md_warn = str(msg_md or "")
            else:
                md_deleted = not _path_exists(target)
        except Exception as exc:
            md_warn = str(exc)
            md_deleted = False

    try:
        _library_store().delete_by_path(pdf_path)
    except Exception:
        pass

    warnings: list[str] = []
    if (not pdf_ok) and pdf_err:
        warnings.append(f"pdf: {pdf_err}")
    if bool(body.also_md) and (not md_deleted) and md_warn:
        warnings.append(f"md: {md_warn}")
    return {
        "ok": bool(pdf_ok) and (not bool(body.also_md) or bool(md_deleted)),
        "pdf_deleted": bool(pdf_ok),
        "md_deleted": bool(md_deleted) if bool(body.also_md) else False,
        "removed_queued": int(removed_queued),
        "warnings": warnings,
        "needs_reindex": bool(pdf_ok),
    }


@router.post("/reindex")
def reindex():
    s = get_settings()
    md_d = _md_dir()
    pdf_d = _pdf_dir()
    ingest_py = _ingest_py_path()
    if not ingest_py.exists():
        return {"ok": False, "error": "ingest.py not found", "refsync": None, "refsync_error": ""}
    result = subprocess.run(
        [os.sys.executable, str(ingest_py), "--src", str(md_d), "--db", str(s.db_dir), "--incremental", "--prune"],
        capture_output=True, text=True, timeout=300,
    )
    ok = result.returncode == 0
    refsync: dict | None = None
    refsync_error = ""
    if ok:
        try:
            try:
                budget_s = float(os.environ.get("KB_CROSSREF_BUDGET_S", "45") or 45.0)
            except Exception:
                budget_s = 45.0
            try:
                workers = int(os.environ.get("KB_REFSYNC_WORKERS", "6") or 6)
            except Exception:
                workers = 6
            refsync = start_reference_sync(
                src_root=md_d,
                db_dir=Path(s.db_dir).expanduser(),
                pdf_root=pdf_d,
                library_db_path=Path(s.library_db_path).expanduser(),
                incremental=True,
                enable_title_lookup=True,
                crossref_time_budget_s=float(max(5.0, budget_s)),
                doi_prefetch_workers=int(max(1, min(16, workers))),
            )
        except Exception as exc:
            refsync_error = str(exc)
    return {
        "ok": bool(ok),
        "stdout": result.stdout[-500:],
        "stderr": result.stderr[-500:],
        "refsync": refsync,
        "refsync_error": refsync_error,
    }
