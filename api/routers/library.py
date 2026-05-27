from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
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
from kb.converter.quality_acceptance import summarize_conversion_quality
from kb.converter.quality_repair import conversion_repair_strategy_for_issue, repair_markdown_quality
from kb.converter.structured_index_batch import rebuild_structured_indices_for_root
from kb.library_store import LibraryStore
from kb.pdf_tools import PdfMetaSuggestion, extract_pdf_meta_suggestion, run_pdf_to_md, open_in_explorer
from kb.reference_sync import start_reference_sync

router = APIRouter(prefix="/api/library", tags=["library"])
_RENAME_SUGGEST_CACHE: dict[str, dict] = {}
_CONVERSION_QUALITY_CACHE: dict[str, tuple[int, int, dict]] = {}
_RESEARCH_QA_EVAL_ROOT = Path("test_results") / "research_qa_eval"


def _suggestion_basis_meta(suggestion: PdfMetaSuggestion, *, venue: str, year: str, title: str) -> dict:
    match_method = str(getattr(suggestion, "match_method", "") or "").strip()
    year_source = str(getattr(suggestion, "year_source", "") or "").strip()
    if match_method == "doi":
        basis_label = "DOI确认"
        basis_detail = "题录由 DOI 直连确认。"
    elif match_method == "crossref_strong":
        basis_label = "Crossref强匹配"
        basis_detail = "题录由 Crossref 强匹配确认。"
    elif match_method == "crossref_weak" and year_source == "filename" and year:
        basis_label = "年份沿用文件名"
        basis_detail = "Crossref 未形成强确认，年份沿用原文件名。"
    elif match_method == "crossref_weak":
        basis_label = "Crossref弱匹配"
        basis_detail = "Crossref 未形成强确认，年份未直接采用。"
    elif year_source == "filename" and year:
        basis_label = "年份沿用文件名"
        basis_detail = "年份沿用原文件名，未额外做强校验。"
    elif year_source == "heuristic" and year:
        basis_label = "首页文本提取"
        basis_detail = "年份根据 PDF 首页文本规则提取。"
    elif year_source == "llm" and year:
        basis_label = "LLM提取"
        basis_detail = "年份根据首页内容由 LLM 抽取。"
    else:
        basis_label = "启发式建议"
        basis_detail = "建议名来自标题/会议信息提取，未形成强确认。"
    return {
        "venue": str(venue or ""),
        "year": str(year or ""),
        "title": str(title or ""),
        "match_method": match_method,
        "year_source": year_source,
        "basis_label": basis_label,
        "basis_detail": basis_detail,
    }


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


def _strip_known_source_ext(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return ""
    s = re.sub(r"\.en\.md$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\.md$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\.pdf$", "", s, flags=re.IGNORECASE)
    return s.strip()


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


def _merge_task_map_entry(mapping: dict[str, dict], key: str, info: dict) -> None:
    k = str(key or "").strip()
    if not k:
        return
    prev = mapping.get(k)
    if not isinstance(prev, dict):
        mapping[k] = dict(info)
        return
    merged = dict(prev)
    merged["queued"] = bool(prev.get("queued")) or bool(info.get("queued"))
    merged["running"] = bool(prev.get("running")) or bool(info.get("running"))
    merged["replace"] = bool(prev.get("replace")) or bool(info.get("replace"))
    prev_q = int(prev.get("queue_pos") or 0)
    next_q = int(info.get("queue_pos") or 0)
    if next_q > 0 and ((prev_q <= 0) or (next_q < prev_q)):
        merged["queue_pos"] = next_q
    else:
        merged["queue_pos"] = prev_q
    next_tid = str(info.get("task_id") or "").strip()
    if next_tid and (not str(prev.get("task_id") or "").strip()):
        merged["task_id"] = next_tid
    for field in ("cur_page_done", "cur_page_total", "cur_page_msg"):
        if field in info:
            merged[field] = info.get(field)
    mapping[k] = merged


def _conversion_quality_issue(code: str, label: str, *, severity: str = "warning", count: int = 0) -> dict:
    strategy = conversion_repair_strategy_for_issue(code)
    return {
        "code": str(code or ""),
        "label": str(label or ""),
        "severity": str(severity or "warning"),
        "count": int(count or 0),
        "repairable": bool(strategy.get("safe")),
        "repair_strategy": str(strategy.get("label") or ""),
        "repair_steps": list(strategy.get("strategies") or []),
    }


def _conversion_quality_summary(md_path: str | Path) -> dict | None:
    path = Path(md_path)
    try:
        if not _path_is_file(path):
            return None
    except Exception:
        if not path.exists() or not path.is_file():
            return None

    try:
        stat = path.stat()
        cache_key = str(path.expanduser().resolve())
        cached = _CONVERSION_QUALITY_CACHE.get(cache_key)
        if cached and cached[0] == int(stat.st_mtime_ns) and cached[1] == int(stat.st_size):
            return dict(cached[2])

        metrics = summarize_conversion_quality(path)
        metric_view = {
            "chars": int(metrics.chars),
            "headings": int(metrics.heading_count),
            "page_markers": int(metrics.page_marker_count),
            "page_marker_gaps": int(metrics.page_marker_gap_count),
            "figures": int(metrics.image_count),
            "missing_images": int(metrics.missing_image_count),
            "captions": int(metrics.caption_count),
            "tables": int(metrics.table_block_count),
            "display_math": int(metrics.display_math_block_count),
            "inline_math": int(metrics.inline_math_count),
            "unclosed_display_math": int(metrics.unclosed_display_math_block_count),
            "references": int(metrics.extracted_reference_count),
            "reference_lines": int(metrics.reference_line_count),
            "body_citations": int(metrics.body_citation_expanded_index_count),
            "mojibake": int(metrics.mojibake_count),
            "analyzer_errors": int(metrics.analyzer_error_count),
            "analyzer_warnings": int(metrics.analyzer_warning_count),
        }
        issues: list[dict] = []
        score = 100

        def add_issue(code: str, label: str, *, severity: str = "warning", count: int = 0, penalty: int = 6) -> None:
            nonlocal score
            n = max(1, int(count or 0))
            issues.append(_conversion_quality_issue(code, label, severity=severity, count=count))
            score -= min(36, max(0, int(penalty)) * min(n, 4))

        if metrics.missing_image_count > 0:
            add_issue("missing_images", "Missing image assets", severity="error", count=metrics.missing_image_count, penalty=10)
        if metrics.unclosed_display_math_block_count > 0:
            add_issue("unclosed_display_math", "Unclosed display math", severity="error", count=metrics.unclosed_display_math_block_count, penalty=18)
        if metrics.mojibake_count > 0:
            add_issue("mojibake", "Encoding artifacts", severity="error", count=metrics.mojibake_count, penalty=12)
        if metrics.analyzer_error_count > 0:
            add_issue("analyzer_errors", "Markdown analyzer errors", severity="error", count=metrics.analyzer_error_count, penalty=12)
        if metrics.heading_count <= 1:
            add_issue("weak_structure", "Weak heading structure", count=metrics.heading_count, penalty=8)
        if not metrics.has_abstract_heading:
            add_issue("missing_abstract", "Missing abstract heading", penalty=5)
        if metrics.page_marker_count <= 0:
            add_issue("missing_page_markers", "Missing page anchors", penalty=8)
        if metrics.page_marker_gap_count > 0:
            add_issue("page_marker_gaps", "Page anchor gaps", count=metrics.page_marker_gap_count, penalty=6)
        if metrics.extracted_reference_count <= 0 and metrics.reference_line_count <= 0:
            add_issue("missing_references", "Missing reference list", penalty=12)
        if metrics.image_count > 0 and metrics.caption_count <= 0:
            add_issue("missing_captions", "Figures lack captions", count=metrics.image_count, penalty=5)
        if metrics.analyzer_warning_count > 3:
            add_issue("analyzer_warnings", "Markdown analyzer warnings", count=metrics.analyzer_warning_count, penalty=3)
        if metrics.heading_level_jump_count > 0:
            add_issue("heading_level_jumps", "Heading level jumps", count=metrics.heading_level_jump_count, penalty=4)

        hard_issue = any(str(item.get("severity") or "") == "error" for item in issues)
        status = "error" if hard_issue else ("warning" if issues else "good")
        label = "Needs repair" if status == "error" else ("Needs review" if status == "warning" else "Ready")
        score = max(0, min(100, int(score)))
        summary = (
            f"{label} | Q{score} | "
            f"{metrics.page_marker_count} pages | "
            f"{metrics.extracted_reference_count or metrics.reference_line_count} refs | "
            f"{metrics.image_count} figures | "
            f"{metrics.display_math_block_count + metrics.inline_math_count} math"
        )
        result = {
            "status": status,
            "label": label,
            "score": score,
            "summary": summary,
            "has_review_issue": bool(issues),
            "issues": issues[:8],
            "metrics": metric_view,
        }
        _CONVERSION_QUALITY_CACHE[cache_key] = (int(stat.st_mtime_ns), int(stat.st_size), dict(result))
        return result
    except Exception as exc:
        return {
            "status": "error",
            "label": "Quality scan failed",
            "score": 0,
            "summary": "Quality scan failed",
            "has_review_issue": True,
            "issues": [
                _conversion_quality_issue(
                    "quality_scan_failed",
                    str(exc)[:160] or "Quality scan failed",
                    severity="error",
                    count=1,
                )
            ],
            "metrics": {},
        }


def _clear_conversion_quality_cache(md_path: str | Path) -> None:
    try:
        cache_key = str(Path(md_path).expanduser().resolve())
        _CONVERSION_QUALITY_CACHE.pop(cache_key, None)
    except Exception:
        return


def _path_is_within(path_obj: Path, roots: list[Path]) -> bool:
    try:
        path = Path(path_obj).expanduser().resolve(strict=False)
    except Exception:
        return False
    for root in roots:
        try:
            root_resolved = Path(root).expanduser().resolve(strict=False)
            path.relative_to(root_resolved)
            return True
        except Exception:
            continue
    return False


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        try:
            key = str(path.expanduser().resolve(strict=False)).lower()
        except Exception:
            key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _infer_pdf_from_md_source(md_path: Path, *, pdf_dir: Path, source_name: str = "") -> Path | None:
    stems = [
        _strip_known_source_ext(md_path.name),
        _strip_known_source_ext(md_path.parent.name),
        _strip_known_source_ext(source_name),
    ]
    candidates: list[Path] = []
    source_leaf = Path(str(source_name or "").replace("\\", "/")).name
    if source_leaf.lower().endswith(".pdf"):
        candidates.append(pdf_dir / source_leaf)
    for stem in stems:
        if stem:
            candidates.append(pdf_dir / f"{stem}.pdf")
    for candidate in _dedupe_paths(candidates):
        if not _path_is_within(candidate, [pdf_dir]):
            continue
        try:
            if _path_is_file(candidate):
                return candidate
        except Exception:
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def _resolve_quality_source(*, source_path: str, source_name: str = "") -> dict:
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    roots = [pdf_d, md_d]
    raw = str(source_path or "").strip()
    name_hint = str(source_name or "").strip()
    leaf = Path(raw.replace("\\", "/")).name if raw else ""
    stem = _strip_known_source_ext(leaf or name_hint)
    candidates: list[Path] = []

    if raw:
        raw_path = Path(raw).expanduser()
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(pdf_d / raw_path.name)
            candidates.append(md_d / raw_path.name)
            if stem:
                candidates.append(md_d / stem / f"{stem}.en.md")
                candidates.append(md_d / stem / f"{stem}.md")

    if leaf:
        candidates.append(pdf_d / leaf)
        candidates.append(md_d / leaf)
    if stem:
        candidates.append(pdf_d / f"{stem}.pdf")
        candidates.append(md_d / stem / f"{stem}.en.md")
        candidates.append(md_d / stem / f"{stem}.md")

    md_path: Path | None = None
    pdf_path: Path | None = None
    md_exists = False

    for candidate in _dedupe_paths(candidates):
        suffix = candidate.suffix.lower()
        if suffix == ".md":
            if not _path_is_within(candidate, [md_d]):
                continue
            try:
                if _path_is_file(candidate):
                    md_path = candidate
                    md_exists = True
                    pdf_path = _infer_pdf_from_md_source(candidate, pdf_dir=pdf_d, source_name=name_hint)
                    break
            except Exception:
                if candidate.exists() and candidate.is_file():
                    md_path = candidate
                    md_exists = True
                    pdf_path = _infer_pdf_from_md_source(candidate, pdf_dir=pdf_d, source_name=name_hint)
                    break
        if suffix == ".pdf":
            if not _path_is_within(candidate, roots):
                continue
            pdf_path = candidate
            _, md_main, exists = _resolve_md_output_paths(md_d, candidate)
            if exists:
                md_path = md_main
                md_exists = True
                break
            if md_path is None:
                md_path = md_main

    quality = _conversion_quality_summary(md_path) if (md_path is not None and md_exists) else None
    return {
        "source_path": raw,
        "source_name": name_hint or leaf,
        "pdf_path": str(pdf_path or ""),
        "md_path": str(md_path or ""),
        "md_exists": bool(md_exists),
        "conversion_quality": quality,
    }


def _compact_active_tasks(snap: dict) -> list[dict]:
    items: list[dict] = []
    for task in list((snap or {}).get("active_tasks") or []):
        if not isinstance(task, dict):
            continue
        items.append(
            {
                "task_id": str(task.get("_tid") or ""),
                "name": str(task.get("name") or ""),
                "pdf": str(task.get("pdf") or ""),
                "replace": bool(task.get("replace", False)),
                "cur_page_done": int(task.get("cur_page_done", 0) or 0),
                "cur_page_total": int(task.get("cur_page_total", 0) or 0),
                "cur_page_msg": str(task.get("cur_page_msg") or ""),
            }
        )
    return items


def _is_pdf_active_in_snapshot(*, snap: dict, pdf_path: Path, pdf_name: str) -> bool:
    pdf_key = _normalized_path_key(pdf_path)
    for task in _compact_active_tasks(snap):
        task_name = str(task.get("name") or "").strip()
        task_pdf = str(task.get("pdf") or "").strip()
        task_key = _normalized_path_key(task_pdf)
        if task_name and (task_name == pdf_name):
            return True
        if pdf_key and task_key and (task_key == pdf_key):
            return True
    if bool((snap or {}).get("running")):
        current_name = str((snap or {}).get("current") or "").strip()
        if current_name and (current_name == pdf_name):
            return True
    return False


def _build_task_maps_from_snapshot(snap: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    by_path: dict[str, dict] = {}
    by_name: dict[str, dict] = {}
    for task in _compact_active_tasks(snap):
        task_pdf = str(task.get("pdf") or "").strip()
        if not task_pdf:
            continue
        task_name = str(task.get("name") or Path(task_pdf).name).strip()
        info = {
            "queued": False,
            "running": True,
            "replace": bool(task.get("replace", False)),
            "queue_pos": 0,
            "task_id": str(task.get("task_id") or ""),
            "cur_page_done": int(task.get("cur_page_done", 0) or 0),
            "cur_page_total": int(task.get("cur_page_total", 0) or 0),
            "cur_page_msg": str(task.get("cur_page_msg") or ""),
        }
        key = _normalized_path_key(task_pdf)
        if key:
            _merge_task_map_entry(by_path, key, info)
        if task_name:
            _merge_task_map_entry(by_name, task_name, info)

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
            _merge_task_map_entry(by_path, key, info)
        if task_name:
            _merge_task_map_entry(by_name, task_name, info)

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
        _merge_task_map_entry(by_name, current_name, cur)
    return by_path, by_name


def _library_file_item(
    pdf: Path,
    *,
    md_root: Path,
    task_by_path: dict[str, dict],
    task_by_name: dict[str, dict],
    meta_rec: dict | None = None,
) -> dict:
    md_folder, md_main, md_exists = _resolve_md_output_paths(md_root, pdf)
    key = _normalized_path_key(pdf)
    info = task_by_path.get(key) if key else None
    if not isinstance(info, dict):
        info = task_by_name.get(pdf.name) if isinstance(task_by_name.get(pdf.name), dict) else {}
    queued = bool((info or {}).get("queued"))
    running = bool((info or {}).get("running"))
    replace_task = bool((info or {}).get("replace"))
    queue_pos = int((info or {}).get("queue_pos") or 0)
    cur_page_done = int((info or {}).get("cur_page_done") or 0)
    cur_page_total = int((info or {}).get("cur_page_total") or 0)
    cur_page_msg = str((info or {}).get("cur_page_msg") or "")
    task_state = "running" if running else ("queued" if queued else "idle")
    queued_or_running = bool(queued or running)
    reconverting = bool(replace_task and queued_or_running)
    category = "converted" if (md_exists and (not reconverting) and (not queued_or_running)) else "pending"
    conversion_quality = _conversion_quality_summary(md_main) if md_exists else None
    if task_state == "running":
        status = "running_reconvert" if replace_task else "running"
    elif task_state == "queued":
        status = "queued_reconvert" if replace_task else "queued"
    else:
        status = "converted" if category == "converted" else "pending"
    return {
        "name": pdf.name,
        "path": str(pdf),
        "sha1": str((meta_rec or {}).get("sha1") or ""),
        "md_exists": bool(md_exists),
        "md_path": str(md_main) if md_exists else "",
        "md_folder": str(md_folder),
        "conversion_quality": conversion_quality,
        "category": category,
        "task_state": task_state,
        "status": status,
        "replace_task": bool(replace_task),
        "queue_pos": int(queue_pos),
        "cur_page_done": int(cur_page_done),
        "cur_page_total": int(cur_page_total),
        "cur_page_msg": cur_page_msg,
        "paper_category": str((meta_rec or {}).get("paper_category") or ""),
        "reading_status": str((meta_rec or {}).get("reading_status") or ""),
        "note": str((meta_rec or {}).get("note") or ""),
        "user_tags": list((meta_rec or {}).get("user_tags") or []),
        "has_suggestions": bool((meta_rec or {}).get("has_suggestions")),
        "suggested_category": str((meta_rec or {}).get("suggested_category") or ""),
        "suggested_tags": list((meta_rec or {}).get("suggested_tags") or []),
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
    meta_by_path = _library_store().list_records_by_paths(view)
    items = [
        _library_file_item(
            pdf,
            md_root=md_dir,
            task_by_path=task_by_path,
            task_by_name=task_by_name,
            meta_rec=meta_by_path.get(str(pdf)),
        )
        for pdf in view
    ]

    pending = sum(1 for item in items if str(item.get("category") or "") == "pending")
    converted = sum(1 for item in items if str(item.get("category") or "") == "converted")
    queued = sum(1 for item in items if str(item.get("task_state") or "") == "queued")
    running = sum(1 for item in items if str(item.get("task_state") or "") == "running")
    reconverting = sum(1 for item in items if bool(item.get("replace_task")) and str(item.get("task_state") or "") in {"queued", "running"})
    quality_review = sum(1 for item in items if bool(((item.get("conversion_quality") or {}) if isinstance(item.get("conversion_quality"), dict) else {}).get("has_review_issue")))
    quality_ready = sum(1 for item in items if str(((item.get("conversion_quality") or {}) if isinstance(item.get("conversion_quality"), dict) else {}).get("status") or "") == "good")

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
            "quality_review": int(quality_review),
            "quality_ready": int(quality_ready),
        },
        "truncated": bool(limit > 0 and len(pdfs_all) > len(view)),
        "scope": "all" if limit <= 0 else str(limit),
        "queue": {
            "running": bool(snap.get("running", False)) or bool(list(snap.get("active_tasks") or [])),
            "active_count": int(snap.get("active_count", len(list(snap.get("active_tasks") or []))) or 0),
            "current": str(snap.get("current", "")),
            "done": int(snap.get("done", 0) or 0),
            "total": int(snap.get("total", 0) or 0),
            "active_tasks": _compact_active_tasks(snap),
        },
    }


def _quality_status_rank(status: str) -> int:
    s = str(status or "").strip().lower()
    if s == "error":
        return 3
    if s == "warning":
        return 2
    if s == "good":
        return 1
    return 0


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_mtime(path: Path | None) -> float:
    if path is None:
        return 0.0
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _latest_artifact_file(root: Path, filename: str) -> Path | None:
    try:
        if not root.exists():
            return None
        candidates = [p for p in root.rglob(filename) if p.is_file()]
    except Exception:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda p: (_safe_mtime(p), str(p)), reverse=True)
    return candidates[0]


def _read_json_artifact(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_jsonl_artifact(path: Path | None, *, limit: int = 10000) -> list[dict]:
    if path is None:
        return []
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if len(rows) >= limit:
                    break
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        return []
    return rows


def _latest_research_qa_artifacts() -> tuple[Path | None, Path | None]:
    summary_path = _latest_artifact_file(_RESEARCH_QA_EVAL_ROOT, "summary.json")
    raw_path: Path | None = None
    if summary_path is not None:
        sibling_raw = summary_path.parent / "raw_results.jsonl"
        if sibling_raw.is_file():
            raw_path = sibling_raw
    if raw_path is None:
        raw_path = _latest_artifact_file(_RESEARCH_QA_EVAL_ROOT, "raw_results.jsonl")
    return summary_path, raw_path


def _quality_failure_name(item) -> str:
    if isinstance(item, dict):
        for key in ("name", "code", "check"):
            value = str(item.get(key) or "").strip()
            if value:
                return value
        detail = item.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip().split(":", 1)[0]
    value = str(item or "").strip()
    return value.split(":", 1)[0] if value else "unknown"


def _quality_failure_detail(item) -> str:
    if not isinstance(item, dict):
        return str(item or "").strip()[:240]
    detail = item.get("detail")
    if detail in (None, ""):
        return ""
    if isinstance(detail, (dict, list)):
        try:
            return json.dumps(detail, ensure_ascii=False)[:240]
        except Exception:
            return str(detail)[:240]
    return str(detail or "").strip()[:240]


def _quality_failure_domain(name: str) -> str:
    key = str(name or "").strip().lower()
    if key in {"citation_card_quality", "citation_shelf_quality", "refs_card_copy_quality", "system_b_audit"}:
        return "citation_cards"
    if key.startswith("citation_") or key.startswith("system_b_") or key.startswith("ref_card_") or key.startswith("shelf_"):
        return "citation_cards"
    return "research_qa"


def _list_strings(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _expected_doc_ids(expected: dict) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in ("requiredRefDocIds", "requiredCitationDocIds", "requiredSystemBDocIds"):
        for doc_id in _list_strings((expected or {}).get(key)):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(doc_id)
    return out


def _compact_text(*values, limit: int = 0) -> str:
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        return text[:limit] if limit > 0 else text
    return ""


def _list_dict_items(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _research_qa_citation_details(row: dict) -> list[dict]:
    details: list[dict] = []

    def add_many(value) -> None:
        details.extend(_list_dict_items(value))

    message = row.get("assistant_message") if isinstance(row.get("assistant_message"), dict) else {}
    add_many((message or {}).get("cite_details"))
    add_many((message or {}).get("citation_details"))

    meta = (message or {}).get("meta") if isinstance((message or {}).get("meta"), dict) else {}
    contract = (meta or {}).get("paper_guide_contracts") if isinstance((meta or {}).get("paper_guide_contracts"), dict) else {}
    render_packet = (contract or {}).get("render_packet") if isinstance((contract or {}).get("render_packet"), dict) else {}
    add_many((render_packet or {}).get("cite_details"))
    add_many((render_packet or {}).get("citation_details"))

    final_payload = row.get("final_payload") if isinstance(row.get("final_payload"), dict) else {}
    add_many((final_payload or {}).get("cite_details"))
    add_many((final_payload or {}).get("citation_details"))

    seen: set[tuple[str, str, str, str, str]] = set()
    out: list[dict] = []
    for item in details:
        key = (
            str(item.get("num") or item.get("ref_num") or ""),
            str(item.get("anchor") or item.get("anchor_id") or ""),
            str(item.get("source_path") or ""),
            str(item.get("title") or item.get("source_name") or ""),
            "b" if bool(item.get("is_inpaper")) else "a",
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _research_qa_ref_hits(row: dict) -> list[dict]:
    refs_payload = row.get("refs_payload") if isinstance(row.get("refs_payload"), dict) else {}
    if not refs_payload:
        return []

    packs: list[dict | list] = []
    user_msg_id = row.get("user_msg_id")
    if user_msg_id not in (None, ""):
        pack = refs_payload.get(str(user_msg_id))
        if isinstance(pack, (dict, list)):
            packs.append(pack)
    if isinstance(refs_payload.get("hits"), list):
        packs.append(refs_payload)
    for value in refs_payload.values():
        if isinstance(value, dict) and isinstance(value.get("hits"), list):
            packs.append(value)
        elif isinstance(value, list):
            packs.append(value)

    hits: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for pack in packs:
        raw_hits = pack.get("hits") if isinstance(pack, dict) else pack
        for item in _list_dict_items(raw_hits):
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            ui_meta = item.get("ui_meta") if isinstance(item.get("ui_meta"), dict) else {}
            key = (
                str(item.get("id") or item.get("chunk_id") or ""),
                str((ui_meta or {}).get("source_path") or (meta or {}).get("source_path") or ""),
                str(item.get("text") or "")[:80],
            )
            if key in seen:
                continue
            seen.add(key)
            hits.append(item)
    return hits


def _research_qa_citation_diagnostics(row: dict, *, limit: int = 8) -> list[dict]:
    out: list[dict] = []
    for item in _research_qa_citation_details(row)[:limit]:
        is_system_b = bool(item.get("is_inpaper")) or str(item.get("route") or "").strip().lower() == "system_b"
        out.append(
            {
                "route": "system_b" if is_system_b else "system_a",
                "num": _safe_int(item.get("num") or item.get("ref_num"), 0),
                "anchor": _compact_text(item.get("anchor"), item.get("anchor_id"), item.get("block_id"), limit=120),
                "title": _compact_text(item.get("title"), item.get("source_name"), limit=180),
                "source_name": _compact_text(item.get("source_name"), limit=180),
                "source_path": _compact_text(item.get("source_path"), limit=260),
                "heading_path": _compact_text(item.get("heading_path"), item.get("location_label"), limit=180),
                "evidence_quote": _compact_text(item.get("evidence_quote"), item.get("citation_context"), limit=260),
                "answer_claim": _compact_text(item.get("answer_claim"), limit=220),
                "support_relation": _compact_text(item.get("support_relation"), item.get("user_question_relation"), limit=180),
                "trace": _compact_text(item.get("citation_context_source"), item.get("mapping_source"), item.get("anchor_kind"), limit=120),
            }
        )
    return out


def _research_qa_ref_diagnostics(row: dict, *, limit: int = 8) -> list[dict]:
    out: list[dict] = []
    for hit in _research_qa_ref_hits(row)[:limit]:
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        citation_meta = (ui_meta or {}).get("citation_meta") if isinstance((ui_meta or {}).get("citation_meta"), dict) else {}
        score = (ui_meta or {}).get("score")
        if score in (None, ""):
            score = hit.get("score")
        out.append(
            {
                "title": _compact_text((ui_meta or {}).get("display_name"), (citation_meta or {}).get("title"), (meta or {}).get("source_name"), limit=180),
                "source_name": _compact_text((ui_meta or {}).get("source_name"), (meta or {}).get("source_name"), limit=180),
                "source_path": _compact_text((ui_meta or {}).get("source_path"), (citation_meta or {}).get("source_path"), (meta or {}).get("source_path"), limit=260),
                "heading_path": _compact_text((ui_meta or {}).get("heading_path"), (meta or {}).get("heading_path"), limit=180),
                "score": round(_safe_float(score, 0.0), 3),
                "summary_line": _compact_text((ui_meta or {}).get("summary_line"), limit=220),
                "why_line": _compact_text((ui_meta or {}).get("why_line"), limit=220),
                "polish_status": _compact_text((ui_meta or {}).get("polish_status"), limit=80),
                "ref_pack_state": _compact_text((meta or {}).get("ref_pack_state"), limit=80),
                "evidence_quote": _compact_text(hit.get("text"), limit=260),
            }
        )
    return out


def _research_qa_source_diagnostics(citations: list[dict], refs: list[dict], *, limit: int = 8) -> list[dict]:
    by_key: dict[str, dict] = {}

    def add_source(item: dict, role: str) -> None:
        source_path = _compact_text(item.get("source_path"), limit=260)
        source_name = _compact_text(item.get("source_name"), item.get("title"), limit=180)
        title = _compact_text(item.get("title"), item.get("source_name"), limit=180)
        key = (source_path or source_name or title).lower()
        if not key:
            return
        cur = by_key.get(key)
        if cur is None:
            cur = {
                "source_path": source_path,
                "source_name": source_name,
                "title": title,
                "roles": [],
            }
            by_key[key] = cur
        if role not in cur["roles"]:
            cur["roles"].append(role)

    for item in citations:
        route = "system_b" if str(item.get("route") or "") == "system_b" else "system_a"
        add_source(item, f"citation:{route}")
    for item in refs:
        add_source(item, "reference_basket")

    out: list[dict] = []
    for item in list(by_key.values())[:limit]:
        resolved = _resolve_quality_source(
            source_path=str(item.get("source_path") or ""),
            source_name=str(item.get("source_name") or item.get("title") or ""),
        )
        quality = resolved.get("conversion_quality") if isinstance(resolved.get("conversion_quality"), dict) else {}
        issues = []
        for issue in list((quality or {}).get("issues") or [])[:4]:
            if not isinstance(issue, dict):
                continue
            issues.append(
                {
                    "code": _compact_text(issue.get("code"), limit=80),
                    "label": _compact_text(issue.get("label"), issue.get("code"), limit=120),
                    "severity": _compact_text(issue.get("severity"), limit=40) or "warning",
                    "count": _safe_int(issue.get("count"), 0),
                }
            )
        pdf_path = _compact_text((resolved or {}).get("pdf_path"), limit=260)
        out.append(
            {
                "source_path": str(item.get("source_path") or ""),
                "source_name": str(item.get("source_name") or ""),
                "title": str(item.get("title") or ""),
                "roles": list(item.get("roles") or []),
                "pdf_path": pdf_path,
                "md_path": _compact_text((resolved or {}).get("md_path"), limit=260),
                "md_exists": bool((resolved or {}).get("md_exists")),
                "repairable": bool(pdf_path),
                "needs_repair": bool((quality or {}).get("has_review_issue")),
                "quality_status": _compact_text((quality or {}).get("status"), limit=40) or "unknown",
                "quality_score": _safe_int((quality or {}).get("score"), 0),
                "quality_summary": _compact_text((quality or {}).get("summary"), limit=220),
                "quality_issues": issues,
            }
        )
    return out


def _research_qa_root_causes(
    *,
    failures: list[dict],
    missing_expected_doc_ids: list[str],
    citation_count: int,
    system_b_count: int,
    ref_hit_count: int,
    source_diagnostics: list[dict],
    rerun_status: dict | None = None,
) -> list[dict]:
    names = {str(item.get("name") or "").strip().lower() for item in failures}
    out: list[dict] = []
    seen: set[str] = set()

    def add(code: str, label: str, *, severity: str = "warning", detail: str = "", action: str = "inspect") -> None:
        if code in seen:
            return
        seen.add(code)
        out.append(
            {
                "code": code,
                "label": label,
                "severity": severity,
                "detail": detail[:240],
                "action": action,
            }
        )

    bad_sources = [
        item for item in source_diagnostics
        if bool(item.get("needs_repair")) or str(item.get("quality_status") or "").lower() in {"error", "warning"}
    ]
    if bad_sources:
        worst = "error" if any(str(item.get("quality_status") or "").lower() == "error" for item in bad_sources) else "warning"
        add(
            "source_conversion_quality",
            "Source conversion needs repair",
            severity=worst,
            detail=f"{len(bad_sources)} related sources have conversion-quality issues.",
            action="repair_sources",
        )

    if missing_expected_doc_ids or "refs_include_required_docs" in names:
        add(
            "retrieval_missing_expected_docs",
            "Retrieval missed required documents",
            severity="error",
            detail="Missing expected docs: " + (" / ".join(missing_expected_doc_ids) if missing_expected_doc_ids else "see QA failure detail"),
            action="rebuild_index",
        )

    if citation_count <= 0 or "citations_include_required_docs" in names or "citation_include_required_docs" in names:
        add(
            "citation_missing_expected_docs",
            "Answer citations missed required evidence",
            severity="error",
            detail=f"Captured citations: {citation_count}.",
            action="inspect_replay",
        )

    if any(name in names for name in {"citation_card_quality", "citation_shelf_quality", "refs_card_copy_quality"}):
        add(
            "citation_card_quality",
            "Citation card or basket copy is weak",
            severity="error",
            detail="Card title, source, summary, why-line, or shelf rendering failed acceptance.",
            action="inspect_cards",
        )

    if any(name == "system_b_audit" or name.startswith("system_b_") for name in names):
        add(
            "system_b_mapping",
            "System B in-paper reference mapping needs review",
            severity="error",
            detail=f"System B citations captured: {system_b_count}.",
            action="inspect_system_b",
        )

    if ref_hit_count <= 0:
        add(
            "empty_reference_basket",
            "Reference basket returned no usable hits",
            severity="error",
            detail="No retrieval refs were captured for this QA case.",
            action="rebuild_index",
        )

    latest_rerun_status = str((rerun_status or {}).get("last_status") or "").lower()
    latest_rerun_error_kind = str((rerun_status or {}).get("error_kind") or "").lower()
    latest_rerun_error_detail = _compact_text((rerun_status or {}).get("error_detail"), limit=180)
    if latest_rerun_status == "error":
        if latest_rerun_error_kind == "connection":
            add(
                "research_qa_service_unreachable",
                "Research QA service is unreachable",
                severity="warning",
                detail=latest_rerun_error_detail or "The last rerun could not connect to the QA API service.",
                action="retry_rerun",
            )
        elif latest_rerun_error_kind == "timeout":
            add(
                "research_qa_rerun_timeout",
                "Research QA rerun timed out",
                severity="warning",
                detail=latest_rerun_error_detail or "The last rerun exceeded the configured timeout.",
                action="retry_rerun",
            )
        else:
            add(
                "research_qa_rerun_error",
                "Research QA rerun errored",
                severity="warning",
                detail=latest_rerun_error_detail or "The last rerun ended before quality checks completed.",
                action="inspect_replay",
            )

    if not out:
        add(
            "answer_quality",
            "Answer quality regression needs replay",
            severity="warning",
            detail="Open replay and compare answer claims against retrieved evidence.",
            action="inspect_replay",
        )

    out.sort(key=lambda item: (-_quality_status_rank(str(item.get("severity") or "")), str(item.get("code") or "")))
    return out[:5]


def _research_qa_repair_plan_steps(
    *,
    cause_codes: set[str],
    cause_actions: set[str],
    needs_repair_sources: list[dict],
    missing_expected_doc_ids: list[str],
) -> list[dict]:
    steps: list[dict] = []
    if needs_repair_sources:
        steps.append(
            {
                "kind": "repair_sources",
                "label": "Repair source conversions",
                "source_count": len(needs_repair_sources),
            }
        )
    if "citation_card_quality" in cause_codes or "system_b_mapping" in cause_codes:
        steps.append(
            {
                "kind": "repair_shelf_metadata",
                "label": "Repair citation and shelf metadata",
            }
        )
    if "rebuild_index" in cause_actions or "retrieval_missing_expected_docs" in cause_codes or "empty_reference_basket" in cause_codes or missing_expected_doc_ids:
        steps.append(
            {
                "kind": "rebuild_index",
                "label": "Rebuild retrieval index",
            }
        )
    if steps or "retry_rerun" in cause_actions:
        steps.append(
            {
                "kind": "rerun_case",
                "label": "Rerun QA acceptance",
            }
        )
    return steps


def _research_qa_repair_actions(
    *,
    root_causes: list[dict],
    source_diagnostics: list[dict],
    missing_expected_doc_ids: list[str],
) -> list[dict]:
    cause_codes = {str(item.get("code") or "") for item in root_causes}
    cause_actions = {str(item.get("action") or "") for item in root_causes}
    repairable_sources = [item for item in source_diagnostics if bool(item.get("repairable"))]
    needs_repair_sources = [item for item in repairable_sources if bool(item.get("needs_repair")) or str(item.get("quality_status") or "").lower() in {"error", "warning"}]
    plan_steps = _research_qa_repair_plan_steps(
        cause_codes=cause_codes,
        cause_actions=cause_actions,
        needs_repair_sources=needs_repair_sources,
        missing_expected_doc_ids=missing_expected_doc_ids,
    )
    actions: list[dict] = []
    if plan_steps:
        step_labels = [str(item.get("label") or item.get("kind") or "").strip() for item in plan_steps]
        actions.append(
            {
                "id": "apply_repair_plan",
                "kind": "apply_repair_plan",
                "label": "Fix from source",
                "severity": "error" if any(str(item.get("severity") or "").lower() == "error" for item in root_causes) else "warning",
                "enabled": True,
                "source_count": len(needs_repair_sources),
                "detail": "Run the diagnostic repair plan: " + " -> ".join([item for item in step_labels if item]),
                "steps": plan_steps,
                "acceptance": "The case is rerun after repairs so the quality center can verify the fix.",
            }
        )
    actions.append(
        {
            "id": "open_replay",
            "kind": "open_replay",
            "label": "Open replay",
            "severity": "warning",
            "enabled": True,
            "detail": "Inspect the failed answer, refs, and citation cards.",
        }
    )
    actions.append(
        {
            "id": "rerun_case",
            "kind": "rerun_case",
            "label": "Rerun case",
            "severity": "error" if missing_expected_doc_ids else "warning",
            "enabled": True,
            "detail": "Run this QA case again after repair or index refresh.",
        }
    )

    if needs_repair_sources:
        actions.append(
            {
                "id": "repair_sources",
                "kind": "repair_sources",
                "label": "Repair sources",
                "severity": "error",
                "enabled": True,
                "source_count": len(needs_repair_sources),
                "detail": f"Reconvert {len(needs_repair_sources)} related sources with conversion-quality issues.",
            }
        )

    if "rebuild_index" in cause_actions or "retrieval_missing_expected_docs" in cause_codes or missing_expected_doc_ids:
        actions.append(
            {
                "id": "rebuild_index",
                "kind": "rebuild_index",
                "label": "Rebuild index",
                "severity": "error",
                "enabled": True,
                "detail": "Refresh chunks, BM25 index, and reference sync after source repair.",
            }
        )

    if "citation_card_quality" in cause_codes or "system_b_mapping" in cause_codes:
        actions.append(
            {
                "id": "open_raw",
                "kind": "open_artifact",
                "target": "raw",
                "label": "Open raw QA",
                "severity": "warning",
                "enabled": True,
                "detail": "Inspect raw cite_details, refs payload, and quality checks.",
            }
        )

    actions.append(
        {
            "id": "open_report",
            "kind": "open_artifact",
            "target": "report",
            "label": "Open report",
            "severity": "warning",
            "enabled": True,
            "detail": "Open the latest QA report.",
        }
    )
    return actions[:6]


def _research_qa_rerun_history_path() -> Path:
    return _RESEARCH_QA_EVAL_ROOT / "rerun_history.jsonl"


def _quality_action_history_path() -> Path:
    return _RESEARCH_QA_EVAL_ROOT / "action_history.jsonl"


def _research_qa_rerun_history_rows(*, limit: int = 1000) -> list[dict]:
    rows = _read_jsonl_artifact(_research_qa_rerun_history_path(), limit=limit)
    rows.sort(key=lambda item: (_safe_int(item.get("finished_at"), 0), _safe_int(item.get("started_at"), 0)), reverse=True)
    return rows


def _research_qa_rerun_history_by_case(rows: list[dict] | None = None) -> dict[str, list[dict]]:
    by_case: dict[str, list[dict]] = {}
    for row in list(rows or _research_qa_rerun_history_rows()):
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            continue
        by_case.setdefault(case_id, []).append(row)
    for items in by_case.values():
        items.sort(key=lambda item: (_safe_int(item.get("finished_at"), 0), _safe_int(item.get("started_at"), 0)), reverse=True)
    return by_case


def _research_qa_rerun_case_status(case_id: str, rows: list[dict]) -> dict:
    case_rows = [
        item for item in list(rows or [])
        if str(item.get("case_id") or "").strip() == str(case_id or "").strip()
    ]
    case_rows.sort(key=lambda item: (_safe_int(item.get("finished_at"), 0), _safe_int(item.get("started_at"), 0)), reverse=True)
    if not case_rows:
        return {
            "available": False,
            "run_count": 0,
            "last_status": "",
            "last_quality_ok": False,
            "last_finished_at": 0,
            "last_latency_ms": 0,
            "last_passed_at": 0,
            "consecutive_failed": 0,
            "failure_names": [],
            "report_path": "",
            "raw_path": "",
            "error_kind": "",
            "error_detail": "",
        }
    last = case_rows[0]
    consecutive_failed = 0
    last_passed_at = 0
    for row in case_rows:
        ok = bool(row.get("quality_ok")) or str(row.get("status") or "").lower() == "passed"
        if ok:
            if not last_passed_at:
                last_passed_at = _safe_int(row.get("finished_at"), 0)
            if consecutive_failed == 0:
                break
        elif not last_passed_at:
            consecutive_failed += 1
    failures = [
        _quality_failure_name(item)
        for item in list(last.get("failures") or [])
        if str(_quality_failure_name(item) or "").strip()
    ]
    error_kind = str(last.get("error_kind") or "")
    error_detail = _compact_text(last.get("error_detail"), limit=240)
    if str(last.get("status") or "").lower() == "error" and not error_kind:
        error_kind, error_detail = _research_qa_transient_error(
            "\n".join([str(last.get("stderr_tail") or ""), str(last.get("stdout_tail") or "")])
        )
    return {
        "available": True,
        "run_count": len(case_rows),
        "last_status": str(last.get("status") or "").strip(),
        "last_quality_ok": bool(last.get("quality_ok")),
        "last_finished_at": _safe_int(last.get("finished_at"), 0),
        "last_latency_ms": _safe_int(last.get("latency_ms"), 0),
        "last_passed_at": int(last_passed_at),
        "consecutive_failed": int(consecutive_failed),
        "failure_names": failures[:6],
        "report_path": str(last.get("report_path") or ""),
        "raw_path": str(last.get("raw_path") or ""),
        "error_kind": error_kind,
        "error_detail": error_detail,
    }


def _research_qa_rerun_history_summary(rows: list[dict] | None = None) -> dict:
    items = list(rows or _research_qa_rerun_history_rows())
    if not items:
        return {
            "available": False,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "error": 0,
            "case_count": 0,
            "latest_finished_at": 0,
            "latest_status": "",
            "top_failures": [],
        }
    passed = sum(1 for item in items if bool(item.get("quality_ok")) or str(item.get("status") or "").lower() == "passed")
    error = sum(1 for item in items if str(item.get("status") or "").lower() == "error")
    failed = sum(1 for item in items if str(item.get("status") or "").lower() == "failed")
    failure_counter: Counter = Counter()
    for item in items:
        for failure in list(item.get("failures") or []):
            failure_counter[_quality_failure_name(failure)] += 1
    latest = items[0]
    return {
        "available": True,
        "total": len(items),
        "passed": int(passed),
        "failed": int(failed),
        "error": int(error),
        "case_count": len({str(item.get("case_id") or "") for item in items if str(item.get("case_id") or "").strip()}),
        "latest_finished_at": _safe_int(latest.get("finished_at"), 0),
        "latest_status": str(latest.get("status") or "").strip(),
        "top_failures": _counter_items(failure_counter),
    }


def _append_research_qa_rerun_history(result: dict) -> None:
    try:
        path = _research_qa_rerun_history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "case_id": str(result.get("case_id") or ""),
            "status": str(result.get("status") or ""),
            "quality_ok": bool(result.get("quality_ok")),
            "returncode": _safe_int(result.get("returncode"), 0),
            "failures": list(result.get("failures") or [])[:8],
            "output_dir": str(result.get("output_dir") or ""),
            "report_path": str(result.get("report_path") or ""),
            "raw_path": str(result.get("raw_path") or ""),
            "error_kind": str(result.get("error_kind") or ""),
            "error_detail": _compact_text(result.get("error_detail"), limit=240),
            "stdout_tail": _tail_text(str(result.get("stdout_tail") or ""), limit=800),
            "stderr_tail": _tail_text(str(result.get("stderr_tail") or ""), limit=800),
            "started_at": _safe_int(result.get("started_at"), 0),
            "finished_at": _safe_int(result.get("finished_at"), 0),
            "latency_ms": _safe_int(result.get("latency_ms"), 0),
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        return


def _compact_json_value(value, *, limit: int = 40):
    if isinstance(value, dict):
        out = {}
        for key, item in list(value.items())[:limit]:
            clean_key = _compact_text(key, limit=80)
            if not clean_key:
                continue
            if isinstance(item, (str, int, float, bool)) or item is None:
                out[clean_key] = _compact_text(item, limit=240) if isinstance(item, str) else item
            elif isinstance(item, list):
                out[clean_key] = [_compact_json_value(v, limit=limit) for v in item[:12]]
            elif isinstance(item, dict):
                out[clean_key] = _compact_json_value(item, limit=limit)
            else:
                out[clean_key] = _compact_text(item, limit=240)
        return out
    if isinstance(value, list):
        return [_compact_json_value(item, limit=limit) for item in value[:limit]]
    if isinstance(value, str):
        return _compact_text(value, limit=500)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _compact_text(value, limit=240)


def _quality_action_history_rows(*, limit: int = 40) -> list[dict]:
    rows = _read_jsonl_artifact(_quality_action_history_path(), limit=1000)
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        stage_key = str(row.get("stage_key") or "").strip()
        if not stage_key:
            continue
        out.append(
            {
                "id": str(row.get("id") or ""),
                "stage_key": stage_key,
                "stage_label": str(row.get("stage_label") or stage_key),
                "action": str(row.get("action") or ""),
                "status": str(row.get("status") or "info"),
                "summary": _compact_text(row.get("summary"), limit=500),
                "detail": _compact_text(row.get("detail"), limit=500),
                "target_ids": _list_strings(row.get("target_ids"))[:12],
                "metrics": row.get("metrics") if isinstance(row.get("metrics"), dict) else {},
                "before": row.get("before") if isinstance(row.get("before"), dict) else {},
                "after": row.get("after") if isinstance(row.get("after"), dict) else {},
                "delta": row.get("delta") if isinstance(row.get("delta"), dict) else {},
                "improved": row.get("improved") if isinstance(row.get("improved"), bool) else None,
                "verification": row.get("verification") if isinstance(row.get("verification"), dict) else {},
                "created_at": _safe_int(row.get("created_at"), 0),
            }
        )
    out.sort(key=lambda item: (_safe_int(item.get("created_at"), 0), str(item.get("id") or "")), reverse=True)
    return out[: max(0, min(200, int(limit)))]


def _append_quality_action_history(record: dict) -> dict:
    now = int(time.time())
    stage_key = _compact_text(record.get("stage_key"), limit=80).lower()
    stage_label = _compact_text(record.get("stage_label"), stage_key, limit=120)
    action = _compact_text(record.get("action"), limit=120)
    status = _compact_text(record.get("status"), limit=40).lower() or "info"
    if status not in {"success", "warning", "error", "info", "good"}:
        status = "info"
    row = {
        "id": _compact_text(record.get("id"), limit=80) or uuid.uuid4().hex,
        "stage_key": stage_key,
        "stage_label": stage_label or stage_key,
        "action": action,
        "status": status,
        "summary": _compact_text(record.get("summary"), limit=500),
        "detail": _compact_text(record.get("detail"), limit=500),
        "target_ids": _list_strings(record.get("target_ids"))[:12],
        "metrics": _compact_json_value(record.get("metrics") if isinstance(record.get("metrics"), dict) else {}, limit=20),
        "before": _compact_json_value(record.get("before") if isinstance(record.get("before"), dict) else {}, limit=20),
        "after": _compact_json_value(record.get("after") if isinstance(record.get("after"), dict) else {}, limit=20),
        "delta": _compact_json_value(record.get("delta") if isinstance(record.get("delta"), dict) else {}, limit=20),
        "improved": record.get("improved") if isinstance(record.get("improved"), bool) else None,
        "verification": _compact_json_value(record.get("verification") if isinstance(record.get("verification"), dict) else {}, limit=20),
        "created_at": _safe_int(record.get("created_at"), now) or now,
    }
    if not row["stage_key"]:
        raise HTTPException(400, "stage_key is required")
    if not row["summary"]:
        raise HTTPException(400, "summary is required")
    try:
        path = _quality_action_history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"failed to write quality action history: {exc}") from exc
    return row


def _counter_items(counter: Counter, *, limit: int = 6) -> list[dict]:
    return [
        {"name": str(name), "count": int(count)}
        for name, count in counter.most_common(limit)
        if str(name or "").strip()
    ]


def _quality_status_from_counts(total: int, failed: int) -> str:
    if total <= 0:
        return "unknown"
    if failed > 0:
        return "error"
    return "good"


def _latest_research_qa_quality_summary() -> dict:
    summary_path, raw_path = _latest_research_qa_artifacts()
    if summary_path is None and raw_path is None:
        return {
            "available": False,
            "status": "unknown",
            "summary": {"total": 0, "passed": 0, "failed": 0},
            "top_failures": [],
            "latest_path": "",
            "report_path": "",
            "updated_at": 0,
        }

    summary = _read_json_artifact(summary_path)
    rows = _read_jsonl_artifact(raw_path)
    row_total = len(rows)
    row_failed = sum(
        1
        for row in rows
        if isinstance(row.get("quality"), dict) and not bool((row.get("quality") or {}).get("ok"))
    )
    total = _safe_int(summary.get("total"), row_total)
    passed = _safe_int(summary.get("passed"), max(0, total - row_failed))
    failed = _safe_int(summary.get("failed"), row_failed)

    failures: Counter = Counter()
    for row in rows:
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        for item in list((quality or {}).get("failures") or []):
            failures[_quality_failure_name(item)] += 1

    anchor = summary_path or raw_path
    report_path = anchor.parent / "report.md" if anchor is not None else None
    return {
        "available": True,
        "status": _quality_status_from_counts(total, failed),
        "summary": {
            "total": int(total),
            "passed": int(passed),
            "failed": int(failed),
        },
        "top_failures": _counter_items(failures),
        "latest_path": str(anchor.parent) if anchor is not None else "",
        "report_path": str(report_path) if report_path is not None and report_path.is_file() else "",
        "updated_at": int(max(_safe_mtime(summary_path), _safe_mtime(raw_path))),
    }


def _latest_research_qa_failure_cases(*, limit: int = 12, rerun_history: list[dict] | None = None) -> list[dict]:
    _, raw_path = _latest_research_qa_artifacts()
    rows = _read_jsonl_artifact(raw_path)
    rerun_rows = list(rerun_history or _research_qa_rerun_history_rows())
    cases: list[dict] = []
    for row in rows:
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        if bool(quality.get("ok")):
            continue
        raw_failures = [item for item in list(quality.get("failures") or []) if isinstance(item, dict)]
        failures: list[dict] = []
        for failure in raw_failures[:8]:
            name = _quality_failure_name(failure)
            failures.append(
                {
                    "name": name,
                    "domain": _quality_failure_domain(name),
                    "detail": _quality_failure_detail(failure),
                }
            )
        expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
        ref_doc_ids = _list_strings(quality.get("ref_doc_ids"))
        citation_doc_ids = _list_strings(quality.get("citation_doc_ids"))
        expected_doc_ids = _expected_doc_ids(expected)
        observed_doc_ids = set([*ref_doc_ids, *citation_doc_ids])
        missing_expected_doc_ids = [doc_id for doc_id in expected_doc_ids if doc_id not in observed_doc_ids]
        citation_diagnostics = _research_qa_citation_diagnostics(row)
        ref_diagnostics = _research_qa_ref_diagnostics(row)
        citation_routes = Counter(str(item.get("route") or "system_a") for item in citation_diagnostics)
        if not citation_diagnostics:
            quality_system_b_count = _safe_int(quality.get("system_b_count"), 0)
            quality_citation_count = _safe_int(quality.get("citation_count"), 0)
            citation_routes["system_b"] = quality_system_b_count
            citation_routes["system_a"] = max(0, quality_citation_count - quality_system_b_count)
        citation_count = _safe_int(quality.get("citation_count"), 0)
        system_b_count = _safe_int(quality.get("system_b_count"), 0)
        ref_hit_count = _safe_int(quality.get("ref_hit_count"), 0)
        source_diagnostics = _research_qa_source_diagnostics(citation_diagnostics, ref_diagnostics)
        rerun_status = _research_qa_rerun_case_status(str(row.get("id") or "").strip(), rerun_rows)
        root_causes = _research_qa_root_causes(
            failures=failures,
            missing_expected_doc_ids=missing_expected_doc_ids,
            citation_count=citation_count,
            system_b_count=system_b_count,
            ref_hit_count=ref_hit_count,
            source_diagnostics=source_diagnostics,
            rerun_status=rerun_status,
        )
        repair_actions = _research_qa_repair_actions(
            root_causes=root_causes,
            source_diagnostics=source_diagnostics,
            missing_expected_doc_ids=missing_expected_doc_ids,
        )
        doc_ids = []
        seen_doc_ids: set[str] = set()
        for doc_id in [*expected_doc_ids, *ref_doc_ids, *citation_doc_ids]:
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            doc_ids.append(doc_id)
        cases.append(
            {
                "id": str(row.get("id") or "").strip(),
                "question": str(row.get("question") or "").strip(),
                "status": str(row.get("status") or "").strip(),
                "conv_id": str(row.get("conv_id") or "").strip(),
                "latency_ms": _safe_int(row.get("latency_ms"), 0),
                "failures": failures,
                "failure_names": [str(item.get("name") or "") for item in failures if str(item.get("name") or "").strip()],
                "expected_doc_ids": expected_doc_ids,
                "ref_doc_ids": ref_doc_ids,
                "citation_doc_ids": citation_doc_ids,
                "missing_expected_doc_ids": missing_expected_doc_ids,
                "doc_ids": doc_ids,
                "citation_count": citation_count,
                "system_b_count": system_b_count,
                "ref_hit_count": ref_hit_count,
                "diagnostic_summary": {
                    "citation_routes": {
                        "system_a": int(citation_routes.get("system_a", 0)),
                        "system_b": int(citation_routes.get("system_b", 0)),
                    },
                    "missing_expected_doc_count": len(missing_expected_doc_ids),
                    "citation_diagnostic_count": len(citation_diagnostics),
                    "ref_diagnostic_count": len(ref_diagnostics),
                },
                "citation_diagnostics": citation_diagnostics,
                "ref_diagnostics": ref_diagnostics,
                "source_diagnostics": source_diagnostics,
                "root_causes": root_causes,
                "repair_actions": repair_actions,
                "rerun_status": rerun_status,
                "answer_preview": str(quality.get("answer_preview") or "").strip()[:360],
            }
        )
        if len(cases) >= int(limit):
            break
    return cases


def _latest_citation_card_quality_summary() -> dict:
    _, raw_path = _latest_research_qa_artifacts()
    rows = _read_jsonl_artifact(raw_path)
    if raw_path is None or not rows:
        return {
            "available": False,
            "status": "unknown",
            "summary": {
                "tracked_checks": 0,
                "failed_checks": 0,
                "citation_card_failed": 0,
                "shelf_failed": 0,
                "ref_card_failed": 0,
                "system_b_failed": 0,
            },
            "top_failures": [],
            "latest_path": "",
            "updated_at": 0,
        }

    tracked = {
        "citation_card_quality": "citation_card_failed",
        "citation_shelf_quality": "shelf_failed",
        "refs_card_copy_quality": "ref_card_failed",
        "system_b_audit": "system_b_failed",
    }
    summary = {
        "tracked_checks": 0,
        "failed_checks": 0,
        "citation_card_failed": 0,
        "shelf_failed": 0,
        "ref_card_failed": 0,
        "system_b_failed": 0,
    }
    failures: Counter = Counter()
    for row in rows:
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        for check in list((quality or {}).get("checks") or []):
            if not isinstance(check, dict):
                continue
            name = str(check.get("name") or "").strip()
            bucket = tracked.get(name)
            if not bucket:
                continue
            summary["tracked_checks"] = int(summary["tracked_checks"]) + 1
            if bool(check.get("ok")):
                continue
            summary["failed_checks"] = int(summary["failed_checks"]) + 1
            summary[bucket] = int(summary[bucket]) + 1
            failures[name] += 1

    available = int(summary.get("tracked_checks") or 0) > 0
    return {
        "available": available,
        "status": _quality_status_from_counts(
            int(summary.get("tracked_checks") or 0),
            int(summary.get("failed_checks") or 0),
        ),
        "summary": summary,
        "top_failures": _counter_items(failures),
        "latest_path": str(raw_path.parent),
        "updated_at": int(_safe_mtime(raw_path)),
    }


def _quality_priority_actions(
    *,
    conversion_status: str,
    recommended: list[dict],
    research_qa: dict,
    citation_cards: dict,
) -> list[dict]:
    actions: list[dict] = []
    if recommended:
        actions.append(
            {
                "domain": "conversion",
                "severity": "error" if conversion_status == "error" else "warning",
                "label": "Repair conversion quality",
                "count": len(recommended),
                "detail": str((recommended[0] or {}).get("summary") or ""),
            }
        )

    qa_summary = research_qa.get("summary") if isinstance(research_qa.get("summary"), dict) else {}
    qa_failed = _safe_int(qa_summary.get("failed"), 0)
    if not bool(research_qa.get("available")):
        actions.append(
            {
                "domain": "research_qa",
                "severity": "warning",
                "label": "Run research QA regression",
                "count": 0,
                "detail": "No latest research QA artifact was found.",
            }
        )
    elif qa_failed > 0:
        actions.append(
            {
                "domain": "research_qa",
                "severity": "error",
                "label": "Fix failed research QA cases",
                "count": qa_failed,
                "detail": str((research_qa.get("top_failures") or [{}])[0].get("name") or ""),
            }
        )

    card_summary = citation_cards.get("summary") if isinstance(citation_cards.get("summary"), dict) else {}
    card_failed = _safe_int(card_summary.get("failed_checks"), 0)
    if bool(citation_cards.get("available")) and card_failed > 0:
        actions.append(
            {
                "domain": "citation_cards",
                "severity": "error" if str(citation_cards.get("status") or "") == "error" else "warning",
                "label": "Fix citation and card quality",
                "count": card_failed,
                "detail": str((citation_cards.get("top_failures") or [{}])[0].get("name") or ""),
            }
        )

    actions.sort(key=lambda item: (-_quality_status_rank(str(item.get("severity") or "")), -_safe_int(item.get("count"), 0), str(item.get("domain") or "")))
    return actions[:6]


def _quality_full_chain_stage(
    key: str,
    label: str,
    status: str,
    *,
    detail: str = "",
    action: str = "",
    count: int = 0,
    blocking: bool = False,
    metrics: dict | None = None,
) -> dict:
    clean_status = str(status or "").strip().lower() or "unknown"
    return {
        "key": str(key or "").strip(),
        "label": str(label or "").strip(),
        "status": clean_status,
        "detail": str(detail or "").strip(),
        "action": str(action or "").strip(),
        "count": max(0, _safe_int(count, 0)),
        "blocking": bool(blocking),
        "metrics": metrics if isinstance(metrics, dict) else {},
    }


def _quality_full_chain_root_causes(*, conversion_issues: list[dict], failure_cases: list[dict]) -> list[dict]:
    stats: dict[str, dict] = {}

    def add(code: str, label: str, domain: str, *, severity: str = "warning", count: int = 1) -> None:
        clean_code = str(code or label or "").strip()
        clean_label = str(label or code or "").strip()
        clean_domain = str(domain or "quality").strip()
        if not clean_code or not clean_label:
            return
        key = f"{clean_domain}:{clean_code.lower()}"
        cur = stats.get(key) or {
            "code": clean_code,
            "label": clean_label,
            "domain": clean_domain,
            "count": 0,
            "severity": str(severity or "warning").strip().lower() or "warning",
        }
        cur["count"] = _safe_int(cur.get("count"), 0) + max(1, _safe_int(count, 1))
        if _quality_status_rank(str(severity or "")) > _quality_status_rank(str(cur.get("severity") or "")):
            cur["severity"] = str(severity or "warning").strip().lower() or "warning"
        stats[key] = cur

    for issue in list(conversion_issues or []):
        if not isinstance(issue, dict):
            continue
        add(
            str(issue.get("code") or issue.get("label") or ""),
            str(issue.get("label") or issue.get("code") or ""),
            "conversion",
            severity=str(issue.get("severity") or "warning"),
            count=max(_safe_int(issue.get("papers"), 0), _safe_int(issue.get("count"), 0), 1),
        )

    for case in list(failure_cases or []):
        if not isinstance(case, dict):
            continue
        case_causes = [item for item in list(case.get("root_causes") or []) if isinstance(item, dict)]
        if case_causes:
            seen_in_case: set[str] = set()
            for cause in case_causes:
                code = str(cause.get("code") or cause.get("label") or "").strip()
                if not code or code in seen_in_case:
                    continue
                seen_in_case.add(code)
                add(
                    code,
                    str(cause.get("label") or code),
                    _quality_failure_domain(code) if code.startswith(("citation_", "system_b_", "shelf_", "ref_card_")) else "research_qa",
                    severity=str(cause.get("severity") or "warning"),
                )
        else:
            for failure in list(case.get("failures") or []):
                if not isinstance(failure, dict):
                    continue
                name = _quality_failure_name(failure)
                add(name, name, _quality_failure_domain(name), severity="error")

    out = list(stats.values())
    out.sort(
        key=lambda item: (
            -_quality_status_rank(str(item.get("severity") or "")),
            -_safe_int(item.get("count"), 0),
            str(item.get("domain") or ""),
            str(item.get("label") or "").lower(),
        )
    )
    return out[:8]


def _quality_full_chain_check(
    *,
    conversion_domain: dict,
    research_qa: dict,
    citation_cards: dict,
    failure_cases: list[dict],
    recommended: list[dict],
    rerun_summary: dict,
    priority_actions: list[dict],
    conversion_issues: list[dict],
) -> dict:
    conversion_summary = conversion_domain.get("summary") if isinstance(conversion_domain.get("summary"), dict) else {}
    conversion_status = str(conversion_domain.get("status") or "unknown").strip().lower() or "unknown"
    conversion_review = _safe_int(conversion_summary.get("review"), 0)
    conversion_unknown = _safe_int(conversion_summary.get("unknown"), 0)
    conversion_avg = _safe_int(conversion_summary.get("avg_score"), 0)
    conversion_count = len(recommended) if recommended else conversion_review
    conversion_detail = (
        f"{conversion_count} sources need conversion repair"
        if conversion_count > 0
        else (f"{conversion_unknown} converted sources are not assessed" if conversion_unknown > 0 else f"Q{conversion_avg} average conversion score")
    )

    qa_summary = research_qa.get("summary") if isinstance(research_qa.get("summary"), dict) else {}
    qa_available = bool(research_qa.get("available"))
    qa_total = _safe_int(qa_summary.get("total"), 0)
    qa_passed = _safe_int(qa_summary.get("passed"), 0)
    qa_failed = _safe_int(qa_summary.get("failed"), 0)
    qa_status = "warning"
    qa_detail = "No latest research QA artifact was found"
    if qa_available:
        qa_status = "error" if qa_failed > 0 else "good"
        qa_detail = f"{qa_failed} failed / {qa_total} QA cases" if qa_failed > 0 else f"{qa_passed}/{qa_total} QA cases passed"

    retrieval_count = 0
    for case in list(failure_cases or []):
        if not isinstance(case, dict):
            continue
        names = {str(item or "").strip().lower() for item in list(case.get("failure_names") or [])}
        cause_codes = {str(item.get("code") or "").strip().lower() for item in list(case.get("root_causes") or []) if isinstance(item, dict)}
        if case.get("missing_expected_doc_ids") or "refs_include_required_docs" in names or "retrieval_missing_expected_docs" in cause_codes:
            retrieval_count += 1
    retrieval_status = "error" if retrieval_count > 0 else ("warning" if not qa_available else "good")
    retrieval_detail = (
        f"{retrieval_count} QA cases missed required retrieval docs"
        if retrieval_count > 0
        else ("Waiting for QA regression evidence" if not qa_available else "Required-doc retrieval checks passed")
    )

    card_summary = citation_cards.get("summary") if isinstance(citation_cards.get("summary"), dict) else {}
    card_available = bool(citation_cards.get("available"))
    shelf_failed = _safe_int(card_summary.get("shelf_failed"), 0)
    card_failed = (
        _safe_int(card_summary.get("citation_card_failed"), 0)
        + _safe_int(card_summary.get("ref_card_failed"), 0)
        + _safe_int(card_summary.get("system_b_failed"), 0)
    )
    card_status = "error" if card_failed > 0 else ("warning" if not card_available else "good")
    card_detail = (
        f"{card_failed} citation/card checks failed"
        if card_failed > 0
        else ("Waiting for citation-card acceptance results" if not card_available else "Citation cards passed acceptance")
    )
    shelf_status = "error" if shelf_failed > 0 else ("warning" if not card_available else "good")
    shelf_detail = (
        f"{shelf_failed} literature basket checks failed"
        if shelf_failed > 0
        else ("Waiting for shelf acceptance results" if not card_available else "Literature basket checks passed")
    )

    rerun_available = bool((rerun_summary or {}).get("available"))
    rerun_failed = _safe_int((rerun_summary or {}).get("failed"), 0) + _safe_int((rerun_summary or {}).get("error"), 0)
    latest_rerun_status = str((rerun_summary or {}).get("latest_status") or "").strip().lower()
    has_failures = bool(failure_cases) or qa_failed > 0 or card_failed > 0 or shelf_failed > 0 or conversion_count > 0
    if not rerun_available and has_failures:
        repair_status = "warning"
        repair_detail = "No rerun verification history yet"
    elif rerun_failed > 0 or latest_rerun_status in {"failed", "error"}:
        repair_status = "warning"
        repair_detail = f"{rerun_failed} failed/error reruns; latest {latest_rerun_status or 'unknown'}"
    elif rerun_available:
        repair_status = "good"
        repair_detail = "Latest repair loop has passing evidence"
    else:
        repair_status = "good"
        repair_detail = "No failed QA cases require rerun verification"

    stages = [
        _quality_full_chain_stage(
            "conversion",
            "PDF conversion",
            conversion_status,
            detail=conversion_detail,
            action="repair_conversion" if conversion_count > 0 else "monitor_conversion",
            count=conversion_count,
            blocking=conversion_status == "error",
            metrics={"review": conversion_review, "unknown": conversion_unknown, "avg_score": conversion_avg},
        ),
        _quality_full_chain_stage(
            "research_qa",
            "Research QA",
            qa_status,
            detail=qa_detail,
            action="fix_failed_qa_cases" if qa_failed > 0 else ("run_research_qa" if not qa_available else "monitor_research_qa"),
            count=qa_failed,
            blocking=qa_status == "error",
            metrics={"total": qa_total, "passed": qa_passed, "failed": qa_failed},
        ),
        _quality_full_chain_stage(
            "retrieval",
            "Retrieval coverage",
            retrieval_status,
            detail=retrieval_detail,
            action="rebuild_index" if retrieval_count > 0 else "monitor_retrieval",
            count=retrieval_count,
            blocking=retrieval_status == "error",
        ),
        _quality_full_chain_stage(
            "citations",
            "Citation cards",
            card_status,
            detail=card_detail,
            action="repair_citation_cards" if card_failed > 0 else "monitor_citation_cards",
            count=card_failed,
            blocking=card_status == "error",
        ),
        _quality_full_chain_stage(
            "shelf",
            "Literature basket",
            shelf_status,
            detail=shelf_detail,
            action="repair_shelf_metadata" if shelf_failed > 0 else "monitor_literature_basket",
            count=shelf_failed,
            blocking=shelf_status == "error",
        ),
        _quality_full_chain_stage(
            "repair_loop",
            "Repair verification",
            repair_status,
            detail=repair_detail,
            action="rerun_failed_cases" if repair_status != "good" else "monitor_repair_loop",
            count=rerun_failed,
            blocking=False,
        ),
    ]

    score = 100
    for stage in stages:
        stage_status = str(stage.get("status") or "")
        if stage_status == "error":
            score -= 18
            if bool(stage.get("blocking")):
                score -= 4
        elif stage_status == "warning":
            score -= 8
        elif stage_status == "unknown":
            score -= 6
    score = max(0, min(100, score))
    worst = max(stages, key=lambda item: _quality_status_rank(str(item.get("status") or ""))) if stages else {}
    full_status = str(worst.get("status") or "unknown")
    blocking_count = sum(1 for stage in stages if bool(stage.get("blocking")))
    if full_status == "good":
        summary = "PDF conversion, QA regression, citations, and literature basket are passing current checks."
    elif blocking_count > 0:
        summary = f"{blocking_count} blocking stages need source-level repair before the app is release-ready."
    else:
        summary = "Full-chain checks need fresh regression evidence before quality can be trusted."

    return {
        "available": True,
        "status": full_status,
        "score": int(score),
        "summary": summary,
        "stages": stages,
        "root_causes": _quality_full_chain_root_causes(
            conversion_issues=conversion_issues,
            failure_cases=failure_cases,
        ),
        "next_actions": list(priority_actions or [])[:4],
    }


def _quality_feature_health_item(
    key: str,
    label: str,
    status: str,
    *,
    score: int = 0,
    summary: str = "",
    detail: str = "",
    action: str = "",
    target_stage: str = "",
    count: int = 0,
    blocking: bool = False,
    metrics: dict | None = None,
) -> dict:
    clean_status = str(status or "").strip().lower() or "unknown"
    return {
        "key": str(key or "").strip(),
        "label": str(label or "").strip(),
        "status": clean_status,
        "score": max(0, min(100, _safe_int(score, 0))),
        "summary": str(summary or "").strip(),
        "detail": str(detail or "").strip(),
        "action": str(action or "").strip(),
        "target_stage": str(target_stage or "").strip(),
        "count": max(0, _safe_int(count, 0)),
        "blocking": bool(blocking),
        "metrics": metrics if isinstance(metrics, dict) else {},
    }


def _feature_score_from_status(status: str, *, good: int = 96, warning: int = 72, error: int = 42, unknown: int = 55) -> int:
    rank = str(status or "").strip().lower()
    if rank == "good":
        return int(good)
    if rank == "warning":
        return int(warning)
    if rank == "error":
        return int(error)
    return int(unknown)


def _quality_feature_health(
    *,
    conversion_domain: dict,
    research_qa: dict,
    citation_cards: dict,
    failure_cases: list[dict],
    rerun_summary: dict,
    full_chain: dict,
) -> dict:
    stages = {
        str(stage.get("key") or ""): stage
        for stage in list((full_chain or {}).get("stages") or [])
        if isinstance(stage, dict)
    }

    conversion_summary = conversion_domain.get("summary") if isinstance(conversion_domain.get("summary"), dict) else {}
    conversion_status = str(conversion_domain.get("status") or "unknown").strip().lower() or "unknown"
    conversion_review = _safe_int(conversion_summary.get("review"), 0)
    conversion_unknown = _safe_int(conversion_summary.get("unknown"), 0)
    conversion_avg = _safe_int(conversion_summary.get("avg_score"), 0)

    qa_summary = research_qa.get("summary") if isinstance(research_qa.get("summary"), dict) else {}
    qa_available = bool(research_qa.get("available"))
    qa_total = _safe_int(qa_summary.get("total"), 0)
    qa_passed = _safe_int(qa_summary.get("passed"), 0)
    qa_failed = _safe_int(qa_summary.get("failed"), 0)
    qa_status = str((stages.get("research_qa") or {}).get("status") or ("warning" if not qa_available else "good")).lower()
    qa_score = int(round(100 * qa_passed / max(1, qa_total))) if qa_available and qa_total > 0 else _feature_score_from_status(qa_status)

    retrieval_stage = stages.get("retrieval") or {}
    retrieval_status = str(retrieval_stage.get("status") or "unknown").lower()
    retrieval_failed = _safe_int(retrieval_stage.get("count"), 0)

    card_summary = citation_cards.get("summary") if isinstance(citation_cards.get("summary"), dict) else {}
    card_available = bool(citation_cards.get("available"))
    citation_card_failed = _safe_int(card_summary.get("citation_card_failed"), 0)
    ref_card_failed = _safe_int(card_summary.get("ref_card_failed"), 0)
    system_b_failed = _safe_int(card_summary.get("system_b_failed"), 0)
    shelf_failed = _safe_int(card_summary.get("shelf_failed"), 0)
    card_failed = citation_card_failed + ref_card_failed + system_b_failed
    card_status = str((stages.get("citations") or {}).get("status") or ("warning" if not card_available else "good")).lower()
    shelf_status = str((stages.get("shelf") or {}).get("status") or ("warning" if not card_available else "good")).lower()

    missing_expected_cases = sum(
        1
        for case in list(failure_cases or [])
        if isinstance(case, dict) and list(case.get("missing_expected_doc_ids") or [])
    )
    citation_missing_cases = sum(
        1
        for case in list(failure_cases or [])
        if isinstance(case, dict)
        and (
            _safe_int(case.get("citation_count"), 0) <= 0
            or any(str(name or "").lower() in {"citations_include_required_docs", "citation_include_required_docs"} for name in list(case.get("failure_names") or []))
        )
    )

    paper_guide_status = "good"
    paper_guide_count = 0
    for status in [conversion_status, retrieval_status, card_status]:
        if _quality_status_rank(status) > _quality_status_rank(paper_guide_status):
            paper_guide_status = status
    if not qa_available and paper_guide_status == "good":
        paper_guide_status = "warning"
    paper_guide_count = conversion_review + retrieval_failed + card_failed

    reader_status = "good"
    reader_count = 0
    if citation_missing_cases > 0 or card_failed > 0:
        reader_status = "error"
        reader_count = citation_missing_cases + card_failed
    elif conversion_review > 0 or conversion_unknown > 0 or not card_available:
        reader_status = "warning"
        reader_count = conversion_review + conversion_unknown

    rerun_failed = _safe_int((rerun_summary or {}).get("failed"), 0) + _safe_int((rerun_summary or {}).get("error"), 0)
    repair_status = str((stages.get("repair_loop") or {}).get("status") or "unknown").lower()

    items = [
        _quality_feature_health_item(
            "pdf_conversion",
            "PDF conversion",
            conversion_status,
            score=conversion_avg if conversion_avg > 0 else _feature_score_from_status(conversion_status),
            summary="Markdown is ready for retrieval" if conversion_status == "good" else f"{conversion_review} sources need conversion review",
            detail=f"{conversion_unknown} converted sources still need quality assessment" if conversion_unknown > 0 else "Readable Markdown, page markers, figures, formulas, and references.",
            action="repair_conversion" if conversion_status != "good" else "review_conversion",
            target_stage="conversion",
            count=conversion_review + conversion_unknown,
            blocking=conversion_status == "error",
            metrics={"review": conversion_review, "unknown": conversion_unknown, "avg_score": conversion_avg},
        ),
        _quality_feature_health_item(
            "general_qa",
            "General QA",
            qa_status,
            score=qa_score,
            summary=f"{qa_failed} failed / {qa_total} research QA cases" if qa_available else "No research QA regression evidence yet",
            detail="Checks whether user questions retrieve the right papers and cite usable evidence.",
            action="fix_failed_qa_cases" if qa_failed > 0 else ("run_research_qa" if not qa_available else "review_research_qa"),
            target_stage="research_qa",
            count=qa_failed,
            blocking=qa_status == "error",
            metrics={"total": qa_total, "passed": qa_passed, "failed": qa_failed},
        ),
        _quality_feature_health_item(
            "paper_guide",
            "Paper Guide",
            paper_guide_status,
            score=_feature_score_from_status(paper_guide_status, good=94, warning=70, error=38),
            summary="Single-paper deep reading is backed by current evidence" if paper_guide_status == "good" else "Deep-reading quality is limited by source, retrieval, or citation failures",
            detail="Depends on conversion quality, focused retrieval, figure/equation/source grounding, and citation surfacing.",
            action="inspect_paper_guide" if paper_guide_status != "good" else "review_paper_guide",
            target_stage="research_qa" if qa_failed > 0 else ("retrieval" if retrieval_failed > 0 else ("citations" if card_failed > 0 else "conversion")),
            count=paper_guide_count,
            blocking=paper_guide_status == "error",
            metrics={"conversion_review": conversion_review, "retrieval_failed": retrieval_failed, "citation_failed": card_failed},
        ),
        _quality_feature_health_item(
            "citation_cards",
            "Citation cards",
            card_status,
            score=_feature_score_from_status(card_status, good=96, warning=74, error=45),
            summary="Citation cards pass current acceptance" if card_status == "good" else f"{card_failed} citation/card checks failed",
            detail="Tracks title, source, evidence quote, claim support, System B mapping, and card copy quality.",
            action="repair_citation_cards" if card_failed > 0 else "review_citation_cards",
            target_stage="citations",
            count=card_failed,
            blocking=card_status == "error",
            metrics={"citation_card_failed": citation_card_failed, "ref_card_failed": ref_card_failed, "system_b_failed": system_b_failed},
        ),
        _quality_feature_health_item(
            "literature_basket",
            "Literature basket",
            shelf_status,
            score=_feature_score_from_status(shelf_status, good=96, warning=73, error=44),
            summary="Basket metadata is export-ready" if shelf_status == "good" else f"{shelf_failed} basket quality checks failed",
            detail="Checks DOI, authors, venue, recommendation reason, source-open, and export readiness.",
            action="repair_shelf_metadata" if shelf_failed > 0 else "review_literature_basket",
            target_stage="shelf",
            count=shelf_failed,
            blocking=shelf_status == "error",
            metrics={"shelf_failed": shelf_failed},
        ),
        _quality_feature_health_item(
            "reader_locate",
            "Reader locate",
            reader_status,
            score=_feature_score_from_status(reader_status, good=94, warning=72, error=43),
            summary="Reader jumps have grounded evidence" if reader_status == "good" else "Reader locate may be affected by weak citations or source conversion",
            detail="Covers citation click-through, source opening, anchors, page markers, and evidence snippets.",
            action="inspect_reader_locate" if reader_status != "good" else "review_reader_locate",
            target_stage="citations" if card_failed > 0 or citation_missing_cases > 0 else "conversion",
            count=reader_count,
            blocking=reader_status == "error",
            metrics={"citation_missing_cases": citation_missing_cases, "conversion_review": conversion_review, "conversion_unknown": conversion_unknown},
        ),
        _quality_feature_health_item(
            "repair_loop",
            "Repair loop",
            repair_status,
            score=_feature_score_from_status(repair_status, good=96, warning=76, error=48),
            summary="Recent repairs have passing evidence" if repair_status == "good" else f"{rerun_failed} failed/error reruns need follow-up",
            detail="Confirms that source repair, metadata repair, reindex, and QA rerun actually improved results.",
            action="rerun_failed_cases" if repair_status != "good" else "review_repair_history",
            target_stage="repair_loop",
            count=rerun_failed,
            blocking=False,
            metrics={"rerun_failed": rerun_failed, "rerun_total": _safe_int((rerun_summary or {}).get("total"), 0)},
        ),
    ]

    worst = max(items, key=lambda item: _quality_status_rank(str(item.get("status") or ""))) if items else {}
    status = str(worst.get("status") or "unknown")
    score = int(round(sum(_safe_int(item.get("score"), 0) for item in items) / max(1, len(items)))) if items else 0
    unhealthy = sum(1 for item in items if str(item.get("status") or "") != "good")
    summary = (
        "All user-facing research workflows pass current health checks."
        if unhealthy <= 0
        else f"{unhealthy} user-facing workflows need attention before the product feels reliable."
    )
    return {
        "available": True,
        "status": status,
        "score": max(0, min(100, score)),
        "summary": summary,
        "items": items,
    }


def _quality_overview_from_listing(listing: dict) -> dict:
    items = [item for item in list((listing or {}).get("items") or []) if isinstance(item, dict)]
    counts = (listing or {}).get("counts") if isinstance((listing or {}).get("counts"), dict) else {}
    assessed = [item for item in items if isinstance(item.get("conversion_quality"), dict)]
    scores: list[int] = []
    issue_stats: dict[str, dict] = {}
    has_error = False

    for item in assessed:
        quality = item.get("conversion_quality") if isinstance(item.get("conversion_quality"), dict) else {}
        try:
            score = int(round(float((quality or {}).get("score") or 0)))
            if score > 0:
                scores.append(max(0, min(100, score)))
        except Exception:
            pass
        seen_in_paper: set[str] = set()
        for raw_issue in list((quality or {}).get("issues") or []):
            if not isinstance(raw_issue, dict):
                continue
            code = str(raw_issue.get("code") or raw_issue.get("label") or "").strip()
            label = str(raw_issue.get("label") or raw_issue.get("code") or "").strip()
            if not code or not label:
                continue
            key = code.lower()
            severity = str(raw_issue.get("severity") or "warning").strip().lower() or "warning"
            try:
                count = max(0, int(raw_issue.get("count") or 0))
            except Exception:
                count = 0
            cur = issue_stats.get(key) or {
                "code": code,
                "label": label,
                "severity": severity,
                "papers": 0,
                "count": 0,
                "repairable": bool(raw_issue.get("repairable")),
                "repair_strategy": str(raw_issue.get("repair_strategy") or ""),
                "repair_steps": list(raw_issue.get("repair_steps") or [])[:8],
            }
            cur["count"] = int(cur.get("count") or 0) + max(1, count)
            if key not in seen_in_paper:
                cur["papers"] = int(cur.get("papers") or 0) + 1
                seen_in_paper.add(key)
            if severity == "error":
                cur["severity"] = "error"
                has_error = True
            issue_stats[key] = cur

    def issue_sort_key(issue: dict) -> tuple[int, int, int, str]:
        severity_weight = 2 if str(issue.get("severity") or "") == "error" else 1
        return (
            -severity_weight,
            -int(issue.get("papers") or 0),
            -int(issue.get("count") or 0),
            str(issue.get("label") or "").lower(),
        )

    top_issues = sorted(issue_stats.values(), key=issue_sort_key)[:8]

    def item_quality(item: dict) -> dict:
        quality = item.get("conversion_quality")
        return quality if isinstance(quality, dict) else {}

    recommended_source = [
        item
        for item in items
        if str(item.get("task_state") or "") == "idle"
        and bool(item_quality(item).get("has_review_issue"))
    ]
    recommended_source.sort(
        key=lambda item: (
            -_quality_status_rank(str(item_quality(item).get("status") or "")),
            int(item_quality(item).get("score") or 0),
            str(item.get("name") or "").lower(),
        )
    )
    recommended: list[dict] = []
    for item in recommended_source[:8]:
        quality = item_quality(item)
        recommended.append(
            {
                "name": str(item.get("name") or ""),
                "path": str(item.get("path") or ""),
                "md_path": str(item.get("md_path") or ""),
                "status": str(quality.get("status") or ""),
                "score": int(quality.get("score") or 0),
                "summary": str(quality.get("summary") or ""),
                "task_state": str(item.get("task_state") or ""),
                "issues": list(quality.get("issues") or [])[:4],
            }
        )

    converted = int(counts.get("converted") or sum(1 for item in items if str(item.get("category") or "") == "converted"))
    review = int(counts.get("quality_review") or 0)
    ready = int(counts.get("quality_ready") or 0)
    unknown = sum(
        1
        for item in items
        if str(item.get("category") or "") == "converted"
        and not isinstance(item.get("conversion_quality"), dict)
    )
    if has_error:
        conversion_status = "error"
    elif review > 0 or unknown > 0:
        conversion_status = "warning"
    else:
        conversion_status = "good"
    conversion_domain = {
        "available": converted > 0,
        "status": conversion_status,
        "summary": {
            "converted": converted,
            "assessed": len(assessed),
            "good": ready,
            "review": review,
            "unknown": int(unknown),
            "avg_score": int(round(sum(scores) / len(scores))) if scores else 0,
        },
        "top_failures": [
            {
                "name": str(issue.get("label") or issue.get("code") or ""),
                "count": int(issue.get("papers") or issue.get("count") or 0),
            }
            for issue in top_issues[:6]
        ],
    }
    research_qa = _latest_research_qa_quality_summary()
    citation_cards = _latest_citation_card_quality_summary()
    rerun_history = _research_qa_rerun_history_rows()
    failure_cases = _latest_research_qa_failure_cases(rerun_history=rerun_history)
    domains = {
        "conversion": conversion_domain,
        "research_qa": research_qa,
        "citation_cards": citation_cards,
    }
    rerun_summary = _research_qa_rerun_history_summary(rerun_history)
    priority_actions = _quality_priority_actions(
        conversion_status=conversion_status,
        recommended=recommended,
        research_qa=research_qa,
        citation_cards=citation_cards,
    )
    full_chain = _quality_full_chain_check(
        conversion_domain=conversion_domain,
        research_qa=research_qa,
        citation_cards=citation_cards,
        failure_cases=failure_cases,
        recommended=recommended,
        rerun_summary=rerun_summary,
        priority_actions=priority_actions,
        conversion_issues=top_issues,
    )
    full_chain["action_history"] = _quality_action_history_rows(limit=12)
    feature_health = _quality_feature_health(
        conversion_domain=conversion_domain,
        research_qa=research_qa,
        citation_cards=citation_cards,
        failure_cases=failure_cases,
        rerun_summary=rerun_summary,
        full_chain=full_chain,
    )

    return {
        "status": str(full_chain.get("status") or conversion_status),
        "summary": {
            "total_view": int(counts.get("total_view") or len(items)),
            "total_all": int(counts.get("total_all") or len(items)),
            "converted": converted,
            "pending": int(counts.get("pending") or 0),
            "queued": int(counts.get("queued") or 0),
            "running": int(counts.get("running") or 0),
            "assessed": len(assessed),
            "good": ready,
            "review": review,
            "unknown": int(unknown),
            "avg_score": int(round(sum(scores) / len(scores))) if scores else 0,
        },
        "top_issues": top_issues,
        "recommended": recommended,
        "domains": domains,
        "full_chain": full_chain,
        "feature_health": feature_health,
        "failure_cases": failure_cases,
        "rerun_summary": rerun_summary,
        "priority_actions": priority_actions,
        "queue": (listing or {}).get("queue") or {},
        "scope": str((listing or {}).get("scope") or ""),
        "truncated": bool((listing or {}).get("truncated")),
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
        "meta": _suggestion_basis_meta(suggestion, venue=venue, year=year, title=title),
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

        explicit_base = Path((base_name or "").strip()).stem
        fallback_title = explicit_base or Path(raw_name_pdf).stem or "Untitled"
        venue = str(getattr(sug, "venue", "") or "").strip()
        year = str(getattr(sug, "year", "") or "").strip()
        title = str(getattr(sug, "title", "") or "").strip() or fallback_title

        if explicit_base:
            base = explicit_base
        else:
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

    explicit_base = Path((base_name or "").strip()).stem
    fallback_title = explicit_base or src_pdf.stem or "Untitled"
    venue = str(getattr(sug, "venue", "") or "").strip()
    year = str(getattr(sug, "year", "") or "").strip()
    title = str(getattr(sug, "title", "") or "").strip() or fallback_title

    if explicit_base:
        base = explicit_base
    else:
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


@router.get("/quality/overview")
def library_quality_overview(scope: str = "all"):
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    pdf_d.mkdir(parents=True, exist_ok=True)
    md_d.mkdir(parents=True, exist_ok=True)
    listing = _collect_library_files(pdf_dir=pdf_d, md_dir=md_d, scope=scope)
    return {
        "ok": True,
        **_quality_overview_from_listing(listing),
    }


class ConvertPendingBody(BaseModel):
    speed_mode: str = "balanced"
    no_llm: bool = False
    limit: int = 0
    replace: bool = True


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
            "meta": _suggestion_basis_meta(sug, venue=venue, year=year, title=title),
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
                    "meta": {
                        "venue": "",
                        "year": "",
                        "title": "",
                        "match_method": "",
                        "year_source": "",
                        "basis_label": "建议生成失败",
                        "basis_detail": "本次未能生成稳定建议，保留原文件名。",
                    },
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
    replace = bool(body.replace)

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
            replace=replace,
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
                replace=True,
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
    replace: bool = True


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
            "running": bool(snap.get("running", False)) or bool(list(snap.get("active_tasks") or [])),
            "done": (not bool(snap.get("running", False))) and (not bool(list(snap.get("active_tasks") or []))) and snap.get("total", 0) > 0,
            "total": snap.get("total", 0),
            "completed": snap.get("done", 0),
            "current": snap.get("current", ""),
            "active_count": int(snap.get("active_count", len(list(snap.get("active_tasks") or []))) or 0),
            "active_tasks": _compact_active_tasks(snap),
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


class OpenQualityArtifactBody(BaseModel):
    domain: str = "research_qa"
    target: str = "report"  # report | folder | raw | summary | runbook


class QualityActionHistoryBody(BaseModel):
    stage_key: str
    stage_label: str = ""
    action: str = ""
    status: str = "info"
    summary: str
    detail: str = ""
    target_ids: list[str] = []
    metrics: dict = {}
    before: dict = {}
    after: dict = {}
    delta: dict = {}
    improved: bool | None = None
    verification: dict = {}
    created_at: int = 0


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


def _quality_artifact_path(domain: str, target: str) -> tuple[str, Path]:
    domain_key = str(domain or "").strip().lower()
    target_key = str(target or "report").strip().lower()
    if domain_key not in {"research_qa", "citation_cards"}:
        raise HTTPException(400, "invalid domain")

    if target_key == "runbook":
        path = (Path("docs") / "RESEARCH_QA_EVAL_RUNBOOK.md").expanduser()
        if not _path_is_file(path):
            raise HTTPException(404, "runbook not found")
        return "runbook", path

    summary_path, raw_path = _latest_research_qa_artifacts()
    anchor = summary_path or raw_path
    if anchor is None:
        raise HTTPException(404, "quality artifact not found")

    if target_key == "folder":
        return "folder", anchor.parent
    if target_key == "raw":
        if raw_path is None or not _path_is_file(raw_path):
            raise HTTPException(404, "raw results not found")
        return "raw", raw_path
    if target_key == "summary":
        if summary_path is None or not _path_is_file(summary_path):
            raise HTTPException(404, "summary not found")
        return "summary", summary_path

    if target_key != "report":
        raise HTTPException(400, "invalid target")
    report_path = anchor.parent / "report.md"
    if _path_is_file(report_path):
        return "report", report_path
    return "folder", anchor.parent


@router.post("/quality/artifact/open")
def open_quality_artifact(body: OpenQualityArtifactBody):
    target, path = _quality_artifact_path(body.domain, body.target)
    open_in_explorer(path)
    return {
        "ok": True,
        "domain": str(body.domain or "").strip().lower(),
        "target": target,
        "path": str(path),
    }


@router.get("/quality/action-history")
def quality_action_history(limit: int = 20):
    return {
        "ok": True,
        "items": _quality_action_history_rows(limit=limit),
    }


@router.post("/quality/action-history")
def append_quality_action_history(body: QualityActionHistoryBody):
    row = _append_quality_action_history(body.model_dump())
    return {
        "ok": True,
        "item": row,
    }


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return Path.cwd()


def _tail_text(value: str, *, limit: int = 2400) -> str:
    text = str(value or "")
    if len(text) <= int(limit):
        return text
    return text[-int(limit):]


def _research_qa_transient_error(text: str) -> tuple[str, str]:
    raw = str(text or "")
    folded = raw.lower()
    if any(token in folded for token in [
        "connection refused",
        "failed to establish a new connection",
        "max retries exceeded",
        "connecterror",
        "httpconnectionpool",
        "econnrefused",
        "err_connection_refused",
        "network is unreachable",
        "temporary failure in name resolution",
    ]):
        return "connection", _compact_text(raw, limit=240)
    if any(token in folded for token in [
        "timed out",
        "timeout",
        "read timed out",
        "deadline exceeded",
    ]):
        return "timeout", _compact_text(raw, limit=240)
    return "", _compact_text(raw, limit=240)


def _extract_research_qa_output_dir(stdout: str, *, fallback_after: float) -> Path | None:
    for match in reversed(list(re.finditer(r"research QA eval finished:\s*(.+)", str(stdout or "")))):
        raw = str(match.group(1) or "").strip()
        if raw:
            return Path(raw)
    try:
        candidates = [p.parent for p in _RESEARCH_QA_EVAL_ROOT.rglob("summary.json") if p.is_dir()]
    except Exception:
        return None
    recent = [p for p in candidates if _safe_mtime(p / "summary.json") >= float(fallback_after) - 2.0]
    if not recent:
        recent = candidates
    if not recent:
        return None
    recent.sort(key=lambda p: (_safe_mtime(p / "summary.json"), str(p)), reverse=True)
    return recent[0]


def _research_qa_rerun_result(*, case_id: str, output_dir: Path | None, returncode: int, stdout: str, stderr: str, started_at: float, finished_at: float) -> dict:
    summary_path = output_dir / "summary.json" if output_dir is not None else None
    raw_path = output_dir / "raw_results.jsonl" if output_dir is not None else None
    report_path = output_dir / "report.md" if output_dir is not None else None
    summary = _read_json_artifact(summary_path)
    rows = _read_jsonl_artifact(raw_path, limit=100)
    row = next((item for item in rows if str(item.get("id") or "") == case_id), rows[0] if rows else {})
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    failures = [
        {
            "name": _quality_failure_name(item),
            "domain": _quality_failure_domain(_quality_failure_name(item)),
            "detail": _quality_failure_detail(item),
        }
        for item in list((quality or {}).get("failures") or [])
    ]
    quality_ok = bool((quality or {}).get("ok")) if quality else False
    if int(returncode) != 0:
        status = "error"
    elif quality:
        status = "passed" if quality_ok else "failed"
    else:
        status = "complete" if int(returncode) == 0 else "error"
    error_kind = ""
    error_detail = ""
    if status == "error":
        error_kind, error_detail = _research_qa_transient_error("\n".join([str(stderr or ""), str(stdout or "")]))
    return {
        "ok": int(returncode) == 0,
        "case_id": case_id,
        "status": status,
        "quality_ok": quality_ok,
        "returncode": int(returncode),
        "summary": summary,
        "failures": failures[:8],
        "output_dir": str(output_dir or ""),
        "report_path": str(report_path) if report_path is not None and _path_is_file(report_path) else "",
        "raw_path": str(raw_path) if raw_path is not None and _path_is_file(raw_path) else "",
        "stdout_tail": _tail_text(stdout),
        "stderr_tail": _tail_text(stderr),
        "error_kind": error_kind,
        "error_detail": error_detail,
        "started_at": int(started_at),
        "finished_at": int(finished_at),
        "latency_ms": int(round((float(finished_at) - float(started_at)) * 1000)),
    }


@router.post("/quality/research-qa/rerun")
def rerun_research_qa_case(body: QualityResearchQaRerunBody):
    case_id = str(body.case_id or "").strip()
    if not case_id:
        raise HTTPException(400, "case_id is required")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", case_id):
        raise HTTPException(400, "invalid case_id")

    repo = _repo_root()
    runner = repo / "tools" / "research_qa" / "run_research_qa_eval.py"
    if not _path_is_file(runner):
        raise HTTPException(404, "research QA runner not found")

    timeout_s = max(10.0, min(900.0, float(body.timeout_s or 180.0)))
    base_url = str(body.base_url or os.environ.get("KB_RESEARCH_QA_BASE_URL") or "http://127.0.0.1:8000").strip().rstrip("/")
    cmd = [
        sys.executable,
        str(runner),
        "--case-id",
        case_id,
        "--out-dir",
        str(_RESEARCH_QA_EVAL_ROOT),
        "--timeout-s",
        str(timeout_s),
        "--top-k",
        str(max(1, min(20, int(body.top_k or 6)))),
        "--max-tokens",
        str(max(256, min(8192, int(body.max_tokens or 1800)))),
        "--base-url",
        base_url,
    ]
    if bool(body.dry_run):
        cmd.append("--dry-run")

    started_at = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout_s + 60.0,
        )
    except subprocess.TimeoutExpired as exc:
        finished_at = time.time()
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        result = _research_qa_rerun_result(
            case_id=case_id,
            output_dir=None,
            returncode=124,
            stdout=stdout,
            stderr=stderr or "research QA rerun timed out",
            started_at=started_at,
            finished_at=finished_at,
        )
        if not bool(body.dry_run):
            _append_research_qa_rerun_history(result)
        return result

    finished_at = time.time()
    output_dir = None if bool(body.dry_run) else _extract_research_qa_output_dir(proc.stdout, fallback_after=started_at)
    result = _research_qa_rerun_result(
        case_id=case_id,
        output_dir=output_dir,
        returncode=int(proc.returncode),
        stdout=str(proc.stdout or ""),
        stderr=str(proc.stderr or ""),
        started_at=started_at,
        finished_at=finished_at,
    )
    if not bool(body.dry_run):
        _append_research_qa_rerun_history(result)
    return result


@router.post("/quality/sources")
def library_source_quality(body: QualitySourcesBody):
    sources = list(body.sources or [])
    if len(sources) > 80:
        sources = sources[:80]
    items: list[dict] = []
    seen: set[str] = set()
    for item in sources:
        source_path = str(item.source_path or "").strip()
        source_name = str(item.source_name or "").strip()
        key = f"{source_path}\n{source_name}".lower()
        if not source_path and not source_name:
            continue
        if key in seen:
            continue
        seen.add(key)
        items.append(_resolve_quality_source(source_path=source_path, source_name=source_name))
    review_count = sum(
        1
        for item in items
        if bool(((item.get("conversion_quality") or {}) if isinstance(item.get("conversion_quality"), dict) else {}).get("has_review_issue"))
    )
    return {
        "ok": True,
        "items": items,
        "review_count": int(review_count),
    }


class DeleteLibraryFileBody(BaseModel):
    pdf_name: str
    also_md: bool = True
    remove_queued: bool = True


class GuideSourceBody(BaseModel):
    pdf_name: str


class QualitySourceItem(BaseModel):
    source_path: str = ""
    source_name: str = ""


class QualitySourcesBody(BaseModel):
    sources: list[QualitySourceItem] = []


class QualityRepairBody(BaseModel):
    pdf_names: list[str] = []
    sources: list[QualitySourceItem] = []
    speed_mode: str = "balanced"
    no_llm: bool = False
    replace: bool = True
    md_autofix: bool = True


class QualityResearchQaRerunBody(BaseModel):
    case_id: str = ""
    base_url: str = ""
    timeout_s: float = 180.0
    top_k: int = 6
    max_tokens: int = 1800
    dry_run: bool = False


class UpdateLibraryMetaBody(BaseModel):
    pdf_name: str = ""
    sha1: str = ""
    path: str = ""
    paper_category: str = ""
    reading_status: str = ""
    note: str = ""
    user_tags: list[str] = []


class BatchUpdateLibraryMetaBody(BaseModel):
    pdf_names: list[str] = []
    sha1s: list[str] = []
    apply_paper_category: bool = False
    paper_category: str = ""
    apply_reading_status: bool = False
    reading_status: str = ""
    add_tags: list[str] = []
    remove_tags: list[str] = []


class RegenerateLibrarySuggestionsBody(BaseModel):
    pdf_names: list[str] = []
    sha1s: list[str] = []


class LibrarySuggestionActionBody(BaseModel):
    pdf_name: str = ""
    sha1: str = ""
    path: str = ""
    category_action: str = ""
    accept_tags: list[str] = []
    dismiss_tags: list[str] = []
    accept_all_tags: bool = False
    dismiss_all_tags: bool = False


@router.post("/quality/repair")
def repair_library_quality(body: QualityRepairBody):
    settings = get_settings()
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    md_d.mkdir(parents=True, exist_ok=True)

    speed_mode = str(body.speed_mode or "balanced").strip() or "balanced"
    no_llm = bool(body.no_llm) or (speed_mode.lower() == "no_llm")
    replace = bool(body.replace)
    snap = _bg_snapshot()
    task_by_path, task_by_name = _build_task_maps_from_snapshot(snap)

    requested = 0
    items: list[dict] = []
    targets: list[dict] = []

    for raw_name in list(body.pdf_names or [])[:200]:
        pdf_name = str(raw_name or "").strip()
        if not pdf_name:
            continue
        requested += 1
        if Path(pdf_name).name != pdf_name:
            items.append({
                "source_path": "",
                "source_name": pdf_name,
                "pdf_name": pdf_name,
                "pdf_path": "",
                "ok": False,
                "enqueued": False,
                "skipped_busy": False,
                "error": "invalid pdf_name",
                "task_id": "",
            })
            continue
        targets.append({
            "source_path": "",
            "source_name": pdf_name,
            "pdf_path": str((pdf_d / pdf_name).expanduser()),
            "md_path": "",
        })

    seen_sources: set[str] = set()
    for source in list(body.sources or [])[:200]:
        source_path = str(source.source_path or "").strip()
        source_name = str(source.source_name or "").strip()
        if not source_path and not source_name:
            continue
        source_key = f"{source_path}\n{source_name}".lower()
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        requested += 1
        resolved = _resolve_quality_source(source_path=source_path, source_name=source_name)
        pdf_path_raw = str(resolved.get("pdf_path") or "").strip()
        if not pdf_path_raw:
            items.append({
                "source_path": source_path,
                "source_name": source_name,
                "pdf_name": "",
                "pdf_path": "",
                "ok": False,
                "enqueued": False,
                "skipped_busy": False,
                "error": "source pdf not found",
                "task_id": "",
            })
            continue
        targets.append({
            "source_path": source_path,
            "source_name": source_name,
            "pdf_path": pdf_path_raw,
            "md_path": str(resolved.get("md_path") or ""),
        })

    enqueued = 0
    repaired = 0
    skipped_busy = 0
    failed = sum(1 for item in items if not bool(item.get("ok")))
    seen_pdf_paths: set[str] = set()
    for target in targets:
        pdf_path = Path(str(target.get("pdf_path") or "")).expanduser()
        pdf_name = pdf_path.name
        key = _normalized_path_key(pdf_path).lower()
        if not key or key in seen_pdf_paths:
            continue
        seen_pdf_paths.add(key)

        base_item = {
            "source_path": str(target.get("source_path") or ""),
            "source_name": str(target.get("source_name") or ""),
            "pdf_name": pdf_name,
            "pdf_path": str(pdf_path),
            "ok": False,
            "enqueued": False,
            "repaired": False,
            "repair_changed": False,
            "repair_applied": [],
            "repair_before_score": 0,
            "repair_after_score": 0,
            "remaining_issue_codes": [],
            "md_path": "",
            "skipped_busy": False,
            "error": "",
            "repair_error": "",
            "task_id": "",
        }
        if (not _path_is_within(pdf_path, [pdf_d])) or (not _path_is_file(pdf_path)):
            items.append({**base_item, "error": "pdf not found"})
            failed += 1
            continue

        task_info = task_by_path.get(_normalized_path_key(pdf_path)) or task_by_name.get(pdf_name) or {}
        if bool(task_info.get("queued")) or bool(task_info.get("running")):
            items.append({**base_item, "skipped_busy": True, "error": "already queued or running"})
            skipped_busy += 1
            continue

        md_path_raw = str(target.get("md_path") or "").strip()
        if md_path_raw:
            md_path = Path(md_path_raw).expanduser()
            md_exists = _path_is_file(md_path)
        else:
            _md_folder, md_path, md_exists = _resolve_md_output_paths(md_d, pdf_path)
        repair_payload: dict = {}
        skip_enqueue_after_repair = False
        if bool(body.md_autofix) and md_exists and _path_is_within(md_path, [md_d]):
            try:
                before_quality = _conversion_quality_summary(md_path) or {}
                before_issues = [
                    str(issue.get("code") or "")
                    for issue in list(before_quality.get("issues") or [])
                    if isinstance(issue, dict) and str(issue.get("code") or "").strip()
                ]
                repair_result = repair_markdown_quality(md_path, issue_codes=before_issues)
                _clear_conversion_quality_cache(md_path)
                after_quality = _conversion_quality_summary(md_path) or {}
                repair_changed = bool(repair_result.get("changed"))
                if repair_changed:
                    repaired += 1
                skip_enqueue_after_repair = repair_changed and not bool(after_quality.get("has_review_issue"))
                before_issue_codes = [
                    str(issue.get("code") or "")
                    for issue in list(before_quality.get("issues") or [])
                    if isinstance(issue, dict) and str(issue.get("code") or "").strip()
                ]
                after_issue_codes = [
                    str(issue.get("code") or "")
                    for issue in list(after_quality.get("issues") or [])
                    if isinstance(issue, dict) and str(issue.get("code") or "").strip()
                ]
                after_issue_set = set(after_issue_codes)
                fixed_issue_codes = [code for code in before_issue_codes if code and code not in after_issue_set]
                repair_payload = {
                    "repaired": repair_changed,
                    "repair_changed": repair_changed,
                    "repair_applied": list(repair_result.get("applied") or [])[:12],
                    "repair_before_score": _safe_int(before_quality.get("score"), 0),
                    "repair_after_score": _safe_int(after_quality.get("score"), 0),
                    "quality_before": before_quality,
                    "quality_after": after_quality,
                    "before_issue_codes": before_issue_codes[:12],
                    "fixed_issue_codes": fixed_issue_codes[:12],
                    "remaining_issue_codes": after_issue_codes[:12],
                    "md_path": str(md_path),
                    "repair_unsafe": bool(repair_result.get("unsafe")),
                    "repair_regression_reasons": list(repair_result.get("regression_reasons") or [])[:8],
                }
            except Exception as exc:
                repair_payload = {
                    "md_path": str(md_path),
                    "repair_error": str(exc)[:240] or "markdown autofix failed",
                }
        elif md_exists:
            repair_payload = {"md_path": str(md_path)}

        if skip_enqueue_after_repair:
            items.append({**base_item, **repair_payload, "ok": True, "enqueued": False})
            continue

        try:
            task = _build_bg_task(
                pdf_path=pdf_path,
                out_root=md_d,
                db_dir=Path(settings.db_dir).expanduser(),
                no_llm=no_llm,
                replace=replace,
                speed_mode=speed_mode,
            )
            _bg_enqueue(task)
            task_id = str(task.get("_tid") or "")
            items.append({**base_item, **repair_payload, "ok": True, "enqueued": True, "task_id": task_id})
            enqueued += 1
        except Exception as exc:
            items.append({**base_item, **repair_payload, "error": str(exc)[:240] or "enqueue failed"})
            failed += 1

    repaired_items = [item for item in items if bool(item.get("repair_changed"))]
    fixed_counter: Counter = Counter()
    remaining_counter: Counter = Counter()
    before_scores: list[int] = []
    after_scores: list[int] = []
    improved = 0
    for item in repaired_items:
        before_score = _safe_int(item.get("repair_before_score"), 0)
        after_score = _safe_int(item.get("repair_after_score"), 0)
        before_scores.append(before_score)
        after_scores.append(after_score)
        if after_score > before_score:
            improved += 1
        for code in _list_strings(item.get("fixed_issue_codes")):
            fixed_counter[code] += 1
        for code in _list_strings(item.get("remaining_issue_codes")):
            remaining_counter[code] += 1
    needs_reindex = bool(repaired_items) or enqueued > 0
    impact = {
        "requested": int(requested),
        "repaired": int(len(repaired_items)),
        "improved": int(improved),
        "enqueued": int(enqueued),
        "skipped_busy": int(skipped_busy),
        "failed": int(failed),
        "needs_reindex": bool(needs_reindex),
        "before_avg_score": int(round(sum(before_scores) / len(before_scores))) if before_scores else 0,
        "after_avg_score": int(round(sum(after_scores) / len(after_scores))) if after_scores else 0,
        "score_delta": (
            int(round(sum(after_scores) / len(after_scores))) - int(round(sum(before_scores) / len(before_scores)))
            if before_scores and after_scores
            else 0
        ),
        "fixed_issue_codes": _counter_items(fixed_counter, limit=8),
        "remaining_issue_codes": _counter_items(remaining_counter, limit=8),
    }

    return {
        "ok": failed == 0,
        "requested": int(requested),
        "enqueued": int(enqueued),
        "repaired": int(repaired),
        "needs_reindex": bool(needs_reindex),
        "impact": impact,
        "skipped_busy": int(skipped_busy),
        "failed": int(failed),
        "items": items,
    }


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
    if _is_pdf_active_in_snapshot(snap=snap, pdf_path=pdf_path, pdf_name=pdf_name):
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


@router.post("/meta/update")
def update_library_meta(body: UpdateLibraryMetaBody):
    pdf_name = str(body.pdf_name or "").strip()
    sha1 = str(body.sha1 or "").strip().lower()
    path_raw = str(body.path or "").strip()
    resolved_path: Path | None = None

    if pdf_name:
        if Path(pdf_name).name != pdf_name:
            raise HTTPException(400, "invalid pdf_name")
        resolved_path = (_pdf_dir() / pdf_name).expanduser()
    elif path_raw:
        resolved_path = Path(path_raw).expanduser()
    elif not sha1:
        raise HTTPException(400, "pdf_name, path, or sha1 required")

    payload = _library_store().upsert_paper_user_meta(
        sha1=sha1,
        path=resolved_path,
        paper_category=str(body.paper_category or ""),
        reading_status=str(body.reading_status or ""),
        note=str(body.note or ""),
        user_tags=list(body.user_tags or []),
    )
    if not payload:
        raise HTTPException(404, "library item not found")
    return {
        "ok": True,
        "sha1": str(payload.get("sha1") or ""),
        "path": str(payload.get("path") or (resolved_path or "")),
        "paper_category": str(payload.get("paper_category") or ""),
        "reading_status": str(payload.get("reading_status") or ""),
        "note": str(payload.get("note") or ""),
        "user_tags": list(payload.get("user_tags") or []),
        "has_suggestions": bool(payload.get("has_suggestions")),
        "suggested_category": str(payload.get("suggested_category") or ""),
        "suggested_tags": list(payload.get("suggested_tags") or []),
    }


@router.post("/meta/batch_update")
def batch_update_library_meta(body: BatchUpdateLibraryMetaBody):
    pdf_names = [str(name or "").strip() for name in list(body.pdf_names or []) if str(name or "").strip()]
    sha1s = [str(value or "").strip().lower() for value in list(body.sha1s or []) if str(value or "").strip()]
    if not pdf_names and not sha1s:
        raise HTTPException(400, "pdf_names or sha1s required")
    if not (
        bool(body.apply_paper_category)
        or bool(body.apply_reading_status)
        or bool(list(body.add_tags or []))
        or bool(list(body.remove_tags or []))
    ):
        raise HTTPException(400, "no batch changes requested")

    paths: list[Path] = []
    for pdf_name in pdf_names:
        if Path(pdf_name).name != pdf_name:
            raise HTTPException(400, f"invalid pdf_name: {pdf_name}")
        paths.append((_pdf_dir() / pdf_name).expanduser())

    payloads = _library_store().batch_update_paper_user_meta(
        sha1s=sha1s,
        paths=paths,
        apply_paper_category=bool(body.apply_paper_category),
        paper_category=str(body.paper_category or ""),
        apply_reading_status=bool(body.apply_reading_status),
        reading_status=str(body.reading_status or ""),
        add_tags=list(body.add_tags or []),
        remove_tags=list(body.remove_tags or []),
    )

    return {
        "ok": True,
        "requested": len(pdf_names) + len(sha1s),
        "updated": len(payloads),
        "items": [
            {
                "name": Path(str(payload.get("path") or "")).name,
                "sha1": str(payload.get("sha1") or ""),
                "path": str(payload.get("path") or ""),
                "paper_category": str(payload.get("paper_category") or ""),
                "reading_status": str(payload.get("reading_status") or ""),
                "note": str(payload.get("note") or ""),
                "user_tags": list(payload.get("user_tags") or []),
            }
            for payload in payloads
        ],
    }


@router.post("/meta/suggestions/regenerate")
def regenerate_library_meta_suggestions(body: RegenerateLibrarySuggestionsBody):
    pdf_names = [str(name or "").strip() for name in list(body.pdf_names or []) if str(name or "").strip()]
    sha1s = [str(value or "").strip().lower() for value in list(body.sha1s or []) if str(value or "").strip()]

    paths: list[Path] = []
    for pdf_name in pdf_names:
        if Path(pdf_name).name != pdf_name:
            raise HTTPException(400, f"invalid pdf_name: {pdf_name}")
        paths.append((_pdf_dir() / pdf_name).expanduser())

    payloads = _library_store().regenerate_paper_suggestions(
        sha1s=sha1s,
        paths=paths,
    )
    return {
        "ok": True,
        "updated": len(payloads),
        "items": [
            {
                "name": Path(str(payload.get("path") or "")).name,
                "sha1": str(payload.get("sha1") or ""),
                "path": str(payload.get("path") or ""),
                "paper_category": str(payload.get("paper_category") or ""),
                "reading_status": str(payload.get("reading_status") or ""),
                "note": str(payload.get("note") or ""),
                "user_tags": list(payload.get("user_tags") or []),
                "has_suggestions": bool(payload.get("has_suggestions")),
                "suggested_category": str(payload.get("suggested_category") or ""),
                "suggested_tags": list(payload.get("suggested_tags") or []),
            }
            for payload in payloads
        ],
    }


@router.post("/meta/suggestions/apply")
def apply_library_meta_suggestions(body: LibrarySuggestionActionBody):
    pdf_name = str(body.pdf_name or "").strip()
    sha1 = str(body.sha1 or "").strip().lower()
    path_raw = str(body.path or "").strip()
    resolved_path: Path | None = None

    if pdf_name:
        if Path(pdf_name).name != pdf_name:
            raise HTTPException(400, "invalid pdf_name")
        resolved_path = (_pdf_dir() / pdf_name).expanduser()
    elif path_raw:
        resolved_path = Path(path_raw).expanduser()
    elif not sha1:
        raise HTTPException(400, "pdf_name, path, or sha1 required")

    category_action = str(body.category_action or "").strip().lower()
    if category_action not in {"", "accept", "dismiss"}:
        raise HTTPException(400, "invalid category_action")

    payload = _library_store().apply_paper_suggestion_actions(
        sha1=sha1,
        path=resolved_path,
        category_action=category_action,
        accept_tags=list(body.accept_tags or []),
        dismiss_tags=list(body.dismiss_tags or []),
        accept_all_tags=bool(body.accept_all_tags),
        dismiss_all_tags=bool(body.dismiss_all_tags),
    )
    if not payload:
        raise HTTPException(404, "library item not found")

    return {
        "ok": True,
        "sha1": str(payload.get("sha1") or ""),
        "path": str(payload.get("path") or (resolved_path or "")),
        "paper_category": str(payload.get("paper_category") or ""),
        "reading_status": str(payload.get("reading_status") or ""),
        "note": str(payload.get("note") or ""),
        "user_tags": list(payload.get("user_tags") or []),
        "has_suggestions": bool(payload.get("has_suggestions")),
        "suggested_category": str(payload.get("suggested_category") or ""),
        "suggested_tags": list(payload.get("suggested_tags") or []),
    }


@router.post("/file/guide_source")
def resolve_library_guide_source(body: GuideSourceBody):
    pdf_name = str(body.pdf_name or "").strip()
    if (not pdf_name) or (Path(pdf_name).name != pdf_name):
        raise HTTPException(400, "invalid pdf_name")
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    pdf_path = (pdf_d / pdf_name).expanduser()
    if (not _path_exists(pdf_path)) or (not _path_is_file(pdf_path)):
        raise HTTPException(404, "pdf not found")

    _md_folder, md_main, md_exists = _resolve_md_output_paths(md_d, pdf_path)
    if (not md_exists) or (not _path_is_file(md_main)):
        raise HTTPException(400, "markdown not ready")

    source_name = _strip_known_source_ext(pdf_name) or pdf_name
    return {
        "ok": True,
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "md_path": str(md_main),
        "md_exists": True,
        # Bind to PDF path; runtime maps to latest markdown on disk.
        "source_path": str(pdf_path),
        "source_name": source_name,
    }


@router.post("/reindex")
def reindex():
    s = get_settings()
    md_d = _md_dir()
    pdf_d = _pdf_dir()
    ingest_py = _ingest_py_path()
    if not ingest_py.exists():
        return {
            "ok": False,
            "error": "ingest.py not found",
            "structured_indices": None,
            "structured_indices_error": "",
            "refsync": None,
            "refsync_error": "",
        }
    structured_indices: dict | None = None
    structured_indices_error = ""
    try:
        structured_indices = rebuild_structured_indices_for_root(md_d, force=False)
    except Exception as exc:
        structured_indices_error = str(exc)
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
        "structured_indices": structured_indices,
        "structured_indices_error": structured_indices_error,
        "refsync": refsync,
        "refsync_error": refsync_error,
    }
