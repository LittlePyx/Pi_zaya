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
from kb.converter.quality_repair import (
    append_conversion_repair_attempt,
    conversion_quality_result_path,
    conversion_repair_strategy_for_issue,
    load_conversion_quality_result,
    plan_conversion_quality_repair,
    repair_markdown_quality,
    write_conversion_quality_result,
)
from kb.converter.quality_center import (
    discover_quality_markdown_files,
    quality_center_summary,
    repair_quality_targets,
    scan_quality_targets,
    source_pdf_for_markdown,
)
from kb.converter.figure_assets import (
    scan_figure_asset_quality,
    summarize_figure_asset_quality_reports,
)
from kb.converter.structured_index_batch import rebuild_structured_indices_for_root
from kb.library_store import LibraryStore
from kb.pdf_tools import PdfMetaSuggestion, extract_pdf_meta_suggestion, run_pdf_to_md, open_in_explorer
from kb.reference_sync import start_reference_sync
from kb.store import compute_doc_id, doc_chunks_path, load_docs_index

router = APIRouter(prefix="/api/library", tags=["library"])
_RENAME_SUGGEST_CACHE: dict[str, dict] = {}
_CONVERSION_QUALITY_CACHE: dict[str, tuple[int, int, int, int, dict]] = {}
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
        "repair_action": str(strategy.get("action") or ""),
        "repair_scope": str(strategy.get("scope") or ""),
        "repair_speed_mode": str(strategy.get("speed_mode") or ""),
    }


def _conversion_quality_report_summary(md_path: Path, report: dict) -> dict | None:
    if not isinstance(report, dict) or not report:
        return None
    repair = report.get("auto_repair") if isinstance(report.get("auto_repair"), dict) else {}
    repair_attempts = [item for item in list(report.get("repair_attempts") or []) if isinstance(item, dict)]
    latest_attempt = report.get("latest_repair_attempt") if isinstance(report.get("latest_repair_attempt"), dict) else (repair_attempts[-1] if repair_attempts else {})
    source_quality = report.get("source_quality") if isinstance(report.get("source_quality"), dict) else {}
    center_summary = quality_center_summary(report)
    try:
        stat = md_path.stat()
        stale = (
            int(report.get("md_mtime_ns") or 0) != int(stat.st_mtime_ns)
            or int(report.get("md_size") or 0) != int(stat.st_size)
        )
    except Exception:
        stale = True
    return {
        "available": True,
        "stale": bool(stale),
        "path": str(conversion_quality_result_path(md_path)),
        "generated_at": str(report.get("generated_at") or ""),
        "auto_repair_enabled": bool(report.get("auto_repair_enabled")),
        "auto_repair_changed": bool(repair.get("changed")),
        "auto_repair_unsafe": bool(repair.get("unsafe")),
        "auto_repair_applied": [str(item) for item in list(repair.get("applied") or []) if str(item or "").strip()][:20],
        "issue_codes_before": [str(item) for item in list(repair.get("issue_codes_before") or []) if str(item or "").strip()][:30],
        "remaining_issue_codes": [str(item) for item in list(repair.get("remaining_issue_codes") or []) if str(item or "").strip()][:30],
        "regression_reasons": [str(item) for item in list(repair.get("regression_reasons") or []) if str(item or "").strip()][:20],
        "repair_plan": report.get("repair_plan") if isinstance(report.get("repair_plan"), dict) else {},
        "repair_attempt_count": len(repair_attempts),
        "latest_repair_attempt": latest_attempt,
        "repair_attempts": repair_attempts[-5:],
        "recommended_action": str(report.get("recommended_action") or ""),
        "needs_reconvert": bool(report.get("needs_reconvert")),
        "source_quality": source_quality,
        "quality_center": center_summary,
        "source_quality_status": str(center_summary.get("status") or ""),
        "source_quality_message": str(center_summary.get("message") or ""),
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
        try:
            report_stat = conversion_quality_result_path(path).stat()
            report_mtime_ns = int(report_stat.st_mtime_ns)
            report_size = int(report_stat.st_size)
        except Exception:
            report_mtime_ns = 0
            report_size = 0
        cached = _CONVERSION_QUALITY_CACHE.get(cache_key)
        if (
            cached
            and len(cached) >= 5
            and cached[0] == int(stat.st_mtime_ns)
            and cached[1] == int(stat.st_size)
            and cached[2] == report_mtime_ns
            and cached[3] == report_size
        ):
            return dict(cached[4])

        metrics = summarize_conversion_quality(path)
        report_payload = load_conversion_quality_result(path)
        conversion_report = _conversion_quality_report_summary(path, report_payload)
        source_quality = (conversion_report or {}).get("source_quality") if isinstance(conversion_report, dict) else {}
        source_document_type = str((source_quality or {}).get("document_type") or "").strip().lower() if isinstance(source_quality, dict) else ""
        references_not_required = source_document_type == "supplementary"
        report_plan = report_payload.get("repair_plan") if isinstance(report_payload.get("repair_plan"), dict) else {}
        report_issue_codes = [
            str(code or "").strip().lower()
            for code in list((report_plan or {}).get("issue_codes") or [])
            if str(code or "").strip()
        ]
        if references_not_required:
            report_issue_codes = [
                code
                for code in report_issue_codes
                if code not in {"missing_abstract", "missing_references", "weak_structure"}
            ]
        report_action = str((report_plan or {}).get("action") or report_payload.get("recommended_action") or "").strip().lower()
        report_stale = bool((conversion_report or {}).get("stale")) if isinstance(conversion_report, dict) else True
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

        def issue_count_for_code(code: str) -> int:
            if code == "missing_images":
                return int(metrics.missing_image_count)
            if code == "unclosed_display_math":
                return int(metrics.unclosed_display_math_block_count)
            if code == "mojibake":
                return int(metrics.mojibake_count)
            if code == "analyzer_errors":
                return int(metrics.analyzer_error_count)
            if code == "weak_structure":
                return int(metrics.heading_count)
            if code == "missing_page_markers":
                return int(metrics.page_marker_count)
            if code == "page_marker_gaps":
                return int(metrics.page_marker_gap_count)
            if code == "missing_references":
                return int(metrics.extracted_reference_count or metrics.reference_line_count)
            if code == "missing_captions":
                return int(metrics.image_count)
            if code == "analyzer_warnings":
                return int(metrics.analyzer_warning_count)
            if code == "heading_level_jumps":
                return int(metrics.heading_level_jump_count)
            return 1

        def issue_label_for_code(code: str) -> str:
            strategy = conversion_repair_strategy_for_issue(code)
            return str(strategy.get("label") or code.replace("_", " ")).strip()

        def issue_severity_for_code(code: str) -> str:
            if code in {"source_text_loss", "missing_images", "mojibake", "analyzer_errors", "quality_scan_failed"}:
                return "error"
            return "warning"

        if conversion_report and not report_stale and (report_issue_codes or report_action == "none"):
            for code in report_issue_codes:
                add_issue(
                    code,
                    issue_label_for_code(code),
                    severity=issue_severity_for_code(code),
                    count=issue_count_for_code(code),
                    penalty=12 if issue_severity_for_code(code) == "error" else 6,
                )
        else:
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
            if (not references_not_required) and metrics.extracted_reference_count <= 0 and metrics.reference_line_count <= 0:
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
        reference_count = metrics.extracted_reference_count or metrics.reference_line_count
        reference_summary = "refs n/a" if references_not_required and reference_count <= 0 else f"{reference_count} refs"
        summary = (
            f"{label} | Q{score} | "
            f"{metrics.page_marker_count} pages | "
            f"{reference_summary} | "
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
            "conversion_report": conversion_report,
        }
        _CONVERSION_QUALITY_CACHE[cache_key] = (
            int(stat.st_mtime_ns),
            int(stat.st_size),
            report_mtime_ns,
            report_size,
            dict(result),
        )
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


def _load_docs_index_state() -> dict:
    db_dir: Path | None = None
    try:
        raw_db_dir = str(get_settings().db_dir or "").strip()
        if raw_db_dir:
            db_dir = Path(raw_db_dir).expanduser().resolve()
    except Exception:
        db_dir = None
    docs: dict = {}
    if db_dir is not None:
        try:
            docs = load_docs_index(db_dir)
        except Exception:
            docs = {}
    by_path: dict[str, tuple[str, dict]] = {}
    for doc_id, rec in (docs or {}).items():
        if not isinstance(rec, dict):
            continue
        key = _normalized_path_key(rec.get("path") or "")
        if key:
            by_path[key] = (str(rec.get("doc_id") or doc_id), rec)
    return {
        "db_dir": db_dir,
        "docs": docs if isinstance(docs, dict) else {},
        "by_path": by_path,
    }


def _library_markdown_index_state(md_path: Path | None, index_state: dict | None) -> dict:
    if not md_path:
        return {
            "index_state": "not_converted",
            "index_status": "",
            "index_ready": False,
            "index_doc_id": "",
            "index_path": "",
            "index_num_chunks": 0,
            "index_chunk_exists": False,
            "quality_gate": None,
        }
    state = index_state if isinstance(index_state, dict) else {}
    db_dir = state.get("db_dir") if isinstance(state.get("db_dir"), Path) else None
    docs = state.get("docs") if isinstance(state.get("docs"), dict) else {}
    by_path = state.get("by_path") if isinstance(state.get("by_path"), dict) else {}
    doc_id = compute_doc_id(md_path)
    rec = docs.get(doc_id) if isinstance(docs.get(doc_id), dict) else None
    record_doc_id = doc_id
    if not isinstance(rec, dict):
        key = _normalized_path_key(md_path)
        matched = by_path.get(key) if key else None
        if isinstance(matched, tuple) and len(matched) >= 2 and isinstance(matched[1], dict):
            record_doc_id = str(matched[0] or doc_id)
            rec = matched[1]

    if not isinstance(rec, dict):
        return {
            "index_state": "not_indexed",
            "index_status": "missing",
            "index_ready": False,
            "index_doc_id": doc_id,
            "index_path": "",
            "index_num_chunks": 0,
            "index_chunk_exists": False,
            "quality_gate": None,
        }

    status = str(rec.get("index_status") or "").strip().lower()
    num_chunks = int(rec.get("num_chunks") or 0)
    if not status:
        status = "ready" if num_chunks > 0 else "unknown"
    chunk_exists = False
    if db_dir is not None:
        try:
            chunk_exists = doc_chunks_path(db_dir, record_doc_id).exists()
        except Exception:
            chunk_exists = False
    quality_gate = rec.get("quality_gate") if isinstance(rec.get("quality_gate"), dict) else None
    if status == "ready" and chunk_exists and num_chunks > 0:
        normalized = "ready"
    elif status.startswith("quality_") or status in {"blocked", "not_indexable"}:
        normalized = "quality_blocked"
    elif status == "ready":
        normalized = "index_stale"
    elif num_chunks <= 0 or not chunk_exists:
        normalized = "not_indexed"
    else:
        normalized = "not_ready"
    return {
        "index_state": normalized,
        "index_status": status,
        "index_ready": normalized == "ready",
        "index_doc_id": record_doc_id,
        "index_path": str(rec.get("path") or ""),
        "index_num_chunks": num_chunks,
        "index_chunk_exists": bool(chunk_exists),
        "quality_gate": quality_gate,
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


def _resolve_library_pdf_path_arg(path_raw: str) -> Path:
    raw = str(path_raw or "").strip()
    if not raw:
        raise HTTPException(400, "path required")
    pdf_d = _pdf_dir()
    try:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = pdf_d / candidate
        resolved = candidate.resolve(strict=False)
    except Exception as exc:
        raise HTTPException(400, f"invalid path: {exc}") from exc
    if resolved.suffix.lower() != ".pdf":
        raise HTTPException(400, "path must point to a PDF")
    if not _path_is_within(resolved, [pdf_d]):
        raise HTTPException(400, "path must be within the configured PDF directory")
    return resolved


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
    docs_index_state: dict | None = None,
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
    md_index = _library_markdown_index_state(md_main if md_exists else None, docs_index_state)
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
        **md_index,
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
    docs_index_state = _load_docs_index_state()
    items = [
        _library_file_item(
            pdf,
            md_root=md_dir,
            task_by_path=task_by_path,
            task_by_name=task_by_name,
            docs_index_state=docs_index_state,
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
    index_ready = sum(1 for item in items if str(item.get("index_state") or "") == "ready")
    index_quality_blocked = sum(1 for item in items if str(item.get("index_state") or "") == "quality_blocked")
    index_stale = sum(1 for item in items if str(item.get("index_state") or "") in {"index_stale", "not_indexed", "not_ready"})

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
            "index_ready": int(index_ready),
            "index_quality_blocked": int(index_quality_blocked),
            "index_stale": int(index_stale),
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


def _research_qa_quality_issue_rows(summary: dict, key: str = "failures") -> dict[int, list[dict]]:
    rows: dict[int, list[dict]] = {}
    source = summary if isinstance(summary, dict) else {}
    for item in list(source.get(key) or []):
        if not isinstance(item, dict):
            continue
        index = _safe_int(item.get("index"), 0)
        if index <= 0:
            continue
        issue = {
            "name": _compact_text(item.get("name"), limit=120),
            "field": _compact_text(item.get("field"), limit=120),
            "detail": _compact_text(item.get("detail"), limit=220),
            "severity": _compact_text(item.get("severity"), limit=40) or ("warning" if key == "warnings" else "error"),
        }
        if not issue["name"]:
            continue
        rows.setdefault(index, []).append(issue)
    return rows


_SHELF_EXPORT_FIELD_ALIASES = {
    "author": "authors",
    "authors": "authors",
    "venue": "venue",
    "journal": "venue",
    "conference": "venue",
    "year": "year",
    "doi": "doi",
    "title": "title",
    "source": "source",
    "source_path": "source",
    "source_name": "source",
}


def _research_qa_shelf_issue_field(issue: dict) -> str:
    name = str((issue or {}).get("name") or "").strip().lower()
    field = str((issue or {}).get("field") or "").strip().lower()
    if name.startswith("shelf_export_missing_"):
        field = name.removeprefix("shelf_export_missing_")
    elif name == "shelf_missing_author_hint":
        field = "authors"
    elif name == "shelf_missing_venue_hint":
        field = "venue"
    elif name == "shelf_missing_year_hint":
        field = "year"
    elif name in {"shelf_missing_doi", "shelf_doi_not_promoted"}:
        field = "doi"
    elif name in {"shelf_missing_source_identity", "shelf_source_not_clickable", "shelf_system_a_missing_source"}:
        field = "source"
    elif name in {"shelf_title_too_short", "shelf_weak_generic_title"}:
        field = "title"
    return _SHELF_EXPORT_FIELD_ALIASES.get(field, "")


def _research_qa_shelf_missing_fields(issues: list[dict]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for issue in list(issues or []):
        if not isinstance(issue, dict):
            continue
        field = _research_qa_shelf_issue_field(issue)
        if not field or field in seen:
            continue
        seen.add(field)
        fields.append(field)
    return fields


def _research_qa_shelf_missing_field_counts(citations: list[dict]) -> list[dict]:
    counter: Counter = Counter()
    for item in list(citations or []):
        if not isinstance(item, dict):
            continue
        for field in _list_strings(item.get("metadata_missing_fields")):
            counter[field] += 1
    return _counter_items(counter, limit=8)


def _research_qa_shelf_missing_field_counts_from_quality(shelf_quality: dict) -> list[dict]:
    counter: Counter = Counter()
    source = shelf_quality if isinstance(shelf_quality, dict) else {}
    for issue in [*_list_dict_items(source.get("failures")), *_list_dict_items(source.get("warnings"))]:
        field = _research_qa_shelf_issue_field(issue)
        if field:
            counter[field] += 1
    return _counter_items(counter, limit=8)


def _research_qa_quality_gate_summary(quality: dict) -> dict:
    source = quality if isinstance(quality, dict) else {}
    citation_quality = source.get("citation_quality") if isinstance(source.get("citation_quality"), dict) else {}
    shelf_quality = source.get("citation_shelf_quality") if isinstance(source.get("citation_shelf_quality"), dict) else {}
    ref_card_quality = source.get("ref_card_quality") if isinstance(source.get("ref_card_quality"), dict) else {}
    system_b_audit = source.get("system_b_audit") if isinstance(source.get("system_b_audit"), dict) else {}
    missing_fields = _research_qa_shelf_missing_field_counts_from_quality(shelf_quality)
    return {
        "citation_card_failure_count": len(_list_dict_items(citation_quality.get("failures"))),
        "citation_card_warning_count": len(_list_dict_items(citation_quality.get("warnings"))),
        "shelf_failure_count": len(_list_dict_items(shelf_quality.get("failures"))),
        "shelf_warning_count": len(_list_dict_items(shelf_quality.get("warnings"))),
        "shelf_metadata_ready_count": _safe_int(shelf_quality.get("metadata_ready_count"), 0),
        "shelf_export_ready_count": _safe_int(shelf_quality.get("export_ready_count"), 0),
        "shelf_summary_export_ready_count": _safe_int(shelf_quality.get("summary_export_ready_count"), 0),
        "shelf_doi_count": _safe_int(shelf_quality.get("doi_count"), 0),
        "shelf_source_clickable_count": _safe_int(shelf_quality.get("source_clickable_count"), 0),
        "shelf_review_count": _safe_int(shelf_quality.get("review_count"), 0),
        "shelf_missing_export_fields": missing_fields,
        "ref_card_failure_count": len(_list_dict_items(ref_card_quality.get("failures"))),
        "ref_card_warning_count": len(_list_dict_items(ref_card_quality.get("warnings"))),
        "system_b_needs_review_count": _safe_int(system_b_audit.get("needs_review_count"), 0),
        "system_b_answer_context_only_count": _safe_int(system_b_audit.get("answer_context_only_count"), 0),
        "system_b_reference_index_fallback_count": _safe_int(system_b_audit.get("reference_index_fallback_count"), 0),
    }


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
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    citation_quality = quality.get("citation_quality") if isinstance(quality.get("citation_quality"), dict) else {}
    shelf_quality = quality.get("citation_shelf_quality") if isinstance(quality.get("citation_shelf_quality"), dict) else {}
    card_failures = _research_qa_quality_issue_rows(citation_quality, "failures")
    card_warnings = _research_qa_quality_issue_rows(citation_quality, "warnings")
    shelf_failures = _research_qa_quality_issue_rows(shelf_quality, "failures")
    shelf_warnings = _research_qa_quality_issue_rows(shelf_quality, "warnings")
    for index, item in enumerate(_research_qa_citation_details(row)[:limit], start=1):
        is_system_b = bool(item.get("is_inpaper")) or str(item.get("route") or "").strip().lower() == "system_b"
        quality_issues = [
            *card_failures.get(index, []),
            *card_warnings.get(index, []),
        ]
        shelf_quality_issues = [
            *shelf_failures.get(index, []),
            *shelf_warnings.get(index, []),
        ]
        metadata_missing_fields = _research_qa_shelf_missing_fields(shelf_quality_issues)
        raw_text = _compact_text(item.get("raw"), item.get("cite_fmt"), item.get("citeFmt"), item.get("card_reference_entry"), item.get("cardReferenceEntry"), limit=420)
        summary_quality = item.get("summary_quality") if isinstance(item.get("summary_quality"), dict) else (
            item.get("summaryQuality") if isinstance(item.get("summaryQuality"), dict) else {}
        )
        out.append(
            {
                "route": "system_b" if is_system_b else "system_a",
                "num": _safe_int(item.get("num") or item.get("ref_num"), 0),
                "anchor": _compact_text(item.get("anchor"), item.get("anchor_id"), item.get("block_id"), limit=120),
                "title": _compact_text(item.get("title"), item.get("source_name"), limit=180),
                "source_name": _compact_text(item.get("source_name"), limit=180),
                "source_path": _compact_text(item.get("source_path"), limit=260),
                "authors": _compact_text(item.get("authors"), item.get("external_authors"), item.get("externalAuthors"), limit=220),
                "venue": _compact_text(item.get("venue"), item.get("external_venue"), item.get("externalVenue"), limit=180),
                "year": _compact_text(item.get("year"), item.get("published_year"), item.get("publishedYear"), limit=24),
                "doi": _compact_text(item.get("doi"), item.get("external_doi"), item.get("externalDoi"), limit=160),
                "doi_url": _compact_text(item.get("doi_url"), item.get("doiUrl"), item.get("external_doi_url"), item.get("externalDoiUrl"), limit=220),
                "raw": raw_text,
                "cite_fmt": _compact_text(item.get("cite_fmt"), item.get("citeFmt"), limit=420),
                "summary_line": _compact_text(item.get("summary_line"), item.get("summaryLine"), item.get("card_takeaway"), item.get("upstream_work_role"), item.get("user_question_relation"), limit=260),
                "summary_quality": dict(summary_quality),
                "heading_path": _compact_text(item.get("heading_path"), item.get("location_label"), limit=180),
                "evidence_quote": _compact_text(item.get("evidence_quote"), item.get("citation_context"), limit=260),
                "answer_claim": _compact_text(item.get("answer_claim"), limit=220),
                "support_relation": _compact_text(item.get("support_relation"), item.get("user_question_relation"), limit=180),
                "trace": _compact_text(item.get("citation_context_source"), item.get("mapping_source"), item.get("anchor_kind"), limit=120),
                "quality_issues": quality_issues,
                "shelf_quality_issues": shelf_quality_issues,
                "metadata_missing_fields": metadata_missing_fields,
                "metadata_repairable": bool(is_system_b and metadata_missing_fields),
                "quality_issue_count": len(quality_issues) + len(shelf_quality_issues),
            }
        )
    return out


def _research_qa_ref_diagnostics(row: dict, *, limit: int = 8) -> list[dict]:
    out: list[dict] = []
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    ref_card_quality = quality.get("ref_card_quality") if isinstance(quality.get("ref_card_quality"), dict) else {}
    ref_failures = _research_qa_quality_issue_rows(ref_card_quality, "failures")
    ref_warnings = _research_qa_quality_issue_rows(ref_card_quality, "warnings")
    for index, hit in enumerate(_research_qa_ref_hits(row)[:limit], start=1):
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        citation_meta = (ui_meta or {}).get("citation_meta") if isinstance((ui_meta or {}).get("citation_meta"), dict) else {}
        summary_quality = (ui_meta or {}).get("summary_quality") if isinstance((ui_meta or {}).get("summary_quality"), dict) else {}
        score = (ui_meta or {}).get("score")
        if score in (None, ""):
            score = hit.get("score")
        quality_issues = [
            *ref_failures.get(index, []),
            *ref_warnings.get(index, []),
        ]
        out.append(
            {
                "title": _compact_text((ui_meta or {}).get("display_name"), (citation_meta or {}).get("title"), (meta or {}).get("source_name"), limit=180),
                "source_name": _compact_text((ui_meta or {}).get("source_name"), (meta or {}).get("source_name"), limit=180),
                "source_path": _compact_text((ui_meta or {}).get("source_path"), (citation_meta or {}).get("source_path"), (meta or {}).get("source_path"), limit=260),
                "authors": _compact_text((citation_meta or {}).get("authors"), (ui_meta or {}).get("authors"), limit=220),
                "venue": _compact_text((citation_meta or {}).get("venue"), (ui_meta or {}).get("venue"), limit=180),
                "year": _compact_text((citation_meta or {}).get("year"), (ui_meta or {}).get("year"), limit=24),
                "doi": _compact_text((citation_meta or {}).get("doi"), (ui_meta or {}).get("doi"), limit=160),
                "doi_url": _compact_text((citation_meta or {}).get("doi_url"), (citation_meta or {}).get("doiUrl"), (ui_meta or {}).get("doi_url"), (ui_meta or {}).get("doiUrl"), limit=220),
                "raw": _compact_text((citation_meta or {}).get("raw"), (citation_meta or {}).get("cite_fmt"), (ui_meta or {}).get("raw"), hit.get("text"), limit=420),
                "cite_fmt": _compact_text((citation_meta or {}).get("cite_fmt"), (citation_meta or {}).get("citeFmt"), limit=420),
                "heading_path": _compact_text((ui_meta or {}).get("heading_path"), (meta or {}).get("heading_path"), limit=180),
                "score": round(_safe_float(score, 0.0), 3),
                "summary_line": _compact_text((ui_meta or {}).get("summary_line"), limit=220),
                "summary_quality": dict(summary_quality),
                "why_line": _compact_text((ui_meta or {}).get("why_line"), limit=220),
                "polish_status": _compact_text((ui_meta or {}).get("polish_status"), limit=80),
                "ref_pack_state": _compact_text((meta or {}).get("ref_pack_state"), limit=80),
                "evidence_quote": _compact_text(hit.get("text"), limit=260),
                "quality_issues": quality_issues,
                "quality_issue_count": len(quality_issues),
            }
        )
    return out


def _research_qa_shelf_metadata_repair_targets(citations: list[dict], refs: list[dict], *, limit: int = 12) -> list[dict]:
    targets: list[dict] = []
    seen: set[str] = set()

    def add(item: dict, *, kind: str) -> None:
        source_path = _compact_text(item.get("source_path"), limit=260)
        source_name = _compact_text(item.get("source_name"), item.get("title"), limit=180)
        title = _compact_text(item.get("title"), item.get("source_name"), limit=180)
        raw = _compact_text(item.get("raw"), item.get("cite_fmt"), item.get("evidence_quote"), limit=420)
        key = "|".join([source_path, source_name, title, raw[:120]]).lower()
        if not key.strip("|") or key in seen:
            return
        seen.add(key)
        target = {
            "key": _compact_text(item.get("key"), item.get("anchor"), f"{kind}:{len(targets) + 1}", limit=160),
            "source_path": source_path,
            "source_name": source_name,
            "title": title,
            "authors": _compact_text(item.get("authors"), limit=220),
            "venue": _compact_text(item.get("venue"), limit=180),
            "year": _compact_text(item.get("year"), limit=24),
            "doi": _compact_text(item.get("doi"), limit=160),
            "doi_url": _compact_text(item.get("doi_url"), limit=220),
            "raw": raw,
            "cite_fmt": _compact_text(item.get("cite_fmt"), limit=420),
            "summary_line": _compact_text(item.get("summary_line"), item.get("why_line"), limit=260),
            "summary_quality": dict(item.get("summary_quality") or {}) if isinstance(item.get("summary_quality"), dict) else {},
            "metadata_missing_fields": _list_strings(item.get("metadata_missing_fields")),
            "repair_target_kind": kind,
        }
        targets.append(target)

    for item in list(citations or []):
        if not isinstance(item, dict):
            continue
        route = str(item.get("route") or "").strip().lower()
        missing_fields = _list_strings(item.get("metadata_missing_fields"))
        has_shelf_issue = any(
            str(issue.get("name") or "").startswith("shelf_")
            for issue in _list_dict_items(item.get("shelf_quality_issues"))
        )
        if route == "system_b" and (missing_fields or has_shelf_issue):
            add(item, kind="system_b_citation")
        if len(targets) >= int(limit):
            return targets[:limit]

    if targets:
        return targets[:limit]

    for item in list(refs or []):
        if not isinstance(item, dict):
            continue
        if _list_dict_items(item.get("quality_issues")):
            add(item, kind="reference_card")
        if len(targets) >= int(limit):
            break
    return targets[:limit]


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
    citation_diagnostics: list[dict] | None = None,
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

    shelf_missing_fields = _research_qa_shelf_missing_field_counts(list(citation_diagnostics or []))
    if shelf_missing_fields:
        field_detail = ", ".join(f"{item.get('name')} x{item.get('count')}" for item in shelf_missing_fields[:5])
        add(
            "shelf_metadata_export_fields",
            "Literature basket metadata is not export-ready",
            severity="error",
            detail=f"Missing structured fields: {field_detail}.",
            action="repair_shelf_metadata",
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
    shelf_metadata_targets: list[dict] | None = None,
    shelf_missing_fields: list[dict] | None = None,
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
    if "citation_card_quality" in cause_codes or "system_b_mapping" in cause_codes or "shelf_metadata_export_fields" in cause_codes:
        steps.append(
            {
                "kind": "repair_shelf_metadata",
                "label": "Repair citation and shelf metadata",
                "target_count": len(list(shelf_metadata_targets or [])),
                "missing_fields": list(shelf_missing_fields or [])[:8],
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
    shelf_metadata_targets: list[dict] | None = None,
    shelf_missing_fields: list[dict] | None = None,
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
        shelf_metadata_targets=shelf_metadata_targets,
        shelf_missing_fields=shelf_missing_fields,
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
                "detail": (
                    "Run the diagnostic repair plan: " + " -> ".join([item for item in step_labels if item])
                    + (
                        "; metadata fields "
                        + ", ".join(f"{field.get('name')} x{field.get('count')}" for field in list(shelf_missing_fields or [])[:5])
                        if shelf_missing_fields else ""
                    )
                ),
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

    if "citation_card_quality" in cause_codes or "system_b_mapping" in cause_codes or "shelf_metadata_export_fields" in cause_codes:
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


def _quality_repair_runs_path() -> Path:
    return _RESEARCH_QA_EVAL_ROOT / "repair_runs.jsonl"


def _reader_locate_events_path() -> Path:
    return _RESEARCH_QA_EVAL_ROOT / "reader_locate_events.jsonl"


_READER_LOCATE_GOOD_STATUSES = {"exact", "block"}
_READER_LOCATE_DEGRADED_STATUSES = {"fuzzy", "section", "source_only"}
_READER_LOCATE_STATUSES = _READER_LOCATE_GOOD_STATUSES | _READER_LOCATE_DEGRADED_STATUSES | {"failed"}
_READER_LOCATE_PRECISIONS = {"exact_anchor", "block", "phrase", "fuzzy", "section", "source_only", "failed"}


def _reader_locate_source_key(row: dict) -> str:
    for field in ("pdf_path", "md_path", "source_path"):
        value = str((row or {}).get(field) or "").strip()
        key = _normalized_path_key(value)
        if key:
            return key.lower()
    name = str((row or {}).get("source_name") or "").strip().lower()
    return name


def _reader_locate_identity(row: dict) -> str:
    feedback_key = str((row or {}).get("locate_feedback_key") or "").strip()
    if feedback_key:
        return f"feedback:{feedback_key}"
    parts = [
        _reader_locate_source_key(row),
        str((row or {}).get("locate_request_id") or "").strip(),
        str((row or {}).get("block_id") or "").strip(),
        str((row or {}).get("anchor_id") or "").strip(),
        str((row or {}).get("heading_path") or "").strip(),
    ]
    clean = "|".join(part for part in parts if part)
    return clean or str((row or {}).get("id") or "")


def _reader_locate_recommended_action(row: dict) -> str:
    status = str((row or {}).get("status") or "").strip().lower()
    precision = str((row or {}).get("precision") or "").strip().lower()
    md_exists = bool((row or {}).get("md_exists"))
    strict = bool((row or {}).get("strict_locate"))
    if not md_exists:
        return "reconvert_source"
    if status == "failed" or precision == "failed":
        return "repair_conversion_and_reindex"
    if status == "source_only" or precision == "source_only":
        return "rebuild_source_anchors"
    if status in {"fuzzy", "section"} or precision in {"fuzzy", "section"} or (strict and status not in _READER_LOCATE_GOOD_STATUSES):
        return "repair_anchors_and_evidence"
    return "review_reader_locate"


def _normalize_reader_locate_event(row: dict) -> dict:
    if not isinstance(row, dict):
        return {}
    source_path = _compact_text(row.get("source_path"), limit=500)
    source_name = _compact_text(row.get("source_name"), limit=240)
    resolved = _resolve_quality_source(source_path=source_path, source_name=source_name)
    status = _compact_text(row.get("status"), limit=40).lower()
    precision = _compact_text(row.get("precision"), limit=40).lower()
    if status not in _READER_LOCATE_STATUSES:
        status = "exact" if bool(row.get("ok")) else "failed"
    if precision not in _READER_LOCATE_PRECISIONS:
        precision = "phrase" if status == "exact" else status
    ok = bool(row.get("ok")) if "ok" in row else status in _READER_LOCATE_GOOD_STATUSES | _READER_LOCATE_DEGRADED_STATUSES
    repairable = bool(row.get("repairable")) or status == "failed" or (
        bool(row.get("strict_locate")) and status not in _READER_LOCATE_GOOD_STATUSES
    )
    created_at = _safe_int(row.get("created_at"), 0) or int(time.time())
    out = {
        "id": _compact_text(row.get("id"), limit=80) or uuid.uuid4().hex,
        "created_at": created_at,
        "source_path": source_path,
        "source_name": source_name or str(resolved.get("source_name") or ""),
        "pdf_path": str(resolved.get("pdf_path") or ""),
        "md_path": str(resolved.get("md_path") or ""),
        "md_exists": bool(resolved.get("md_exists")),
        "locate_feedback_key": _compact_text(row.get("locate_feedback_key"), limit=160),
        "locate_request_id": _safe_int(row.get("locate_request_id"), 0),
        "status": status,
        "precision": precision,
        "ok": bool(ok),
        "repairable": bool(repairable),
        "strict_locate": bool(row.get("strict_locate")),
        "hint": _compact_text(row.get("hint"), limit=300),
        "reason": _compact_text(row.get("reason"), limit=500),
        "active_alt_index": _safe_int(row.get("active_alt_index"), 0),
        "block_id": _compact_text(row.get("block_id"), limit=160),
        "anchor_id": _compact_text(row.get("anchor_id"), limit=160),
        "anchor_kind": _compact_text(row.get("anchor_kind"), limit=80),
        "heading_path": _compact_text(row.get("heading_path"), limit=300),
    }
    if not out["source_path"] and not out["source_name"] and not out["locate_feedback_key"]:
        return {}
    out["source_key"] = _reader_locate_source_key(out)
    out["recommended_action"] = _reader_locate_recommended_action(out)
    return out


def _reader_locate_event_rows(*, limit: int = 1000) -> list[dict]:
    rows = _read_jsonl_artifact(_reader_locate_events_path(), limit=max(1000, min(10000, int(limit or 1000) * 4)))
    latest_by_identity: dict[str, dict] = {}
    for raw in rows:
        row = _normalize_reader_locate_event(raw)
        identity = _reader_locate_identity(row)
        if not identity:
            continue
        prev = latest_by_identity.get(identity)
        if not prev or _safe_int(row.get("created_at"), 0) >= _safe_int(prev.get("created_at"), 0):
            latest_by_identity[identity] = row
    out = list(latest_by_identity.values())
    out.sort(key=lambda item: (_safe_int(item.get("created_at"), 0), str(item.get("id") or "")), reverse=True)
    return out[: max(0, min(2000, int(limit or 1000)))]


def _append_reader_locate_event(record: dict) -> dict:
    row = _normalize_reader_locate_event(record)
    if not row:
        raise HTTPException(400, "source_path, source_name, or locate_feedback_key is required")
    try:
        path = _reader_locate_events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"failed to write reader locate event: {exc}") from exc
    return row


def _reader_locate_quality_summary(rows: list[dict] | None = None) -> dict:
    items = list(rows or _reader_locate_event_rows(limit=1000))
    if not items:
        return {
            "available": False,
            "status": "unknown",
            "summary": {
                "total": 0,
                "exact": 0,
                "block": 0,
                "degraded": 0,
                "failed": 0,
                "repairable": 0,
                "strict_miss": 0,
                "affected_sources": 0,
            },
            "top_failures": [],
            "recommended_sources": [],
            "latest": [],
        }

    total = len(items)
    exact = sum(1 for item in items if str(item.get("status") or "") == "exact")
    block = sum(1 for item in items if str(item.get("status") or "") == "block")
    degraded = sum(1 for item in items if str(item.get("status") or "") in _READER_LOCATE_DEGRADED_STATUSES)
    failed = sum(1 for item in items if str(item.get("status") or "") == "failed" or str(item.get("precision") or "") == "failed")
    strict_miss = sum(
        1
        for item in items
        if bool(item.get("strict_locate")) and str(item.get("status") or "") not in _READER_LOCATE_GOOD_STATUSES
    )
    repairable = sum(
        1
        for item in items
        if bool(item.get("repairable")) or str(item.get("status") or "") == "failed" or bool(item.get("strict_locate")) and str(item.get("status") or "") not in _READER_LOCATE_GOOD_STATUSES
    )

    reason_counter: Counter = Counter()
    source_stats: dict[str, dict] = {}
    for item in items:
        status = str(item.get("status") or "")
        if status in _READER_LOCATE_DEGRADED_STATUSES or status == "failed":
            reason = str(item.get("reason") or item.get("hint") or status).strip() or status
            reason_counter[reason] += 1
        source_key = _reader_locate_source_key(item)
        if not source_key:
            continue
        cur = source_stats.get(source_key) or {
            "source_path": str(item.get("source_path") or ""),
            "source_name": str(item.get("source_name") or ""),
            "pdf_path": str(item.get("pdf_path") or ""),
            "md_path": str(item.get("md_path") or ""),
            "md_exists": bool(item.get("md_exists")),
            "total": 0,
            "failed": 0,
            "degraded": 0,
            "repairable": 0,
            "strict_miss": 0,
            "latest_status": "",
            "latest_precision": "",
            "latest_reason": "",
            "latest_at": 0,
            "recommended_action": "",
        }
        cur["total"] = _safe_int(cur.get("total"), 0) + 1
        if status == "failed":
            cur["failed"] = _safe_int(cur.get("failed"), 0) + 1
        if status in _READER_LOCATE_DEGRADED_STATUSES:
            cur["degraded"] = _safe_int(cur.get("degraded"), 0) + 1
        if bool(item.get("repairable")) or status == "failed":
            cur["repairable"] = _safe_int(cur.get("repairable"), 0) + 1
        if bool(item.get("strict_locate")) and status not in _READER_LOCATE_GOOD_STATUSES:
            cur["strict_miss"] = _safe_int(cur.get("strict_miss"), 0) + 1
        created_at = _safe_int(item.get("created_at"), 0)
        problem = (
            status == "failed"
            or status in _READER_LOCATE_DEGRADED_STATUSES
            or bool(item.get("repairable"))
            or (bool(item.get("strict_locate")) and status not in _READER_LOCATE_GOOD_STATUSES)
        )
        if problem:
            cur["recommended_action"] = str(item.get("recommended_action") or _reader_locate_recommended_action(item))
            if not str(cur.get("latest_reason") or "").strip():
                cur["latest_reason"] = str(item.get("reason") or item.get("hint") or "")
        if created_at >= _safe_int(cur.get("latest_at"), 0):
            cur["latest_status"] = status
            cur["latest_precision"] = str(item.get("precision") or "")
            if problem or not str(cur.get("latest_reason") or "").strip():
                cur["latest_reason"] = str(item.get("reason") or item.get("hint") or "")
            cur["latest_at"] = created_at
            if not str(cur.get("recommended_action") or "").strip():
                cur["recommended_action"] = str(item.get("recommended_action") or _reader_locate_recommended_action(item))
        source_stats[source_key] = cur

    recommended_sources = [
        item
        for item in source_stats.values()
        if _safe_int(item.get("failed"), 0) > 0
        or _safe_int(item.get("degraded"), 0) > 0
        or _safe_int(item.get("repairable"), 0) > 0
        or _safe_int(item.get("strict_miss"), 0) > 0
    ]
    recommended_sources.sort(
        key=lambda item: (
            -_safe_int(item.get("failed"), 0),
            -_safe_int(item.get("strict_miss"), 0),
            -_safe_int(item.get("degraded"), 0),
            -_safe_int(item.get("repairable"), 0),
            -_safe_int(item.get("latest_at"), 0),
            str(item.get("source_name") or "").lower(),
        )
    )
    status = "error" if failed > 0 or strict_miss > 0 else ("warning" if degraded > 0 or repairable > 0 else "good")
    return {
        "available": True,
        "status": status,
        "summary": {
            "total": int(total),
            "exact": int(exact),
            "block": int(block),
            "degraded": int(degraded),
            "failed": int(failed),
            "repairable": int(repairable),
            "strict_miss": int(strict_miss),
            "affected_sources": len(source_stats),
        },
        "top_failures": [{"name": name, "count": count} for name, count in reason_counter.most_common(6)],
        "recommended_sources": recommended_sources[:8],
        "latest": items[:10],
    }


def _reader_locate_all_event_rows(*, limit: int = 4000) -> list[dict]:
    rows = _read_jsonl_artifact(_reader_locate_events_path(), limit=max(1000, min(10000, int(limit or 4000))))
    out: list[dict] = []
    for raw in rows:
        row = _normalize_reader_locate_event(raw)
        if row:
            out.append(row)
    out.sort(key=lambda item: (_safe_int(item.get("created_at"), 0), str(item.get("id") or "")), reverse=True)
    return out


def _reader_locate_event_problem(row: dict) -> bool:
    status = str((row or {}).get("status") or "").strip().lower()
    precision = str((row or {}).get("precision") or "").strip().lower()
    strict = bool((row or {}).get("strict_locate"))
    return (
        status == "failed"
        or precision == "failed"
        or status in _READER_LOCATE_DEGRADED_STATUSES
        or precision in _READER_LOCATE_DEGRADED_STATUSES
        or bool((row or {}).get("repairable"))
        or (strict and status not in _READER_LOCATE_GOOD_STATUSES)
    )


def _reader_locate_event_tokens(row: dict) -> set[str]:
    tokens: set[str] = set()
    for field in ("source_path", "source_name", "pdf_path", "md_path"):
        text = str((row or {}).get(field) or "").strip()
        if not text:
            continue
        folded = text.replace("\\", "/").lower()
        tokens.add(folded)
        base = Path(folded).name
        if base:
            tokens.add(base)
            stem = _strip_known_source_ext(base).lower()
            if stem:
                tokens.add(stem)
        key = _normalized_path_key(text).replace("\\", "/").lower()
        if key:
            tokens.add(key)
    return {token for token in tokens if token}


def _reader_locate_run_problem_events(run: dict, rows: list[dict]) -> list[dict]:
    run_tokens = _quality_repair_run_match_tokens(run)
    if not run_tokens:
        return []
    run_created_at = _safe_int((run or {}).get("created_at"), 0)
    latest_problem_by_identity: dict[str, dict] = {}
    for row in rows:
        if not _reader_locate_event_problem(row):
            continue
        created_at = _safe_int(row.get("created_at"), 0)
        if run_created_at > 0 and created_at > run_created_at:
            continue
        if not run_tokens.intersection(_reader_locate_event_tokens(row)):
            continue
        identity = _reader_locate_identity(row)
        if not identity:
            continue
        prev = latest_problem_by_identity.get(identity)
        if not prev or created_at >= _safe_int(prev.get("created_at"), 0):
            latest_problem_by_identity[identity] = row
    out = list(latest_problem_by_identity.values())
    out.sort(key=lambda item: (_safe_int(item.get("created_at"), 0), str(item.get("id") or "")), reverse=True)
    return out[:40]


def _reader_locate_source_problem_events(
    *,
    source_path: str = "",
    source_name: str = "",
    pdf_path: str = "",
    md_path: str = "",
    limit: int = 20,
) -> list[dict]:
    target_tokens = _reader_locate_event_tokens(
        {
            "source_path": source_path,
            "source_name": source_name,
            "pdf_path": pdf_path,
            "md_path": md_path,
        }
    )
    if not target_tokens:
        return []
    rows = _reader_locate_all_event_rows(limit=4000)
    latest_problem_by_identity: dict[str, dict] = {}
    for row in rows:
        if not _reader_locate_event_problem(row):
            continue
        if not target_tokens.intersection(_reader_locate_event_tokens(row)):
            continue
        identity = _reader_locate_identity(row)
        if not identity:
            continue
        prev = latest_problem_by_identity.get(identity)
        if not prev or _safe_int(row.get("created_at"), 0) >= _safe_int(prev.get("created_at"), 0):
            latest_problem_by_identity[identity] = row
    out = list(latest_problem_by_identity.values())
    out = [
        problem
        for problem in out
        if not _reader_locate_newer_good_event(
            problem,
            rows,
            after_at=_safe_int(problem.get("created_at"), 0),
        )
    ]
    out.sort(key=lambda item: (_safe_int(item.get("created_at"), 0), str(item.get("id") or "")), reverse=True)
    return out[: max(0, min(80, int(limit or 20)))]


def _reader_locate_newer_good_event(problem: dict, rows: list[dict], *, after_at: int) -> dict:
    identity = _reader_locate_identity(problem)
    problem_source_key = str(problem.get("source_key") or _reader_locate_source_key(problem)).lower()
    candidates: list[dict] = []
    for row in rows:
        created_at = _safe_int(row.get("created_at"), 0)
        if after_at > 0 and created_at <= after_at:
            continue
        status = str(row.get("status") or "").strip().lower()
        if status not in _READER_LOCATE_GOOD_STATUSES:
            continue
        same_identity = bool(identity) and _reader_locate_identity(row) == identity
        same_source = problem_source_key and str(row.get("source_key") or _reader_locate_source_key(row)).lower() == problem_source_key
        if same_identity or (not identity.startswith("feedback:") and same_source):
            candidates.append(row)
    candidates.sort(key=lambda item: (_safe_int(item.get("created_at"), 0), str(item.get("id") or "")), reverse=True)
    return candidates[0] if candidates else {}


def _fold_reader_locate_heading(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
    text = re.sub(r"[#*_`~>\[\]().,:;|\\/-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _reader_locate_source_target_check(row: dict) -> dict:
    source_path = str((row or {}).get("source_path") or row.get("md_path") or row.get("pdf_path") or "").strip()
    source_name = str((row or {}).get("source_name") or "").strip()
    resolved = _resolve_quality_source(source_path=source_path, source_name=source_name)
    md_path_raw = str(resolved.get("md_path") or "").strip()
    if not bool(resolved.get("md_exists")) or not md_path_raw:
        return {
            "status": "failed",
            "reason": "markdown source is missing after repair",
            "md_path": md_path_raw,
            "source_quality_status": "missing",
            "source_quality_score": 0,
            "checks": {"md_exists": False},
        }

    md_path = Path(md_path_raw).expanduser()
    quality = _conversion_quality_summary(md_path) or {}
    quality_status = str(quality.get("status") or "unknown").strip().lower()
    quality_score = _safe_int(quality.get("score"), 0)
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"markdown source cannot be read: {exc}",
            "md_path": md_path_raw,
            "source_quality_status": quality_status,
            "source_quality_score": quality_score,
            "checks": {"md_exists": True, "readable": False},
        }

    anchor_id = str((row or {}).get("anchor_id") or "").strip()
    block_id = str((row or {}).get("block_id") or "").strip()
    heading_path = str((row or {}).get("heading_path") or "").strip()
    target_ids = [value for value in [anchor_id, block_id] if value]
    missing_ids = [value for value in target_ids if value not in text]
    found_ids = [value for value in target_ids if value and value in text]

    headings = [_fold_reader_locate_heading(match.group(1)) for match in re.finditer(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$", text)]
    heading_parts = [
        _fold_reader_locate_heading(part)
        for part in re.split(r"\s*(?:/|>|→|\||::)\s*", heading_path)
        if _fold_reader_locate_heading(part)
    ]
    heading_found = False
    if heading_parts:
        for part in reversed(heading_parts[-3:]):
            if any(part and (part == heading or part in heading or heading in part) for heading in headings):
                heading_found = True
                break

    checks = {
        "md_exists": True,
        "readable": True,
        "found_ids": found_ids,
        "missing_ids": missing_ids,
        "heading_found": heading_found,
        "source_quality_status": quality_status,
        "source_quality_score": quality_score,
    }
    if missing_ids:
        return {
            "status": "failed",
            "reason": "target anchor/block is still missing after repair",
            "md_path": md_path_raw,
            "source_quality_status": quality_status,
            "source_quality_score": quality_score,
            "checks": checks,
        }
    if found_ids or heading_found:
        return {
            "status": "passed",
            "reason": "target anchor/block or heading is present after repair",
            "md_path": md_path_raw,
            "source_quality_status": quality_status,
            "source_quality_score": quality_score,
            "checks": checks,
        }
    if quality_status == "good" and not bool(quality.get("has_review_issue")):
        return {
            "status": "needs_reader_reopen",
            "reason": "source quality is ready, but this event has no backend-verifiable target id",
            "md_path": md_path_raw,
            "source_quality_status": quality_status,
            "source_quality_score": quality_score,
            "checks": checks,
        }
    return {
        "status": "failed",
        "reason": "source still has conversion quality issues and no target anchor was verified",
        "md_path": md_path_raw,
        "source_quality_status": quality_status,
        "source_quality_score": quality_score,
        "checks": checks,
    }


def _reader_locate_repair_verification(run: dict) -> dict:
    rows = _reader_locate_all_event_rows(limit=4000)
    problems = _reader_locate_run_problem_events(run, rows)
    if not problems:
        return {
            "type": "reader_locate_repair",
            "status": "skipped",
            "quality_ok": False,
            "target_count": 0,
            "passed": 0,
            "failed": 0,
            "needs_reader_reopen": 0,
            "detail": "No matching Reader locate failure or degraded event was linked to this repair run.",
        }
    after_at = _safe_int((run or {}).get("updated_at"), 0) or _safe_int((run or {}).get("created_at"), 0)
    checked: list[dict] = []
    passed = 0
    failed = 0
    needs_reopen = 0
    for problem in problems[:20]:
        newer_good = _reader_locate_newer_good_event(problem, rows, after_at=after_at)
        if newer_good:
            status = "passed"
            reason = "a newer exact/block reader locate event verified this source"
            check = {
                "status": status,
                "reason": reason,
                "source_path": str(problem.get("source_path") or ""),
                "source_name": str(problem.get("source_name") or ""),
                "locate_feedback_key": str(problem.get("locate_feedback_key") or ""),
                "previous_status": str(problem.get("status") or ""),
                "verified_by_event_at": _safe_int(newer_good.get("created_at"), 0),
                "recommended_action": str(problem.get("recommended_action") or ""),
            }
        else:
            target_check = _reader_locate_source_target_check(problem)
            status = str(target_check.get("status") or "failed")
            reason = str(target_check.get("reason") or "")
            check = {
                "status": status,
                "reason": reason,
                "source_path": str(problem.get("source_path") or ""),
                "source_name": str(problem.get("source_name") or ""),
                "locate_feedback_key": str(problem.get("locate_feedback_key") or ""),
                "previous_status": str(problem.get("status") or ""),
                "previous_precision": str(problem.get("precision") or ""),
                "recommended_action": str(problem.get("recommended_action") or ""),
                "md_path": str(target_check.get("md_path") or ""),
                "source_quality_status": str(target_check.get("source_quality_status") or ""),
                "source_quality_score": _safe_int(target_check.get("source_quality_score"), 0),
                "checks": target_check.get("checks") if isinstance(target_check.get("checks"), dict) else {},
            }
        if status == "passed":
            passed += 1
        elif status == "needs_reader_reopen":
            needs_reopen += 1
        else:
            failed += 1
        checked.append(check)
    if failed > 0:
        status = "failed"
    elif needs_reopen > 0:
        status = "needs_reader_reopen"
    else:
        status = "passed"
    return {
        "type": "reader_locate_repair",
        "status": status,
        "quality_ok": status == "passed",
        "target_count": len(problems),
        "passed": int(passed),
        "failed": int(failed),
        "needs_reader_reopen": int(needs_reopen),
        "checked": checked[:12],
        "detail": (
            f"Reader locate verification passed for {passed}/{len(problems)} targets."
            if status == "passed"
            else (
                f"Reader locate needs a user reopen for {needs_reopen}/{len(problems)} targets."
                if status == "needs_reader_reopen"
                else f"Reader locate verification still failing for {failed}/{len(problems)} targets."
            )
        ),
    }


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


def _quality_repair_run_status(*, enqueued: int, repaired: int, failed: int, needs_reindex: bool) -> tuple[str, str]:
    if failed > 0 and repaired <= 0 and enqueued <= 0:
        return "failed", "repair_failed"
    if enqueued > 0:
        return "queued", "source_reconversion_queued"
    if needs_reindex:
        return "reindex_pending", "reindex_pending"
    return "completed", "repair_complete"


def _normalize_quality_repair_run(row: dict) -> dict:
    if not isinstance(row, dict):
        return {}
    run_id = _compact_text(row.get("run_id"), limit=80)
    if not run_id:
        return {}
    impact = row.get("impact") if isinstance(row.get("impact"), dict) else {}
    return {
        "run_id": run_id,
        "status": _compact_text(row.get("status"), limit=80) or "info",
        "phase": _compact_text(row.get("phase"), limit=120),
        "created_at": _safe_int(row.get("created_at"), 0),
        "updated_at": _safe_int(row.get("updated_at"), 0),
        "requested": _safe_int(row.get("requested"), 0),
        "enqueued": _safe_int(row.get("enqueued"), 0),
        "repaired": _safe_int(row.get("repaired"), 0),
        "failed": _safe_int(row.get("failed"), 0),
        "skipped_busy": _safe_int(row.get("skipped_busy"), 0),
        "needs_reindex": bool(row.get("needs_reindex")),
        "reindexed": row.get("reindexed") if isinstance(row.get("reindexed"), bool) else None,
        "target_names": _list_strings(row.get("target_names"))[:40],
        "target_sources": _list_strings(row.get("target_sources"))[:40],
        "impact": _compact_json_value(impact, limit=30) if impact else {},
        "verification": _compact_json_value(row.get("verification") if isinstance(row.get("verification"), dict) else {}, limit=30),
        "detail": _compact_text(row.get("detail"), limit=500),
    }


def _quality_repair_run_rows(*, limit: int = 40) -> list[dict]:
    rows = _read_jsonl_artifact(_quality_repair_runs_path(), limit=1000)
    latest: dict[str, dict] = {}
    for raw in rows:
        row = _normalize_quality_repair_run(raw)
        run_id = str(row.get("run_id") or "")
        if not run_id:
            continue
        prev = latest.get(run_id)
        if not prev or _safe_int(row.get("updated_at"), 0) >= _safe_int(prev.get("updated_at"), 0):
            latest[run_id] = row
    out = list(latest.values())
    out.sort(key=lambda item: (_safe_int(item.get("updated_at"), 0), _safe_int(item.get("created_at"), 0)), reverse=True)
    return out[: max(0, min(200, int(limit)))]


def _quality_repair_run_by_id(run_id: str) -> dict:
    target = str(run_id or "").strip()
    if not target:
        return {}
    for row in _quality_repair_run_rows(limit=200):
        if str(row.get("run_id") or "") == target:
            return row
    return {}


def _append_quality_repair_run(record: dict) -> dict:
    now = int(time.time())
    run_id = _compact_text(record.get("run_id"), limit=80) or uuid.uuid4().hex
    row = _normalize_quality_repair_run({
        "run_id": run_id,
        "status": record.get("status") or "info",
        "phase": record.get("phase") or "",
        "created_at": _safe_int(record.get("created_at"), now) or now,
        "updated_at": _safe_int(record.get("updated_at"), now) or now,
        "requested": record.get("requested"),
        "enqueued": record.get("enqueued"),
        "repaired": record.get("repaired"),
        "failed": record.get("failed"),
        "skipped_busy": record.get("skipped_busy"),
        "needs_reindex": bool(record.get("needs_reindex")),
        "reindexed": record.get("reindexed") if isinstance(record.get("reindexed"), bool) else None,
        "target_names": record.get("target_names") if isinstance(record.get("target_names"), list) else [],
        "target_sources": record.get("target_sources") if isinstance(record.get("target_sources"), list) else [],
        "impact": record.get("impact") if isinstance(record.get("impact"), dict) else {},
        "verification": record.get("verification") if isinstance(record.get("verification"), dict) else {},
        "detail": record.get("detail") or "",
    })
    try:
        path = _quality_repair_runs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:
        raise HTTPException(500, f"failed to write quality repair run: {exc}") from exc
    return row


def _quality_repair_run_has_active_sources(run: dict) -> bool:
    snap = _bg_snapshot()
    task_by_path, task_by_name = _build_task_maps_from_snapshot(snap)
    target_names = {name.strip() for name in _list_strings((run or {}).get("target_names")) if name.strip()}
    target_sources = {source.strip() for source in _list_strings((run or {}).get("target_sources")) if source.strip()}

    def _task_active(info: dict | None) -> bool:
        return isinstance(info, dict) and (bool(info.get("queued")) or bool(info.get("running")))

    for name in target_names:
        if _task_active(task_by_name.get(name)):
            return True
        base = Path(name.replace("\\", "/")).name
        if base and _task_active(task_by_name.get(base)):
            return True
    for raw in target_sources:
        key = _normalized_path_key(raw)
        if key and _task_active(task_by_path.get(key)):
            return True
        base = Path(raw.replace("\\", "/")).name
        if base and _task_active(task_by_name.get(base)):
            return True
    return False


def _quality_repair_run_update_record(current: dict, **patch) -> dict:
    merged = dict(current or {})
    merged.update({key: value for key, value in patch.items() if value is not None})
    merged["updated_at"] = int(time.time())
    if isinstance(merged.get("reindexed"), bool):
        impact = dict(merged.get("impact") or {})
        impact["reindexed"] = bool(merged.get("reindexed"))
        merged["impact"] = impact
    return _append_quality_repair_run(merged)


def _quality_repair_run_match_tokens(run: dict) -> set[str]:
    tokens: set[str] = set()
    for raw in [*_list_strings((run or {}).get("target_names")), *_list_strings((run or {}).get("target_sources"))]:
        text = str(raw or "").strip()
        if not text:
            continue
        folded = text.replace("\\", "/").lower()
        tokens.add(folded)
        base = Path(folded).name
        if base:
            tokens.add(base)
            stem = Path(base).stem
            if stem:
                tokens.add(stem)
        key = _normalized_path_key(text).lower()
        if key:
            tokens.add(key.replace("\\", "/"))
    return {token for token in tokens if token}


def _quality_repair_case_match_tokens(case: dict) -> set[str]:
    tokens: set[str] = set()
    for source in list((case or {}).get("source_diagnostics") or []):
        if not isinstance(source, dict):
            continue
        for field in ("source_path", "md_path", "pdf_path", "source_name", "title"):
            text = str(source.get(field) or "").strip()
            if not text:
                continue
            folded = text.replace("\\", "/").lower()
            tokens.add(folded)
            base = Path(folded).name
            if base:
                tokens.add(base)
                stem = Path(base).stem
                if stem:
                    tokens.add(stem)
            key = _normalized_path_key(text).lower()
            if key:
                tokens.add(key.replace("\\", "/"))
    for doc_id in _list_strings((case or {}).get("doc_ids")):
        if doc_id:
            tokens.add(str(doc_id).strip().lower())
    return {token for token in tokens if token}


def _quality_repair_run_candidate_cases(run: dict, *, limit: int = 3) -> list[dict]:
    run_tokens = _quality_repair_run_match_tokens(run)
    if not run_tokens:
        return []
    matched: list[tuple[int, dict]] = []
    for case in _latest_research_qa_failure_cases(limit=40):
        case_tokens = _quality_repair_case_match_tokens(case)
        if not case_tokens:
            continue
        overlap = run_tokens.intersection(case_tokens)
        if overlap:
            matched.append((len(overlap), case))
    matched.sort(key=lambda item: (-item[0], str((item[1] or {}).get("id") or "")))
    out: list[dict] = []
    seen: set[str] = set()
    for _score, case in matched:
        case_id = str(case.get("id") or "").strip()
        if not case_id or case_id in seen:
            continue
        seen.add(case_id)
        out.append(case)
        if len(out) >= int(limit):
            break
    return out


def _quality_repair_run_source_inputs(run: dict) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in [*_list_strings((run or {}).get("target_sources")), *_list_strings((run or {}).get("target_names"))]:
        text = str(raw or "").strip()
        if not text:
            continue
        name = Path(text.replace("\\", "/")).name or text
        key = f"{text}\n{name}".lower()
        if key in seen:
            continue
        seen.add(key)
        values.append((text, name))
    return values


def _quality_repair_run_source_verification(run: dict) -> dict:
    if _safe_int((run or {}).get("enqueued"), 0) <= 0 and str((run or {}).get("phase") or "").strip().lower() != "source_reconversion_queued":
        return {}

    items: list[dict] = []
    issue_counter: Counter = Counter()
    seen_targets: set[str] = set()
    for source_path, source_name in _quality_repair_run_source_inputs(run):
        resolved = _resolve_quality_source(source_path=source_path, source_name=source_name)
        pdf_path_raw = str(resolved.get("pdf_path") or "").strip()
        md_path_raw = str(resolved.get("md_path") or "").strip()
        md_exists = bool(resolved.get("md_exists")) and bool(md_path_raw)
        pdf_exists = bool(pdf_path_raw and _path_is_file(Path(pdf_path_raw).expanduser()))
        if not md_exists and not pdf_exists:
            continue

        target_key = _normalized_path_key(md_path_raw or pdf_path_raw).lower()
        if not target_key or target_key in seen_targets:
            continue
        seen_targets.add(target_key)

        if not md_exists:
            issue_counter["missing_markdown"] += 1
            items.append({
                "source_path": source_path,
                "source_name": source_name,
                "pdf_path": pdf_path_raw,
                "md_path": md_path_raw,
                "status": "missing_markdown",
                "score": 0,
                "issue_codes": ["missing_markdown"],
                "action": "reconvert",
            })
            continue

        md_path = Path(md_path_raw).expanduser()
        pdf_path = Path(pdf_path_raw).expanduser() if pdf_exists else None
        try:
            write_conversion_quality_result(md_path, source_pdf_path=pdf_path)
            _clear_conversion_quality_cache(md_path)
            quality = _conversion_quality_summary(md_path) or {}
            report = quality.get("conversion_report") if isinstance(quality.get("conversion_report"), dict) else {}
            plan = report.get("repair_plan") if isinstance(report.get("repair_plan"), dict) else {}
            issue_codes = [
                str(issue.get("code") or "").strip().lower()
                for issue in list(quality.get("issues") or [])
                if isinstance(issue, dict) and str(issue.get("code") or "").strip()
            ]
            if not issue_codes:
                issue_codes = [
                    str(code or "").strip().lower()
                    for code in list((plan or {}).get("issue_codes") or [])
                    if str(code or "").strip()
                ]
            for code in issue_codes:
                issue_counter[code] += 1
            action = str((plan or {}).get("action") or report.get("recommended_action") or "").strip().lower()
            if not action:
                action = "none" if not issue_codes else "review"
            status = "ready" if not issue_codes and str(quality.get("status") or "") == "good" else (action if action != "none" else str(quality.get("status") or "ready"))
            items.append({
                "source_path": source_path,
                "source_name": source_name,
                "pdf_path": pdf_path_raw,
                "md_path": str(md_path),
                "status": status,
                "score": _safe_int(quality.get("score"), 0),
                "issue_codes": issue_codes[:12],
                "action": action,
                "summary": str(quality.get("summary") or ""),
            })
        except Exception as exc:
            issue_counter["quality_scan_failed"] += 1
            items.append({
                "source_path": source_path,
                "source_name": source_name,
                "pdf_path": pdf_path_raw,
                "md_path": md_path_raw,
                "status": "quality_scan_failed",
                "score": 0,
                "issue_codes": ["quality_scan_failed"],
                "action": "review",
                "error": str(exc)[:240],
            })

    target_count = len(items)
    if target_count <= 0:
        return {}
    ready = sum(1 for item in items if str(item.get("status") or "").lower() == "ready")
    autofix = sum(1 for item in items if str(item.get("action") or "").lower() == "autofix")
    reconvert = sum(1 for item in items if str(item.get("action") or "").lower() == "reconvert")
    review = sum(1 for item in items if str(item.get("action") or "").lower() == "review")
    quality_ok = ready == target_count and reconvert == 0 and review == 0 and autofix == 0
    status = "passed" if quality_ok else ("failed" if reconvert > 0 or review > 0 else "retryable")
    return {
        "type": "conversion_source_quality",
        "status": status,
        "quality_ok": bool(quality_ok),
        "target_count": int(target_count),
        "ready": int(ready),
        "autofix": int(autofix),
        "reconvert": int(reconvert),
        "review": int(review),
        "issue_codes": _counter_items(issue_counter, limit=12),
        "items": items[:20],
        "detail": (
            f"Conversion source quality passed for {ready}/{target_count} target(s)."
            if quality_ok
            else f"Conversion source quality still has {target_count - ready}/{target_count} target(s) needing action."
        ),
    }


def _quality_repair_run_verification_from_rerun(result: dict) -> dict:
    failures = [
        str(item.get("name") or "").strip()
        for item in list((result or {}).get("failures") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    return {
        "type": "research_qa_rerun",
        "case_id": str((result or {}).get("case_id") or "").strip(),
        "status": str((result or {}).get("status") or "").strip(),
        "quality_ok": bool((result or {}).get("quality_ok")),
        "failure_count": len(failures),
        "failures": failures[:8],
        "error_kind": str((result or {}).get("error_kind") or "").strip(),
        "error_detail": _compact_text((result or {}).get("error_detail"), limit=240),
        "report_path": str((result or {}).get("report_path") or "").strip(),
        "raw_path": str((result or {}).get("raw_path") or "").strip(),
        "finished_at": _safe_int((result or {}).get("finished_at"), 0),
    }


def _quality_repair_run_verification_patch(run: dict, *, body) -> dict:
    if body is not None and not bool(getattr(body, "verify", True)):
        return {}
    reader_verification = _reader_locate_repair_verification(run)
    reader_has_targets = _safe_int(reader_verification.get("target_count"), 0) > 0
    reader_status = str(reader_verification.get("status") or "").strip().lower()
    source_verification = _quality_repair_run_source_verification(run)
    source_has_targets = _safe_int(source_verification.get("target_count"), 0) > 0
    source_ok = (not source_has_targets) or bool(source_verification.get("quality_ok"))
    case_id = str(getattr(body, "case_id", "") or "").strip() if body is not None else ""
    candidate_cases = _quality_repair_run_candidate_cases(run, limit=3)
    if not case_id and candidate_cases:
        case_id = str(candidate_cases[0].get("id") or "").strip()
    if not case_id:
        if source_has_targets and not reader_has_targets:
            if bool(source_verification.get("quality_ok")):
                return {
                    "status": "completed",
                    "verification": source_verification,
                    "detail": str(source_verification.get("detail") or "Conversion source quality verification passed."),
                }
            return {
                "status": "warning",
                "phase": "source_quality_failed",
                "verification": source_verification,
                "detail": str(source_verification.get("detail") or "Conversion source quality still needs attention."),
            }
        if reader_has_targets:
            if source_has_targets and not source_ok:
                combined = {
                    "type": "combined_repair_verification",
                    "status": "failed",
                    "quality_ok": False,
                    "source_quality": source_verification,
                    "reader_locate": reader_verification,
                }
                return {
                    "status": "warning",
                    "phase": "source_quality_failed",
                    "verification": combined,
                    "detail": str(source_verification.get("detail") or "Conversion source quality still needs attention."),
                }
            if bool(reader_verification.get("quality_ok")) or reader_status == "passed":
                verification = (
                    {
                        "type": "combined_repair_verification",
                        "status": "passed",
                        "quality_ok": True,
                        "source_quality": source_verification,
                        "reader_locate": reader_verification,
                    }
                    if source_has_targets
                    else reader_verification
                )
                return {
                    "status": "completed",
                    "phase": "verification_passed",
                    "verification": verification,
                    "detail": str(reader_verification.get("detail") or "Reader locate verification passed."),
                }
            if reader_status == "needs_reader_reopen":
                verification = (
                    {
                        "type": "combined_repair_verification",
                        "status": "needs_reader_reopen",
                        "quality_ok": False,
                        "source_quality": source_verification,
                        "reader_locate": reader_verification,
                    }
                    if source_has_targets
                    else reader_verification
                )
                return {
                    "status": "warning",
                    "phase": "verification_needs_reader_reopen",
                    "verification": verification,
                    "detail": str(reader_verification.get("detail") or "Reader locate needs a user reopen to confirm exact positioning."),
                }
            verification = (
                {
                    "type": "combined_repair_verification",
                    "status": "failed",
                    "quality_ok": False,
                    "source_quality": source_verification,
                    "reader_locate": reader_verification,
                }
                if source_has_targets
                else reader_verification
            )
            return {
                "status": "warning",
                "phase": "verification_failed",
                "verification": verification,
                "detail": str(reader_verification.get("detail") or "Reader locate verification still failing."),
            }
        return {
            "verification": {
                "type": "research_qa_rerun",
                "status": "skipped",
                "quality_ok": False,
                "candidate_count": 0,
                "detail": "No matching failed Research QA case was linked to this source repair.",
            },
        }
    try:
        result = _run_research_qa_case(
            case_id=case_id,
            base_url=str(getattr(body, "base_url", "") or ""),
            timeout_s=float(getattr(body, "timeout_s", 180.0) or 180.0),
            top_k=int(getattr(body, "top_k", 6) or 6),
            max_tokens=int(getattr(body, "max_tokens", 1800) or 1800),
            dry_run=bool(getattr(body, "dry_run", False)),
            record_history=not bool(getattr(body, "dry_run", False)),
        )
    except HTTPException as exc:
        result = {
            "case_id": case_id,
            "status": "error",
            "quality_ok": False,
            "failures": [],
            "error_kind": "research_qa_runner_unavailable",
            "error_detail": str(exc.detail or exc.status_code)[:240],
            "finished_at": int(time.time()),
        }
    except Exception as exc:
        result = {
            "case_id": case_id,
            "status": "error",
            "quality_ok": False,
            "failures": [],
            "error_kind": "exception",
            "error_detail": str(exc)[:240],
            "finished_at": int(time.time()),
        }
    verification = _quality_repair_run_verification_from_rerun(result)
    if reader_has_targets:
        qa_ok = bool(verification.get("quality_ok")) or str(verification.get("status") or "").lower() == "passed"
        reader_ok = bool(reader_verification.get("quality_ok")) or reader_status == "passed"
        combined_ok = bool(qa_ok and reader_ok and source_ok)
        combined = {
            "type": "combined_repair_verification",
            "status": "passed" if combined_ok else ("blocked" if str(verification.get("status") or "").lower() == "error" else "failed"),
            "quality_ok": bool(combined_ok),
            "source_quality": source_verification if source_has_targets else {},
            "research_qa": verification,
            "reader_locate": reader_verification,
        }
        if combined_ok:
            return {
                "status": "completed",
                "phase": "verification_passed",
                "verification": combined,
                "detail": f"Research QA and Reader locate verification passed for {case_id}.",
            }
        if source_has_targets and not source_ok:
            return {
                "status": "warning",
                "phase": "source_quality_failed",
                "verification": combined,
                "detail": str(source_verification.get("detail") or "Conversion source quality still needs attention."),
            }
        if qa_ok and reader_status == "needs_reader_reopen":
            return {
                "status": "warning",
                "phase": "verification_needs_reader_reopen",
                "verification": combined,
                "detail": str(reader_verification.get("detail") or "Reader locate needs a user reopen to confirm exact positioning."),
            }
        if qa_ok:
            return {
                "status": "warning",
                "phase": "verification_failed",
                "verification": combined,
                "detail": str(reader_verification.get("detail") or "Reader locate verification still failing."),
            }
    research_status = str(verification.get("status") or "").lower()
    research_error_kind = str(verification.get("error_kind") or "").strip()
    research_failures = _list_strings(verification.get("failures"))
    if bool(verification.get("quality_ok")) or research_status == "passed":
        if source_has_targets and not source_ok:
            return {
                "status": "warning",
                "phase": "source_quality_failed",
                "verification": {
                    "type": "combined_repair_verification",
                    "status": "failed",
                    "quality_ok": False,
                    "source_quality": source_verification,
                    "research_qa": verification,
                },
                "detail": str(source_verification.get("detail") or "Conversion source quality still needs attention."),
            }
        return {
            "status": "completed",
            "phase": "verification_passed",
            "verification": (
                {
                    "type": "combined_repair_verification",
                    "status": "passed",
                    "quality_ok": True,
                    "source_quality": source_verification,
                    "research_qa": verification,
                }
                if source_has_targets
                else verification
            ),
            "detail": f"Verification passed for Research QA case {case_id}.",
        }
    if source_has_targets:
        verification = {
            "type": "combined_repair_verification",
            "status": "blocked" if research_status == "error" else "failed",
            "quality_ok": False,
            "source_quality": source_verification,
            "research_qa": verification,
        }
    if research_status == "error" and research_error_kind:
        return {
            "status": "warning",
            "phase": "verification_blocked",
            "verification": verification,
            "detail": f"Research QA verification could not complete: {research_error_kind}.",
        }
    return {
        "status": "warning",
        "phase": "verification_failed",
        "verification": verification,
        "detail": "Research QA verification still failing: " + (" / ".join(research_failures[:4]) if research_failures else research_status or "unknown"),
    }


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
        shelf_metadata_repair_targets = _research_qa_shelf_metadata_repair_targets(citation_diagnostics, ref_diagnostics)
        shelf_metadata_missing_fields = _research_qa_shelf_missing_field_counts(citation_diagnostics)
        rerun_status = _research_qa_rerun_case_status(str(row.get("id") or "").strip(), rerun_rows)
        root_causes = _research_qa_root_causes(
            failures=failures,
            missing_expected_doc_ids=missing_expected_doc_ids,
            citation_count=citation_count,
            system_b_count=system_b_count,
            ref_hit_count=ref_hit_count,
            source_diagnostics=source_diagnostics,
            citation_diagnostics=citation_diagnostics,
            rerun_status=rerun_status,
        )
        repair_actions = _research_qa_repair_actions(
            root_causes=root_causes,
            source_diagnostics=source_diagnostics,
            missing_expected_doc_ids=missing_expected_doc_ids,
            shelf_metadata_targets=shelf_metadata_repair_targets,
            shelf_missing_fields=shelf_metadata_missing_fields,
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
                    "shelf_metadata_repair_target_count": len(shelf_metadata_repair_targets),
                    **_research_qa_quality_gate_summary(quality),
                },
                "citation_diagnostics": citation_diagnostics,
                "ref_diagnostics": ref_diagnostics,
                "shelf_metadata_repair_targets": shelf_metadata_repair_targets,
                "shelf_metadata_missing_fields": shelf_metadata_missing_fields,
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
                "shelf_item_count": 0,
                "shelf_metadata_ready_count": 0,
                "shelf_export_ready_count": 0,
                "shelf_summary_export_ready_count": 0,
                "shelf_doi_count": 0,
                "shelf_source_clickable_count": 0,
                "shelf_review_count": 0,
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
        "shelf_item_count": 0,
        "shelf_metadata_ready_count": 0,
        "shelf_export_ready_count": 0,
        "shelf_summary_export_ready_count": 0,
        "shelf_doi_count": 0,
        "shelf_source_clickable_count": 0,
        "shelf_review_count": 0,
    }
    failures: Counter = Counter()
    for row in rows:
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        shelf_quality = quality.get("citation_shelf_quality") if isinstance(quality.get("citation_shelf_quality"), dict) else {}
        if shelf_quality:
            summary["shelf_item_count"] = int(summary["shelf_item_count"]) + _safe_int(shelf_quality.get("count"), 0)
            summary["shelf_metadata_ready_count"] = int(summary["shelf_metadata_ready_count"]) + _safe_int(shelf_quality.get("metadata_ready_count"), 0)
            summary["shelf_export_ready_count"] = int(summary["shelf_export_ready_count"]) + _safe_int(shelf_quality.get("export_ready_count"), 0)
            summary["shelf_summary_export_ready_count"] = int(summary["shelf_summary_export_ready_count"]) + _safe_int(shelf_quality.get("summary_export_ready_count"), 0)
            summary["shelf_doi_count"] = int(summary["shelf_doi_count"]) + _safe_int(shelf_quality.get("doi_count"), 0)
            summary["shelf_source_clickable_count"] = int(summary["shelf_source_clickable_count"]) + _safe_int(shelf_quality.get("source_clickable_count"), 0)
            summary["shelf_review_count"] = int(summary["shelf_review_count"]) + _safe_int(shelf_quality.get("review_count"), 0)
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
    reader_locate: dict | None = None,
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

    reader = reader_locate if isinstance(reader_locate, dict) else {}
    reader_summary = reader.get("summary") if isinstance(reader.get("summary"), dict) else {}
    reader_failed = _safe_int(reader_summary.get("failed"), 0)
    reader_degraded = _safe_int(reader_summary.get("degraded"), 0)
    reader_strict_miss = _safe_int(reader_summary.get("strict_miss"), 0)
    reader_problem_count = reader_failed + reader_degraded + reader_strict_miss
    if bool(reader.get("available")) and reader_problem_count > 0:
        recommended_sources = list(reader.get("recommended_sources") or [])
        first_source = recommended_sources[0] if recommended_sources and isinstance(recommended_sources[0], dict) else {}
        actions.append(
            {
                "domain": "reader_locate",
                "severity": "error" if reader_failed > 0 or reader_strict_miss > 0 else "warning",
                "label": "Repair reader locate sources",
                "count": len(recommended_sources) or reader_problem_count,
                "detail": str(first_source.get("latest_reason") or first_source.get("recommended_action") or "Reader locate degraded"),
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
    shelf_items = _safe_int(card_summary.get("shelf_item_count"), 0)
    shelf_export_ready = _safe_int(card_summary.get("shelf_export_ready_count"), 0)
    shelf_summary_export_ready = _safe_int(card_summary.get("shelf_summary_export_ready_count"), 0)
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
    shelf_export_detail = (
        f"export-ready {shelf_export_ready}/{shelf_items}, summaries {shelf_summary_export_ready}/{shelf_items}"
        if shelf_items > 0
        else ""
    )
    shelf_detail = (
        f"{shelf_failed} literature basket checks failed" + (f"; {shelf_export_detail}" if shelf_export_detail else "")
        if shelf_failed > 0
        else (
            "Waiting for shelf acceptance results"
            if not card_available
            else (f"Literature basket checks passed; {shelf_export_detail}" if shelf_export_detail else "Literature basket checks passed")
        )
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
            metrics={
                "items": shelf_items,
                "export_ready": shelf_export_ready,
                "summary_export_ready": shelf_summary_export_ready,
            },
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
    reader_locate: dict | None = None,
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
    shelf_items = _safe_int(card_summary.get("shelf_item_count"), 0)
    shelf_metadata_ready = _safe_int(card_summary.get("shelf_metadata_ready_count"), 0)
    shelf_export_ready = _safe_int(card_summary.get("shelf_export_ready_count"), 0)
    shelf_summary_export_ready = _safe_int(card_summary.get("shelf_summary_export_ready_count"), 0)
    shelf_doi = _safe_int(card_summary.get("shelf_doi_count"), 0)
    shelf_source_clickable = _safe_int(card_summary.get("shelf_source_clickable_count"), 0)
    shelf_review = _safe_int(card_summary.get("shelf_review_count"), 0)
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

    reader_quality = reader_locate if isinstance(reader_locate, dict) else {}
    reader_summary = reader_quality.get("summary") if isinstance(reader_quality.get("summary"), dict) else {}
    reader_available = bool(reader_quality.get("available"))
    reader_failed = _safe_int(reader_summary.get("failed"), 0)
    reader_degraded = _safe_int(reader_summary.get("degraded"), 0)
    reader_strict_miss = _safe_int(reader_summary.get("strict_miss"), 0)
    reader_repairable = _safe_int(reader_summary.get("repairable"), 0)
    reader_count = reader_failed + reader_degraded + reader_strict_miss
    if reader_available:
        reader_status = str(reader_quality.get("status") or "unknown").strip().lower() or "unknown"
    else:
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
            summary=(
                f"Basket export-ready {shelf_export_ready}/{shelf_items}"
                if shelf_status == "good" and shelf_items > 0
                else ("Basket metadata is export-ready" if shelf_status == "good" else f"{shelf_failed} basket checks failed; export-ready {shelf_export_ready}/{shelf_items}")
            ),
            detail="Checks structured DOI, authors, venue, grounded summary, source-open, and export readiness.",
            action="repair_shelf_metadata" if shelf_failed > 0 else "review_literature_basket",
            target_stage="shelf",
            count=shelf_failed,
            blocking=shelf_status == "error",
            metrics={
                "shelf_failed": shelf_failed,
                "items": shelf_items,
                "metadata_ready": shelf_metadata_ready,
                "export_ready": shelf_export_ready,
                "summary_export_ready": shelf_summary_export_ready,
                "doi": shelf_doi,
                "source_clickable": shelf_source_clickable,
                "review": shelf_review,
            },
        ),
        _quality_feature_health_item(
            "reader_locate",
            "Reader locate",
            reader_status,
            score=_feature_score_from_status(reader_status, good=94, warning=72, error=43),
            summary=(
                "Reader jumps have grounded evidence"
                if reader_status == "good"
                else (
                    f"{reader_count} real reader jumps need source repair"
                    if reader_available
                    else "Reader locate may be affected by weak citations or source conversion"
                )
            ),
            detail=(
                f"Observed {reader_failed} failed, {reader_degraded} degraded, {reader_repairable} repairable locate results."
                if reader_available
                else "Covers citation click-through, source opening, anchors, page markers, and evidence snippets."
            ),
            action="repair_reader_locate" if reader_available and reader_status != "good" else ("inspect_reader_locate" if reader_status != "good" else "review_reader_locate"),
            target_stage="reader_locate" if reader_available else ("citations" if card_failed > 0 or citation_missing_cases > 0 else "conversion"),
            count=reader_count,
            blocking=reader_status == "error",
            metrics={
                "citation_missing_cases": citation_missing_cases,
                "conversion_review": conversion_review,
                "conversion_unknown": conversion_unknown,
                "failed": reader_failed,
                "degraded": reader_degraded,
                "repairable": reader_repairable,
                "strict_miss": reader_strict_miss,
            },
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
    reader_locate = _reader_locate_quality_summary()
    rerun_history = _research_qa_rerun_history_rows()
    failure_cases = _latest_research_qa_failure_cases(rerun_history=rerun_history)
    reader_locate_summary = reader_locate.get("summary") if isinstance(reader_locate.get("summary"), dict) else {}
    domains = {
        "conversion": conversion_domain,
        "research_qa": research_qa,
        "citation_cards": citation_cards,
        "reader_locate": {
            "available": bool(reader_locate.get("available")),
            "status": str(reader_locate.get("status") or "unknown"),
            "summary": reader_locate_summary,
            "top_failures": list(reader_locate.get("top_failures") or [])[:6],
        },
    }
    rerun_summary = _research_qa_rerun_history_summary(rerun_history)
    priority_actions = _quality_priority_actions(
        conversion_status=conversion_status,
        recommended=recommended,
        research_qa=research_qa,
        citation_cards=citation_cards,
        reader_locate=reader_locate,
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
        reader_locate=reader_locate,
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
        "reader_locate": reader_locate,
        "full_chain": full_chain,
        "feature_health": feature_health,
        "failure_cases": failure_cases,
        "rerun_summary": rerun_summary,
        "repair_runs": _quality_repair_run_rows(limit=8),
        "priority_actions": priority_actions,
        "queue": (listing or {}).get("queue") or {},
        "scope": str((listing or {}).get("scope") or ""),
        "truncated": bool((listing or {}).get("truncated")),
    }


def _figure_asset_scan_payload(
    md_paths: list[Path],
    *,
    pdf_root: Path,
    target_dpi: int = 0,
    include_all: bool = False,
    max_errors: int = 20,
) -> dict:
    reports: list[dict] = []
    visible_items: list[dict] = []
    errors: list[dict] = []
    for md_path in md_paths:
        try:
            source_pdf = source_pdf_for_markdown(md_path, pdf_root)
            report = scan_figure_asset_quality(
                md_path,
                source_pdf_path=source_pdf,
                target_dpi=int(target_dpi or 0) or None,
            )
            source_name = _strip_known_source_ext(md_path.parent.name or md_path.stem)
            report = {
                **report,
                "source_name": source_name,
                "pdf_name": Path(str(source_pdf or "")).name if source_pdf else "",
                "pdf_path": str(source_pdf or ""),
            }
            reports.append(report)
            if bool(include_all) or int(report.get("issue_count") or 0) > 0:
                visible_items.append(report)
        except Exception as exc:
            if len(errors) < max(0, int(max_errors)):
                errors.append({"path": str(md_path), "error": str(exc)[:400]})

    summary = summarize_figure_asset_quality_reports(reports)
    if reports:
        summary["target_dpi"] = int(reports[0].get("target_dpi") or 0)
    return {
        **summary,
        "failed": len(errors),
        "errors": errors,
        "items": visible_items,
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


class ReaderLocateQualityBody(BaseModel):
    source_path: str = ""
    source_name: str = ""
    locate_feedback_key: str = ""
    locate_request_id: int = 0
    status: str = ""
    precision: str = ""
    ok: bool = False
    repairable: bool = False
    strict_locate: bool = False
    hint: str = ""
    reason: str = ""
    active_alt_index: int = 0
    block_id: str = ""
    anchor_id: str = ""
    anchor_kind: str = ""
    heading_path: str = ""


class QualityRepairRunUpdateBody(BaseModel):
    status: str = ""
    phase: str = ""
    reindexed: bool | None = None
    detail: str = ""
    metrics: dict = {}


class QualityRepairRunAdvanceBody(BaseModel):
    verify: bool = True
    case_id: str = ""
    base_url: str = ""
    timeout_s: float = 180.0
    top_k: int = 6
    max_tokens: int = 1800
    dry_run: bool = False


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
    row = _append_quality_action_history(body.model_dump() if hasattr(body, "model_dump") else body.dict())
    return {
        "ok": True,
        "item": row,
    }


@router.get("/quality/reader-locate")
def quality_reader_locate_events(limit: int = 40):
    rows = _reader_locate_event_rows(limit=max(1, min(200, int(limit or 40))))
    return {
        "ok": True,
        "items": rows,
        "summary": _reader_locate_quality_summary(rows),
    }


@router.post("/quality/reader-locate")
def record_quality_reader_locate(body: ReaderLocateQualityBody):
    row = _append_reader_locate_event(body.model_dump() if hasattr(body, "model_dump") else body.dict())
    return {
        "ok": True,
        "item": row,
        "summary": _reader_locate_quality_summary(),
    }


@router.get("/quality/repair-runs")
def quality_repair_runs(limit: int = 20):
    return {
        "ok": True,
        "items": _quality_repair_run_rows(limit=limit),
    }


@router.get("/quality/repair-runs/{run_id}")
def quality_repair_run(run_id: str):
    row = _quality_repair_run_by_id(run_id)
    if not row:
        raise HTTPException(404, "quality repair run not found")
    return {
        "ok": True,
        "item": row,
    }


@router.post("/quality/repair-runs/{run_id}")
def update_quality_repair_run(run_id: str, body: QualityRepairRunUpdateBody):
    current = _quality_repair_run_by_id(run_id)
    if not current:
        raise HTTPException(404, "quality repair run not found")
    patch = body.model_dump()
    update = {
        "status": str(patch.get("status") or current.get("status") or "info"),
        "phase": str(patch.get("phase") or current.get("phase") or ""),
        "detail": str(patch.get("detail") or current.get("detail") or ""),
    }
    if isinstance(patch.get("reindexed"), bool):
        update["reindexed"] = bool(patch.get("reindexed"))
    if isinstance(patch.get("metrics"), dict) and patch.get("metrics"):
        impact = dict(current.get("impact") or {})
        impact["update_metrics"] = patch.get("metrics")
        update["impact"] = impact
    row = _quality_repair_run_update_record(current, **update)
    return {
        "ok": True,
        "item": row,
    }


@router.post("/quality/repair-runs/{run_id}/advance")
def advance_quality_repair_run(run_id: str, body: QualityRepairRunAdvanceBody = QualityRepairRunAdvanceBody()):
    current = _quality_repair_run_by_id(run_id)
    if not current:
        raise HTTPException(404, "quality repair run not found")

    status = str(current.get("status") or "").strip().lower()
    phase = str(current.get("phase") or "").strip().lower()
    has_verification = isinstance(current.get("verification"), dict) and bool(current.get("verification"))
    if status == "completed" and (phase in {"repair_complete", "verification_passed"} or (phase == "reindex_complete" and has_verification)):
        return {
            "ok": True,
            "advanced": False,
            "waiting": False,
            "item": current,
            "reindex": None,
            "detail": "quality repair run is already complete",
        }

    advanced = False
    if status == "queued" or phase == "source_reconversion_queued":
        if _quality_repair_run_has_active_sources(current):
            row = _quality_repair_run_update_record(
                current,
                status="queued",
                phase="source_reconversion_queued",
                detail="Source reconversion is still running; continue after conversion finishes.",
            )
            return {
                "ok": True,
                "advanced": False,
                "waiting": True,
                "item": row,
                "reindex": None,
                "detail": "source reconversion is still active",
            }
        if bool(current.get("needs_reindex")):
            current = _quality_repair_run_update_record(
                current,
                status="reindex_pending",
                phase="reindex_pending",
                detail="Source reconversion is no longer active; index refresh is pending.",
            )
            status = "reindex_pending"
            phase = "reindex_pending"
            advanced = True
        else:
            row = _quality_repair_run_update_record(
                current,
                status="completed",
                phase="repair_complete",
                reindexed=False,
                detail="Source repair completed; index refresh was not required.",
            )
            return {
                "ok": True,
                "advanced": True,
                "waiting": False,
                "item": row,
                "reindex": None,
                "detail": "quality repair run completed",
            }

    needs_reindex = bool(current.get("needs_reindex")) and current.get("reindexed") is not True
    should_reindex = needs_reindex and (
        status in {"reindex_pending", "warning"}
        or phase in {"reindex_pending", "reindex_failed", "source_reconversion_queued"}
    )
    if should_reindex:
        result = _run_library_reindex()
        ok = bool(result.get("ok"))
        verification_patch = _quality_repair_run_verification_patch(current, body=body) if ok else {}
        row = _quality_repair_run_update_record(
            current,
            status=verification_patch.get("status") or ("completed" if ok else "warning"),
            phase=verification_patch.get("phase") or ("reindex_complete" if ok else "reindex_failed"),
            reindexed=ok,
            verification=verification_patch.get("verification"),
            detail=(
                verification_patch.get("detail")
                if verification_patch.get("detail")
                else "Index refresh completed after source repair."
                if ok
                else str(result.get("error") or result.get("stderr") or "Index refresh failed after source repair.")[:240]
            ),
        )
        return {
            "ok": bool(ok),
            "advanced": True,
            "waiting": False,
            "item": row,
            "reindex": result,
            "detail": "index refresh completed" if ok else "index refresh failed",
        }

    if (
        bool(getattr(body, "verify", True))
        and status == "completed"
        and phase == "reindex_complete"
        and not (isinstance(current.get("verification"), dict) and current.get("verification"))
    ):
        verification_patch = _quality_repair_run_verification_patch(current, body=body)
        if verification_patch.get("verification"):
            row = _quality_repair_run_update_record(
                current,
                status=verification_patch.get("status") or current.get("status") or "completed",
                phase=verification_patch.get("phase") or current.get("phase") or "reindex_complete",
                verification=verification_patch.get("verification"),
                detail=verification_patch.get("detail") or current.get("detail") or "",
            )
            return {
                "ok": str(row.get("status") or "").lower() == "completed",
                "advanced": True,
                "waiting": False,
                "item": row,
                "reindex": None,
                "detail": str(row.get("detail") or "verification completed"),
            }

    return {
        "ok": True,
        "advanced": advanced,
        "waiting": False,
        "item": current,
        "reindex": None,
        "detail": "quality repair run has no pending automatic step",
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


def _run_research_qa_case(
    *,
    case_id: str,
    base_url: str = "",
    timeout_s: float = 180.0,
    top_k: int = 6,
    max_tokens: int = 1800,
    dry_run: bool = False,
    record_history: bool = True,
) -> dict:
    case_id = str(case_id or "").strip()
    if not case_id:
        raise HTTPException(400, "case_id is required")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", case_id):
        raise HTTPException(400, "invalid case_id")

    repo = _repo_root()
    runner = repo / "tools" / "research_qa" / "run_research_qa_eval.py"
    if not _path_is_file(runner):
        raise HTTPException(404, "research QA runner not found")

    timeout_s = max(10.0, min(900.0, float(timeout_s or 180.0)))
    base_url = str(base_url or os.environ.get("KB_RESEARCH_QA_BASE_URL") or "http://127.0.0.1:8000").strip().rstrip("/")
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
        str(max(1, min(20, int(top_k or 6)))),
        "--max-tokens",
        str(max(256, min(8192, int(max_tokens or 1800)))),
        "--base-url",
        base_url,
    ]
    if bool(dry_run):
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
        if bool(record_history) and not bool(dry_run):
            _append_research_qa_rerun_history(result)
        return result

    finished_at = time.time()
    output_dir = None if bool(dry_run) else _extract_research_qa_output_dir(proc.stdout, fallback_after=started_at)
    result = _research_qa_rerun_result(
        case_id=case_id,
        output_dir=output_dir,
        returncode=int(proc.returncode),
        stdout=str(proc.stdout or ""),
        stderr=str(proc.stderr or ""),
        started_at=started_at,
        finished_at=finished_at,
    )
    if bool(record_history) and not bool(dry_run):
        _append_research_qa_rerun_history(result)
    return result


@router.post("/quality/research-qa/rerun")
def rerun_research_qa_case(body: QualityResearchQaRerunBody):
    return _run_research_qa_case(
        case_id=body.case_id,
        base_url=body.base_url,
        timeout_s=body.timeout_s,
        top_k=body.top_k,
        max_tokens=body.max_tokens,
        dry_run=body.dry_run,
        record_history=not bool(body.dry_run),
    )


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


class FigureAssetQualityScanBody(BaseModel):
    limit: int = 1000
    include_all: bool = False
    target_dpi: int = 0


class FigureAssetRefreshBody(BaseModel):
    pdf_names: list[str] = []
    sources: list[QualitySourceItem] = []
    limit: int = 200
    speed_mode: str = "balanced"
    no_llm: bool = False
    replace: bool = True
    target_dpi: int = 0


class QualityConversionBatchBody(BaseModel):
    repair: bool = False
    rebuild_indices: bool = True
    limit: int = 1000


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
    auto_apply_empty: bool = False


class LibrarySuggestionActionBody(BaseModel):
    pdf_name: str = ""
    sha1: str = ""
    path: str = ""
    category_action: str = ""
    accept_tags: list[str] = []
    dismiss_tags: list[str] = []
    accept_all_tags: bool = False
    dismiss_all_tags: bool = False


@router.post("/quality/conversion/batch")
def conversion_quality_batch(body: QualityConversionBatchBody):
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    try:
        limit = int(body.limit or 1000)
    except Exception:
        limit = 1000
    limit = max(1, min(limit, 5000))
    targets = discover_quality_markdown_files(md_d, limit=limit)
    if bool(body.repair):
        stats = repair_quality_targets(
            targets,
            pdf_root=pdf_d,
            rebuild_indices=bool(body.rebuild_indices),
        )
        mode = "repair"
    else:
        stats = scan_quality_targets(targets, pdf_root=pdf_d)
        mode = "scan"
    needs_reindex = bool(int(stats.get("changed", 0) or 0) > 0 or int(stats.get("rebuilt", 0) or 0) > 0)
    try:
        for target in targets:
            _clear_conversion_quality_cache(target)
    except Exception:
        pass
    return {
        "ok": True,
        "mode": mode,
        "target_count": len(targets),
        "limit": limit,
        "needs_reindex": needs_reindex,
        **stats,
    }


@router.post("/quality/figure-assets/scan")
def figure_asset_quality_scan(body: FigureAssetQualityScanBody):
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    try:
        limit = int(body.limit or 1000)
    except Exception:
        limit = 1000
    limit = max(1, min(limit, 5000))
    targets = discover_quality_markdown_files(md_d, limit=limit)
    payload = _figure_asset_scan_payload(
        targets,
        pdf_root=pdf_d,
        target_dpi=int(body.target_dpi or 0),
        include_all=bool(body.include_all),
    )
    return {
        "ok": True,
        "target_count": len(targets),
        "limit": limit,
        **payload,
    }


@router.post("/quality/figure-assets/refresh")
def refresh_figure_assets(body: FigureAssetRefreshBody):
    settings = get_settings()
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    md_d.mkdir(parents=True, exist_ok=True)

    speed_mode = str(body.speed_mode or "balanced").strip() or "balanced"
    no_llm = bool(body.no_llm) or speed_mode.lower() == "no_llm"
    replace = bool(body.replace)
    try:
        limit = int(body.limit or 200)
    except Exception:
        limit = 200
    limit = max(1, min(limit, 5000))

    explicit_targets: list[Path] = []
    explicit_pdf_by_md: dict[str, Path] = {}
    errors: list[dict] = []

    for raw_name in list(body.pdf_names or []):
        pdf_name = str(raw_name or "").strip()
        if (not pdf_name) or Path(pdf_name).name != pdf_name:
            errors.append({"name": pdf_name, "error": "invalid pdf_name"})
            continue
        pdf_path = (pdf_d / pdf_name).expanduser()
        if not _path_is_file(pdf_path):
            errors.append({"name": pdf_name, "error": "pdf not found"})
            continue
        _md_folder, md_main, md_exists = _resolve_md_output_paths(md_d, pdf_path)
        if not md_exists:
            errors.append({"name": pdf_name, "pdf_path": str(pdf_path), "error": "markdown not found"})
            continue
        explicit_targets.append(md_main)
        explicit_pdf_by_md[str(md_main)] = pdf_path

    for source in list(body.sources or []):
        resolved = _resolve_quality_source(
            source_path=str(source.source_path or ""),
            source_name=str(source.source_name or ""),
        )
        md_path_raw = str(resolved.get("md_path") or "").strip()
        pdf_path_raw = str(resolved.get("pdf_path") or "").strip()
        if not md_path_raw or not bool(resolved.get("md_exists")):
            errors.append({
                "source_path": str(source.source_path or ""),
                "source_name": str(source.source_name or ""),
                "error": "markdown not found",
            })
            continue
        md_path = Path(md_path_raw).expanduser()
        explicit_targets.append(md_path)
        if pdf_path_raw:
            explicit_pdf_by_md[str(md_path)] = Path(pdf_path_raw).expanduser()

    explicit_targets = _dedupe_paths(explicit_targets)
    targets = explicit_targets or discover_quality_markdown_files(md_d, limit=limit)
    scan = _figure_asset_scan_payload(
        targets,
        pdf_root=pdf_d,
        target_dpi=int(body.target_dpi or 0),
        include_all=False,
    )
    scan_errors = list(scan.get("errors") or [])
    if scan_errors:
        errors.extend(scan_errors[: max(0, 20 - len(errors))])

    snap = _bg_snapshot()
    task_by_path, task_by_name = _build_task_maps_from_snapshot(snap)
    enqueued = 0
    skipped_busy = 0
    failed = 0
    items: list[dict] = []
    seen_pdf_keys: set[str] = set()

    for report in list(scan.get("items") or []):
        if enqueued >= limit:
            break
        if not isinstance(report, dict):
            continue
        if not bool(report.get("refresh_recommended")):
            continue
        md_path = Path(str(report.get("md_path") or "")).expanduser()
        pdf_path = explicit_pdf_by_md.get(str(md_path))
        if pdf_path is None:
            raw_pdf = str(report.get("pdf_path") or "").strip()
            pdf_path = Path(raw_pdf).expanduser() if raw_pdf else source_pdf_for_markdown(md_path, pdf_d)
        issue_counts = report.get("issue_counts") if isinstance(report.get("issue_counts"), dict) else {}
        issue_codes = [str(code) for code in issue_counts.keys() if str(code or "").strip()]
        base_item = {
            "source_name": str(report.get("source_name") or ""),
            "pdf_name": Path(str(pdf_path or "")).name if pdf_path else str(report.get("pdf_name") or ""),
            "pdf_path": str(pdf_path or ""),
            "md_path": str(md_path),
            "issue_count": int(report.get("issue_count") or 0),
            "issue_codes": issue_codes,
            "enqueued": False,
            "skipped_busy": False,
            "task_id": "",
            "error": "",
        }
        if pdf_path is None or (not _path_is_file(pdf_path)):
            items.append({**base_item, "error": "source pdf not found"})
            failed += 1
            continue
        pdf_key = _normalized_path_key(pdf_path)
        if pdf_key and pdf_key in seen_pdf_keys:
            continue
        seen_pdf_keys.add(pdf_key)
        task_info = task_by_path.get(pdf_key) if pdf_key else None
        if not isinstance(task_info, dict):
            task_info = task_by_name.get(pdf_path.name) if isinstance(task_by_name.get(pdf_path.name), dict) else {}
        if bool(task_info.get("queued")) or bool(task_info.get("running")):
            items.append({**base_item, "skipped_busy": True, "error": "already queued or running"})
            skipped_busy += 1
            continue

        try:
            task = _build_bg_task(
                pdf_path=pdf_path,
                out_root=md_d,
                db_dir=Path(settings.db_dir).expanduser(),
                no_llm=no_llm,
                replace=replace,
                speed_mode=speed_mode,
                repair_context={
                    "action": "reconvert",
                    "scope": "figure_assets",
                    "reason": "figure asset quality refresh",
                    "source": "figure_asset_quality_refresh",
                    "issue_codes": issue_codes,
                },
            )
            _bg_enqueue(task)
            task_id = str(task.get("_tid") or "")
            try:
                append_conversion_repair_attempt(
                    md_path,
                    event="figure_asset_refresh_queued",
                    status="queued",
                    action="reconvert",
                    scope="figure_assets",
                    speed_mode=speed_mode,
                    issue_codes=issue_codes,
                    task_id=task_id,
                    source="figure_asset_quality_refresh",
                    reason="figure asset quality refresh",
                    detail="Figure asset scan queued source reconversion to refresh image crops and DPI.",
                    extra={
                        "replace": replace,
                        "no_llm": no_llm,
                        "target_dpi": int(scan.get("target_dpi") or body.target_dpi or 0),
                        "issue_counts": issue_counts,
                    },
                )
            except Exception:
                pass
            items.append({**base_item, "enqueued": True, "task_id": task_id})
            enqueued += 1
        except Exception as exc:
            items.append({**base_item, "error": str(exc)[:240] or "enqueue failed"})
            failed += 1

    return {
        "ok": failed == 0,
        "requested": len(targets),
        "scanned": int(scan.get("scanned") or 0),
        "figures": int(scan.get("figures") or 0),
        "docs_with_issues": int(scan.get("docs_with_issues") or 0),
        "refresh_recommended": int(scan.get("refresh_recommended") or 0),
        "issue_counts": scan.get("issue_counts") or {},
        "severity_counts": scan.get("severity_counts") or {},
        "enqueued": int(enqueued),
        "skipped_busy": int(skipped_busy),
        "failed": int(failed),
        "errors": errors[:20],
        "items": items,
    }


@router.post("/quality/repair")
def repair_library_quality(body: QualityRepairBody):
    settings = get_settings()
    pdf_d = _pdf_dir()
    md_d = _md_dir()
    md_d.mkdir(parents=True, exist_ok=True)

    speed_mode = str(body.speed_mode or "balanced").strip() or "balanced"
    requested_no_llm = bool(body.no_llm) or (speed_mode.lower() == "no_llm")
    replace = bool(body.replace)
    snap = _bg_snapshot()
    task_by_path, task_by_name = _build_task_maps_from_snapshot(snap)

    def _planned_queue_settings(plan: dict) -> tuple[str, bool, bool]:
        plan_speed = str((plan or {}).get("speed_mode") or "").strip()
        queue_speed = plan_speed or speed_mode
        queue_no_llm = bool((plan or {}).get("no_llm"))
        if requested_no_llm:
            queue_speed = speed_mode
            queue_no_llm = True
        if queue_speed.lower() == "no_llm":
            queue_no_llm = True
        queue_replace = bool((plan or {}).get("replace")) or replace
        return queue_speed, queue_no_llm, queue_replace

    requested = 0
    items: list[dict] = []
    targets: list[dict] = []
    repair_run_id = uuid.uuid4().hex

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
        md_path_raw = str(resolved.get("md_path") or "").strip()
        md_exists = bool(resolved.get("md_exists")) and bool(md_path_raw)
        if not pdf_path_raw and not md_exists:
            items.append({
                "source_path": source_path,
                "source_name": source_name,
                "pdf_name": "",
                "pdf_path": "",
                "ok": False,
                "enqueued": False,
                "skipped_busy": False,
                "error": "source pdf or markdown not found",
                "task_id": "",
            })
            continue
        targets.append({
            "source_path": source_path,
            "source_name": source_name,
            "pdf_path": pdf_path_raw,
            "md_path": md_path_raw,
        })

    enqueued = 0
    repaired = 0
    skipped_busy = 0
    failed = sum(1 for item in items if not bool(item.get("ok")))
    seen_target_paths: set[str] = set()
    for target in targets:
        pdf_path_raw = str(target.get("pdf_path") or "").strip()
        md_path_raw = str(target.get("md_path") or "").strip()
        pdf_path = Path(pdf_path_raw).expanduser() if pdf_path_raw else None
        pdf_name = pdf_path.name if pdf_path is not None else (
            str(target.get("source_name") or "").strip() or Path(md_path_raw.replace("\\", "/")).name
        )
        dedupe_key = _normalized_path_key(pdf_path) if pdf_path is not None else _normalized_path_key(md_path_raw)
        key = dedupe_key.lower()
        if not key or key in seen_target_paths:
            continue
        seen_target_paths.add(key)

        base_item = {
            "source_path": str(target.get("source_path") or ""),
            "source_name": str(target.get("source_name") or ""),
            "pdf_name": pdf_name,
            "pdf_path": str(pdf_path or ""),
            "ok": False,
            "enqueued": False,
            "repaired": False,
            "repair_changed": False,
            "repair_applied": [],
            "repair_before_score": 0,
            "repair_after_score": 0,
            "remaining_issue_codes": [],
            "repair_plan": {},
            "planned_action": "",
            "planned_scope": "",
            "planned_speed_mode": "",
            "planned_no_llm": False,
            "repair_attempt": {},
            "md_path": "",
            "skipped_busy": False,
            "error": "",
            "repair_error": "",
            "task_id": "",
        }
        pdf_available = bool(
            pdf_path is not None
            and _path_is_within(pdf_path, [pdf_d])
            and _path_is_file(pdf_path)
        )
        if pdf_path is not None and not pdf_available:
            items.append({**base_item, "error": "pdf not found"})
            failed += 1
            continue

        if pdf_available and pdf_path is not None:
            task_info = task_by_path.get(_normalized_path_key(pdf_path)) or task_by_name.get(pdf_name) or {}
            if bool(task_info.get("queued")) or bool(task_info.get("running")):
                items.append({**base_item, "skipped_busy": True, "error": "already queued or running"})
                skipped_busy += 1
                continue

        if md_path_raw:
            md_path = Path(md_path_raw).expanduser()
            md_exists = _path_is_file(md_path)
        elif pdf_path is not None:
            _md_folder, md_path, md_exists = _resolve_md_output_paths(md_d, pdf_path)
        else:
            md_path = Path()
            md_exists = False
        reader_locate_problems = _reader_locate_source_problem_events(
            source_path=str(target.get("source_path") or ""),
            source_name=str(target.get("source_name") or ""),
            pdf_path=str(pdf_path or ""),
            md_path=str(md_path) if md_exists else str(md_path_raw or ""),
            limit=20,
        )
        reader_locate_recommended_actions = sorted(
            set(
                str(item.get("recommended_action") or _reader_locate_recommended_action(item))
                for item in reader_locate_problems
                if str(item.get("recommended_action") or _reader_locate_recommended_action(item)).strip()
            )
        )
        repair_payload: dict = {}
        active_plan: dict = {}
        if bool(body.md_autofix) and md_exists and _path_is_within(md_path, [md_d]):
            try:
                before_quality = _conversion_quality_summary(md_path) or {}
                before_issues = [
                    str(issue.get("code") or "")
                    for issue in list(before_quality.get("issues") or [])
                    if isinstance(issue, dict) and str(issue.get("code") or "").strip()
                ]
                source_pdf_for_quality = pdf_path if (pdf_available and pdf_path is not None) else None
                repair_result = repair_markdown_quality(
                    md_path,
                    issue_codes=before_issues,
                    source_pdf_path=source_pdf_for_quality,
                )
                try:
                    write_conversion_quality_result(
                        md_path,
                        auto_repair_result=repair_result,
                        source_pdf_path=source_pdf_for_quality,
                    )
                except Exception:
                    pass
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
                after_plan = plan_conversion_quality_repair(
                    after_issue_codes,
                    metrics=after_quality.get("metrics") if isinstance(after_quality.get("metrics"), dict) else {},
                )
                active_plan = after_plan
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
                    "repair_plan": active_plan,
                    "planned_action": str(active_plan.get("action") or ""),
                    "planned_scope": str(active_plan.get("scope") or ""),
                    "md_path": str(md_path),
                    "repair_unsafe": bool(repair_result.get("unsafe")),
                    "repair_regression_reasons": list(repair_result.get("regression_reasons") or [])[:8],
                }
                try:
                    repair_payload["repair_attempt"] = append_conversion_repair_attempt(
                        md_path,
                        event="markdown_autofix",
                        status="partial" if str(active_plan.get("action") or "").lower() == "reconvert" else "success",
                        action=str(active_plan.get("action") or "autofix"),
                        scope=str(active_plan.get("scope") or "markdown"),
                        speed_mode=str(active_plan.get("speed_mode") or ""),
                        issue_codes=before_issue_codes,
                        source="library_quality_repair",
                        reason=str(active_plan.get("reason") or ""),
                        detail=(
                            "Markdown auto-repair applied before source reconversion."
                            if str(active_plan.get("action") or "").lower() == "reconvert"
                            else "Markdown auto-repair resolved conversion-quality issues."
                        ),
                        extra={
                            "changed": repair_changed,
                            "applied": list(repair_result.get("applied") or [])[:12],
                            "fixed_issue_codes": fixed_issue_codes[:12],
                            "remaining_issue_codes": after_issue_codes[:12],
                        },
                    )
                except Exception:
                    pass
            except Exception as exc:
                active_plan = plan_conversion_quality_repair(["quality_scan_failed"])
                repair_payload = {
                    "md_path": str(md_path),
                    "repair_plan": active_plan,
                    "planned_action": str(active_plan.get("action") or ""),
                    "planned_scope": str(active_plan.get("scope") or ""),
                    "repair_error": str(exc)[:240] or "markdown autofix failed",
                }
        elif md_exists:
            before_quality = _conversion_quality_summary(md_path) or {}
            before_issue_codes = [
                str(issue.get("code") or "")
                for issue in list(before_quality.get("issues") or [])
                if isinstance(issue, dict) and str(issue.get("code") or "").strip()
            ]
            active_plan = plan_conversion_quality_repair(
                before_issue_codes,
                metrics=before_quality.get("metrics") if isinstance(before_quality.get("metrics"), dict) else {},
            )
            repair_payload = {
                "md_path": str(md_path),
                "quality_before": before_quality,
                "remaining_issue_codes": before_issue_codes[:12],
                "repair_plan": active_plan,
                "planned_action": str(active_plan.get("action") or ""),
                "planned_scope": str(active_plan.get("scope") or ""),
            }
        else:
            active_plan = plan_conversion_quality_repair(["missing_markdown"])
            repair_payload = {
                "repair_plan": active_plan,
                "planned_action": str(active_plan.get("action") or ""),
                "planned_scope": str(active_plan.get("scope") or ""),
            }

        plan_action = str(active_plan.get("action") or "").strip().lower()
        reader_locate_reindex_required = bool(reader_locate_problems) and md_exists and plan_action in {"", "none"}
        if reader_locate_problems:
            repair_payload["reader_locate_problem_count"] = len(reader_locate_problems)
            repair_payload["reader_locate_recommended_actions"] = reader_locate_recommended_actions[:8]
            repair_payload["reader_locate_problem_keys"] = [
                str(item.get("locate_feedback_key") or item.get("id") or "")
                for item in reader_locate_problems[:8]
                if str(item.get("locate_feedback_key") or item.get("id") or "").strip()
            ]
        if reader_locate_reindex_required:
            repair_payload["reader_locate_reindex_required"] = True
            repair_payload["planned_action"] = "reindex"
            repair_payload["planned_scope"] = "source_blocks"
            repair_payload["repair_plan"] = {
                **dict(active_plan or {}),
                "action": "reindex",
                "scope": "source_blocks",
                "reason": "Reader locate reported a degraded or failed target; rebuild structured source anchors and indexes.",
                "issue_codes": list(active_plan.get("issue_codes") or []),
                "reader_locate_recommended_actions": reader_locate_recommended_actions[:8],
            }
        if md_exists and plan_action != "reconvert":
            if md_exists and _path_is_within(md_path, [md_d]):
                try:
                    attempt_event = "reader_locate_reindex_required" if reader_locate_reindex_required else "repair_closed"
                    repair_payload["repair_attempt"] = append_conversion_repair_attempt(
                        md_path,
                        event=attempt_event,
                        status=(
                            "reindex_pending"
                            if reader_locate_reindex_required
                            else ("success" if plan_action in {"", "none"} else str(plan_action or "success"))
                        ),
                        action="reindex" if reader_locate_reindex_required else str(active_plan.get("action") or "none"),
                        scope="source_blocks" if reader_locate_reindex_required else str(active_plan.get("scope") or ""),
                        speed_mode=str(active_plan.get("speed_mode") or ""),
                        issue_codes=list(active_plan.get("issue_codes") or []),
                        source="reader_locate_quality" if reader_locate_reindex_required else "library_quality_repair",
                        reason=(
                            "Reader locate failure/degradation requires source index rebuild."
                            if reader_locate_reindex_required
                            else str(active_plan.get("reason") or "")
                        ),
                        detail=(
                            "Reader locate failed or degraded while Markdown quality is otherwise closed; rebuild structured source anchors."
                            if reader_locate_reindex_required
                            else "No source reconversion required after quality repair planning."
                        ),
                        extra=(
                            {
                                "reader_locate_problem_count": len(reader_locate_problems),
                                "reader_locate_recommended_actions": reader_locate_recommended_actions[:8],
                            }
                            if reader_locate_reindex_required
                            else None
                        ),
                    )
                except Exception:
                    pass
            items.append({**base_item, **repair_payload, "ok": True, "enqueued": False})
            continue

        if not pdf_available or pdf_path is None:
            items.append({
                **base_item,
                **repair_payload,
                "error": "source pdf not found for reconversion",
            })
            failed += 1
            continue

        try:
            queue_speed_mode, queue_no_llm, queue_replace = _planned_queue_settings(active_plan)
            repair_payload["planned_speed_mode"] = queue_speed_mode
            repair_payload["planned_no_llm"] = bool(queue_no_llm)
            repair_context = {
                "action": str(active_plan.get("action") or "reconvert"),
                "scope": str(active_plan.get("scope") or ""),
                "reason": str(active_plan.get("reason") or ""),
                "source": "library_quality_repair",
                "repair_run_id": repair_run_id,
                "issue_codes": list(active_plan.get("issue_codes") or []),
            }
            task = _build_bg_task(
                pdf_path=pdf_path,
                out_root=md_d,
                db_dir=Path(settings.db_dir).expanduser(),
                no_llm=queue_no_llm,
                replace=queue_replace,
                speed_mode=queue_speed_mode,
                repair_context=repair_context,
            )
            _bg_enqueue(task)
            task_id = str(task.get("_tid") or "")
            if md_exists and _path_is_within(md_path, [md_d]):
                try:
                    repair_payload["repair_attempt"] = append_conversion_repair_attempt(
                        md_path,
                        event="reconvert_queued",
                        status="queued",
                        action=str(active_plan.get("action") or "reconvert"),
                        scope=str(active_plan.get("scope") or ""),
                        speed_mode=queue_speed_mode,
                        issue_codes=list(active_plan.get("issue_codes") or []),
                        task_id=task_id,
                        source="library_quality_repair",
                        reason=str(active_plan.get("reason") or ""),
                        detail="Source reconversion was queued from conversion-quality repair planning.",
                        extra={"replace": queue_replace, "no_llm": queue_no_llm},
                    )
                except Exception:
                    pass
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
    reader_locate_reindex_items = [item for item in items if bool(item.get("reader_locate_reindex_required"))]
    needs_reindex = bool(repaired_items) or enqueued > 0 or bool(reader_locate_reindex_items)
    impact = {
        "requested": int(requested),
        "repaired": int(len(repaired_items)),
        "improved": int(improved),
        "enqueued": int(enqueued),
        "skipped_busy": int(skipped_busy),
        "failed": int(failed),
        "needs_reindex": bool(needs_reindex),
        "reader_locate_reindex": int(len(reader_locate_reindex_items)),
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
    run_status, run_phase = _quality_repair_run_status(
        enqueued=int(enqueued),
        repaired=int(repaired),
        failed=int(failed),
        needs_reindex=bool(needs_reindex),
    )
    repair_run = _append_quality_repair_run({
        "run_id": repair_run_id,
        "status": run_status,
        "phase": run_phase,
        "requested": int(requested),
        "enqueued": int(enqueued),
        "repaired": int(repaired),
        "failed": int(failed),
        "skipped_busy": int(skipped_busy),
        "needs_reindex": bool(needs_reindex),
        "target_names": [
            str(item.get("pdf_name") or item.get("source_name") or "")
            for item in items
            if str(item.get("pdf_name") or item.get("source_name") or "").strip()
        ],
        "target_sources": [
            str(item.get("source_path") or item.get("md_path") or item.get("pdf_path") or "")
            for item in items
            if str(item.get("source_path") or item.get("md_path") or item.get("pdf_path") or "").strip()
        ],
        "impact": impact,
        "detail": (
            "Source reconversion queued; index refresh should run after conversion."
            if enqueued > 0
            else (
                "Reader locate source anchors need index refresh."
                if reader_locate_reindex_items and not repaired_items
                else ("Markdown source repair completed; index refresh is pending." if needs_reindex else "No source repair required.")
            )
        ),
    })

    return {
        "ok": failed == 0,
        "repair_run_id": repair_run_id,
        "repair_run": repair_run,
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
        resolved_path = _resolve_library_pdf_path_arg(path_raw)
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
        auto_apply_empty=bool(body.auto_apply_empty),
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
        resolved_path = _resolve_library_pdf_path_arg(path_raw)
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


def _run_library_reindex() -> dict:
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
    result = subprocess.run(
        [os.sys.executable, str(ingest_py), "--src", str(md_d), "--db", str(s.db_dir), "--incremental", "--prune"],
        capture_output=True, text=True, timeout=300,
    )
    ok = result.returncode == 0
    if ok:
        try:
            structured_indices = rebuild_structured_indices_for_root(md_d, force=False)
        except Exception as exc:
            structured_indices_error = str(exc)
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


@router.post("/reindex")
def reindex():
    return _run_library_reindex()
