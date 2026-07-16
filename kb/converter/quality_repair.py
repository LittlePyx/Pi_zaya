from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from .post_processing import postprocess_markdown
from .post_references import _is_post_references_resume_heading_line
from .quality_acceptance import summarize_conversion_quality
from .quality_compare import compare_markdown_quality
from .reference_markdown import (
    fix_references_format,
    normalize_references_page_text,
    _is_plausible_reference_number,
    _looks_like_author_year_reference_text,
)
from .pdf_reference_text import reference_ordered_page_text
from .reference_page_vl import reference_markdown_entry_count
from .tables import markdown_table_issue_counts, normalize_markdown_tables_document
from kb.reference_index import extract_references_map_from_md

try:
    import fitz
except ImportError:
    fitz = None


PAGE_MARKER_RE = re.compile(r"<!--\s*kb_page:\s*(\d+)\s*-->", re.IGNORECASE)
DISPLAY_MATH_DELIMITER_RE = re.compile(r"^\s*\$\$\s*$")
IMAGE_LINE_RE = re.compile(r"^(\s*)!\[([^\]]*)]\(([^)]+)\)\s*$")
CAPTION_LINE_RE = re.compile(
    r"^\s*(?:\*{1,2}\s*)?(?:fig(?:ure)?\.?|table|algorithm)\s*(?:S?\d+[A-Za-z]?|[A-Za-z](?:\.\d+)?|[IVXLC]+)\b",
    re.IGNORECASE,
)
REFERENCES_HEADING_RE = re.compile(r"^#{1,6}\s+(?:References|Bibliography)\s*$", re.IGNORECASE)
BODY_SECTION_HEADING_RE = re.compile(
    r"^#{1,6}\s+(?:\d+(?:\.\d+)*\.?\s+|[IVXLC]+\.\s+)?"
    r"(?:introduction|background|related\s+work|theory|principle|comparison|method(?:s|ology)?|"
    r"experiment(?:s|al)?|results?|discussion|conclusions?|implementation|analysis|system|structure)\b",
    re.IGNORECASE,
)


_PAGE_MARKER_OFFSETS_CACHE_MAX_ITEMS = 32
_PAGE_MARKER_OFFSETS_CACHE_LOCK = threading.Lock()
_PAGE_MARKER_OFFSETS_CACHE: dict[tuple[str, int, int, int, str, bool], dict[int, int]] = {}


def _page_marker_offsets_cache_key(
    md_text: str,
    pdf_path: Path,
    *,
    snap_to_line_start: bool,
) -> tuple[str, int, int, int, str, bool] | None:
    try:
        path = Path(pdf_path).expanduser().resolve()
        stat = path.stat()
        text = str(md_text or "")
        digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
        return (
            os.path.normcase(str(path)),
            int(stat.st_mtime_ns),
            int(stat.st_size),
            len(text),
            digest,
            bool(snap_to_line_start),
        )
    except Exception:
        return None


def _get_cached_page_marker_offsets(
    key: tuple[str, int, int, int, str, bool] | None,
) -> dict[int, int] | None:
    if key is None:
        return None
    with _PAGE_MARKER_OFFSETS_CACHE_LOCK:
        cached = _PAGE_MARKER_OFFSETS_CACHE.get(key)
        return dict(cached) if cached is not None else None


def _cache_page_marker_offsets(
    key: tuple[str, int, int, int, str, bool] | None,
    offsets: dict[int, int],
) -> None:
    if key is None:
        return
    with _PAGE_MARKER_OFFSETS_CACHE_LOCK:
        _PAGE_MARKER_OFFSETS_CACHE[key] = dict(offsets)
        while len(_PAGE_MARKER_OFFSETS_CACHE) > _PAGE_MARKER_OFFSETS_CACHE_MAX_ITEMS:
            oldest = next(iter(_PAGE_MARKER_OFFSETS_CACHE), None)
            if oldest is None:
                break
            _PAGE_MARKER_OFFSETS_CACHE.pop(oldest, None)


def _reference_map_missing_numbers(ref_map: dict[int, str]) -> list[int]:
    refs: set[int] = set()
    for raw_key, raw_value in (ref_map if isinstance(ref_map, dict) else {}).items():
        try:
            n = int(raw_key)
        except Exception:
            continue
        if n <= 0 or not str(raw_value or "").strip():
            continue
        refs.add(n)
    if len(refs) < 8:
        return []
    first = min(refs)
    last = max(refs)
    if first != 1 or last < 8:
        return []
    return sorted(int(n) for n in (set(range(first, last + 1)) - refs))


def _reference_gap_is_material(ref_map: dict[int, str]) -> bool:
    missing = _reference_map_missing_numbers(ref_map)
    if not missing:
        return False
    last = 0
    for raw_key in (ref_map if isinstance(ref_map, dict) else {}).keys():
        try:
            last = max(last, int(raw_key))
        except Exception:
            continue
    if last < 8:
        return False
    # A small tail gap can come from intentionally omitted supplement-only refs.
    # Interior gaps are almost always conversion damage.
    return any(int(n) < last for n in missing)


def _reference_map_has_short_truncated_entries(ref_map: dict[int, str]) -> bool:
    rows: list[tuple[int, str]] = []
    for raw_key, raw_value in (ref_map if isinstance(ref_map, dict) else {}).items():
        try:
            n = int(raw_key)
        except Exception:
            continue
        text = re.sub(r"\s+", " ", str(raw_value or "")).strip()
        if n > 0 and text:
            rows.append((n, text))
    if len(rows) < 8:
        return False
    bad_short = 0
    for _, text in sorted(rows):
        body = re.sub(r"^\s*(?:\[\s*\d{1,4}\s*\]|\d{1,4}[.)])\s*", "", text).strip(" .;:,")
        if not body:
            continue
        if re.search(r"\b(?:18|19|20)\d{2}\b|https?://|www\.|\bdoi\s*:|10\.\d{4,9}/", body, flags=re.IGNORECASE):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", body)
        if len(body) <= 32 and len(words) <= 5 and re.search(r"\bet\s+al\.?$", body, flags=re.IGNORECASE):
            return True
        if (
            len(body) <= 140
            and len(words) >= 4
            and (
                body.endswith(("-", ","))
                or re.search(r"\b(?:a|an|and|based|for|from|in|of|on|over|the|to|using|with)\s*$", body, re.IGNORECASE)
            )
        ):
            bad_short += 1
            continue
        if len(body) <= 90 and len(words) >= 4:
            bad_short += 1
    return bad_short >= 2


def _reference_map_has_inflated_tail(before_map: dict[int, str], recovered_map: dict[int, str]) -> bool:
    before_nums = sorted(
        int(k)
        for k, v in (before_map if isinstance(before_map, dict) else {}).items()
        if str(v or "").strip() and str(k).isdigit() and int(k) > 0
    )
    recovered_nums = sorted(
        int(k)
        for k, v in (recovered_map if isinstance(recovered_map, dict) else {}).items()
        if str(v or "").strip() and str(k).isdigit() and int(k) > 0
    )
    if len(before_nums) < 12 or len(recovered_nums) < 8:
        return False
    if len(recovered_nums) >= len(before_nums):
        return False
    if (len(recovered_nums) / max(1, len(before_nums))) < 0.45:
        return False
    if len(before_nums) < int(len(recovered_nums) * 1.35):
        return False
    if _reference_map_missing_numbers(recovered_map):
        return False
    if min(recovered_nums) != 1 or max(recovered_nums) < len(recovered_nums) * 0.85:
        return False
    recovered_max = max(recovered_nums)
    tail_nums = [n for n in before_nums if n > recovered_max]
    return len(tail_nums) >= max(5, int(len(before_nums) * 0.20))


CONVERSION_QUALITY_RESULT_FILENAME = "conversion_quality_result.json"
MAX_CONVERSION_REPAIR_ATTEMPTS = 30
PAGE_ALIGNMENT_NGRAMS = (8, 6)
PAGE_ALIGNMENT_DEFAULT_NGRAM = PAGE_ALIGNMENT_NGRAMS[0]
SOURCE_PAGE_COVERAGE_THRESHOLD = 0.66
SOURCE_PAGE_MIN_RARE_TOKENS = 60
SOURCE_PAGE_SEGMENT_COVERAGE_THRESHOLD = 0.32
PAGE_ALIGNMENT_ANCHOR_DRIFT_CHARS = 1200
PAGE_ALIGNMENT_BEAM_SIZE = 300
PAGE_ALIGNMENT_BEAM_PER_MATCH = 80
LEADING_PAGE_ALIGNMENT_MAX_OFFSET = 900
LEADING_PAGE_ALIGNMENT_WINDOW_CHARS = 3200
LEADING_PAGE_DROP_MAX_PREVIOUS_COVERAGE = 0.50
LEADING_PAGE_DROP_MIN_COVERAGE_MARGIN = 0.12
PAGE_ALIGNMENT_STOP_WORDS = {
    "the", "of", "and", "in", "to", "for", "with", "on", "by", "from", "that", "this",
    "these", "those", "using", "used", "into", "such", "which", "were", "their", "have",
    "has", "been", "between", "through", "under", "over", "figure", "table", "results",
    "method", "methods", "abstract", "introduction",
}


CONVERSION_REPAIR_STRATEGIES: dict[str, dict[str, Any]] = {
    "missing_markdown": {
        "label": "Convert the PDF because no Markdown output exists",
        "safe": False,
        "action": "reconvert",
        "scope": "document",
        "speed_mode": "normal",
        "reason": "No converted Markdown exists, so the source PDF must be converted.",
        "strategies": [],
    },
    "missing_images": {
        "label": "Reconvert the paper to rebuild missing image assets",
        "safe": False,
        "action": "reconvert",
        "scope": "assets",
        "speed_mode": "normal",
        "reason": "Markdown references image assets that are missing on disk.",
        "strategies": ["repair_missing_image_links"],
    },
    "mojibake": {
        "label": "Reconvert with vision mode to avoid encoding artifacts",
        "safe": False,
        "action": "reconvert",
        "scope": "document",
        "speed_mode": "normal",
        "reason": "Encoding artifacts usually originate in extraction and should not be indexed.",
        "strategies": [],
    },
    "weak_structure": {
        "label": "Reconvert with higher-quality structure extraction",
        "safe": False,
        "action": "reconvert",
        "scope": "document",
        "speed_mode": "normal",
        "reason": "The output has too little heading structure for reliable retrieval and location.",
        "strategies": [],
    },
    "missing_references": {
        "label": "Reconvert to recover the reference section",
        "safe": False,
        "action": "reconvert",
        "scope": "references",
        "speed_mode": "normal",
        "reason": "The bibliography is missing, which breaks upstream citations and metadata repair.",
        "strategies": [],
    },
    "source_text_loss": {
        "label": "Reconvert because body text appears to be missing from the Markdown",
        "safe": False,
        "action": "reconvert",
        "scope": "document",
        "speed_mode": "normal",
        "reason": "The converted Markdown is much shorter than the source PDF or reaches references before recovering the body.",
        "strategies": [],
    },
    "missing_source_pages": {
        "label": "Recover missing PDF page text into Markdown",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "One or more source PDF pages have too little text represented in Markdown.",
        "strategies": ["recover_missing_source_pages"],
    },
    "source_page_marker_alignment": {
        "label": "Realign Markdown page anchors from the source PDF",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "The source PDF page starts do not line up with the Markdown page anchors.",
        "strategies": ["realign_page_markers_from_pdf", "recover_missing_source_pages"],
    },
    "reference_index_truncated": {
        "label": "Rebuild the reference section from the source PDF text layer",
        "safe": True,
        "action": "autofix",
        "scope": "references",
        "reason": "The visible reference lines are interrupted or not extractable as a complete reference index.",
        "strategies": ["pdf_reference_backfill"],
    },
    "references_before_body": {
        "label": "Move an early References block behind recovered body sections",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["move_early_references_to_end"],
    },
    "analyzer_errors": {
        "label": "Run deterministic Markdown post-processing",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["normalize_markdown_tables", "postprocess_markdown", "balance_display_math", "figure_metadata_captions"],
    },
    "analyzer_warnings": {
        "label": "Normalize headings, captions, tables, and layout noise",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["normalize_markdown_tables", "postprocess_markdown", "figure_metadata_captions", "pdf_text_captions"],
    },
    "collapsed_table_rows": {
        "label": "Recover collapsed Markdown table rows",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "Multiple logical data rows were packed into cells with literal HTML break markers.",
        "strategies": ["normalize_markdown_tables"],
    },
    "duplicate_table_representations": {
        "label": "Remove a nearby lower-quality duplicate table",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "The same table data appears in nearby compact and structured representations.",
        "strategies": ["normalize_markdown_tables"],
    },
    "missing_abstract": {
        "label": "Infer and insert Abstract heading from front matter",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["abstract_heading_only", "postprocess_markdown"],
    },
    "missing_page_markers": {
        "label": "Insert a fallback page anchor at the Markdown start",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["ensure_page_anchor"],
    },
    "page_marker_gaps": {
        "label": "Normalize duplicate or out-of-order page anchors",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["realign_page_markers_from_pdf", "normalize_page_markers", "postprocess_markdown"],
    },
    "missing_captions": {
        "label": "Recover visible captions from alt text and figure metadata sidecars",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["postprocess_markdown", "figure_metadata_captions", "pdf_text_captions"],
    },
    "unclosed_display_math": {
        "label": "Close a trailing display-math block when the delimiter is unbalanced",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["balance_display_math"],
    },
    "heading_level_jumps": {
        "label": "Rebalance heading levels using the established heading policy",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["normalize_heading_levels"],
    },
    "collapsed_heading_hierarchy": {
        "label": "Promote collapsed review-paper headings for better navigation",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "A review-style paper has many section headings collapsed under H3, which weakens reader navigation and retrieval heading paths.",
        "strategies": ["promote_collapsed_review_headings"],
    },
    "stray_inline_math": {
        "label": "Clean stray inline math markup",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "Prose contains bare LaTeX or OCR math leftovers that should be normalized as inline math.",
        "strategies": ["postprocess_markdown", "promote_collapsed_review_headings"],
    },
}


def conversion_repair_strategy_for_issue(code: str) -> dict[str, Any]:
    clean = str(code or "").strip().lower()
    strategy = CONVERSION_REPAIR_STRATEGIES.get(clean)
    if not strategy:
        return {
            "label": "",
            "safe": False,
            "strategies": [],
        }
    return {
        "label": str(strategy.get("label") or ""),
        "safe": bool(strategy.get("safe")),
        "action": str(strategy.get("action") or ("autofix" if bool(strategy.get("safe")) else "")),
        "scope": str(strategy.get("scope") or ""),
        "speed_mode": str(strategy.get("speed_mode") or ""),
        "reason": str(strategy.get("reason") or ""),
        "strategies": [str(item) for item in list(strategy.get("strategies") or []) if str(item or "").strip()],
    }


def conversion_quality_result_path(md_path: Path | str) -> Path:
    return Path(md_path).expanduser().parent / CONVERSION_QUALITY_RESULT_FILENAME


def plan_conversion_quality_repair(issue_codes: list[str] | None, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    codes = []
    seen: set[str] = set()
    for raw in list(issue_codes or []):
        code = str(raw or "").strip().lower()
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    issue_actions: list[dict[str, Any]] = []
    reconvert_actions: list[dict[str, Any]] = []
    autofix_actions: list[dict[str, Any]] = []
    review_actions: list[dict[str, Any]] = []
    for code in codes:
        strategy = conversion_repair_strategy_for_issue(code)
        action = str(strategy.get("action") or "").strip() or ("autofix" if bool(strategy.get("safe")) else "review")
        row = {
            "code": code,
            "action": action,
            "safe": bool(strategy.get("safe")),
            "scope": str(strategy.get("scope") or ""),
            "label": str(strategy.get("label") or code),
            "reason": str(strategy.get("reason") or ""),
            "speed_mode": str(strategy.get("speed_mode") or ""),
            "strategies": list(strategy.get("strategies") or []),
        }
        issue_actions.append(row)
        if action == "reconvert":
            reconvert_actions.append(row)
        elif action == "autofix":
            autofix_actions.append(row)
        else:
            review_actions.append(row)

    if reconvert_actions:
        scope_priority = {"document": 4, "assets": 3, "references": 2, "markdown": 1}
        primary = sorted(
            reconvert_actions,
            key=lambda item: scope_priority.get(str(item.get("scope") or ""), 0),
            reverse=True,
        )[0]
        return {
            "action": "reconvert",
            "scope": str(primary.get("scope") or "document"),
            "speed_mode": str(primary.get("speed_mode") or "normal") or "normal",
            "no_llm": False,
            "replace": True,
            "md_autofix_first": bool(autofix_actions),
            "reason": str(primary.get("reason") or primary.get("label") or "Source conversion needs repair."),
            "issue_codes": codes,
            "reconvert_issue_codes": [str(item.get("code") or "") for item in reconvert_actions],
            "autofix_issue_codes": [str(item.get("code") or "") for item in autofix_actions],
            "review_issue_codes": [str(item.get("code") or "") for item in review_actions],
            "issue_actions": issue_actions,
            "metrics": dict(metrics or {}),
        }
    if autofix_actions:
        return {
            "action": "autofix",
            "scope": "markdown",
            "speed_mode": "",
            "no_llm": False,
            "replace": False,
            "md_autofix_first": True,
            "reason": "Remaining issues are covered by deterministic Markdown repair.",
            "issue_codes": codes,
            "reconvert_issue_codes": [],
            "autofix_issue_codes": [str(item.get("code") or "") for item in autofix_actions],
            "review_issue_codes": [str(item.get("code") or "") for item in review_actions],
            "issue_actions": issue_actions,
            "metrics": dict(metrics or {}),
        }
    if review_actions:
        return {
            "action": "review",
            "scope": "manual",
            "speed_mode": "",
            "no_llm": False,
            "replace": False,
            "md_autofix_first": False,
            "reason": "The remaining issues do not have a safe automated repair yet.",
            "issue_codes": codes,
            "reconvert_issue_codes": [],
            "autofix_issue_codes": [],
            "review_issue_codes": [str(item.get("code") or "") for item in review_actions],
            "issue_actions": issue_actions,
            "metrics": dict(metrics or {}),
        }
    return {
        "action": "none",
        "scope": "",
        "speed_mode": "",
        "no_llm": False,
        "replace": False,
        "md_autofix_first": False,
        "reason": "No conversion quality issues detected.",
        "issue_codes": [],
        "reconvert_issue_codes": [],
        "autofix_issue_codes": [],
        "review_issue_codes": [],
        "issue_actions": [],
        "metrics": dict(metrics or {}),
    }


def _current_markdown_stat(path: Path) -> dict[str, int]:
    try:
        stat = path.stat()
        return {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}
    except Exception:
        return {"mtime_ns": 0, "size": 0}


def load_conversion_quality_result(md_path: Path | str) -> dict[str, Any]:
    report_path = conversion_quality_result_path(md_path)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def append_conversion_repair_attempt(
    md_path: Path | str,
    *,
    event: str,
    status: str,
    action: str = "",
    scope: str = "",
    speed_mode: str = "",
    issue_codes: list[str] | None = None,
    task_id: str = "",
    repair_run_id: str = "",
    source: str = "",
    reason: str = "",
    detail: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    report_path = conversion_quality_result_path(path)
    report = load_conversion_quality_result(path)
    if not report and path.exists():
        try:
            report = write_conversion_quality_result(path)
        except Exception:
            report = {}
    if not isinstance(report, dict):
        report = {}

    row = {
        "event": str(event or "").strip() or "repair",
        "status": str(status or "").strip() or "info",
        "action": str(action or "").strip(),
        "scope": str(scope or "").strip(),
        "speed_mode": str(speed_mode or "").strip(),
        "issue_codes": [str(item or "").strip() for item in list(issue_codes or []) if str(item or "").strip()][:30],
        "task_id": str(task_id or "").strip(),
        "repair_run_id": str(repair_run_id or "").strip(),
        "source": str(source or "").strip(),
        "reason": str(reason or "").strip()[:500],
        "detail": str(detail or "").strip()[:800],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if isinstance(extra, dict) and extra:
        row["extra"] = {
            str(k): v
            for k, v in extra.items()
            if isinstance(k, str) and k and isinstance(v, (str, int, float, bool, type(None), list, dict))
        }

    attempts = report.get("repair_attempts") if isinstance(report.get("repair_attempts"), list) else []
    attempts = [item for item in attempts if isinstance(item, dict)]
    attempts.append(row)
    report["repair_attempts"] = attempts[-MAX_CONVERSION_REPAIR_ATTEMPTS:]
    report["latest_repair_attempt"] = row
    report.setdefault("schema_version", 1)
    report.setdefault("kind", "conversion_quality_result")
    report["md_path"] = str(path)
    try:
        stat = path.stat()
        report["md_mtime_ns"] = int(stat.st_mtime_ns)
        report["md_size"] = int(stat.st_size)
    except Exception:
        report.setdefault("md_mtime_ns", 0)
        report.setdefault("md_size", 0)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return row


def write_conversion_quality_result(
    md_path: Path | str,
    *,
    auto_repair_result: dict[str, Any] | None = None,
    auto_repair_enabled: bool = True,
    source_pdf_path: Path | str | None = None,
    allow_source_pdf_inference: bool = True,
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    report_path = conversion_quality_result_path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    metrics = _metric_view(path, text)
    source_quality = _source_quality_view(
        path,
        text,
        metrics,
        source_pdf_path=source_pdf_path,
        allow_source_pdf_inference=allow_source_pdf_inference,
    )
    repair = dict(auto_repair_result or {})
    repair.pop("repaired_text", None)
    remaining = [
        str(code or "").strip().lower()
        for code in list(repair.get("remaining_issue_codes") or _issue_codes_from_context(path, text, metrics, source_quality=source_quality))
        if str(code or "").strip()
    ]
    md_stat = _current_markdown_stat(path)
    prev = load_conversion_quality_result(path)
    exhausted_issue_codes = _persistent_source_autofix_codes(repair) if auto_repair_result is not None else set()
    if auto_repair_result is None:
        prev_stat_matches = (
            int(prev.get("md_mtime_ns") or 0) == md_stat["mtime_ns"]
            and int(prev.get("md_size") or 0) == md_stat["size"]
        )
        prev_auto_repair = prev.get("auto_repair") if isinstance(prev.get("auto_repair"), dict) else {}
        if prev_stat_matches:
            exhausted_issue_codes = {
                str(code or "").strip().lower()
                for code in list((prev_auto_repair or {}).get("exhausted_issue_codes") or [])
                if str(code or "").strip().lower() in remaining
            }
    repair_plan = plan_conversion_quality_repair(remaining, metrics=metrics)
    if exhausted_issue_codes:
        repair_plan = _escalate_persistent_source_autofix(
            repair_plan,
            {
                "issue_codes_before": sorted(exhausted_issue_codes),
                "remaining_issue_codes": remaining,
            },
            source_available=bool(source_quality.get("source_pdf_available")),
        )
    recommended_action = str(repair_plan.get("action") or "review")
    prev_attempts = prev.get("repair_attempts") if isinstance(prev.get("repair_attempts"), list) else []
    prev_attempts = [item for item in prev_attempts if isinstance(item, dict)][-MAX_CONVERSION_REPAIR_ATTEMPTS:]
    latest_attempt = prev.get("latest_repair_attempt") if isinstance(prev.get("latest_repair_attempt"), dict) else (prev_attempts[-1] if prev_attempts else {})
    payload = {
        "schema_version": 1,
        "kind": "conversion_quality_result",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "md_path": str(path),
        "md_mtime_ns": md_stat["mtime_ns"],
        "md_size": md_stat["size"],
        "auto_repair_enabled": bool(auto_repair_enabled),
        "auto_repair": {
            "changed": bool(repair.get("changed")),
            "unsafe": bool(repair.get("unsafe")),
            "applied": [str(item) for item in list(repair.get("applied") or []) if str(item or "").strip()][:20],
            "attempted_issue_codes": [
                str(item)
                for item in list(repair.get("attempted_issue_codes") or [])
                if str(item or "").strip()
            ][:30],
            "issue_codes_before": [str(item) for item in list(repair.get("issue_codes_before") or []) if str(item or "").strip()][:30],
            "issue_codes_after": [str(item) for item in list(repair.get("issue_codes_after") or []) if str(item or "").strip()][:30],
            "remaining_issue_codes": remaining[:30],
            "exhausted_issue_codes": sorted(exhausted_issue_codes)[:30],
            "regression_reasons": [str(item) for item in list(repair.get("regression_reasons") or []) if str(item or "").strip()][:20],
        },
        "repair_plan": repair_plan,
        "repair_attempts": prev_attempts,
        "latest_repair_attempt": latest_attempt,
        "recommended_action": recommended_action,
        "needs_reconvert": recommended_action == "reconvert",
        "metrics": metrics,
        "source_quality": source_quality,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


_PERSISTENT_SOURCE_AUTOFIX_ISSUES = {
    "page_marker_gaps",
    "source_page_marker_alignment",
}


def _persistent_source_autofix_codes(repair_result: Mapping[str, Any]) -> set[str]:
    attempted = {
        str(code or "").strip().lower()
        for code in list(repair_result.get("attempted_issue_codes") or repair_result.get("issue_codes_before") or [])
        if str(code or "").strip()
    }
    remaining = {
        str(code or "").strip().lower()
        for code in list(repair_result.get("remaining_issue_codes") or [])
        if str(code or "").strip()
    }
    return attempted & remaining & _PERSISTENT_SOURCE_AUTOFIX_ISSUES


def _escalate_persistent_source_autofix(
    repair_plan: dict[str, Any],
    repair_result: Mapping[str, Any],
    *,
    source_available: bool,
) -> dict[str, Any]:
    """Stop offering a deterministic repair when the same source issue survives it."""
    persistent = _persistent_source_autofix_codes(repair_result)
    if not persistent:
        return repair_plan

    plan = dict(repair_plan)
    issue_actions: list[dict[str, Any]] = []
    for raw in list(plan.get("issue_actions") or []):
        row = dict(raw) if isinstance(raw, Mapping) else {}
        code = str(row.get("code") or "").strip().lower()
        if code in persistent:
            row.update(
                {
                    "action": "reconvert",
                    "safe": False,
                    "scope": "document",
                    "reason": "Page anchors remain inconsistent after deterministic Markdown repair.",
                    "speed_mode": "normal",
                    "strategies": ["source_reconversion"],
                }
            )
        issue_actions.append(row)

    autofix_codes = [
        str(code)
        for code in list(plan.get("autofix_issue_codes") or [])
        if str(code).strip().lower() not in persistent
    ]
    if source_available:
        plan.update(
            {
                "action": "reconvert",
                "scope": "document",
                "speed_mode": "normal",
                "no_llm": False,
                "replace": True,
                "md_autofix_first": bool(autofix_codes),
                "reason": "Page anchors remain inconsistent after deterministic repair; rerun source conversion.",
                "reconvert_issue_codes": sorted(persistent),
                "autofix_issue_codes": autofix_codes,
                "issue_actions": issue_actions,
            }
        )
    else:
        for row in issue_actions:
            if str(row.get("code") or "").strip().lower() in persistent:
                row.update(
                    {
                        "action": "review",
                        "scope": "manual",
                        "speed_mode": "",
                        "reason": "Page anchors remain inconsistent, but the source PDF is unavailable.",
                        "strategies": [],
                    }
                )
        review_codes = list(plan.get("review_issue_codes") or []) + sorted(persistent)
        plan.update(
            {
                "action": "review",
                "scope": "manual",
                "speed_mode": "",
                "no_llm": False,
                "replace": False,
                "md_autofix_first": bool(autofix_codes),
                "reason": "Page anchors need review because the source PDF is unavailable.",
                "reconvert_issue_codes": [],
                "autofix_issue_codes": autofix_codes,
                "review_issue_codes": list(dict.fromkeys(str(code) for code in review_codes)),
                "issue_actions": issue_actions,
            }
        )
    return plan


def _metric_view(md_path: Path, text: str) -> dict[str, Any]:
    metrics = summarize_conversion_quality(md_path, text)
    return asdict(metrics)


def _issue_codes_from_metrics(metrics: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if int(metrics.get("missing_image_count") or 0) > 0:
        out.append("missing_images")
    if int(metrics.get("unclosed_display_math_block_count") or 0) > 0:
        out.append("unclosed_display_math")
    if int(metrics.get("mojibake_count") or 0) > 0:
        out.append("mojibake")
    if int(metrics.get("analyzer_error_count") or 0) > 0:
        out.append("analyzer_errors")
    if int(metrics.get("heading_count") or 0) <= 1:
        out.append("weak_structure")
    if not bool(metrics.get("has_abstract_heading")):
        out.append("missing_abstract")
    if int(metrics.get("page_marker_count") or 0) <= 0:
        out.append("missing_page_markers")
    if int(metrics.get("page_marker_gap_count") or 0) > 0:
        out.append("page_marker_gaps")
    if int(metrics.get("extracted_reference_count") or 0) <= 0 and int(metrics.get("reference_line_count") or 0) <= 0:
        out.append("missing_references")
    if int(metrics.get("image_count") or 0) > 0 and int(metrics.get("caption_count") or 0) <= 0:
        out.append("missing_captions")
    if int(metrics.get("analyzer_warning_count") or 0) > 3:
        out.append("analyzer_warnings")
    if int(metrics.get("heading_level_jump_count") or 0) > 0:
        out.append("heading_level_jumps")
    return out


def _dedupe_codes(codes: list[str]) -> list[str]:
    out: list[str] = []
    for raw in codes:
        code = str(raw or "").strip().lower()
        if code and code not in out:
            out.append(code)
    return out


def _collapsed_heading_hierarchy_likely(text: str, metrics: dict[str, Any], quality: dict[str, Any]) -> bool:
    if str((quality or {}).get("document_type") or "").strip().lower() != "review":
        return False
    heading_count = int(metrics.get("heading_count") or 0)
    h2_count = int(metrics.get("h2_count") or 0)
    h3_plus_count = int(metrics.get("h3_plus_count") or 0)
    if heading_count < 10 or h3_plus_count < 8:
        return False
    if h2_count >= 8:
        return False
    if h3_plus_count < max(8, h2_count * 2):
        return False
    # Require at least a few topical H3 headings. This avoids flagging papers
    # that only use H3 for appendices, captions, or small front-matter blocks.
    topical = 0
    for line in str(text or "").splitlines():
        match = re.match(r"^###\s+(.+?)\s*$", line)
        if not match:
            continue
        title = str(match.group(1) or "").strip()
        if _review_heading_promotable(title):
            topical += 1
        if topical >= 4:
            return True
    return False


_STRAY_INLINE_LATEX_RE = re.compile(
    r"\\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|chi|psi|omega|"
    r"Theta|Sigma|hat|mathbf|boldsymbol)"
    r"(?:\s*(?:[_^]\{[^{}\n]{1,60}\}|[_^][A-Za-z0-9]))?"
    r"(?:\s*(?:[<>=]|\\leq|\\geq|\\neq|\\approx|\\in|\\notin)\s*"
    r"(?:[-+]?\d+(?:\.\d+)?|[A-Za-z](?:[_^]\{[^{}\n]{1,60}\}|[_^][A-Za-z0-9])?|"
    r"\\[A-Za-z]+(?:\{[^{}\n]{1,60}\})?|\[[^\]\n]{1,80}\]))+",
)
_STRAY_INLINE_CITATION_DOLLAR_RE = re.compile(
    r"\[\s*\d{1,4}(?:\s*[,;\u2013-]\s*\d{1,4})*\s*\]\$\s*(?=(?:and|or|the|this|that|these|those|[A-Z]))"
)
_STRAY_INLINE_CDOT_RE = re.compile(r"\bloss\s*\(\s*c(?:dot|[.\u00b7\u22c5])\s*\)", re.IGNORECASE)
_STRAY_INLINE_UNCLOSED_SENTENCE_RE = re.compile(
    r"\$(\\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|chi|psi|omega|"
    r"Theta|Sigma|hat|mathbf|boldsymbol)[^$\n]{1,180}?)([.?!])(?=\s+[A-Z])"
)


def _stray_inline_math_likely(text: str) -> bool:
    if not text or ("\\" not in text and "$" not in text and "cdot" not in text):
        return False
    in_fence = False
    in_math = False
    in_refs = False
    for line in str(text or "").splitlines():
        st = line.strip()
        if re.match(r"^\s*```", line):
            in_fence = not in_fence
            continue
        if st == "$$":
            in_math = not in_math
            continue
        if REFERENCES_HEADING_RE.match(st):
            in_refs = True
            continue
        if in_fence or in_math or in_refs:
            continue
        if re.match(r"^#{1,6}\s+", st) or re.match(r"^\s*!\[[^\]]*\]\([^)]+\)", st):
            continue
        if (
            _STRAY_INLINE_CITATION_DOLLAR_RE.search(line)
            or _STRAY_INLINE_CDOT_RE.search(line)
            or _STRAY_INLINE_UNCLOSED_SENTENCE_RE.search(line)
        ):
            return True
        probe = re.sub(r"\$[^$\n]*\$", " ", line)
        if _STRAY_INLINE_LATEX_RE.search(probe):
            return True
    return False


def _issue_codes_from_context(
    md_path: Path,
    text: str,
    metrics: dict[str, Any],
    *,
    source_quality: dict[str, Any] | None = None,
) -> list[str]:
    quality = source_quality if isinstance(source_quality, dict) else _source_quality_view(md_path, text, metrics)
    codes = _issue_codes_from_metrics(metrics)
    if bool(quality.get("abstract_not_applicable")):
        codes = [code for code in codes if code not in {"missing_abstract", "missing_references", "weak_structure"}]
    elif "missing_abstract" in codes and not bool(quality.get("abstract_autofix_likely")):
        codes = [code for code in codes if code != "missing_abstract"]
    if bool(quality.get("source_text_loss")):
        codes.insert(0, "source_text_loss")
        codes = [code for code in codes if code not in {"missing_abstract", "weak_structure"}]
    if int(quality.get("missing_source_page_count") or 0) > 0:
        codes.append("missing_source_pages")
    if int(quality.get("source_page_anchor_issue_count") or 0) > 0:
        codes.append("source_page_marker_alignment")
    if bool(quality.get("reference_index_truncated")):
        codes.append("reference_index_truncated")
    if bool(quality.get("references_before_body")):
        codes.append("references_before_body")
    if _collapsed_heading_hierarchy_likely(text, metrics, quality):
        codes.append("collapsed_heading_hierarchy")
    if _stray_inline_math_likely(text):
        codes.append("stray_inline_math")
    table_issues = markdown_table_issue_counts(text)
    if int(table_issues.get("literal_break_count") or 0) > 0 or int(table_issues.get("collapsed_row_count") or 0) > 0:
        codes.append("collapsed_table_rows")
    if int(table_issues.get("duplicate_table_count") or 0) > 0:
        codes.append("duplicate_table_representations")
    return _dedupe_codes(codes)


def _ensure_page_anchor(md: str) -> tuple[str, bool]:
    text = str(md or "")
    if PAGE_MARKER_RE.search(text):
        return text, False
    stripped = text.lstrip()
    prefix_len = len(text) - len(stripped)
    fixed = f"{text[:prefix_len]}<!-- kb_page: 1 -->\n\n{stripped}" if stripped else "<!-- kb_page: 1 -->"
    return fixed, fixed != text


def _normalize_page_marker_sequence(md: str) -> tuple[str, bool]:
    text = str(md or "")
    matches = list(PAGE_MARKER_RE.finditer(text))
    if len(matches) <= 1:
        return text, False
    removable_spans: list[tuple[int, int]] = []
    for previous, current in zip(matches, matches[1:]):
        if int(previous.group(1)) != int(current.group(1)):
            continue
        if text[int(previous.end()) : int(current.start())].strip():
            continue
        removable_spans.append((int(current.start()), int(current.end())))
    if not removable_spans:
        return text, False
    fixed = text
    for start, end in reversed(removable_spans):
        fixed = fixed[:start] + fixed[end:]
    return fixed, fixed != text


def _balance_display_math(md: str) -> tuple[str, bool]:
    lines = str(md or "").splitlines()
    out: list[str] = []
    in_math = False
    changed = False
    structural_re = re.compile(r"^\s*(?:#{1,6}\s+\S.*|!\[[^\]]*]\([^)]+\)|\|.+\|)\s*$")
    for line in lines:
        if DISPLAY_MATH_DELIMITER_RE.match(line or ""):
            in_math = not in_math
            out.append(line)
            continue
        if in_math and structural_re.match(line or ""):
            out.append("$$")
            changed = True
            in_math = False
        out.append(line)
    if in_math:
        out.append("$$")
        changed = True
    fixed = "\n".join(out)
    return fixed, changed and fixed != str(md or "")


def _asset_name_from_image_target(target: str) -> str:
    raw = str(target or "").strip().strip("<>")
    raw = raw.split("#", 1)[0].split("?", 1)[0]
    if not raw:
        return ""
    return Path(unquote(raw.replace("\\", "/"))).name


def _normalize_caption_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    text = re.sub(r"^\*+|\*+$", "", text).strip()
    if not CAPTION_LINE_RE.match(text):
        return ""
    if re.fullmatch(r"(?i)(?:fig(?:ure)?\.?|table|algorithm)\s*\d*[A-Za-z]?", text):
        return ""
    if len(text.split()) < 4 and len(text) < 32:
        return ""
    return text


def _format_caption_line(caption: str) -> str:
    text = _normalize_caption_text(caption)
    if not text:
        return ""
    m = re.match(r"(?i)^(fig(?:ure)?\.?|table|algorithm)\s*([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?)\.?\s*(.*)$", text)
    if not m:
        return f"*{text}*"
    kind_raw = m.group(1).lower()
    kind = "Figure" if kind_raw.startswith("fig") else ("Table" if kind_raw.startswith("table") else "Algorithm")
    ident = m.group(2)
    tail = (m.group(3) or "").strip()
    prefix = f"**{kind} {ident}.**"
    return f"{prefix} {tail}".rstrip()


def _load_figure_metadata_captions(md_path: Path) -> dict[str, str]:
    assets_dir = md_path.parent / "assets"
    if not assets_dir.exists():
        return {}
    out: dict[str, str] = {}

    def add(asset_name: str, caption: Any) -> None:
        name = str(asset_name or "").strip()
        line = _format_caption_line(str(caption or ""))
        if name and line and name not in out:
            out[name] = line

    for path in [assets_dir / "figure_index.json", *sorted(assets_dir.glob("page_*_fig_index.json"))]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        figures = payload.get("figures") if isinstance(payload, dict) else None
        if not isinstance(figures, list):
            continue
        for item in figures:
            if not isinstance(item, dict):
                continue
            add(str(item.get("asset_name") or item.get("asset_name_raw") or item.get("asset_name_alias") or ""), item.get("caption"))

    for path in sorted(assets_dir.glob("*.meta.json"))[:500]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        add(str(payload.get("asset_name") or payload.get("asset_name_raw") or f"{path.stem}.png"), payload.get("caption"))

    return out


def _has_caption_nearby(lines: list[str], image_idx: int) -> bool:
    for idx in range(image_idx + 1, min(len(lines), image_idx + 5)):
        st = (lines[idx] or "").strip()
        if not st:
            continue
        return bool(CAPTION_LINE_RE.match(st))
    return False


def _inject_figure_metadata_captions(md_path: Path, md: str) -> tuple[str, bool]:
    captions = _load_figure_metadata_captions(md_path)
    if not captions:
        return md, False
    lines = str(md or "").splitlines()
    out: list[str] = []
    changed = False
    for idx, line in enumerate(lines):
        out.append(line)
        m = IMAGE_LINE_RE.match(line or "")
        if not m or _has_caption_nearby(lines, idx):
            continue
        asset_name = _asset_name_from_image_target(m.group(3) or "")
        caption_line = captions.get(asset_name) or _format_caption_line(m.group(2) or "")
        if not caption_line:
            continue
        out.append("")
        out.append(caption_line)
        changed = True
    return "\n".join(out), changed


def _asset_page_and_fig(asset_name: str) -> tuple[int, int]:
    m = re.search(r"page_(\d+)_fig_(\d+)", str(asset_name or ""), flags=re.IGNORECASE)
    if not m:
        return (0, 0)
    try:
        return (int(m.group(1)), int(m.group(2)))
    except Exception:
        return (0, 0)


def _caption_fig_no_from_text(text: str) -> int:
    m = re.match(r"^\s*(?:\*{1,2}\s*)?fig(?:ure)?\.?\s*S?(\d{1,4})\b", str(text or ""), flags=re.IGNORECASE)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def _pdf_page_caption_candidates(page: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        text = _normalize_caption_text(raw)
        if not text:
            return
        key = re.sub(r"\s+", " ", text).strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        out.append({"fig_no": _caption_fig_no_from_text(text), "caption": text})

    try:
        data = page.get_text("dict") or {}
        for block in data.get("blocks", []) or []:
            parts: list[str] = []
            for line in block.get("lines", []) or []:
                spans = line.get("spans", []) or []
                raw = "".join(str(span.get("text", "")) for span in spans)
                if raw.strip():
                    parts.append(raw.strip())
            if parts:
                add(" ".join(parts))
    except Exception:
        pass

    try:
        for raw in str(page.get_text("text") or "").splitlines():
            add(raw)
    except Exception:
        pass
    return out


def _load_pdf_text_captions_by_page(source_pdf_path: Path | str | None) -> dict[int, list[dict[str, Any]]]:
    if fitz is None or not source_pdf_path:
        return {}
    path = Path(source_pdf_path).expanduser()
    try:
        if not path.exists() or not path.is_file():
            return {}
    except Exception:
        return {}
    out: dict[int, list[dict[str, Any]]] = {}
    try:
        doc = fitz.open(str(path))
    except Exception:
        return {}
    try:
        for page_index in range(len(doc)):
            try:
                candidates = _pdf_page_caption_candidates(doc.load_page(page_index))
            except Exception:
                candidates = []
            if candidates:
                out[page_index + 1] = candidates
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return out


def _choose_pdf_caption_for_image(
    *,
    asset_name: str,
    alt_text: str,
    captions_by_page: dict[int, list[dict[str, Any]]],
) -> str:
    page_no, page_fig_index = _asset_page_and_fig(asset_name)
    if page_no <= 0:
        return ""
    candidates = list(captions_by_page.get(page_no) or [])
    if not candidates:
        return ""
    alt_fig_no = _caption_fig_no_from_text(alt_text)
    if alt_fig_no > 0:
        for item in candidates:
            try:
                if int(item.get("fig_no") or 0) == alt_fig_no:
                    return _format_caption_line(str(item.get("caption") or ""))
            except Exception:
                continue
    numbered = [item for item in candidates if int(item.get("fig_no") or 0) > 0]
    if len(numbered) == 1:
        return _format_caption_line(str(numbered[0].get("caption") or ""))
    if page_fig_index > 0 and page_fig_index <= len(numbered):
        return _format_caption_line(str(numbered[page_fig_index - 1].get("caption") or ""))
    if len(candidates) == 1:
        return _format_caption_line(str(candidates[0].get("caption") or ""))
    return ""


def _inject_pdf_text_captions(md_path: Path, md: str, source_pdf_path: Path | str | None = None) -> tuple[str, bool]:
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    captions_by_page = _load_pdf_text_captions_by_page(pdf_path)
    if not captions_by_page:
        return md, False
    lines = str(md or "").splitlines()
    out: list[str] = []
    changed = False
    for idx, line in enumerate(lines):
        out.append(line)
        m = IMAGE_LINE_RE.match(line or "")
        if not m or _has_caption_nearby(lines, idx):
            continue
        asset_name = _asset_name_from_image_target(m.group(3) or "")
        caption_line = _choose_pdf_caption_for_image(
            asset_name=asset_name,
            alt_text=m.group(2) or "",
            captions_by_page=captions_by_page,
        )
        if not caption_line:
            continue
        out.append("")
        out.append(caption_line)
        changed = True
    return "\n".join(out), changed


def _markdown_image_target_path(md_path: Path, target: str) -> Path | None:
    raw = str(target or "").strip().strip("<>")
    if not raw or raw.startswith(("http://", "https://", "data:", "#")):
        return None
    raw = raw.split("#", 1)[0].split("?", 1)[0]
    if not raw:
        return None
    decoded = unquote(raw.replace("\\", "/"))
    p = Path(decoded)
    return p if p.is_absolute() else (md_path.parent / p)


def _target_page_hint(target: str) -> int:
    raw = str(target or "")
    patterns = [
        r"(?:^|[_\-/])p0*(\d{1,5})(?:[_\-.]|$)",
        r"(?:^|[_\-/])page[_-]?0*(\d{1,5})(?:[_\-.]|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, raw, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


def _target_region_hint(target: str) -> int:
    raw = str(target or "")
    for pattern in [r"(?:^|[_\-/])r0*(\d{1,4})(?:[_\-.]|$)", r"(?:^|[_\-/])fig0*(\d{1,4})(?:[_\-.]|$)"]:
        m = re.search(pattern, raw, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


def _image_assets_by_page(assets_dir: Path) -> dict[int, list[Path]]:
    out: dict[int, list[Path]] = {}
    try:
        children = sorted(assets_dir.iterdir())
    except Exception:
        return out
    for child in children:
        if not child.is_file() or child.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        page_no, _fig_no = _asset_page_and_fig(child.name)
        if page_no <= 0:
            continue
        out.setdefault(page_no, []).append(child)
    return out


def _next_caption_fig_no(lines: list[str], image_idx: int) -> int:
    for idx in range(image_idx + 1, min(len(lines), image_idx + 5)):
        st = (lines[idx] or "").strip()
        if not st:
            continue
        if CAPTION_LINE_RE.match(st):
            return _caption_fig_no_from_text(st)
        return 0
    return 0


def _format_asset_target_like(original_target: str, asset: Path, assets_dir: Path) -> str:
    raw = str(original_target or "").strip()
    normalized = raw.replace("\\", "/")
    name = asset.name
    if normalized.startswith("./assets/"):
        return f"./assets/{name}"
    if normalized.startswith("assets/"):
        return f"assets/{name}"
    try:
        rel = asset.relative_to(assets_dir.parent).as_posix()
        return f"./{rel}" if normalized.startswith("./") else rel
    except Exception:
        return f"./assets/{name}"


def _choose_replacement_image_asset(
    *,
    candidates: list[Path],
    target: str,
    alt_text: str,
    nearby_caption_fig_no: int,
) -> Path | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    wanted_fig = _caption_fig_no_from_text(alt_text) or int(nearby_caption_fig_no or 0)
    if wanted_fig > 0:
        matches = [p for p in candidates if _asset_page_and_fig(p.name)[1] == wanted_fig]
        if len(matches) == 1:
            return matches[0]

    region = _target_region_hint(target)
    if region > 0:
        matches = [p for p in candidates if _asset_page_and_fig(p.name)[1] == region]
        if len(matches) == 1:
            return matches[0]
        if len(candidates) == 1:
            return candidates[0]
    return None


def _repair_missing_image_links(md_path: Path, md: str) -> tuple[str, bool]:
    assets_dir = md_path.parent / "assets"
    if not assets_dir.exists():
        return md, False
    by_page = _image_assets_by_page(assets_dir)
    if not by_page:
        return md, False

    lines = str(md or "").splitlines()
    out: list[str] = []
    current_page = 0
    changed = False
    for idx, line in enumerate(lines):
        page_match = PAGE_MARKER_RE.search(line or "")
        if page_match:
            try:
                current_page = int(page_match.group(1))
            except Exception:
                current_page = 0

        m = IMAGE_LINE_RE.match(line or "")
        if not m:
            out.append(line)
            continue

        target = m.group(3) or ""
        target_path = _markdown_image_target_path(md_path, target)
        if target_path is None or target_path.exists():
            out.append(line)
            continue

        page_no = _target_page_hint(target) or current_page
        replacement = _choose_replacement_image_asset(
            candidates=list(by_page.get(page_no) or []),
            target=target,
            alt_text=m.group(2) or "",
            nearby_caption_fig_no=_next_caption_fig_no(lines, idx),
        )
        if replacement is None:
            out.append(line)
            continue

        new_target = _format_asset_target_like(target, replacement, assets_dir)
        out.append(f"{m.group(1) or ''}![{m.group(2) or ''}]({new_target})")
        changed = True

    fixed = "\n".join(out)
    return fixed, changed and fixed != str(md or "")


def _looks_like_abstract_candidate_line(text: str) -> bool:
    st = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(st) < 140:
        return False
    if st.startswith(("![", "|", "<!--")) or re.match(r"^#{1,6}\s+", st):
        return False
    low = st.lower()
    if any(
        needle in low
        for needle in [
            "contents lists available",
            "article info",
            "keywords:",
            "ocis codes",
            "doi:",
            "copyright",
            "creative commons",
            "published by",
            "received:",
            "accepted:",
        ]
    ):
        return False
    words = re.findall(r"[A-Za-z]{3,}", st)
    if len(words) < 22:
        return False
    signal_hits = len(
        re.findall(
            r"\b(?:we|this|our|paper|study|work|approach|method|results?|demonstrate|"
            r"propose|show|capable|promise|however)\b",
            low,
        )
    )
    return signal_hits >= 1 or len(words) >= 35


def _insert_abstract_heading_only(md: str) -> tuple[str, bool]:
    text = str(md or "")
    if re.search(r"(?mi)^#{1,6}\s+Abstract\s*$", text):
        return text, False
    lines = text.splitlines()
    out: list[str] = []
    changed = False
    spaced_re = re.compile(r"^(?:#*\s*)?A\s*B\s*S\s*T\s*R\s*A\s*C\s*T\s*$", re.IGNORECASE)
    inline_re = re.compile(r"^\s*(?:\*{1,2})?\s*Abstract\s*(?:\*{1,2})?\s*:?\s*(.*)$", re.IGNORECASE)

    for line in lines:
        st = (line or "").strip()
        if spaced_re.match(st):
            out.append("## Abstract")
            changed = True
            continue
        m_inline = inline_re.match(st)
        if m_inline and not re.match(r"^#{1,6}\s+", st):
            tail = (m_inline.group(1) or "").strip()
            out.append("## Abstract")
            if tail:
                out.extend(["", tail])
            changed = True
            continue
        out.append(line)
    if changed:
        return "\n".join(out), True

    first_section_idx = len(lines)
    section_re = re.compile(
        r"^#{1,6}\s+(?:"
        r"\d+(?:\.\d+)*\.?\s+\S+|[IVXLC]+\.\s+\S+|"
        r"(?:introduction|background|related\s+work|results?|method(?:s|ology)?|discussion|"
        r"conclusions?|references|bibliography|structure\b|article\b))",
        re.IGNORECASE,
    )
    for idx, line in enumerate(lines):
        if section_re.match((line or "").strip()):
            first_section_idx = idx
            break

    candidate_idx = -1
    for idx in range(0, first_section_idx):
        if _looks_like_abstract_candidate_line((lines[idx] or "").strip()):
            candidate_idx = idx
            break
    if candidate_idx < 0:
        return text, False

    insert = ["## Abstract", ""]
    if candidate_idx > 0 and (lines[candidate_idx - 1] or "").strip():
        insert.insert(0, "")
    fixed_lines = list(lines)
    fixed_lines[candidate_idx:candidate_idx] = insert
    return "\n".join(fixed_lines), True


def _pdf_source_stats(source_pdf_path: Path | str | None) -> dict[str, Any]:
    if fitz is None or not source_pdf_path:
        return {"available": False, "page_count": 0, "text_chars": 0}
    path = Path(source_pdf_path).expanduser()
    try:
        if not path.exists() or not path.is_file():
            return {"available": False, "page_count": 0, "text_chars": 0, "path": str(path)}
    except Exception:
        return {"available": False, "page_count": 0, "text_chars": 0, "path": str(path)}
    text_chars = 0
    page_count = 0
    try:
        doc = fitz.open(str(path))
    except Exception:
        return {"available": False, "page_count": 0, "text_chars": 0, "path": str(path)}
    try:
        page_count = len(doc)
        for page_index in range(page_count):
            try:
                text_chars += len(str(doc.load_page(page_index).get_text("text") or "").strip())
            except Exception:
                continue
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return {"available": True, "page_count": int(page_count), "text_chars": int(text_chars), "path": str(path)}


def _document_profile(md_path: Path, text: str) -> dict[str, Any]:
    name = f"{md_path.parent.name} {md_path.name}".lower()
    file_name = md_path.name.lower()
    front = re.sub(r"\s+", " ", str(text or "")[:5000]).lower()
    file_title_hint = re.sub(r"^.*?\b(?:19|20)\d{2}\b[\s._-]*", "", file_name).strip()
    is_supplement = bool(
        re.search(r"\b(?:supplement|supplementary|supplemental)\b", name)
        or re.search(r"\b(?:supplement|supplementary|supplemental)\s+(?:document|material|information|doi)\b", front)
        or "parent article doi" in front
    )
    is_review = bool(
        re.search(r"\b(?:review|survey)\b", file_title_hint)
        or re.search(r"\b(?:review article|survey article|systematic review|brief review|comprehensive review)\b", front)
    )
    doc_type = "supplementary" if is_supplement else ("review" if is_review else "research_article")
    return {
        "document_type": doc_type,
        "abstract_required": not is_supplement,
        "abstract_not_applicable": bool(is_supplement),
    }


def _reference_layout(text: str) -> dict[str, Any]:
    lines = str(text or "").splitlines()
    ref_idx = -1
    for idx, line in enumerate(lines):
        if REFERENCES_HEADING_RE.match((line or "").strip()):
            ref_idx = idx
            break
    if ref_idx < 0:
        return {
            "references_line": -1,
            "references_char": -1,
            "references_char_ratio": 1.0,
            "body_heading_after_references_line": -1,
            "reference_line_count_before_body": 0,
            "references_before_body": False,
        }
    char_pos = len("\n".join(lines[:ref_idx]))
    char_ratio = float(char_pos / max(1, len(str(text or ""))))
    ref_count = 0
    body_after = -1
    for idx in range(ref_idx + 1, len(lines)):
        st = (lines[idx] or "").strip()
        if re.match(r"^\[\d{1,4}\]\s+", st):
            ref_count += 1
            continue
        if BODY_SECTION_HEADING_RE.match(st):
            body_after = idx
            break
    before_body = bool(ref_idx >= 0 and body_after > ref_idx and ref_count >= 3 and char_ratio < 0.2)
    return {
        "references_line": int(ref_idx + 1),
        "references_char": int(char_pos),
        "references_char_ratio": round(char_ratio, 4),
        "body_heading_after_references_line": int(body_after + 1) if body_after >= 0 else -1,
        "reference_line_count_before_body": int(ref_count),
        "references_before_body": before_body,
    }


def _source_text_loss_likely(text: str, metrics: dict[str, Any], pdf_stats: dict[str, Any], ref_layout: dict[str, Any]) -> bool:
    if not bool(pdf_stats.get("available")):
        return False
    pdf_pages = int(pdf_stats.get("page_count") or 0)
    pdf_text_chars = int(pdf_stats.get("text_chars") or 0)
    md_chars = int(metrics.get("chars") or len(str(text or "")))
    ref_lines = int(metrics.get("reference_line_count") or 0)
    non_ref_body_chars = md_chars
    ref_pos = int(ref_layout.get("references_char") or -1)
    if ref_pos >= 0:
        non_ref_body_chars = max(0, ref_pos)
    has_body_after_refs = int(ref_layout.get("body_heading_after_references_line") or -1) > 0
    if pdf_pages >= 4 and pdf_text_chars >= 5000 and md_chars < int(pdf_text_chars * 0.22):
        return True
    if pdf_pages >= 4 and ref_lines >= 5 and non_ref_body_chars < max(900, pdf_pages * 180) and not has_body_after_refs:
        return True
    if pdf_pages >= 6 and int(metrics.get("heading_count") or 0) <= 1 and ref_lines >= 8:
        return True
    return False


def _page_alignment_quality(metrics: dict[str, Any], pdf_stats: dict[str, Any]) -> dict[str, Any]:
    pdf_pages = int(pdf_stats.get("page_count") or 0)
    markers = int(metrics.get("page_marker_count") or 0)
    ratio = float(markers / max(1, pdf_pages)) if pdf_pages > 0 else 0.0
    if pdf_pages <= 0:
        confidence = "unknown"
    elif markers <= 0:
        confidence = "missing"
    elif ratio >= 0.8:
        confidence = "high"
    elif ratio >= 0.5:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "pdf_pages": int(pdf_pages),
        "page_marker_count": int(markers),
        "matched_page_ratio": round(ratio, 4) if pdf_pages > 0 else 0.0,
        "page_alignment_confidence": confidence,
    }


def _rare_source_tokens(text: str) -> set[str]:
    return {
        tok
        for tok, _ in _word_tokens_with_offsets(text)
        if len(tok) >= 5 and tok not in PAGE_ALIGNMENT_STOP_WORDS and not tok.isdigit()
    }


def _page_marker_occurrences(text: str) -> list[dict[str, int]]:
    matches = list(PAGE_MARKER_RE.finditer(str(text or "")))
    out: list[dict[str, int]] = []
    for idx, match in enumerate(matches):
        try:
            page_no = int(match.group(1))
        except Exception:
            continue
        segment_end = int(matches[idx + 1].start()) if idx + 1 < len(matches) else len(str(text or ""))
        out.append(
            {
                "page": int(page_no),
                "start": int(match.start()),
                "end": int(match.end()),
                "segment_start": int(match.end()),
                "segment_end": int(segment_end),
            }
        )
    return out


def _source_page_coverage_quality(text: str, pdf_path: Path | None) -> dict[str, Any]:
    if fitz is None or pdf_path is None:
        return {
            "min_source_page_coverage": 0.0,
            "missing_source_page_count": 0,
            "missing_source_pages": [],
            "source_page_coverage_threshold": SOURCE_PAGE_COVERAGE_THRESHOLD,
        }
    try:
        path = Path(pdf_path).expanduser()
        if not path.exists() or not path.is_file():
            return {
                "min_source_page_coverage": 0.0,
                "missing_source_page_count": 0,
                "missing_source_pages": [],
                "source_page_coverage_threshold": SOURCE_PAGE_COVERAGE_THRESHOLD,
            }
    except Exception:
        return {
            "min_source_page_coverage": 0.0,
            "missing_source_page_count": 0,
            "missing_source_pages": [],
            "source_page_coverage_threshold": SOURCE_PAGE_COVERAGE_THRESHOLD,
        }

    md_tokens = _rare_source_tokens(text)
    if len(md_tokens) < SOURCE_PAGE_MIN_RARE_TOKENS:
        return {
            "min_source_page_coverage": 0.0,
            "missing_source_page_count": 0,
            "missing_source_pages": [],
            "source_page_coverage_threshold": SOURCE_PAGE_COVERAGE_THRESHOLD,
        }
    marker_pages = {
        int(match.group(1))
        for match in PAGE_MARKER_RE.finditer(str(text or ""))
        if str(match.group(1) or "").isdigit()
    }
    marker_segments: dict[int, tuple[int, int]] = {}
    for item in _page_marker_occurrences(text):
        page_no = int(item.get("page") or 0)
        if page_no > 0 and page_no not in marker_segments:
            marker_segments[page_no] = (
                int(item.get("segment_start") or 0),
                int(item.get("segment_end") or 0),
            )
    inferred_offsets: dict[int, int] = {}
    try:
        inferred_offsets = _page_marker_offsets_from_pdf_text(str(text or ""), path, snap_to_line_start=False)
    except Exception:
        inferred_offsets = {}

    def local_segment_for_page(page_no: int) -> str:
        if page_no in marker_segments:
            start, end = marker_segments[page_no]
            return str(text or "")[max(0, start) : max(0, end)]
        known = sorted((int(page), int(offset)) for page, offset in inferred_offsets.items() if int(page) > 0)
        previous = [(page, offset) for page, offset in known if page < page_no]
        following = [(page, offset) for page, offset in known if page > page_no]
        if not previous or not following:
            return ""
        start = previous[-1][1]
        end = following[0][1]
        if end <= start:
            return ""
        return str(text or "")[max(0, start) : max(0, end)]

    low_pages: list[dict[str, Any]] = []
    min_coverage = 1.0
    assessed = 0
    try:
        doc = fitz.open(str(path))
    except Exception:
        return {
            "min_source_page_coverage": 0.0,
            "missing_source_page_count": 0,
            "missing_source_pages": [],
            "source_page_coverage_threshold": SOURCE_PAGE_COVERAGE_THRESHOLD,
        }
    try:
        for page_index in range(len(doc)):
            page_no = page_index + 1
            try:
                page_text = str(doc.load_page(page_index).get_text("text") or "")
            except Exception:
                page_text = ""
            if marker_pages and page_no < min(marker_pages) and _pdf_page_looks_like_download_landing_page(page_text):
                continue
            page_tokens = _rare_source_tokens(page_text)
            if len(page_tokens) < SOURCE_PAGE_MIN_RARE_TOKENS:
                continue
            assessed += 1
            coverage = len(page_tokens.intersection(md_tokens)) / max(1, len(page_tokens))
            local_segment = local_segment_for_page(page_no)
            local_coverage: float | None = None
            if local_segment:
                local_tokens = _rare_source_tokens(local_segment)
                local_coverage = len(page_tokens.intersection(local_tokens)) / max(1, len(page_tokens))
            min_coverage = min(min_coverage, coverage)
            local_low_without_inferred_anchor = (
                local_coverage is not None
                and local_coverage < SOURCE_PAGE_SEGMENT_COVERAGE_THRESHOLD
                and page_no not in set(inferred_offsets)
            )
            if page_no in marker_pages and not local_low_without_inferred_anchor:
                continue
            if coverage >= SOURCE_PAGE_COVERAGE_THRESHOLD and not local_low_without_inferred_anchor:
                continue
            low_pages.append(
                {
                    "page": int(page_no),
                    "coverage": round(float(coverage), 4),
                    "local_coverage": round(float(local_coverage), 4) if local_coverage is not None else None,
                    "source_token_count": int(len(page_tokens)),
                    "has_page_marker": bool(page_no in marker_pages),
                    "reason": "low_local_page_overlap" if local_low_without_inferred_anchor else "low_text_overlap",
                }
            )
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return {
        "min_source_page_coverage": round(float(min_coverage if assessed > 0 else 0.0), 4),
        "missing_source_page_count": int(len(low_pages)),
        "missing_source_pages": low_pages[:20],
        "source_page_coverage_threshold": SOURCE_PAGE_COVERAGE_THRESHOLD,
    }


def _source_page_anchor_alignment_quality(text: str, pdf_path: Path | None) -> dict[str, Any]:
    empty = {
        "source_page_anchor_issue_count": 0,
        "missing_source_page_marker_count": 0,
        "misaligned_source_page_marker_count": 0,
        "source_page_anchor_issues": [],
        "source_page_anchor_alignment_threshold": PAGE_ALIGNMENT_ANCHOR_DRIFT_CHARS,
    }
    if fitz is None or pdf_path is None:
        return empty
    path = Path(pdf_path).expanduser()
    try:
        if not path.exists() or not path.is_file():
            return empty
    except Exception:
        return empty
    occurrences = _page_marker_occurrences(text)
    if not occurrences:
        return empty

    marker_offsets: dict[int, int] = {}
    marker_segments: dict[int, tuple[int, int]] = {}
    for item in occurrences:
        page_no = int(item.get("page") or 0)
        if page_no <= 0 or page_no in marker_offsets:
            continue
        marker_offsets[page_no] = int(item.get("start") or 0)
        marker_segments[page_no] = (
            int(item.get("segment_start") or 0),
            int(item.get("segment_end") or 0),
        )

    inferred = _page_marker_offsets_from_pdf_text(text, path, snap_to_line_start=False)
    inferred_pages = {int(page) for page in inferred if int(page) > 0}
    marker_pages = set(marker_offsets)

    issues: list[dict[str, Any]] = []
    for page_no in sorted(page for page in inferred_pages - marker_pages if page > 1):
        issues.append(
            {
                "page": int(page_no),
                "reason": "source_page_text_present_without_anchor",
                "inferred_offset": int(inferred.get(page_no) or 0),
            }
        )

    try:
        doc = fitz.open(str(path))
    except Exception:
        doc = None
    if doc is not None:
        try:
            for page_no, (segment_start, segment_end) in sorted(marker_segments.items()):
                if page_no <= 0 or page_no > len(doc):
                    continue
                try:
                    page_text = str(doc.load_page(page_no - 1).get_text("text") or "")
                except Exception:
                    page_text = ""
                page_tokens = _rare_source_tokens(page_text)
                if len(page_tokens) < SOURCE_PAGE_MIN_RARE_TOKENS:
                    continue
                segment = str(text or "")[max(0, segment_start) : max(0, segment_end)]
                segment_tokens = _rare_source_tokens(segment)
                coverage = len(page_tokens.intersection(segment_tokens)) / max(1, len(page_tokens))
                inferred_offset = inferred.get(page_no)
                marker_offset = marker_offsets.get(page_no)
                inferred_far = (
                    inferred_offset is None
                    or marker_offset is None
                    or abs(int(marker_offset) - int(inferred_offset)) > PAGE_ALIGNMENT_ANCHOR_DRIFT_CHARS
                )
                if coverage < SOURCE_PAGE_SEGMENT_COVERAGE_THRESHOLD and inferred_far:
                    issues.append(
                        {
                            "page": int(page_no),
                            "reason": "page_anchor_segment_low_source_overlap",
                            "segment_coverage": round(float(coverage), 4),
                        }
                    )
        finally:
            try:
                doc.close()
            except Exception:
                pass

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for item in issues:
        page_no = int(item.get("page") or 0)
        reason = str(item.get("reason") or "")
        key = (page_no, reason)
        if page_no <= 0 or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    missing_count = sum(1 for item in deduped if str(item.get("reason") or "") == "source_page_text_present_without_anchor")
    return {
        "source_page_anchor_issue_count": int(len(deduped)),
        "missing_source_page_marker_count": int(missing_count),
        "misaligned_source_page_marker_count": int(len(deduped) - missing_count),
        "source_page_anchor_issues": deduped[:20],
        "source_page_anchor_alignment_threshold": PAGE_ALIGNMENT_ANCHOR_DRIFT_CHARS,
    }


def _reference_index_truncated(text: str, metrics: dict[str, Any]) -> bool:
    reference_lines = int(metrics.get("reference_line_count") or 0)
    max_index = int(metrics.get("max_reference_index") or 0)
    extracted = int(metrics.get("extracted_reference_count") or 0)
    if reference_lines < 8 or max_index < 8:
        return False
    if extracted <= 0:
        return True
    ref_map = extract_references_map_from_md(text)
    if _reference_gap_is_material(ref_map):
        return True
    if _reference_map_has_short_truncated_entries(ref_map):
        return True
    expected = max(reference_lines, max_index)
    return extracted < max(5, int(expected * 0.55))


def _source_quality_view(
    md_path: Path,
    text: str,
    metrics: dict[str, Any],
    *,
    source_pdf_path: Path | str | None = None,
    allow_source_pdf_inference: bool = True,
) -> dict[str, Any]:
    pdf_path = (
        Path(source_pdf_path).expanduser()
        if source_pdf_path
        else (_guess_source_pdf_for_md(md_path) if allow_source_pdf_inference else None)
    )
    pdf_stats = _pdf_source_stats(pdf_path)
    profile = _document_profile(md_path, text)
    ref_layout = _reference_layout(text)
    abstract_candidate_text, abstract_changed = _insert_abstract_heading_only(text)
    _ = abstract_candidate_text
    abstract_autofix_likely = bool(abstract_changed)
    source_text_loss = False if bool(profile.get("abstract_not_applicable")) else _source_text_loss_likely(text, metrics, pdf_stats, ref_layout)
    page_quality = _page_alignment_quality(metrics, pdf_stats)
    page_coverage = _source_page_coverage_quality(text, pdf_path if pdf_path and bool(pdf_stats.get("available")) else None)
    page_anchor_quality = _source_page_anchor_alignment_quality(text, pdf_path if pdf_path and bool(pdf_stats.get("available")) else None)
    return {
        **profile,
        "source_pdf_path": str((pdf_stats or {}).get("path") or (pdf_path or "")),
        "source_pdf_available": bool(pdf_stats.get("available")),
        "pdf_page_count": int(pdf_stats.get("page_count") or 0),
        "pdf_text_chars": int(pdf_stats.get("text_chars") or 0),
        **page_quality,
        **page_coverage,
        **page_anchor_quality,
        **ref_layout,
        "abstract_autofix_likely": abstract_autofix_likely,
        "source_text_loss": bool(source_text_loss),
        "reference_index_truncated": _reference_index_truncated(text, metrics),
    }


def _move_early_references_to_end(md: str) -> tuple[str, bool]:
    lines = str(md or "").splitlines()
    ref_idx = -1
    for idx, line in enumerate(lines):
        if REFERENCES_HEADING_RE.match((line or "").strip()):
            ref_idx = idx
            break
    if ref_idx < 0:
        return str(md or ""), False
    ref_count = 0
    body_idx = -1
    for idx in range(ref_idx + 1, len(lines)):
        st = (lines[idx] or "").strip()
        if re.match(r"^\[\d{1,4}\]\s+", st):
            ref_count += 1
            continue
        if BODY_SECTION_HEADING_RE.match(st):
            body_idx = idx
            break
    if body_idx <= ref_idx or ref_count < 3:
        return str(md or ""), False
    char_pos = len("\n".join(lines[:ref_idx]))
    if char_pos > int(max(1, len(str(md or ""))) * 0.2):
        return str(md or ""), False
    head = lines[:ref_idx]
    ref_block = lines[ref_idx:body_idx]
    body = lines[body_idx:]

    def _trim_blank_edges(items: list[str]) -> list[str]:
        out = list(items)
        while out and not (out[0] or "").strip():
            out.pop(0)
        while out and not (out[-1] or "").strip():
            out.pop()
        return out

    fixed_parts = _trim_blank_edges(head) + [""] + _trim_blank_edges(body) + [""] + _trim_blank_edges(ref_block)
    fixed = "\n".join(fixed_parts).strip() + "\n"
    return fixed, fixed != str(md or "")


def _normalize_markdown_tables_only(md: str) -> tuple[str, bool]:
    fixed = normalize_markdown_tables_document(md)
    return fixed, fixed != str(md or "")


def _normalize_heading_level_jumps(md: str) -> tuple[str, bool]:
    lines = str(md or "").splitlines()
    out: list[str] = []
    previous_level = 0
    in_fence = False
    in_math = False
    changed = False

    for line in lines:
        stripped = str(line or "").strip()
        if re.match(r"^\s*```", line):
            in_fence = not in_fence
            out.append(line)
            continue
        if stripped == "$$":
            in_math = not in_math
            out.append(line)
            continue
        if in_fence or in_math:
            out.append(line)
            continue

        match = re.match(r"^(#{1,6})(\s+.+)$", line)
        if not match:
            out.append(line)
            continue

        level = len(match.group(1))
        target_level = level
        if previous_level > 0 and level > previous_level + 1:
            target_level = min(6, previous_level + 1)
        if target_level != level:
            out.append("#" * target_level + match.group(2))
            changed = True
        else:
            out.append(line)
        previous_level = target_level

    fixed = "\n".join(out)
    if str(md or "").endswith("\n"):
        fixed += "\n"
    return fixed, changed and fixed != str(md or "")


_REVIEW_PROMOTABLE_HEADING_RE = re.compile(
    r"\b(?:"
    r"problem|statement|classical|spatial|domain|filter(?:ing)?|variational|regularization|"
    r"non[-\s]?local|sparse|representation|low[-\s]?rank|minimization|transform|technique|"
    r"adaptive|bm3d|cnn|mlp|deep\s+learning|experiments?|metrics?|comparison|methods?|models?|"
    r"performance|conclusions?"
    r")\b",
    re.IGNORECASE,
)


def _review_heading_promotable(title: str) -> bool:
    text = re.sub(r"\s+", " ", str(title or "")).strip()
    if len(text) < 3 or len(text) > 110:
        return False
    if REFERENCES_HEADING_RE.match(f"## {text}") or re.match(r"^(?:abstract|keywords?)$", text, re.IGNORECASE):
        return False
    if CAPTION_LINE_RE.match(text):
        return False
    if re.search(r"[=\\^_{}]|^\(?\d{1,3}\)?\s*$", text):
        return False
    if re.search(r"[.!?]\s+\S", text):
        return False
    return bool(_REVIEW_PROMOTABLE_HEADING_RE.search(text))


def _promote_collapsed_review_headings(md: str) -> tuple[str, bool]:
    text = str(md or "")
    lines = text.splitlines()
    h2_count = sum(1 for line in lines if re.match(r"^##\s+\S", line))
    h3_count = sum(1 for line in lines if re.match(r"^###\s+\S", line))
    if h3_count < 8 or h2_count >= 8 or h3_count < max(8, h2_count * 2):
        return text, False

    out: list[str] = []
    in_fence = False
    in_math = False
    changed = False
    for line in lines:
        stripped = str(line or "").strip()
        if re.match(r"^\s*```", line):
            in_fence = not in_fence
            out.append(line)
            continue
        if stripped == "$$":
            in_math = not in_math
            out.append(line)
            continue
        if in_fence or in_math:
            out.append(line)
            continue

        match = re.match(r"^(###)(\s+)(.+?)\s*$", line)
        if match and _review_heading_promotable(match.group(3)):
            out.append(f"##{match.group(2)}{match.group(3).strip()}")
            changed = True
            continue
        out.append(line)

    fixed = "\n".join(out)
    if text.endswith("\n"):
        fixed += "\n"
    return fixed, changed and fixed != text


def _regression_reasons(base_text: str, candidate_text: str) -> list[str]:
    comparison = compare_markdown_quality(base_text, candidate_text)
    base = comparison.get("base") if isinstance(comparison.get("base"), dict) else {}
    cand = comparison.get("candidate") if isinstance(comparison.get("candidate"), dict) else {}
    flags = comparison.get("regression_flags") if isinstance(comparison.get("regression_flags"), dict) else {}
    reasons = [str(key) for key, value in flags.items() if bool(value)]
    if "tables_dropped" in reasons:
        base_table_issues = markdown_table_issue_counts(base_text)
        candidate_table_issues = markdown_table_issue_counts(candidate_text)
        if (
            int(base_table_issues.get("duplicate_table_count") or 0)
            > int(candidate_table_issues.get("duplicate_table_count") or 0)
            and int(candidate_table_issues.get("literal_break_count") or 0) == 0
        ):
            reasons = [reason for reason in reasons if reason != "tables_dropped"]
    base_chars = int(base.get("chars") or 0)
    cand_chars = int(cand.get("chars") or 0)
    if reasons == ["analyzer_warnings_increased"]:
        base_no_markers = re.sub(r"\s*<!--\s*kb_page:\s*\d+\s*-->\s*", " ", str(base_text or ""))
        cand_no_markers = re.sub(r"\s*<!--\s*kb_page:\s*\d+\s*-->\s*", " ", str(candidate_text or ""))
        marker_only_change = (
            re.sub(r"\s+", " ", base_no_markers).strip()
            == re.sub(r"\s+", " ", cand_no_markers).strip()
        )
        structural_gain = bool(
            (not bool(base.get("has_abstract_heading"))) and bool(cand.get("has_abstract_heading"))
        ) or int(cand.get("caption_count") or 0) > int(base.get("caption_count") or 0)
        no_content_loss = (
            int(cand.get("chars") or 0) >= int(int(base.get("chars") or 0) * 0.9)
            and int(cand.get("image_count") or 0) >= int(base.get("image_count") or 0)
            and int(cand.get("table_block_count") or 0) >= int(base.get("table_block_count") or 0)
            and int(cand.get("reference_line_count") or 0) >= int(base.get("reference_line_count") or 0)
        )
        base_warning_count = int(base.get("analyzer_warning_count") or 0)
        cand_warning_count = int(cand.get("analyzer_warning_count") or 0)
        low_warning_delta = cand_warning_count <= max(3, base_warning_count + 2)
        content_backfill_gain = base_chars > 0 and cand_chars >= base_chars + max(1000, int(base_chars * 0.08))
        if marker_only_change or (structural_gain and no_content_loss) or (content_backfill_gain and no_content_loss and low_warning_delta):
            reasons = []
    if base_chars > 1000 and cand_chars < int(base_chars * 0.82):
        reasons.append("content_shrank_too_much")
    return reasons


def _source_pdf_name_candidates(md_path: Path) -> list[str]:
    names: list[str] = []
    parent_name = str(md_path.parent.name or "").strip()
    if parent_name:
        names.append(parent_name)
    stem = str(md_path.stem or "").strip()
    if stem.endswith(".en"):
        stem = stem[:-3]
    if stem and stem not in names:
        names.append(stem)
    out: list[str] = []
    for name in names:
        clean = name[:-4] if name.lower().endswith(".pdf") else name
        if clean and f"{clean}.pdf" not in out:
            out.append(f"{clean}.pdf")
    return out


def _candidate_pdf_roots(md_path: Path) -> list[Path]:
    roots: list[Path] = []

    def add(raw: str | Path | None) -> None:
        if raw is None:
            return
        try:
            p = Path(raw).expanduser()
        except Exception:
            return
        if p and p not in roots:
            roots.append(p)

    add(md_path.parent)
    add(md_path.parent.parent)
    add(os.environ.get("KB_PDF_DIR"))

    pref_paths = [
        Path.cwd() / "user_prefs.json",
        Path(__file__).resolve().parents[2] / "user_prefs.json",
        md_path.parent.parent / "user_prefs.json",
    ]
    for pref_path in pref_paths:
        try:
            payload = json.loads(pref_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            add(payload.get("pdf_dir"))
    return roots


def _guess_source_pdf_for_md(md_path: Path) -> Path | None:
    names = _source_pdf_name_candidates(md_path)
    if not names:
        return None
    for root in _candidate_pdf_roots(md_path):
        for name in names:
            candidate = root / name
            try:
                if candidate.exists() and candidate.is_file():
                    return candidate
            except Exception:
                continue
    lowered = {name.lower() for name in names}
    for root in _candidate_pdf_roots(md_path):
        try:
            for candidate in root.glob("*.pdf"):
                if candidate.name.lower() in lowered:
                    return candidate
        except Exception:
            continue
    return None


def _word_tokens_with_offsets(text: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for m in re.finditer(r"[^\W_]+", str(text or ""), flags=re.UNICODE):
        raw = unicodedata.normalize("NFKC", m.group(0) or "").lower()
        raw = raw.replace("\u00ad", "")
        for sub in re.finditer(r"[a-z][a-z0-9]{2,}|\d{2,}", raw):
            out.append((sub.group(0), int(m.start())))
    return out


def _page_alignment_candidates(tokens: list[tuple[str, int]], width: int = PAGE_ALIGNMENT_DEFAULT_NGRAM) -> list[tuple[tuple[str, ...], int, int]]:
    if len(tokens) < width:
        return []
    total = len(tokens)
    starts: list[int] = []
    starts.extend(range(0, min(total, 220), 8 if width <= 6 else 12))
    starts.extend(range(max(0, total // 3 - 80), min(total, total // 3 + 100), 16 if width <= 6 else 20))
    starts.extend(range(max(0, (2 * total) // 3 - 80), min(total, (2 * total) // 3 + 100), 16 if width <= 6 else 20))
    out: list[tuple[tuple[str, ...], int, int]] = []
    seen: set[tuple[str, ...]] = set()
    rare_min = 2 if width <= 6 else 3
    for start in starts:
        if start + width > total:
            continue
        gram = tuple(tok for tok, _ in tokens[start : start + width])
        if gram in seen:
            continue
        seen.add(gram)
        rare = sum(
            1
            for tok in gram
            if len(tok) >= 5 and tok not in PAGE_ALIGNMENT_STOP_WORDS and not tok.isdigit()
        )
        if rare < rare_min:
            continue
        out.append((gram, rare, start))
    return out


def _line_start_for_offset(text: str, offset: int) -> int:
    pos = max(0, min(len(text), int(offset)))
    return str(text or "").rfind("\n", 0, pos) + 1


def _page_alignment_md_grams(md_tokens: list[tuple[str, int]], width: int) -> dict[tuple[str, ...], list[int]]:
    md_grams: dict[tuple[str, ...], list[int]] = {}
    for idx in range(0, len(md_tokens) - width + 1):
        gram = tuple(tok for tok, _ in md_tokens[idx : idx + width])
        bucket = md_grams.setdefault(gram, [])
        if len(bucket) < 30:
            bucket.append(idx)
    return md_grams


def _page_alignment_page_candidates(
    page_tokens: list[tuple[str, int]],
    md_grams: dict[tuple[str, ...], list[int]],
    width: int,
) -> list[tuple[int, int, int]]:
    by_md_index: dict[int, tuple[int, int]] = {}
    for gram, rare, page_token_start in _page_alignment_candidates(page_tokens, width):
        for md_token_index in md_grams.get(gram, []):
            if md_token_index < 10:
                continue
            old = by_md_index.get(md_token_index)
            if old is None or rare > old[0] or (rare == old[0] and page_token_start < old[1]):
                by_md_index[md_token_index] = (rare, page_token_start)
    choices = [(idx, rare, start) for idx, (rare, start) in by_md_index.items()]
    choices.sort(key=lambda item: (int(item[0]), int(item[2]), -int(item[1])))
    return choices[:160]


def _select_page_alignment_offsets(
    md_tokens: list[tuple[str, int]],
    page_tokens_by_index: list[list[tuple[str, int]]],
    width: int,
) -> dict[int, int]:
    if len(md_tokens) < width:
        return {1: 0}
    md_grams = _page_alignment_md_grams(md_tokens, width)
    page_candidates = [
        []
        if page_index == 0
        else _page_alignment_page_candidates(page_tokens, md_grams, width)
        for page_index, page_tokens in enumerate(page_tokens_by_index)
    ]

    # State: matched pages, quality score, last Markdown token index, selected (page_no, token_index) pairs.
    states: list[tuple[int, float, int, tuple[tuple[int, int], ...]]] = [(0, 0.0, -1, tuple())]
    for page_no in range(2, len(page_tokens_by_index) + 1):
        new_states = list(states)
        for matched, quality, last_token_index, offsets in states:
            for md_token_index, rare, page_token_start in page_candidates[page_no - 1]:
                if md_token_index <= last_token_index:
                    continue
                score = quality + float(rare * 4.0) - float(page_token_start) * 0.4
                new_states.append(
                    (
                        matched + 1,
                        score,
                        int(md_token_index),
                        offsets + ((int(page_no), int(md_token_index)),),
                    )
                )
        if len(new_states) > PAGE_ALIGNMENT_BEAM_SIZE:
            new_states.sort(key=lambda item: (int(item[0]), float(item[1]), -int(item[2])), reverse=True)
            kept: list[tuple[int, float, int, tuple[tuple[int, int], ...]]] = []
            buckets: dict[int, int] = {}
            for state in new_states:
                matched = int(state[0])
                count = buckets.get(matched, 0)
                if len(kept) >= PAGE_ALIGNMENT_BEAM_SIZE:
                    break
                if count >= PAGE_ALIGNMENT_BEAM_PER_MATCH:
                    continue
                kept.append(state)
                buckets[matched] = count + 1
            states = kept
        else:
            states = new_states

    best = max(states, key=lambda item: (int(item[0]), float(item[1]), -int(item[2])), default=states[0])
    offsets: dict[int, int] = {1: 0}
    for page_no, md_token_index in best[3]:
        try:
            offsets[int(page_no)] = int(md_tokens[int(md_token_index)][1])
        except Exception:
            continue
    return offsets


def _source_page_token_coverage(page_text: str, segment_text: str) -> float:
    source_tokens = _rare_source_tokens(page_text)
    if not source_tokens:
        return 0.0
    segment_tokens = _rare_source_tokens(segment_text)
    if not segment_tokens:
        return 0.0
    return len(source_tokens.intersection(segment_tokens)) / max(1, len(source_tokens))


def _pdf_page_looks_like_download_landing_page(page_text: str) -> bool:
    text = _clean_pdf_page_block_text(page_text)
    if not text:
        return False
    low = text.lower()
    signals = 0
    for pattern in (
        "latest updates",
        "pdf download",
        "total citations",
        "total downloads",
        "citation in bibtex",
        "open access support",
        "dl.acm.org",
    ):
        if pattern in low:
            signals += 1
    if signals >= 2:
        return True
    return bool(signals >= 1 and "published" in low and ("doi" in low or "acm" in low))


def _adjust_offsets_for_skipped_leading_pdf_pages(
    md_text: str,
    pdf_path: Path,
    offsets: dict[int, int],
) -> dict[int, int]:
    """
    Some publisher PDFs include a download/landing page before the actual paper.
    If conversion intentionally starts at the article page, the alignment code may
    still synthesize page 1 at offset 0 and then split the true first article page.
    Prefer the first strongly matched later source page when it starts near the
    beginning and earlier PDF pages only weakly match the Markdown head.
    """
    if fitz is None or not offsets:
        return offsets
    later = sorted((int(page), int(offset)) for page, offset in offsets.items() if int(page) > 1)
    if not later:
        return offsets
    first_page, first_offset = later[0]
    if first_page <= 1 or first_offset < 0 or first_offset > LEADING_PAGE_ALIGNMENT_MAX_OFFSET:
        return offsets

    head_len = max(LEADING_PAGE_ALIGNMENT_WINDOW_CHARS, first_offset + LEADING_PAGE_ALIGNMENT_WINDOW_CHARS)
    head = str(md_text or "")[: min(len(str(md_text or "")), int(head_len))]
    if len(_rare_source_tokens(head)) < 20:
        return offsets

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return offsets
    try:
        if first_page > len(doc):
            return offsets
        first_text = str(doc.load_page(first_page - 1).get_text("text") or "")
        first_coverage = _source_page_token_coverage(first_text, head)
        previous_coverages: list[float] = []
        for page_index in range(0, max(0, first_page - 1)):
            previous_text = str(doc.load_page(page_index).get_text("text") or "")
            previous_coverages.append(_source_page_token_coverage(previous_text, head))
    except Exception:
        return offsets
    finally:
        try:
            doc.close()
        except Exception:
            pass

    best_previous = max(previous_coverages, default=0.0)
    if (
        first_coverage >= SOURCE_PAGE_SEGMENT_COVERAGE_THRESHOLD
        and best_previous < LEADING_PAGE_DROP_MAX_PREVIOUS_COVERAGE
        and (first_coverage - best_previous) >= LEADING_PAGE_DROP_MIN_COVERAGE_MARGIN
    ):
        adjusted = {int(page): int(offset) for page, offset in offsets.items() if int(page) >= first_page}
        adjusted[first_page] = 0
        return dict(sorted(adjusted.items(), key=lambda item: int(item[0])))
    return offsets


def _page_marker_offsets_from_pdf_text(
    md_text: str,
    pdf_path: Path,
    *,
    snap_to_line_start: bool = True,
) -> dict[int, int]:
    if fitz is None:
        return {}
    path = Path(pdf_path).expanduser()
    try:
        if not path.exists() or not path.is_file():
            return {}
    except Exception:
        return {}

    cache_key = _page_marker_offsets_cache_key(
        md_text,
        path,
        snap_to_line_start=snap_to_line_start,
    )
    cached = _get_cached_page_marker_offsets(cache_key)
    if cached is not None:
        return cached

    md_tokens = _word_tokens_with_offsets(md_text)
    if len(md_tokens) < min(PAGE_ALIGNMENT_NGRAMS):
        return {}

    offsets: dict[int, int] = {1: 0}
    pdf_page_count = 0
    try:
        doc = fitz.open(str(path))
    except Exception:
        return offsets
    try:
        pdf_page_count = len(doc)
        page_tokens_by_index: list[list[tuple[str, int]]] = []
        for page_index in range(pdf_page_count):
            try:
                page_text = doc.load_page(page_index).get_text("text")
            except Exception:
                page_text = ""
            page_tokens_by_index.append(_word_tokens_with_offsets(page_text))

        best_offsets: dict[int, int] = {1: 0}
        for width in PAGE_ALIGNMENT_NGRAMS:
            current_offsets = _select_page_alignment_offsets(md_tokens, page_tokens_by_index, width)

            if len(current_offsets) > len(best_offsets):
                best_offsets = current_offsets
        offsets = _adjust_offsets_for_skipped_leading_pdf_pages(str(md_text or ""), path, best_offsets)
    finally:
        try:
            doc.close()
        except Exception:
            pass

    matched_later_pages = len([p for p in offsets if p > 1])
    if matched_later_pages <= 0:
        offsets = {1: 0}
    elif pdf_page_count >= 6 and matched_later_pages < 2:
        offsets = {1: 0}
    elif snap_to_line_start:
        offsets = {
            int(page_no): _line_start_for_offset(md_text, int(offset))
            for page_no, offset in offsets.items()
        }
    _cache_page_marker_offsets(cache_key, offsets)
    return offsets


def _recover_page_markers_from_pdf_text(md_text: str, md_path: Path, source_pdf_path: Path | str | None = None) -> tuple[str, bool]:
    text = str(md_text or "")
    if PAGE_MARKER_RE.search(text):
        return text, False
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    if not pdf_path:
        return text, False
    offsets = _page_marker_offsets_from_pdf_text(text, pdf_path)
    if not offsets:
        return text, False
    fixed = text
    for page_no, offset in sorted(offsets.items(), key=lambda item: int(item[1]), reverse=True):
        marker = f"<!-- kb_page: {int(page_no)} -->"
        pos = max(0, min(len(fixed), int(offset)))
        needs_after = pos < len(fixed) and fixed[pos : pos + 2] != "\n\n"
        insert = marker + ("\n\n" if needs_after else "\n")
        if pos > 0 and not fixed[:pos].endswith("\n\n"):
            insert = "\n" + insert
        fixed = fixed[:pos] + insert + fixed[pos:]
    return fixed, fixed != text


def _insert_page_marker_at_offset(md_text: str, offset: int, page_no: int) -> str:
    text = str(md_text or "")
    pos = max(0, min(len(text), int(offset)))
    marker = f"<!-- kb_page: {int(page_no)} -->"
    left = text[:pos]
    right = text[pos:]
    insert = marker
    if left and not left.endswith("\n\n"):
        insert = ("\n" if left.endswith("\n") else "\n\n") + insert
    if right and not right.startswith("\n\n"):
        insert = insert + ("\n" if right.startswith("\n") else "\n\n")
    if not left and right and not right.startswith("\n"):
        insert += "\n\n"
    return left + insert + right


def _page_marker_insert_offset(md_text: str, offset: int) -> int:
    text = str(md_text or "")
    pos = max(0, min(len(text), int(offset)))
    line_start = _line_start_for_offset(text, pos)
    line_end = text.find("\n", pos)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end]
    if re.match(r"\s*\[\s*\d{1,4}\s*]\s+", line):
        return int(line_start)
    return int(pos)


def _is_existing_converter_page_asset(md_path: Path, raw_target: str) -> bool:
    target = unquote(str(raw_target or "").strip())
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")].strip()
    else:
        target = re.split(r"\s+[\"']", target, maxsplit=1)[0].strip()
    target = target.split("#", 1)[0].split("?", 1)[0].strip()
    if not target or re.match(r"^(?:[a-z][a-z0-9+.-]*:|//)", target, flags=re.IGNORECASE):
        return False
    relative = Path(target)
    if relative.is_absolute():
        return False
    try:
        candidate = (md_path.parent / relative).resolve()
        assets_root = (md_path.parent / "assets").resolve()
        candidate.relative_to(assets_root)
        return candidate.is_file()
    except Exception:
        return False


def _realign_page_markers_from_pdf_text(md_text: str, md_path: Path, source_pdf_path: Path | str | None = None) -> tuple[str, bool]:
    text = str(md_text or "")
    if not PAGE_MARKER_RE.search(text):
        return text, False
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    if not pdf_path:
        return text, False
    markerless = PAGE_MARKER_RE.sub("", text)
    offsets = _page_marker_offsets_from_pdf_text(markerless, pdf_path, snap_to_line_start=False)
    matched_later = len([page for page in offsets if int(page) > 1])
    if matched_later <= 0:
        return text, False
    pdf_page_count = 0
    if fitz is not None:
        try:
            doc = fitz.open(str(pdf_path))
            try:
                pdf_page_count = len(doc)
            finally:
                doc.close()
        except Exception:
            pdf_page_count = 0
    if pdf_page_count >= 6:
        required = max(2, int((pdf_page_count - 1) * 0.45))
        if matched_later < required:
            return text, False

    # Text alignment cannot place an image-only page. Converter-owned asset names
    # retain the physical PDF page number, so use the first page asset as a safe
    # anchor for otherwise unmatched pages.
    if pdf_page_count > 0:
        asset_pattern = re.compile(
            r"(?im)^.*?!\[[^\]]*]\(([^\n)]*?page[_-](\d{1,5})[_-][^\n)]*)\).*$"
        )
        for match in asset_pattern.finditer(markerless):
            page_no = int(match.group(2))
            if (
                1 <= page_no <= pdf_page_count
                and page_no not in offsets
                and _is_existing_converter_page_asset(md_path, match.group(1))
            ):
                offsets[page_no] = _line_start_for_offset(markerless, int(match.start()))

    cleaned_offsets: dict[int, int] = {}
    last_offset = -1
    for page_no, offset in sorted(offsets.items(), key=lambda item: (int(item[0]), int(item[1]))):
        page = int(page_no)
        pos = _page_marker_insert_offset(markerless, int(offset))
        if page <= 0 or pos < last_offset:
            continue
        if page > 1 and pos == last_offset:
            continue
        cleaned_offsets[page] = pos
        last_offset = pos
    if len(cleaned_offsets) <= 1:
        return text, False

    fixed = markerless
    for page_no, offset in sorted(cleaned_offsets.items(), key=lambda item: int(item[1]), reverse=True):
        fixed = _insert_page_marker_at_offset(fixed, int(offset), int(page_no))
    return fixed, fixed != text


def _clean_pdf_page_block_text(raw: str) -> str:
    lines = [line.strip() for line in str(raw or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    text = "\n".join(lines)
    def join_hyphen(match: re.Match) -> str:
        left = str(match.group(1) or "")
        right = str(match.group(2) or "")
        if left.lower() in {"what", "goes"}:
            return f"{left}-{right}"
        return f"{left}{right}"

    text = re.sub(r"([A-Za-z]{2,})-\s*\n\s*([a-z][A-Za-z]*)", join_hyphen, text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _pdf_page_has_references_heading_text(text: str) -> bool:
    return bool(
        re.search(
            r"(?mi)^\s*(?:references?|bibliography|references?\s+and\s+links|literature\s+cited)\s*$",
            str(text or ""),
        )
    )


PDF_REFERENCE_START_LINE_RE = re.compile(r"^\s*(?:\[\s*(\d{1,4})\s*\]|(\d{1,4})[.)])\s+\S")
PDF_REFERENCE_STANDALONE_NUMBER_RE = re.compile(r"^\s*\[?(\d{1,4})\]?[.)]\s*$")


def _pdf_page_reference_start_numbers(text: str) -> list[int]:
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    numbers: list[int] = []
    for idx, raw in enumerate(lines):
        line = str(raw or "").strip()
        if not line:
            continue
        match = PDF_REFERENCE_START_LINE_RE.match(line)
        if match and _is_plausible_reference_number(match.group(1) or match.group(2)):
            if _pdf_reference_window_has_signal(lines, idx):
                numbers.append(int(match.group(1) or match.group(2)))
            continue
        standalone = PDF_REFERENCE_STANDALONE_NUMBER_RE.match(line)
        if standalone and _is_plausible_reference_number(standalone.group(1)):
            if _pdf_reference_window_has_signal(lines, idx):
                numbers.append(int(standalone.group(1)))
    return numbers


def _pdf_reference_window_has_signal(lines: list[str], idx: int) -> bool:
    window = " ".join(str(line or "").strip() for line in lines[idx : idx + 6])
    window = re.sub(r"\s+", " ", window).strip()
    if not window:
        return False
    if re.search(r"\b(?:18|19|20)\d{2}\b|https?://|www\.|\bdoi\s*:|10\.\d{4,9}/|arxiv", window, re.IGNORECASE):
        return True
    if re.search(
        r"\b(?:journal|proceedings?|proc\.?|conference|ieee|acm|spie|springer|wiley|"
        r"elsevier|opt\.|optics?|photonics?|phys\.|nature|science|letters?|review|"
        r"commun\.|express|trans\.|vol\.|pp\.)\b",
        window,
        re.IGNORECASE,
    ):
        return True
    first = re.sub(r"\s+", " ", str(lines[idx] or "")).strip()
    if PDF_REFERENCE_STANDALONE_NUMBER_RE.match(first) and idx + 1 < len(lines):
        first = re.sub(r"\s+", " ", str(lines[idx + 1] or "")).strip()
    return bool(
        first.count(",") >= 2
        and re.match(r"^[A-Z][A-Za-z'\-]{1,40}(?:\s+[A-Z]\.?)?(?:,|\s)", first)
    )


def _pdf_page_has_reference_block_text(text: str) -> bool:
    numbers = _pdf_page_reference_start_numbers(text)
    if len(numbers) < 3:
        return False
    increasing_pairs = sum(1 for left, right in zip(numbers, numbers[1:]) if int(right) > int(left))
    return increasing_pairs >= max(1, len(numbers) - 2)


def _trim_pdf_page_text_to_first_reference(text: str) -> str:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")
    for idx, raw_line in enumerate(lines):
        line = str(raw_line or "").strip()
        match = PDF_REFERENCE_START_LINE_RE.match(line)
        standalone = PDF_REFERENCE_STANDALONE_NUMBER_RE.match(line)
        raw_num = (match.group(1) or match.group(2)) if match else (standalone.group(1) if standalone else None)
        if raw_num and _is_plausible_reference_number(raw_num) and _pdf_reference_window_has_signal(lines, idx):
            return "\n".join(lines[idx:]).strip()
    return raw.strip()


def _drop_pdf_reference_running_lines(text: str) -> str:
    lines: list[str] = []
    raw_lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for idx, raw in enumerate(raw_lines):
        line = str(raw or "").strip()
        prev_line = str(raw_lines[idx - 1] or "").strip() if idx > 0 else ""
        next_line = str(raw_lines[idx + 1] or "").strip() if idx + 1 < len(raw_lines) else ""
        if not line:
            lines.append(raw)
            continue
        if re.fullmatch(r"Research\s+Article", line, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"(?:TOPICAL\s+REVIEW|FRONTIERS\s+OF\s+PHYSICS)", line, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\d{4,6}-\d{1,3}", line):
            continue
        if re.fullmatch(r".+\bet\s+al\.,\s+Front\.\s+Phys\.\s+\d.+", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^Vol\.\s+\d+\b.*\b(?:Optica|Optics?|Photonics?)\b", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^#\d+\b", line):
            continue
        if re.fullmatch(r"https?://doi\.org/\S+", line, flags=re.IGNORECASE) and (
            re.match(r"^#\d+\b", prev_line) or re.match(r"^Journal\s+©\s+\d{4}", next_line, flags=re.IGNORECASE)
        ):
            continue
        if re.fullmatch(r"Journal\s+©\s+\d{4}", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^Received\s+\d{1,2}\s+\w+\s+\d{4}\b", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^\(?C\)?\s+\d{4}\s+OSA\b", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^\d{1,2}\s+\w+\s+\d{4}\s*/\s*Vol\.", line, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\d{1,4}", line):
            continue
        lines.append(raw)
    return "\n".join(lines).strip()


def _merge_pdf_page_paragraphs(parts: list[str]) -> list[str]:
    out: list[str] = []
    for raw in parts:
        part = str(raw or "").strip()
        if not part:
            continue
        if not out:
            out.append(part)
            continue
        prev = out[-1].rstrip()
        if prev.endswith("-"):
            out[-1] = prev[:-1] + part.lstrip()
            continue
        if not re.search(r"[.!?]\s*$", prev) and re.match(r"^[a-z]", part):
            out[-1] = prev + " " + part
            continue
        out.append(part)
    return out


def _pdf_page_fallback_markdown(pdf_path: Path, page_no: int) -> str:
    if fitz is None:
        return ""
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return ""
    try:
        if page_no <= 0 or page_no > len(doc):
            return ""
        page = doc.load_page(page_no - 1)
        W = float(page.rect.width)
        H = float(page.rect.height)
        main_blocks: list[tuple[int, float, str]] = []
        footnote_blocks: list[tuple[int, float, str]] = []
        for block in list(page.get_text("blocks") or []):
            if len(block) < 5:
                continue
            try:
                x0, y0, x1, y1 = float(block[0]), float(block[1]), float(block[2]), float(block[3])
            except Exception:
                continue
            text = _clean_pdf_page_block_text(str(block[4] or ""))
            if not text:
                continue
            if y1 < H * 0.07:
                continue
            if re.fullmatch(r"(?:references|bibliography)", text, re.IGNORECASE):
                continue
            if re.match(r"^\d{1,4}\.\s+[A-Z]", text) and y0 > H * 0.65:
                continue
            is_footnote = bool(y0 > H * 0.74 and x1 < W * 0.58 and len(text) >= 40)
            width = max(0.0, x1 - x0)
            if width >= W * 0.62:
                col = 0
            else:
                col = 0 if x0 < W * 0.50 else 1
            if is_footnote:
                footnote_blocks.append((col, y0, text))
            else:
                main_blocks.append((col, y0, text))
        main = _merge_pdf_page_paragraphs([text for _, _, text in sorted(main_blocks, key=lambda item: (item[0], item[1]))])
        footnotes = [text for _, _, text in sorted(footnote_blocks, key=lambda item: (item[0], item[1]))]
    finally:
        try:
            doc.close()
        except Exception:
            pass
    if not main and not footnotes:
        return ""
    parts = [f"<!-- kb_page: {int(page_no)} -->"]
    parts.extend(main)
    parts.extend(f"> {note}" for note in footnotes)
    return "\n\n".join(part for part in parts if part).strip()


def _line_start_before(text: str, offset: int) -> int:
    return str(text or "").rfind("\n", 0, max(0, int(offset))) + 1


def _insertion_offset_for_missing_page(text: str, page_no: int) -> int:
    markers: list[tuple[int, int, int]] = []
    for match in PAGE_MARKER_RE.finditer(str(text or "")):
        try:
            start = int(match.start())
            markers.append((int(match.group(1)), start, _line_start_before(text, start)))
        except Exception:
            continue
    ref_match = re.search(r"(?mi)^#{1,6}\s+References\s*$", str(text or ""))
    ref_offset = int(ref_match.start()) if ref_match else -1
    body_next_offsets = [
        line_start
        for page, _offset, line_start in markers
        if page > page_no and line_start >= 0 and (ref_offset < 0 or line_start < ref_offset)
    ]
    if body_next_offsets:
        return min(body_next_offsets)
    if ref_offset >= 0:
        reference_pages = [page for page, offset, _line_start in markers if offset >= ref_offset]
        first_reference_page = min(reference_pages) if reference_pages else 0
        if first_reference_page > 0 and page_no < first_reference_page:
            return ref_offset
    next_offsets = [line_start for page, _offset, line_start in markers if page > page_no and line_start >= 0]
    if next_offsets:
        return min(next_offsets)
    if ref_offset >= 0:
        reference_pages = [page for page, offset, _line_start in markers if offset >= ref_offset]
        first_reference_page = min(reference_pages) if reference_pages else 0
        if first_reference_page <= 0 or page_no < first_reference_page:
            return ref_offset
    return len(str(text or ""))


def _recover_missing_source_pages_from_pdf_text(md_text: str, md_path: Path, source_pdf_path: Path | str | None = None) -> tuple[str, bool]:
    text = str(md_text or "")
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    if not pdf_path:
        return text, False
    coverage = _source_page_coverage_quality(text, pdf_path)
    pages = [
        int(item.get("page") or 0)
        for item in list(coverage.get("missing_source_pages") or [])
        if int(item.get("page") or 0) > 0
    ]
    if not pages:
        return text, False
    fixed = text
    for page_no in sorted(set(pages), reverse=True):
        fallback = _pdf_page_fallback_markdown(pdf_path, page_no)
        if not fallback:
            continue
        pos = _insertion_offset_for_missing_page(fixed, page_no)
        insert = fallback.strip() + "\n\n"
        if pos > 0 and not fixed[:pos].endswith("\n\n"):
            insert = "\n" + insert
        fixed = fixed[:pos] + insert + fixed[pos:]
    return fixed, fixed != text


def _extract_pdf_reference_markdown(pdf_path: Path) -> tuple[str, int]:
    if fitz is None:
        return "", 0
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return "", 0
    page_texts: list[tuple[int, str]] = []
    try:
        in_references = False
        for page_index in range(len(doc)):
            try:
                page = doc.load_page(page_index)
                plain_text = str(page.get_text("text") or "")
                page_text = reference_ordered_page_text(page, fallback_text=plain_text).strip()
            except Exception:
                page_text = ""
            if not page_text:
                continue
            has_heading = _pdf_page_has_references_heading_text(page_text)
            has_reference_block = _pdf_page_has_reference_block_text(page_text)
            if not in_references:
                if not has_reference_block:
                    continue
                in_references = True
                if not has_heading:
                    page_text = _trim_pdf_page_text_to_first_reference(page_text)
            elif not has_heading and not has_reference_block:
                break
            page_text = _drop_pdf_reference_running_lines(page_text)
            if page_text:
                page_texts.append((page_index + 1, page_text))
    finally:
        try:
            doc.close()
        except Exception:
            pass
    if not page_texts:
        return "", 0
    raw_lines: list[str] = ["## References"]
    for page_no, page_text in page_texts:
        normalized = normalize_references_page_text(page_text)
        if not re.match(r"(?i)^#{1,6}\s+References\b", normalized.strip()):
            normalized = "## References\n\n" + normalized.lstrip()
        normalized = re.sub(r"(?im)^#{1,6}\s+References\b.*$", "## References", normalized.strip(), count=1)
        body_lines = normalized.splitlines()
        if body_lines and re.match(r"(?i)^#{1,6}\s+References\b", body_lines[0].strip()):
            body_lines = body_lines[1:]
        body_lines = [line for line in body_lines if str(line or "").strip()]
        if not body_lines:
            continue
        raw_lines.extend(["", f"<!-- kb_page: {int(page_no)} -->", *body_lines])
    formatted = fix_references_format("\n".join(raw_lines)).strip()
    return formatted.strip(), reference_markdown_entry_count(formatted)


def _replace_references_section(md: str, references_md: str) -> str:
    text = str(md or "")
    lines = text.splitlines()
    ref_idx = -1
    for idx, line in enumerate(lines):
        if REFERENCES_HEADING_RE.match((line or "").strip()):
            ref_idx = idx
            break
    refs = str(references_md or "").strip()
    if not refs:
        return text
    if ref_idx < 0:
        return (text.rstrip() + "\n\n" + refs).strip()

    start_idx = ref_idx
    stray_start = ref_idx
    stray_count = 0
    idx = ref_idx - 1
    while idx >= 0 and ref_idx - idx <= 12:
        st = (lines[idx] or "").strip()
        if not st:
            idx -= 1
            continue
        if _looks_markdown_reference_payload_line(st):
            stray_start = idx
            stray_count += 1
            idx -= 1
            continue
        break
    if stray_count > 0:
        start_idx = stray_start

    tail_idx = len(lines)
    ref_signal = 0
    non_ref_run = 0
    non_ref_start = -1
    for idx in range(ref_idx + 1, len(lines)):
        st = (lines[idx] or "").strip()
        if not st:
            continue
        if REFERENCES_HEADING_RE.match(st):
            continue
        if _looks_markdown_reference_payload_line(st):
            ref_signal += 1
            non_ref_run = 0
            non_ref_start = -1
            continue
        if ref_signal >= 1 and _is_post_references_resume_heading_line(st):
            tail_idx = idx
            break
        if ref_signal >= 3 and _post_reference_body_heading_line(st):
            tail_idx = non_ref_start if non_ref_start >= 0 else idx
            break
        if ref_signal >= 8:
            if non_ref_run == 0:
                non_ref_start = idx
            non_ref_run += 1
            if non_ref_run >= 8 and non_ref_start >= 0:
                tail_idx = non_ref_start
                break
    tail_lines = lines[tail_idx:] if tail_idx < len(lines) else []
    if tail_lines and _post_reference_tail_should_precede_references(tail_lines):
        clean_tail = _clean_post_reference_body_tail(tail_lines)
        refs_for_tail = _drop_leading_duplicate_reference_page_marker(
            refs,
            _last_page_marker_in_lines([*lines[:start_idx], *clean_tail]),
        )
        parts = [*lines[:start_idx], *clean_tail, "", *refs_for_tail.splitlines()]
        return "\n".join(parts).strip()

    parts = [*lines[:start_idx], *refs.splitlines()]
    if tail_idx < len(lines):
        parts.extend(["", *tail_lines])
    return "\n".join(parts).strip()


def _last_page_marker_before_references(md: str) -> int:
    text = str(md or "")
    ref_match = re.search(r"(?mi)^#{1,6}\s+References\s*$", text)
    ref_offset = int(ref_match.start()) if ref_match else len(text)
    last_page = 0
    for match in PAGE_MARKER_RE.finditer(text):
        if int(match.start()) >= ref_offset:
            continue
        try:
            last_page = int(match.group(1))
        except Exception:
            continue
    return last_page


def _last_page_marker_in_lines(lines: list[str]) -> int:
    last_page = 0
    for line in list(lines or []):
        for match in PAGE_MARKER_RE.finditer(str(line or "")):
            try:
                last_page = int(match.group(1))
            except Exception:
                continue
    return last_page


def _clean_post_reference_body_tail(lines: list[str]) -> list[str]:
    raw = [str(line or "") for line in list(lines or [])]

    def next_nonempty(start: int) -> str:
        for candidate in raw[start + 1 :]:
            if str(candidate or "").strip():
                return str(candidate or "").strip()
        return ""

    out: list[str] = []
    for idx, line in enumerate(raw):
        st = line.strip()
        if _looks_markdown_reference_payload_line(st):
            continue
        if PAGE_MARKER_RE.fullmatch(st) and _looks_markdown_reference_payload_line(next_nonempty(idx)):
            continue
        out.append(line)

    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    compact: list[str] = []
    blank_run = 0
    for line in out:
        if not line.strip():
            blank_run += 1
            if blank_run > 2:
                continue
        else:
            blank_run = 0
        compact.append(line)
    return compact


def _drop_leading_duplicate_reference_page_marker(references_md: str, page_no: int) -> str:
    if page_no <= 0:
        return str(references_md or "")
    pattern = rf"(?im)^(#{{1,6}}\s+References\s*)\n\s*<!--\s*kb_page:\s*{int(page_no)}\s*-->\s*\n*"
    return re.sub(pattern, r"\1\n\n", str(references_md or "").strip(), count=1)


def _looks_markdown_reference_payload_line(line: str) -> bool:
    st = str(line or "").strip()
    if not st or PAGE_MARKER_RE.fullmatch(st):
        return False
    if _looks_like_author_year_reference_text(st):
        return True
    match = re.match(r"^\s*(?:\[\s*(\d{1,4})\s*\]|(\d{1,4})[.)])\s+(.+)$", st)
    if not match:
        return False
    try:
        n = int(match.group(1) or match.group(2) or 0)
    except Exception:
        return False
    if n <= 0 or 1800 <= n <= 2099:
        return False
    body = str(match.group(3) or "").strip()
    if len(body) < 20:
        return False
    if re.match(
        r"^(?:abstract|introduction|background|related\s+work|method(?:s|ology)?|"
        r"experiment(?:s|al)?|results?|discussion|conclusions?|funding|"
        r"acknowledg(?:e)?ments?|appendix|supplementary|supplemental)\b",
        body,
        re.IGNORECASE,
    ):
        return False
    return bool(
        re.search(r"\b(?:18|19|20)\d{2}\b", body)
        or re.search(r"\b(?:doi\s*:|10\.\d{4,9}/|arxiv\s*:)", body, re.IGNORECASE)
        or re.search(
            r"\b(?:journal|proceedings?|proc\.?|opt\.|phys\.|nat\.|nature|science|"
            r"ieee|acm|appl\.|express|letters?|review|commun\.|photonics)\b",
            body,
            re.IGNORECASE,
        )
    )


def _post_reference_body_heading_line(line: str) -> bool:
    st = str(line or "").strip()
    if not st:
        return False
    st = re.sub(r"^#{1,6}\s+", "", st).strip()
    return bool(
        re.fullmatch(
            r"(?:\d+(?:\.\d+)*\.?\s+)?(?:conclusions?|method(?:s|ology)?|discussion|"
            r"funding|acknowledg(?:e)?ments?)\s*[:.]?",
            st,
            re.IGNORECASE,
        )
    )


def _post_reference_tail_should_precede_references(lines: list[str]) -> bool:
    sample = "\n".join(str(line or "") for line in list(lines or [])[:30])
    if re.search(r"\b(?:supplementary|supplemental|appendix|appendices)\b", sample, re.IGNORECASE):
        return False
    return bool(
        re.search(
            r"(?mi)^\s*(?:#{1,6}\s+)?(?:\d+(?:\.\d+)*\.?\s+)?(?:conclusions?|method(?:s|ology)?|"
            r"discussion|funding|acknowledg(?:e)?ments?)\b",
            sample,
        )
    )


def _backfill_references_from_pdf_text(md_text: str, md_path: Path, source_pdf_path: Path | str | None = None) -> tuple[str, bool]:
    text = str(md_text or "")
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    if not pdf_path:
        return text, False
    before_map = extract_references_map_from_md(text)
    before_extracted = len(before_map)
    before_missing = set(_reference_map_missing_numbers(before_map))
    references_md, recovered_count = _extract_pdf_reference_markdown(pdf_path)
    recovered_map = extract_references_map_from_md(references_md)
    recovered_missing = set(_reference_map_missing_numbers(recovered_map))
    fills_existing_gap = bool(before_missing and len(recovered_missing) < len(before_missing))
    inflated_tail = _reference_map_has_inflated_tail(before_map, recovered_map)
    if not references_md or recovered_count < 5:
        return text, False
    if recovered_count < before_extracted and not (fills_existing_gap or inflated_tail):
        return text, False
    if before_missing and not fills_existing_gap:
        return text, False
    references_md = _drop_leading_duplicate_reference_page_marker(
        references_md,
        _last_page_marker_before_references(text),
    )
    fixed = _replace_references_section(text, references_md)
    return fixed, fixed != text


def repair_markdown_text(
    md_path: Path | str,
    md_text: str,
    *,
    issue_codes: list[str] | None = None,
    default_to_postprocess: bool = False,
    source_pdf_path: Path | str | None = None,
    allow_source_pdf_inference: bool = True,
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    before_text = str(md_text or "")
    before_metrics = _metric_view(path, before_text)
    source_repairs_enabled = bool(source_pdf_path) or bool(allow_source_pdf_inference)
    before_source_quality = _source_quality_view(
        path,
        before_text,
        before_metrics,
        source_pdf_path=source_pdf_path,
        allow_source_pdf_inference=allow_source_pdf_inference,
    )
    requested_codes = [str(code or "").strip().lower() for code in list(issue_codes or []) if str(code or "").strip()]
    before_issue_codes = _issue_codes_from_context(path, before_text, before_metrics, source_quality=before_source_quality)
    active_codes = requested_codes or before_issue_codes
    active_strategy_names: set[str] = set()
    for code in active_codes:
        for name in conversion_repair_strategy_for_issue(code).get("strategies") or []:
            active_strategy_names.add(str(name))
    if not active_strategy_names and default_to_postprocess:
        active_strategy_names.update({"postprocess_markdown"})

    text = before_text
    applied: list[str] = []

    if active_strategy_names:
        if "ensure_page_anchor" in active_strategy_names:
            changed = False
            if source_repairs_enabled:
                text, changed = _recover_page_markers_from_pdf_text(text, path, source_pdf_path)
                if changed:
                    applied.append("recover_page_markers_from_pdf")
            if not changed:
                text, changed = _ensure_page_anchor(text)
                if changed:
                    applied.append("ensure_page_anchor")

        if source_repairs_enabled and "realign_page_markers_from_pdf" in active_strategy_names:
            text, changed = _realign_page_markers_from_pdf_text(text, path, source_pdf_path)
            if changed:
                applied.append("realign_page_markers_from_pdf")

        if "normalize_page_markers" in active_strategy_names:
            text, changed = _normalize_page_marker_sequence(text)
            if changed:
                applied.append("normalize_page_markers")

        if "balance_display_math" in active_strategy_names:
            text, changed = _balance_display_math(text)
            if changed:
                applied.append("balance_display_math")

        if "figure_metadata_captions" in active_strategy_names:
            text, changed = _inject_figure_metadata_captions(path, text)
            if changed:
                applied.append("figure_metadata_captions")

        if source_repairs_enabled and "pdf_text_captions" in active_strategy_names:
            text, changed = _inject_pdf_text_captions(path, text, source_pdf_path)
            if changed:
                applied.append("pdf_text_captions")

        if "repair_missing_image_links" in active_strategy_names:
            text, changed = _repair_missing_image_links(path, text)
            if changed:
                applied.append("repair_missing_image_links")

        if "abstract_heading_only" in active_strategy_names:
            text, changed = _insert_abstract_heading_only(text)
            if changed:
                applied.append("abstract_heading_only")

        if "move_early_references_to_end" in active_strategy_names:
            text, changed = _move_early_references_to_end(text)
            if changed:
                applied.append("move_early_references_to_end")

        if "normalize_markdown_tables" in active_strategy_names:
            text, changed = _normalize_markdown_tables_only(text)
            if changed:
                applied.append("normalize_markdown_tables")

        if "normalize_heading_levels" in active_strategy_names:
            text, changed = _normalize_heading_level_jumps(text)
            if changed:
                applied.append("normalize_heading_levels")

        if source_repairs_enabled and "recover_missing_source_pages" in active_strategy_names:
            text, changed = _recover_missing_source_pages_from_pdf_text(text, path, source_pdf_path)
            if changed:
                applied.append("recover_missing_source_pages")

        if source_repairs_enabled and "pdf_reference_backfill" in active_strategy_names:
            text, changed = _backfill_references_from_pdf_text(text, path, source_pdf_path)
            if changed:
                applied.append("pdf_reference_backfill")

        if "postprocess_markdown" in active_strategy_names:
            postprocessed = postprocess_markdown(text)
            if postprocessed != text:
                text = postprocessed
                applied.append("postprocess_markdown")

        if "promote_collapsed_review_headings" in active_strategy_names:
            text, changed = _promote_collapsed_review_headings(text)
            if changed:
                applied.append("promote_collapsed_review_headings")

        # Post-processing can remove leading comments when legacy files are oddly shaped;
        # preserve at least one stable reader anchor for quality-center repair.
        text, changed = _ensure_page_anchor(text)
        if changed:
            applied.append("ensure_page_anchor")

    after_metrics = _metric_view(path, text)
    after_source_quality = _source_quality_view(
        path,
        text,
        after_metrics,
        source_pdf_path=source_pdf_path,
        allow_source_pdf_inference=allow_source_pdf_inference,
    )
    after_issue_codes = _issue_codes_from_context(path, text, after_metrics, source_quality=after_source_quality)
    changed_text = text != before_text
    regression_reasons = _regression_reasons(before_text, text) if changed_text else []
    if (
        changed_text
        and regression_reasons
        and "pdf_reference_backfill" in applied
        and set(regression_reasons).issubset({"reference_lines_dropped", "reference_index_regressed"})
        and _reference_map_has_inflated_tail(
            extract_references_map_from_md(before_text),
            extract_references_map_from_md(text),
        )
    ):
        regression_reasons = []
    safe_to_use = changed_text and not regression_reasons
    attempted_applied = list(applied)
    final_applied = list(applied)

    if changed_text and regression_reasons:
        fallback_text = before_text
        fallback_applied: list[str] = []
        for label in applied:
            candidate = fallback_text
            changed = False
            if source_repairs_enabled and label == "recover_page_markers_from_pdf":
                candidate, changed = _recover_page_markers_from_pdf_text(fallback_text, path, source_pdf_path)
            elif label == "ensure_page_anchor":
                candidate, changed = _ensure_page_anchor(fallback_text)
            elif source_repairs_enabled and label == "realign_page_markers_from_pdf":
                candidate, changed = _realign_page_markers_from_pdf_text(fallback_text, path, source_pdf_path)
            elif label == "normalize_page_markers":
                candidate, changed = _normalize_page_marker_sequence(fallback_text)
            elif label == "figure_metadata_captions":
                candidate, changed = _inject_figure_metadata_captions(path, fallback_text)
            elif source_repairs_enabled and label == "pdf_text_captions":
                candidate, changed = _inject_pdf_text_captions(path, fallback_text, source_pdf_path)
            elif label == "repair_missing_image_links":
                candidate, changed = _repair_missing_image_links(path, fallback_text)
            elif label == "abstract_heading_only":
                candidate, changed = _insert_abstract_heading_only(fallback_text)
            elif label == "move_early_references_to_end":
                candidate, changed = _move_early_references_to_end(fallback_text)
            elif label == "normalize_markdown_tables":
                candidate, changed = _normalize_markdown_tables_only(fallback_text)
            elif label == "normalize_heading_levels":
                candidate, changed = _normalize_heading_level_jumps(fallback_text)
            elif label == "promote_collapsed_review_headings":
                candidate, changed = _promote_collapsed_review_headings(fallback_text)
            elif source_repairs_enabled and label == "recover_missing_source_pages":
                candidate, changed = _recover_missing_source_pages_from_pdf_text(fallback_text, path, source_pdf_path)
            elif source_repairs_enabled and label == "pdf_reference_backfill":
                candidate, changed = _backfill_references_from_pdf_text(fallback_text, path, source_pdf_path)
            if not changed or candidate == fallback_text:
                continue
            step_reasons = _regression_reasons(fallback_text, candidate)
            if step_reasons:
                continue
            fallback_text = candidate
            fallback_applied.append(label)
        if fallback_text != before_text:
            fallback_reasons = _regression_reasons(before_text, fallback_text)
            if not fallback_reasons:
                text = fallback_text
                final_applied = fallback_applied
                after_metrics = _metric_view(path, text)
                after_source_quality = _source_quality_view(
                    path,
                    text,
                    after_metrics,
                    source_pdf_path=source_pdf_path,
                    allow_source_pdf_inference=allow_source_pdf_inference,
                )
                after_issue_codes = _issue_codes_from_context(path, text, after_metrics, source_quality=after_source_quality)
                changed_text = True
                regression_reasons = []
                safe_to_use = True

    final_text = text if safe_to_use or not changed_text else before_text
    rolled_back = bool(final_text == before_text and attempted_applied and changed_text and regression_reasons)
    if rolled_back:
        final_applied = []

    return {
        "ok": bool(safe_to_use or not changed_text),
        "changed": bool(safe_to_use),
        "unsafe": bool(changed_text and regression_reasons),
        "path": str(path),
        "backup_path": "",
        "applied": final_applied,
        "attempted_applied": attempted_applied,
        "rolled_back": rolled_back,
        "attempted_issue_codes": active_codes,
        "issue_codes_before": before_issue_codes,
        "issue_codes_after": after_issue_codes if safe_to_use or not changed_text else before_issue_codes,
        "remaining_issue_codes": after_issue_codes if safe_to_use or not changed_text else before_issue_codes,
        "regression_reasons": regression_reasons,
        "before": before_metrics,
        "after": after_metrics if safe_to_use or not changed_text else before_metrics,
        "source_quality_before": before_source_quality,
        "source_quality_after": after_source_quality if safe_to_use or not changed_text else before_source_quality,
        "repaired_text": final_text,
    }


def repair_markdown_quality(
    md_path: Path | str,
    *,
    issue_codes: list[str] | None = None,
    create_backup: bool = True,
    source_pdf_path: Path | str | None = None,
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    before_text = path.read_text(encoding="utf-8", errors="replace")
    result = repair_markdown_text(
        path,
        before_text,
        issue_codes=issue_codes,
        default_to_postprocess=True,
        source_pdf_path=source_pdf_path,
    )
    text = str(result.get("repaired_text") or before_text)
    safe_to_write = bool(result.get("changed"))

    backup_path = ""
    if safe_to_write:
        if create_backup:
            backup = path.with_suffix(path.suffix + ".bak")
            try:
                shutil.copyfile(path, backup)
                backup_path = str(backup)
            except Exception:
                backup_path = ""
        path.write_text(text, encoding="utf-8")

    out = dict(result)
    out["backup_path"] = backup_path
    out.pop("repaired_text", None)
    return out
