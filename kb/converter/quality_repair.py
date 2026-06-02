from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from .post_processing import postprocess_markdown
from .quality_acceptance import summarize_conversion_quality
from .quality_compare import compare_markdown_quality
from .reference_markdown import fix_references_format, normalize_references_page_text
from .reference_page_vl import reference_markdown_entry_count
from .tables import normalize_markdown_table_block
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
CONVERSION_QUALITY_RESULT_FILENAME = "conversion_quality_result.json"
MAX_CONVERSION_REPAIR_ATTEMPTS = 30
PAGE_ALIGNMENT_NGRAMS = (8, 6)
PAGE_ALIGNMENT_DEFAULT_NGRAM = PAGE_ALIGNMENT_NGRAMS[0]
SOURCE_PAGE_COVERAGE_THRESHOLD = 0.66
SOURCE_PAGE_MIN_RARE_TOKENS = 60
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
        "strategies": ["normalize_page_markers", "postprocess_markdown"],
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
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    report_path = conversion_quality_result_path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    metrics = _metric_view(path, text)
    source_quality = _source_quality_view(path, text, metrics, source_pdf_path=source_pdf_path)
    repair = dict(auto_repair_result or {})
    repair.pop("repaired_text", None)
    remaining = [
        str(code or "").strip().lower()
        for code in list(repair.get("remaining_issue_codes") or _issue_codes_from_context(path, text, metrics, source_quality=source_quality))
        if str(code or "").strip()
    ]
    repair_plan = plan_conversion_quality_repair(remaining, metrics=metrics)
    recommended_action = str(repair_plan.get("action") or "review")
    md_stat = _current_markdown_stat(path)
    prev = load_conversion_quality_result(path)
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
            "issue_codes_before": [str(item) for item in list(repair.get("issue_codes_before") or []) if str(item or "").strip()][:30],
            "issue_codes_after": [str(item) for item in list(repair.get("issue_codes_after") or []) if str(item or "").strip()][:30],
            "remaining_issue_codes": remaining[:30],
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
    if bool(quality.get("reference_index_truncated")):
        codes.append("reference_index_truncated")
    if bool(quality.get("references_before_body")):
        codes.append("references_before_body")
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
    pages: list[int] = []
    for match in matches:
        try:
            pages.append(int(match.group(1)))
        except Exception:
            pages.append(0)
    has_duplicate = len(set(pages)) != len(pages)
    has_backward = any(cur <= prev for prev, cur in zip(pages, pages[1:]))
    if not has_duplicate and not has_backward:
        return text, False

    next_pages: list[int] = []
    previous = 0
    for raw in pages:
        current = int(raw or 0)
        if current <= previous:
            current = previous + 1
        next_pages.append(current)
        previous = current

    idx = 0

    def repl(_match: re.Match[str]) -> str:
        nonlocal idx
        page_no = next_pages[idx]
        idx += 1
        return f"<!-- kb_page: {page_no} -->"

    fixed = PAGE_MARKER_RE.sub(repl, text)
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
            page_tokens = _rare_source_tokens(page_text)
            if len(page_tokens) < SOURCE_PAGE_MIN_RARE_TOKENS:
                continue
            assessed += 1
            coverage = len(page_tokens.intersection(md_tokens)) / max(1, len(page_tokens))
            min_coverage = min(min_coverage, coverage)
            if page_no in marker_pages:
                continue
            if coverage >= SOURCE_PAGE_COVERAGE_THRESHOLD:
                continue
            low_pages.append(
                {
                    "page": int(page_no),
                    "coverage": round(float(coverage), 4),
                    "source_token_count": int(len(page_tokens)),
                    "reason": "missing_page_anchor_and_low_text_overlap",
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


def _reference_index_truncated(text: str, metrics: dict[str, Any]) -> bool:
    reference_lines = int(metrics.get("reference_line_count") or 0)
    max_index = int(metrics.get("max_reference_index") or 0)
    extracted = int(metrics.get("extracted_reference_count") or 0)
    if reference_lines < 8 or max_index < 8:
        return False
    if extracted <= 0:
        return True
    expected = max(reference_lines, max_index)
    return extracted < max(5, int(expected * 0.55))


def _source_quality_view(
    md_path: Path,
    text: str,
    metrics: dict[str, Any],
    *,
    source_pdf_path: Path | str | None = None,
) -> dict[str, Any]:
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    pdf_stats = _pdf_source_stats(pdf_path)
    profile = _document_profile(md_path, text)
    ref_layout = _reference_layout(text)
    abstract_candidate_text, abstract_changed = _insert_abstract_heading_only(text)
    _ = abstract_candidate_text
    abstract_autofix_likely = bool(abstract_changed)
    source_text_loss = False if bool(profile.get("abstract_not_applicable")) else _source_text_loss_likely(text, metrics, pdf_stats, ref_layout)
    page_quality = _page_alignment_quality(metrics, pdf_stats)
    page_coverage = _source_page_coverage_quality(text, pdf_path if pdf_path and bool(pdf_stats.get("available")) else None)
    return {
        **profile,
        "source_pdf_path": str((pdf_stats or {}).get("path") or (pdf_path or "")),
        "source_pdf_available": bool(pdf_stats.get("available")),
        "pdf_page_count": int(pdf_stats.get("page_count") or 0),
        "pdf_text_chars": int(pdf_stats.get("text_chars") or 0),
        **page_quality,
        **page_coverage,
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
    lines = str(md or "").splitlines()
    out: list[str] = []
    table_buf: list[str] = []

    def flush_table() -> None:
        nonlocal table_buf
        if not table_buf:
            return
        block = "\n".join(table_buf)
        out.extend(normalize_markdown_table_block(block).splitlines())
        table_buf = []

    for line in lines:
        stripped = str(line or "").lstrip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            table_buf.append(line)
            continue
        flush_table()
        out.append(line)
    flush_table()

    fixed = "\n".join(out)
    if str(md or "").endswith("\n"):
        fixed += "\n"
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


def _regression_reasons(base_text: str, candidate_text: str) -> list[str]:
    comparison = compare_markdown_quality(base_text, candidate_text)
    base = comparison.get("base") if isinstance(comparison.get("base"), dict) else {}
    cand = comparison.get("candidate") if isinstance(comparison.get("candidate"), dict) else {}
    flags = comparison.get("regression_flags") if isinstance(comparison.get("regression_flags"), dict) else {}
    reasons = [str(key) for key, value in flags.items() if bool(value)]
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


def _page_marker_offsets_from_pdf_text(md_text: str, pdf_path: Path) -> dict[int, int]:
    if fitz is None:
        return {}
    path = Path(pdf_path).expanduser()
    try:
        if not path.exists() or not path.is_file():
            return {}
    except Exception:
        return {}

    md_tokens = _word_tokens_with_offsets(md_text)
    if len(md_tokens) < min(PAGE_ALIGNMENT_NGRAMS):
        return {}

    offsets: dict[int, int] = {1: 0}
    previous_token_index = -1
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
            if len(md_tokens) < width:
                continue
            md_grams: dict[tuple[str, ...], list[int]] = {}
            for idx in range(0, len(md_tokens) - width + 1):
                gram = tuple(tok for tok, _ in md_tokens[idx : idx + width])
                bucket = md_grams.setdefault(gram, [])
                if len(bucket) < 30:
                    bucket.append(idx)

            current_offsets: dict[int, int] = {1: 0}
            previous_token_index = -1
            for page_index, page_tokens in enumerate(page_tokens_by_index):
                choices: list[tuple[int, int, int]] = []
                for gram, rare, page_token_start in _page_alignment_candidates(page_tokens, width):
                    for md_token_index in md_grams.get(gram, []):
                        if md_token_index <= previous_token_index:
                            continue
                        choices.append((md_token_index, -rare, page_token_start))
                if not choices:
                    continue
                choices.sort()
                md_token_index = int(choices[0][0])
                if page_index > 0 and md_token_index < 10:
                    continue
                previous_token_index = md_token_index
                page_no = page_index + 1
                if page_no > 1:
                    current_offsets[page_no] = _line_start_for_offset(md_text, int(md_tokens[md_token_index][1]))

            if len(current_offsets) > len(best_offsets):
                best_offsets = current_offsets
            matched_later = len([p for p in current_offsets if p > 1])
            required_good = 1 if pdf_page_count <= 2 else max(2, int((pdf_page_count - 1) * 0.55))
            if matched_later >= required_good:
                offsets = current_offsets
                break
        else:
            offsets = best_offsets
    finally:
        try:
            doc.close()
        except Exception:
            pass

    matched_later_pages = len([p for p in offsets if p > 1])
    if matched_later_pages <= 0:
        return {1: 0}
    if pdf_page_count >= 6 and matched_later_pages < 2:
        return {1: 0}
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
    return bool(re.search(r"(?mi)^\s*(?:references|bibliography)\s*$", str(text or "")))


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
        start_index = -1
        for page_index in range(len(doc)):
            try:
                page_text = str(doc.load_page(page_index).get_text("text") or "")
            except Exception:
                page_text = ""
            if _pdf_page_has_references_heading_text(page_text):
                start_index = page_index
                break
        if start_index < 0:
            return "", 0
        for page_index in range(start_index, len(doc)):
            try:
                page_text = str(doc.load_page(page_index).get_text("text") or "").strip()
            except Exception:
                page_text = ""
            if page_text:
                page_texts.append((page_index + 1, page_text))
    finally:
        try:
            doc.close()
        except Exception:
            pass
    if not page_texts:
        return "", 0
    out_lines: list[str] = ["## References"]
    for page_no, page_text in page_texts:
        normalized = normalize_references_page_text(page_text)
        if not re.match(r"(?i)^#{1,6}\s+References\b", normalized.strip()):
            normalized = "## References\n\n" + normalized.lstrip()
        normalized = re.sub(r"(?im)^#{1,6}\s+References\b.*$", "## References", normalized.strip(), count=1)
        page_formatted = fix_references_format(normalized).strip()
        body_lines = page_formatted.splitlines()
        if body_lines and re.match(r"(?i)^#{1,6}\s+References\b", body_lines[0].strip()):
            body_lines = body_lines[1:]
        body_lines = [line for line in body_lines if str(line or "").strip()]
        if not body_lines:
            continue
        out_lines.extend(["", f"<!-- kb_page: {int(page_no)} -->", *body_lines])
    formatted = "\n".join(out_lines)
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
    return "\n".join([*lines[:ref_idx], *refs.splitlines()]).strip()


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


def _drop_leading_duplicate_reference_page_marker(references_md: str, page_no: int) -> str:
    if page_no <= 0:
        return str(references_md or "")
    pattern = rf"(?im)^(#{{1,6}}\s+References\s*)\n\s*<!--\s*kb_page:\s*{int(page_no)}\s*-->\s*\n*"
    return re.sub(pattern, r"\1\n\n", str(references_md or "").strip(), count=1)


def _backfill_references_from_pdf_text(md_text: str, md_path: Path, source_pdf_path: Path | str | None = None) -> tuple[str, bool]:
    text = str(md_text or "")
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    if not pdf_path:
        return text, False
    before_extracted = len(extract_references_map_from_md(text))
    references_md, recovered_count = _extract_pdf_reference_markdown(pdf_path)
    if not references_md or recovered_count < 5 or recovered_count < before_extracted:
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
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    before_text = str(md_text or "")
    before_metrics = _metric_view(path, before_text)
    before_source_quality = _source_quality_view(path, before_text, before_metrics, source_pdf_path=source_pdf_path)
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
            text, changed = _recover_page_markers_from_pdf_text(text, path, source_pdf_path)
            if changed:
                applied.append("recover_page_markers_from_pdf")
            else:
                text, changed = _ensure_page_anchor(text)
                if changed:
                    applied.append("ensure_page_anchor")

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

        if "pdf_text_captions" in active_strategy_names:
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

        if "recover_missing_source_pages" in active_strategy_names:
            text, changed = _recover_missing_source_pages_from_pdf_text(text, path, source_pdf_path)
            if changed:
                applied.append("recover_missing_source_pages")

        if "pdf_reference_backfill" in active_strategy_names:
            text, changed = _backfill_references_from_pdf_text(text, path, source_pdf_path)
            if changed:
                applied.append("pdf_reference_backfill")

        if "postprocess_markdown" in active_strategy_names:
            postprocessed = postprocess_markdown(text)
            if postprocessed != text:
                text = postprocessed
                applied.append("postprocess_markdown")

        # Post-processing can remove leading comments when legacy files are oddly shaped;
        # preserve at least one stable reader anchor for quality-center repair.
        text, changed = _ensure_page_anchor(text)
        if changed:
            applied.append("ensure_page_anchor")

    after_metrics = _metric_view(path, text)
    after_source_quality = _source_quality_view(path, text, after_metrics, source_pdf_path=source_pdf_path)
    after_issue_codes = _issue_codes_from_context(path, text, after_metrics, source_quality=after_source_quality)
    changed_text = text != before_text
    regression_reasons = _regression_reasons(before_text, text) if changed_text else []
    safe_to_use = changed_text and not regression_reasons
    final_applied = list(applied)

    if changed_text and regression_reasons:
        fallback_text = before_text
        fallback_applied: list[str] = []
        for label in applied:
            candidate = fallback_text
            changed = False
            if label == "recover_page_markers_from_pdf":
                candidate, changed = _recover_page_markers_from_pdf_text(fallback_text, path, source_pdf_path)
            elif label == "ensure_page_anchor":
                candidate, changed = _ensure_page_anchor(fallback_text)
            elif label == "normalize_page_markers":
                candidate, changed = _normalize_page_marker_sequence(fallback_text)
            elif label == "figure_metadata_captions":
                candidate, changed = _inject_figure_metadata_captions(path, fallback_text)
            elif label == "pdf_text_captions":
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
            elif label == "recover_missing_source_pages":
                candidate, changed = _recover_missing_source_pages_from_pdf_text(fallback_text, path, source_pdf_path)
            elif label == "pdf_reference_backfill":
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
                after_source_quality = _source_quality_view(path, text, after_metrics, source_pdf_path=source_pdf_path)
                after_issue_codes = _issue_codes_from_context(path, text, after_metrics, source_quality=after_source_quality)
                changed_text = True
                regression_reasons = []
                safe_to_use = True

    final_text = text if safe_to_use or not changed_text else before_text

    return {
        "ok": bool(safe_to_use or not changed_text),
        "changed": bool(safe_to_use),
        "unsafe": bool(changed_text and regression_reasons),
        "path": str(path),
        "backup_path": "",
        "applied": final_applied,
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
