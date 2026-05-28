from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from .post_processing import postprocess_markdown
from .quality_acceptance import summarize_conversion_quality
from .quality_compare import compare_markdown_quality


PAGE_MARKER_RE = re.compile(r"<!--\s*kb_page:\s*(\d+)\s*-->", re.IGNORECASE)
DISPLAY_MATH_DELIMITER_RE = re.compile(r"^\s*\$\$\s*$")
IMAGE_LINE_RE = re.compile(r"^(\s*)!\[([^\]]*)]\(([^)]+)\)\s*$")
CAPTION_LINE_RE = re.compile(
    r"^\s*(?:\*{1,2}\s*)?(?:fig(?:ure)?\.?|table|algorithm)\s*(?:\d+|[A-Za-z](?:\.\d+)?|[IVXLC]+)\b",
    re.IGNORECASE,
)
CONVERSION_QUALITY_RESULT_FILENAME = "conversion_quality_result.json"


CONVERSION_REPAIR_STRATEGIES: dict[str, dict[str, Any]] = {
    "analyzer_errors": {
        "label": "Run deterministic Markdown post-processing",
        "safe": True,
        "strategies": ["postprocess_markdown", "balance_display_math", "figure_metadata_captions"],
    },
    "analyzer_warnings": {
        "label": "Normalize headings, captions, tables, and layout noise",
        "safe": True,
        "strategies": ["postprocess_markdown", "figure_metadata_captions"],
    },
    "missing_abstract": {
        "label": "Infer and insert Abstract heading from front matter",
        "safe": True,
        "strategies": ["postprocess_markdown"],
    },
    "missing_page_markers": {
        "label": "Insert a fallback page anchor at the Markdown start",
        "safe": True,
        "strategies": ["ensure_page_anchor"],
    },
    "page_marker_gaps": {
        "label": "Normalize duplicate and out-of-place page anchors",
        "safe": True,
        "strategies": ["postprocess_markdown"],
    },
    "missing_captions": {
        "label": "Recover visible captions from alt text and figure metadata sidecars",
        "safe": True,
        "strategies": ["postprocess_markdown", "figure_metadata_captions"],
    },
    "unclosed_display_math": {
        "label": "Close a trailing display-math block when the delimiter is unbalanced",
        "safe": True,
        "strategies": ["balance_display_math"],
    },
    "heading_level_jumps": {
        "label": "Rebalance heading levels using the established heading policy",
        "safe": True,
        "strategies": ["postprocess_markdown"],
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
        "strategies": [str(item) for item in list(strategy.get("strategies") or []) if str(item or "").strip()],
    }


def conversion_quality_result_path(md_path: Path | str) -> Path:
    return Path(md_path).expanduser().parent / CONVERSION_QUALITY_RESULT_FILENAME


def _recommended_action_for_issues(issue_codes: list[str]) -> str:
    codes = [str(code or "").strip().lower() for code in list(issue_codes or []) if str(code or "").strip()]
    if not codes:
        return "none"
    if all(bool(conversion_repair_strategy_for_issue(code).get("safe")) for code in codes):
        return "autofix_available"
    if any(not bool(conversion_repair_strategy_for_issue(code).get("safe")) for code in codes):
        return "reconvert"
    return "review"


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


def write_conversion_quality_result(
    md_path: Path | str,
    *,
    auto_repair_result: dict[str, Any] | None = None,
    auto_repair_enabled: bool = True,
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    report_path = conversion_quality_result_path(path)
    metrics = _metric_view(path, path.read_text(encoding="utf-8", errors="replace"))
    repair = dict(auto_repair_result or {})
    repair.pop("repaired_text", None)
    remaining = [
        str(code or "").strip().lower()
        for code in list(repair.get("remaining_issue_codes") or _issue_codes_from_metrics(metrics))
        if str(code or "").strip()
    ]
    recommended_action = _recommended_action_for_issues(remaining)
    md_stat = _current_markdown_stat(path)
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
        "recommended_action": recommended_action,
        "needs_reconvert": recommended_action == "reconvert",
        "metrics": metrics,
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


def _ensure_page_anchor(md: str) -> tuple[str, bool]:
    text = str(md or "")
    if PAGE_MARKER_RE.search(text):
        return text, False
    stripped = text.lstrip()
    prefix_len = len(text) - len(stripped)
    fixed = f"{text[:prefix_len]}<!-- kb_page: 1 -->\n\n{stripped}" if stripped else "<!-- kb_page: 1 -->"
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


def _regression_reasons(base_text: str, candidate_text: str) -> list[str]:
    comparison = compare_markdown_quality(base_text, candidate_text)
    base = comparison.get("base") if isinstance(comparison.get("base"), dict) else {}
    cand = comparison.get("candidate") if isinstance(comparison.get("candidate"), dict) else {}
    flags = comparison.get("regression_flags") if isinstance(comparison.get("regression_flags"), dict) else {}
    reasons = [str(key) for key, value in flags.items() if bool(value)]
    base_chars = int(base.get("chars") or 0)
    cand_chars = int(cand.get("chars") or 0)
    if base_chars > 1000 and cand_chars < int(base_chars * 0.82):
        reasons.append("content_shrank_too_much")
    return reasons


def repair_markdown_text(
    md_path: Path | str,
    md_text: str,
    *,
    issue_codes: list[str] | None = None,
    default_to_postprocess: bool = False,
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    before_text = str(md_text or "")
    before_metrics = _metric_view(path, before_text)
    requested_codes = [str(code or "").strip().lower() for code in list(issue_codes or []) if str(code or "").strip()]
    before_issue_codes = _issue_codes_from_metrics(before_metrics)
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
            text, changed = _ensure_page_anchor(text)
            if changed:
                applied.append("ensure_page_anchor")

        if "balance_display_math" in active_strategy_names:
            text, changed = _balance_display_math(text)
            if changed:
                applied.append("balance_display_math")

        if "figure_metadata_captions" in active_strategy_names:
            text, changed = _inject_figure_metadata_captions(path, text)
            if changed:
                applied.append("figure_metadata_captions")

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
    after_issue_codes = _issue_codes_from_metrics(after_metrics)
    changed_text = text != before_text
    regression_reasons = _regression_reasons(before_text, text) if changed_text else []
    safe_to_use = changed_text and not regression_reasons
    final_text = text if safe_to_use or not changed_text else before_text

    return {
        "ok": bool(safe_to_use or not changed_text),
        "changed": bool(safe_to_use),
        "unsafe": bool(changed_text and regression_reasons),
        "path": str(path),
        "backup_path": "",
        "applied": applied,
        "issue_codes_before": before_issue_codes,
        "issue_codes_after": after_issue_codes if safe_to_use or not changed_text else before_issue_codes,
        "remaining_issue_codes": after_issue_codes if safe_to_use or not changed_text else before_issue_codes,
        "regression_reasons": regression_reasons,
        "before": before_metrics,
        "after": after_metrics if safe_to_use or not changed_text else before_metrics,
        "repaired_text": final_text,
    }


def repair_markdown_quality(
    md_path: Path | str,
    *,
    issue_codes: list[str] | None = None,
    create_backup: bool = True,
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    before_text = path.read_text(encoding="utf-8", errors="replace")
    result = repair_markdown_text(
        path,
        before_text,
        issue_codes=issue_codes,
        default_to_postprocess=True,
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
