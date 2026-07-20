from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from kb.converter.quality_compare import summarize_markdown_quality
from kb.converter.tables import markdown_table_issue_counts
from kb.converter.text_utils import count_mojibake
from kb.reference_index import extract_references_map_from_md


_PAGE_MARKER_RE = re.compile(r"<!--\s*kb_page:\s*(\d+)\s*-->", re.IGNORECASE)
_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_DISPLAY_MATH_DELIMITER_RE = re.compile(r"^\s*\$\$\s*$")
_CONVERSION_RETRY_MARKER_RE = re.compile(r"<!--\s*kb:conversion_retry\s+(.+?)\s*-->", re.IGNORECASE)
_CONVERSION_MARKER_ATTR_RE = re.compile(r"([A-Za-z_][\w-]*)=(?:\"([^\"]*)\"|'([^']*)'|([^\s]+))")
_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]\n]{1,160}\]\([^)\n]+\)")
_PROSE_DISPLAY_MATH_WORD_RE = re.compile(r"\b[A-Za-z]{2,}\b")
_PROSE_DISPLAY_MATH_COMMON_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "denotes",
    "for",
    "from",
    "in",
    "indicates",
    "is",
    "of",
    "or",
    "refers",
    "represents",
    "that",
    "the",
    "this",
    "to",
    "where",
    "which",
    "with",
}


@dataclass(frozen=True)
class ConversionQualityMetrics:
    chars: int
    lines: int
    nonempty_lines: int
    heading_count: int
    h1_count: int
    h2_count: int
    h3_plus_count: int
    heading_level_jump_count: int
    has_abstract_heading: bool
    page_marker_count: int
    page_marker_min: int
    page_marker_max: int
    page_marker_gap_count: int
    image_count: int
    missing_image_count: int
    caption_count: int
    table_block_count: int
    table_literal_break_count: int
    collapsed_table_row_count: int
    ambiguous_table_break_row_count: int
    duplicate_table_count: int
    fragmented_table_column_count: int
    fragmented_table_duplicate_count: int
    display_math_block_count: int
    unclosed_display_math_block_count: int
    prose_dominant_display_math_block_count: int
    display_math_markdown_link_count: int
    inline_math_count: int
    conversion_retry_marker_count: int
    conversion_retry_math_text_count: int
    conversion_retry_equation_count: int
    conversion_retry_other_count: int
    conversion_retry_kind_counts: dict[str, int]
    reference_line_count: int
    extracted_reference_count: int
    max_reference_index: int
    body_citation_marker_count: int
    body_citation_expanded_index_count: int
    mojibake_count: int
    analyzer_error_count: int
    analyzer_warning_count: int


def _resolve_repo_path(path_value: str | Path, *, repo_root: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    root = repo_root or Path.cwd()
    return root / path


def _heading_level_jump_count(md_text: str) -> int:
    levels = [len(match.group(1)) for match in _HEADING_RE.finditer(md_text or "")]
    jumps = 0
    previous = 0
    for level in levels:
        if previous > 0 and level > previous + 1:
            jumps += 1
        previous = level
    return jumps


def _page_marker_stats(md_text: str) -> tuple[int, int, int, int]:
    pages = []
    for match in _PAGE_MARKER_RE.finditer(md_text or ""):
        try:
            pages.append(int(match.group(1)))
        except Exception:
            continue
    if not pages:
        return 0, 0, 0, 0
    unique = sorted(set(pages))
    duplicates = len(pages) - len(unique)
    backwards = sum(1 for prev, cur in zip(pages, pages[1:]) if cur < prev)
    return len(pages), unique[0], unique[-1], duplicates + backwards


def _display_math_unclosed_count(md_text: str) -> int:
    in_block = False
    unclosed = 0
    for raw in str(md_text or "").splitlines():
        if not _DISPLAY_MATH_DELIMITER_RE.match(raw):
            continue
        in_block = not in_block
    if in_block:
        unclosed += 1
    return unclosed


def _display_math_blocks(md_text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] | None = None
    for raw in str(md_text or "").splitlines():
        if _DISPLAY_MATH_DELIMITER_RE.match(raw):
            if current is None:
                current = []
            else:
                blocks.append("\n".join(current))
                current = None
            continue
        if current is not None:
            current.append(raw)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _display_math_content_issue_counts(md_text: str) -> tuple[int, int]:
    prose_dominant = 0
    markdown_links = 0
    for block in _display_math_blocks(md_text):
        markdown_links += len(_MARKDOWN_LINK_RE.findall(block))
        # Ignore TeX command names and count natural-language words that remain.
        # Requiring both a long phrase and several connective/definition words
        # avoids flagging ordinary equations with descriptive variable names.
        prose_probe = re.sub(r"\\[A-Za-z]+\*?", " ", block)
        words = [word.lower() for word in _PROSE_DISPLAY_MATH_WORD_RE.findall(prose_probe)]
        common_count = sum(1 for word in words if word in _PROSE_DISPLAY_MATH_COMMON_WORDS)
        alpha_chars = sum(len(word) for word in words)
        if len(words) >= 8 and common_count >= 3 and alpha_chars >= 40:
            prose_dominant += 1
    return prose_dominant, markdown_links


def _conversion_retry_stats(md_text: str) -> tuple[int, int, int, int, dict[str, int]]:
    kind_counts: dict[str, int] = {}
    for marker in _CONVERSION_RETRY_MARKER_RE.finditer(str(md_text or "")):
        attrs = {
            item.group(1).lower(): next((value for value in item.groups()[1:] if value is not None), "")
            for item in _CONVERSION_MARKER_ATTR_RE.finditer(marker.group(1))
        }
        kind = str(attrs.get("kind") or "unknown").strip().lower() or "unknown"
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    math_text = int(kind_counts.get("math_text") or 0)
    equation = int(kind_counts.get("equation") or 0)
    total = sum(kind_counts.values())
    other = max(0, total - math_text - equation)
    return total, math_text, equation, other, dict(sorted(kind_counts.items()))


def _image_target_path(md_path: Path, target: str) -> Path | None:
    raw = str(target or "").strip().strip("<>")
    if not raw or raw.startswith(("http://", "https://", "data:", "#")):
        return None
    raw = raw.split("#", 1)[0].split("?", 1)[0]
    if not raw:
        return None
    return (md_path.parent / unquote(raw)).resolve()


def _missing_image_count(md_path: Path, md_text: str) -> int:
    missing = 0
    for match in _IMAGE_RE.finditer(md_text or ""):
        target = _image_target_path(md_path, match.group(1))
        if target is not None and not target.exists():
            missing += 1
    return missing


def summarize_conversion_quality(md_path: Path, md_text: str | None = None) -> ConversionQualityMetrics:
    path = Path(md_path)
    text = path.read_text(encoding="utf-8", errors="replace") if md_text is None else str(md_text or "")
    base = summarize_markdown_quality(text)
    page_count, page_min, page_max, page_gaps = _page_marker_stats(text)
    references = extract_references_map_from_md(text)
    table_issues = markdown_table_issue_counts(text)
    prose_display_math_count, display_math_link_count = _display_math_content_issue_counts(text)
    retry_count, retry_math_text_count, retry_equation_count, retry_other_count, retry_kind_counts = (
        _conversion_retry_stats(text)
    )
    detected_reference_count = max(int(base.reference_line_count), len(references))
    max_reference_index = (
        max(references.keys(), default=0)
        if references
        else int(base.max_reference_index)
    )
    return ConversionQualityMetrics(
        chars=base.chars,
        lines=base.lines,
        nonempty_lines=base.nonempty_lines,
        heading_count=base.heading_count,
        h1_count=base.h1_count,
        h2_count=base.h2_count,
        h3_plus_count=base.h3_plus_count,
        heading_level_jump_count=_heading_level_jump_count(text),
        has_abstract_heading=base.has_abstract_heading,
        page_marker_count=page_count,
        page_marker_min=page_min,
        page_marker_max=page_max,
        page_marker_gap_count=page_gaps,
        image_count=base.image_count,
        missing_image_count=_missing_image_count(path, text),
        caption_count=base.caption_count,
        table_block_count=base.table_block_count,
        table_literal_break_count=int(table_issues.get("literal_break_count") or 0),
        collapsed_table_row_count=int(table_issues.get("collapsed_row_count") or 0),
        ambiguous_table_break_row_count=int(table_issues.get("ambiguous_break_row_count") or 0),
        duplicate_table_count=int(table_issues.get("duplicate_table_count") or 0),
        fragmented_table_column_count=int(table_issues.get("fragmented_column_count") or 0),
        fragmented_table_duplicate_count=int(table_issues.get("fragmented_duplicate_count") or 0),
        display_math_block_count=base.display_math_block_count,
        unclosed_display_math_block_count=_display_math_unclosed_count(text),
        prose_dominant_display_math_block_count=prose_display_math_count,
        display_math_markdown_link_count=display_math_link_count,
        inline_math_count=base.inline_math_count,
        conversion_retry_marker_count=retry_count,
        conversion_retry_math_text_count=retry_math_text_count,
        conversion_retry_equation_count=retry_equation_count,
        conversion_retry_other_count=retry_other_count,
        conversion_retry_kind_counts=retry_kind_counts,
        reference_line_count=detected_reference_count,
        extracted_reference_count=len(references),
        max_reference_index=max_reference_index,
        body_citation_marker_count=base.body_citation_marker_count,
        body_citation_expanded_index_count=base.body_citation_expanded_index_count,
        mojibake_count=count_mojibake(text),
        analyzer_error_count=base.analyzer_error_count,
        analyzer_warning_count=base.analyzer_warning_count,
    )


def _as_int(mapping: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(mapping.get(key, default) or default)
    except Exception:
        return int(default)


def _as_bool(mapping: dict[str, Any], key: str, default: bool = False) -> bool:
    value = mapping.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _contains_failures(md_text: str, checks: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for item in list(checks.get("must_contain_text") or []):
        needle = str(item or "")
        if needle and needle not in md_text:
            failures.append(f"missing_text:{needle[:80]}")
    occurrence_checks = checks.get("min_text_occurrences")
    if isinstance(occurrence_checks, dict):
        for item, raw_expected in occurrence_checks.items():
            needle = str(item or "")
            if not needle:
                continue
            try:
                expected = max(0, int(raw_expected or 0))
            except Exception:
                expected = 0
            actual = md_text.count(needle)
            if actual < expected:
                failures.append(f"text_occurrences:{needle[:60]}:{actual}<{expected}")
    for item in list(checks.get("must_not_contain_text") or []):
        needle = str(item or "")
        if needle and needle in md_text:
            failures.append(f"forbidden_text_present:{needle[:80]}")
    stripped = str(md_text or "").lstrip()
    for item in list(checks.get("must_start_with") or []):
        needle = str(item or "")
        if needle and not stripped.startswith(needle):
            failures.append(f"missing_prefix:{needle[:80]}")
    for item in list(checks.get("must_not_start_with") or []):
        needle = str(item or "")
        if needle and stripped.startswith(needle):
            failures.append(f"forbidden_prefix_present:{needle[:80]}")
    cursor = -1
    for item in list(checks.get("ordered_text") or []):
        needle = str(item or "")
        if not needle:
            continue
        pos = md_text.find(needle, cursor + 1)
        if pos < 0:
            failures.append(f"ordered_text_missing:{needle[:80]}")
            continue
        cursor = pos
    return failures


def evaluate_conversion_quality(
    md_path: Path,
    *,
    checks: dict[str, Any] | None = None,
    md_text: str | None = None,
) -> dict[str, Any]:
    path = Path(md_path)
    text = path.read_text(encoding="utf-8", errors="replace") if md_text is None else str(md_text or "")
    cfg = dict(checks or {})
    metrics = summarize_conversion_quality(path, text)
    values = asdict(metrics)
    failures: list[str] = []

    minimum_checks = {
        "min_chars": "chars",
        "min_nonempty_lines": "nonempty_lines",
        "min_headings": "heading_count",
        "min_h1": "h1_count",
        "min_h2": "h2_count",
        "min_page_markers": "page_marker_count",
        "min_images": "image_count",
        "min_captions": "caption_count",
        "min_tables": "table_block_count",
        "min_display_math": "display_math_block_count",
        "min_inline_math": "inline_math_count",
        "min_references": "extracted_reference_count",
        "min_reference_lines": "reference_line_count",
        "min_max_reference_index": "max_reference_index",
        "min_body_citations": "body_citation_marker_count",
        "min_body_citation_indices": "body_citation_expanded_index_count",
    }
    for check_key, metric_key in minimum_checks.items():
        if check_key not in cfg:
            continue
        expected = _as_int(cfg, check_key)
        actual = int(values.get(metric_key) or 0)
        if actual < expected:
            failures.append(f"{metric_key}:{actual}<{expected}")

    maximum_checks = {
        "max_missing_images": "missing_image_count",
        "max_page_marker_gaps": "page_marker_gap_count",
        "max_heading_level_jumps": "heading_level_jump_count",
        "max_unclosed_display_math": "unclosed_display_math_block_count",
        "max_prose_dominant_display_math": "prose_dominant_display_math_block_count",
        "max_display_math_markdown_links": "display_math_markdown_link_count",
        "max_conversion_retries": "conversion_retry_marker_count",
        "max_conversion_retry_math_text": "conversion_retry_math_text_count",
        "max_conversion_retry_equation": "conversion_retry_equation_count",
        "max_mojibake": "mojibake_count",
        "max_analyzer_errors": "analyzer_error_count",
        "max_analyzer_warnings": "analyzer_warning_count",
    }
    for check_key, metric_key in maximum_checks.items():
        if check_key not in cfg:
            continue
        expected = _as_int(cfg, check_key)
        actual = int(values.get(metric_key) or 0)
        if actual > expected:
            failures.append(f"{metric_key}:{actual}>{expected}")

    if _as_bool(cfg, "require_abstract_heading") and not metrics.has_abstract_heading:
        failures.append("missing_abstract_heading")

    failures.extend(_contains_failures(text, cfg))

    return {
        "ok": not failures,
        "path": str(path),
        "metrics": values,
        "failures": failures,
    }


def load_quality_manifest(path: Path | str, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or Path.cwd()
    manifest_path = _resolve_repo_path(path, repo_root=root)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    defaults = dict(data.get("defaults") or {})
    cases: list[dict[str, Any]] = []
    for raw_case in list(data.get("cases") or []):
        if not isinstance(raw_case, dict):
            continue
        case = dict(raw_case)
        checks = dict(defaults)
        checks.update(case.get("checks") or {})
        case["checks"] = checks
        md_path = _resolve_repo_path(str(case.get("md_path") or ""), repo_root=root)
        case["_md_abspath"] = str(md_path)
        case["_exists"] = md_path.exists()
        cases.append(case)
    out = dict(data)
    out["cases"] = cases
    out["_manifest_path"] = str(manifest_path)
    return out
