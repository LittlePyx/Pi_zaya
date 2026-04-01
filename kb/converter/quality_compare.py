from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .md_analyzer import MarkdownAnalyzer


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", flags=re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_REF_HEADING_RE = re.compile(r"^#{1,6}\s+References?\s*$", flags=re.IGNORECASE | re.MULTILINE)
_REF_LINE_RE = re.compile(r"^\[(\d{1,4})\]\s+", flags=re.MULTILINE)
_CAPTION_RE = re.compile(r"^\s*(?:\*\*)?(?:Figure|Fig\.|Table|Algorithm)\b", flags=re.IGNORECASE)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")
_CITATION_RE = re.compile(r"\[(\d{1,4}(?:\s*-\s*\d{1,4})?(?:\s*,\s*\d{1,4}(?:\s*-\s*\d{1,4})?)*)\]")


@dataclass(frozen=True)
class MarkdownQualitySummary:
    chars: int
    lines: int
    nonempty_lines: int
    heading_count: int
    h1_count: int
    h2_count: int
    h3_plus_count: int
    has_abstract_heading: bool
    image_count: int
    caption_count: int
    table_block_count: int
    display_math_block_count: int
    inline_math_count: int
    references_heading_count: int
    reference_line_count: int
    max_reference_index: int
    body_citation_marker_count: int
    body_citation_range_count: int
    body_citation_expanded_index_count: int
    analyzer_issue_count: int
    analyzer_error_count: int
    analyzer_warning_count: int
    analyzer_info_count: int
    analyzer_category_counts: dict[str, int]


def _count_display_math_blocks(md_text: str) -> int:
    lines = md_text.splitlines()
    in_block = False
    count = 0
    for raw in lines:
        stripped = (raw or "").strip()
        if not stripped:
            continue
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            count += 1
            continue
        if stripped == "$$":
            if in_block:
                count += 1
                in_block = False
            else:
                in_block = True
    return count


def _strip_display_math_blocks(md_text: str) -> str:
    lines = md_text.splitlines()
    out: list[str] = []
    in_block = False
    for raw in lines:
        stripped = (raw or "").strip()
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            out.append("")
            continue
        if stripped == "$$":
            in_block = not in_block
            out.append("")
            continue
        if in_block:
            out.append("")
            continue
        out.append(raw)
    return "\n".join(out)


def _count_table_blocks(md_text: str) -> int:
    count = 0
    run = 0
    for raw in md_text.splitlines():
        stripped = (raw or "").strip()
        is_table_row = stripped.startswith("|") and stripped.count("|") >= 2 and (not stripped.startswith("```"))
        if is_table_row:
            run += 1
            continue
        if run >= 2:
            count += 1
        run = 0
    if run >= 2:
        count += 1
    return count


def _find_reference_section_start(lines: list[str]) -> int:
    for idx, raw in enumerate(lines):
        if _REF_HEADING_RE.match((raw or "").strip()):
            return idx
    return -1


def _expand_citation_token(token: str) -> list[int]:
    out: list[int] = []
    for part in re.split(r"\s*,\s*", str(token or "").strip()):
        if not part:
            continue
        m = re.match(r"^(\d{1,4})\s*-\s*(\d{1,4})$", part)
        if m:
            lo = int(m.group(1))
            hi = int(m.group(2))
            if lo <= hi and (hi - lo) <= 200:
                out.extend(range(lo, hi + 1))
            continue
        if re.match(r"^\d{1,4}$", part):
            out.append(int(part))
    return out


def summarize_markdown_quality(md_text: str) -> MarkdownQualitySummary:
    text = str(md_text or "")
    lines = text.splitlines()
    body_ref_idx = _find_reference_section_start(lines)
    if body_ref_idx >= 0:
        body_text = "\n".join(lines[:body_ref_idx])
        ref_lines = lines[body_ref_idx + 1 :]
    else:
        body_text = text
        ref_lines = []

    headings = list(_HEADING_RE.finditer(text))
    h1_count = sum(1 for m in headings if len(m.group(1)) == 1)
    h2_count = sum(1 for m in headings if len(m.group(1)) == 2)
    h3_plus_count = sum(1 for m in headings if len(m.group(1)) >= 3)

    body_citation_markers = list(_CITATION_RE.finditer(body_text))
    expanded_indices: set[int] = set()
    body_citation_range_count = 0
    for match in body_citation_markers:
        token = str(match.group(1) or "")
        if "-" in token:
            body_citation_range_count += 1
        expanded_indices.update(_expand_citation_token(token))

    reference_numbers = [int(m.group(1)) for m in _REF_LINE_RE.finditer("\n".join(ref_lines))]
    display_math_blocks = _count_display_math_blocks(text)
    inline_math = _INLINE_MATH_RE.findall(_strip_display_math_blocks(text))

    analyzer = MarkdownAnalyzer()
    issues = analyzer.analyze(text)
    category_counts: dict[str, int] = {}
    error_count = 0
    warning_count = 0
    info_count = 0
    for issue in issues:
        category = str(issue.category or "unknown")
        category_counts[category] = int(category_counts.get(category, 0)) + 1
        sev = str(issue.severity or "").lower()
        if sev == "error":
            error_count += 1
        elif sev == "warning":
            warning_count += 1
        else:
            info_count += 1

    return MarkdownQualitySummary(
        chars=len(text),
        lines=len(lines),
        nonempty_lines=sum(1 for line in lines if (line or "").strip()),
        heading_count=len(headings),
        h1_count=h1_count,
        h2_count=h2_count,
        h3_plus_count=h3_plus_count,
        has_abstract_heading=bool(re.search(r"^#{1,6}\s+Abstract\s*$", text, flags=re.IGNORECASE | re.MULTILINE)),
        image_count=len(_IMAGE_RE.findall(text)),
        caption_count=sum(1 for raw in lines if _CAPTION_RE.match((raw or "").strip())),
        table_block_count=_count_table_blocks(text),
        display_math_block_count=display_math_blocks,
        inline_math_count=len(inline_math),
        references_heading_count=len(_REF_HEADING_RE.findall(text)),
        reference_line_count=len(reference_numbers),
        max_reference_index=max(reference_numbers) if reference_numbers else 0,
        body_citation_marker_count=len(body_citation_markers),
        body_citation_range_count=body_citation_range_count,
        body_citation_expanded_index_count=len(expanded_indices),
        analyzer_issue_count=len(issues),
        analyzer_error_count=error_count,
        analyzer_warning_count=warning_count,
        analyzer_info_count=info_count,
        analyzer_category_counts=category_counts,
    )


def compare_markdown_quality(base_text: str, candidate_text: str) -> dict[str, Any]:
    base = summarize_markdown_quality(base_text)
    candidate = summarize_markdown_quality(candidate_text)

    def _delta(name: str) -> int:
        return int(getattr(candidate, name)) - int(getattr(base, name))

    regression_flags = {
        "missing_abstract_heading": bool(base.has_abstract_heading and (not candidate.has_abstract_heading)),
        "reference_lines_dropped": bool(candidate.reference_line_count + 2 < base.reference_line_count),
        "reference_index_regressed": bool(candidate.max_reference_index + 1 < base.max_reference_index),
        "tables_dropped": bool(candidate.table_block_count < base.table_block_count),
        "images_dropped": bool(candidate.image_count < base.image_count),
        "display_math_dropped": bool(candidate.display_math_block_count < base.display_math_block_count),
        "analyzer_errors_increased": bool(candidate.analyzer_error_count > base.analyzer_error_count),
        "analyzer_warnings_increased": bool(candidate.analyzer_warning_count > base.analyzer_warning_count),
    }

    return {
        "similarity_ratio": round(difflib.SequenceMatcher(None, base_text, candidate_text).ratio(), 6),
        "exact_match": bool(base_text == candidate_text),
        "base": asdict(base),
        "candidate": asdict(candidate),
        "delta": {
            "chars": _delta("chars"),
            "lines": _delta("lines"),
            "heading_count": _delta("heading_count"),
            "image_count": _delta("image_count"),
            "caption_count": _delta("caption_count"),
            "table_block_count": _delta("table_block_count"),
            "display_math_block_count": _delta("display_math_block_count"),
            "inline_math_count": _delta("inline_math_count"),
            "reference_line_count": _delta("reference_line_count"),
            "max_reference_index": _delta("max_reference_index"),
            "body_citation_marker_count": _delta("body_citation_marker_count"),
            "body_citation_expanded_index_count": _delta("body_citation_expanded_index_count"),
            "analyzer_issue_count": _delta("analyzer_issue_count"),
            "analyzer_error_count": _delta("analyzer_error_count"),
            "analyzer_warning_count": _delta("analyzer_warning_count"),
        },
        "regression_flags": regression_flags,
    }


def render_quality_comparison_report(
    *,
    base_label: str,
    candidate_label: str,
    comparison: dict[str, Any],
) -> str:
    base = comparison.get("base") or {}
    candidate = comparison.get("candidate") or {}
    delta = comparison.get("delta") or {}
    flags = comparison.get("regression_flags") or {}

    lines = [
        "# Markdown Quality Comparison",
        "",
        f"- Baseline: {base_label}",
        f"- Candidate: {candidate_label}",
        f"- Similarity: {comparison.get('similarity_ratio')}",
        f"- Exact match: {comparison.get('exact_match')}",
        "",
        "## Key Metrics",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    metrics = [
        "chars",
        "lines",
        "heading_count",
        "image_count",
        "caption_count",
        "table_block_count",
        "display_math_block_count",
        "inline_math_count",
        "reference_line_count",
        "max_reference_index",
        "body_citation_marker_count",
        "body_citation_expanded_index_count",
        "analyzer_issue_count",
        "analyzer_error_count",
        "analyzer_warning_count",
    ]
    for key in metrics:
        lines.append(
            f"| {key} | {base.get(key, 0)} | {candidate.get(key, 0)} | {delta.get(key, 0)} |"
        )

    lines.extend(
        [
            "",
            "## Boolean Checks",
            "",
            f"- baseline has abstract heading: {base.get('has_abstract_heading')}",
            f"- candidate has abstract heading: {candidate.get('has_abstract_heading')}",
            "",
            "## Regression Flags",
            "",
        ]
    )
    for key, value in flags.items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Analyzer Categories",
            "",
            f"- baseline: {json.dumps(base.get('analyzer_category_counts') or {}, ensure_ascii=False, sort_keys=True)}",
            f"- candidate: {json.dumps(candidate.get('analyzer_category_counts') or {}, ensure_ascii=False, sort_keys=True)}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def compare_markdown_files(*, base_path: Path, candidate_path: Path) -> dict[str, Any]:
    base_text = Path(base_path).read_text(encoding="utf-8")
    candidate_text = Path(candidate_path).read_text(encoding="utf-8")
    comparison = compare_markdown_quality(base_text, candidate_text)
    comparison["base_path"] = str(Path(base_path))
    comparison["candidate_path"] = str(Path(candidate_path))
    comparison["report_markdown"] = render_quality_comparison_report(
        base_label=str(Path(base_path).name),
        candidate_label=str(Path(candidate_path).name),
        comparison=comparison,
    )
    return comparison
