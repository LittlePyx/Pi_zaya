from __future__ import annotations

import re
import threading
from collections import Counter
from typing import Optional, List
from pathlib import Path

try:
    import fitz
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from .text_utils import _normalize_text
from .geometry_utils import _bbox_width, _rect_area, _rect_intersection_area, _overlap_1d


# PyMuPDF's table finder uses shared native state and can fail with
# ``not a textpage of this page`` when invoked concurrently, even when each
# worker owns a separate document. Keep only the native finder call serialized;
# the rest of page preparation and table rendering remains parallel.
_PYMUPDF_TABLE_FINDER_LOCK = threading.Lock()


def _ensure_pdfplumber_module():
    global pdfplumber
    if pdfplumber is not None:
        return pdfplumber
    try:
        import pdfplumber as _pdfplumber
    except Exception as e:
        raise RuntimeError("`pdfplumber` package is not available.") from e
    pdfplumber = _pdfplumber
    return pdfplumber


def _escape_md_table_cell(value: str) -> str:
    cell = _normalize_text(value or "")
    if not cell:
        return ""
    cell = cell.replace("\r", "\n")
    cell = re.sub(r"\s*\n\s*", "<br>", cell)
    cell = cell.replace("|", r"\|")
    return cell.strip()


def _table_rows_to_markdown(rows_raw) -> Optional[str]:
    if not rows_raw or not isinstance(rows_raw, list):
        return None

    rows: list[list[str]] = []
    for row in rows_raw:
        if not isinstance(row, (list, tuple)):
            continue
        cells = [_escape_md_table_cell("" if c is None else str(c)) for c in row]
        rows.append(cells)

    # Drop empty rows.
    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 2:
        return None

    width = max(len(r) for r in rows)
    if width < 2:
        return None
    rows = [r + [""] * (width - len(r)) for r in rows]

    # Drop columns that are empty across all rows.
    keep_cols = [i for i in range(width) if any(rows[r][i].strip() for r in range(len(rows)))]
    if len(keep_cols) < 2:
        return None
    rows = [[r[i] for i in keep_cols] for r in rows]
    width = len(rows[0])

    # Some detectors prepend an almost-empty row; skip it if it looks like noise.
    if len(rows) >= 3:
        first_non_empty = sum(1 for c in rows[0] if c.strip())
        second_non_empty = sum(1 for c in rows[1] if c.strip())
        if first_non_empty <= 1 and second_non_empty >= 2:
            rows = rows[1:]

    if len(rows) < 2:
        return None

    header = rows[0]
    if not any(c.strip() for c in header):
        header = [f"col_{i + 1}" for i in range(width)]

    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    return normalize_markdown_table_block("\n".join(md_lines))


def _markdown_table_quality_score(md: str) -> float:
    lines = [ln.strip() for ln in (md or "").splitlines() if ln.strip()]
    if len(lines) < 3:
        return 0.0
    width = max(0, lines[0].count("|") - 1)
    body_rows = max(0, len(lines) - 2)
    non_empty_cells = 0
    for ln in lines:
        parts = [p.strip() for p in ln.strip("|").split("|")]
        non_empty_cells += sum(1 for p in parts if p and p != "---")
    return float(width * body_rows) + float(non_empty_cells) * 0.08


def _is_markdown_table_sane(md: str) -> bool:
    lines = [ln.strip() for ln in (md or "").splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    width = max(0, lines[0].count("|") - 1)
    if width < 2 or width > 32:
        return False
    cells: list[str] = []
    cols: list[list[str]] = [[] for _ in range(width)]
    numeric_cells = 0
    row_non_empty: list[int] = []
    filled_slots = 0
    total_slots = 0
    for ln in lines[2:]:
        parts = [p.strip() for p in ln.strip("|").split("|")]
        if len(parts) != width:
            return False
        non_empty_in_row = 0
        for ci, p in enumerate(parts):
            if p:
                non_empty_in_row += 1
                cols[ci].append(p)
                cells.append(p)
                if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|e[+-]?\d+)?", p, flags=re.IGNORECASE) or len(
                    re.findall(r"(?<![A-Za-z0-9])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|e[+-]?\d+)?", p, flags=re.IGNORECASE)
                ) >= 2:
                    numeric_cells += 1
        row_non_empty.append(non_empty_in_row)
        filled_slots += non_empty_in_row
        total_slots += width
    if not cells:
        return False
    if not row_non_empty:
        return False
    # Reject highly sparse grids (common false positives from figure axis text).
    fill_ratio = filled_slots / max(1, total_slots)
    if fill_ratio < 0.36:
        return False
    sparse_rows = sum(1 for n in row_non_empty if n <= max(1, width // 3))
    if sparse_rows / max(1, len(row_non_empty)) > 0.34:
        return False
    if numeric_cells == 0 and len(lines) >= 5:
        return False
    tiny_ratio = sum(1 for c in cells if len(c) <= 1) / max(1, len(cells))
    if tiny_ratio > 0.35:
        return False
    tiny_alpha_ratio = sum(1 for c in cells if re.fullmatch(r"[A-Za-z]{1,2}", c)) / max(1, len(cells))
    if tiny_alpha_ratio > 0.22:
        return False
    if len(lines) > 12 and tiny_alpha_ratio > 0.10:
        return False
    if width >= 5 and tiny_alpha_ratio > 0.14:
        return False
    for col_cells in cols:
        if len(col_cells) < 3:
            continue
        tiny_col = sum(1 for c in col_cells if len(c) <= 1) / max(1, len(col_cells))
        if tiny_col > 0.78:
            return False
        tiny_alpha_col = sum(1 for c in col_cells if re.fullmatch(r"[A-Za-z]{1,2}", c)) / max(1, len(col_cells))
        if tiny_alpha_col > 0.58:
            return False
    long_phrase_ratio = sum(1 for c in cells if len(re.findall(r"[A-Za-z]{2,}", c)) >= 8) / max(1, len(cells))
    if long_phrase_ratio > 0.55:
        return False
    if long_phrase_ratio > 0.38 and (numeric_cells / max(1, len(cells))) < 0.12:
        return False
    return True


def _split_md_table_cells(line: str) -> list[str]:
    text = (line or "").strip()
    if not text.startswith("|"):
        return []
    inner = text.strip("|")
    parts = re.split(r"(?<!\\)\|", inner)
    return [p.strip() for p in parts]


def _looks_separator_cell(cell: str) -> bool:
    return bool(re.fullmatch(r":?-{3,}:?", (cell or "").strip()))


def _looks_numeric_table_cell(cell: str) -> bool:
    t = (cell or "").strip()
    if not t:
        return False
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|e[+-]?\d+)?", t, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"\d+\s*/\s*\d+", t):
        return True
    return False


def _looks_value_header_cell(cell: str) -> bool:
    t = (cell or "").strip()
    if not t:
        return False
    if re.fullmatch(r"(?:\d+(?:\.\d+)?)%?", t):
        return True
    if re.fullmatch(r"\d+\s*/\s*\d+", t):
        return True
    return False


def _expand_multicolumn_cells(cells: list[str]) -> list[str]:
    out: list[str] = []
    for cell in cells:
        t = (cell or "").strip()
        m = re.fullmatch(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}", t)
        if not m:
            out.append(t)
            continue
        span = max(1, int(m.group(1)))
        label = _normalize_text(m.group(2) or "").strip()
        out.append(label)
        out.extend([""] * (span - 1))
    return out


def _table_numeric_suffix_width(cells: list[str]) -> int:
    width = 0
    for cell in reversed(cells):
        if _looks_numeric_table_cell(cell):
            width += 1
            continue
        break
    return width


def _table_value_header_suffix_width(cells: list[str]) -> int:
    width = 0
    for cell in reversed(cells):
        if _looks_value_header_cell(cell):
            width += 1
            continue
        break
    return width


def _normalize_sparse_prefix(cells: list[str], *, target_width: int, suffix_width: int) -> list[str]:
    if len(cells) >= target_width or suffix_width <= 0 or suffix_width >= len(cells):
        return cells + [""] * max(0, target_width - len(cells))
    prefix = cells[:-suffix_width]
    suffix = cells[-suffix_width:]
    target_prefix = max(0, target_width - suffix_width)
    if len(prefix) < target_prefix:
        head = prefix[:1]
        rest = prefix[1:]
        prefix = head + ([""] * (target_prefix - len(prefix))) + rest
    return (prefix + suffix + [""] * target_width)[:target_width]


def _normalize_sparse_header_row(
    cells: list[str],
    *,
    target_width: int,
    value_header_start: int | None,
) -> list[str]:
    cells = [(c or "").strip() for c in cells]
    if len(cells) >= target_width:
        return cells[:target_width]
    if value_header_start is None:
        return cells + [""] * (target_width - len(cells))

    labels = [c for c in cells if c]
    if not labels:
        return [""] * target_width

    span_label = next(
        (lab for lab in labels if re.search(r"(?i)(sampling|ratio|measurement|frequency)", lab)),
        None,
    )
    base = [""] * target_width
    if span_label:
        others = labels.copy()
        others.remove(span_label)
        if others:
            start = max(0, value_header_start - len(others))
            for i, lab in enumerate(others):
                if start + i < target_width:
                    base[start + i] = lab
        if value_header_start < target_width:
            base[value_header_start] = span_label
        return base

    if len(cells) == target_width - 1:
        head = cells[:1]
        rest = cells[1:]
        return (head + [""] + rest + [""] * target_width)[:target_width]

    return cells + [""] * (target_width - len(cells))


def _render_markdown_table(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    norm_rows = [r + [""] * (width - len(r)) for r in rows]
    md_lines = [
        "| " + " | ".join(norm_rows[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in norm_rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(md_lines)


_HTML_TABLE_BREAK_RE = re.compile(r"\s*<br\s*/?>\s*", flags=re.IGNORECASE)
_TABLE_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|e[+-]?\d+)?",
    flags=re.IGNORECASE,
)
_TABLE_VALUE_TOKEN_RE = re.compile(
    r"(?:\*{1,2})?[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|e[+-]?\d+)?(?:\*{1,2})?",
    flags=re.IGNORECASE,
)
_KNOWN_METRIC_LABEL_RE = re.compile(
    r"(?:PSNR|SSIM|LPIPS|RMSE|MSE|MAE|FID|F1|AP|AR|IOU|AUC|ACCURACY|PRECISION|RECALL)[↑↓]?",
    flags=re.IGNORECASE,
)
_TABLE_DEDUPE_WORD_STOP = {
    "and", "col", "column", "down", "lpips", "method", "metric", "ours", "psnr",
    "result", "results", "score", "ssim", "table", "the", "up", "value", "values",
}


def _split_html_table_breaks(cell: str) -> list[str]:
    text = str(cell or "").strip()
    if not _HTML_TABLE_BREAK_RE.search(text):
        return [text]
    parts = [part.strip() for part in _HTML_TABLE_BREAK_RE.split(text)]
    while parts and not parts[0]:
        parts.pop(0)
    while parts and not parts[-1]:
        parts.pop()
    return parts or [""]


def _collapsed_table_row_segment_count(row: list[str]) -> int:
    split_cells = [_split_html_table_breaks(cell) for cell in row]
    multi_counts = [len(parts) for parts in split_cells if len(parts) >= 2]
    if not multi_counts:
        return 0
    dominant_count, dominant_cells = Counter(multi_counts).most_common(1)[0]
    non_empty_cells = sum(1 for cell in row if str(cell or "").strip())
    if dominant_count < 2 or dominant_cells < max(2, (non_empty_cells + 1) // 2):
        return 0
    if any(count != dominant_count for count in multi_counts):
        return 0

    first_cell_aligned = bool(split_cells and len(split_cells[0]) == dominant_count)
    numeric_aligned_columns = sum(
        1
        for parts in split_cells[1:]
        if len(parts) == dominant_count and all(_TABLE_NUMBER_RE.search(part or "") for part in parts)
    )
    if numeric_aligned_columns <= 0 and not (first_cell_aligned and dominant_cells >= 3):
        return 0
    return dominant_count


def _ambiguous_table_break_row(row: list[str]) -> bool:
    if _collapsed_table_row_segment_count(row) >= 2:
        return False
    split_cells = [_split_html_table_breaks(cell) for cell in row]
    first_count = len(split_cells[0]) if split_cells else 1
    numeric_counts = [
        len(parts)
        for parts in split_cells[1:]
        if len(parts) >= 2 and all(_TABLE_NUMBER_RE.search(part or "") for part in parts)
    ]
    if len(set(numeric_counts)) > 1:
        return True
    return first_count >= 2 and any(count != first_count for count in numeric_counts)


def markdown_table_block_has_ambiguous_breaks(md: str) -> bool:
    lines = [line for line in str(md or "").splitlines() if line.strip()]
    rows = [
        _split_md_table_cells(line)
        for line in lines
        if line.lstrip().startswith("|")
        and not all(_looks_separator_cell(cell) or not cell for cell in _split_md_table_cells(line))
    ]
    return any(_ambiguous_table_break_row(row) for row in rows[1:])


_FRAGMENTED_HEADER_SAFE_TOKENS = {
    "accuracy", "ap", "ar", "auc", "batch", "depth", "dice", "epe", "epoch",
    "f1", "fid", "fps", "gopro", "iou", "lpips", "mae", "method", "mse",
    "nmse", "precision", "psnr", "recall", "rmse", "sam", "sidd", "size",
    "ssim", "time", "wer",
}
_FRAGMENTED_HEADER_STEMS = {"ate", "latenc", "mpl", "ncy", "ness"}


def _fragmented_header_evidence(header: list[str]) -> bool:
    suspicious = 0
    explicit_hyphen_fragment = False
    for cell in header[2:]:
        raw = str(cell or "").strip()
        if not raw or re.search(r"\d", raw):
            continue
        clean = re.sub(r"[^A-Za-z-]+", "", raw).strip()
        token = clean.strip("-").lower()
        if not token or token in _FRAGMENTED_HEADER_SAFE_TOKENS:
            continue
        hyphen_fragment = bool(re.fullmatch(r"[A-Za-z]{2,6}-", clean))
        stem_fragment = token in _FRAGMENTED_HEADER_STEMS
        if not (hyphen_fragment or stem_fragment):
            continue
        suspicious += 1
        explicit_hyphen_fragment = explicit_hyphen_fragment or hyphen_fragment
    return suspicious >= 3 or (explicit_hyphen_fragment and suspicious >= 2)


def _fragmented_table_column_score(rows: list[list[str]]) -> int:
    if len(rows) < 5:
        return 0
    width = max((len(row) for row in rows), default=0)
    if width < 8:
        return 0

    header = rows[0] + ([""] * max(0, width - len(rows[0])))
    if not _fragmented_header_evidence(header):
        return 0

    split_rows = 0
    for row in rows[1:]:
        clean_cells = [re.sub(r"[*_`]", "", str(cell or "")).strip() for cell in row]
        has_split_decimal = any(
            re.fullmatch(r"\d{1,3}", left or "")
            and re.match(r"^\.\d{1,3}(?:\s|$)", right or "")
            for left, right in zip(clean_cells, clean_cells[1:])
        )
        if has_split_decimal:
            split_rows += 1
    return split_rows if split_rows >= 3 else 0


def markdown_table_block_is_fragmented(md: str) -> bool:
    lines = [line for line in str(md or "").splitlines() if line.strip()]
    if len(lines) < 2 or not all(line.lstrip().startswith("|") for line in lines):
        return False
    rows = [
        _split_md_table_cells(line)
        for line in lines
        if not all(_looks_separator_cell(cell) or not cell for cell in _split_md_table_cells(line))
    ]
    return _fragmented_table_column_score(rows) > 0


def _expand_collapsed_table_rows(rows: list[list[str]]) -> list[list[str]]:
    if len(rows) < 2:
        return rows
    expanded: list[list[str]] = [list(rows[0])]
    for row in rows[1:]:
        segment_count = _collapsed_table_row_segment_count(row)
        if segment_count <= 0:
            expanded.append(list(row))
            continue
        split_cells = [_split_html_table_breaks(cell) for cell in row]
        for segment_index in range(segment_count):
            split_row: list[str] = []
            for parts in split_cells:
                if len(parts) == segment_count:
                    split_row.append(parts[segment_index])
                elif segment_index == 0:
                    split_row.append(parts[0] if parts else "")
                else:
                    split_row.append("")
            expanded.append(split_row)
    return expanded


def _metric_header_parts(cell: str) -> tuple[str, list[str]] | None:
    parts = _split_html_table_breaks(cell)
    if len(parts) < 2 or not parts[0]:
        return None
    metric_text = " ".join(parts[1:]).strip()
    labels = [match.group(0) for match in _KNOWN_METRIC_LABEL_RE.finditer(metric_text)]
    if len(labels) < 2:
        labels = [token for token in re.split(r"\s+", metric_text) if token]
    if len(labels) < 2 or len(labels) > 8:
        return None
    return parts[0], labels


def _metric_value_parts(cell: str, *, width: int) -> list[str] | None:
    text = str(cell or "").strip()
    if not text:
        return [""] * width
    values = [match.group(0) for match in _TABLE_VALUE_TOKEN_RE.finditer(text)]
    leftover = _TABLE_VALUE_TOKEN_RE.sub(" ", text)
    leftover = re.sub(r"[\s,;/]+", "", leftover)
    if leftover or len(values) != width:
        return None
    return values


def _expand_compact_metric_columns(rows: list[list[str]]) -> list[list[str]]:
    if len(rows) < 2 or len(rows[0]) < 2:
        return rows
    header = rows[0]
    parsed_headers = [_metric_header_parts(cell) for cell in header[1:]]
    if not parsed_headers or any(item is None for item in parsed_headers):
        return rows
    metric_widths = [len(item[1]) for item in parsed_headers if item is not None]
    if not metric_widths or len(set(metric_widths)) != 1:
        return rows
    metric_width = metric_widths[0]

    expanded_data: list[list[str]] = []
    for row in rows[1:]:
        if len(row) != len(header):
            return rows
        expanded_row = [row[0]]
        for cell in row[1:]:
            values = _metric_value_parts(cell, width=metric_width)
            if values is None:
                return rows
            expanded_row.extend(values)
        expanded_data.append(expanded_row)

    expanded_header = [header[0]]
    metric_header = [""]
    for parsed in parsed_headers:
        assert parsed is not None
        scene, labels = parsed
        expanded_header.extend([scene] + ([""] * (metric_width - 1)))
        metric_header.extend(labels)
    return [expanded_header, metric_header, *expanded_data]


def _flatten_html_table_breaks(cell: str) -> str:
    parts = [part for part in _split_html_table_breaks(cell) if part]
    return " · ".join(parts)


def normalize_markdown_table_block(md: str) -> str:
    lines = [ln.rstrip() for ln in (md or "").splitlines() if ln.strip()]
    if len(lines) < 2 or not all(ln.lstrip().startswith("|") for ln in lines):
        return md

    rows: list[list[str]] = []
    for line in lines:
        cells = _expand_multicolumn_cells(_split_md_table_cells(line))
        if not cells:
            return md
        if cells and all(_looks_separator_cell(c) or not c for c in cells):
            continue
        rows.append(cells)
    if len(rows) < 2:
        return md
    if any(_ambiguous_table_break_row(row) for row in rows[1:]):
        return md

    rows = _expand_collapsed_table_rows(rows)
    rows = _expand_compact_metric_columns(rows)
    rows = [[_flatten_html_table_breaks(cell) for cell in row] for row in rows]

    target_width = max(len(r) for r in rows)
    if target_width < 2:
        return md

    numeric_suffixes = [_table_numeric_suffix_width(r) for r in rows]
    value_header_suffixes = [_table_value_header_suffix_width(r) for r in rows]
    tail_candidates = [w for w in numeric_suffixes if w >= 2]
    if not tail_candidates:
        tail_candidates = [w for w in value_header_suffixes if w >= 2]
    tail_width = max(tail_candidates, key=tail_candidates.count) if tail_candidates else 0
    value_header_start = (target_width - tail_width) if tail_width >= 2 else None

    normalized: list[list[str]] = []
    for row, numeric_tail, value_tail in zip(rows, numeric_suffixes, value_header_suffixes):
        if tail_width >= 2 and numeric_tail >= 2:
            normalized.append(_normalize_sparse_prefix(row, target_width=target_width, suffix_width=numeric_tail))
            continue
        if tail_width >= 2 and value_tail >= 2:
            normalized.append(_normalize_sparse_prefix(row, target_width=target_width, suffix_width=value_tail))
            continue
        normalized.append(
            _normalize_sparse_header_row(
                row,
                target_width=target_width,
                value_header_start=value_header_start,
            )
        )

    return _render_markdown_table(normalized)


def _markdown_table_spans(lines: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, line in enumerate(lines + [""]):
        stripped = str(line or "").lstrip()
        is_table_line = stripped.startswith("|") and stripped.count("|") >= 2
        if is_table_line and start is None:
            start = index
            continue
        if is_table_line:
            continue
        if start is not None:
            spans.append((start, index))
            start = None
    return spans


def _normalize_markdown_table_blocks_document(md: str) -> str:
    text = str(md or "")
    trailing_newline = text.endswith("\n")
    lines = text.splitlines()
    spans = _markdown_table_spans(lines)
    if not spans:
        return text
    out: list[str] = []
    cursor = 0
    for start, end in spans:
        out.extend(lines[cursor:start])
        out.extend(normalize_markdown_table_block("\n".join(lines[start:end])).splitlines())
        cursor = end
    out.extend(lines[cursor:])
    fixed = "\n".join(out)
    if trailing_newline:
        fixed += "\n"
    return fixed


def _normalized_table_number(value: str) -> str:
    text = str(value or "").strip().lower().rstrip("%")
    sign = ""
    if text.startswith(("+", "-")):
        sign, text = text[0], text[1:]
    if text.startswith("0."):
        text = text[1:]
    return sign + text


def _markdown_table_signature(lines: list[str], start: int, end: int) -> dict:
    block_lines = lines[start:end]
    rows = [
        _split_md_table_cells(line)
        for line in block_lines
        if not all(_looks_separator_cell(cell) or not cell for cell in _split_md_table_cells(line))
    ]
    cells = [cell for row in rows for cell in row]
    plain = " ".join(cells)
    without_citations = re.sub(r"\[(?:\d{1,4}(?:\s*[,;\u2013-]\s*\d{1,4})*)\]", " ", plain)
    numbers = Counter(_normalized_table_number(match.group(0)) for match in _TABLE_NUMBER_RE.finditer(without_citations))
    words = {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+_-]{2,}", without_citations.lower())
        if token not in _TABLE_DEDUPE_WORD_STOP
    }
    width = max((len(row) for row in rows), default=0)
    non_empty = sum(1 for cell in cells if str(cell or "").strip())
    truncated_decimal_cells = sum(
        1
        for cell in cells
        if re.fullmatch(r"(?:\*{1,2})?[+-]?\d+\.(?:\*{1,2})?", str(cell or "").strip())
    )
    score = float(width * max(1, len(rows))) + float(non_empty) * 0.2
    return {
        "start": start,
        "end": end,
        "numbers": numbers,
        "words": words,
        "score": score,
        "truncated_decimal_cells": truncated_decimal_cells,
    }


def _nearby_duplicate_table_pairs(md: str) -> list[tuple[dict, dict]]:
    lines = str(md or "").splitlines()
    signatures = [_markdown_table_signature(lines, start, end) for start, end in _markdown_table_spans(lines)]
    pairs: list[tuple[dict, dict]] = []
    for left_index, left in enumerate(signatures):
        for right in signatures[left_index + 1:]:
            if int(right["start"]) - int(left["end"]) > 24:
                break
            left_numbers = left["numbers"]
            right_numbers = right["numbers"]
            left_total = sum(left_numbers.values())
            right_total = sum(right_numbers.values())
            if min(left_total, right_total) < 6:
                continue
            shared_numbers = sum((left_numbers & right_numbers).values())
            numeric_coverage = shared_numbers / max(1, max(left_total, right_total))
            left_words = set(left["words"])
            right_words = set(right["words"])
            shared_words = len(left_words & right_words)
            word_coverage = shared_words / max(1, min(len(left_words), len(right_words)))
            exact_duplicate = bool(
                numeric_coverage >= 0.90
                and shared_words >= 2
                and (word_coverage >= 0.50 or numeric_coverage >= 0.98)
            )
            smaller, larger = (
                (left, right)
                if float(left["score"]) <= float(right["score"])
                else (right, left)
            )
            partial_numeric_coverage = _fragment_aware_numeric_coverage(
                smaller["numbers"],
                larger["numbers"],
            )
            truncated_partial_duplicate = bool(
                int(smaller.get("truncated_decimal_cells") or 0) >= 2
                and sum(smaller["numbers"].values()) >= 6
                and float(smaller["score"]) <= float(larger["score"]) * 0.85
                and partial_numeric_coverage >= 0.85
                and shared_words >= 2
                and word_coverage >= 0.60
            )
            if not (exact_duplicate or truncated_partial_duplicate):
                continue
            pairs.append((left, right))
    return pairs


def _number_is_fragment_of_any(value: str, candidates: Counter[str]) -> bool:
    digits = re.sub(r"\D", "", str(value or ""))
    if not digits:
        return False
    for candidate in candidates:
        candidate_digits = re.sub(r"\D", "", str(candidate or ""))
        if not candidate_digits:
            continue
        shorter, longer = sorted((digits, candidate_digits), key=len)
        if shorter == longer:
            return True
        if shorter in longer and (len(shorter) >= 2 or len(longer) <= 4):
            return True
    return False


def _fragment_aware_numeric_coverage(numbers: Counter[str], candidates: Counter[str]) -> float:
    total = sum(numbers.values())
    if total <= 0:
        return 0.0
    covered = 0
    for value, count in numbers.items():
        exact = min(int(count), int(candidates.get(value, 0)))
        covered += exact
        remaining = int(count) - exact
        if remaining > 0 and _number_is_fragment_of_any(value, candidates):
            covered += remaining
    return covered / total


def _fragmented_aggregate_duplicate_ranges(md: str) -> set[tuple[int, int]]:
    lines = str(md or "").splitlines()
    spans = _markdown_table_spans(lines)
    signatures = [_markdown_table_signature(lines, start, end) for start, end in spans]
    fragmented = {
        index
        for index, (start, end) in enumerate(spans)
        if markdown_table_block_is_fragmented("\n".join(lines[start:end]))
    }
    drop_ranges: set[tuple[int, int]] = set()
    for index in sorted(fragmented):
        start, end = spans[index]
        nearby_indices = [
            candidate
            for candidate, (_, candidate_end) in enumerate(spans[:index])
            if candidate not in fragmented and start - candidate_end <= 30
        ][-4:]
        if len(nearby_indices) < 2:
            continue

        aggregate_numbers: Counter[str] = Counter()
        aggregate_words: set[str] = set()
        for candidate in nearby_indices:
            aggregate_numbers += signatures[candidate]["numbers"]
            aggregate_words.update(signatures[candidate]["words"])

        signature = signatures[index]
        shared_numbers = sum((signature["numbers"] & aggregate_numbers).values())
        numeric_coverage = _fragment_aware_numeric_coverage(signature["numbers"], aggregate_numbers)
        meaningful_words = {word for word in signature["words"] if len(word) >= 5}
        shared_words = meaningful_words & aggregate_words
        word_coverage = len(shared_words) / max(1, len(meaningful_words))
        if (
            shared_numbers < 12
            or numeric_coverage < 0.90
            or len(shared_words) < 4
            or word_coverage < 0.55
        ):
            continue
        drop_ranges.add((start, end))
    return drop_ranges


def dedupe_nearby_markdown_tables(md: str) -> str:
    text = str(md or "")
    trailing_newline = text.endswith("\n")
    lines = text.splitlines()
    drop_ranges: set[tuple[int, int]] = set(_fragmented_aggregate_duplicate_ranges(text))
    for left, right in _nearby_duplicate_table_pairs(text):
        left_range = (int(left["start"]), int(left["end"]))
        right_range = (int(right["start"]), int(right["end"]))
        if left_range in drop_ranges or right_range in drop_ranges:
            continue
        if float(left["score"]) > float(right["score"]):
            drop_ranges.add(right_range)
        else:
            drop_ranges.add(left_range)
    if not drop_ranges:
        return text
    drop_lines = {index for start, end in drop_ranges for index in range(start, end)}
    fixed = "\n".join(line for index, line in enumerate(lines) if index not in drop_lines)
    if trailing_newline:
        fixed += "\n"
    return fixed


def normalize_markdown_tables_document(md: str) -> str:
    normalized = _normalize_markdown_table_blocks_document(md)
    return dedupe_nearby_markdown_tables(normalized)


def markdown_table_issue_counts(md: str) -> dict[str, int]:
    text = str(md or "")
    lines = text.splitlines()
    literal_break_count = 0
    collapsed_row_count = 0
    ambiguous_break_row_count = 0
    fragmented_column_count = 0
    for start, end in _markdown_table_spans(lines):
        block = "\n".join(lines[start:end])
        if markdown_table_block_is_fragmented(block):
            fragmented_column_count += 1
        for line in lines[start:end]:
            literal_break_count += len(_HTML_TABLE_BREAK_RE.findall(line))
        rows = [
            _split_md_table_cells(line)
            for line in lines[start:end]
            if not all(_looks_separator_cell(cell) or not cell for cell in _split_md_table_cells(line))
        ]
        for row in rows[1:]:
            if _collapsed_table_row_segment_count(row) >= 2:
                collapsed_row_count += 1
            elif _ambiguous_table_break_row(row):
                ambiguous_break_row_count += 1
    normalized = _normalize_markdown_table_blocks_document(text)
    duplicate_count = len(_nearby_duplicate_table_pairs(normalized))
    fragmented_duplicate_count = len(_fragmented_aggregate_duplicate_ranges(text))
    return {
        "literal_break_count": literal_break_count,
        "collapsed_row_count": collapsed_row_count,
        "ambiguous_break_row_count": ambiguous_break_row_count,
        "duplicate_table_count": duplicate_count,
        "fragmented_column_count": fragmented_column_count,
        "fragmented_duplicate_count": fragmented_duplicate_count,
    }


def _extract_tables_by_pdfplumber(pdf_path: Optional[Path], page_index: int) -> list[tuple["fitz.Rect", str]]:
    if fitz is None or (pdf_path is None):
        return []
    try:
        pdm = _ensure_pdfplumber_module()
    except Exception:
        return []
    out: list[tuple[fitz.Rect, str]] = []
    try:
        with pdm.open(str(pdf_path)) as pd:
            if page_index < 0 or page_index >= len(pd.pages):
                return []
            pg = pd.pages[page_index]
            tables = pg.find_tables(
                table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "intersection_tolerance": 3,
                    "min_words_vertical": 2,
                    "min_words_horizontal": 1,
                }
            )
            for tb in tables:
                try:
                    bbox = tuple(float(x) for x in tb.bbox)
                    rect = fitz.Rect(bbox)
                except Exception:
                    continue
                try:
                    rows = tb.extract()
                except Exception:
                    rows = None
                md = _table_rows_to_markdown(rows) if rows is not None else None
                if not md:
                    continue
                if not _is_markdown_table_sane(md):
                    continue
                out.append((rect, md))
    except Exception:
        return []
    return out


def _page_maybe_has_table_from_dict(page_dict: dict) -> bool:
    """
    Fast page-level gate to avoid expensive table finder calls on obvious non-table pages.
    """
    blocks = page_dict.get("blocks", []) if isinstance(page_dict, dict) else []
    if not blocks:
        return False
    numeric_rows = 0
    delimiter_rows = 0
    scanned = 0
    for b in blocks:
        if "lines" not in b:
            continue
        for l in (b.get("lines", []) or []):
            spans = l.get("spans", []) or []
            if not spans:
                continue
            text = _normalize_text("".join(str(s.get("text", "")) for s in spans))
            if not text:
                continue
            scanned += 1
            if re.match(r"^\s*Table\s+(?:\d+|[IVXLC]+)\b", text, flags=re.IGNORECASE):
                return True
            if len(text) > 180:
                if scanned >= 260:
                    break
                continue
            cols = [c for c in re.split(r"\t+|\s{2,}", text.strip()) if c.strip()]
            nums = re.findall(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|e[+-]?\d+)?", text, flags=re.IGNORECASE)
            has_delim = ("|" in text) or (len(cols) >= 3)
            if has_delim:
                delimiter_rows += 1
            if len(nums) >= 2 and (has_delim or len(cols) >= 2):
                numeric_rows += 1

            if numeric_rows >= 3 and delimiter_rows >= 2:
                return True
            if delimiter_rows >= 6 and numeric_rows >= 2:
                return True
            if scanned >= 260:
                break
        if scanned >= 260:
            break
    return False

def table_text_to_markdown(text: str) -> Optional[str]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    def split_cols(s: str) -> list[str]:
        return [c.strip() for c in re.split(r"\t+|\s{2,}", s.strip()) if c.strip()]

    def rows_to_md(rows: list[list[str]]) -> Optional[str]:
        rows = [r for r in rows if r and any(c.strip() for c in r)]
        if len(rows) < 2:
            return None
        width = max(len(r) for r in rows)
        if width < 2:
            return None
        rows = [r + [""] * (width - len(r)) for r in rows]
        # Drop columns empty across all rows.
        keep = [i for i in range(width) if any(rows[r][i].strip() for r in range(len(rows)))]
        if len(keep) < 2:
            return None
        rows = [[_escape_md_table_cell(r[i]) for i in keep] for r in rows]
        width = len(rows[0])
        header = rows[0]
        if not any(c.strip() for c in header):
            header = [f"col_{i + 1}" for i in range(width)]
        md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
        for r in rows[1:]:
            md_lines.append("| " + " | ".join(r) + " |")
        return "\n".join(md_lines)

    rows = [split_cols(ln) for ln in lines]
    rows = [r for r in rows if r]
    md_basic = rows_to_md(rows)
    if md_basic:
        return md_basic

    # Fallback: infer column boundaries from aligned whitespace runs (common in text-extracted tables).
    norm = [ln.replace("\t", "    ") for ln in lines]
    max_len = max(len(ln) for ln in norm)
    padded = [ln.ljust(max_len) for ln in norm]
    cut_votes: list[int] = []
    for ln in padded:
        for m in re.finditer(r"\s{2,}", ln):
            cut_votes.append(int((m.start() + m.end()) / 2))
    if not cut_votes:
        return None
    cut_votes.sort()
    clusters: list[list[int]] = []
    for pos in cut_votes:
        if not clusters or abs(pos - clusters[-1][-1]) > 2:
            clusters.append([pos])
        else:
            clusters[-1].append(pos)
    cuts = [int(round(sum(g) / len(g))) for g in clusters if len(g) >= max(2, len(padded) // 5)]
    cuts = sorted({c for c in cuts if 2 <= c <= max_len - 2})
    if not cuts:
        return None
    rows_aligned: list[list[str]] = []
    for ln in padded:
        start = 0
        row: list[str] = []
        for c in cuts + [max_len]:
            cell = ln[start:c].strip()
            row.append(cell)
            start = c
        if sum(1 for cell in row if cell.strip()) >= 2:
            rows_aligned.append(row)
    return rows_to_md(rows_aligned)


def _table_from_numeric_pattern(lines: list[str]) -> Optional[str]:
    """
    Recover simple text tables like:
      Method PSNR SSIM LPIPS
      Ours 33.8 0.95 0.08
    where spacing is collapsed and first column may contain words.
    """
    if len(lines) < 3:
        return None
    num_re = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|e[+-]?\d+)?$", re.IGNORECASE)
    token_rows = [re.findall(r"\S+", ln.strip()) for ln in lines if ln.strip()]
    if len(token_rows) < 3:
        return None

    num_counts = [sum(1 for t in row if num_re.fullmatch(t)) for row in token_rows]
    data_counts = [c for c in num_counts if c >= 2]
    if len(data_counts) < 2:
        return None
    # Robust central tendency without importing statistics.
    data_counts.sort()
    n_num = data_counts[len(data_counts) // 2]
    if n_num < 2:
        return None

    rows: list[list[str]] = []
    for row in token_rows:
        if len(row) < n_num + 1:
            continue
        first_num_idx = next((i for i, t in enumerate(row) if num_re.fullmatch(t)), None)
        if first_num_idx is None:
            # header row: use last n_num tokens as metric columns
            label = " ".join(row[: len(row) - n_num]).strip()
            vals = row[len(row) - n_num :]
            if (not label) or len(vals) != n_num:
                continue
            rows.append([label] + vals)
            continue
        nums = [t for t in row[first_num_idx:] if num_re.fullmatch(t)]
        if len(nums) < n_num:
            continue
        label = " ".join(row[:first_num_idx]).strip()
        if not label:
            continue
        rows.append([label] + nums[:n_num])

    if len(rows) < 3:
        return None
    width = 1 + n_num
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = [_escape_md_table_cell(c) for c in rows[0]]
    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        md_lines.append("| " + " | ".join(_escape_md_table_cell(c) for c in r) + " |")
    out = "\n".join(md_lines)
    if not _is_markdown_table_sane(out):
        return None
    return out

def _extract_tables_by_layout(
    page,
    *,
    pdf_path: Optional[Path] = None,
    page_index: int = 0,
    visual_rects: Optional[list["fitz.Rect"]] = None,
    use_pdfplumber_fallback: bool = False,
    page_dict: dict | None = None,
) -> list[tuple["fitz.Rect", str]]:
    if fitz is None:
        return []

    page_w = float(page.rect.width)
    page_h = float(page.rect.height)
    page_area = max(1.0, page_w * page_h)
    vis_rects = visual_rects or []
    has_table_hint = getattr(page, "has_table_hint", False)

    # Heuristic: detect captions like "Table 1:" to anchor the table search.
    caption_rects: list[fitz.Rect] = []
    try:
        d = page_dict if page_dict is not None else page.get_text("dict")
        for b in d.get("blocks", []):
            if "lines" not in b:
                continue
            bbox = b.get("bbox")
            lines: list[str] = []
            for l in b.get("lines", []):
                spans = l.get("spans", [])
                if not spans:
                    continue
                line = "".join(str(s.get("text", "")) for s in spans)
                line = _normalize_text(line)
                if line:
                    lines.append(line)
            txt = _normalize_text(" ".join(lines))
            if re.match(r"^\s*Table\s+(?:\d+|[IVXLC]+)\b", txt, flags=re.IGNORECASE):
                caption_rects.append(fitz.Rect(tuple(float(x) for x in bbox)))
    except Exception:
        caption_rects = []

    def _has_nearby_table_caption(rect: "fitz.Rect") -> bool:
        if not caption_rects:
            return False
        for cr in caption_rects:
            hov = _overlap_1d(float(rect.x0), float(rect.x1), float(cr.x0) - 18.0, float(cr.x1) + 18.0)
            min_hov = max(10.0, min(float(rect.width), float(cr.width) + 36.0) * 0.15)
            if hov < min_hov:
                continue
            vgap = min(abs(float(rect.y0) - float(cr.y1)), abs(float(cr.y0) - float(rect.y1)))
            if vgap <= max(110.0, page_h * 0.24):
                return True
        return False

    primary_kwargs = [{"vertical_strategy": "lines", "horizontal_strategy": "lines"}]
    if has_table_hint:
        primary_kwargs.extend(
            [
                {"vertical_strategy": "lines", "horizontal_strategy": "text", "min_words_horizontal": 1, "text_tolerance": 2.0},
                {"vertical_strategy": "text", "horizontal_strategy": "lines", "min_words_vertical": 2, "text_tolerance": 2.0},
            ]
        )

    candidates: list[tuple[fitz.Rect, str, float]] = []
    kwargs_seq = primary_kwargs
    
    import time
    table_extract_start = time.time()
    max_table_extract_time = 8.0  # Max 8 seconds for table extraction per page
    
    for strategy_idx, kwargs in enumerate(kwargs_seq):
        if time.time() - table_extract_start > max_table_extract_time:
            print(f"      [Table extraction] Timeout after {strategy_idx}/{len(kwargs_seq)} strategies, skipping remaining", flush=True)
            break
        
        uses_text_strategy = ("text" in str(kwargs.get("vertical_strategy", "")).lower()) or (
            "text" in str(kwargs.get("horizontal_strategy", "")).lower()
        )
        strategy_start = time.time()
        try:
            with _PYMUPDF_TABLE_FINDER_LOCK:
                table_finder = page.find_tables(**kwargs)
            strategy_time = time.time() - strategy_start
            if strategy_time > 1.0:
                print(f"      [Table extraction] Strategy {strategy_idx+1} ({kwargs.get('vertical_strategy', '?')}/{kwargs.get('horizontal_strategy', '?')}): {strategy_time:.2f}s (SLOW!)", flush=True)
        except Exception as e:
            strategy_time = time.time() - strategy_start
            if strategy_time > 0.5:
                print(f"      [Table extraction] Strategy {strategy_idx+1} FAILED after {strategy_time:.2f}s: {e}", flush=True)
            continue
        tables = getattr(table_finder, "tables", table_finder)
        if not tables:
            continue

        for tb in tables:
            try:
                rect = fitz.Rect(getattr(tb, "bbox", None))
            except Exception:
                continue
            if _rect_area(rect) < page_area * 0.0035:
                continue
            if float(rect.width) < page_w * 0.12 or float(rect.height) < page_h * 0.04:
                continue
            if _rect_area(rect) > page_area * 0.55:
                continue

            md = None
            try:
                md = _table_rows_to_markdown(tb.extract())
            except Exception:
                md = None
            if not md:
                try:
                    raw_clip = page.get_text("text", clip=rect)
                except Exception:
                    raw_clip = ""
                md = table_text_to_markdown(raw_clip) if raw_clip else None
                if (not md) and raw_clip:
                    md = _table_from_numeric_pattern([ln for ln in raw_clip.splitlines() if ln.strip()])
            if not md:
                continue
            if not _is_markdown_table_sane(md):
                continue
            near_caption = _has_nearby_table_caption(rect)
            if vis_rects and (not near_caption):
                vis_overlap = max(
                    (
                        _rect_intersection_area(rect, vr) / max(1.0, min(_rect_area(rect), _rect_area(vr)))
                        for vr in vis_rects
                    ),
                    default=0.0,
                )
                # Strongly suppress figure-overlapping table false positives.
                if vis_overlap >= 0.62:
                    continue
                if vis_overlap >= 0.45 and (not has_table_hint):
                    continue
            if uses_text_strategy and (not near_caption):
                # Text-based strategies are prone to chart-axis false positives.
                # Keep only compact, denser candidates when no table caption is nearby.
                if _rect_area(rect) > page_area * 0.18:
                    continue
                if len([ln for ln in md.splitlines() if ln.strip()]) < 4:
                    continue
            if (not has_table_hint) and (not near_caption) and _rect_area(rect) > page_area * 0.08:
                continue
            score = _markdown_table_quality_score(md)
            if score <= 0.0:
                continue
            candidates.append((rect, md, score))

    if (not candidates) and use_pdfplumber_fallback and (pdf_path is not None) and has_table_hint:
        for rect, md in _extract_tables_by_pdfplumber(pdf_path, page_index):
            score = _markdown_table_quality_score(md)
            if score > 0.0:
                candidates.append((rect, md, score))

    if not candidates:
        return []

    # De-duplicate overlapping candidates, keeping the better table representation.
    uniq: list[tuple[fitz.Rect, str, float]] = []
    for rect, md, score in sorted(candidates, key=lambda x: (x[0].y0, x[0].x0, -x[2])):
        replaced = False
        for i, (r0, md0, s0) in enumerate(uniq):
            inter = _rect_intersection_area(rect, r0)
            denom = max(1.0, min(_rect_area(rect), _rect_area(r0)))
            if inter / denom >= 0.72:
                if score > s0:
                    uniq[i] = (rect, md, score)
                replaced = True
                break
        if not replaced:
            uniq.append((rect, md, score))

    uniq.sort(key=lambda x: (x[0].y0, x[0].x0))
    return [(r, md) for r, md, _ in uniq]


def _latex_array_to_markdown_table(latex: str) -> Optional[str]:
    # Accept \begin{array}{|l|c|...|} ... \end{array} or tabular-like.
    m = re.search(r"\\begin\{array\}\{[^}]*\}(.*?)\\end\{array\}", latex, flags=re.S)
    if not m:
        return None
    body = m.group(1)
    body = body.replace("\\hline", "")
    body = body.strip()
    # split rows
    raw_rows = [r.strip() for r in body.split("\\\\") if r.strip()]
    if len(raw_rows) < 2:
        return None
    rows: list[list[str]] = []
    for r in raw_rows:
        cols = [c.strip() for c in r.split("&")]
        cols = [re.sub(r"\\text\{([^}]*)\}", r"\1", c) for c in cols]
        cols = [re.sub(r"\s+", " ", c).strip() for c in cols]
        rows.append(cols)
    width = max(len(r) for r in rows)
    if width < 2:
        return None
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = rows[0]
    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        md_lines.append("| " + " | ".join(r) + " |")
    return "\n".join(md_lines)

def _convert_latex_array_math_blocks(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "$$":
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        buf: list[str] = []
        while j < len(lines) and lines[j].strip() != "$$":
            buf.append(lines[j])
            j += 1
        if j >= len(lines):
            out.append(lines[i])
            out.extend(buf)
            break
        latex = "\n".join(buf)
        table = _latex_array_to_markdown_table(latex)
        if table:
            out.extend(table.splitlines())
            out.append("")
        else:
            out.append("$$")
            out.extend(buf)
            out.append("$$")
        i = j + 1
    return "\n".join(out)
