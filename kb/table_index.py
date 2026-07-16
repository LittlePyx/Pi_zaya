from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Iterable

from .source_blocks import build_source_blocks, doc_id_for_path, normalize_inline_markdown


_TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
_CELL_BREAK_RE = re.compile(r"\s*<br\s*/?>\s*|\r?\n", flags=re.I)
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?")
_METRIC_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:LPIPS|PSNR|SSIM|RMSE|NMSE|MSE|MAE|SNR|FID|"
    r"mAP(?:@[.\d:]+)?|AP(?:50|75)?|F1|AUC|IoU|Dice|Accuracy|Precision|Recall|"
    r"FPS|FLOPs|Params?|Latency|Time)(?![A-Za-z0-9])",
    flags=re.I,
)
_DIRECTION_WORDS = {"↑": "higher is better", "↓": "lower is better"}


def _clean_cell(value: Any) -> str:
    text = html.unescape(str(value or "")).replace("\u00a0", " ")
    text = text.replace(r"\|", "|")
    return normalize_inline_markdown(text).strip()


def _cell_segments(value: Any) -> list[str]:
    segments = [_clean_cell(part) for part in _CELL_BREAK_RE.split(str(value or ""))]
    return [segment for segment in segments if segment]


def _split_markdown_row(line: str) -> list[str]:
    source = str(line or "").strip()
    if source.startswith("|"):
        source = source[1:]
    if source.endswith("|") and not source.endswith(r"\|"):
        source = source[:-1]

    cells: list[str] = []
    buf: list[str] = []
    escaped = False
    code_ticks = 0
    idx = 0
    while idx < len(source):
        char = source[idx]
        if escaped:
            buf.append(char)
            escaped = False
            idx += 1
            continue
        if char == "\\":
            buf.append(char)
            escaped = True
            idx += 1
            continue
        if char == "`":
            run = 1
            while idx + run < len(source) and source[idx + run] == "`":
                run += 1
            if code_ticks == 0:
                code_ticks = run
            elif code_ticks == run:
                code_ticks = 0
            buf.extend("`" * run)
            idx += run
            continue
        if char == "|" and code_ticks == 0:
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(char)
        idx += 1
    cells.append("".join(buf).strip())
    return cells


def _is_separator_row(cells: Iterable[str]) -> bool:
    rows = [str(cell or "").replace(" ", "") for cell in cells]
    return bool(rows) and all(bool(_TABLE_SEPARATOR_RE.fullmatch(cell)) for cell in rows)


def _expand_logical_rows(cells: list[str]) -> list[list[str]]:
    segmented = [_cell_segments(cell) for cell in cells]
    segment_count = max((len(parts) for parts in segmented), default=0)
    if segment_count <= 1:
        return [[parts[0] if parts else "" for parts in segmented]]

    first_count = len(segmented[0]) if segmented else 0
    parallel_data_cells = sum(1 for parts in segmented[1:] if len(parts) == segment_count)
    can_expand = first_count == segment_count and parallel_data_cells >= 1
    if not can_expand:
        return [[" / ".join(parts) for parts in segmented]]

    out: list[list[str]] = []
    for logical_index in range(segment_count):
        row: list[str] = []
        for parts in segmented:
            if len(parts) == segment_count:
                row.append(parts[logical_index])
            elif len(parts) == 1:
                row.append(parts[0])
            else:
                row.append(" / ".join(parts))
        out.append(row)
    return out


def _header_continuation_score(cells: list[str], next_cells: list[str]) -> int:
    cleaned = [" ".join(_cell_segments(cell)).strip() for cell in cells]
    if not cleaned or cleaned[0] or sum(1 for value in cleaned if value) < 2:
        return 0
    surface = " ".join(cleaned)
    percentage_count = len(re.findall(r"(?<!\d)\d+(?:\.\d+)?%", surface))
    metric_count = len(list(_METRIC_RE.finditer(surface)))
    if percentage_count < 2 and metric_count < 2:
        return 0
    next_cleaned = [" ".join(_cell_segments(cell)).strip() for cell in next_cells]
    if not next_cleaned or not any(next_cleaned):
        return 0
    next_numeric_count = len(list(_NUMBER_RE.finditer(" ".join(next_cleaned))))
    return percentage_count + metric_count + (1 if next_numeric_count >= 2 else 0)


def _combine_header_rows(headers: list[str], continuation: list[str]) -> list[str]:
    width = max(len(headers), len(continuation))
    main = list(headers) + [""] * max(0, width - len(headers))
    sub = list(continuation) + [""] * max(0, width - len(continuation))

    propagated: list[str] = []
    active = ""
    for value in main:
        if value:
            active = value
        propagated.append(active)

    ratio_group = next(
        (value for value in main if re.search(r"(?:sampling\s+ratio|\bCS\s+ratio\b|\bSR\b|ratio)", value, flags=re.I)),
        "",
    )
    percent_header = sum(1 for value in sub if re.fullmatch(r"\d+(?:\.\d+)?%", value)) >= 2
    out: list[str] = []
    for idx in range(width):
        parent = main[idx]
        child = sub[idx]
        if child:
            if percent_header and re.fullmatch(r"\d+(?:\.\d+)?%", child) and ratio_group:
                out.append(f"{ratio_group} {child}".strip())
            else:
                out.append(" ".join(part for part in (parent or propagated[idx], child) if part).strip())
        else:
            out.append(parent)
    return out


def parse_markdown_table(raw_table: str) -> tuple[list[str], list[list[str]]]:
    lines = [line.strip() for line in str(raw_table or "").splitlines() if line.strip()]
    parsed = [_split_markdown_row(line) for line in lines]
    if len(parsed) < 2 or not _is_separator_row(parsed[1]):
        return [], []

    width = max(len(row) for row in parsed)
    headers = [" ".join(_cell_segments(cell)).strip() for cell in parsed[0]]
    headers.extend([""] * max(0, width - len(headers)))
    data_start = 2
    if len(parsed) >= 4 and _header_continuation_score(parsed[2], parsed[3]) > 0:
        continuation = [" ".join(_cell_segments(cell)).strip() for cell in parsed[2]]
        headers = _combine_header_rows(headers[:width], continuation)
        data_start = 3

    rows: list[list[str]] = []
    for raw_row in parsed[data_start:]:
        padded = list(raw_row) + [""] * max(0, width - len(raw_row))
        for logical_row in _expand_logical_rows(padded[:width]):
            if any(str(value or "").strip() for value in logical_row):
                rows.append(logical_row)
    return headers[:width], rows


def _metric_specs(header: str) -> list[dict[str, str]]:
    text = str(header or "").strip()
    matches = list(_METRIC_RE.finditer(text))
    if not matches:
        return []
    prefix = text[: matches[0].start()].strip(" \t:/,;-_")
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in matches:
        name = str(match.group(0) or "").strip()
        suffix = text[match.end() : match.end() + 3]
        direction = next((symbol for symbol in ("↑", "↓") if symbol in suffix), "")
        label = " ".join(part for part in (prefix, name, direction) if part).strip()
        key = label.lower()
        if not label or key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "label": label, "direction": direction})
    return out


def _metric_facts(header: str, value: str) -> list[dict[str, str]]:
    specs = _metric_specs(header)
    numbers = [str(match.group(0) or "").strip() for match in _NUMBER_RE.finditer(str(value or ""))]
    if not specs or len(specs) != len(numbers):
        return []
    return [
        {
            "metric": spec["name"],
            "label": spec["label"],
            "value": number,
            "direction": spec["direction"],
        }
        for spec, number in zip(specs, numbers)
    ]


def _label_column_count(headers: list[str]) -> int:
    for index, header in enumerate(headers):
        if index <= 0:
            continue
        if _metric_specs(header):
            return index
        if re.search(r"\b(?:sampling\s+ratio|CS\s+ratio|SR)\b.*\d+(?:\.\d+)?%", header, flags=re.I):
            return index
        if re.search(r"\d+(?:\.\d+)?%", header):
            return index
    return 1


def _table_location_prefix(table: dict[str, Any]) -> str:
    number = int(table.get("table_number") or 0)
    # `table_index` is only the parser's ordinal.  It must never be presented as
    # an authored table number: converted papers can contain an uncaptioned or
    # duplicated table between Table 5 and Table 6, and calling that duplicate
    # "Table 6" makes the retrieved evidence factually misleading.
    label = f"Table {number}" if number > 0 else "Table data"
    heading = str(table.get("heading_path") or "").strip()
    caption = str(table.get("caption") or "").strip()
    parts = [label.rstrip(" .")]
    if heading:
        parts.append(heading.rstrip(" ."))
    if caption and caption.lower() not in {heading.lower(), label.lower()}:
        parts.append(caption.rstrip(" ."))
    return ". ".join(parts)


def _table_metric_context(table: dict[str, Any]) -> str:
    caption = re.sub(r"\[[^\]]+\]", " ", str(table.get("caption") or ""))
    caption = re.sub(r"\s+", " ", caption).strip(" .:;,-")
    patterns = (
        r"\bresults?\s+(?:on|for)\s+([A-Za-z0-9][A-Za-z0-9+_.-]*)",
        r"\b(?:on|for)\s+([A-Za-z0-9][A-Za-z0-9+_.-]*)\s+(?:dataset|benchmark)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, caption, flags=re.I)
        if match:
            return str(match.group(1) or "").strip()
    return ""


def _contextual_metric_label(table: dict[str, Any], label: str) -> str:
    metric_label = str(label or "").strip()
    context = _table_metric_context(table)
    if not context or context.lower() in metric_label.lower():
        return metric_label
    return f"{context} {metric_label}".strip()


def _table_subject_kind(label: str) -> str:
    surface = str(label or "").strip()
    if re.search(
        r"\b(?:blocks?|layers?|depth|width|variant|setting|configuration|component|activation|sigma|"
        r"sampling\s+ratio|CS\s+ratio|SR|patch(?:es)?)\b|"
        r"块数|层数|深度|宽度|变体|设置|配置|组件|激活|采样率",
        surface,
        flags=re.I,
    ):
        return "variant"
    if re.search(r"\b(?:method|model|network|algorithm|architecture)\b|方法|模型|网络|算法|架构", surface, flags=re.I):
        return "method"
    return ""


def _build_table_record(block: dict[str, Any], *, table_index: int) -> dict[str, Any] | None:
    headers, raw_rows = parse_markdown_table(str(block.get("raw_text") or ""))
    if not headers or not raw_rows:
        return None

    number = int(block.get("number") or block.get("table_number") or 0)
    table: dict[str, Any] = {
        "table_index": int(table_index),
        "table_number": number,
        "caption": str(block.get("caption_text") or "").strip(),
        "heading_path": str(block.get("heading_path") or "").strip(),
        "block_id": str(block.get("block_id") or "").strip(),
        "anchor_id": str(block.get("anchor_id") or "").strip(),
        "line_start": int(block.get("line_start") or 0),
        "line_end": int(block.get("line_end") or 0),
        "headers": headers,
        "rows": [],
    }
    page_start = int(block.get("page_start") or 0)
    page_end = int(block.get("page_end") or page_start or 0)
    if page_start > 0:
        table["page_start"] = page_start
        table["page_end"] = page_end or page_start

    label_column_count = min(max(1, _label_column_count(headers)), max(1, len(headers) - 1))
    row_headers = [headers[index] or f"Row field {index + 1}" for index in range(label_column_count)]
    row_header = " / ".join(row_headers)
    row_subject_kind = _table_subject_kind(row_header)
    carried_labels = [""] * label_column_count
    metric_series: dict[str, dict[str, Any]] = {}
    for row_index, values in enumerate(raw_rows, start=1):
        label_parts: list[str] = []
        for label_index in range(label_column_count):
            value = str(values[label_index] if label_index < len(values) else "").strip()
            if value:
                carried_labels[label_index] = value
            effective = value or carried_labels[label_index]
            if effective:
                label_parts.append(effective)
        label = " / ".join(label_parts) or f"Row {row_index}"
        transposed_specs = _metric_specs(label) if label_column_count == 1 else []
        transposed_metric_row = bool(transposed_specs) and sum(
            1 for value in values[label_column_count:] if str(value or "").strip()
        ) >= 2
        cells: list[dict[str, Any]] = []
        fact_texts: list[str] = []
        for column_index in range(label_column_count, max(len(headers), len(values))):
            header = str(headers[column_index] if column_index < len(headers) else "").strip() or f"Column {column_index + 1}"
            value = str(values[column_index] if column_index < len(values) else "").strip()
            if not value:
                continue
            facts = _metric_facts(label if transposed_metric_row else header, value)
            cell: dict[str, Any] = {"column_index": column_index, "column": header, "value": value}
            if facts:
                cell_facts: list[dict[str, str]] = []
                for fact in facts:
                    fact_label = (
                        _contextual_metric_label(table, fact["label"])
                        if transposed_metric_row
                        else fact["label"]
                    )
                    cell_fact = {**fact, "label": fact_label}
                    cell_facts.append(cell_fact)
                    if transposed_metric_row:
                        fact_texts.append(f"{header} {fact_label} = {fact['value']}")
                    else:
                        fact_texts.append(f"{fact_label} = {fact['value']}")
                    series = metric_series.setdefault(
                        fact_label,
                        {
                            "label": fact_label,
                            "metric": fact["metric"],
                            "direction": fact["direction"],
                            "subject_label": headers[0] if transposed_metric_row else row_header,
                            "subject_kind": (
                                _table_subject_kind(headers[0])
                                if transposed_metric_row
                                else row_subject_kind
                            ),
                            "values": [],
                        },
                    )
                    series["values"].append(
                        {
                            "row_label": header if transposed_metric_row else label,
                            "value": fact["value"],
                        }
                    )
                cell["metrics"] = cell_facts
            else:
                fact_texts.append(f"{header} = {value}")
                series = metric_series.setdefault(
                    header,
                    {
                        "label": header,
                        "metric": header,
                        "direction": "",
                        "subject_label": row_header,
                        "subject_kind": row_subject_kind,
                        "values": [],
                    },
                )
                series["values"].append({"row_label": label, "value": value})
            cells.append(cell)

        prefix = _table_location_prefix(table)
        facts_surface = "; ".join(fact_texts)
        search_text = f"{prefix}. {row_header}: {label}. {facts_surface}".strip()
        table["rows"].append(
            {
                "row_index": row_index,
                "row_label": label,
                "cells": cells,
                "search_text": search_text[:6000],
                "locate_anchor": f"{row_header}: {label}",
            }
        )

    table["row_count"] = len(table["rows"])
    table["metric_series"] = list(metric_series.values())
    return table


def build_table_index_payload(
    md_path: Path | str,
    blocks: list[dict[str, Any]],
    *,
    version: int,
) -> dict[str, Any]:
    path = Path(str(md_path or "")).expanduser()
    tables: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict) or str(block.get("kind") or "").strip().lower() != "table":
            continue
        record = _build_table_record(block, table_index=len(tables) + 1)
        if record is not None:
            tables.append(record)
    return {
        "version": int(version),
        "doc_id": doc_id_for_path(path),
        "doc_path": str(path.resolve(strict=False)),
        "table_count": len(tables),
        "row_count": sum(int(table.get("row_count") or 0) for table in tables),
        "tables": tables,
    }


def table_index_to_chunks(payload: dict[str, Any], *, source_path: str, schema_version: int) -> list[dict]:
    chunks: list[dict] = []
    for table in list(payload.get("tables") or []):
        if not isinstance(table, dict):
            continue
        base_meta: dict[str, Any] = {
            "source_path": source_path,
            "heading_path": str(table.get("heading_path") or "").strip(),
            "table_index": int(table.get("table_index") or 0),
            "table_number": int(table.get("table_number") or 0),
            "block_id": str(table.get("block_id") or "").strip(),
            "table_block_id": str(table.get("block_id") or "").strip(),
            "anchor_id": str(table.get("anchor_id") or "").strip(),
            "line_start": int(table.get("line_start") or 0),
            "line_end": int(table.get("line_end") or 0),
            "chunk_schema_version": int(schema_version),
        }
        if int(table.get("page_start") or 0) > 0:
            base_meta["page_start"] = int(table.get("page_start") or 0)
            base_meta["page_end"] = int(table.get("page_end") or table.get("page_start") or 0)

        for row in list(table.get("rows") or []):
            if not isinstance(row, dict):
                continue
            text = str(row.get("search_text") or "").strip()
            if not text:
                continue
            meta = {
                **base_meta,
                "structured_kind": "table_row",
                "table_row_index": int(row.get("row_index") or 0),
                "table_row_label": str(row.get("row_label") or "").strip(),
                "table_locate_anchor": str(row.get("locate_anchor") or "").strip(),
                "char_len": len(text),
            }
            chunks.append({"text": text, "meta": meta})

        prefix = _table_location_prefix(table)
        for series in list(table.get("metric_series") or []):
            if not isinstance(series, dict):
                continue
            label = str(series.get("label") or "").strip()
            values = [
                f"{str(item.get('row_label') or '').strip()} = {str(item.get('value') or '').strip()}"
                for item in list(series.get("values") or [])
                if isinstance(item, dict) and str(item.get("row_label") or "").strip() and str(item.get("value") or "").strip()
            ]
            if not label or not values:
                continue
            direction = _DIRECTION_WORDS.get(str(series.get("direction") or ""), "")
            direction_text = f" ({direction})" if direction else ""
            text = f"{prefix}. {label}{direction_text}: {'; '.join(values)}"[:6000]
            meta = {
                **base_meta,
                "structured_kind": "table_metric",
                "table_metric": str(series.get("metric") or label).strip(),
                "table_metric_label": label,
                "table_metric_direction": str(series.get("direction") or ""),
                "table_subject_label": str(series.get("subject_label") or "").strip(),
                "table_subject_kind": str(series.get("subject_kind") or "").strip(),
                "char_len": len(text),
            }
            chunks.append({"text": text, "meta": meta})
    return chunks


def table_chunks_from_markdown(md_text: str, *, source_path: str, schema_version: int) -> list[dict]:
    source = str(md_text or "")
    if not re.search(r"(?m)^\s*\|.*\|\s*$", source):
        return []
    blocks = build_source_blocks(
        source,
        doc_id=doc_id_for_path(source_path),
    )
    payload = build_table_index_payload(source_path, blocks, version=schema_version)
    return table_index_to_chunks(payload, source_path=source_path, schema_version=schema_version)
