from __future__ import annotations

from pathlib import Path
import re
from typing import Any


def _positive_int(value: Any) -> int:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def system_a_source_key(detail: dict | None) -> str:
    """Return a stable document key for one rendered System-A citation."""

    row = detail if isinstance(detail, dict) else {}
    source_path = str(row.get("source_path") or row.get("sourcePath") or "").strip()
    if source_path:
        normalized = source_path.replace("/", "\\").casefold()
        parts = [part for part in normalized.split("\\") if part]
        if len(parts) >= 2:
            return "\\".join(parts[-2:])
        if parts:
            return parts[-1]
    source_name = str(row.get("source_name") or row.get("sourceName") or "").strip().casefold()
    if not source_name:
        return ""
    name = Path(source_name).name or source_name
    return re.sub(r"(?i)(?:\.en)?\.md$|\.pdf$", "", name).strip()


def _system_a_detail(detail: dict | None) -> bool:
    row = detail if isinstance(detail, dict) else {}
    return (
        str(row.get("citation_route") or "").strip().lower() == "system_a"
        and bool(system_a_source_key(row))
    )


def _link_position(markdown: str, anchor: str) -> int:
    anchor_text = str(anchor or "").strip()
    if not anchor_text:
        return -1
    return str(markdown or "").find(f"](#{anchor_text}")


def _detail_original_numbers(detail: dict) -> list[int]:
    traced_values = [
        detail.get("answer_hit_num"),
        detail.get("original_num"),
        *list(detail.get("answer_hit_linked_nums") or []),
    ]
    fallback_values = [
        detail.get("num"),
        *list(detail.get("linked_nums") or []),
    ]
    values = traced_values if any(_positive_int(value) for value in traced_values) else fallback_values
    out: list[int] = []
    for value in values:
        number = _positive_int(value)
        if number > 0 and number not in out:
            out.append(number)
    return out


def remap_system_a_citations_for_display(
    markdown: str,
    cite_details: list[dict] | None,
) -> tuple[str, list[dict], list[dict]]:
    """Make visible System-A numbers contiguous by cited document.

    Retrieval-hit numbers are an internal generation coordinate. Reference
    cards are grouped by document, so exposing those raw coordinates can leave
    an answer with ``[4]`` while the matching card is labelled ``#1``. This
    function runs only after every link has already been grounded. It changes
    the visible label and public detail number while preserving the original
    hit number and exact anchor for traceability. System-B bibliography numbers
    are deliberately left untouched.
    """

    text = str(markdown or "")
    rows = [dict(item) for item in list(cite_details or []) if isinstance(item, dict)]
    eligible: list[tuple[int, int, str, dict]] = []
    for index, row in enumerate(rows):
        if not _system_a_detail(row):
            continue
        anchor = str(row.get("anchor") or "").strip()
        position = _link_position(text, anchor)
        eligible.append((position, index, system_a_source_key(row), row))
    if not eligible:
        return text, rows, []

    eligible.sort(
        key=lambda item: (
            item[0] if item[0] >= 0 else 10**12,
            _positive_int(item[3].get("num")) or 10**9,
            item[1],
        )
    )
    display_by_source: dict[str, int] = {}
    registry_by_source: dict[str, dict] = {}
    for _position, _index, source_key, row in eligible:
        if source_key not in display_by_source:
            display_by_source[source_key] = len(display_by_source) + 1
            registry_by_source[source_key] = {
                "display_num": display_by_source[source_key],
                "source_key": source_key,
                "source_path": str(row.get("source_path") or row.get("sourcePath") or "").strip(),
                "source_name": str(row.get("source_name") or row.get("sourceName") or "").strip(),
                "original_nums": [],
            }
        original_numbers = _detail_original_numbers(row)
        registry_numbers = registry_by_source[source_key]["original_nums"]
        for number in original_numbers:
            if number not in registry_numbers:
                registry_numbers.append(number)

    remapped: list[dict] = []
    for row in rows:
        if not _system_a_detail(row):
            remapped.append(row)
            continue
        source_key = system_a_source_key(row)
        display_num = int(display_by_source[source_key])
        original_numbers = _detail_original_numbers(row)
        original_num = _positive_int(row.get("answer_hit_num")) or _positive_int(row.get("original_num"))
        if original_num <= 0:
            original_num = _positive_int(row.get("num"))
        next_row = dict(row)
        if original_num > 0:
            next_row.setdefault("answer_hit_num", original_num)
            next_row.setdefault("original_num", original_num)
        if original_numbers:
            next_row["answer_hit_linked_nums"] = original_numbers
        next_row["display_num"] = display_num
        next_row["num"] = display_num
        next_row["linked_nums"] = [display_num]
        anchor = str(next_row.get("anchor") or "").strip()
        if anchor:
            text = re.sub(
                rf"\[\d{{1,4}}\](?=\(\#{re.escape(anchor)}(?:\s|\)))",
                f"[{display_num}]",
                text,
            )
        remapped.append(next_row)

    known_anchors = {
        str(row.get("anchor") or "").strip()
        for row in remapped
        if str(row.get("anchor") or "").strip()
    }
    # Citation rendering can deduplicate detail rows by document while the
    # final Markdown legitimately links the same paper at multiple answer
    # positions.  Give each rendered System-A anchor a detail alias so the
    # frontend styles and handles every occurrence instead of falling back to
    # a plain Markdown link.
    for match in re.finditer(
        r'\[(\d{1,4})\]\(\#([^\s)]+)(?:\s+"([^"]*)")?\)',
        text,
    ):
        linked_num = _positive_int(match.group(1))
        anchor = str(match.group(2) or "").strip()
        title = str(match.group(3) or "").strip().lower()
        if (
            not anchor
            or anchor in known_anchors
            or anchor.startswith("kb-cite-reader-")
            or "source:" not in title
        ):
            continue
        candidates = [
            row
            for row in remapped
            if _system_a_detail(row)
            and (
                _positive_int(row.get("num")) == linked_num
                or linked_num in _detail_original_numbers(row)
            )
        ]
        source_keys = {system_a_source_key(row) for row in candidates if system_a_source_key(row)}
        if len(source_keys) != 1 or not candidates:
            continue
        display_num = _positive_int(candidates[0].get("num"))
        if display_num <= 0:
            continue
        text = re.sub(
            rf"\[{linked_num}\](?=\(\#{re.escape(anchor)}(?:\s|\)))",
            f"[{display_num}]",
            text,
            count=1,
        )
        alias = dict(candidates[0])
        alias["anchor"] = anchor
        alias["citation_occurrence_alias"] = True
        remapped.append(alias)
        known_anchors.add(anchor)

    registry = sorted(
        registry_by_source.values(),
        key=lambda item: int(item.get("display_num") or 0),
    )
    return text, remapped, registry


__all__ = [
    "remap_system_a_citations_for_display",
    "system_a_source_key",
]
