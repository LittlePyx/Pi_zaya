from __future__ import annotations

import re
from typing import Any


def _line_text_from_dict_line(line: dict[str, Any]) -> str:
    parts: list[str] = []
    for span in list(line.get("spans") or []):
        text = str(span.get("text") or "")
        if text:
            parts.append(text)
    text = "".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _looks_two_column(lines: list[tuple[float, float, float, float, str]], page_width: float) -> bool:
    if page_width <= 0 or len(lines) < 12:
        return False
    content = [
        item
        for item in lines
        if item[1] >= 35.0
        and item[3] <= 760.0
        and not re.fullmatch(r"\d{1,6}-\d{1,3}", item[4].strip())
    ]
    if len(content) < 12:
        return False
    mid = page_width * 0.5
    left = [item for item in content if ((item[1] + item[2]) / 2.0) < mid - 12.0]
    right = [item for item in content if ((item[1] + item[2]) / 2.0) > mid + 12.0]
    if len(left) < 6 or len(right) < 6:
        return False
    left_x1 = max(item[2] for item in left)
    right_x0 = min(item[1] for item in right)
    return (right_x0 - left_x1) >= 6.0


def reference_ordered_page_text(page, *, fallback_text: str | None = None) -> str:
    """
    Extract page text in layout order suitable for references.

    PyMuPDF's plain text mode can interleave two-column reference lists by row
    (left ref 7, right ref 18, left ref 8...), which corrupts bibliography
    reconstruction. For detected two-column pages, emit full left column then
    full right column while preserving line breaks.
    """
    fallback = str(fallback_text or "")
    try:
        data = page.get_text("dict") or {}
    except Exception:
        return fallback

    try:
        page_width = float(getattr(getattr(page, "rect", None), "width", 0.0) or 0.0)
    except Exception:
        page_width = 0.0

    lines: list[tuple[float, float, float, float, str]] = []
    for block in list(data.get("blocks") or []):
        if "lines" not in block:
            continue
        for line in list(block.get("lines") or []):
            text = _line_text_from_dict_line(line)
            if not text:
                continue
            try:
                x0, y0, x1, y1 = [float(v) for v in line.get("bbox")]
            except Exception:
                continue
            if x1 <= x0 or y1 <= y0:
                continue
            lines.append((y0, x0, x1, y1, text))

    if len(lines) < 4:
        return fallback

    if _looks_two_column(lines, page_width):
        mid = page_width * 0.5

        def key(item: tuple[float, float, float, float, str]) -> tuple[int, float, float]:
            y0, x0, x1, _y1, _text = item
            col = 0 if ((x0 + x1) / 2.0) < mid else 1
            return (col, round(y0, 1), round(x0, 1))

        ordered = sorted(lines, key=key)
    else:
        ordered = sorted(lines, key=lambda item: (round(item[0], 1), round(item[1], 1)))

    text = "\n".join(item[4] for item in ordered).strip()
    return text or fallback
