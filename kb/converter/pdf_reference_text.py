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
    if (right_x0 - left_x1) >= 6.0:
        return True

    # A page that changes from body/table content into a two-column bibliography
    # can contain a few centred table cells or spanning headers. Those lines make
    # the strict column-edge gap negative even though normal prose still forms
    # two strong x-position clusters.
    narrow = [
        item
        for item in content
        if (item[2] - item[1]) <= page_width * 0.48
        and len(item[4]) >= 8
        and re.search(r"[A-Za-z]{3}", item[4])
    ]
    left_anchors = [item for item in narrow if item[1] <= page_width * 0.20]
    right_anchors = [item for item in narrow if item[1] >= page_width * 0.48]
    return len(left_anchors) >= 4 and len(right_anchors) >= 4


def consecutive_reference_chain_positions(numbers: list[int]) -> list[int]:
    """Return positions for the longest n, n+1, ... subsequence.

    A PDF reference block can contain a standalone volume or page number that
    resembles a reference marker (for example ``1, 2, 13, 3, 4``). Ignore such
    values instead of rejecting an otherwise complete bibliography.
    """
    clean = [int(value) for value in numbers]
    candidates: list[list[int]] = []
    for start, first in enumerate(clean):
        if first <= 0:
            continue
        expected = first
        positions = [start]
        for idx in range(start + 1, len(clean)):
            value = clean[idx]
            if value == expected + 1:
                positions.append(idx)
                expected = value
        candidates.append(positions)
    if not candidates:
        return []
    candidates.sort(
        key=lambda positions: (
            len(positions),
            int(bool(positions) and clean[positions[0]] == 1),
            -int(positions[0] if positions else len(clean)),
        ),
        reverse=True,
    )
    best = candidates[0]
    # Interleaved two-column extraction can manufacture two equally plausible
    # chains (for example 1,20,2,21,3,22). Without layout confirmation, treating
    # either chain as a bibliography would silently reorder or drop entries.
    best_positions = set(best)
    for competitor in candidates[1:]:
        if len(competitor) < 3:
            break
        if best_positions.isdisjoint(competitor):
            return []
    return best


_BRACKETED_REFERENCE_START_RE = re.compile(r"^\s*\[\s*\d{1,4}\s*]\s+\S")
_STANDALONE_PERIOD_NUMBER_RE = re.compile(r"^\s*\d{1,4}\.\s*$")


def merge_standalone_reference_continuations(text: str) -> str:
    """Join volume/page-only lines to the prior bracketed reference.

    Wiley-style references use ``[n]`` markers, but a wrapped final page number
    can be extracted as a standalone ``13.`` line. It is a continuation, not a
    new reference numbered 13.
    """
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if sum(1 for line in lines if _BRACKETED_REFERENCE_START_RE.match(line)) < 3:
        return "\n".join(lines).strip()
    out: list[str] = []
    for idx, line in enumerate(lines):
        match = _STANDALONE_PERIOD_NUMBER_RE.match(line)
        if match and out:
            previous = out[-1].rstrip()
            active_number = 0
            for prior in reversed(out):
                active_match = _BRACKETED_REFERENCE_START_RE.match(prior)
                if not active_match:
                    continue
                number_match = re.match(r"^\s*\[\s*(\d{1,4})\s*]", prior)
                active_number = int(number_match.group(1)) if number_match else 0
                break
            standalone_number = int(re.search(r"\d{1,4}", line).group(0))
            following = [str(item or "").strip() for item in lines[idx + 1 : idx + 5] if str(item or "").strip()]
            next_line = following[0] if following else ""
            next_starts_reference_body = bool(
                next_line
                and not _BRACKETED_REFERENCE_START_RE.match(next_line)
                and re.match(r"^[A-Z][A-Za-z'\-]{1,40}(?:,|\s+[A-Z]\.?\b)", next_line)
                and re.search(
                    r"\b(?:18|19|20)\d{2}\b|\b(?:journal|nature|science|opt\.|phys\.|ieee|acm|doi)\b",
                    " ".join(following),
                    flags=re.IGNORECASE,
                )
            )
            is_next_reference = bool(
                active_number > 0
                and standalone_number == active_number + 1
                and next_starts_reference_body
            )
            if previous.endswith((",", ";", ":")) and not is_next_reference:
                out[-1] = f"{previous} {line.strip()}".strip()
                continue
        out.append(line)
    return "\n".join(out).strip()


def is_ambiguous_reference_running_line(line: str) -> bool:
    """Return whether text needs page-position or pre-marker context to drop."""
    text = re.sub(r"\s+", " ", str(line or "")).strip()
    return bool(
        re.fullmatch(r"Article", text, flags=re.IGNORECASE)
        or re.match(r"^Laser Photonics Rev\.\s+20\d{2}\b", text, flags=re.IGNORECASE)
    )


def is_reference_running_line(line: str) -> bool:
    """Return whether a line is a publisher header/footer inside References."""
    text = re.sub(r"\s+", " ", str(line or "")).strip()
    if not text:
        return False
    return bool(
        re.fullmatch(r"Article", text, flags=re.IGNORECASE)
        or re.match(r"^NATURE COMMUNICATIONS\s*\|", text, flags=re.IGNORECASE)
        or re.match(r"^Laser Photonics Rev\.\s+20\d{2}\b", text, flags=re.IGNORECASE)
        or re.fullmatch(r"\d{5,12}\s*\(\d+\s+of\s+\d+\)", text, flags=re.IGNORECASE)
        or re.match(r"^\d{8},\s*20\d{2},.*\bDownloaded from\b", text, flags=re.IGNORECASE)
        or re.match(r"^(?:©\s*)?20\d{2}\s+Wiley(?:-VCH)?\b", text, flags=re.IGNORECASE)
    )


def trim_reference_publisher_tail(text: str) -> str:
    """Remove publisher boilerplate appended after the final reference.

    Nature-family PDFs often place a publisher note, licence, affiliations,
    and contact details below the last bibliography entry. Layout extraction
    flattens that material into the final reference unless the boundary is
    restored before reference formatting.
    """
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for idx, raw in enumerate(lines):
        line = re.sub(r"\s+", " ", str(raw or "")).strip()
        if re.match(r"^Publisher['’]s note\b", line, flags=re.IGNORECASE):
            return "\n".join(lines[:idx]).strip()
    return "\n".join(lines).strip()


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
        page_height = float(getattr(getattr(page, "rect", None), "height", 0.0) or 0.0)
    except Exception:
        page_width = 0.0
        page_height = 0.0

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
            if (
                is_reference_running_line(text)
                and page_height > 0
                and (y0 <= 45.0 or y1 >= page_height - 35.0)
            ):
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
