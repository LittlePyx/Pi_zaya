from __future__ import annotations

import re
from collections.abc import Callable


_PAGE_MARKER_RE = re.compile(r"<!--\s*kb_page:\s*(\d+)\s*-->", re.IGNORECASE)
_DISPLAY_MATH_DELIMITER_RE = re.compile(r"(?<!\\)\$\$")
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
_FORMULA_ANCHOR_RE = re.compile(
    r"(?:\\[A-Za-z]+|[=_^{}]|\\begin\{|\\end\{|\\tag\{|\\frac\b|\\sum\b|"
    r"\\int\b|\\mathbb\b|\\mathrm\b|\\left\b|\\right\b|\\mid\b|\\Vert\b)"
)


def _delimiter_spans_outside_fences(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    in_fence = False
    fence_char = ""
    offset = 0
    for line in str(text or "").splitlines(keepends=True):
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            marker_char = marker[0]
            if not in_fence:
                in_fence = True
                fence_char = marker_char
            elif marker_char == fence_char:
                in_fence = False
                fence_char = ""
            offset += len(line)
            continue
        if not in_fence:
            spans.extend(
                (offset + match.start(), offset + match.end())
                for match in _DISPLAY_MATH_DELIMITER_RE.finditer(line)
            )
        offset += len(line)
    return spans


def normalize_display_math_delimiter_lines(text: str) -> str:
    """Put unescaped ``$$`` delimiters on dedicated Markdown lines.

    Vision models often emit ``prose: $$``. The legacy math cleanup only tracks
    standalone delimiters, so normalizing this representation is required
    before any stateful cleanup runs.
    """

    source = str(text or "")
    out: list[str] = []
    in_fence = False
    fence_char = ""
    for line in source.splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            marker_char = marker[0]
            if not in_fence:
                in_fence = True
                fence_char = marker_char
            elif marker_char == fence_char:
                in_fence = False
                fence_char = ""
            out.append(line)
            continue
        if in_fence or not _DISPLAY_MATH_DELIMITER_RE.search(line) or line.strip() == "$$":
            out.append(line)
            continue

        cursor = 0
        for match in _DISPLAY_MATH_DELIMITER_RE.finditer(line):
            fragment = line[cursor : match.start()].rstrip()
            if fragment:
                out.append(fragment)
            out.append("$$")
            cursor = match.end()
        tail = line[cursor:].lstrip()
        if tail:
            out.append(tail)
    fixed = "\n".join(out)
    if source.endswith("\n"):
        fixed += "\n"
    return fixed


def _page_segments(text: str, *, default_page: int = 1) -> list[tuple[int, str]]:
    source = normalize_display_math_delimiter_lines(str(text or ""))
    matches = list(_PAGE_MARKER_RE.finditer(source))
    if not matches:
        return [(max(1, int(default_page)), source)]

    segments: list[tuple[int, str]] = []
    if source[: matches[0].start()]:
        segments.append((max(1, int(default_page)), source[: matches[0].start()]))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        segments.append((int(match.group(1)), source[match.start() : end]))
    return segments


def transform_markdown_pages(
    text: str,
    transform: Callable[[str], str],
    *,
    default_page: int = 1,
) -> str:
    """Apply a stateful Markdown transform independently to every page.

    Math cleanup routines maintain delimiter state while scanning. Running one
    over a whole document lets malformed output on one physical page affect
    every later page. Splitting at preserved page anchors contains that state.
    """

    return "".join(
        transform(segment)
        for _, segment in _page_segments(text, default_page=default_page)
    )


def unclosed_display_math_pages(text: str, *, default_page: int = 1) -> list[int]:
    """Return physical pages whose unescaped ``$$`` delimiters are unbalanced.

    Counting is deliberately page-local. A missing delimiter on one page must
    never be paired with a delimiter on a later page, because that silently
    turns ordinary prose and page anchors into one enormous math block.
    """

    pages: list[int] = []
    for page_no, segment in _page_segments(text, default_page=default_page):
        if len(_delimiter_spans_outside_fences(segment)) % 2:
            pages.append(int(page_no))
    return pages


def _looks_formula_block(text: str) -> bool:
    probe = str(text or "").strip()
    if not probe:
        return False
    anchors = len(_FORMULA_ANCHOR_RE.findall(probe))
    prose_words = re.findall(r"\b[A-Za-z]{3,}\b", re.sub(r"\\[A-Za-z]+", " ", probe))
    # Compact equations such as ``x = y`` have only one strong anchor but no
    # prose. Treat those as safe while keeping prose-only tails unresolved.
    return bool(anchors >= 1 and (anchors >= 4 or len(prose_words) <= 8))


def _repair_unclosed_segment(
    segment: str,
    *,
    page_no: int,
    add_audit_marker: bool,
) -> tuple[str, bool, bool]:
    """Close the last page-local display block.

    Returns ``(text, changed, safely_repaired)``. The common VL failure leaves
    one final equation opener followed by a contiguous equation paragraph and
    then prose (or EOF). We close after that first equation paragraph. If the
    boundary is not formula-shaped, we still contain the block at the page end
    and emit an unresolved retry marker so later pages remain intact and the
    quality gate cannot silently accept the page.
    """

    spans = _delimiter_spans_outside_fences(segment)
    if not spans or len(spans) % 2 == 0:
        return segment, False, True

    start, end = spans[-1]
    prefix = segment[:start]
    tail = segment[end:]

    line_start = prefix.rfind("\n") + 1
    opener_prefix = prefix[line_start:]
    if opener_prefix.strip():
        prefix = prefix.rstrip() + "\n\n"

    leading_match = re.match(r"[ \t\r\n]*", tail)
    leading = leading_match.group(0) if leading_match else ""
    body = tail[len(leading) :]
    if not body:
        equation = ""
        remainder = ""
        separator = ""
        safe = False
    else:
        boundary = re.search(r"\r?\n[ \t]*\r?\n", body)
        if boundary:
            equation = body[: boundary.start()].rstrip()
            separator = boundary.group(0)
            remainder = body[boundary.end() :]
        else:
            equation = body.rstrip()
            separator = ""
            remainder = ""
        safe = _looks_formula_block(equation)

    if safe:
        repaired = prefix + "$$\n"
        if equation:
            repaired += equation + "\n"
        repaired += "$$"
        audit_marker = f"<!-- kb:page_math_boundary_repair page={int(page_no)} -->"
        if add_audit_marker and audit_marker not in segment:
            repaired += f"\n{audit_marker}"
        if remainder:
            repaired += (separator or "\n\n") + remainder.lstrip("\r\n")
        elif segment.endswith("\n"):
            repaired += "\n"
        return repaired, repaired != segment, True

    # Uncertain boundary: contain the damage within this physical page and
    # leave an actionable marker. This is safer than allowing every following
    # page to be parsed as display math.
    repaired = prefix + "$$" + tail.rstrip() + "\n$$"
    retry_marker = (
        f"<!-- kb:conversion_retry kind=math_text page={int(page_no)} "
        "reason=unclosed_display_math -->"
    )
    if retry_marker not in segment:
        repaired += f"\n{retry_marker}"
    if segment.endswith("\n"):
        repaired += "\n"
    return repaired, repaired != segment, False


def repair_page_display_math_boundaries(
    text: str,
    *,
    default_page: int = 1,
    add_audit_marker: bool = True,
) -> tuple[str, list[int], list[int]]:
    """Repair or contain unclosed display math independently on every page.

    The second return value lists safely repaired pages; the third lists pages
    that were only contained and therefore carry a conversion-retry marker.
    """

    source = str(text or "")
    repaired_segments: list[str] = []
    repaired_pages: list[int] = []
    unresolved_pages: list[int] = []
    for page_no, segment in _page_segments(source, default_page=default_page):
        repaired, changed, safe = _repair_unclosed_segment(
            segment,
            page_no=page_no,
            add_audit_marker=add_audit_marker,
        )
        repaired_segments.append(repaired)
        if changed:
            if safe:
                repaired_pages.append(int(page_no))
            else:
                unresolved_pages.append(int(page_no))
    return "".join(repaired_segments), repaired_pages, unresolved_pages
