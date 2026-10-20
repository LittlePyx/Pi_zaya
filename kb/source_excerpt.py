from __future__ import annotations

import re
from pathlib import Path

from kb.source_blocks import load_source_blocks_cached


def scoped_source_blocks(record: dict) -> list[dict]:
    """Never repair a stale locator by searching a different page or section."""
    source = str(record.get("source_path") or "")
    if not source or not Path(source).is_file():
        return []
    try:
        page = int(record.get("page_start") or 0)
        end = int(record.get("page_end") or page)
    except (ValueError, TypeError):
        return []
    heading = str(record.get("heading_path") or "").strip()
    if page <= 0 or not heading:
        return []
    return [
        b
        for b in load_source_blocks_cached(Path(source))
        if int(b.get("page_start") or 0) >= page
        and int(b.get("page_end") or 0) <= end
        and (
            str(b.get("heading_path") or "").strip() == heading
            or (
                not record.get("block_id")
                and not record.get("anchor_id")
                and str(b.get("heading_path") or "").startswith(heading + " / ")
            )
        )
        and all(
            not record.get(k) or record[k] == b.get(k)
            for k in ("block_id", "anchor_id")
        )
    ]


def literal_excerpt(block: dict, quote: str) -> str:
    """Restore only explicitly omitted text inside one physical source block.

    No token-overlap matching: all excerpt fragments must occur in order.
    Returned text is copied from the source, including its math and punctuation.
    """
    raw = str(block.get("raw_text") or "")
    exact = _literal_excerpt(raw, quote, raw)
    if exact or block.get("kind") == "code":
        return exact
    # A prose finisher may remove paired bold delimiters. Restore source bytes
    # through an offset-preserving surface; no words or math are discarded.
    protected = [m.span() for m in re.finditer(r"\$\$.*?\$\$|\$[^$\n]+\$|`+.*?`+", raw, re.S)]
    surface = list(raw)
    for match in re.finditer(r"(?<![\w\\])\*\*([^*\n]+)\*\*(?!\w)", raw):
        if any(match.start() < end and match.end() > start for start, end in protected):
            continue
        for index in (match.start(), match.start()+1, match.end()-2, match.end()-1):
            surface[index] = " "
    visible = "".join(surface)
    return _literal_excerpt(raw, quote, visible) if visible != raw else ""


def _literal_excerpt(raw: str, quote: str, surface: str) -> str:
    # Collapsing whitespace gives offsets back into the original source. It
    # does not discard math, punctuation, citations, negation or numbers.
    tokens = list(re.finditer(r"\S+", surface))
    flat = " ".join(t[0] for t in tokens)
    offsets = [
        i
        for j, t in enumerate(tokens)
        for i in ([t.start() - 1] if j else []) + list(range(t.start(), t.end()))
    ]
    pieces = [p.strip() for p in re.split(r"…|\.{3,}", quote) if p.strip()]
    if not pieces or any(len(p) < 20 for p in pieces):
        return ""
    start = end = 0
    for index, piece in enumerate(pieces):
        piece = re.sub(r"\s+", " ", piece).strip()
        match = re.search(re.escape(piece), flat[end:], re.IGNORECASE)
        if match is None:
            return ""
        position = end + match.start()
        if index == 0:
            start = position
        end += match.end()
    if quote.rstrip().endswith(("...", "…")):
        # Finish the source sentence instead of printing synthetic ellipsis.
        boundary = re.search(r"[.!?。！？](?=\s|$)", flat[end:])
        end = end + boundary.end() if boundary else len(flat)
    first, last = offsets[start], offsets[end - 1] + 1
    # An exact preview can lose only the opening bold delimiter. Restore it
    # only from a complete prose emphasis pair in this very source span.
    protected = [m.span() for m in re.finditer(r"\$\$.*?\$\$|\$[^$\n]+\$|`+.*?`+", raw, re.S)]
    for emphasis in re.finditer(r"(?<![\w\\])\*\*[^*\n]+\*\*(?!\w)", raw):
        if emphasis.start() + 2 == first and last >= emphasis.end() and not any(emphasis.start() < b and emphasis.end() > a for a, b in protected):
            first = emphasis.start()
    if raw[first-2:first] == "**" and surface[first-2:first] == "  ":
        first -= 2
    if raw[last:last+2] == "**" and surface[last:last+2] == "  ":
        last += 2
    result = raw[first:last].strip()
    return result if len(result) >= 40 else ""
