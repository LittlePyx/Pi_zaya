from __future__ import annotations

import re
from typing import Optional


def _parse_box_heading_line(line: str) -> Optional[tuple[int, int, str]]:
    """
    Parse a standalone Box heading line.
    Returns (heading_level, box_id, title).
    heading_level=0 means plain line without markdown heading markers.
    """
    raw = (line or "").strip()
    if not raw:
        return None
    level = 0
    body = raw
    m_h = re.match(r"^(#{1,6})\s+(.*)$", raw)
    if m_h:
        level = len(m_h.group(1))
        body = (m_h.group(2) or "").strip()
    m = re.match(r"^Box\s*(\d+)\s*(?:[|:-]\s*(.+))?$", body, flags=re.IGNORECASE)
    if not m:
        return None
    box_id = int(m.group(1))
    title = (m.group(2) or "").strip()
    return (level, box_id, title)


def _render_box_block(*, box_id: int, title: str, body_lines: list[str]) -> list[str]:
    """Render extracted Box section as a standalone metadata block in markdown."""
    out: list[str] = []
    out.append(f"<!-- box:start id={box_id} -->")
    out.append("")
    label = f"Box {box_id}"
    if title:
        label = f"{label} - {title}"
    out.append(f"**[{label}]**")

    page_hint = None
    for ln in body_lines:
        m = re.search(r"page_(\d+)_", ln or "", flags=re.IGNORECASE)
        if m:
            try:
                page_hint = int(m.group(1))
            except Exception:
                page_hint = None
            if page_hint is not None:
                break
    if page_hint is not None:
        out.append(f"*Source: p.{page_hint} sidebar*")

    if body_lines:
        out.append("")
        out.extend(body_lines)

    out.append("")
    out.append(f"<!-- box:end id={box_id} -->")
    return out


def _extract_box_sidebars(md: str) -> str:
    """
    Convert standalone "Box n" headings into non-heading sidebar blocks so they
    don't pollute document heading hierarchy in markdown outputs.
    """
    lines = md.splitlines()
    if not lines:
        return md

    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        parsed = _parse_box_heading_line(lines[i])
        if not parsed:
            out.append(lines[i])
            i += 1
            continue

        level, box_id, title = parsed
        j = i + 1
        body: list[str] = []
        while j < n:
            ln = lines[j]
            st = (ln or "").strip()
            m_h = re.match(r"^(#{1,6})\s+(.+)$", st)
            if m_h:
                # Start of next same/higher section ends this box block.
                # For plain (non-#) Box lines, any heading ends the block.
                next_level = len(m_h.group(1))
                if level == 0 or next_level <= level:
                    break
            body.append(ln)
            j += 1

        # Trim redundant blank lines around body content.
        while body and (not body[0].strip()):
            body.pop(0)
        while body and (not body[-1].strip()):
            body.pop()

        rendered = _render_box_block(box_id=box_id, title=title, body_lines=body)
        if out and out[-1].strip():
            out.append("")
        out.extend(rendered)
        if j < n and lines[j].strip():
            out.append("")

        i = j

    return "\n".join(out)


def _looks_like_lowercase_continuation_line(text: str) -> bool:
    st = (text or "").strip()
    if not st:
        return False
    if re.match(r"^(?:#{1,6}\s+|!\[|<!--|```|\$\$)", st):
        return False
    if re.match(r"^(?:[-*+]\s+|\d+\.\s+)", st):
        return False
    if re.match(r"^(?:fig(?:ure)?\.?|table|algorithm)\b", st, re.IGNORECASE):
        return False
    if len(st) < 16:
        return False
    m = re.search(r"[A-Za-z]", st)
    if not m:
        return False
    ch = st[m.start()]
    if not ch.islower():
        return False
    # Continuation fragments usually terminate naturally.
    if not re.search(r"[.!?)]\s*$", st):
        return False
    return True


def _ends_with_dangling_connector(text: str) -> bool:
    st = (text or "").strip()
    if not st:
        return False
    if re.search(r"[.!?;:)\]\}]$", st):
        return False
    low = st.lower()
    return bool(
        re.search(
            r"(?:\bas\s+the|\bof\s+the|\bto\s+the|\bfor\s+the|\bwith\s+the|\bin\s+the|\bon\s+the)\s*$",
            low,
        )
        or re.search(r"(?:\bthat|\bwhich|\bwhose|\bwhere|\bwhen|\band|\bor)\s*$", low)
    )


def _repair_dangling_heading_continuations(md: str) -> str:
    """
    Repair common reading-order glitches where a heading appears before a
    lowercase continuation fragment that should belong to the previous sentence.
    """
    lines = md.splitlines()
    if not lines:
        return md

    heading_re = re.compile(r"^\s*#{1,6}\s+\S")
    max_scan = 28

    i = 0
    while i < len(lines):
        if not heading_re.match(lines[i]):
            i += 1
            continue

        p = i - 1
        while p >= 0:
            stp = (lines[p] or "").strip()
            if not stp:
                p -= 1
                continue
            if stp.startswith("<!-- box:"):
                p -= 1
                continue
            break
        if p < 0 or (not _ends_with_dangling_connector(lines[p])):
            i += 1
            continue

        cand_idx = None
        scanned = 0
        j = i + 1
        while j < len(lines) and scanned < max_scan:
            st = (lines[j] or "").strip()
            if heading_re.match(lines[j]):
                break
            if _looks_like_lowercase_continuation_line(st):
                cand_idx = j
                break
            j += 1
            scanned += 1
        if cand_idx is None:
            i += 1
            continue

        cand_line = lines.pop(cand_idx)
        lines.insert(i, cand_line)

        # If we inserted a continuation fragment, collapse one blank line above it.
        if i - 1 >= 0 and not (lines[i - 1] or "").strip():
            lines.pop(i - 1)
            i -= 1

        # Move past heading to avoid reprocessing loops.
        i += 2

    return "\n".join(lines)

