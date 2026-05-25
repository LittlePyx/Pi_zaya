from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class Block:
    kind: str  # "heading" | "text"
    text: str
    heading_path: str
    page: int | None = None


def _parse_blocks(md: str) -> list[Block]:
    blocks: list[Block] = []
    heading_stack: list[tuple[int, str]] = []
    cur_page: int | None = None

    # Page marker inserted by our converter:
    # <!-- kb_page: 12 -->
    re_page = re.compile(r"^<!--\s*kb_page\s*:\s*(\d+)\s*-->$", flags=re.IGNORECASE)

    def current_heading_path() -> str:
        return " / ".join([t for _, t in heading_stack])

    lines = md.splitlines()
    buf: list[str] = []

    def flush_buf() -> None:
        nonlocal buf
        s = "\n".join(buf).strip("\n")
        if s.strip():
            blocks.append(Block(kind="text", text=s, heading_path=current_heading_path(), page=cur_page))
        buf = []

    for line in lines:
        stripped = line.strip()
        m_page = re_page.match(stripped)
        if m_page:
            # Update current page and drop the marker from text (avoid polluting retrieval).
            flush_buf()
            try:
                cur_page = int(m_page.group(1))
            except Exception:
                cur_page = cur_page
            continue

        if stripped.startswith("#"):
            # Flush previous text block
            flush_buf()

            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped[level:].strip()

            # Maintain stack
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

            blocks.append(Block(kind="heading", text=stripped, heading_path=current_heading_path(), page=cur_page))
            continue

        # Keep paragraph structure; blank lines separate paragraphs.
        if stripped == "":
            flush_buf()
        else:
            buf.append(line)

    flush_buf()
    return blocks


def _semantic_overlap_tail(text: str, overlap: int) -> str:
    src = str(text or "")
    max_len = int(overlap or 0)
    if max_len <= 0 or len(src) <= max_len:
        return src

    start = max(0, len(src) - max_len)
    window = src[start:]
    min_tail = min(80, max(24, max_len // 3))

    boundary_re = re.compile(r"(?:\n\s*\n+|(?<=[。！？.!?])\s+)")
    best = ""
    for match in boundary_re.finditer(window):
        candidate = window[match.end() :].strip()
        if len(candidate) >= min_tail:
            best = candidate
            break
    if best:
        return best

    # If the raw character window starts inside a token, drop the partial token.
    prev = src[start - 1] if start > 0 else ""
    if prev and window and re.match(r"[A-Za-z0-9]", prev) and re.match(r"[A-Za-z0-9]", window[0]):
        trimmed = re.sub(r"^\S+\s*", "", window, count=1).strip()
        if len(trimmed) >= min_tail:
            return trimmed

    return window.strip()


def _merge_blocks_into_chunks(
    blocks: list[Block],
    source_path: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    chunks: list[dict] = []
    cur: list[str] = []
    cur_len = 0
    cur_heading_path = ""
    cur_page_start: int | None = None
    cur_page_end: int | None = None

    def flush(force: bool = False) -> None:
        nonlocal cur, cur_len, cur_heading_path, cur_page_start, cur_page_end
        if not cur:
            return
        text = "\n".join(cur).strip()
        if not text:
            cur = []
            cur_len = 0
            cur_page_start = None
            cur_page_end = None
            return

        meta = {
            "source_path": source_path,
            "heading_path": cur_heading_path,
            "char_len": len(text),
        }
        if cur_page_start is not None:
            meta["page_start"] = int(cur_page_start)
        if cur_page_end is not None:
            meta["page_end"] = int(cur_page_end)

        chunks.append(
            {
                "text": text,
                "meta": meta,
            }
        )

        if force or overlap <= 0:
            cur = []
            cur_len = 0
            cur_page_start = None
            cur_page_end = None
            return

        # Keep tail as overlap, but avoid starting the next chunk in the
        # middle of a word/sentence. Mid-token overlaps make evidence cards look
        # fragmentary even when the original markdown is fine.
        tail = _semantic_overlap_tail(text, overlap)
        cur = [tail]
        cur_len = len(tail)
        # Overlap keeps the same approximate page range.

    for b in blocks:
        if b.kind == "heading":
            # Start a new chunk at headings to help retrieval & navigation.
            flush(force=True)
            cur_heading_path = b.heading_path
            cur = [b.text]
            cur_len = len(b.text)
            cur_page_start = b.page
            cur_page_end = b.page
            continue

        if not cur:
            cur_heading_path = b.heading_path
            cur_page_start = b.page
            cur_page_end = b.page

        if cur_len + len(b.text) + 1 > chunk_size and cur_len > 200:
            flush(force=False)

        cur.append(b.text)
        cur_len += len(b.text) + 1
        if b.page is not None:
            if cur_page_start is None:
                cur_page_start = b.page
                cur_page_end = b.page
            else:
                cur_page_start = min(cur_page_start, b.page)
                cur_page_end = max(cur_page_end or b.page, b.page)

    flush(force=True)
    return chunks


def chunk_markdown(
    md: str,
    source_path: str,
    chunk_size: int = 1400,
    overlap: int = 200,
) -> list[dict]:
    blocks = _parse_blocks(md)
    return _merge_blocks_into_chunks(
        blocks=blocks,
        source_path=source_path,
        chunk_size=chunk_size,
        overlap=overlap,
    )
