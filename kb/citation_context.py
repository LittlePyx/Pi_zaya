from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

from kb.inpaper_citation_grounding import parse_ref_num_set
from kb.source_blocks import normalize_inline_markdown, tokenize_match_text


_PAGE_MARKER_RE = re.compile(r"<!--\s*kb_page:\s*(\d{1,5})\s*-->", re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
_REFERENCE_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*"
    r"(?:references?|bibliography|literature\s+cited|works\s+cited|cited\s+references|"
    r"reference\s+list|\u53c2\u8003\u6587\u732e|\u5f15\u7528\u6587\u732e)\b",
    re.IGNORECASE,
)
_REFERENCE_ENTRY_RE = re.compile(r"^\s*(?:\[\s*\d{1,4}\s*\]|\d{1,4}\s*[\.)])\s+\S+")
_INLINE_REF_RE = re.compile(
    r"(?<!\[)\[(\d{1,4}(?:\s*(?:[-\u2013\u2014\u2212,;\uff0c\u3001\uff1b])\s*\d{1,4})*)\](?!\])"
)
_REF_SPEC_SEPARATOR_RE = re.compile(r"[\uff0c\u3001;\uff1b]")
_BOUNDARY_CHARS = ".!?;\n\u3002\uff01\uff1f\uff1b"


class _BodyBlock(NamedTuple):
    text: str
    heading_path: str
    page: int
    line_start: int
    line_end: int


def _path_cache_key(source_path: str) -> tuple[str, int, int] | None:
    raw = str(source_path or "").strip()
    if not raw:
        return None
    path = Path(raw)
    try:
        if not path.exists():
            return None
        stat = path.stat()
        return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    except Exception:
        return None


@lru_cache(maxsize=96)
def _read_text_cached(path_str: str, mtime_ns: int, size: int) -> str:
    del mtime_ns, size
    try:
        return Path(path_str).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _clean_heading(raw: str) -> str:
    text = normalize_inline_markdown(str(raw or ""))
    text = re.sub(r"\s+", " ", text).strip(" #\t")
    return text[:180]


def _heading_path(stack: list[tuple[int, str]]) -> str:
    return " / ".join(title for _level, title in stack if title)


def _is_reference_entry_line(line: str) -> bool:
    s = str(line or "").strip()
    if not s:
        return False
    return bool(_REFERENCE_ENTRY_RE.match(s))


def _normalize_ref_spec(spec: str) -> str:
    return _REF_SPEC_SEPARATOR_RE.sub(",", str(spec or "").strip())


def _contains_ref_num(marker_spec: str, ref_num: int) -> bool:
    try:
        target = int(ref_num)
    except Exception:
        return False
    if target <= 0:
        return False
    return target in parse_ref_num_set(_normalize_ref_spec(marker_spec), max_items=64)


def _trim_window(text: str, start: int, end: int, *, max_chars: int) -> str:
    s = str(text or "")
    if not s:
        return ""
    limit = max(120, int(max_chars or 520))
    start = max(0, min(len(s), int(start)))
    end = max(start, min(len(s), int(end)))

    left = -1
    for ch in _BOUNDARY_CHARS:
        left = max(left, s.rfind(ch, 0, start))
    left = 0 if left < 0 else left + 1

    right_candidates = [s.find(ch, end) for ch in _BOUNDARY_CHARS]
    right_candidates = [x for x in right_candidates if x >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(s)

    if right - left > limit:
        center = (start + end) // 2
        half = limit // 2
        left = max(0, center - half)
        right = min(len(s), left + limit)
        left = max(0, min(left, max(0, right - limit)))

    chunk = re.sub(r"\s+", " ", s[left:right]).strip()
    if not chunk:
        return ""
    if left > 0:
        chunk = "..." + chunk.lstrip()
    if right < len(s):
        chunk = chunk.rstrip() + "..."
    return chunk[:limit].strip()


def _score_context(context: str, *, answer_context: str, heading_path: str, page: int, line_start: int) -> float:
    score = 5.0
    if heading_path:
        score += 0.8
    if page > 0:
        score += 0.5
    if line_start > 0:
        score += max(0.0, 0.4 - min(line_start, 400) / 1000.0)
    answer_tokens = set(tokenize_match_text(answer_context))
    context_tokens = set(tokenize_match_text(context))
    if answer_tokens and context_tokens:
        overlap = answer_tokens & context_tokens
        score += min(4.0, len(overlap) * 0.55)
        if len(overlap) >= 3:
            score += 0.6
    return score


@lru_cache(maxsize=96)
def _body_blocks_cached(path_str: str, mtime_ns: int, size: int) -> tuple[_BodyBlock, ...]:
    text = _read_text_cached(path_str, mtime_ns, size)
    if not text:
        return ()

    blocks: list[_BodyBlock] = []
    paragraph: list[str] = []
    paragraph_start = 0
    paragraph_page = 0
    paragraph_heading = ""
    current_page = 0
    heading_stack: list[tuple[int, str]] = []
    in_fence = False

    def flush(line_no: int) -> None:
        nonlocal paragraph, paragraph_start, paragraph_page, paragraph_heading
        if not paragraph:
            return
        raw = "\n".join(paragraph).strip()
        paragraph = []
        cleaned = normalize_inline_markdown(raw)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) < 20:
            return
        if _is_reference_entry_line(cleaned):
            return
        blocks.append(
            _BodyBlock(
                text=cleaned,
                heading_path=paragraph_heading,
                page=paragraph_page,
                line_start=int(paragraph_start),
                line_end=int(line_no),
            )
        )

    for line_no, line in enumerate(text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), start=1):
        if _FENCE_RE.match(line):
            flush(line_no - 1)
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        page_match = _PAGE_MARKER_RE.search(line)
        if page_match:
            flush(line_no - 1)
            try:
                current_page = int(page_match.group(1))
            except Exception:
                current_page = 0
            continue

        if _REFERENCE_HEADING_RE.match(line):
            flush(line_no - 1)
            break

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush(line_no - 1)
            level = len(str(heading_match.group(1) or ""))
            title = _clean_heading(str(heading_match.group(2) or ""))
            if title:
                heading_stack = [(lv, tx) for lv, tx in heading_stack if lv < level]
                heading_stack.append((level, title))
            continue

        if not str(line or "").strip():
            flush(line_no - 1)
            continue
        if _is_reference_entry_line(line):
            flush(line_no - 1)
            continue

        if not paragraph:
            paragraph_start = line_no
            paragraph_page = int(current_page or 0)
            paragraph_heading = _heading_path(heading_stack)
        paragraph.append(line)

    flush(10**9)
    return tuple(blocks)


def extract_inpaper_reference_context(
    source_path: str,
    ref_num: int,
    *,
    answer_context: str = "",
    max_chars: int = 520,
) -> dict[str, Any]:
    """Find the citing-paper context where bibliography reference ``ref_num`` is used."""

    key = _path_cache_key(source_path)
    if not key:
        return {}
    path_str, mtime_ns, size = key
    try:
        target = int(ref_num)
    except Exception:
        return {}
    if target <= 0:
        return {}

    best: dict[str, Any] = {}
    best_score = float("-inf")
    for block in _body_blocks_cached(path_str, mtime_ns, size):
        for match in _INLINE_REF_RE.finditer(block.text):
            if not _contains_ref_num(match.group(1), target):
                continue
            context = _trim_window(block.text, int(match.start()), int(match.end()), max_chars=max_chars)
            if not context:
                continue
            score = _score_context(
                context,
                answer_context=str(answer_context or ""),
                heading_path=block.heading_path,
                page=block.page,
                line_start=block.line_start,
            )
            location_parts = [part for part in (block.heading_path, f"p. {block.page}" if block.page > 0 else "") if part]
            candidate = {
                "citation_context": context,
                "citation_context_source": "source_markdown",
                "heading_path": block.heading_path,
                "location_label": " · ".join(location_parts),
                "page_start": int(block.page or 0),
                "page_end": int(block.page or 0),
                "line_start": int(block.line_start),
                "line_end": int(block.line_end),
                "anchor_kind": "paragraph",
                "citation_context_quality": "matched_ref_marker",
                "citation_context_score": round(float(score), 3),
            }
            if score > best_score:
                best_score = score
                best = candidate
    return best
