from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

from kb.evidence_text import looks_author_list_context, looks_bibliography_entry_context
from kb.inpaper_citation_grounding import parse_ref_num_set
from kb.source_blocks import normalize_inline_markdown, tokenize_match_text


_PAGE_MARKER_RE = re.compile(r"<!--\s*kb_page:\s*(\d{1,5})\s*-->", re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
_REFERENCE_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*"
    r"(?:\d{1,3}\s*[.)]?\s*)?(?:references?|bibliography|literature\s+cited|works\s+cited|"
    r"cited\s+references|reference\s+list|references\s+and\s+notes|"
    r"\u53c2\u8003\u6587\u732e|\u5f15\u7528\u6587\u732e)\s*[:\uff1a]?\s*$",
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


def _reference_index_cache_key(source_path: str) -> tuple[str, int, int] | None:
    raw = str(source_path or "").strip()
    if not raw:
        return None
    index_path = Path(raw).parent / "assets" / "reference_index.json"
    try:
        if not index_path.exists():
            return None
        stat = index_path.stat()
        return (str(index_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    except Exception:
        return None


@lru_cache(maxsize=96)
def _read_reference_index_cached(path_str: str, mtime_ns: int, size: int) -> dict[str, Any]:
    del mtime_ns, size
    try:
        data = json.loads(Path(path_str).read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _to_int(value: Any) -> int:
    try:
        out = int(value or 0)
    except Exception:
        return 0
    return out if out > 0 else 0


def _precomputed_reference_mentions(source_path: str, ref_num: int) -> list[dict[str, Any]]:
    key = _reference_index_cache_key(source_path)
    if not key:
        return []
    path_str, mtime_ns, size = key
    payload = _read_reference_index_cached(path_str, mtime_ns, size)
    references = payload.get("references")
    if not isinstance(references, list):
        return []
    target = _to_int(ref_num)
    if target <= 0:
        return []
    for item in references:
        if not isinstance(item, dict):
            continue
        if _to_int(item.get("ref_num")) != target:
            continue
        rows = item.get("citation_mentions")
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
        return []
    return []


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


def _looks_like_inline_reference_marker(text: str, match: re.Match[str]) -> bool:
    """Reject bracketed numbers that are probably notation, not bibliography refs."""

    s = str(text or "")
    try:
        start = int(match.start())
        end = int(match.end())
    except Exception:
        return True

    before = s[start - 1] if start > 0 else ""
    after = s[end] if end < len(s) else ""

    # Examples from converted PDFs: ``s [2]ISM`` / ``s[2]ISM`` are notation
    # fragments for s2ISM, not citations to bibliography reference 2.
    if after and re.match(r"[A-Za-z0-9_]", after):
        return False
    if before and re.match(r"[A-Za-z0-9_]", before):
        if not before.isspace():
            return False
    return True


def _looks_invalid_source_context(context: str) -> bool:
    text = re.sub(r"\s+", " ", str(context or "")).strip()
    if not text:
        return True
    without_markers = _INLINE_REF_RE.sub(" ", text)
    without_markers = re.sub(r"^[.\s]+|[.\s]+$", "", without_markers)
    if len(tokenize_match_text(without_markers)) < 4:
        return True
    return looks_author_list_context(text) or looks_bibliography_entry_context(text)


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

    structured_best: dict[str, Any] = {}
    structured_best_score = float("-inf")
    for row in _precomputed_reference_mentions(source_path, target):
        context = re.sub(r"\s+", " ", str(row.get("citation_context") or "")).strip()
        if not context:
            continue
        limit = max(180, int(max_chars or 520))
        if len(context) > limit:
            context = context[: max(0, limit - 3)].rstrip() + "..."
        if _looks_invalid_source_context(context):
            continue
        heading_path = str(row.get("heading_path") or "").strip()
        page = _to_int(row.get("page_start") or row.get("page"))
        line_start = _to_int(row.get("line_start"))
        line_end = _to_int(row.get("line_end"))
        score = _score_context(
            context,
            answer_context=str(answer_context or ""),
            heading_path=heading_path,
            page=page,
            line_start=line_start,
        )
        location_label = str(row.get("location_label") or "").strip()
        if not location_label:
            location_parts = [part for part in (heading_path, f"p. {page}" if page > 0 else "") if part]
            location_label = " / ".join(location_parts)
        candidate = {
            "citation_context": context,
            "citation_context_source": "structured_reference_index",
            "heading_path": heading_path,
            "location_label": location_label,
            "page_start": int(page or 0),
            "page_end": _to_int(row.get("page_end")) or int(page or 0),
            "line_start": int(line_start or 0),
            "line_end": int(line_end or 0),
            "anchor_kind": str(row.get("anchor_kind") or "paragraph").strip() or "paragraph",
            "citation_context_quality": "precomputed_ref_marker",
            "citation_context_score": round(float(score), 3),
        }
        for key_name in ("block_id", "anchor_id"):
            value = str(row.get(key_name) or "").strip()
            if value:
                candidate[key_name] = value
        if score > structured_best_score:
            structured_best_score = score
            structured_best = candidate
    if structured_best:
        return structured_best

    best: dict[str, Any] = {}
    best_score = float("-inf")
    for block in _body_blocks_cached(path_str, mtime_ns, size):
        for match in _INLINE_REF_RE.finditer(block.text):
            if not _looks_like_inline_reference_marker(block.text, match):
                continue
            if not _contains_ref_num(match.group(1), target):
                continue
            context = _trim_window(block.text, int(match.start()), int(match.end()), max_chars=max_chars)
            if not context:
                continue
            if _looks_invalid_source_context(context):
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
                "location_label": " / ".join(location_parts),
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
