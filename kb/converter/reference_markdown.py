from __future__ import annotations

import re

from .formula_markdown import formula_to_plain_text
from .text_utils import _normalize_text


_REF_HEADING_RE = re.compile(r"^#+\s+References?\s*$|^References?\s*$", re.IGNORECASE)
_REF_START_INLINE_RE = re.compile(r"^(?:\[\d{1,4}\]|\d{1,4}[.)])\s+")
_REF_BACKREF_SUFFIX_RE = re.compile(
    r"(\b(?:19|20)\d{2}\.)(?:\s+\d{1,3}\s*(?:,\s*\d{1,3}){0,12})\s*$"
)
_YEAR_BACKREF_LINE_RE = re.compile(
    r"^(?P<year>(?:18|19|20)\d{2})\.\s+\d{1,3}(?:\s*,\s*\d{1,3}){0,12}\s*$"
)
_REF_SECTION_STOP_RE = re.compile(
    r"^(?:Acknowledg(?:e)?ments?|Author contributions?|Competing interests?|"
    r"Additional information|Supplementary information|Supplementary materials?|"
    r"Supplemental information|Supplemental materials?|Correspondence and requests|"
    r"Data availability|Code availability|Ethics declarations?)$",
    re.IGNORECASE,
)
_STANDALONE_REF_NUMBER_RE = re.compile(r"^\[?(\d{1,4})\]?[.)]\s*$")
_INLINE_REF_START_RE = re.compile(r"^(?:\[(\d{1,4})\]|(\d{1,4})[.)])\s+(.+)$")
_MID_REF_START_RE = re.compile(r"(?<!\S)(?:\[(\d{1,4})\]|(\d{1,4})[.)])\s+([A-Z][^.]{10,})")
_AUTHOR_YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}[a-z]?\.\s+\S")
_PAGE_MARKER_LINE_RE = re.compile(r"^<!--\s*kb_page:\s*\d+\s*-->$", re.IGNORECASE)
_INLINE_PAGE_MARKER_RE = re.compile(r"(<!--\s*kb_page:\s*\d+\s*-->)", re.IGNORECASE)


def _is_plausible_reference_number(value: str | int | None) -> bool:
    try:
        n = int(str(value or "").strip())
    except Exception:
        return False
    if n <= 0:
        return False
    # Years inside titles/venues, e.g. "In 2019 IEEE ...", are not reference ids.
    if 1800 <= n <= 2099:
        return False
    return n <= 999


def _is_reference_start_line(text: str) -> bool:
    src = str(text or "").strip()
    if _YEAR_BACKREF_LINE_RE.match(src):
        return False
    match = _INLINE_REF_START_RE.match(src)
    if not match:
        return False
    return _is_plausible_reference_number(match.group(1) or match.group(2))


def _looks_like_author_list_prefix(prefix: str) -> bool:
    text = re.sub(r"\s+", " ", str(prefix or "")).strip(" ,;:")
    if len(text) < 8 or len(text) > 360:
        return False
    low = text.lower()
    if re.search(
        r"\b(?:conference|proceedings?|transactions?|journal|volume|vol\.?|"
        r"pages?|springer|wiley|ieee|acm|cvpr|iccv|eccv|siggraph)\b",
        low,
    ):
        return False
    words = re.findall(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`.-]{1,}\b", text)
    if len(words) < 2:
        return False
    name_word = r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`.-]{1,}"
    starts_like_author_list = re.match(
        rf"^{name_word}(?:\s+(?:[A-Z]\.?|{name_word})){{0,5}}(?:\s*,|\s+and\b)",
        text,
    )
    return bool(starts_like_author_list and ("," in text or re.search(r"\band\b", text, flags=re.IGNORECASE)))


def _looks_like_author_year_reference_text(text: str) -> bool:
    src = re.sub(r"\s+", " ", str(text or "")).strip()
    if not src or _is_reference_start_line(src):
        return False
    match = _AUTHOR_YEAR_RE.search(src)
    if not match:
        return False
    prefix = src[: int(match.start())]
    return _looks_like_author_list_prefix(prefix)


def _author_year_entry_start_offsets(text: str) -> list[int]:
    src = re.sub(r"\s+", " ", str(text or "")).strip()
    if not src:
        return []
    name_word = r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`.-]{1,}"
    author_start_re = re.compile(
        rf"{name_word}(?:\s+(?:[A-Z]\.?|{name_word})){{0,5}}(?:\s*,|\s+and\b)"
    )
    starts: list[int] = []
    search_start = 0
    for year_match in _AUTHOR_YEAR_RE.finditer(src):
        best_start: int | None = None
        for author_match in author_start_re.finditer(src[search_start : int(year_match.start())]):
            candidate_start = search_start + int(author_match.start())
            candidate_prefix = src[candidate_start : int(year_match.start())]
            if _looks_like_author_list_prefix(candidate_prefix):
                best_start = candidate_start
                break
        if best_start is not None and all(abs(best_start - existing) > 12 for existing in starts):
            starts.append(best_start)
        search_start = max(search_start, int(year_match.end()))
    return sorted(starts)


def _split_author_year_collapsed_line(text: str) -> list[str]:
    src = re.sub(r"\s+", " ", str(text or "")).strip()
    if not src:
        return []
    starts = _author_year_entry_start_offsets(src)
    if not starts:
        return [src]
    boundaries = list(starts)
    if starts[0] > 0:
        boundaries = [0, *boundaries]
    out: list[str] = []
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(src)
        part = src[max(0, start) : max(0, end)].strip(" ,;")
        if part:
            out.append(part)
    return out


def _line_looks_like_author_year_entry_start(line: str) -> bool:
    text = re.sub(r"\s+", " ", str(line or "")).strip()
    if not text:
        return False
    if _looks_like_author_year_reference_text(text):
        return True
    if _AUTHOR_YEAR_RE.search(text):
        return False
    return _looks_like_author_list_prefix(text)


def _looks_like_reference_running_line(line: str) -> bool:
    text = re.sub(r"\s+", " ", str(line or "")).strip()
    if not text:
        return False
    if text == "•":
        return True
    if re.fullmatch(r"\d{1,5}:\d{1,5}", text):
        return True
    if re.match(r"^ACM Trans\. Graph\.,\s+Vol\.", text, flags=re.IGNORECASE):
        return True
    if re.match(r"^Publication date:", text, flags=re.IGNORECASE):
        return True
    return False


def _reference_lines_look_author_year(ref_lines: list[tuple[int, str]]) -> bool:
    lines = [
        str(line or "").strip()
        for _, line in list(ref_lines or [])
        if str(line or "").strip()
        and not _PAGE_MARKER_LINE_RE.match(str(line or "").strip())
        and not _REF_HEADING_RE.match(str(line or "").strip())
    ]
    if not lines:
        return False
    numeric_starts = sum(1 for line in lines if _is_reference_start_line(line))
    author_year_hits = 0
    for idx, line in enumerate(lines):
        if not _AUTHOR_YEAR_RE.search(line):
            continue
        if _looks_like_author_year_reference_text(line):
            author_year_hits += max(1, len(_AUTHOR_YEAR_RE.findall(line)))
            continue
        window = _join_reference_fragments(lines[max(0, idx - 2) : idx + 1])
        if _looks_like_author_year_reference_text(window):
            author_year_hits += 1
    return bool(author_year_hits >= 2 and numeric_starts < author_year_hits)


def _is_year_backref_continuation_line(text: str) -> bool:
    return bool(_YEAR_BACKREF_LINE_RE.match(str(text or "").strip()))


def _strip_reference_backref_suffix(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    return _REF_BACKREF_SUFFIX_RE.sub(r"\1", src)


def _join_reference_fragments(parts: list[str]) -> str:
    chunks = [str(item or "").strip() for item in list(parts or []) if str(item or "").strip()]
    if not chunks:
        return ""
    text = " ".join(chunks)
    text = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", text)
    text = re.sub(r"(https?://\S*/)\s+(?=\S)", r"\1", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _strip_reference_backref_suffix(text)
    return text.strip()


def _should_keep_reference_open_on_blank(
    current_ref: list[str],
    next_nonempty: str,
) -> bool:
    tail = str((current_ref or [])[-1] or "").strip() if current_ref else ""
    if not tail:
        return False
    nxt = str(next_nonempty or "").strip()
    if not nxt or _is_reference_start_line(nxt):
        return False
    if tail.endswith("-"):
        return True
    if not re.search(r"[.!?]\s*$", tail):
        return True
    return False


def normalize_references_page_text(page_text: str) -> str:
    lines_out: list[str] = []
    raw_lines = str(page_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized_lines = [_normalize_text(raw or "").strip() for raw in raw_lines]
    heading_indices = [idx for idx, line in enumerate(normalized_lines) if _REF_HEADING_RE.match(line)]
    saw_heading = bool(heading_indices)
    if heading_indices:
        raw_lines = raw_lines[int(heading_indices[0]) + 1 :]

    reference_start_seen = 0
    for raw in raw_lines:
        line = _normalize_text(raw or "").strip()
        if not line:
            lines_out.append("")
            continue
        if _REF_HEADING_RE.match(line):
            continue
        if _REF_SECTION_STOP_RE.match(line) and reference_start_seen >= 2:
            break
        if re.fullmatch(r"\d{1,4}", line):
            continue
        if (
            re.fullmatch(r"[A-Z][A-Z\s&,\-]{6,80}", line)
            and not _is_reference_start_line(line)
            and not re.search(r"\b(?:18|19|20)\d{2}\b", line)
        ):
            continue
        if re.search(r"\bpage\s+\d+\s+of\s+\d+\b", line, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"www\.[^\s]+", line, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"[A-Z][A-Z\s&,\-]{3,90}\s+www\.[^\s]+", line):
            continue
        if _is_reference_start_line(line) or _STANDALONE_REF_NUMBER_RE.match(line):
            reference_start_seen += 1
        lines_out.append(line)

    while lines_out and not str(lines_out[0] or "").strip():
        lines_out.pop(0)
    while lines_out and not str(lines_out[-1] or "").strip():
        lines_out.pop()

    if saw_heading:
        return "# References\n\n" + "\n".join(lines_out)
    return "\n".join(lines_out)


def fix_references_format(md: str) -> str:
    """
    Fix references section formatting:
    - Remove formula blocks ($$...$$) and code blocks (```...```) from references
    - Ensure each reference is on a separate line
    - Ensure references are numbered (add numbers if missing)
    - Convert formulas in references to plain text
    """
    lines = md.splitlines()
    result = []
    in_references = False
    ref_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^#+\s+References?\s*$", stripped, re.IGNORECASE) or re.match(r"^References?\s*$", stripped, re.IGNORECASE):
            in_references = True
            result.append(line)
            continue

        if in_references:
            if stripped.startswith("#") and not re.match(r"^#+\s+References?\s*$", stripped, re.IGNORECASE):
                heading_level = len(stripped) - len(stripped.lstrip("#"))
                if heading_level <= 2:
                    result.extend(format_references_block(ref_lines))
                    ref_lines = []
                    in_references = False
                    result.append(line)
                    continue

            ref_lines.append((i, line))
        else:
            result.append(line)

    if ref_lines:
        result.extend(format_references_block(ref_lines))

    return "\n".join(result)


def format_references_block(ref_lines: list[tuple[int, str]]) -> list[str]:
    """Format a block of reference lines - ensure each reference is on a separate line with numbering."""
    if _reference_lines_look_author_year(ref_lines):
        return _format_author_year_references_block(ref_lines)

    formatted = []
    current_ref = []
    ref_num = 1
    current_ref_number: int | None = None
    in_code_block = False
    in_display_math = False

    for idx, (_, line) in enumerate(ref_lines):
        stripped = line.strip()

        if re.match(r"^<!--\s*kb_page:\s*\d+\s*-->$", stripped, re.IGNORECASE):
            if current_ref:
                ref_text = _join_reference_fragments(current_ref)
                if ref_text:
                    out_num = int(current_ref_number or ref_num)
                    formatted.append(format_single_reference(ref_text, out_num))
                    ref_num = out_num + 1
                current_ref = []
                current_ref_number = None
            formatted.append(stripped)
            continue

        if not stripped:
            next_nonempty = ""
            for _, candidate in ref_lines[idx + 1 :]:
                candidate_text = str(candidate or "").strip()
                if candidate_text:
                    next_nonempty = candidate_text
                    break
            if current_ref and _should_keep_reference_open_on_blank(current_ref, next_nonempty):
                continue
            if current_ref:
                ref_text = _join_reference_fragments(current_ref)
                if ref_text:
                    out_num = int(current_ref_number or ref_num)
                    formatted.append(format_single_reference(ref_text, out_num))
                    ref_num = out_num + 1
                current_ref = []
                current_ref_number = None
            continue

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        if stripped == "$$":
            in_display_math = not in_display_math
            continue
        if in_display_math:
            plain_math = formula_to_plain_text(stripped)
            if plain_math:
                current_ref.append(plain_math)
            continue

        if stripped.startswith("$$"):
            if stripped.endswith("$$") and len(stripped) > 4:
                formula_text = stripped[2:-2].strip()
                plain_text = formula_to_plain_text(formula_text)
                if plain_text:
                    current_ref.append(plain_text)
            continue

        if stripped.startswith("$") and stripped.endswith("$") and stripped.count("$") == 2 and len(stripped) > 2:
            stripped = formula_to_plain_text(stripped[1:-1].strip())
            if not stripped:
                continue

        if "$" in stripped:
            stripped = re.sub(r"\$([^$]+)\$", lambda m: formula_to_plain_text(m.group(1)), stripped)
            stripped = re.sub(r"\$\$([^$]+)\$\$", lambda m: formula_to_plain_text(m.group(1)), stripped)
            stripped = stripped.replace("$", "").strip()

        if current_ref and _is_year_backref_continuation_line(stripped):
            current_ref.append(stripped)
            continue

        standalone_match = _STANDALONE_REF_NUMBER_RE.match(stripped)
        if standalone_match and _is_plausible_reference_number(standalone_match.group(1)):
            if current_ref:
                ref_text = _join_reference_fragments(current_ref)
                if ref_text:
                    out_num = int(current_ref_number or ref_num)
                    formatted.append(format_single_reference(ref_text, out_num))
                    ref_num = out_num + 1
                current_ref = []
                current_ref_number = None

            current_ref_number = int(standalone_match.group(1))
            continue

        ref_match = _INLINE_REF_START_RE.match(stripped)
        if ref_match and _is_plausible_reference_number(ref_match.group(1) or ref_match.group(2)):
            if current_ref:
                ref_text = _join_reference_fragments(current_ref)
                if ref_text:
                    out_num = int(current_ref_number or ref_num)
                    formatted.append(format_single_reference(ref_text, out_num))
                    ref_num = out_num + 1
                current_ref = []
                current_ref_number = None

            ref_content = ref_match.group(3).strip()
            current_ref_number = int(ref_match.group(1) or ref_match.group(2))
            if ref_content:
                current_ref.append(ref_content)
        else:
            mid_ref_match = None
            for candidate in _MID_REF_START_RE.finditer(stripped):
                raw_num = candidate.group(1) or candidate.group(2)
                if _is_plausible_reference_number(raw_num):
                    mid_ref_match = candidate
                    break
            if mid_ref_match and current_ref:
                before_ref = stripped[:mid_ref_match.start()].strip()
                if before_ref:
                    current_ref.append(before_ref)
                ref_text = _join_reference_fragments(current_ref)
                if ref_text:
                    out_num = int(current_ref_number or ref_num)
                    formatted.append(format_single_reference(ref_text, out_num))
                    ref_num = out_num + 1
                current_ref_number = None
                raw_num = mid_ref_match.group(1) or mid_ref_match.group(2)
                current_ref_number = int(raw_num)
                current_ref = [mid_ref_match.group(3).strip()]
            else:
                current_ref.append(stripped)

    if current_ref:
        ref_text = _join_reference_fragments(current_ref)
        if ref_text:
            out_num = int(current_ref_number or ref_num)
            formatted.append(format_single_reference(ref_text, out_num))

    return formatted


def _format_author_year_references_block(ref_lines: list[tuple[int, str]]) -> list[str]:
    formatted: list[str] = []
    current_ref: list[str] = []

    def has_year(parts: list[str]) -> bool:
        return bool(_AUTHOR_YEAR_RE.search(_join_reference_fragments(parts)))

    def flush_current() -> None:
        nonlocal current_ref
        ref_text = _join_reference_fragments(current_ref)
        current_ref = []
        if ref_text and _looks_like_author_year_reference_text(ref_text):
            formatted.append(ref_text)

    for _, line in list(ref_lines or []):
        stripped = str(line or "").strip()
        if not stripped:
            if current_ref and has_year(current_ref):
                flush_current()
            continue
        if _PAGE_MARKER_LINE_RE.match(stripped):
            if current_ref:
                flush_current()
            formatted.append(stripped)
            continue
        if _REF_HEADING_RE.match(stripped):
            continue
        if _looks_like_reference_running_line(stripped):
            continue
        if stripped.startswith("```"):
            continue
        if stripped == "$$":
            continue
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            stripped = formula_to_plain_text(stripped[2:-2].strip())
        elif stripped.startswith("$") and stripped.endswith("$") and stripped.count("$") == 2 and len(stripped) > 2:
            stripped = formula_to_plain_text(stripped[1:-1].strip())
        elif "$" in stripped:
            stripped = re.sub(r"\$([^$]+)\$", lambda m: formula_to_plain_text(m.group(1)), stripped)
            stripped = re.sub(r"\$\$([^$]+)\$\$", lambda m: formula_to_plain_text(m.group(1)), stripped)
            stripped = stripped.replace("$", "").strip()
        if not stripped:
            continue

        for piece in _INLINE_PAGE_MARKER_RE.split(stripped):
            piece = str(piece or "").strip()
            if not piece:
                continue
            if _PAGE_MARKER_LINE_RE.match(piece):
                if current_ref:
                    flush_current()
                formatted.append(piece)
                continue
            for fragment in _split_author_year_collapsed_line(piece):
                if not fragment:
                    continue
                if not current_ref and not _line_looks_like_author_year_entry_start(fragment):
                    continue
                if current_ref and has_year(current_ref) and _line_looks_like_author_year_entry_start(fragment):
                    flush_current()
                current_ref.append(fragment)

    if current_ref:
        flush_current()

    return formatted


def format_single_reference(text: str, num: int) -> str:
    """Format a single reference with proper numbering."""
    text = _join_reference_fragments([text])
    text = re.sub(r"\$([^$]+)\$", lambda m: formula_to_plain_text(m.group(1)), text)
    text = re.sub(r"\$\$([^$]+)\$\$", lambda m: formula_to_plain_text(m.group(1)), text)

    leading_number = re.match(r"^\[?(\d+)\]?\s+", text)
    if leading_number and _is_plausible_reference_number(leading_number.group(1)):
        return re.sub(r"^\[?(\d+)\]?\s+", r"[\1] ", text)

    return f"[{num}] {text}"
