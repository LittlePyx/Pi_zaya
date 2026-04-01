from __future__ import annotations

import re

from .formula_markdown import formula_to_plain_text
from .text_utils import _normalize_text


_REF_HEADING_RE = re.compile(r"^#+\s+References?\s*$|^References?\s*$", re.IGNORECASE)
_REF_START_INLINE_RE = re.compile(r"^\[?\d+\]?[.)]?\s+")
_REF_BACKREF_SUFFIX_RE = re.compile(
    r"(\b(?:19|20)\d{2}\.)(?:\s+\d{1,3}\s*(?:,\s*\d{1,3}){0,12})\s*$"
)
_YEAR_BACKREF_LINE_RE = re.compile(
    r"^(?P<year>(?:18|19|20)\d{2})\.\s+\d{1,3}(?:\s*,\s*\d{1,3}){0,12}\s*$"
)


def _is_reference_start_line(text: str) -> bool:
    src = str(text or "").strip()
    if _YEAR_BACKREF_LINE_RE.match(src):
        return False
    return bool(_REF_START_INLINE_RE.match(src))


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
    saw_heading = False
    for raw in str(page_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = _normalize_text(raw or "").strip()
        if not line:
            lines_out.append("")
            continue
        if _REF_HEADING_RE.match(line):
            saw_heading = True
            continue
        if re.fullmatch(r"\d{1,4}", line):
            continue
        if re.search(r"\bpage\s+\d+\s+of\s+\d+\b", line, flags=re.IGNORECASE):
            continue
        if "www." in line.lower():
            continue
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
    formatted = []
    current_ref = []
    ref_num = 1
    current_ref_number: int | None = None
    in_code_block = False
    in_display_math = False

    for idx, (_, line) in enumerate(ref_lines):
        stripped = line.strip()

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

        ref_match = re.match(r"^(\[?\d+\]?)[\.\s]+(.+)$", stripped)
        if ref_match:
            if current_ref:
                ref_text = _join_reference_fragments(current_ref)
                if ref_text:
                    out_num = int(current_ref_number or ref_num)
                    formatted.append(format_single_reference(ref_text, out_num))
                    ref_num = out_num + 1
                current_ref = []
                current_ref_number = None

            ref_content = ref_match.group(2).strip()
            raw_num = str(ref_match.group(1) or "").strip().strip("[]")
            try:
                current_ref_number = int(raw_num)
            except Exception:
                current_ref_number = None
            if ref_content:
                current_ref.append(ref_content)
        else:
            mid_ref_match = re.search(r"\s+(\[?\d+\]?)[\.\s]+([A-Z][^.]{10,})", stripped)
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
                raw_num = str(mid_ref_match.group(1) or "").strip().strip("[]")
                try:
                    current_ref_number = int(raw_num)
                except Exception:
                    current_ref_number = None
                current_ref = [mid_ref_match.group(2).strip()]
            else:
                current_ref.append(stripped)

    if current_ref:
        ref_text = _join_reference_fragments(current_ref)
        if ref_text:
            out_num = int(current_ref_number or ref_num)
            formatted.append(format_single_reference(ref_text, out_num))

    return formatted


def format_single_reference(text: str, num: int) -> str:
    """Format a single reference with proper numbering."""
    text = _join_reference_fragments([text])
    text = re.sub(r"\$([^$]+)\$", lambda m: formula_to_plain_text(m.group(1)), text)
    text = re.sub(r"\$\$([^$]+)\$\$", lambda m: formula_to_plain_text(m.group(1)), text)

    if re.match(r"^\[?\d+\]?\s+", text):
        return re.sub(r"^\[?(\d+)\]?\s+", r"[\1] ", text)

    return f"[{num}] {text}"
