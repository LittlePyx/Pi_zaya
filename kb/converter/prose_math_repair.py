from __future__ import annotations

from pathlib import Path
import re

from .pdf_inline_prose import recover_inline_prose_fragments
from .quality_acceptance import is_prose_dominant_display_math_block

DISPLAY_MATH_DELIMITER_RE = re.compile(r"^\s*\$\$\s*$")


def _unwrap_prose_dominant_display_math(md: str, pdf_path: Path | str | None = None) -> tuple[str, bool]:
    """Remove display delimiters only from blocks classified as prose-heavy.

    The block body is preserved exactly so mixed prose/equation evidence is
    never discarded. Genuine equation-only blocks retain their delimiters.
    """

    original = str(md or "")
    structured, _ = recover_inline_prose_fragments(original, pdf_path) if pdf_path else (original, False)
    lines = structured.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        if not DISPLAY_MATH_DELIMITER_RE.match(lines[index]):
            out.append(lines[index])
            index += 1
            continue
        end = index + 1
        while end < len(lines) and not DISPLAY_MATH_DELIMITER_RE.match(lines[end]):
            end += 1
        if end >= len(lines):
            out.extend(lines[index:])
            break
        block_lines = lines[index + 1 : end]
        if is_prose_dominant_display_math_block("\n".join(block_lines)):
            out.extend(block_lines)
        else:
            out.extend(lines[index : end + 1])
        index = end + 1
    repaired = "\n".join(out)
    if original.endswith("\n"):
        repaired += "\n"
    return repaired, repaired != original
