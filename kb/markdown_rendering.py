from __future__ import annotations

"""
Backend-facing markdown rendering helpers.

The current implementations are shared with the retired Streamlit UI module.
Keep backend code depending on this module so those helpers can be moved here
without changing API or generation code again.
"""

import re

from ui.chat_widgets import (
    _md_to_plain_text,
    _normalize_copy_citation_links,
    _normalize_math_markdown,
)


_SIGNED_BINARY_VECTOR_RE = re.compile(
    r"\[\s*(?P<left>[+-]?1)\s*[,，]\s*(?P<right>[+-]?1)\s*\]"
)


def normalize_signed_binary_vectors(markdown: str) -> str:
    """Keep ``[1, -1]`` bit alphabets out of numeric-citation parsing."""

    def _replace_prose(value: str) -> str:
        def _replace(match: re.Match) -> str:
            left = str(match.group("left") or "")
            right = str(match.group("right") or "")
            if not (left.startswith("-") or right.startswith("-")):
                return match.group(0)
            left = left if left.startswith(("+", "-")) else f"+{left}"
            right = right if right.startswith(("+", "-")) else f"+{right}"
            return f"({left}, {right})"

        return _SIGNED_BINARY_VECTOR_RE.sub(_replace, value)

    lines: list[str] = []
    in_fence = False
    for raw_line in str(markdown or "").splitlines(keepends=True):
        if raw_line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            lines.append(raw_line)
            continue
        if in_fence or "[" not in raw_line:
            lines.append(raw_line)
            continue
        inline_parts = re.split(r"(`+[^`\n]*?`+)", raw_line)
        lines.append(
            "".join(
                part if index % 2 else _replace_prose(part)
                for index, part in enumerate(inline_parts)
            )
        )
    return "".join(lines)


__all__ = [
    "_md_to_plain_text",
    "_normalize_copy_citation_links",
    "_normalize_math_markdown",
    "normalize_signed_binary_vectors",
]
