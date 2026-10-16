from __future__ import annotations

import re
from collections.abc import Callable


# Protect literal notation before presentation-only Markdown cleanup. This does
# not interpret, repair, or add mathematics absent from the source.
_MATH_RE = re.compile(
    r"\$\$[^$]{1,1200}\$\$|(?<![\\$])\$(?!\$)[^$\n]{1,1200}\$(?!\$)|"
    r"\\\([^\n]{1,1200}?\\\)|\\\[[^\n]{1,1200}?\\\]|"
    r"\|\|[^|\n]{1,160}\|\||"
    r"\b[A-Za-z]{1,3}\s*(?:\([^()\n]*\|[^()\n]*\)|\[[^\[\]\n]*\|[^\[\]\n]*\])|"
    r"(?<![\w|])\|[^|\s]{1,120}\|(?![\w|])"
)


def protect_display_math(value: str) -> tuple[str, Callable[[str], str]]:
    literals: list[str] = []
    prefix = "KBMATHLITERAL"
    while prefix in value:
        prefix += "X"

    def protect(match: re.Match[str]) -> str:
        literal = match.group(0)
        if re.fullmatch(r"\$\s*[A-Za-z0-9]+\s*\$", literal):
            # Plain inline identifiers keep the existing readable prose form;
            # preserve delimiters when notation carries operators or styling.
            return literal
        if literal.startswith("|") and not re.search(r"[A-Za-z0-9\\]", literal):
            return literal
        if literal.startswith("||") or ("|" in literal and not literal.startswith(("$", "\\", "|"))):
            # Bare bar notation must have symbolic operands, not prose/table
            # cells such as "Options (red | blue)" or "|| Method ||".
            operands = re.sub(r"\\[A-Za-z]+|[_^]\{[^}]*\}", "", literal)
            if re.search(r"[A-Za-z]{2,}", operands):
                return literal
        token = f"{prefix}{len(literals)}END"
        literals.append(literal)
        return token

    protected = _MATH_RE.sub(protect, value)

    def restore(text: str) -> str:
        for index, literal in enumerate(literals):
            text = text.replace(f"{prefix}{index}END", literal)
        return text

    return protected, restore
