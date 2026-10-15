from __future__ import annotations

import re
from collections.abc import Iterator


def source_equation_literals(text: str) -> Iterator[str]:
    """Read explicit displays and plain-text finite-set assignments literally.

    The converter can preserve an inline set assignment as prose, with only its
    styled left-hand symbol inside dollars. Escape the balanced *plain-text*
    braces as visible set delimiters; do not reinterpret braces inside TeX math.
    No variables, set members, or missing delimiters are inferred.
    """
    yield from (match[1] for match in re.finditer(r"\$\$(.*?)\$\$", text, re.S))
    prose = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    pattern = re.compile(r"(?<![A-Za-z\\])(?P<lhs>\$\\mathcal\{[A-Za-z]\}\$)\s*=\s*(?P<set>\{)")
    for match in pattern.finditer(prose):
        start = match.start("set")
        depth = 0
        for end in range(start, min(len(prose), start + 240)):
            char = prose[end]
            if not re.fullmatch(r"[{}A-Za-z0-9,.\s+\-]", char):
                break
            depth += (char == "{") - (char == "}")
            if depth == 0:
                rhs = prose[start:end + 1]
                # Restrict this path to enumerations, rather than prose in braces.
                if "," in rhs and not re.search(r"[A-Za-z]{2,}", rhs):
                    rhs = rhs.replace("{", r"\{").replace("}", r"\}")
                    yield match["lhs"].strip("$") + "=" + rhs
                break
