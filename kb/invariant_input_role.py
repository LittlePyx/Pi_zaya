from __future__ import annotations

import re

from kb.citation_marker_context import numeric_marker_is_content


_RELATION = re.compile(
    r"\bSince\s+(?P<function>[A-Za-z])\s+has\s+to\s+use\s+information\s+from\s+outside\s+of\s+"
    r"each\s+subset\s+of\s+dimensions\s+(?P<outside>[A-Za-z])\s+to\s+predict\s+the\s+values\s+"
    r"inside\s+of\s+(?P<inside>[A-Za-z])\b",
    re.I,
)


def clarify_supported_input_role(answer: str, hits: list[dict]) -> tuple[str, int]:
    """Name a function argument's role without changing its subset dependency.

    Require the cited source's outside-to-inside relation for the same function
    and subset. Do not edit source quotations, code, equations or uncited prose.
    """
    output = []
    count = 0
    in_fence = False
    for line in answer.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        notation = re.search(
            r"\b([A-Za-z])\s*\(\s*([A-Za-z])\s*\)\s*_\s*(?:\{\s*([A-Za-z])\s*\}|([A-Za-z])(?![A-Za-z]))",
            line,
        )
        if (
            in_fence
            or "`" in line
            or stripped.startswith((">", "|", "~~~", '"', "'", "“", "‘"))
            or notation is None
        ):
            output.append(line)
            continue
        function, argument = notation[1], notation[2]
        subset = notation[3] or notation[4]
        sources = [
            hits[int(m[1]) - 1].get("text", "")
            for m in re.finditer(r"(?<!\[)\[(\d{1,4})\](?!\])", line)
            if 0 < int(m[1]) <= len(hits)
            and not numeric_marker_is_content(line, m.start())
        ]
        supported = any(
            m["function"] == function and m["outside"] == m["inside"] == subset
            for text in sources
            for m in _RELATION.finditer(re.sub(r"[$}{]", "", str(text)))
        )
        if supported:
            pattern = re.compile(
                r"(?P<prefix>(?i:does\s+not\s+depend\s+on\s+(?:the\s+)?))(?P<value>(?i:value))"
                rf"(?P<rest>\s+(?i:of)\s+\$?\s*{argument}\s*_\s*(?:\{{\s*{subset}\s*\}}|{subset})\s*\$?)"
            )

            def replace(match: re.Match) -> str:
                nonlocal count
                if numeric_marker_is_content(line, match.start()):
                    return match[0]
                count += 1
                return match["prefix"] + "input " + match["value"] + match["rest"]

            line = pattern.sub(replace, line)
        output.append(line)
    return "".join(output), count
