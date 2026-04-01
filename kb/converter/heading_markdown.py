from __future__ import annotations

import re


_UNICODE_MATHISH_RE = re.compile(r"[\u0370-\u03FF\u2200-\u22FF]", re.IGNORECASE)


def fix_heading_structure(md: str) -> str:
    """Fix heading hierarchy to ensure proper structure."""
    lines = md.splitlines()
    out = []
    heading_stack = [0]

    for line in lines:
        stripped = line.strip()
        match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if not match:
            out.append(line)
            continue

        level = len(match.group(1))
        text = match.group(2).strip()

        if (
            re.search(r"^\s*[A-Z]?\s*\d+\s*[a-z]", text)
            or re.search(r"^\s*\d+\s*[a-z]+\s*[+\-]", text)
            or (len(text) <= 15 and re.search(r"\d+.*[a-z]|[a-z].*\d+", text) and "=" in text)
            or _UNICODE_MATHISH_RE.search(text)
            or (len(text) <= 20 and re.search(r"[+\-*/^_{}\[\]()]", text) and not re.search(r"[A-Z]{3,}", text))
        ):
            out.append(f"$$\n{text}\n$$")
            continue

        if level > heading_stack[-1] + 1:
            level = heading_stack[-1] + 1

        while heading_stack and heading_stack[-1] >= level:
            heading_stack.pop()
        heading_stack.append(level)

        if len(text) > 200:
            out.append(text)
        else:
            out.append("#" * level + " " + text)

    return "\n".join(out)
