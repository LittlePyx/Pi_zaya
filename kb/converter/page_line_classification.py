from __future__ import annotations

import re


def looks_like_mixed_math_prose_line(text: str) -> bool:
    """Identify prose sentences that contain a short inline equation.

    Treating the whole line as display math makes the equation crop discard the
    surrounding condition, especially in two-column proposition statements.
    """

    value = re.sub(r"\s+", " ", str(text or "")).strip()
    words = re.findall(r"\b[A-Za-z]{2,}\b", value)
    equation_anchors = re.findall(r"=|∈|≤|≥|\\(?:sum|int|frac|sqrt)\b", value)
    return bool(
        len(words) >= 8
        and 1 <= len(equation_anchors) <= 4
        and re.search(r"[,;:]", value)
        and re.search(r"\b(?:and|where|suppose|such that|conditional|provided|when|if)\b", value, re.I)
    )
