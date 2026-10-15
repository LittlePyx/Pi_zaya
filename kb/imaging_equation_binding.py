from __future__ import annotations

import re


def has_masked_frame_sum(evidence: str) -> bool:
    """Recognize an explicit sum of element-wise products, not mask-after-sum."""
    surface = str(evidence or "")
    if not re.search(r"\bmask[s]?\b|掩[模码膜]", surface, re.I):
        return False
    for body in re.findall(r"\$\$(.*?)\$\$|\\\[(.*?)\\\]", surface, re.S):
        equation = next((part for part in body if part), "")
        equation = re.sub(r"\\(?:mathbf|mathrm|mathit|boldsymbol)\{([^{}]+)\}", r"\1", equation)
        match = re.search(r"\\sum(?![A-Za-z])", equation)
        if not match:
            continue
        tail = equation[match.end():]
        tail = re.sub(r"^\s*(?:(?:_\{[^{}]*\}|\^\{[^{}]*\}|_\w|\^\w)\s*)+", "", tail)
        operand = re.split(r"[+);=]", tail, maxsplit=1)[0]
        if r"\odot" in operand:
            return True
    return False
