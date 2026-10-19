from __future__ import annotations

import re


def named_metrics_covered(claim: str, evidence: str) -> bool:
    return all(not re.search(rf'\b{metric}\b', claim, re.I) or re.search(rf'\b{metric}\b', evidence, re.I) for metric in ('PSNR', 'SSIM', 'LPIPS'))


def qualitative_outperformance_covered(claim: str, evidence: str) -> bool:
    """A baseline list does not itself assert an experimental improvement."""
    requested = list(re.finditer(r'outperform\w*|优于|胜过', claim, re.I))
    if not requested:
        return True

    def positive(text: str, match: re.Match) -> bool:
        prefix = re.split(r'[.;。\n]|\b(?:but|whereas)\b', text[:match.end()], flags=re.I)[-1][-96:]
        prefix = re.sub(r'\bnot\s+only\b', '', prefix, flags=re.I)
        return not re.search(r'\b(?:not|never|cannot|unable)\b|不|未|无法|不能', prefix, re.I)

    supplied = list(re.finditer(r'outperform\w*|(?:our\s+(?:method|approach)|this\s+method|we)[^.;\n]{0,80}(?:better\s+than|superior\s+to)|优于|胜过', evidence, re.I))
    return all(any(positive(claim, wanted) == positive(evidence, actual) for actual in supplied) for wanted in requested)
