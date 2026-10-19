from __future__ import annotations

import re


_REQUIREMENTS = (
    r'statistical\s+model\s+of\s+(?:the\s+)?noise\s+distribution|噪声分布(?:的)?统计模型',
    r'(?:single|one)\s+noisy\s+reali[sz]ation|(?:单个|一个)(?:的)?(?:含噪|带噪|噪声)(?:实现|观测)|(?:一份|单次)(?:含噪|带噪)观测',
)


def noise_model_prerequisites_covered(claim: str, evidence: str) -> bool:
    """Augmenting inputs or listing noise types does not state data/model needs."""
    return all(not re.search(pattern, claim, re.I) or re.search(pattern, evidence, re.I)
               for pattern in _REQUIREMENTS)


def objective_term_covered(claim: str, evidence: str) -> bool:
    """A variable definition alone does not locate a term inside an objective."""
    if not re.search(r'constant\s+term|常数项', claim, re.I) or not re.search(r'loss|objective|损失|目标函数', claim, re.I):
        return True
    return bool(re.search(r'loss|objective|损失|目标函数|\\(?:sum|min)\b', evidence, re.I))
