from __future__ import annotations

import re
from kb.citation_sentences import citation_sentence_spans


IID_PATTERN = r"\bi\.\s*i\.\s*d\.|\biid\b|\bindependent(?:ly)?\s+and\s+identically\s+distributed\b|独立同分布"
_PAIR = re.compile(r"\b([A-Z])\s*(?:,\s*|and\s+|和|与)\s*([A-Z])\b")
_PIXEL_SCOPE = re.compile(r"(?:同一幅|单幅|图内|像素).{0,32}分量|pixel[- ]wise|noise\s+components", re.I)
_RAW_NOISE = re.compile(
    r"(?:raw\s+(?:network\s+)?output|uncorrected\s+output)"
    r"(?:\s+of\s+(?:the|our)\s+network)?\s+(?:is|remains?)\s+"
    r"(?P<state>not\s+(?:still\s+)?noisy|(?:still\s+)?noisy|not\s+(?:a\s+)?clean(?:\s+image)?)", re.I,
)


def _negated_relation_context(text: str, match: re.Match[str]) -> bool:
    start = next((a for a, b in citation_sentence_spans(text) if a <= match.start() < b), 0)
    prefix = text[max(start, match.start() - 80):match.start()]
    return bool(re.search(r"\b(?:not|never|cannot|false)\b|并非|不(?:是|要求|需要)", prefix, re.I))


def _relation_pair(text: str) -> re.Match[str] | None:
    normalized = re.sub(r"\\(?:boldsymbol|mathbf|mathcal|mathrm)\s*\{([A-Z])\}", r"\1", text)
    return _PAIR.search(normalized.replace('$', ''))


def iid_relation_covered(claim: str, evidence: str) -> bool:
    requested = re.search(IID_PATTERN, claim, re.I)
    if not requested:
        return True
    # Negated/qualified assertions cannot gain positive support through an
    # abbreviation alias. Ordinary polarity checks still run independently.
    if _negated_relation_context(claim, requested):
        return False
    claim_pair = _relation_pair(claim)
    sample_scope = re.search(r"额外|合成|生成|\b(?:additional|synthetic|generated)\b", claim, re.I)
    spans = citation_sentence_spans(evidence)
    for actual in re.finditer(IID_PATTERN, evidence, re.I):
        if _negated_relation_context(evidence, actual):
            continue
        start, end = next(((a, b) for a, b in spans if a <= actual.start() < b), (0, len(evidence)))
        statement = evidence[start:end]
        if _PIXEL_SCOPE.search(claim) and not _PIXEL_SCOPE.search(statement):
            continue
        source_pair = _relation_pair(statement)
        if claim_pair and (not source_pair or set(claim_pair.groups()) != set(source_pair.groups())):
            continue
        if sample_scope and not (source_pair or re.search(
            rf"(?:two|both)\s+(?:(?:{IID_PATTERN})\s+)?(?:noise\s+)?(?:samples|draws)", statement, re.I,
        )):
            continue
        return True
    return False


def raw_output_noise_covered(claim: str, evidence: str) -> bool:
    wanted = _RAW_NOISE.search(claim)
    if not wanted:
        return True
    supplied = _RAW_NOISE.search(evidence)
    if not supplied or any(m['state'].lower().startswith(('not noisy','not still noisy')) for m in (wanted,supplied)):
        return False
    return 'clean' in wanted['state'].lower() or 'clean' not in supplied['state'].lower()


def sampling_ability_covered(claim: str, evidence: str) -> bool:
    """Not needing a sampling prior does not establish inability to sample."""
    sampling = r"sampl\w*\s+from\s+(?:the\s+)?noise\s+distribution|从噪声分布中?采样"
    inability = r"不具备.{0,16}(?:能力|这一点)|无法.{0,25}采样|\b(?:cannot|unable\s+to)\b.{0,45}sampl|\b(?:lack\w*|not\s+(?:have|possess))\b.{0,25}ability"
    if not (re.search(sampling, claim, re.I) and re.search(inability, claim, re.I)):
        return True
    return bool(re.search(
        r"\b(?:cannot|unable\s+to)\b[^.;\n]{0,45}sampl|"
        r"\b(?:lack\w*|not\s+(?:have|possess))\s+(?:the\s+)?ability\s+to\s+sampl|"
        r"(?:不具备|没有|缺乏)[^。；\n]{0,30}采样能力|无法从噪声分布中?采样",
        evidence, re.I,
    ))
