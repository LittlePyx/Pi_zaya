from __future__ import annotations

import re


IID_PATTERN = r"\bi\.\s*i\.\s*d\.|\biid\b|\bindependent(?:ly)?\s+and\s+identically\s+distributed\b|独立同分布"
_PAIR = re.compile(r"\b([A-Z])\s*(?:,\s*|and\s+|和|与)\s*([A-Z])\b")
_RAW_NOISE = re.compile(
    r"(?:raw\s+(?:network\s+)?output|uncorrected\s+output)"
    r"(?:\s+of\s+(?:the|our)\s+network)?\s+(?:is|remains?)\s+"
    r"(?P<state>not\s+(?:still\s+)?noisy|(?:still\s+)?noisy|not\s+(?:a\s+)?clean(?:\s+image)?)", re.I,
)


def iid_relation_covered(claim: str, evidence: str) -> bool:
    requested = re.search(IID_PATTERN, claim, re.I)
    if not requested:
        return True
    actual = re.search(IID_PATTERN, evidence, re.I)
    if not actual:
        return False
    # Negated/qualified assertions cannot gain positive support through an
    # abbreviation alias. Ordinary polarity checks still run independently.
    if any(re.search(r"\b(?:not|never|cannot|false)\b|并非|不(?:是|要求|需要)", text[max(0, match.start()-80):match.start()], re.I)
           for text,match in ((claim,requested),(evidence,actual))):
        return False
    claim_pair = _PAIR.search(claim.replace('$',''))
    source_pair = _PAIR.search(evidence.replace('$',''))
    if claim_pair:
        return bool(source_pair and set(claim_pair.groups())==set(source_pair.groups()))
    sample_scope = re.search(r"额外|合成|生成|\b(?:additional|synthetic|generated)\b", claim, re.I)
    if sample_scope:
        return bool(source_pair or re.search(r"(?:two|both)\s+(?:noise\s+)?(?:samples|draws)",evidence,re.I))
    return True


def raw_output_noise_covered(claim: str, evidence: str) -> bool:
    wanted = _RAW_NOISE.search(claim)
    if not wanted:
        return True
    supplied = _RAW_NOISE.search(evidence)
    if not supplied or any(m['state'].lower().startswith(('not noisy','not still noisy')) for m in (wanted,supplied)):
        return False
    return 'clean' in wanted['state'].lower() or 'clean' not in supplied['state'].lower()
