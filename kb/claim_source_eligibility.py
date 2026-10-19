from __future__ import annotations

import re


def retrieval_limit_claim(claim: str) -> bool:
    """An evidence-window limitation cannot be supported as a paper assertion."""
    plain = re.sub(r'^[\s*#>\-]+', '', str(claim or ''))
    return bool(re.match(
        r'(?:not\s+(?:directly\s+)?(?:stated|specified|provided|shown|reported)\s+in\s+'
        r'(?:the\s+)?(?:(?:current|currently|selected)\s+)?(?:retrieved|cited)\s+'
        r'(?:evidence|snippets?|excerpts?|passages?)\s*[:：]|'
        r'(?:the\s+)?(?:(?:current|currently)\s+)?(?:retrieved|cited|selected|provided)\s+'
        r'(?:(?:retrieved|cited|selected|provided)\s+)?(?:evidence|snippets?|excerpts?|passages?)\s+'
        r'(?:cannot|can\x27t|does\s+not|do\s+not|doesn\x27t|don\x27t)\s+'
        r'(?:establish|confirm|determine|prove|support|provide|show|specify|state|describe|report|give)\b|'
        r'(?:当前|本轮)(?:引用|检索)(?:到的)?(?:证据|片段|内容|结果)'
        r'(?:并)?(?:无法|不能|不足以|未直接|未|没有))', plain, re.I,
    ))


def incomplete_source_prose(evidence: str) -> bool:
    """Trailing logical connectives signal a missing continuation, not proof."""
    return bool(re.search(
        r'\band\s+(?:therefore|thus|hence)(?:\s+and\s+(?:therefore|thus|hence))?\s*[.,]?\s*$',
        str(evidence or ''), re.I,
    ))


def complete_source_prefix(evidence: str) -> str:
    """Keep an existing complete statement before a lost mathematical continuation."""
    text=str(evidence or '')
    match=re.search(r'\s+and\s+(?:therefore|thus|hence)(?:\s+and\s+(?:therefore|thus|hence))?\s*[.,]?\s*$',text,re.I)
    prefix=text[:match.start()].rstrip() if match else text
    return prefix if re.search(r'\b(?:is|are|was|were)\b',prefix,re.I) and prefix.endswith('.') else text
