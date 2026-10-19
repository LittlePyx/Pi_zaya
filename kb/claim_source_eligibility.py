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
        r'(?:establish|confirm|determine|prove|support|provide|show|specify|state|describe|detail|report|give|contain|include)\b|'
        r'(?:the\s+)?(?:current\s+)?(?:retrieved|cited|selected|provided)\s+'
        r'(?:evidence|snippets?|excerpts?|passages?)\s+(?:contains?|includes?|reports?)\s+no\b|'
        r'no\s+(?:retrieved|cited|selected|provided)\s+(?:snippet|excerpt|passage)\s+'
        r'(?:reports?|shows?|provides?|states?|describes?)\b|'
        r'(?:that\s+difference\s+alone\s+means|this\s+means)\s+[^.!?]{0,350}?not\s+'
        r'something\s+(?:these|the\s+(?:retrieved|selected|cited))\s+(?:snippets|excerpts|passages)\s+can\s+'
        r'(?:demonstrate|show|establish)\b|'
        r'this\s+is\s+(?:a|the)\s+boundary\s+of\s+(?:the\s+)?(?:retrieved|selected|cited)\s+(?:snippets|excerpts|passages|evidence)\b|'
        r'absence\s+of\s+[^.!?]{0,160}\s+in\s+(?:these|the\s+(?:retrieved|selected|cited))\s+'
        r'(?:snippets|excerpts|passages)\s+does\s+not\s+(?:prove|establish|show)\b|'
        r'(?:当前|本轮)(?:引用|检索)(?:到的)?(?:证据|片段|内容|结果)'
        r'(?:并)?(?:无法|不能|不足以|未直接|未|没有))', plain, re.I,
    ))


def incomplete_source_prose(evidence: str) -> bool:
    """Trailing logical connectives signal a missing continuation, not proof."""
    return bool(re.search(
        r'\band\s+(?:therefore|thus|hence)(?:\s+and\s+(?:therefore|thus|hence))?\s*[.,]?\s*$',
        str(evidence or ''), re.I,
    ))


def omitted_derivation_covered(claim: str, evidence: str) -> bool:
    """Only an explicit omission statement can establish a missing proof."""
    def explicit_omission(text: str) -> bool:
        for sentence in re.split(r'[;。!?！？；\n]', text):
            if re.search(r"\b(?:do(?:es)?\s+not|never)\s+omit|\bnot\s+omitted|(?:并未|没有)省略", sentence, re.I):
                continue
            missing = r'(?:未|没有|不)(?:展示|给出|提供|包含|涉及)|\b(?:not\s+(?:show\w*|provid\w*|present\w*|includ\w*)|omit\w*)\b'
            proof = r'推导|证明|\b(?:proof|derivation)\b'
            if re.search(rf'(?:{missing})[^.;。!?]{{0,60}}(?:{proof})|(?:{proof})[^.;。!?]{{0,60}}(?:{missing})', sentence, re.I):
                return True
        return False
    return not explicit_omission(claim) or explicit_omission(evidence)


def complete_source_prefix(evidence: str) -> str:
    """Keep an existing complete statement before a lost mathematical continuation."""
    text=str(evidence or '')
    match=re.search(r'\s+and\s+(?:therefore|thus|hence)(?:\s+and\s+(?:therefore|thus|hence))?\s*[.,]?\s*$',text,re.I)
    prefix=text[:match.start()].rstrip() if match else text
    return prefix if re.search(r'\b(?:is|are|was|were)\b',prefix,re.I) and prefix.endswith('.') else text
