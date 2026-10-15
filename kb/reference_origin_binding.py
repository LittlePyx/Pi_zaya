from __future__ import annotations

import re


def origin_topic_identifier(topic: str) -> str:
    aliases = re.findall(r"\(([A-Z][A-Za-z0-9-]{1,24})\)", topic)
    return aliases[0] if len(aliases) == 1 else topic


def origin_sentence_score(line: str, topic: str) -> float:
    if not re.search(rf"(?i)(?<![A-Za-z0-9]){re.escape(topic)}(?![A-Za-z0-9])", line):
        return float("-inf")
    score = 1.0
    prior = re.search(
        r"(?i)upstream\s+paper|prior\s+work|previous\s+work|comes?\s+from|"
        r"introduced\s+by|proposed\s+by|did\s+not\s+invent|reuses?.{0,36}(?:prior|previous)|"
        r"上游|先前工作|已有(?:工作|方法|技术)|既有(?:优化)?(?:方法|工作|技术)|来自|提出者|(?:不是|并非).{0,36}(?:发明|原创)",
        line,
    )
    if prior:
        score += 12.0
    if re.search(r"(?i)(?:不是|并非).{0,36}(?:发明|原创)|\b(?:did|does|is)\s+not.{0,24}(?:invent|original)", line):
        score += 8.0
    if re.search(r"(?i)\bet\s+al\.", line):
        score += 4.0
    if re.search(r"(?i)open[- ]domain\s+question\s+answering", line):
        score += 5.0
    return score
