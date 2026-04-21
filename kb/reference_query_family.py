from __future__ import annotations

import re


_MULTI_PAPER_LIST_PATTERNS = (
    "which papers",
    "what papers",
    "what other papers",
    "other papers",
    "papers in my library",
    "papers in your library",
    "\u6709\u54ea\u4e9b\u6587\u7ae0",
    "\u6709\u54ea\u51e0\u7bc7\u6587\u7ae0",
    "\u54ea\u4e9b\u6587\u7ae0",
    "\u54ea\u4e9b\u8bba\u6587",
    "\u6709\u54ea\u51e0\u7bc7\u8bba\u6587",
    "\u54ea\u51e0\u7bc7\u6587\u7ae0",
    "\u54ea\u51e0\u7bc7\u8bba\u6587",
    "\u54ea\u4e9b\u6587\u732e",
    "\u54ea\u51e0\u7bc7\u6587\u732e",
)

_SINGLE_PAPER_PICK_PATTERNS = (
    "which paper in my library",
    "which paper in your library",
    "what paper in my library",
    "what paper in your library",
    "\u54ea\u7bc7\u6587\u7ae0",
    "\u54ea\u7bc7\u8bba\u6587",
    "\u54ea\u7bc7\u6587\u732e",
)

_REFERENCE_COMPARE_PATTERNS = (
    r"\b(compare|compares|compared|comparison|versus|vs\.?)\b",
    r"\u6bd4\u8f83|\u5bf9\u6bd4",
)

_REFERENCE_DEFINITION_PATTERNS = (
    r"\b(defin(?:e|es|ed|ition)|what\s+is|introduced?\s+as)\b",
    r"\u5b9a\u4e49|\u662f\u4ec0\u4e48|\u600e\u4e48\u5b9a\u4e49",
)

_REFERENCE_DISCUSSION_PATTERNS = (
    r"\b(discuss(?:es|ed)?|mention(?:s|ed)?|cover(?:s|ed)?|address(?:es|ed)?|describe(?:s|d)?)\b",
    r"\u63d0\u5230|\u63d0\u53ca|\u8ba8\u8bba|\u6d89\u53ca|\u4ecb\u7ecd",
)

_REFERENCE_LOCATE_PATTERNS = (
    r"\b(where\s+(?:is|was|are)|point\s+me|locate|source\s+section)\b",
    r"\u54ea\u91cc|\u54ea\u4e2a\u7ae0\u8282|\u5b9a\u4f4d|\u51fa\u5904|\u6e90\u7ae0\u8282",
)


def _prompt_matches_any_pattern(prompt: str, patterns: tuple[str, ...]) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def prompt_explicitly_requests_multi_paper_list(prompt: str) -> bool:
    text = str(prompt or "").strip().lower()
    if not text:
        return False
    if any(pat in text for pat in _MULTI_PAPER_LIST_PATTERNS):
        return True
    return bool(
        re.search(
            r"\bwhich\s+papers\b|\bwhat\s+papers\b|\bwhich\s+other\s+papers\b|\bwhat\s+other\s+papers\b|\bother\s+papers\b",
            text,
            flags=re.I,
        )
    )


def prompt_explicitly_requests_single_paper_pick(prompt: str) -> bool:
    text = str(prompt or "").strip().lower()
    if not text:
        return False
    if prompt_explicitly_requests_multi_paper_list(text):
        return False
    if any(pat in text for pat in _SINGLE_PAPER_PICK_PATTERNS):
        return True
    return bool(re.search(r"\bwhich\s+paper\b|\bwhat\s+paper\b", text, flags=re.I))


def prompt_prefers_zh(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    zh_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_word_count = len(re.findall(r"[A-Za-z]{2,}", text))
    return bool(zh_count > 0 and zh_count >= max(2, ascii_word_count))


def extract_multi_paper_topic(prompt: str) -> str:
    text = str(prompt or "").strip()
    if not text:
        return ""
    patterns = (
        r"(?:\u63d0\u5230(?:\u4e86)?|\u63d0\u53ca(?:\u4e86)?|\u8ba8\u8bba(?:\u4e86)?|\u6d89\u53ca(?:\u4e86)?|\u5b9a\u4e49(?:\u4e86)?)\s*(.+?)(?:[\uff0c\u3002\uff1f?]|$)",
        r"(?:mention(?:s|ed)?|discuss(?:es|ed)?|define(?:s|d)?)\s+(.+?)(?:[?.,]|$)",
    )
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if not m:
            continue
        topic = re.sub(r"\s+", " ", str(m.group(1) or "").strip(" \uff0c\uff1b,.\u3002\uff1f?!\uff1a:"))
        if topic:
            return topic
    return ""


def prompt_requests_reference_compare(prompt: str) -> bool:
    return _prompt_matches_any_pattern(prompt, _REFERENCE_COMPARE_PATTERNS)


def prompt_requests_reference_definition(prompt: str) -> bool:
    return _prompt_matches_any_pattern(prompt, _REFERENCE_DEFINITION_PATTERNS)


def prompt_requests_reference_discussion(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    if prompt_requests_reference_compare(text) or prompt_requests_reference_definition(text):
        return True
    if prompt_explicitly_requests_single_paper_pick(text) or prompt_explicitly_requests_multi_paper_list(text):
        return True
    if _prompt_matches_any_pattern(text, _REFERENCE_LOCATE_PATTERNS):
        return True
    return _prompt_matches_any_pattern(text, _REFERENCE_DISCUSSION_PATTERNS)


def prompt_reference_focus_action(prompt: str) -> str:
    text = str(prompt or "").strip()
    if not text:
        return ""
    if prompt_requests_reference_compare(text):
        return "compare"
    if prompt_requests_reference_definition(text):
        return "define"
    if prompt_requests_reference_discussion(text):
        return "discuss"
    return ""


def prompt_requires_reference_focus_match(prompt: str) -> bool:
    return bool(prompt_reference_focus_action(prompt))


def prompt_targets_sci_topic(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    return bool(
        re.search(
            r"(?:(?<![A-Za-z])SCI(?![A-Za-z])|Snapshot\s+Compressive\s+Imaging|\u5355\u6b21\u66dd\u5149\u538b\u7f29\u6210\u50cf)",
            text,
            flags=re.I,
        )
    )
