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
    "\u54ea\u51e0\u7bc7",
)

_ANSWER_AUDIT_PATTERNS = (
    r"\b(?:audit|review|check|verify|critique)\s+(?:the\s+)?(?:previous|last|prior)\s+answer\b",
    r"\b(?:previous|last|prior)\s+answer\b.{0,80}\b(?:audit|review|check|verify|correct)\b",
    r"\u5ba1\u67e5(?:\u4e0a\u4e00\u6761|\u4e0a\u4e2a|\u524d\u4e00\u6761)?\u56de\u7b54",
    r"(?:\u6838\u5bf9|\u68c0\u67e5|\u9a8c\u8bc1)(?:\u4e0a\u4e00\u6761|\u4e0a\u4e2a|\u524d\u4e00\u6761|\u8be5)\u56de\u7b54",
    r"\u4e0d\u8981\u91cd\u65b0\u751f\u6210",
)

_REQUESTED_PAPER_COUNT_PATTERNS = (
    r"(?:\b(?:exactly|only|top|choose|select|recommend|list|give\s+me)\s+)(\d{1,2})\s+(?:papers?|articles?|studies|references?)\b",
    r"(?:\u53ea(?:\u7528|\u8981|\u9009)|\u8bf7(?:\u9009|\u5217\u51fa|\u63a8\u8350|\u7ed9)|\u5217\u51fa|\u63a8\u8350|\u9009\u51fa|\u6700\u76f8\u5173\u7684)\s*(\d{1,2}|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u4e24]{1,3})\s*\u7bc7(?:\u8bba\u6587|\u6587\u7ae0|\u6587\u732e)?",
)

_CJK_PAPER_COUNT_DIGITS = {
    "\u4e00": 1,
    "\u4e8c": 2,
    "\u4e24": 2,
    "\u4e09": 3,
    "\u56db": 4,
    "\u4e94": 5,
    "\u516d": 6,
    "\u4e03": 7,
    "\u516b": 8,
    "\u4e5d": 9,
}

_MULTI_PAPER_SYNTHESIS_PATTERNS = (
    r"\b(?:roadmap|lineage|reading\s+route|reading\s+order|how\s+to\s+read|how\s+.*relat(?:e|es)|position(?:ing)?|background)\b",
    r"\bfrom\b.{0,80}\bto\b",
    r"\u8bfb\u4e66\u8def\u7ebf|\u9605\u8bfb\u8def\u7ebf|\u5148\u8bfb|\u600e\u4e48\u8bfb|\u642d\u914d\u8bfb|\u5efa\u7acb\u4e3b\u7ebf",
    r"\u4e3b\u7ebf|\u8109\u7edc|\u53d1\u5c55\u8def\u7ebf|\u8fd9\u6761\u7ebf|\u4ece.{0,40}\u5230",
    r"\u8fd9\u4e9b.{0,80}(?:\u65b9\u6cd5|\u6280\u672f|\u6587\u732e|\u8bba\u6587).{0,40}(?:\u5206\u522b|\u5173\u7cfb|\u89e3\u51b3)",
    r"(?:\u4e0e|\u548c).{0,40}(?:\u5173\u7cfb|\u533a\u522b|\u642d\u914d)",
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


def prompt_requests_answer_audit(prompt: str) -> bool:
    return _prompt_matches_any_pattern(prompt, _ANSWER_AUDIT_PATTERNS)


def _parse_requested_paper_count_token(value: str) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        count = int(raw)
    elif raw == "\u5341":
        count = 10
    elif "\u5341" in raw:
        left, _, right = raw.partition("\u5341")
        tens = _CJK_PAPER_COUNT_DIGITS.get(left, 1) if left else 1
        ones = _CJK_PAPER_COUNT_DIGITS.get(right, 0) if right else 0
        count = tens * 10 + ones
    else:
        count = _CJK_PAPER_COUNT_DIGITS.get(raw, 0)
    return count if 1 <= count <= 20 else None


def extract_requested_paper_count(prompt: str) -> int | None:
    text = str(prompt or "").strip()
    if not text or prompt_requests_answer_audit(text):
        return None
    for pattern in _REQUESTED_PAPER_COUNT_PATTERNS:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        count = _parse_requested_paper_count_token(match.group(1))
        if count is not None:
            return count
    return None


def prompt_explicitly_requests_multi_paper_list(prompt: str) -> bool:
    text = str(prompt or "").strip().lower()
    if not text:
        return False
    if prompt_requests_answer_audit(text):
        return False
    requested_count = extract_requested_paper_count(text)
    if requested_count is not None:
        return requested_count > 1
    if any(pat in text for pat in _MULTI_PAPER_LIST_PATTERNS):
        return True
    return bool(
        re.search(
            r"\bwhich\s+papers\b|\bwhat\s+papers\b|\bwhich\s+other\s+papers\b|\bwhat\s+other\s+papers\b|\bother\s+papers\b",
            text,
            flags=re.I,
        )
    )


def prompt_likely_multi_paper_synthesis(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    if prompt_explicitly_requests_multi_paper_list(text):
        return True
    return any(re.search(pattern, text, flags=re.I | re.S) for pattern in _MULTI_PAPER_SYNTHESIS_PATTERNS)


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
        r"(?:directly\s+|systematically\s+)?compar(?:e|es|ed|ing)\s+(.+?)(?:[?.,]|$)",
        r"comparison\s+between\s+(.+?)(?:[?.,]|$)",
        r"where\s+(?:is|was|are|were)\s+(.+?)\s+(?:discussed|mentioned|defined|described|located)(?:[?.,]|$)",
        r"(?:\u76f4\u63a5|\u7cfb\u7edf)?(?:\u6bd4\u8f83|\u5bf9\u6bd4)\s*(.+?)(?:[\uff0c\u3002\uff1f?]|$)",
        r"(.+?)\s*(?:\u5728\u54ea\u91cc|\u54ea\u4e2a\u7ae0\u8282|\u5982\u4f55)(?:\u8ba8\u8bba|\u63d0\u5230|\u5b9a\u4e49|\u5b9a\u4f4d)",
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
    return bool(prompt_reference_focus_action(prompt) and extract_multi_paper_topic(prompt))


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


def prompt_likely_cross_paper_refs(prompt: str) -> bool:
    """True when the prompt asks about library papers beyond the currently bound paper."""
    text = str(prompt or "").strip().lower()
    if not text:
        return False
    if prompt_explicitly_requests_multi_paper_list(text):
        return True
    return bool(
        re.search(
            r"\bwhich other papers?\b|\bother papers?\b|\bbesides (?:this|that) paper\b|"
            r"\banother paper\b|"
            r"\u9664\u6b64\u4e4b\u5916|\u9664(?:\u4e86)?(?:"
            r"\u8fd9\u7bc7|\u90a3\u7bc7)|"
            r"\u5176\u4ed6\u8bba\u6587|\u522b\u7684\u8bba\u6587|"
            r"\u8fd8\u6709\u54ea\u4e9b\u8bba\u6587|\u53e6\u4e00\u7bc7\u8bba\u6587",
            text,
            flags=re.I,
        )
    )
