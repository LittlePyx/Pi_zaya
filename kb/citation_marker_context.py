from __future__ import annotations

import re


_PROTECTED = re.compile(
    r"```.*?```|~~~.*?~~~|`[^`\n]*`|(?<!\\)\$\$.*?(?<!\\)\$\$"
    r"|(?<!\\)\$(?!\$)[^\n$]*?(?<!\\)\$|\\\(.*?\\\)|\\\[.*?\\\]",
    re.DOTALL,
)


def numeric_marker_is_content(text: str, start: int) -> bool:
    """Protect math/code and explicit numeric intervals from citation rewriting."""
    for match in _PROTECTED.finditer(text):
        if match.start() <= start < match.end():
            return True
    prefix = text[max(0, start - 96) : start]
    if re.search(r"(?:\\in|∈|range\s+of)\s*$", prefix, re.I):
        return True
    # Bare groups can mix invalid zero with valid source IDs. A pair needs
    # an interval cue; a noun such as "noise level" can precede a real cite.
    if re.search(
        r"(?:\b(?:range|interval)(?:\s+(?:of|is))?|\b(?:outside|inside|within|between)|"
        r"(?:范围|区间)(?:为|是)?)\s*[:：]?\s*$",
        prefix, re.I,
    ) and re.match(r"\[\s*[+-]?\d+(?:\.\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?\s*\]", text[start:]):
        return True
    return False


def strip_citation_offset(answer: str, offset: int, pattern: re.Pattern[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        if numeric_marker_is_content(answer, match.start()):
            return match.group(0)
        numbers = [
            int(value)
            for value in re.split(r"\s*(?:-|–|—|,|;|；|、)\s*", match.group(1))
            if value.strip()
        ]
        if not numbers or any(number < offset for number in numbers):
            return match.group(0)
        return "[" + ",".join(str(number - offset) for number in numbers) + "]"

    return pattern.sub(replace, answer)


def strip_citation_decoration(text: str, pattern: re.Pattern[str], replacement: str = " ") -> str:
    return pattern.sub(lambda match: match.group(0) if numeric_marker_is_content(text, match.start()) else replacement, text)
