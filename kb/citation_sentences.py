from __future__ import annotations

import re

from kb.citation_marker_context import numeric_marker_is_content
from kb.citation_table_context import cited_table_cell_span

_CITATION_TARGET = r'\(\#[^\s)]+(?:\s+"[^"]*")?\)'
_DEFINITION_BRIDGE = (
    r",(?=\s*(?:\[\d{1,4}\](?:" + _CITATION_TARGET + r")?\s*)*"
    r"(?:(?:giving|yielding|producing|creating|forming|(?:the\s+(?:method|procedure)|we(?:\s+then)?)\s+(?:creates?|forms?))"
    r"\b[^.;\n]{0,160}(?:\\triangleq|:=|≜|\b[A-Z]\s*=)|"
    r"where\s+\$?[A-Z]\$?\s+is\s+(?:an?\s+)?(?:additional\s+)?synthetic\s+noise\s+sample\b))"
)


def split_definition_bridges(text: str) -> list[str]:
    boundaries = [0]
    for match in re.finditer(_DEFINITION_BRIDGE, text, re.I):
        position = match.start()
        if (not numeric_marker_is_content(text, position)
                and text[:position].count('(') == text[:position].count(')')
                and text[:position].count('"') % 2 == 0):
            suffix = re.match(r'(?:\s*\[\d{1,4}\](?:' + _CITATION_TARGET + r')?)*', text[match.end():])
            boundaries.append(match.end() + suffix.end())
    boundaries.append(len(text))
    return [text[a:b].strip() for a, b in zip(boundaries, boundaries[1:]) if text[a:b].strip()]


def citation_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Keep closing quotes/citations with their sentence and protect abbreviations."""
    boundaries = [0]
    links = [m.span() for m in re.finditer(r'\[\d{1,4}\]' + _CITATION_TARGET + r'|\[[^\]\n]*\]\([^\n)]*\)', text)]
    for match in re.finditer(r'[。！？!?；;\n]|[，,](?=\s*(?:其中|where\b)[^。；;\n]{0,180}(?:表示|定义为|denotes?\b|defined\b))|,(?=\s*(?:\[\d{1,4}\](?:\(\#[^\n)]*\))?\s*)?\s+with\b)|\.(?=\s|["\u201d\u2019\']|\[|$)' + '|' + _DEFINITION_BRIDGE, text, re.I):
        position = match.start()
        if numeric_marker_is_content(text, position) or any(a <= position < b for a, b in links):
            continue
        if match[0] in ',，' and (text[:position].count('(') != text[:position].count(')')
                                or text[:position].count('"') % 2):
            continue
        if match[0] == '.' and re.search(
            r'(?i)(?:\b(?:e\.g|i\.e|et\s+al|eq|fig|dr|vs)|\b[A-Z])\.$', text[:position+1],
        ):
            continue
        end = match.end()
        suffix = re.match(r'''["\u201d\u2019']*(?:\s*\[\d{1,4}\](?:''' + _CITATION_TARGET + r')?)*', text[end:])
        end += suffix.end()
        if end > boundaries[-1]:
            boundaries.append(end)
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    return list(zip(boundaries, boundaries[1:]))


def citation_sentence_context(text: str, start: int, end: int) -> str:
    cell = cited_table_cell_span(text,start,end)
    if cell:
        left,right = cell
        return citation_sentence_context(text[left:right],start-left,end-left)
    left, right = next(((a, b) for a, b in citation_sentence_spans(text)
                        if a <= start < b and end <= b), (0, len(text)))
    return text[left:right].strip()
