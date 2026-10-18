from __future__ import annotations

import re

from kb.citation_marker_context import numeric_marker_is_content
from kb.citation_table_context import cited_table_cell_span


def citation_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Keep closing quotes/citations with their sentence and protect abbreviations."""
    boundaries = [0]
    links = [m.span() for m in re.finditer(r'\[[^\]\n]*\]\([^\n)]*\)', text)]
    for match in re.finditer(r'[。！？!?；;\n]|,(?=\s*(?:\[\d{1,4}\](?:\(\#[^\n)]*\))?\s*)?\s+with\b)|\.(?=\s|["\u201d\u2019\']|\[|$)', text):
        position = match.start()
        if numeric_marker_is_content(text, position) or any(a <= position < b for a, b in links):
            continue
        if match[0] == ',' and (text[:position].count('(') != text[:position].count(')')
                                or text[:position].count('"') % 2):
            continue
        if match[0] == '.' and re.search(
            r'(?i)(?:\b(?:e\.g|i\.e|et\s+al|eq|fig|dr|vs)|\b[A-Z])\.$', text[:position+1],
        ):
            continue
        end = match.end()
        suffix = re.match(r'''["\u201d\u2019']*(?:\s*\[\d{1,4}\](?:\(\#[^\n)]*\))?)*''', text[end:])
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
