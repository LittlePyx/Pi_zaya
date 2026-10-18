from __future__ import annotations

import re

from kb.citation_marker_context import numeric_marker_is_content


def cited_table_cell_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    if numeric_marker_is_content(text, start):
        return None
    left = text.rfind('\n', 0, start) + 1
    right = text.find('\n', end)
    right = len(text) if right < 0 else right
    row = text[left:right]
    if not (row.strip().startswith('|') and row.strip().endswith('|')):
        return None
    previous = text[:left].splitlines()
    found_header = False
    for line in reversed(previous):
        if not line.strip().startswith('|'):
            break
        if re.fullmatch(r'\s*\|(?:\s*:?-{3,}:?\s*\|){2,}\s*', line):
            found_header = True
            break
    if not found_header:
        return None
    pipes = [left+i for i,c in enumerate(row) if c=='|' and (i==0 or row[i-1]!='\\')
             and not numeric_marker_is_content(text,left+i)]
    return next(((a+1,b) for a,b in zip(pipes,pipes[1:]) if a < start and end <= b),None)


def cited_table_cell_claims(text: str) -> list[str]:
    cells = []
    for marker in re.finditer(r'(?<!\[)\[\d{1,4}\](?!\])',text):
        span = cited_table_cell_span(text,*marker.span())
        if not span:
            continue
        cell = text[span[0]:span[1]].strip()
        coordinates = set(re.findall(r'(?<!\[)\[(\d{1,4})\](?!\])',cell))
        if len(coordinates)==1 and cell not in cells:
            cells.append(cell)
    return cells
