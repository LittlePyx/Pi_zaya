from __future__ import annotations

from pathlib import Path
import re


def _key(text: str) -> str:
    # Preserve variables, operators, case and font commands. Only Markdown
    # math delimiters and PDF spacing differ in this structural comparison.
    return re.sub(r'\s+', '', text.replace('$', ''))


def _line_text(line: dict) -> str:
    return ''.join(str(s.get('text') or '') for s in line.get('spans', []))


def _ordinary_context(text: str) -> bool:
    return bool(text.strip() and not text.lstrip().startswith(
        ('#', '>', '|', '```', '~~~', '<!--', '![', '$$'))
        and not text.startswith(('    ', '\t')))


def _same_paragraph(block: dict, index: int, before: str, body: str, after: str) -> bool:
    lines = block.get('lines') or []
    if not 0 < index < len(lines) - 1:
        return False
    line = lines[index]
    spans = line.get('spans') or []
    if not spans or any(l.get('wmode', 0) != 0 or tuple(l.get('dir', (1, 0))) != (1, 0) for l in lines):
        return False
    first, last = spans[0], spans[-1]
    size = float(first.get('size') or 0)
    if size <= 0 or first.get('font') != last.get('font'):
        return False
    if abs(float(first['origin'][1]) - float(last['origin'][1])) > size * .25:
        return False
    if not all(any(s.get('font') == first.get('font') for s in lines[i].get('spans', [])) for i in (index - 1, index + 1)):
        return False
    keys = [_key(_line_text(l)) for l in lines]
    whole = ''.join(keys)
    combined = _key(before + body + after)
    if not combined or whole.count(combined) != 1:
        return False
    start = whole.index(combined)
    end = start + len(combined)
    offsets = [0]
    for key in keys:
        offsets.append(offsets[-1] + len(key))
    if start not in offsets or end not in offsets:
        return False
    left, right = offsets.index(start), offsets.index(end) - 1
    if not left < index < right:
        return False
    margin = min(float(l['bbox'][0]) for l in lines)
    indent = float(lines[left]['bbox'][0]) - margin
    if not 4 <= indent <= size * 2.5:
        return False
    for i in range(left + 1, right + 1):
        # No new indented paragraph, column jump, or display-equation gap.
        if abs(float(lines[i]['bbox'][0]) - margin) > 2:
            return False
        gap = float(lines[i]['bbox'][1]) - float(lines[i - 1]['bbox'][1])
        if not 0 < gap <= size * 1.65:
            return False
    return True


def recover_inline_prose_fragments(text: str, pdf_path: Path | str) -> tuple[str, bool]:
    """Restore a PDF-proven prose paragraph split by false display math.

    All three existing Markdown fragments must match a unique same-page PDF
    line flow inside one indented paragraph. Preserve every content token;
    remove only the false display delimiters and intervening blank lines.
    """
    if '$$' not in text:
        return text, False
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
    except (ImportError, OSError, ValueError, RuntimeError):
        return text, False
    parts = re.split(r'(\n[ \t]*\n)', text)
    page = 0
    fence = ''
    in_math = False
    page_blocks: dict[int, list[dict]] = {}
    changes: list[tuple[int, str]] = []
    occupied: set[int] = set()
    with doc:
        for index in range(0, len(parts), 2):
            part = parts[index]
            stripped = part.strip()
            fenced_before, math_before = bool(fence), in_math
            for line in part.splitlines():
                if line.startswith(('    ', '\t')):
                    continue
                boundary = re.match(r'^(`{3,}|~{3,})', line.strip())
                if boundary:
                    if not fence:
                        fence = boundary[1]
                    elif boundary[1][0] == fence[0] and len(boundary[1]) >= len(fence):
                        fence = ''
                    continue
                if fence:
                    continue
                marker = re.fullmatch(r'<!--\s*kb_page:\s*(\d+)\s*-->', line.strip())
                if marker and not in_math:
                    page = int(marker[1])
                if line.strip() == '$$':
                    in_math = not in_math
            if not fence and not fenced_before and re.match(r'^#{1,6}\s+(?:References|Bibliography)\s*$', stripped, re.I):
                break
            match = re.fullmatch(r'\$\$\s*\n([^\n]+)\n\s*\$\$', stripped)
            if fence or fenced_before or math_before or not match or not 1 <= page <= len(doc) or index < 2 or index + 2 >= len(parts):
                continue
            if any(i in occupied for i in range(index - 2, index + 3)) or part.startswith(('    ', '\t')):
                continue
            before, body, after = parts[index - 2].strip(), match[1].strip(), parts[index + 2].strip()
            if not (_ordinary_context(parts[index - 2]) and _ordinary_context(parts[index + 2])):
                continue
            prefix = re.match(r'^[A-Za-z][A-Za-z ,\x27-]{8,}', body)
            suffix = re.search(r'[A-Za-z][A-Za-z ,\x27-]{5,}[.!?]?$', body)
            if not prefix or not suffix or not re.search(r'≜|:=|=', body):
                continue
            if any(len(re.findall(r'\b[A-Za-z]{2,}\b', m[0])) < 2 for m in (prefix, suffix)):
                continue
            if page not in page_blocks:
                try:
                    page_blocks[page] = [b for b in doc[page - 1].get_text('dict').get('blocks', []) if b.get('type') == 0]
                except (RuntimeError, ValueError):
                    continue
            matches = [(b, i) for b in page_blocks[page]
                       for i, l in enumerate(b.get('lines', [])) if _key(_line_text(l)) == _key(body)]
            if len(matches) != 1 or not _same_paragraph(*matches[0], before, body, after):
                continue
            leading = parts[index - 2][:len(parts[index - 2]) - len(parts[index - 2].lstrip())]
            trailing = parts[index + 2][len(parts[index + 2].rstrip()):]
            joined = leading + ' '.join((before, body, after)) + trailing
            changes.append((index, joined))
            occupied.update(range(index - 2, index + 3))
    for index, joined in changes:
        parts[index - 2] = joined
        for i in range(index - 1, index + 3):
            parts[i] = ''
    fixed = ''.join(parts)
    return fixed, fixed != text
