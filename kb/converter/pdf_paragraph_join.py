from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from statistics import median


_LIGATURES = str.maketrans({'ﬀ':'ff','ﬁ':'fi','ﬂ':'fl','ﬃ':'ffi','ﬄ':'ffl'})


def _glyph_key(text: str) -> str:
    # This is a comparison surface only. Output keeps every existing token,
    # including math font commands, variables, indices and operators.
    text = re.sub(r'\\mathcal\s*\{([A-Za-z])\}', r'\1', text)
    return re.sub(r'\s+', '', text.translate(_LIGATURES).replace('$', ''))


def _line_text(line: dict) -> str:
    return ''.join(s.get('text', '') for s in line.get('spans') or [])


def _mathcal_fonts_agree(raw: str, glyphs: list[tuple[str, str]]) -> bool:
    expected = {len(_glyph_key(raw[:m.start()])) for m in re.finditer(r'\\mathcal\s*\{[A-Za-z]\}', raw)}
    actual = {i for i,(char,font) in enumerate(glyphs) if char.isascii() and char.isalpha()
              and re.search(r'cmsy|caligraphic|calligraphic', font, re.I)}
    return expected == actual


def _paragraphs(page) -> tuple[Counter, dict]:
    keys = []
    styles = {}
    payload = page.get_text('dict')
    if not isinstance(payload, dict):
        return Counter(), styles
    for block in payload.get('blocks') or []:
        lines = block.get('lines') or []
        if len(lines) < 3 or any(l.get('wmode', 0) or tuple(l.get('dir', (1,0))) != (1,0) for l in lines):
            continue
        margin = min(float(l['bbox'][0]) for l in lines)
        groups = []
        for line in lines:
            if not groups or float(line['bbox'][0]) > margin + 4:
                groups.append([])
            groups[-1].append(line)
        for group in groups:
            if len(group) < 3 or not margin + 4 < float(group[0]['bbox'][0]) < margin + 22:
                continue
            if any(abs(float(l['bbox'][0])-margin) > 2 for l in group[1:]):
                continue
            sizes = [float(s['size']) for l in group for s in l.get('spans', [])]
            if not sizes:
                continue
            size = median(sizes)
            primary = [[float(s['origin'][1]) for s in l.get('spans', []) if float(s['size']) >= .8*size] for l in group]
            if not all(primary):
                continue
            baselines = [median(origins) for origins in primary]
            if any(not .8*size < b-a < 1.8*size for a,b in zip(baselines,baselines[1:])):
                continue
            text = ' '.join(_line_text(l) for l in group)
            if len(re.findall(r'\b[A-Za-z]{3,}\b', text)) < 15:
                continue
            key = _glyph_key(text)
            glyphs = [(char, str(s.get('font') or '')) for l in group for s in l.get('spans', [])
                      for char in str(s.get('text') or '').translate(_LIGATURES) if not char.isspace() and char != '$']
            if key != ''.join(char for char,_ in glyphs):
                continue
            keys.append(key)
            styles[key] = glyphs
    return Counter(keys), styles


def _fragment(text: str) -> bool:
    stripped = text.strip()
    if not stripped or text.startswith(('    ','\t')):
        return False
    if stripped.startswith('$'):
        return bool(re.fullmatch(r'\$[^$\n]+\$', stripped))
    return not stripped.startswith(('#','>','|','```','~~~','<!--','![','<'))


def join_verified_pdf_paragraphs(text: str, pdf_path: Path | str) -> tuple[str, bool]:
    """Join only complete, unique, same-page PDF paragraphs with normal geometry.

    No fuzzy text match, word recovery, formula transcription, cross-column,
    cross-paragraph, display-equation, reference or protected-code joining.
    """
    if '\n\n' not in text:
        return text, False
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
    except (ImportError, OSError, ValueError, RuntimeError):
        return text, False
    parts = re.split(r'(\n[ \t]*\n)', text)
    pages = {}
    page = 0
    fence = ''
    in_math = False
    occupied = set()
    changes = []
    with doc:
        for i in range(0,len(parts),2):
            part = parts[i]
            protected = bool(fence) or in_math
            for line in part.splitlines():
                boundary = re.match(r'^(`{3,}|~{3,})', line.strip())
                if boundary:
                    if not fence: fence = boundary[1]
                    elif boundary[1][0] == fence[0] and len(boundary[1]) >= len(fence): fence = ''
                    continue
                if fence: continue
                marker = re.fullmatch(r'<!--\s*kb_page:\s*(\d+)\s*-->', line.strip())
                if marker and not in_math: page = int(marker[1])
                if line.strip() == '$$': in_math = not in_math
            if re.match(r'^#{1,6}\s+(?:\d+(?:\.\d+)*\.?\s+)?(?:References|Bibliography)\s*$', part.strip(),re.I):
                break
            if protected or fence or in_math or i in occupied or not 1 <= page <= len(doc) or not _fragment(part):
                continue
            if page not in pages:
                try:
                    pages[page] = _paragraphs(doc[page-1])
                except (RuntimeError, ValueError, TypeError, KeyError, AttributeError):
                    pages[page] = (Counter(), {})
            counts, styles = pages[page]
            candidates = {k for k,n in counts.items() if n == 1 and k.startswith(_glyph_key(part))}
            if not candidates:
                continue
            for end in range(i,min(len(parts),i+16),2):
                if end in occupied or not _fragment(parts[end]):
                    break
                raw = ' '.join(parts[j].strip() for j in range(i,end+1,2))
                if len(raw) > 1800:
                    break
                key = _glyph_key(raw)
                if key in candidates:
                    if end > i and _mathcal_fonts_agree(raw, styles[key]):
                        leading = part[:len(part)-len(part.lstrip())]
                        tail = parts[end][len(parts[end].rstrip()):]
                        changes.append((i,end,leading+raw+tail))
                        occupied.update(range(i,end+1))
                    break
                if not any(k.startswith(key) for k in candidates):
                    break
    for start,end,joined in changes:
        parts[start] = joined
        for i in range(start+1,end+1):parts[i] = ''
    result = ''.join(parts)
    return result, result != text


def recover_pdf_paragraph_structure(text: str, pdf_path: Path | str) -> tuple[str, bool]:
    from kb.converter.pdf_inline_prose import recover_inline_prose_fragments

    fixed, inline_changed = recover_inline_prose_fragments(text, pdf_path)
    fixed, paragraph_changed = join_verified_pdf_paragraphs(fixed, pdf_path)
    return fixed, inline_changed or paragraph_changed
