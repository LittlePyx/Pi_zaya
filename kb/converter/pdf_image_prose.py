from __future__ import annotations

import re
from pathlib import Path

try:
    import fitz
except ImportError:
    fitz = None


def recover_pdf_prose_structure(text: str, md_path: Path, pdf_path: Path | str) -> tuple[str, bool]:
    from kb.converter.pdf_paragraph_join import recover_pdf_paragraph_structure

    fixed, structure_changed = recover_pdf_paragraph_structure(text, pdf_path)
    fixed, image_changed = recover_equation_image_prose(fixed, md_path, pdf_path)
    return fixed, structure_changed or image_changed


def recover_equation_image_prose(text: str, md_path: Path, pdf_path: Path | str) -> tuple[str, bool]:
    """Recover only a literal prose continuation hidden by an equation image.

    A unique same-page prefix must end at a PDF line boundary. The next line
    must contain ordinary prose immediately followed by formula punctuation;
    the formula remains in its existing image. No equation transcription,
    fuzzy matching, cross-page joining, or connective cleanup is performed.
    """
    original = text
    if '![Equation](' not in text:
        return text, False
    lines = text.splitlines(keepends=True)
    page_number = 0
    in_fence = False
    in_math = False
    if fitz is None:
        return text, False
    try:
        document = fitz.open(str(pdf_path))
    except Exception:
        return text, False
    with document as doc:
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('```', '~~~')):
                in_fence = not in_fence
            if in_fence:
                continue
            if stripped == '$$':
                in_math = not in_math
                continue
            if in_math:
                continue
            if line.startswith(('    ', '\t')):
                continue
            marker = re.fullmatch(r'<!--\s*kb_page:\s*(\d+)\s*-->', stripped)
            if marker:
                page_number = int(marker[1])
                continue
            if not 1 <= page_number <= len(doc) or len(stripped) < 40 or not re.fullmatch(r"[A-Za-z][A-Za-z ,.'’\-]+", stripped):
                continue
            next_index = index + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            image = re.fullmatch(r'!\[Equation\]\(([^)]+)\)', lines[next_index].strip()) if next_index < len(lines) else None
            if not image:
                continue
            asset = (md_path.parent / image[1]).resolve()
            if not asset.is_relative_to(md_path.parent.resolve()) or not asset.is_file():
                continue
            prefix = re.compile(r'(?<!\S)' + re.escape(stripped).replace(r'\ ', r'\s+') + r'[ \t]*\n')
            matches = [(str(block[4]), match) for block in doc[page_number - 1].get_text('blocks') if len(block) >= 5 for match in prefix.finditer(str(block[4]))]
            if len(matches) != 1:
                continue
            block_text, match = matches[0]
            following = block_text[match.end():].split('\n', 1)[0]
            # Formula punctuation belongs to the retained image, not the new prose.
            continuation = re.match(r"([a-z][A-Za-z ,’'\-]{23,}[,:])\s+(?=[A-Za-z]+[\[({=])", following)
            if not continuation:
                continue
            words = continuation[1]
            if len(words.split()) < 4 or words in text:
                continue
            newline = '\r\n' if line.endswith('\r\n') else '\n' if line.endswith('\n') else ''
            lines[index] = stripped + ' ' + words + newline
    fixed = ''.join(lines)
    return fixed, fixed != original
