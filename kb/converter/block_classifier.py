from __future__ import annotations

import re
from typing import List

from .text_utils import _normalize_text
from .heuristics import _looks_like_equation_text

_TOC_SECTION_PAT = re.compile(r"(?<!\d)(\d+\.\d+(?:\.\d+)*)(?!\d)")

def _looks_like_toc_lines(lines: list[str]) -> bool:
    """
    Heuristic detection for Table-of-Contents-like blocks.
    """
    if not lines or len(lines) < 2:
        return False
    # Many TOCs contain section-like numbers with dots and a trailing page number.
    toc_like = 0
    for ln in lines[:20]:
        t = _normalize_text(ln)
        if not t:
            continue
        if _TOC_SECTION_PAT.search(t) and re.search(r"\b\d{1,4}\s*$", t):
            toc_like += 1
    return toc_like >= 2


def _split_toc_run_line(line: str) -> list[str]:
    """
    Split a single long TOC run into multiple lines.
    """
    s = _normalize_text(line)
    if not s:
        return []
    ms = list(_TOC_SECTION_PAT.finditer(s))
    if len(ms) < 2:
        return [s]
    # Must also end like a TOC (page number at end), otherwise it's likely normal prose.
    if not re.search(r"\b\d{1,4}\s*$", s):
        return [s]
    parts: list[str] = []
    for i, m in enumerate(ms):
        start = m.start()
        end = ms[i + 1].start() if i + 1 < len(ms) else len(s)
        chunk = s[start:end].strip()
        if chunk:
            parts.append(chunk)
    return parts or [s]


def _strip_trailing_page_no_from_heading_text(text: str) -> str:
    """
    Fix common TOC heading artifacts like:
      "2 Working mechanisms of single photon detector 2"
    where the trailing number is a page number, not part of the title.
    """
    t = _normalize_text(text)
    if not t:
        return t
    # Only strip small trailing integers (avoid removing years like 2024).
    m = re.match(r"^(\d+(?:\.\d+)*\s+.+?)\s+(\d{1,3})\s*$", t)
    if not m:
        return t
    try:
        n = int(m.group(2))
    except Exception:
        return t
    if 0 <= n <= 500:
        return m.group(1).rstrip()
    return t


def _looks_like_code_block(lines: list[str]) -> bool:
    joined = "\n".join(lines)
    if len(lines) < 3:
        return False

    score = 0
    if ("->" in joined) or ("<-" in joined) or (":=" in joined):
        score += 2
    if re.search(r"(?mi)^\s*(while|for|if|function)\b", joined):
        score += 2
    if re.search(r"(?mi)\bend (if|for|while|function)\b", joined):
        score += 1
    if re.search(r"(?mi)\breturn\b", joined):
        score += 1
    if sum(1 for x in lines if (len(x) - len(x.lstrip(" "))) >= 2) >= max(2, len(lines) // 3):
        score += 1

    return score >= 3


def _looks_like_table_block(lines: list[str]) -> bool:
    if len(lines) < 3:
        return False
    if any("|" in x for x in lines) and sum(1 for x in lines if "|" in x) >= 2:
        return True

    def split_cols(s: str) -> list[str]:
        return [c.strip() for c in re.split(r"\t+|\s{2,}", s.strip()) if c.strip()]

    col_counts = [len(split_cols(x)) for x in lines]
    rich_rows = sum(1 for c in col_counts if c >= 3)
    if rich_rows >= 3:
        return True

    numeric_rows = 0
    for x in lines:
        cols = split_cols(x)
        if len(cols) >= 3:
            nums = sum(1 for c in cols if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?%?", c))
            if nums >= 2:
                numeric_rows += 1
    return numeric_rows >= 3


def _looks_like_math_block(lines: list[str]) -> bool:
    # Drop tiny stray fragments that MuPDF often emits around equations.
    cleaned: list[str] = []
    for ln in lines:
        tln = _normalize_text(ln)
        if not tln:
            continue
        if len(tln) <= 1 and re.fullmatch(r"[-\u2013\u2014\u2212]", tln):
            continue
        cleaned.append(tln)
    joined = " ".join(cleaned) if cleaned else " ".join(lines)
    # avoid pseudo-code triggers
    if re.search(r"(?mi)^\s*(while|for|if|function)\b", joined):
        return False
    t = _normalize_text(joined)
    if not t:
        return False
    # Avoid classifying equation numbers as math.
    if re.fullmatch(r"\(\s*\d{1,4}\s*\)", t):
        return False
    
    # Check for common math patterns (more lenient for single lines)
    # Patterns like "δ + log k", "α =", "I 1 = { 1 , 2 , . . . , n }"
    if len(cleaned) == 1:
        # Single line - be more lenient
        # Check for Greek letters, math operators, set notation
        if re.search(r'[αβγδεζηθικλμνξοπρστυφχψω]', t, re.IGNORECASE):
            if re.search(r'[=+\-*/^_{}\[\]()]', t) or re.search(r'\b(log|exp|sin|cos|tan|max|min)\b', t, re.IGNORECASE):
                return True
        # Check for set notation
        if re.search(r'\{.*\}', t) and re.search(r'[=∈⊂⊆]', t):
            return True
        # Check for variable assignments with math operators
        if re.match(r'^[A-Za-z]\s*[0-9]*\s*[=+\-*/]', t) or re.match(r'^[A-Za-z]\s*=\s*', t):
            return True
    
    # lots of math operators / greek / symbols
    # NOTE: single-line display equations are common (e.g., d?/ds), so don't require 2+ lines.
    math_chars = re.findall(r"[=+\-*/^_{}()\[\]<>?]|[\u2200-\u22ff]|[\u0370-\u03ff]|[\ufe00-\ufe0f]", t)
    threshold = 3 if len(cleaned) <= 1 else 6  # Lower threshold for single lines
    if len(math_chars) < threshold and not _looks_like_equation_text(t):
        return False
    alpha_chars = re.findall(r"[A-Za-z]", t)
    # display equations usually have fewer natural-language words
    ratio = (len(alpha_chars) / max(1, len(t)))
    # Guard against misclassifying long body paragraphs that contain a few Greek letters.
    wordish = re.findall(r"[A-Za-z]{2,}", t)
    if len(wordish) >= 40:
        return False
    if len(wordish) >= 24 and len(math_chars) < 18 and "=" not in t and ("\\sum" not in t) and ("\\int" not in t):
        return False
    if len(cleaned) <= 1:
        return ratio < 0.60  # More lenient for single lines
    # multi-line equation blocks can contain connector words like "with/and"; allow a bit more letters.
    if len(math_chars) >= 10:
        return True
    return ratio < 0.45
