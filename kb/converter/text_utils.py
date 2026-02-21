from __future__ import annotations

import hashlib
import re
import unicodedata

LIGATURES = {
    "\ufb01": "fi",   # \ufb01
    "\ufb02": "fl",   # \ufb02
    "\ufb00": "ff",   # \ufb00
    "\ufb03": "ffi",  # \ufb03
    "\ufb04": "ffl",  # \ufb04
    "\ue03c": "tt",   # private-use ligature seen in some ACM PDFs
}


def _build_mojibake_repl() -> dict[str, str]:
    """
    Build replacements for strings where UTF-8 bytes were decoded as cp1252/latin1.
    Example: 鈥?-> 芒鈧?
    """
    canonical: dict[str, str] = {
        "\u201c": "\"",   # left double quote
        "\u201d": "\"",   # right double quote
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u2013": "-",    # en dash
        "\u2014": "-",    # em dash
        "\u2022": "-",    # bullet
        "\u2026": "...",  # ellipsis
    }
    out: dict[str, str] = {}
    for ch, repl in canonical.items():
        for codec in ("cp1252", "latin1"):
            try:
                bad = ch.encode("utf-8").decode(codec)
            except Exception:
                continue
            if bad and bad != ch:
                out[bad] = repl
    return out


_MOJIBAKE_REPL: dict[str, str] = _build_mojibake_repl()

# PDF font mapping sometimes substitutes Greek/math symbols with random CJK glyphs.
# These are paper-dependent; keep to the ones we have observed repeatedly.
_GARBLED_SYMBOL_REPL: dict[str, str] = {
    # Common mojibake patterns from PDFs
    "ďŹ": "fi",
    "Ď": "σ",  # Often sigma
    "Î´": "δ",
    "Îą": "α",
    "âĽ": "≤",  # Less than or equal
    "âĺ¤": "≥",  # Greater than or equal
    "â": "∈",  # Often element-of, but context-dependent
    "ˆ": "^",  # Hat/caret in math context
    "âĺ": "→",  # Arrow
    "âĺ": "→",  # Arrow variant
}


def _fix_common_mojibake(s: str) -> str:
    if not s:
        return ""
    for k, v in _MOJIBAKE_REPL.items():
        if k in s:
            s = s.replace(k, v)
    return s


def _fix_garbled_symbols(s: str) -> str:
    if not s:
        return ""
    for k, v in _GARBLED_SYMBOL_REPL.items():
        if k in s:
            s = s.replace(k, v)
    return s


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for k, v in LIGATURES.items():
        s = s.replace(k, v)
    s = _fix_common_mojibake(s)
    s = _fix_garbled_symbols(s)
    s = (
        s.replace("\u201c", "\"")
        .replace("\u201d", "\"")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u807d", " ")
    )
    s = re.sub(r"[ 	]+", " ", s)
    return s.strip()


def _normalize_line_keep_indent(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u807d", " ")
    indent_len = len(s) - len(s.lstrip(" "))
    indent = " " * indent_len
    body = s.lstrip(" ")
    body = unicodedata.normalize("NFKC", body)
    for k, v in LIGATURES.items():
        body = body.replace(k, v)
    body = _fix_common_mojibake(body)
    body = _fix_garbled_symbols(body)
    body = (
        body.replace("\u201c", "\"")
        .replace("\u201d", "\"")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    body = re.sub(r"[ 	]+", " ", body).rstrip()
    return (indent + body).rstrip()


def _join_lines_preserving_words(lines: list[str]) -> str:
    out: list[str] = []
    for line in lines:
        line = _normalize_text(line)
        if not line:
            continue
        if not out:
            out.append(line)
            continue
        prev = out[-1]
        if prev.endswith("-") and line and line[0].islower():
            out[-1] = prev[:-1] + line
        else:
            out[-1] = prev + " " + line
    return _normalize_text(" ".join(out))


def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()[:12]


def _is_letter(ch: str) -> bool:
    if not ch:
        return False
    try:
        return unicodedata.category(ch).startswith("L")
    except Exception:
        return False


def _common_prefix_length(s1: str, s2: str) -> int:
    i = 0
    while i < min(len(s1), len(s2)) and s1[i] == s2[i]:
        i += 1
    return i
