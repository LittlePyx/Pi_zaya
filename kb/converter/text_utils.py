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
    Mojibake keys are generated below from canonical Unicode punctuation so the
    repair table does not need unreadable hand-written samples.
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

# Some PDF text layers detach an acute accent from the following letter.  NFKC
# also turns the spacing acute accent (U+00B4) into a space plus U+0301, so keep
# this rule deliberately narrow: an accent must occur between two letters.
_DETACHED_ACUTE_RE = re.compile(
    r"(?P<left>[^\W\d_])(?:(?:[ \t]+\u0301)|(?:[ \t]*[\u00b4\u02ca][ \t]*))(?P<right>[^\W\d_])"
)

_KNOWN_MOJIBAKE_CODEPOINTS = (
    "\ufffd",
    "\u951b",
    "\u9428",
    "\u7ecb",
    "\u9225",
    "\u9286",
    "\u7039",
    "\u6d93",
    "\u6769",
    "\u934f",
    "\u7ed4",
    "\u6d63",
    "\u93c4",
    "\u5bee",
    "\u95c2",
    "\u71b6",
    "\u52ec",
    "\u579a",
    "\u70b2",
    "\u53e7",
)


def _build_mojibake_detection_re() -> re.Pattern[str]:
    generated = sorted(_MOJIBAKE_REPL, key=len, reverse=True)
    literal_patterns = [re.escape(value) for value in (*generated, *_KNOWN_MOJIBAKE_CODEPOINTS)]
    common_utf8_decode_patterns = [
        # UTF-8 bytes decoded as Latin-1/cp1252, including names such as
        # ``FranÃ§ois`` and stray non-breaking-space markers such as ``Â±``.
        r"Ã[\x80-\u00bf]",
        r"Â[\x80-\u00bf]",
        # Broken punctuation/math sequences commonly start with UTF-8 E2.
        # Requiring a non-ASCII follower avoids matching ordinary words with â.
        r"â[\x80-\u00bf\u0100-\u017f\u02c0-\u02ff\u2000-\u206f]{1,3}",
        r"ð\u0178[\x80-\u00bf\u0100-\u017f\u2000-\u206f]?",
        r"ï¿½",
        _DETACHED_ACUTE_RE.pattern,
    ]
    return re.compile("|".join([*literal_patterns, *common_utf8_decode_patterns]))


_MOJIBAKE_DETECTION_RE = _build_mojibake_detection_re()

# PDF font mapping sometimes substitutes Greek/math symbols with random CJK glyphs.
# These are paper-dependent; keep to the ones we have observed repeatedly.
_GARBLED_SYMBOL_REPL: dict[str, str] = {
    # Common mojibake patterns from PDFs
    "ďŹ": "fi",
    "Ď": "σ",   # Often sigma
    "Î´": "δ",
    "Îą": "α",
    "âĽ": "≤",
    "âĺ¤": "≥",
    "âĺ": "→",
    "â€“": "-",
    "â€”": "-",
    "â€˜": "'",
    "â€™": "'",
    "â€œ": "\"",
    "â€": "\"",
    "â€¦": "...",
    "脳": "×",
    "渭": "μ",
    "鈥檚": "'s",
    "聽": " ",
    "ˆ": "^",
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


def _fix_detached_acute(s: str) -> str:
    if not s:
        return ""

    def _reattach(match: re.Match[str]) -> str:
        return f"{match.group('left')}{match.group('right')}\u0301"

    return unicodedata.normalize("NFC", _DETACHED_ACUTE_RE.sub(_reattach, s))


def count_mojibake(s: str) -> int:
    """Count known mojibake sequences without double-counting overlaps."""
    if not s:
        return 0
    return sum(1 for _ in _MOJIBAKE_DETECTION_RE.finditer(s))


def contains_mojibake(s: str) -> bool:
    """Return whether text contains a known mojibake sequence."""
    return bool(s and _MOJIBAKE_DETECTION_RE.search(s))


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _fix_detached_acute(s)
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
    body = _fix_detached_acute(body)
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


def _looks_like_body_figure_reference_sentence(text: str) -> bool:
    """
    Distinguish narrative body sentences like:
      "Figure 1b, c compare ..."
    from real captions like:
      "Figure 1. Optical setup ..."

    We keep this intentionally narrow so we do not suppress legitimate captions.
    """
    t = _normalize_text(text or "").strip()
    if not t:
        return False
    probe = re.sub(r"^\*{1,2}\s*", "", t)
    probe = re.sub(r"\s*\*{1,2}\s*", "", probe)
    probe = re.sub(r"^\s*#{1,6}\s*", "", probe)
    probe = re.sub(r"(?i)^(figure|fig\.?)\s*(\d+[A-Za-z])\.\s*,\s*", r"\1 \2, ", probe)
    verb_alt = (
        r"(?:compare|compares|show|shows|illustrate|illustrates|"
        r"demonstrate|demonstrates|depict|depicts|present|presents|"
        r"highlight|highlights|summarize|summarizes)"
    )
    patterns = [
        rf"^\s*(?:fig(?:ure)?\.?)\s*\d+[A-Za-z](?:\s*,\s*[A-Za-z])+\s+{verb_alt}\b",
        rf"^\s*(?:fig(?:ure)?\.?)\s*\d+[A-Za-z]\s+{verb_alt}\b",
        rf"^\s*(?:fig(?:ure)?\.?)\s*\d+\([A-Za-z]\)\s+{verb_alt}\b",
    ]
    return any(re.match(pat, probe, flags=re.IGNORECASE) for pat in patterns)
