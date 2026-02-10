# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


LIGATURES = {
    "\ufb01": "fi",   # \ufb01
    "\ufb02": "fl",   # \ufb02
    "\ufb00": "ff",   # \ufb00
    "\ufb03": "ffi",  # \ufb03
    "\ufb04": "ffl",  # \ufb04
    "\ue03c": "tt",   # private-use ligature seen in some ACM PDFs
}

NOISE_LINE_PATTERNS = [
    r"^ACM Trans\. Graph\., Vol\.",
    r"^Publication date:\s+",
    r"^ACM Transactions on Graphics\b",
    r"^Latest updates:\s*",
    r"^RESEARCH-ARTICLE\b",
    r"^PDF Download\b",
    r"^Total Citations:\b",
    r"^Total Downloads:\b",
    r"^Citation in BibTeX format\b",
    r"^Open Access Support provided by:\s*$",
    r"^3D Gaussian Splatting for Real-Time Radiance Field Rendering\s*$",
    r"^\s*\d+:\d+\s*$",  # e.g. "139:2" page labels in ACM PDFs
    r"^\s*-\s+Bernhard Kerbl, Georgios Kopanas,.*$",  # running author header
    r"^\s*-\s*$",  # stray single dash line
]


# Common mojibake / glyph-substitution artifacts we see in academic PDFs.
# Keep this list small and high-precision; these replacements should almost never be wrong in English papers.
_MOJIBAKE_REPL: dict[str, str] = {
    # Accented letters (common in author affiliations / names)
    "Universit茅": "Université",
    "C么te": "Côte",
    "Leimk眉hler": "Leimkühler",
    "Sch眉tz": "Schütz",
    "M眉ller": "Müller",
    "R眉ckert": "Rückert",
    "Cl茅ment": "Clément",
    "Zollh枚fer": "Zollhöfer",
    "Nie脽ner": "Nießner",
    "脰ztireli": "Öztireli",
    # Typical punctuation / symbols
    "鈥?": "–",
    "鈭?": "∼",
    "鈫?": "→",
    "鈫": "→",
    "鈥": "–",
}

# PDF font mapping sometimes substitutes Greek/math symbols with random CJK glyphs.
# These are paper-dependent; keep to the ones we have observed repeatedly.
_GARBLED_SYMBOL_REPL: dict[str, str] = {
    "伪": "α",
    "蟽": "σ",
    "未": "δ",
    "危": "Σ",
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
        s.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace(" ", " ")
    )
    s = re.sub(r"[ 	]+", " ", s)
    return s.strip()


def _normalize_line_keep_indent(s: str) -> str:
    if not s:
        return ""
    s = s.replace(" ", " ")
    indent_len = len(s) - len(s.lstrip(" "))
    indent = " " * indent_len
    body = s.lstrip(" ")
    body = unicodedata.normalize("NFKC", body)
    for k, v in LIGATURES.items():
        body = body.replace(k, v)
    body = _fix_common_mojibake(body)
    body = _fix_garbled_symbols(body)
    body = (
        body.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
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


_TOC_SECTION_PAT = re.compile(r"(?<!\d)(\d+\.\d+(?:\.\d+)*)(?!\d)")


def _looks_like_toc_lines(lines: list[str]) -> bool:
    """
    Heuristic detection for Table-of-Contents-like blocks.

    Typical symptom we want to fix:
    - PDF text extractor gives multiple TOC entries as multiple lines
      but our paragraph join collapses them into ONE long line.

    We keep line breaks for blocks that look like:
      "2.1.1 ... 2"
      "2.1.2 ... 4"
      "2.1.3 ... 5"
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

    Example input:
      "2.1.1 AAA 2 2.1.2 BBB 4 2.1.3 CCC 5"
    Output:
      ["2.1.1 AAA 2", "2.1.2 BBB 4", "2.1.3 CCC 5"]
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


def _bbox_width(bbox: Iterable[float]) -> float:
    x0, _, x1, _ = bbox
    return float(x1) - float(x0)


def _bbox_height(bbox: Iterable[float]) -> float:
    _, y0, _, y1 = bbox
    return float(y1) - float(y0)

def _rect_area(rect) -> float:
    try:
        return float(rect.get_area())
    except Exception:
        try:
            return max(0.0, float(rect.width) * float(rect.height))
        except Exception:
            return 0.0


@dataclass(frozen=True)
class LlmConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    request_sleep_s: float = 0.0


@dataclass(frozen=True)
class ConvertConfig:
    pdf_path: Path
    out_dir: Path
    translate_zh: bool
    start_page: int
    end_page: int
    skip_existing: bool
    keep_debug: bool
    llm: Optional[LlmConfig]
    llm_classify: bool = True
    llm_render_page: bool = False
    llm_classify_only_if_needed: bool = True
    classify_batch_size: int = 40
    image_scale: float = 2.0
    image_alpha: bool = False
    eq_image_fallback: bool = False
    global_noise_scan: bool = True
    llm_repair: bool = True
    llm_repair_body_math: bool = False


@dataclass(frozen=True)
class TextBlock:
    bbox: tuple[float, float, float, float]
    text: str
    max_font_size: float
    is_bold: bool
    insert_image: Optional[str] = None
    is_code: bool = False
    is_table: bool = False
    is_math: bool = False
    heading_level: Optional[int] = None


def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()[:12]


def _is_letter(ch: str) -> bool:
    if not ch:
        return False
    try:
        return unicodedata.category(ch).startswith("L")
    except Exception:
        return False


def _looks_like_equation_text(s: str) -> bool:
    t = _normalize_text(s)
    if not t:
        return False
    if re.search(r"[=^_\\]", t):
        return True
    if re.search(r"[\u2200-\u22ff]", t):
        return True
    if re.search(r"[∑Σ∏√∫∞≈≠≤≥]", t):
        return True
    if re.search(r"\b(?:arg\s*min|arg\s*max|exp|log|sin|cos|tan)\b", t, flags=re.IGNORECASE):
        return True
    # function-like: G(x), f(\theta), etc.
    if re.match(r"^[A-Za-z]{1,6}\s*\([^)]*\)\s*[=+\-*/^_]", t):
        return True
    return False


def _is_balanced_latex(s: str) -> bool:
    # Basic sanity: balanced braces and begin/end.
    if s.count("{") != s.count("}"):
        return False
    begins = re.findall(r"\\begin\{([^}]+)\}", s)
    ends = re.findall(r"\\end\{([^}]+)\}", s)
    return Counter(begins) == Counter(ends)


def _is_noise_line(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return False
    # Also match after removing Markdown heading markers, since some converters
    # may incorrectly promote boilerplate to headings.
    t2 = re.sub(r"^#+\s*", "", t).strip()
    for pat in NOISE_LINE_PATTERNS:
        if re.match(pat, t) or re.match(pat, t2):
            return True
    return False


def build_repeated_noise_texts(doc) -> set[str]:
    if fitz is None:
        return set()
    total = len(doc)
    if total <= 1:
        return set()
    counts: Counter[str] = Counter()

    for page in doc:
        H = float(page.rect.height)
        top_band = 120.0
        bottom_band = 120.0
        d = page.get_text("dict")
        for b in d.get("blocks", []):
            if "lines" not in b:
                continue
            bbox = tuple(float(x) for x in b.get("bbox", (0, 0, 0, 0)))
            rect = fitz.Rect(bbox)
            if not (rect.y1 < top_band or rect.y0 > H - bottom_band):
                continue
            lines: list[str] = []
            for l in b.get("lines", []) or []:
                spans = l.get("spans", []) or []
                if not spans:
                    continue
                line = "".join(str(s.get("text", "")) for s in spans)
                line = _normalize_text(line)
                if line:
                    lines.append(line)
            txt = _join_lines_preserving_words(lines)
            if 10 <= len(txt) <= 140:
                counts[txt] += 1

    threshold = max(3, int(total * 0.4))
    noise = {t for t, n in counts.items() if n >= threshold}
    noise.update({t for t in counts if _is_noise_line(t)})
    return noise


def _page_has_references_heading(page) -> bool:
    """
    Best-effort detection of a REFERENCES page.

    IMPORTANT: many two-column PDFs do not emit a clean line "REFERENCES" in get_text("text"),
    so we also check the structured text dict for a heading-like span near the top of the page.
    """
    # 1) Fast path: plain text contains an isolated line.
    try:
        t = page.get_text("text") or ""
        if re.search(r"(?mi)^\s*REFERENCES\s*$", t):
            return True
    except Exception:
        pass

    # 2) Structured path: find a span/line equal to REFERENCES near the top.
    try:
        d = page.get_text("dict") or {}
        H = float(page.rect.height)
        for b in d.get("blocks", []) or []:
            if "lines" not in b:
                continue
            bbox = b.get("bbox")
            if not bbox:
                continue
            y0 = float(bbox[1])
            # heading typically appears early in the page
            if y0 > max(240.0, H * 0.45):
                continue
            for l in b.get("lines", []) or []:
                spans = l.get("spans", []) or []
                if not spans:
                    continue
                line = "".join(str(s.get("text", "")) for s in spans)
                line = _normalize_text(line).strip()
                if not line:
                    continue
                if re.fullmatch(r"REFERENCES", line, flags=re.IGNORECASE):
                    return True
    except Exception:
        pass

    return False


def _page_looks_like_references_content(page) -> bool:
    """Heuristic for pages where the REFERENCES heading is missing (continued pages, odd layouts)."""
    try:
        t = page.get_text("text") or ""
    except Exception:
        t = ""
    t = _normalize_text(t)
    if not t.strip():
        return False
    # Many references: lots of years, commas, and URLs/DOIs.
    years = re.findall(r"(?:19|20)\d{2}", t)
    if len(years) < 18:
        return False
    comma_lines = sum(1 for ln in t.splitlines() if "," in ln and len(ln.strip()) >= 25)
    if comma_lines < 10:
        return False
    if ("doi" in t.lower()) or ("http" in t.lower()) or ("arxiv" in t.lower()):
        return True
    # fallback: pure density of years is already a strong signal
    return len(years) >= 28


def _is_frontmatter_noise_line(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return True
    front_pat = [
        r"^Latest updates:\s*",
        r"^PDF Download\b",
        r"^Total Citations\b",
        r"^Total Downloads\b",
        r"^Published:\b",
        r"^Citation in BibTeX format\b",
        r"^Open Access Support provided by:\b",
        r"^RESEARCH-ARTICLE\b",
    ]
    return any(re.match(p, t, flags=re.IGNORECASE) for p in front_pat)


def _looks_like_code_block(lines: list[str]) -> bool:
    joined = "\n".join(lines)
    if len(lines) < 3:
        return False

    score = 0
    if "←" in joined or "<-" in joined or ":=" in joined:
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
        if len(tln) <= 1 and re.fullmatch(r"[-–—−«»]", tln):
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
    # lots of math operators / greek / symbols
    # NOTE: single-line display equations are common (e.g., dΣ'/ds), so don't require 2+ lines.
    math_chars = re.findall(r"[=+\-*/^_{}()\[\]<>Σσμλαδτφ∑∏√≈≤≥∞∥]|[\u2200-\u22ff]|[\u0370-\u03ff]|[\ufe00-\ufe0f]", t)
    threshold = 4 if len(cleaned) <= 1 else 6
    if len(math_chars) < threshold and not _looks_like_equation_text(t):
        return False
    alpha_chars = re.findall(r"[A-Za-z]", t)
    # display equations usually have fewer natural-language words
    ratio = (len(alpha_chars) / max(1, len(t)))
    # Guard against misclassifying long body paragraphs that contain a few Greek letters.
    wordish = re.findall(r"[A-Za-z]{2,}", t)
    if len(wordish) >= 40:
        return False
    if len(wordish) >= 24 and len(math_chars) < 18 and "=" not in t and "∑" not in t and "∏" not in t:
        return False
    if len(cleaned) <= 1:
        return ratio < 0.55
    # multi-line equation blocks can contain connector words like "with/and"; allow a bit more letters.
    if len(math_chars) >= 10:
        return True
    return ratio < 0.45


def detect_body_font_size(doc) -> float:
    sizes: list[float] = []
    for page in doc:
        d = page.get_text("dict")
        for b in d.get("blocks", []):
            for l in b.get("lines", []) or []:
                for s in l.get("spans", []) or []:
                    try:
                        sizes.append(round(float(s.get("size", 0.0)), 1))
                    except Exception:
                        continue
    if not sizes:
        return 10.0
    return float(Counter(sizes).most_common(1)[0][0])


def detect_header_tag(
    *,
    page_index: int,
    text: str,
    max_size: float,
    is_bold: bool,
    body_size: float,
    page_width: float,
    bbox: tuple[float, float, float, float],
) -> Optional[str]:
    t = _normalize_text(text)
    if len(t) < 3 or len(t) > 160:
        return None

    # Strict heading policy:
    # - Prefer numbered section headings (1, 2.1, 7.4, ...)
    # - Allow a small set of all-caps paper headings only when visually emphasized
    is_spanning = _bbox_width(bbox) > page_width * 0.65

    if t.upper() == "APPENDIX" and is_spanning and max_size >= body_size - 0.3:
        return "[H1]"

    if _is_numbered_heading_text(t):
        m = re.match(r"^(\d+(?:\.\d+)*)\s+(.+)$", t)
        if m:
            title = m.group(2).strip()
            if len(title) <= 140 and not title.endswith(".") and not _looks_like_equation_text(title):
                level = m.group(1).count(".") + 1
                level = max(1, min(3, level))
                # require it's at least body size (avoid footnotes polluting headings)
                if max_size >= body_size - 0.3:
                    return f"[H{level}]"

    # Appendix letter headings (A ..., B.1 ..., etc.)
    if _is_appendix_heading_text(t):
        m2 = re.match(r"^([A-Z])((?:\.\d+)+)?\s+(.+)$", t)
        if m2 and is_spanning and max_size >= body_size - 0.3 and len(t) <= 160 and not t.endswith("."):
            title = (m2.group(3) or "").strip()
            if title and not _looks_like_equation_text(title):
                suffix = m2.group(2) or ""
                if suffix:
                    level = min(3, suffix.count(".") + 1)
                    return f"[H{level}]"
                return "[H1]"

    # Keyword headings. Some PDFs do not bold/resize "REFERENCES", so allow it more loosely.
    keywords_strict = {"ABSTRACT", "ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS"}
    if (
        t.upper() in keywords_strict
        and is_spanning
        and is_bold
        and max_size >= body_size + 0.8
        and len(t) <= 40
    ):
        return "[H1]"
    if t.upper() == "REFERENCES" and is_spanning and max_size >= body_size - 0.2 and len(t) <= 40:
        return "[H1]"

    return None


def _pick_column_range(rect: fitz.Rect, page_width: float) -> tuple[float, float]:
    mid = page_width / 2.0
    if rect.width >= page_width * 0.55:
        return (0.0, page_width)
    return (0.0, mid) if rect.x0 < mid else (mid, page_width)


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _math_fragment_score(b: "TextBlock", *, body_size: float) -> int:
    """
    Score whether a math block is likely a fragment (limits, lone operators, etc.) that should be merged.
    Keep this conservative to avoid merging distinct equations.
    """
    try:
        t = _normalize_text(b.text)
    except Exception:
        t = (b.text or "").strip()
    if not t:
        return 0
    if len(t) <= 10:
        return 2
    # Tiny bbox height often indicates superscripts/subscripts split into their own blocks.
    # Keep this strict: normal equation lines are often ~2x body font height.
    if _bbox_height(b.bbox) <= max(6.0, float(body_size) * 0.85):
        return 1
    # Very short operator-like strings.
    if len(t) <= 18 and re.fullmatch(r"[A-Za-z0-9=+*/^_()Σ∑∏√\\\-]+", t):
        return 1
    return 0


def merge_adjacent_math_blocks(
    blocks: list[TextBlock],
    *,
    max_vgap: float,
    body_size: float,
    min_x_overlap: float = 0.2,
) -> list[TextBlock]:
    if not blocks:
        return []
    merged: list[TextBlock] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if not b.is_math:
            merged.append(b)
            i += 1
            continue

        x0, y0, x1, y1 = b.bbox
        parts = [b.text]
        max_size = b.max_font_size
        bold = b.is_bold
        j = i + 1
        while j < len(blocks):
            nb = blocks[j]
            if not nb.is_math:
                break
            # check vertical gap
            vgap = float(nb.bbox[1]) - float(y1)
            if vgap > max_vgap:
                break
            # only merge if at least one side looks like a fragment (prevents merging distinct equations)
            if _math_fragment_score(b, body_size=float(body_size)) <= 0 and _math_fragment_score(nb, body_size=float(body_size)) <= 0:
                break
            # check horizontal overlap
            overlap = _overlap_1d(float(x0), float(x1), float(nb.bbox[0]), float(nb.bbox[2]))
            denom = max(1.0, min(_bbox_width(b.bbox), _bbox_width(nb.bbox)))
            if overlap / denom < min_x_overlap:
                break
            parts.append(nb.text)
            x0 = min(float(x0), float(nb.bbox[0]))
            y0 = min(float(y0), float(nb.bbox[1]))
            x1 = max(float(x1), float(nb.bbox[2]))
            y1 = max(float(y1), float(nb.bbox[3]))
            max_size = max(max_size, nb.max_font_size)
            bold = bold or nb.is_bold
            j += 1

        merged.append(
            TextBlock(
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                text="\n".join([p for p in parts if p.strip()]),
                max_font_size=max_size,
                is_bold=bold,
                insert_image=b.insert_image,
                is_code=b.is_code,
                is_table=b.is_table,
                is_math=True,
                heading_level=b.heading_level,
            )
        )
        i = j

    return merged


def merge_math_blocks_by_proximity(
    blocks: list[TextBlock],
    *,
    body_size: float,
    page_width: float,
) -> list[TextBlock]:
    """
    Merge math blocks that belong to the same displayed equation but were split by PDF extraction.
    This is a geometry-only merge (no regex / content heuristics) to avoid fragile post-processing.
    """
    if not blocks:
        return []

    math_indices = [i for i, b in enumerate(blocks) if b.is_math]
    if len(math_indices) < 2:
        return blocks

    def find(par, x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def union(par, a, b):
        ra, rb = find(par, a), find(par, b)
        if ra != rb:
            par[rb] = ra

    # union-find over math indices
    par = {i: i for i in math_indices}

    def bbox_center(b: TextBlock) -> tuple[float, float]:
        x0, y0, x1, y1 = b.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def overlap_ratio_1d(a0, a1, b0, b1) -> float:
        ov = _overlap_1d(a0, a1, b0, b1)
        denom = max(1.0, min((a1 - a0), (b1 - b0)))
        return float(ov) / float(denom)

    for ia in range(len(math_indices)):
        i = math_indices[ia]
        bi = blocks[i]
        ri = fitz.Rect(bi.bbox)
        cxi, cyi = bbox_center(bi)
        for ja in range(ia + 1, len(math_indices)):
            j = math_indices[ja]
            bj = blocks[j]
            rj = fitz.Rect(bj.bbox)
            cxj, cyj = bbox_center(bj)

            # avoid merging across columns/pages: too far apart horizontally
            if abs(cxj - cxi) > page_width * 0.45:
                continue

            x_overlap = overlap_ratio_1d(ri.x0, ri.x1, rj.x0, rj.x1)
            y_overlap = overlap_ratio_1d(ri.y0, ri.y1, rj.y0, rj.y1)

            # stacked parts of one equation (typically tiny superscripts/subscripts around tall operators)
            vgap = max(0.0, rj.y0 - ri.y1, ri.y0 - rj.y1)
            if x_overlap >= 0.25 and vgap <= max(6.0, float(body_size) * 1.0):
                if _math_fragment_score(bi, body_size=float(body_size)) > 0 or _math_fragment_score(bj, body_size=float(body_size)) > 0:
                    union(par, i, j)
                    continue

            # side-by-side parts on the same equation row (e.g., split tokens / right-hand definition)
            hgap = max(0.0, rj.x0 - ri.x1, ri.x0 - rj.x1)
            if y_overlap >= 0.45 and hgap <= max(18.0, float(body_size) * 6.0):
                if _math_fragment_score(bi, body_size=float(body_size)) > 0 or _math_fragment_score(bj, body_size=float(body_size)) > 0:
                    union(par, i, j)

    # build components
    groups: dict[int, list[int]] = {}
    for i in math_indices:
        r = find(par, i)
        groups.setdefault(r, []).append(i)

    # build merged blocks; keep stable placement at the smallest original index
    replaced: dict[int, TextBlock] = {}
    removed: set[int] = set()
    for root, idxs in groups.items():
        if len(idxs) <= 1:
            continue
        idxs_sorted = sorted(idxs)
        # order fragments top-to-bottom, then left-to-right (helps LLM reconstruction)
        frags = sorted(
            (blocks[k] for k in idxs_sorted),
            key=lambda b: (float(b.bbox[1]), float(b.bbox[0])),
        )
        parts = [b.text for b in frags if b.text.strip()]
        if not parts:
            continue
        x0 = min(float(b.bbox[0]) for b in frags)
        y0 = min(float(b.bbox[1]) for b in frags)
        x1 = max(float(b.bbox[2]) for b in frags)
        y1 = max(float(b.bbox[3]) for b in frags)
        merged = TextBlock(
            bbox=(x0, y0, x1, y1),
            text="\n".join(parts),
            max_font_size=max(float(b.max_font_size) for b in frags),
            is_bold=any(bool(b.is_bold) for b in frags),
            insert_image=frags[0].insert_image,
            is_code=False,
            is_table=False,
            is_math=True,
            heading_level=None,
        )
        replaced[idxs_sorted[0]] = merged
        removed.update(idxs_sorted[1:])

    if not replaced and not removed:
        return blocks

    out: list[TextBlock] = []
    for i, b in enumerate(blocks):
        if i in removed:
            continue
        if i in replaced:
            out.append(replaced[i])
        else:
            out.append(b)
    return out


def attach_equation_numbers(
    blocks: list[TextBlock],
    *,
    body_size: float,
    page_width: float,
) -> tuple[list[TextBlock], dict[int, str]]:
    """
    Attach right-side equation numbers like "(8)" to the nearest displayed-math block.
    Returns (new_blocks, eqno_by_block_index).
    """
    if not blocks:
        return blocks, {}

    eqno_re = re.compile(r"^\(\s*(\d{1,4})\s*\)$")

    candidates: list[int] = []
    for i, b in enumerate(blocks):
        if b.is_math or b.is_table or b.is_code:
            continue
        t = _normalize_text(b.text)
        if not t:
            continue
        if not eqno_re.fullmatch(t):
            continue
        # equation numbers are usually small-ish
        if float(b.max_font_size) > float(body_size) + 1.2:
            continue
        candidates.append(i)

    if not candidates:
        return blocks, {}

    def center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x0, y0, x1, y1 = bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    math_idxs = [i for i, b in enumerate(blocks) if b.is_math]
    eqno_by_block: dict[int, str] = {}
    remove_idx: set[int] = set()

    for ci in candidates:
        cb = blocks[ci]
        m = eqno_re.fullmatch(_normalize_text(cb.text))
        if not m:
            continue
        num = m.group(1)
        cxc, cyc = center(cb.bbox)

        best = None
        best_score = None
        for mi in math_idxs:
            mb = blocks[mi]
            mx, my = center(mb.bbox)
            # must be on the right side of the equation row (works for both columns)
            if cxc < mx:
                continue
            # must be reasonably aligned vertically
            if abs(cyc - my) > max(14.0, float(body_size) * 2.4):
                continue
            # avoid attaching across the whole page
            if abs(cxc - mx) > page_width * 0.55:
                continue
            # prefer the rightmost and closest-in-y candidate
            score = (cxc / max(1.0, page_width)) - 0.002 * abs(cyc - my)
            if best is None or (best_score is not None and score > best_score):
                best = mi
                best_score = score

        if best is None:
            continue
        # don't overwrite an existing number on the same math block
        if best in eqno_by_block:
            continue
        eqno_by_block[best] = num
        remove_idx.add(ci)

    if not remove_idx:
        return blocks, eqno_by_block

    # Build new blocks list and remap eqno indices.
    new_blocks: list[TextBlock] = []
    old_to_new: dict[int, int] = {}
    for i, b in enumerate(blocks):
        if i in remove_idx:
            continue
        old_to_new[i] = len(new_blocks)
        new_blocks.append(b)

    new_eqno: dict[int, str] = {}
    for old_i, n in eqno_by_block.items():
        ni = old_to_new.get(old_i)
        if ni is not None:
            new_eqno[ni] = n
    return new_blocks, new_eqno


def extract_eqno_from_math_text(
    blocks: list[TextBlock],
    eqno_by_block: dict[int, str],
) -> tuple[list[TextBlock], dict[int, str]]:
    """
    Some PDFs embed the equation number (e.g., '(3)') inside the same extracted math text block.
    Pull it out into eqno_by_block and strip it from the math text so renderers don't treat it as math content.
    """
    if not blocks:
        return blocks, eqno_by_block
    out: list[TextBlock] = []
    eqno_by_block = dict(eqno_by_block or {})
    tail_re = re.compile(r"\(\s*(\d{1,4})\s*\)\s*$")
    line_re = re.compile(r"^\(\s*(\d{1,4})\s*\)$")
    for i, b in enumerate(blocks):
        if not b.is_math:
            out.append(b)
            continue
        txt_lines = [ln.rstrip() for ln in (b.text or "").splitlines() if ln.strip()]
        if not txt_lines:
            out.append(b)
            continue
        # 1) Prefer a standalone "(n)" line anywhere (common when MuPDF splits tokens).
        standalone = [line_re.fullmatch(ln.strip()) for ln in txt_lines]
        standalone_nums = [m.group(1) for m in standalone if m]
        if standalone_nums and i not in eqno_by_block:
            eqno_by_block[i] = standalone_nums[-1]
            txt_lines = [ln for ln in txt_lines if not line_re.fullmatch(ln.strip())]
        # 2) Otherwise, try a trailing "(n)" on the last non-empty line.
        if i not in eqno_by_block:
            last = txt_lines[-1].strip() if txt_lines else ""
            m = tail_re.search(last)
            if m:
                eqno_by_block[i] = m.group(1)
                txt_lines[-1] = tail_re.sub("", last).rstrip()
                txt_lines = [ln for ln in txt_lines if ln.strip()]
        if txt_lines != [ln.rstrip() for ln in (b.text or "").splitlines() if ln.strip()]:
            out.append(
                TextBlock(
                    bbox=b.bbox,
                    text="\n".join(txt_lines),
                    max_font_size=b.max_font_size,
                    is_bold=b.is_bold,
                    insert_image=b.insert_image,
                    is_code=b.is_code,
                    is_table=b.is_table,
                    is_math=b.is_math,
                    heading_level=b.heading_level,
                )
            )
        else:
            out.append(b)
    return out, eqno_by_block


def drop_spurious_math_fragments(blocks: list[TextBlock]) -> list[TextBlock]:
    """
    Some PDFs split one display equation into a real math block plus one or more tiny "math-looking"
    body blocks (e.g. "i∈N c_i α_i"). These fragments should be dropped (they are duplicates),
    otherwise they pollute the surrounding paragraph.
    """
    if not blocks:
        return blocks
    out: list[TextBlock] = []
    sym_re = re.compile(r"[=^_\\]|[∑Σ∏√∫∞≤≥≠≈∥∈αβγδεζηθικλμνξοπρστυφχψω]")
    for i, b in enumerate(blocks):
        if b.is_math or b.is_table or b.is_code:
            out.append(b)
            continue
        t = _normalize_text(b.text).strip()
        if not t:
            continue
        # lone dashes created by bullet splitting / extraction noise
        if t in {"-", "–", "—"}:
            continue
        if len(t) <= 40 and sym_re.search(t) and _looks_like_equation_text(t):
            neigh_math = False
            for di in (-2, -1, 1, 2):
                j = i + di
                if 0 <= j < len(blocks) and blocks[j].is_math:
                    neigh_math = True
                    break
            if neigh_math:
                continue
        out.append(b)
    return out


def extract_figures_by_captions(
    page,
    blocks: list[TextBlock],
    asset_dir: Path,
    page_index: int,
    *,
    image_scale: float = 2.0,
    image_alpha: bool = False,
) -> tuple[dict[int, str], list["fitz.Rect"]]:
    caption_re = re.compile(r"^\s*(?:Fig\.|FIG\.|Figure|FIGURE)\s*([0-9]+)", re.IGNORECASE)
    img_infos = page.get_image_info() or []
    img_rects = [fitz.Rect(i["bbox"]) for i in img_infos if "bbox" in i]

    out: dict[int, str] = {}
    covered: list[fitz.Rect] = []
    for bi, b in enumerate(blocks):
        m = caption_re.match(b.text)
        if not m:
            continue

        fig_num = m.group(1)
        caption_rect = fitz.Rect(b.bbox)
        col_x0, col_x1 = _pick_column_range(caption_rect, page.rect.width)

        candidates: list[tuple[float, fitz.Rect]] = []
        for r in img_rects:
            if r.y1 > caption_rect.y0 + 10:
                continue
            overlap = _overlap_1d(r.x0, r.x1, col_x0, col_x1)
            if overlap < min(r.width, (col_x1 - col_x0)) * 0.2:
                continue
            gap = caption_rect.y0 - r.y1
            candidates.append((gap, r))

        # Dynamic crop: capture the full column region immediately above the caption.
        # Rationale: `get_image_info()` bboxes are frequently too tight (vector overlays, labels),
        # so we prioritize recall even if we include a bit of extra whitespace.
        max_up = min(page.rect.height * 0.80, caption_rect.y0)
        y0 = max(0.0, caption_rect.y0 - max_up)
        # Find previous text block in same column to cap crop top.
        prev_y1: Optional[float] = None
        for pj in range(bi - 1, -1, -1):
            pr = fitz.Rect(blocks[pj].bbox)
            # same column overlap
            overlap = _overlap_1d(pr.x0, pr.x1, col_x0, col_x1)
            if overlap >= min(pr.width, (col_x1 - col_x0)) * 0.3 and pr.y1 <= caption_rect.y0:
                # Only cap by real paragraph text; do NOT cap by short labels that are often part of the figure.
                txt = _normalize_text(blocks[pj].text)
                if len(txt) >= 90 and not caption_re.match(txt) and not _looks_like_equation_text(txt):
                    prev_y1 = float(pr.y1)
                    break
        if prev_y1 is not None:
            y0 = max(y0, prev_y1 + 6.0)

        # If we have image rectangles, expand upward to include their full extent (avoid cutting tops).
        if candidates:
            img_top = min(r.y0 for _, r in candidates)
            y0 = min(y0, max(0.0, float(img_top) - 10.0))
            for _, r in candidates:
                covered.append(fitz.Rect(r))

        pad = 12.0
        crop = fitz.Rect(
            max(0.0, col_x0 - pad),
            max(0.0, y0 - pad),
            min(page.rect.width, col_x1 + pad),
            min(page.rect.height, caption_rect.y1 + pad),
        )

        img_name = f"figure_{fig_num}_p{page_index + 1:03d}.png"
        pix = page.get_pixmap(matrix=fitz.Matrix(float(image_scale), float(image_scale)), clip=crop, alpha=bool(image_alpha))
        pix.save(str(asset_dir / img_name))
        out[bi] = img_name

    return out, covered


def extract_images_fallback(
    page,
    blocks: list[TextBlock],
    asset_dir: Path,
    page_index: int,
    *,
    covered_rects: list["fitz.Rect"],
    image_scale: float = 2.0,
    image_alpha: bool = False,
) -> dict[int, str]:
    """
    Best-effort capture of images that don't have detectable captions.
    Goal: maximize recall (avoid missing figures), while filtering obvious tiny logos/icons.
    """
    img_infos = page.get_image_info() or []
    img_rects = [fitz.Rect(i["bbox"]) for i in img_infos if "bbox" in i]
    if not img_rects:
        return {}

    page_w = float(page.rect.width)
    page_h = float(page.rect.height)
    page_area = max(1.0, page_w * page_h)

    def is_covered(r: fitz.Rect) -> bool:
        ra = max(1.0, float(r.width) * float(r.height))
        for c in covered_rects or []:
            inter = r.intersect(c)
            ia = max(0.0, float(inter.width) * float(inter.height))
            if ia / ra >= 0.55:
                return True
        return False

    # anchor each image to the nearest following text block (caption/paragraph) to keep reading flow
    anchors = sorted([(i, fitz.Rect(b.bbox)) for i, b in enumerate(blocks)], key=lambda x: (x[1].y0, x[1].x0))

    out: dict[int, str] = {}
    auto_i = 0
    for r in img_rects:
        if is_covered(r):
            continue
        # Filter tiny icons / decorative rules.
        area = max(0.0, float(r.width) * float(r.height))
        if area < page_area * 0.006:
            continue
        if float(r.width) < page_w * 0.14 or float(r.height) < page_h * 0.08:
            continue
        # Filter very thin rules / separators.
        if float(r.height) < page_h * 0.02 and float(r.width) > page_w * 0.7:
            continue
        # Avoid headers/footers.
        if r.y1 < 60.0 or r.y0 > page_h - 60.0:
            continue

        # find nearest block below the image (prefer same column)
        mid = page_w / 2.0
        r_col = 0 if r.x0 < mid else 1
        best_i = None
        best_gap = None
        for bi, br in anchors:
            b_col = 0 if br.x0 < mid else 1
            if b_col != r_col:
                continue
            gap = float(br.y0) - float(r.y1)
            if gap < -15.0:
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_i = bi
        if best_i is None:
            # fallback: anchor to the last block
            best_i = anchors[-1][0] if anchors else 0

        pad = 6.0
        crop = fitz.Rect(
            max(0.0, r.x0 - pad),
            max(0.0, r.y0 - pad),
            min(page_w, r.x1 + pad),
            min(page_h, r.y1 + pad),
        )
        auto_i += 1
        img_name = f"image_auto_{auto_i:02d}_p{page_index + 1:03d}.png"
        pix = page.get_pixmap(matrix=fitz.Matrix(float(image_scale), float(image_scale)), clip=crop, alpha=bool(image_alpha))
        pix.save(str(asset_dir / img_name))
        out.setdefault(best_i, img_name)
    return out


def extract_equation_images_for_garbled_math(
    page,
    blocks: list[TextBlock],
    asset_dir: Path,
    page_index: int,
    *,
    eqno_by_block: dict[int, str],
    image_scale: float = 2.0,
    image_alpha: bool = False,
) -> dict[int, str]:
    """
    If extracted math is obviously garbled, render the equation region as an image.

    This avoids emitting wrong/broken LaTeX and stays strictly faithful to the PDF.
    """
    out: dict[int, str] = {}
    if fitz is None:
        return out

    page_w = float(page.rect.width)
    page_h = float(page.rect.height)
    pad = 8.0

    for bi, b in enumerate(blocks):
        if not b.is_math:
            continue
        if not _math_text_looks_garbled(b.text):
            continue
        rect = fitz.Rect(b.bbox)
        col_x0, col_x1 = _pick_column_range(rect, page_w)
        crop = fitz.Rect(
            max(0.0, col_x0 - pad),
            max(0.0, rect.y0 - pad),
            min(page_w, col_x1 + pad),
            min(page_h, rect.y1 + pad),
        )

        eqno = (eqno_by_block or {}).get(bi, "").strip()
        if eqno and re.fullmatch(r"\d{1,4}", eqno):
            img_name = f"equation_{eqno}_p{page_index + 1:03d}_b{bi:02d}.png"
        else:
            img_name = f"equation_auto_p{page_index + 1:03d}_b{bi:02d}.png"

        try:
            pix = page.get_pixmap(
                matrix=fitz.Matrix(float(image_scale), float(image_scale)),
                clip=crop,
                alpha=bool(image_alpha),
            )
            pix.save(str(asset_dir / img_name))
            out[bi] = img_name
        except Exception:
            continue

    return out


def extract_text_blocks(
    page,
    *,
    body_size: float,
    noise_texts: Optional[set[str]] = None,
    page_index: int = 0,
    drop_frontmatter: bool = True,
    relax_small_text_filter: bool = False,
    preserve_body_linebreaks: bool = False,
) -> list[TextBlock]:
    d = page.get_text("dict")
    blocks: list[TextBlock] = []
    W, H = float(page.rect.width), float(page.rect.height)
    # References often sit closer to the top/bottom in two-column layouts. Relax bands there.
    header_y = 25.0 if relax_small_text_filter else 75.0
    footer_y = H - (25.0 if relax_small_text_filter else 70.0)

    img_infos = page.get_image_info() or []
    img_rects = [fitz.Rect(i["bbox"]) for i in img_infos if "bbox" in i]

    for b in d.get("blocks", []):
        if "lines" not in b:
            continue
        bbox = tuple(float(x) for x in b.get("bbox", (0, 0, 0, 0)))
        rect = fitz.Rect(bbox)

        if rect.y1 < header_y or rect.y0 > footer_y:
            continue

        raw_lines: list[str] = []
        max_size = 0.0
        bold = False
        for l in b.get("lines", []) or []:
            spans = l.get("spans", []) or []
            if not spans:
                continue
            line = "".join(str(s.get("text", "")) for s in spans)
            if line.strip():
                raw_lines.append(line)
            for s in spans:
                try:
                    max_size = max(max_size, float(s.get("size", 0.0)))
                except Exception:
                    pass
                try:
                    if int(s.get("flags", 0)) & (2**4):
                        bold = True
                except Exception:
                    pass

        probe = [_normalize_text(x) for x in raw_lines if _normalize_text(x)]
        if not probe:
            continue

        # If a text block is almost fully inside an image rect, it's often garbage (publisher logos),
        # but captions may also intersect (depending on how the PDF encodes bbox). Keep captions and equation numbers.
        if img_rects:
            block_area = max(1.0, _rect_area(rect))
            inter = max((_rect_area(rect.intersect(r)) for r in img_rects), default=0.0)
            inside_ratio = inter / block_area
            if inside_ratio > 0.78:
                joined_probe = _normalize_text(" ".join(probe))
                if not re.match(r"^\s*(?:Fig\.|FIG\.|Figure|FIGURE)\s*[0-9]+", joined_probe, re.IGNORECASE) and not re.fullmatch(
                    r"\(\s*\d{1,4}\s*\)", joined_probe
                ):
                    # still allow real math/code blocks which sometimes intersect due to bbox quirks
                    if max_size < body_size - 0.2 and not _looks_like_math_block(probe) and not _looks_like_code_block(probe):
                        continue

        # Extra non-body filter: small text near margins/bands (journal footers, figure credits, etc.)
        # Disable/relax this in REFERENCES pages to avoid dropping right-column items and page-continued refs.
        if (not relax_small_text_filter) and max_size < body_size - 0.6:
            near_top = rect.y1 < 160.0
            near_bottom = rect.y0 > H - 160.0
            near_side = rect.x0 < 40.0 or rect.x1 > W - 40.0
            # Keep equation numbers even if they sit near the right margin.
            joined_probe = _normalize_text(" ".join(probe))
            # Keep figure captions even when small and close to margins (common in two-column PDFs).
            is_eqno = bool(re.fullmatch(r"\(\s*\d{1,4}\s*\)", joined_probe))
            is_caption = bool(re.match(r"^\s*(?:Fig\.|FIG\.|Figure|FIGURE)\s*[0-9]+", joined_probe, re.IGNORECASE))
            if (near_top or near_bottom or near_side) and (not is_eqno) and (not is_caption):
                continue

        is_table = _looks_like_table_block(probe)
        is_math = (not is_table) and _looks_like_math_block(probe)
        is_code = (not is_table) and (not is_math) and _looks_like_code_block(probe)

        if is_table or is_math or is_code or preserve_body_linebreaks or _looks_like_toc_lines(probe):
            norm_lines = [_normalize_line_keep_indent(x) for x in raw_lines]
            text = "\n".join([ln.rstrip() for ln in norm_lines if ln.strip()])
        else:
            text = _join_lines_preserving_words(raw_lines)
        if not text:
            continue
        if noise_texts and text in noise_texts:
            continue
        if _is_noise_line(text):
            continue
        if drop_frontmatter and page_index <= 2 and _is_frontmatter_noise_line(text):
            continue

        blocks.append(
            TextBlock(
                bbox=bbox,
                text=text,
                max_font_size=max_size,
                is_bold=bold,
                is_code=is_code,
                is_table=is_table,
                is_math=is_math,
            )
        )

    return blocks


def sort_blocks_reading_order(blocks: list[TextBlock], page_width: float) -> list[TextBlock]:
    if not blocks:
        return []

    mid = page_width / 2.0
    spanning_threshold = page_width * 0.6
    col_split = mid - 10.0

    by_y = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

    out: list[TextBlock] = []
    segment: list[TextBlock] = []

    def flush_segment():
        nonlocal segment
        if not segment:
            return
        left = [b for b in segment if b.bbox[0] < col_split]
        right = [b for b in segment if b.bbox[0] >= col_split]
        out.extend(left)
        out.extend(right)
        segment = []

    for b in by_y:
        if _bbox_width(b.bbox) >= spanning_threshold:
            flush_segment()
            out.append(b)
        else:
            segment.append(b)

    flush_segment()
    return out


def _tagged_page_text(
    *,
    page_index: int,
    page: fitz.Page,
    blocks: list[TextBlock],
    body_size: float,
    figures_by_block: dict[int, str],
    eqno_by_block: Optional[dict[int, str]] = None,
) -> str:
    W = float(page.rect.width)
    out: list[str] = []
    eqno_by_block = eqno_by_block or {}

    for bi, b in enumerate(blocks):
        img = figures_by_block.get(bi)
        if img:
            out.append(f"[IMAGE: {img}]")
            out.append("")

        if b.heading_level:
            lvl = int(b.heading_level)
            if 1 <= lvl <= 3 and (_is_numbered_heading_text(b.text.strip()) or _is_appendix_heading_text(b.text.strip())):
                out.append(f"[H{lvl}] {b.text.strip()}")
                out.append("")
                continue

        if b.is_table:
            out.append("[TABLE]")
            out.append(b.text)
            out.append("[/TABLE]")
            out.append("")
            continue

        if b.is_math:
            if not b.text.strip():
                continue
            out.append("[MATH]")
            eqno = eqno_by_block.get(bi)
            if eqno:
                out.append(f"(EQNO {eqno})")
            out.append(b.text)
            out.append("[/MATH]")
            out.append("")
            continue

        if b.is_code:
            out.append("[CODE]")
            out.append(b.text)
            out.append("[/CODE]")
            out.append("")
            continue

        tag = detect_header_tag(
            page_index=page_index,
            text=b.text,
            max_size=b.max_font_size,
            is_bold=b.is_bold,
            body_size=body_size,
            page_width=W,
            bbox=b.bbox,
        )
        if tag:
            out.append(f"{tag} {b.text}")
        else:
            out.append(b.text)
        out.append("")

    return "\n".join(out).strip() + "\n"


def _chunk_by_blank_lines(s: str, max_chars: int) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paras:
        if cur and cur_len + len(p) + 2 > max_chars:
            chunks.append("\n\n".join(cur))
            cur = [p]
            cur_len = len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks or [s]


def _fix_spaced_urls(line: str) -> str:
    line = re.sub(r"(https?://)\s+", r"\1", line)
    line = re.sub(r"(doi\.org/)\s+", r"\1", line)
    line = re.sub(r"(doi\.org/10\.)\s+", r"\1", line)
    line = re.sub(r"(doi:\s*10\.)\s+", r"\1", line, flags=re.IGNORECASE)
    if "doi.org/" in line:
        line = re.sub(r"(doi\.org/\S*?\d)\.\s+(\d)", r"\1.\2", line)
        line = re.sub(r"(doi\.org/\S+)\s+(\d)", r"\1\2", line)
    return line


def _cleanup_noise_lines(md: str) -> str:
    out: list[str] = []
    for line in md.splitlines():
        # repair known ligatures / private-use glyphs even in already-generated Markdown
        for k, v in LIGATURES.items():
            line = line.replace(k, v)
        if _is_noise_line(line):
            continue
        out.append(_fix_spaced_urls(line.rstrip()))
    return "\n".join(out).strip() + "\n"


def _split_inline_bullets(md: str) -> str:
    out: list[str] = []
    for line in md.splitlines():
        if "•" not in line:
            out.append(line)
            continue
        parts = [p.strip() for p in line.split("•") if p.strip()]
        if len(parts) <= 1:
            out.append(line.replace("•", "- "))
            continue
        for p in parts:
            out.append(f"- {p}")
    return "\n".join(out)


def _repair_split_citations_around_images(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    open_bracket_re = re.compile(r"\[[^\]]*$")
    img_re = re.compile(r"^!\[[^\]]*\]\((?:\./)?assets/[^)]+\)\s*$")
    fence_re = re.compile(r"^\s*```")
    in_fence = False

    while i < len(lines):
        line = lines[i]
        if fence_re.match(line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue

        if (not in_fence) and open_bracket_re.search(line):
            j = i + 1
            imgs: list[str] = []
            blanks: list[str] = []
            while j < len(lines) and (not lines[j].strip() or img_re.match(lines[j].strip())):
                if img_re.match(lines[j].strip()):
                    imgs.append(lines[j].strip())
                else:
                    blanks.append(lines[j])
                j += 1
            if j < len(lines) and "]" in lines[j]:
                merged = (line.rstrip() + " " + lines[j].lstrip()).rstrip()
                out.append(merged)
                if imgs:
                    out.append("")
                    out.extend(imgs)
                    out.append("")
                i = j + 1
                continue

        out.append(line)
        i += 1

    return "\n".join(out)


def _fence_algorithms(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_fence = False
    algo_re = re.compile(r"^\*\*Algorithm\s+\d+\*\*", re.IGNORECASE)
    end_re = re.compile(r"^\s*end\s+(?:function|while)\b", re.IGNORECASE)
    heading_re = re.compile(r"^#{1,3}\s+")

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue

        if (not in_fence) and algo_re.match(line.strip()):
            out.append(line)
            i += 1
            # keep immediate blank line if present
            if i < len(lines) and not lines[i].strip():
                out.append(lines[i])
                i += 1
            # already fenced?
            if i < len(lines) and lines[i].strip().startswith("```"):
                continue

            out.append("```")
            while i < len(lines):
                cur = lines[i]
                if heading_re.match(cur):
                    break
                out.append(cur)
                i += 1
                if end_re.match(cur.strip()):
                    # close after a following blank line (if any)
                    while i < len(lines) and not lines[i].strip():
                        out.append(lines[i])
                        i += 1
                    break
            out.append("```")
            continue

        out.append(line)
        i += 1

    return "\n".join(out)


def _convert_fenced_blocks(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    in_fence = False
    fence_buf: list[str] = []

    def is_number(s: str) -> bool:
        s = s.strip()
        return bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?", s))

    def split_cols(s: str) -> list[str]:
        return [c.strip() for c in re.split(r"\t+|\s{2,}", s.strip()) if c.strip()]

    def looks_like_math_block(block: list[str]) -> bool:
        joined = " ".join(block)
        if "\\begin{" in joined or "\\frac" in joined or "\\sum" in joined or "\\prod" in joined:
            return True
        math_chars = re.findall(r"[=+\-*/^_{}()\[\]<>Σσμλαδτφ∑∏√≈≤≥∞]", joined)
        if len(math_chars) < 8:
            return False
        words = re.findall(r"[A-Za-z]{3,}", joined)
        return len(words) <= 3

    def try_flattened_table(block: list[str]) -> Optional[str]:
        seq = [ln.strip() for ln in block if ln.strip()]
        if len(seq) < 12:
            return None

        # Try to reconstruct a "flattened" table:
        #   headers (N lines)
        #   (row_label + N numeric values) repeated
        for n in range(3, 16):
            if len(seq) < n + (n + 1) * 3:
                continue
            headers = seq[:n]
            if any(is_number(h) for h in headers):
                continue
            rest = seq[n:]

            rows: list[tuple[str, list[str]]] = []
            k = 0
            ok = True
            while k < len(rest):
                label = rest[k]
                if is_number(label):
                    ok = False
                    break
                if k + 1 + n > len(rest):
                    ok = False
                    break
                vals = rest[k + 1 : k + 1 + n]
                if not all(is_number(v) for v in vals):
                    ok = False
                    break
                rows.append((label, vals))
                k = k + 1 + n
            if not ok or len(rows) < 3:
                continue

            md_lines: list[str] = []
            md_lines.append("|  | " + " | ".join(headers) + " |")
            md_lines.append("| --- | " + " | ".join(["---"] * n) + " |")
            for label, vals in rows:
                md_lines.append("| " + label + " | " + " | ".join(vals) + " |")
            return "\n".join(md_lines)

        # Standard grid table (space-separated columns).
        col_counts = [len(split_cols(x)) for x in seq]
        if sum(1 for c in col_counts if c >= 3) >= 3:
            rows = [split_cols(x) for x in seq if split_cols(x)]
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            header = rows[0]
            md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
            for r in rows[1:]:
                md_lines.append("| " + " | ".join(r) + " |")
            return "\n".join(md_lines)

        return None

    def flush_fence() -> None:
        nonlocal fence_buf
        if not fence_buf:
            out.append("```")
            out.append("```")
            return

        block = fence_buf
        fence_buf = []
        if looks_like_math_block(block):
            out.append("$$")
            out.extend([ln.rstrip() for ln in block if ln.strip()])
            out.append("$$")
            return
        table_md = try_flattened_table(block)
        if table_md:
            out.extend(table_md.splitlines())
            return
        out.append("```")
        out.extend(block)
        out.append("```")

    for line in lines:
        if line.strip().startswith("```"):
            if not in_fence:
                in_fence = True
                fence_buf = []
            else:
                in_fence = False
                flush_fence()
            continue

        if in_fence:
            fence_buf.append(line.rstrip())
        else:
            out.append(line)

    if in_fence:
        # unclosed fence; keep as-is
        out.append("```")
        out.extend(fence_buf)
    return "\n".join(out)


def table_text_to_markdown(text: str) -> Optional[str]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    def split_cols(s: str) -> list[str]:
        return [c.strip() for c in re.split(r"\t+|\s{2,}", s.strip()) if c.strip()]

    rows = [split_cols(ln) for ln in lines]
    rows = [r for r in rows if r]
    if not rows:
        return None
    width = max(len(r) for r in rows)
    if width < 2:
        return None
    # pad
    rows = [r + [""] * (width - len(r)) for r in rows]

    header = rows[0]
    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        md_lines.append("| " + " | ".join(r) + " |")
    return "\n".join(md_lines)


def _latex_array_to_markdown_table(latex: str) -> Optional[str]:
    # Accept \begin{array}{|l|c|...|} ... \end{array} or tabular-like.
    m = re.search(r"\\begin\{array\}\{[^}]*\}(.*?)\\end\{array\}", latex, flags=re.S)
    if not m:
        return None
    body = m.group(1)
    body = body.replace("\\hline", "")
    body = body.strip()
    # split rows
    raw_rows = [r.strip() for r in body.split("\\\\") if r.strip()]
    if len(raw_rows) < 2:
        return None
    rows: list[list[str]] = []
    for r in raw_rows:
        cols = [c.strip() for c in r.split("&")]
        cols = [re.sub(r"\\text\{([^}]*)\}", r"\1", c) for c in cols]
        cols = [re.sub(r"\s+", " ", c).strip() for c in cols]
        rows.append(cols)
    width = max(len(r) for r in rows)
    if width < 2:
        return None
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = rows[0]
    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        md_lines.append("| " + " | ".join(r) + " |")
    return "\n".join(md_lines)


def _convert_latex_array_math_blocks(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "$$":
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        buf: list[str] = []
        while j < len(lines) and lines[j].strip() != "$$":
            buf.append(lines[j])
            j += 1
        if j >= len(lines):
            out.append(lines[i])
            out.extend(buf)
            break
        latex = "\n".join(buf)
        table = _latex_array_to_markdown_table(latex)
        if table:
            out.extend(table.splitlines())
            out.append("")
        else:
            out.append("$$")
            out.extend(buf)
            out.append("$$")
        i = j + 1
    return "\n".join(out)


def _reflow_hard_wrapped_paragraphs(md: str) -> str:
    """
    PDFs often hard-wrap paragraphs at a fixed width. Merge those lines back into
    normal Markdown paragraphs, while preserving structure (headings, lists, code, math, tables, images).
    """
    lines = md.splitlines()
    out: list[str] = []
    buf: list[str] = []

    fence_re = re.compile(r"^\s*```")
    heading_re = re.compile(r"^\s*#{1,6}\s+")
    list_re = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")
    quote_re = re.compile(r"^\s*>")
    table_re = re.compile(r"^\s*\|")
    img_re = re.compile(r"^\s*!\[[^\]]*\]\((?:\./)?assets/[^)]+\)\s*$")

    in_fence = False
    in_math = False
    in_refs = False

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        # Join with spaces, but preserve hyphenation across line breaks.
        parts: list[str] = []
        for ln in buf:
            ln = ln.strip()
            if not ln:
                continue
            if parts and parts[-1].endswith("-") and ln and ln[0].islower():
                parts[-1] = parts[-1][:-1] + ln
            else:
                parts.append(ln)
        out.append(" ".join(parts).strip())
        buf = []

    for ln in lines:
        s = ln.rstrip("\n\r")
        st = s.strip()

        # Do NOT reflow REFERENCES; we want to preserve original line breaks for reliable splitting/formatting.
        if (not in_refs) and re.fullmatch(r"(?:#\s+)?REFERENCES", st, flags=re.IGNORECASE):
            flush()
            in_refs = True
            out.append(s if st.startswith("#") else "REFERENCES")
            continue
        if in_refs:
            # Stop at appendix-like headings.
            if st and re.fullmatch(r"[A-Z]", st) and len(st) == 1:
                flush()
                out.append(s)
                in_refs = False
                continue
            if heading_re.match(st) and (re.search(r"\bAPPENDIX\b", st, flags=re.IGNORECASE) or re.match(r"^#\s*[A-Z]\b", st)):
                flush()
                out.append(s)
                in_refs = False
                continue
            out.append(s)
            continue
        if fence_re.match(st):
            flush()
            in_fence = not in_fence
            out.append(s)
            continue
        if in_fence:
            out.append(s)
            continue

        if st == "$$":
            flush()
            in_math = not in_math
            out.append(s)
            continue
        if in_math:
            out.append(s)
            continue

        # Paragraph boundary
        if not st:
            flush()
            out.append("")
            continue

        # Keep structural lines as-is
        if heading_re.match(st) or list_re.match(st) or quote_re.match(st) or table_re.match(st) or img_re.match(st):
            flush()
            out.append(s)
            continue

        buf.append(s)

    flush()
    return "\n".join(out)


def _format_references(md: str) -> str:
    lines = md.splitlines()
    heading_re = re.compile(r"^#{1,3}\s+REFERENCES\s*$", re.IGNORECASE)
    plain_re = re.compile(r"^\s*REFERENCES\s*$", re.IGNORECASE)
    top_heading_re = re.compile(r"^#\s+")
    idx_line_re = re.compile(r"^\s*(\d+)\.\s+\S")

    def _is_ref_index_line(ln: str) -> bool:
        m = idx_line_re.match(ln.strip())
        if not m:
            return False
        try:
            n = int(m.group(1))
        except Exception:
            return False
        # Guard: years like "2022. ..." are NOT list indices.
        return 1 <= n <= 600

    i = 0
    out: list[str] = []
    found = False
    ref_block: list[str] = []
    tail_after_refs: list[str] = []

    while i < len(lines):
        line = lines[i]
        if not found:
            if heading_re.match(line.strip()) or plain_re.match(line):
                found = True
                out.append("# REFERENCES")
            else:
                out.append(line)
            i += 1
            continue

        # found references: collect until an appendix-like heading starts.
        # Some converters/LLMs may accidentally emit headings in the middle (page headers, etc.) - ignore those.
        if top_heading_re.match(line) and not heading_re.match(line.strip()):
            title = re.sub(r"^#\s+", "", line.strip()).strip()
            if _is_appendix_heading_text(title) or _is_numbered_heading_text(title):
                tail_after_refs = lines[i:]
                break
            # ignore spurious headings inside references (do not stop collection)
            i += 1
            continue
        # Some PDFs (especially two-column) lose heading markers and emit appendix as:
        # "A" newline "DETAILS OF ...". Stop references before that.
        if line.strip() and re.fullmatch(r"[A-Z]", line.strip()):
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt and nxt.upper() == nxt and len(nxt) >= 6:
                tail_after_refs = lines[i:]
                break
        if heading_re.match(line.strip()):
            # skip repeated "REFERENCES" headings
            i += 1
            continue
        ref_block.append(line)
        i += 1

    if not found:
        return md

    # Decide whether we're looking at a true numbered reference list (1., 2., 3., ...)
    idxs: list[int] = []
    for ln in ref_block:
        m = idx_line_re.match(ln.strip())
        if not m:
            continue
        try:
            idxs.append(int(m.group(1)))
        except Exception:
            continue
    numbered_mode = False
    if idxs:
        small = [n for n in idxs if 1 <= n <= 50]
        if len(idxs) >= 10 and 1 in idxs and len(small) >= 5:
            numbered_mode = True

    # If we already have a solid numbered list, do not rewrite it (rewriting can break good references).
    num_cnt = sum(1 for ln in ref_block if re.match(r"^\s*\d+\.\s+\S", ln.strip()))
    other_cnt = sum(1 for ln in ref_block if ln.strip() and not re.match(r"^\s*\d+\.\s+\S", ln.strip()) and len(ln.strip()) >= 20)
    if numbered_mode and num_cnt >= 12 and other_cnt < 10:
        out.extend(ref_block)
        out.extend(tail_after_refs)
        return "\n".join(out)

    # If reference lines are already numbered, preserve entries but renumber sequentially.
    numbered_items: list[str] = []
    cur: list[str] = []
    for ln in ref_block:
        if re.match(r"^#\s*\d+\s*$", ln.strip()):
            continue
        m = idx_line_re.match(ln.strip())
        if numbered_mode and m and _is_ref_index_line(ln):
            if cur:
                numbered_items.append(" ".join(x.strip() for x in cur if x.strip()))
                cur = []
            cur.append(ln.strip()[m.end(1) + 1 :].strip())
        else:
            cur.append(ln)
    if cur:
        numbered_items.append(" ".join(x.strip() for x in cur if x.strip()))

    def split_raw_refs(raw: str) -> list[str]:
        raw = _fix_spaced_urls(re.sub(r"\s+", " ", raw)).strip()
        if not raw:
            return []
        boundary = re.compile(r"(?<=\.)\s+(?=[A-Z].{0,240}?\b(?:19|20)\d{2}\.)")
        pieces: list[str] = []
        last = 0
        for m in boundary.finditer(raw):
            pieces.append(raw[last : m.start()].strip())
            last = m.start()
        pieces.append(raw[last:].strip())
        pieces = [p for p in pieces if len(p) >= 10]
        merged: list[str] = []
        cont_re = re.compile(r"^(In|Proceedings|Association|IEEE|ACM|Springer|Wiley|Elsevier|arXiv|http|https)\b")
        for p in pieces:
            if merged and (cont_re.match(p) or re.match(r"^[a-z]", p)):
                merged[-1] = (merged[-1].rstrip() + " " + p.lstrip()).strip()
            else:
                merged.append(p)
        pieces = merged
        if len(pieces) <= 1:
            alt = re.split(r"(?<=\.)\s+(?=[A-Z])", raw)
            pieces = [p.strip() for p in alt if len(p.strip()) >= 10]
        return pieces

    if numbered_mode and any(x.strip() for x in numbered_items) and any(_is_ref_index_line(x) for x in ref_block):
        items = [split_raw_refs(x)[0] if len(split_raw_refs(x)) == 1 else x for x in numbered_items]
        items = [x.strip() for x in items if len(x.strip()) >= 10]
    else:
        raw_lines = [x.strip() for x in ref_block if x.strip()]
        # NOTE: use single backslashes in raw strings (e.g. r"\b", r"\d") so regex works as intended.
        cont_re = re.compile(
            r"^(In|Proceedings|Proc\.?|Association|IEEE|ACM|Springer|Wiley|Elsevier|arXiv|CVPR|ICCV|ECCV|SIGGRAPH|Computer Graphics Forum|http|https)\b",
            re.IGNORECASE,
        )
        year_re = re.compile(r"(?:19|20)\d{2}")
        non_author_prefix = re.compile(
            r"^(European|Computer\s+Vision|Conference|Transactions|Association|Eurographics|Euro|Graphics|ACM|IEEE)\b",
            re.IGNORECASE,
        )

        # Glue common reference line-break artifacts:
        # - Hyphenation: "Lem-" + "pitsky" => "Lem-pitsky" (or "Euro-" + "graphics" => "Eurographics" best-effort)
        # - Author initials split: "Jonathan T." + "Barron, Ben ..." => "Jonathan T. Barron, Ben ..."
        glued: list[str] = []
        k = 0
        initials_re = re.compile(r"^[A-Z][A-Za-z]{1,}\s+[A-Z]\.$")
        while k < len(raw_lines):
            ln = raw_lines[k].strip()
            if k + 1 < len(raw_lines):
                nxt = raw_lines[k + 1].strip()

                # Hyphenated line break: merge without space. If next begins with lowercase, drop the hyphen.
                if ln.endswith("-") and nxt and nxt[0].isalpha():
                    if nxt[0].islower():
                        ln = (ln[:-1] + nxt).strip()
                    else:
                        ln = (ln + nxt).strip()
                    k += 1
                    glued.append(ln)
                    k += 1
                    continue

                # Only glue when the left part looks like a name + initial (avoid merging page ranges like "712.").
                if initials_re.match(ln) and nxt and nxt[0].isupper() and "," in nxt:
                    ln = (ln + " " + nxt).strip()
                    k += 1

            glued.append(ln)
            k += 1
        raw_lines = glued

        def _is_authorish_line(ln: str) -> bool:
            ln = ln.strip()
            if len(ln) < 12:
                return False
            if not ln[0].isalpha() or not ln[0].isupper():
                return False
            if cont_re.match(ln):
                return False
            if non_author_prefix.match(ln):
                return False
            # Author lines usually contain commas / "and" / "et al."
            low = ln.lower()
            if ("et al" not in low) and (" and " not in low):
                # Without "and"/"et al", require a stronger comma signal to avoid matching venue lines like
                # "ACM Trans. Graph. 41, 4, Article ...".
                if ln.count(",") < 3:
                    return False
            else:
                if ln.count(",") < 1:
                    return False
            # Require a comma that looks like it separates names, not numbers ("41, 4").
            if not re.search(r",\s*[A-Z]", ln[:90]):
                return False
            # Entry starts are very unlikely to contain digits immediately (helps avoid "ACM Trans. Graph. 41, 4, ...").
            if re.search(r"\d", ln[:28]):
                return False
            return True

        def looks_like_entry_start(i: int) -> bool:
            ln = raw_lines[i].strip()
            if not _is_authorish_line(ln):
                return False

            # Avoid splitting inside a multi-line author list: if the previous line ends with a comma or hyphen,
            # this line is very likely a continuation, even if it contains a year.
            if i > 0:
                prev = raw_lines[i - 1].strip()
                if prev.endswith(",") or prev.endswith("-"):
                    return False

            look = ln
            for j in range(1, 5):
                if i + j < len(raw_lines):
                    look += " " + raw_lines[i + j]
            if not year_re.search(look):
                return False
            return True

        items2: list[str] = []
        cur2: list[str] = []
        for i, ln in enumerate(raw_lines):
            if re.fullmatch(r"\d{1,4}", ln):
                continue
            if looks_like_entry_start(i) and cur2:
                items2.append(" ".join(cur2).strip())
                cur2 = [ln]
            else:
                cur2.append(ln)
        if cur2:
            items2.append(" ".join(cur2).strip())

        items2 = [_fix_spaced_urls(re.sub(r"\s+", " ", x)).strip() for x in items2 if len(x.strip()) >= 10]
        # Fix common spacing artifacts after hyphenation across lines: "Euro- graphics" -> "Euro-graphics".
        items2 = [re.sub(r"(\w)-\s+(\w)", r"\1-\2", x) for x in items2]
        # Merge obvious continuation fragments (conference/location/page-range lines without a year).
        merged2: list[str] = []
        for it in items2:
            if not merged2:
                merged2.append(it)
                continue
            if len(it) < 60 or not year_re.search(it) or cont_re.match(it):
                merged2[-1] = (merged2[-1].rstrip() + " " + it.lstrip()).strip()
            else:
                merged2.append(it)
        items2 = merged2
        # If segmentation succeeded, keep it; otherwise fallback to regex splitting.
        if len(items2) >= 12:
            items = items2
        else:
            raw = " ".join(raw_lines)
            items = split_raw_refs(raw)

    out.append("")
    for idx, item in enumerate(items, start=1):
        out.append(f"{idx}. {item}")
    out.append("")
    out.extend(tail_after_refs)
    return "\n".join(out)


def _extract_references_from_pdf(doc) -> list[str]:
    """Extract REFERENCES from the PDF using layout (columns + hanging indent).

    This is intentionally rule-based and conservative: it tries to avoid mixing in appendix text
    (which can share the same page as the tail of REFERENCES in two-column layouts).
    """
    if fitz is None:
        return []

    year_re = re.compile(r"\b(?:19|20)\d{2}\b")

    def _iter_page_lines(page) -> list[tuple[float, float, str]]:
        d = page.get_text("dict")
        lines: list[tuple[float, float, str]] = []
        for b in d.get("blocks", []):
            if b.get("type") != 0:
                continue
            for ln in b.get("lines", []):
                t = "".join(sp.get("text", "") for sp in ln.get("spans", [])).strip()
                if not t:
                    continue
                x0, y0, _, _ = ln.get("bbox", (0.0, 0.0, 0.0, 0.0))
                lines.append((float(y0), float(x0), _normalize_text(t)))
        lines.sort(key=lambda z: (z[0], z[1]))
        return lines

    def _is_authorish_line(t: str) -> bool:
        st = t.strip()
        if len(st) < 18:
            return False
        if st.upper() == "REFERENCES":
            return False
        if re.match(r"^\d{1,3}:\d+\s*$", st):
            return False
        if st.startswith("ACM Trans. Graph"):
            return False
        if not st[:1].isalpha() or not st[:1].isupper():
            return False
        if st.count(",") < 1:
            return False
        if not year_re.search(st):
            return False
        return True

    def _is_mathish_line(t: str) -> bool:
        st = t.strip()
        if not st:
            return False
        if re.search(r"[Σ∂=]", st):
            return True
        if "\\begin{" in st or "\\end{" in st:
            return True
        return False

    def _extract_from_page(page) -> tuple[list[str], bool]:
        W = float(page.rect.width)
        all_lines = _iter_page_lines(page)

        # group by column (2-col PDFs); then decide which columns are actually "references-like"
        cols: dict[int, list[tuple[float, float, str]]] = {0: [], 1: []}
        for y, x, t in all_lines:
            cols[0 if x < W / 2 else 1].append((y, x, t))

        saw_appendix = False
        page_entries: list[str] = []

        for col_id in (0, 1):  # left then right
            L = cols.get(col_id) or []
            if not L:
                continue

            # Stop REFERENCES within this column at appendix-like headings "A" newline "DETAILS OF ..."
            stop_y: Optional[float] = None
            for i, (y, _x, t) in enumerate(L):
                st = t.strip()
                if st and re.fullmatch(r"[A-Z]", st):
                    nxt = (L[i + 1][2].strip() if i + 1 < len(L) else "")
                    if nxt and nxt.upper() == nxt and len(nxt) >= 6:
                        stop_y = y - 1e-3
                        saw_appendix = True
                        break

            L2 = [(y, x, t) for (y, x, t) in L if (stop_y is None or y < stop_y)]
            if not L2:
                continue

            # Score using the "references" region only (before any appendix content on the same column).
            author_cnt = sum(1 for _, __, t in L2 if _is_authorish_line(t))
            math_cnt = sum(1 for _, __, t in L2 if _is_mathish_line(t))
            # If this column has little-to-no reference signal, skip it (prevents appendix leakage).
            if author_cnt < 4 or math_cnt > 8:
                continue

            # base indent for entry starts (hanging indent layout)
            cand = [
                x
                for (y, x, t) in L2
                if t.strip().upper() != "REFERENCES"
                and not re.match(r"^\d{1,3}:\d+\s*$", t.strip())
                and "Publication date:" not in t
                and "ACM Trans. Graph" not in t
            ]
            if not cand:
                continue
            base_x = min(cand)
            eps = 3.5

            cur: list[str] = []
            for _y, x, t in L2:
                st = t.strip()
                if not st:
                    continue
                if st.upper() == "REFERENCES":
                    continue
                if re.match(r"^\d{1,3}:\d+\s*$", st):
                    continue
                if "ACM Trans. Graph" in st and "Publication date" in st:
                    continue
                if "Bernhard Kerbl" in st and "Drettakis" in st and len(st) < 160:
                    continue
                if re.fullmatch(r"\d{1,4}\.", st):
                    # stray line like "712." from page ranges
                    continue

                is_new = (x <= base_x + eps) and st[:1].isalpha() and st[:1].isupper()
                if is_new and cur:
                    page_entries.append(" ".join(cur).strip())
                    cur = [st]
                    continue

                if cur and cur[-1].endswith("-") and st[:1].isalpha():
                    cur[-1] = cur[-1][:-1] + st
                else:
                    cur.append(st)

            if cur:
                page_entries.append(" ".join(cur).strip())

        return page_entries, saw_appendix

    items: list[str] = []
    in_refs = False
    for page_index in range(len(doc)):
        page = doc[page_index]
        if not in_refs:
            if _page_has_references_heading(page):
                in_refs = True
            else:
                continue

        page_items, saw_appendix = _extract_from_page(page)
        items.extend(page_items)

        # If this page contains appendix start and we managed to extract some references from it,
        # we're very likely at the end of the REFERENCES section.
        if saw_appendix and page_items:
            break

        # If the page no longer looks like references, stop.
        if not page_items and not _page_looks_like_references_content(page):
            break

    # Normalize + de-duplicate
    cleaned: list[str] = []
    seen: set[str] = set()
    for it in items:
        it = _fix_spaced_urls(re.sub(r"\s+", " ", _normalize_text(it))).strip()
        if len(it) < 10:
            continue
        key = re.sub(r"\W+", "", it).lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(it)
    return cleaned


def _inject_references_section(md: str, refs: list[str]) -> str:
    """Replace (or append) the REFERENCES section in markdown with the given list."""
    if not refs:
        return md
    lines = md.splitlines()
    heading_re = re.compile(r"^#{1,3}\s+REFERENCES\s*$", re.IGNORECASE)
    plain_re = re.compile(r"^\s*REFERENCES\s*$", re.IGNORECASE)
    top_heading_re = re.compile(r"^#\s+")

    start = None
    for i, ln in enumerate(lines):
        if heading_re.match(ln.strip()) or plain_re.match(ln):
            start = i
            break

    ref_md = ["# REFERENCES", ""]
    for i, it in enumerate(refs, start=1):
        ref_md.append(f"{i}. {it}")
    ref_md.append("")

    if start is None:
        return (md.rstrip() + "\n\n" + "\n".join(ref_md)).strip() + "\n"

    # Find the end of the references section (appendix-like heading).
    end = len(lines)
    i = start + 1
    while i < len(lines):
        ln = lines[i]
        st = ln.strip()
        if top_heading_re.match(ln) and not heading_re.match(st):
            title = re.sub(r"^#\s+", "", st).strip()
            if _is_appendix_heading_text(title) or _is_numbered_heading_text(title):
                end = i
                break
        if st and re.fullmatch(r"[A-Z]", st):
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt and nxt.upper() == nxt and len(nxt) >= 6:
                end = i
                break
        i += 1

    new_lines = []
    new_lines.extend(lines[:start])
    new_lines.extend(ref_md)
    new_lines.extend(lines[end:])
    return "\n".join(new_lines).strip() + "\n"


_ALLOWED_UNNUMBERED_HEADINGS = {
    "ABSTRACT",
    "REFERENCES",
    "ACKNOWLEDGMENTS",
    "ACKNOWLEDGEMENTS",
    "APPENDIX",
}


def _is_numbered_heading_text(title: str) -> bool:
    t = _normalize_text(title)
    m = re.match(r"^(\d+(?:\.\d+)*)\s+(.+)$", t)
    if not m:
        return False
    rest = (m.group(2) or "").strip()
    if not rest:
        return False
    # Avoid equations like "2(x)^T ..." and other math lines.
    if not _is_letter(rest[0]):
        return False
    if _looks_like_equation_text(rest):
        return False
    return True


def _is_appendix_heading_text(title: str) -> bool:
    # Common appendix heading styles:
    # - "APPENDIX"
    # - "A DETAILS OF ..."
    # - "B.1 Something" / "C.2.1 Something"
    t = title.strip()
    if t.upper() == "APPENDIX":
        return True
    if re.match(r"^[A-Z]\s+\S", t):
        return True
    if re.match(r"^[A-Z](?:\.\d+)+\s+\S", t):
        return True
    return False


def _fix_split_numbered_headings(md: str) -> str:
    """Fix headings that were split across lines by PDF extraction.

    Common patterns:
      - "2" newline "RELATED WORK"
      - "# 8" newline "DISCUSSION AND CONCLUSIONS"
      - "2.3" newline "Point-Based Rendering and Radiance Fields"
      - "A" newline "DETAILS OF ..."
    """
    lines = md.splitlines()
    out: list[str] = []
    in_code = False
    i = 0
    while i < len(lines):
        ln = lines[i]
        st = ln.strip()

        if st.startswith("```"):
            in_code = not in_code
            out.append(ln)
            i += 1
            continue
        if in_code:
            out.append(ln)
            i += 1
            continue

        # "# 8" + next line => "# 8 TITLE"
        m = re.fullmatch(r"#\s*(\d+(?:\.\d+)*)\s*$", st)
        if m and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if nxt and nxt.upper() == nxt and len(nxt) >= 4:
                lvl = min(6, m.group(1).count(".") + 1)
                out.append("#" * lvl + f" {m.group(1)} {nxt}".rstrip())
                i += 2
                continue

        # "2.3" + next line => "## 2.3 Title"
        m = re.fullmatch(r"(\d+(?:\.\d+)*)\s*$", st)
        if m and i + 1 < len(lines):
            # Guard against DOI fragments like "3530127" or "344779.344936" being mis-promoted to headings.
            tok = m.group(1)
            parts = tok.split(".")
            try:
                first_n = int(parts[0])
            except Exception:
                first_n = 10**9
            if len(tok) > 6 or first_n > 200 or any(len(p) > 2 for p in parts):
                out.append(ln)
                i += 1
                continue
            nxt = lines[i + 1].strip()
            if nxt and (nxt[0].isalpha() or nxt[0].isdigit()):
                lvl = min(6, m.group(1).count(".") + 1)
                out.append("#" * lvl + f" {m.group(1)} {nxt}".rstrip())
                i += 2
                continue

        # Appendix "A" + next line => "# A TITLE"
        if re.fullmatch(r"[A-Z]", st) and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if nxt and nxt.upper() == nxt and len(nxt) >= 6:
                out.append(f"# {st} {nxt}".rstrip())
                i += 2
                continue

        out.append(ln)
        i += 1

    return "\n".join(out)


def _enforce_heading_policy(md: str) -> str:
    """Demote any invented headings that are not numbered section titles.

    Some LLMs will occasionally promote short labels (e.g. figure callout text)
    to headings. For our strict policy we only keep:
      - numbered section headings: 1, 2.3, 7.4.1 ...
      - a small set of paper-level headings like ABSTRACT/REFERENCES.
    Everything else is converted back to plain text.
    """

    lines = md.splitlines()
    out: list[str] = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if not m:
            out.append(line)
            continue

        title = m.group(2).strip()
        if not title:
            continue

        if _is_numbered_heading_text(title) or _is_appendix_heading_text(title):
            out.append(line)
            continue

        if title.upper() in _ALLOWED_UNNUMBERED_HEADINGS:
            out.append("# " + title.upper())
            continue

        # Demote invented headings to plain text.
        out.append(title)

    return "\n".join(out)


def postprocess_markdown(md: str) -> str:
    md = _cleanup_noise_lines(md)
    md = _reflow_hard_wrapped_paragraphs(md)
    md = _fix_split_numbered_headings(md)
    md = _enforce_heading_policy(md)
    md = _split_inline_bullets(md)
    # Bullet splitting can create new noise lines (e.g., page headers with "•"). Clean again.
    md = _cleanup_noise_lines(md)
    # Noise cleanup can reveal split headings; fix again.
    md = _fix_split_numbered_headings(md)
    md = _enforce_heading_policy(md)
    md = _repair_split_citations_around_images(md)
    md = _fence_algorithms(md)
    md = _convert_fenced_blocks(md)
    md = _convert_latex_array_math_blocks(md)
    md = _fix_tag_in_aligned_in_md(md)
    md = _dedupe_consecutive_display_math(md)
    return md.strip() + "\n"


def _fix_tag_in_aligned_in_md(md: str) -> str:
    """
    KaTeX disallows \\tag inside the `aligned` environment. Instead of rewriting environments (fragile),
    move \\tag{n} to the top level of the display-math block: `$$ ... \\tag{n} $$`.
    """
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    tag_re = re.compile(r"\\tag\{([^}]+)\}")

    while i < len(lines):
        if lines[i].strip() != "$$":
            out.append(lines[i])
            i += 1
            continue

        j = i + 1
        buf: list[str] = []
        while j < len(lines) and lines[j].strip() != "$$":
            buf.append(lines[j])
            j += 1
        if j >= len(lines):
            out.append(lines[i])
            out.extend(buf)
            break

        math_text = "\n".join(buf)
        # If a tag was emitted as "\\tag{n}" (newline + tag), normalize to "\tag{n}".
        math_text = math_text.replace("\\\\tag{", "\\tag{")
        if "\\begin{aligned}" in math_text and "\\tag" in math_text:
            tags = tag_re.findall(math_text)
            # remove all tags (we'll re-add one at the end)
            math_text = tag_re.sub("", math_text)
            # if multiple tags existed, keep the last one (best effort)
            if tags:
                math_text = math_text.rstrip() + f"\n\\tag{{{tags[-1].strip()}}}\n"

        out.append("$$")
        out.extend([ln.rstrip() for ln in math_text.splitlines()])
        out.append("$$")
        i = j + 1

    return "\n".join(out)


def _dedupe_consecutive_display_math(md: str) -> str:
    # Remove accidental duplicated display-math blocks (common when extraction splits/duplicates equations).
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    # Common fragment: "G(x)=e^{-1}" appears when a single equation is split across lines/blocks.
    # We'll drop it when an adjacent display-math block has the same LHS and looks more complete.
    frag_re = re.compile(
        r"^\s*([A-Za-z][A-Za-z0-9]*\([^)]*\))\s*=\s*e\s*\^\s*\{\s*-\s*\d+\s*\}\s*$"
    )

    def _lhs(expr: str) -> Optional[str]:
        m = re.match(r"^\s*([A-Za-z][A-Za-z0-9]*\([^)]*\))\s*=", expr.strip())
        return m.group(1) if m else None

    def _last_display_block() -> Optional[tuple[int, str]]:
        # return (start_index_in_out, content_string) for the last $$...$$ block in out
        if not out or out[-1].strip() != "$$":
            return None
        k = len(out) - 2
        buf: list[str] = []
        while k >= 0 and out[k].strip() != "$$":
            buf.append(out[k].rstrip())
            k -= 1
        if k < 0:
            return None
        content = "\n".join(reversed([ln for ln in buf if ln.strip()])).strip()
        return (k, content)

    while i < len(lines):
        if lines[i].strip() != "$$":
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        buf: list[str] = []
        while j < len(lines) and lines[j].strip() != "$$":
            buf.append(lines[j].rstrip())
            j += 1
        if j >= len(lines):
            out.append(lines[i])
            out.extend(buf)
            break
        cur = "\n".join([ln for ln in buf if ln.strip()]).strip()
        # skip empty math blocks entirely
        if not cur:
            i = j + 1
            continue

        # check for duplicate or fragment-vs-complete with immediately previous display block
        last = _last_display_block()
        if last is not None:
            k, prev = last
            if prev and prev == cur:
                i = j + 1
                continue
            prev_lhs = _lhs(prev)
            cur_lhs = _lhs(cur)
            if prev_lhs and cur_lhs and prev_lhs == cur_lhs:
                # If previous is an obvious fragment like "G(x)=e^{-1}", drop it and keep current.
                if frag_re.fullmatch(prev.strip()) and len(cur) >= len(prev) + 12:
                    # remove previous block: out[k:] starts with "$$"
                    out = out[:k]
                # If current is the fragment and previous looks longer, skip current.
                elif frag_re.fullmatch(cur.strip()) and len(prev) >= len(cur) + 12:
                    i = j + 1
                    continue
        out.append("$$")
        out.extend(cur.splitlines())
        out.append("$$")
        i = j + 1
    return "\n".join(out)


def _normalize_display_latex(latex: str, *, eq_number: Optional[str] = None) -> Optional[str]:
    s = _normalize_text(latex or "")
    if not s:
        return None
    # Remove accidental fences some models may add.
    s = re.sub(r"^\s*\$\$\s*", "", s)
    s = re.sub(r"\s*\$\$\s*$", "", s)
    s = re.sub(r"^\s*\\\[\s*", "", s)
    s = re.sub(r"\s*\\\]\s*$", "", s)
    s = s.strip()

    # Never keep model-produced tags; we attach numbering ourselves.
    s = re.sub(r"\\tag\{[^}]+\}", "", s).strip()

    if eq_number:
        n = str(eq_number).strip()
        if n:
            # If an outer equation/align environment exists, inject tag before its \\end.
            if re.search(r"\\begin\{align\*?\}", s) and re.search(r"\\end\{align\*?\}", s):
                s = re.sub(r"(\\end\{align\*?\})", f"\\tag{{{n}}}\n\\1", s, count=1).strip()
            elif re.search(r"\\begin\{equation\*?\}", s) and re.search(r"\\end\{equation\*?\}", s):
                s = re.sub(r"(\\end\{equation\*?\})", f"\\tag{{{n}}}\n\\1", s, count=1).strip()
            else:
                # KaTeX supports top-level \\tag inside $$...$$; avoid placing \\tag inside `aligned`.
                s = s.rstrip() + f"\n\\tag{{{n}}}"

    # Conservative fragment guard: drop bare operators without any real body.
    if len(s) <= 28 and re.search(r"\\(?:sum|prod|int)\b", s) and not re.search(r"[=A-Za-z0-9\\\\]", s.replace("\\sum", "").replace("\\prod", "").replace("\\int", "")):
        return None

    return s.strip() or None


def _math_text_looks_garbled(s: str) -> bool:
    """
    Decide whether extracted math text is too garbled to be shown as LaTeX/Unicode math.
    If so, we'll prefer rendering the equation region as an image (strictly faithful to the PDF).
    """
    t = _normalize_text(s or "").strip()
    if not t:
        return True
    # Common junk glyphs from broken PDF text extraction
    junk = ["©", "®", "¬", "", "", "�", "︁", "︀", "Ö"]
    if any(j in t for j in junk):
        return True
    # Overly fragmented operator-only lines (sums/products without bodies)
    if len(t) <= 40 and re.search(r"[∑∏∫]", t) and not re.search(r"[=A-Za-z0-9]", t):
        return True
    # Many lines but very low alnum density => usually broken layout
    alnum = sum(1 for ch in t if ch.isalnum())
    if len(t) > 120 and alnum / max(1, len(t)) < 0.18:
        return True
    return False


class PdfToMarkdown:
    def __init__(self, cfg: ConvertConfig):
        self.cfg = cfg
        self._client = None
        self._temp_dir: Optional[Path] = None
        self._repairs_dir: Optional[Path] = None
        if cfg.llm:
            if OpenAI is None:
                raise RuntimeError("`openai` package is not available, but LLM is configured.")
            self._client = OpenAI(api_key=cfg.llm.api_key, base_url=cfg.llm.base_url)

    def _call_llm_convert(self, tagged_text: str, page_number: int) -> str:
        if not self.cfg.llm or not self._client:
            return self._fallback_convert_tags(tagged_text)

        llm = self.cfg.llm
        base_instructions = f"""
You are a Data Conversion Engine. Convert the tagged text into Markdown.

MANDATORY RULES:
1) TAGS:
   - Convert `[H1] Text` -> `# Text`
   - Convert `[H2] Text` -> `## Text`
   - Convert `[H3] Text` -> `### Text`
  - Convert `[IMAGE: filename]` -> `![Figure](./assets/filename)`
   - Convert `[CODE] ... [/CODE]` into a fenced code block using triple backticks.
   - Convert `[TABLE] ... [/TABLE]` into a Markdown table (use `|` pipes and `---` separator).
   - Convert `[MATH] ... [/MATH]` into display math: `$$ ... $$` (keep LaTeX if present).
2) VERBATIM: Keep all non-tag text. Do not summarize. Do not drop paragraphs.
3) NO INVENTION: Do not invent headings. If a line doesn't have an [H] tag, it's not a heading.
4) MATH: Repair broken math into LaTeX `$$ ... $$`. Add blank lines around block math.
   - If you see a line like `(EQNO 8)` inside a [MATH] block, that is the equation number. Convert it to `\\tag{{8}}` and REMOVE the `(EQNO ...)` line.
   - `\\tag{{n}}` MUST be top-level in the display math (e.g., after `\\end{{aligned}}`), never inside `aligned`.
   - Do NOT put `\\tag{{}}` outside the `$$ ... $$` math block.
   - If multiple consecutive [MATH] blocks are fragments of the same equation, MERGE them into ONE complete equation and output it once.
   - Do NOT output empty math blocks.
   - Do NOT output fragment-only sums/products/limits (e.g., a lone \sum with limits and no summand).
   - If the same equation appears twice, output it once.
   - Inline math must use $...$ ONLY. Display math must use $$...$$ ONLY.
   - Do not place raw LaTeX outside math fences.
5) TABLES: Convert text grids into Markdown tables. Do not use images for tables.
 6) FIGURES:
    - For EVERY `[IMAGE: filename]`, you MUST output `![Figure](./assets/filename)` exactly once.
    - Keep figure captions (lines starting with `Fig.` / `Figure`) directly under the image. Bold the `Fig. X.` prefix.
7) REFERENCES: Under `# REFERENCES`, format references as a numbered list (one entry per item). Preserve content; only add line breaks/list markers.
8) BULLETS: Replace `•` bullets with Markdown list items (`- `).
9) OUTPUT: Markdown only. No commentary.

PAGE: {page_number}
""".strip()

        out_parts: list[str] = []
        chunks = _chunk_by_blank_lines(tagged_text, max_chars=12_000)
        for chunk_i, chunk in enumerate(chunks, start=1):
            if llm.request_sleep_s > 0:
                time.sleep(llm.request_sleep_s)
            chunk_header = "" if len(chunks) == 1 else f"\n\nCHUNK {chunk_i}/{len(chunks)}"
            prompt = f"""{base_instructions}{chunk_header}

TEXT:
---
{chunk}
---
""".strip()
            resp = self._client.chat.completions.create(
                model=llm.model,
                messages=[
                    {"role": "system", "content": "You are a strict robotic text-to-markdown converter. Follow tags verbatim."},
                    {"role": "user", "content": prompt},
                ],
                temperature=llm.temperature,
                max_tokens=llm.max_tokens,
            )
            out_parts.append(resp.choices[0].message.content or "")
        return "\n\n".join(p.strip() for p in out_parts if p.strip()).strip()

    def _repair_cache_path(self, *, kind: str, page_number: int, block_index: int, raw: str) -> Optional[Path]:
        if not self._repairs_dir:
            return None
        h = _hash_text(raw)
        return self._repairs_dir / f"p{page_number:03d}.b{block_index:03d}.{kind}.{h}.json"

    def _load_cached_repair(self, path: Optional[Path]) -> Optional[str]:
        if not path or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            out = payload.get("output")
            if isinstance(out, str) and out.strip():
                return out.strip()
        except Exception:
            return None
        return None

    def _save_cached_repair(self, path: Optional[Path], *, kind: str, raw: str, output: str) -> None:
        if not path:
            return
        try:
            path.write_text(
                json.dumps(
                    {"kind": kind, "input_hash": _hash_text(raw), "output": output},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            return

    def _call_llm_repair_table(self, raw: str, *, page_number: int, block_index: int) -> Optional[str]:
        if not self.cfg.llm or not self._client or not self.cfg.llm_repair:
            return None
        cache = self._repair_cache_path(kind="table", page_number=page_number, block_index=block_index, raw=raw)
        cached = self._load_cached_repair(cache)
        if cached:
            return cached

        prompt = (
            "Convert the following extracted table text into a Markdown table.\n"
            "Rules:\n"
            "- Output ONLY the Markdown table (pipes + header separator row).\n"
            "- Preserve numbers/units/arrows exactly.\n"
            "- Do NOT wrap in code fences.\n"
            "- If the input is actually not a table, output an empty string.\n\n"
            + raw.strip()
        )
        llm = self.cfg.llm
        if llm.request_sleep_s > 0:
            time.sleep(llm.request_sleep_s)
        resp = self._client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": "You convert tables to Markdown. Output only the table."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=min(llm.max_tokens, 4096),
        )
        out = (resp.choices[0].message.content or "").strip()
        # Minimal validation: must have pipes and a separator row.
        if "|" not in out or not re.search(r"(?m)^\|\s*---", out):
            return None
        self._save_cached_repair(cache, kind="table", raw=raw, output=out)
        return out

    def _call_llm_repair_math(
        self,
        raw: str,
        *,
        page_number: int,
        block_index: int,
        context_before: str = "",
        context_after: str = "",
        eq_number: Optional[str] = None,
    ) -> Optional[str]:
        if not self.cfg.llm or not self._client or not self.cfg.llm_repair:
            return None
        cache = self._repair_cache_path(kind="math", page_number=page_number, block_index=block_index, raw=raw)
        cached = self._load_cached_repair(cache)
        if cached:
            return cached

        ctx = ""
        cb = context_before.strip()
        ca = context_after.strip()
        if cb:
            ctx += f"Context before:\n{cb}\n\n"
        if ca:
            ctx += f"Context after:\n{ca}\n\n"

        eq_hint = f"Equation number (for your reference only): ({eq_number})\n\n" if eq_number else ""
        prompt = (
            "Convert the following extracted equation(s) into valid LaTeX.\n"
            "Rules:\n"
            "- Output ONLY LaTeX (no Markdown, no $$).\n"
            "- Ensure braces are balanced and every \\begin has a matching \\end.\n"
            "- Prefer \\begin{aligned}...\\end{aligned} for multi-line.\n"
            "- The input may contain fragmented or duplicated pieces of ONE equation. Merge fragments into a complete equation and remove duplicates.\n"
            "- DO NOT output equation numbers like (3) or any \\tag{...}. (We will attach numbering outside the environment.)\n"
            "- Use context to infer missing symbols if needed.\n"
            "- If you are not confident, output an empty string.\n\n"
            + eq_hint
            + ctx
            + "Extracted math:\n"
            + raw.strip()
        )
        llm = self.cfg.llm
        if llm.request_sleep_s > 0:
            time.sleep(llm.request_sleep_s)
        resp = self._client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": "You convert extracted math into valid LaTeX. Output LaTeX only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=min(llm.max_tokens, 2048),
        )
        out = (resp.choices[0].message.content or "").strip()
        if not out:
            return None
        if not _is_balanced_latex(out):
            return None
        self._save_cached_repair(cache, kind="math", raw=raw, output=out)
        return out

    def _call_llm_polish_code(self, raw: str, *, page_number: int, block_index: int) -> Optional[str]:
        if not self.cfg.llm or not self._client or not self.cfg.llm_repair:
            return None
        cache = self._repair_cache_path(kind="code", page_number=page_number, block_index=block_index, raw=raw)
        cached = self._load_cached_repair(cache)
        if cached:
            return cached

        prompt = (
            "Clean up the following pseudocode so it matches the original PDF formatting as closely as possible.\n"
            "Rules:\n"
            "- Output ONLY the code body (no Markdown fences).\n"
            "- Preserve line breaks and indentation as much as possible.\n"
            "- Preserve symbols (←, →, ⇒, ⊲, Σ, μ, ∇, ∥, etc.). Do NOT translate them.\n"
            "- Only fix obvious mojibake/ligatures and spacing; do not rewrite or paraphrase.\n\n"
            + raw.rstrip()
        )
        llm = self.cfg.llm
        if llm.request_sleep_s > 0:
            time.sleep(llm.request_sleep_s)
        resp = self._client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": "You clean up pseudocode. Output code only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=min(llm.max_tokens, 2048),
        )
        out = (resp.choices[0].message.content or "").strip("\n")
        if not out.strip():
            return None
        self._save_cached_repair(cache, kind="code", raw=raw, output=out)
        return out

    def _call_llm_repair_body_paragraph(self, raw: str, *, page_number: int, block_index: int) -> Optional[str]:
        if not self.cfg.llm or not self._client or not (self.cfg.llm_repair and self.cfg.llm_repair_body_math):
            return None
        t = raw.strip()
        if len(t) < 20 or len(t) > 1200:
            return None
        # include Greek letters (often appear as unicode in extracted text)
        if not re.search(r"[=^_\\]|[\u2200-\u22ff]|[∑Σ∏√∫∞≤≥≠≈∥]|[\u0370-\u03ff]", t):
            return None
        cache = self._repair_cache_path(kind="bodymath", page_number=page_number, block_index=block_index, raw=t)
        cached = self._load_cached_repair(cache)
        if cached:
            return cached

        prompt = (
            "Fix inline math in this paragraph from a research paper.\n"
            "Rules:\n"
            "- Output a SINGLE Markdown paragraph (no headings, no bullet lists, no code fences).\n"
            "- Keep all normal words and citations unchanged.\n"
            "- Convert mangled math into valid LaTeX using $...$ for inline math ONLY.\n"
            "- Do NOT use \\( ... \\) and do NOT use $$ ... $$.\n"
            "- If you are not confident, output the original paragraph unchanged.\n\n"
            + t
        )
        llm = self.cfg.llm
        if llm.request_sleep_s > 0:
            time.sleep(llm.request_sleep_s)
        resp = self._client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": "You fix inline math. Output one Markdown paragraph only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=min(llm.max_tokens, 2048),
        )
        out = (resp.choices[0].message.content or "").strip()
        # safety: no headings/bullets/fences
        if re.search(r"(?m)^\s*(#|```|\- |\* )", out):
            return None
        if not out:
            return None
        self._save_cached_repair(cache, kind="bodymath", raw=t, output=out)
        return out

    def _call_llm_split_references(self, raw: str, *, paper_name: str = "") -> Optional[str]:
        if not self.cfg.llm or not self._client or not self.cfg.llm_repair:
            return None
        t = raw.strip()
        if len(t) < 200:
            return None
        cache = self._repair_cache_path(kind="refs", page_number=999, block_index=0, raw=(paper_name + "\n" + t))
        cached = self._load_cached_repair(cache)
        if cached:
            return cached

        prompt = (
            "Split the following REFERENCES section text into a numbered Markdown list.\n"
            "Rules:\n"
            "- Output ONLY lines like \"1. ...\", \"2. ...\".\n"
            "- Preserve author names, venues, years, and DOIs/URLs.\n"
            "- Keep the original order.\n"
            "- Do not invent missing references.\n\n"
            + t
        )
        llm = self.cfg.llm
        if llm.request_sleep_s > 0:
            time.sleep(llm.request_sleep_s)
        resp = self._client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": "You split references into a numbered Markdown list."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=min(llm.max_tokens, 4096),
        )
        out = (resp.choices[0].message.content or "").strip()
        if not out:
            return None
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if sum(1 for ln in lines if re.match(r"^\d+\.\s+\S", ln)) < 5:
            return None
        self._save_cached_repair(cache, kind="refs", raw=(paper_name + "\n" + t), output="\n".join(lines))
        return "\n".join(lines)

    def _repair_references_with_llm(self, md: str, *, paper_name: str = "") -> str:
        if not self.cfg.llm or not self._client or not self.cfg.llm_repair:
            return md
        lines = md.splitlines()
        # find references heading
        start = None
        for i, ln in enumerate(lines):
            if re.match(r"^#\s+REFERENCES\s*$", ln.strip(), flags=re.IGNORECASE):
                start = i + 1
                break
        if start is None:
            return md
        end = len(lines)
        for j in range(start, len(lines)):
            if re.match(r"^#\s+\S", lines[j]) and not re.match(r"^#\s+REFERENCES\s*$", lines[j].strip(), flags=re.IGNORECASE):
                end = j
                break
        ref_block = "\n".join(lines[start:end]).strip()
        if not ref_block:
            return md
        # If it's already a numbered list with enough items, keep.
        if sum(1 for ln in ref_block.splitlines() if re.match(r"^\s*\d+\.\s+\S", ln)) >= 10:
            return md
        # Otherwise ask LLM to split.
        split = self._call_llm_split_references(ref_block, paper_name=paper_name)
        if not split:
            return md
        new_lines = lines[: start] + [""] + split.splitlines() + [""] + lines[end:]
        return "\n".join(new_lines)

    def _render_blocks_to_markdown(
        self,
        blocks: list[TextBlock],
        figs_by_block: dict[int, str],
        *,
        page_number: int,
        eqno_by_block: Optional[dict[int, str]] = None,
        eqimg_by_block: Optional[dict[int, str]] = None,
    ) -> str:
        out: list[str] = []
        eqno_by_block = eqno_by_block or {}
        eqimg_by_block = eqimg_by_block or {}

        def caption_to_md(txt: str) -> str:
            t = txt.strip()
            m = re.match(r"^(Fig\.|Figure)\s*([0-9]+)\.?\s*(.*)$", t, flags=re.IGNORECASE)
            if not m:
                return t
            prefix = f"{m.group(1)} {m.group(2)}."
            rest = m.group(3).strip()
            if rest:
                return f"**{prefix}** {rest}"
            return f"**{prefix}**"

        for bi, b in enumerate(blocks):
            img = figs_by_block.get(bi)
            if img:
                out.append(f"![Figure](./assets/{img})")
                # If this block is a caption, render it right after the image.
                if re.match(r"^\s*(?:Fig\.|FIG\.|Figure|FIGURE)\s*[0-9]+", b.text.strip(), re.IGNORECASE):
                    out.append(caption_to_md(b.text))
                    out.append("")
                    continue
                out.append("")

            if b.heading_level and 1 <= int(b.heading_level) <= 3 and (
                _is_numbered_heading_text(b.text) or _is_appendix_heading_text(b.text)
            ):
                lvl = int(b.heading_level)
                out.append("#" * lvl + " " + b.text.strip())
                out.append("")
                continue

            if b.is_table:
                table_md = table_text_to_markdown(b.text)
                if not table_md:
                    # Some publishers encode tables in math/array form or flatten columns badly.
                    table_md = self._call_llm_repair_table(b.text, page_number=page_number, block_index=bi)
                if table_md:
                    out.extend(table_md.splitlines())
                else:
                    # Last resort: keep plain text (NOT a code fence) to avoid "table as code" pollution.
                    out.extend([ln.rstrip() for ln in b.text.splitlines() if ln.strip()])
                out.append("")
                continue

            if b.is_math:
                # If extracted math is clearly garbled (common in --no-llm mode), render it as an image instead of
                # emitting broken LaTeX/Unicode fragments.
                eq_img = eqimg_by_block.get(bi)
                if eq_img:
                    out.append(f"![Equation](./assets/{eq_img})")
                    out.append("")
                    continue
                # Some PDFs encode tables as LaTeX arrays inside math blocks.
                if re.search(r"\\begin\{(?:array|tabular)\}", b.text):
                    table_md = _latex_array_to_markdown_table(b.text)
                    if not table_md:
                        table_md = self._call_llm_repair_table(b.text, page_number=page_number, block_index=bi)
                    if table_md:
                        out.extend(table_md.splitlines())
                        out.append("")
                        continue

                raw_math = "\n".join([ln.rstrip() for ln in b.text.splitlines() if ln.strip()])
                eqno = eqno_by_block.get(bi)
                latex: Optional[str] = None
                # Add nearby context to help reconstruction when extraction is fragmented.
                ctx_before = ""
                ctx_after = ""
                for pj in range(bi - 1, -1, -1):
                    pb = blocks[pj]
                    if pb.is_table or pb.is_math or pb.is_code:
                        continue
                    t = pb.text.strip()
                    if len(t) >= 20:
                        ctx_before = t[-400:]
                        break
                for nj in range(bi + 1, len(blocks)):
                    nb = blocks[nj]
                    if nb.is_table or nb.is_math or nb.is_code:
                        continue
                    t = nb.text.strip()
                    if len(t) >= 20:
                        ctx_after = t[:400]
                        break
                if self.cfg.llm and self.cfg.llm_repair:
                    latex = self._call_llm_repair_math(
                        raw_math or b.text,
                        page_number=page_number,
                        block_index=bi,
                        context_before=ctx_before,
                        context_after=ctx_after,
                        eq_number=eqno,
                    )
                if not latex and raw_math and _is_balanced_latex(raw_math) and re.search(r"\\[A-Za-z]+", raw_math):
                    latex = raw_math
                latex_norm = _normalize_display_latex(latex if latex is not None else raw_math, eq_number=eqno)
                if latex_norm:
                    out.append("$$")
                    out.append(latex_norm)
                    out.append("$$")
                else:
                    # Avoid emitting broken LaTeX that crashes renderers.
                    out.extend([ln.rstrip() for ln in b.text.splitlines() if ln.strip()])
                out.append("")
                continue

            if b.is_code:
                code_text = b.text
                if re.search(r"[⊲←→⇒∥∇μΣ]", code_text) or "鈫?" in code_text or "�" in code_text or re.search(r"(?mi)^\s*(while|for|if|function)\b", code_text):
                    polished = self._call_llm_polish_code(code_text, page_number=page_number, block_index=bi)
                    if polished:
                        code_text = polished
                out.append("```")
                out.extend([ln.rstrip() for ln in code_text.splitlines()])
                out.append("```")
                out.append("")
                continue

            para = b.text.strip()
            repaired = self._call_llm_repair_body_paragraph(para, page_number=page_number, block_index=bi)
            out.append(repaired if repaired else para)
            out.append("")

        return "\n".join(out).strip()

    def _extract_json_array(self, s: str) -> Optional[list]:
        if not s:
            return None
        start = s.find("[")
        end = s.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        blob = s[start : end + 1]
        try:
            return json.loads(blob)
        except Exception:
            return None

    def _call_llm_classify_blocks(self, blocks: list[TextBlock], page_number: int, page) -> Optional[list[dict]]:
        if not self.cfg.llm or not self._client or not self.cfg.llm_classify:
            return None

        W, H = float(page.rect.width), float(page.rect.height)
        llm = self.cfg.llm

        def pack(bs: list[TextBlock], offset: int) -> list[dict]:
            out = []
            for i, b in enumerate(bs):
                txt = b.text.strip()
                if len(txt) > 800:
                    txt = txt[:800] + "…"
                out.append(
                    {
                        "i": offset + i,
                        "text": txt,
                        "font": round(float(b.max_font_size), 2),
                        "bold": bool(b.is_bold),
                        "bbox": [round(float(x), 2) for x in b.bbox],
                        "page": page_number,
                        "page_wh": [round(W, 2), round(H, 2)],
                    }
                )
            return out

        system = "You are a strict PDF block classifier. Output JSON only."

        def make_prompt(items: list[dict]) -> str:
            return f"""
Classify each block from a research paper PDF page.

Return a JSON array with EXACTLY the same number of items as the input.
Each output item MUST be an object with keys:
- i: integer (copy input i)
- action: \"keep\" or \"drop\"
- kind: \"heading\" | \"body\" | \"table\" | \"math\" | \"code\" | \"caption\"
- heading_level: 1 | 2 | 3 | null  (only for kind=heading)
- text: string (cleaned text; keep meaning; fix mojibake/ligatures; keep spacing for tables)

STRICT RULES:
1) Headings: kind=\"heading\" ONLY if text matches a real paper section heading:
   - Numbered: ^\\d+(\\.\\d+)*\\s+<LETTER>  (examples: \"1 INTRODUCTION\", \"5.2 Adaptive Control\").
     NOT a heading if the title starts with \"(\" or looks like an equation (contains \"=\", \"^\", \"_\", \"\\\\\", Σ, ∑, etc.).
   - Appendix letters: ^[A-Z](?:\\.\\d+)*\\s+<LETTER> (examples: \"A DETAILS...\", \"B.1 ...\").
     NOT a heading if it looks like an equation.
   - The literal word \"APPENDIX\" (as a standalone heading).
2) Drop boilerplate/noise: journal headers/footers, page numbers, navigation like \"Latest updates\", \"RESEARCH-ARTICLE\",
   download/citation stats, copyright blocks, publisher notices.
3) Table vs code vs math:
   - table: rows/columns or matrices of numbers; DO NOT label tables as code.
   - math: mostly equations/symbols; output plain text lines (we will wrap with $$ later).
   - code: ONLY pseudocode/algorithms (keywords like while/for/if/function, arrows like ←, indentation).
4) Captions: if a block starts with \"Fig.\" or \"Figure\" and a number, set kind=\"caption\" and keep that prefix exactly.
5) Never invent content. Do not merge blocks. Do not reorder. Do not add items.

INPUT JSON:
{json.dumps(items, ensure_ascii=False)}
""".strip()

        # chunk by count to reduce risk of truncation
        batch_size = max(10, int(self.cfg.classify_batch_size))
        all_results: list[dict] = []
        for start in range(0, len(blocks), batch_size):
            sub = blocks[start : start + batch_size]
            items = pack(sub, offset=start)
            if llm.request_sleep_s > 0:
                time.sleep(llm.request_sleep_s)
            resp = self._client.chat.completions.create(
                model=llm.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": make_prompt(items)},
                ],
                temperature=0.0,
                max_tokens=llm.max_tokens,
            )
            content = resp.choices[0].message.content or ""
            arr = self._extract_json_array(content)
            if not isinstance(arr, list) or len(arr) != len(items):
                return None
            all_results.extend(arr)
        return all_results

    def _call_llm_translate_zh(self, md: str) -> str:
        if not self.cfg.llm or not self._client:
            return md
        llm = self.cfg.llm
        prompt = (
            "Translate to Chinese. Keep ALL Markdown structure (#, $$, images, code fences) exactly. "
            "Do not translate author names, venues, citations, or LaTeX.\n\n"
            + md
        )
        if llm.request_sleep_s > 0:
            time.sleep(llm.request_sleep_s)
        resp = self._client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "system", "content": "Translator mode."}, {"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or md).strip()

    def _fallback_convert_tags(self, tagged_text: str) -> str:
        out: list[str] = []
        in_code = False
        in_table = False
        in_math = False
        table_lines: list[str] = []
        math_lines: list[str] = []

        def flush_table():
            nonlocal table_lines
            if not table_lines:
                return
            rows: list[list[str]] = []
            for ln in table_lines:
                cols = [c.strip() for c in re.split(r"\t+|\s{2,}", ln.strip()) if c.strip()]
                if cols:
                    rows.append(cols)
            if not rows:
                out.extend(table_lines)
                table_lines = []
                return
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            header = rows[0]
            out.append("| " + " | ".join(header) + " |")
            out.append("| " + " | ".join(["---"] * width) + " |")
            for r in rows[1:]:
                out.append("| " + " | ".join(r) + " |")
            table_lines = []

        def flush_math():
            nonlocal math_lines
            if not math_lines:
                return
            out.append("$$")
            out.extend([ln.rstrip() for ln in math_lines if ln.strip()])
            out.append("$$")
            math_lines = []

        for line in tagged_text.splitlines():
            raw = line.rstrip("\n\r")
            if raw.strip() == "[TABLE]":
                if in_math:
                    flush_math()
                    in_math = False
                in_table = True
                table_lines = []
                continue
            if raw.strip() == "[/TABLE]":
                if in_table:
                    flush_table()
                    in_table = False
                continue
            if raw.strip() == "[MATH]":
                if in_table:
                    flush_table()
                    in_table = False
                in_math = True
                math_lines = []
                continue
            if raw.strip() == "[/MATH]":
                if in_math:
                    flush_math()
                    in_math = False
                continue
            if raw.strip() == "[CODE]":
                if not in_code:
                    out.append("```")
                    in_code = True
                continue
            if raw.strip() == "[/CODE]":
                if in_code:
                    out.append("```")
                    in_code = False
                continue
            if in_table:
                if raw.strip():
                    table_lines.append(raw)
                continue
            if in_math:
                if raw.strip():
                    math_lines.append(raw)
                continue
            if in_code:
                out.append(raw.rstrip())
                continue

            line = raw.rstrip()
            if line.startswith("[H1] "):
                h = _strip_trailing_page_no_from_heading_text(line[len("[H1] ") :])
                out.append("# " + h)
            elif line.startswith("[H2] "):
                h = _strip_trailing_page_no_from_heading_text(line[len("[H2] ") :])
                out.append("## " + h)
            elif line.startswith("[H3] "):
                h = _strip_trailing_page_no_from_heading_text(line[len("[H3] ") :])
                out.append("### " + h)
            else:
                m = re.match(r"^\[IMAGE:\s*([^\]]+)\]\s*$", line)
                if m:
                    # Use "./assets/..." for better compatibility across Markdown renderers.
                    out.append(f"![Figure](./assets/{m.group(1).strip()})")
                else:
                    # Fix TOC runs that got collapsed into a single long line.
                    parts = _split_toc_run_line(line)
                    if len(parts) > 1:
                        out.extend(parts)
                    else:
                        out.append(line)
        return "\n".join(out).strip()

    def convert(self) -> None:
        if fitz is None:
            raise SystemExit("Missing dependency `PyMuPDF` (import name: `fitz`). Install it, then retry.")
        pdf_path = self.cfg.pdf_path
        paper_name = pdf_path.stem
        save_dir = self.cfg.out_dir / paper_name
        assets_dir = save_dir / "assets"
        temp_dir = save_dir / "temp"
        assets_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.keep_debug or self.cfg.skip_existing or (self.cfg.llm and (self.cfg.llm_classify or self.cfg.llm_repair)):
            temp_dir.mkdir(parents=True, exist_ok=True)

        with fitz.open(pdf_path) as doc:
            body_size = detect_body_font_size(doc)
            total_pages = len(doc)
            noise_texts = build_repeated_noise_texts(doc) if self.cfg.global_noise_scan else set()
            start = max(0, int(self.cfg.start_page))
            end = min(total_pages, int(self.cfg.end_page) if self.cfg.end_page >= 0 else total_pages)

            print(f"Detected body font size: {body_size} | pages: {total_pages} | range: {start+1}-{end}")

            en_pages: list[str] = []
            zh_pages: list[str] = []
            in_references = False
            warned_garbled_math = False

            for page_index in range(start, end):
                pnum = page_index + 1
                print(f"Processing page {pnum}/{total_pages} ...")

                raw_out = temp_dir / f"p{pnum:03d}.tagged.txt"
                en_out = temp_dir / f"p{pnum:03d}.en.md"
                zh_out = temp_dir / f"p{pnum:03d}.zh.md"
                cls_out = temp_dir / f"p{pnum:03d}.cls.json"

                if self.cfg.skip_existing and en_out.exists() and (not self.cfg.translate_zh or zh_out.exists()):
                    en_md0 = en_out.read_text(encoding="utf-8", errors="replace")
                    if "<!-- kb_page:" not in en_md0[:120]:
                        en_md0 = f"<!-- kb_page: {pnum} -->\n\n" + en_md0.lstrip()
                    en_pages.append(en_md0)
                    if self.cfg.translate_zh:
                        zh_md0 = zh_out.read_text(encoding="utf-8", errors="replace")
                        if "<!-- kb_page:" not in zh_md0[:120]:
                            zh_md0 = f"<!-- kb_page: {pnum} -->\n\n" + zh_md0.lstrip()
                        zh_pages.append(zh_md0)
                    continue

                page = doc[page_index]
                # References often use smaller fonts and appear in two columns near margins.
                # If we're in REFERENCES (or this page introduces it), relax the small-text margin filter.
                if _page_has_references_heading(page) or _page_looks_like_references_content(page):
                    in_references = True
                blocks = extract_text_blocks(
                    page,
                    body_size=body_size,
                    noise_texts=noise_texts,
                    page_index=page_index,
                    relax_small_text_filter=bool(in_references),
                    preserve_body_linebreaks=bool(in_references),
                )
                blocks = sort_blocks_reading_order(blocks, page_width=float(page.rect.width))

                # Optional LLM-based block classification (noise/table/math/code/heading).
                need_classify = bool(self.cfg.llm and self.cfg.llm_classify)
                if need_classify and self.cfg.llm_classify_only_if_needed:
                    if page_index > 2 and not any((b.is_table or b.is_math or b.is_code) for b in blocks):
                        need_classify = False

                llm_cls = None
                if need_classify:
                    if cls_out.exists():
                        try:
                            payload = json.loads(cls_out.read_text(encoding="utf-8", errors="replace"))
                            if isinstance(payload, list):
                                llm_cls = payload
                            elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
                                llm_cls = payload["items"]
                        except Exception:
                            llm_cls = None
                    if llm_cls is None:
                        llm_cls = self._call_llm_classify_blocks(blocks, pnum, page)
                        if llm_cls:
                            try:
                                cls_out.write_text(json.dumps({"page": pnum, "items": llm_cls}, ensure_ascii=False), encoding="utf-8")
                            except Exception:
                                pass
                if llm_cls:
                    # Map by index, don't trust ordering.
                    by_i: dict[int, dict] = {}
                    for r in llm_cls:
                        try:
                            by_i[int(r.get("i"))] = r
                        except Exception:
                            continue
                    updated: list[TextBlock] = []
                    caption_re = re.compile(r"^\s*(?:Fig\.|FIG\.|Figure|FIGURE)\s*([0-9]+)", re.IGNORECASE)
                    for i, b in enumerate(blocks):
                        r = by_i.get(i)
                        if not isinstance(r, dict):
                            updated.append(b)
                            continue
                        try:
                            action = str(r.get("action", "keep"))
                            kind = str(r.get("kind", "body"))
                            text = str(r.get("text", b.text)).strip() or b.text
                            heading_level = r.get("heading_level", None)
                        except Exception:
                            updated.append(b)
                            continue

                        # Never drop figure captions (used for figure cropping/placement).
                        if action == "drop" and caption_re.match(b.text.strip()):
                            action = "keep"
                            kind = "caption"

                        if action == "drop":
                            continue

                        is_table = kind == "table"
                        is_math = kind == "math"
                        is_code = kind == "code"
                        hl: Optional[int] = None
                        if kind == "heading":
                            try:
                                hl = int(heading_level) if heading_level is not None else None
                            except Exception:
                                hl = None
                            if not (_is_numbered_heading_text(text) or _is_appendix_heading_text(text)):
                                hl = None
                        updated.append(
                            TextBlock(
                                bbox=b.bbox,
                                text=text,
                                max_font_size=b.max_font_size,
                                is_bold=b.is_bold,
                                insert_image=b.insert_image,
                                is_code=is_code,
                                is_table=is_table,
                                is_math=is_math,
                                heading_level=hl,
                            )
                        )
                    blocks = updated

                # Heuristic heading pass (for when classifier is off or conservative).
                headed: list[TextBlock] = []
                for b in blocks:
                    if b.heading_level is None:
                        tag = detect_header_tag(
                            page_index=page_index,
                            text=b.text,
                            max_size=b.max_font_size,
                            is_bold=b.is_bold,
                            body_size=body_size,
                            page_width=float(page.rect.width),
                            bbox=b.bbox,
                        )
                        if tag:
                            try:
                                lvl = int(tag.strip("[]H"))
                            except Exception:
                                lvl = None
                            headed.append(
                                TextBlock(
                                    bbox=b.bbox,
                                    text=b.text,
                                    max_font_size=b.max_font_size,
                                    is_bold=b.is_bold,
                                    insert_image=b.insert_image,
                                    is_code=b.is_code,
                                    is_table=b.is_table,
                                    is_math=b.is_math,
                                    heading_level=lvl,
                                )
                            )
                            continue
                    headed.append(b)
                blocks = headed
                # Merge fragmented display-math blocks before rendering (LLM will fix math inside).
                blocks = merge_math_blocks_by_proximity(blocks, body_size=float(body_size), page_width=float(page.rect.width))
                blocks = merge_adjacent_math_blocks(
                    blocks,
                    max_vgap=max(12.0, float(body_size) * 1.6),
                    body_size=float(body_size),
                    min_x_overlap=0.15,
                )
                # Re-sort after merges (bbox changed).
                blocks = sort_blocks_reading_order(blocks, page_width=float(page.rect.width))

                # If LLM is disabled and equation-image fallback is off, warn early that math may be garbled.
                if (self.cfg.llm is None) and (not self.cfg.eq_image_fallback) and (not warned_garbled_math):
                    try:
                        if any(b.is_math and _math_text_looks_garbled(b.text) for b in blocks):
                            print(
                                "WARNING: Detected garbled math in PDF extraction. "
                                "To get correct LaTeX, enable LLM (DeepSeek) or pass --eq-image-fallback as a last resort."
                            )
                            warned_garbled_math = True
                    except Exception:
                        pass

                # Attach equation numbers like "(8)" to math blocks and remove standalone number blocks.
                blocks, eqno_by_block = attach_equation_numbers(
                    blocks,
                    body_size=float(body_size),
                    page_width=float(page.rect.width),
                )
                blocks, eqno_by_block = extract_eqno_from_math_text(blocks, eqno_by_block)
                blocks = drop_spurious_math_fragments(blocks)

                figs, covered = extract_figures_by_captions(
                    page,
                    blocks,
                    assets_dir,
                    page_index,
                    image_scale=self.cfg.image_scale,
                    image_alpha=self.cfg.image_alpha,
                )
                # Fallback: capture uncaptioned images to avoid missing figures (max-recall mode).
                extra_imgs = extract_images_fallback(
                    page,
                    blocks,
                    assets_dir,
                    page_index,
                    covered_rects=covered,
                    image_scale=self.cfg.image_scale,
                    image_alpha=self.cfg.image_alpha,
                )
                for k, v in extra_imgs.items():
                    figs.setdefault(k, v)

                # Optional fallback: render obviously-garbled display math as equation images.
                eq_imgs: dict[int, str] = {}
                if self.cfg.eq_image_fallback and not (
                    self.cfg.llm and (self.cfg.llm_repair or self.cfg.llm_render_page)
                ):
                    eq_imgs = extract_equation_images_for_garbled_math(
                        page,
                        blocks,
                        assets_dir,
                        page_index,
                        eqno_by_block=eqno_by_block,
                        image_scale=self.cfg.image_scale,
                        image_alpha=self.cfg.image_alpha,
                    )

                # Ensure repair cache directory exists (when debug/classify is on we already created temp_dir).
                self._temp_dir = temp_dir
                self._repairs_dir = temp_dir / "repairs"
                if self.cfg.llm and self.cfg.llm_repair:
                    self._repairs_dir.mkdir(parents=True, exist_ok=True)

                tagged = ""
                if self.cfg.keep_debug or self.cfg.llm_render_page:
                    tagged = _tagged_page_text(
                        page_index=page_index,
                        page=page,
                        blocks=blocks,
                        body_size=body_size,
                        figures_by_block=figs,
                        eqno_by_block=eqno_by_block,
                    )

                if self.cfg.keep_debug:
                    raw_out.write_text(tagged, encoding="utf-8")

                if self.cfg.llm_render_page:
                    en_md = self._call_llm_convert(tagged, pnum)
                else:
                    en_md = self._render_blocks_to_markdown(
                        blocks,
                        figs,
                        page_number=pnum,
                        eqno_by_block=eqno_by_block,
                        eqimg_by_block=eq_imgs,
                    )

                # Safety: ensure extracted images aren't silently dropped.
                for img_name in figs.values():
                    if f"./assets/{img_name}" not in en_md and f"assets/{img_name}" not in en_md:
                        en_md = en_md.rstrip() + f"\n\n![Figure](./assets/{img_name})\n"
                for img_name in eq_imgs.values():
                    if f"./assets/{img_name}" not in en_md and f"assets/{img_name}" not in en_md:
                        en_md = en_md.rstrip() + f"\n\n![Equation](./assets/{img_name})\n"

                if self.cfg.keep_debug:
                    en_out.write_text(en_md, encoding="utf-8")

                # Add a lightweight page marker so the KB can "定位到具体页" later.
                en_pages.append(f"<!-- kb_page: {pnum} -->\n\n" + en_md.lstrip())

                if self.cfg.translate_zh:
                    zh_md = self._call_llm_translate_zh(en_md)
                    if self.cfg.keep_debug:
                        zh_out.write_text(zh_md, encoding="utf-8")
                    zh_pages.append(f"<!-- kb_page: {pnum} -->\n\n" + zh_md.lstrip())

            en_full = postprocess_markdown("\n\n".join(en_pages))
            # Prefer PDF-layout-based REFERENCES extraction (more reliable for two-column + cross-page refs).
            pdf_refs = _extract_references_from_pdf(doc)
            if pdf_refs:
                en_full = _inject_references_section(en_full, pdf_refs)
            else:
                # Fallback: optionally refine via LLM then heuristically reformat.
                en_full = self._repair_references_with_llm(en_full, paper_name=paper_name)
                en_full = _format_references(en_full)
            (save_dir / f"{paper_name}.en.md").write_text(en_full, encoding="utf-8")
            # Write a simple manifest so missing images are obvious (and easy to bulk-import elsewhere).
            try:
                pngs = sorted(p.name for p in assets_dir.glob("*.png"))
                if pngs:
                    manifest = "\n".join([f"- ![asset](./assets/{n})" for n in pngs]) + "\n"
                    (save_dir / "assets_manifest.md").write_text(manifest, encoding="utf-8")
            except Exception:
                pass
            if self.cfg.translate_zh:
                zh_full = postprocess_markdown("\n\n".join(zh_pages))
                (save_dir / f"{paper_name}.zh.md").write_text(zh_full, encoding="utf-8")

        print(f"Done. Output: {save_dir}")


def _parse_args(argv: Optional[list[str]] = None) -> ConvertConfig:
    ap = argparse.ArgumentParser(description="Convert a research PDF into (high-fidelity) Markdown with assets.")
    ap.add_argument("--pdf", required=True, help="Input PDF path")
    ap.add_argument("--out", required=True, help="Output folder (paper-stem subfolder will be created)")
    ap.add_argument("--translate-zh", action="store_true", help="Also output Chinese Markdown")
    ap.add_argument("--start-page", type=int, default=0, help="0-based start page (default 0)")
    ap.add_argument("--end-page", type=int, default=-1, help="0-based end page (exclusive); -1 means end")
    ap.add_argument("--skip-existing", action="store_true", help="Skip pages if temp outputs already exist")
    ap.add_argument("--keep-debug", action="store_true", help="Write per-page tagged + md into temp/")

    # DeepSeek's OpenAI-compatible endpoint uses the /v1 prefix.
    ap.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    ap.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    ap.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help="Env var name holding the API key (default: DEEPSEEK_API_KEY)",
    )
    ap.add_argument("--no-llm", action="store_true", help="Disable LLM (only tag->markdown conversion)")
    ap.add_argument("--no-llm-classify", action="store_true", help="Disable LLM block classification (keep heuristics)")
    ap.add_argument("--no-llm-repair", action="store_true", help="Disable LLM repair for tables/math/pseudocode")
    ap.add_argument("--llm-repair-body-math", action="store_true", help="Also repair inline math-like body paragraphs (slower)")
    ap.add_argument(
        "--no-llm-repair-body-math",
        action="store_true",
        help="Disable inline math repair in body paragraphs (faster, lower fidelity)",
    )
    ap.add_argument("--llm-render-page", action="store_true", help="Use LLM to render the whole tagged page (best for math)")
    ap.add_argument("--no-llm-render-page", action="store_true", help="Disable page-level LLM rendering")
    ap.add_argument("--classify-batch-size", type=int, default=40, help="LLM classify blocks batch size (higher=fewer calls)")
    ap.add_argument("--classify-always", action="store_true", help="Always classify every page with LLM (slower)")
    ap.add_argument("--image-scale", type=float, default=2.0, help="Image render scale for figure crops (lower=faster)")
    ap.add_argument("--image-alpha", action="store_true", help="Render images with alpha channel (slower)")
    ap.add_argument(
        "--eq-image-fallback",
        action="store_true",
        help="Fallback: render garbled display-math as equation images (only affects --no-llm style runs)",
    )
    ap.add_argument("--no-global-noise-scan", action="store_true", help="Skip global header/footer scan (faster, less clean)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between LLM requests")
    args = ap.parse_args(argv)

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    llm: Optional[LlmConfig]
    if args.no_llm:
        llm = None
    else:
        api_key = os.environ.get(str(args.api_key_env), "").strip()
        # Users often set env vars with quotes in cmd.exe: set DEEPSEEK_API_KEY="sk-...".
        # Strip a single pair of surrounding quotes to avoid server-side auth failures.
        if (api_key.startswith('"') and api_key.endswith('"')) or (api_key.startswith("'") and api_key.endswith("'")):
            api_key = api_key[1:-1].strip()
        if not api_key:
            raise SystemExit(f"Missing API key: set env {args.api_key_env} or pass --no-llm")
        base_url = str(args.base_url).strip().rstrip("/")
        # Be forgiving: many people set DEEPSEEK_BASE_URL=https://api.deepseek.com
        # but the OpenAI-compatible API is under /v1.
        if "api.deepseek.com" in base_url and not base_url.endswith("/v1"):
            base_url = base_url + "/v1"

        llm = LlmConfig(
            api_key=api_key,
            base_url=base_url,
            model=str(args.model),
            request_sleep_s=float(args.sleep),
        )

    return ConvertConfig(
        pdf_path=pdf_path,
        out_dir=out_dir,
        translate_zh=bool(args.translate_zh),
        start_page=int(args.start_page),
        end_page=int(args.end_page),
        skip_existing=bool(args.skip_existing),
        keep_debug=bool(args.keep_debug),
        llm=llm,
        llm_classify=(not bool(args.no_llm_classify)),
        # Default to block-level rendering (more deterministic for images/captions); opt-in page-level with --llm-render-page.
        llm_render_page=bool(args.llm_render_page) and (not bool(args.no_llm_render_page)),
        llm_classify_only_if_needed=(not bool(args.classify_always)),
        classify_batch_size=int(args.classify_batch_size),
        image_scale=float(args.image_scale),
        image_alpha=bool(args.image_alpha),
        eq_image_fallback=bool(args.eq_image_fallback),
        global_noise_scan=(not bool(args.no_global_noise_scan)),
        llm_repair=(not bool(args.no_llm_repair)),
        # Prefer higher fidelity by default when LLM is enabled; can be turned off via --no-llm-repair-body-math.
        llm_repair_body_math=bool(args.llm_repair_body_math) or (llm is not None and not bool(args.no_llm_repair_body_math)),
    )


def main() -> None:
    cfg = _parse_args()
    PdfToMarkdown(cfg).convert()


if __name__ == "__main__":
    main()
