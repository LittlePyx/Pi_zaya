# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

# Lazy imports for heavy optional deps. This avoids high startup overhead in no-LLM runs.
OpenAI = None  # type: ignore[assignment]
pdfplumber = None  # type: ignore[assignment]
_PROGRESS_PRINT_LOCK = threading.Lock()
ASSET_REV_TAG = "r2"


def _progress_log(msg: str) -> None:
    with _PROGRESS_PRINT_LOCK:
        print(msg, flush=True)


def _ensure_openai_class():
    global OpenAI
    if OpenAI is not None:
        return OpenAI
    try:
        from openai import OpenAI as _OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("`openai` package is not available, but LLM is configured.") from e
    OpenAI = _OpenAI
    return OpenAI


def _ensure_pdfplumber_module():
    global pdfplumber
    if pdfplumber is not None:
        return pdfplumber
    try:
        import pdfplumber as _pdfplumber  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("`pdfplumber` package is not available.") from e
    pdfplumber = _pdfplumber
    return pdfplumber


def _as_fs_path(p: Path) -> Path:
    """
    Normalize Windows long paths for filesystem I/O.
    Keeps a normal Path on non-Windows platforms.
    """
    pp = Path(p)
    if os.name != "nt":
        return pp
    s = str(pp)
    if s.startswith("\\\\?\\"):
        return pp
    try:
        s = str(pp.resolve())
    except Exception:
        s = str(pp)
    if s.startswith("\\\\"):
        # UNC path: \\server\share -> \\?\UNC\server\share
        return Path("\\\\?\\UNC\\" + s.lstrip("\\"))
    return Path("\\\\?\\" + s)


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
    # Keep this conservative to avoid over-correction.
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


def _rect_intersection_area(a, b) -> float:
    """Intersection area without mutating either input rect."""
    if fitz is None:
        return 0.0
    try:
        inter = fitz.Rect(a) & fitz.Rect(b)
    except Exception:
        return 0.0
    return _rect_area(inter)


@dataclass(frozen=True)
class LlmConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    request_sleep_s: float = 0.0
    timeout_s: float = 45.0
    max_retries: int = 0


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
    image_scale: float = 2.2
    image_alpha: bool = False
    detect_tables: bool = True
    table_pdfplumber_fallback: bool = False
    eq_image_fallback: bool = False
    global_noise_scan: bool = True
    llm_repair: bool = True
    llm_repair_body_math: bool = False
    llm_smart_math_repair: bool = True
    llm_auto_page_render_threshold: int = 12
    llm_workers: int = 1
    workers: int = 1
    speed_mode: str = "balanced"


@dataclass(frozen=True)
class TextBlock:
    bbox: tuple[float, float, float, float]
    text: str
    max_font_size: float
    is_bold: bool
    insert_image: Optional[str] = None
    is_code: bool = False
    is_table: bool = False
    table_markdown: Optional[str] = None
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


_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*)(?:[.):]|\s)\s*(?P<rest>.+)$"
)
_APPENDIX_HEADING_RE = re.compile(
    r"^(?P<letter>[A-Z])(?P<suffix>(?:\.\d+)*)\s+(?P<rest>.+)$"
)


def _parse_numbered_heading_level(title: str) -> Optional[int]:
    t = _normalize_text(title or "").strip()
    if not t:
        return None
    m = _NUMBERED_HEADING_RE.match(t)
    if not m:
        return None
    rest = (m.group("rest") or "").strip()
    if not rest:
        return None
    if not _is_letter(rest[0]):
        return None
    if _looks_like_equation_text(rest):
        return None

    parts = (m.group("num") or "").split(".")
    if not parts:
        return None
    try:
        first_n = int(parts[0])
    except Exception:
        return None
    # Guard against year-like or DOI-like prefixes being mistaken as headings.
    if first_n <= 0 or first_n > 200:
        return None
    if len(parts) == 1 and len(parts[0]) > 2:
        return None
    if any((not p) or len(p) > 2 for p in parts[1:]):
        return None
    return len(parts)


def _parse_appendix_heading_level(title: str) -> Optional[int]:
    t = _normalize_text(title or "").strip()
    if not t:
        return None
    if t.upper() == "APPENDIX":
        return 1
    m = _APPENDIX_HEADING_RE.match(t)
    if not m:
        return None
    rest = (m.group("rest") or "").strip()
    if not rest:
        return None
    if _looks_like_equation_text(rest):
        return None
    suffix = m.group("suffix") or ""
    return 1 + int(suffix.count("."))


def _looks_like_equation_text(s: str) -> bool:
    t = _normalize_text(s)
    if not t:
        return False
    if re.search(r"[=^_\\]", t):
        return True
    if re.search(r"[\u2200-\u22ff]", t):
        return True
    if re.search(r"[\u2248\u2260\u2264\u2265\u2211\u220f]", t):
        return True
    if re.search(r"\b(?:arg\s*min|arg\s*max|exp|log|sin|cos|tan)\b", t, flags=re.IGNORECASE):
        return True
    # function-like: G(x), f(\theta), etc.
    if re.match(r"^[A-Za-z]{1,6}\s*\([^)]*\)\s*[=+\-*/^_]", t):
        return True
    return False


_COMMON_SECTION_HEADINGS = {
    "ABSTRACT",
    "INTRODUCTION",
    "RELATED WORK",
    "BACKGROUND",
    "PRELIMINARIES",
    "PRELIMINARY",
    "METHOD",
    "METHODS",
    "METHODOLOGY",
    "APPROACH",
    "MODEL",
    "MODELS",
    "EXPERIMENT",
    "EXPERIMENTS",
    "RESULT",
    "RESULTS",
    "DISCUSSION",
    "CONCLUSION",
    "CONCLUSIONS",
    "LIMITATIONS",
    "ABLATION",
    "ABLATION STUDY",
    "IMPLEMENTATION DETAILS",
    "EVALUATION",
    "DATASET",
    "DATASETS",
    "ACKNOWLEDGMENTS",
    "ACKNOWLEDGEMENTS",
    "REFERENCES",
    "APPENDIX",
}


def _strip_heading_prefix(text: str) -> str:
    t = _normalize_text(text or "").strip()
    if not t:
        return ""
    t = re.sub(r"^\d+(?:\.\d+)*(?:[.)]|闁?)\s*", "", t)
    t = re.sub(r"^[A-Z](?:\.\d+)*\s+", "", t)
    return t.strip()


def _is_common_section_heading(text: str) -> bool:
    t = _strip_heading_prefix(text)
    if not t:
        return False
    t = re.sub(r"\s+", " ", t).strip().upper()
    if t in _COMMON_SECTION_HEADINGS:
        return True
    fuzzy_prefixes = (
        "RELATED WORK",
        "METHOD",
        "METHODS",
        "EXPERIMENT",
        "RESULT",
        "DISCUSSION",
        "CONCLUSION",
        "BACKGROUND",
        "PRELIMINARY",
        "IMPLEMENTATION",
        "ABLATION",
        "LIMITATION",
    )
    return any(t.startswith(p + " ") for p in fuzzy_prefixes)


def _is_reasonable_heading_text(title: str) -> bool:
    t = _normalize_text(title or "").strip()
    if not t:
        return False
    if _parse_numbered_heading_level(t) is not None or _parse_appendix_heading_level(t) is not None:
        return True
    if t.upper() in _COMMON_SECTION_HEADINGS:
        return True
    if _looks_like_equation_text(t):
        return False
    if re.match(r"^\s*(?:Fig\.|Figure|Table|Algorithm)\s*\d+", t, flags=re.IGNORECASE):
        return False
    if re.search(r"\b(?:doi|arxiv|https?://)\b", t, flags=re.IGNORECASE):
        return False
    if re.search(r"\[[0-9,\-\s]+\]", t):
        return False
    if "," in t and re.search(r"\b(?:19|20)\d{2}\b", t):
        return False
    if len(t) > 120:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]*", t)
    if not words or len(words) > 16:
        return False
    # Long sentence-like fragments are usually body text, not headings.
    if len(words) >= 7 and re.search(r"\b(?:we|our|this|that|these|those|is|are|was|were|have|has)\b", t, flags=re.IGNORECASE):
        return False
    if len(words) >= 8 and t.endswith("."):
        return False
    if len(words) >= 10 and re.search(r"[,:;]\s", t):
        return False

    alpha = [ch for ch in t if ch.isalpha()]
    if not alpha:
        return False
    upper_ratio = sum(1 for ch in alpha if ch.isupper()) / max(1, len(alpha))
    titlecase_ratio = sum(1 for w in words if w[:1].isupper()) / max(1, len(words))
    if upper_ratio >= 0.72 or titlecase_ratio >= 0.68:
        return True
    # Sentence-case headings are common in modern papers (not all-caps/title-case).
    if (
        len(words) <= 9
        and t[:1].isupper()
        and (not t.endswith("."))
        and (not re.search(r"\b(?:doi|arxiv|https?://)\b", t, flags=re.IGNORECASE))
    ):
        return True
    return len(words) <= 10 and t[:1].isupper() and (not t.endswith("."))


def _suggest_heading_level(
    *,
    text: str,
    max_size: float,
    is_bold: bool,
    body_size: float,
    page_width: float,
    bbox: tuple[float, float, float, float],
    page_index: int,
) -> Optional[int]:
    t = _normalize_text(text or "").strip()
    delta = float(max_size) - float(body_size)
    words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]*", t)
    style_fallback = bool(
        t
        and len(words) <= 10
        and t[:1].isupper()
        and (not t.endswith("."))
        and (delta >= 0.55 or (is_bold and delta >= 0.20))
        and (not _looks_like_equation_text(t))
        and (not re.match(r"^\s*(?:Fig\.|Figure|Table|Algorithm)\s*\d+", t, flags=re.IGNORECASE))
    )
    if not (_is_reasonable_heading_text(t) or style_fallback):
        return None

    numbered_level = _parse_numbered_heading_level(t)
    if numbered_level is not None:
        return max(1, min(3, int(numbered_level)))
    appendix_level = _parse_appendix_heading_level(t)
    if appendix_level is not None:
        return max(1, min(3, int(appendix_level)))

    is_spanning = _bbox_width(bbox) >= page_width * 0.62
    is_common = _is_common_section_heading(t)

    if is_common and (delta >= -0.2):
        if delta >= 2.6 or (is_spanning and page_index <= 1 and delta >= 1.2):
            return 1
        if delta >= 1.2:
            return 2
        if delta >= 0.45 or is_bold:
            return 3
    if delta >= 1.6 and (is_bold or is_spanning):
        return 1
    if delta >= 0.9 and (is_bold or is_spanning):
        return 2
    if style_fallback and (delta >= 0.55 or is_bold):
        return 3
    if delta >= 0.35 and is_bold and len(t) <= 90:
        return 3
    return None


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


def _noise_template_key(text: str) -> str:
    t = _normalize_text(text or "")
    if not t:
        return ""
    t = t.lower()
    t = re.sub(r"\b(?:19|20)\d{2}\b", "#", t)
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_repeated_noise_texts(doc) -> set[str]:
    if fitz is None:
        return set()
    total = len(doc)
    if total <= 1:
        return set()
    counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    template_to_texts: dict[str, set[str]] = {}

    # For long papers/books, scanning every page here is expensive.
    # Sample pages evenly; repeated headers/footers still remain detectable.
    if total <= 36:
        scan_indices = list(range(total))
    else:
        target_n = min(36, total)
        step = max(1, total // target_n)
        scan_indices = list(range(0, total, step))[:target_n]
        # Ensure tail pages are represented for footer/header variants.
        if (total - 1) not in scan_indices:
            scan_indices.append(total - 1)
        scan_indices = sorted(set(scan_indices))

    scan_total = max(1, len(scan_indices))
    for pi in scan_indices:
        page = doc[pi]
        H = float(page.rect.height)
        top_band = 85.0
        bottom_band = 95.0
        side_band = 28.0
        W = float(page.rect.width)
        d = page.get_text("dict")
        for b in d.get("blocks", []):
            if "lines" not in b:
                continue
            bbox = tuple(float(x) for x in b.get("bbox", (0, 0, 0, 0)))
            rect = fitz.Rect(bbox)
            in_top_bottom = rect.y1 < top_band or rect.y0 > H - bottom_band
            in_side = rect.x0 < side_band or rect.x1 > W - side_band
            if not (in_top_bottom or in_side):
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
                key = _noise_template_key(txt)
                if key:
                    template_counts[key] += 1
                    template_to_texts.setdefault(key, set()).add(txt)

    threshold = max(3, int(scan_total * 0.4))
    noise = {t for t, n in counts.items() if n >= threshold}
    tmpl_threshold = max(3, int(scan_total * 0.35))
    for key, n in template_counts.items():
        if n < tmpl_threshold:
            continue
        for t in template_to_texts.get(key, set()):
            if 6 <= len(t) <= 180:
                noise.add(t)
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


def _build_refs_mode_by_page(doc) -> list[bool]:
    """
    Build per-page REFERENCES mode without "sticky-to-end" behavior.

    Older logic toggled refs-mode once and kept it for all following pages, which
    breaks journals where a references block appears early. Here we classify each
    page independently, with only a one-page bridge for occasional OCR misses.
    """
    total = int(len(doc) or 0)
    if total <= 0:
        return []

    seed: list[bool] = [False] * total
    for i in range(total):
        try:
            p = doc[i]
            seed[i] = bool(_page_has_references_heading(p) or _page_looks_like_references_content(p))
        except Exception:
            seed[i] = False

    refs_mode = list(seed)
    if total >= 3:
        for i in range(1, total - 1):
            # Bridge one isolated miss between two reference-like pages.
            if (not refs_mode[i]) and refs_mode[i - 1] and refs_mode[i + 1]:
                refs_mode[i] = True

    return refs_mode


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
    # lots of math operators / greek / symbols
    # NOTE: single-line display equations are common (e.g., d鍗?/ds), so don't require 2+ lines.
    math_chars = re.findall(r"[=+\-*/^_{}()\[\]<>鍗辫熃娓綅浼湭锜胯爜閳垟鍨欓埈姘ｅ閳倐澧洪埈鐐╁灳]|[\u2200-\u22ff]|[\u0370-\u03ff]|[\ufe00-\ufe0f]", t)
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
    if len(wordish) >= 24 and len(math_chars) < 18 and "=" not in t and ("\\sum" not in t) and ("\\int" not in t):
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

    level = _suggest_heading_level(
        text=t,
        max_size=max_size,
        is_bold=is_bold,
        body_size=body_size,
        page_width=page_width,
        bbox=bbox,
        page_index=page_index,
    )
    if level is not None:
        return f"[H{max(1, min(3, int(level)))}]"

    return None


def _pick_column_range(rect: fitz.Rect, page_width: float) -> tuple[float, float]:
    mid = page_width / 2.0
    if rect.width >= page_width * 0.55:
        return (0.0, page_width)
    return (0.0, mid) if rect.x0 < mid else (mid, page_width)


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _union_rect(rects: list["fitz.Rect"]) -> Optional["fitz.Rect"]:
    if not rects:
        return None
    x0 = min(float(r.x0) for r in rects)
    y0 = min(float(r.y0) for r in rects)
    x1 = max(float(r.x1) for r in rects)
    y1 = max(float(r.y1) for r in rects)
    return fitz.Rect(x0, y0, x1, y1)


def _merge_nearby_visual_rects(rects: list["fitz.Rect"], *, page_w: float, page_h: float) -> list["fitz.Rect"]:
    if fitz is None or (not rects):
        return rects
    merged = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out: list[fitz.Rect] = []
        used = [False] * len(merged)
        for i, a in enumerate(merged):
            if used[i]:
                continue
            cur = fitz.Rect(a)
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                b = merged[j]
                x_gap = max(0.0, max(float(cur.x0), float(b.x0)) - min(float(cur.x1), float(b.x1)))
                y_gap = max(0.0, max(float(cur.y0), float(b.y0)) - min(float(cur.y1), float(b.y1)))
                x_ov = _overlap_1d(float(cur.x0), float(cur.x1), float(b.x0), float(b.x1))
                y_ov = _overlap_1d(float(cur.y0), float(cur.y1), float(b.y0), float(b.y1))
                y_min = max(1.0, min(float(cur.height), float(b.height)))
                x_min = max(1.0, min(float(cur.width), float(b.width)))
                close_h = (y_ov / y_min >= 0.45) and (x_gap <= page_w * 0.035)
                close_v = (x_ov / x_min >= 0.45) and (y_gap <= page_h * 0.028)
                touching = _rect_intersection_area(cur, b) > 0.0
                if close_h or close_v or touching:
                    cur = fitz.Rect(
                        min(float(cur.x0), float(b.x0)),
                        min(float(cur.y0), float(b.y0)),
                        max(float(cur.x1), float(b.x1)),
                        max(float(cur.y1), float(b.y1)),
                    )
                    used[j] = True
                    changed = True
            out.append(cur)
        merged = out
    return merged


def _is_caption_like_text(text: str) -> bool:
    t = _normalize_text(text or "").strip()
    if not t:
        return False
    return bool(
        re.match(
            r"^\s*(?:Fig\.|Figure|Table|Algorithm)\s*(?:\d+|[IVXLC]+)\b",
            t,
            flags=re.IGNORECASE,
        )
    )


def _collect_image_regions(page) -> list[dict[str, object]]:
    if fitz is None:
        return []
    out: list[dict[str, object]] = []
    for info in (page.get_image_info() or []):
        if "bbox" not in info:
            continue
        try:
            r = fitz.Rect(info["bbox"])
        except Exception:
            continue
        if r.width <= 1.0 or r.height <= 1.0:
            continue
        w = 0
        h = 0
        try:
            w = int(info.get("width") or 0)
        except Exception:
            w = 0
        try:
            h = int(info.get("height") or 0)
        except Exception:
            h = 0
        out.append({"rect": r, "width": w, "height": h})
    return out


def _collect_image_rects(page) -> list["fitz.Rect"]:
    if fitz is None:
        return []
    out: list[fitz.Rect] = []
    for it in _collect_image_regions(page):
        try:
            out.append(fitz.Rect(it["rect"]))
        except Exception:
            continue
    return out


def _collect_visual_rects(page, *, image_rects: Optional[list["fitz.Rect"]] = None) -> list["fitz.Rect"]:
    """
    Collect visual regions that may represent figures/charts:
    - embedded image bboxes
    - large vector drawing bboxes (for plots not embedded as images)
    """
    if fitz is None:
        return []

    page_w = float(page.rect.width)
    page_h = float(page.rect.height)
    page_area = max(1.0, page_w * page_h)
    out: list[fitz.Rect] = []

    out.extend([fitz.Rect(r) for r in (image_rects if image_rects is not None else _collect_image_rects(page))])

    # Vector charts/diagrams may not appear in get_image_info().
    try:
        for d in (page.get_drawings() or []):
            r0 = d.get("rect")
            if not r0:
                continue
            try:
                r = fitz.Rect(r0)
            except Exception:
                continue
            area = max(0.0, float(r.width) * float(r.height))
            if area < page_area * 0.003:
                continue
            if area > page_area * 0.92:
                continue
            if float(r.width) < page_w * 0.12 or float(r.height) < page_h * 0.05:
                continue
            # Skip decorative rules.
            if float(r.height) < page_h * 0.02 and float(r.width) > page_w * 0.7:
                continue
            out.append(r)
    except Exception:
        pass

    # De-duplicate near-identical rectangles.
    seen: set[tuple[int, int, int, int]] = set()
    uniq: list[fitz.Rect] = []
    for r in out:
        key = (
            int(round(float(r.x0) * 2)),
            int(round(float(r.y0) * 2)),
            int(round(float(r.x1) * 2)),
            int(round(float(r.y1) * 2)),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return _merge_nearby_visual_rects(uniq, page_w=page_w, page_h=page_h)


def _pick_render_scale(page_rect: "fitz.Rect", crop_rect: "fitz.Rect", *, base_scale: float, min_scale: float = 1.35) -> float:
    page_area = max(1.0, float(page_rect.width) * float(page_rect.height))
    crop_area = max(1.0, float(crop_rect.width) * float(crop_rect.height))
    ratio = crop_area / page_area
    scale = float(base_scale)
    # Keep large crops readable: avoid aggressive down-sampling that blurs plot labels.
    if ratio >= 0.72:
        scale = float(base_scale) * 0.92
    elif ratio >= 0.52:
        scale = float(base_scale) * 0.96
    elif ratio >= 0.34:
        scale = float(base_scale) * 1.00
    return max(float(min_scale), min(float(base_scale), float(scale)))


def _intrinsic_image_scale_hint(
    crop_rect: "fitz.Rect",
    image_regions: list[dict[str, object]],
    *,
    overlap_threshold: float = 0.70,
) -> float:
    if fitz is None or (not image_regions):
        return 0.0
    crop_area = max(1.0, _rect_area(crop_rect))
    best = 0.0
    for it in image_regions:
        try:
            r = fitz.Rect(it["rect"])
        except Exception:
            continue
        inter = _rect_intersection_area(crop_rect, r)
        if (inter / crop_area) < float(overlap_threshold):
            continue
        try:
            w_px = int(it.get("width") or 0)
        except Exception:
            w_px = 0
        try:
            h_px = int(it.get("height") or 0)
        except Exception:
            h_px = 0
        sx = (float(w_px) / max(1.0, float(r.width))) if w_px > 0 else 0.0
        sy = (float(h_px) / max(1.0, float(r.height))) if h_px > 0 else 0.0
        best = max(best, sx, sy)
    return float(best)


def _render_clip_pixmap(
    page,
    crop_rect: "fitz.Rect",
    *,
    base_scale: float,
    image_alpha: bool = False,
    min_scale: float = 1.45,
    max_scale: float = 4.8,
    min_long_edge_px: int = 2200,
    min_short_edge_px: int = 980,
    max_pixels: int = 16_000_000,
    image_regions: Optional[list[dict[str, object]]] = None,
):
    crop = fitz.Rect(crop_rect)
    scale = _pick_render_scale(page.rect, crop, base_scale=float(base_scale), min_scale=float(min_scale))

    w_pt = max(1.0, float(crop.width))
    h_pt = max(1.0, float(crop.height))
    long_pt = max(w_pt, h_pt)
    short_pt = min(w_pt, h_pt)
    scale = max(scale, float(min_long_edge_px) / long_pt, float(min_short_edge_px) / short_pt)

    if image_regions:
        scale = max(scale, _intrinsic_image_scale_hint(crop, image_regions))

    scale = min(float(max_scale), max(float(min_scale), float(scale)))
    pred_pixels = (w_pt * scale) * (h_pt * scale)
    if pred_pixels > float(max_pixels):
        shrink = (float(max_pixels) / max(1.0, pred_pixels)) ** 0.5
        scale = max(float(min_scale), float(scale) * float(shrink))

    return page.get_pixmap(matrix=fitz.Matrix(float(scale), float(scale)), clip=crop, alpha=bool(image_alpha))


def _escape_md_table_cell(value: str) -> str:
    cell = _normalize_text(value or "")
    if not cell:
        return ""
    cell = cell.replace("\r", "\n")
    cell = re.sub(r"\s*\n\s*", "<br>", cell)
    cell = cell.replace("|", r"\|")
    return cell.strip()


def _table_rows_to_markdown(rows_raw) -> Optional[str]:
    if not rows_raw or not isinstance(rows_raw, list):
        return None

    rows: list[list[str]] = []
    for row in rows_raw:
        if not isinstance(row, (list, tuple)):
            continue
        cells = [_escape_md_table_cell("" if c is None else str(c)) for c in row]
        rows.append(cells)

    # Drop empty rows.
    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 2:
        return None

    width = max(len(r) for r in rows)
    if width < 2:
        return None
    rows = [r + [""] * (width - len(r)) for r in rows]

    # Drop columns that are empty across all rows.
    keep_cols = [i for i in range(width) if any(rows[r][i].strip() for r in range(len(rows)))]
    if len(keep_cols) < 2:
        return None
    rows = [[r[i] for i in keep_cols] for r in rows]
    width = len(rows[0])

    # Some detectors prepend an almost-empty row; skip it if it looks like noise.
    if len(rows) >= 3:
        first_non_empty = sum(1 for c in rows[0] if c.strip())
        second_non_empty = sum(1 for c in rows[1] if c.strip())
        if first_non_empty <= 1 and second_non_empty >= 2:
            rows = rows[1:]

    if len(rows) < 2:
        return None

    header = rows[0]
    if not any(c.strip() for c in header):
        header = [f"col_{i + 1}" for i in range(width)]

    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(md_lines)


def _markdown_table_quality_score(md: str) -> float:
    lines = [ln.strip() for ln in (md or "").splitlines() if ln.strip()]
    if len(lines) < 3:
        return 0.0
    width = max(0, lines[0].count("|") - 1)
    body_rows = max(0, len(lines) - 2)
    non_empty_cells = 0
    for ln in lines:
        parts = [p.strip() for p in ln.strip("|").split("|")]
        non_empty_cells += sum(1 for p in parts if p and p != "---")
    return float(width * body_rows) + float(non_empty_cells) * 0.08


def _is_markdown_table_sane(md: str) -> bool:
    lines = [ln.strip() for ln in (md or "").splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    width = max(0, lines[0].count("|") - 1)
    if width < 2 or width > 18:
        return False
    cells: list[str] = []
    cols: list[list[str]] = [[] for _ in range(width)]
    numeric_cells = 0
    row_non_empty: list[int] = []
    filled_slots = 0
    total_slots = 0
    for ln in lines[2:]:
        parts = [p.strip() for p in ln.strip("|").split("|")]
        if len(parts) != width:
            return False
        non_empty_in_row = 0
        for ci, p in enumerate(parts):
            if p:
                non_empty_in_row += 1
                cols[ci].append(p)
                cells.append(p)
                if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|e[+-]?\d+)?", p, flags=re.IGNORECASE):
                    numeric_cells += 1
        row_non_empty.append(non_empty_in_row)
        filled_slots += non_empty_in_row
        total_slots += width
    if not cells:
        return False
    if not row_non_empty:
        return False
    # Reject highly sparse grids (common false positives from figure axis text).
    fill_ratio = filled_slots / max(1, total_slots)
    if fill_ratio < 0.36:
        return False
    sparse_rows = sum(1 for n in row_non_empty if n <= max(1, width // 3))
    if sparse_rows / max(1, len(row_non_empty)) > 0.34:
        return False
    if numeric_cells == 0 and len(lines) >= 5:
        return False
    tiny_ratio = sum(1 for c in cells if len(c) <= 1) / max(1, len(cells))
    if tiny_ratio > 0.35:
        return False
    tiny_alpha_ratio = sum(1 for c in cells if re.fullmatch(r"[A-Za-z]{1,2}", c)) / max(1, len(cells))
    if tiny_alpha_ratio > 0.22:
        return False
    if len(lines) > 12 and tiny_alpha_ratio > 0.10:
        return False
    if width >= 5 and tiny_alpha_ratio > 0.14:
        return False
    for col_cells in cols:
        if len(col_cells) < 3:
            continue
        tiny_col = sum(1 for c in col_cells if len(c) <= 1) / max(1, len(col_cells))
        if tiny_col > 0.78:
            return False
        tiny_alpha_col = sum(1 for c in col_cells if re.fullmatch(r"[A-Za-z]{1,2}", c)) / max(1, len(col_cells))
        if tiny_alpha_col > 0.58:
            return False
    long_phrase_ratio = sum(1 for c in cells if len(re.findall(r"[A-Za-z]{2,}", c)) >= 8) / max(1, len(cells))
    if long_phrase_ratio > 0.55:
        return False
    if long_phrase_ratio > 0.38 and (numeric_cells / max(1, len(cells))) < 0.12:
        return False
    return True


def _extract_tables_by_pdfplumber(pdf_path: Optional[Path], page_index: int) -> list[tuple["fitz.Rect", str]]:
    if fitz is None or (pdf_path is None):
        return []
    try:
        pdm = _ensure_pdfplumber_module()
    except Exception:
        return []
    out: list[tuple[fitz.Rect, str]] = []
    try:
        with pdm.open(str(pdf_path)) as pd:
            if page_index < 0 or page_index >= len(pd.pages):
                return []
            pg = pd.pages[page_index]
            tables = pg.find_tables(
                table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "intersection_tolerance": 3,
                    "min_words_vertical": 2,
                    "min_words_horizontal": 1,
                }
            )
            for tb in tables:
                try:
                    bbox = tuple(float(x) for x in tb.bbox)
                    rect = fitz.Rect(bbox)
                except Exception:
                    continue
                try:
                    rows = tb.extract()
                except Exception:
                    rows = None
                md = _table_rows_to_markdown(rows) if rows is not None else None
                if not md:
                    continue
                if not _is_markdown_table_sane(md):
                    continue
                out.append((rect, md))
    except Exception:
        return []
    return out


def _page_maybe_has_table_from_dict(page_dict: dict) -> bool:
    """
    Fast page-level gate to avoid expensive table finder calls on obvious non-table pages.
    """
    blocks = page_dict.get("blocks", []) if isinstance(page_dict, dict) else []
    if not blocks:
        return False
    numeric_rows = 0
    delimiter_rows = 0
    scanned = 0
    for b in blocks:
        if "lines" not in b:
            continue
        for l in (b.get("lines", []) or []):
            spans = l.get("spans", []) or []
            if not spans:
                continue
            text = _normalize_text("".join(str(s.get("text", "")) for s in spans))
            if not text:
                continue
            scanned += 1
            if re.match(r"^\s*Table\s+(?:\d+|[IVXLC]+)\b", text, flags=re.IGNORECASE):
                return True
            if len(text) > 180:
                if scanned >= 260:
                    break
                continue
            cols = [c for c in re.split(r"\t+|\s{2,}", text.strip()) if c.strip()]
            nums = re.findall(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|e[+-]?\d+)?", text, flags=re.IGNORECASE)
            has_delim = ("|" in text) or (len(cols) >= 3)
            if has_delim:
                delimiter_rows += 1
            if len(nums) >= 2 and (has_delim or len(cols) >= 2):
                numeric_rows += 1

            if numeric_rows >= 3 and delimiter_rows >= 2:
                return True
            if delimiter_rows >= 6 and numeric_rows >= 2:
                return True
            if scanned >= 260:
                break
        if scanned >= 260:
            break
    return False


def _extract_tables_by_layout(
    page,
    *,
    pdf_path: Optional[Path] = None,
    page_index: int = 0,
    visual_rects: Optional[list["fitz.Rect"]] = None,
    use_pdfplumber_fallback: bool = False,
) -> list[tuple["fitz.Rect", str]]:
    """
    Prefer PyMuPDF's structural table detector to avoid treating tables as plain paragraphs.
    """
    if fitz is None or (not hasattr(page, "find_tables")):
        return []

    page_w = float(page.rect.width)
    page_h = float(page.rect.height)
    page_area = max(1.0, page_w * page_h)
    vis_rects = [fitz.Rect(r) for r in (visual_rects or [])]

    try:
        quick_text = _normalize_text(page.get_text("text") or "")
    except Exception:
        quick_text = ""
    table_keyword_hint = bool(re.search(r"(?mi)^\s*table\s+(?:\d+|[ivxlc]+)\b", quick_text))
    delimiter_lines = sum(1 for ln in quick_text.splitlines() if re.search(r"(?:\t| {3,})", ln))
    has_table_hint = bool(table_keyword_hint or delimiter_lines >= 3)

    caption_rects: list[fitz.Rect] = []
    try:
        d = page.get_text("dict")
        for b in d.get("blocks", []):
            if "lines" not in b:
                continue
            bbox = b.get("bbox")
            if not bbox:
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
            if re.match(r"^\s*Table\s+(?:\d+|[IVXLC]+)\b", txt, flags=re.IGNORECASE):
                caption_rects.append(fitz.Rect(tuple(float(x) for x in bbox)))
    except Exception:
        caption_rects = []

    def _has_nearby_table_caption(rect: "fitz.Rect") -> bool:
        if not caption_rects:
            return False
        for cr in caption_rects:
            hov = _overlap_1d(float(rect.x0), float(rect.x1), float(cr.x0) - 18.0, float(cr.x1) + 18.0)
            min_hov = max(10.0, min(float(rect.width), float(cr.width) + 36.0) * 0.15)
            if hov < min_hov:
                continue
            vgap = min(abs(float(rect.y0) - float(cr.y1)), abs(float(cr.y0) - float(rect.y1)))
            if vgap <= max(110.0, page_h * 0.24):
                return True
        return False

    primary_kwargs = [{"vertical_strategy": "lines", "horizontal_strategy": "lines"}]
    if has_table_hint:
        primary_kwargs.extend(
            [
                {"vertical_strategy": "lines", "horizontal_strategy": "text", "min_words_horizontal": 1, "text_tolerance": 2.0},
                {"vertical_strategy": "text", "horizontal_strategy": "lines", "min_words_vertical": 2, "text_tolerance": 2.0},
            ]
        )

    candidates: list[tuple[fitz.Rect, str, float]] = []
    kwargs_seq = primary_kwargs
    for kwargs in kwargs_seq:
        uses_text_strategy = ("text" in str(kwargs.get("vertical_strategy", "")).lower()) or (
            "text" in str(kwargs.get("horizontal_strategy", "")).lower()
        )
        try:
            table_finder = page.find_tables(**kwargs)
        except Exception:
            continue
        tables = getattr(table_finder, "tables", table_finder)
        if not tables:
            continue

        for tb in tables:
            try:
                rect = fitz.Rect(getattr(tb, "bbox", None))
            except Exception:
                continue
            if _rect_area(rect) < page_area * 0.0035:
                continue
            if float(rect.width) < page_w * 0.12 or float(rect.height) < page_h * 0.04:
                continue
            if _rect_area(rect) > page_area * 0.55:
                continue

            md = None
            try:
                md = _table_rows_to_markdown(tb.extract())
            except Exception:
                md = None
            if not md:
                try:
                    raw_clip = page.get_text("text", clip=rect)
                except Exception:
                    raw_clip = ""
                md = table_text_to_markdown(raw_clip) if raw_clip else None
                if (not md) and raw_clip:
                    md = _table_from_numeric_pattern([ln for ln in raw_clip.splitlines() if ln.strip()])
            if not md:
                continue
            if not _is_markdown_table_sane(md):
                continue
            near_caption = _has_nearby_table_caption(rect)
            if vis_rects and (not near_caption):
                vis_overlap = max(
                    (
                        _rect_intersection_area(rect, vr) / max(1.0, min(_rect_area(rect), _rect_area(vr)))
                        for vr in vis_rects
                    ),
                    default=0.0,
                )
                # Strongly suppress figure-overlapping table false positives.
                if vis_overlap >= 0.62:
                    continue
                if vis_overlap >= 0.45 and (not has_table_hint):
                    continue
            if uses_text_strategy and (not near_caption):
                # Text-based strategies are prone to chart-axis false positives.
                # Keep only compact, denser candidates when no table caption is nearby.
                if _rect_area(rect) > page_area * 0.18:
                    continue
                if len([ln for ln in md.splitlines() if ln.strip()]) < 4:
                    continue
            if (not has_table_hint) and (not near_caption) and _rect_area(rect) > page_area * 0.08:
                continue
            score = _markdown_table_quality_score(md)
            if score <= 0.0:
                continue
            candidates.append((rect, md, score))

    if (not candidates) and use_pdfplumber_fallback and (pdf_path is not None) and has_table_hint:
        for rect, md in _extract_tables_by_pdfplumber(pdf_path, page_index):
            score = _markdown_table_quality_score(md)
            if score > 0.0:
                candidates.append((rect, md, score))

    if not candidates:
        return []

    # De-duplicate overlapping candidates, keeping the better table representation.
    uniq: list[tuple[fitz.Rect, str, float]] = []
    for rect, md, score in sorted(candidates, key=lambda x: (x[0].y0, x[0].x0, -x[2])):
        replaced = False
        for i, (r0, md0, s0) in enumerate(uniq):
            inter = _rect_intersection_area(rect, r0)
            denom = max(1.0, min(_rect_area(rect), _rect_area(r0)))
            if inter / denom >= 0.72:
                if score > s0:
                    uniq[i] = (rect, md, score)
                replaced = True
                break
        if not replaced:
            uniq.append((rect, md, score))

    uniq.sort(key=lambda x: (x[0].y0, x[0].x0))
    return [(r, md) for r, md, _ in uniq]


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
    if len(t) <= 18 and re.fullmatch(r"[A-Za-z0-9=+*/^_()鍗遍埈鎴斿灆閳瓡\\-]+", t):
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
                table_markdown=b.table_markdown,
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
                    table_markdown=b.table_markdown,
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
    body blocks (e.g. "i閳湤 c_i 浼猒i"). These fragments should be dropped (they are duplicates),
    otherwise they pollute the surrounding paragraph.
    """
    if not blocks:
        return blocks
    out: list[TextBlock] = []
    sym_re = re.compile(r"[=^_\\]|[閳嚦锝傚灆閳埃鍩嗛埈鐐╁⒐閳儮澧甸埉鍫氬灳閳熂钄藉弬婢勬婊存鑲鸿倽鍒綍鏅冨槈杞跨枤鑲偓锜昏熃锜胯爛锠佽爞锠勮爡]")
    for i, b in enumerate(blocks):
        if b.is_math or b.is_table or b.is_code:
            out.append(b)
            continue
        t = _normalize_text(b.text).strip()
        if not t:
            continue
        # lone dashes created by bullet splitting / extraction noise
        if t in {"-", "\u2013", "\u2014"}:
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
    visual_rects: Optional[list["fitz.Rect"]] = None,
    image_scale: float = 2.0,
    image_alpha: bool = False,
) -> tuple[dict[int, str], list["fitz.Rect"]]:
    caption_re = re.compile(r"^\s*(?:Fig\.|FIG\.|Figure|FIGURE)\s*([0-9]+)", re.IGNORECASE)
    visual_rects = [fitz.Rect(r) for r in (visual_rects or _collect_visual_rects(page))]
    image_regions = _collect_image_regions(page)
    page_w = float(page.rect.width)
    page_h = float(page.rect.height)

    out: dict[int, str] = {}
    covered: list[fitz.Rect] = []
    for bi, b in enumerate(blocks):
        m = caption_re.match(b.text)
        if not m:
            continue

        fig_num = m.group(1)
        caption_rect = fitz.Rect(b.bbox)
        spanning_caption = float(caption_rect.width) >= page_w * 0.5
        if spanning_caption:
            col_x0, col_x1 = 0.0, page_w
        else:
            col_x0, col_x1 = _pick_column_range(caption_rect, page.rect.width)
        col_w = max(1.0, float(col_x1 - col_x0))

        # Some PDFs place a full-width figure with a caption that looks single-column.
        # If nearby visuals clearly occupy both columns, upgrade to full-width search.
        if not spanning_caption:
            near: list[tuple[float, fitz.Rect]] = []
            for r in visual_rects:
                if r.y1 > caption_rect.y0 + 14.0:
                    continue
                gap = float(caption_rect.y0) - float(r.y1)
                if gap < -8.0 or gap > page_h * 0.55:
                    continue
                near.append((gap, r))
            if near:
                near.sort(key=lambda x: x[0])
                g0 = float(near[0][0])
                near = [(g, r) for g, r in near if float(g) <= g0 + page_h * 0.18]
                mid = page_w * 0.5
                has_left = any(((float(r.x0) + float(r.x1)) * 0.5) < mid - page_w * 0.06 for _, r in near)
                has_right = any(((float(r.x0) + float(r.x1)) * 0.5) > mid + page_w * 0.06 for _, r in near)
                if has_left and has_right:
                    spanning_caption = True
                    col_x0, col_x1 = 0.0, page_w
                    col_w = page_w

        candidates: list[tuple[float, fitz.Rect]] = []
        for r in visual_rects:
            if r.y1 > caption_rect.y0 + 12.0:
                continue
            gap = float(caption_rect.y0) - float(r.y1)
            if gap < -8.0 or gap > page_h * 0.62:
                continue
            # Keep visuals that overlap caption column (double-column with single-column figures)
            # or overlap the caption text range itself.
            overlap_col = _overlap_1d(float(r.x0), float(r.x1), float(col_x0), float(col_x1))
            overlap_cap = _overlap_1d(float(r.x0), float(r.x1), float(caption_rect.x0) - 14.0, float(caption_rect.x1) + 14.0)
            min_col_overlap = max(12.0, min(float(r.width), col_w) * (0.08 if spanning_caption else 0.16))
            min_cap_overlap = max(8.0, min(float(r.width), max(20.0, float(caption_rect.width))) * (0.10 if spanning_caption else 0.14))
            if (overlap_col < min_col_overlap) and (overlap_cap < min_cap_overlap):
                # Keep likely neighboring panels for multi-panel figures.
                large_panel = float(r.width) >= page_w * 0.18 and float(r.height) >= page_h * 0.10
                if (not spanning_caption) and large_panel and gap <= page_h * 0.26:
                    candidates.append((gap, r))
                    continue
                continue
            candidates.append((gap, r))

        selected_rects: list[fitz.Rect] = []
        if candidates:
            # Keep the nearest visual group to avoid swallowing unrelated visuals higher on page.
            candidates.sort(key=lambda x: x[0])
            min_gap = float(candidates[0][0])
            keep_gap = min(
                page_h * (0.42 if spanning_caption else 0.30),
                min_gap + max(96.0, page_h * (0.22 if spanning_caption else 0.14)),
            )
            selected_rects = [r for g, r in candidates if float(g) <= keep_gap]

            # Expand with nearby rects that belong to the same multi-panel visual region.
            changed = True
            while changed and selected_rects:
                changed = False
                u0 = _union_rect(selected_rects)
                if u0 is None:
                    break
                for g, r in candidates:
                    if r in selected_rects:
                        continue
                    if float(g) > (keep_gap + (160.0 if spanning_caption else 110.0)):
                        continue
                    ov = _overlap_1d(float(r.x0), float(r.x1), float(u0.x0), float(u0.x1))
                    yov = _overlap_1d(float(r.y0), float(r.y1), float(u0.y0), float(u0.y1))
                    vgap = max(0.0, float(u0.y0) - float(r.y1), float(r.y0) - float(u0.y1))
                    near_stack = (
                        ov >= max(8.0, min(float(r.width), float(u0.width)) * 0.16)
                        and vgap <= max(44.0, page_h * 0.085)
                    )
                    if (
                        ov >= max(10.0, min(float(r.width), float(u0.width)) * 0.20)
                        or yov >= max(10.0, min(float(r.height), float(u0.height)) * 0.35)
                        or near_stack
                    ):
                        selected_rects.append(r)
                        changed = True

            # Final pass over all visual rects to avoid dropping an adjacent panel due strict candidate gating.
            changed = True
            while changed and selected_rects:
                changed = False
                u0 = _union_rect(selected_rects)
                if u0 is None:
                    break
                for r in visual_rects:
                    if r in selected_rects:
                        continue
                    if r.y1 > caption_rect.y0 + 14.0:
                        continue
                    gap = float(caption_rect.y0) - float(r.y1)
                    if gap < -8.0 or gap > page_h * 0.62:
                        continue
                    ov = _overlap_1d(float(r.x0), float(r.x1), float(u0.x0), float(u0.x1))
                    yov = _overlap_1d(float(r.y0), float(r.y1), float(u0.y0), float(u0.y1))
                    hgap = max(0.0, float(u0.x0) - float(r.x1), float(r.x0) - float(u0.x1))
                    vgap = max(0.0, float(u0.y0) - float(r.y1), float(r.y0) - float(u0.y1))
                    if ov >= max(8.0, min(float(r.width), float(u0.width)) * 0.12):
                        selected_rects.append(r)
                        changed = True
                        continue
                    if yov >= max(8.0, min(float(r.height), float(u0.height)) * 0.25):
                        if hgap <= max(34.0, page_w * 0.055):
                            selected_rects.append(r)
                            changed = True
                            continue
                    if vgap <= max(30.0, page_h * 0.06) and hgap <= max(36.0, page_w * 0.06):
                        selected_rects.append(r)
                        changed = True
                        continue

            # Very relaxed bridge pass for multi-panel figures:
            # if a panel sits in the same y-band and close in x, include it to avoid half-cut crops.
            changed = True
            while changed and selected_rects:
                changed = False
                u0 = _union_rect(selected_rects)
                if u0 is None:
                    break
                y0_band = max(0.0, float(u0.y0) - page_h * 0.12)
                y1_band = min(page_h, float(u0.y1) + page_h * 0.12)
                for r in visual_rects:
                    if r in selected_rects:
                        continue
                    if r.y1 > caption_rect.y0 + 14.0:
                        continue
                    gap = float(caption_rect.y0) - float(r.y1)
                    if gap < -8.0 or gap > page_h * 0.62:
                        continue
                    y_band_overlap = _overlap_1d(float(r.y0), float(r.y1), y0_band, y1_band)
                    hgap = max(0.0, float(u0.x0) - float(r.x1), float(r.x0) - float(u0.x1))
                    if y_band_overlap >= max(12.0, float(r.height) * 0.20) and hgap <= max(84.0, page_w * 0.14):
                        selected_rects.append(r)
                        changed = True

        if selected_rects:
            u = _union_rect(selected_rects)
            assert u is not None
            pad_x = max(8.0, min(18.0, float(u.width) * 0.04))
            pad_y = max(8.0, min(20.0, float(u.height) * 0.07))
            crop = fitz.Rect(
                max(0.0, float(u.x0) - pad_x),
                max(0.0, float(u.y0) - pad_y),
                min(page_w, float(u.x1) + pad_x),
                min(page_h, float(caption_rect.y1) + 10.0),
            )
            covered.extend([fitz.Rect(r) for r in selected_rects])
        else:
            # No reliable visual bbox: use conservative column fallback around caption.
            max_up = min(page_h * (0.68 if spanning_caption else 0.56), float(caption_rect.y0))
            y0 = max(0.0, float(caption_rect.y0) - max_up)
            prev_y1: Optional[float] = None
            for pj in range(bi - 1, -1, -1):
                pr = fitz.Rect(blocks[pj].bbox)
                overlap = _overlap_1d(float(pr.x0), float(pr.x1), float(col_x0), float(col_x1))
                min_overlap = max(10.0, min(float(pr.width), col_w) * (0.14 if spanning_caption else 0.30))
                if overlap < min_overlap:
                    continue
                if float(pr.y1) > float(caption_rect.y0):
                    continue
                txt = _normalize_text(blocks[pj].text)
                if len(txt) >= 90 and (not caption_re.match(txt)) and (not _looks_like_equation_text(txt)):
                    prev_y1 = float(pr.y1)
                    break
            if prev_y1 is not None:
                y0 = max(y0, prev_y1 + 6.0)
            pad = 10.0
            crop = fitz.Rect(
                max(0.0, float(col_x0) - pad),
                max(0.0, y0 - pad),
                min(page_w, float(col_x1) + pad),
                min(page_h, float(caption_rect.y1) + pad),
            )

        img_name = f"figure_{fig_num}_p{page_index + 1:03d}_{ASSET_REV_TAG}.png"
        pix = _render_clip_pixmap(
            page,
            crop,
            base_scale=float(image_scale),
            image_alpha=bool(image_alpha),
            min_scale=1.55,
            max_scale=4.8,
            min_long_edge_px=2300,
            min_short_edge_px=1024,
            image_regions=image_regions,
        )
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
    visual_rects: Optional[list["fitz.Rect"]] = None,
    image_scale: float = 2.0,
    image_alpha: bool = False,
) -> dict[int, str]:
    """
    Best-effort capture of images that don't have detectable captions.
    Goal: maximize recall (avoid missing figures), while filtering obvious tiny logos/icons.
    """
    img_rects = [fitz.Rect(r) for r in (visual_rects or _collect_visual_rects(page))]
    image_regions = _collect_image_regions(page)
    if not img_rects:
        return {}

    page_w = float(page.rect.width)
    page_h = float(page.rect.height)
    page_area = max(1.0, page_w * page_h)

    def is_covered(r: fitz.Rect) -> bool:
        ra = max(1.0, float(r.width) * float(r.height))
        for c in covered_rects or []:
            ia = _rect_intersection_area(r, c)
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
        img_name = f"image_auto_{auto_i:02d}_p{page_index + 1:03d}_{ASSET_REV_TAG}.png"
        pix = _render_clip_pixmap(
            page,
            crop,
            base_scale=float(image_scale),
            image_alpha=bool(image_alpha),
            min_scale=1.45,
            max_scale=4.6,
            min_long_edge_px=1900,
            min_short_edge_px=860,
            image_regions=image_regions,
        )
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
            img_name = f"equation_{eqno}_p{page_index + 1:03d}_b{bi:02d}_{ASSET_REV_TAG}.png"
        else:
            img_name = f"equation_auto_p{page_index + 1:03d}_b{bi:02d}_{ASSET_REV_TAG}.png"

        try:
            pix = _render_clip_pixmap(
                page,
                crop,
                base_scale=float(image_scale),
                image_alpha=bool(image_alpha),
                min_scale=1.6,
                max_scale=5.2,
                min_long_edge_px=1800,
                min_short_edge_px=720,
                image_regions=None,
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
    pdf_path: Optional[Path] = None,
    image_rects: Optional[list["fitz.Rect"]] = None,
    visual_rects: Optional[list["fitz.Rect"]] = None,
    drop_frontmatter: bool = True,
    relax_small_text_filter: bool = False,
    preserve_body_linebreaks: bool = False,
    detect_tables: bool = True,
    table_pdfplumber_fallback: bool = False,
) -> list[TextBlock]:
    d = page.get_text("dict")
    blocks: list[TextBlock] = []
    W, H = float(page.rect.width), float(page.rect.height)
    # References often sit closer to the top/bottom in two-column layouts. Relax bands there.
    header_y = 25.0 if relax_small_text_filter else 75.0
    footer_y = H - (25.0 if relax_small_text_filter else 70.0)

    img_rects = [fitz.Rect(r) for r in (image_rects if image_rects is not None else _collect_image_rects(page))]
    vis_rects = [fitz.Rect(r) for r in (visual_rects or _collect_visual_rects(page, image_rects=img_rects))]
    can_have_table = bool(detect_tables) and _page_maybe_has_table_from_dict(d)
    table_regions = (
        _extract_tables_by_layout(
            page,
            pdf_path=pdf_path,
            page_index=page_index,
            visual_rects=vis_rects,
            use_pdfplumber_fallback=bool(table_pdfplumber_fallback),
        )
        if can_have_table
        else []
    )
    table_rects = [r for r, _ in table_regions]
    if table_rects and vis_rects:
        vis_rects = [
            vr
            for vr in vis_rects
            if max((_rect_intersection_area(vr, tr) / max(1.0, min(_rect_area(vr), _rect_area(tr))) for tr in table_rects), default=0.0)
            < 0.5
        ]

    for b in d.get("blocks", []):
        if "lines" not in b:
            continue
        bbox = tuple(float(x) for x in b.get("bbox", (0, 0, 0, 0)))
        rect = fitz.Rect(bbox)

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
        joined_probe = _normalize_text(" ".join(probe))
        is_table_caption = bool(re.match(r"^\s*Table\s+(?:\d+|[IVXLC]+)\b", joined_probe, re.IGNORECASE))
        is_caption_like = _is_caption_like_text(joined_probe)

        # Table detector already extracted this region as structured rows.
        # Skip overlapping text blocks to avoid duplicate "table as paragraph" output.
        if table_rects and (not is_table_caption):
            block_area = max(1.0, _rect_area(rect))
            inter_table = max((_rect_intersection_area(rect, tr) for tr in table_rects), default=0.0)
            if inter_table / block_area >= 0.62:
                continue

        # Text inside figure/plot visuals is frequently noisy OCR and should not pollute body/table parsing.
        if vis_rects and (not is_caption_like):
            block_area = max(1.0, _rect_area(rect))
            cx = (float(rect.x0) + float(rect.x1)) / 2.0
            cy = (float(rect.y0) + float(rect.y1)) / 2.0
            center_inside_visual = any((vr.x0 <= cx <= vr.x1) and (vr.y0 <= cy <= vr.y1) for vr in vis_rects)
            inter_parts = [_rect_intersection_area(rect, vr) for vr in vis_rects]
            inter_vis_max = max(inter_parts, default=0.0)
            inter_vis_sum = min(block_area, sum(inter_parts))
            vis_ratio = inter_vis_sum / block_area
            single_char_like = sum(1 for ln in probe if len(re.findall(r"[A-Za-z0-9]", ln)) <= 1)
            short_word_count = len(re.findall(r"[A-Za-z]{2,}", joined_probe))
            if vis_ratio >= 0.62 or inter_vis_max / block_area >= 0.75:
                # Keep only table-like text strongly (for rare drawing-based tables).
                if not _looks_like_table_block(probe):
                    continue
            if center_inside_visual and max_size <= body_size + 0.8 and short_word_count <= 18:
                # Typical axis labels / panel labels / embedded figure text.
                if (not is_table_caption) and (not _is_reasonable_heading_text(joined_probe)):
                    continue
            if center_inside_visual and len(joined_probe) <= 28 and (" " not in joined_probe):
                # Vertical side labels often become one short token without spaces.
                if len(re.findall(r"[A-Za-z]", joined_probe)) >= 2:
                    continue
            if center_inside_visual and _looks_like_table_block(probe) and (not is_table_caption):
                # Figure-internal labels can look tabular; suppress unless explicitly a Table caption block.
                if not re.search(r"\b(?:method|dataset|psnr|ssim|fid|table)\b", joined_probe, flags=re.IGNORECASE):
                    continue
            if vis_ratio >= 0.45 and (single_char_like >= max(2, len(probe) // 2)):
                continue

        # Header/footer band filter, but keep real structural content.
        if rect.y1 < header_y or rect.y0 > footer_y:
            is_caption0 = bool(re.match(r"^\s*(?:Fig\.|FIG\.|Figure|FIGURE)\s*[0-9]+", joined_probe, re.IGNORECASE))
            is_eqno0 = bool(re.fullmatch(r"\(\s*\d{1,4}\s*\)", joined_probe))
            is_struct_heading0 = bool(_is_numbered_heading_text(joined_probe) or _is_appendix_heading_text(joined_probe))
            if (not is_caption0) and (not is_eqno0) and (not is_struct_heading0):
                if max_size < body_size + 0.2:
                    continue

        # If a text block is almost fully inside an image rect, it's often garbage (publisher logos),
        # but captions may also intersect (depending on how the PDF encodes bbox). Keep captions and equation numbers.
        if img_rects:
            block_area = max(1.0, _rect_area(rect))
            inter = max((_rect_intersection_area(rect, r) for r in img_rects), default=0.0)
            inside_ratio = inter / block_area
            if inside_ratio > 0.78:
                if not re.match(r"^\s*(?:Fig\.|FIG\.|Figure|FIGURE)\s*[0-9]+", joined_probe, re.IGNORECASE) and not re.fullmatch(
                    r"\(\s*\d{1,4}\s*\)", joined_probe
                ):
                    # still allow real math/code blocks which sometimes intersect due to bbox quirks
                    if max_size < body_size - 0.2 and not _looks_like_math_block(probe) and not _looks_like_code_block(probe):
                        continue

        # Extra non-body filter: small text near margins/bands (journal footers, figure credits, etc.)
        # Disable/relax this in REFERENCES pages to avoid dropping right-column items and page-continued refs.
        if (not relax_small_text_filter) and max_size < body_size - 0.6:
            near_top = rect.y1 < 110.0
            near_bottom = rect.y0 > H - 110.0
            near_side = rect.x0 < 28.0 or rect.x1 > W - 28.0
            # Keep equation numbers even if they sit near the right margin.
            # Keep figure captions even when small and close to margins (common in two-column PDFs).
            is_eqno = bool(re.fullmatch(r"\(\s*\d{1,4}\s*\)", joined_probe))
            is_caption = bool(re.match(r"^\s*(?:Fig\.|FIG\.|Figure|FIGURE)\s*[0-9]+", joined_probe, re.IGNORECASE))
            is_heading = bool(_is_numbered_heading_text(joined_probe) or _is_appendix_heading_text(joined_probe))
            short_margin_noise = len(joined_probe) <= 90
            if (near_top or near_bottom or near_side) and short_margin_noise and (not is_eqno) and (not is_caption) and (not is_heading):
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
        # Extra cleanup for narrow-margin vertical artifacts (common in figure side labels).
        if (rect.x0 < 20.0 or rect.x1 > W - 20.0) and _bbox_width(bbox) < max(16.0, W * 0.03):
            if len(re.findall(r"[A-Za-z]", text)) <= 6:
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

    for table_rect, table_md in table_regions:
        blocks.append(
            TextBlock(
                bbox=(float(table_rect.x0), float(table_rect.y0), float(table_rect.x1), float(table_rect.y1)),
                text=table_md,
                max_font_size=body_size,
                is_bold=False,
                is_table=True,
                table_markdown=table_md,
            )
        )

    return blocks


def _detect_column_split_x(blocks: list[TextBlock], page_width: float) -> Optional[float]:
    """
    Detect the x-position separating left/right columns.
    Returns None for likely single-column layouts.
    """
    if not blocks:
        return None

    candidates = [
        (float(b.bbox[0]) + float(b.bbox[2])) / 2.0
        for b in blocks
        if _bbox_width(b.bbox) < page_width * 0.62
    ]
    if len(candidates) < 4:
        return None
    centers = sorted(candidates)

    best_gap = 0.0
    best_mid = None
    lo = page_width * 0.22
    hi = page_width * 0.78
    for i in range(len(centers) - 1):
        a, b = centers[i], centers[i + 1]
        mid = (a + b) / 2.0
        if mid < lo or mid > hi:
            continue
        gap = float(b - a)
        if gap > best_gap:
            best_gap = gap
            best_mid = mid

    if best_mid is None or best_gap < page_width * 0.08:
        return None

    left_n = sum(1 for c in centers if c < best_mid)
    right_n = len(centers) - left_n
    if left_n < 2 or right_n < 2:
        return None
    return float(best_mid)


def sort_blocks_reading_order(blocks: list[TextBlock], page_width: float) -> list[TextBlock]:
    if not blocks:
        return []

    col_split = _detect_column_split_x(blocks, page_width=page_width)
    if col_split is None:
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

    spanning_threshold = page_width * 0.62
    cross_margin = max(8.0, page_width * 0.015)
    by_y = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

    out: list[TextBlock] = []
    segment: list[TextBlock] = []

    def flush_segment():
        nonlocal segment
        if not segment:
            return
        left = [b for b in segment if ((float(b.bbox[0]) + float(b.bbox[2])) / 2.0) < col_split]
        right = [b for b in segment if ((float(b.bbox[0]) + float(b.bbox[2])) / 2.0) >= col_split]
        left.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        right.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        out.extend(left)
        out.extend(right)
        segment = []

    for b in by_y:
        x0, _, x1, _ = b.bbox
        crosses_split = float(x0) < (col_split - cross_margin) and float(x1) > (col_split + cross_margin)
        if _bbox_width(b.bbox) >= spanning_threshold or crosses_split:
            flush_segment()
            out.append(b)
        else:
            segment.append(b)

    flush_segment()
    return out


def promote_heading_blocks(
    blocks: list[TextBlock],
    *,
    body_size: float,
    page_width: float,
    page_index: int,
) -> list[TextBlock]:
    """
    Promote likely heading lines to heading blocks using style + context.
    """
    if not blocks:
        return []

    out: list[TextBlock] = []
    for i, b in enumerate(blocks):
        if b.heading_level is not None or b.is_table or b.is_math or b.is_code:
            out.append(b)
            continue
        t = _normalize_text(b.text).strip()
        words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]*", t)
        style_candidate = bool(
            t
            and len(words) <= 10
            and t[:1].isupper()
            and (not t.endswith("."))
            and (float(b.max_font_size) - float(body_size) >= 0.55 or (b.is_bold and float(b.max_font_size) - float(body_size) >= 0.20))
            and (not _looks_like_equation_text(t))
            and (not re.match(r"^\s*(?:Fig\.|Figure|Table|Algorithm)\s*\d+", t, flags=re.IGNORECASE))
        )
        if not (_is_reasonable_heading_text(t) or style_candidate):
            out.append(b)
            continue

        # Require local context: a heading is usually followed by non-trivial body text.
        next_body_len = 0
        for j in range(i + 1, min(len(blocks), i + 6)):
            nb = blocks[j]
            if nb.is_table or nb.is_math or nb.is_code:
                continue
            nt = _normalize_text(nb.text).strip()
            if nt:
                next_body_len = len(nt)
                break
        if next_body_len < 40 and (not _is_common_section_heading(t)) and (not style_candidate):
            out.append(b)
            continue

        lvl = _suggest_heading_level(
            text=t,
            max_size=b.max_font_size,
            is_bold=b.is_bold,
            body_size=body_size,
            page_width=page_width,
            bbox=b.bbox,
            page_index=page_index,
        )
        if lvl is None:
            out.append(b)
            continue

        out.append(
            TextBlock(
                bbox=b.bbox,
                text=b.text,
                max_font_size=b.max_font_size,
                is_bold=b.is_bold,
                insert_image=b.insert_image,
                is_code=b.is_code,
                is_table=b.is_table,
                table_markdown=b.table_markdown,
                is_math=b.is_math,
                heading_level=max(1, min(3, int(lvl))),
            )
        )
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
            title = _normalize_text(b.text).strip()
            lvl = int(b.heading_level)
            numbered_level = _parse_numbered_heading_level(title)
            appendix_level = _parse_appendix_heading_level(title)
            if numbered_level is not None:
                lvl = int(numbered_level)
            elif appendix_level is not None:
                lvl = int(appendix_level)
            if 1 <= lvl <= 3 and _is_reasonable_heading_text(title):
                if title.upper() in _ALLOWED_UNNUMBERED_HEADINGS:
                    lvl = 1
                    title = title.upper()
                out.append(f"[H{lvl}] {title}")
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
        marker = None
        if "\u2022" in line:
            marker = "\u2022"
        elif "\u00b7" in line:
            marker = "\u00b7"
        if marker is None:
            out.append(line)
            continue
        parts = [p.strip() for p in line.split(marker) if p.strip()]
        if len(parts) <= 1:
            out.append(line.replace(marker, "- "))
            continue
        for p in parts:
            out.append(f"- {p}")
    return "\n".join(out)


def _convert_caption_following_tabular_lines(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_code = False
    in_math = False
    caption_re = re.compile(r"^\s*(?:\*\*)?Table\s+(?:\d+|[IVXLC]+)\.?", re.IGNORECASE)
    heading_re = re.compile(r"^\s*#{1,6}\s+")

    def split_cols(s: str) -> list[str]:
        return [c.strip() for c in re.split(r"\t+|\s{2,}", s.strip()) if c.strip()]

    while i < len(lines):
        line = lines[i]
        st = line.strip()
        if st.startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if st == "$$":
            in_math = not in_math
            out.append(line)
            i += 1
            continue
        if in_code or in_math:
            out.append(line)
            i += 1
            continue

        if caption_re.match(st):
            out.append(line)
            j = i + 1
            probe: list[str] = []
            while j < len(lines):
                cur = lines[j]
                cur_st = cur.strip()
                if cur_st.startswith("```") or cur_st == "$$":
                    break
                if heading_re.match(cur) or cur_st.startswith("!") or cur_st.startswith("<!-- kb_page:"):
                    break
                if caption_re.match(cur_st):
                    break
                if cur_st:
                    probe.append(cur)
                if len(probe) >= 18:
                    break
                j += 1
            candidate = [ln for ln in probe if len(split_cols(ln)) >= 2]
            numeric_rows = sum(
                1
                for ln in candidate
                if len(re.findall(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|e[+-]?\d+)?", ln, flags=re.IGNORECASE)) >= 2
            )
            if len(candidate) >= 3 and numeric_rows >= 1:
                table_md = table_text_to_markdown("\n".join(candidate))
                if not table_md:
                    table_md = _table_from_numeric_pattern(candidate)
                if table_md and _is_markdown_table_sane(table_md):
                    out.append("")
                    out.extend(table_md.splitlines())
                    out.append("")
                    i = j
                    continue

            i += 1
            continue

        out.append(line)
        i += 1

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
        math_chars = re.findall(r"[=+\-*/^_{}()\[\]<>]|[\u2200-\u22ff]|[\u0370-\u03ff]", joined)
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

    def rows_to_md(rows: list[list[str]]) -> Optional[str]:
        rows = [r for r in rows if r and any(c.strip() for c in r)]
        if len(rows) < 2:
            return None
        width = max(len(r) for r in rows)
        if width < 2:
            return None
        rows = [r + [""] * (width - len(r)) for r in rows]
        # Drop columns empty across all rows.
        keep = [i for i in range(width) if any(rows[r][i].strip() for r in range(len(rows)))]
        if len(keep) < 2:
            return None
        rows = [[_escape_md_table_cell(r[i]) for i in keep] for r in rows]
        width = len(rows[0])
        header = rows[0]
        if not any(c.strip() for c in header):
            header = [f"col_{i + 1}" for i in range(width)]
        md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
        for r in rows[1:]:
            md_lines.append("| " + " | ".join(r) + " |")
        return "\n".join(md_lines)

    rows = [split_cols(ln) for ln in lines]
    rows = [r for r in rows if r]
    md_basic = rows_to_md(rows)
    if md_basic:
        return md_basic

    # Fallback: infer column boundaries from aligned whitespace runs (common in text-extracted tables).
    norm = [ln.replace("\t", "    ") for ln in lines]
    max_len = max(len(ln) for ln in norm)
    padded = [ln.ljust(max_len) for ln in norm]
    cut_votes: list[int] = []
    for ln in padded:
        for m in re.finditer(r"\s{2,}", ln):
            cut_votes.append(int((m.start() + m.end()) / 2))
    if not cut_votes:
        return None
    cut_votes.sort()
    clusters: list[list[int]] = []
    for pos in cut_votes:
        if not clusters or abs(pos - clusters[-1][-1]) > 2:
            clusters.append([pos])
        else:
            clusters[-1].append(pos)
    cuts = [int(round(sum(g) / len(g))) for g in clusters if len(g) >= max(2, len(padded) // 5)]
    cuts = sorted({c for c in cuts if 2 <= c <= max_len - 2})
    if not cuts:
        return None
    rows_aligned: list[list[str]] = []
    for ln in padded:
        start = 0
        row: list[str] = []
        for c in cuts + [max_len]:
            cell = ln[start:c].strip()
            row.append(cell)
            start = c
        if sum(1 for cell in row if cell.strip()) >= 2:
            rows_aligned.append(row)
    return rows_to_md(rows_aligned)


def _table_from_numeric_pattern(lines: list[str]) -> Optional[str]:
    """
    Recover simple text tables like:
      Method PSNR SSIM LPIPS
      Ours 33.8 0.95 0.08
    where spacing is collapsed and first column may contain words.
    """
    if len(lines) < 3:
        return None
    num_re = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|e[+-]?\d+)?$", re.IGNORECASE)
    token_rows = [re.findall(r"\S+", ln.strip()) for ln in lines if ln.strip()]
    if len(token_rows) < 3:
        return None

    num_counts = [sum(1 for t in row if num_re.fullmatch(t)) for row in token_rows]
    data_counts = [c for c in num_counts if c >= 2]
    if len(data_counts) < 2:
        return None
    # Robust central tendency without importing statistics.
    data_counts.sort()
    n_num = data_counts[len(data_counts) // 2]
    if n_num < 2:
        return None

    rows: list[list[str]] = []
    for row in token_rows:
        if len(row) < n_num + 1:
            continue
        first_num_idx = next((i for i, t in enumerate(row) if num_re.fullmatch(t)), None)
        if first_num_idx is None:
            # header row: use last n_num tokens as metric columns
            label = " ".join(row[: len(row) - n_num]).strip()
            vals = row[len(row) - n_num :]
            if (not label) or len(vals) != n_num:
                continue
            rows.append([label] + vals)
            continue
        nums = [t for t in row[first_num_idx:] if num_re.fullmatch(t)]
        if len(nums) < n_num:
            continue
        label = " ".join(row[:first_num_idx]).strip()
        if not label:
            continue
        rows.append([label] + nums[:n_num])

    if len(rows) < 3:
        return None
    width = 1 + n_num
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = [_escape_md_table_cell(c) for c in rows[0]]
    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        md_lines.append("| " + " | ".join(_escape_md_table_cell(c) for c in r) + " |")
    out = "\n".join(md_lines)
    if not _is_markdown_table_sane(out):
        return None
    return out


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
    top_heading_re = re.compile(r"^#{1,6}\s+")
    idx_line_re = re.compile(r"^\s*(\d+)\.\s+\S")
    plain_section_re = re.compile(
        r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
        r"(?:introduction|background|related work|method(?:s|ology)?|"
        r"experiment(?:s|al)?|results?|discussion|conclusion|appendix|"
        r"acknowledg(?:e)?ments?)\b",
        re.IGNORECASE,
    )

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

    def _looks_refish_line(ln: str) -> bool:
        s = _normalize_text(ln or "").strip()
        if not s:
            return False
        if _is_ref_index_line(s) or re.match(r"^\s*\[\d+\]\s+\S", s):
            return True
        years = re.findall(r"(?:19|20)\d{2}", s)
        if years and (s.count(",") >= 2) and (len(s) >= 20):
            return True
        low = s.lower()
        if ("doi" in low or "http" in low or "arxiv" in low) and len(s) >= 20:
            return True
        return False

    i = 0
    out: list[str] = []
    found = False
    ref_block: list[str] = []
    tail_after_refs: list[str] = []
    refish_hits = 0
    non_ref_run = 0

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
            title = re.sub(r"^#{1,6}\s+", "", line.strip()).strip()
            if _is_appendix_heading_text(title) or _is_numbered_heading_text(title):
                tail_after_refs = lines[i:]
                break
            # Front-references safe exit: when substantial references were already collected,
            # and a normal body section appears, stop collecting references.
            if plain_section_re.match(title):
                tail_after_refs = lines[i:]
                break
            # ignore spurious headings inside references (do not stop collection)
            i += 1
            continue
        # Non-markdown section cue (e.g. "1. Introduction") can indicate body starts here.
        if plain_section_re.match(line.strip()) and len(ref_block) >= 3:
            tail_after_refs = lines[i:]
            break
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

        # If we're deep into a references run but hit a long sequence of non-reference lines,
        # this is likely normal body content after an early references block.
        st_line = line.strip()
        is_refish = _looks_refish_line(st_line)
        if st_line and (not is_refish) and (refish_hits >= 8) and ((non_ref_run + 1) >= 8):
            tail_after_refs = lines[i:]
            break

        ref_block.append(line)
        if st_line:
            if is_refish:
                refish_hits += 1
                non_ref_run = 0
            else:
                non_ref_run += 1
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
        if re.search(r"[鍗遍埈?]", st):
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
    top_heading_re = re.compile(r"^#{1,6}\s+")

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
            title = re.sub(r"^#{1,6}\s+", "", st).strip()
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
    return _parse_numbered_heading_level(title) is not None


def _is_appendix_heading_text(title: str) -> bool:
    return _parse_appendix_heading_level(title) is not None


def _is_strict_heading_text(title: str) -> bool:
    t = _normalize_text(title or "").strip()
    if not t:
        return False
    if _is_numbered_heading_text(t) or _is_appendix_heading_text(t):
        return True
    return t.upper() in _ALLOWED_UNNUMBERED_HEADINGS


def _normalize_heading_level_from_markdown(prefix: str) -> int:
    try:
        lvl = int(max(1, min(3, len(prefix))))
    except Exception:
        lvl = 2
    return lvl


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

        # "# 8" / "# 2.1." + next line => heading line with proper level.
        m = re.fullmatch(r"#\s*(\d+(?:\.\d+)*)(?:[.):]?)\s*$", st)
        if m and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if nxt and nxt.upper() == nxt and len(nxt) >= 4:
                lvl = min(6, m.group(1).count(".") + 1)
                out.append("#" * lvl + f" {m.group(1)} {nxt}".rstrip())
                i += 2
                continue

        # "2.3" / "2.3." + next line => heading line with proper level.
        m = re.fullmatch(r"(\d+(?:\.\d+)*)(?:[.):]?)\s*$", st)
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

        numbered_level = _parse_numbered_heading_level(title)
        if numbered_level is not None:
            lvl = max(1, min(3, int(numbered_level)))
            out.append("#" * lvl + " " + title)
            continue

        appendix_level = _parse_appendix_heading_level(title)
        if appendix_level is not None:
            lvl = max(1, min(3, int(appendix_level)))
            if title.upper() in _ALLOWED_UNNUMBERED_HEADINGS:
                out.append("# " + title.upper())
            else:
                out.append("#" * lvl + " " + title)
            continue

        if title.upper() in _ALLOWED_UNNUMBERED_HEADINGS:
            out.append("# " + title.upper())
            continue

        if _is_reasonable_heading_text(title):
            lvl = _normalize_heading_level_from_markdown(m.group(1))
            if _is_common_section_heading(title):
                lvl = 1 if lvl <= 2 else 2
            out.append("#" * lvl + " " + title)
            continue

        # Demote obvious invented headings to plain text.
        out.append(title)

    return "\n".join(out)


def postprocess_markdown(md: str) -> str:
    md = _cleanup_noise_lines(md)
    md = _convert_caption_following_tabular_lines(md)
    md = _reflow_hard_wrapped_paragraphs(md)
    md = _fix_split_numbered_headings(md)
    md = _enforce_heading_policy(md)
    md = _split_inline_bullets(md)
    # Bullet splitting can create new noise lines (e.g., page headers with "閳?). Clean again.
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
    # Some extractions lose one side of \left...\right. Strip extras to avoid renderer errors.
    lft = len(re.findall(r"\\left\b", s))
    rgt = len(re.findall(r"\\right\b", s))
    if lft != rgt:
        if lft > rgt:
            for _ in range(lft - rgt):
                s2 = re.sub(r"\\left\b\s*", "", s, count=1)
                if s2 == s:
                    break
                s = s2
        else:
            for _ in range(rgt - lft):
                s2 = re.sub(r"\\right\b\s*", "", s, count=1)
                if s2 == s:
                    break
                s = s2

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

    # Last gate before writing into $$...$$: reject obviously unbalanced LaTeX.
    if not _is_balanced_latex(s):
        return None

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
    junk_codepoints = (
        0x0010,  # DLE control char
        0x0011,  # DC1 control char
        0x951F,  # 閿?
        0x9514,  # 閿?
        0x8130,  # 鑴?
        0x6F0F,  # 婕?
        0x5E90,  # 搴?
        0x5362,  # 鍗?
    )
    if any(chr(cp) in t for cp in junk_codepoints):
        return True
    # Overly fragmented operator-only lines (sums/products without bodies)
    if len(t) <= 40 and re.search(r"[\u2211\u220f\u221a]", t) and not re.search(r"[=A-Za-z0-9]", t):
        return True
    # Many lines but very low alnum density => usually broken layout
    alnum = sum(1 for ch in t if ch.isalnum())
    if len(t) > 120 and alnum / max(1, len(t)) < 0.18:
        return True
    # Broken LaTeX structure almost always fails Markdown math renderers.
    if "\\" in t:
        if not _is_balanced_latex(t):
            return True
        lft = len(re.findall(r"\\left\b", t))
        rgt = len(re.findall(r"\\right\b", t))
        if lft != rgt:
            return True
    return False


class PdfToMarkdown:
    def __init__(self, cfg: ConvertConfig):
        self.cfg = cfg
        self._client = None
        self._OpenAIClass = None
        self._thread_local = threading.local()
        self._temp_dir: Optional[Path] = None
        self._repairs_dir: Optional[Path] = None
        if cfg.llm:
            OpenAIClass = _ensure_openai_class()
            self._OpenAIClass = OpenAIClass
            self._client = OpenAIClass(api_key=cfg.llm.api_key, base_url=cfg.llm.base_url)

    def _llm_create(
        self,
        *,
        messages: list[dict],
        temperature: float,
        max_tokens: Optional[int] = None,
    ):
        if not self.cfg.llm or not self._client:
            raise RuntimeError("LLM is not configured")
        llm = self.cfg.llm
        client = self._client
        if self._OpenAIClass is not None:
            c = getattr(self._thread_local, "client", None)
            if c is None:
                try:
                    c = self._OpenAIClass(api_key=llm.api_key, base_url=llm.base_url)
                    self._thread_local.client = c
                except Exception:
                    c = self._client
            client = c or self._client
        timeout_s = max(8.0, float(getattr(llm, "timeout_s", 60.0) or 60.0))
        retries = max(0, int(getattr(llm, "max_retries", 1) or 0))
        mt = int(max_tokens if max_tokens is not None else llm.max_tokens)
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                kwargs = {
                    "model": llm.model,
                    "messages": messages,
                    "temperature": float(temperature),
                    "timeout": timeout_s,
                }
                if mt > 0:
                    kwargs["max_tokens"] = mt
                return client.chat.completions.create(**kwargs)
            except Exception as e:
                last_err = e
                if attempt >= retries:
                    break
                time.sleep(min(2.5, 0.5 * (attempt + 1)))
        raise last_err if last_err is not None else RuntimeError("LLM request failed")

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
8) BULLETS: Replace `閳ヮ晢 bullets with Markdown list items (`- `).
9) FIDELITY: Keep all non-tag text exactly once (no omissions, no paraphrase, no added facts).
10) OUTPUT: Markdown only. No commentary.

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
            try:
                resp = self._llm_create(
                    messages=[
                        {"role": "system", "content": "You are a strict robotic text-to-markdown converter. Follow tags verbatim."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=llm.temperature,
                    max_tokens=llm.max_tokens,
                )
                out_parts.append(resp.choices[0].message.content or "")
            except Exception as e:
                print(f"WARNING: LLM render fallback on page {page_number} chunk {chunk_i}: {e}")
                out_parts.append(self._fallback_convert_tags(chunk))
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
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You convert tables to Markdown. Output only the table."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=min(llm.max_tokens, 4096),
            )
        except Exception:
            return None
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
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You convert extracted math into valid LaTeX. Output LaTeX only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=min(llm.max_tokens, 2048),
            )
        except Exception:
            return None
        out = (resp.choices[0].message.content or "").strip()
        if not out:
            return None
        if not _is_balanced_latex(out):
            return None
        self._save_cached_repair(cache, kind="math", raw=raw, output=out)
        return out

    def _call_llm_repair_math_from_image(
        self,
        png_bytes: bytes,
        *,
        page_number: int,
        block_index: int,
        eq_number: Optional[str] = None,
    ) -> Optional[str]:
        """
        Vision-based math recovery: screenshot the equation region and ask a VL model for LaTeX.
        Works only when the backend supports image_url inputs.
        """
        if not self.cfg.llm or not self._client or not self.cfg.llm_repair:
            return None
        if not png_bytes:
            return None
        try:
            import hashlib

            h = hashlib.sha1(png_bytes).hexdigest()[:16]
            cache = self._repairs_dir / f"mathv_p{int(page_number):03d}_b{int(block_index):03d}_{h}.json"
            cached = self._load_cached_repair(cache)
            if cached:
                return cached
        except Exception:
            cache = None

        b64 = base64.b64encode(png_bytes).decode("ascii")
        data_url = "data:image/png;base64," + b64
        eq_hint = f"(Equation number: {eq_number})" if eq_number else ""
        prompt = (
            f"Recover the LaTeX for this equation image from a research paper PDF page {page_number} {eq_hint}.\n"
            "Return ONLY the LaTeX for the equation body.\n"
            "- No $/$$ delimiters\n"
            "- No \\begin{equation}/align environments\n"
            "- No explanations\n"
            "Be exact and faithful to the image.\n"
        )
        llm = self.cfg.llm
        if llm.request_sleep_s > 0:
            time.sleep(llm.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You are a LaTeX math expert. Return only LaTeX."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=min(getattr(llm, "max_tokens", 2048) or 2048, 900),
            )
        except Exception as e:
            try:
                debug_vision = bool(int(os.environ.get("KB_PDF_DEBUG_VISION_MATH", "0") or "0")) or bool(
                    getattr(self.cfg, "keep_debug", False)
                )
            except Exception:
                debug_vision = False
            if debug_vision:
                try:
                    mname = str(getattr(self.cfg.llm, "model", "") or "")
                except Exception:
                    mname = ""
                _progress_log(f"VISION_MATH_CALL_FAILED page={page_number} block={block_index} model={mname!r} err={e}")
            return None
        out = (resp.choices[0].message.content or "").strip()
        if out.startswith("```"):
            m = re.search(r"```(?:\\w+)?\\n(.*?)```", out, flags=re.S)
            if m:
                out = (m.group(1) or "").strip()
        if out.startswith("$$") and out.endswith("$$"):
            out = out[2:-2].strip()
        if re.search(r"\\begin\\{equation\\}|\\begin\\{align\\}", out):
            return None
        if cache and out:
            try:
                self._save_cached_repair(cache, kind="mathv", raw=f"<image:{len(png_bytes)} bytes>", output=out)
            except Exception:
                pass
        return out or None

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
            "- Preserve symbols (閳? 閳? 閳? 閳? 鍗? 娓? 閳? 閳? etc.). Do NOT translate them.\n"
            "- Only fix obvious mojibake/ligatures and spacing; do not rewrite or paraphrase.\n\n"
            + raw.rstrip()
        )
        llm = self.cfg.llm
        if llm.request_sleep_s > 0:
            time.sleep(llm.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You clean up pseudocode. Output code only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=min(llm.max_tokens, 2048),
            )
        except Exception:
            return None
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
        if not re.search(r"[=^_\\]|[\u2200-\u22ff]|[閳嚦锝傚灆閳埃鍩嗛埈鐐╁⒐閳儮澧甸埉鍫氬灳]|[\u0370-\u03ff]", t):
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
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You fix inline math. Output one Markdown paragraph only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=min(llm.max_tokens, 2048),
            )
        except Exception:
            return None
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
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You split references into a numbered Markdown list."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=min(llm.max_tokens, 4096),
            )
        except Exception:
            return None
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
            if re.match(r"^#{1,6}\s+REFERENCES\s*$", ln.strip(), flags=re.IGNORECASE):
                start = i + 1
                break
        if start is None:
            return md
        end = len(lines)
        for j in range(start, len(lines)):
            if re.match(r"^#{1,6}\s+\S", lines[j]) and not re.match(r"^#{1,6}\s+REFERENCES\s*$", lines[j].strip(), flags=re.IGNORECASE):
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

    def _code_needs_llm_polish(self, code_text: str) -> bool:
        t = code_text or ""
        if not t.strip():
            return False
        return bool(
            re.search(r"[\u2190-\u21ff]|[\u25b2-\u25ff]", t)
            or ("->" in t)
            or ("<-" in t)
            or (":=" in t)
            or re.search(r"(?mi)^\s*(while|for|if|function)\b", t)
        )

    def _should_llm_repair_math_block(self, raw_math: str, *, block: TextBlock) -> bool:
        if not (self.cfg.llm and self.cfg.llm_repair):
            return False
        if not self.cfg.llm_smart_math_repair:
            return True

        t = _normalize_text(raw_math or block.text).strip()
        if not t:
            return False
        if _math_text_looks_garbled(t):
            return True

        has_tex_cmd = bool(re.search(r"\\[A-Za-z]+", t))
        has_tex_env = bool(re.search(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", t))
        has_math_symbol = bool(
            re.search(r"(?:[=^_]|[\u2200-\u22ff]|[\u2248\u2260\u2264\u2265\u2211\u220f]|(?:->|<-|=>|<=))", t)
        )
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        line_count = len(lines)
        token_count = len(re.findall(r"\S+", t))

        if has_tex_cmd:
            if not _is_balanced_latex(t):
                return True
            # Already-healthy LaTeX is usually better kept as-is.
            return False

        if has_tex_env and (not _is_balanced_latex(t)):
            return True

        # Multi-line non-LaTeX equations are often fragmented and benefit from LLM reconstruction.
        if line_count >= 2 and has_math_symbol:
            return True

        # Dense symbolic text without TeX commands is likely a degraded equation extraction.
        symbol_count = len(re.findall(r"[=^_+\-*/()[\]{}<>]", t))
        if has_math_symbol and token_count >= 10 and symbol_count >= 4:
            return True

        return False

    def _run_llm_jobs_parallel(self, jobs: list[tuple[str, int, Callable[[], Optional[str]]]]) -> list[Optional[str]]:
        if not jobs:
            return []
        worker_cap = max(1, int(self.cfg.llm_workers))
        if worker_cap <= 1 or len(jobs) <= 1:
            out_seq: list[Optional[str]] = []
            for _, _, fn in jobs:
                try:
                    out_seq.append(fn())
                except Exception:
                    out_seq.append(None)
            return out_seq

        max_workers = min(worker_cap, len(jobs))
        out: list[Optional[str]] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fut_to_idx = {executor.submit(fn): i for i, (_, _, fn) in enumerate(jobs)}
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                try:
                    out[i] = fut.result()
                except Exception:
                    out[i] = None
        return out

    def _estimate_llm_repair_job_counts(
        self,
        *,
        blocks: list[TextBlock],
    ) -> tuple[int, int, int, int]:
        """
        Estimate how many block-level LLM repair calls this page would trigger.
        Returns (table_n, math_n, code_n, body_n).
        """
        if not (self.cfg.llm and self.cfg.llm_repair):
            return 0, 0, 0, 0
        table_n = 0
        math_n = 0
        code_n = 0
        body_n = 0
        for b in blocks:
            if b.is_table:
                table_md = (b.table_markdown or "").strip()
                if table_md and _is_markdown_table_sane(table_md):
                    continue
                table_n += 1
                continue
            if b.is_math:
                if re.search(r"\\begin\{(?:array|tabular)\}", b.text):
                    continue
                raw_math = "\n".join([ln.rstrip() for ln in b.text.splitlines() if ln.strip()])
                if self._should_llm_repair_math_block(raw_math, block=b):
                    math_n += 1
                continue
            if b.is_code and self._code_needs_llm_polish(b.text):
                code_n += 1
                continue
            if self.cfg.llm_repair_body_math and (not b.heading_level):
                body_n += 1
        return table_n, math_n, code_n, body_n

    def _should_auto_use_page_llm(
        self,
        *,
        blocks: list[TextBlock],
    ) -> tuple[bool, str]:
        """
        On very noisy pages, block-level LLM repairs can explode into many requests and become slow.
        Switch to page-level LLM rendering when estimated block-level calls exceed a threshold.
        """
        if not self.cfg.llm:
            return False, ""
        if self.cfg.llm_render_page:
            return False, ""
        threshold = max(0, int(self.cfg.llm_auto_page_render_threshold))
        if threshold <= 0:
            return False, ""
        table_n, math_n, code_n, body_n = self._estimate_llm_repair_job_counts(blocks=blocks)
        total_n = int(table_n + math_n + code_n + body_n)
        if total_n >= threshold:
            return True, (
                f"repair_jobs={total_n} (table={table_n}, math={math_n}, code={code_n}, body={body_n}) "
                f">= threshold={threshold}"
            )
        return False, ""

    def _prefetch_llm_block_repairs(
        self,
        *,
        blocks: list[TextBlock],
        page_number: int,
        eqno_by_block: Optional[dict[int, str]] = None,
    ) -> tuple[dict[int, str], dict[int, str], dict[int, str], dict[int, str]]:
        table_by_block: dict[int, str] = {}
        math_by_block: dict[int, str] = {}
        code_by_block: dict[int, str] = {}
        body_by_block: dict[int, str] = {}
        if not (self.cfg.llm and self.cfg.llm_repair):
            return table_by_block, math_by_block, code_by_block, body_by_block

        eqno_by_block = eqno_by_block or {}

        def _ctx_before(i: int) -> str:
            for pj in range(i - 1, -1, -1):
                pb = blocks[pj]
                if pb.is_table or pb.is_math or pb.is_code:
                    continue
                t = pb.text.strip()
                if len(t) >= 20:
                    return t[-400:]
            return ""

        def _ctx_after(i: int) -> str:
            for nj in range(i + 1, len(blocks)):
                nb = blocks[nj]
                if nb.is_table or nb.is_math or nb.is_code:
                    continue
                t = nb.text.strip()
                if len(t) >= 20:
                    return t[:400]
            return ""

        jobs: list[tuple[str, int, Callable[[], Optional[str]]]] = []
        for bi, b in enumerate(blocks):
            if b.is_table:
                table_md = (b.table_markdown or "").strip() or table_text_to_markdown(b.text)
                if table_md and _is_markdown_table_sane(table_md):
                    continue
                raw = b.text
                jobs.append(
                    (
                        "table",
                        bi,
                        lambda raw=raw, bi=bi: self._call_llm_repair_table(raw, page_number=page_number, block_index=bi),
                    )
                )
                continue

            if b.is_math:
                if re.search(r"\\begin\{(?:array|tabular)\}", b.text):
                    # This path is handled inline and may become a table.
                    continue
                raw_math = "\n".join([ln.rstrip() for ln in b.text.splitlines() if ln.strip()])
                if not self._should_llm_repair_math_block(raw_math, block=b):
                    continue
                eqno = eqno_by_block.get(bi)
                ctx_before = _ctx_before(bi)
                ctx_after = _ctx_after(bi)
                jobs.append(
                    (
                        "math",
                        bi,
                        lambda raw_math=raw_math, fallback_text=b.text, bi=bi, ctx_before=ctx_before, ctx_after=ctx_after, eqno=eqno: self._call_llm_repair_math(
                            raw_math or fallback_text,
                            page_number=page_number,
                            block_index=bi,
                            context_before=ctx_before,
                            context_after=ctx_after,
                            eq_number=eqno,
                        ),
                    )
                )
                continue

            if b.is_code and self._code_needs_llm_polish(b.text):
                raw_code = b.text
                jobs.append(
                    (
                        "code",
                        bi,
                        lambda raw_code=raw_code, bi=bi: self._call_llm_polish_code(raw_code, page_number=page_number, block_index=bi),
                    )
                )
                continue

            if self.cfg.llm_repair_body_math and (not b.heading_level):
                para = b.text.strip()
                jobs.append(
                    (
                        "body",
                        bi,
                        lambda para=para, bi=bi: self._call_llm_repair_body_paragraph(para, page_number=page_number, block_index=bi),
                    )
                )

        if not jobs:
            return table_by_block, math_by_block, code_by_block, body_by_block

        results = self._run_llm_jobs_parallel(jobs)
        for (kind, bi, _), out in zip(jobs, results):
            if not out:
                continue
            if kind == "table":
                table_by_block[bi] = out
            elif kind == "math":
                math_by_block[bi] = out
            elif kind == "code":
                code_by_block[bi] = out
            elif kind == "body":
                body_by_block[bi] = out
        return table_by_block, math_by_block, code_by_block, body_by_block

    def _render_blocks_to_markdown(
        self,
        blocks: list[TextBlock],
        figs_by_block: dict[int, str],
        *,
        page=None,
        page_number: int,
        eqno_by_block: Optional[dict[int, str]] = None,
        eqimg_by_block: Optional[dict[int, str]] = None,
    ) -> str:
        out: list[str] = []
        eqno_by_block = eqno_by_block or {}
        eqimg_by_block = eqimg_by_block or {}
        llm_table_by_block, llm_math_by_block, llm_code_by_block, llm_body_by_block = self._prefetch_llm_block_repairs(
            blocks=blocks,
            page_number=page_number,
            eqno_by_block=eqno_by_block,
        )

        def _vision_math_enabled() -> bool:
            try:
                if (not self.cfg.llm) or (not self._client) or (not self.cfg.llm_repair):
                    return False
                if page is None:
                    return False
                raw = str(os.environ.get("KB_PDF_LLM_VISION_MATH", "") or "").strip().lower()
                if raw:
                    return raw in {"1", "true", "yes", "y", "on"}
                m = str(getattr(self.cfg.llm, "model", "") or "").strip().lower()
                return ("vl" in m) or ("vision" in m)
            except Exception:
                return False

        def _debug_vision_math() -> bool:
            try:
                return bool(int(os.environ.get("KB_PDF_DEBUG_VISION_MATH", "0") or "0")) or bool(
                    getattr(self.cfg, "keep_debug", False)
                )
            except Exception:
                return False

        def _looks_like_broken_display_math(raw_math: str, latex: str) -> bool:
            rm = (raw_math or "").strip()
            lt = (latex or "").strip()
            if not rm or not lt:
                return False
            if ("=" in rm) and ("=" not in lt):
                return True
            if len(rm) >= 60 and len(lt) <= max(24, int(len(rm) * 0.45)):
                return True
            if any(x in rm for x in ["∑", "Σ", "\\sum", "∫", "\\int", "||", "‖"]) and not any(
                y in lt for y in ["\\sum", "\\int", "\\left\\|", "\\|", "\\lVert", "\\rVert"]
            ):
                return True
            return False

        def _screenshot_eq_png(bbox: tuple[float, float, float, float]) -> bytes:
            if fitz is None or page is None:
                return b""
            try:
                rect = fitz.Rect(bbox)
            except Exception:
                return b""
            if rect.width <= 2 or rect.height <= 2:
                return b""
            # Pad more vertically to include equation number.
            pad_x = max(2.0, float(rect.width) * 0.06)
            pad_y = max(2.0, float(rect.height) * 0.22)
            try:
                crop = fitz.Rect(
                    max(0.0, float(rect.x0) - pad_x),
                    max(0.0, float(rect.y0) - pad_y),
                    min(float(page.rect.width), float(rect.x1) + pad_x),
                    min(float(page.rect.height), float(rect.y1) + pad_y),
                )
            except Exception:
                crop = rect
            try:
                pix = _render_clip_pixmap(
                    page,
                    crop,
                    base_scale=float(getattr(self.cfg, "image_scale", 2.2) or 2.2),
                    image_alpha=bool(getattr(self.cfg, "image_alpha", False)),
                    min_scale=1.8,
                    max_scale=5.2,
                    min_long_edge_px=1600,
                    min_short_edge_px=520,
                    image_regions=None,
                )
                return pix.tobytes("png")
            except Exception:
                return b""

        def _ctx_before(i: int) -> str:
            for pj in range(i - 1, -1, -1):
                pb = blocks[pj]
                if pb.is_table or pb.is_math or pb.is_code:
                    continue
                t = pb.text.strip()
                if len(t) >= 20:
                    return t[-400:]
            return ""

        def _ctx_after(i: int) -> str:
            for nj in range(i + 1, len(blocks)):
                nb = blocks[nj]
                if nb.is_table or nb.is_math or nb.is_code:
                    continue
                t = nb.text.strip()
                if len(t) >= 20:
                    return t[:400]
            return ""

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

        caption_table_re = re.compile(r"^\s*Table\s+(?:\d+|[IVXLC]+)\b", re.IGNORECASE)
        fig_caption_re = re.compile(r"^\s*(?:Fig\.|Figure)\s*\d+", re.IGNORECASE)
        num_re = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|e[+-]?\d+)?", re.IGNORECASE)
        auto_table_by_caption: dict[int, tuple[str, set[int]]] = {}
        auto_skip_indices: set[int] = set()
        for i, b in enumerate(blocks):
            if i in auto_skip_indices:
                continue
            if b.is_table or b.is_math or b.is_code or b.heading_level:
                continue
            t = _normalize_text(b.text).strip()
            if not caption_table_re.match(t):
                continue
            rows: list[str] = []
            consumed: list[int] = []
            j = i + 1
            while j < len(blocks) and len(consumed) < 12:
                nb = blocks[j]
                if nb.is_table or nb.is_math or nb.is_code or nb.heading_level:
                    break
                nt = _normalize_text(nb.text).strip()
                if not nt:
                    j += 1
                    continue
                if caption_table_re.match(nt) or fig_caption_re.match(nt):
                    break
                if len(nt) > 140:
                    break
                tokens = re.findall(r"\S+", nt)
                nums = num_re.findall(nt)
                looks_row = (len(nums) >= 2) or (2 <= len(tokens) <= 10 and (not nt.endswith(".")))
                if looks_row:
                    rows.append(nt)
                    consumed.append(j)
                    j += 1
                    continue
                if rows:
                    break
                j += 1

            if len(rows) < 3:
                continue
            table_md = table_text_to_markdown("\n".join(rows))
            if not table_md:
                table_md = _table_from_numeric_pattern(rows)
            if not table_md or (not _is_markdown_table_sane(table_md)):
                continue
            auto_table_by_caption[i] = (table_md, set(consumed))
            auto_skip_indices.update(consumed)

        for bi, b in enumerate(blocks):
            if bi in auto_skip_indices:
                continue
            img = figs_by_block.get(bi)
            if img:
                out.append(f"![Figure](./assets/{img})")
                # If this block is a caption, render it right after the image.
                if re.match(r"^\s*(?:Fig\.|FIG\.|Figure|FIGURE)\s*[0-9]+", b.text.strip(), re.IGNORECASE):
                    out.append(caption_to_md(b.text))
                    out.append("")
                    continue
                out.append("")

            if bi in auto_table_by_caption:
                table_md, _ = auto_table_by_caption[bi]
                out.append(_normalize_text(b.text).strip())
                out.append("")
                out.extend(table_md.splitlines())
                out.append("")
                continue

            if b.heading_level and 1 <= int(b.heading_level) <= 3 and _is_reasonable_heading_text(b.text):
                title = _normalize_text(b.text).strip()
                lvl = int(b.heading_level)
                numbered_level = _parse_numbered_heading_level(title)
                appendix_level = _parse_appendix_heading_level(title)
                if numbered_level is not None:
                    lvl = int(numbered_level)
                elif appendix_level is not None:
                    lvl = int(appendix_level)
                lvl = max(1, min(3, lvl))
                if title.upper() in _ALLOWED_UNNUMBERED_HEADINGS:
                    lvl = 1
                    title = title.upper()
                out.append("#" * lvl + " " + title)
                out.append("")
                continue

            if b.is_table:
                table_md = (b.table_markdown or "").strip() or None
                if not table_md:
                    table_md = table_text_to_markdown(b.text)
                if not table_md:
                    table_md = llm_table_by_block.get(bi)
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
                # Keep equation image as a late fallback (after LaTeX repair/normalization attempt).
                eq_img = eqimg_by_block.get(bi)
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
                latex: Optional[str] = llm_math_by_block.get(bi)
                if (not latex) and self.cfg.llm and self.cfg.llm_repair:
                    # Smart mode only falls back to LLM when local normalization fails.
                    needs_late_call = (not self.cfg.llm_smart_math_repair)
                    if self.cfg.llm_smart_math_repair:
                        needs_late_call = _normalize_display_latex(raw_math, eq_number=eqno) is None
                    if needs_late_call:
                        latex = self._call_llm_repair_math(
                            raw_math or b.text,
                            page_number=page_number,
                            block_index=bi,
                            context_before=_ctx_before(bi),
                            context_after=_ctx_after(bi),
                            eq_number=eqno,
                        )
                if not latex and raw_math and _is_balanced_latex(raw_math) and re.search(r"\\[A-Za-z]+", raw_math):
                    latex = raw_math
                latex_norm = _normalize_display_latex(latex if latex is not None else raw_math, eq_number=eqno)

                # VL vision retry for display-like equations when text LaTeX is missing or looks broken.
                if _vision_math_enabled():
                    try:
                        ms = (raw_math or "").strip()
                        sym_n = len(re.findall(r"[=+\\-*/^_{}\\\\\\[\\]]|[∈≤≥≈×·Σ∑∫∞→←⇔⇒]", ms))
                        complex_tok = bool(
                            re.search(r"[∑Σ∫∞≤≥≈≠→←⇔⇒√]|\\\\(?:frac|sqrt|sum|int|prod|log|exp|left|right|begin)\\b", ms)
                        )
                        is_wide = False
                        try:
                            rr = fitz.Rect(b.bbox)
                            is_wide = float(rr.width) >= float(page.rect.width) * 0.55
                        except Exception:
                            is_wide = False
                        displayish = (
                            ("\n" in ms)
                            or ("=" in ms)
                            or (len(ms) >= 60)
                            or complex_tok
                            or (sym_n >= 10 and len(ms) >= 25)
                            or (is_wide and len(ms) >= 18)
                        )
                        should_try = displayish and ((not latex_norm) or _looks_like_broken_display_math(ms, latex_norm or ""))
                        if _debug_vision_math() and not should_try:
                            why = []
                            if not displayish:
                                why.append(f"not_displayish(sym_n={sym_n},wide={int(is_wide)},len={len(ms)})")
                            if latex_norm and (not _looks_like_broken_display_math(ms, latex_norm)):
                                why.append("latex_not_suspicious")
                            _progress_log(f"VISION_MATH_SKIP page={page_number} block={bi} reason={','.join(why) or 'unknown'}")

                        if should_try:
                            png = _screenshot_eq_png(b.bbox)
                            if png:
                                if _debug_vision_math():
                                    _progress_log(f"VISION_MATH_CALL page={page_number} block={bi} src_len={len(ms)}")
                                vlatex = self._call_llm_repair_math_from_image(
                                    png,
                                    page_number=page_number,
                                    block_index=bi,
                                    eq_number=eqno,
                                )
                                vnorm = _normalize_display_latex(vlatex or "", eq_number=eqno) if vlatex else None
                                if vnorm:
                                    _progress_log(f"VISION_MATH_OK page={page_number} block={bi} eq={eqno or ''}")
                                    latex_norm = vnorm
                    except Exception as e:
                        if _debug_vision_math():
                            _progress_log(f"VISION_MATH_ERROR page={page_number} block={bi} err={e}")

                if latex_norm:
                    out.append("$$")
                    out.append(latex_norm)
                    out.append("$$")
                else:
                    if eq_img:
                        out.append(f"![Equation](./assets/{eq_img})")
                    else:
                        # Avoid emitting broken LaTeX that crashes renderers.
                        out.extend([ln.rstrip() for ln in b.text.splitlines() if ln.strip()])
                out.append("")
                continue

            if b.is_code:
                code_text = b.text
                polished = llm_code_by_block.get(bi)
                if not polished and self._code_needs_llm_polish(code_text):
                    polished = self._call_llm_polish_code(code_text, page_number=page_number, block_index=bi)
                if polished:
                    code_text = polished
                out.append("```")
                out.extend([ln.rstrip() for ln in code_text.splitlines()])
                out.append("```")
                out.append("")
                continue

            para = b.text.strip()
            repaired = llm_body_by_block.get(bi)
            if repaired is None:
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
                    txt = txt[:800] + "..."
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
     NOT a heading if the title starts with \"(\" or looks like an equation (contains \"=\", \"^\", \"_\", \"\\\\\", 鍗? 閳? etc.).
   - Appendix letters: ^[A-Z](?:\\.\\d+)*\\s+<LETTER> (examples: \"A DETAILS...\", \"B.1 ...\").
     NOT a heading if it looks like an equation.
   - The literal word \"APPENDIX\" (as a standalone heading).
2) Drop boilerplate/noise: journal headers/footers, page numbers, navigation like \"Latest updates\", \"RESEARCH-ARTICLE\",
   download/citation stats, copyright blocks, publisher notices.
3) Table vs code vs math:
   - table: rows/columns or matrices of numbers; DO NOT label tables as code.
   - math: mostly equations/symbols; output plain text lines (we will wrap with $$ later).
   - code: ONLY pseudocode/algorithms (keywords like while/for/if/function, arrows like 閳? indentation).
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
            try:
                resp = self._llm_create(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": make_prompt(items)},
                    ],
                    temperature=0.0,
                    max_tokens=llm.max_tokens,
                )
            except Exception:
                return None
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
        try:
            resp = self._llm_create(
                messages=[{"role": "system", "content": "Translator mode."}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=llm.max_tokens,
            )
        except Exception:
            return md
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

    def _process_page_without_llm_with_open_doc(
        self,
        *,
        doc,
        pdf_path: Path,
        page_index: int,
        body_size: float,
        noise_texts: set[str],
        in_references: bool,
        assets_dir: Path,
        temp_dir: Path,
    ) -> tuple[int, str, bool]:
        pnum = page_index + 1
        raw_out = temp_dir / f"p{pnum:03d}.tagged.txt"
        en_out = temp_dir / f"p{pnum:03d}.en.md"

        page = doc[page_index]
        image_rects = _collect_image_rects(page)
        visual_rects = _collect_visual_rects(page, image_rects=image_rects)
        blocks = extract_text_blocks(
            page,
            body_size=body_size,
            noise_texts=noise_texts,
            page_index=page_index,
            pdf_path=pdf_path,
            image_rects=image_rects,
            visual_rects=visual_rects,
            relax_small_text_filter=bool(in_references),
            preserve_body_linebreaks=bool(in_references),
            detect_tables=bool(self.cfg.detect_tables),
            table_pdfplumber_fallback=bool(self.cfg.table_pdfplumber_fallback),
        )
        blocks = sort_blocks_reading_order(blocks, page_width=float(page.rect.width))

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
                            table_markdown=b.table_markdown,
                            is_math=b.is_math,
                            heading_level=lvl,
                        )
                    )
                    continue
            headed.append(b)
        blocks = promote_heading_blocks(
            headed,
            body_size=float(body_size),
            page_width=float(page.rect.width),
            page_index=page_index,
        )

        blocks = merge_math_blocks_by_proximity(blocks, body_size=float(body_size), page_width=float(page.rect.width))
        blocks = merge_adjacent_math_blocks(
            blocks,
            max_vgap=max(12.0, float(body_size) * 1.6),
            body_size=float(body_size),
            min_x_overlap=0.15,
        )
        blocks = sort_blocks_reading_order(blocks, page_width=float(page.rect.width))

        warned_garbled_math = any(b.is_math and _math_text_looks_garbled(b.text) for b in blocks)

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
            visual_rects=visual_rects,
            image_scale=self.cfg.image_scale,
            image_alpha=self.cfg.image_alpha,
        )
        extra_imgs = extract_images_fallback(
            page,
            blocks,
            assets_dir,
            page_index,
            covered_rects=covered,
            visual_rects=visual_rects,
            image_scale=self.cfg.image_scale,
            image_alpha=self.cfg.image_alpha,
        )
        for k, v in extra_imgs.items():
            figs.setdefault(k, v)

        eq_imgs: dict[int, str] = {}
        need_eq_image_fallback = bool(self.cfg.eq_image_fallback)
        if need_eq_image_fallback:
            eq_imgs = extract_equation_images_for_garbled_math(
                page,
                blocks,
                assets_dir,
                page_index,
                eqno_by_block=eqno_by_block,
                image_scale=self.cfg.image_scale,
                image_alpha=self.cfg.image_alpha,
            )

        tagged = ""
        if self.cfg.keep_debug:
            tagged = _tagged_page_text(
                page_index=page_index,
                page=page,
                blocks=blocks,
                body_size=body_size,
                figures_by_block=figs,
                eqno_by_block=eqno_by_block,
            )
            raw_out.write_text(tagged, encoding="utf-8")

        en_md = self._render_blocks_to_markdown(
            blocks,
            figs,
            page=page,
            page_number=pnum,
            eqno_by_block=eqno_by_block,
            eqimg_by_block=eq_imgs,
        )

        for img_name in figs.values():
            if f"./assets/{img_name}" not in en_md and f"assets/{img_name}" not in en_md:
                en_md = en_md.rstrip() + f"\n\n![Figure](./assets/{img_name})\n"
        for img_name in eq_imgs.values():
            if f"./assets/{img_name}" not in en_md and f"assets/{img_name}" not in en_md:
                en_md = en_md.rstrip() + f"\n\n![Equation](./assets/{img_name})\n"

        if self.cfg.keep_debug:
            en_out.write_text(en_md, encoding="utf-8")

        return pnum, en_md, bool(warned_garbled_math)

    def _process_page_batch_without_llm(
        self,
        *,
        pdf_path: Path,
        page_indices: list[int],
        total_pages: int,
        body_size: float,
        noise_texts: set[str],
        refs_mode_by_page: list[bool],
        assets_dir: Path,
        temp_dir: Path,
    ) -> list[tuple[int, str, bool]]:
        if not page_indices:
            return []
        out: list[tuple[int, str, bool]] = []
        with fitz.open(pdf_path) as d:
            for pi in page_indices:
                pnum = pi + 1
                _progress_log(f"Processing page {pnum}/{total_pages} ...")
                out.append(
                    self._process_page_without_llm_with_open_doc(
                        doc=d,
                        pdf_path=pdf_path,
                        page_index=pi,
                        body_size=body_size,
                        noise_texts=noise_texts,
                        in_references=refs_mode_by_page[pi],
                        assets_dir=assets_dir,
                        temp_dir=temp_dir,
                    )
                )
                _progress_log(f"Finished page {pnum}/{total_pages}")
        return out

    def _process_page_with_llm_with_open_doc(
        self,
        *,
        doc,
        pdf_path: Path,
        page_index: int,
        total_pages: int,
        body_size: float,
        noise_texts: set[str],
        refs_mode_by_page: list[bool],
        assets_dir: Path,
        temp_dir: Path,
    ) -> tuple[int, str, Optional[str], bool]:
        pnum = page_index + 1
        raw_out = temp_dir / f"p{pnum:03d}.tagged.txt"
        en_out = temp_dir / f"p{pnum:03d}.en.md"
        zh_out = temp_dir / f"p{pnum:03d}.zh.md"
        cls_out = temp_dir / f"p{pnum:03d}.cls.json"

        if self.cfg.skip_existing and en_out.exists() and (not self.cfg.translate_zh or zh_out.exists()):
            en_md0 = en_out.read_text(encoding="utf-8", errors="replace")
            if "<!-- kb_page:" not in en_md0[:120]:
                en_md0 = f"<!-- kb_page: {pnum} -->\n\n" + en_md0.lstrip()
            zh_md0: Optional[str] = None
            if self.cfg.translate_zh:
                z0 = zh_out.read_text(encoding="utf-8", errors="replace")
                if "<!-- kb_page:" not in z0[:120]:
                    z0 = f"<!-- kb_page: {pnum} -->\n\n" + z0.lstrip()
                zh_md0 = z0
            return pnum, en_md0, zh_md0, False

        page = doc[page_index]
        image_rects = _collect_image_rects(page)
        visual_rects = _collect_visual_rects(page, image_rects=image_rects)
        in_references = bool(refs_mode_by_page[page_index]) if (0 <= page_index < len(refs_mode_by_page)) else False
        blocks = extract_text_blocks(
            page,
            body_size=body_size,
            noise_texts=noise_texts,
            page_index=page_index,
            pdf_path=pdf_path,
            image_rects=image_rects,
            visual_rects=visual_rects,
            relax_small_text_filter=bool(in_references),
            preserve_body_linebreaks=bool(in_references),
            detect_tables=bool(self.cfg.detect_tables),
            table_pdfplumber_fallback=bool(self.cfg.table_pdfplumber_fallback),
        )
        blocks = sort_blocks_reading_order(blocks, page_width=float(page.rect.width))

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
                    numbered_level = _parse_numbered_heading_level(text)
                    appendix_level = _parse_appendix_heading_level(text)
                    if numbered_level is not None:
                        hl = max(1, min(3, int(numbered_level)))
                    elif appendix_level is not None:
                        hl = max(1, min(3, int(appendix_level)))
                    else:
                        t_upper = _normalize_text(text).strip().upper()
                        if t_upper in _ALLOWED_UNNUMBERED_HEADINGS:
                            hl = 1
                        else:
                            try:
                                hl = int(heading_level) if heading_level is not None else None
                            except Exception:
                                hl = None
                            if hl is not None:
                                hl = max(1, min(3, int(hl)))
                updated.append(
                    TextBlock(
                        bbox=b.bbox,
                        text=text,
                        max_font_size=b.max_font_size,
                        is_bold=b.is_bold,
                        insert_image=b.insert_image,
                        is_code=is_code,
                        is_table=is_table,
                        table_markdown=(b.table_markdown if is_table else None),
                        is_math=is_math,
                        heading_level=hl,
                    )
                )
            blocks = updated

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
                            table_markdown=b.table_markdown,
                            is_math=b.is_math,
                            heading_level=lvl,
                        )
                    )
                    continue
            headed.append(b)
        blocks = promote_heading_blocks(
            headed,
            body_size=float(body_size),
            page_width=float(page.rect.width),
            page_index=page_index,
        )
        blocks = merge_math_blocks_by_proximity(blocks, body_size=float(body_size), page_width=float(page.rect.width))
        blocks = merge_adjacent_math_blocks(
            blocks,
            max_vgap=max(12.0, float(body_size) * 1.6),
            body_size=float(body_size),
            min_x_overlap=0.15,
        )
        blocks = sort_blocks_reading_order(blocks, page_width=float(page.rect.width))

        warned_garbled_math = False
        if (self.cfg.llm is None) and (not self.cfg.eq_image_fallback):
            warned_garbled_math = any(b.is_math and _math_text_looks_garbled(b.text) for b in blocks)

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
            visual_rects=visual_rects,
            image_scale=self.cfg.image_scale,
            image_alpha=self.cfg.image_alpha,
        )
        extra_imgs = extract_images_fallback(
            page,
            blocks,
            assets_dir,
            page_index,
            covered_rects=covered,
            visual_rects=visual_rects,
            image_scale=self.cfg.image_scale,
            image_alpha=self.cfg.image_alpha,
        )
        for k, v in extra_imgs.items():
            figs.setdefault(k, v)

        eq_imgs: dict[int, str] = {}
        if bool(self.cfg.eq_image_fallback):
            eq_imgs = extract_equation_images_for_garbled_math(
                page,
                blocks,
                assets_dir,
                page_index,
                eqno_by_block=eqno_by_block,
                image_scale=self.cfg.image_scale,
                image_alpha=self.cfg.image_alpha,
            )

        self._temp_dir = temp_dir
        self._repairs_dir = temp_dir / "repairs"
        if self.cfg.llm and self.cfg.llm_repair:
            self._repairs_dir.mkdir(parents=True, exist_ok=True)

        auto_page_llm, auto_page_llm_reason = self._should_auto_use_page_llm(blocks=blocks)
        use_page_llm = bool(self.cfg.llm_render_page or auto_page_llm)
        if auto_page_llm:
            _progress_log(f"Page {pnum}: auto page-LLM enabled ({auto_page_llm_reason})")

        tagged = ""
        if self.cfg.keep_debug or use_page_llm:
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

        if use_page_llm:
            en_md = self._call_llm_convert(tagged, pnum)
        else:
            en_md = self._render_blocks_to_markdown(
                blocks,
                figs,
                page=page,
                page_number=pnum,
                eqno_by_block=eqno_by_block,
                eqimg_by_block=eq_imgs,
            )

        for img_name in figs.values():
            if f"./assets/{img_name}" not in en_md and f"assets/{img_name}" not in en_md:
                en_md = en_md.rstrip() + f"\n\n![Figure](./assets/{img_name})\n"
        for img_name in eq_imgs.values():
            if f"./assets/{img_name}" not in en_md and f"assets/{img_name}" not in en_md:
                en_md = en_md.rstrip() + f"\n\n![Equation](./assets/{img_name})\n"

        en_body = en_md
        if self.cfg.keep_debug:
            en_out.write_text(en_body, encoding="utf-8")
        en_md = f"<!-- kb_page: {pnum} -->\n\n" + en_body.lstrip()

        zh_md: Optional[str] = None
        if self.cfg.translate_zh:
            zh_body = self._call_llm_translate_zh(en_body)
            zh_md = zh_body
            if "<!-- kb_page:" not in zh_md[:120]:
                zh_md = f"<!-- kb_page: {pnum} -->\n\n" + zh_md.lstrip()
            if self.cfg.keep_debug:
                zh_out.write_text(zh_md, encoding="utf-8")

        return pnum, en_md, zh_md, bool(warned_garbled_math)

    def _process_page_batch_with_llm(
        self,
        *,
        pdf_path: Path,
        page_indices: list[int],
        total_pages: int,
        body_size: float,
        noise_texts: set[str],
        refs_mode_by_page: list[bool],
        assets_dir: Path,
        temp_dir: Path,
    ) -> list[tuple[int, str, Optional[str], bool]]:
        if not page_indices:
            return []
        out: list[tuple[int, str, Optional[str], bool]] = []
        with fitz.open(pdf_path) as d:
            for pi in page_indices:
                pnum = pi + 1
                _progress_log(f"Processing page {pnum}/{total_pages} ...")
                out.append(
                    self._process_page_with_llm_with_open_doc(
                        doc=d,
                        pdf_path=pdf_path,
                        page_index=pi,
                        total_pages=total_pages,
                        body_size=body_size,
                        noise_texts=noise_texts,
                        refs_mode_by_page=refs_mode_by_page,
                        assets_dir=assets_dir,
                        temp_dir=temp_dir,
                    )
                )
                _progress_log(f"Finished page {pnum}/{total_pages}")
        return out

    def _convert_parallel_without_llm(
        self,
        *,
        doc,
        pdf_path: Path,
        paper_name: str,
        save_dir: Path,
        assets_dir: Path,
        temp_dir: Path,
        body_size: float,
        total_pages: int,
        start: int,
        end: int,
        noise_texts: set[str],
    ) -> None:
        refs_mode_by_page = _build_refs_mode_by_page(doc)

        en_by_index: dict[int, str] = {}
        todo_pages: list[int] = []
        for page_index in range(start, end):
            pnum = page_index + 1
            en_out = temp_dir / f"p{pnum:03d}.en.md"
            zh_out = temp_dir / f"p{pnum:03d}.zh.md"
            if self.cfg.skip_existing and en_out.exists() and (not self.cfg.translate_zh or zh_out.exists()):
                en_md0 = en_out.read_text(encoding="utf-8", errors="replace")
                if "<!-- kb_page:" not in en_md0[:120]:
                    en_md0 = f"<!-- kb_page: {pnum} -->\n\n" + en_md0.lstrip()
                en_by_index[page_index] = en_md0
            else:
                todo_pages.append(page_index)

        workers = max(1, min(int(self.cfg.workers), max(1, len(todo_pages))))
        print(f"Parallel page workers: {workers}")
        chunks: list[list[int]] = [[] for _ in range(workers)]
        for i, pi in enumerate(sorted(todo_pages)):
            chunks[i % workers].append(pi)
        chunks = [c for c in chunks if c]

        warned_once = False
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._process_page_batch_without_llm,
                    pdf_path=pdf_path,
                    page_indices=chunk,
                    total_pages=total_pages,
                    body_size=body_size,
                    noise_texts=noise_texts,
                    refs_mode_by_page=refs_mode_by_page,
                    assets_dir=assets_dir,
                    temp_dir=temp_dir,
                ): tuple(chunk)
                for chunk in chunks
            }
            for fut in as_completed(futures):
                batch = fut.result()
                for pnum, en_md, warned in batch:
                    if warned and not warned_once:
                        print(
                            "WARNING: Detected garbled math in PDF extraction. "
                            "Use --eq-image-fallback for strict visual math fallback."
                        )
                        warned_once = True
                    pi = pnum - 1
                    en_by_index[pi] = f"<!-- kb_page: {pnum} -->\n\n" + en_md.lstrip()

        en_pages = [en_by_index[i] for i in range(start, end) if i in en_by_index]
        en_full = postprocess_markdown("\n\n".join(en_pages))
        pdf_refs = _extract_references_from_pdf(doc)
        if pdf_refs:
            en_full = _inject_references_section(en_full, pdf_refs)
        else:
            en_full = _format_references(en_full)
        (save_dir / f"{paper_name}.en.md").write_text(en_full, encoding="utf-8")

        try:
            pngs = sorted(p.name for p in assets_dir.glob("*.png"))
            if pngs:
                manifest = "\n".join([f"- ![asset](./assets/{n})" for n in pngs]) + "\n"
                (save_dir / "assets_manifest.md").write_text(manifest, encoding="utf-8")
        except Exception:
            pass

    def convert(self) -> None:
        if fitz is None:
            raise SystemExit("Missing dependency `PyMuPDF` (import name: `fitz`). Install it, then retry.")
        pdf_path = self.cfg.pdf_path
        paper_name = pdf_path.stem
        save_dir_ui = self.cfg.out_dir / paper_name
        save_dir = _as_fs_path(save_dir_ui)
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

            can_fast_no_llm = (self.cfg.llm is None) and (not self.cfg.translate_zh) and (not self.cfg.llm_render_page)
            if can_fast_no_llm:
                self._convert_parallel_without_llm(
                    doc=doc,
                    pdf_path=pdf_path,
                    paper_name=paper_name,
                    save_dir=save_dir,
                    assets_dir=assets_dir,
                    temp_dir=temp_dir,
                    body_size=float(body_size),
                    total_pages=total_pages,
                    start=start,
                    end=end,
                    noise_texts=noise_texts,
                )
                print(f"Done. Output: {save_dir_ui}")
                return
            refs_mode_by_page = _build_refs_mode_by_page(doc)

            en_by_index: dict[int, str] = {}
            zh_by_index: dict[int, str] = {}
            todo_pages: list[int] = []
            for page_index in range(start, end):
                pnum = page_index + 1
                en_out = temp_dir / f"p{pnum:03d}.en.md"
                zh_out = temp_dir / f"p{pnum:03d}.zh.md"
                if self.cfg.skip_existing and en_out.exists() and (not self.cfg.translate_zh or zh_out.exists()):
                    en_md0 = en_out.read_text(encoding="utf-8", errors="replace")
                    if "<!-- kb_page:" not in en_md0[:120]:
                        en_md0 = f"<!-- kb_page: {pnum} -->\n\n" + en_md0.lstrip()
                    en_by_index[page_index] = en_md0
                    if self.cfg.translate_zh:
                        zh_md0 = zh_out.read_text(encoding="utf-8", errors="replace")
                        if "<!-- kb_page:" not in zh_md0[:120]:
                            zh_md0 = f"<!-- kb_page: {pnum} -->\n\n" + zh_md0.lstrip()
                        zh_by_index[page_index] = zh_md0
                else:
                    todo_pages.append(page_index)

            workers = max(1, min(int(self.cfg.workers), max(1, len(todo_pages))))
            print(f"Parallel page workers: {workers}")
            if self.cfg.llm:
                print(
                    f"LLM concurrency: page_workers={workers}, llm_workers={int(self.cfg.llm_workers)}, "
                    f"max_inflight~{workers * max(1, int(self.cfg.llm_workers))}"
                )
            warned_once = False
            if workers > 1 and todo_pages:
                chunks: list[list[int]] = [[] for _ in range(workers)]
                for i, pi in enumerate(sorted(todo_pages)):
                    chunks[i % workers].append(pi)
                chunks = [c for c in chunks if c]
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            self._process_page_batch_with_llm,
                            pdf_path=pdf_path,
                            page_indices=chunk,
                            total_pages=total_pages,
                            body_size=float(body_size),
                            noise_texts=noise_texts,
                            refs_mode_by_page=refs_mode_by_page,
                            assets_dir=assets_dir,
                            temp_dir=temp_dir,
                        ): tuple(chunk)
                        for chunk in chunks
                    }
                    for fut in as_completed(futures):
                        batch = fut.result()
                        for pnum, en_md, zh_md, warned in batch:
                            if warned and not warned_once:
                                print(
                                    "WARNING: Detected garbled math in PDF extraction. "
                                    "To get correct LaTeX, enable LLM (DeepSeek) or pass --eq-image-fallback as a last resort."
                                )
                                warned_once = True
                            pi = pnum - 1
                            en_by_index[pi] = en_md
                            if zh_md is not None:
                                zh_by_index[pi] = zh_md
            else:
                for page_index in todo_pages:
                    pnum = page_index + 1
                    print(f"Processing page {pnum}/{total_pages} ...")
                    pnum, en_md, zh_md, warned = self._process_page_with_llm_with_open_doc(
                        doc=doc,
                        pdf_path=pdf_path,
                        page_index=page_index,
                        total_pages=total_pages,
                        body_size=float(body_size),
                        noise_texts=noise_texts,
                        refs_mode_by_page=refs_mode_by_page,
                        assets_dir=assets_dir,
                        temp_dir=temp_dir,
                    )
                    if warned and not warned_once:
                        print(
                            "WARNING: Detected garbled math in PDF extraction. "
                            "To get correct LaTeX, enable LLM (DeepSeek) or pass --eq-image-fallback as a last resort."
                        )
                        warned_once = True
                    en_by_index[page_index] = en_md
                    if zh_md is not None:
                        zh_by_index[page_index] = zh_md

            en_pages = [en_by_index[i] for i in range(start, end) if i in en_by_index]
            zh_pages = [zh_by_index[i] for i in range(start, end) if i in zh_by_index]
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

        print(f"Done. Output: {save_dir_ui}")


def _parse_args(argv: Optional[list[str]] = None) -> ConvertConfig:
    ap = argparse.ArgumentParser(description="Convert a research PDF into (high-fidelity) Markdown with assets.")
    ap.add_argument(
        "--profile",
        type=str,
        default="locked",
        choices=["locked", "custom"],
        help="Conversion profile: locked (recommended, enforces stable high-quality defaults) or custom (respect current env/flags).",
    )
    ap.add_argument("--pdf", required=True, help="Input PDF path")
    ap.add_argument("--out", required=True, help="Output folder (paper-stem subfolder will be created)")
    ap.add_argument("--translate-zh", action="store_true", help="Also output Chinese Markdown")
    ap.add_argument("--start-page", type=int, default=0, help="0-based start page (default 0)")
    ap.add_argument("--end-page", type=int, default=-1, help="0-based end page (exclusive); -1 means end")
    ap.add_argument("--skip-existing", action="store_true", help="Skip pages if temp outputs already exist")
    ap.add_argument("--keep-debug", action="store_true", help="Write per-page tagged + md into temp/")

    # OpenAI-compatible endpoints.
    # If QWEN_API_KEY is set, default to Qwen's compatible-mode endpoint and a vision-capable model.
    if os.environ.get("QWEN_API_KEY"):
        default_base_url = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        default_model = os.environ.get("QWEN_MODEL", os.environ.get("OPENAI_MODEL", "qwen3-vl-plus"))
        default_key_env = os.environ.get("QWEN_API_KEY_ENV", "QWEN_API_KEY")
    else:
        # DeepSeek's OpenAI-compatible endpoint uses the /v1 prefix.
        default_base_url = os.environ.get("DEEPSEEK_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1"))
        default_model = os.environ.get("DEEPSEEK_MODEL", os.environ.get("OPENAI_MODEL", "deepseek-chat"))
        default_key_env = os.environ.get("DEEPSEEK_API_KEY_ENV", "DEEPSEEK_API_KEY")

    ap.add_argument("--base-url", default=default_base_url)
    ap.add_argument("--model", default=default_model)
    ap.add_argument(
        "--api-key-env",
        default=default_key_env,
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
    ap.add_argument(
        "--auto-page-llm-threshold",
        type=int,
        default=int(os.environ.get("KB_PDF_AUTO_PAGE_LLM_THRESHOLD", "12")),
        help="Auto-switch to page-level LLM if estimated block-level LLM repair calls on a page exceed this threshold (0=disable)",
    )
    ap.add_argument("--llm-workers", type=int, default=0, help="Concurrent LLM repair requests per page (0=auto)")
    ap.add_argument(
        "--no-llm-smart-math-repair",
        action="store_true",
        help="Disable smart gating and call LLM for most math blocks (slower, sometimes higher recall)",
    )
    ap.add_argument("--classify-batch-size", type=int, default=40, help="LLM classify blocks batch size (higher=fewer calls)")
    ap.add_argument("--classify-always", action="store_true", help="Always classify every page with LLM (slower)")
    ap.add_argument("--speed-mode", type=str, default="normal", choices=["normal", "ultra_fast", "no_llm", "full_llm", "balanced", "fast"], help="Speed mode: normal (vision-direct, max parallelism), ultra_fast (faster, lower quality), no_llm (basic text extraction), full_llm (legacy), balanced (legacy), fast (legacy)")
    ap.add_argument("--image-scale", type=float, default=2.2, help="Image render scale for figure/equation crops (lower=faster)")
    ap.add_argument("--image-alpha", action="store_true", help="Render images with alpha channel (slower)")
    ap.add_argument("--no-table-detect", action="store_true", help="Disable structural table detection from PDF layout")
    ap.add_argument("--table-pdfplumber-fallback", action="store_true", help="Use pdfplumber as slow fallback for hard table pages")
    ap.add_argument(
        "--eq-image-fallback",
        action="store_true",
        help="Fallback: render garbled display-math as equation images when LaTeX is unreliable",
    )
    ap.add_argument("--no-global-noise-scan", action="store_true", help="Skip global header/footer scan (faster, less clean)")
    ap.add_argument("--fast", action="store_true", help="Speed-first mode: lighter image scale and disable expensive fallbacks")
    ap.add_argument("--workers", type=int, default=0, help="Page worker threads for no-LLM mode (0=auto)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between LLM requests")
    ap.add_argument(
        "--llm-timeout",
        type=float,
        default=float(os.environ.get("KB_PDF_LLM_TIMEOUT_S", os.environ.get("DEEPSEEK_TIMEOUT_S", "120"))),
        help="Per-request LLM timeout seconds",
    )
    ap.add_argument("--llm-retries", type=int, default=int(os.environ.get("DEEPSEEK_RETRIES", "0")), help="Retries for each LLM request")
    args = ap.parse_args(argv)

    # Stable profile that freezes key guardrails to avoid quality regressions
    # caused by ad-hoc environment toggles.
    if str(getattr(args, "profile", "locked") or "locked").strip().lower() == "locked":
        locked_env = {
            "KB_PDF_LEGACY_EXTRA_CLEANUP": "0",
            "KB_PDF_VISION_MATH_QUALITY_GATE": "1",
            "KB_PDF_VISION_EMPTY_RETRY": "3",
            "KB_PDF_VISION_EMPTY_RETRY_BACKOFF_S": "1.5",
            "KB_PDF_VISION_REFS_COLUMN_MODE": "1",
            "KB_PDF_VISION_FRAGMENT_FALLBACK": "0",
            "KB_PDF_VISION_FORMULA_OVERLAY": "0",
        }
        for k, v in locked_env.items():
            os.environ[k] = v

        # Keep quality-first route deterministic under locked profile.
        if str(args.speed_mode).lower() != "normal":
            print(
                f"[PROFILE] locked: forcing --speed-mode normal (ignore {args.speed_mode!r})",
                flush=True,
            )
            args.speed_mode = "normal"

        # Timeout guardrail to reduce random mid-run aborts on heavy pages.
        try:
            args.llm_timeout = max(120.0, float(args.llm_timeout))
        except Exception:
            args.llm_timeout = 120.0

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    llm: Optional[LlmConfig]
    # If speed_mode is "no_llm", force no_llm to True
    if args.speed_mode == "no_llm":
        args.no_llm = True
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
            timeout_s=float(args.llm_timeout),
            max_retries=max(0, int(args.llm_retries)),
        )

    raw_workers = int(args.workers)
    workers = raw_workers
    llm_enabled = (not bool(args.no_llm))
    if workers <= 0:
        cpu = os.cpu_count() or 4
        if llm_enabled:
            # LLM mode: too many page workers often causes provider-side throttling and stalls.
            workers = max(1, min(3, cpu // 2 if cpu >= 4 else 1))
        else:
            workers = max(1, min(10, cpu - 1 if cpu > 2 else cpu))

    raw_llm_workers = int(args.llm_workers)
    llm_workers = raw_llm_workers
    if llm_workers <= 0:
        cpu = os.cpu_count() or 4
        # Keep conservative default to avoid API-side throttling.
        llm_workers = max(1, min(3, cpu // 2 if cpu >= 4 else 1))
    if args.no_llm or args.no_llm_repair:
        llm_workers = 1
    # If both values are auto-picked, cap aggregate LLM in-flight requests.
    # This stabilizes throughput under provider QPS limits.
    if llm_enabled and raw_workers <= 0 and raw_llm_workers <= 0:
        max_inflight = 6
        if workers * llm_workers > max_inflight:
            llm_workers = max(1, max_inflight // max(1, workers))

    image_scale = float(args.image_scale)
    table_pdfplumber_fallback = bool(args.table_pdfplumber_fallback)
    global_noise_scan = (not bool(args.no_global_noise_scan))
    if bool(args.fast):
        if abs(image_scale - 2.2) < 1e-6:
            image_scale = 1.8
        global_noise_scan = False
        table_pdfplumber_fallback = False
        if workers < 2 and (os.cpu_count() or 1) > 2:
            workers = min(10, max(2, (os.cpu_count() or 4) - 1))
        if (not args.no_llm) and (not args.no_llm_repair) and llm_workers < 2:
            llm_workers = 2

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
        image_scale=float(image_scale),
        image_alpha=bool(args.image_alpha),
        detect_tables=(not bool(args.no_table_detect)),
        table_pdfplumber_fallback=bool(table_pdfplumber_fallback),
        eq_image_fallback=bool(args.eq_image_fallback),
        global_noise_scan=bool(global_noise_scan),
        llm_repair=(not bool(args.no_llm_repair)),
        # Keep off by default to avoid long stalls; users can opt-in via --llm-repair-body-math.
        llm_repair_body_math=bool(args.llm_repair_body_math) and (not bool(args.no_llm_repair_body_math)),
        llm_smart_math_repair=(not bool(args.no_llm_smart_math_repair)),
        llm_auto_page_render_threshold=max(0, int(args.auto_page_llm_threshold)),
        llm_workers=max(1, int(llm_workers)),
        workers=max(1, int(workers)),
        speed_mode=str(args.speed_mode) if hasattr(args, 'speed_mode') else 'balanced',
    )


def main() -> None:
    """Main entry point - uses test_converter.py logic directly."""
    cfg = _parse_args()
    
    # DIRECTLY USE test_converter.py LOGIC - proven to work
    import sys
    import os
    from pathlib import Path
    
    # Add project root to path (exactly like test_converter.py)
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    print("=" * 80, flush=True)
    print("USING test_converter.py LOGIC DIRECTLY", flush=True)
    print(f"Project root: {project_root}", flush=True)
    print("=" * 80, flush=True)
    
    try:
        # Import exactly like test_converter.py
        from kb.converter import PDFConverter, ConvertConfig
        from dataclasses import replace
        
        print("Successfully imported PDFConverter from kb.converter", flush=True)
        
        # Get parameters from cfg (parsed from command line)
        pdf_path = Path(cfg.pdf_path)
        out_dir = Path(cfg.out_dir)
        
        if not pdf_path.exists():
            print(f"Error: PDF file not found: {pdf_path}", flush=True)
            raise SystemExit(1)
        
        # Create output directory (paper_name subdirectory)
        paper_name = pdf_path.stem
        output_dir = out_dir / paper_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"PDF: {pdf_path}", flush=True)
        print(f"Output dir: {output_dir}", flush=True)
        
        # Get speed mode
        speed_mode = getattr(cfg, 'speed_mode', 'balanced')
        print(f"Speed mode: {speed_mode}", flush=True)
        try:
            locked_keys = [
                "KB_PDF_LEGACY_EXTRA_CLEANUP",
                "KB_PDF_VISION_MATH_QUALITY_GATE",
                "KB_PDF_VISION_EMPTY_RETRY",
                "KB_PDF_VISION_EMPTY_RETRY_BACKOFF_S",
                "KB_PDF_VISION_REFS_COLUMN_MODE",
                "KB_PDF_VISION_FRAGMENT_FALLBACK",
                "KB_PDF_VISION_FORMULA_OVERLAY",
            ]
            kv = " ".join([f"{k}={os.environ.get(k, '')}" for k in locked_keys])
            print(f"[PROFILE] active guardrails: {kv}", flush=True)
        except Exception:
            pass

        # IMPORTANT: keep parsed cfg fields (workers/llm_workers, llm_repair flags, batch sizes, etc).
        # Previously we rebuilt ConvertConfig with only a subset of fields, silently dropping those knobs.
        print("Creating ConvertConfig (preserving parsed args)...", flush=True)
        new_cfg = replace(cfg, pdf_path=pdf_path, out_dir=output_dir, speed_mode=str(speed_mode))
        # Type-guard for older modules / hot-reloads where cfg isn't a dataclass instance.
        if not isinstance(new_cfg, ConvertConfig):
            new_cfg = ConvertConfig(
                pdf_path=pdf_path,
                out_dir=output_dir,
                translate_zh=bool(getattr(cfg, "translate_zh", False)),
                start_page=int(getattr(cfg, "start_page", 0) or 0),
                end_page=int(getattr(cfg, "end_page", -1) or -1),
                skip_existing=bool(getattr(cfg, "skip_existing", False)),
                keep_debug=bool(getattr(cfg, "keep_debug", False)),
                llm=getattr(cfg, "llm", None),
                llm_classify=bool(getattr(cfg, "llm_classify", True)),
                llm_render_page=bool(getattr(cfg, "llm_render_page", False)),
                llm_classify_only_if_needed=bool(getattr(cfg, "llm_classify_only_if_needed", True)),
                classify_batch_size=int(getattr(cfg, "classify_batch_size", 40) or 40),
                image_scale=float(getattr(cfg, "image_scale", 2.2) or 2.2),
                image_alpha=bool(getattr(cfg, "image_alpha", False)),
                detect_tables=bool(getattr(cfg, "detect_tables", True)),
                table_pdfplumber_fallback=bool(getattr(cfg, "table_pdfplumber_fallback", False)),
                eq_image_fallback=bool(getattr(cfg, "eq_image_fallback", False)),
                global_noise_scan=bool(getattr(cfg, "global_noise_scan", True)),
                llm_repair=bool(getattr(cfg, "llm_repair", True)),
                llm_repair_body_math=bool(getattr(cfg, "llm_repair_body_math", False)),
                llm_smart_math_repair=bool(getattr(cfg, "llm_smart_math_repair", True)),
                llm_auto_page_render_threshold=int(getattr(cfg, "llm_auto_page_render_threshold", 12) or 12),
                llm_workers=int(getattr(cfg, "llm_workers", 1) or 1),
                workers=int(getattr(cfg, "workers", 1) or 1),
                speed_mode=str(speed_mode),
            )

        # Create converter
        print("Creating PDFConverter...", flush=True)
        converter = PDFConverter(new_cfg)
        converter.dpi = 200
        converter.analyze_quality = True
        print("PDFConverter created (dpi=200, analyze_quality=True)", flush=True)
        
        # Convert EXACTLY like test_converter.py
        print(f"Converting {pdf_path.name}...", flush=True)
        print("=" * 80, flush=True)
        
        converter.convert(str(pdf_path), str(output_dir))
        
        print("=" * 80, flush=True)
        print(f"\n[OK] Conversion completed!", flush=True)
        print(f"  Output: {output_dir / 'output.md'}", flush=True)
        
        # Keep conversion success even on long Windows paths.
        # output.md is already sufficient for UI discovery (_resolve_md_output_paths falls back to any *.md).
        output_md = output_dir / "output.md"
        try:
            output_exists = bool(_as_fs_path(output_md).exists())
        except Exception:
            output_exists = output_md.exists()
        if output_exists:
            final_md = output_dir / f"{paper_name}.en.md"
            src_p = _as_fs_path(output_md)
            dst_p = _as_fs_path(final_md)
            moved = False
            try:
                if dst_p.exists():
                    dst_p.unlink()
            except Exception:
                pass
            try:
                src_p.replace(dst_p)
                moved = True
                print(f"  Renamed to: {final_md}", flush=True)
            except Exception as e:
                # Fallback: copy then keep source as-is.
                try:
                    import shutil
                    shutil.copyfile(str(src_p), str(dst_p))
                    moved = True
                    print(f"  Rename failed ({e}); copied to: {final_md}", flush=True)
                except Exception as e2:
                    print(f"  Skip rename/copy to .en.md ({e2}); keep output.md", flush=True)
            if moved:
                # Best effort: remove output.md when canonical file is ready.
                try:
                    if src_p.exists():
                        src_p.unlink()
                except Exception:
                    pass
        
        if (output_dir / "quality_report.md").exists():
            print(f"  Quality report: {output_dir / 'quality_report.md'}", flush=True)
        
    except ImportError as e:
        print(f"CRITICAL ERROR: Cannot import converter ({e})", flush=True)
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
    except Exception as e:
        print(f"ERROR during conversion: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
