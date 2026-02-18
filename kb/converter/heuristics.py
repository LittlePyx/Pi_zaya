from __future__ import annotations

import re
from typing import Optional
from collections import Counter
try:
    import fitz
except ImportError:
    fitz = None

from .text_utils import _normalize_text, _is_letter, _join_lines_preserving_words
from .geometry_utils import _bbox_width


_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*)(?:[.):]|\s)\s*(?P<rest>.+)$"
)
_APPENDIX_HEADING_RE = re.compile(
    r"^(?P<letter>[A-Z])(?P<suffix>(?:\.\d+)*)\s+(?P<rest>.+)$"
)
_ROMAN_HEADING_RE = re.compile(
    r"^(?P<roman>[IVXLC]+)\.\s+(?P<rest>.+)$",
    re.IGNORECASE
)
_LETTER_HEADING_RE = re.compile(
    r"^(?P<letter>[A-Z])\.\s+(?P<rest>.+)$"
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
    # Import here to avoid circular dependency
    from .block_classifier import _looks_like_math_block
    if _looks_like_equation_text(t) or _looks_like_math_block([t]):
        return False
    
    # Additional checks: exclude math-like patterns that look like formulas
    # Patterns like "M 6 n 1", "6 n ,", "2 s + k" are likely formulas, not headings
    if re.match(r'^[A-Z]\s*\d+\s*[a-z]', t):  # "M 6 n"
        return False
    if re.match(r'^\d+\s*[a-z]\s*[,.]?\s*$', t):  # "6 n ,"
        return False
    if re.match(r'^\d+\s*[a-z]+\s*[+\-]\s*[a-z]', t):  # "2 s + k"
        return False
    # Very short text with numbers and letters is likely a formula
    if len(t) <= 25 and re.search(r'\d+.*[a-z]|[a-z].*\d+', t) and not re.search(r'[A-Z]{2,}', t):
        # But allow if it's a common heading word
        if t.upper() not in _COMMON_SECTION_HEADINGS:
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
    if not t:
        return None
    
    # STRICT CHECK: If it looks like a formula, don't classify as heading
    from .block_classifier import _looks_like_math_block
    if _looks_like_equation_text(t) or _looks_like_math_block([t]):
        return None
    
    # Additional formula pattern checks (must be before other checks)
    if re.match(r'^[A-Z]\s*\d+\s*[a-z]', t):  # "M 6 n"
        return None
    if re.match(r'^\d+\s*[a-z]\s*[,.]?\s*$', t):  # "6 n ,"
        return None
    if re.match(r'^\d+\s*[a-z]+\s*[+\-]', t):  # "2 s + k"
        return None
    # Very short text with numbers and letters is likely a formula
    if len(t) <= 25 and re.search(r'\d+.*[a-z]|[a-z].*\d+', t) and not re.search(r'[A-Z]{2,}', t):
        # But allow if it's a common heading word
        if t.upper() not in _COMMON_SECTION_HEADINGS:
            return None
    
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


_AFFILIATION_KEYWORDS = {
    "university",
    "institute",
    "department",
    "school",
    "college",
    "faculty",
    "laboratory",
    "lab",
    "academy",
    "hospital",
    "centre",
    "center",
    "ministry",
    "state key",
}

_JOURNAL_FOOTER_KEYWORDS = {
    "vol.",
    "volume",
    "issue",
    "issn",
    "eissn",
    "all rights reserved",
    "open access",
    "creative commons",
    "published by",
    "copyright",
}


def _is_non_body_metadata_text(
    text: str,
    *,
    page_index: int = 0,
    y0: float = 0.0,
    y1: float = 0.0,
    page_height: float = 0.0,
    max_font_size: float = 0.0,
    body_font_size: float = 0.0,
    is_references_page: bool = False,
) -> bool:
    """
    Detect non-body metadata blocks that should not be emitted as main article text:
    - author affiliations/contact blocks
    - journal header/footer/copyright boilerplate
    - DOI/ORCID/contact snippets outside references
    """
    t = _normalize_text(text or "").strip()
    if not t:
        return False
    low = t.lower()

    if is_references_page and re.match(r"^\[\d{1,4}\]\s+", t):
        return False
    if _is_caption_like_text(t):
        return False
    # Markdown/ocr variants of figure captions: *Figure 5..., **Figure 5..., [Figure 5]...
    if re.match(r"^\s*[*_`>#\[\(]*\s*(?:fig\.?|figure|table|algorithm)\s*(?:\d+|[ivxlc]+)\b", t, flags=re.IGNORECASE):
        return False
    if ("reproduced with permission" in low) and (("figure" in low) or ("fig." in low)):
        return False
    word_n = len(re.findall(r"[A-Za-z]{2,}", t))
    edge_zone = False
    if page_height > 0:
        edge_zone = (y1 <= page_height * 0.18) or (y0 >= page_height * 0.82)

    has_email = bool(re.search(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", t))
    has_orcid = "orcid" in low
    has_url = bool(re.search(r"https?://|www\.[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", low))
    has_doi = bool(re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", t)) or ("doi:" in low) or ("doi.org/" in low)
    has_page_counter = bool(re.search(r"\(\s*\d+\s+of\s+\d+\s*\)", low))

    if has_email or has_orcid:
        return True
    if has_page_counter:
        return True
    if has_doi and (edge_zone or word_n <= 28):
        return True
    if has_url and edge_zone:
        return True
    if re.search(r"\b(?:all rights reserved|creative commons|open access|copyright|published by)\b", low):
        return True
    if re.search(r"\b(?:received|accepted|published online|article history)\b", low) and (edge_zone or page_index <= 1):
        return True

    aff_n = sum(1 for k in _AFFILIATION_KEYWORDS if k in low)
    journal_n = sum(1 for k in _JOURNAL_FOOTER_KEYWORDS if k in low)
    zip_n = len(re.findall(r"\b\d{5,6}\b", t))
    country_n = len(
        re.findall(
            r"\b(?:china|usa|united states|uk|england|france|germany|canada|australia|japan|korea|singapore|india)\b",
            low,
        )
    )
    name_pair_n = len(re.findall(r"\b[A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20}\b", t))
    initial_name_n = len(re.findall(r"\b(?:[A-Z]\.\s*){1,3}[A-Z][a-z]{1,24}\b", t))
    verb_n = len(
        re.findall(
            r"\b(?:is|are|was|were|be|been|being|have|has|had|can|may|will|would|should|could|do|does|did|show|shows|shown|propose|proposed|present|presents|discuss|discusses|demonstrate|improve|use|used)\b",
            low,
        )
    )

    if aff_n >= 3 and word_n <= 130 and verb_n <= 5:
        return True
    if (page_index <= 1 or edge_zone) and aff_n >= 2 and (zip_n + country_n >= 1 or name_pair_n + initial_name_n >= 2):
        return True
    if edge_zone and journal_n >= 2 and word_n <= 90:
        return True
    if aff_n >= 2 and has_url:
        return True

    # Running header/footer in smaller font near page edge.
    if edge_zone and body_font_size > 0 and max_font_size > 0:
        if max_font_size <= body_font_size * 0.92 and (has_url or has_doi or journal_n >= 1):
            return True

    # Extremely long metadata merges (common OCR/LLM collapse case).
    if len(t) >= 260 and (aff_n >= 2 or has_url or has_doi) and verb_n <= 6:
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
    """
    heading_pat = re.compile(
        r"^\s*(?:REFERENCES|BIBLIOGRAPHY|REFERENCES\s+AND\s+NOTES|LITERATURE\s+CITED)\s*$",
        flags=re.IGNORECASE,
    )
    # 1) Fast path: plain text contains an isolated line.
    try:
        t = page.get_text("text") or ""
        if any(heading_pat.match((ln or "").strip()) for ln in t.splitlines()):
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
                if heading_pat.fullmatch(line):
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

    lines = [ln.strip() for ln in t.splitlines() if (ln or "").strip()]
    if len(lines) < 10:
        return False
    # Many references: lots of years, commas, and reference-like lead markers.
    years = re.findall(r"(?:19|20)\d{2}", t)
    if len(years) < 18:
        return False
    comma_lines = sum(1 for ln in lines if "," in ln and len(ln) >= 25)
    if comma_lines < 10:
        return False

    ref_line_pat = re.compile(r"^\s*(?:\[\s*\d{1,4}\s*\]|\d{1,3}\.\s+[A-Z])")
    ref_like_lines = [i for i, ln in enumerate(lines) if ref_line_pat.match(ln)]
    if len(ref_like_lines) < max(6, int(len(lines) * 0.18)):
        # Dense references can still be wrapped strangely; keep a permissive fallback below.
        if len(years) < 28:
            return False

    # Avoid classifying mixed body+references pages as full references pages.
    first_ref = min(ref_like_lines) if ref_like_lines else len(lines)
    if first_ref >= 6:
        bodyish_before = 0
        for ln in lines[:first_ref]:
            words = re.findall(r"[A-Za-z]{3,}", ln)
            stop_n = len(
                re.findall(
                    r"\b(?:the|and|with|from|that|this|these|into|through|between|while|where|which|during|using)\b",
                    ln,
                    flags=re.IGNORECASE,
                )
            )
            if len(words) >= 8 and stop_n >= 2 and not ref_line_pat.match(ln):
                bodyish_before += 1
        if bodyish_before >= 5:
            return False

    if ("doi" in t.lower()) or ("http" in t.lower()) or ("arxiv" in t.lower()):
        return True
    # Fallback: pure density of years + many reference-like lines.
    return len(years) >= 28 and len(ref_like_lines) >= 8


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
