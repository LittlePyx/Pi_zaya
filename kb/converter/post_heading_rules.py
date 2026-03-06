from __future__ import annotations

import re
from typing import Optional

from .text_utils import _normalize_text

_ALLOWED_UNNUMBERED_HEADINGS = {
    "ABSTRACT",
    "ACKNOWLEDGMENTS",
    "ACKNOWLEDGEMENT",
    "REFERENCES",
    "BIBLIOGRAPHY",
    "APPENDIX",
    "APPENDICES",
    "INTRODUCTION",
    "CONCLUSION",
    "CONCLUSIONS",
    "RELATED WORK",
    "METHOD",
    "METHODS",
    "RESULTS",
    "DISCUSSION",
    "SUPPLEMENTARY MATERIAL",
    "SUPPLEMENTAL MATERIAL",
    "SUPPLEMENTARY",
    "SUPPLEMENTAL",
}

_JOURNAL_METADATA_HEADINGS = {
    "REVIEW ARTICLE",
    "NATURAL PHOTONICS",
    "NATURE PHOTONICS",
}


def _is_reasonable_heading_text(t: str) -> bool:
    if len(t) < 2:
        return False
    # If it's very long, probably not a heading
    if len(t) > 200:
        return False
    # If it contains many math symbols, surely not a heading
    if re.search(r"[=\\^_]", t):
        return False
    return True


def _parse_numbered_heading_level(text: str) -> Optional[int]:
    """
    Detect '1. Introduction', '2.1 Method', 'A. Appendix' etc.
    """
    t = text.lstrip()
    m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.*)", t)
    if not m:
        return None
    nums = m.group(1).split(".")
    nums = [x for x in nums if x]
    return min(4, max(1, len(nums)))


def _parse_appendix_heading_level(text: str) -> Optional[int]:
    t = text.lstrip()
    if re.match(r"^Appendix\s+[A-Z](?:\.\d+)*", t, re.IGNORECASE):
        return 1
    m = re.match(r"^([A-Z])(?:\.(\d+))*\.?\s+(.*)", t)
    if m:
        count = 0
        if m.group(1):
            count += 1
        if m.group(2):
            count += 1
        return min(4, max(1, count))
    return None


def _is_caption_heading_text(title: str) -> bool:
    t = _normalize_text(title or "").strip()
    if not t:
        return False
    if re.match(r"^(?:figure|fig\.?|table|algorithm)\s*(?:\d+|[ivxlc]+)\b", t, re.IGNORECASE):
        return True
    if re.match(r"^(?:figure|fig\.?|table|algorithm)\s+\d+\s+caption\b", t, re.IGNORECASE):
        return True
    return False


def _is_journal_metadata_heading(title: str) -> bool:
    t = _normalize_text(title or "").strip()
    if not t:
        return False
    t = re.sub(r"\s+", " ", t).strip().upper()
    if t in _JOURNAL_METADATA_HEADINGS:
        return True
    if re.fullmatch(r"(?:NATURE|SCIENCE)\s+[A-Z][A-Z\s\-]{2,40}", t):
        return True
    if re.fullmatch(r"[A-Z][A-Z\s&\-]{5,40}", t) and ("ARTICLE" in t):
        if t not in _ALLOWED_UNNUMBERED_HEADINGS:
            return True
    return False


def _is_common_section_heading(title: str) -> bool:
    t = _normalize_text(title).upper()
    return any(x in t for x in _ALLOWED_UNNUMBERED_HEADINGS)


def _enforce_heading_policy(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []

    headings: list[tuple[int, str]] = []
    for line in lines:
        m = re.match(r"^(#+)\s+(.*)", line)
        if not m:
            continue
        headings.append((len(m.group(1)), (m.group(2) or "").strip()))

    def _is_numbered_or_structural(t: str) -> bool:
        if _parse_numbered_heading_level(t) is not None:
            return True
        if _parse_appendix_heading_level(t) is not None:
            return True
        if re.match(r"^[IVX]+\.\s+", t, re.IGNORECASE):
            return True
        if _is_common_section_heading(t):
            return True
        if _is_caption_heading_text(t):
            return True
        return False

    def _heading_key(t: str) -> str:
        k = _normalize_text(t or "").strip().lower()
        k = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", k)
        k = re.sub(r"^[ivx]+\.\s+", "", k, flags=re.IGNORECASE)
        k = re.sub(r"^[a-z]\.\s+", "", k, flags=re.IGNORECASE)
        k = re.sub(r"[^a-z0-9]+", " ", k)
        k = re.sub(r"\s+", " ", k).strip()
        return k

    has_explicit_title = False
    numbered_heading_count = 0
    for _, ht in headings:
        if _parse_numbered_heading_level(ht) is not None:
            numbered_heading_count += 1
    numbered_style_doc = numbered_heading_count >= 2
    if headings:
        first_title = headings[0][1]
        if first_title and (not _is_numbered_or_structural(first_title)):
            has_explicit_title = True

    seen_h1 = False
    seen_numbered = False
    seen_structural_keys: set[str] = set()
    for line in lines:
        if _is_journal_metadata_heading(line.strip()):
            continue
        m = re.match(r"^(#+)\s+(.*)", line)
        if not m:
            out.append(line)
            continue

        lvl = len(m.group(1))
        title = (m.group(2) or "").strip()
        if not title:
            out.append(line)
            continue

        if _is_caption_heading_text(title):
            out.append(title)
            continue
        if _is_journal_metadata_heading(title):
            continue

        authorish = False
        if ("@" in title) or ("†" in title):
            authorish = True
        if (len(re.findall(r"\d", title)) >= 2) and (title.count(",") >= 2):
            authorish = True
        if re.search(
            r"\((?:incomplete\s+visible|partially\s+visible|not\s+fully\s+visible)\)|\b(?:unreadable|illegible)\b",
            title,
            flags=re.IGNORECASE,
        ):
            authorish = True
        if re.match(r"^\d{1,3}\.?\s+(?:[A-Z]\.\s*){1,6}(?:[A-Z][A-Za-z'\-]+\.?)?(?:\s*\(|\s*,|$)", title):
            authorish = True
        cap_words = re.findall(r"\b[A-Z][a-z]{1,}\b", title)
        numbered_like = bool(
            _parse_numbered_heading_level(title) is not None
            or re.match(r"^[IVX]+\.\s+", title, re.IGNORECASE)
            or _parse_appendix_heading_level(title) is not None
        )
        if (len(cap_words) >= 4) and (len(re.findall(r"\d", title)) >= 2) and (not numbered_like):
            authorish = True
        if authorish:
            out.append(title)
            continue

        title_key = _heading_key(title)
        if numbered_like:
            seen_numbered = True
            if title_key:
                seen_structural_keys.add(title_key)
        elif _is_common_section_heading(title):
            if title_key:
                seen_structural_keys.add(title_key)
        else:
            if numbered_style_doc and seen_numbered:
                overlap = False
                if title_key:
                    for k0 in seen_structural_keys:
                        if title_key == k0:
                            overlap = True
                            break
                        if len(title_key) >= 8 and len(k0) >= 8 and (title_key in k0 or k0 in title_key):
                            overlap = True
                            break
                if overlap:
                    continue
                out.append(title)
                continue

        desired = _parse_numbered_heading_level(title)
        if desired is None:
            if re.match(r"^[IVX]+\.\s+", title, re.IGNORECASE):
                desired = 2
            else:
                desired = _parse_appendix_heading_level(title)
        if desired is not None:
            if has_explicit_title:
                desired = desired + 1
            lvl = int(min(6, max(1, desired)))
        elif _is_common_section_heading(title):
            lvl = 2 if has_explicit_title else 1

        if re.fullmatch(r"[\d\.]+", title):
            out.append(title)
            continue

        if len(title) > 150:
            out.append(title)
            continue

        if lvl == 1:
            if seen_h1:
                lvl = 2
            else:
                seen_h1 = True

        out.append("#" * lvl + " " + title)
    return "\n".join(out)

