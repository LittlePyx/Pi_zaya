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
    # Filter empty strings from '1.' -> ['1', '']
    nums = [x for x in nums if x]
    return min(4, max(1, len(nums)))

def _parse_appendix_heading_level(text: str) -> Optional[int]:
    t = text.lstrip()
    # "Appendix A", "Appendix A.1"
    if re.match(r"^Appendix\s+[A-Z](?:\.\d+)*", t, re.IGNORECASE):
        # We can treat this as level 1 usually
        return 1
    # "A. Proof of..."
    m = re.match(r"^([A-Z])(?:\.(\d+))*\.?\s+(.*)", t)
    if m:
        # A -> 1, A.1 -> 2
        groups = [g for g in m.groups() if g is not None]
        # groups[0] is letter, groups[1] is number...
        # It's heuristic.
        count = 0
        if m.group(1): count += 1
        if m.group(2): count += 1
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

def _cleanup_noise_lines(md: str) -> str:
    lines = md.splitlines()
    out = []
    # Simple deduplication of blank lines and noise
    for ln in lines:
        if not ln.strip():
            if out and not out[-1].strip():
                continue
            out.append(ln)
            continue
        out.append(ln)
    return "\n".join(out)


def _drop_ocr_placeholder_lines(md: str) -> str:
    if not md:
        return md
    pat = re.compile(
        r"\((?:incomplete\s+visible|partially\s+visible|not\s+fully\s+visible)\)|\b(?:unreadable|illegible)\b",
        flags=re.IGNORECASE,
    )
    lines = md.splitlines()
    out: list[str] = []
    in_fence = False
    in_math = False
    for ln in lines:
        st = ln.strip()
        if re.match(r"^\s*```", ln):
            in_fence = not in_fence
            out.append(ln)
            continue
        if st == "$$":
            in_math = not in_math
            out.append(ln)
            continue
        if (not in_fence) and (not in_math) and pat.search(st):
            continue
        out.append(ln)
    return "\n".join(out)

def _fix_malformed_code_fences(md: str) -> str:
    """
    Recover from OCR/VL malformed fenced code blocks, e.g.:
      ```
      10010001
      10101100 ```
    where the closing fence is not on a standalone line and causes the rest
    of the document to render as a giant code block in Markdown viewers.
    """
    lines = md.splitlines()
    out: list[str] = []
    in_fence = False

    open_re = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*$")
    close_only_re = re.compile(r"^\s*```\s*$")
    close_tail_re = re.compile(r"^(.*\S)\s+```\s*$")
    heading_re = re.compile(r"^\s*#{1,6}\s+\S")
    image_re = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")

    for ln in lines:
        st = ln.strip()

        if not in_fence:
            if open_re.match(st):
                out.append(st)
                in_fence = True
                continue
            # Stray trailing fences outside code blocks: drop only the fence tail.
            m_tail = close_tail_re.match(ln)
            if m_tail:
                out.append((m_tail.group(1) or "").rstrip())
                continue
            out.append(ln)
            continue

        # Inside fence:
        # If structural markdown starts, the fence was likely malformed/open too long.
        if heading_re.match(st) or image_re.match(st):
            out.append("```")
            in_fence = False
            out.append(ln)
            continue

        if close_only_re.match(st):
            out.append("```")
            in_fence = False
            continue

        # Handle inline closing fence at end of content line.
        m_close_tail = close_tail_re.match(ln)
        if m_close_tail:
            out.append((m_close_tail.group(1) or "").rstrip())
            out.append("```")
            in_fence = False
            continue

        out.append(ln)

    # Ensure fenced block balance.
    if in_fence:
        out.append("```")

    return "\n".join(out)

def _convert_caption_following_tabular_lines(md: str) -> str:
    # Logic to attach captions to tables if they got separated
    # This is a bit complex in regex, maybe simplified version here
    return md

def _reflow_hard_wrapped_paragraphs(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    buf: list[str] = []

    fence_re = re.compile(r"^\s*```")
    heading_re = re.compile(r"^\s*#{1,6}\s+")
    list_re = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")
    table_re = re.compile(r"^\s*\|")
    math_marker = re.compile(r"^\s*\$\$")

    in_fence = False
    in_math = False

    def flush_buf():
        nonlocal buf
        if not buf:
            return
        # Join with space
        merged = " ".join(buf).replace("  ", " ")
        out.append(merged)
        buf = []

    for line in lines:
        s = line.strip()
        if fence_re.match(line):
            flush_buf()
            in_fence = not in_fence
            out.append(line)
            continue
        if math_marker.match(line):
            flush_buf()
            in_math = not in_math
            out.append(line)
            continue
        
        if in_fence or in_math:
            out.append(line)
            continue
        
        if not s:
            flush_buf()
            out.append(line)
            continue
        
        if (
            heading_re.match(line)
            or list_re.match(line)
            or table_re.match(line)
            or _looks_like_promotable_numbered_heading_line(s)
        ):
            flush_buf()
            out.append(line)
            continue
            
        # Image link
        if s.startswith("![") and s.endswith(")"):
            flush_buf()
            out.append(line)
            continue
            
        # Otherwise, regular text line, buffer it
        buf.append(s)

    flush_buf()
    return "\n".join(out)

def _split_inline_heading_markers(md: str) -> str:
    """
    Put inline heading markers onto a new line.
    Example:
      "... (20 of 21) ## ADVANCED SCIENCE NEWS ..."
      ->
      "... (20 of 21)"
      "## ADVANCED SCIENCE NEWS ..."
    """
    # Common publisher footer pattern:
    # "... (20 of 21) ADVANCED SCIENCE NEWS ..."
    # Force a line break after the page marker before all-caps titles.
    md = re.sub(
        r"(\(\s*\d+\s+of\s+\d+\s*\))\s+(?=(?:##\s+)?[A-Z][A-Z ]{5,}\b)",
        r"\1\n",
        md,
    )

    lines = md.splitlines()
    out: list[str] = []
    in_fence = False
    in_math = False

    for line in lines:
        s = line.strip()
        if re.match(r"^\s*```", line):
            in_fence = not in_fence
            out.append(line)
            continue
        if s == "$$":
            in_math = not in_math
            out.append(line)
            continue
        if in_fence or in_math:
            out.append(line)
            continue

        cur = line
        # Repeatedly split if there are multiple inline headings in one line.
        while True:
            m = re.search(r"\s(#{1,6}\s+\S)", cur)
            if not m:
                break
            idx = m.start(1)
            if idx <= 0:
                break
            left = cur[:idx].rstrip()
            right = cur[idx:].lstrip()
            # Avoid splitting URL anchors like ".../path#section".
            if idx > 0 and cur[idx - 1] in "/_-":
                break
            if left:
                out.append(left)
            cur = right
            if re.match(r"^\s*#{1,6}\s+", cur):
                break
        out.append(cur)

    return "\n".join(out)

def _fix_split_numbered_headings(md: str) -> str:
    # "1\nIntroduction" -> "1 Introduction"
    # Copied from legacy converter logic roughly
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_fence = False
    in_math = False
    while i < len(lines):
        line = lines[i]
        st = line.strip()
        if re.match(r"^\s*```", line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if st == "$$":
            in_math = not in_math
            out.append(line)
            i += 1
            continue
        if in_fence or in_math:
            out.append(line)
            i += 1
            continue

        if i + 1 < len(lines):
            next_line = lines[i+1]
            next_st = next_line.strip()
            # If line is just a small section token (number/letter), not arbitrary digits.
            is_small_number_token = bool(
                re.fullmatch(r"\d{1,2}(?:\.\d{1,2}){0,3}\.?", st)
            )
            is_letter_token = bool(re.fullmatch(r"[A-Z]\.?", st))
            next_is_heading_like = bool(
                re.match(
                    r"^(?:[A-Z][A-Za-z0-9][^\n]{0,140}|"
                    r"(?:abstract|introduction|related work|method(?:s|ology)?|"
                    r"experiment(?:s|al)?|results?|discussion|conclusion|references|appendix)\b)",
                    next_st,
                    re.IGNORECASE,
                )
            )
            next_is_structure = bool(
                re.match(r"^(?:#{1,6}\s+|```|\$\$|!\[[^\]]*\]\([^)]+\)|\|)", next_st)
            )
            if (is_small_number_token or is_letter_token):
                if next_st and next_is_heading_like and (not next_is_structure) and (not _parse_numbered_heading_level(next_st)):
                    # Merge split heading token + title line.
                    out.append(f"{st} {next_st}")
                    i += 2
                    continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _looks_like_promotable_numbered_heading_line(line: str) -> bool:
    t = _normalize_text(line or "").strip()
    if not t:
        return False
    m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$", t)
    if not m:
        return False
    head = (m.group(2) or "").strip()
    if not head:
        return False
    # OCR/VL placeholders should never be promoted to headings.
    if re.search(
        r"\((?:incomplete\s+visible|partially\s+visible|not\s+fully\s+visible)\)|\b(?:unreadable|illegible)\b",
        head,
        flags=re.IGNORECASE,
    ):
        return False
    # Avoid converting body sentences like "1. this method ...".
    if re.match(r"^[a-z]", head):
        return False
    # Reject references-like initials (e.g., "4. J. R. ...") that often appear
    # when references are partially visible on dense pages.
    if re.match(r"^(?:[A-Z]\.\s*){1,6}(?:[A-Z][A-Za-z'\-]+\.?)?(?:\s*\(|\s*,|$)", head):
        long_words = re.findall(r"[A-Za-z]{4,}", head)
        if len(long_words) <= 1:
            return False
    if len(head) > 140:
        return False
    if head.endswith((".", "!", "?", ";", ":")):
        return False
    if (head.count(",") >= 3) or ("http" in head.lower()) or ("doi" in head.lower()) or ("@" in head):
        return False
    if _is_caption_heading_text(head):
        return False
    return True


def _promote_bare_numbered_headings(md: str) -> str:
    """
    Promote standalone numbered headings that VL OCR emitted without markdown markers:
      "2.1. Methods" -> "### 2.1. Methods" (title-aware section depth)
    """
    lines = md.splitlines()
    out: list[str] = []
    in_fence = False
    in_math = False
    in_refs = False

    for ln in lines:
        st = ln.strip()
        if re.match(r"^\s*```", ln):
            in_fence = not in_fence
            out.append(ln)
            continue
        if st == "$$":
            in_math = not in_math
            out.append(ln)
            continue
        if re.match(r"^#{1,6}\s+References\b", st, re.IGNORECASE):
            in_refs = True
            out.append(ln)
            continue
        if in_fence or in_math or in_refs:
            out.append(ln)
            continue
        if re.match(r"^\s*#{1,6}\s+", ln):
            out.append(ln)
            continue

        if _looks_like_promotable_numbered_heading_line(st):
            m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$", st)
            if m:
                nums = [x for x in (m.group(1) or "").split(".") if x]
                depth = max(1, len(nums))
                lvl = min(6, depth + 1)  # reserve H1 for document title
                out.append("#" * lvl + " " + st)
                continue

        out.append(ln)

    return "\n".join(out)

def _unwrap_math_wrapped_headings(md: str) -> str:
    """
    Fix headings that were wrapped into inline-math by OCR/LLM, e.g.:
      $1. Introduction$  ->  # 1. Introduction
      $5. Conclusion$    ->  # 5. Conclusion
    Keep it deterministic and conservative.
    """
    lines = md.splitlines()
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        m = re.match(r"^\$(.+)\$$", s)
        if not m:
            out.append(ln)
            continue
        inner = (m.group(1) or "").strip()
        if not inner:
            out.append(ln)
            continue
        # Looks like a section heading (numbered / roman / appendix / common names)
        looks = False
        if re.match(r"^\d+(?:\.\d+)*\.?\s+\S+", inner):
            looks = True
        elif re.match(r"^[IVX]+\.\s+\S+", inner, re.IGNORECASE):
            looks = True
        elif re.match(r"^[A-Z]\.\s+\S+", inner):
            looks = True
        elif _is_common_section_heading(inner):
            looks = True
        # Reject if it looks like actual math
        if looks and (len(re.findall(r"[=^_{}\\\[\]]", inner)) == 0) and re.search(r"[A-Za-z]{3,}", inner):
            lvl = _parse_numbered_heading_level(inner) or _parse_appendix_heading_level(inner) or (2 if re.match(r"^[IVX]+\.", inner, re.IGNORECASE) else 1)
            lvl = int(min(4, max(1, lvl)))
            out.append("#" * lvl + " " + inner)
        else:
            out.append(ln)
    return "\n".join(out)

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

    # If the first heading is title-like (non-numbered), reserve H1 for it.
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
        m = re.match(r"^(#+)\s+(.*)", line)
        if not m:
            out.append(line)
            continue

        lvl = len(m.group(1))
        title = (m.group(2) or "").strip()
        if not title:
            out.append(line)
            continue

        # Figure/Table/Algorithm captions are not section headings.
        if _is_caption_heading_text(title):
            out.append(title)
            continue

        # Demote obvious author/affiliation lines misclassified as headings.
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
            # In numbered papers, ad-hoc unnumbered headings appearing after numbered
            # sections are usually OCR/VL heading hallucinations from captions/prose.
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
                    # Drop duplicated structural heading variants entirely.
                    continue
                # Keep content but demote from heading to plain text.
                out.append(title)
                continue

        # Normalize numbered / roman / appendix heading levels deterministically.
        desired = _parse_numbered_heading_level(title)
        if desired is None:
            if re.match(r"^[IVX]+\.\s+", title, re.IGNORECASE):
                desired = 2
            else:
                desired = _parse_appendix_heading_level(title)
        if desired is not None:
            if has_explicit_title:
                # Keep document title as H1, shift numbered hierarchy by +1.
                desired = desired + 1
            lvl = int(min(6, max(1, desired)))
        elif _is_common_section_heading(title):
            lvl = 2 if has_explicit_title else 1

        # If title is numeric only, demote
        if re.fullmatch(r"[\d\.]+", title):
            out.append(title)
            continue

        # Demote likely false headings (too long)
        if len(title) > 150:
            out.append(title)
            continue

        # Keep only one H1 in final output.
        if lvl == 1:
            if seen_h1:
                lvl = 2
            else:
                seen_h1 = True

        out.append("#" * lvl + " " + title)
    return "\n".join(out)

def _format_references(md: str) -> str:
    """
    Deterministic references formatter (fast, no LLM):
    - Find the References heading
    - Ensure each entry starts on its own line: [n] ...
    - Merge wrapped lines into the previous entry
    """
    sup_cite_pat = r"\[\s*\d{1,4}(?:\s*[,;\u2013\-]\s*\d{1,4})*\s*\]"

    def _unwrap_sup_cites(s: str) -> str:
        t = s or ""
        # $^{[12]}$ / ^{[12]} -> [12]
        t = re.sub(
            rf"\$\s*\^\{{\s*({sup_cite_pat})\s*\}}\s*\$",
            lambda m: (m.group(1) or "").strip(),
            t,
        )
        t = re.sub(
            rf"\^\{{\s*({sup_cite_pat})\s*\}}",
            lambda m: (m.group(1) or "").strip(),
            t,
        )
        return t

    def _looks_reference_payload_line(s: str) -> bool:
        t = (s or "").strip()
        if not t:
            return False
        if re.match(r"^\[\d{1,4}\]\s+[A-Z]", t):
            return True
        marker_n = len(re.findall(r"\[\d{1,4}\]", t))
        if marker_n >= 4 and re.search(r"\b(?:19|20)\d{2}\b", t):
            return True
        if marker_n >= 10:
            return True
        return False

    heading_any_re = re.compile(r"^#{1,6}\s+")
    plain_section_re = re.compile(
        r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
        r"(?:introduction|background|related work|method(?:s|ology)?|"
        r"experiment(?:s|al)?|results?|discussion|conclusion|appendix|"
        r"acknowledg(?:e)?ments?)\b",
        re.IGNORECASE,
    )

    def _heading_title(ln: str) -> str:
        return re.sub(r"^#{1,6}\s+", "", (ln or "").strip()).strip()

    lines = [_unwrap_sup_cites(ln) for ln in md.splitlines()]
    if not lines:
        return md

    ref_i = None
    inferred_heading = False
    for i, ln in enumerate(lines):
        if re.match(r"^#{1,6}\s+References\b", ln.strip(), re.IGNORECASE):
            ref_i = i
            break
        if re.match(r"^#{1,6}\s+Bibliography\b", ln.strip(), re.IGNORECASE):
            ref_i = i
            break
    if ref_i is None:
        # Fallback: infer references start when heading is missing.
        # We only trigger when there are many reference-like leading markers,
        # and they are concentrated in the latter part of the document.
        cand_idx: list[int] = []
        ref_start_re = re.compile(r"^\[\d{1,4}\]\s+[A-Z]")
        for i, ln in enumerate(lines):
            s = ln.strip()
            if ref_start_re.match(s):
                cand_idx.append(i)
        if len(cand_idx) >= 3:
            tail_threshold = int(len(lines) * 0.45)
            tail_cands = [i for i in cand_idx if i >= tail_threshold]
            if len(tail_cands) >= 3:
                ref_i = max(0, tail_cands[0] - 1)
                inferred_heading = True

        # Some VL outputs collapse many references into dense marker runs like:
        #   [1] [2] ... [24] [1] Author..., especially near section boundaries.
        if ref_i is None:
            dense_idx: list[int] = []
            for i, ln in enumerate(lines):
                s = ln.strip()
                if not s:
                    continue
                if not re.match(r"^(?:\[\d{1,4}\]\s*){2,}", s):
                    continue
                marker_n = len(re.findall(r"\[\d{1,4}\]", s))
                if marker_n >= 8 and re.search(r"\b(?:19|20)\d{2}\b", s):
                    dense_idx.append(i)
            if dense_idx:
                tail_gate = int(len(lines) * 0.45)
                tail_dense = [i for i in dense_idx if i >= tail_gate]
                if tail_dense:
                    ref_i = max(0, tail_dense[0] - 1)
                    inferred_heading = True

        if ref_i is None:
            # Some converters collapse many references into one long line.
            # Use document-level marker density as fallback.
            doc = "\n".join(lines)
            all_markers = list(re.finditer(r"\[\d{1,4}\]\s+[A-Z]", doc))
            if len(all_markers) < 6:
                return md
            first_pos = int(all_markers[0].start())
            ratio_gate = 0.35 if len(doc) >= 4000 else 0.15
            if first_pos < int(len(doc) * ratio_gate):
                return md
            ref_start_line = doc[:first_pos].count("\n")
            ref_i = max(0, int(ref_start_line))
            inferred_heading = True

    # Some outputs place "References" late (or infer it late), with one or more
    # reference payload lines right before the heading/start index.
    # Pull those lines into the references tail in both explicit and inferred cases.
    tail_start = ref_i if inferred_heading else (ref_i + 1)
    k = ref_i - 1
    pre_tail: list[str] = []
    while k >= 0:
        s = lines[k].strip()
        if not s:
            if pre_tail:
                pre_tail.insert(0, lines[k])
                k -= 1
                continue
            break
        if _looks_reference_payload_line(s):
            pre_tail.insert(0, lines[k])
            k -= 1
            continue
        break

    if pre_tail:
        head = lines[: k + 1]
        tail = pre_tail + lines[tail_start:]
    else:
        head = lines[:ref_i]
        tail = lines[tail_start:]

    # Bound the references tail to avoid swallowing body content when a paper
    # places a references block early (or headings are partially missing).
    tail_end = len(lines)
    if not inferred_heading:
        ref_signal = 0
        non_ref_run = 0
        non_ref_start = -1
        for j in range(tail_start, len(lines)):
            st = lines[j].strip()
            if not st:
                continue
            if re.match(r"^#{1,6}\s+(?:References|Bibliography)\b", st, re.IGNORECASE):
                continue

            # Strong section boundary (markdown heading).
            if heading_any_re.match(st):
                title = _heading_title(st)
                if (
                    plain_section_re.match(title)
                    or re.match(r"^(?:\d+(?:\.\d+)*|[IVXLCM]+)\.?\s+", title, re.IGNORECASE)
                    or re.match(r"^(?:appendix|acknowledg(?:e)?ments?)\b", title, re.IGNORECASE)
                ):
                    tail_end = j
                    break

            # Plain-text section boundary (no markdown heading marker).
            if plain_section_re.match(st) and ref_signal >= 3:
                tail_end = j
                break

            if _looks_reference_payload_line(st):
                ref_signal += 1
                non_ref_run = 0
                non_ref_start = -1
            else:
                if non_ref_run == 0:
                    non_ref_start = j
                non_ref_run += 1
                # After enough reference signal, a long run of non-reference lines
                # means we've crossed back into normal body content.
                if ref_signal >= 8 and non_ref_run >= 8 and non_ref_start >= tail_start:
                    tail_end = non_ref_start
                    break

    if tail_end < tail_start:
        tail_end = tail_start
    body_tail = lines[tail_end:]
    if pre_tail:
        tail = pre_tail + lines[tail_start:tail_end]
    else:
        tail = lines[tail_start:tail_end]
    head.append("## References")

    blob = "\n".join(tail).strip()
    if not blob:
        if body_tail:
            return "\n".join(head + [""] + body_tail)
        return "\n".join(head)

    # Normalize superscript-citation wrappers before reference parsing.
    blob = _unwrap_sup_cites(blob)

    # References must be plain text: aggressively unwrap math delimiters and
    # display-math fences that vision models may emit by mistake.
    # 1) Convert $$...$$ blocks into plain text (single/multi-line).
    blob = re.sub(r"\$\$\s*(.*?)\s*\$\$", lambda m: " " + re.sub(r"\s+", " ", (m.group(1) or "").strip()) + " ", blob, flags=re.DOTALL)
    # 2) Convert inline $...$ into plain text.
    blob = re.sub(r"\$([^$\n]{1,400})\$", lambda m: (m.group(1) or "").strip(), blob)
    # 3) Drop orphan fence lines and residual dollar signs.
    blob = re.sub(r"(?m)^\s*\$\$\s*$", "", blob)
    blob = blob.replace("$", "")
    blob = re.sub(r"[ \t]{2,}", " ", blob)

    # Split multiple [n] items that got collapsed into one line.
    blob = re.sub(r"\s+(?=\[\d+\])", "\n", blob)
    # Normalize leading markers like "1] ..." or "1. ..."
    blob = re.sub(r"(?m)^\s*(\d+)\]\s*", r"[\1] ", blob)
    blob = re.sub(r"(?m)^\s*(\d+)\.\s+", r"[\1] ", blob)

    def _trim_reference_noise(entry: str) -> str:
        s = re.sub(r"\s+", " ", (entry or "")).strip()
        if not s:
            return s

        # Drop explicit OCR placeholders from clipped column crops.
        if re.search(
            r"\((?:incomplete\s+visible|partially\s+visible|not\s+fully\s+visible)\)|\b(?:unreadable|illegible)\b",
            s,
            flags=re.IGNORECASE,
        ):
            return ""

        # Normalize stray superscript macro text leaked from OCR.
        s = re.sub(r"\\?textsuperscript\{([^{}]{0,120})\}", r"\1", s, flags=re.IGNORECASE)
        s = s.replace("©", " ")
        # Drop model/explanation artifacts that should never appear in references.
        s = re.sub(r"(?i)\bthere is no math equation in the provided garbled block[^.]*\.", " ", s)
        s = re.sub(r"(?i)\bno latex equation can be recovered\.?", " ", s)
        s = re.sub(r"\s{2,}", " ", s).strip()

        # References should not contain math operators/equations.
        # If hard-math tokens appear after the first year, trim that tail.
        try:
            hard_math_m = re.search(
                r"(\\operatorname\*?\{|\\arg(?:min|max)|\\frac|\\sum|\\int|\\left|\\right|\\mathbf\{|\\mathbb\{|\\begin\{|\\end\{|\\tag\{|\\\|)",
                s,
            )
            if hard_math_m:
                ym0 = re.search(r"\b(?:19|20)\d{2}\b", s)
                if ym0 and hard_math_m.start() > ym0.start():
                    s = s[: hard_math_m.start()].rstrip(" ,;:-")
        except Exception:
            pass

        # Find a likely citation terminus (year, volume, page/article id).
        end = -1
        first_start = None
        end_pats = [
            r"\b(?:19|20)\d{2}\s*,\s*\d+\s*,\s*[A-Za-z]?\d+[A-Za-z0-9\-]*\s*\.",
            r"\b(?:19|20)\d{2}\s*,\s*\d+\s*,\s*[A-Za-z]?\d+[A-Za-z0-9\-]*\b",
            r"\b(?:19|20)\d{2}\s*,\s*\d+\s*\.",
            r"\b\d+\s*\(\s*\d+\s*\)\s*,\s*[A-Za-z]?\d+[A-Za-z0-9\-]*\s*\\?\s*\(\s*(?:19|20)\d{2}\s*\)",
            r"\b\d+\s*\(\s*\d+\s*\)\s*,\s*[A-Za-z]?\d+[A-Za-z0-9\-]*\s*\(\s*(?:19|20)\d{2}\s*\)",
        ]
        for pat in end_pats:
            for m in re.finditer(pat, s):
                st = int(m.start())
                ed = int(m.end())
                if first_start is None or st < first_start:
                    first_start = st
                    end = ed
                elif st == first_start:
                    # Prefer the longer candidate at the same start.
                    end = max(end, ed)

        tail = s[end:].strip() if (end > 0 and end < len(s)) else ""
        noise_markers = [
            r"\bAcknowledg(?:e)?ments?\b",
            r"\bConflict(?:s)? of Interest\b",
            r"\bKeywords?\b",
            r"\bBiographies?\b",
            r"\bAbout the Authors?\b",
            r"\bSupplementary\b",
            r"\bSupporting Information\b",
            r"\bThis work was supported\b",
            r"\bReceived\b|\bAccepted\b|\bPublished\b",
            r"\bwww\.[^\s]+",
            r"https?://[^\s]+",
            r"(?:copyright|\(c\)|©)\s*\d{4}",
            r"\(\s*\d+\s+of\s+\d+\s*\)",
            r"\b(?:advancedsciencenews|lpr-journal)\b",
        ]
        tail_keep = re.compile(
            r"^(?:doi\b|https?://(?:dx\.)?doi\.org/|arxiv\s*:|e-?print\b|pmid\b|isbn\b|issn\b)",
            flags=re.IGNORECASE,
        )

        def _is_prose_like_tail(t: str) -> bool:
            tt = re.sub(r"\[\s*\d{1,4}(?:\s*[,\u2013\-]\s*\d{1,4})*\s*\]", " ", t)
            tt = re.sub(r"\{[^{}]{0,120}\}", " ", tt)
            tt = re.sub(r"\\[A-Za-z]+", " ", tt)
            tt = re.sub(r"\s+", " ", tt).strip()
            if not tt:
                return False
            words = re.findall(r"[A-Za-z]{2,}", tt)
            if len(words) < 8:
                return False
            stop = {
                "the", "and", "of", "to", "in", "for", "with", "from", "that", "this", "these",
                "those", "is", "are", "was", "were", "be", "been", "being", "it", "its", "on",
                "as", "at", "by", "can", "could", "will", "would", "should", "into", "such",
                "more", "than", "through", "their", "there", "also",
            }
            stop_n = sum(1 for w in words if w.lower() in stop)
            lower_start = bool(re.match(r"^[a-z]", tt))
            sentence_like = bool(re.search(r"[.!?]\s+[A-Z][a-z]{2,}", tt))
            cite_clusters = len(re.findall(r"\[\s*\d{1,4}(?:\s*[,\u2013\-]\s*\d{1,4})*\s*\]", t))
            if lower_start and len(words) >= 7 and stop_n >= 2:
                return True
            if len(words) >= 18 and stop_n >= 4:
                return True
            if stop_n >= 5 and (lower_start or sentence_like):
                return True
            if cite_clusters >= 2 and stop_n >= 3 and len(words) >= 10:
                return True
            return False

        cut_pos: int | None = None
        if end > 0 and tail:
            if not tail_keep.match(tail):
                # Typical merged-next-reference signature: author initials at tail start.
                if re.match(r"^(?:[A-Z]\.\s*){1,5}[A-Z][A-Za-z'\-]+,", tail):
                    cut_pos = end
                # Long natural-language tails after a complete citation are contamination.
                if cut_pos is None and _is_prose_like_tail(tail):
                    cut_pos = end
                # Publisher/footer/section artifacts after citation terminus.
                if cut_pos is None:
                    for pat in noise_markers:
                        if re.search(pat, tail, flags=re.IGNORECASE):
                            cut_pos = end
                            break
                # Another author-list chunk after citation terminus usually means merged contamination.
                if cut_pos is None:
                    m_tail_auth = re.search(r"(?:^|\s)(?:[A-Z]\.\s*){1,4}[A-Z][A-Za-z'\-]+,\s", tail)
                    if m_tail_auth and len(tail) >= 48:
                        cut_pos = end + int(m_tail_auth.start())
                # Keep strict upper bound to avoid giant merged blocks.
                if cut_pos is None and len(tail) >= 120:
                    cut_pos = end

        # If no robust terminus was matched, use year-anchor fallback on very long entries.
        if cut_pos is None and end < 0 and len(s) >= 320:
            ym = re.search(r"\(\s*(?:19|20)\d{2}\s*\)|\b(?:19|20)\d{2}\b", s)
            if ym:
                tail2 = s[ym.end() :].strip()
                if len(tail2) >= 120 and not tail_keep.match(tail2):
                    cut_pos = int(ym.end())

        if cut_pos is None:
            # Fallback: obvious publisher/footer markers anywhere.
            marker_pos: list[int] = []
            for pat in noise_markers:
                m = re.search(pat, s, flags=re.IGNORECASE)
                if m:
                    marker_pos.append(int(m.start()))
            if marker_pos:
                cut_pos = min(marker_pos)

        if cut_pos is not None and cut_pos > 0:
            s = s[:cut_pos].rstrip(" ,;:-")

        # Drop marker-only shards (e.g., "[1] [2] [3]") and other non-reference leftovers.
        try:
            year_n = len(re.findall(r"\b(?:19|20)\d{2}\b", s))
            word_n = len(re.findall(r"[A-Za-z]{2,}", s))
            if year_n == 0 and word_n < 3:
                return ""
        except Exception:
            pass

        return s

    entries: list[str] = []
    cur: list[str] | None = None
    start_re = re.compile(r"^\[(\d+)\]\s+")
    for raw in blob.splitlines():
        s = raw.strip()
        if not s:
            continue
        m = start_re.match(s)
        if m:
            if cur:
                entries.append(" ".join(cur).strip())
            cur = [s]
            continue
        # Ignore garbage before first reference marker
        if cur is None:
            continue
        cur.append(s)
    if cur:
        entries.append(" ".join(cur).strip())

    if not entries:
        out0 = head + [""] + tail
        if body_tail:
            out0.extend([""] + body_tail)
        return "\n".join(out0)

    entries = [_trim_reference_noise(e) for e in entries if (e or "").strip()]
    entries = [e for e in entries if (e or "").strip()]

    # Sort by reference number when possible; this fixes reading-order shuffles.
    parsed: list[tuple[int, str]] = []
    unknown: list[str] = []
    for e in entries:
        # OCR sometimes prepends an article-id-like marker before the real reference number.
        # Example: [30390] [189] ...  -> keep [189].
        e = re.sub(r"^\[(\d{3,6})\]\s+\[(\d{1,4})\]\s+", r"[\2] ", e)
        m = start_re.match(e)
        if not m:
            unknown.append(e)
            continue
        try:
            n = int(m.group(1))
        except Exception:
            unknown.append(e)
            continue
        if n <= 0 or n > 2000:
            unknown.append(e)
            continue
        parsed.append((n, e))

    parsed.sort(key=lambda x: x[0])
    seen_nums: set[int] = set()
    out_refs: list[str] = []
    for n, e in parsed:
        if n in seen_nums:
            continue
        seen_nums.add(n)
        out_refs.append(e)
    # Keep any unknown tail lines at the end (rare)
    out_refs.extend(unknown)

    out1 = head + [""] + out_refs
    if body_tail:
        out1.extend([""] + body_tail)
    return "\n".join(out1)

def fix_math_markdown(md: str) -> str:
    """
    Deterministic (non-LLM) math cleanup for Markdown.

    Fixes common converter failure patterns:
    - Non-math text accidentally wrapped in $$...$$ (citations, prose fragments)
    - Prose/captions leaking into display-math blocks (e.g. "where ...", "Figure 3 ...")
    - Garbled / unicode math symbols and spacing: ⊙, ∈, ˆ, \\inR, etc.
    - Inline-math runs split across multiple lines ($N X$ / $X$ / ...), merge into one display block.
    """

    if not md or "$$" not in md:
        return md

    _GREEK_MAP = {
        r"\sigma": "σ",
        r"\delta": "δ",
        r"\lambda": "λ",
        r"\theta": "θ",
        r"\mu": "μ",
        r"\pi": "π",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\omega": "ω",
    }

    def _latex_text_to_plain(s: str) -> str:
        """
        Best-effort cleanup for stray LaTeX text macros that leak into plain text.
        Keep conservative: do not attempt to reformat true math.
        """
        if not s:
            return ""
        t = str(s)
        # Unwrap common text macros
        for _ in range(3):
            t2 = re.sub(r"\\text\{([^{}]{0,220})\}", r"\1", t)
            t2 = re.sub(r"\\mathrm\{([^{}]{0,220})\}", r"\1", t2)
            t2 = re.sub(r"\\mathbf\{([^{}]{0,220})\}", r"\1", t2)
            if t2 == t:
                break
            t = t2

        # Convert cite macros to numeric brackets when possible: \cite{52} -> [52]
        def _cite_repl(m: re.Match) -> str:
            inner = (m.group(1) or "").strip().replace(" ", "")
            if not inner:
                return ""
            if re.fullmatch(r"\d+(?:,\d+)*", inner):
                return "[" + inner + "]"
            return m.group(0)
        t = re.sub(r"~?\\cite\{([^}]{1,120})\}", _cite_repl, t)

        # Spacing macros
        t = t.replace("~", " ")
        t = re.sub(r"\\quad|\\,|\\;|\\:|\\!", " ", t)
        # Escaped line breaks/spaces from OCR-ized LaTeX.
        t = t.replace("\\\\", " ")
        t = t.replace("\\ ", " ")

        # Greek letters that appear in prose due to extraction glitches
        for k, v in _GREEK_MAP.items():
            t = t.replace(k, v)

        # Drop residual LaTeX slashes in plain-text context.
        t = t.replace("\\", " ")
        return re.sub(r"\s{2,}", " ", t).strip()

    def _has_hard_math_anchors(s: str) -> bool:
        """
        Detect strong equation anchors.
        Keep this stricter than generic symbol checks so text-like blocks wrapped
        in \text{...} can be unwrapped back to prose.
        """
        if not s:
            return False
        return bool(
            re.search(
                r"(?:=|[<>]|\\frac|\\sum|\\int|\\prod|\\sqrt|\\arg|\\min|\\max|\\partial|\\nabla|\\times|\\cdot|\\leq|\\geq|\\neq|\\approx|\^|_[{A-Za-z0-9])",
                s,
                flags=re.IGNORECASE,
            )
        )

    def _norm_math(s: str) -> str:
        t = s
        # Normalize common unicode math operators.
        t = t.replace("⊙", r"\odot")
        t = t.replace("∈", r"\in")
        t = t.replace("×", r"\times")
        t = t.replace("·", r"\cdot")
        # Fix hat artifacts, but do NOT rewrite normal superscripts like "^N".
        # Only convert explicit hat-like glyphs or spaced caret forms.
        t = re.sub(r"[ˆ]\s*([A-Za-z])\b", r"\\hat{\1}", t)
        t = re.sub(r"(?<![A-Za-z0-9}])\^\s+([A-Za-z])\b", r"\\hat{\1}", t)
        # Fix common spacing issues.
        t = re.sub(r"\\in\s*R\b", r"\\in R", t)
        t = re.sub(r"\\inR\b", r"\\in R", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _split_prose_tail(s: str) -> tuple[str, str]:
        """
        Split 'math prefix' and 'prose tail' inside a math string.
        Returns (math_part, tail_part).
        """
        if not s:
            return "", ""
        ss = s.strip()
        low = ss.lower()

        # Captions
        cap = re.search(r"(?i)(\*?Figure\s+\d+\b|\bFig\.?\s*\d+\b|\bTable\s+\d+\b)", ss)
        cap_pos = cap.start() if cap else -1

        # Where-explanations (accept space/newline separators)
        where_pos = -1
        for tok in (" where ", "\nwhere ", "\r\nwhere "):
            p = low.find(tok)
            if p >= 0:
                wstart = p + tok.find("where")
                if where_pos < 0 or wstart < where_pos:
                    where_pos = wstart
        if low.startswith("where "):
            where_pos = 0

        cut_pos = -1
        for p in (cap_pos, where_pos):
            if p is None or p < 0:
                continue
            if cut_pos < 0 or p < cut_pos:
                cut_pos = p

        if cut_pos <= 0:
            return ss, ""

        tail = ss[cut_pos:].strip()
        # Only treat as prose tail if it looks like natural language.
        tail_words = len(re.findall(r"\b[A-Za-z]{2,}\b", tail))
        if tail_words < 8 and not re.match(r"(?i)^(figure|fig\.?|table)\b", tail):
            return ss, ""

        return ss[:cut_pos].strip(), tail

    def _strip_text_macros(s: str) -> str:
        """Remove text-like LaTeX wrappers so math-anchor checks are not fooled."""
        if not s:
            return ""
        t = str(s)
        for _ in range(3):
            t2 = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]{0,260})\}", r"\1", t)
            t2 = re.sub(r"\\cite\{([^{}]{0,260})\}", r" ", t2)
            if t2 == t:
                break
            t = t2
        return t

    def _looks_like_not_math(block: str) -> bool:
        """
        Detect blocks wrapped in $$...$$ that are clearly prose/citations rather than equations.
        Keep it conservative: false negatives are OK.
        """
        t = (block or "").strip()
        if not t:
            return True
        # Text-wrapped pseudo equations from OCR/VL, e.g.
        # \text{...}[12]\text{...}[13]
        if "\\text{" in t:
            plain = _latex_text_to_plain(t)
            plain_words = len(re.findall(r"\b[A-Za-z]{2,}\b", plain))
            cite_like = bool(re.search(r"\[\s*\d{1,4}(?:\s*[,\u2013\-]\s*\d{1,4})*", plain))
            # Unwrap when it's clearly prose/citation text rather than equation syntax.
            if (plain_words >= 6 and not _has_hard_math_anchors(t)) or (plain_words >= 4 and cite_like and not _has_hard_math_anchors(t)):
                return True
        # Obvious prose markers
        if re.search(r"(?i)\b(in this paper|we propose|however|moreover|there are|these methods)\b", t):
            return True
        # Definition/prose lines often leak into math blocks in PDFs.
        # If it reads like a sentence, treat it as not-math even if it contains \odot/\in.
        if re.search(r"(?i)\b(represents|denotes|corresponding to|pixel value|element\s*-\s*wise)\b", t):
            if not _has_hard_math_anchors(t):
                return True
        # Citation fragments wrapped in $$...$$ are almost always wrong.
        if ("[" in t) and ("]" in t) and re.search(r"\d", t) and not _has_hard_math_anchors(t):
            return True
        # \text{...} / \cite{...} inside display math is often from a sentence fragment, not an equation.
        if ("\\text{" in t or "\\cite{" in t) and ("=" not in t) and not _has_hard_math_anchors(t):
            return True
        # Citation-only lines / bracketed refs
        if re.match(r"^\[\s*\d+", t) and not _has_hard_math_anchors(t):
            return True
        # Mostly words, few math anchors
        word_n = len(re.findall(r"\b[A-Za-z]{2,}\b", t))
        # Don't count citation brackets [] as math symbols.
        t_probe = _strip_text_macros(t)
        sym_n = len(re.findall(r"[=+\-*/^_]|\\frac|\\sum|\\int|\\prod|\\sqrt|\\times|\\cdot|\\odot|\\in|\\partial|\\nabla", t_probe))
        if word_n >= 10 and sym_n <= 1:
            return True
        # Fragmentary short math blocks with unbalanced delimiters are usually extraction junk.
        if len(t) <= 40:
            if (t.count("(") != t.count(")")) or (t.count("{") != t.count("}")) or (t.count("[") != t.count("]")):
                return True
        return False

    # Normalize single-line display math to fenced form:
    #   $$ ... $$ -> $$\n...\n$$
    # This prevents later inline-math regex from collapsing it into $...$.
    lines_raw = md.splitlines()
    lines: list[str] = []
    for ln in lines_raw:
        st = ln.strip()
        if st.startswith("$$") and st.endswith("$$") and st != "$$":
            inner = st[2:-2].strip()
            if inner:
                lines.extend(["$$", inner, "$$"])
                continue
        lines.append(ln)

    # 1) Merge runs of split inline-math lines into one display-math block.
    merged_lines: list[str] = []
    i = 0
    inline_only = re.compile(r"^\s*\$[^$]{1,160}\$\s*$")
    eqno = re.compile(r"^\s*[,;:]?\s*\(\s*\d{1,4}\s*\)\s*$")
    while i < len(lines):
        s = lines[i].strip()
        if inline_only.match(lines[i]) or eqno.match(lines[i]) or re.fullmatch(r"\d{1,4}", s):
            # Try collect a run
            j = i
            parts: list[str] = []
            tag_n: str | None = None
            seen_inline = 0
            while j < len(lines) and (inline_only.match(lines[j]) or eqno.match(lines[j]) or not lines[j].strip() or re.fullmatch(r"\d{1,4}", lines[j].strip())):
                sj = lines[j].strip()
                if not sj:
                    j += 1
                    continue
                if eqno.match(lines[j]):
                    m = re.search(r"(\d{1,4})", sj)
                    if m:
                        tag_n = m.group(1)
                    j += 1
                    continue
                if inline_only.match(lines[j]):
                    inner = sj[1:-1].strip()
                    inner = _norm_math(inner)
                    parts.append(inner)
                    seen_inline += 1
                    j += 1
                    continue
                if re.fullmatch(r"\d{1,4}", sj) and tag_n is None:
                    tag_n = sj
                j += 1

                if seen_inline >= 4:
                    # don't over-eagerly consume huge regions
                    break

            if seen_inline >= 2:
                merged = " ".join(p for p in parts if p).strip()
                if tag_n:
                    merged = (merged + rf" \tag{{{tag_n}}}").strip()
                merged_lines.extend(["$$", merged, "$$", ""])
                i = j
                continue

        merged_lines.append(lines[i])
        i += 1

    # 2) Fix display-math blocks: strip prose/captions, normalize symbols, and unwrap false-math blocks.
    out: list[str] = []
    i = 0
    while i < len(merged_lines):
        if merged_lines[i].strip() != "$$":
            out.append(merged_lines[i])
            i += 1
            continue

        # Collect $$ block
        i += 1
        buf: list[str] = []
        while i < len(merged_lines) and merged_lines[i].strip() != "$$":
            buf.append(merged_lines[i])
            i += 1
        if i < len(merged_lines) and merged_lines[i].strip() == "$$":
            i += 1

        raw = "\n".join(buf).strip()
        # Remove nested inline $...$ inside display math.
        raw = raw.replace("$", "")
        raw = raw.replace("\uFFFD", " ")  # replacement char
        raw = raw.strip()

        # CRITICAL: Be very conservative - only unwrap if it's clearly prose/citation
        # Since we improved the LLM prompt, most $$ blocks should be real formulas
        # Only unwrap if it's OBVIOUSLY not math (e.g., full sentences with no math symbols)
        if _looks_like_not_math(raw):
            # Keep as math only when strong equation anchors remain after stripping text wrappers.
            raw_probe = _strip_text_macros(raw)
            if _has_hard_math_anchors(raw_probe):
                math_part, tail = _split_prose_tail(raw)
                math_part = _norm_math(math_part)
                if math_part:
                    out.extend(["$$", math_part, "$$"])
                if tail:
                    out.append("")
                    out.append(_normalize_text(tail))
                out.append("")
                continue
            if raw:
                out.append(_normalize_text(_latex_text_to_plain(raw)))
            out.append("")
            continue

        math_part, tail = _split_prose_tail(raw)
        math_part = _norm_math(math_part)
        # Attach equation number tags like ", (5)" that often get separated into the next line(s).
        # We look ahead in the original merged_lines stream (not yet consumed into `out`) by peeking
        # at the next non-empty line after the closing $$.
        tag_n: str | None = None
        try:
            j = i
            while j < len(merged_lines) and not merged_lines[j].strip():
                j += 1
            if j < len(merged_lines):
                s2 = merged_lines[j].strip()
                m = re.fullmatch(r"[,;:]?\s*\(\s*(\d{1,4})\s*\)\s*", s2)
                if m:
                    tag_n = m.group(1)
                    # Consume that line
                    i = j + 1
        except Exception:
            tag_n = None

        if tag_n and math_part:
            # Avoid duplicating tag if already present.
            if "\\tag{" not in math_part:
                math_part = (math_part + rf" \tag{{{tag_n}}}").strip()
        if math_part:
            out.extend(["$$", math_part, "$$"])
        if tail:
            out.append("")
            out.append(_normalize_text(tail))
        out.append("")

    # 3) Normalize ALL inline-math segments ($...$) anywhere in the document.
    # This fixes cases like: N X $ˆ Y(r) -$  ->  N X $\hat{Y}(r) -$
    out_s = "\n".join(out)
    try:
        def _repl_inline(m: re.Match) -> str:
            inner = (m.group(1) or "").strip()
            if not inner:
                return "$$"
            # Inline math that is actually citation/prose -> unwrap.
            inner_no_text = re.sub(r"\\(?:cite|text|mathrm|mathbf)\{[^}]{0,200}\}", "", inner)
            inner_no_text = inner_no_text.replace("~", " ")
            hard_math = bool(re.search(r"[=^_{}]|\\frac|\\sum|\\int|\\prod|\\sqrt|\\odot|\\in|\\times", inner_no_text))
            # Broken citations often appear as "[12, 13]" or even "word[12, 13" (missing closing bracket).
            # Be permissive as long as it contains numeric bracket and no hard math anchors.
            looks_like_bracket_cite = bool(re.search(r"\[\s*\d", inner)) and bool(re.search(r"\d", inner))
            if (not hard_math) and (looks_like_bracket_cite or ("\\cite{" in inner) or ("\\text{" in inner)):
                return _latex_text_to_plain(inner)
            # Inline math that is actually citation/prose -> unwrap.
            low = inner.lower()
            if re.fullmatch(r"\\hat\{[A-Za-z]\}_\{\[\d+\]\}", inner.replace(" ", "")):
                # e.g. \hat{C}_{[19]} is almost certainly a broken citation.
                mm = re.search(r"\[(\d+)\]", inner)
                if mm:
                    return f"[{mm.group(1)}]"
            if looks_like_bracket_cite and (not hard_math):
                return _latex_text_to_plain(inner)
            word_n = len(re.findall(r"\b[A-Za-z]{2,}\b", inner))
            sym_n = len(re.findall(r"[=+\-*/^_{}\\]|\\frac|\\sum|\\int|\\odot|\\in|\\times", inner))
            if word_n >= 6 and (sym_n == 0 or (sym_n <= 2 and ("\\cite{" in inner or "\\text{" in inner))):
                return _latex_text_to_plain(inner)
            return "$" + _norm_math(inner) + "$"
        out_s = re.sub(r"\$([^$\n]{1,220})\$", _repl_inline, out_s)
    except Exception:
        pass

    # 4) Drop tiny math shards that sit immediately next to a real display equation.
    # These are usually extraction artifacts like:
    #   N X $ˆ Y(r) -$
    #   X
    #   $$
    #   L = ...
    #   $$
    try:
        ls = out_s.splitlines()
        cleaned: list[str] = []
        k = 0
        while k < len(ls):
            s0 = ls[k].strip()
            if (s0 == "X" or s0 == "N X" or re.match(r"^N\s+X\s+\$[^$]+\$$", s0)) and (k + 1 < len(ls)):
                # Look ahead for a display equation starting soon.
                look = "\n".join(ls[k : min(len(ls), k + 10)])
                if re.search(r"(?m)^\$\$\s*$\n\s*L\s*=", look):
                    k += 1
                    continue
            cleaned.append(ls[k])
            k += 1
        out_s = "\n".join(cleaned)
    except Exception:
        pass

    out_s = out_s.rstrip() + "\n"

    # 5) Drop tiny orphan lines right next to display equations.
    # Common in two-column PDFs: stray tokens like "Z t f", "t n", "R t" extracted as separate lines.
    try:
        ls2 = out_s.splitlines()
        cleaned2: list[str] = []
        k = 0
        while k < len(ls2):
            s0 = (ls2[k] or "").strip()
            if s0 and len(s0) <= 12 and re.fullmatch(r"[A-Za-z]\s*[A-Za-z](?:\s*[A-Za-z]){0,3}", s0):
                look_fwd = "\n".join(ls2[k : min(len(ls2), k + 4)])
                look_back = "\n".join(ls2[max(0, k - 4) : k + 1])
                # Drop if a display-math block starts soon, or we just ended one.
                if re.search(r"(?m)^\$\$\s*$", look_fwd) or re.search(r"(?m)^\$\$\s*$", look_back):
                    k += 1
                    continue
            cleaned2.append(ls2[k])
            k += 1
        out_s = "\n".join(cleaned2).rstrip() + "\n"
    except Exception:
        pass

    # 6) Ensure $$ fences are balanced.
    # Unbalanced $$ can cause the rest of the document to be interpreted as math,
    # which breaks rendering and prevents later cleanup passes from running.
    try:
        ls3 = out_s.splitlines()
        stack: list[int] = []
        keep = [True] * len(ls3)
        for i, ln in enumerate(ls3):
            if (ln or "").strip() != "$$":
                continue
            if not stack:
                stack.append(i)  # opening
            else:
                stack.pop()      # closing
        # Any remaining openings are unmatched; drop those fence lines.
        for i in stack:
            if 0 <= i < len(keep):
                keep[i] = False
        if stack:
            ls3 = [ln for i, ln in enumerate(ls3) if keep[i]]
            out_s = "\n".join(ls3).rstrip() + "\n"
    except Exception:
        pass

    return out_s


def _cleanup_stray_latex_in_text(md: str) -> str:
    """
    Cleanup LaTeX text macros leaked into plain text, without touching:
    - display math blocks ($$...$$)
    - references section
    - images / headings
    """
    if not md or ("\\" not in md and "$" not in md):
        return md
    lines = md.splitlines()
    out: list[str] = []
    in_math = False
    in_refs = False
    for ln in lines:
        s = ln.rstrip("\n")
        st = s.strip()
        if st == "$$":
            in_math = not in_math
            out.append(s)
            continue
        if re.match(r"^#{1,6}\s+References\b", st, re.IGNORECASE):
            in_refs = True
            out.append(s)
            continue
        if in_refs or in_math:
            out.append(s)
            continue
        if re.match(r"^#{1,6}\s+", st) or re.match(r"^\s*!\[[^\]]*\]\([^)]+\)", st):
            out.append(s)
            continue

        t = s
        # Some PDFs leak italic/emphasis markers as stray `$` in plain text, e.g.:
        #   $representation$[11, 34]$. Others ...$
        # These are not math; strip dollars when the line looks like prose/citations.
        if "$" in t:
            hard_math_line = bool(re.search(r"[=^_{}]|\\frac|\\sum|\\int|\\prod|\\sqrt|\\odot|\\in|\\times", t))
            citey_line = bool(re.search(r"\[\s*\d", t)) or ("\\cite{" in t)
            if (not hard_math_line) and citey_line and (t.count("$") >= 2):
                t = t.replace("$", "")

        # Unwrap line-level $...$ that is clearly prose/citations (not math).
        # This catches a few broken cases that can escape fix_math_markdown's inline pass.
        try:
            m0 = re.fullmatch(r"\s*\$([^$\n]{1,220})\$\s*", t)
        except Exception:
            m0 = None
        if m0:
            inner0 = (m0.group(1) or "").strip()
            inner_probe = re.sub(r"\\(?:cite|text|mathrm|mathbf)\{[^}]{0,200}\}", "", inner0)
            hard_math0 = bool(re.search(r"[=^_{}]|\\frac|\\sum|\\int|\\prod|\\sqrt|\\odot|\\in|\\times", inner_probe))
            looks_like_cite0 = bool(re.search(r"\[\s*\d", inner0)) or ("\\cite{" in inner0) or ("\\text{" in inner0)
            if looks_like_cite0 and (not hard_math0):
                t = inner0

        for _ in range(3):
            t2 = re.sub(r"\\text\{([^{}]{0,220})\}", r"\1", t)
            t2 = re.sub(r"\\mathrm\{([^{}]{0,220})\}", r"\1", t2)
            t2 = re.sub(r"\\mathbf\{([^{}]{0,220})\}", r"\1", t2)
            if t2 == t:
                break
            t = t2

        def _cite_repl(m: re.Match) -> str:
            inner = (m.group(1) or "").strip().replace(" ", "")
            if re.fullmatch(r"\d+(?:,\d+)*", inner or ""):
                return "[" + inner + "]"
            return m.group(0)
        t = re.sub(r"~?\\cite\{([^}]{1,120})\}", _cite_repl, t)
        t = re.sub(r"\$(\[[0-9,\s]+\])\$", r"\1", t)
        out.append(t)
    return "\n".join(out)


def _normalize_typora_math_expr(expr: str) -> str:
    """
    Normalize math commands to Typora/KaTeX-friendly forms without changing meaning.
    """
    t = (expr or "")
    if not t:
        return ""

    # 1) Strip/convert environments that are often problematic in Typora.
    t = re.sub(r"\\begin\{equation\*?\}", "", t)
    t = re.sub(r"\\end\{equation\*?\}", "", t)
    t = re.sub(r"\\begin\{align\*?\}", r"\\begin{aligned}", t)
    t = re.sub(r"\\end\{align\*?\}", r"\\end{aligned}", t)

    # 2) Convert old/deprecated TeX commands to modern KaTeX-friendly forms.
    t = re.sub(r"\\mbox\s*\{", r"\\text{", t)
    t = re.sub(r"\\Bbb(?=[^A-Za-z]|$)", r"\\mathbb", t)
    t = re.sub(r"\\mathbbm(?=[^A-Za-z]|$)", r"\\mathbb", t)
    t = re.sub(r"\\dfrac(?=[^A-Za-z]|$)", r"\\frac", t)
    t = re.sub(r"\\tfrac(?=[^A-Za-z]|$)", r"\\frac", t)
    t = re.sub(r"\\cfrac(?=[^A-Za-z]|$)", r"\\frac", t)

    # Old font switches.
    t = re.sub(r"\\rm\s*\{", r"\\mathrm{", t)
    t = re.sub(r"\\bf\s*\{", r"\\mathbf{", t)
    t = re.sub(r"\\it\s*\{", r"\\mathit{", t)
    t = re.sub(r"\\cal\s*\{", r"\\mathcal{", t)
    t = re.sub(r"\\tt\s*\{", r"\\mathtt{", t)

    # 3) Expand DeclareMathOperator macros into operatorname and remove declarations.
    op_defs: list[tuple[str, str]] = []
    for m in re.finditer(r"\\DeclareMathOperator(\*?)\{\\([A-Za-z]{1,40})\}\{([^{}]{1,80})\}", t):
        star = "*" if (m.group(1) or "") == "*" else ""
        name = m.group(2) or ""
        body = (m.group(3) or "").strip()
        if name and body:
            op_defs.append((name, rf"\\operatorname{star}{{{body}}}"))
    t = re.sub(r"\\DeclareMathOperator\*?\{\\[A-Za-z]{1,40}\}\{[^{}]{1,80}\}", "", t)
    for name, repl in op_defs:
        t = re.sub(rf"\\{name}(?=[^A-Za-z]|$)", repl, t)

    # 4) Typora/KaTeX often does not recognize \argmin/\argmax directly.
    # Use explicit operator form for stable rendering.
    t = re.sub(r"\\arg\\min(?=[^A-Za-z]|$)", r"\\operatorname*{arg\\,min}", t)
    t = re.sub(r"\\arg\\max(?=[^A-Za-z]|$)", r"\\operatorname*{arg\\,max}", t)
    t = re.sub(r"\\arg\s*min(?=[^A-Za-z]|$)", r"\\operatorname*{arg\\,min}", t)
    t = re.sub(r"\\arg\s*max(?=[^A-Za-z]|$)", r"\\operatorname*{arg\\,max}", t)

    # 4.5) Typora/KaTeX does not allow \tag{} inside aligned/alignedat/split.
    # Move tag to the outer display-math level:
    #   \begin{aligned} ... \tag{10} \end{aligned}
    # ->\begin{aligned} ... \end{aligned} \tag{10}
    def _pull_tag_out_of_aligned(m: re.Match) -> str:
        env = (m.group(1) or "").strip()
        body = m.group(2) or ""
        tag_m = re.search(r"\\tag\{([^{}]{1,40})\}", body)
        if not tag_m:
            return m.group(0)
        tag_no = (tag_m.group(1) or "").strip()
        body_no_tag = re.sub(r"\s*\\tag\{[^{}]{1,40}\}\s*", " ", body)
        body_no_tag = re.sub(r"\s{2,}", " ", body_no_tag).strip()
        return rf"\begin{{{env}}} {body_no_tag} \end{{{env}}} \tag{{{tag_no}}}"

    t = re.sub(
        r"\\begin\{(aligned|alignedat|split)\}([\s\S]*?)\\end\{\1\}",
        _pull_tag_out_of_aligned,
        t,
        flags=re.IGNORECASE,
    )

    # 5) Unicode math symbols/Greek letters -> LaTeX.
    char_map: dict[str, str] = {
        "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\epsilon",
        "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta", "ι": r"\iota", "κ": r"\kappa",
        "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi", "π": r"\pi",
        "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
        "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
        "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda", "Ξ": r"\Xi",
        "Π": r"\Pi", "Σ": r"\Sigma", "Υ": r"\Upsilon", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
        "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx", "∈": r"\in", "∉": r"\notin",
        "⊂": r"\subset", "⊆": r"\subseteq", "⊃": r"\supset", "⊇": r"\supseteq",
        "∪": r"\cup", "∩": r"\cap", "∑": r"\sum", "∏": r"\prod", "∫": r"\int",
        "∂": r"\partial", "∇": r"\nabla", "∞": r"\infty", "→": r"\to", "←": r"\leftarrow",
        "↔": r"\leftrightarrow", "×": r"\times", "·": r"\cdot", "±": r"\pm", "∓": r"\mp", "√": r"\sqrt",
    }

    for ch, cmd in char_map.items():
        # If command is followed by letters, add a separating space to avoid command swallowing.
        t = re.sub(re.escape(ch) + r"(?=[A-Za-z])", lambda _m: cmd + " ", t)
        t = t.replace(ch, cmd)

    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def _normalize_math_for_typora(md: str) -> str:
    """
    Apply Typora compatibility normalization to both display and inline math.
    """
    if not md or ("\\" not in md and "$" not in md):
        return md

    lines = md.splitlines()
    out: list[str] = []
    in_fence = False
    in_display_math = False

    for ln in lines:
        st = ln.strip()
        if re.match(r"^\s*```", ln):
            in_fence = not in_fence
            out.append(ln)
            continue
        if in_fence:
            out.append(ln)
            continue
        if st == "$$":
            in_display_math = not in_display_math
            out.append(ln)
            continue

        if in_display_math:
            out.append(_normalize_typora_math_expr(ln))
            continue

        # Inline math on regular lines.
        ln2 = re.sub(
            r"\$([^$\n]{1,600})\$",
            lambda m: "$" + _normalize_typora_math_expr(m.group(1) or "") + "$",
            ln,
        )
        out.append(ln2)

    return "\n".join(out)

def _normalize_body_citations_to_superscript(md: str) -> str:
    """
    Normalize in-body numeric citations to a consistent superscript form:
      [188] -> $^{[188]}$
      [190-195] -> $^{[190-195]}$

    Keep references list entries unchanged.
    """
    if not md or "[" not in md:
        return md

    def _norm_cite(tok: str) -> str:
        t = (tok or "").strip()
        if not (t.startswith("[") and t.endswith("]")):
            return t
        inner = t[1:-1]
        inner = inner.replace("\u2013", "-")
        inner = re.sub(r"\s*[,;]\s*", ",", inner)
        inner = re.sub(r"\s*-\s*", "-", inner)
        inner = re.sub(r"\s+", "", inner)
        return f"[{inner}]"

    cite_pat = re.compile(r"\[\s*\d{1,4}(?:\s*[,;\u2013\-]\s*\d{1,4})*\s*\](?!\()")
    sup_pat = re.compile(r"\$\s*\^\{\s*(\[\s*\d{1,4}(?:\s*[,;\u2013\-]\s*\d{1,4})*\s*\])\s*\}\s*\$")
    bare_sup_pat = re.compile(r"\^\{\s*(\[\s*\d{1,4}(?:\s*[,;\u2013\-]\s*\d{1,4})*\s*\])\s*\}")

    lines = md.splitlines()
    out: list[str] = []
    in_refs = False
    in_fence = False
    in_math = False

    for ln in lines:
        st = ln.strip()
        if re.match(r"^\s*```", ln):
            in_fence = not in_fence
            out.append(ln)
            continue
        if st == "$$":
            in_math = not in_math
            out.append(ln)
            continue
        if re.match(r"^#{1,6}\s+References\b", st, re.IGNORECASE):
            in_refs = True
            out.append(ln)
            continue

        if in_refs or in_fence or in_math:
            out.append(ln)
            continue
        # Guardrail: references-like lines can appear even when heading detection misses.
        # Do not re-wrap these into superscript form again.
        try:
            leading_ref_like = bool(
                re.match(r"^\s*(?:\[\s*\d{1,4}\s*\]|\$\s*\^\{\s*\[\s*\d{1,4}\s*\]\s*\}\s*\$)", st)
            )
            if leading_ref_like:
                many_markers = len(re.findall(r"\[\s*\d{1,4}\s*\]", st)) >= 3
                has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", st))
                if many_markers or has_year:
                    out.append(ln)
                    continue
        except Exception:
            pass
        if re.match(r"^#{1,6}\s+", st) or re.match(r"^\s*!\[[^\]]*\]\([^)]+\)", st) or st.startswith("|"):
            out.append(ln)
            continue

        placeholders: dict[str, str] = {}

        def _stash(value: str) -> str:
            key = f"__CITESUP_{len(placeholders)}__"
            placeholders[key] = value
            return key

        def _sup_repl(m: re.Match) -> str:
            return _stash(f"$^{{{_norm_cite(m.group(1) or '')}}}$")

        t = ln
        # Normalize escaped citation brackets from OCR/PDF text layer: \[12-14\] -> [12-14]
        t = re.sub(r"\\\[\s*(\d{1,4}(?:\s*[,;\u2013\-]\s*\d{1,4})*)\s*\\\]", r"[\1]", t)
        t = sup_pat.sub(_sup_repl, t)
        t = bare_sup_pat.sub(_sup_repl, t)
        t = cite_pat.sub(lambda m: f"$^{{{_norm_cite(m.group(0) or '')}}}$", t)

        for k, v in placeholders.items():
            t = t.replace(k, v)
        out.append(t)

    return "\n".join(out)


def _normalize_figure_caption_blocks(md: str) -> str:
    """
    Normalize figure/table caption presentation:
    - Keep captions visually distinct from body text (italicized line).
    - If image alt contains a full caption but no visible caption line follows,
      inject one deterministic caption line after the image.
    """
    if not md:
        return md

    image_re = re.compile(r"^\s*!\[([^\]]*)\]\([^)]+\)\s*$")
    caption_re = re.compile(
        r"^\s*(?:\*{1,2}\s*)?(?:fig(?:ure)?\.?|table)\s*(?:\d+[A-Za-z]?|[IVXLC]+)\b",
        flags=re.IGNORECASE,
    )

    def _caption_id(text: str) -> Optional[str]:
        m = re.match(
            r"^\s*(?:\*{1,2}\s*)?(?:fig(?:ure)?\.?|table)\s*(\d+[A-Za-z]?|[IVXLC]+)\b",
            _normalize_text(text or "").strip(),
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        return (m.group(1) or "").strip().lower()

    def _caption_from_alt(alt: str) -> Optional[str]:
        t = _normalize_text(alt or "").strip()
        if not t:
            return None
        if not caption_re.match(t):
            return None
        # Skip generic alt text like "Figure" / "Fig. 3".
        if re.fullmatch(r"(?i)(?:figure|fig\.?|table)\s*\d*[A-Za-z]?", t):
            return None
        words = re.findall(r"[A-Za-z]{2,}", t)
        if len(words) < 5:
            return None
        if len(t) < 24:
            return None
        return t

    def _format_caption_line(line: str) -> str:
        t = (line or "").strip()
        # Remove one layer of markdown emphasis and normalize spacing.
        t = re.sub(r"^\*{1,2}\s*", "", t)
        t = re.sub(r"\s*\*{1,2}$", "", t)
        t = re.sub(r"\s{2,}", " ", t).strip()
        if not t:
            return ""
        return f"*{t}*"

    lines = md.splitlines()
    out: list[str] = []
    in_fence = False
    in_math = False
    in_refs = False
    pending_alt_caption: Optional[str] = None
    pending_alt_id: Optional[str] = None
    pending_after_image = False

    for ln in lines:
        st = ln.strip()

        if re.match(r"^\s*```", ln):
            if pending_after_image and pending_alt_caption:
                out.append(f"*{pending_alt_caption}*")
                out.append("")
            pending_alt_caption = None
            pending_alt_id = None
            pending_after_image = False
            in_fence = not in_fence
            out.append(ln)
            continue
        if st == "$$":
            if pending_after_image and pending_alt_caption:
                out.append(f"*{pending_alt_caption}*")
                out.append("")
            pending_alt_caption = None
            pending_alt_id = None
            pending_after_image = False
            in_math = not in_math
            out.append(ln)
            continue
        if re.match(r"^#{1,6}\s+References\b", st, re.IGNORECASE):
            if pending_after_image and pending_alt_caption:
                out.append(f"*{pending_alt_caption}*")
                out.append("")
            pending_alt_caption = None
            pending_alt_id = None
            pending_after_image = False
            in_refs = True
            out.append(ln)
            continue

        if in_fence or in_math or in_refs:
            out.append(ln)
            continue

        if pending_after_image:
            if not st:
                out.append(ln)
                continue
            if caption_re.match(st):
                cur_cap_id = _caption_id(st)
                # If the detected caption clearly belongs to another figure/table,
                # do not consume it for the previous image.
                if pending_alt_id and cur_cap_id and (cur_cap_id != pending_alt_id):
                    if pending_alt_caption:
                        out.append(f"*{pending_alt_caption}*")
                        out.append("")
                    pending_alt_caption = None
                    pending_alt_id = None
                    pending_after_image = False
                    # Continue processing current line normally below.
                else:
                    cap = _format_caption_line(ln)
                    if cap:
                        out.append(cap)
                    pending_alt_caption = None
                    pending_alt_id = None
                    pending_after_image = False
                    continue
            # No explicit caption followed image: inject from alt if we have one.
            if pending_after_image and pending_alt_caption:
                out.append(f"*{pending_alt_caption}*")
                out.append("")
            pending_alt_caption = None
            pending_alt_id = None
            pending_after_image = False

        m_img = image_re.match(ln)
        if m_img:
            out.append(ln)
            pending_alt_caption = _caption_from_alt(m_img.group(1) or "")
            pending_alt_id = _caption_id(m_img.group(1) or "") if pending_alt_caption else None
            pending_after_image = True
            continue

        out.append(ln)

    if pending_after_image and pending_alt_caption:
        out.append(f"*{pending_alt_caption}*")

    return "\n".join(out)

def postprocess_markdown(md: str) -> str:
    md = _cleanup_noise_lines(md)
    md = _drop_ocr_placeholder_lines(md)
    md = _fix_malformed_code_fences(md)
    md = _convert_caption_following_tabular_lines(md)
    md = _reflow_hard_wrapped_paragraphs(md)
    md = _split_inline_heading_markers(md)
    md = fix_math_markdown(md)
    md = _normalize_math_for_typora(md)
    md = _cleanup_stray_latex_in_text(md)
    md = _fix_split_numbered_headings(md)
    md = _promote_bare_numbered_headings(md)
    md = _unwrap_math_wrapped_headings(md)
    md = _enforce_heading_policy(md)
    md = _format_references(md)
    md = _normalize_figure_caption_blocks(md)
    md = _normalize_body_citations_to_superscript(md)
    md = _split_inline_heading_markers(md)
    return md
