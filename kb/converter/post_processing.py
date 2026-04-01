from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .post_heading_rules import (
    _enforce_heading_policy,
    _is_caption_heading_text,
    _is_common_section_heading,
    _is_journal_metadata_heading,
    _parse_appendix_heading_level,
    _parse_numbered_heading_level,
)
from .post_math_rules import (
    _cleanup_stray_latex_in_text,
    _normalize_math_for_typora,
    fix_math_markdown,
)
from .post_references import (
    _format_references,
    _is_post_references_resume_heading_line,
    _is_references_heading_line,
)
from .post_layout_repairs import (
    _extract_box_sidebars,
    _repair_dangling_heading_continuations,
)
from .tables import normalize_markdown_table_block
from .text_utils import _looks_like_body_figure_reference_sentence, _normalize_text





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


def _normalize_markdown_tables(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    table_buf: list[str] = []

    def flush_table() -> None:
        nonlocal table_buf
        if not table_buf:
            return
        block = "\n".join(table_buf)
        out.extend(normalize_markdown_table_block(block).splitlines())
        table_buf = []

    for line in lines:
        if line.lstrip().startswith("|"):
            table_buf.append(line)
            continue
        flush_table()
        out.append(line)

    flush_table()
    return "\n".join(out)


def _drop_standalone_journal_metadata_lines(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    out: list[str] = []
    for line in lines:
        st = (line or "").strip()
        if st and (_is_journal_metadata_heading(st) or _looks_like_running_journal_header_line(st)):
            continue
        out.append(line)
    return "\n".join(out)


def _looks_like_running_journal_header_line(text: str) -> bool:
    t = _normalize_text(text or "").strip()
    if not t:
        return False
    low = t.lower()
    has_page_marker = bool(re.search(r"\bpage\s+\d+\s+of\s+\d+\b", low))
    has_journal_name = bool(
        re.search(
            r"\b(?:light:\s*science\s*&\s*applications|optics express|natural photonics|nature photonics|advanced science news)\b",
            low,
        )
    )
    has_volume_marker = bool(re.search(r"\(\d{4}\)\s+\d+[:\s]\d+", t))
    has_author_pair = bool(re.search(r"\b[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)\b", t))
    return bool(has_page_marker and (has_journal_name or has_volume_marker or has_author_pair))

def _reflow_hard_wrapped_paragraphs(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    buf: list[str] = []

    fence_re = re.compile(r"^\s*```")
    heading_re = re.compile(r"^\s*#{1,6}\s+")
    list_re = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")
    table_re = re.compile(r"^\s*\|")
    math_marker = re.compile(r"^\s*\$\$")
    caption_like_re = re.compile(r"^\s*(?:\*{1,2}\s*)?(?:fig(?:ure)?\.?|table)\s*(?:\d+[A-Za-z]?|[A-Za-z](?:\.\d+)?|[IVXLC]+)\b", re.IGNORECASE)
    italic_line_re = re.compile(r"^\s*\*[^*].*\*\s*$")

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
            or caption_like_re.match(line)
            or italic_line_re.match(line)
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
    # Numeric table rows like "22.85 .4057 ..." are not section headings.
    numish_n = len(re.findall(r"\b\d+(?:\.\d+)?\b", head))
    word_n = len(re.findall(r"\b[A-Za-z]{2,}\b", head))
    short_alpha_n = len(re.findall(r"\b[A-Za-z]\b", head))
    if word_n == 0 and short_alpha_n >= 1:
        return False
    if any(ch in head for ch in "()[]{}") and word_n <= 1:
        return False
    if numish_n >= 3 and word_n <= 1:
        return False
    if numish_n >= 5:
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
        if _is_references_heading_line(st):
            in_refs = True
            out.append(ln)
            continue
        if in_refs and _is_post_references_resume_heading_line(st):
            in_refs = False
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


def _promote_document_title_line(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    first_content_idx: int | None = None
    for idx, line in enumerate(lines):
        if line.strip():
            first_content_idx = idx
            break
    if first_content_idx is None:
        return md

    first_line = lines[first_content_idx].strip()
    if re.match(r"^#{1,6}\s+", first_line):
        return md
    if first_line.startswith("![") or first_line.startswith("|") or first_line in {"$$", "```"}:
        return md
    if len(first_line) < 12 or len(first_line) > 220:
        return md
    if "@" in first_line:
        return md
    if _is_caption_heading_text(first_line):
        return md
    if _is_journal_metadata_heading(first_line):
        return md
    if re.search(r"\b(?:department|university|institute|school|laboratory|lab)\b", first_line, flags=re.IGNORECASE):
        return md
    if first_line.count(",") >= 3 or first_line.count(";") >= 2:
        return md
    if len(re.findall(r"[A-Za-z]{2,}", first_line)) < 4:
        return md

    saw_anchor_heading = False
    for line in lines[first_content_idx + 1 : first_content_idx + 16]:
        st = line.strip()
        if not st:
            continue
        if re.match(r"^#{1,6}\s+", st):
            title = re.sub(r"^#{1,6}\s+", "", st).strip()
            if _is_common_section_heading(title) or _parse_numbered_heading_level(title) is not None:
                saw_anchor_heading = True
                break
        if _looks_like_promotable_numbered_heading_line(st):
            saw_anchor_heading = True
            break
    if not saw_anchor_heading:
        return md

    lines[first_content_idx] = "# " + first_line
    return "\n".join(lines)


def _normalize_body_citation_markers(md: str) -> str:
    """
    Normalize in-body numeric citations to plain bracket markers:
      [ 188 ] -> [188]
      [190-195] -> [190-195]
      $^{[199, 200]}$ -> [199,200]

    Keep references list entries unchanged.
    """
    bare_citation_signal = bool(
        re.search(r"\bVOC\d{4}\s+\d{1,4}\b", md or "", flags=re.IGNORECASE)
        or re.search(r"\b(?:transformer\s+)?framework\s+\d{1,4}\s*,\s*\d{1,4}\b", md or "", flags=re.IGNORECASE)
        or re.search(r"\btasks?\s+\d{1,4}\s*[\u2013\u2014\-]\s*\d{1,4}\b", md or "", flags=re.IGNORECASE)
    )
    if not md or (
        not any(tok in md for tok in ("[", "<sup", "^{", "\\textsuperscript"))
        and not re.search(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]", md)
        and not bare_citation_signal
    ):
        return md

    def _norm_cite_inner(inner: str) -> str:
        inner = (inner or "").strip()
        inner = inner.replace("\u2013", "-")
        inner = inner.replace("\u2014", "-")
        inner = re.sub(r"\s*[,;]\s*", ",", inner)
        inner = re.sub(r"\s*-\s*", "-", inner)
        inner = re.sub(r"\s+", "", inner)
        return inner

    def _norm_cite(tok: str) -> str:
        t = (tok or "").strip()
        if t.startswith("[") and t.endswith("]"):
            inner = t[1:-1]
        else:
            inner = t
        inner = _norm_cite_inner(inner)
        return f"[{inner}]"

    cite_pat = re.compile(r"\[\s*\d{1,4}(?:\s*[,;\u2013\-]\s*\d{1,4})*\s*\](?!\()")
    sup_pat = re.compile(r"\$\s*\^\{\s*(\[\s*\d{1,4}(?:\s*[,;\u2013\-]\s*\d{1,4})*\s*\])\s*\}\s*\$")
    bare_sup_pat = re.compile(r"\^\{\s*(\[\s*\d{1,4}(?:\s*[,;\u2013\-]\s*\d{1,4})*\s*\])\s*\}")
    html_sup_pat = re.compile(r"<sup>\s*(\d{1,4}(?:\s*[,;\u2013\u2014\-]\s*\d{1,4})*)\s*</sup>", re.IGNORECASE)
    math_plain_sup_pat = re.compile(r"\$\s*\^\{\s*(\d{1,4}(?:\s*[,;\u2013\u2014\-]\s*\d{1,4})*)\s*\}\s*\$")
    bare_plain_sup_pat = re.compile(r"(?<!\w)\^\{\s*(\d{1,4}(?:\s*[,;\u2013\u2014\-]\s*\d{1,4})*)\s*\}")
    latex_textsup_pat = re.compile(r"\\textsuperscript\{\s*(\d{1,4}(?:\s*[,;\u2013\u2014\-]\s*\d{1,4})*)\s*\}")

    sup_digit_map = str.maketrans({
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
    })
    unicode_sup_pat = re.compile(r"([⁰¹²³⁴⁵⁶⁷⁸⁹]{1,6})")

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
        if _is_references_heading_line(st):
            in_refs = True
            out.append(ln)
            continue
        if in_refs and _is_post_references_resume_heading_line(st):
            in_refs = False

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
            key = f"__CITEBR_{len(placeholders)}__"
            placeholders[key] = value
            return key

        def _sup_repl(m: re.Match) -> str:
            return _stash(_norm_cite(m.group(1) or ""))

        t = ln
        # Normalize escaped citation brackets from OCR/PDF text layer: \[12-14\] -> [12-14]
        t = re.sub(r"\\\[\s*(\d{1,4}(?:\s*[,;\u2013\-]\s*\d{1,4})*)\s*\\\]", r"[\1]", t)
        t = sup_pat.sub(_sup_repl, t)
        t = bare_sup_pat.sub(_sup_repl, t)
        t = html_sup_pat.sub(lambda m: _stash(_norm_cite(m.group(1) or "")), t)
        t = math_plain_sup_pat.sub(lambda m: _stash(_norm_cite(m.group(1) or "")), t)
        t = bare_plain_sup_pat.sub(lambda m: _stash(_norm_cite(m.group(1) or "")), t)
        t = latex_textsup_pat.sub(lambda m: _stash(_norm_cite(m.group(1) or "")), t)
        t = cite_pat.sub(lambda m: _norm_cite(m.group(0) or ""), t)
        t = unicode_sup_pat.sub(lambda m: _stash(_norm_cite((m.group(1) or "").translate(sup_digit_map))), t)
        # Some VL outputs drop the bracket/superscript wrapper entirely and leave
        # citation numbers inline, e.g. "VOC2007 31 and VOC2012 32 datasets" or
        # "transformer framework 33 , 34". Keep this intentionally narrow so we
        # do not rewrite ordinary prose numbers.
        t = re.sub(
            r"\b(VOC\d{4})\s+(\d{1,4})(?=\s+(?:and\b|datasets?\b|images?\b|data\b))",
            lambda m: f"{m.group(1)} {_stash(_norm_cite(m.group(2) or ''))}",
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(
            r"\b((?:transformer\s+)?framework)\s+(\d{1,4})\s*,\s*(\d{1,4})(?=\s+[A-Za-z])",
            lambda m: f"{m.group(1)} {_stash(_norm_cite(f'{m.group(2)},{m.group(3)}'))}",
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(
            r"\b(tasks?)\s+(\d{1,4})\s*[\u2013\u2014\-]\s*(\d{1,4})(?=\s*(?:[.,;:]\s*)?[A-Za-z])",
            lambda m: f"{m.group(1)} {_stash(_norm_cite(f'{m.group(2)}-{m.group(3)}'))}",
            t,
            flags=re.IGNORECASE,
        )

        for k, v in placeholders.items():
            t = t.replace(k, v)
        t = re.sub(
            r"(?<=[A-Za-z0-9\)\*])(\[(?:\d{1,4}(?:[,\-]\d{1,4})*)\])",
            r" \1",
            t,
        )
        out.append(t)

    return "\n".join(out)


def _merge_obvious_body_continuation_lines(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    out: list[str] = []
    in_refs = False
    in_fence = False
    in_math = False
    image_re = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")
    caption_re = re.compile(r"^\s*\*\*(?:Figure|Table)\s+[A-Za-z0-9]+\.\*\*")
    heading_re = re.compile(r"^\s*#{1,6}\s+")
    table_re = re.compile(r"^\s*\|")
    ref_entry_re = re.compile(r"^\s*\[\s*\d{1,4}\s*\]\s+")

    def _is_body_line(s: str) -> bool:
        st = (s or "").strip()
        if not st:
            return False
        if image_re.match(st) or caption_re.match(st) or heading_re.match(st) or table_re.match(st):
            return False
        if re.match(r"^\*[^*].*\*$", st):
            return False
        if ref_entry_re.match(st):
            return False
        if re.match(r"^\s*```", st) or st == "$$":
            return False
        if _looks_like_formulaish_heading_text(st):
            return False
        return True

    def _looks_incomplete_prefix(s: str) -> bool:
        st = _normalize_text(s or "").strip()
        if not _is_body_line(st):
            return False
        if st.endswith("-"):
            return True
        if st.endswith((".", "!", "?", ":", ";")):
            return False
        return bool(
            re.search(
                r"\b(?:the|a|an|this|that|these|those|of|with|for|from|to|in|on|by|as|despite|due|because|same|see)\s*$",
                st,
                flags=re.IGNORECASE,
            )
        )

    def _looks_continuation(s: str) -> bool:
        st = _normalize_text(s or "").strip()
        if not _is_body_line(st):
            return False
        return bool(
            re.match(r'^[a-z(\"\']', st)
            or re.match(r"^(?:absence|square|which|that|while|where|with|and|or|for|to|in|on|by)\b", st)
            or re.match(r"^(?:voc\d{4}|pascal)\b", st, flags=re.IGNORECASE)
        )

    def _join_lines(left: str, right: str) -> str:
        l = (left or "").rstrip()
        r = (right or "").lstrip()
        if l.endswith("-"):
            return l[:-1] + r
        return (l + " " + r).strip()

    i = 0
    while i < len(lines):
        ln = lines[i]
        st = ln.strip()
        if re.match(r"^\s*```", ln):
            in_fence = not in_fence
            out.append(ln)
            i += 1
            continue
        if st == "$$":
            in_math = not in_math
            out.append(ln)
            i += 1
            continue
        if _is_references_heading_line(st):
            in_refs = True
            out.append(ln)
            i += 1
            continue
        if in_refs and _is_post_references_resume_heading_line(st):
            in_refs = False
        if in_refs or in_fence or in_math or not _is_body_line(ln):
            out.append(ln)
            i += 1
            continue

        merged = ln
        j = i + 1
        while True:
            blank_count = 0
            k = j
            while k < len(lines) and not (lines[k] or "").strip():
                blank_count += 1
                k += 1
            if blank_count > 1 or k >= len(lines) or not _is_body_line(lines[k]):
                break
            if not (_looks_incomplete_prefix(merged) or _looks_continuation(lines[k])):
                break
            merged = _join_lines(merged, lines[k])
            j = k + 1
        out.append(merged)
        i = j

    return "\n".join(out)


def _repair_sentence_split_by_figure_blocks(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    out: list[str] = []
    image_re = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")
    caption_re = re.compile(r"^\s*\*\*(?:Figure|Table)\s+[A-Za-z0-9]+\.\*\*")
    heading_re = re.compile(r"^\s*#{1,6}\s+")
    table_re = re.compile(r"^\s*\|")

    def _is_body_line(s: str) -> bool:
        st = (s or "").strip()
        if not st:
            return False
        if image_re.match(st) or caption_re.match(st) or heading_re.match(st) or table_re.match(st):
            return False
        if re.match(r"^\*[^*].*\*$", st):
            return False
        if re.match(r"^\s*```", st) or st == "$$":
            return False
        return True

    def _looks_incomplete_prefix(s: str) -> bool:
        st = _normalize_text(s or "").strip()
        if not _is_body_line(st):
            return False
        if st.endswith((".", "!", "?", ":", ";")):
            return False
        if st.endswith("-"):
            return True
        return bool(re.search(r"\b(?:the|a|an|this|that|these|those|of|with|for|from|to|in|on|by|as|despite|due|because)\s*$", st, flags=re.IGNORECASE))

    def _looks_continuation(s: str) -> bool:
        st = _normalize_text(s or "").strip()
        if not _is_body_line(st):
            return False
        return bool(
            re.match(r'^[a-z(\"\']', st)
            or re.match(r"^(?:absence|square|which|that|while|where|with|and|or|for|to|in|on|by)\b", st)
            or re.match(r"^(?:voc\d{4}|pascal)\b", st, flags=re.IGNORECASE)
        )

    def _join_lines(left: str, right: str) -> str:
        l = (left or "").rstrip()
        r = (right or "").lstrip()
        if l.endswith("-"):
            return l[:-1] + r
        return (l + " " + r).strip()

    i = 0
    while i < len(lines):
        st = (lines[i] or "").strip()
        if not image_re.match(st):
            out.append(lines[i])
            i += 1
            continue

        block: list[str] = [lines[i]]
        j = i + 1
        while j < len(lines):
            s = (lines[j] or "").strip()
            if (not s) or caption_re.match(s):
                block.append(lines[j])
                j += 1
                continue
            break

        next_idx = j
        while next_idx < len(lines) and not (lines[next_idx] or "").strip():
            next_idx += 1
        if (not out) or next_idx >= len(lines) or (not _looks_continuation(lines[next_idx])):
            out.extend(block)
            i = j
            continue

        prev_idx = len(out) - 1
        while prev_idx >= 0 and not (out[prev_idx] or "").strip():
            prev_idx -= 1
        if prev_idx < 0 or not _looks_incomplete_prefix(out[prev_idx]):
            out.extend(block)
            i = j
            continue

        para_lines: list[str] = []
        k = next_idx
        while k < len(lines):
            s = (lines[k] or "").strip()
            if not s:
                break
            if not _is_body_line(s):
                break
            para_lines.append(lines[k])
            k += 1
        if not para_lines:
            out.extend(block)
            i = j
            continue

        out[prev_idx] = _join_lines(out[prev_idx], para_lines[0])
        out.extend(para_lines[1:])
        if out and out[-1].strip():
            out.append("")
        out.extend(block)
        i = k
        continue

    return "\n".join(out)


def _looks_like_formulaish_heading_text(text: str) -> bool:
    t = _normalize_text(text or "").strip()
    if not t:
        return False
    mathish = bool(re.search(r"[=^_{}\\|<>~∑⊛⊙∘¼½¾×÷±·⋅]", t))
    controlish = bool(re.search(r"[\x00-\x1f\ufffd]", t))
    if _is_common_section_heading(t):
        return False
    if _parse_numbered_heading_level(t) is not None:
        return False
    if _is_caption_heading_text(t):
        return False
    if not (mathish or controlish):
        if _parse_appendix_heading_level(t) is not None:
            return False
        return False
    word_count = len(re.findall(r"[A-Za-z]{2,}", t))
    weird_count = sum(1 for ch in t if not (ch.isalnum() or ch.isspace() or ch in "-,:;()'./&[]"))
    return bool(word_count <= 6 or weird_count >= 3)


def _demote_formulaish_headings(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    out: list[str] = []
    in_fence = False
    in_math = False
    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")

    def _next_non_empty(idx: int) -> str:
        for j in range(idx + 1, len(lines)):
            st = (lines[j] or "").strip()
            if st:
                return st
        return ""

    for idx, ln in enumerate(lines):
        st = ln.strip()
        if re.match(r"^\s*```", ln):
            in_fence = not in_fence
            out.append(ln)
            continue
        if st == "$$":
            in_math = not in_math
            out.append(ln)
            continue
        if in_fence or in_math:
            out.append(ln)
            continue
        m = heading_re.match(ln)
        if not m:
            out.append(ln)
            continue
        title = (m.group(2) or "").strip()
        if not _looks_like_formulaish_heading_text(title):
            out.append(ln)
            continue
        next_st = _normalize_text(_next_non_empty(idx))
        if next_st and (next_st[:1].islower() or re.match(r"^(?:with|where|and|or|for|to|in|on|by)\b", next_st, flags=re.IGNORECASE)):
            out.append(title)
            continue
        out.append(title)
    return "\n".join(out)


def _normalize_common_rendering_artifacts(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    out: list[str] = []
    in_fence = False
    unit_letters = "mWlLsSA"
    for ln in lines:
        if re.match(r"^\s*```", ln):
            in_fence = not in_fence
            out.append(ln)
            continue
        if in_fence:
            out.append(ln)
            continue
        t = unicodedata.normalize("NFKC", ln)
        t = t.replace("\\mum", "\\mu m")
        t = t.replace("掳C", "°C")
        t = re.sub(
            rf"\\mu\s+(?!\\mathrm\{{)([{unit_letters}])\b",
            lambda m: f"\\mu\\mathrm{{{m.group(1)}}}",
            t,
        )
        t = re.sub(
            rf"\\mu\$(?!\\mathrm\{{)([{unit_letters}])\b",
            lambda m: f"\\mu\\mathrm{{{m.group(1)}}}$",
            t,
        )
        t = re.sub(
            rf"(?<!\$)\b(\d+(?:\.\d+)?)\s*μ\s*([{unit_letters}])\b",
            lambda m: f"${m.group(1)}\\,\\mu\\mathrm{{{m.group(2)}}}$",
            t,
        )
        t = re.sub(
            rf"(?<!\$)\b(\d+(?:\.\d+)?)\s*μm\b",
            lambda m: f"${m.group(1)}\\,\\mu\\mathrm{{m}}$",
            t,
        )
        t = re.sub(
            rf"\$(\d+(?:\.\d+)?)\s*\\,?\s*\\mu(?:\\mathrm\{{([{unit_letters}])\}})?\$\s*([{unit_letters}])\b",
            lambda m: f"${m.group(1)}\\,\\mu\\mathrm{{{m.group(2) or m.group(3)}}}$",
            t,
        )
        t = re.sub(
            rf"(?<!\$)\b(\d+(?:\.\d+)?)\s+\$\\mu(?:\\mathrm\{{([{unit_letters}])\}})?\$\s*([{unit_letters}])\b",
            lambda m: f"${m.group(1)}\\,\\mu\\mathrm{{{m.group(2) or m.group(3)}}}$",
            t,
        )
        t = re.sub(
            rf"(?<!\$)\b(\d+(?:\.\d+)?)\s+\$\\mu\\mathrm\{{([{unit_letters}])\}}\$",
            lambda m: f"${m.group(1)}\\,\\mu\\mathrm{{{m.group(2)}}}$",
            t,
        )
        t = re.sub(r"\biISM-\s+APR\b", "iISM-APR", t)
        t = re.sub(r"\b([A-Za-z]{3,})-\s+([a-z]{2,})\b", lambda m: m.group(1) + m.group(2), t)
        t = re.sub(r"\b([A-Za-z]{2,})\s+fi\s+([A-Za-z]{2,})\b", lambda m: m.group(1) + "fi" + m.group(2), t)
        t = re.sub(r"\b([A-Za-z]{2,})\s+fl\s+([A-Za-z]{2,})\b", lambda m: m.group(1) + "fl" + m.group(2), t)
        t = re.sub(r"\b([A-Za-z]{2,})-\s*fi\s+([A-Za-z]{2,})\b", lambda m: m.group(1) + "-fi" + m.group(2), t)
        t = re.sub(r"\b([A-Za-z]{2,})-\s*fl\s+([A-Za-z]{2,})\b", lambda m: m.group(1) + "-fl" + m.group(2), t)
        t = re.sub(r"\bonflat-fielded\b", "on flat-fielded", t)
        t = re.sub(r"\b(Fig\.|Figure)\s+(\d+)\s+([a-z])\b", r"\1 \2\3", t)
        t = re.sub(r"\biISM\s+[’']\s+s\b", "iISM's", t)
        t = re.sub(r"(?<=[A-Za-z])\((see\s+(?:eq|fig(?:ure)?)\b)", lambda m: " (" + m.group(1), t, flags=re.IGNORECASE)
        t = re.sub(r"\(\s*([A-Za-z])\s*(\d+)\s*[−-]\s*([A-Za-z])\s*(\d+)\s*\)", lambda m: f"({m.group(1)}{m.group(2)}–{m.group(3)}{m.group(4)})" if m.group(1).lower() == m.group(3).lower() else m.group(0), t)
        out.append(t)
    return "\n".join(out)


def _normalize_figure_caption_blocks(md: str) -> str:
    """
    Normalize figure/table caption presentation:
    - Use a canonical Markdown caption form for figures/tables:
      `**Figure 3.** ...` / `**Table A.** ...`
    - If image alt contains a full caption but no visible caption line follows,
      inject one deterministic caption line after the image.
    """
    if not md:
        return md

    image_re = re.compile(r"^\s*!\[([^\]]*)\]\([^)]+\)\s*$")
    caption_re = re.compile(
        r"^\s*(?:\*{1,2}\s*)?(?:fig(?:ure)?\.?|table)\s*(?:\d+[A-Za-z]?|[A-Za-z](?:\.\d+)?|[IVXLC]+)\b",
        flags=re.IGNORECASE,
    )

    def _caption_id(text: str) -> Optional[str]:
        m = re.match(
            r"^\s*(?:\*{1,2}\s*)?(?:fig(?:ure)?\.?|table)\s*(\d+[A-Za-z]?|[A-Za-z](?:\.\d+)?|[IVXLC]+)\b",
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
        t = _normalize_text(line or "").strip()
        # Remove one layer of markdown emphasis and normalize spacing.
        t = re.sub(r"^\*{1,2}\s*", "", t)
        t = re.sub(r"\s*\*{1,2}$", "", t)
        t = re.sub(r"(?i)\b((?:fig(?:ure)?\.?|table)\s*[A-Za-z0-9]+)\.\s*\*{1,2}\s*", r"\1. ", t)
        t = re.sub(r"\.\*{1,2}\s+\*{1,2}([a-z])\*\*", r". **\1**", t, flags=re.IGNORECASE)
        t = re.sub(r"\s{2,}", " ", t).strip()
        if not t:
            return ""
        m = re.match(
            r"^(fig(?:ure)?\.?|table)\s*([A-Za-z0-9]+)\s*(?:[.:]\s*|\-\s*|\s+)?(.*)$",
            t,
            flags=re.IGNORECASE,
        )
        if not m:
            return t
        kind_raw = (m.group(1) or "").strip().lower()
        ident = (m.group(2) or "").strip()
        tail = (m.group(3) or "").strip()
        tail = re.sub(r"^\*{1,2}\s*", "", tail).strip()
        tail = re.sub(r"^[\|\s\.\:\-]+", "", tail).strip()
        kind = "Table" if kind_raw.startswith("table") else "Figure"
        if tail:
            return f"**{kind} {ident}.** {tail}"
        return f"**{kind} {ident}.**"

    lines = md.splitlines()
    out: list[str] = []
    in_fence = False
    in_math = False
    in_refs = False
    pending_alt_caption: Optional[str] = None
    pending_alt_id: Optional[str] = None
    pending_after_image = False
    last_image_line: Optional[str] = None

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
        if _is_references_heading_line(st):
            if pending_after_image and pending_alt_caption:
                out.append(f"*{pending_alt_caption}*")
                out.append("")
            pending_alt_caption = None
            pending_alt_id = None
            pending_after_image = False
            in_refs = True
            out.append(ln)
            continue
        if in_refs and _is_post_references_resume_heading_line(st):
            in_refs = False

        if in_fence or in_math or in_refs:
            out.append(ln)
            continue

        if pending_after_image:
            if not st:
                out.append(ln)
                continue
            if caption_re.match(st) and (not _looks_like_body_figure_reference_sentence(st)):
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
                out.append(_format_caption_line(pending_alt_caption))
                out.append("")
            pending_alt_caption = None
            pending_alt_id = None
            pending_after_image = False

        m_img = image_re.match(ln)
        if m_img:
            norm_img = st
            if last_image_line and norm_img == last_image_line:
                continue
            out.append(ln)
            pending_alt_caption = _caption_from_alt(m_img.group(1) or "")
            pending_alt_id = _caption_id(m_img.group(1) or "") if pending_alt_caption else None
            pending_after_image = True
            last_image_line = norm_img
            continue

        if st:
            last_image_line = None

        if caption_re.match(st) and (not _looks_like_body_figure_reference_sentence(st)):
            cap = _format_caption_line(ln)
            if cap:
                out.append(cap)
                continue

        out.append(ln)

    if pending_after_image and pending_alt_caption:
        out.append(_format_caption_line(pending_alt_caption))

    return "\n".join(out)


def _repair_body_figure_reference_captions(md: str) -> str:
    if not md:
        return md
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
        if in_fence or in_math:
            out.append(ln)
            continue
        if not _looks_like_body_figure_reference_sentence(st):
            out.append(ln)
            continue
        repaired = _normalize_text(st)
        repaired = re.sub(r"^\*{1,2}\s*", "", repaired)
        repaired = re.sub(r"\s*\*{1,2}\s*", "", repaired)
        repaired = re.sub(
            r"(?i)^(figure|fig\.?)\s*(\d+[A-Za-z])\.\s*,\s*",
            lambda m: f"Figure {m.group(2)}, ",
            repaired,
        )
        repaired = re.sub(r"(?i)^fig\.\s*", "Figure ", repaired)
        out.append(repaired)
    return "\n".join(out)


def _dedupe_nearby_repeated_captions(md: str) -> str:
    if not md:
        return md

    lines = md.splitlines()
    caption_re = re.compile(r"^\s*\*\*(Figure|Table)\s+([A-Za-z0-9]+)\.\*\*\s*(.*)$", re.IGNORECASE)
    image_re = re.compile(r"^\s*!\[([^\]]*)\]\([^)]+\)\s*$")

    def _caption_key(line: str) -> tuple[str, str, str] | None:
        m = caption_re.match((line or "").strip())
        if not m:
            return None
        kind = (m.group(1) or "").strip().lower()
        ident = (m.group(2) or "").strip().lower()
        tail = re.sub(r"\s+", " ", _normalize_text(m.group(3) or "").strip()).lower()
        return kind, ident, tail

    def _caption_related_continuation(idx: int) -> str:
        parts: list[str] = []
        for k in range(idx + 1, min(len(lines), idx + 5)):
            st = (lines[k] or "").strip()
            if not st:
                if parts:
                    break
                continue
            if image_re.match(st) or caption_re.match(st) or re.match(r"^#{1,6}\s+", st) or st.startswith("|"):
                break
            parts.append(_normalize_text(st))
        return " ".join(parts).strip()

    def _caption_quality(idx: int) -> int:
        key = _caption_key(lines[idx])
        if key is None:
            return -10_000
        _, _, tail = key
        continuation = _caption_related_continuation(idx)
        blob = " ".join(x for x in [tail, continuation] if x).strip()
        score = len(blob)
        score += 40 if continuation else 0
        score -= 80 * len(re.findall(r"[脳渭鈥脦膹藛蠁�]", blob))
        return score

    def _tails_look_duplicate(t1: str, t2: str) -> bool:
        if t1 == t2:
            return True
        if t1 and t2 and (t1.startswith(t2) or t2.startswith(t1)):
            return True
        words1 = set(re.findall(r"[a-z]{4,}", t1))
        words2 = set(re.findall(r"[a-z]{4,}", t2))
        if len(words1) < 3 or len(words2) < 3:
            return False
        overlap = len(words1 & words2)
        return overlap >= 4 and overlap / max(1, min(len(words1), len(words2))) >= 0.6

    def _image_ident(line: str) -> tuple[str, str] | None:
        m = image_re.match((line or "").strip())
        if not m:
            return None
        alt = _normalize_text(m.group(1) or "").strip()
        m2 = re.match(r"^(Figure|Table)\s+([A-Za-z0-9]+)\b", alt, re.IGNORECASE)
        if not m2:
            return None
        return (m2.group(1) or "").strip().lower(), (m2.group(2) or "").strip().lower()

    drop: set[int] = set()
    caption_idxs = [i for i, ln in enumerate(lines) if _caption_key(ln) is not None]
    for pos, i in enumerate(caption_idxs):
        if i in drop:
            continue
        key_i = _caption_key(lines[i])
        if key_i is None:
            continue
        for j in caption_idxs[pos + 1 :]:
            if j - i > 12:
                break
            if j in drop:
                continue
            key_j = _caption_key(lines[j])
            if key_j is None:
                continue
            if key_j[:2] != key_i[:2]:
                continue
            if not _tails_look_duplicate(key_i[2], key_j[2]):
                continue
            kind, ident, _ = key_i
            matching_images = [
                k
                for k in range(i + 1, j)
                if _image_ident(lines[k]) == (kind, ident)
            ]
            if matching_images:
                drop.add(i)
            else:
                if _caption_quality(j) >= _caption_quality(i):
                    drop.add(i)
                else:
                    drop.add(j)
            break

    if not drop:
        return md
    return "\n".join(ln for idx, ln in enumerate(lines) if idx not in drop)


def _normalize_plural_figure_reference_lines(md: str) -> str:
    if not md:
        return md
    out = md
    out = re.sub(r"(?mi)^\s*\*\*Figure s\.\*\*\s*", "Figures ", out)
    out = re.sub(r"(?mi)^\s*\*\*Figures\.\*\*\s*", "Figures ", out)
    return out


def _fill_empty_image_alt_from_following_caption(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    image_re = re.compile(r"^\s*!\[\s*\]\(([^)]+)\)\s*$")
    caption_re = re.compile(r"^\s*\*\*(Figure|Table)\s+([A-Za-z0-9]+)\.\*\*")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m_img = image_re.match(line)
        if not m_img:
            out.append(line)
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            m_cap = caption_re.match(lines[j].strip())
            if m_cap:
                kind = m_cap.group(1)
                ident = m_cap.group(2)
                out.append(f"![{kind} {ident}]({m_img.group(1)})")
                i += 1
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _tighten_image_caption_spacing(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    image_re = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")
    caption_re = re.compile(r"^\s*\*\*(?:Figure|Table)\s+[A-Za-z0-9]+\.\*\*")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if not image_re.match(line):
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines) and caption_re.match(lines[j].strip()):
            out.append("")
            i = j
            continue
        i += 1
    return "\n".join(out)


def _pull_images_to_matching_previous_captions(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    image_re = re.compile(r"^\s*!\[(Figure|Table)\s+([A-Za-z0-9]+)\]\([^)]+\)\s*$")
    caption_re = re.compile(r"^\s*\*\*(Figure|Table)\s+([A-Za-z0-9]+)\.\*\*")
    heading_re = re.compile(r"^\s*#{1,6}\s+")
    table_re = re.compile(r"^\s*\|")

    def _has_matching_caption_below(start: int, kind: str, ident: str) -> bool:
        non_empty = 0
        j = start + 1
        while j < len(lines):
            raw = lines[j] or ""
            s = raw.strip()
            if not s:
                j += 1
                continue
            if heading_re.match(raw) or image_re.match(raw) or table_re.match(raw):
                return False
            m_cap = caption_re.match(s)
            if m_cap:
                return m_cap.group(1).lower() == kind and m_cap.group(2).lower() == ident
            non_empty += 1
            if non_empty > 2:
                return False
            j += 1
        return False

    changed = False
    i = 0
    while i < len(lines):
        raw = lines[i] or ""
        m_img = image_re.match(raw)
        if not m_img:
            i += 1
            continue
        kind = (m_img.group(1) or "").strip().lower()
        ident = (m_img.group(2) or "").strip().lower()
        if _has_matching_caption_below(i, kind, ident):
            i += 1
            continue

        non_empty = 0
        cap_idx = None
        j = i - 1
        while j >= 0:
            prev_raw = lines[j] or ""
            s = prev_raw.strip()
            if not s:
                j -= 1
                continue
            if image_re.match(prev_raw) or heading_re.match(prev_raw) or table_re.match(prev_raw):
                break
            m_cap = caption_re.match(s)
            if m_cap:
                if m_cap.group(1).lower() == kind and m_cap.group(2).lower() == ident:
                    cap_idx = j
                break
            non_empty += 1
            if non_empty > 6:
                break
            j -= 1

        if cap_idx is None or i <= cap_idx + 1:
            i += 1
            continue

        image_line = lines.pop(i)
        lines.insert(cap_idx, image_line)
        changed = True
        i = cap_idx + 1

    if not changed:
        return md
    return "\n".join(lines)


def _normalize_structural_labels(md: str) -> str:
    if not md:
        return md
    replacements = {
        r"(?mi)^#{1,6}\s+A\s+B\s+S\s+T\s+R\s+A\s+C\s+T\s*$": "## Abstract",
        r"(?mi)^#{1,6}\s+R\s+E\s+F\s+E\s+R\s+E\s+N\s+C\s+E\s+S\s*$": "## References",
        r"(?mi)^#{1,6}\s+K\s+E\s+Y\s+W\s+O\s+R\s+D\s+S\s*$": "## Keywords",
    }
    out = md
    for pat, repl in replacements.items():
        out = re.sub(pat, repl, out)
    return out


def _strip_frontmatter_metadata_lines(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    out: list[str] = []
    heading_seen = False
    heading_re = re.compile(r"^\s*#{1,6}\s+\S")
    drop_frontmatter_re = re.compile(
        r"^(?:"
        r"Article|"
        r"https?://doi\.org/\S+|"
        r"doi:\S+|"
        r"Published online:.*|"
        r"Received:\s+.*|"
        r"Accepted:\s+.*|"
        r"Check for updates|"
        r"Optics\s*&\s*Laser\s*Technology\s+\d+\s*\(\d{4}\)\s*\d+|"
        r"Vol\.\s*\d+\s*,\s*No\.\s*\d+\s*\|.*OPTICS\s+EXPRESS.*|"
        r".*Optical Society of America.*|"
        r".*\bOCIS codes:\b.*"
        r")$",
        flags=re.IGNORECASE,
    )
    for idx, line in enumerate(lines):
        st = line.strip()
        if heading_re.match(st):
            heading_seen = True
        if idx < 30 and drop_frontmatter_re.match(st):
            continue
        if idx < 30 and st == "---":
            continue
        out.append(line)
    return "\n".join(out)


def _demote_panel_headings(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    out: list[str] = []
    for line in lines:
        m = re.match(r"^(#{2,6})\s+(.+)$", line)
        if not m:
            out.append(line)
            continue
        title = (m.group(2) or "").strip()
        if re.match(r"^[a-z]\s+[A-Z][A-Za-z0-9].{3,120}$", title):
            out.append(title)
            continue
        out.append(line)
    return "\n".join(out)


def _rebalance_custom_headings_within_structural_sections(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    out: list[str] = []
    current_structural_level: int | None = None
    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if not m:
            out.append(line)
            continue
        level = len(m.group(1))
        title = (m.group(2) or "").strip()
        is_numbered = _parse_numbered_heading_level(title) is not None
        is_appendix = _parse_appendix_heading_level(title) is not None
        is_structural = _is_common_section_heading(title)

        if is_structural or is_numbered or is_appendix:
            current_structural_level = level
            out.append(line)
            continue

        if current_structural_level is not None:
            if level <= current_structural_level:
                new_level = min(6, current_structural_level + 1)
                out.append("#" * new_level + " " + title)
                continue
            if level > current_structural_level + 1:
                out.append("#" * min(6, current_structural_level + 1) + " " + title)
                continue
        out.append(line)
    return "\n".join(out)


def _drop_empty_figure_duplicate_headings(md: str) -> str:
    if not md:
        return md

    stop_words = {
        "the", "and", "for", "with", "from", "into", "onto", "this", "that", "these", "those",
        "reported", "their", "then", "than", "while", "where", "which", "using", "used", "use",
        "over", "under", "between", "through", "about", "network",
    }

    def _title_words(text: str) -> set[str]:
        norm = _normalize_text(text or "").lower()
        words = re.findall(r"[a-z]{4,}", norm)
        return {w for w in words if w not in stop_words}

    def _looks_custom_heading(title: str) -> bool:
        if not title:
            return False
        if _is_common_section_heading(title):
            return False
        if _parse_numbered_heading_level(title) is not None:
            return False
        if _parse_appendix_heading_level(title) is not None:
            return False
        if _is_caption_heading_text(title):
            return False
        return True

    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if not m:
            out.append(line)
            i += 1
            continue
        if i == 0:
            out.append(line)
            i += 1
            continue

        title = (m.group(2) or "").strip()
        title_words = _title_words(title)
        if not _looks_custom_heading(title) or len(title_words) < 3:
            out.append(line)
            i += 1
            continue

        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            out.append(line)
            i += 1
            continue
        first_following = (lines[j] or "").strip()
        if not (
            re.match(r"^#{1,6}\s+", first_following)
            or first_following.startswith("![")
            or re.match(r"^\*\*(?:Figure|Fig\.?|Table)\b", first_following, re.IGNORECASE)
        ):
            out.append(line)
            i += 1
            continue

        window_end = min(len(lines), i + 40)
        saw_image = False
        duplicate_caption = False
        for k in range(j, window_end):
            st = (lines[k] or "").strip()
            if not st:
                continue
            if st.startswith("!["):
                saw_image = True
                continue
            if re.match(r"^#{1,6}\s+", st):
                if k == j:
                    continue
                continue
            if re.match(r"^\*\*(?:Figure|Fig\.?|Table)\b", st, re.IGNORECASE):
                cand_words = _title_words(st)
                if len(cand_words) >= 3:
                    overlap = len(title_words & cand_words)
                    if overlap >= 3 and overlap / max(1, len(title_words)) >= 0.5:
                        duplicate_caption = True
                continue
            cand_words = _title_words(st)
            if len(cand_words) < 3:
                continue
            overlap = len(title_words & cand_words)
            if overlap >= 3 and overlap / max(1, len(title_words)) >= 0.5:
                duplicate_caption = True
        if saw_image and duplicate_caption:
            i += 1
            continue

        out.append(line)
        i += 1
    return "\n".join(out)


def _cleanup_author_superscript_noise(md: str) -> str:
    if not md:
        return md
    def _digits_only(raw: str) -> str:
        return re.sub(r"\s+", "", raw or "")
    out = md
    out = re.sub(r"\$\^\{ID\s*([0-9,\s]+)\}\$", lambda m: "$^{" + _digits_only(m.group(1)) + "}$", out)
    out = re.sub(r"\^\{ID\s*([0-9,\s]+)\}", lambda m: "^{" + _digits_only(m.group(1)) + "}", out)
    return out


def _insert_missing_abstract_heading(md: str) -> str:
    if not md:
        return md
    if re.search(r"(?mi)^#{1,6}\s+Abstract\s*$", md):
        return md
    lines = md.splitlines()
    out: list[str] = []
    inserted = False
    first_h2_seen = False
    long_para_re = re.compile(r"[A-Za-z]{4,}")
    for idx, line in enumerate(lines):
        st = line.strip()
        if re.match(r"^##\s+", st):
            first_h2_seen = True
        if (
            (not inserted)
            and (not first_h2_seen)
            and idx > 0
            and len(st) >= 240
            and long_para_re.search(st)
            and not st.startswith("![")
            and not st.startswith("|")
            and not st.startswith("**Figure")
        ):
            prev_nonempty = next((x.strip() for x in reversed(out) if x.strip()), "")
            if prev_nonempty and re.match(r"^(?:#\s+.+|.+\$\^\{?.+|.+&.+|Published online:.*)$", prev_nonempty):
                out.append("## Abstract")
                out.append("")
                inserted = True
        out.append(line)
    return "\n".join(out)


def _drop_abstract_leading_metadata_lines(md: str) -> str:
    if not md:
        return md

    aff_kw_re = re.compile(
        r"\b("
        r"dept\.?|department|school|college|university|institute|laborator(?:y|ies)|"
        r"faculty|hospital|centre|center|orcid|e-?mail|correspond(?:ing)?\s+author|"
        r"equal\s+contribution|affiliation"
        r")\b",
        re.IGNORECASE,
    )
    lead_marker_re = re.compile(r"^(?:\$\^\{[^}]{1,20}\}\$|\^\{[^}]{1,20}\}|[*\u2020\u2021\u00a7#]+)\s*")
    abstract_heading_re = re.compile(r"^#{1,6}\s+Abstract\s*$", re.IGNORECASE)

    def _looks_like_abstract_metadata_line(text: str) -> bool:
        st = _normalize_text(text or "").strip()
        if not st:
            return False
        if re.match(r"^\*{0,2}\s*abstract\b", st, re.IGNORECASE):
            return False
        if re.match(r"^\*{0,2}\s*index\s+terms?\b", st, re.IGNORECASE):
            return False
        if st.startswith("![") or st.startswith("|") or re.match(r"^#{1,6}\s+", st):
            return False

        compact = lead_marker_re.sub("", st).strip()
        has_aff_kw = bool(aff_kw_re.search(compact))
        has_email = ("@" in compact) or bool(re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", compact))
        has_semicolon_block = compact.count(";") >= 2
        has_many_markers = len(re.findall(r"\$\^\{[^}]{1,20}\}\$|\^\{[^}]{1,20}\}", st)) >= 2
        starts_like_affiliation = bool(re.match(r"^(?:dept\.?|department|school|college|university|institute|laborator(?:y|ies)|faculty|hospital|centre|center)\b", compact, re.IGNORECASE))

        if has_email:
            return True
        if has_aff_kw and (has_many_markers or has_semicolon_block or starts_like_affiliation):
            return True
        return False

    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if not abstract_heading_re.match((line or "").strip()):
            i += 1
            continue

        i += 1
        while i < len(lines) and not (lines[i] or "").strip():
            out.append(lines[i])
            i += 1

        while i < len(lines):
            st = (lines[i] or "").strip()
            if not st:
                out.append(lines[i])
                i += 1
                continue
            if _looks_like_abstract_metadata_line(st):
                i += 1
                continue
            break
        continue

    return "\n".join(out)


def _promote_known_plain_subheadings(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    out: list[str] = []
    in_fence = False
    in_math = False
    in_refs = False
    heading_re = re.compile(r"^\s*#{1,6}\s+")
    structural_re = re.compile(r"^\s*(?:!\[[^\]]*\]\([^)]+\)|\||```|\$\$)")
    known_titles = {
        "network training",
        "image reconstruction",
        "loss function",
        "data availability",
        "code availability",
        "competing interests",
        "author contributions",
        "additional information",
    }

    def _next_nonempty(idx: int) -> str:
        j = idx + 1
        while j < len(lines):
            st = str(lines[j] or "").strip()
            if st:
                return st
            j += 1
        return ""

    for idx, line in enumerate(lines):
        st = str(line or "").strip()
        if re.match(r"^\s*```", line):
            in_fence = not in_fence
            out.append(line)
            continue
        if st == "$$":
            in_math = not in_math
            out.append(line)
            continue
        if _is_references_heading_line(st):
            in_refs = True
            out.append(line)
            continue
        if in_refs and _is_post_references_resume_heading_line(st):
            in_refs = False
        if in_fence or in_math or in_refs:
            out.append(line)
            continue
        if (not st) or heading_re.match(st) or structural_re.match(st):
            out.append(line)
            continue
        if len(st) > 80:
            out.append(line)
            continue
        low = _normalize_text(st).strip().lower()
        if low not in known_titles:
            out.append(line)
            continue
        nxt = _next_nonempty(idx)
        if (not nxt) or heading_re.match(nxt) or structural_re.match(nxt):
            out.append(line)
            continue
        out.append(f"### {st}")
    return "\n".join(out)


def _move_early_references_block_to_end(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    ref_re = re.compile(r"^##\s+References\s*$", re.IGNORECASE)
    body_heading_re = re.compile(
        r"^##\s+(?:\d+(?:\.\d+)*\.?\s+.+|Introduction|Background|Related Work|Methods?|Results?|Discussion|Conclusion)\b",
        re.IGNORECASE,
    )
    ref_idx = next((i for i, ln in enumerate(lines) if ref_re.match(ln.strip())), None)
    if ref_idx is None:
        return md
    first_body_idx = next((i for i, ln in enumerate(lines) if body_heading_re.match(ln.strip())), None)
    if first_body_idx is None or ref_idx > first_body_idx:
        return md
    end_idx = None
    for i in range(ref_idx + 1, len(lines)):
        if body_heading_re.match(lines[i].strip()):
            end_idx = i
            break
    if end_idx is None:
        return md
    ref_payload_n = sum(1 for ln in lines[ref_idx + 1 : end_idx] if re.match(r"^\[\d{1,4}\]", ln.strip()))
    if ref_payload_n < 5:
        return md
    moved = lines[ref_idx:end_idx]
    kept = lines[:ref_idx] + lines[end_idx:]
    while kept and not kept[-1].strip():
        kept.pop()
    if kept:
        kept.append("")
    kept.extend(moved)
    return "\n".join(kept)


def _fix_known_safe_ocr_terms(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    out: list[str] = []
    in_fence = False
    in_math = False
    caption_line_re = re.compile(r"^\s*\*\*(?:Figure|Table)\s+[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?\.\*\*", re.IGNORECASE)
    safe_prefix_re = re.compile(
        r"\b("
        r"anti|be|co|de|inter|intra|macro|micro|multi|non|over|post|pre|re|semi|sub|super|trans|ultra|un|under"
        r")-\s+([a-z]{3,})\b",
        flags=re.IGNORECASE,
    )

    def _fix_safe_split_prefix_words(text: str) -> str:
        return safe_prefix_re.sub(lambda m: f"{m.group(1)}{m.group(2)}", text)

    replacements = {
        "EffciientSCI": "EfficientSCI",
    }

    for line in lines:
        if re.match(r"^\s*```", line):
            in_fence = not in_fence
            out.append(line)
            continue
        st = line.strip()
        if st == "$$":
            in_math = not in_math
            out.append(line)
            continue
        if in_fence or in_math:
            out.append(line)
            continue

        fixed = str(line)
        for src, dst in replacements.items():
            fixed = fixed.replace(src, dst)
        fixed = _fix_safe_split_prefix_words(fixed)
        if caption_line_re.match(fixed.strip()):
            fixed = re.sub(r"\s+([,.;:])", r"\1", fixed)
        out.append(fixed)

    return "\n".join(out)

def postprocess_markdown(md: str) -> str:
    md = _cleanup_noise_lines(md)
    md = _drop_standalone_journal_metadata_lines(md)
    md = _drop_ocr_placeholder_lines(md)
    md = _fix_malformed_code_fences(md)
    md = _convert_caption_following_tabular_lines(md)
    md = _normalize_markdown_tables(md)
    md = _reflow_hard_wrapped_paragraphs(md)
    md = _split_inline_heading_markers(md)
    md = fix_math_markdown(md)
    md = _normalize_math_for_typora(md)
    md = _cleanup_stray_latex_in_text(md)
    md = _fix_split_numbered_headings(md)
    md = _promote_document_title_line(md)
    md = _promote_bare_numbered_headings(md)
    md = _unwrap_math_wrapped_headings(md)
    md = _normalize_structural_labels(md)
    md = _strip_frontmatter_metadata_lines(md)
    md = _cleanup_author_superscript_noise(md)
    md = _insert_missing_abstract_heading(md)
    md = _drop_abstract_leading_metadata_lines(md)
    md = _promote_known_plain_subheadings(md)
    md = _enforce_heading_policy(md)
    md = _demote_formulaish_headings(md)
    md = _demote_panel_headings(md)
    md = _rebalance_custom_headings_within_structural_sections(md)
    md = _drop_empty_figure_duplicate_headings(md)
    md = _extract_box_sidebars(md)
    md = _repair_dangling_heading_continuations(md)
    md = _format_references(md)
    md = _move_early_references_block_to_end(md)
    md = _repair_body_figure_reference_captions(md)
    md = _normalize_figure_caption_blocks(md)
    md = _fill_empty_image_alt_from_following_caption(md)
    md = _tighten_image_caption_spacing(md)
    md = _dedupe_nearby_repeated_captions(md)
    md = _pull_images_to_matching_previous_captions(md)
    md = _tighten_image_caption_spacing(md)
    md = _merge_obvious_body_continuation_lines(md)
    md = _repair_sentence_split_by_figure_blocks(md)
    md = _normalize_plural_figure_reference_lines(md)
    md = _normalize_body_citation_markers(md)
    md = _normalize_common_rendering_artifacts(md)
    md = _fix_known_safe_ocr_terms(md)
    md = _split_inline_heading_markers(md)
    return md

