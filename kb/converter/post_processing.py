from __future__ import annotations

import re
from typing import Optional

from .post_heading_rules import (
    _enforce_heading_policy,
    _is_caption_heading_text,
    _is_common_section_heading,
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
from .text_utils import _normalize_text





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
        tail = re.sub(r"^[\s\.\:\-]+", "", tail).strip()
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
                out.append(_format_caption_line(pending_alt_caption))
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

        if caption_re.match(st):
            cap = _format_caption_line(ln)
            if cap:
                out.append(cap)
                continue

        out.append(ln)

    if pending_after_image and pending_alt_caption:
        out.append(_format_caption_line(pending_alt_caption))

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
    md = _extract_box_sidebars(md)
    md = _repair_dangling_heading_continuations(md)
    md = _format_references(md)
    md = _normalize_figure_caption_blocks(md)
    md = _normalize_body_citations_to_superscript(md)
    md = _split_inline_heading_markers(md)
    return md

