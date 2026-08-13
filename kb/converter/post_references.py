from __future__ import annotations

import re

from .post_heading_rules import _parse_appendix_heading_level
from .reference_markdown import (
    _format_author_year_references_block,
    _is_plausible_reference_number,
    _is_year_backref_continuation_line,
    _join_reference_fragments,
    _reference_lines_look_author_year,
    _strip_reference_backref_suffix,
)

def _is_references_heading_line(text: str) -> bool:
    st = (text or "").strip()
    if not st:
        return False
    return bool(
        re.match(r"^#{1,6}\s+(?:References|Bibliography)\b", st, re.IGNORECASE)
        or re.match(r"^(?:References|Bibliography)\s*$", st, re.IGNORECASE)
    )

def _is_post_references_resume_heading_line(text: str) -> bool:
    """
    Detect a likely section boundary after a References/Bibliography block.
    This is intentionally conservative and primarily targets appendices /
    supplementary sections, which are commonly placed after references.
    """
    st = (text or "").strip()
    if not st:
        return False
    if _is_references_heading_line(st):
        return False
    ref_start_match = bool(re.match(r"^\s*(?:\[\s*\d{1,4}\s*\]|\d{1,4}\.\s+[A-Z])", st))

    # Markdown headings: any explicit appendix/supplementary heading should end refs mode.
    if re.match(r"^#{1,6}\s+", st):
        title = re.sub(r"^#{1,6}\s+", "", st).strip()
        if re.fullmatch(r"[A-Z]\.?", title):
            return True
        if re.match(
            r"^(?:appendix|appendices|supplementary(?:\s+material)?|supplemental(?:\s+material)?|annex)\b",
            title,
            re.IGNORECASE,
        ):
            return True
        if _parse_appendix_heading_level(title) is not None:
            return True
        return False

    # Plain-text appendix/supplementary headings (common after OCR/VL).
    if re.fullmatch(
        r"(?:supplementary|supplemental)(?:\s+(?:materials?|information|notes?)(?:\s+[A-Z0-9.]+)?)?",
        st,
        re.IGNORECASE,
    ):
        return True
    if re.fullmatch(
        r"(?:appendix|appendices|annex)(?:\s+[A-Z0-9IVXLCM]+(?:[.:]\s*[^\n]{1,100})?)?",
        st,
        re.IGNORECASE,
    ):
        return True
    # Some OCR/VL outputs merge the paper title and "Supplementary Material" onto one line.
    # Example: "SCIGS: ... Snapshot Compressive Image Supplementary Material"
    if re.search(r"\bsupplementary\s+material\b", st, re.IGNORECASE):
        # Avoid misclassifying reference prose lines mentioning supplementary material.
        if ref_start_match:
            # If this is a merged mega-line containing multiple references plus a supplementary header,
            # treat it as a true resume boundary.
            ref_markers = len(re.findall(r"\[\s*\d{1,4}\s*\]", st))
            if ref_markers >= 2 and len(st) >= 220:
                return True
    if re.search(r"\bsupplemental\s+material\b", st, re.IGNORECASE):
        if ref_start_match:
            ref_markers = len(re.findall(r"\[\s*\d{1,4}\s*\]", st))
            if ref_markers >= 2 and len(st) >= 220:
                return True
    # Catch merged headers like "SCIGS ... A. Additional Experiments" on one OCR line.
    if re.search(r"\b[A-Z]\.\s+Additional\s+Experiments\b", st, re.IGNORECASE):
        if ref_start_match:
            ref_markers = len(re.findall(r"\[\s*\d{1,4}\s*\]", st))
            if ref_markers >= 2 and len(st) >= 180:
                return True
        elif not re.search(r"\b(?:doi|arxiv|proc(?:eedings)?\.?)\b", st, re.IGNORECASE):
            return True
    # For normal reference entries, keep them in refs mode.
    if ref_start_match:
        return False
    if re.match(
        r"^[A-Z]\.\s+(?:appendix|additional|supplementary|supplemental|proof(?:s)?|derivation(?:s)?|"
        r"implementation(?:\s+details?)?|experiment(?:s)?|results?|ablation(?:s)?|details?|extra)\b",
        st,
        re.IGNORECASE,
    ):
        return True
    return False

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
        # Multi-panel figure/table captions often contain many citation markers
        # and copyright years.  That density looks reference-like, but treating a
        # late caption as the start of an unheaded bibliography can move every
        # following body section behind a synthetic ``References`` heading.
        if re.match(
            r"^(?:!\[[^\]]*\]\([^)]+\)\s*)?(?:\*{1,2}\s*)?(?:fig(?:ure)?\.?|table)\s*\w+\b",
            t,
            flags=re.IGNORECASE,
        ):
            return False
        if _looks_bare_numbered_reference_payload_line(t):
            return True
        if re.match(r"^\[\d{1,4}\]\s+[A-Z]", t):
            return True
        marker_n = len(re.findall(r"\[\d{1,4}\]", t))
        if marker_n >= 4 and re.search(r"\b(?:19|20)\d{2}\b", t):
            return True
        if marker_n >= 10:
            return True
        return False

    def _looks_bare_numbered_reference_payload_line(s: str) -> bool:
        t = (s or "").strip()
        if not t:
            return False
        m = re.match(r"^(\d{1,4})\.\s+(.+)$", t)
        if not m:
            return False
        if not _is_plausible_reference_number(m.group(1)):
            return False
        if _is_year_backref_continuation_line(t):
            return False
        body = (m.group(2) or "").strip()
        if len(body) < 24:
            return False
        if re.match(
            r"^(?:introduction|background|related\s+work|method(?:s|ology)?|"
            r"experiment(?:s|al)?|results?|discussion|conclusion|appendix|"
            r"acknowledg(?:e)?ments?|supplementary|supplemental)\b",
            body,
            re.IGNORECASE,
        ):
            return False
        has_year_or_id = bool(
            re.search(r"\b(?:18|19|20)\d{2}\b", body)
            or re.search(r"\b(?:doi\s*:|10\.\d{4,9}/|arxiv\s*:)", body, re.IGNORECASE)
        )
        if not has_year_or_id:
            return False
        has_reference_shape = bool(
            re.search(r"(?:[A-Z]\.\s*){1,5}[A-Z][A-Za-z'\-]+", body)
            or re.search(
                r"\b(?:journal|proceedings?|proc\.?|opt\.|phys\.|nat\.|nature|science|"
                r"ieee|acm|appl\.|express|letters?|review|commun\.|photonics|"
                r"arxiv|doi)\b",
                body,
                re.IGNORECASE,
            )
            or ("," in body and len(re.findall(r"[A-Za-z]{2,}", body)) >= 5)
        )
        return has_reference_shape

    heading_any_re = re.compile(r"^#{1,6}\s+")
    plain_section_re = re.compile(
        r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
        r"(?:introduction|background|related work|method(?:s|ology)?|"
        r"experiment(?:s|al)?|results?|discussion|conclusion|"
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
        if _is_references_heading_line(ln):
            ref_i = i
            break
    if ref_i is None:
        # Fallback: infer references start when heading is missing.
        # We only trigger when there are many reference-like leading markers,
        # and they are concentrated in the latter part of the document.
        cand_idx: list[int] = []
        for i, ln in enumerate(lines):
            s = ln.strip()
            if _looks_reference_payload_line(s):
                cand_idx.append(i)
        if len(cand_idx) >= 3:
            tail_threshold = int(len(lines) * 0.45)
            tail_cands = [i for i in cand_idx if i >= tail_threshold]
            if len(tail_cands) >= 3:
                ref_i = tail_cands[0]
                if ref_i > 0 and not lines[ref_i - 1].strip():
                    ref_i -= 1
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
                    ref_i = tail_dense[0]
                    if ref_i > 0 and not lines[ref_i - 1].strip():
                        ref_i -= 1
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
    # Some converters insert one or more spacer blank lines immediately before
    # a late "References" heading. Skip those first so we can still pull the
    # dense pre-heading reference payload into the references block.
    leading_blank_gap = 0
    while k >= 0 and not lines[k].strip() and leading_blank_gap < 4:
        leading_blank_gap += 1
        k -= 1
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
    # The same boundary scan is required when the References heading itself was
    # inferred: supplementary material commonly begins immediately after the
    # numbered list even when OCR omitted the heading above that list.
    tail_end = len(lines)
    ref_signal = 0
    non_ref_run = 0
    non_ref_start = -1
    for j in range(tail_start, len(lines)):
        st = lines[j].strip()
        if not st:
            continue
        if re.match(r"^#{1,6}\s+(?:References|Bibliography)\b", st, re.IGNORECASE):
            continue

        # Strong boundary: Appendix/Supplementary section after references.
        # This must work even when the references section is short (<3 entries).
        if _is_post_references_resume_heading_line(st):
            tail_end = j
            break

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
    # A resume section is a sibling of References, even if earlier heading
    # normalization demoted it while the References heading was still missing.
    # Restore that hierarchy so structured indices do not report supplementary
    # formulas and figures as children of the bibliography.
    for body_idx, body_line in enumerate(body_tail[:12]):
        body_st = (body_line or "").strip()
        if not body_st or re.fullmatch(r"<!--\s*kb_page:\s*\d+\s*-->", body_st, flags=re.IGNORECASE):
            continue
        if _is_post_references_resume_heading_line(body_st):
            body_title = re.sub(r"^#{1,6}\s+", "", body_st).strip()
            body_tail[body_idx] = f"## {body_title}"
        break
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

    author_year_ref_lines = list(enumerate(tail))
    if _reference_lines_look_author_year(author_year_ref_lines):
        author_year_refs = _format_author_year_references_block(author_year_ref_lines)
        out_author_year = head + [""] + author_year_refs
        if body_tail:
            out_author_year.extend([""] + body_tail)
        return "\n".join(out_author_year)

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

    # Page anchors are structural metadata, not part of a bibliography entry.
    # Put inline anchors on their own line before parsing so wrapped reference
    # text cannot absorb and later discard them.
    page_marker_pat = r"<!--\s*kb_page:\s*\d+\s*-->"
    blob = re.sub(
        rf"\s*({page_marker_pat})\s*",
        lambda match: f"\n{(match.group(1) or '').strip()}\n",
        blob,
        flags=re.IGNORECASE,
    ).strip()

    # Split multiple [n] items that got collapsed into one line.
    blob = re.sub(r"\s+(?=\[\d+\])", "\n", blob)
    # Some reference pages emit the next entry as a bare numbered marker after a
    # normal sentence terminus, e.g. "... (2020). 21. Author ...". Split these
    # before marker normalization so they can become standalone [21] entries.
    blob = re.sub(r"(?<=[\.\?\!])\s+(?=\d{1,4}\.\s+[A-Z])", "\n", blob)
    # Normalize leading markers like "1] ..." or "1. ...", but do not
    # reinterpret detached year/backref lines like "2020. 2, 5" as references.
    norm_blob_lines: list[str] = []
    for raw_blob_line in blob.splitlines():
        st = (raw_blob_line or "").strip()
        if _is_year_backref_continuation_line(st):
            norm_blob_lines.append(st)
            continue
        bracket_match = re.match(r"^\s*(\d+)\]\s*", st)
        if bracket_match and _is_plausible_reference_number(bracket_match.group(1)):
            st = re.sub(r"^\s*(\d+)\]\s*", r"[\1] ", st)
        dot_match = re.match(r"^\s*(\d+)\.\s+", st)
        if dot_match and _is_plausible_reference_number(dot_match.group(1)):
            st = re.sub(r"^\s*(\d+)\.\s+", r"[\1] ", st)
        norm_blob_lines.append(st)
    blob = "\n".join(norm_blob_lines)

    # Inline rescue: some preprocessors/OCR outputs collapse the first lines of
    # supplementary material into the tail of the last reference line.
    # When a strong post-references marker appears inside the blob after multiple
    # reference markers, split it back out into body_tail so it won't be trimmed
    # as reference noise.
    try:
        inline_resume_pat = re.compile(
            r"(?mi)^\s*(?:"
            r"(?:supplementary|supplemental)(?:\s+(?:materials?|information|notes?)(?:\s+[A-Z0-9.]+)?)?"
            r"|(?:appendix|appendices)(?:\s+[A-Z0-9IVXLCM]+(?:[.:]\s*[^\n]{1,100})?)?"
            r"|[A-Z]\.\s+Additional\s+Experiments"
            r")\s*$",
        )
        m_inline_resume = inline_resume_pat.search(blob)
        if m_inline_resume:
            pre_blob = blob[: int(m_inline_resume.start())]
            ref_marker_n = len(re.findall(r"\[\s*\d{1,4}\s*\]", pre_blob))
            if ref_marker_n >= 2:
                inline_tail = blob[int(m_inline_resume.start()) :].strip()
                blob = pre_blob.rstrip()
                if inline_tail:
                    body_tail = [inline_tail] + list(body_tail or [])
    except Exception:
        pass

    def _trim_reference_noise(entry: str) -> str:
        s = re.sub(r"\s+", " ", (entry or "")).strip()
        if not s:
            return s
        s = _strip_reference_backref_suffix(s)

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
                r"(\\operatorname\*?\{|\\arg(?:min|max)|\\frac|\\sum|\\int|\\left|\\right|\\mathbf\{|\\mathbb\{|\\partial|\\nabla|\\begin\{|\\end\{|\\tag\{|\\\|)",
                s,
            )
            if not hard_math_m:
                hard_math_m = re.search(
                    r"(?:(?<![A-Za-z0-9])[=^_]{1,2}(?![A-Za-z])|\\[()\[\]])",
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
            r"\bhttps?://\S+,\s*(?:19|20)\d{2}\.",
            r"\bwww\.[^\s,]+,\s*(?:19|20)\d{2}\.",
            r"\b(?:19|20)\d{2}\s*,\s*\d+\s*,\s*[A-Za-z]?\d+[A-Za-z0-9\-]*\s*\.",
            r"\b(?:19|20)\d{2}\s*,\s*\d+\s*,\s*[A-Za-z]?\d+[A-Za-z0-9\-]*\b",
            r"\b(?:19|20)\d{2}\s*,\s*\d+\s*\.",
            r"\b(?:19|20)\d{2}\.",
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
            r"\bThis work was supported\b",
            r"\bReceived\b|\bAccepted\b|\bPublished\b",
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
    page_markers_by_ref: dict[int, list[str]] = {}
    pending_page_markers: list[str] = []
    trailing_page_markers: list[str] = []
    cur: list[str] | None = None
    start_re = re.compile(r"^\[(\d+)\]\s+")

    def _flush_current_reference() -> None:
        nonlocal cur
        if not cur:
            cur = None
            return
        joined = _join_reference_fragments(cur)
        if joined:
            entries.append(joined)
        cur = None

    for raw in blob.splitlines():
        s = raw.strip()
        if not s:
            continue
        if re.fullmatch(page_marker_pat, s, flags=re.IGNORECASE):
            _flush_current_reference()
            pending_page_markers.append(s)
            continue
        if re.fullmatch(r"\d{1,4}", s):
            continue
        if re.search(r"\bpage\s+\d+\s+of\s+\d+\b", s, flags=re.IGNORECASE):
            continue
        m = start_re.match(s)
        if m:
            _flush_current_reference()
            if pending_page_markers:
                ref_no = int(m.group(1))
                page_markers_by_ref.setdefault(ref_no, []).extend(pending_page_markers)
                pending_page_markers = []
            cur = [s]
            continue
        # Ignore garbage before first reference marker
        if cur is None:
            continue
        cur.append(s)
    _flush_current_reference()
    trailing_page_markers = list(pending_page_markers)

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
        if not _is_plausible_reference_number(n):
            unknown.append(e)
            continue
        parsed.append((n, e))

    parsed.sort(key=lambda x: x[0])
    seen_nums: set[int] = set()
    out_refs: list[str] = []
    used_marker_anchors: set[int] = set()
    for n, e in parsed:
        if n in seen_nums:
            continue
        seen_nums.add(n)
        for marker in page_markers_by_ref.get(n, []):
            if marker not in out_refs:
                out_refs.append(marker)
        used_marker_anchors.add(n)
        out_refs.append(e)
    # Keep any unknown tail lines at the end (rare)
    out_refs.extend(unknown)
    for ref_no, markers in page_markers_by_ref.items():
        if ref_no in used_marker_anchors:
            continue
        for marker in markers:
            if marker not in out_refs:
                out_refs.append(marker)
    for marker in trailing_page_markers:
        if marker not in out_refs:
            out_refs.append(marker)

    out1 = head + [""] + out_refs
    if body_tail:
        out1.extend([""] + body_tail)
    return "\n".join(out1)
