from __future__ import annotations

import re

from .post_heading_rules import (
    _is_common_section_heading,
    _parse_appendix_heading_level,
    _parse_numbered_heading_level,
)
from .post_references import (
    _is_post_references_resume_heading_line,
    _is_references_heading_line,
)
from .text_utils import _normalize_text

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
            t2 = re.sub(r"\\textit\{([^{}]{0,220})\}", r"\1", t2)
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
            t2 = re.sub(r"\\(?:text|textit|mathrm|mathbf)\{([^{}]{0,260})\}", r"\1", t)
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
        if "\\textit{" in t and (not _has_hard_math_anchors(t)):
            plain = _latex_text_to_plain(t)
            plain_words = len(re.findall(r"\b[A-Za-z]{2,}\b", plain))
            if plain_words >= 3:
                return True
        if ("@" in t) or ("http://" in t.lower()) or ("https://" in t.lower()) or ("www." in t.lower()):
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
        unit_like = bool(
            re.search(
                r"(?i)\b(?:nm|mm|cm|mW|kW|Hz|kHz|MHz|GHz|fps|dB|ms|us|μs|ns|kg|mg|mol|mmol|kV|mA|V|A)\b",
                t,
            )
        )
        if word_n >= 10 and sym_n <= 1:
            return True
        if word_n >= 8 and unit_like and sym_n <= 1:
            return True
        # Fragmentary short math blocks with unbalanced delimiters are usually extraction junk.
        if len(t) <= 40:
            if (t.count("(") != t.count(")")) or (t.count("{") != t.count("}")) or (t.count("[") != t.count("]")):
                return True
        return False

    def _looks_like_sentence_style_math_block(block: str) -> bool:
        """
        Detect display-math blocks that are really sentence fragments with a little
        inline math mixed in, for example:
          about 1.4 \\text{ Airy units ... } 1\\,\\text{AU} = ...
        These should be unwrapped back to prose instead of preserved as equations.
        """
        t = (block or "").strip()
        if not t:
            return False
        plain = _latex_text_to_plain(t)
        plain_norm = re.sub(r"\s+", " ", plain).strip()
        if not plain_norm:
            return False
        word_n = len(re.findall(r"\b[A-Za-z]{2,}\b", plain_norm))
        if word_n < 8:
            return False
        text_macro_n = len(re.findall(r"\\text\{", t))
        if text_macro_n <= 0:
            return False
        if re.search(r"\\(?:sum|int|prod|frac|sqrt|begin\{(?:equation|align|gather|multline))", t, flags=re.IGNORECASE):
            return False
        first_eq = t.find("=")
        if first_eq >= 0 and first_eq <= 12:
            return False
        if re.match(r"(?i)^(?:about|where|with|for|which|when|the|this|these|those)\b", plain_norm):
            return True
        return plain_norm[:1].islower()

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
        sentence_style_math = _looks_like_sentence_style_math_block(raw)
        if sentence_style_math or _looks_like_not_math(raw):
            if sentence_style_math:
                if raw:
                    out.append(_normalize_text(_latex_text_to_plain(raw)))
                out.append("")
                continue
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
        if _is_references_heading_line(st):
            in_refs = True
            out.append(s)
            continue
        if in_refs and _is_post_references_resume_heading_line(st):
            in_refs = False
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
            word_n0 = len(re.findall(r"\b[A-Za-z]{2,}\b", inner0))
            looks_like_prose_fragment0 = bool(
                re.match(r"^[a-z][A-Za-z]*(?:\(|\b)", inner0)
                or re.search(r"(?i)\b(?:see|where|with|and|or|for|to|in|on|by|as|because)\b", inner0)
                or (word_n0 >= 2 and re.search(r"[.,;:()]", inner0))
            )
            if (looks_like_cite0 or looks_like_prose_fragment0) and (not hard_math0):
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
