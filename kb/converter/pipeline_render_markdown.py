from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

try:
    import fitz
except ImportError:
    fitz = None

from .models import TextBlock
from .text_utils import _normalize_text

_UNICODE_MATH_SYMBOLS_RE = r"[∑∏∫√∞≈≠≤≥±×·⋅∈∂∇]"


def _has_mathish_symbol(text: str) -> bool:
    return bool(re.search(rf"[=^_\\]|{_UNICODE_MATH_SYMBOLS_RE}", text or ""))


def _count_mathish_symbols(text: str) -> int:
    return len(re.findall(rf"[=+\-*/^_{{}}\\\[\]]|{_UNICODE_MATH_SYMBOLS_RE}", text or ""))


def _has_complex_math_token(text: str) -> bool:
    return bool(
        re.search(
            rf"{_UNICODE_MATH_SYMBOLS_RE}|\\(?:frac|sqrt|sum|int|prod|log|exp|left|right|begin)\b",
            text or "",
        )
    )


def _looks_like_short_prose_math_fragment(text: str) -> bool:
    ss = (text or "").strip()
    if not ss or len(ss) > 48:
        return False
    if not re.search(r"[A-Za-z]{2,}", ss):
        return False
    if re.match(r"(?i)^(?:see\s+)?eq\.?~?\s*\(?\d+\)?\.?$", ss):
        return True
    if _has_complex_math_token(ss):
        return False
    if re.search(r"[=^_{}\\\[\]]", ss):
        return False
    return bool(re.match(r"^[a-z]{2,}", ss) and (re.search(r"(?i)\bsee\s+eq\b", ss) or re.search(r"[().,;:]", ss)))


def render_blocks_to_markdown(
    self,
    blocks: List[TextBlock],
    page_index: int,
    *,
    page=None,
    assets_dir: Path | None = None,
    is_references_page: bool = False,
) -> str:
    import time
    render_start = time.time()
    llm_call_count = 0
    llm_total_time = 0.0

    eq_img_idx = 0

    def _ctx_from_neighbor_blocks(idx: int, *, direction: int) -> str:
        """
        Build a small, useful context snippet for LLM math repair from nearby raw blocks.
        direction: -1 for before, +1 for after
        """
        try:
            acc: list[str] = []
            j = idx + direction
            # Take up to 2 blocks of context
            while 0 <= j < len(blocks) and len(acc) < 2:
                nb = blocks[j]
                t = (nb.text or "").strip()
                if not t:
                    j += direction
                    continue
                # Skip structural noise
                if nb.is_table or nb.is_code:
                    j += direction
                    continue
                # Avoid dumping huge paragraphs
                if len(t) > 240:
                    t = t[:240] + "..."
                acc.append(t)
                j += direction
            return "\n".join(acc) if direction > 0 else "\n".join(reversed(acc))
        except Exception:
            return ""

    def _extract_math_raw_from_page(page, bbox: tuple[float, float, float, float]) -> str:
        """
        Extract math-like text from `page` inside `bbox` using span-level geometry (rawdict),
        attempting to preserve superscript/subscript structure.

        This is critical for full_llm quality: `get_text('dict')` often loses layout cues,
        producing garbled math that even an LLM can't reliably reconstruct.
        """
        try:
            clip = fitz.Rect(bbox)
            if clip.width <= 2 or clip.height <= 2:
                return ""
        except Exception:
            return ""

        try:
            d = page.get_text("rawdict")
        except Exception:
            try:
                d = page.get_text("dict")
            except Exception:
                return ""

        spans: list[tuple[float, float, float, float, float, str]] = []
        try:
            for b0 in (d.get("blocks") or []):
                for ln in (b0.get("lines") or []):
                    for sp in (ln.get("spans") or []):
                        txt = str(sp.get("text") or "")
                        if not txt.strip():
                            # Some PDFs store glyphs only in `chars` with empty `text`.
                            chars = sp.get("chars") or []
                            if chars:
                                try:
                                    txt = "".join(str(ch.get("c") or "") for ch in chars)
                                except Exception:
                                    txt = ""
                        if not txt.strip():
                            continue
                        sb = sp.get("bbox")
                        if not sb:
                            continue
                        try:
                            r = fitz.Rect(tuple(float(x) for x in sb))
                        except Exception:
                            continue
                        if not r.intersects(clip):
                            continue
                        size = float(sp.get("size") or 0.0)
                        spans.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1), size, txt))
        except Exception:
            spans = []

        if not spans:
            return ""

        # Sort top-to-bottom, left-to-right
        spans.sort(key=lambda x: (x[1], x[0]))

        # Group spans into lines by y proximity
        lines: list[list[tuple[float, float, float, float, float, str]]] = []
        cur: list[tuple[float, float, float, float, float, str]] = []
        cur_y = None
        for sp in spans:
            y0, y1, size = sp[1], sp[3], sp[4]
            cy = (y0 + y1) / 2.0
            if cur_y is None:
                cur_y = cy
                cur = [sp]
                continue
            tol = max(2.0, (size or 10.0) * 0.65)
            if abs(cy - cur_y) <= tol:
                cur.append(sp)
                # update running center y
                cur_y = (cur_y * 0.7) + (cy * 0.3)
            else:
                lines.append(cur)
                cur = [sp]
                cur_y = cy
        if cur:
            lines.append(cur)

        def _median(xs: list[float]) -> float:
            if not xs:
                return 0.0
            xs2 = sorted(xs)
            mid = len(xs2) // 2
            return xs2[mid] if (len(xs2) % 2 == 1) else (xs2[mid - 1] + xs2[mid]) / 2.0

        out_lines: list[str] = []
        for ln in lines:
            ln.sort(key=lambda x: x[0])
            centers = [((x[1] + x[3]) / 2.0) for x in ln]
            sizes = [float(x[4] or 0.0) for x in ln]
            base_c = _median(centers)
            base_s = _median([s for s in sizes if s > 0.0]) or 0.0
            # Estimate line height
            heights = [max(1.0, float(x[3] - x[1])) for x in ln]
            lh = _median(heights) or 10.0

            parts: list[str] = []
            prev_x1 = None
            for x0, y0, x1, y1, size, txt in ln:
                t = str(txt)
                # Normalize some common math glyphs early
                t = (
                    t.replace("\u2299", r"\odot")
                    .replace("\u2208", r"\in")
                    .replace("\u00d7", r"\times")
                    .replace("\u00b7", r"\cdot")
                )
                t = re.sub(r"\s+", " ", t).strip()
                if not t:
                    continue
                c = (y0 + y1) / 2.0
                is_small = (base_s > 0.0) and (size > 0.0) and (size <= base_s * 0.88)
                # Sup/sub classification by relative center y
                super_th = base_c - (lh * 0.22)
                sub_th = base_c + (lh * 0.22)
                if is_small and c < super_th:
                    t = "^{" + t + "}"
                elif is_small and c > sub_th:
                    t = "_{" + t + "}"

                # Insert a space if there is a large horizontal gap between spans
                if prev_x1 is not None:
                    gap = float(x0) - float(prev_x1)
                    if gap > max(2.0, (base_s or 10.0) * 0.4):
                        parts.append(" ")
                parts.append(t)
                prev_x1 = x1

            line_s = "".join(parts).strip()
            if line_s:
                out_lines.append(line_s)

        return "\n".join(out_lines).strip()

    def _save_eq_image(bbox: tuple[float, float, float, float]) -> str | None:
        nonlocal eq_img_idx
        if (not self.cfg.eq_image_fallback) or (page is None) or (assets_dir is None):
            return None
        try:
            assets_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None
        try:
            r = fitz.Rect(bbox)
        except Exception:
            return None
        # Pad a bit to include equation number / surrounding symbols
        try:
            pad_x = max(2.0, float(r.width) * 0.04)
            pad_y = max(2.0, float(r.height) * 0.10)
            clip = fitz.Rect(
                max(0.0, float(r.x0) - pad_x),
                max(0.0, float(r.y0) - pad_y),
                min(float(page.rect.width), float(r.x1) + pad_x),
                min(float(page.rect.height), float(r.y1) + pad_y),
            )
            if clip.width <= 2 or clip.height <= 2:
                return None
        except Exception:
            clip = r
        eq_img_idx += 1
        img_name = f"page_{page_index+1}_eq_{eq_img_idx}.png"
        img_path = assets_dir / img_name
        try:
            pix = page.get_pixmap(clip=clip, dpi=int(getattr(self, "dpi", 200) or 200))
            pix.save(img_path)
            if (not img_path.exists()) or (img_path.stat().st_size < 256):
                return None
        except Exception:
            return None
        return f"![Equation](./assets/{img_name})"

    def _vision_math_enabled() -> bool:
        """
        Whether to use vision-capable math recovery from equation screenshots.
        Priority:
        - explicit env KB_PDF_LLM_VISION_MATH
        - otherwise auto-enable for VL/vision models (e.g. qwen3-vl-plus)
        """
        try:
            if (not self.cfg.llm) or (not self.llm_worker) or (not getattr(self.llm_worker, "_client", None)):
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

    def _looks_like_broken_display_math(math_src: str, latex_text: str) -> bool:
        """
        Decide if a non-empty latex_text is still likely wrong enough to justify a vision retry.
        Keep conservative to control cost; only for display-ish math.
        """
        ms = (math_src or "").strip()
        lt = (latex_text or "").strip()
        if not ms or not lt:
            return False
        if ("=" in ms) and ("=" not in lt):
            return True
        if len(ms) >= 55 and len(lt) <= max(24, int(len(ms) * 0.45)):
            return True
        if any(x in ms for x in ["\u2211", "\u221e", "\\sum", "\u222b", "\\int", "||", "\u2014"]) and not any(
            y in lt for y in ["\\sum", "\\int", "\\left\\|", "\\|", "\\lVert", "\\rVert"]
        ):
            return True
        if len(lt) >= 120 and len(re.findall(r"\b[A-Za-z]{3,}\b", lt)) >= 10:
            return True
        return False

    def _debug_vision_math() -> bool:
        try:
            return bool(int(os.environ.get("KB_PDF_DEBUG_VISION_MATH", "0") or "0")) or bool(
                getattr(self.cfg, "keep_debug", False)
            )
        except Exception:
            return False

    def _vision_math_policy() -> str:
        """
        Control when to use vision math recovery.
        - env KB_PDF_VISION_MATH_POLICY: off|fallback|prefer|force
        - additionally, KB_PDF_LLM_VISION_MATH can be set to "prefer"/"force" (backward compatible with boolean).
        """
        try:
            raw2 = str(os.environ.get("KB_PDF_VISION_MATH_POLICY", "") or "").strip().lower()
            if raw2 in {"off", "0", "false", "none"}:
                return "off"
            if raw2 in {"fallback", "prefer", "force"}:
                return raw2
            raw = str(os.environ.get("KB_PDF_LLM_VISION_MATH", "") or "").strip().lower()
            if raw in {"prefer", "force"}:
                return raw
            return "fallback"
        except Exception:
            return "fallback"

    def _expand_math_group_bbox(idx: int, bb: tuple[float, float, float, float]) -> "fitz.Rect":
        """
        Expand a bbox to include neighboring math-ish fragments that belong to the same display equation.
        This dramatically improves VL quality on PDFs where a single equation is split into many tiny blocks.
        """
        try:
            if fitz is None:
                return fitz.Rect(bb)  # type: ignore[union-attr]
        except Exception:
            return fitz.Rect(bb)  # type: ignore[union-attr]
        try:
            base = fitz.Rect(bb)
        except Exception:
            base = fitz.Rect(0, 0, 0, 0)
        if base.width <= 1 or base.height <= 1:
            return base

        def _is_mathish_block(b2: TextBlock) -> bool:
            try:
                if bool(getattr(b2, "is_math", False)):
                    return True
            except Exception:
                pass
            t2 = (getattr(b2, "text", "") or "").strip()
            if not t2:
                return False
            if len(t2) <= 10 and re.fullmatch(r"[A-Za-z0-9\s\(\)\[\]\{\}=+\-*/^_.,\\]+", t2):
                return True
            if re.fullmatch(r"[A-Za-z]\s+[A-Za-z]", t2):
                return True
            if any(
                ch in t2
                for ch in [
                    "\u2212",
                    "\u2264",
                    "\u2265",
                    "\u2260",
                    "\u00d7",
                    "\u00b7",
                    "\u221e",
                    "\u2208",
                    "\u2209",
                    "\u2211",
                    "\u2192",
                    "\u2194",
                    "\u2200",
                    "\u2203",
                ]
            ):
                return True
            if ("\\" in t2) or ("^" in t2) or ("_" in t2):
                return True
            return False

        def _can_merge(r0: "fitz.Rect", r1: "fitz.Rect") -> bool:
            try:
                y_ol = min(float(r0.y1), float(r1.y1)) - max(float(r0.y0), float(r1.y0))
                y_gap = max(0.0, max(float(r1.y0) - float(r0.y1), float(r0.y0) - float(r1.y1)))
                x_ol = min(float(r0.x1), float(r1.x1)) - max(float(r0.x0), float(r1.x0))
                x_ol = max(0.0, float(x_ol))
                min_w = max(1.0, min(float(r0.width), float(r1.width)))
                x_ratio = x_ol / min_w
                tiny = (float(r0.width) < 60.0) or (float(r1.width) < 60.0)
                return (y_ol > 0.0) or (y_gap <= max(3.0, min(float(r0.height), float(r1.height)) * 0.95) and (x_ratio >= 0.12 or tiny))
            except Exception:
                return False

        r = base
        # Look forward/backward a few blocks.
        max_hops = 6
        for j in range(idx + 1, min(len(blocks), idx + 1 + max_hops)):
            nb = blocks[j]
            if not _is_mathish_block(nb):
                break
            try:
                r2 = fitz.Rect(getattr(nb, "bbox", None) or bb)
            except Exception:
                break
            if _can_merge(r, r2):
                r |= r2
                continue
            # Stop if it jumps too far downward.
            try:
                if float(r2.y0) - float(r.y1) > max(18.0, float(r.height) * 1.8):
                    break
            except Exception:
                break
        for j in range(idx - 1, max(-1, idx - 1 - max_hops), -1):
            pb = blocks[j]
            if not _is_mathish_block(pb):
                break
            try:
                r2 = fitz.Rect(getattr(pb, "bbox", None) or bb)
            except Exception:
                break
            if _can_merge(r, r2):
                r |= r2
                continue
            try:
                if float(r.y0) - float(r2.y1) > max(18.0, float(r.height) * 1.8):
                    break
            except Exception:
                break
        return r
    
    out = []
    try:
        if bool(int(os.environ.get("KB_PDF_DEBUG_MATH_BLOCKS", "0") or "0")):
            print(f"[DEBUG] Page {page_index+1} blocks dump (n={len(blocks)}):", flush=True)
            for i0, b0 in enumerate(blocks[:180]):
                try:
                    t0 = (b0.text or "").strip().replace("\n", "\\n")
                except Exception:
                    t0 = ""
                try:
                    bb0 = getattr(b0, "bbox", None)
                except Exception:
                    bb0 = None
                try:
                    is_m = bool(getattr(b0, "is_math", False))
                except Exception:
                    is_m = False
                if (not is_m) and (not _has_mathish_symbol(t0)):
                    continue
                print(f"[DEBUG]  idx={i0:03d} is_math={int(is_m)} bbox={bb0} text={ascii(t0[:140])}", flush=True)
    except Exception:
        pass
    block_times = []
    for block_idx, b in enumerate(blocks):
        block_start = time.time()
        # Check if this is an image block (images are stored as text blocks with markdown image syntax)
        if b.text and (b.text.startswith("![") or re.match(r'^!\[.*?\]\(.*?\)', b.text)):
            # This is an image block - output it directly
            out.append(b.text)
            out.append("")  # Add blank line after image
            block_time = time.time() - block_start
            if block_time > 0.1:
                block_times.append((block_idx, "image", block_time))
            continue
        
        if b.heading_level:
            # [H1] -> #
            lvl = int(b.heading_level.replace("[H", "").replace("]", ""))
            heading_text = b.text.strip()
            # Render-stage LLM calls are expensive and often redundant with Step 6 (LLM classify).
            # Keep render deterministic: apply strict heuristics only.
            if '@' in heading_text or re.search(r'\b(?:university|dept|department|institute|email|zhejiang|westlake)\b', heading_text, re.IGNORECASE):
                out.append(heading_text)
                continue

            if re.search(r'[鈫戔啌]', heading_text) or re.search(r'\b(?:PSNR|SSIM|LPIPS)\b', heading_text, re.IGNORECASE):
                out.append(heading_text)
                continue

            # Quick heuristic: very short or math-like -> not heading
            if len(heading_text) <= 5 or not re.search(r'[A-Za-z]{3,}', heading_text):
                out.append(heading_text)
                continue

            math_symbol_count = len(re.findall(r'[+\-*/=^_{}\[\]()]', heading_text))
            if math_symbol_count > 2:
                out.append(heading_text)
                continue

            # Normalize for duplicate check
            normalized_heading = re.sub(r'^[IVX]+\.\s*', '', heading_text, flags=re.IGNORECASE).strip()
            normalized_heading = re.sub(r'^[A-Z]\.\s*', '', normalized_heading).strip()
            # Remove full hierarchical numeric prefixes: "5.6. Title" -> "Title"
            normalized_heading = re.sub(r'^\d+(?:\.\d+)*\.?\s*', '', normalized_heading).strip()

            if normalized_heading in self.seen_headings:
                out.append(heading_text)
                continue

            # Heuristic level determination based on numbering pattern
            numbered_match = re.match(r'^(\d+(?:\.\d+)*)\.?\s+', heading_text)
            if numbered_match:
                num_parts = numbered_match.group(1).split('.')
                if len(num_parts) == 1:
                    lvl = 1
                elif len(num_parts) == 2:
                    lvl = 2
                else:
                    lvl = 3
            else:
                letter_match = re.match(r'^[A-Z]\.\s+', heading_text)
                if letter_match:
                    lvl = 2
                else:
                    lvl = min(3, lvl)

            while self.heading_stack and self.heading_stack[-1][0] >= lvl:
                self.heading_stack.pop()
            self.heading_stack.append((lvl, heading_text))
            self.seen_headings.add(normalized_heading)
            out.append("#" * lvl + " " + heading_text)
        elif b.is_table:
            if b.table_markdown:
                out.append(b.table_markdown)
            else:
                out.append(b.text) # Fallback
        elif b.is_code:
            out.append("```\n" + b.text + "\n```")
        elif b.is_math:
            math_block_start = time.time()
            math_text_len = len(b.text)
            text_stripped = b.text.strip()

            if is_references_page:
                out.append(_normalize_text(text_stripped))
                out.append("")
                continue

            # Some PDFs pack "equation + where explanation + figure caption" into one block/line.
            # If we treat the whole thing as math, it becomes a giant broken $$...$$ block.
            # Split out obvious prose/caption tails and only repair/render the math prefix.
            prose_tail = ""
            math_src = text_stripped
            # For display equations, prefer a rawdict span-based extraction inside the bbox.
            # This preserves superscript/subscript cues and improves LLM repair quality.
            try:
                if page is not None:
                    bb = getattr(b, "bbox", None)
                    if bb and isinstance(bb, tuple) and len(bb) == 4:
                        raw2 = _extract_math_raw_from_page(page, bb)
                        # Only replace if it looks non-trivial.
                        if raw2 and len(raw2) >= max(12, int(len(math_src) * 0.85)):
                            math_src = raw2
                            text_stripped = math_src.strip()
            except Exception:
                pass
            try:
                cap_m = re.search(r"(?i)(\*?Figure\s+\d+\b|\bFig\.?\s*\d+\b|\bTable\s+\d+\b)", math_src)
                if cap_m:
                    prose_tail = math_src[cap_m.start():].strip()
                    math_src = math_src[:cap_m.start()].strip()
            except Exception:
                pass
            try:
                low0 = math_src.lower()
                wpos = -1
                for tok in (" where ", "\nwhere ", "\r\nwhere "):
                    p = low0.find(tok)
                    if p >= 0:
                        wstart = p + tok.find("where")
                        if wpos < 0 or wstart < wpos:
                            wpos = wstart
                if wpos < 0 and low0.startswith("where "):
                    wpos = 0
                if wpos >= 0:
                    tail = math_src[wpos:].strip()
                    tail_words = len(re.findall(r"\b[A-Za-z]{2,}\b", tail))
                    if tail_words >= 8:
                        prose_tail = (tail + ("\n" + prose_tail if prose_tail else "")).strip()
                        math_src = math_src[:wpos].strip()
            except Exception:
                pass

            if not math_src and prose_tail:
                out.append(prose_tail)
                out.append("")
                continue
            if math_src:
                text_stripped = math_src

            try:
                prev_idx = len(out) - 1
                while prev_idx >= 0 and not (out[prev_idx] or "").strip():
                    prev_idx -= 1
                if prev_idx >= 0:
                    prev_line = (out[prev_idx] or "").rstrip()
                    if prev_line.endswith("-") and _looks_like_short_prose_math_fragment(text_stripped):
                        out[prev_idx] = prev_line[:-1] + text_stripped.lstrip()
                        if prose_tail:
                            out.append(prose_tail)
                            out.append("")
                        continue
            except Exception:
                pass

            def _strip_prose_tail_from_math(s: str) -> str:
                ss = (s or "").strip()
                if not ss:
                    return ss
                try:
                    cap_m2 = re.search(r"(?i)(\*?Figure\s+\d+\b|\bFig\.?\s*\d+\b|\bTable\s+\d+\b)", ss)
                    if cap_m2 and cap_m2.start() > 0:
                        ss = ss[:cap_m2.start()].strip()
                except Exception:
                    pass
                try:
                    low1 = ss.lower()
                    wpos2 = -1
                    for tok in (" where ", "\nwhere ", "\r\nwhere "):
                        p = low1.find(tok)
                        if p >= 0:
                            wstart = p + tok.find("where")
                            if wpos2 < 0 or wstart < wpos2:
                                wpos2 = wstart
                    if wpos2 < 0 and low1.startswith("where "):
                        wpos2 = 0
                    if wpos2 > 0:
                        tail2 = ss[wpos2:].strip()
                        tail_words2 = len(re.findall(r"\b[A-Za-z]{2,}\b", tail2))
                        if tail_words2 >= 8:
                            ss = ss[:wpos2].strip()
                except Exception:
                    pass
                return ss

            # Fast heuristic: some headings can be misclassified as math. Handle without LLM.
            looks_like_heading = bool(
                re.match(r'^(?:\d+(?:\.\d+)*\.?|[A-Z]|[IVX]+)\.?\s+\S+', text_stripped)
                or re.match(r'^(?:abstract|introduction|related work|method|methods|experiments?|results?|discussion|conclusion|references|appendix)\b', text_stripped, re.IGNORECASE)
            )
            if looks_like_heading and (len(text_stripped) >= 6) and re.search(r'[A-Za-z]{3,}', text_stripped) and (len(re.findall(r'[+\-*/=^_{}\[\]()]', text_stripped)) <= 1):
                heading_text = text_stripped
                # Determine level by numbering pattern (keep it simple here)
                lvl2 = 2
                numbered_match = re.match(r'^(\d+(?:\.\d+)*)\.?\s+', heading_text)
                if numbered_match:
                    n_parts = numbered_match.group(1).split('.')
                    lvl2 = 1 if len(n_parts) == 1 else (2 if len(n_parts) == 2 else 3)
                elif re.match(r'^[A-Z]\.\s+', heading_text):
                    lvl2 = 2
                normalized_heading = re.sub(r'^[IVX]+\.\s*', '', heading_text, flags=re.IGNORECASE).strip()
                normalized_heading = re.sub(r'^[A-Z]\.\s*', '', normalized_heading).strip()
                normalized_heading = re.sub(r'^\d+(?:\.\d+)*\.?\s*', '', normalized_heading).strip()
                if normalized_heading not in self.seen_headings:
                    while self.heading_stack and self.heading_stack[-1][0] >= lvl2:
                        self.heading_stack.pop()
                    self.heading_stack.append((lvl2, heading_text))
                    self.seen_headings.add(normalized_heading)
                    out.append("#" * lvl2 + " " + heading_text)
                    continue
            
            # Guard: if a "math" block is actually prose (common misclassification),
            # render it as plain text and skip all math repair.
            try:
                word_n = len(re.findall(r"\b\w+\b", text_stripped))
                letters_n = len(re.findall(r"[A-Za-z]", text_stripped))
                math_sym_n = len(re.findall(r"[=+\-*/^_{}\\\[\]]", text_stripped))
                has_sentence = (". " in text_stripped) or ("? " in text_stripped) or ("! " in text_stripped)
                # Long, wordy, low-math-symbol content is almost surely not an equation.
                if (
                    len(text_stripped) >= 120
                    and word_n >= 18
                    and letters_n >= 60
                    and math_sym_n <= 6
                    and has_sentence
                ):
                    out.append(text_stripped)
                    continue
            except Exception:
                pass

            # LLM said it's not a heading, or LLM unavailable - proceed with math rendering
            # Quick heuristic: very short or no letters -> definitely math
            if len(text_stripped) <= 5 or not re.search(r'[A-Za-z]', text_stripped):
                # Too short or no letters - definitely math
                pass

            # If the extracted "math" looks extremely fragmentary (common in PDF text extraction),
            # avoid LLM guessing; rely on lightweight rule fixes instead.
            try:
                frag = False
                if len(text_stripped) <= 14 and re.search(r"[A-Za-z]", text_stripped):
                    frag = True
                # e.g., "N X", "r 鈭?R", "C(r) 2"
                if re.match(r"^[A-Za-z]\s+[A-Za-z]$", text_stripped):
                    frag = True
                if ("\u2212" in text_stripped or "-" in text_stripped) and len(text_stripped) <= 20:
                    frag = True
            except Exception:
                pass
            
            # It's actually math, proceed with math rendering
            # ALWAYS try LLM repair first for better quality (user said speed is OK)
            latex_text = None
            speed_cfg = getattr(self, "_active_speed_config", None) or {}
            # Check both use_llm_for_all and use_llm_in_render (balanced mode disables render LLM)
            use_llm_in_render = speed_cfg.get("use_llm_in_render", speed_cfg.get("use_llm_for_all", True)) if self.cfg.llm else False
            # Don't ask the LLM to "repair" tiny fragments; it tends to hallucinate.
            prefer_llm_repair = (
                (len(text_stripped) >= 18)
                or ("\n" in b.text)
                or ("=" in text_stripped)
                or any(x in text_stripped for x in ["\\sum", "\\int", "\\frac", "\\sqrt"])
            )
            
            if prefer_llm_repair and use_llm_in_render and self.cfg.llm and self.llm_worker._client:
                try:
                    # First attempt: standard repair
                    ctx_before = _ctx_from_neighbor_blocks(block_idx, direction=-1)
                    ctx_after = _ctx_from_neighbor_blocks(block_idx, direction=+1)
                    t0 = time.time()
                    repaired = self.llm_worker.call_llm_repair_math(
                        math_src,
                        page_number=page_index,
                        block_index=block_idx,
                        context_before=ctx_before,
                        context_after=ctx_after,
                    )
                    llm_call_count += 1
                    llm_total_time += (time.time() - t0)
                    if repaired:
                        latex_text = repaired
                    else:
                        # Second attempt: more aggressive repair for inline math
                        # Check if it looks like inline math (short, no line breaks)
                        if len(b.text.strip()) <= 50 and "\n" not in b.text:
                            # Try with a more specific prompt for inline math
                            prompt = f"""Convert this garbled inline math expression to proper LaTeX.
The expression is: {math_src}

Requirements:
- Use proper LaTeX syntax (e.g., \\hat{{C}} not 藛 C, C(r)^2 not C ( r ) 2)
- Remove extra spaces
- Fix subscripts and superscripts properly
- Return ONLY the LaTeX code without $ delimiters

LaTeX:"""
                            try:
                                t1 = time.time()
                                resp = self.llm_worker._llm_create(
                                    messages=[
                                        {"role": "system", "content": "You are a LaTeX math expert specializing in inline math expressions."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    temperature=0.0,
                                    max_tokens=200,
                                )
                                llm_call_count += 1
                                llm_total_time += (time.time() - t1)
                                repaired2 = (resp.choices[0].message.content or "").strip()
                                # Remove $ if present
                                if repaired2.startswith("$") and repaired2.endswith("$"):
                                    repaired2 = repaired2[1:-1].strip()
                                if repaired2:
                                    latex_text = repaired2
                            except Exception:
                                pass
                except Exception:
                    pass

            # Vision math recovery (if enabled): screenshot the equation and ask a VL model for exact LaTeX.
            # Important: do NOT require latex_text to be empty; we also retry when it looks suspiciously broken.
            if _vision_math_enabled() and self.cfg.llm and self.llm_worker._client and page is not None:
                try:
                    bb = getattr(b, "bbox", None)
                    if bb and isinstance(bb, tuple) and len(bb) == 4:
                        # "display-ish" heuristic (controls cost): include common non-'=' display equations too.
                        # Many papers have short display equations without '=' (e.g., sums / norms / constraints).
                        ms = (math_src or "").strip()
                        sym_n = _count_mathish_symbols(ms)
                        complex_tok = _has_complex_math_token(ms)
                        r = _expand_math_group_bbox(block_idx, bb)
                        is_wide = False
                        try:
                            is_wide = float(r.width) >= float(page.rect.width) * 0.55
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
                        policy = _vision_math_policy()
                        should_try = False
                        if policy == "off":
                            should_try = False
                        elif policy == "force":
                            # Force: try vision whenever we have a bbox and the snippet looks math-ish enough.
                            should_try = bool(displayish or complex_tok or sym_n >= 6 or len(ms) >= 18)
                        elif policy == "prefer":
                            # Prefer: for display-ish math, try vision first even if text repair produced something.
                            should_try = bool(displayish)
                        else:
                            # Fallback: only when text repair failed or looks broken.
                            should_try = bool(displayish and (latex_text is None or _looks_like_broken_display_math(ms, latex_text)))
                        if _debug_vision_math() and not should_try:
                            try:
                                mname = str(getattr(self.cfg.llm, "model", "") or "")
                            except Exception:
                                mname = ""
                            why = []
                            if not displayish:
                                why.append(f"not_displayish(sym_n={sym_n},wide={int(is_wide)},len={len(ms)})")
                            if (latex_text is not None) and (not _looks_like_broken_display_math(ms, latex_text)):
                                why.append("latex_not_suspicious")
                            why.append(f"policy={policy}")
                            why_s = ",".join(why) or "unknown"
                            print(
                                f"[VISION_MATH] skip page={page_index+1} block={block_idx+1} model={mname!a} reason={why_s} src_snip={ms[:80]!a}",
                                flush=True,
                            )

                        if should_try:
                            # pad to capture delimiters / equation number area
                            pad_x = max(2.0, float(r.width) * 0.06)
                            pad_y = max(2.0, float(r.height) * 0.20)
                            # If policy is force and the extracted math looks fragmentary, expand the crop more.
                            # This helps when PDF text extraction misses most of the equation but the glyphs
                            # are still visible on the page.
                            try:
                                fraggy = (policy == "force") and (
                                    (len(ms) <= 40)
                                    or (float(r.width) < float(page.rect.width) * 0.38)
                                    or (float(r.height) < 18.0)
                                )
                            except Exception:
                                fraggy = False
                            if fraggy:
                                try:
                                    pad_x = max(pad_x, float(page.rect.width) * 0.10)
                                    pad_y = max(pad_y, 28.0)
                                except Exception:
                                    pad_x = max(pad_x, 24.0)
                                    pad_y = max(pad_y, 28.0)
                            clip = fitz.Rect(
                                max(0.0, float(r.x0) - pad_x),
                                max(0.0, float(r.y0) - pad_y),
                                min(float(page.rect.width), float(r.x1) + pad_x),
                                min(float(page.rect.height), float(r.y1) + pad_y),
                            )
                            if clip.width > 4 and clip.height > 4:
                                # Qwen VL models may enforce minimum image dimensions (e.g. >10px).
                                # Expand clip to satisfy a conservative minimum in pixels at the chosen DPI.
                                try:
                                    dpi0 = int(getattr(self, "dpi", 200) or 200)
                                except Exception:
                                    dpi0 = 200
                                try:
                                    min_px = int(os.environ.get("KB_PDF_VISION_MIN_PX", "12") or "12")
                                except Exception:
                                    min_px = 12
                                try:
                                    min_pt = (72.0 * float(min_px)) / max(50.0, float(dpi0))
                                    if clip.width < min_pt or clip.height < min_pt:
                                        ex = max(0.0, (min_pt - float(clip.width)) / 2.0)
                                        ey = max(0.0, (min_pt - float(clip.height)) / 2.0)
                                        clip = fitz.Rect(
                                            max(0.0, float(clip.x0) - ex),
                                            max(0.0, float(clip.y0) - ey),
                                            min(float(page.rect.width), float(clip.x1) + ex),
                                            min(float(page.rect.height), float(clip.y1) + ey),
                                        )
                                except Exception:
                                    pass
                                if _debug_vision_math():
                                    try:
                                        mname = str(getattr(self.cfg.llm, "model", "") or "")
                                    except Exception:
                                        mname = ""
                                    print(
                                        f"[VISION_MATH] call page={page_index+1} block={block_idx+1} model={mname!a} policy={policy} clip=({clip.x0:.1f},{clip.y0:.1f},{clip.x1:.1f},{clip.y1:.1f}) src_len={len(ms)} dpi={dpi0}",
                                        flush=True,
                                    )
                                pix = page.get_pixmap(clip=clip, dpi=dpi0)
                                try:
                                    if (int(getattr(pix, "width", 0) or 0) < int(min_px)) or (
                                        int(getattr(pix, "height", 0) or 0) < int(min_px)
                                    ):
                                        if _debug_vision_math():
                                            print(
                                                f"[VISION_MATH] skip_small_image page={page_index+1} block={block_idx+1} pix=({pix.width}x{pix.height}) min_px={min_px}",
                                                flush=True,
                                            )
                                        pix = None
                                except Exception:
                                    pass
                                if pix is not None:
                                    v_start = time.time()
                                    png = pix.tobytes("png")
                                    repaired_v = self.llm_worker.call_llm_repair_math_from_image(
                                        png,
                                        page_number=page_index,
                                        block_index=block_idx,
                                    )
                                    llm_call_count += 1
                                    llm_total_time += (time.time() - v_start)
                                    if repaired_v:
                                        if _debug_vision_math():
                                            print(
                                                f"[VISION_MATH] ok page={page_index+1} block={block_idx+1} out_len={len(repaired_v)}",
                                                flush=True,
                                            )
                                        latex_text = repaired_v
                except Exception as e:
                    if _debug_vision_math():
                        try:
                            mname = str(getattr(self.cfg.llm, "model", "") or "")
                        except Exception:
                            mname = ""
                        print(
                            f"[VISION_MATH] error page={page_index+1} block={block_idx+1} model={mname!a} err={e!a}",
                            flush=True,
                        )
            
            # If LLM didn't help, use heuristic conversion
            if not latex_text:
                convert_start = time.time()
                latex_text = self._convert_formula_to_latex(math_src)
                convert_time = time.time() - convert_start
                if convert_time > 0.5:
                    print(f"      [Page {page_index+1}] Block {block_idx+1} _convert_formula_to_latex: {convert_time:.2f}s (SLOW!), text_len={len(b.text)}", flush=True)

            # If still looks bad, fall back to equation image to preserve correctness.
            try:
                if self.cfg.eq_image_fallback:
                    bad = False
                    s = (latex_text or "").strip()
                    # too long text-like output or contains many normal words
                    if len(s) >= 140 and len(re.findall(r"\b[A-Za-z]{3,}\b", s)) >= 10:
                        bad = True
                    # contains obvious prose markers
                    if any(x in s.lower() for x in ["the ", "appears", "likely", "interpretation", "here is"]):
                        bad = True
                    if bad:
                        img_md = _save_eq_image(getattr(b, "bbox", None) or getattr(b, "bbox", (0, 0, 0, 0)))
                        if img_md:
                            out.append(img_md)
                            out.append("")
                            continue
            except Exception:
                pass
            
            if latex_text:
                # Clean up latex_text: remove trailing commas, fix nested $, etc.
                latex_text = latex_text.strip()
                latex_text = _strip_prose_tail_from_math(latex_text)
                # Remove trailing commas and spaces
                latex_text = re.sub(r',\s*$', '', latex_text)
                # Fix nested $ symbols (should not have $ inside $...$)
                latex_text = latex_text.replace('$', '')
                
                # Check if it's inline or display math
                # Display math if:
                # - Contains line breaks
                # - Is long (> 60 chars)
                # - Contains = (equations)
                # - Contains complex structures (sum, integral, etc.)
                has_break = "\n" in math_src
                has_equals = "=" in latex_text
                has_complex = any(op in latex_text for op in ['\\sum', '\\int', '\\prod', '\\frac', '\\sqrt', '\\exp', '\\log'])
                is_long = len(latex_text) > 60
                
                is_inline = not (has_break or has_equals or has_complex or is_long)
                
                if is_inline:
                    # Inline math: use LLM to polish for better quality (only if enabled in speed mode)
                    speed_cfg = getattr(self, "_active_speed_config", None) or {}
                    use_llm_in_render = speed_cfg.get("use_llm_in_render", speed_cfg.get("use_llm_for_all", True)) if self.cfg.llm else False
                    # Avoid polishing very short expressions; LLM often over-edits/hallucinates.
                    if use_llm_in_render and self.cfg.llm and self.llm_worker._client and len(latex_text.strip()) >= 12:
                        try:
                            # More aggressive LLM polish for inline math
                            polish_prompt = f"""Convert this inline math expression to proper LaTeX. The expression may contain garbled characters or incorrect formatting.

Input: {latex_text}

Requirements:
1. Fix garbled characters (e.g., "藛 C" -> "\\hat{{C}}", "C ( r ) 2" -> "C(r)^2")
2. Use proper LaTeX syntax:
   - Subscripts: x_i not x i
   - Superscripts: x^2 not x 2
   - Functions: \\hat{{C}} not 藛 C
   - Remove extra spaces
3. Ensure proper grouping with braces when needed
4. Return ONLY the cleaned LaTeX code without $ delimiters

LaTeX:"""
                            resp = self.llm_worker._llm_create(
                                messages=[
                                    {"role": "system", "content": "You are a LaTeX math expert specializing in inline math expressions. Always return clean, correct LaTeX."},
                                    {"role": "user", "content": polish_prompt}
                                ],
                                temperature=0.0,
                                max_tokens=300,
                            )
                            polished = (resp.choices[0].message.content or "").strip()
                            # Remove $ if present
                            if polished.startswith("$") and polished.endswith("$"):
                                polished = polished[1:-1].strip()
                            elif polished.startswith("$$") and polished.endswith("$$"):
                                polished = polished[2:-2].strip()
                            # Remove any markdown code fences
                            if polished.startswith("```"):
                                polished = re.sub(r'^```(?:\w+)?\n?', '', polished)
                                polished = re.sub(r'\n?```$', '', polished)
                            if polished and len(polished) > 0:
                                latex_text = polished
                        except Exception as e:
                            # If LLM fails, use the original
                            pass
                    
                    # Inline math: ensure no nested $ and proper formatting
                    out.append(f"${latex_text}$")
                else:
                    # Display math
                    out.append(f"$$\n{latex_text}\n$$")
            else:
                # Fallback to original text if conversion fails
                # Clean up the text first
                fallback_text = math_src.strip()
                fallback_text = _strip_prose_tail_from_math(fallback_text)
                fallback_text = re.sub(r',\s*$', '', fallback_text)
                fallback_text = fallback_text.replace('$', '')
                out.append(f"$$\n{fallback_text}\n$$")
            # Emit any prose/caption tail AFTER the math block.
            if prose_tail:
                out.append(prose_tail)
            math_block_time = time.time() - math_block_start
            if math_block_time > 0.1:
                block_times.append((block_idx, f"math({math_text_len}chars)", math_block_time))
        elif b.is_caption:
            # Captions: italicize and add proper spacing
            caption_text = b.text.strip()
            # Remove leading "Fig." or "Figure" if already in markdown format
            if not caption_text.startswith("*"):
                out.append(f"*{caption_text}*")
            else:
                out.append(caption_text)
        else:
            # Regular text - use LLM to fix mojibake if available
            text = b.text
            if self.cfg.llm and hasattr(self.llm_worker, '_client') and self.llm_worker._client:
                # Check if text has mojibake
                if any(pattern in text for pattern in ['膹殴', '膸', '脦麓', '脦膮', '芒']):
                    try:
                        t2 = time.time()
                        repaired = self.llm_worker.call_llm_repair_body_paragraph(
                            text,
                            page_number=page_index,
                            block_index=len(out)
                        )
                        llm_call_count += 1
                        llm_total_time += (time.time() - t2)
                        if repaired:
                            text = repaired
                    except Exception:
                        pass
            out.append(text)
        out.append("")
    
    render_time = time.time() - render_start
    if llm_call_count > 0:
        avg_llm_time = llm_total_time / llm_call_count
        print(f"    [Page {page_index+1}] Render: {render_time:.2f}s total, {llm_call_count} LLM calls, {llm_total_time:.2f}s LLM time (avg {avg_llm_time:.2f}s/call)", flush=True)
    else:
        print(f"    [Page {page_index+1}] Render: {render_time:.2f}s (no LLM calls)", flush=True)
    
    # Report slow blocks
    if block_times:
        slow_blocks = sorted(block_times, key=lambda x: x[2], reverse=True)[:5]
        for block_idx, block_type, block_time in slow_blocks:
            print(f"      [Page {page_index+1}] Slow block {block_idx+1} ({block_type}): {block_time:.2f}s", flush=True)
    
    return "\n".join(out)


