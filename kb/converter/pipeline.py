from __future__ import annotations

import os
import shutil
import re
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import fitz
except ImportError:
    fitz = None

from .config import ConvertConfig
from .models import TextBlock
from .geometry_utils import _rect_area, _union_rect, _rect_intersection_area, _overlap_1d
from .text_utils import _normalize_text

# Greek letter to LaTeX mapping (expanded)
GREEK_TO_LATEX = {
    # Lowercase
    "\u03b1": r"\alpha",
    "\u03b2": r"\beta",
    "\u03b3": r"\gamma",
    "\u03b4": r"\delta",
    "\u03b5": r"\epsilon",
    "\u03b6": r"\zeta",
    "\u03b7": r"\eta",
    "\u03b8": r"\theta",
    "\u03b9": r"\iota",
    "\u03ba": r"\kappa",
    "\u03bb": r"\lambda",
    "\u03bc": r"\mu",
    "\u03bd": r"\nu",
    "\u03be": r"\xi",
    "\u03bf": r"o",  # omicron (rarely used)
    "\u03c0": r"\pi",
    "\u03c1": r"\rho",
    "\u03c3": r"\sigma",
    "\u03c4": r"\tau",
    "\u03c5": r"\upsilon",
    "\u03c6": r"\phi",
    "\u03c7": r"\chi",
    "\u03c8": r"\psi",
    "\u03c9": r"\omega",
    # Uppercase
    "\u0391": r"A",  # Alpha
    "\u0392": r"B",  # Beta
    "\u0393": r"\Gamma",
    "\u0394": r"\Delta",
    "\u0395": r"E",  # Epsilon
    "\u0396": r"Z",  # Zeta
    "\u0397": r"H",  # Eta
    "\u0398": r"\Theta",
    "\u0399": r"I",  # Iota
    "\u039a": r"K",  # Kappa
    "\u039b": r"\Lambda",
    "\u039c": r"M",  # Mu
    "\u039d": r"N",  # Nu
    "\u039e": r"\Xi",
    "\u039f": r"O",  # Omicron
    "\u03a0": r"\Pi",
    "\u03a1": r"P",  # Rho
    "\u03a3": r"\Sigma",
    "\u03a4": r"T",  # Tau
    "\u03a5": r"\Upsilon",
    "\u03a6": r"\Phi",
    "\u03a7": r"X",  # Chi
    "\u03a8": r"\Psi",
    "\u03a9": r"\Omega",
}

# Math symbol to LaTeX mapping
MATH_SYMBOL_TO_LATEX = {
    "\u2264": r"\leq",
    "\u2265": r"\geq",
    "\u2260": r"\neq",
    "\u2248": r"\approx",
    "\u2208": r"\in",
    "\u2209": r"\notin",
    "\u2211": r"\sum",
    "\u220f": r"\prod",
    "\u222b": r"\int",
    "\u221e": r"\infty",
    "\u2192": r"\to",
    "\u00d7": r"\times",
    "\u00b7": r"\cdot",
    "\u2212": r"-",  # minus sign
}
from .layout_analysis import (
    detect_body_font_size,
    build_repeated_noise_texts,
    _pick_render_scale,
    _collect_visual_rects,
    _is_frontmatter_noise_line,
    _pick_column_range,
    sort_blocks_reading_order,
    _detect_column_split_x,
)
from .geometry_utils import _bbox_width
from .heuristics import (
    _suggest_heading_level,
    _is_noise_line,
    _is_non_body_metadata_text,
    _page_has_references_heading,
    _page_looks_like_references_content,
    detect_header_tag,
)
from .tables import _extract_tables_by_layout, _is_markdown_table_sane, _page_maybe_has_table_from_dict, table_text_to_markdown
from .block_classifier import _looks_like_math_block, _looks_like_code_block
from .llm_worker import LLMWorker
from .post_processing import postprocess_markdown
from .md_analyzer import MarkdownAnalyzer
from .pipeline_vision_direct import process_batch_vision_direct
from .pipeline_render_markdown import render_blocks_to_markdown


class PDFConverter:
    def __init__(self, cfg: ConvertConfig):
        self.cfg = cfg
        self.llm_worker = LLMWorker(cfg)
        self.noise_texts: Set[str] = set()
        # Store optional config attributes
        self.dpi = getattr(cfg, 'dpi', 200)
        self.analyze_quality = getattr(cfg, 'analyze_quality', True)
        # Track seen headings to avoid duplicates
        self.seen_headings: Set[str] = set()
        # Track heading hierarchy across pages
        self.heading_stack: List[Tuple[int, str]] = []  # (level, text)

    def convert(self, pdf_path: str, save_dir: str) -> None:
        """Convert PDF to Markdown using the new converter."""
        print("=" * 60, flush=True)
        print("NEW PDFConverter starting...", flush=True)
        print("=" * 60, flush=True)
        
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) not installed.")
            
        pdf_path = Path(pdf_path).resolve()
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        assets_dir = save_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        
        print(f"Opening PDF: {pdf_path}", flush=True)
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"PDF opened successfully, {total_pages} pages", flush=True)
        try:
            self._cleanup_stale_page_assets(assets_dir=assets_dir, total_pages=total_pages)
        except Exception as e:
            print(f"[WARN] stale asset cleanup skipped: {e}", flush=True)
        
        # Output total pages for progress tracking (must match expected format)
        print(f"Detected body font size: 12.0 | pages: {total_pages} | range: 1-{total_pages}", flush=True)
        print(f"Starting conversion of {total_pages} pages...", flush=True)
        
        # Pre-scan for noise
        self.noise_texts = build_repeated_noise_texts(doc)
        print(f"Detected {len(self.noise_texts)} repeated lines as noise.", flush=True)
        
        # Process pages
        # For simplicity, we'll use a sequential loop or simple batching here.
        # The original code had complex batching. We'll simplify to page-by-page for now
        # or use ThreadPool if configured.
        
        md_pages = [None] * total_pages
        speed_mode = getattr(self.cfg, 'speed_mode', 'normal')
        speed_config = self._get_speed_mode_config(speed_mode, total_pages)
        self._active_speed_config = speed_config
        
        # Use LLM config from cfg (already set by the CLI entrypoint)
        use_llm = False
        llm_config = self.cfg.llm
        
        # If LLM config is provided, use it
        if llm_config:
            use_llm = True
            print(f"Using LLM ({llm_config.model}) for processing", flush=True)
        else:
            print("LLM not configured, using fast mode", flush=True)
        
        # Only use vision-direct mode: screenshot 鈫?VL 鈫?Markdown
        speed_mode = getattr(self.cfg, 'speed_mode', 'normal')
        
        if speed_mode == 'no_llm':
            # No LLM mode: use basic text extraction (fallback)
            print("[MODE] No LLM: basic text extraction only", flush=True)
            md_pages = self._process_batch_no_llm(doc, pdf_path, assets_dir)
        else:
            # Vision-direct mode with LLM (normal or ultra_fast)
            if not use_llm or not llm_config:
                print("[WARN] LLM not configured, falling back to no_llm mode", flush=True)
                md_pages = self._process_batch_no_llm(doc, pdf_path, assets_dir)
            else:
                mode_name = "ultra_fast" if speed_mode == "ultra_fast" else "normal"
                print(f"[MODE] Vision-direct ({mode_name}): each page screenshot 鈫?VL model 鈫?Markdown", flush=True)
                md_pages = self._process_batch_vision_direct(doc, pdf_path, assets_dir, speed_mode=speed_mode)
            
        final_md = "\n\n".join(filter(None, md_pages))

        # Post-process: run as best-effort to avoid whole-job failure on one cleanup stage.
        # In production this is better than aborting after long page processing.
        try:
            final_md = postprocess_markdown(final_md)
        except Exception as e:
            print(f"[WARN] postprocess_markdown failed, keep raw markdown: {e}", flush=True)
        if self._legacy_extra_cleanup_enabled():
            # Legacy post-fixes are kept behind an opt-in gate only.
            # They can be helpful for some old outputs but may over-modify modern VL pages.
            try:
                final_md = self._fix_heading_structure(final_md)
            except Exception as e:
                print(f"[WARN] _fix_heading_structure failed, keep previous markdown: {e}", flush=True)
            try:
                final_md = self._fix_references_format(final_md)
            except Exception as e:
                print(f"[WARN] _fix_references_format failed, keep previous markdown: {e}", flush=True)
        try:
            final_md = self._repair_broken_image_links(final_md, save_dir=save_dir, assets_dir=assets_dir)
        except Exception as e:
            print(f"[WARN] _repair_broken_image_links failed, keep previous markdown: {e}", flush=True)
        try:
            final_md = self._inject_title_from_pdf_metadata(final_md, doc)
        except Exception as e:
            print(f"[WARN] title metadata injection skipped: {e}", flush=True)
        
        # Write output
        out_file = save_dir / "output.md"
        out_file.write_text(final_md, encoding="utf-8")
        print(f"Saved to {out_file}", flush=True)
        print(f"Conversion completed successfully!", flush=True)
        
        # Analyze quality and generate report
        if self.analyze_quality:
            try:
                analyzer = MarkdownAnalyzer()
                issues = analyzer.analyze(final_md, out_file)
                if issues:
                    report = analyzer.generate_report()
                    report_file = save_dir / "quality_report.md"
                    report_file.write_text(report, encoding="utf-8")
                    print(f"Quality report saved to {report_file}")
                    print(f"Found {len(issues)} quality issues")
                else:
                    print("[OK] No quality issues detected")
            except Exception as e:
                print(f"[WARN] quality analysis failed: {e}", flush=True)

    def _inject_title_from_pdf_metadata(self, md: str, doc) -> str:
        if not md:
            return md
        try:
            title = str((doc.metadata or {}).get("title") or "").strip()
        except Exception:
            title = ""
        if not title:
            return md
        if len(title) < 8:
            return md
        lines = md.splitlines()
        first_nonempty = next((ln.strip() for ln in lines if ln.strip()), "")
        if re.match(r"^#\s+", first_nonempty):
            return md
        hay = "\n".join(lines[:40]).lower()
        if title.lower() in hay:
            return md
        prefix = f"# {title}\n\n"
        return prefix + md.lstrip()

    def _cleanup_stale_page_assets(self, *, assets_dir: Path, total_pages: int) -> None:
        """
        Remove stale page-scoped assets in the target conversion range so reruns
        don't keep old split/cropped figure files that are no longer referenced.
        """
        start = max(0, int(getattr(self.cfg, "start_page", 0) or 0))
        end = int(getattr(self.cfg, "end_page", -1) or -1)
        if end < 0:
            end = int(total_pages)
        end = min(int(total_pages), int(end))
        if start >= end:
            return

        page_set = {int(i) for i in range(start + 1, end + 1)}  # filenames are 1-based
        pat = re.compile(
            r"^page_(\d+)_(?:fig|eq)_\d+\.(?:png|jpg|jpeg|webp|gif)$",
            re.IGNORECASE,
        )
        pat_meta = re.compile(
            r"^page_(\d+)_fig_\d+\.meta\.json$",
            re.IGNORECASE,
        )
        pat_page_index = re.compile(
            r"^page_(\d+)_fig_index\.json$",
            re.IGNORECASE,
        )
        removed = 0
        for p in assets_dir.glob("page_*"):
            try:
                m = pat.match(p.name) or pat_meta.match(p.name) or pat_page_index.match(p.name)
                if not m:
                    continue
                page_no = int(m.group(1))
                if page_no not in page_set:
                    continue
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                continue
        if removed > 0:
            print(
                f"[ASSET_CLEAN] removed {removed} stale page asset(s) in range {start+1}-{end}",
                flush=True,
            )

    def _collect_non_body_metadata_rects(
        self,
        page,
        *,
        page_index: int,
        is_references_page: bool = False,
    ) -> List[fitz.Rect]:
        """
        Source-level metadata detector (author/affiliation/journal boilerplate).
        Returns text rectangles to suppress before OCR/LLM conversion.
        """
        if fitz is None:
            return []
        try:
            d = page.get_text("dict") or {}
        except Exception:
            return []
        blocks = d.get("blocks", []) or []
        if not blocks:
            return []

        page_h = float(page.rect.height)
        # Per-page body-size hint (mode) for small-font footer/header filtering.
        size_freq: dict[float, int] = {}
        for b in blocks:
            for l in (b.get("lines", []) or []):
                for s in (l.get("spans", []) or []):
                    try:
                        sz = round(float(s.get("size") or 0.0), 1)
                    except Exception:
                        sz = 0.0
                    if sz <= 0.0:
                        continue
                    size_freq[sz] = int(size_freq.get(sz, 0)) + 1
        body_size_hint = 0.0
        if size_freq:
            body_size_hint = float(sorted(size_freq.items(), key=lambda kv: kv[1], reverse=True)[0][0])

        rects: list[fitz.Rect] = []
        for b in blocks:
            if "lines" not in b:
                continue
            try:
                rect = fitz.Rect(b.get("bbox"))
            except Exception:
                continue
            if rect.width <= 0.0 or rect.height <= 0.0:
                continue

            lines: list[str] = []
            max_size = 0.0
            for l in (b.get("lines", []) or []):
                spans = l.get("spans", []) or []
                parts: list[str] = []
                for s in spans:
                    t = (s.get("text") or "")
                    if t and t.strip():
                        parts.append(t)
                    try:
                        sz = float(s.get("size") or 0.0)
                    except Exception:
                        sz = 0.0
                    if sz > max_size:
                        max_size = sz
                if parts:
                    lines.append(" ".join(parts).strip())
            if not lines:
                continue
            txt = _normalize_text(" ".join(lines)).strip()
            if not txt:
                continue

            if _is_non_body_metadata_text(
                txt,
                page_index=page_index,
                y0=float(rect.y0),
                y1=float(rect.y1),
                page_height=page_h,
                max_font_size=float(max_size),
                body_font_size=float(body_size_hint),
                is_references_page=bool(is_references_page),
            ):
                rects.append(rect)

        if len(rects) <= 1:
            return rects

        # Merge overlapping / near-touching metadata regions.
        rects = sorted(rects, key=lambda r: (float(r.y0), float(r.x0)))
        merged: list[fitz.Rect] = []
        for r in rects:
            if not merged:
                merged.append(fitz.Rect(r))
                continue
            last = merged[-1]
            inter = _rect_intersection_area(last, r)
            x_ov = _overlap_1d(float(last.x0), float(last.x1), float(r.x0), float(r.x1))
            min_w = max(1.0, min(float(last.width), float(r.width)))
            v_gap = max(0.0, max(float(last.y0), float(r.y0)) - min(float(last.y1), float(r.y1)))
            if inter > 0.0 or (x_ov / min_w >= 0.25 and v_gap <= 12.0):
                merged[-1] = fitz.Rect(
                    min(float(last.x0), float(r.x0)),
                    min(float(last.y0), float(r.y0)),
                    max(float(last.x1), float(r.x1)),
                    max(float(last.y1), float(r.y1)),
                )
            else:
                merged.append(fitz.Rect(r))
        return merged

    def _mask_rects_on_png(
        self,
        png_bytes: bytes,
        *,
        rects: List[fitz.Rect],
        page_width: float,
        page_height: float,
    ) -> bytes:
        """White-out metadata rectangles before sending screenshot to VL model."""
        if not png_bytes or not rects:
            return png_bytes
        try:
            import io
            from PIL import Image, ImageDraw

            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            draw = ImageDraw.Draw(img)
            sx = float(img.width) / max(1.0, float(page_width))
            sy = float(img.height) / max(1.0, float(page_height))
            pad = max(1, int(round(min(img.width, img.height) * 0.002)))
            for r in rects:
                x0 = int(max(0, min(img.width, round(float(r.x0) * sx) - pad)))
                y0 = int(max(0, min(img.height, round(float(r.y0) * sy) - pad)))
                x1 = int(max(0, min(img.width, round(float(r.x1) * sx) + pad)))
                y1 = int(max(0, min(img.height, round(float(r.y1) * sy) + pad)))
                if x1 <= x0 or y1 <= y0:
                    continue
                draw.rectangle([x0, y0, x1, y1], fill="white")
            out = io.BytesIO()
            img.save(out, format="PNG", optimize=True, compress_level=3)
            return out.getvalue()
        except Exception:
            return png_bytes

    @staticmethod
    def _looks_axis_or_panel_text(text: str) -> bool:
        t = _normalize_text(text or "").strip()
        if not t:
            return False
        low = t.lower()
        if re.fullmatch(r"\(?[a-z]\)?", t, flags=re.IGNORECASE):
            return True
        if len(t) <= 24 and re.fullmatch(r"[\d\.\-\+\s,;:()/%]+", t):
            return True
        if len(t) <= 32 and re.search(r"\d", t):
            return True
        if len(t) <= 32 and re.search(r"\b(?:wavelength|intensity|frequency|time|pixel|nm|hz|a\.?u\.?|x|y)\b", low):
            return True
        return False

    def _collect_page_text_line_boxes(self, page) -> list[tuple[fitz.Rect, str]]:
        if fitz is None:
            return []
        out: list[tuple[fitz.Rect, str]] = []
        try:
            d = page.get_text("dict") or {}
        except Exception:
            return out
        for b in (d.get("blocks", []) or []):
            for l in (b.get("lines", []) or []):
                spans = l.get("spans", []) or []
                txt = "".join(str(s.get("text", "")) for s in spans)
                txt = _normalize_text(txt).strip()
                if not txt:
                    continue
                try:
                    lb = l.get("bbox") or b.get("bbox")
                    r = fitz.Rect(lb)
                except Exception:
                    continue
                if r.width <= 0.0 or r.height <= 0.0:
                    continue
                out.append((r, txt))
        return out

    def _expanded_visual_crop_rect(
        self,
        *,
        rect: fitz.Rect,
        page_w: float,
        page_h: float,
        is_full_width: bool,
        line_boxes: list[tuple[fitz.Rect, str]] | None = None,
    ) -> fitz.Rect:
        """
        Expand figure crops conservatively, with extra bottom room for axes/ticks/panel labels,
        while avoiding accidental bleed into nearby body paragraphs.
        """
        r = fitz.Rect(rect)
        if r.width <= 0.0 or r.height <= 0.0:
            return r

        pad_x = max(2.0, float(page_w) * (0.004 if not is_full_width else 0.003))
        pad_top = max(2.0, float(page_h) * 0.006)
        pad_bottom = max(8.0, float(page_h) * (0.024 if not is_full_width else 0.020))

        lines = line_boxes or []
        probe_down = max(18.0, float(page_h) * 0.10)
        axis_bottom = float(r.y1)
        boundary_y0 = float(page_h) + 1.0

        for lb, txt in lines:
            if float(lb.y0) < float(r.y1) - 1.0:
                continue
            gap = float(lb.y0) - float(r.y1)
            if gap > probe_down:
                continue
            x_ov = _overlap_1d(float(r.x0), float(r.x1), float(lb.x0), float(lb.x1))
            min_w = max(1.0, min(float(r.width), float(lb.width)))
            if (x_ov / min_w) < 0.45:
                continue
            if self._looks_axis_or_panel_text(txt):
                axis_bottom = max(axis_bottom, float(lb.y1))
            else:
                # Long sentence-like text below figure is likely caption/body boundary.
                if len(txt) >= 24 and (len(re.findall(r"[A-Za-z]{2,}", txt)) >= 4):
                    boundary_y0 = min(boundary_y0, float(lb.y0))

        if axis_bottom > float(r.y1):
            pad_bottom = max(pad_bottom, axis_bottom - float(r.y1) + 2.0)

        x0 = max(0.0, float(r.x0) - pad_x)
        y0 = max(0.0, float(r.y0) - pad_top)
        y1 = min(float(page_h), float(r.y1) + pad_bottom)
        if boundary_y0 <= float(page_h):
            y1 = min(y1, max(float(r.y1) + 2.0, boundary_y0 - 1.0))
        if y1 <= y0:
            y1 = min(float(page_h), y0 + max(4.0, float(r.height) * 0.4))
        x1 = min(float(page_w), float(r.x1) + pad_x)
        if x1 <= x0:
            x1 = min(float(page_w), x0 + max(4.0, float(r.width) * 0.4))
        return fitz.Rect(x0, y0, x1, y1)

    def _vision_formula_overlay_enabled(self) -> bool:
        try:
            # Default OFF: keep structure/layout stable unless formula overlay is explicitly enabled.
            raw = str(os.environ.get("KB_PDF_VISION_FORMULA_OVERLAY", "0") or "0").strip().lower()
        except Exception:
            raw = "0"
        return raw in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _legacy_extra_cleanup_enabled() -> bool:
        try:
            # Default OFF: keep one deterministic cleanup pipeline (postprocess_markdown).
            # The legacy passes can over-adjust headings/references on clean VL output.
            raw = str(os.environ.get("KB_PDF_LEGACY_EXTRA_CLEANUP", "0") or "0").strip().lower()
        except Exception:
            raw = "0"
        return raw in {"1", "true", "yes", "y", "on"}

    def _vision_fragment_fallback_enabled(self) -> bool:
        try:
            # Default OFF to preserve full-page VL pipeline semantics.
            raw = str(os.environ.get("KB_PDF_VISION_FRAGMENT_FALLBACK", "0") or "0").strip().lower()
        except Exception:
            raw = "0"
        return raw in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _looks_like_overlay_math_line(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        low = t.lower()
        # Fast reject of obvious prose/caption lines.
        if low.startswith(("fig.", "figure ", "table ", "where ", "this ", "we ", "in this ", "note that ")):
            return False
        if ("@" in t) or ("http://" in low) or ("https://" in low) or ("www." in low):
            return False
        if re.search(r"\[\s*\d{1,4}(?:\s*,\s*\d{1,4})*\s*\]", t):
            return False
        word_n = len(re.findall(r"\b[A-Za-z]{2,}\b", t))
        strong_anchor_n = len(
            re.findall(
                r"(?:=|\^|_|\\frac|\\sum|\\int|\\prod|\\sqrt|\\min|\\max|\\arg|\\left|\\right|\\cdot|\\times|\\partial|\\nabla|\\mathbf|\\mathcal|\\hat|\\bar|\\tilde|/)",
                t,
                flags=re.IGNORECASE,
            )
        )
        bracket_anchor_n = len(re.findall(r"[\(\)\{\}\[\]]", t))
        # Strong equation anchors.
        if strong_anchor_n >= 2:
            return True
        # Equation-number-like fragment.
        if re.fullmatch(r"[\),;:\s]*\(\s*\d{1,4}\s*\)\s*", t):
            return True
        # Tiny fragment lines commonly produced from display equations.
        if len(t) <= 40:
            if re.fullmatch(r"[A-Za-z]|\d{1,3}", t):
                return True
            if re.search(r"\bDNN\b", t, flags=re.IGNORECASE):
                return True
            if any(ch in t for ch in "=+-*/^_") and word_n <= 5:
                return True
        # Avoid turning long prose into formulas.
        if word_n >= 12 and strong_anchor_n <= 1 and re.search(r"[.,:;]", t):
            return False
        if word_n >= 7 and strong_anchor_n == 0:
            return False
        if strong_anchor_n == 0 and bracket_anchor_n >= 2 and word_n >= 4:
            return False
        return False

    @staticmethod
    def _is_display_math_candidate_text(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        low = t.lower()
        if low.startswith(("fig.", "figure ", "table ", "where ")):
            return False
        if ("@" in t) or ("http://" in low) or ("https://" in low) or ("www." in low):
            return False
        if re.search(r"\[\s*\d{1,4}(?:\s*,\s*\d{1,4})*\s*\]", t):
            return False
        word_n = len(re.findall(r"\b[A-Za-z]{2,}\b", t))
        strong_anchor_n = len(
            re.findall(
                r"(?:=|\^|_|\\frac|\\sum|\\int|\\prod|\\sqrt|\\min|\\max|\\arg|\\left|\\right|\\cdot|\\times|\\partial|\\nabla|\\mathbf|\\mathcal|\\hat|\\bar|\\tilde|/)",
                t,
                flags=re.IGNORECASE,
            )
        )
        has_eqno = bool(re.search(r"\(\s*\d{1,4}\s*\)\s*$", t))
        if has_eqno:
            return True
        if ("\n" in t) and (strong_anchor_n >= 1):
            return True
        if strong_anchor_n >= 2 and len(t) >= 12:
            return True
        if word_n <= 5 and len(t) <= 48 and any(ch in t for ch in "=+-*/^_"):
            return True
        return False

    def _collect_display_math_candidates(
        self,
        page,
        *,
        page_index: int,
        is_references_page: bool,
    ) -> List[dict]:
        """
        Detect display-equation regions from page text geometry.
        We use this to mask formulas before full-page OCR and recover them separately.
        """
        if is_references_page:
            return []
        if fitz is None:
            return []
        try:
            d = page.get_text("dict") or {}
        except Exception:
            return []
        blocks = d.get("blocks", []) or []
        if not blocks:
            return []

        page_w = float(page.rect.width)
        page_h = float(page.rect.height)
        head_y = page_h * 0.08
        foot_y = page_h * 0.93

        line_items: list[tuple[fitz.Rect, str]] = []
        for b in blocks:
            if "lines" not in b:
                continue
            for ln in (b.get("lines", []) or []):
                spans = ln.get("spans", []) or []
                parts: list[str] = []
                for s in spans:
                    t = (s.get("text") or "")
                    if t and t.strip():
                        parts.append(t)
                txt = " ".join(parts).strip()
                if not txt:
                    continue
                try:
                    r = fitz.Rect(ln.get("bbox"))
                except Exception:
                    continue
                if r.width <= 0.0 or r.height <= 0.0:
                    continue
                if r.y1 < head_y or r.y0 > foot_y:
                    continue
                if not self._looks_like_overlay_math_line(txt):
                    continue
                line_items.append((r, txt))

        if not line_items:
            return []

        line_items = sorted(line_items, key=lambda it: (float(it[0].y0), float(it[0].x0)))
        groups: list[list[tuple[fitz.Rect, str]]] = []
        cur: list[tuple[fitz.Rect, str]] = []
        for r, t in line_items:
            if not cur:
                cur = [(r, t)]
                continue
            last_r = cur[-1][0]
            y_gap = max(0.0, float(r.y0) - float(last_r.y1))
            x_ol = _overlap_1d(float(last_r.x0), float(last_r.x1), float(r.x0), float(r.x1))
            min_w = max(1.0, min(float(last_r.width), float(r.width)))
            x_ratio = x_ol / min_w
            near = y_gap <= max(12.0, min(float(last_r.height), float(r.height)) * 1.2)
            if near and (x_ratio >= 0.12 or float(r.width) <= 70.0 or float(last_r.width) <= 70.0):
                cur.append((r, t))
            else:
                groups.append(cur)
                cur = [(r, t)]
        if cur:
            groups.append(cur)

        candidates: list[dict] = []
        for gi, g in enumerate(groups):
            rect = _union_rect([x[0] for x in g]) or g[0][0]
            text = "\n".join(x[1] for x in g).strip()
            if not text:
                continue
            if not self._is_display_math_candidate_text(text):
                continue
            # Avoid huge prose regions.
            word_n = len(re.findall(r"\b[A-Za-z]{2,}\b", text))
            if word_n >= 28 and len(text) >= 180:
                continue
            # Display-ish geometry: not too tiny and not whole paragraph width+height.
            if rect.width < page_w * 0.08:
                continue
            if rect.height > page_h * 0.35:
                continue
            candidates.append({"group_index": gi, "rect": fitz.Rect(rect), "text": text})

        if not candidates:
            return []

        # Merge overlapping candidates.
        candidates = sorted(candidates, key=lambda c: (float(c["rect"].y0), float(c["rect"].x0)))
        merged: list[dict] = []
        for c in candidates:
            if not merged:
                merged.append(c)
                continue
            last = merged[-1]
            r0 = last["rect"]
            r1 = c["rect"]
            inter = _rect_intersection_area(r0, r1)
            x_ol = _overlap_1d(float(r0.x0), float(r0.x1), float(r1.x0), float(r1.x1))
            min_w = max(1.0, min(float(r0.width), float(r1.width)))
            y_gap = max(0.0, max(float(r0.y0), float(r1.y0)) - min(float(r0.y1), float(r1.y1)))
            if inter > 0.0 or (x_ol / min_w >= 0.22 and y_gap <= 10.0):
                last["rect"] = fitz.Rect(
                    min(float(r0.x0), float(r1.x0)),
                    min(float(r0.y0), float(r1.y0)),
                    max(float(r0.x1), float(r1.x1)),
                    max(float(r0.y1), float(r1.y1)),
                )
                last["text"] = (str(last.get("text") or "") + "\n" + str(c.get("text") or "")).strip()
            else:
                merged.append(c)

        try:
            max_n = int(os.environ.get("KB_PDF_VISION_FORMULA_MAX_PER_PAGE", "14") or "14")
        except Exception:
            max_n = 14
        max_n = max(0, min(40, max_n))
        return merged[:max_n]

    def _extract_formula_placeholders_for_page(
        self,
        page,
        *,
        page_index: int,
        candidates: List[dict],
        base_dpi: int,
    ) -> tuple[list[tuple[fitz.Rect, str]], dict[str, str]]:
        """
        Build placeholder labels and recover each display equation via equation-image OCR.
        Returns:
        - list of (rect, placeholder_token) for masking
        - mapping {placeholder_token: rendered_display_math_markdown}
        """
        if not candidates:
            return [], {}
        if (not self.cfg.llm) or (not self.llm_worker) or (not getattr(self.llm_worker, "_client", None)):
            return [], {}

        try:
            formula_dpi = int(os.environ.get("KB_PDF_VISION_FORMULA_DPI", "") or "0")
        except Exception:
            formula_dpi = 0
        if formula_dpi <= 0:
            formula_dpi = max(320, int(base_dpi))
        formula_dpi = max(220, min(600, formula_dpi))

        placeholder_rects: list[tuple[fitz.Rect, str]] = []
        placeholder_map: dict[str, str] = {}
        for idx, cand in enumerate(candidates, start=1):
            try:
                r = fitz.Rect(cand.get("rect"))
            except Exception:
                continue
            if r.width <= 2.0 or r.height <= 2.0:
                continue

            pad_x = max(3.0, float(r.width) * 0.08)
            pad_y = max(4.0, float(r.height) * 0.24)
            clip = fitz.Rect(
                max(0.0, float(r.x0) - pad_x),
                max(0.0, float(r.y0) - pad_y),
                min(float(page.rect.width), float(r.x1) + pad_x),
                min(float(page.rect.height), float(r.y1) + pad_y),
            )
            if clip.width <= 4.0 or clip.height <= 4.0:
                continue

            try:
                pix = page.get_pixmap(clip=clip, dpi=formula_dpi)
                png = pix.tobytes("png")
            except Exception:
                continue

            latex = self.llm_worker.call_llm_repair_math_from_image(
                png,
                page_number=page_index,
                block_index=int(cand.get("group_index", idx - 1) or 0),
            )
            if not latex:
                continue

            latex = (latex or "").strip().replace("$", "")
            if len(latex) < 3:
                continue

            # Keep equation numbering when line text has a visible "(N)" and LaTeX lacks \tag.
            try:
                if "\\tag{" not in latex:
                    m_no = re.search(r"\(\s*(\d{1,4})\s*\)\s*$", str(cand.get("text") or ""))
                    if m_no:
                        latex = f"{latex} \\tag{{{m_no.group(1)}}}"
            except Exception:
                pass

            token = f"[[EQ_{idx}]]"
            placeholder_rects.append((r, token))
            placeholder_map[token] = f"$$\n{latex}\n$$"

        return placeholder_rects, placeholder_map

    def _mask_rects_with_labels_on_png(
        self,
        png_bytes: bytes,
        *,
        labeled_rects: List[tuple[fitz.Rect, str]],
        page_width: float,
        page_height: float,
    ) -> bytes:
        """White-out rectangles and draw placeholder labels so VL keeps equation positions."""
        if not png_bytes or not labeled_rects:
            return png_bytes
        try:
            import io
            from PIL import Image, ImageDraw, ImageFont

            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            draw = ImageDraw.Draw(img)
            sx = float(img.width) / max(1.0, float(page_width))
            sy = float(img.height) / max(1.0, float(page_height))
            for r, label in labeled_rects:
                x0 = int(max(0, min(img.width, round(float(r.x0) * sx))))
                y0 = int(max(0, min(img.height, round(float(r.y0) * sy))))
                x1 = int(max(0, min(img.width, round(float(r.x1) * sx))))
                y1 = int(max(0, min(img.height, round(float(r.y1) * sy))))
                if x1 <= x0 or y1 <= y0:
                    continue
                draw.rectangle([x0, y0, x1, y1], fill="white")

                # Draw a visible placeholder token in the masked region.
                box_h = max(10, y1 - y0)
                font_size = max(12, min(36, int(box_h * 0.45)))
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except Exception:
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                    except Exception:
                        font = ImageFont.load_default()
                label_s = str(label or "").strip()
                if not label_s:
                    continue
                try:
                    tb = draw.textbbox((0, 0), label_s, font=font)
                    tw = max(1, int(tb[2] - tb[0]))
                    th = max(1, int(tb[3] - tb[1]))
                except Exception:
                    tw = max(1, len(label_s) * 8)
                    th = max(1, font_size)
                tx = x0 + max(2, ((x1 - x0) - tw) // 2)
                ty = y0 + max(2, ((y1 - y0) - th) // 2)
                draw.text((tx, ty), label_s, fill="black", font=font)

            out = io.BytesIO()
            img.save(out, format="PNG", optimize=True, compress_level=3)
            return out.getvalue()
        except Exception:
            return png_bytes

    @staticmethod
    def _formula_placeholder_hint(placeholder_map: dict[str, str]) -> str:
        if not placeholder_map:
            return ""
        keys = list(placeholder_map.keys())
        if not keys:
            return ""
        shown = ", ".join(keys[:8])
        more = len(keys) - min(len(keys), 8)
        if more > 0:
            shown = f"{shown} (+{more} more)"
        return (
            "The page image contains equation placeholders. "
            f"Keep placeholder tokens EXACTLY as written: {shown}. "
            "Do not alter, translate, split, merge, or delete these tokens."
        )

    @staticmethod
    def _restore_formula_placeholders(md: str, placeholder_map: dict[str, str]) -> str:
        if not md or not placeholder_map:
            return md
        out = md
        for token, formula_md in placeholder_map.items():
            tok = str(token or "").strip()
            if not tok:
                continue
            # Accept exact token and whitespace-tolerant bracket variant only.
            core = re.sub(r"[\[\]\s]", "", tok)
            if not core:
                continue
            pat = re.compile(rf"(?i)(?:{re.escape(tok)}|\[\[\s*{re.escape(core)}\s*\]\])")
            # IMPORTANT: use a function replacement to avoid re-sub parsing backslashes
            # in LaTeX (e.g., \sum, \mu, \text) as escape sequences.
            out = pat.sub(lambda _m, rep=str(formula_md): rep, out, count=1)
        return out

    @staticmethod
    def _looks_fragmented_math_output(md: str) -> bool:
        """
        Detect page-level VL markdown where one equation is exploded into many tiny pieces,
        e.g. isolated lines like "N", "( DNN ( u n", ")(12)" with several tiny $$ blocks.
        This is a quality gate for deciding whether to retry/fallback the whole page.
        """
        text = (md or "").strip()
        if not text:
            return False

        lines = text.splitlines()
        if len(lines) < 8:
            return False

        in_math = False
        cur_block: list[str] = []
        math_blocks: list[str] = []
        non_math_lines: list[str] = []

        for ln in lines:
            st = (ln or "").strip()
            if st == "$$":
                if in_math:
                    blk = " ".join(x for x in cur_block if x).strip()
                    if blk:
                        math_blocks.append(blk)
                    cur_block = []
                in_math = not in_math
                continue
            if in_math:
                cur_block.append(st)
            else:
                non_math_lines.append(st)

        if in_math and cur_block:
            blk = " ".join(x for x in cur_block if x).strip()
            if blk:
                math_blocks.append(blk)

        if not math_blocks:
            return False

        def _math_anchor_count(s: str) -> int:
            return len(
                re.findall(
                    r"(?:=|\^|_|\\frac|\\sum|\\int|\\prod|\\sqrt|\\min|\\max|\\arg|\\left|\\right|\\cdot|\\times|\\nabla|\\partial|\||\{|\})",
                    s,
                    flags=re.IGNORECASE,
                )
            )

        tiny_math = 0
        long_prose_in_math = 0
        for b in math_blocks:
            bb = re.sub(r"\s+", " ", b).strip()
            if not bb:
                continue
            word_n = len(re.findall(r"[A-Za-z]{2,}", bb))
            anchor_n = _math_anchor_count(bb)
            if len(bb) <= 42 or (word_n <= 4 and len(bb) <= 80):
                tiny_math += 1
            # Sentence-like long text wrapped in $$ is usually a bad VL split.
            if word_n >= 18 and anchor_n <= 3 and re.search(r"[.,:;]", bb):
                long_prose_in_math += 1

        eqno_lines = 0
        fragment_lines = 0
        for st in non_math_lines:
            if not st:
                continue
            # Equation number fragments: "(12)", ")(16)", ", (7)"
            if re.fullmatch(r"[\),;:\s]*\(\s*\d{1,4}\s*\)\s*", st):
                eqno_lines += 1
                continue
            # Isolated symbol/variable line, e.g. "N"
            if re.fullmatch(r"[A-Za-z]|\d{1,3}", st):
                fragment_lines += 1
                continue
            # OCR-spaced variable fragments with brackets/operators.
            spaced_var = bool(re.search(r"(?:\b[A-Za-z]\b\s+){2,}\b[A-Za-z]\b", st))
            has_math_punct = any(ch in st for ch in "()[]{}_=+-/*")
            if len(st) <= 46 and has_math_punct and (
                spaced_var or re.search(r"\bDNN\b", st, flags=re.IGNORECASE)
            ):
                fragment_lines += 1

        if long_prose_in_math >= 1 and tiny_math >= 2:
            return True
        # Mixed case: one seemingly complete equation plus many orphan shard lines
        # (e.g., "M", "(DNN(u*),v*)", ")(15)", "n", "n").
        if len(math_blocks) >= 1 and fragment_lines >= 2 and (eqno_lines >= 1 or tiny_math >= 1):
            return True
        # Two or more tiny math blocks plus orphan equation-number/fragment lines
        # is a strong signal that one display equation was split apart.
        if len(math_blocks) >= 2 and tiny_math >= 2 and (fragment_lines >= 2 or eqno_lines >= 1):
            return True
        if len(math_blocks) >= 3 and tiny_math >= 3 and (fragment_lines >= 2 or eqno_lines >= 1):
            return True
        if tiny_math >= 4 and fragment_lines >= 2:
            return True
        return False

    def _vision_math_quality_gate_enabled(self) -> bool:
        try:
            raw = str(os.environ.get("KB_PDF_VISION_MATH_QUALITY_GATE", "1") or "1").strip().lower()
        except Exception:
            raw = "1"
        return raw in {"1", "true", "yes", "y", "on"}

    def _vision_math_retry_hint(self, page_hint: str) -> str:
        extra = (
            "Formula quality warning: previous OCR produced fragmented equations. "
            "You MUST output each complete equation as ONE coherent expression (inline or one $$...$$ block). "
            "Never output isolated fragments such as single symbols/indices/limits (e.g., 'N', 'i=1', '(12)') on separate lines."
        )
        base = (page_hint or "").strip()
        return (base + " " + extra).strip() if base else extra

    @staticmethod
    def _vision_empty_retry_attempts() -> int:
        try:
            raw = str(os.environ.get("KB_PDF_VISION_EMPTY_RETRY", "2") or "2").strip()
            n = int(raw)
        except Exception:
            n = 2
        return max(0, min(5, n))

    @staticmethod
    def _vision_empty_retry_backoff_s() -> float:
        try:
            raw = str(os.environ.get("KB_PDF_VISION_EMPTY_RETRY_BACKOFF_S", "1.2") or "1.2").strip()
            v = float(raw)
        except Exception:
            v = 1.2
        return max(0.0, min(8.0, v))

    @staticmethod
    def _vision_references_column_mode_enabled() -> bool:
        try:
            raw = str(os.environ.get("KB_PDF_VISION_REFS_COLUMN_MODE", "1") or "1").strip().lower()
        except Exception:
            raw = "1"
        return raw in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _vision_layout_crop_mode_enabled() -> bool:
        try:
            raw = str(os.environ.get("KB_PDF_VISION_LAYOUT_CROP_MODE", "0") or "0").strip().lower()
        except Exception:
            raw = "0"
        return raw in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _is_probable_page_number_block(block: TextBlock, *, page_w: float, page_h: float) -> bool:
        try:
            x0, y0, x1, y1 = (float(v) for v in block.bbox)
        except Exception:
            return False
        txt = _normalize_text(getattr(block, "text", "") or "").strip()
        if not re.fullmatch(r"\d{1,4}", txt):
            return False
        if (y0 + y1) * 0.5 < page_h * 0.92:
            return False
        cx = (x0 + x1) * 0.5
        if abs(cx - (page_w * 0.5)) > page_w * 0.12:
            return False
        return (x1 - x0) <= page_w * 0.10

    @staticmethod
    def _structured_crop_lane_for_bbox(
        bbox: tuple[float, float, float, float],
        *,
        page_w: float,
        col_split: float,
    ) -> str:
        x0, _, x1, _ = (float(v) for v in bbox)
        w = max(0.0, x1 - x0)
        cross_margin = max(18.0, page_w * 0.11)
        crosses = x0 < (col_split - cross_margin) and x1 > (col_split + cross_margin)
        if w >= page_w * 0.72 or (crosses and w >= page_w * 0.48):
            return "full"
        return "left" if ((x0 + x1) * 0.5) < col_split else "right"

    @staticmethod
    def _expand_structured_crop_rect(
        rect: "fitz.Rect",
        *,
        lane: str,
        page_w: float,
        page_h: float,
        col_split: float,
    ) -> "fitz.Rect":
        r = fitz.Rect(rect)
        x_pad = max(6.0, page_w * 0.012)
        y_pad = max(4.0, page_h * 0.008)
        gutter = max(8.0, page_w * 0.018)
        x0 = max(0.0, float(r.x0) - x_pad)
        y0 = max(0.0, float(r.y0) - y_pad)
        x1 = min(float(page_w), float(r.x1) + x_pad)
        y1 = min(float(page_h), float(r.y1) + y_pad)
        if lane == "left":
            x1 = min(x1, float(col_split) - gutter)
        elif lane == "right":
            x0 = max(x0, float(col_split) + gutter)
        return fitz.Rect(x0, y0, max(x0 + 1.0, x1), max(y0 + 1.0, y1))

    @staticmethod
    def _fallback_markdown_from_blocks(blocks: list[TextBlock]) -> str:
        if not blocks:
            return ""
        parts: list[str] = []
        for block in blocks:
            if bool(getattr(block, "is_table", False)) and getattr(block, "table_markdown", None):
                md = str(block.table_markdown or "").strip()
                if md:
                    parts.append(md)
                continue
            txt = str(getattr(block, "text", "") or "").strip()
            if (not txt) or txt == "[TABLE]":
                continue
            lvl = str(getattr(block, "heading_level", "") or "").strip()
            m_lvl = re.fullmatch(r"\[H([1-6])\]", lvl)
            if m_lvl:
                parts.append(("#" * int(m_lvl.group(1))) + f" {txt}")
            elif bool(getattr(block, "is_caption", False)) and (not txt.startswith("*")):
                parts.append(f"*{txt}*")
            else:
                parts.append(txt)
        return "\n\n".join(p for p in parts if p.strip()).strip()

    def _convert_page_with_layout_crops(
        self,
        *,
        page,
        page_index: int,
        total_pages: int,
        page_hint: str,
        speed_mode: str,
        pdf_path: Path,
        assets_dir: Path,
        image_names: list[str],
    ) -> Optional[str]:
        if page is None or image_names:
            return None
        if page_index == 0:
            return None
        if not self._vision_layout_crop_mode_enabled():
            return None

        try:
            page_w = float(page.rect.width)
            page_h = float(page.rect.height)
        except Exception:
            return None
        if page_w <= 2 or page_h <= 2:
            return None

        try:
            page_dict = page.get_text("dict") or {}
        except Exception:
            page_dict = {}
        has_table_hint = bool(_page_maybe_has_table_from_dict(page_dict))
        if not has_table_hint:
            return None

        try:
            setattr(page, "has_table_hint", bool(has_table_hint))
        except Exception:
            pass

        try:
            tables = _extract_tables_by_layout(
                page,
                pdf_path=pdf_path,
                page_index=page_index,
                visual_rects=[],
                use_pdfplumber_fallback=True,
            )
        except Exception:
            tables = []
        if not tables:
            return None

        try:
            body_size = detect_body_font_size([page])
        except Exception:
            body_size = 10.0

        try:
            blocks = self._extract_text_blocks(
                page,
                page_index=page_index,
                body_size=body_size,
                tables=tables,
                visual_rects=[],
                assets_dir=assets_dir,
                is_references_page=False,
            )
        except Exception:
            return None
        blocks = [
            b for b in (blocks or [])
            if not self._is_probable_page_number_block(b, page_w=page_w, page_h=page_h)
        ]
        if len(blocks) < 3:
            return None

        col_split = _detect_column_split_x(blocks, page_width=page_w)
        if col_split is None:
            return None

        annotated: list[dict] = []
        for block in blocks:
            try:
                bbox = tuple(float(v) for v in block.bbox)
            except Exception:
                continue
            lane = self._structured_crop_lane_for_bbox(
                bbox,
                page_w=page_w,
                col_split=float(col_split),
            )
            annotated.append(
                {
                    "block": block,
                    "lane": lane,
                    "rect": fitz.Rect(bbox),
                }
            )
        if len(annotated) < 3:
            return None

        def _union_items(items: list[dict]) -> "fitz.Rect":
            r = fitz.Rect(items[0]["rect"])
            for row in items[1:]:
                r |= fitz.Rect(row["rect"])
            return r

        def _cluster_by_gap(items: list[dict], *, max_gap: float) -> list[list[dict]]:
            if not items:
                return []
            items = sorted(items, key=lambda row: (float(row["rect"].y0), float(row["rect"].x0)))
            groups: list[list[dict]] = [[items[0]]]
            for row in items[1:]:
                prev = groups[-1][-1]
                gap = float(row["rect"].y0) - float(prev["rect"].y1)
                if gap <= max_gap:
                    groups[-1].append(row)
                else:
                    groups.append([row])
            return groups

        full_items = [row for row in annotated if row["lane"] == "full"]
        full_groups = _cluster_by_gap(full_items, max_gap=max(8.0, page_h * 0.015))
        full_group_rects = [{"blocks": grp, "rect": _union_items(grp)} for grp in full_groups if grp]

        regions: list[dict] = []
        order_idx = 0
        cursor_y = 0.0

        def _append_interval_regions(y_start: float, y_end: float) -> None:
            nonlocal order_idx
            if y_end <= y_start + 2.0:
                return
            segment_rows = []
            for row in annotated:
                if row["lane"] == "full":
                    continue
                cy = (float(row["rect"].y0) + float(row["rect"].y1)) * 0.5
                if y_start <= cy <= y_end:
                    segment_rows.append(row)
            if not segment_rows:
                return
            for lane in ("left", "right"):
                lane_rows = [row for row in segment_rows if row["lane"] == lane]
                if not lane_rows:
                    continue
                order_idx += 1
                rect = self._expand_structured_crop_rect(
                    _union_items(lane_rows),
                    lane=lane,
                    page_w=page_w,
                    page_h=page_h,
                    col_split=float(col_split),
                )
                regions.append(
                    {
                        "order": order_idx,
                        "lane": lane,
                        "rect": rect,
                        "blocks": [row["block"] for row in lane_rows],
                        "fallback_md": self._fallback_markdown_from_blocks([row["block"] for row in lane_rows]),
                    }
                )

        for fg in sorted(full_group_rects, key=lambda row: (float(row["rect"].y0), float(row["rect"].x0))):
            _append_interval_regions(cursor_y, float(fg["rect"].y0) - 2.0)
            order_idx += 1
            regions.append(
                {
                    "order": order_idx,
                    "lane": "full",
                    "rect": self._expand_structured_crop_rect(
                        fg["rect"],
                        lane="full",
                        page_w=page_w,
                        page_h=page_h,
                        col_split=float(col_split),
                    ),
                    "blocks": [row["block"] for row in fg["blocks"]],
                    "fallback_md": self._fallback_markdown_from_blocks([row["block"] for row in fg["blocks"]]),
                }
            )
            cursor_y = float(fg["rect"].y1) + 2.0

        _append_interval_regions(cursor_y, float(page_h))

        regions = [row for row in regions if row.get("fallback_md") or row.get("blocks")]
        if len(regions) <= 1:
            return None

        try:
            raw_dpi = str(os.environ.get("KB_PDF_VISION_DPI", "") or "").strip()
            dpi = int(raw_dpi) if raw_dpi else 0
        except Exception:
            dpi = 0
        if dpi <= 0:
            try:
                dpi = int((getattr(self, "_active_speed_config", None) or {}).get("dpi", 220) or 220)
            except Exception:
                dpi = int(getattr(self, "dpi", 220) or 220)
        dpi = max(220, min(420, int(dpi)))

        try:
            print(
                f"[VISION_DIRECT][LAYOUT] page {page_index+1}: structured crop mode enabled ({len(regions)} crops, dpi={int(dpi)})",
                flush=True,
            )
        except Exception:
            pass

        def _ocr_region(row: dict) -> tuple[int, str]:
            rid = int(row["order"])
            lane = str(row["lane"])
            rect = fitz.Rect(row["rect"])
            fallback_md = str(row.get("fallback_md", "") or "").strip()
            try:
                pix = page.get_pixmap(clip=rect, dpi=int(dpi), alpha=False)
                png = pix.tobytes("png")
            except Exception:
                return rid, fallback_md
            if not png:
                return rid, fallback_md
            hint = (
                (page_hint + " " if page_hint else "")
                + f"This is structured crop {rid}/{len(regions)} from a paper page. "
                  f"It covers the {lane} layout region only. "
                  "Output only the content visible in this crop, in normal reading order, as Markdown. "
                  "Preserve formulas, tables, captions, and headings exactly when present. "
                  "Do not invent or repeat content from other regions."
            )
            t0 = time.time()
            md_part = self.llm_worker.call_llm_page_to_markdown(
                png,
                page_number=page_index,
                total_pages=total_pages,
                hint=hint,
                speed_mode=speed_mode,
                is_references_page=False,
            )
            md_part = (md_part or "").strip() or fallback_md

            def _table_presence_key(table_md: str) -> str:
                lines = [ln.strip() for ln in (table_md or "").splitlines() if ln.strip()]
                for ln in lines:
                    if ln.startswith("|") and "---" in ln:
                        continue
                    if ln.startswith("|"):
                        key = re.sub(r"[|`*_]", " ", _normalize_text(ln))
                        key = re.sub(r"\s+", " ", key).strip().lower()
                        if key:
                            return key[:160]
                key = re.sub(r"[|`*_]", " ", _normalize_text(table_md or ""))
                key = re.sub(r"\s+", " ", key).strip().lower()
                return key[:160]

            table_rows: list[tuple[int, str]] = []
            for pos, block in enumerate(row.get("blocks") or []):
                if bool(getattr(block, "is_table", False)) and getattr(block, "table_markdown", None):
                    table_rows.append((pos, str(block.table_markdown or "").strip()))
            if table_rows:
                md_norm = re.sub(r"\s+", " ", _normalize_text(md_part or "")).strip().lower()
                missing_tables: list[tuple[int, str]] = []
                for pos, table_md in table_rows:
                    key = _table_presence_key(table_md)
                    if key and key in md_norm:
                        continue
                    missing_tables.append((pos, table_md))
                if missing_tables:
                    first_non_table_idx = next(
                        (
                            idx for idx, block in enumerate(row.get("blocks") or [])
                            if not bool(getattr(block, "is_table", False))
                        ),
                        len(row.get("blocks") or []),
                    )
                    prefix_tables = [table_md for pos, table_md in missing_tables if pos <= first_non_table_idx]
                    suffix_tables = [table_md for pos, table_md in missing_tables if pos > first_non_table_idx]
                    md_part = "\n\n".join(
                        part for part in (
                            *(prefix_tables or []),
                            md_part,
                            *(suffix_tables or []),
                        )
                        if part
                    ).strip()
            try:
                print(
                    f"[VISION_DIRECT][LAYOUT] page {page_index+1} crop {rid}/{len(regions)} ({lane}) done ({time.time()-t0:.1f}s, {len(md_part)} chars)",
                    flush=True,
                )
            except Exception:
                pass
            return rid, md_part

        ordered_md: dict[int, str] = {}
        max_workers = min(4, max(1, len(regions)))
        if max_workers <= 1:
            for row in regions:
                rid, md_part = _ocr_region(row)
                if md_part:
                    ordered_md[rid] = md_part
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = [pool.submit(_ocr_region, row) for row in regions]
                for fut in as_completed(futs):
                    try:
                        rid, md_part = fut.result()
                    except Exception:
                        continue
                    if md_part:
                        ordered_md[rid] = md_part

        parts = [ordered_md[k].strip() for k in sorted(ordered_md.keys()) if ordered_md.get(k)]
        if not parts:
            return None
        merged = "\n\n".join(part for part in parts if part).strip()
        return merged or None

    @staticmethod
    def _is_reference_placeholder_line(text: str) -> bool:
        t = _normalize_text(text or "").strip().lower()
        if not t:
            return False
        if "(incomplete visible)" in t:
            return True
        if "(partially visible)" in t:
            return True
        if "(not fully visible)" in t:
            return True
        if "[unreadable]" in t or "[illegible]" in t:
            return True
        return False

    def _sanitize_reference_crop_markdown(self, md: str) -> str:
        if not md:
            return ""
        out: list[str] = []
        for ln in (md or "").splitlines():
            s = (ln or "").strip()
            if not s:
                continue
            if self._is_reference_placeholder_line(s):
                continue
            # References crop should not contain markdown headings/code/maths.
            if s.startswith("```"):
                continue
            s = s.replace("$$", "").replace("$", "").strip()
            if re.match(r"^#{1,6}\s+", s):
                s = re.sub(r"^#{1,6}\s+", "", s).strip()
            out.append(s)
        return "\n".join(out).strip()

    def _merge_reference_crop_markdowns(self, parts: list[str]) -> str:
        """
        Merge per-column references OCR with de-duplication and stable numbering.
        """
        if not parts:
            return ""
        numbered_best: dict[int, str] = {}
        extras: list[str] = []
        extra_seen: set[str] = set()

        for part in parts:
            for raw in (part or "").splitlines():
                s = _normalize_text(raw or "").strip()
                if not s:
                    continue
                if self._is_reference_placeholder_line(s):
                    continue
                m = re.match(r"^\[?(\d{1,4})\]?[.)]?\s+(.+)$", s)
                if m:
                    try:
                        n = int(m.group(1))
                    except Exception:
                        n = -1
                    body = (m.group(2) or "").strip()
                    if n <= 0 or n > 2000 or (not body):
                        continue
                    line = f"[{n}] {body}"
                    prev = numbered_best.get(n)
                    if (prev is None) or (len(line) > len(prev)):
                        numbered_best[n] = line
                    continue
                key = re.sub(r"\s+", " ", s).strip().lower()
                if key in extra_seen:
                    continue
                extra_seen.add(key)
                extras.append(s)

        out: list[str] = []
        if numbered_best:
            for n in sorted(numbered_best.keys()):
                out.append(numbered_best[n])
        out.extend(extras)
        return "\n".join(out).strip()

    def _build_reference_column_crop_rects(
        self,
        *,
        page,
        page_w: float,
        page_h: float,
    ) -> list[fitz.Rect]:
        """
        Build robust references column crops using text-driven split when possible.
        Falls back to symmetric half-page crops.
        """
        top_pad = float(page_h) * 0.015
        bot_pad = float(page_h) * 0.02
        y0 = max(0.0, top_pad)
        y1 = max(y0 + 1.0, float(page_h) - bot_pad)

        mid = float(page_w) * 0.5
        overlap = float(page_w) * 0.045
        fallback = [
            fitz.Rect(0.0, y0, min(float(page_w), mid + overlap), y1),
            fitz.Rect(max(0.0, mid - overlap), y0, float(page_w), y1),
        ]

        try:
            d = page.get_text("dict") or {}
            line_boxes: list[tuple[float, float, float, float]] = []
            for b in d.get("blocks", []) or []:
                if "lines" not in b:
                    continue
                for l in (b.get("lines", []) or []):
                    bbox = l.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    spans = l.get("spans", []) or []
                    txt = _normalize_text("".join(str(s.get("text", "")) for s in spans)).strip()
                    if len(txt) < 2:
                        continue
                    try:
                        x0, y0b, x1, y1b = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                    except Exception:
                        continue
                    if (x1 - x0) <= 8.0 or (y1b - y0b) <= 2.0:
                        continue
                    if y1b < y0 or y0b > y1:
                        continue
                    # Ignore full-width lines when estimating column split.
                    if (x1 - x0) >= float(page_w) * 0.75:
                        continue
                    line_boxes.append((x0, y0b, x1, y1b))

            if len(line_boxes) < 10:
                return fallback

            centers = sorted(((x0 + x1) * 0.5 for x0, _, x1, _ in line_boxes))
            best_gap = 0.0
            best_mid = None
            lo = float(page_w) * 0.22
            hi = float(page_w) * 0.78
            for i in range(len(centers) - 1):
                a = float(centers[i])
                b = float(centers[i + 1])
                g = b - a
                m = (a + b) * 0.5
                if m < lo or m > hi:
                    continue
                if g > best_gap:
                    best_gap = g
                    best_mid = m

            if best_mid is None or best_gap < float(page_w) * 0.075:
                return fallback

            gutter = max(10.0, float(page_w) * 0.022)
            left_x1 = max(0.0, min(float(page_w), float(best_mid) - gutter))
            right_x0 = max(0.0, min(float(page_w), float(best_mid) + gutter))
            if left_x1 <= float(page_w) * 0.18 or right_x0 >= float(page_w) * 0.82:
                return fallback
            if (right_x0 - left_x1) < float(page_w) * 0.04:
                return fallback

            return [
                fitz.Rect(0.0, y0, left_x1, y1),
                fitz.Rect(right_x0, y0, float(page_w), y1),
            ]
        except Exception:
            return fallback

    def _convert_references_page_with_column_vl(
        self,
        *,
        page,
        page_index: int,
        total_pages: int,
        page_hint: str,
        speed_mode: str,
    ) -> Optional[str]:
        """
        Dense references pages are likely to timeout as one large VL request.
        Use two overlapping column crops and OCR each crop separately.
        """
        if page is None:
            return None
        try:
            page_w = float(page.rect.width)
            page_h = float(page.rect.height)
        except Exception:
            return None
        if page_w <= 2 or page_h <= 2:
            return None

        try:
            raw_dpi = str(os.environ.get("KB_PDF_VISION_DPI", "") or "").strip()
            dpi = int(raw_dpi) if raw_dpi else 0
        except Exception:
            dpi = 0
        if dpi <= 0:
            try:
                dpi = int((getattr(self, "_active_speed_config", None) or {}).get("dpi", 220) or 220)
            except Exception:
                dpi = int(getattr(self, "dpi", 220) or 220)
        # Dense reference text tends to be tiny; keep a higher floor for OCR fidelity.
        dpi = max(240, min(500, int(dpi)))

        rects = self._build_reference_column_crop_rects(
            page=page,
            page_w=float(page_w),
            page_h=float(page_h),
        )

        try:
            print(
                f"[VISION_DIRECT][REFS] page {page_index+1}: column mode enabled ({len(rects)} crops, dpi={int(dpi)})",
                flush=True,
            )
        except Exception:
            pass

        part_payloads: list[tuple[int, bytes]] = []
        for idx, rect in enumerate(rects, start=1):
            try:
                pix = page.get_pixmap(clip=rect, dpi=int(dpi), alpha=False)
                part_png = pix.tobytes("png")
                if part_png:
                    part_payloads.append((idx, part_png))
            except Exception:
                continue

        def _ocr_ref_crop(idx: int, part_png: bytes) -> tuple[int, Optional[str]]:
            t0 = time.time()
            col_hint = (
                (page_hint + " " if page_hint else "")
                + f"This is column crop {idx}/2 of a references page. "
                  "Output only complete references fully visible in this crop, one per line. "
                  "Do NOT output placeholders like '(incomplete visible)' or 'unreadable'. "
                  "If an entry is clipped/uncertain, skip it."
            )
            part_md = self.llm_worker.call_llm_page_to_markdown(
                part_png,
                page_number=page_index,
                total_pages=total_pages,
                hint=col_hint,
                speed_mode=speed_mode,
                is_references_page=True,
            )
            part_md = self._sanitize_reference_crop_markdown((part_md or "").strip())
            part_md = part_md or None
            try:
                elapsed = time.time() - t0
                s = f"{len(part_md)} chars" if part_md else "empty"
                print(
                    f"[VISION_DIRECT][REFS] page {page_index+1} crop {idx}/2 done ({elapsed:.1f}s, {s})",
                    flush=True,
                )
            except Exception:
                pass
            return idx, part_md

        ordered: dict[int, str] = {}
        if len(part_payloads) <= 1:
            for idx, part_png in part_payloads:
                i2, md2 = _ocr_ref_crop(idx, part_png)
                if md2:
                    ordered[i2] = md2
        else:
            with ThreadPoolExecutor(max_workers=min(2, len(part_payloads))) as pool:
                futs = [pool.submit(_ocr_ref_crop, idx, part_png) for idx, part_png in part_payloads]
                for fut in as_completed(futs):
                    try:
                        i2, md2 = fut.result()
                    except Exception:
                        continue
                    if md2:
                        ordered[i2] = md2

        parts = [ordered[k] for k in sorted(ordered.keys()) if ordered.get(k)]

        if not parts:
            return None
        merged = self._merge_reference_crop_markdowns(parts)
        return merged or None

    def _convert_page_with_vision_guardrails(
        self,
        *,
        png_bytes: bytes,
        page,
        page_index: int,
        total_pages: int,
        page_hint: str,
        speed_mode: str,
        is_references_page: bool,
        pdf_path: Path,
        assets_dir: Path,
        image_names: Optional[list[str]] = None,
        formula_placeholders: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Run vision-direct page OCR with a math quality gate:
        1) normal VL page conversion
        2) retry once with stricter hint if math is fragmented
        3) fallback to block pipeline when fragmentation persists
        """
        if is_references_page and self._vision_references_column_mode_enabled():
            try:
                md_ref = self._convert_references_page_with_column_vl(
                    page=page,
                    page_index=page_index,
                    total_pages=total_pages,
                    page_hint=page_hint,
                    speed_mode=speed_mode,
                )
                if md_ref:
                    return md_ref
            except Exception as e:
                print(
                    f"[VISION_DIRECT] references column OCR failed on page {page_index+1}: {e}",
                    flush=True,
                )

        if not is_references_page:
            try:
                md_layout = self._convert_page_with_layout_crops(
                    page=page,
                    page_index=page_index,
                    total_pages=total_pages,
                    page_hint=page_hint,
                    speed_mode=speed_mode,
                    pdf_path=pdf_path,
                    assets_dir=assets_dir,
                    image_names=image_names or [],
                )
                if md_layout:
                    return md_layout
            except Exception as e:
                print(
                    f"[VISION_DIRECT] structured layout OCR failed on page {page_index+1}: {e}",
                    flush=True,
                )

        md = self.llm_worker.call_llm_page_to_markdown(
            png_bytes,
            page_number=page_index,
            total_pages=total_pages,
            hint=page_hint,
            speed_mode=speed_mode,
            is_references_page=is_references_page,
        )
        if md and formula_placeholders:
            md = self._restore_formula_placeholders(md, formula_placeholders)
        if not md:
            last_vl_err = ""
            try:
                last_vl_err = str(self.llm_worker.get_last_vl_error_code() or "").strip().lower()
            except Exception:
                last_vl_err = ""
            if last_vl_err == "timeout":
                print(
                    f"[VISION_DIRECT] VL hard-timeout on page {page_index+1}, skip empty retries and fallback",
                    flush=True,
                )
                retry_n = 0
            elif last_vl_err == "unsupported_vision":
                print(
                    f"[VISION_DIRECT] provider/model does not support image payloads on page {page_index+1}, skip empty retries and fallback",
                    flush=True,
                )
                retry_n = 0
            else:
                retry_n = self._vision_empty_retry_attempts()
            retry_sleep = self._vision_empty_retry_backoff_s()
            for k in range(1, retry_n + 1):
                try:
                    if retry_sleep > 0:
                        time.sleep(retry_sleep)
                except Exception:
                    pass
                print(
                    f"[VISION_DIRECT] VL empty on page {page_index+1}, retry {k}/{retry_n}",
                    flush=True,
                )
                retry_hint = (
                    (page_hint + " " if page_hint else "")
                    + "Previous attempt returned empty. OCR the full page and return complete Markdown only."
                )
                md = self.llm_worker.call_llm_page_to_markdown(
                    png_bytes,
                    page_number=page_index,
                    total_pages=total_pages,
                    hint=retry_hint,
                    speed_mode=speed_mode,
                    is_references_page=is_references_page,
                )
                if md:
                    if formula_placeholders:
                        md = self._restore_formula_placeholders(md, formula_placeholders)
                    break

        if not md:
            print(
                f"[VISION_DIRECT] VL returned empty for page {page_index+1}, falling back to extraction pipeline",
                flush=True,
            )
            try:
                return self._process_page(page, page_index=page_index, pdf_path=pdf_path, assets_dir=assets_dir)
            except Exception as e:
                print(
                    f"[VISION_DIRECT] fallback extraction failed on page {page_index+1}: {e}",
                    flush=True,
                )
                return None

        # References pages should not contain math; skip fragmented-math checks there.
        if is_references_page or not self._vision_math_quality_gate_enabled():
            return md

        if not self._looks_fragmented_math_output(md):
            return md

        print(
            f"[VISION_DIRECT] fragmented math detected on page {page_index+1}, retrying with strict formula hint",
            flush=True,
        )
        md_retry = self.llm_worker.call_llm_page_to_markdown(
            png_bytes,
            page_number=page_index,
            total_pages=total_pages,
            hint=self._vision_math_retry_hint(page_hint),
            speed_mode=speed_mode,
            is_references_page=is_references_page,
        )
        if md_retry and formula_placeholders:
            md_retry = self._restore_formula_placeholders(md_retry, formula_placeholders)
        if md_retry and (not self._looks_fragmented_math_output(md_retry)):
            print(
                f"[VISION_DIRECT] page {page_index+1} math recovered after retry",
                flush=True,
            )
            return md_retry

        if not self._vision_fragment_fallback_enabled():
            print(
                f"[VISION_DIRECT] fragmented math persists on page {page_index+1}, keep VL output (fallback disabled)",
                flush=True,
            )
            # Keep VL output by default; block-level fallback can alter heading/layout style.
            return md_retry or md
        print(
            f"[VISION_DIRECT] fragmented math persists on page {page_index+1}, using extraction fallback",
            flush=True,
        )
        try:
            fallback_md = self._process_page(page, page_index=page_index, pdf_path=pdf_path, assets_dir=assets_dir)
            if fallback_md:
                return fallback_md
        except Exception as e:
            print(
                f"[VISION_DIRECT] fallback extraction failed on page {page_index+1}: {e}",
                flush=True,
            )

        # Last resort: keep the best VL output we have rather than dropping the page.
        return md_retry or md

    def _repair_broken_image_links(self, md: str, *, save_dir: Path, assets_dir: Path) -> str:
        """
        Repair broken Markdown image links by remapping them to existing files in ./assets.
        This is a best-effort pass for vision-direct outputs where models may emit
        synthetic names like `figure_5.png` while extracted files are `page_*_fig_*.png`.
        """
        if not md:
            return md

        img_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
        matches = list(img_re.finditer(md))
        if not matches:
            return md

        # Candidate image files under assets.
        cand_files = []
        try:
            for p in sorted(assets_dir.glob("*")):
                if (not p.is_file()):
                    continue
                if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                    continue
                cand_files.append(p)
        except Exception:
            cand_files = []
        if not cand_files:
            return md

        page_fig_re = re.compile(r"page_(\d+)_fig_(\d+)", flags=re.IGNORECASE)

        def _asset_key(p: Path) -> tuple[int, int, str]:
            m = page_fig_re.search(p.stem)
            if not m:
                return (10**9, 10**9, p.name.lower())
            try:
                return (int(m.group(1)), int(m.group(2)), p.name.lower())
            except Exception:
                return (10**9, 10**9, p.name.lower())

        cand_files = sorted(cand_files, key=_asset_key)
        cand_index = {p.name: i for i, p in enumerate(cand_files)}

        # Track refs (in document order) and pre-resolve ones that already exist.
        refs: list[dict] = []
        used_names: set[str] = set()
        for m in matches:
            alt = m.group(1) or ""
            raw_path = (m.group(2) or "").strip().strip('"').strip("'")
            # Normalize markdown that accidentally had a space after `!`.
            # The regex only captures valid `![`, so we only fix path here.
            resolved_name = None
            if not (raw_path.startswith("http://") or raw_path.startswith("https://")):
                p_local = (save_dir / raw_path).resolve()
                if p_local.exists():
                    resolved_name = p_local.name
                else:
                    # Try direct by basename in assets.
                    b = Path(raw_path).name
                    if (assets_dir / b).exists():
                        resolved_name = b
            if resolved_name:
                used_names.add(resolved_name)
            refs.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "alt": alt,
                    "raw_path": raw_path,
                    "resolved_name": resolved_name,
                }
            )

        # First pass: figure-number-aware matching for unresolved refs.
        for r in refs:
            if r["resolved_name"]:
                continue
            b = Path(str(r["raw_path"])).name
            fig_no = None
            m_fig = re.search(r"figure[_\-\s]*(\d+)", b, flags=re.IGNORECASE)
            if m_fig:
                try:
                    fig_no = int(m_fig.group(1))
                except Exception:
                    fig_no = None
            chosen = None
            if fig_no is not None:
                # Prefer assets whose filename contains `fig_<n>`.
                exact = [
                    p
                    for p in cand_files
                    if re.search(rf"(?:^|_)fig_{fig_no}(?:_|$|\.)", p.name, flags=re.IGNORECASE)
                ]
                if exact:
                    for p in exact:
                        if p.name not in used_names:
                            chosen = p
                            break
                    if chosen is None:
                        chosen = exact[0]
            if chosen is None:
                # Basename similarity fallback.
                low = b.lower()
                sim = [p for p in cand_files if (low and low in p.name.lower())]
                if sim:
                    for p in sim:
                        if p.name not in used_names:
                            chosen = p
                            break
                    if chosen is None:
                        chosen = sim[0]
            if chosen is not None:
                r["resolved_name"] = chosen.name
                used_names.add(chosen.name)

        # Second pass: positional interpolation for still-unresolved refs.
        unresolved_idx = [i for i, r in enumerate(refs) if not r["resolved_name"]]
        if unresolved_idx:
            assigned_page_by_pos: dict[int, int] = {}
            for i, r in enumerate(refs):
                nm = r.get("resolved_name")
                if not nm:
                    continue
                p = assets_dir / str(nm)
                m_pg = page_fig_re.search(p.stem)
                if m_pg:
                    try:
                        assigned_page_by_pos[i] = int(m_pg.group(1))
                    except Exception:
                        pass

            avail = [p for p in cand_files if p.name not in used_names]
            for i in unresolved_idx:
                if not avail:
                    break
                prev_pages = [assigned_page_by_pos[j] for j in assigned_page_by_pos if j < i]
                next_pages = [assigned_page_by_pos[j] for j in assigned_page_by_pos if j > i]
                prev_p = max(prev_pages) if prev_pages else None
                next_p = min(next_pages) if next_pages else None

                def _page_of(p: Path) -> int:
                    m = page_fig_re.search(p.stem)
                    if not m:
                        return 10**9
                    try:
                        return int(m.group(1))
                    except Exception:
                        return 10**9

                choice = None
                if prev_p is not None and next_p is not None:
                    in_range = [p for p in avail if prev_p <= _page_of(p) <= next_p]
                    if in_range:
                        mid = (prev_p + next_p) / 2.0
                        choice = min(in_range, key=lambda p: abs(_page_of(p) - mid))
                if choice is None and prev_p is not None:
                    after = [p for p in avail if _page_of(p) >= prev_p]
                    if after:
                        choice = min(after, key=_page_of)
                if choice is None and next_p is not None:
                    before = [p for p in avail if _page_of(p) <= next_p]
                    if before:
                        choice = max(before, key=_page_of)
                if choice is None:
                    choice = avail[0]

                refs[i]["resolved_name"] = choice.name
                used_names.add(choice.name)
                try:
                    avail.remove(choice)
                except Exception:
                    pass

        # Rebuild markdown with remapped links.
        out = md
        shift = 0
        repaired = 0
        for r in refs:
            nm = r.get("resolved_name")
            if not nm:
                continue
            new_ref = f"![{r['alt']}](./assets/{nm})"
            s0 = int(r["start"]) + shift
            e0 = int(r["end"]) + shift
            old_ref = out[s0:e0]
            if old_ref != new_ref:
                out = out[:s0] + new_ref + out[e0:]
                shift += len(new_ref) - (e0 - s0)
                repaired += 1

        if repaired > 0:
            print(f"[IMAGE_FIX] repaired {repaired} image link(s)", flush=True)
        return out

    @staticmethod
    def _inject_missing_page_image_links(
        md: str,
        *,
        page_index: int,
        image_names: list[str],
        is_references_page: bool = False,
    ) -> str:
        """
        In vision-direct mode, VL output can omit image markdown even when page images
        were extracted to ./assets. Inject deterministic links as a fallback.
        """
        if not md:
            return md
        if is_references_page:
            return md
        if not image_names:
            return md

        # Keep stable order by fig index.
        def _img_key(nm: str) -> tuple[int, str]:
            m = re.search(r"_fig_(\d+)", nm, flags=re.IGNORECASE)
            if m:
                try:
                    return (int(m.group(1)), nm.lower())
                except Exception:
                    pass
            return (10**9, nm.lower())

        ordered = sorted({str(n).strip() for n in image_names if str(n).strip()}, key=_img_key)
        if not ordered:
            return md

        link_re = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
        existing_basenames: set[str] = set()
        for m in link_re.finditer(md):
            raw = (m.group(1) or "").strip().strip('"').strip("'")
            if not raw:
                continue
            existing_basenames.add(Path(raw).name)

        missing = [nm for nm in ordered if nm not in existing_basenames]
        if not missing:
            return md

        lines = md.splitlines()
        cap_idx: list[int] = []
        cap_fig_no: str | None = None
        for i, ln in enumerate(lines):
            st = _normalize_text(ln or "").strip()
            m = re.match(r"^(?:Figure|Fig\.?)\s*(\d+)\b", st, flags=re.IGNORECASE)
            if m:
                cap_idx.append(i)
                if cap_fig_no is None:
                    cap_fig_no = m.group(1)

        # Conservative: only inject when page text looks figure-related.
        looks_figure_page = bool(cap_idx) or bool(re.search(r"\bFigure\s+\d+\b", md, flags=re.IGNORECASE))
        if not looks_figure_page:
            return md

        if cap_idx:
            insert_at = cap_idx[0]
        else:
            insert_at = 0

        alt = f"Figure {cap_fig_no}" if cap_fig_no else "Figure"
        inject_block: list[str] = [f"![{alt}](./assets/{nm})" for nm in missing]
        if inject_block:
            inject_block.append("")
            lines = lines[:insert_at] + inject_block + lines[insert_at:]
            try:
                print(
                    f"[IMAGE_FIX] page {page_index+1}: injected {len(missing)} missing image link(s)",
                    flush=True,
                )
            except Exception:
                pass
        return "\n".join(lines)

    @staticmethod
    def _normalize_page_local_image_link_order(
        md: str,
        *,
        page_index: int,
        image_names: list[str],
    ) -> str:
        """
        Normalize current-page figure link ordering to match extracted asset order.

        Vision models may reference valid assets but swap `fig_1` / `fig_2`.
        When the page contains the exact same set of current-page assets, remap
        links in document order to the sorted extracted order.
        """
        if not md:
            return md
        if len(image_names or []) < 2:
            return md

        page_no = int(page_index) + 1

        def _img_key(nm: str) -> tuple[int, str]:
            m = re.search(r"_fig_(\d+)", str(nm), flags=re.IGNORECASE)
            if m:
                try:
                    return (int(m.group(1)), str(nm).lower())
                except Exception:
                    pass
            return (10**9, str(nm).lower())

        ordered_assets = sorted({str(n).strip() for n in image_names if str(n).strip()}, key=_img_key)
        if len(ordered_assets) < 2:
            return md

        img_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
        page_fig_re = re.compile(rf"^page_{page_no}_fig_(\d+)\.[A-Za-z0-9]+$", flags=re.IGNORECASE)
        refs: list[dict] = []
        for m in img_re.finditer(md):
            raw_path = (m.group(2) or "").strip().strip('"').strip("'")
            base = Path(raw_path).name
            if not page_fig_re.match(base):
                continue
            refs.append({"start": m.start(), "end": m.end(), "alt": m.group(1) or "", "path": raw_path, "base": base})

        if len(refs) < 2:
            return md
        if len(refs) != len(ordered_assets):
            return md

        current_order = [str(r["base"]) for r in refs]
        if set(current_order) != set(ordered_assets):
            return md
        if current_order == ordered_assets:
            return md

        out = md
        shift = 0
        changed = 0
        for r, target_name in zip(refs, ordered_assets):
            new_ref = f"![{r['alt']}](./assets/{target_name})"
            s0 = int(r["start"]) + shift
            e0 = int(r["end"]) + shift
            old_ref = out[s0:e0]
            if old_ref == new_ref:
                continue
            out = out[:s0] + new_ref + out[e0:]
            shift += len(new_ref) - (e0 - s0)
            changed += 1
        if changed:
            try:
                print(f"[IMAGE_ORDER] page {page_no}: normalized {changed} figure link(s)", flush=True)
            except Exception:
                pass
        return out

    @staticmethod
    def _extract_figure_number_from_text(text: str) -> Optional[int]:
        t = _normalize_text(text or "").strip()
        if not t:
            return None
        m = re.search(
            r"\b(?:fig(?:ure)?\.?)\s*(\d{1,4})(?:[A-Za-z])?\b",
            t,
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    @staticmethod
    def _figure_remap_debug_enabled() -> bool:
        try:
            return bool(int(os.environ.get("KB_PDF_DEBUG_FIG_REMAP", "0") or "0"))
        except Exception:
            return False

    @staticmethod
    def _reorder_page_figure_pairs_by_number(
        md: str,
        *,
        page_index: int,
    ) -> str:
        """
        Reorder adjacent figure image+caption pairs for the same page by figure number.
        This keeps visual order stable when VL outputs Fig.3 before Fig.2 on one page.
        """
        if not md:
            return md

        page_no = int(page_index) + 1
        img_re = re.compile(
            rf"^\s*!\[[^\]]*\]\(\./assets/(page_{page_no}_fig_\d+\.[^)]+)\)\s*$",
            flags=re.IGNORECASE,
        )
        cap_re = re.compile(
            r"^\s*(?:[*_`>#\[\(]\s*)*(?:fig(?:ure)?\.?)\s*(\d{1,4})(?:[A-Za-z])?\b",
            flags=re.IGNORECASE,
        )
        heading_re = re.compile(r"^\s*#{1,6}\s+")

        lines = md.splitlines()
        n = len(lines)
        blocks: list[dict] = []
        i = 0
        while i < n:
            m_img = img_re.match(lines[i] or "")
            if not m_img:
                i += 1
                continue

            # Find nearby caption line after the image.
            caption_idx = None
            fig_no = None
            j = i + 1
            scanned = 0
            while j < n and scanned < 8:
                s = (lines[j] or "").strip()
                if not s:
                    j += 1
                    scanned += 1
                    continue
                if img_re.match(lines[j] or "") or heading_re.match(lines[j] or ""):
                    break
                m_cap = cap_re.match(_normalize_text(s))
                if m_cap:
                    caption_idx = j
                    try:
                        fig_no = int(m_cap.group(1))
                    except Exception:
                        fig_no = None
                    break
                # Non-caption content appears before caption -> skip this image block.
                break

            if caption_idx is None or fig_no is None:
                i += 1
                continue

            # Include caption continuation lines conservatively.
            end = caption_idx + 1
            cont = 0
            while end < n:
                s = (lines[end] or "").strip()
                if not s:
                    end += 1
                    break
                if img_re.match(lines[end] or "") or heading_re.match(lines[end] or ""):
                    break
                if cont >= 2:
                    break
                end += 1
                cont += 1

            blocks.append(
                {
                    "start": int(i),
                    "end": int(end),
                    "fig_no": int(fig_no),
                    "img_name": str(m_img.group(1)),
                }
            )
            i = end

        if len(blocks) < 2:
            return md

        # Group adjacent blocks where separators are only blank lines.
        runs: list[list[dict]] = []
        cur: list[dict] = [blocks[0]]
        for b in blocks[1:]:
            prev = cur[-1]
            gap_lines = lines[int(prev["end"]):int(b["start"])]
            only_blank_gap = all((not (gl or "").strip()) for gl in gap_lines)
            if only_blank_gap:
                cur.append(b)
            else:
                if len(cur) >= 2:
                    runs.append(cur)
                cur = [b]
        if len(cur) >= 2:
            runs.append(cur)
        if not runs:
            return md

        changed = 0
        out_lines: list[str] = []
        cursor = 0
        for run in runs:
            seg_start = int(run[0]["start"])
            seg_end = int(run[-1]["end"])
            out_lines.extend(lines[cursor:seg_start])
            nums = [int(x["fig_no"]) for x in run]
            if nums == sorted(nums):
                out_lines.extend(lines[seg_start:seg_end])
                cursor = seg_end
                continue

            run_sorted = sorted(run, key=lambda x: (int(x["fig_no"]), int(x["start"])))
            chunk: list[str] = []
            for idx, rb in enumerate(run_sorted):
                chunk.extend(lines[int(rb["start"]):int(rb["end"])])
                if idx != len(run_sorted) - 1 and (not chunk or chunk[-1].strip()):
                    chunk.append("")
            out_lines.extend(chunk)
            cursor = seg_end
            changed += 1

        out_lines.extend(lines[cursor:])
        if changed:
            try:
                print(f"[FIG_ORDER] page {page_no}: reordered {changed} figure block run(s) by figure number", flush=True)
            except Exception:
                pass
        return "\n".join(out_lines)

    @staticmethod
    def _remap_page_image_links_by_caption(
        md: str,
        *,
        page_index: int,
        figure_meta_by_asset: Optional[dict[str, dict]] = None,
    ) -> str:
        """
        Use per-page figure metadata (asset -> figure number) to remap image links
        according to nearby caption numbers in markdown.
        """
        if not md:
            return md
        if not figure_meta_by_asset:
            return md

        fig_to_asset: dict[int, str] = {}
        ambiguous: set[int] = set()
        for asset_name, meta in (figure_meta_by_asset or {}).items():
            if not isinstance(meta, dict):
                continue
            try:
                fig_no = int(meta.get("fig_no"))
            except Exception:
                continue
            if fig_no in fig_to_asset and fig_to_asset.get(fig_no) != asset_name:
                ambiguous.add(fig_no)
                continue
            fig_to_asset[fig_no] = str(asset_name)
        for n in ambiguous:
            fig_to_asset.pop(n, None)
        if not fig_to_asset:
            return md

        page_no = int(page_index) + 1
        page_fig_re = re.compile(rf"^page_{page_no}_fig_(\d+)\.[A-Za-z0-9]+$", flags=re.IGNORECASE)
        img_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
        heading_re = re.compile(r"^\s*#{1,6}\s+")
        cap_re = re.compile(
            r"^\s*(?:[*_`>#\[\(]\s*)*(?:fig(?:ure)?\.?)\s*(\d{1,4})(?:[A-Za-z])?\b",
            flags=re.IGNORECASE,
        )

        lines = md.splitlines()
        changed = 0
        debug = PDFConverter._figure_remap_debug_enabled()
        trace: list[str] = []
        if debug:
            try:
                pairs = ", ".join(f"{k}->{v}" for k, v in sorted(fig_to_asset.items(), key=lambda x: x[0]))
                trace.append(f"[FIG_DEBUG] page {page_no} mapping: {pairs or '<empty>'}")
            except Exception:
                pass

        def _find_nearby_caption_fig(line_idx: int) -> Optional[int]:
            def _scan(*, delta: int, max_steps: int, max_non_empty: int) -> Optional[int]:
                non_empty = 0
                for step in range(1, max_steps + 1):
                    j = line_idx + delta * step
                    if not (0 <= j < len(lines)):
                        break
                    raw = lines[j] or ""
                    s = raw.strip()
                    if not s:
                        continue
                    if heading_re.match(raw) or img_re.search(raw):
                        break
                    non_empty += 1
                    m = cap_re.match(_normalize_text(s))
                    if m:
                        try:
                            return int(m.group(1))
                        except Exception:
                            return None
                    if non_empty >= max_non_empty:
                        break
                return None

            # Prefer caption below image, then above.
            down = _scan(delta=1, max_steps=8, max_non_empty=3)
            if down is not None:
                return down
            up = _scan(delta=-1, max_steps=6, max_non_empty=2)
            if up is not None:
                return up
            return None

        out_lines: list[str] = []
        for idx, line in enumerate(lines):
            matches = list(img_re.finditer(line))
            if not matches:
                out_lines.append(line)
                continue
            out = line
            shift = 0
            for m in matches:
                alt = m.group(1) or ""
                raw_path = (m.group(2) or "").strip().strip('"').strip("'")
                base = Path(raw_path).name
                if not page_fig_re.match(base):
                    continue

                fig_no = PDFConverter._extract_figure_number_from_text(alt)
                if fig_no is None:
                    fig_no = _find_nearby_caption_fig(idx)
                if fig_no is None:
                    if debug:
                        trace.append(
                            f"[FIG_DEBUG] page {page_no} line {idx+1}: skip {base} (no fig number from alt/caption)"
                        )
                    continue

                target = fig_to_asset.get(int(fig_no))
                if not target or target == base:
                    if debug:
                        target_show = target if target else "<none>"
                        trace.append(
                            f"[FIG_DEBUG] page {page_no} line {idx+1}: keep {base} (fig_no={fig_no}, target={target_show})"
                        )
                    continue

                new_ref = f"![{alt}](./assets/{target})"
                s0 = int(m.start()) + shift
                e0 = int(m.end()) + shift
                old_ref = out[s0:e0]
                if old_ref == new_ref:
                    continue
                out = out[:s0] + new_ref + out[e0:]
                shift += len(new_ref) - (e0 - s0)
                changed += 1
                if debug:
                    trace.append(
                        f"[FIG_DEBUG] page {page_no} line {idx+1}: remap {base} -> {target} (fig_no={fig_no})"
                    )
            out_lines.append(out)

        if debug and trace:
            for row in trace:
                try:
                    print(row, flush=True)
                except Exception:
                    pass
        if changed:
            try:
                print(f"[IMAGE_REMAP] page {page_no}: remapped {changed} figure link(s) by caption metadata", flush=True)
            except Exception:
                pass
        return "\n".join(out_lines)

    def _extract_page_figure_caption_candidates(self, page) -> list[dict]:
        if fitz is None or page is None:
            return []
        out: list[dict] = []
        try:
            d = page.get_text("dict") or {}
        except Exception:
            return out

        cap_start_re = re.compile(
            r"^\s*(?:fig(?:ure)?\.?)\s*(\d{1,4}[A-Za-z]?)\b",
            flags=re.IGNORECASE,
        )
        for b in d.get("blocks", []) or []:
            if "lines" not in b:
                continue
            bbox = b.get("bbox")
            if not bbox:
                continue
            txt_parts: list[str] = []
            for l in b.get("lines", []) or []:
                spans = l.get("spans", []) or []
                if not spans:
                    continue
                line = "".join(str(s.get("text", "")) for s in spans)
                line = _normalize_text(line).strip()
                if line:
                    txt_parts.append(line)
            if not txt_parts:
                continue
            txt = _normalize_text(" ".join(txt_parts)).strip()
            if not txt:
                continue
            m = cap_start_re.match(txt)
            if not m:
                continue
            fig_ident = (m.group(1) or "").strip()
            fig_no = None
            try:
                fig_no = int(re.match(r"^(\d+)", fig_ident).group(1))  # type: ignore[union-attr]
            except Exception:
                fig_no = None
            out.append(
                {
                    "fig_no": fig_no,
                    "fig_ident": fig_ident,
                    "caption": txt,
                    "bbox": [float(x) for x in bbox],
                }
            )
        out.sort(key=lambda x: (float(x["bbox"][1]), float(x["bbox"][0])))
        return out

    def _match_figure_entries_with_captions(
        self,
        *,
        page,
        figure_entries: list[dict],
        caption_candidates: list[dict],
    ) -> list[dict]:
        if fitz is None or page is None:
            return figure_entries
        if not figure_entries or not caption_candidates:
            return figure_entries

        page_h = float(page.rect.height or 0.0)
        max_below_gap = max(120.0, page_h * 0.34)
        max_above_gap = max(96.0, page_h * 0.22)

        def _score(entry: dict, cap: dict) -> float:
            rb = entry.get("crop_bbox") or entry.get("bbox")
            cb = cap.get("bbox")
            if not rb or not cb:
                return -10_000.0
            try:
                rx0, ry0, rx1, ry1 = [float(v) for v in rb]
                cx0, cy0, cx1, cy1 = [float(v) for v in cb]
            except Exception:
                return -10_000.0

            hov = _overlap_1d(rx0, rx1, cx0 - 18.0, cx1 + 18.0)
            min_w = max(1.0, min(rx1 - rx0, (cx1 - cx0) + 36.0))
            hov_ratio = hov / min_w

            below_gap = cy0 - ry1
            above_gap = ry0 - cy1
            gap_penalty = 500.0
            if below_gap >= -20.0 and below_gap <= max_below_gap:
                gap_penalty = max(0.0, below_gap) * 0.9
            elif above_gap >= -20.0 and above_gap <= max_above_gap:
                # Caption above figure is less common; allow but penalize more.
                gap_penalty = 60.0 + max(0.0, above_gap) * 1.2

            rcx = (rx0 + rx1) / 2.0
            ccx = (cx0 + cx1) / 2.0
            x_penalty = abs(rcx - ccx) * 0.06

            score = (hov_ratio * 140.0) - gap_penalty - x_penalty
            if hov_ratio < 0.08:
                score -= 140.0
            return score

        used_entries: set[int] = set()
        assigned: dict[int, dict] = {}
        for cap in caption_candidates:
            best_idx = None
            best_score = -10_000.0
            for i, ent in enumerate(figure_entries):
                if i in used_entries:
                    continue
                s = _score(ent, cap)
                if s > best_score:
                    best_score = s
                    best_idx = i
            if best_idx is None:
                continue
            # Conservative threshold to avoid spurious matches.
            if best_score < -18.0:
                continue
            used_entries.add(best_idx)
            assigned[best_idx] = cap

        out: list[dict] = []
        for i, ent in enumerate(figure_entries):
            row = dict(ent)
            cap = assigned.get(i)
            if cap:
                row["fig_no"] = cap.get("fig_no")
                row["fig_ident"] = cap.get("fig_ident")
                row["caption"] = cap.get("caption")
                row["caption_bbox"] = cap.get("bbox")
            out.append(row)
        return out

    def _persist_page_figure_metadata(
        self,
        *,
        assets_dir: Path,
        page_index: int,
        figure_entries: list[dict],
    ) -> dict[str, dict]:
        """
        Persist per-page figure metadata (asset/crop/caption binding) and return
        a fast lookup map by asset filename.
        """
        page_no = int(page_index) + 1
        if not figure_entries:
            # Keep output deterministic: remove stale page metadata if present.
            meta_path = assets_dir / f"page_{page_no}_fig_index.json"
            try:
                meta_path.unlink(missing_ok=True)
            except Exception:
                pass
            return {}

        rows: list[dict] = []
        by_asset: dict[str, dict] = {}
        for idx, item in enumerate(figure_entries, start=1):
            asset_name = str(item.get("asset_name") or "").strip()
            if not asset_name:
                continue
            row = {
                "page": page_no,
                "index": int(idx),
                "asset_name": asset_name,
                "fig_no": item.get("fig_no"),
                "fig_ident": item.get("fig_ident"),
                "caption": item.get("caption"),
                "bbox": item.get("bbox"),
                "crop_bbox": item.get("crop_bbox"),
                "caption_bbox": item.get("caption_bbox"),
            }
            rows.append(row)
            by_asset[asset_name] = row

            sidecar = assets_dir / f"{Path(asset_name).stem}.meta.json"
            try:
                sidecar.write_text(
                    json.dumps(row, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

        meta_path = assets_dir / f"page_{page_no}_fig_index.json"
        try:
            payload = {
                "page": page_no,
                "figures": rows,
            }
            meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return by_asset

    @staticmethod
    def _build_page_figure_mapping_hint(figure_meta_by_asset: Optional[dict[str, dict]]) -> str:
        if not figure_meta_by_asset:
            return ""
        rows: list[tuple[int, str]] = []
        for asset, meta in figure_meta_by_asset.items():
            if not isinstance(meta, dict):
                continue
            try:
                n = int(meta.get("fig_no"))
            except Exception:
                continue
            rows.append((n, str(asset)))
        if not rows:
            return ""
        rows.sort(key=lambda x: (x[0], x[1].lower()))
        seen: set[int] = set()
        pairs: list[str] = []
        for n, asset in rows:
            if n in seen:
                continue
            seen.add(n)
            pairs.append(f"Figure {n} -> ./assets/{asset}")
        if not pairs:
            return ""
        return "Figure-to-asset mapping for this page: " + "; ".join(pairs) + "."

    def _detect_table_rects_for_fallback(
        self,
        page,
        *,
        page_index: int,
        pdf_path: Path,
        visual_rects: Optional[list["fitz.Rect"]] = None,
    ) -> list["fitz.Rect"]:
        """
        Find table rectangles for screenshot fallback, without requiring markdown
        table extraction success.
        """
        if fitz is None or page is None or (not hasattr(page, "find_tables")):
            return []

        W = float(page.rect.width or 0.0)
        H = float(page.rect.height or 0.0)
        page_area = max(1.0, W * H)
        vis_rects = list(visual_rects or [])

        d0 = None
        has_table_hint = False
        try:
            d0 = page.get_text("dict")
            if isinstance(d0, dict):
                has_table_hint = bool(_page_maybe_has_table_from_dict(d0))
        except Exception:
            d0 = None
            has_table_hint = False

        caption_rects: list["fitz.Rect"] = []
        if isinstance(d0, dict):
            try:
                for b in d0.get("blocks", []):
                    if "lines" not in b:
                        continue
                    bbox = b.get("bbox")
                    if not bbox:
                        continue
                    txt_parts: list[str] = []
                    for l in b.get("lines", []):
                        spans = l.get("spans", [])
                        if not spans:
                            continue
                        line = "".join(str(s.get("text", "")) for s in spans)
                        line = _normalize_text(line)
                        if line:
                            txt_parts.append(line)
                    txt = _normalize_text(" ".join(txt_parts)).strip()
                    if re.match(r"^Table\s+(?:\d+|[A-Za-z]|[IVXLC]+)\b", txt, flags=re.IGNORECASE):
                        caption_rects.append(fitz.Rect(tuple(float(x) for x in bbox)))
            except Exception:
                caption_rects = []

        def _near_caption(rect: "fitz.Rect") -> bool:
            for cr in caption_rects:
                hov = _overlap_1d(float(rect.x0), float(rect.x1), float(cr.x0) - 18.0, float(cr.x1) + 18.0)
                if hov < max(10.0, min(float(rect.width), float(cr.width) + 36.0) * 0.15):
                    continue
                # Usually table is right below caption, but allow moderate gap.
                if float(rect.y0) >= float(cr.y0) - 24.0 and (float(rect.y0) - float(cr.y1)) <= max(140.0, H * 0.25):
                    return True
            return False

        kwargs_seq = [{"vertical_strategy": "lines", "horizontal_strategy": "lines"}]
        if has_table_hint:
            kwargs_seq.extend(
                [
                    {"vertical_strategy": "lines", "horizontal_strategy": "text", "min_words_horizontal": 1, "text_tolerance": 2.0},
                    {"vertical_strategy": "text", "horizontal_strategy": "lines", "min_words_vertical": 2, "text_tolerance": 2.0},
                ]
            )

        candidates: list[tuple["fitz.Rect", float]] = []
        for kwargs in kwargs_seq:
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
                if rect.width <= 0 or rect.height <= 0:
                    continue
                area = _rect_area(rect)
                if area < page_area * 0.0035:
                    continue
                if area > page_area * 0.70:
                    continue
                if float(rect.width) < W * 0.15 or float(rect.height) < H * 0.04:
                    continue
                near_cap = _near_caption(rect)
                if vis_rects and (not near_cap):
                    vis_overlap = max(
                        (
                            _rect_intersection_area(rect, vr) / max(1.0, min(_rect_area(rect), _rect_area(vr)))
                            for vr in vis_rects
                        ),
                        default=0.0,
                    )
                    if vis_overlap >= 0.55:
                        continue
                score = float(area / page_area)
                if near_cap:
                    score += 10.0
                candidates.append((rect, score))

        if not candidates:
            return []

        uniq: list[tuple["fitz.Rect", float]] = []
        for rect, score in sorted(candidates, key=lambda x: (x[0].y0, x[0].x0, -x[1])):
            replaced = False
            for j, (r0, s0) in enumerate(uniq):
                inter = _rect_intersection_area(rect, r0)
                denom = max(1.0, min(_rect_area(rect), _rect_area(r0)))
                if (inter / denom) >= 0.72:
                    if score > s0:
                        uniq[j] = (rect, score)
                    replaced = True
                    break
            if not replaced:
                uniq.append((rect, score))
        uniq.sort(key=lambda x: (x[0].y0, x[0].x0))
        return [r for r, _ in uniq]

    def _inject_table_image_fallbacks(
        self,
        md: str,
        *,
        page,
        page_index: int,
        pdf_path: Path,
        assets_dir: Path,
        visual_rects: Optional[list["fitz.Rect"]] = None,
        is_references_page: bool = False,
    ) -> str:
        """
        For difficult tables that VL fails to render into markdown, insert a table
        screenshot fallback (saved as `page_X_table_Y.png`) before the table caption.
        """
        if not md or is_references_page:
            return md
        if not re.search(r"(?mi)^\s*(?:\*{1,2}\s*)?Table\s+(?:\d+|[A-Za-z]|[IVXLC]+)\b", md):
            return md

        lines = md.splitlines()
        if not lines:
            return md

        cap_re = re.compile(
            r"^\s*(?:\*{1,2}\s*)?Table\s+([A-Za-z0-9]+)\s*(?:[.:]\s*|\-\s*|\s+)?(.*)$",
            flags=re.IGNORECASE,
        )
        md_table_re = re.compile(r"^\s*\|.*\|\s*$")
        table_img_re = re.compile(rf"!\[[^\]]*\]\(\./assets/page_{page_index+1}_table_\d+\.[^)]+\)", flags=re.IGNORECASE)

        caption_rows: list[dict] = []
        for idx, ln in enumerate(lines):
            m = cap_re.match(ln.strip())
            if not m:
                continue
            ident = (m.group(1) or "").strip()
            label = f"Table {ident}" if ident else "Table"
            caption_rows.append({"idx": idx, "label": label})
        if not caption_rows:
            return md

        def _needs_table_image_at(idx: int) -> bool:
            lo = max(0, idx - 4)
            hi = min(len(lines), idx + 9)
            for j in range(lo, hi):
                s = lines[j].strip()
                if not s:
                    continue
                if table_img_re.search(s):
                    return False
                if md_table_re.match(s):
                    return False
            return True

        missing_caps = [c for c in caption_rows if _needs_table_image_at(int(c["idx"]))]
        if not missing_caps:
            return md

        table_rects = self._detect_table_rects_for_fallback(
            page,
            page_index=page_index,
            pdf_path=pdf_path,
            visual_rects=visual_rects,
        )
        if not table_rects:
            try:
                print(f"[TABLE_FALLBACK] page {page_index+1}: no table rects detected for {len(missing_caps)} caption(s)", flush=True)
            except Exception:
                pass
            return md

        page_no = int(page_index) + 1
        out_lines = list(lines)
        offset = 0
        inserted = 0
        dpi = max(144, int(getattr(self, "dpi", 200) or 200))

        for table_idx, (cap, rect) in enumerate(zip(missing_caps, table_rects), start=1):
            pad_x = max(4.0, float(rect.width) * 0.015)
            pad_y = max(4.0, float(rect.height) * 0.015)
            clip = fitz.Rect(
                max(0.0, float(rect.x0) - pad_x),
                max(0.0, float(rect.y0) - pad_y),
                min(float(page.rect.width), float(rect.x1) + pad_x),
                min(float(page.rect.height), float(rect.y1) + pad_y),
            )
            img_name = f"page_{page_no}_table_{table_idx}.png"
            img_path = assets_dir / img_name
            try:
                pix = page.get_pixmap(clip=clip, dpi=dpi, alpha=False)
                pix.save(img_path)
                if (not img_path.exists()) or img_path.stat().st_size < 256:
                    continue
            except Exception as e:
                try:
                    print(f"[TABLE_FALLBACK] page {page_no}: failed to save {img_name}: {e}", flush=True)
                except Exception:
                    pass
                continue

            label = str(cap.get("label") or "Table").strip()
            ins_at = int(cap["idx"]) + offset
            block = [
                f"![{label}](./assets/{img_name})",
                f"<!-- kb:asset kind=table_image_fallback page={page_no} index={table_idx} label={label} -->",
                "",
            ]
            out_lines[ins_at:ins_at] = block
            offset += len(block)
            inserted += 1

        if inserted:
            try:
                print(f"[TABLE_FALLBACK] page {page_no}: inserted {inserted} table screenshot fallback(s)", flush=True)
            except Exception:
                pass
        return "\n".join(out_lines)

    def _postprocess_vision_page_markdown(
        self,
        md: str,
        *,
        page,
        page_index: int,
        pdf_path: Path,
        assets_dir: Path,
        image_names: list[str],
        figure_meta_by_asset: Optional[dict[str, dict]] = None,
        visual_rects: Optional[list["fitz.Rect"]] = None,
        is_references_page: bool = False,
    ) -> str:
        if not md:
            return md
        md = self._inject_missing_page_image_links(
            md,
            page_index=page_index,
            image_names=image_names,
            is_references_page=is_references_page,
        )
        md = self._remap_page_image_links_by_caption(
            md,
            page_index=page_index,
            figure_meta_by_asset=figure_meta_by_asset,
        )
        md = self._reorder_page_figure_pairs_by_number(
            md,
            page_index=page_index,
        )
        has_caption_meta = bool(
            figure_meta_by_asset
            and any((isinstance(v, dict) and v.get("fig_no") is not None) for v in figure_meta_by_asset.values())
        )
        if not has_caption_meta:
            md = self._normalize_page_local_image_link_order(
                md,
            page_index=page_index,
            image_names=image_names,
        )
        md = self._cleanup_page_local_image_markdown(
            md,
            page_index=page_index,
        )
        md = self._inject_table_image_fallbacks(
            md,
            page=page,
            page_index=page_index,
            pdf_path=pdf_path,
            assets_dir=assets_dir,
            visual_rects=visual_rects,
            is_references_page=is_references_page,
        )
        return md

    @staticmethod
    def _cleanup_page_local_image_markdown(md: str, *, page_index: int) -> str:
        if not md:
            return md

        lines = md.splitlines()

        def _line_norm(s: str) -> str:
            s = _normalize_text(s or "")
            s = re.sub(r"[*_`>#\[\]()]", "", s)
            s = re.sub(r"\s+", " ", s).strip().lower()
            return s

        img_re = re.compile(r"!\[([^\]]*)\]\((\.?/)?assets/[^)]+\)", flags=re.IGNORECASE)
        for idx, raw in enumerate(lines):
            m = img_re.search(raw or "")
            if not m:
                continue
            alt = (m.group(1) or "").strip()
            if not alt:
                continue
            fig_no = None
            for probe in range(idx, min(len(lines), idx + 4)):
                cap_line = _normalize_text(lines[probe] or "").strip()
                m_cap = re.match(r"^\*{0,2}(?:Figure|Fig\.?)\s*(\d+)\b", cap_line, flags=re.IGNORECASE)
                if m_cap:
                    fig_no = m_cap.group(1)
                    break
            if fig_no is None:
                m_alt = re.search(r"\b(?:Figure|Fig\.?)\s*(\d+)\b", alt, flags=re.IGNORECASE)
                if m_alt:
                    fig_no = m_alt.group(1)
            if fig_no and (len(alt) > 32 or re.search(r"[.:,;]", alt)):
                new_ref = re.sub(
                    r"!\[[^\]]*\]",
                    f"![Figure {fig_no}]",
                    raw,
                    count=1,
                )
                lines[idx] = new_ref

        cleaned: list[str] = []
        recent_caption_keys: list[str] = []
        for raw in lines:
            st = _normalize_text(raw or "").strip()
            if not st:
                cleaned.append(raw)
                continue

            if page_index == 0:
                wide_gap = bool(re.search(r"(?:\s|\u2000|\u2001|\u2002|\u2003|\u2004|\u2005|\u2006|\u2007|\u2008|\u2009|\u200A|\u3000){6,}", raw or ""))
                bold_chunks = len(re.findall(r"\*\*[^*]{2,80}\*\*", raw or ""))
                wordish = len(re.findall(r"[A-Za-z]{2,}", st))
                if wide_gap and bold_chunks >= 2 and wordish <= 8 and "." not in st:
                    continue

            if re.match(r"^\*{0,2}(?:Figure|Fig\.?)\s*\d+\b", st, flags=re.IGNORECASE):
                key = _line_norm(st)
                if key and key in recent_caption_keys[-3:]:
                    continue
                recent_caption_keys.append(key)
            cleaned.append(raw)

        out = "\n".join(cleaned)
        out = re.sub(r"(\*\*(?:Figure|Fig\.?)\s*\d+\.?\*\*)\s+\*\*\s*", r"\1 ", out, flags=re.IGNORECASE)
        out = re.sub(r"(\*\*Table\s*\d+\.?\*\*)\s+\*\*\s*", r"\1 ", out, flags=re.IGNORECASE)
        return out

    # Removed: _process_batch_fast and _process_batch_llm (old text extraction methods)
    # Now only using vision-direct mode (_process_batch_vision_direct) and no-LLM mode (_process_batch_no_llm)

    # ------------------------------------------------------------------
    # Vision-direct mode: screenshot each page 鈫?VL model 鈫?Markdown
    # ------------------------------------------------------------------
    def _process_batch_vision_direct(self, doc, pdf_path: Path, assets_dir: Path, speed_mode: str = 'normal') -> List[Optional[str]]:
        return process_batch_vision_direct(
            self,
            doc,
            pdf_path,
            assets_dir,
            speed_mode=speed_mode,
        )

    def _process_batch_no_llm(self, doc, pdf_path: Path, assets_dir: Path) -> List[Optional[str]]:
        """
        Deterministic fallback path when LLM is disabled/unavailable.
        Process pages one by one using local extraction logic.
        """
        total_pages = len(doc)
        results: List[Optional[str]] = [None] * total_pages

        start = max(0, int(getattr(self.cfg, "start_page", 0) or 0))
        end = int(getattr(self.cfg, "end_page", -1) or -1)
        if end < 0:
            end = total_pages
        end = min(total_pages, end)
        if start >= end:
            return results

        print(f"[NO_LLM] Processing pages {start+1}-{end} with local extraction", flush=True)
        for i in range(start, end):
            t0 = time.time()
            try:
                page = doc.load_page(i)
                print(f"Processing page {i+1}/{total_pages} (no-llm) ...", flush=True)
                results[i] = self._process_page(page, page_index=i, pdf_path=pdf_path, assets_dir=assets_dir)
                print(f"Finished page {i+1}/{total_pages} ({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"[NO_LLM] error page {i+1}: {e}", flush=True)
                results[i] = None
        return results

    def _process_page(self, page, page_index: int, pdf_path: Path, assets_dir: Path) -> str:
        import time
        page_start = time.time()
        
        # 1. Analyze Layout
        step_start = time.time()
        body_size = detect_body_font_size([page]) # Heuristic was on full doc, but per-page is okay fallback
        print(f"  [Page {page_index+1}] Step 1 (layout analysis): {time.time()-step_start:.2f}s", flush=True)
        
        # 2. Check if this is a references page (for special handling)
        step_start = time.time()
        is_references_page = _page_has_references_heading(page) or _page_looks_like_references_content(page)
        print(f"  [Page {page_index+1}] Step 2 (refs check): {time.time()-step_start:.2f}s", flush=True)
        
        # 3. Extract specific rects (excluding header/footer regions)
        step_start = time.time()
        W = float(page.rect.width)
        H = float(page.rect.height)
        header_threshold = H * 0.12
        footer_threshold = H * 0.88
        visual_rects = _collect_visual_rects(page)
        try:
            cap_candidates = self._extract_page_figure_caption_candidates(page)
        except Exception:
            cap_candidates = []

        def _near_caption(rect: "fitz.Rect") -> bool:
            for cap in (cap_candidates or []):
                try:
                    cr = fitz.Rect(cap.get("bbox"))
                except Exception:
                    continue
                if cr.width <= 0 or cr.height <= 0:
                    continue
                x_ov = _overlap_1d(float(rect.x0), float(rect.x1), float(cr.x0), float(cr.x1))
                min_w = max(1.0, min(float(rect.width), float(cr.width)))
                if (x_ov / min_w) < 0.30:
                    continue
                below_gap = float(cr.y0) - float(rect.y1)
                above_gap = float(rect.y0) - float(cr.y1)
                if 0.0 <= below_gap <= max(80.0, H * 0.10):
                    return True
                if 0.0 <= above_gap <= max(40.0, H * 0.05):
                    return True
            return False

        # Filter visual rects in header/footer (unless they're large figures)
        visual_rects = [
            r for r in visual_rects 
            if not (r.y1 < header_threshold or r.y0 > footer_threshold) or _rect_area(r) > (W * H * 0.15)
        ]
        filtered_visual_rects = []
        for r in visual_rects:
            area_ratio = _rect_area(r) / max(1.0, (W * H))
            wide_banner = float(r.width) >= W * 0.55 and float(r.height) <= H * 0.18 and float(r.y0) <= H * 0.28
            tiny_badge = area_ratio <= 0.015 and float(r.height) <= H * 0.08 and float(r.y0) <= H * 0.45
            if page_index == 0 and (wide_banner or tiny_badge) and (not _near_caption(r)):
                continue
            filtered_visual_rects.append(r)
        visual_rects = filtered_visual_rects
        print(f"  [Page {page_index+1}] Step 3 (visual rects): {time.time()-step_start:.2f}s, found {len(visual_rects)} rects", flush=True)
        
        # 4. Extract Tables
        step_start = time.time()
        # We need to map table rects to avoid processing them as text
        try:
            # Fast hint gate to enable more aggressive table strategies on table-heavy pages.
            # This significantly improves table extraction for dense CVPR/ICCV two-column PDFs.
            try:
                d0 = page.get_text("dict")
                has_table_hint = _page_maybe_has_table_from_dict(d0) if isinstance(d0, dict) else False
            except Exception:
                has_table_hint = False
            try:
                setattr(page, "has_table_hint", bool(has_table_hint))
            except Exception:
                pass

            tables_found = _extract_tables_by_layout(
                page, 
                pdf_path=pdf_path, 
                page_index=page_index,
                visual_rects=visual_rects,
                # Only enable pdfplumber fallback when page likely has a table.
                # If pdfplumber isn't installed, the fallback is a no-op.
                use_pdfplumber_fallback=bool(has_table_hint),
            )
            table_time = time.time() - step_start
            if table_time > 2.0:
                print(f"  [Page {page_index+1}] Step 4 (table extraction): {table_time:.2f}s (SLOW!), found {len(tables_found)} tables", flush=True)
            else:
                print(f"  [Page {page_index+1}] Step 4 (table extraction): {table_time:.2f}s, found {len(tables_found)} tables", flush=True)
        except Exception as e:
            print(f"  [Page {page_index+1}] Step 4 (table extraction) FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
            tables_found = []
        
        # 5. Extract Text Blocks
        step_start = time.time()
        blocks = self._extract_text_blocks(
            page, 
            page_index=page_index, 
            body_size=body_size,
            tables=tables_found,
            visual_rects=visual_rects,
            assets_dir=assets_dir,
            is_references_page=is_references_page
        )
        print(f"  [Page {page_index+1}] Step 5 (text blocks): {time.time()-step_start:.2f}s, found {len(blocks)} blocks", flush=True)

        # 5.5 Merge split math fragments BEFORE any LLM work / rendering.
        # PDF extraction frequently splits a single equation into many tiny blocks ("N X", ", (5)", "r 鈭?R").
        # If we try to repair each fragment in isolation, LaTeX quality collapses.
        step_start = time.time()
        try:
            blocks = self._merge_adjacent_math_fragments(blocks, page_wh=(page.rect.width, page.rect.height))
        except Exception:
            # Never fail conversion due to a heuristic merge.
            pass
        print(f"  [Page {page_index+1}] Step 5.5 (merge math frags): {time.time()-step_start:.2f}s, now {len(blocks)} blocks", flush=True)
        try:
            if bool(int(os.environ.get("KB_PDF_DEBUG_MATH", "0") or "0")):
                math_n0 = sum(1 for b in blocks if bool(getattr(b, "is_math", False)))
                code_n0 = sum(1 for b in blocks if bool(getattr(b, "is_code", False)))
                table_n0 = sum(1 for b in blocks if bool(getattr(b, "is_table", False)))
                print(
                    f"  [Page {page_index+1}] Debug: blocks math={math_n0} code={code_n0} table={table_n0}",
                    flush=True,
                )
        except Exception:
            pass
        
        # 6. LLM Classification / Repair
        step_start = time.time()
        speed_cfg = getattr(self, "_active_speed_config", None) or {}
        if self.cfg.llm and speed_cfg.get("use_llm_for_all", True):
            # Enhance blocks with LLM
            blocks = self._enhance_blocks_with_llm(blocks, page_index, page)
            print(f"  [Page {page_index+1}] Step 6 (LLM enhance): {time.time()-step_start:.2f}s", flush=True)
        else:
            print(f"  [Page {page_index+1}] Step 6 (LLM enhance): skipped (no LLM or disabled)", flush=True)
        try:
            if bool(int(os.environ.get("KB_PDF_DEBUG_MATH", "0") or "0")):
                math_n1 = sum(1 for b in blocks if bool(getattr(b, "is_math", False)))
                print(f"  [Page {page_index+1}] Debug: after enhance math={math_n1}", flush=True)
        except Exception:
            pass

        # Re-run math fragment merge AFTER LLM classify.
        # The classifier often flips is_math flags on blocks that heuristics missed; merging again
        # prevents one display equation from being rendered as many tiny $$ blocks (bad LaTeX).
        try:
            if self.cfg.llm and speed_cfg.get("use_llm_for_all", True):
                step_start2 = time.time()
                blocks = self._merge_adjacent_math_fragments(blocks, page_wh=(page.rect.width, page.rect.height))
                if bool(int(os.environ.get("KB_PDF_DEBUG_MATH", "0") or "0")):
                    math_n2 = sum(1 for b in blocks if bool(getattr(b, "is_math", False)))
                    print(
                        f"  [Page {page_index+1}] Debug: post-enhance merge math={math_n2} blocks={len(blocks)} ({time.time()-step_start2:.2f}s)",
                        flush=True,
                    )
        except Exception:
            pass
            
        # 7. Render
        step_start = time.time()
        result = self._render_blocks_to_markdown(
            blocks,
            page_index,
            page=page,
            assets_dir=assets_dir,
            is_references_page=is_references_page,
        )
        print(f"  [Page {page_index+1}] Step 7 (render): {time.time()-step_start:.2f}s", flush=True)
        print(f"  [Page {page_index+1}] TOTAL: {time.time()-page_start:.2f}s", flush=True)
        return result

    def _merge_adjacent_math_fragments(self, blocks: List[TextBlock], *, page_wh: tuple[float, float]) -> List[TextBlock]:
        """
        Merge adjacent math fragments split by PDF extraction.

        Typical failure mode (your screenshot):
        - one display equation becomes multiple blocks: "N X", ", (5)", "r 鈭?R", ...
        If we repair each in isolation, LaTeX becomes nonsense.
        """
        if not blocks:
            return blocks

        page_w = float(page_wh[0] or 1.0)

        def _t(b: TextBlock) -> str:
            return (b.text or "").strip()

        def _is_eq_no_text(s: str) -> bool:
            # "(5)" / "(EQNO 5)" / ", (5)"
            ss = s.strip()
            if re.fullmatch(r"[,;:]?\s*\(\s*(?:EQNO\s+)?\d{1,4}\s*\)\s*", ss, flags=re.IGNORECASE):
                return True
            if re.fullmatch(r"\(\s*\d{1,4}\s*\)", ss):
                return True
            return False

        def _is_tiny_connector(s: str) -> bool:
            ss = s.strip()
            return bool(re.fullmatch(r"[,.;:]", ss))

        def _looks_mathish_text(s: str) -> bool:
            ss = s.strip()
            if not ss:
                return False
            # very short math-y tokens
            if any(
                ch in ss
                for ch in [
                    "=",
                    "-",
                    "\u2212",
                    "<",
                    ">",
                    "\u2264",
                    "\u2265",
                    "\u2192",
                    "\u00d7",
                    "\u00b7",
                    "\u2211",
                    "\u2208",
                    "\u2209",
                ]
            ):
                return True
            # LaTeX-ish / OCR math
            if "\\" in ss or "^" in ss or "_" in ss:
                return True
            # bracket-heavy + sparse words
            if len(ss) <= 28 and re.search(r"[(){}\[\]]", ss) and not re.search(r"[A-Za-z]{3,}", ss):
                return True
            # "N X" / "r 鈭?R"
            if re.fullmatch(r"[A-Za-z]\s+[A-Za-z]", ss):
                return True
            if ("\u2212" in ss or "-" in ss) and len(ss) <= 24:
                return True
            return False

        def _looks_proseish_text(s: str) -> bool:
            """
            Detect paragraph-like text that may contain a couple math symbols,
            but should NOT be merged into a formula block.
            """
            ss = s.strip()
            if len(ss) < 80:
                return False
            try:
                word_n = len(re.findall(r"\b[A-Za-z]{2,}\b", ss))
                math_sym_n = len(re.findall(r"[=+\-*/^_{}\\\\\\[\\]]|[鈭堚墹鈮モ増脳路危鈭戔埆]", ss))
                has_sentence = (". " in ss) or ("? " in ss) or ("! " in ss)
                if word_n >= 14 and (math_sym_n <= 10 or has_sentence):
                    return True
            except Exception:
                return False
            return False

        def _y_overlap_ratio(r1: "fitz.Rect", r2: "fitz.Rect") -> float:
            y0 = max(float(r1.y0), float(r2.y0))
            y1 = min(float(r1.y1), float(r2.y1))
            ov = max(0.0, y1 - y0)
            denom = max(1e-6, min(float(r1.height), float(r2.height)))
            return ov / denom

        def _x_overlap_ratio(r1: "fitz.Rect", r2: "fitz.Rect") -> float:
            x0 = max(float(r1.x0), float(r2.x0))
            x1 = min(float(r1.x1), float(r2.x1))
            ov = max(0.0, x1 - x0)
            denom = max(1e-6, min(float(r1.width), float(r2.width)))
            return ov / denom

        merged: list[TextBlock] = []
        i = 0
        while i < len(blocks):
            b = blocks[i]
            bt = _t(b)
            if not bt:
                merged.append(b)
                i += 1
                continue

            # Start a merge group only when current block is math-ish (or already classified math).
            if not (b.is_math or _looks_mathish_text(bt)):
                merged.append(b)
                i += 1
                continue

            # Do not merge tables/code/captions/headings into math.
            if b.is_table or b.is_code or b.is_caption or b.heading_level:
                merged.append(b)
                i += 1
                continue

            group_texts = [bt]
            group_rect = fitz.Rect(b.bbox)
            last_rect = group_rect

            j = i + 1
            while j < len(blocks):
                nb = blocks[j]
                nt = _t(nb)
                if not nt:
                    j += 1
                    continue

                # Stop at structural blocks.
                if nb.is_table or nb.is_code or nb.is_caption or nb.heading_level:
                    break

                # Avoid swallowing normal paragraphs into an equation merge.
                # (The paragraph often contains symbols like "r鈭圧" which are math-ish, but it's prose.)
                if (not nb.is_math) and _looks_proseish_text(nt):
                    break

                nr = fitz.Rect(nb.bbox)

                # Same-line merge: strong y-overlap, small x-gap.
                y_ov = _y_overlap_ratio(last_rect, nr)
                x_gap = float(nr.x0) - float(last_rect.x1)
                same_line = (y_ov >= 0.65) and (x_gap <= max(10.0, page_w * 0.03))

                # Stacked-line merge (multi-line equation): x overlap, small y-gap.
                y_gap = float(nr.y0) - float(last_rect.y1)
                x_ov = _x_overlap_ratio(group_rect, nr)
                stacked = (y_gap >= -2.0) and (y_gap <= max(14.0, (float(last_rect.height) + float(nr.height)) * 0.35)) and (x_ov >= 0.25)

                # Allow merging tiny connector / equation-number fragments near math.
                is_ok_token = nb.is_math or _looks_mathish_text(nt) or _is_eq_no_text(nt) or _is_tiny_connector(nt)

                if is_ok_token and (same_line or stacked):
                    # Preserve line breaks for stacked parts; spaces for same-line.
                    if stacked and not same_line:
                        group_texts.append("\n" + nt)
                    else:
                        group_texts.append(" " + nt)
                    group_rect = _union_rect(group_rect, nr)
                    last_rect = nr
                    j += 1
                    continue

                break

            if j == i + 1:
                merged.append(b)
                i += 1
                continue

            merged_text = "".join(group_texts).strip()
            bd = b.model_dump()
            bd["bbox"] = tuple(group_rect)
            bd["text"] = merged_text
            bd["is_math"] = True
            bd["is_code"] = False
            bd["is_caption"] = False
            bd["heading_level"] = None
            merged.append(TextBlock(**bd))
            i = j

        return merged

    def _extract_text_blocks(
        self, 
        page, 
        page_index: int, 
        body_size: float, 
        tables: List[Tuple[fitz.Rect, str]], 
        visual_rects: List[fitz.Rect],
        assets_dir: Path,
        is_references_page: bool = False
    ) -> List[TextBlock]:
        import time
        extract_start = time.time()
        
        text_blocks = []
        # Get raw blocks
        step_start = time.time()
        page_dict = page.get_text("dict")
        raw_blocks = page_dict.get("blocks", [])
        print(f"    [Page {page_index+1}] get_text('dict'): {time.time()-step_start:.2f}s, {len(raw_blocks)} raw blocks", flush=True)
        
        W = float(page.rect.width)
        H = float(page.rect.height)
        
        # Mask out tables and figures
        ignore_rects = [r for r, _ in tables] + visual_rects
        
        for b in raw_blocks:
            bbox = fitz.Rect(b["bbox"])
            
            # Check overlap with tables/figures
            is_masked = False
            for ig in ignore_rects:
                if _rect_intersection_area(bbox, ig) > _rect_area(bbox) * 0.5:
                    is_masked = True
                    break
            if is_masked:
                continue
                
            # Process lines: keep MuPDF line structure. A single raw block often contains
            # both display equations and a following prose explanation ("where ...").
            # If we flatten everything with spaces, the equation and prose become inseparable.
            max_size = 0.0
            is_bold = False

            if "lines" not in b:
                continue

            from .heuristics import _looks_like_equation_text, _is_caption_like_text

            line_items: list[tuple[fitz.Rect, str]] = []

            def _line_overlaps_visual(line_rect: "fitz.Rect", line_text: str) -> bool:
                try:
                    if _is_caption_like_text(line_text):
                        return False
                except Exception:
                    pass
                line_area = max(1.0, _rect_area(line_rect))
                for vr in visual_rects:
                    try:
                        inter = _rect_intersection_area(line_rect, vr)
                    except Exception:
                        inter = 0.0
                    if inter <= 0.0:
                        continue
                    if (inter / line_area) >= 0.40:
                        return True
                return False

            for l in b["lines"]:
                spans = l.get("spans") or []
                parts: list[str] = []
                for s in spans:
                    t = (s.get("text") or "")
                    if not t.strip():
                        continue
                    # Track font size/bold
                    try:
                        size = float(s.get("size") or 0.0)
                    except Exception:
                        size = 0.0
                    try:
                        font = str(s.get("font") or "").lower()
                    except Exception:
                        font = ""
                    if size > max_size:
                        max_size = size
                    if ("bold" in font) or ("dubai-bold" in font):
                        is_bold = True
                    parts.append(t)
                line_text = " ".join(parts).strip()
                if not line_text:
                    continue
                try:
                    lb = fitz.Rect(l.get("bbox"))
                except Exception:
                    lb = bbox
                if _line_overlaps_visual(lb, line_text):
                    continue
                line_items.append((lb, line_text))

            if not line_items:
                continue

            # Split this raw block into sub-blocks by line-type (math-like vs prose-like).
            # This prevents "equation + where paragraph" from becoming one giant math block.

            def _is_math_line(txt: str) -> bool:
                tt = (txt or "").strip()
                if not tt:
                    return False
                low = tt.lower()
                if is_references_page:
                    if re.match(r"^\s*(?:\[\s*\d{1,4}\s*\]|\d{1,4}\.\s+[A-Z])", tt):
                        return False
                    if re.search(r"\b(?:19|20)\d{2}\b", tt) and len(re.findall(r"\b[A-Za-z]{2,}\b", tt)) >= 4:
                        return False
                # Captions should never be merged into equations.
                try:
                    if _is_caption_like_text(tt):
                        return False
                except Exception:
                    pass
                # Prose explanation lines commonly start with "where/with/and" and include many words.
                try:
                    word_n0 = len(re.findall(r"\b[A-Za-z]{2,}\b", tt))
                    if (low.startswith("where ") or low.startswith("with ") or low.startswith("and ")) and word_n0 >= 6:
                        return False
                    # Long sentence-like lines with few hard equation anchors are usually prose.
                    if word_n0 >= 14 and ("=" not in tt) and ("\\sum" not in tt) and ("\\int" not in tt):
                        return False
                except Exception:
                    pass
                if _looks_like_math_block([tt]):
                    return True
                if _looks_like_equation_text(tt):
                    return True
                # Extra: short symbol-heavy lines are usually equation lines
                sym_n = len(re.findall(r"[=+\-*/^_{}\\\\\\[\\]]|[鈭堚墹鈮モ増脳路危鈭戔埆]", tt))
                word_n = len(re.findall(r"\b[A-Za-z]{2,}\b", tt))
                if len(tt) <= 80 and sym_n >= 3 and word_n <= 10:
                    return True
                return False

            groups: list[tuple[bool, list[tuple[fitz.Rect, str]]]] = []
            cur_is_math = _is_math_line(line_items[0][1])
            cur: list[tuple[fitz.Rect, str]] = []
            for lr, lt in line_items:
                m = _is_math_line(lt)
                if cur and (m != cur_is_math):
                    groups.append((cur_is_math, cur))
                    cur = []
                    cur_is_math = m
                cur.append((lr, lt))
            if cur:
                groups.append((cur_is_math, cur))

            for group_is_math_hint, group_lines in groups:
                try:
                    rects = [rr for rr, _ in group_lines]
                    group_bbox = _union_rect(rects) or rects[0]
                except Exception:
                    group_bbox = group_lines[0][0]

                # For heuristics like heading/caption/noise detection we want a flat string.
                full_text_space = " ".join(x for _, x in group_lines).strip()
                if not full_text_space:
                    continue
                pre_is_caption = _is_caption_like_text(full_text_space)

                # Source-level suppression: strip author/affiliation/journal metadata blocks
                # before they enter body reconstruction.
                if not pre_is_caption:
                    if _is_non_body_metadata_text(
                        full_text_space,
                        page_index=page_index,
                        y0=float(group_bbox.y0),
                        y1=float(group_bbox.y1),
                        page_height=float(H),
                        max_font_size=float(max_size),
                        body_font_size=float(body_size),
                        is_references_page=bool(is_references_page),
                    ):
                        continue
            
                # Check if this looks like author information (should not be heading)
                # Pattern: "Name 1 , Name 2 , Name 3 , and Name 4 ,"
                if re.search(r'^\s*[A-Z][a-z]+\s+\d+\s*,.*\d+\s*,.*and.*\d+\s*,?\s*$', full_text_space):
                    tb = TextBlock(
                        bbox=tuple(group_bbox),
                        text=full_text_space,
                        max_font_size=max_size,
                        is_bold=is_bold,
                        heading_level=None,
                        is_math=False,
                        is_code=False,
                        is_caption=False
                    )
                    text_blocks.append(tb)
                    continue
            
                # Detect math formulas, code blocks, and captions early
                is_math = False
                is_code = False
                is_caption = False

                # Check if this is a caption (Figure/Table caption)
                from .heuristics import _is_caption_like_text
                is_caption = bool(pre_is_caption or _is_caption_like_text(full_text_space))

                # Determine math/codelike. Prefer line-level hint to avoid misclassifying prose lines.
                if not is_caption:
                    group_line_texts = [x for _, x in group_lines]
                    is_math = bool(group_is_math_hint) or _looks_like_math_block(group_line_texts)
                    if not is_math:
                        is_code = _looks_like_code_block(group_line_texts)
                # Preserve line breaks for math blocks for better repair/rendering.
                full_text = ("\n".join(x for _, x in group_lines) if is_math else full_text_space).strip()
                if not full_text:
                    continue
            
                # Filter noise - check if in header/footer region
                is_header_footer = False
                header_threshold = H * 0.12  # Top 12% of page
                footer_threshold = H * 0.88  # Bottom 12% of page
                if group_bbox.y1 < header_threshold or group_bbox.y0 > footer_threshold:
                    is_header_footer = True
                    # Still allow if it's a major heading (e.g., section title at top of page)
                    if not (max_size > body_size + 1.5 and is_bold and len(full_text_space) < 100):
                        if full_text_space in self.noise_texts or _is_noise_line(full_text_space):
                            continue
            
                if full_text_space in self.noise_texts:
                    continue
                if _is_frontmatter_noise_line(full_text_space):
                    continue
                if _is_noise_line(full_text_space) and not is_header_footer:
                    continue
                
                # Detect heading (heuristic-based, LLM will refine later if available)
                # Skip heading detection if this is already classified as math, code, or caption
                heading_level = None
                if not is_math and not is_code and not is_caption:
                    # Additional check: very short text with numbers is likely formula, not heading
                    text_stripped = full_text_space.strip()
                    if len(text_stripped) <= 25:
                        # Check for formula-like patterns more strictly
                        if re.search(r'^\s*[A-Z]?\s*\d+\s*[a-z]', text_stripped) or \
                           re.search(r'^\s*\d+\s*[a-z]+\s*[+\-]', text_stripped) or \
                           re.search(r'^\s*[a-z]\s*\d+', text_stripped) or \
                           (re.search(r'\d+.*[a-z]|[a-z].*\d+', text_stripped) and not re.search(r'[A-Z]{2,}', text_stripped) and '=' not in text_stripped):
                            is_math = True  # Reclassify as math
                            is_caption = False
                            full_text = ("\n".join(x for _, x in group_lines)).strip()

                    if not is_math:
                        heading_tag = detect_header_tag(
                            page_index=page_index,
                            text=full_text_space,
                            max_size=max_size,
                            is_bold=is_bold,
                            body_size=body_size,
                            page_width=W,
                            bbox=tuple(group_bbox),
                        )
                        if heading_tag:
                            heading_level = heading_tag
                
                # Create Block with detected types
                tb = TextBlock(
                    bbox=tuple(group_bbox),
                    text=full_text,
                    max_font_size=max_size,
                    is_bold=is_bold,
                    heading_level=heading_level,
                    is_math=is_math,
                    is_code=is_code,
                    is_caption=is_caption
                )
                text_blocks.append(tb)

        # Insert Tables as Blocks
        for rect, md in tables:
            tb = TextBlock(
                bbox=tuple(rect),
                text="[TABLE]",
                max_font_size=body_size,
                is_table=True,
                table_markdown=md
            )
            text_blocks.append(tb)
            
        # Insert Images (Visual Rects) as Blocks
        step_start = time.time()
        # Filter out header/footer regions and crop properly
        header_threshold = H * 0.12
        footer_threshold = H * 0.88
        side_margin = W * 0.05  # 5% margin on sides
        
        # Detect column layout for proper image handling
        col_split = _detect_column_split_x(text_blocks, page_width=W) if text_blocks else None
        spanning_threshold = W * 0.55  # Full-width images span both columns
        line_boxes = self._collect_page_text_line_boxes(page)
        
        img_count = 0
        figure_entries: list[dict] = []
        for rect_idx, rect in enumerate(visual_rects):
            img_step_start = time.time()
            # Check if this is a full-width image (spans both columns or most of page)
            is_full_width = _rect_area(rect) >= (W * H * 0.40) or _bbox_width(tuple(rect)) >= spanning_threshold
            
            # Skip if in header/footer region (likely page numbers, headers)
            if rect.y1 < header_threshold or rect.y0 > footer_threshold:
                # Allow if it's a large figure (likely a real figure, not header/footer)
                if not is_full_width and _rect_area(rect) < (W * H * 0.15):  # Less than 15% of page area
                    continue
            
            # Skip small edge artifacts (unless it's a full-width image)
            if not is_full_width:
                if rect.x0 < side_margin and rect.width < W * 0.1:
                    continue
                if rect.x1 > W - side_margin and rect.width < W * 0.1:
                    continue
            
            # For full-width images in double-column layout, ensure we capture the full width
            if is_full_width and col_split:
                # Expand rect to full width if it's close to spanning
                if _bbox_width(tuple(rect)) < W * 0.85:
                    # It might be a full-width image that was detected as two separate rects
                    # Keep original rect but ensure proper cropping
                    pass
            
            cropped_rect = self._expanded_visual_crop_rect(
                rect=rect,
                page_w=W,
                page_h=H,
                is_full_width=bool(is_full_width),
                line_boxes=line_boxes,
            )
            
            if cropped_rect.width <= 0 or cropped_rect.height <= 0:
                continue
                
            # Save image with proper DPI from config
            # Use a stable per-page index to avoid filename collisions/overwrites on Windows.
            img_name = f"page_{page_index+1}_fig_{rect_idx+1}.png"
            img_path = assets_dir / img_name
            used_clip = cropped_rect
            
            # Use configured DPI
            dpi = self.dpi
            try:
                pixmap_start = time.time()
                pix = page.get_pixmap(clip=cropped_rect, dpi=dpi)
                pixmap_time = time.time() - pixmap_start
                save_start = time.time()
                pix.save(img_path)
                save_time = time.time() - save_start
                if pixmap_time > 1.0 or save_time > 1.0:
                    print(f"      [Page {page_index+1}] Image {rect_idx+1}/{len(visual_rects)}: get_pixmap={pixmap_time:.2f}s, save={save_time:.2f}s, size={cropped_rect.width:.0f}x{cropped_rect.height:.0f}", flush=True)
            except Exception as e:
                print(f"      [Page {page_index+1}] Image {rect_idx+1} get_pixmap failed: {e}", flush=True)
                # Fallback to original rect if crop fails
                try:
                    used_clip = rect
                    pix = page.get_pixmap(clip=rect, dpi=dpi)
                    pix.save(img_path)
                except Exception:
                    continue

            # Guard: don't emit broken markdown links.
            try:
                if (not img_path.exists()) or (img_path.stat().st_size < 256):
                    continue
            except Exception:
                continue
            
            tb = TextBlock(
                bbox=tuple(rect),
                text=f"![Figure](./assets/{img_name})",
                max_font_size=body_size
            )
            text_blocks.append(tb)
            figure_entries.append(
                {
                    "asset_name": img_name,
                    "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                    "crop_bbox": [float(used_clip.x0), float(used_clip.y0), float(used_clip.x1), float(used_clip.y1)],
                }
            )
            img_count += 1
            if time.time() - img_step_start > 0.5:
                print(f"      [Page {page_index+1}] Image {rect_idx+1} processing took {time.time()-img_step_start:.2f}s", flush=True)
        
        print(f"    [Page {page_index+1}] Image processing: {time.time()-step_start:.2f}s, processed {img_count}/{len(visual_rects)} images", flush=True)
        try:
            if figure_entries:
                cap_candidates = self._extract_page_figure_caption_candidates(page)
                figure_entries = self._match_figure_entries_with_captions(
                    page=page,
                    figure_entries=figure_entries,
                    caption_candidates=cap_candidates,
                )
            self._persist_page_figure_metadata(
                assets_dir=assets_dir,
                page_index=page_index,
                figure_entries=figure_entries,
            )
        except Exception as e:
            print(f"    [Page {page_index+1}] Figure metadata persist skipped: {e}", flush=True)

        # Sort reading order
        sort_start = time.time()
        sorted_blocks = sort_blocks_reading_order(text_blocks, page_width=W)
        print(f"    [Page {page_index+1}] Sort: {time.time()-sort_start:.2f}s", flush=True)
        print(f"    [Page {page_index+1}] _extract_text_blocks TOTAL: {time.time()-extract_start:.2f}s", flush=True)
        return sorted_blocks

    def _enhance_blocks_with_llm(self, blocks: List[TextBlock], page_index: int, page) -> List[TextBlock]:
        # Call classifier
        classified = self.llm_worker.call_llm_classify_blocks(
            blocks, 
            page_number=page_index, 
            page_wh=(page.rect.width, page.rect.height)
        )
        if not classified:
            return blocks
        
        # Index classifications by block index to avoid O(n^2) scans.
        cls_by_i: dict[int, dict] = {}
        try:
            for it in classified:
                ii = it.get("i")
                if isinstance(ii, int):
                    cls_by_i[ii] = it
        except Exception:
            cls_by_i = {}

        # Apply classifications - create new blocks since TextBlock is immutable
        enhanced_blocks = []
        for i, b in enumerate(blocks):
            item = cls_by_i.get(i)
            
            if item:
                kind = item.get("kind")
                # Create new block with updated properties
                block_dict = b.model_dump()
                if kind == "heading":
                    lvl = item.get("heading_level")
                    if lvl:
                        block_dict["heading_level"] = f"[H{lvl}]"
                    # Clear other flags when classified as heading
                    block_dict["is_math"] = False
                    block_dict["is_code"] = False
                    block_dict["is_caption"] = False
                elif kind == "table":
                    block_dict["is_table"] = True
                    block_dict["is_math"] = False
                    block_dict["is_code"] = False
                elif kind == "code":
                    block_dict["is_code"] = True
                    block_dict["is_math"] = False
                    block_dict["is_caption"] = False
                elif kind == "math":
                    block_dict["is_math"] = True
                    block_dict["is_code"] = False
                    block_dict["heading_level"] = None  # Clear heading if it's math
                elif kind == "caption":
                    block_dict["is_caption"] = True
                    block_dict["is_math"] = False
                    block_dict["is_code"] = False
                    block_dict["heading_level"] = None  # Clear heading if it's caption
                enhanced_blocks.append(TextBlock(**block_dict))
            else:
                enhanced_blocks.append(b)
                    
        return enhanced_blocks

    def _convert_formula_to_latex(self, text: str) -> str:
        """Convert formula text to LaTeX format."""
        if not text:
            return ""
        
        # Fast path: if text is very long, skip expensive processing
        if len(text) > 1000:
            # For very long formulas, just do basic cleanup and return
            t = text.strip()
            t = re.sub(r'\(\s*(?:EQNO\s+)?\d+\s*\)\s*$', '', t).strip()
            return t
        
        # Normalize text but preserve line structure for multi-line formulas
        # Skip expensive _normalize_text for math formulas - just basic cleanup
        lines = text.splitlines()
        normalized_lines = []
        for line in lines:
            t = line.strip()
            if t:
                normalized_lines.append(t)
        
        if not normalized_lines:
            return ""
        
        t = " ".join(normalized_lines)
        
        # Remove equation numbers at the end like "(8)" or "(EQNO 8)"
        t = re.sub(r'\(\s*(?:EQNO\s+)?\d+\s*\)\s*$', '', t).strip()
        
        # Replace Greek letters first (before other processing)
        # Use simple replace (fast enough for small dicts)
        for greek, latex in GREEK_TO_LATEX.items():
            if greek in t:  # Only check if present (faster)
                t = t.replace(greek, latex)
        
        # Replace math symbols
        for symbol, latex in MATH_SYMBOL_TO_LATEX.items():
            if symbol in t:  # Only check if present (faster)
                t = t.replace(symbol, latex)

        # Fix common OCR spacing around parentheses: "C ( r )" -> "C(r)"
        # Do this early so subsequent sub/superscript rules see a cleaner token stream.
        try:
            t = re.sub(r"\s*\(\s*", "(", t)
            t = re.sub(r"\s*\)\s*", ")", t)
        except Exception:
            pass

        # Fix superscripts in a safe, targeted way:
        # - "C(r) 2" -> "C(r)^2"
        # Only trigger when a digit follows a closing bracket (very likely exponent, not index).
        try:
            t = re.sub(r"([\)\]\}])\s+(\d{1,2})\b", r"\1^{\2}", t)
        except Exception:
            pass

        # Fix "hat" when OCR emits a caret with whitespace: "^ C" or "藛 C" (normalized to "^ C")
        # This avoids confusing "^" exponent usage because we REQUIRE whitespace after caret.
        try:
            t = re.sub(r"\^\s+([A-Za-z])\b", r"\\hat{\1}", t)
        except Exception:
            pass
        
        # Normalize whitespace but be careful with subscripts
        # First, protect potential subscripts/superscripts
        # Pattern: letter followed by space and then digit or lowercase letter
        # But only if not already in LaTeX format
        
        # Fix common math functions - compile regex once for speed
        if not hasattr(self, '_math_func_regexes'):
            self._math_func_regexes = {
                'log': re.compile(r'\blog\b'),
                'exp': re.compile(r'\bexp\b'),
                'sin': re.compile(r'\bsin\b'),
                'cos': re.compile(r'\bcos\b'),
                'tan': re.compile(r'\btan\b'),
                'max': re.compile(r'\bmax\b'),
                'min': re.compile(r'\bmin\b'),
                'ln': re.compile(r'\bln\b'),
                'sqrt': re.compile(r'\bsqrt\b'),
            }
            self._math_func_replacements = {
                'log': r'\\log',
                'exp': r'\\exp',
                'sin': r'\\sin',
                'cos': r'\\cos',
                'tan': r'\\tan',
                'max': r'\\max',
                'min': r'\\min',
                'ln': r'\\ln',
                'sqrt': r'\\sqrt',
            }
        
        for func, regex in self._math_func_regexes.items():
            t = regex.sub(self._math_func_replacements[func], t)
        
        # Fix subscripts more carefully
        # Pattern: variable name (single letter or word) followed by space and digit/single letter
        # Only if it's not already in LaTeX format
        if '_{' not in t and '_' not in t:
            # Simple case: "x 1" -> "x_1", but be careful
            # Only do this for single letters followed by single digits/letters
            # Use compiled regex for speed
            if not hasattr(self, '_subscript_regex1'):
                self._subscript_regex1 = re.compile(r'\b([a-zA-Z])\s+(\d+)\b')
                self._subscript_regex2 = re.compile(r'\b([a-zA-Z])\s+([a-z])\b(?!\w)')
            t = self._subscript_regex1.sub(r'\1_{\2}', t)
            # For single lowercase letters as subscripts: "x i" -> "x_i" (but not "x in")
            t = self._subscript_regex2.sub(r'\1_{\2}', t)
        
        # Normalize remaining whitespace - use compiled regex
        if not hasattr(self, '_whitespace_regex'):
            self._whitespace_regex = re.compile(r'\s+')
            self._operator_regexes = {
                '=': re.compile(r'\s*=\s*'),
                '+': re.compile(r'\s*\+\s*'),
                '-': re.compile(r'\s*-\s*'),
                '*': re.compile(r'\s*\*\s*'),
                '/': re.compile(r'\s*/\s*'),
            }
        t = self._whitespace_regex.sub(' ', t).strip()
        
        # Fix common operators that might have been split
        t = self._operator_regexes['='].sub(' = ', t)
        t = self._operator_regexes['+'].sub(' + ', t)
        t = self._operator_regexes['-'].sub(' - ', t)
        # Replace * with \cdot, but escape the backslash properly
        t = self._operator_regexes['*'].sub(r' \\cdot ', t)
        t = self._operator_regexes['/'].sub(' / ', t)
        
        # Clean up extra spaces around operators
        t = self._whitespace_regex.sub(' ', t)
        
        return t

    def _render_blocks_to_markdown(
        self,
        blocks: List[TextBlock],
        page_index: int,
        *,
        page=None,
        assets_dir: Path | None = None,
        is_references_page: bool = False,
    ) -> str:
        return render_blocks_to_markdown(
            self,
            blocks,
            page_index,
            page=page,
            assets_dir=assets_dir,
            is_references_page=is_references_page,
        )

    def _merge_split_formulas(self, md: str) -> str:
        """Merge consecutive formula blocks that were split across lines."""
        lines = md.splitlines()
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check for display math block start
            if stripped == "$$":
                # Collect formula content
                formula_parts = []
                i += 1
                
                # Collect until we find closing $$
                while i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.strip()
                    
                    if next_stripped == "$$":
                        # Found closing $$
                        i += 1
                        break
                    elif next_stripped:
                        formula_parts.append(next_stripped)
                    i += 1
                
                # Merge formula parts
                if formula_parts:
                    merged = " ".join(formula_parts)
                    # Clean up: remove duplicate spaces, fix common issues
                    merged = re.sub(r'\s+', ' ', merged)
                    result.append("$$")
                    result.append(merged)
                    result.append("$$")
                    result.append("")
            else:
                # Regular line - check if it's inline math
                if re.search(r'\$[^$]+\$', line):
                    result.append(line)
                else:
                    result.append(line)
                i += 1
        
        # Second pass: merge consecutive $$ blocks and inline math that should be together
        final_result = []
        i = 0
        while i < len(result):
            line = result[i]
            stripped = line.strip()
            
            if stripped == "$$":
                # Start collecting consecutive formula blocks
                formula_blocks = []
                i += 1
                
                # Collect this formula
                current_formula = []
                while i < len(result) and result[i].strip() != "$$":
                    if result[i].strip():
                        current_formula.append(result[i].strip())
                    i += 1
                
                if i < len(result) and result[i].strip() == "$$":
                    i += 1
                    if current_formula:
                        formula_blocks.append(" ".join(current_formula))
                
                # Check if next blocks are also formulas (within 3 lines, including inline math)
                blank_count = 0
                inline_math_blocks = []
                while i < len(result) and blank_count < 3:
                    next_line = result[i]
                    next_stripped = next_line.strip()
                    
                    if next_stripped == "":
                        blank_count += 1
                        i += 1
                    elif next_stripped == "$$":
                        # Another display math block - merge it
                        i += 1
                        next_formula = []
                        while i < len(result) and result[i].strip() != "$$":
                            if result[i].strip():
                                next_formula.append(result[i].strip())
                            i += 1
                        if i < len(result) and result[i].strip() == "$$":
                            i += 1
                            if next_formula:
                                formula_blocks.append(" ".join(next_formula))
                        blank_count = 0
                    elif re.match(r'^\$[^$]+\$$', next_stripped):
                        # Inline math - collect it if it looks like part of a larger formula
                        inline_math_blocks.append(next_stripped)
                        i += 1
                        blank_count = 0
                    else:
                        # Check if this line looks like a formula fragment (short, contains math symbols)
                        if len(next_stripped) < 50 and re.search(r'[+\-*/=<>鈮も墺鈭堚垜鈭蔽参澄次滴肝晃枷€蟽蟿蠁蠅]', next_stripped):
                            # Might be a formula fragment
                            inline_math_blocks.append(next_stripped)
                            i += 1
                            blank_count = 0
                        else:
                            break
                
                # Merge all formula blocks
                all_parts = formula_blocks + inline_math_blocks
                if all_parts:
                    merged = " ".join(all_parts)
                    merged = re.sub(r'\s+', ' ', merged)
                    final_result.append("$$")
                    final_result.append(merged)
                    final_result.append("$$")
                    final_result.append("")
            else:
                final_result.append(line)
                i += 1
        
        return "\n".join(final_result)

    def _merge_fragmented_formulas(self, md: str) -> str:
        """
        Aggressively merge fragmented formula pieces that are on separate lines.
        Detects formula fragments (like 危, i=1, N, T(x,y), etc.) and merges them into complete formulas.
        """
        lines = md.splitlines()
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check if this line looks like a formula fragment
            is_formula_fragment = self._is_formula_fragment(stripped)
            
            if is_formula_fragment:
                # Collect consecutive formula fragments
                formula_parts = [stripped]
                j = i + 1
                
                # Look ahead for more formula fragments (up to 10 lines)
                while j < len(lines) and j < i + 10:
                    next_line = lines[j]
                    next_stripped = next_line.strip()
                    
                    # Skip empty lines
                    if not next_stripped:
                        j += 1
                        continue
                    
                    # Check if next line is also a formula fragment
                    if self._is_formula_fragment(next_stripped):
                        formula_parts.append(next_stripped)
                        j += 1
                    else:
                        # Check if it's a continuation (has math-like content but not a full formula)
                        if self._looks_like_formula_continuation(next_stripped, formula_parts):
                            formula_parts.append(next_stripped)
                            j += 1
                        else:
                            break
                
                # Merge the fragments into a single formula
                if len(formula_parts) > 1:
                    merged_formula = ' '.join(formula_parts)
                    # Clean up the merged formula
                    merged_formula = re.sub(r'\s+', ' ', merged_formula)  # Normalize spaces
                    merged_formula = re.sub(r'\s*([+\-*/=,()\[\]{}])\s*', r'\1', merged_formula)  # Remove spaces around operators
                    merged_formula = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1\2', merged_formula)  # Remove spaces between letters
                    # Wrap in $$ if not already wrapped
                    if not merged_formula.startswith('$$'):
                        merged_formula = f"$${merged_formula}$$"
                    result.append(merged_formula)
                    i = j
                    continue
            
            result.append(line)
            i += 1
        
        return "\n".join(result)
    
    def _is_formula_fragment(self, text: str) -> bool:
        """Check if a line is a formula fragment (part of a larger formula)."""
        if not text or len(text) < 1:
            return False
        
        # Already wrapped in $$ - might be complete or incomplete
        if text.startswith('$$') or text.endswith('$$'):
            return True
        
        # Check for common formula fragment patterns
        fragment_patterns = [
            r'^[鈭戔埆鈭徫Ｎ燷',  # Sum/integral/product symbols
            r'^[a-zA-Z]_\{[^}]+\}',  # Subscript like x_{i}
            r'^[a-zA-Z]\^\{[^}]+\}',  # Superscript like x^{2}
            r'^[a-zA-Z]\^[0-9]',  # Superscript like x^2
            r'^[a-zA-Z]_[0-9a-z]',  # Subscript like x_i
            r'^\\[a-zA-Z]+\{',  # LaTeX command with brace
            r'^[+\-*/=]',  # Operators at start
            r'^[()\[\]{}]',  # Brackets at start
            r'^[伪尾纬未蔚味畏胃喂魏位渭谓尉慰蟺蟻蟽蟿蠀蠁蠂蠄蠅]',  # Greek letters
            r'^[0-9]+\s*[+\-*/=]',  # Number followed by operator
            r'^[a-zA-Z]\([^)]*\)',  # Function call like T(x,y)
            r'^<[^>]+>',  # Angle brackets like <B_i>
            r'^[a-zA-Z]\s*=\s*[0-9]',  # Assignment like i=1
        ]
        
        for pattern in fragment_patterns:
            if re.match(pattern, text):
                return True
        
        # Check if it's a very short line with math-like content
        if len(text) <= 15 and re.search(r'[+\-*/^_{}\[\]()=<>鈭戔埆鈭徫Ｎ犖?蠅]', text):
            # But exclude if it looks like regular text
            if not re.search(r'\b(the|and|or|is|are|was|were|in|on|at|to|for|of|with|by)\b', text, re.IGNORECASE):
                return True
        
        return False
    
    def _looks_like_formula_continuation(self, text: str, existing_parts: list[str]) -> bool:
        """Check if text looks like a continuation of the formula fragments."""
        if not text:
            return False
        
        # Check for math-like content
        has_math = bool(re.search(r'[+\-*/^_{}\[\]()=<>鈭戔埆鈭徫Ｎ犖?蠅\\]', text))
        
        # Check if it doesn't look like regular text
        has_text_words = bool(re.search(r'\b(the|and|or|is|are|was|were|in|on|at|to|for|of|with|by)\b', text, re.IGNORECASE))
        
        # If it has math content and no text words, it's likely a continuation
        if has_math and not has_text_words:
            return True
        
        # Check if it's a bracket or operator that continues the formula
        if re.match(r'^[()\[\]{}+\-*/=,<>]', text.strip()):
            return True
        
        return False

    def _basic_formula_cleanup(self, md: str) -> str:
        """
        Basic formula cleanup - only remove obviously broken fragments.
        CRITICAL: Preserve all $$...$$ and $...$ blocks - do NOT remove them.
        Rely on improved prompt for correct formula formatting.
        """
        lines = md.splitlines()
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # CRITICAL: Preserve $$ markers - they are essential for formula rendering
            # Only skip if it's clearly a broken empty marker
            if stripped == '$$ $$' or (stripped == '$$' and i + 1 < len(lines) and lines[i + 1].strip() == '$$'):
                # Empty $$ block - skip it
                i += 1
                continue
            
            # Skip obviously broken non-formula fragments (but NOT $$ markers)
            if stripped in [')(', '()', ')((', '()(']:
                i += 1
                continue
            
            # Skip very short lines that are just fragments (like "N", "i=1" alone)
            # BUT: if it's inside a $$ block, keep it
            if len(stripped) <= 5 and re.match(r'^[Nn]$|^i\s*=\s*1$|^\\sum\s*$$$', stripped):
                # Check if we're inside a $$ block
                # Look backwards for opening $$
                in_math_block = False
                for j in range(i - 1, max(-1, i - 20), -1):
                    if j < 0:
                        break
                    if lines[j].strip() == '$$':
                        in_math_block = True
                        break
                
                # If not in math block, check if next line is also a fragment
                if not in_math_block:
                    if i + 1 < len(lines):
                        next_stripped = lines[i + 1].strip()
                        if next_stripped and len(next_stripped) <= 10 and re.search(r'[+\-*/=<>鈭戔埆鈭徫Ｎ犖?蠅\\]', next_stripped):
                            i += 1
                            continue
                    i += 1
                    continue
            
            # Fix lines with multiple $$ blocks (like "$$)(2)$$ B=HT=...")
            # But preserve the $$ markers
            if stripped.count('$$') >= 4:
                # Extract all formula-like parts
                parts = re.split(r'\$\$', stripped)
                formula_parts = []
                for p in parts:
                    p = p.strip()
                    if p and not re.match(r'^[)(]+$', p):  # Skip pure brackets
                        # Remove strange bracket patterns
                        p = re.sub(r'^[)(]+', '', p)
                        p = re.sub(r'[)(]+$', '', p)
                        if p and len(p) > 2:
                            formula_parts.append(p)
                
                if formula_parts:
                    # Try to merge into one formula
                    merged = ' '.join(formula_parts)
                    merged = re.sub(r'\s+', ' ', merged)
                    if len(merged) > 10:
                        line = f"$${merged}$$"
            
            # CRITICAL: Always preserve the line, especially if it contains $ or $$
            result.append(line)
            i += 1
        
        return "\n".join(result)

    def _fix_heading_structure(self, md: str) -> str:
        """Fix heading hierarchy to ensure proper structure."""
        lines = md.splitlines()
        out = []
        heading_stack = [0]  # Track heading levels
        
        for line in lines:
            stripped = line.strip()
            
            # Check if it's a heading
            match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                
                # Skip if heading looks like a formula (more patterns)
                is_formula = False
                if re.search(r'^\s*[A-Z]?\s*\d+\s*[a-z]', text) or \
                   re.search(r'^\s*\d+\s*[a-z]+\s*[+\-]', text) or \
                   (len(text) <= 15 and re.search(r'\d+.*[a-z]|[a-z].*\d+', text) and '=' in text) or \
                   re.search(r'[伪尾纬未蔚味畏胃喂魏位渭谓尉慰蟺蟻蟽蟿蠀蠁蠂蠄蠅]', text, re.IGNORECASE) or \
                   (len(text) <= 20 and re.search(r'[+\-*/^_{}\[\]()]', text) and not re.search(r'[A-Z]{3,}', text)):
                    # This is likely a formula, not a heading - convert to math
                    out.append(f"$$\n{text}\n$$")
                    continue
                
                # Fix skipped levels (e.g., H1 -> H3 should become H1 -> H2)
                if level > heading_stack[-1] + 1:
                    # Skip was detected, fix it
                    level = heading_stack[-1] + 1
                
                # Update stack
                while len(heading_stack) > 0 and heading_stack[-1] >= level:
                    heading_stack.pop()
                heading_stack.append(level)
                
                # Ensure heading text is reasonable
                if len(text) > 200:
                    # Very long heading - might be body text
                    out.append(text)
                else:
                    out.append("#" * level + " " + text)
            else:
                out.append(line)
        
        return "\n".join(out)

    def _is_likely_formula(self, text: str) -> bool:
        """Check if text is likely a formula (not regular text)."""
        if not text or len(text.strip()) < 2:
            return False
        
        text = text.strip()
        
        # If it's wrapped in $$ or $, it's definitely a formula
        if (text.startswith('$$') and text.endswith('$$')) or \
           (text.startswith('$') and text.endswith('$') and text.count('$') == 2):
            return True
        
        # Check for common formula patterns
        formula_indicators = [
            r'\\[a-zA-Z]+\{',  # LaTeX commands like \frac{, \sum_{
            r'[a-zA-Z]_\{[^}]+\}',  # Subscripts like x_{i}
            r'[a-zA-Z]\^\{[^}]+\}',  # Superscripts like x^{2}
            r'\\[a-zA-Z]+',  # LaTeX commands like \alpha, \beta
            r'[伪尾纬未蔚味畏胃喂魏位渭谓尉慰蟺蟻蟽蟿蠀蠁蠂蠄蠅]',  # Greek letters
            r'[鈭戔埆鈭忊垶鈮も墺鈮犫増鈭堚垑鈯傗埅鈭',  # Math symbols
        ]
        
        has_formula_pattern = any(re.search(pattern, text) for pattern in formula_indicators)
        
        # Check if it's likely regular text (has common words, punctuation)
        text_indicators = [
            r'\b(the|and|or|is|are|was|were|in|on|at|to|for|of|with|by)\b',  # Common words
            r'[.!?,;:]',  # Punctuation
            r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # Capitalized words (names)
        ]
        
        has_text_pattern = any(re.search(pattern, text, re.IGNORECASE) for pattern in text_indicators)
        
        # If it has formula patterns but no text patterns, it's likely a formula
        if has_formula_pattern and not has_text_pattern:
            return True
        
        # If it's very short and has math-like characters, might be formula
        if len(text) < 20 and re.search(r'[+\-*/^_{}\[\]()=]', text) and not has_text_pattern:
            return True
        
        return False
    
    def _fix_vision_formula_errors(self, md: str) -> str:
        """
        Fix common formula errors from vision model output:
        - Missing subscripts: \alphaj -> \alpha_j
        - Missing superscripts: x2 -> x^2
        - Split formulas (merged back together)
        - Prime symbols with subscripts: G' i -> G'_i
        - Unrendered symbols (鈻? etc.)
        - Formatting issues
        - Chinese text mixed in formulas
        - Table formatting
        - Remove incorrectly identified formulas (text that was marked as formula)
        """
        lines = md.splitlines()
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check for incorrectly identified formulas (text wrapped in $$ that shouldn't be)
            if stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 4:
                formula_content = stripped[2:-2].strip()
                # Check if it's actually text, not a formula
                if not self._is_likely_formula(formula_content):
                    # This is text, not a formula - remove the $$ wrapper
                    result.append(formula_content)
                    i += 1
                    continue
            
            # Merge split display formulas ($$...$$ that were broken across lines)
            if stripped.startswith('$$') and not stripped.endswith('$$'):
                # Start of a split formula - collect until we find the closing $$
                formula_parts = [line]
                i += 1
                found_closing = False
                while i < len(lines):
                    next_line = lines[i]
                    formula_parts.append(next_line)
                    if next_line.strip().endswith('$$'):
                        found_closing = True
                        break
                    i += 1
                
                if found_closing:
                    # Merge into single line
                    merged = ' '.join(p.strip() for p in formula_parts)
                    # Verify it's actually a formula before keeping it
                    formula_content = merged.replace('$$', '').strip()
                    if self._is_likely_formula(formula_content):
                        result.append(merged)
                    else:
                        # It's text, not a formula - remove $$ wrapper
                        result.append(formula_content)
                else:
                    # Never found closing $$ - might be incorrectly identified
                    # Check if it's actually text
                    formula_content = ' '.join(p.strip() for p in formula_parts).replace('$$', '').strip()
                    if self._is_likely_formula(formula_content):
                        result.append(' '.join(p.strip() for p in formula_parts))
                    else:
                        result.append(formula_content)
                i += 1
                continue
            
            # Only process lines that contain formulas
            if '$' in line:
                # Check for incorrectly identified inline formulas
                # Pattern: $text$ where text is not actually a formula
                inline_formula_pattern = r'\$([^$]+)\$'
                matches = list(re.finditer(inline_formula_pattern, line))
                for match in reversed(matches):  # Process in reverse to maintain positions
                    formula_content = match.group(1)
                    if not self._is_likely_formula(formula_content):
                        # This is text, not a formula - remove $ wrapper
                        line = line[:match.start()] + formula_content + line[match.end():]
                
                # Now process actual formulas
                if '$' in line:  # Check again after removing false positives
                    # Fix prime symbols with subscripts: G' i -> G'_i, G'_{low} -> G'_{low} (already correct)
                    line = re.sub(r"([A-Za-z])'\s+([a-z])(?![_^{])", r"\1'_{\2}", line)
                    line = re.sub(r"([A-Za-z])'\s+_\{([^}]+)\}", r"\1'_{\2}", line)
                    
                    # Fix missing subscripts in Greek letters: \alphaj -> \alpha_j
                    line = re.sub(r'\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|chi|psi|omega)([a-z])(?![_^{])', r'\\\1_{\2}', line)
                    
                    # Fix missing subscripts: \partial c j -> \partial c_j
                    line = re.sub(r'\\(partial)\s+([a-z])\s+([a-z])(?![_^{])', r'\\\1 \2_{\3}', line)
                    
                    # Fix: \alphaj (no backslash before j) -> \alpha_j
                    line = re.sub(r'\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|chi|psi|omega)([a-z])(?![_^{\\])', r'\\\1_{\2}', line)
                    
                    # Remove Chinese text from inside formulas
                    line = re.sub(r'(\$[^$]*)[鍏紡]+([^$]*\$)', r'\1\2', line)
                    line = re.sub(r'(\$\$[^$]*)[鍏紡]+([^$]*\$\$)', r'\1\2', line)
                    
                    # Remove unrendered box symbols (鈻? from formulas
                    line = re.sub(r'(\$[^$]*)[鈻+([^$]*\$)', r'\1\2', line)
                    line = re.sub(r'(\$\$[^$]*)[鈻+([^$]*\$\$)', r'\1\2', line)
                    
                    # Fix spaces before subscripts/superscripts: x _i -> x_i, x ^2 -> x^2
                    line = re.sub(r'([a-zA-Z])\s+_(\w)', r'\1_\2', line)
                    line = re.sub(r'([a-zA-Z])\s+\^(\w)', r'\1^\2', line)
                    
                    # Fix inline formulas that were split: $... $ ...$ -> $... ...$
                    if line.count('$') >= 2 and line.count('$') % 2 == 0:
                        # Try to merge split inline formulas on the same line
                        line = re.sub(r'\$\s+([^$]+)\s+\$', r'$ \1 $', line)
                    
                    # Merge adjacent inline formulas that should be one
                    # Pattern: $formula1$ $formula2$ -> $formula1 formula2$ (if they're related)
                    line = re.sub(r'\$([^$]+)\$\s+\$([^$]+)\$', lambda m: 
                        f'${m.group(1)} {m.group(2)}$' if self._is_likely_formula(m.group(1) + ' ' + m.group(2)) 
                        else f'${m.group(1)}$ ${m.group(2)}$', line)
            
            # Fix table formatting: ensure proper separator and alignment
            if '|' in line and line.strip().startswith('|'):
                # Check if this looks like a table row
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if len(cells) >= 2:
                    # Ensure proper table format
                    if '---' not in line and i + 1 < len(lines):
                        # Check if next line is separator
                        next_line = lines[i + 1].strip()
                        if '|' in next_line and '---' not in next_line:
                            # Missing separator - insert one
                            sep = '| ' + ' | '.join(['---'] * len(cells)) + ' |'
                            result.append(line)
                            result.append(sep)
                            i += 1
                            continue
            
            result.append(line)
            i += 1
        
        return "\n".join(result)

    def _fix_references_format(self, md: str) -> str:
        """
        Fix references section formatting:
        - Remove formula blocks ($$...$$) and code blocks (```...```) from references
        - Ensure each reference is on a separate line
        - Ensure references are numbered (add numbers if missing)
        - Convert formulas in references to plain text
        """
        lines = md.splitlines()
        result = []
        in_references = False
        ref_start_idx = None
        ref_lines = []
        
        # Find References section
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check if this is a References heading
            if re.match(r'^#+\s+References?\s*$', stripped, re.IGNORECASE) or \
               re.match(r'^References?\s*$', stripped, re.IGNORECASE):
                in_references = True
                ref_start_idx = i
                result.append(line)  # Keep the heading
                continue
            
            if in_references:
                # Check if we've reached the end of references (new major heading)
                if stripped.startswith('#') and not re.match(r'^#+\s+References?\s*$', stripped, re.IGNORECASE):
                    # Check if it's a major section (not just a subheading in references)
                    heading_level = len(stripped) - len(stripped.lstrip('#'))
                    if heading_level <= 2:  # H1 or H2 - likely end of references
                        # Process collected references
                        result.extend(self._format_references_block(ref_lines))
                        ref_lines = []
                        in_references = False
                        result.append(line)  # Add the new heading
                        continue
                
                # Collect reference lines
                ref_lines.append((i, line))
            else:
                result.append(line)
        
        # Process any remaining references
        if ref_lines:
            result.extend(self._format_references_block(ref_lines))
        
        return "\n".join(result)
    
    def _format_references_block(self, ref_lines: list[tuple[int, str]]) -> list[str]:
        """Format a block of reference lines - ensure each reference is on a separate line with numbering."""
        formatted = []
        current_ref = []
        ref_num = 1
        in_code_block = False
        in_display_math = False
        
        for i, line in ref_lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                if current_ref:
                    # End of current reference - finish it
                    ref_text = ' '.join(current_ref).strip()
                    if ref_text:
                        formatted.append(self._format_single_reference(ref_text, ref_num))
                        ref_num += 1
                    current_ref = []
                continue
            
            # Remove code blocks completely (including multi-line fenced blocks)
            if stripped.startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            # Remove multi-line display-math blocks inside references.
            if stripped == '$$':
                in_display_math = not in_display_math
                continue
            if in_display_math:
                plain_math = self._formula_to_plain_text(stripped)
                if plain_math:
                    current_ref.append(plain_math)
                continue
            
            # Remove display math blocks ($$...$$) - convert to text
            if stripped.startswith('$$'):
                if stripped.endswith('$$') and len(stripped) > 4:
                    # Single-line formula - convert to text
                    formula_text = stripped[2:-2].strip()
                    plain_text = self._formula_to_plain_text(formula_text)
                    if plain_text:
                        current_ref.append(plain_text)
                # Multi-line formulas - skip the opening $$, collect content until closing $$
                continue
            
            # Whole-line inline math wrappers: `$...$` should be plain text in references.
            if stripped.startswith('$') and stripped.endswith('$') and stripped.count('$') == 2 and len(stripped) > 2:
                stripped = self._formula_to_plain_text(stripped[1:-1].strip())
                if not stripped:
                    continue
            
            # Remove inline math ($...$) but keep the content as text
            if '$' in stripped:
                # Replace $...$ with plain text
                stripped = re.sub(r'\$([^$]+)\$', lambda m: self._formula_to_plain_text(m.group(1)), stripped)
                stripped = re.sub(r'\$\$([^$]+)\$\$', lambda m: self._formula_to_plain_text(m.group(1)), stripped)
                stripped = stripped.replace('$', '').strip()
            
            # Check if this line starts a new reference
            # Patterns: [1] text, [1]. text, 1. text, 1] text
            ref_match = re.match(r'^(\[?\d+\]?)[\.\s]+(.+)$', stripped)
            if ref_match:
                # This starts a new reference
                if current_ref:
                    # Finish the previous reference first
                    ref_text = ' '.join(current_ref).strip()
                    if ref_text:
                        formatted.append(self._format_single_reference(ref_text, ref_num))
                        ref_num += 1
                    current_ref = []
                
                # Start new reference
                ref_content = ref_match.group(2).strip()
                if ref_content:
                    current_ref.append(ref_content)
            else:
                # Check if line contains a reference number pattern in the middle (e.g., "text [2] more text")
                # This might indicate a new reference started
                mid_ref_match = re.search(r'\s+(\[?\d+\]?)[\.\s]+([A-Z][^.]{10,})', stripped)
                if mid_ref_match and current_ref:
                    # Likely a new reference started mid-line
                    # Finish previous reference
                    before_ref = stripped[:mid_ref_match.start()].strip()
                    if before_ref:
                        current_ref.append(before_ref)
                    ref_text = ' '.join(current_ref).strip()
                    if ref_text:
                        formatted.append(self._format_single_reference(ref_text, ref_num))
                        ref_num += 1
                    
                    # Start new reference
                    current_ref = [mid_ref_match.group(2).strip()]
                else:
                    # Continuation of current reference
                    current_ref.append(stripped)
        
        # Add the last reference
        if current_ref:
            ref_text = ' '.join(current_ref).strip()
            if ref_text:
                formatted.append(self._format_single_reference(ref_text, ref_num))
        
        return formatted
    
    def _format_single_reference(self, text: str, num: int) -> str:
        """Format a single reference with proper numbering."""
        # Clean up the text
        text = text.strip()
        
        # Remove any remaining math notation
        text = re.sub(r'\$([^$]+)\$', lambda m: self._formula_to_plain_text(m.group(1)), text)
        text = re.sub(r'\$\$([^$]+)\$\$', lambda m: self._formula_to_plain_text(m.group(1)), text)
        
        # Check if it already has a number
        if re.match(r'^\[?\d+\]?\s+', text):
            # Already numbered, just ensure proper format
            text = re.sub(r'^\[?(\d+)\]?\s+', r'[\1] ', text)
            return text
        
        # Add number if missing
        return f"[{num}] {text}"
    
    def _formula_to_plain_text(self, formula: str) -> str:
        """Convert LaTeX formula to plain text for references."""
        if not formula:
            return ""
        
        # Remove LaTeX commands but keep the content
        text = formula
        
        # Convert subscripts: x_i -> x i or xi
        text = re.sub(r'_\{([^}]+)\}', r' \1', text)
        text = re.sub(r'_([a-z0-9])', r' \1', text)
        
        # Convert superscripts: x^2 -> x2
        text = re.sub(r'\^\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\^([a-z0-9])', r'\1', text)
        
        # Remove LaTeX commands but keep Greek letter names
        text = re.sub(r'\\alpha', 'alpha', text)
        text = re.sub(r'\\beta', 'beta', text)
        text = re.sub(r'\\gamma', 'gamma', text)
        text = re.sub(r'\\delta', 'delta', text)
        text = re.sub(r'\\[a-z]+\{([^}]+)\}', r'\1', text)  # \command{content} -> content
        text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove remaining commands
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _llm_fix_misclassified_headings(self, md: str) -> str:
        """Fix headings that were misclassified as math formulas, and remove non-headings."""
        lines = md.splitlines()
        fixed_lines = []
        seen_headings = set()
        heading_stack = []
        
        print("Checking for misclassified headings...")
        fixed_count = 0
        removed_count = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check if this line looks like a misclassified heading (starts and ends with $, single line)
            if stripped.startswith('$') and stripped.endswith('$'):
                # Count $ symbols - should be exactly 2 (one at start, one at end)
                dollar_count = stripped.count('$')
                if dollar_count == 2:
                    text_content = stripped[1:-1].strip()
                    
                    # First, use aggressive heuristic check for obvious headings
                    is_heading = False
                    level = 1
                    
                    # Pattern: "1. Introduction", "2. Related Work", etc.
                    match1 = re.match(r'^(\d+)\.\s+(.+)$', text_content)
                    if match1:
                        is_heading = True
                        level = 1
                        heading_text = text_content
                    
                    # Pattern: "3.1. Background", "4.1. Experimental Setup", etc.
                    match2 = re.match(r'^(\d+)\.(\d+)\.\s+(.+)$', text_content)
                    if match2:
                        is_heading = True
                        level = 2
                        heading_text = text_content
                    
                    # Pattern: "3.1.1. Details", etc.
                    match3 = re.match(r'^(\d+)\.(\d+)\.(\d+)\.\s+(.+)$', text_content)
                    if match3:
                        is_heading = True
                        level = 3
                        heading_text = text_content
                    
                    # Pattern: "A. Appendix", etc.
                    match4 = re.match(r'^([A-Z])\.\s+(.+)$', text_content)
                    if match4:
                        is_heading = True
                        level = 2
                        heading_text = text_content
                    
                    # If heuristic says it's a heading, use LLM to confirm and refine
                    if is_heading:
                        if self.cfg.llm and hasattr(self.llm_worker, '_client') and self.llm_worker._client:
                            try:
                                llm_result = self.llm_worker.call_llm_confirm_and_level_heading(
                                    heading_text,
                                    page_number=0
                                )
                                if llm_result and llm_result.get('is_heading'):
                                    heading_text = llm_result.get('text', heading_text)
                                    level = llm_result.get('level', level)
                            except Exception:
                                pass  # Use heuristic level if LLM fails
                        
                        # Normalize for duplicate check
                        normalized = re.sub(r'^[IVX]+\.\s*', '', heading_text, flags=re.IGNORECASE).strip()
                        normalized = re.sub(r'^[A-Z]\.\s*', '', normalized).strip()
                        normalized = re.sub(r'^\d+(?:\.\d+)*\.?\s*', '', normalized).strip()
                        
                        if normalized not in seen_headings:
                            # Update heading stack
                            while heading_stack and heading_stack[-1] >= level:
                                heading_stack.pop()
                            heading_stack.append(level)
                            seen_headings.add(normalized)
                            fixed_lines.append("#" * level + " " + heading_text)
                            fixed_count += 1
                            continue
            
            # Check if this is a heading that should be removed (author/affiliation lines)
            if stripped.startswith('#'):
                heading_match = re.match(r'^#+\s+(.+)$', stripped)
                if heading_match:
                    heading_text = heading_match.group(1).strip()
                    # Check if it's an author/affiliation line
                    is_author_line = (
                        '@' in heading_text or
                        re.search(r'\b(?:university|dept|department|institute|email|zhejiang|westlake)\b', heading_text, re.IGNORECASE) or
                        re.search(r'^\w+\s+\w+.*\d+.*\d+', heading_text)  # "Name 1, 2 Name 2" pattern
                    )
                    
                    if is_author_line:
                        # Remove heading markdown, keep as regular text
                        fixed_lines.append(heading_text)
                        removed_count += 1
                        continue
            
            # Not a misclassified heading, keep original line
            fixed_lines.append(line)
        
        if fixed_count > 0:
            print(f"Fixed {fixed_count} misclassified headings")
        if removed_count > 0:
            print(f"Removed {removed_count} non-heading titles")
        
        return "\n".join(fixed_lines)

    def _llm_fix_inline_formulas(self, md: str) -> str:
        """Fix inline formulas with batch LLM processing for speed."""
        if not self.cfg.llm:
            return md
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client for inline formulas: {e}")
                return md
        
        # Collect all inline formulas first
        import re
        pattern = r'\$([^$]+)\$'
        all_formulas = []
        formula_positions = []  # List of (line_idx, match_start, match_end, formula_text)
        
        lines = md.splitlines()
        for line_idx, line in enumerate(lines):
            if '$' in line and not line.strip().startswith('$$'):
                matches = list(re.finditer(pattern, line))  # Use original line, not stripped
                for match in matches:
                    formula_text = match.group(1)
                    # Skip if it's a display formula marker
                    if formula_text.startswith('$') or formula_text.endswith('$'):
                        continue
                    # Skip if it looks like already correct LaTeX (has backslashes and no garbled chars)
                    # But include if it has garbled chars like 藛
                    has_garbled = any(c in formula_text for c in ['藛', ' ']) and '\\' not in formula_text
                    if '\\' in formula_text and '\\hat' in formula_text and not has_garbled:
                        continue
                    all_formulas.append(formula_text)
                    formula_positions.append((line_idx, match.start(), match.end(), formula_text))
        
        if not all_formulas:
            return md
        
        print(f"Fixing {len(all_formulas)} inline formulas in batch...")
        
        # Batch process ALL formulas at once for speed
        fixed_formulas = {}
        try:
            prompt = f"""Fix these {len(all_formulas)} inline math expressions to proper LaTeX. Return a JSON array with the fixed formulas.

CRITICAL FIXES for each formula:
1. Fix garbled Unicode: "藛" -> "\\hat", "C ( r )" -> "C(r)", "C ( r ) 2" -> "C(r)^2"
2. Use proper LaTeX: subscripts x_i, superscripts x^2, functions \\hat{{C}}
3. Remove ALL extra spaces between symbols
4. Group properly with braces

Input formulas:
{json.dumps(all_formulas, ensure_ascii=False)}

Return JSON array with fixed formulas in the same order, e.g.:
["\\hat{{C}}(r) - C(r)^2", "x_1 + x_2", ...]

Return ONLY the JSON array, no other text:"""
            
            resp = self.llm_worker._llm_create(
                messages=[
                    {"role": "system", "content": "You are a LaTeX math expert. Return only valid JSON array with fixed formulas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=3000,
            )
            result_text = (resp.choices[0].message.content or "").strip()
            # Remove markdown code fences
            if result_text.startswith("```"):
                result_text = re.sub(r'^```(?:\w+)?\n?', '', result_text)
                result_text = re.sub(r'\n?```$', '', result_text)
            
            fixed_batch = json.loads(result_text)
            
            # Map back to original formulas
            for i, fixed_formula in enumerate(fixed_batch):
                if i < len(all_formulas):
                    original = all_formulas[i]
                    # Remove $ if present
                    fixed = str(fixed_formula).strip()
                    if fixed.startswith("$") and fixed.endswith("$"):
                        fixed = fixed[1:-1].strip()
                    elif fixed.startswith("$$") and fixed.endswith("$$"):
                        fixed = fixed[2:-2].strip()
                    # Additional cleanup for common issues
                    fixed = fixed.replace('\\hat C', '\\hat{C}').replace('\\hat{ C', '\\hat{C')
                    fixed = re.sub(r'C\s*\(\s*r\s*\)\s*2', r'C(r)^2', fixed)
                    fixed = re.sub(r'C\s*\(\s*r\s*\)', r'C(r)', fixed)
                    fixed = re.sub(r'\s+', ' ', fixed)  # Remove extra spaces
                    fixed_formulas[original] = fixed
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract formulas manually
            print(f"JSON parse failed, trying manual extraction: {e}")
            # Try to extract from text response
            if '[' in result_text and ']' in result_text:
                try:
                    # Try to find JSON array in the response
                    start = result_text.find('[')
                    end = result_text.rfind(']') + 1
                    if start >= 0 and end > start:
                        json_text = result_text[start:end]
                        fixed_batch = json.loads(json_text)
                        for i, fixed_formula in enumerate(fixed_batch):
                            if i < len(all_formulas):
                                original = all_formulas[i]
                                fixed = str(fixed_formula).strip()
                                if fixed.startswith("$"):
                                    fixed = fixed.strip('$').strip()
                                fixed = fixed.replace('\\hat C', '\\hat{C}')
                                fixed = re.sub(r'C\s*\(\s*r\s*\)\s*2', r'C(r)^2', fixed)
                                fixed = re.sub(r'C\s*\(\s*r\s*\)', r'C(r)', fixed)
                                fixed_formulas[original] = fixed
                except Exception:
                    pass
        except Exception as e:
            # If batch fails, keep originals
            print(f"Failed to fix inline formulas: {e}")
            pass
        
        # Apply fixes to lines - use positions for efficient replacement
        fixed_lines = lines.copy()
        fixed_count = 0
        # Process from end to start to preserve indices
        for line_idx, match_start, match_end, formula_text in reversed(formula_positions):
            if formula_text in fixed_formulas:
                fixed = fixed_formulas[formula_text]
                if fixed and fixed != formula_text:
                    # Replace in the line
                    line = fixed_lines[line_idx]
                    fixed_lines[line_idx] = line[:match_start] + f"${fixed}$" + line[match_end:]
                    fixed_count += 1
        
        if fixed_count > 0:
            print(f"Fixed {fixed_count} inline formulas")
        
        return "\n".join(fixed_lines)

    def _llm_fix_references(self, md: str) -> str:
        """Fix references section formatting with LLM."""
        if not self.cfg.llm:
            return md
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client for references: {e}")
                return md
        
        # Find References section
        lines = md.splitlines()
        ref_start = None
        for i, line in enumerate(lines):
            if re.match(r'^#+\s+References', line, re.IGNORECASE):
                ref_start = i
                break
        
        if ref_start is None:
            return md
        
        # Quick check: if references already formatted (has [1] pattern at start), skip
        # Check the FIRST non-empty line after References
        first_ref_line = None
        for idx in range(ref_start+1, min(ref_start+10, len(lines))):
            line = lines[idx].strip()
            if line:
                first_ref_line = line
                break
        if first_ref_line and re.match(r'^\[\d+\]', first_ref_line):
            # Check if all or most references are formatted (sample check)
            formatted_count = 0
            total_count = 0
            for idx in range(ref_start+1, min(ref_start+30, len(lines))):
                line = lines[idx].strip()
                if line:
                    total_count += 1
                    if re.match(r'^\[\d+\]', line):
                        formatted_count += 1
            # If 80%+ are formatted, skip
            if total_count > 0 and formatted_count / total_count >= 0.8:
                return md  # Already formatted
        
        print("Fixing references formatting...")
        # Extract references section
        ref_lines = lines[ref_start+1:]
        ref_text = "\n".join(ref_lines)
        
        # Process references in larger chunks for speed (reduce LLM calls)
        max_ref_chunk = 20000  # Larger chunks = fewer LLM calls
        if len(ref_text) <= max_ref_chunk:
            # Single chunk - fastest
            formatted_refs = self._llm_format_references_chunk(ref_text)
            if formatted_refs:
                return "\n".join(lines[:ref_start+1]) + "\n\n" + formatted_refs
        else:
            # Multiple chunks - but limit to max 2 chunks for speed
            ref_chunks = []
            # Split into 2 chunks max
            mid_point = len(ref_lines) // 2
            ref_chunks.append("\n".join(ref_lines[:mid_point]))
            ref_chunks.append("\n".join(ref_lines[mid_point:]))
            
            # Process chunks in parallel to reduce tail latency when page conversion is already at 100%.
            formatted_chunks = [None] * len(ref_chunks)
            with ThreadPoolExecutor(max_workers=min(2, len(ref_chunks))) as ex:
                fut_to_idx = {
                    ex.submit(self._llm_format_references_chunk, chunk): i
                    for i, chunk in enumerate(ref_chunks)
                }
                for fut in as_completed(fut_to_idx):
                    i = fut_to_idx[fut]
                    print(f"Processing references chunk {i+1}/{len(ref_chunks)}...", end='\r')
                    try:
                        formatted = fut.result()
                    except Exception:
                        formatted = None
                    formatted_chunks[i] = formatted if formatted else ref_chunks[i]
            print()  # New line
            
            if formatted_chunks:
                return "\n".join(lines[:ref_start+1]) + "\n\n" + "\n\n".join(formatted_chunks)
        
        return md
    
    def _llm_format_references_chunk(self, ref_text: str) -> Optional[str]:
        """Format a chunk of references using LLM."""
        try:
            prompt = f"""Format this references section properly. Each reference should be on its own line, properly formatted.

Requirements:
1. Each reference should start with [number] followed by a space
2. Format should be: [number] Author names. Title. Conference/Journal, pages, year.
3. Fix any garbled text and mojibake (e.g., "Milo拧 Ha拧an" not garbled)
4. Ensure proper spacing and punctuation
5. Keep all citation numbers exactly as they appear
6. Each reference on a separate line
7. Remove any duplicate references
8. Fix special characters properly (e.g., "庐" not garbled)

References section:
{ref_text}

Return ONLY the formatted references, one per line, no explanations:"""
            
            resp = self.llm_worker._llm_create(
                messages=[
                    {"role": "system", "content": "You are an expert at formatting academic references. Return properly formatted references with correct Unicode characters."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=6000,
            )
            formatted_refs = (resp.choices[0].message.content or "").strip()
            # Remove markdown code fences if present
            if formatted_refs.startswith("```"):
                formatted_refs = re.sub(r'^```(?:\w+)?\n?', '', formatted_refs)
                formatted_refs = re.sub(r'\n?```$', '', formatted_refs)
            
            return formatted_refs if formatted_refs else None
        except Exception as e:
            print(f"Failed to format references chunk: {e}")
            return None

    def _llm_fix_tables(self, md: str) -> str:
        """Fix tables formatting with LLM."""
        if not self.cfg.llm:
            return md
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client for tables: {e}")
                return md
        
        lines = md.splitlines()
        fixed_lines = []
        in_table = False
        table_lines = []
        table_start = None
        fixed_table_count = 0
        
        print("Checking for tables to fix...")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check if this is a table line (starts with |)
            if stripped.startswith('|') and '|' in stripped[1:]:
                if not in_table:
                    in_table = True
                    table_start = i
                    table_lines = [line]
                else:
                    table_lines.append(line)
            else:
                # End of table
                if in_table:
                    # Process the table
                    table_text = "\n".join(table_lines)
                    # Quick check: if table already looks good (has --- separator), skip LLM
                    if '---' in table_text and len(table_lines) >= 3:
                        fixed_lines.extend(table_lines)
                    elif len(table_lines) >= 3:  # At least header, separator, one row
                        try:
                            prompt = f"""Fix this Markdown table. Ensure proper formatting, alignment, and fix any garbled text.

Requirements:
1. Ensure all rows have the same number of columns
2. Fix any garbled text or mojibake
3. Ensure proper alignment with | separators
4. Keep header row and separator row (---)
5. Fix any spacing issues

Table:
{table_text}

Return ONLY the fixed table in Markdown format, no explanations:"""
                            
                            resp = self.llm_worker._llm_create(
                                messages=[
                                    {"role": "system", "content": "You are an expert at formatting Markdown tables. Return properly formatted tables."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.0,
                                max_tokens=2000,
                            )
                            fixed_table = (resp.choices[0].message.content or "").strip()
                            # Remove markdown code fences if present
                            if fixed_table.startswith("```"):
                                fixed_table = re.sub(r'^```(?:\w+)?\n?', '', fixed_table)
                                fixed_table = re.sub(r'\n?```$', '', fixed_table)
                            
                            if fixed_table and fixed_table != table_text:
                                fixed_lines.extend(fixed_table.splitlines())
                                fixed_table_count += 1
                            else:
                                fixed_lines.extend(table_lines)
                        except Exception as e:
                            # Keep original if LLM fails
                            fixed_lines.extend(table_lines)
                    else:
                        fixed_lines.extend(table_lines)
                    
                    in_table = False
                    table_lines = []
                
                fixed_lines.append(line)
        
        # Handle table at end of document
        if in_table and table_lines:
            fixed_lines.extend(table_lines)
        
        if fixed_table_count > 0:
            print(f"Fixed {fixed_table_count} tables")
        
        return "\n".join(fixed_lines)
    
    def _llm_fix_tables_with_screenshot(self, md: str, pdf_path: Path, save_dir: Path) -> str:
        """Fix tables formatting with LLM, or screenshot if too difficult."""
        if not self.cfg.llm:
            return self._llm_fix_tables(md)
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client for tables: {e}")
                return md
        
        lines = md.splitlines()
        fixed_lines = []
        in_table = False
        table_lines = []
        table_start = None
        table_caption = None
        fixed_table_count = 0
        screenshot_count = 0
        table_num = 1
        
        print("Checking for tables to fix (with screenshot fallback)...")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check for table caption (before table)
            if i > 0 and (stripped.lower().startswith('table ') or '*Table' in stripped or 'Table' in stripped):
                table_caption = stripped
            # Check if this is a table line (starts with |)
            if stripped.startswith('|') and '|' in stripped[1:]:
                if not in_table:
                    in_table = True
                    table_start = i
                    table_lines = [line]
                else:
                    table_lines.append(line)
            else:
                # End of table
                if in_table:
                    table_text = "\n".join(table_lines)
                    # Quick check: if table already looks good (has --- separator), skip
                    has_separator = any('---' in tl for tl in table_lines)
                    has_multiple_rows = len([l for l in table_lines if l.strip().startswith('|')]) >= 3
                    if has_separator and has_multiple_rows:
                        fixed_lines.extend(table_lines)
                    elif len(table_lines) >= 3:  # At least header, separator, one row
                        try:
                            # Try LLM fix first
                            prompt = f"""Fix this Markdown table. Ensure proper formatting, alignment, and fix any garbled text.

Requirements:
1. Ensure all rows have the same number of columns
2. Fix any garbled text or mojibake
3. Ensure proper alignment with | separators
4. Keep header row and separator row (---)
5. Fix any spacing issues

Table:
{table_text}

Return ONLY the fixed table in Markdown format, no explanations. If the table is too complex or corrupted, return "SCREENSHOT" as the only word:"""
                            
                            resp = self.llm_worker._llm_create(
                                messages=[
                                    {"role": "system", "content": "You are an expert at formatting Markdown tables. Return properly formatted tables or 'SCREENSHOT' if too difficult."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.0,
                                max_tokens=2000,
                            )
                            fixed_table = (resp.choices[0].message.content or "").strip()
                            # Remove markdown code fences if present
                            if fixed_table.startswith("```"):
                                fixed_table = re.sub(r'^```(?:\w+)?\n?', '', fixed_table)
                                fixed_table = re.sub(r'\n?```$', '', fixed_table)
                            
                            # Check if LLM says to screenshot
                            if fixed_table.upper() == "SCREENSHOT" or "SCREENSHOT" in fixed_table.upper():
                                # Screenshot the table from PDF
                                screenshot_path = self._screenshot_table_from_pdf(pdf_path, table_num, save_dir, table_caption)
                                if screenshot_path:
                                    fixed_lines.append(f"![Table {table_num}](./assets/{screenshot_path.name})")
                                    if table_caption:
                                        fixed_lines.append(f"*{table_caption}*")
                                    screenshot_count += 1
                                    table_num += 1
                                else:
                                    # Fallback: keep original
                                    fixed_lines.extend(table_lines)
                            elif fixed_table and fixed_table != table_text:
                                fixed_lines.extend(fixed_table.splitlines())
                                if table_caption:
                                    fixed_lines.append(f"*{table_caption}*")
                                fixed_table_count += 1
                            else:
                                # LLM couldn't fix, try screenshot
                                screenshot_path = self._screenshot_table_from_pdf(pdf_path, table_num, save_dir, table_caption)
                                if screenshot_path:
                                    fixed_lines.append(f"![Table {table_num}](./assets/{screenshot_path.name})")
                                    if table_caption:
                                        fixed_lines.append(f"*{table_caption}*")
                                    screenshot_count += 1
                                    table_num += 1
                                else:
                                    fixed_lines.extend(table_lines)
                        except Exception as e:
                            # LLM failed, try screenshot
                            print(f"LLM table fix failed, trying screenshot: {e}")
                            screenshot_path = self._screenshot_table_from_pdf(pdf_path, table_num, save_dir, table_caption)
                            if screenshot_path:
                                fixed_lines.append(f"![Table {table_num}](./assets/{screenshot_path.name})")
                                if table_caption:
                                    fixed_lines.append(f"*{table_caption}*")
                                screenshot_count += 1
                                table_num += 1
                            else:
                                fixed_lines.extend(table_lines)
                    else:
                        fixed_lines.extend(table_lines)
                    
                    in_table = False
                    table_lines = []
                    table_caption = None
                
                fixed_lines.append(line)
        
        # Add any remaining table lines
        if in_table and table_lines:
            fixed_lines.extend(table_lines)
        
        if fixed_table_count > 0:
            print(f"Fixed {fixed_table_count} tables")
        if screenshot_count > 0:
            print(f"Screenshot {screenshot_count} difficult tables")
        
        return "\n".join(fixed_lines)
    
    def _screenshot_table_from_pdf(self, pdf_path: Path, table_num: int, save_dir: Path, caption: Optional[str] = None) -> Optional[Path]:
        """Screenshot a table from PDF (simplified - would need page detection in real implementation)."""
        if fitz is None:
            return None
        try:
            doc = fitz.open(pdf_path)
            assets_dir = save_dir / "assets"
            assets_dir.mkdir(exist_ok=True)
            
            # For now, just take a screenshot of the first page (in real implementation, would detect table location)
            # This is a placeholder - real implementation would need to detect which page has the table
            if len(doc) > 0:
                page = doc[0]  # Placeholder: use first page
                img_name = f"table_{table_num}.png"
                img_path = assets_dir / img_name
                pix = page.get_pixmap(dpi=self.dpi)
                pix.save(img_path)
                doc.close()
                return img_path
        except Exception as e:
            print(f"Failed to screenshot table: {e}")
        return None
    
    def _llm_fix_references_with_crossref(self, md: str, save_dir: Path) -> str:
        """Fix references formatting with LLM and enrich with Crossref metadata."""
        if not self.cfg.llm:
            return self._llm_fix_references(md)
        
        # First, format references normally
        formatted_md = self._llm_fix_references(md)
        
        # Then, enrich with Crossref metadata
        try:
            from kb.citation_meta import fetch_best_crossref_meta
            
            # Find References section
            lines = formatted_md.splitlines()
            ref_start = None
            for i, line in enumerate(lines):
                if re.match(r'^#+\s+References', line, re.IGNORECASE):
                    ref_start = i
                    break
            
            if ref_start is None:
                return formatted_md
            
            # Extract references and enrich with Crossref
            ref_lines = lines[ref_start+1:]
            enriched_refs = []
            crossref_metadata = {}
            
            print("Enriching references with Crossref metadata...")
            for ref_line in ref_lines[:50]:  # Limit to first 50 for speed
                ref_line = ref_line.strip()
                if not ref_line or not re.match(r'^\[\d+\]', ref_line):
                    enriched_refs.append(ref_line)
                    continue
                
                # Extract title from reference (simplified)
                # Try to extract title between first period and next period or comma
                title_match = re.search(r'\]\s*[^.]*\.\s*([^.,]+(?:\.|,))', ref_line)
                if title_match:
                    title = title_match.group(1).strip().rstrip('.,')
                    if len(title) > 10:  # Reasonable title length
                        try:
                            meta = fetch_best_crossref_meta(query_title=title, min_score=0.85)
                            if meta:
                                ref_num = re.match(r'^\[(\d+)\]', ref_line).group(1)
                                crossref_metadata[ref_num] = meta
                                # Add DOI if available
                                if meta.get('doi'):
                                    ref_line = f"{ref_line} DOI: {meta['doi']}"
                        except Exception:
                            pass
                
                enriched_refs.append(ref_line)
            
            # Save Crossref metadata to JSON file
            if crossref_metadata:
                metadata_file = save_dir / "crossref_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(crossref_metadata, f, indent=2, ensure_ascii=False)
                print(f"Saved Crossref metadata for {len(crossref_metadata)} references to {metadata_file}")
            
            # Reconstruct markdown
            return "\n".join(lines[:ref_start+1]) + "\n\n" + "\n".join(enriched_refs) + "\n\n" + "\n".join(lines[ref_start+1+len(ref_lines):])
        except Exception as e:
            print(f"Crossref enrichment failed: {e}, using formatted references")
            return formatted_md

    def _llm_fix_display_math(self, md: str) -> str:
        """Fix display math blocks - remove nested $ symbols, fix formatting, and add equation numbers."""
        if not self.cfg.llm:
            return md

        # Safety: by default, do NOT ask the LLM to rewrite display math blocks.
        # Some models may "explain" formulas or convert norms (|...|) into Markdown tables.
        # Enable explicitly if you really want it:
        #   KB_PDF_ENABLE_LLM_DISPLAY_MATH_FIX=1
        def _env_bool(name: str, default: bool = False) -> bool:
            try:
                raw = str(os.environ.get(name, "") or "").strip().lower()
                if not raw:
                    return bool(default)
                return raw in {"1", "true", "yes", "y", "on"}
            except Exception:
                return bool(default)

        enable_llm_fix = _env_bool("KB_PDF_ENABLE_LLM_DISPLAY_MATH_FIX", False)
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client for display math: {e}")
                return md
        
        lines = md.splitlines()
        fixed_lines = []
        in_display_math = False
        math_lines = []
        fixed_count = 0
        # NOTE: We do NOT auto-number equations here. PDFs already have numbering and
        # hallucinated renumbering is worse than leaving it as-is.
        
        print("Fixing display math blocks...")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "$$":
                if in_display_math:
                    # End of display math block
                    math_text = "\n".join(math_lines)
                    
                    # Check if there are nested $ symbols or \[ \]
                    if enable_llm_fix and ('$' in math_text or '\\[' in math_text or '\\]' in math_text):
                        try:
                            prompt = f"""Fix this display math block. Remove any nested $ symbols and fix formatting.

Input:
{math_text}

Requirements:
1. Remove ALL $ symbols inside the math block (display math uses $$...$$, no $ inside)
2. Remove any \\[ or \\] symbols (use $$ only for display math)
3. Fix garbled characters (e.g., "藛" -> "\\hat", "C ( r )" -> "C(r)")
4. Use proper LaTeX syntax (e.g., "Z_{{t}} f" -> "\\int_{{t_0}}^{{t_1}}" where t_0 and t_1 are time bounds)
5. Return ONLY the cleaned math content without $$ or \\[ \\] delimiters

LaTeX:"""
                            
                            resp = self.llm_worker._llm_create(
                                messages=[
                                    {"role": "system", "content": "You are a LaTeX math expert. Fix display math blocks by removing nested $ symbols and fixing formatting."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.0,
                                max_tokens=1000,
                            )
                            fixed_math = (resp.choices[0].message.content or "").strip()
                            # Remove $ if present
                            if fixed_math.startswith("$$") and fixed_math.endswith("$$"):
                                fixed_math = fixed_math[2:-2].strip()
                            elif fixed_math.startswith("$") and fixed_math.endswith("$"):
                                fixed_math = fixed_math[1:-1].strip()
                            # Remove markdown code fences
                            if fixed_math.startswith("```"):
                                fixed_math = re.sub(r'^```(?:\w+)?\n?', '', fixed_math)
                                fixed_math = re.sub(r'\n?```$', '', fixed_math)
                            
                            if fixed_math and fixed_math != math_text:
                                # Remove any \[ or \] if present (should use $$ only)
                                fixed_math = fixed_math.replace('\\[', '').replace('\\]', '')
                                fixed_lines.append("$$")
                                fixed_lines.append(fixed_math)
                                fixed_lines.append("$$")
                                fixed_count += 1
                            else:
                                # Just remove nested $ symbols and \[ \] manually
                                cleaned_math = math_text.replace('$', '').replace('\\[', '').replace('\\]', '')
                                fixed_lines.append("$$")
                                fixed_lines.append(cleaned_math)
                                fixed_lines.append("$$")
                        except Exception:
                            # Fallback: just remove nested $ symbols and \[ \]
                            cleaned_math = math_text.replace('$', '').replace('\\[', '').replace('\\]', '')
                            fixed_lines.append("$$")
                            fixed_lines.append(cleaned_math)
                            fixed_lines.append("$$")
                    else:
                        # Clean up: remove empty lines and fix formatting
                        cleaned_lines = [l for l in math_lines if l.strip()]
                        if cleaned_lines:
                            # Basic cleanup only: remove nested delimiters if they leaked into the block.
                            cleaned_lines = [ln.replace('$', '').replace('\\[', '').replace('\\]', '') for ln in cleaned_lines]
                            # Only add $$ if we actually have content
                            fixed_lines.append("$$")
                            fixed_lines.extend(cleaned_lines)
                            fixed_lines.append("$$")
                        else:
                            # Empty math block, skip it - don't add anything
                            pass
                    
                    in_display_math = False
                    math_lines = []
                else:
                    # Start of display math block
                    # Check if next line is also $$ (duplicate)
                    if i + 1 < len(lines) and lines[i + 1].strip() == "$$":
                        # Skip this $$, it's a duplicate - don't add it
                        continue
                    in_display_math = True
                    # Don't add $$ yet, wait for content
            elif in_display_math:
                math_lines.append(line)
            else:
                fixed_lines.append(line)
        
        if fixed_count > 0:
            print(f"Fixed {fixed_count} display math blocks")
        
        return "\n".join(fixed_lines)

    def _llm_light_cleanup(self, md: str) -> str:
        """Light LLM cleanup - only fix remaining mojibake."""
        if not self.cfg.llm:
            return md
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client: {e}")
                return md
        
        # Only process if there are obvious mojibake issues
        mojibake_patterns = ['膹殴', '膸', '脦麓', '脦膮', '芒慕', '芒暮陇', '芒', '藛', '芒暮']
        has_mojibake = any(pattern in md for pattern in mojibake_patterns)
        
        if not has_mojibake:
            return md
        
        # Process in smaller chunks for speed
        max_chunk_size = 12000
        if len(md) <= max_chunk_size:
            return self._llm_cleanup_chunk(md)
        
        lines = md.splitlines()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1
            if current_size + line_size > max_chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        repaired_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"LLM cleanup chunk {i+1}/{len(chunks)}...", end='\r')
            repaired = self._llm_cleanup_chunk(chunk)
            repaired_chunks.append(repaired)
        print()  # New line
        
        return "\n\n".join(repaired_chunks)
    
    def _llm_cleanup_chunk(self, md_chunk: str) -> str:
        """Light cleanup of a chunk - fix mojibake and improve inline formulas."""
        prompt = f"""Fix issues in this Markdown chunk:

1. FIX MOJIBAKE (garbled Unicode):
   - "膹殴" -> "fi"
   - "膸" -> "蟽" or "蟿"
   - "脦麓" -> "未"
   - "脦膮" -> "伪"
   - "芒" -> correct symbol (鈥? 鈭? 鈭? 鈮? 鈮? etc.)
   - "藛" -> "\\hat" (in math context)

2. FIX INLINE FORMULAS (single $...$):
   - Fix garbled math: "藛 C ( r ) - C ( r ) 2" -> "\\hat{{C}}(r) - C(r)^2"
   - Remove extra spaces: "C ( r )" -> "C(r)"
   - Fix subscripts/superscripts: "x 2" -> "x^2" or "x_2"
   - Use proper LaTeX: "藛" -> "\\hat", "伪" -> "\\alpha"

3. PRESERVE:
   - All headings (do NOT change)
   - All display math blocks ($$...$$)
   - All tables, images, code blocks
   - Document structure

Return ONLY the fixed Markdown, no explanations.

INPUT:
{md_chunk}
"""
        
        try:
            resp = self.llm_worker._llm_create(
                messages=[
                    {"role": "system", "content": "You are a mojibake fixer. Only fix garbled characters."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=min(16384, self.cfg.llm.max_tokens),
            )
            result = resp.choices[0].message.content or md_chunk
            if result.startswith("```"):
                result = re.sub(r'^```(?:\w+)?\n', '', result)
                result = re.sub(r'\n```$', '', result)
            return result.strip()
        except Exception as e:
            print(f"LLM cleanup failed: {e}")
            return md_chunk

    def _llm_postprocess_markdown(self, md: str) -> str:
        """Use LLM to fix mojibake, formulas, and structure in the final markdown."""
        if not self.cfg.llm:
            return md
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client for post-processing: {e}")
                return md
        
        # Split into chunks if too long (LLM has token limits)
        max_chunk_size = 8000  # Leave room for prompt
        if len(md) <= max_chunk_size:
            return self._llm_repair_markdown_chunk(md)
        
        # Process in chunks
        lines = md.splitlines()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            if current_size + line_size > max_chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        # Repair each chunk
        repaired_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"LLM post-processing chunk {i+1}/{len(chunks)}...")
            repaired = self._llm_repair_markdown_chunk(chunk)
            repaired_chunks.append(repaired)
        
        return "\n\n".join(repaired_chunks)
    
    def _llm_repair_markdown_chunk(self, md_chunk: str) -> str:
        """Repair a chunk of markdown using LLM."""
        prompt = f"""You are fixing a Markdown document converted from a PDF. The conversion has many errors that need to be fixed.

CRITICAL TASKS:

1. FIX ALL MOJIBAKE/GARBLED TEXT (This is the MOST IMPORTANT):
   - "膹殴" -> "fi" (e.g., "膹殴exible" -> "flexible", "膹殴rst" -> "first", "膹殴xed" -> "fixed")
   - "膸" -> "蟽" (sigma) when in math context, or "蟿" (tau) when appropriate
   - "脦麓" -> "未" (delta)
   - "脦膮" -> "伪" (alpha)
   - "芒" -> various symbols: "芒慕" -> "鈭? (norm), "芒暮陇" -> "鈮? (leq), "芒暮楼" -> "鈮? (geq), "芒" -> "鈥? (em dash), "芒" -> "鈭? (element of)
   - "膫" -> "脳" (times)
   - "臍" -> "鈮? (not equal) or other symbols
   - Fix ALL garbled characters systematically

2. FIX ALL FORMULAS (CRITICAL):
   - Convert ALL Unicode math symbols to LaTeX:
     * 伪, 尾, 纬, 未, 蔚, 胃, 位, 渭, 蟺, 蟽, 蟿, 蠁, 蠅 -> \\alpha, \\beta, \\gamma, \\delta, \\epsilon, \\theta, \\lambda, \\mu, \\pi, \\sigma, \\tau, \\phi, \\omega
     * 鈮?-> \\leq, 鈮?-> \\geq, 鈮?-> \\neq, 鈭?-> \\in, 鈭?-> \\notin
     * 鈭?-> \\sum, 鈭?-> \\prod, 鈭?-> \\int, 鈭?-> \\infty
   - Fix subscripts: "\\tau 1" -> "\\tau_1", "x i" -> "x_i", "I j" -> "I_j"
   - Fix superscripts: "x 2" -> "x^2" when appropriate
   - Display math (block): use $$...$$ for equations that should be centered
   - Inline math: use $...$ for formulas within text
   - Merge split formulas that belong together
   - Fix spacing: "log k" -> "\\log k", "exp" -> "\\exp", etc.
   - Remove equation numbers from inside formulas

3. FIX HEADING STRUCTURE (CRITICAL):
   - "I. INTRODUCTION" -> "## I. INTRODUCTION" (H2, not H1)
   - "II. MAIN RESULT" -> "## II. MAIN RESULT"
   - "III. SEQUENTIAL..." -> "## III. SEQUENTIAL..."
   - "A. Structured..." -> "### A. Structured..."
   - "B. Sensing..." -> "### B. Sensing..."
   - "IV. PROOF..." -> "## IV. PROOF..."
   - "V. ACKNOWLEDGEMENTS" -> "## V. ACKNOWLEDGEMENTS"
   - "APPENDIX" -> "## APPENDIX" (H2)
   - "A. Proof of..." -> "### A. Proof of..." (H3)
   - "REFERENCES" -> "## REFERENCES" (H2)
   - Remove author names from headings (they should be plain text)
   - Remove duplicate headings completely (if you see the same heading twice, keep only the first occurrence)
   - Remove any heading that appears in the middle of a paragraph (headings should be on their own line)
   - Ensure proper hierarchy: H1 for title only, H2 for main sections (I, II, III, IV, V, APPENDIX, REFERENCES), H3 for subsections (A, B, C, etc.)

4. FIX TEXT CONTENT:
   - "芒" -> "鈥? (em dash) in text
   - Fix all ligatures and special characters
   - Preserve mathematical notation in text

5. PRESERVE:
   - All tables (keep markdown table format)
   - All images (keep ![alt](path) format)
   - All code blocks (keep ``` format)
   - All citations [1], [2], etc.
   - All references

6. OUTPUT REQUIREMENTS:
   - Return ONLY the fixed Markdown
   - NO explanations, NO comments, NO code fences
   - Maintain original paragraph structure
   - Keep all content, just fix the errors
   - Remove duplicate headings (if the same heading appears twice, keep only the first occurrence)
   - Ensure headings appear in logical order (I, II, III, IV, V, then APPENDIX, then REFERENCES)

INPUT MARKDOWN (fix all errors):
{md_chunk}
"""
        
        try:
            resp = self.llm_worker._llm_create(
                messages=[
                    {"role": "system", "content": "You are an expert at fixing PDF-to-Markdown conversion errors, especially mojibake, formula formatting, and document structure."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm.max_tokens,
            )
            result = resp.choices[0].message.content or md_chunk
            # Remove any markdown code fences if LLM added them
            if result.startswith("```"):
                result = re.sub(r'^```(?:\w+)?\n', '', result)
                result = re.sub(r'\n```$', '', result)
            return result.strip()
        except Exception as e:
            print(f"LLM post-processing failed: {e}, using original")
            return md_chunk
    
    def _llm_final_quality_check(self, md: str) -> str:
        """Final comprehensive quality check for full_llm mode - ensure everything is perfect."""
        if not self.cfg.llm:
            return md
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client for final quality check: {e}")
                return md
        
        # Process in chunks if too long
        max_chunk_size = 12000
        if len(md) <= max_chunk_size:
            return self._llm_final_quality_check_chunk(md)
        
        # Split into chunks at logical boundaries (sections)
        lines = md.splitlines()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1
            # Start new chunk at major section boundaries (H2 headings)
            if line.strip().startswith('## ') and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            elif current_size + line_size > max_chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        # Process each chunk
        checked_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Final quality check chunk {i+1}/{len(chunks)}...", end='\r')
            checked = self._llm_final_quality_check_chunk(chunk)
            checked_chunks.append(checked)
        print()  # New line
        
        return "\n\n".join(checked_chunks)
    
    def _llm_final_quality_check_chunk(self, md_chunk: str) -> str:
        """Final comprehensive quality check for a chunk - ensure everything is perfect."""
        prompt = f"""You are performing a FINAL COMPREHENSIVE QUALITY CHECK on a Markdown document converted from PDF. 
This is the LAST pass to ensure EVERYTHING is PERFECT. Fix ALL remaining issues.

CRITICAL QUALITY REQUIREMENTS (must be PERFECT):

1. TITLE/HEADING STRUCTURE (MUST BE PERFECT):
   - Ensure proper hierarchy: H1 for title only, H2 for main sections (I, II, III, IV, V, APPENDIX, REFERENCES), H3 for subsections (A, B, C, etc.)
   - Remove any duplicate headings (keep only first occurrence)
   - Remove author names from headings (they should be plain text)
   - Ensure headings are on their own lines (not in middle of paragraphs)
   - Fix any misclassified headings (e.g., headings wrapped in $...$ should be proper markdown headings)
   - Ensure logical order: I, II, III, IV, V, then APPENDIX, then REFERENCES

2. FORMULAS - BOTH INLINE AND DISPLAY (MUST BE PERFECT):
   - Inline formulas ($...$): Must be proper LaTeX, no garbled Unicode, correct subscripts/superscripts
   - Display formulas ($$...$$): Must be proper LaTeX, no nested $ symbols, correct formatting
   - Fix ALL Unicode math symbols to LaTeX (伪 -> \\alpha, 鈮?-> \\leq, etc.)
   - Remove equation numbers from inside formulas (they should be outside or use \\tag)
   - Merge split formulas that belong together
   - Fix spacing: "log k" -> "\\log k", "exp" -> "\\exp", etc.
   - Ensure proper grouping with braces

3. IMAGES (MUST BE PERFECT):
   - All images must have proper markdown syntax: ![Figure](./assets/filename.png)
   - Ensure figure captions are properly formatted (bold "Fig. X." prefix)
   - Remove any duplicate image references
   - Ensure images are on their own lines with blank lines around them

4. TABLES (MUST BE PERFECT):
   - All tables must have proper Markdown format with | separators
   - All rows must have the same number of columns
   - Must have proper header row and separator row (---)
   - Fix any garbled text in tables
   - Ensure proper alignment

5. REFERENCES (MUST BE PERFECT):
   - Each reference must start with [number] followed by a space
   - Format: [number] Author names. Title. Conference/Journal, pages, year.
   - Fix all garbled text and mojibake
   - Ensure proper spacing and punctuation
   - Each reference on a separate line
   - Remove duplicate references

6. BODY TEXT (MUST BE PERFECT):
   - Fix ALL mojibake/garbled text (膹殴 -> fi, 膸 -> 蟽, 脦麓 -> 未, etc.)
   - Fix all ligatures and special characters
   - Ensure proper paragraph structure
   - Preserve mathematical notation in text
   - Fix em dashes: "芒" -> "鈥?

7. OVERALL STRUCTURE:
   - Ensure proper spacing between sections
   - Remove any empty or duplicate content
   - Ensure logical flow

OUTPUT REQUIREMENTS:
- Return ONLY the perfected Markdown
- NO explanations, NO comments, NO code fences
- Maintain all content, just fix errors and improve quality
- Ensure EVERYTHING is perfect - this is the final pass

INPUT MARKDOWN (make it PERFECT):
{md_chunk}
"""
        
        try:
            resp = self.llm_worker._llm_create(
                messages=[
                    {"role": "system", "content": "You are an expert at perfecting PDF-to-Markdown conversions. This is the final quality check - ensure EVERYTHING is perfect."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm.max_tokens,
            )
            result = resp.choices[0].message.content or md_chunk
            # Remove any markdown code fences if LLM added them
            if result.startswith("```"):
                result = re.sub(r'^```(?:\w+)?\n', '', result)
                result = re.sub(r'\n```$', '', result)
            return result.strip()
        except Exception as e:
            print(f"Final quality check failed: {e}, using original")
            return md_chunk
    
    def _get_speed_mode_config(self, speed_mode: str, total_pages: int) -> dict:
        """Get speed-mode defaults tuned for stable VL OCR quality."""
        import multiprocessing
        cpu_count = max(1, int(multiprocessing.cpu_count() or 1))
        total_pages = max(1, int(total_pages or 1))

        # Keep defaults conservative to reduce timeout/fallback cascades.
        normal_parallel = min(total_pages, max(2, min(10, cpu_count)))
        normal_inflight = max(2, min(10, cpu_count))
        ultra_parallel = min(total_pages, max(2, min(14, cpu_count * 2)))
        ultra_inflight = max(2, min(14, cpu_count * 2))

        configs = {
            'normal': {
                'max_parallel_pages': normal_parallel,
                'max_inflight': normal_inflight,
                'dpi': 220,
                'compress': 2,
                'max_tokens': 4096,
            },
            'ultra_fast': {
                'max_parallel_pages': ultra_parallel,
                'max_inflight': ultra_inflight,
                'dpi': 150,
                'compress': 5,
                'max_tokens': 2048,
            },
            'no_llm': {
                'max_parallel_pages': min(8, cpu_count, total_pages),
                'max_inflight': 1,
                'dpi': 200,
                'compress': 0,
                'max_tokens': 0,
            }
        }

        return configs.get(speed_mode, configs['normal'])

