from __future__ import annotations

import os
import shutil
import re
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple

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
    build_repeated_noise_texts,
    _pick_render_scale,
    _is_frontmatter_noise_line,
    _pick_column_range,
    _detect_column_split_x,
)
from .geometry_utils import _bbox_width
from .heuristics import (
    _suggest_heading_level,
    _is_non_body_metadata_text,
    _looks_like_author_name_line,
)
from .tables import _is_markdown_table_sane, table_text_to_markdown
from .block_classifier import _looks_like_math_block, _looks_like_code_block
from .llm_worker import LLMWorker
from .post_processing import postprocess_markdown
from .md_analyzer import MarkdownAnalyzer
from .pipeline_vision_direct import process_batch_vision_direct
from .pipeline_render_markdown import render_blocks_to_markdown
from .page_asset_cleanup import cleanup_stale_page_assets, cleanup_unreferenced_assets
from .page_image_markdown import (
    cleanup_page_local_image_markdown,
    extract_figure_number_from_text,
    figure_remap_debug_enabled,
    inject_missing_page_image_links,
    inject_page_image_captions_from_meta,
    remap_page_image_links_by_caption,
    repair_broken_image_links,
    reorder_page_figure_pairs_by_number,
    normalize_page_image_caption_order,
    normalize_page_local_image_link_order,
)
from .page_figure_metadata import (
    extract_page_figure_caption_candidates,
    match_figure_entries_with_captions,
    split_visual_rects_by_internal_captions,
    persist_page_figure_metadata,
    build_page_figure_mapping_hint,
    reconcile_figure_metadata_from_markdown,
)
from .structured_indices import rebuild_structured_indices_for_markdown
from .page_table_fallback import detect_table_rects_for_fallback, inject_table_image_fallbacks
from .page_layout_crops import (
    convert_page_with_layout_crops,
    expand_structured_crop_rect,
    fallback_markdown_from_blocks,
    is_probable_page_number_block,
    structured_crop_lane_for_bbox,
)
from .page_local_pipeline import prepare_page_render_input, process_page, render_prepared_page
from .page_text_blocks import extract_text_blocks
from .page_vision_guardrails import convert_page_with_vision_guardrails
from .reference_page_vl import (
    is_reference_placeholder_line,
    sanitize_reference_crop_markdown,
    merge_reference_crop_markdowns,
    build_reference_column_crop_rects,
    convert_references_page_with_column_vl,
)
from .formula_markdown import (
    merge_split_formulas,
    merge_fragmented_formulas,
    is_formula_fragment,
    looks_like_formula_continuation,
    basic_formula_cleanup,
    is_likely_formula,
    fix_vision_formula_errors,
    formula_to_plain_text,
)
from .reference_markdown import (
    fix_references_format,
    format_references_block,
    format_single_reference,
)
from .heading_markdown import fix_heading_structure
from .llm_math_cleanup import llm_fix_inline_formulas, llm_fix_display_math
from .llm_reference_table_cleanup import (
    llm_fix_references,
    llm_polish_references,
    llm_format_references_chunk,
    llm_fix_tables,
    llm_fix_tables_with_screenshot,
)
from .llm_general_cleanup import (
    llm_light_cleanup,
    llm_cleanup_chunk,
    llm_postprocess_markdown,
    llm_repair_markdown_chunk,
    llm_final_quality_check,
    llm_final_quality_check_chunk,
)


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
        
        # Only use vision-direct mode: screenshot -> VL -> Markdown
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
                print(f"[MODE] Vision-direct ({mode_name}): each page screenshot -> VL model -> Markdown", flush=True)
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
        if self._llm_reference_polish_enabled():
            try:
                final_md = self._llm_polish_references(final_md)
            except Exception as e:
                print(f"[WARN] _llm_polish_references failed, keep previous markdown: {e}", flush=True)
        try:
            # Run one final text-level normalization after image/caption/title repairs.
            # Several repair stages can legitimately reinsert blank lines or raw caption
            # text patterns; a final postprocess pass keeps the saved markdown stable.
            final_md = postprocess_markdown(final_md)
        except Exception as e:
            print(f"[WARN] final postprocess_markdown failed, keep current markdown: {e}", flush=True)
        try:
            self._cleanup_unreferenced_assets(final_md, assets_dir=assets_dir)
        except Exception as e:
            print(f"[WARN] unreferenced asset cleanup skipped: {e}", flush=True)
        try:
            self._reconcile_figure_metadata_from_markdown(md=final_md, assets_dir=assets_dir)
        except Exception as e:
            print(f"[WARN] figure metadata reconciliation skipped: {e}", flush=True)
        
        # Write output
        out_file = save_dir / "output.md"
        out_file.write_text(final_md, encoding="utf-8")
        try:
            rebuild_structured_indices_for_markdown(out_file, md_text=final_md, assets_dir=assets_dir)
        except Exception as e:
            print(f"[WARN] structured index build skipped: {e}", flush=True)
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
        cleanup_stale_page_assets(cfg=self.cfg, assets_dir=assets_dir, total_pages=total_pages)

    def _cleanup_unreferenced_assets(self, md: str, *, assets_dir: Path) -> None:
        cleanup_unreferenced_assets(md, assets_dir=assets_dir)

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
            line_items: list[tuple[fitz.Rect, str, float]] = []
            max_size = 0.0
            for l in (b.get("lines", []) or []):
                spans = l.get("spans", []) or []
                parts: list[str] = []
                line_max_size = 0.0
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
                    if sz > line_max_size:
                        line_max_size = sz
                if parts:
                    line_text = " ".join(parts).strip()
                    lines.append(line_text)
                    try:
                        line_rect = fitz.Rect(l.get("bbox"))
                    except Exception:
                        line_rect = fitz.Rect(rect)
                    line_items.append((line_rect, line_text, float(line_max_size or max_size or 0.0)))
            if not lines:
                continue
            txt = _normalize_text(" ".join(lines)).strip()
            if not txt:
                continue

            block_is_metadata = _is_non_body_metadata_text(
                txt,
                page_index=page_index,
                y0=float(rect.y0),
                y1=float(rect.y1),
                page_height=page_h,
                max_font_size=float(max_size),
                body_font_size=float(body_size_hint),
                is_references_page=bool(is_references_page),
            )
            matched_line_rects: list[fitz.Rect] = []
            for line_rect, line_text, line_size in line_items:
                if _is_non_body_metadata_text(
                    line_text,
                    page_index=page_index,
                    y0=float(line_rect.y0),
                    y1=float(line_rect.y1),
                    page_height=page_h,
                    max_font_size=float(line_size),
                    body_font_size=float(body_size_hint),
                    is_references_page=bool(is_references_page),
                ) or _looks_like_author_name_line(
                    line_text,
                    page_index=page_index,
                    y0=float(line_rect.y0),
                    page_height=page_h,
                ):
                    matched_line_rects.append(line_rect)
            if matched_line_rects:
                rects.extend(matched_line_rects)
            elif block_is_metadata and line_items:
                rects.extend(line_rect for line_rect, _, _ in line_items)
            elif block_is_metadata:
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
        if re.match(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)\.?\s+[A-Za-z]", t, flags=re.IGNORECASE):
            return False
        if re.fullmatch(r"\(?[a-z]\)?", t, flags=re.IGNORECASE):
            return True
        if len(t) <= 24 and re.fullmatch(r"[\d\.\-\+\s,;:()/%]+", t):
            return True
        if len(t) <= 32 and re.search(r"\d", t):
            return True
        if len(t) <= 32 and re.search(r"\b(?:wavelength|intensity|frequency|time|pixel|nm|hz|a\.?u\.?|x|y)\b", low):
            return True
        return False

    def _collect_page_text_line_boxes(self, page) -> list[tuple[fitz.Rect, str, float, bool]]:
        if fitz is None:
            return []
        out: list[tuple[fitz.Rect, str, float, bool]] = []
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
                max_font = 0.0
                is_bold = False
                for s in spans:
                    try:
                        max_font = max(max_font, float(s.get("size", 0.0) or 0.0))
                    except Exception:
                        pass
                    try:
                        font_name = str(s.get("font", "") or "").lower()
                        if "bold" in font_name or "black" in font_name or "heavy" in font_name:
                            is_bold = True
                    except Exception:
                        pass
                    try:
                        flags = int(s.get("flags", 0) or 0)
                        if flags & 16:
                            is_bold = True
                    except Exception:
                        pass
                out.append((r, txt, max_font, is_bold))
        return out

    def _expanded_visual_crop_rect(
        self,
        *,
        rect: fitz.Rect,
        page_w: float,
        page_h: float,
        is_full_width: bool,
        line_boxes: list[tuple] | None = None,
    ) -> fitz.Rect:
        """
        Expand figure crops conservatively, with extra bottom room for axes/ticks/panel labels,
        while avoiding accidental bleed into nearby body paragraphs.
        """
        r = fitz.Rect(rect)
        if r.width <= 0.0 or r.height <= 0.0:
            return r

        pad_x = max(2.0, float(page_w) * (0.004 if not is_full_width else 0.003))
        # Keep a more generous top pad: figure-local titles often sit slightly
        # above the detected visual bbox, while body text above can be clipped
        # back using the boundary check below.
        pad_top = max(8.0, float(page_h) * (0.014 if not is_full_width else 0.012))
        pad_bottom = max(8.0, float(page_h) * (0.024 if not is_full_width else 0.020))

        lines = line_boxes or []
        probe_up = max(18.0, float(page_h) * 0.08)
        probe_down = max(18.0, float(page_h) * 0.10)
        very_close_up = max(10.0, float(page_h) * 0.015)
        title_top = float(r.y0)
        boundary_y1 = -1.0
        axis_bottom = float(r.y1)
        boundary_y0 = float(page_h) + 1.0

        upper_lines: list[tuple[fitz.Rect, str, float, bool, float, float]] = []
        for item in lines:
            try:
                lb = fitz.Rect(item[0])
                txt = str(item[1])
            except Exception:
                continue
            try:
                max_font = float(item[2]) if len(item) >= 3 else 0.0
            except Exception:
                max_font = 0.0
            try:
                is_bold = bool(item[3]) if len(item) >= 4 else False
            except Exception:
                is_bold = False
            if float(lb.y1) <= float(r.y0) + 1.0:
                gap_up = float(r.y0) - float(lb.y1)
                if gap_up <= probe_up:
                    x_ov = _overlap_1d(float(r.x0), float(r.x1), float(lb.x0), float(lb.x1))
                    min_w = max(1.0, min(float(r.width), float(lb.width)))
                    upper_lines.append((lb, txt, max_font, is_bold, gap_up, x_ov / min_w))

        for item in lines:
            try:
                lb = fitz.Rect(item[0])
                txt = str(item[1])
            except Exception:
                continue
            if float(lb.y1) <= float(r.y0) + 1.0:
                gap_up = float(r.y0) - float(lb.y1)
                if gap_up <= probe_up:
                    x_ov = _overlap_1d(float(r.x0), float(r.x1), float(lb.x0), float(lb.x1))
                    min_w = max(1.0, min(float(r.width), float(lb.width)))
                    overlap_ratio = x_ov / min_w
                    if overlap_ratio >= 0.45:
                        try:
                            max_font = float(item[2]) if len(item) >= 3 else 0.0
                        except Exception:
                            max_font = 0.0
                        try:
                            is_bold = bool(item[3]) if len(item) >= 4 else False
                        except Exception:
                            is_bold = False
                        word_n = len(re.findall(r"[A-Za-z]{2,}", txt))
                        width_ratio = float(lb.width) / max(1.0, float(r.width))
                        line_center = (float(lb.x0) + float(lb.x1)) * 0.5
                        rect_center = (float(r.x0) + float(r.x1)) * 0.5
                        center_delta = abs(line_center - rect_center)
                        starts_near_left = float(lb.x0) <= float(r.x0) + max(10.0, float(page_w) * 0.012)
                        low_txt = txt.lower()
                        has_panel_peer = any(
                            self._looks_axis_or_panel_text(other_txt)
                            and abs(float(other_lb.y0) - float(lb.y0)) <= max(8.0, float(lb.height))
                            for other_lb, other_txt, *_ in upper_lines
                        )
                        looks_ragged_body_tail = (
                            gap_up <= very_close_up
                            and (not is_bold)
                            and max_font <= 10.5
                            and word_n >= 3
                            and len(txt) >= 12
                            and starts_near_left
                            and width_ratio <= 0.72
                            and "." not in low_txt
                            and ("http://" not in low_txt)
                            and ("https://" not in low_txt)
                            and ("www." not in low_txt)
                        )
                        looks_wide_figure_title = (
                            gap_up <= very_close_up
                            and (is_bold or has_panel_peer or max_font >= 11.5)
                            and word_n >= 4
                            and len(txt) <= 140
                            and "." not in txt
                            and ("http://" not in low_txt)
                            and ("https://" not in low_txt)
                            and ("www." not in low_txt)
                        )
                        if looks_ragged_body_tail:
                            # Short ragged paragraph tails often sit right above a figure and
                            # were previously mistaken for figure-local titles.
                            boundary_y1 = max(boundary_y1, float(lb.y1))
                        elif self._looks_axis_or_panel_text(txt):
                            title_top = min(title_top, float(lb.y0))
                        elif looks_wide_figure_title:
                            title_top = min(title_top, float(lb.y0))
                        elif (
                            gap_up <= very_close_up
                            and
                            2 <= word_n <= 10
                            and len(txt) <= 72
                            and width_ratio <= 0.60
                            and (is_bold or has_panel_peer or center_delta <= max(24.0, float(r.width) * 0.12))
                            and "." not in txt
                            and ("http://" not in low_txt)
                            and ("https://" not in low_txt)
                            and ("www." not in low_txt)
                        ):
                            # Likely a figure-internal title or row/column label.
                            title_top = min(title_top, float(lb.y0))
                        elif len(txt) >= 24 and word_n >= 4:
                            # Longer prose above is likely body text; use it as a stop boundary.
                            boundary_y1 = max(boundary_y1, float(lb.y1))
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

        if title_top < float(r.y0):
            pad_top = max(pad_top, float(r.y0) - title_top + 2.0)
        if axis_bottom > float(r.y1):
            pad_bottom = max(pad_bottom, axis_bottom - float(r.y1) + 2.0)

        x0 = max(0.0, float(r.x0) - pad_x)
        y0 = max(0.0, float(r.y0) - pad_top)
        if boundary_y1 >= 0.0:
            y0 = max(y0, min(float(r.y0) - 2.0, boundary_y1 + 1.0))
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
    def _vision_references_prefer_local_enabled() -> bool:
        try:
            raw = str(os.environ.get("KB_PDF_VISION_REFS_PREFER_LOCAL", "1") or "1").strip().lower()
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
    def _llm_reference_polish_enabled() -> bool:
        try:
            raw = str(os.environ.get("KB_PDF_LLM_REFERENCE_POLISH", "1") or "1").strip().lower()
        except Exception:
            raw = "1"
        return raw in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _is_probable_page_number_block(block: TextBlock, *, page_w: float, page_h: float) -> bool:
        return is_probable_page_number_block(block, page_w=page_w, page_h=page_h)

    @staticmethod
    def _structured_crop_lane_for_bbox(
        bbox: tuple[float, float, float, float],
        *,
        page_w: float,
        col_split: float,
    ) -> str:
        return structured_crop_lane_for_bbox(
            bbox,
            page_w=page_w,
            col_split=col_split,
        )

    @staticmethod
    def _expand_structured_crop_rect(
        rect: "fitz.Rect",
        *,
        lane: str,
        page_w: float,
        page_h: float,
        col_split: float,
    ) -> "fitz.Rect":
        return expand_structured_crop_rect(
            rect,
            lane=lane,
            page_w=page_w,
            page_h=page_h,
            col_split=col_split,
        )

    @staticmethod
    def _fallback_markdown_from_blocks(blocks: list[TextBlock]) -> str:
        return fallback_markdown_from_blocks(blocks)

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
        return convert_page_with_layout_crops(
            self,
            page=page,
            page_index=page_index,
            total_pages=total_pages,
            page_hint=page_hint,
            speed_mode=speed_mode,
            pdf_path=pdf_path,
            assets_dir=assets_dir,
            image_names=image_names,
        )

    @staticmethod
    def _is_reference_placeholder_line(text: str) -> bool:
        return is_reference_placeholder_line(text)

    def _sanitize_reference_crop_markdown(self, md: str) -> str:
        return sanitize_reference_crop_markdown(md)

    def _merge_reference_crop_markdowns(self, parts: list[str]) -> str:
        return merge_reference_crop_markdowns(parts)

    def _build_reference_column_crop_rects(
        self,
        *,
        page,
        page_w: float,
        page_h: float,
    ) -> list[fitz.Rect]:
        return build_reference_column_crop_rects(page=page, page_w=page_w, page_h=page_h)

    def _convert_references_page_with_column_vl(
        self,
        *,
        page,
        page_index: int,
        total_pages: int,
        page_hint: str,
        speed_mode: str,
    ) -> Optional[str]:
        return convert_references_page_with_column_vl(
            self,
            page=page,
            page_index=page_index,
            total_pages=total_pages,
            page_hint=page_hint,
            speed_mode=speed_mode,
        )

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
        max_tokens_override: Optional[int] = None,
        formula_placeholders: Optional[dict[str, str]] = None,
        skip_references_column_mode: bool = False,
    ) -> Optional[str]:
        return convert_page_with_vision_guardrails(
            self,
            png_bytes=png_bytes,
            page=page,
            page_index=page_index,
            total_pages=total_pages,
            page_hint=page_hint,
            speed_mode=speed_mode,
            is_references_page=is_references_page,
            pdf_path=pdf_path,
            assets_dir=assets_dir,
            image_names=image_names,
            max_tokens_override=max_tokens_override,
            formula_placeholders=formula_placeholders,
            skip_references_column_mode=skip_references_column_mode,
        )

    def _repair_broken_image_links(self, md: str, *, save_dir: Path, assets_dir: Path) -> str:
        return repair_broken_image_links(md, save_dir=save_dir, assets_dir=assets_dir)

    @staticmethod
    def _inject_missing_page_image_links(
        md: str,
        *,
        page_index: int,
        image_names: list[str],
        figure_meta_by_asset: Optional[dict[str, dict]] = None,
        is_references_page: bool = False,
    ) -> str:
        return inject_missing_page_image_links(
            md,
            page_index=page_index,
            image_names=image_names,
            figure_meta_by_asset=figure_meta_by_asset,
            is_references_page=is_references_page,
        )

    @staticmethod
    def _inject_page_image_captions_from_meta(
        md: str,
        *,
        page_index: int,
        figure_meta_by_asset: Optional[dict[str, dict]] = None,
    ) -> str:
        return inject_page_image_captions_from_meta(
            md,
            page_index=page_index,
            figure_meta_by_asset=figure_meta_by_asset,
        )

    @staticmethod
    def _normalize_page_image_caption_order(
        md: str,
        *,
        page_index: int,
        figure_meta_by_asset: Optional[dict[str, dict]] = None,
    ) -> str:
        return normalize_page_image_caption_order(
            md,
            page_index=page_index,
            figure_meta_by_asset=figure_meta_by_asset,
        )

    @staticmethod
    def _normalize_page_local_image_link_order(
        md: str,
        *,
        page_index: int,
        image_names: list[str],
    ) -> str:
        return normalize_page_local_image_link_order(
            md,
            page_index=page_index,
            image_names=image_names,
        )

    @staticmethod
    def _extract_figure_number_from_text(text: str) -> Optional[int]:
        return extract_figure_number_from_text(text)

    @staticmethod
    def _figure_remap_debug_enabled() -> bool:
        return figure_remap_debug_enabled()

    @staticmethod
    def _reorder_page_figure_pairs_by_number(
        md: str,
        *,
        page_index: int,
    ) -> str:
        return reorder_page_figure_pairs_by_number(md, page_index=page_index)

    @staticmethod
    def _remap_page_image_links_by_caption(
        md: str,
        *,
        page_index: int,
        figure_meta_by_asset: Optional[dict[str, dict]] = None,
    ) -> str:
        return remap_page_image_links_by_caption(
            md,
            page_index=page_index,
            figure_meta_by_asset=figure_meta_by_asset,
        )

    def _extract_page_figure_caption_candidates(self, page) -> list[dict]:
        return extract_page_figure_caption_candidates(page)

    def _match_figure_entries_with_captions(
        self,
        *,
        page,
        figure_entries: list[dict],
        caption_candidates: list[dict],
    ) -> list[dict]:
        return match_figure_entries_with_captions(
            page=page,
            figure_entries=figure_entries,
            caption_candidates=caption_candidates,
        )

    def _split_visual_rects_by_internal_captions(
        self,
        *,
        page,
        visual_rects: list["fitz.Rect"],
        caption_candidates: list[dict],
    ) -> list["fitz.Rect"]:
        return split_visual_rects_by_internal_captions(
            page=page,
            visual_rects=visual_rects,
            caption_candidates=caption_candidates,
        )

    def _persist_page_figure_metadata(
        self,
        *,
        assets_dir: Path,
        page_index: int,
        figure_entries: list[dict],
    ) -> dict[str, dict]:
        return persist_page_figure_metadata(
            assets_dir=assets_dir,
            page_index=page_index,
            figure_entries=figure_entries,
        )

    @staticmethod
    def _build_page_figure_mapping_hint(figure_meta_by_asset: Optional[dict[str, dict]]) -> str:
        return build_page_figure_mapping_hint(figure_meta_by_asset)

    def _reconcile_figure_metadata_from_markdown(self, *, md: str, assets_dir: Path) -> dict[str, dict]:
        return reconcile_figure_metadata_from_markdown(md=md, assets_dir=assets_dir)

    def _detect_table_rects_for_fallback(
        self,
        page,
        *,
        page_index: int,
        pdf_path: Path,
        visual_rects: Optional[list["fitz.Rect"]] = None,
    ) -> list["fitz.Rect"]:
        return detect_table_rects_for_fallback(
            page,
            page_index=page_index,
            pdf_path=pdf_path,
            visual_rects=visual_rects,
        )

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
        return inject_table_image_fallbacks(
            md,
            page=page,
            page_index=page_index,
            pdf_path=pdf_path,
            assets_dir=assets_dir,
            visual_rects=visual_rects,
            is_references_page=is_references_page,
            dpi=int(getattr(self, "dpi", 200) or 200),
        )

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
            figure_meta_by_asset=figure_meta_by_asset,
            is_references_page=is_references_page,
        )
        md = self._remap_page_image_links_by_caption(
            md,
            page_index=page_index,
            figure_meta_by_asset=figure_meta_by_asset,
        )
        md = self._inject_page_image_captions_from_meta(
            md,
            page_index=page_index,
            figure_meta_by_asset=figure_meta_by_asset,
        )
        md = self._normalize_page_image_caption_order(
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
        return cleanup_page_local_image_markdown(md, page_index=page_index)

    # Removed: _process_batch_fast and _process_batch_llm (old text extraction methods)
    # Now only using vision-direct mode (_process_batch_vision_direct) and no-LLM mode (_process_batch_no_llm)

    # ------------------------------------------------------------------
    # Vision-direct mode: screenshot each page -> VL model -> Markdown
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

        speed_config = self._get_speed_mode_config("no_llm", total_pages)
        raw_workers = str(os.environ.get("KB_PDF_NO_LLM_PAGE_WORKERS", "") or "").strip()
        try:
            num_workers = int(raw_workers) if raw_workers else int(speed_config.get("max_parallel_pages", 1) or 1)
        except Exception:
            num_workers = int(speed_config.get("max_parallel_pages", 1) or 1)
        num_workers = max(1, min(int(num_workers), int(end - start)))
        prepare_docs = threading.local()
        opened_prepare_docs = []
        opened_prepare_docs_lock = threading.Lock()

        def _get_prepare_doc():
            if fitz is None:
                return None
            local_doc = getattr(prepare_docs, "doc", None)
            if local_doc is None:
                local_doc = fitz.open(pdf_path)
                prepare_docs.doc = local_doc
                with opened_prepare_docs_lock:
                    opened_prepare_docs.append(local_doc)
            return local_doc

        def _prepare_page_task(page_idx: int) -> tuple[int, Optional[dict], Optional[str]]:
            try:
                if fitz is not None:
                    local_doc = _get_prepare_doc()
                    page = local_doc.load_page(page_idx)
                else:
                    page = doc.load_page(page_idx)
                prepared = prepare_page_render_input(
                    self,
                    page,
                    page_index=page_idx,
                    pdf_path=pdf_path,
                    assets_dir=assets_dir,
                )
                return page_idx, prepared, None
            except Exception as e:
                return page_idx, None, str(e)

        if num_workers <= 1:
            print(f"[NO_LLM] Processing pages {start+1}-{end} with local extraction (sequential)", flush=True)
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

        print(
            f"[NO_LLM] Processing pages {start+1}-{end} with local extraction "
            f"(parallel prepare={num_workers}, serial render=1)",
            flush=True,
        )
        prepared_by_page: dict[int, Optional[dict]] = {}
        errors_by_page: dict[int, str] = {}
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                future_map = {
                    pool.submit(_prepare_page_task, i): i
                    for i in range(start, end)
                }
                for fut in as_completed(future_map):
                    page_idx = future_map[fut]
                    try:
                        idx, prepared, err = fut.result()
                    except Exception as e:
                        idx, prepared, err = page_idx, None, str(e)
                    if err:
                        errors_by_page[idx] = err
                        prepared_by_page[idx] = None
                    else:
                        prepared_by_page[idx] = prepared
        finally:
            for local_doc in opened_prepare_docs:
                try:
                    local_doc.close()
                except Exception:
                    pass

        for i in range(start, end):
            t0 = time.time()
            if errors_by_page.get(i):
                print(f"[NO_LLM] error page {i+1}: {errors_by_page[i]}", flush=True)
                results[i] = None
                continue
            try:
                page = doc.load_page(i)
                prepared = prepared_by_page.get(i)
                if not prepared:
                    print(f"[NO_LLM] error page {i+1}: empty prepared payload", flush=True)
                    results[i] = None
                    continue
                print(f"Rendering page {i+1}/{total_pages} (no-llm) ...", flush=True)
                results[i] = render_prepared_page(
                    self,
                    prepared=prepared,
                    page=page,
                    page_index=i,
                    assets_dir=assets_dir,
                )
                print(f"Finished page {i+1}/{total_pages} ({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"[NO_LLM] error page {i+1}: {e}", flush=True)
                results[i] = None
        return results

    def _process_page(self, page, page_index: int, pdf_path: Path, assets_dir: Path) -> str:
        return process_page(self, page, page_index=page_index, pdf_path=pdf_path, assets_dir=assets_dir)

    def _merge_adjacent_math_fragments(self, blocks: List[TextBlock], *, page_wh: tuple[float, float]) -> List[TextBlock]:
        """
        Merge adjacent math fragments split by PDF extraction.

        Typical failure mode (your screenshot):
        - one display equation becomes multiple blocks: "N X", ", (5)", "r - R", ...
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
            # "N X" / "r in R"
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
                sym_n = len(re.findall(r"[=+\-*/^_{}\\\[\]]|[\u2208\u2264\u2265\u2248\u00D7\u00B7\u03A3\u2212\u222B]", ss))
                has_sentence = (". " in ss) or ("? " in ss) or ("! " in ss)
                if word_n >= 14 and (sym_n <= 10 or has_sentence):
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
                # (The paragraph often contains symbols like "r in R" which are math-ish, but it's prose.)
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
        return extract_text_blocks(
            self,
            page,
            page_index=page_index,
            body_size=body_size,
            tables=tables,
            visual_rects=visual_rects,
            assets_dir=assets_dir,
            is_references_page=is_references_page,
        )

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

        # Fix "hat" when OCR emits a caret with whitespace: "^ C" or "caret C" (normalized to "^ C")
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
        return merge_split_formulas(md)

    def _merge_fragmented_formulas(self, md: str) -> str:
        return merge_fragmented_formulas(md)
    
    def _is_formula_fragment(self, text: str) -> bool:
        return is_formula_fragment(text)
    
    def _looks_like_formula_continuation(self, text: str, existing_parts: list[str]) -> bool:
        return looks_like_formula_continuation(text, existing_parts)

    def _basic_formula_cleanup(self, md: str) -> str:
        return basic_formula_cleanup(md)

    def _fix_heading_structure(self, md: str) -> str:
        return fix_heading_structure(md)

    def _is_likely_formula(self, text: str) -> bool:
        return is_likely_formula(text)
    
    def _fix_vision_formula_errors(self, md: str) -> str:
        return fix_vision_formula_errors(md)

    def _fix_references_format(self, md: str) -> str:
        return fix_references_format(md)
    
    def _format_references_block(self, ref_lines: list[tuple[int, str]]) -> list[str]:
        return format_references_block(ref_lines)
    
    def _format_single_reference(self, text: str, num: int) -> str:
        return format_single_reference(text, num)
    
    def _formula_to_plain_text(self, formula: str) -> str:
        return formula_to_plain_text(formula)

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
        return llm_fix_inline_formulas(self, md)

    def _llm_fix_references(self, md: str) -> str:
        return llm_fix_references(self, md)

    def _llm_polish_references(self, md: str) -> str:
        return llm_polish_references(self, md)
    
    def _llm_format_references_chunk(self, ref_text: str) -> Optional[str]:
        return llm_format_references_chunk(self, ref_text)

    def _llm_fix_tables(self, md: str) -> str:
        return llm_fix_tables(self, md)
    
    def _llm_fix_tables_with_screenshot(self, md: str, pdf_path: Path, save_dir: Path) -> str:
        return llm_fix_tables_with_screenshot(self, md, pdf_path, save_dir)
    
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
        return llm_fix_display_math(self, md)

    def _llm_light_cleanup(self, md: str) -> str:
        return llm_light_cleanup(self, md)
    
    def _llm_cleanup_chunk(self, md_chunk: str) -> str:
        return llm_cleanup_chunk(self, md_chunk)

    def _llm_postprocess_markdown(self, md: str) -> str:
        return llm_postprocess_markdown(self, md)
    
    def _llm_repair_markdown_chunk(self, md_chunk: str) -> str:
        return llm_repair_markdown_chunk(self, md_chunk)
    
    def _llm_final_quality_check(self, md: str) -> str:
        return llm_final_quality_check(self, md)
    
    def _llm_final_quality_check_chunk(self, md_chunk: str) -> str:
        return llm_final_quality_check_chunk(self, md_chunk)
    
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
