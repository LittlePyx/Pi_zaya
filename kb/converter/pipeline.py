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
from .geometry_utils import _rect_area, _union_rect
from .text_utils import _normalize_text, _common_prefix_length
from .layout_analysis import (
    detect_body_font_size,
    build_repeated_noise_texts,
    _pick_render_scale,
    _collect_visual_rects,
    _is_frontmatter_noise_line,
    _pick_column_range,
    sort_blocks_reading_order,
)
from .tables import _extract_tables_by_layout, _is_markdown_table_sane, table_text_to_markdown
from .llm_worker import LLMWorker
from .post_processing import postprocess_markdown


class PDFConverter:
    def __init__(self, cfg: ConvertConfig):
        self.cfg = cfg
        self.llm_worker = LLMWorker(cfg)
        self.noise_texts: Set[str] = set()

    def convert(self, pdf_path: str, save_dir: str) -> None:
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) not installed.")
            
        pdf_path = Path(pdf_path).resolve()
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        assets_dir = save_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Pre-scan for noise
        self.noise_texts = build_repeated_noise_texts(doc)
        print(f"Detected {len(self.noise_texts)} repeated lines as noise.")
        
        # Process pages
        # For simplicity, we'll use a sequential loop or simple batching here.
        # The original code had complex batching. We'll simplify to page-by-page for now
        # or use ThreadPool if configured.
        
        md_pages = [None] * total_pages
        
        if self.cfg.llm:
            # LLM mode might run in parallel if configured
            md_pages = self._process_batch_llm(doc, pdf_path, assets_dir)
        else:
            # Fast mode
            md_pages = self._process_batch_fast(doc, pdf_path, assets_dir)
            
        final_md = "\n\n".join(filter(None, md_pages))
        final_md = postprocess_markdown(final_md)
        
        out_file = save_dir / "output.md"
        out_file.write_text(final_md, encoding="utf-8")
        print(f"Saved to {out_file}")

    def _process_batch_fast(self, doc, pdf_path: Path, assets_dir: Path) -> List[Optional[str]]:
        results = [None] * len(doc)
        for i, page in enumerate(doc):
            print(f"Processing page {i+1}/{len(doc)}...")
            results[i] = self._process_page(
                page, 
                page_index=i, 
                pdf_path=pdf_path, 
                assets_dir=assets_dir
            )
        return results

    def _process_batch_llm(self, doc, pdf_path: Path, assets_dir: Path) -> List[Optional[str]]:
        # Similar to fast but potentially using LLM worker
        # We'll just reuse the sequential loop for now to be safe
        return self._process_batch_fast(doc, pdf_path, assets_dir)

    def _process_page(self, page, page_index: int, pdf_path: Path, assets_dir: Path) -> str:
        # 1. Analyze Layout
        body_size = detect_body_font_size([page]) # Heuristic was on full doc, but per-page is okay fallback
        
        # 2. Extract specific rects
        visual_rects = _collect_visual_rects(page)
        
        # 3. Extract Tables
        # We need to map table rects to avoid processing them as text
        tables_found = _extract_tables_by_layout(
            page, 
            pdf_path=pdf_path, 
            page_index=page_index,
            visual_rects=visual_rects
        )
        
        # 4. Extract Text Blocks
        blocks = self._extract_text_blocks(
            page, 
            page_index=page_index, 
            body_size=body_size,
            tables=tables_found,
            visual_rects=visual_rects,
            assets_dir=assets_dir
        )
        
        # 5. LLM Classification / Repair
        if self.cfg.llm:
            # Enhance blocks with LLM
            blocks = self._enhance_blocks_with_llm(blocks, page_index, page)
            
        # 6. Render
        return self._render_blocks_to_markdown(blocks, page_index)

    def _extract_text_blocks(
        self, 
        page, 
        page_index: int, 
        body_size: float, 
        tables: List[Tuple[fitz.Rect, str]], 
        visual_rects: List[fitz.Rect],
        assets_dir: Path
    ) -> List[TextBlock]:
        
        text_blocks = []
        # Get raw blocks
        page_dict = page.get_text("dict")
        raw_blocks = page_dict.get("blocks", [])
        
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
                
            # Process lines
            lines_text = []
            max_size = 0.0
            is_bold = False
            
            if "lines" not in b:
                continue
                
            for l in b["lines"]:
                for s in l["spans"]:
                    t = s["text"]
                    if not t.strip():
                        continue
                    # Check font size/bold
                    size = s["size"]
                    font = s["font"].lower()
                    if size > max_size:
                        max_size = size
                    if "bold" in font or "dubai-bold" in font: # heuristic
                        is_bold = True
                    lines_text.append(t)
            
            full_text = " ".join(lines_text).strip()
            if not full_text:
                continue
            
            # Filter noise
            if full_text in self.noise_texts:
                continue
            if _is_frontmatter_noise_line(full_text):
                continue
                
            # Create Block
            tb = TextBlock(
                bbox=tuple(bbox),
                text=full_text,
                max_font_size=max_size,
                is_bold=is_bold
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
            
        # Insert Images (Visual Rects) as Blocks?
        # Usually checking for images is done separately, but let's add placeholder
        for rect in visual_rects:
            # We should save image here
            img_name = f"page_{page_index+1}_fig_{int(rect.x0)}_{int(rect.y0)}.png"
            img_path = assets_dir / img_name
            
            # Render rect to image
            pix = page.get_pixmap(clip=rect, dpi=200)
            pix.save(img_path)
            
            tb = TextBlock(
                bbox=tuple(rect),
                text=f"![Figure](./assets/{img_name})",
                max_font_size=body_size
            )
            text_blocks.append(tb)

        # Sort reading order
        sorted_blocks = sort_blocks_reading_order(text_blocks, page_width=W)
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
            
        # Apply classifications
        # This mapping depends on the JSON structure from LLM
        # For now, simplistic application
        for item in classified:
            idx = item.get("i")
            kind = item.get("kind")
            if idx is not None and 0 <= idx < len(blocks):
                b = blocks[idx]
                if kind == "heading":
                    lvl = item.get("heading_level")
                    if lvl:
                        b.heading_level = f"[H{lvl}]"
                elif kind == "table":
                    b.is_table = True
                elif kind == "code":
                    b.is_code = True
                elif kind == "math":
                    b.is_math = True
                    
        return blocks

    def _render_blocks_to_markdown(self, blocks: List[TextBlock], page_index: int) -> str:
        out = []
        for b in blocks:
            if b.heading_level:
                # [H1] -> #
                lvl = int(b.heading_level.replace("[H", "").replace("]", ""))
                out.append("#" * lvl + " " + b.text)
            elif b.is_table:
                if b.table_markdown:
                    out.append(b.table_markdown)
                else:
                    out.append(b.text) # Fallback
            elif b.is_code:
                out.append("```\n" + b.text + "\n```")
            elif b.is_math:
                out.append("$$\n" + b.text + "\n$$")
            else:
                out.append(b.text)
            out.append("")
        return "\n".join(out)
